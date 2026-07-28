"""Unit tests for pure parsing/factory logic that the integration tests skip.

Targets the MCP result parsing (ccc backend), the unified-diff extraction, and the
LLM factory — real logic that was previously only exercised end-to-end (or not).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.live_context.backends.ccc_backend import _extract_results, _result_to_codehit
from lha.live_context.models import CodeHit, ContextBundle, ContextItem, Freshness, Provenance
from lha.llm import DeterministicStub, get_llm
from lha.llm.base import (
    _MAX_IMPLEMENTATION_PROMPT_BYTES,
    _MAX_SOURCE_FILE_BYTES,
    LLMClient,
    _touched_from_diff,
    extract_file_blocks,
    extract_unified_diff,
)


# --- ccc MCP result parsing -------------------------------------------------
def test_extract_results_from_structured_content():
    cr = SimpleNamespace(
        structuredContent={"results": [{"path": "a.py"}, {"path": "b.py"}]}, content=[]
    )
    assert [r["path"] for r in _extract_results(cr)] == ["a.py", "b.py"]


def test_extract_results_from_text_json_blocks():
    cr = SimpleNamespace(
        structuredContent=None, content=[SimpleNamespace(text='[{"path": "a.py"}]')]
    )
    assert _extract_results(cr) == [{"path": "a.py"}]


def test_extract_results_empty_when_nothing_parseable():
    cr = SimpleNamespace(structuredContent=None, content=[SimpleNamespace(text="not json")])
    assert _extract_results(cr) == []


def test_result_to_codehit_maps_fields_and_locator():
    d = {
        "path": "m.py",
        "line_start": 1,
        "line_end": 9,
        "code": "x = 1",
        "language": "python",
        "score": 0.5,
    }
    hit = _result_to_codehit(d, Path("/tmp/somewhere"), now())
    assert isinstance(hit, CodeHit)
    assert hit.text == "x = 1"
    assert hit.line_start == 1 and hit.line_end == 9
    assert hit.language == "python"
    assert hit.provenance.source_kind == "code"
    assert hit.provenance.locator.endswith("m.py:1-9")


def test_result_to_codehit_tolerates_missing_fields():
    hit = _result_to_codehit({}, Path("/tmp/x"), now())  # no path/code/lines
    assert isinstance(hit, CodeHit)
    assert hit.text == ""
    assert hit.line_start is None


# --- unified-diff extraction ------------------------------------------------
def test_extract_unified_diff_from_fence():
    text = "blah\n```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n```\ntrailing"
    diff = extract_unified_diff(text)
    assert diff.startswith("--- a/x.py")
    assert _touched_from_diff(diff) == ["x.py"]


def test_extract_unified_diff_bare_and_absent():
    assert extract_unified_diff("--- a/x.py\n+++ b/x.py\n").startswith("--- a/x.py")
    assert extract_unified_diff("just prose, no diff") == ""


# --- whole-file rewrite extraction (the robust apply path) ------------------
class _Echo(LLMClient):
    name = "echo"

    def complete(self, system: str, prompt: str) -> str:  # unused here
        return ""


class _PromptCapture(LLMClient):
    name = "capture"

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return ""


def _bundle() -> ContextBundle:
    return ContextBundle(query="q", freshness=Freshness(index_version="v", indexed_at=now()))


def _step() -> Step:
    return Step(step_id="s", kind="code", action="edit_code", goal="g")


def test_extract_file_blocks_multiple():
    text = (
        "Here is the fix.\n"
        "### pkg/a.py\n```python\nprint('a')\n```\n"
        "### b.py\n```\nx = 2\n```\n"
    )
    blocks = extract_file_blocks(text)
    assert blocks == {"pkg/a.py": "print('a')", "b.py": "x = 2"}


def test_patch_from_response_builds_one_executable_representation(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    resp = "### m.py\n```python\ndef f():\n    return 2\n```\n"
    patch = _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)
    assert patch.file_contents["m.py"] == "def f():\n    return 2\n"
    assert patch.touched_files == ["m.py"]
    # A Patch has exactly one executable representation.  The harness derives
    # the human review diff later from these bytes plus the persisted backup.
    assert patch.unified_diff == ""


def test_patch_from_response_skips_unchanged(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    resp = "### m.py\n```python\ndef f():\n    return 1\n```\n"  # identical
    patch = _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)
    assert patch.is_empty()


def test_patch_from_response_accepts_new_file_under_new_directory(tmp_path):
    resp = "### pkg/new.py\n```python\nvalue = 1\n```\n"
    patch = _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)
    assert patch.file_contents == {"pkg/new.py": "value = 1\n"}


def test_patch_from_response_single_block_fallback(tmp_path):
    # No '### path', but exactly one non-test source file -> map the lone block to it.
    (tmp_path / "only.py").write_text("v = 0\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_only.py").write_text("def test(): pass\n")
    resp = "```python\nv = 1\n```"
    patch = _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)
    assert patch.file_contents == {"only.py": "v = 1\n"}


def test_patch_from_response_rejects_path_escape(tmp_path):
    resp = "### ../evil.py\n```\nboom\n```\n"
    patch = _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)
    assert patch.is_empty()
    assert not (tmp_path.parent / "evil.py").exists()


def test_repo_prompt_rejects_python_symlink(tmp_path):
    secret = tmp_path.parent / "host-secret"
    secret.write_text("must not reach the model")
    (tmp_path / "leak.py").symlink_to(secret)

    with pytest.raises(ValueError, match="symbolic link"):
        _Echo._read_repo_python(tmp_path)


def test_repo_prompt_rejects_python_hardlink(tmp_path):
    secret = tmp_path.parent / "host-secret"
    secret.write_text("must not reach the model")
    os.link(secret, tmp_path / "leak.py")

    with pytest.raises(ValueError, match="standalone regular file"):
        _Echo._read_repo_python(tmp_path)


def test_patch_from_response_rejects_symlinked_target(tmp_path):
    secret = tmp_path.parent / "host-secret"
    secret.write_text("must not be overwritten")
    (tmp_path / "leak.py").symlink_to(secret)
    resp = "### leak.py\n```python\nvalue = 1\n```\n"

    with pytest.raises(ValueError, match="link"):
        _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)


def test_patch_from_response_rejects_hardlinked_target(tmp_path):
    secret = tmp_path.parent / "host-secret"
    secret.write_text("must not be overwritten")
    os.link(secret, tmp_path / "leak.py")
    resp = "### leak.py\n```python\nvalue = 1\n```\n"

    with pytest.raises(ValueError, match="standalone regular file"):
        _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)


def test_single_block_fallback_rejects_symlinked_source(tmp_path):
    secret = tmp_path.parent / "host-secret"
    secret.write_text("must not be selected")
    (tmp_path / "only.py").symlink_to(secret)

    with pytest.raises(ValueError, match="symbolic link"):
        _Echo._single_block_fallback(tmp_path, "```python\nvalue = 1\n```")


def test_prompt_selects_explicit_target_after_first_twelve_files(tmp_path):
    for ordinal in range(13):
        (tmp_path / f"a{ordinal:02}.py").write_text(f"VALUE = {ordinal}\n")
    (tmp_path / "z_target.py").write_text("TARGET = 'included whole'\n")
    client = _PromptCapture()
    step = Step(
        step_id="s",
        kind="code",
        action="edit_code",
        goal="Fix the retrieved regression.",
    )
    bundle = ContextBundle(
        query="regression",
        freshness=Freshness(index_version="v", indexed_at=now()),
        items=[
            ContextItem(
                text="TARGET is wrong",
                provenance=Provenance(source_kind="code", locator="z_target.py:1"),
            )
        ],
    )

    client.propose_patch(step, bundle, tmp_path)

    prompt = client.prompts[0]
    assert "### z_target.py\nTARGET = 'included whole'\n" in prompt
    assert prompt.index("### z_target.py") < prompt.index("### a00.py")


def test_prompt_rejects_referenced_source_that_cannot_fit_whole(tmp_path):
    (tmp_path / "large.py").write_text("x" * (_MAX_SOURCE_FILE_BYTES + 1))
    client = _PromptCapture()
    step = Step(
        step_id="s",
        kind="code",
        action="edit_code",
        goal="Fix large.py without changing its public API.",
    )

    with pytest.raises(ValueError, match="whole-file prompt limit: large.py"):
        client.propose_patch(step, _bundle(), tmp_path)

    assert client.prompts == []


def test_prompt_omits_oversize_unreferenced_source_instead_of_truncating(tmp_path):
    (tmp_path / "target.py").write_text("VALUE = 'complete-target'\n")
    (tmp_path / "unrelated.py").write_text("SECRET_TAIL_" + "x" * _MAX_SOURCE_FILE_BYTES)
    client = _PromptCapture()
    step = Step(
        step_id="s",
        kind="code",
        action="edit_code",
        goal="Fix target.py.",
    )

    client.propose_patch(step, _bundle(), tmp_path)

    prompt = client.prompts[0]
    assert "### target.py\nVALUE = 'complete-target'\n" in prompt
    assert "### unrelated.py" not in prompt
    assert "SECRET_TAIL_" not in prompt


def test_prompt_path_references_ignore_escape_absolute_and_missing_files(tmp_path):
    (tmp_path / "valid.py").write_text("VALUE = 1\n")
    (tmp_path / "other.py").write_text("OTHER = 1\n")
    outside = tmp_path.parent / "outside.py"
    outside.write_text("HOST_SECRET = 1\n")
    step = Step(
        step_id="s",
        kind="code",
        action="edit_code",
        goal="Fix ../outside.py, /tmp/host.py, missing.py, and valid.py.",
    )

    referenced = _Echo._referenced_repo_python(tmp_path, step, _bundle())

    assert referenced == ["valid.py"]
    client = _PromptCapture()
    client.propose_patch(step, _bundle(), tmp_path)
    assert "HOST_SECRET" not in client.prompts[0]


def test_prompt_selection_is_deterministic_across_creation_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    files = {
        "pkg/target.py": "TARGET = 1\n",
        "pkg/helper.py": "HELPER = 1\n",
        "small.py": "SMALL = 1\n",
        "larger.py": "LARGER = '" + "x" * 100 + "'\n",
    }
    for root, names in ((first, list(files)), (second, list(reversed(files)))):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(files[name])
    step = Step(
        step_id="s",
        kind="code",
        action="edit_code",
        goal="Fix pkg/target.py.",
    )
    prompts = []
    for root in (first, second):
        client = _PromptCapture()
        client.propose_patch(step, _bundle(), root)
        prompts.append(client.prompts[0])

    assert prompts[0] == prompts[1]
    assert prompts[0].index("### pkg/target.py") < prompts[0].index("### pkg/helper.py")
    assert prompts[0].index("### pkg/helper.py") < prompts[0].index("### small.py")


def test_implementation_prompt_has_hard_utf8_byte_budget(tmp_path):
    (tmp_path / "target.py").write_text("TARGET = 'complete'\n")
    for ordinal in range(30):
        (tmp_path / f"module_{ordinal:02}.py").write_text(
            f"VALUE_{ordinal} = '" + "数" * 2500 + "'\n"
        )
    item = ContextItem(
        text="检索" * 50_000,
        provenance=Provenance(
            source_kind="code",
            locator="target.py:1",
            indexed_at=now(),
        ),
    )
    bundle = ContextBundle(
        query="q",
        freshness=Freshness(index_version="v", indexed_at=now()),
        items=[item],
    )
    step = Step(
        step_id="s",
        kind="code",
        action="edit_code",
        goal="Fix target.py. " + "问题" * 20_000,
        prior_failures=["失败" * 30_000],
    )
    client = _PromptCapture()

    client.propose_patch(step, bundle, tmp_path)

    prompt = client.prompts[0]
    assert len(prompt.encode("utf-8")) <= _MAX_IMPLEMENTATION_PROMPT_BYTES
    assert "### target.py\nTARGET = 'complete'\n" in prompt
    assert "[issue text truncated to fixed prompt budget]" in prompt
    assert "[prior failures truncated to fixed prompt budget]" in prompt
    assert "[retrieved snippet truncated to fixed prompt budget]" in prompt


# --- LLM factory ------------------------------------------------------------
def test_get_llm_factory():
    assert isinstance(get_llm(Config(llm_backend="stub")), DeterministicStub)
    # claude_cli/anthropic construct without importing their optional backends
    assert get_llm(Config(llm_backend="claude_cli")).name == "claude_cli"
    assert get_llm(Config(llm_backend="anthropic")).name == "anthropic"
    try:
        get_llm(Config(llm_backend="nope"))
        raise AssertionError("expected ValueError for unknown backend")
    except ValueError:
        pass
