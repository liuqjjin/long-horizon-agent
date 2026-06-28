"""Unit tests for pure parsing/factory logic that the integration tests skip.

Targets the MCP result parsing (ccc backend), the unified-diff extraction, and the
LLM factory — real logic that was previously only exercised end-to-end (or not).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.live_context.backends.ccc_backend import _extract_results, _result_to_codehit
from lha.live_context.models import CodeHit, ContextBundle, Freshness
from lha.llm import DeterministicStub, get_llm
from lha.llm.base import LLMClient, _touched_from_diff, extract_file_blocks, extract_unified_diff


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


def test_patch_from_response_builds_file_contents_and_diff(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    resp = "### m.py\n```python\ndef f():\n    return 2\n```\n"
    patch = _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)
    assert patch.file_contents["m.py"] == "def f():\n    return 2\n"
    assert patch.touched_files == ["m.py"]
    assert patch.unified_diff  # a display diff is recorded
    assert "return 2" in patch.unified_diff


def test_patch_from_response_skips_unchanged(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    resp = "### m.py\n```python\ndef f():\n    return 1\n```\n"  # identical
    patch = _Echo()._patch_from_response(_step(), _bundle(), tmp_path, resp)
    assert patch.is_empty()


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
