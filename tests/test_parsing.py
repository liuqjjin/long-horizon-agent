"""Unit tests for pure parsing/factory logic that the integration tests skip.

Targets the MCP result parsing (ccc backend), the unified-diff extraction, and the
LLM factory — real logic that was previously only exercised end-to-end (or not).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lha.clock import now
from lha.config import Config
from lha.live_context.backends.ccc_backend import _extract_results, _result_to_codehit
from lha.live_context.models import CodeHit
from lha.llm import DeterministicStub, get_llm
from lha.llm.base import _touched_from_diff, extract_unified_diff


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
