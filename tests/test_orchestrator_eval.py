"""Hermetic tests for orchestrator result parsing and the eval report."""

from __future__ import annotations

from dataclasses import dataclass

from lha.eval import EvalReport, EvalResult
from lha.orchestrator import _parse


@dataclass
class _Proc:
    stdout: str
    stderr: str = ""
    returncode: int = 0


def test_orchestrator_parses_result_line():
    out = (
        "some noise\n"
        "[Stats] cocoindex chatter\n"
        '__LHA_RESULT__ {"run_id": "r1", "status": "DONE", "verified": true}\n'
    )
    outcome = _parse("t.yaml", _Proc(out))
    assert outcome.status == "DONE"
    assert outcome.verified is True
    assert outcome.run_id == "r1"


def test_orchestrator_handles_missing_result():
    outcome = _parse("t.yaml", _Proc("just noise, no result line", "stderr tail"))
    assert outcome.status == "ERROR"
    assert "stderr tail" in outcome.detail


def test_orchestrator_flags_crash_after_clean_status():
    out = '__LHA_RESULT__ {"run_id": "r1", "status": "DONE", "verified": true}\n'
    outcome = _parse("t.yaml", _Proc(out, returncode=1))  # crashed after emitting
    assert outcome.status == "ERROR"
    assert outcome.run_id == "r1"


def test_eval_report_score_and_markdown():
    report = EvalReport(
        results=[
            EvalResult("a", "issue-to-PR", True, "ok"),
            EvalResult("b", "freshness", False, "nope"),
        ]
    )
    assert report.score == "1/2"
    assert report.all_passed is False
    md = report.to_markdown()
    assert "PASS" in md and "FAIL" in md
