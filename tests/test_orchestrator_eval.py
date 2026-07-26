"""Hermetic tests for orchestrator result parsing and the eval report."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from lha.config import Config
from lha.eval import EvalReport, EvalResult, _case_resume, _eval_data_root
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


def test_resume_eval_uses_one_persisted_budget_contract(tmp_path, monkeypatch):
    class _TTY:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr("sys.stdin", _TTY())

    def unexpected_prompt(*args, **kwargs):
        raise AssertionError("self-eval must not depend on an interactive terminal")

    monkeypatch.setattr("builtins.input", unexpected_prompt)
    result = _case_resume(
        Config(
            runs_dir=tmp_path / "runs",
            data_dir=Path.cwd() / "data",
            code_backend="null",
            use_skill_memory=False,
        )
    )

    assert result.passed, result.detail
    assert "first=AWAITING_APPROVAL" in result.detail
    assert "resumed=DONE" in result.detail


def test_quick_eval_materializes_packaged_fixtures_outside_checkout(tmp_path, monkeypatch):
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    root = _eval_data_root(Config(runs_dir=tmp_path / "runs"), quick=True)

    assert root == (tmp_path / "runs" / "eval" / "_fixtures").resolve()
    assert (root / "tasks" / "fix_average.yaml").is_file()
    assert (root / "sample_repo" / "tests" / "test_mathutils.py").is_file()
    assert (root / "papers" / "note_srgan.md").is_file()
    assert not list(root.rglob("__pycache__"))


def test_full_eval_outside_checkout_fails_with_an_explicit_boundary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        _eval_data_root(Config(runs_dir=tmp_path / "runs"), quick=False)
    except FileNotFoundError as error:
        assert "--quick" in str(error)
    else:
        raise AssertionError("full installed-package eval unexpectedly found checkout fixtures")


def test_checkout_eval_keeps_using_checkout_data():
    root = _eval_data_root(Config(), quick=True)
    assert root == (Path.cwd() / "data").resolve()


def test_packaged_quick_fixtures_match_the_checkout_corpus():
    packaged = resources.files("lha.resources").joinpath("eval")
    for relative in (
        "tasks/fix_average.yaml",
        "sample_repo/mathutils.py",
        "sample_repo/pyproject.toml",
        "sample_repo/tests/test_mathutils.py",
        "papers/note_srgan.md",
    ):
        assert packaged.joinpath(relative).read_bytes() == (Path("data") / relative).read_bytes()
