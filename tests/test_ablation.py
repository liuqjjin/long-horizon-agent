"""Deterministic tests for the paired verification ablation — no network.

The real ablation runs a live LLM, but every mechanism is pinned here with injected
fake backends: the paired trust/gate contrast, tamper-proof source-only patching,
the repair lift, transient-error handling, aggregation, and resumability.
"""

from __future__ import annotations

from pathlib import Path

from lha.ablation import (
    CONDITIONS,
    AblationReport,
    ConditionStats,
    RunRecord,
    _aggregate,
    _sanitize,
    run_ablation,
)
from lha.artifacts import Patch
from lha.config import Config
from lha.llm.base import LLMClient

_PYPROJECT = (
    '[project]\nname = "buggy"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n\n'
    '[tool.pytest.ini_options]\npythonpath = ["."]\n'
)
_TEST = "from m import f\n\n\ndef test_f():\n    assert f() == 2\n"


def _repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(_PYPROJECT)
    (root / "m.py").write_text("def f():\n    return 1\n")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_m.py").write_text(_TEST)
    return root


def _task(tmp_path: Path, src: Path) -> str:
    y = tmp_path / "task.yaml"
    y.write_text(
        "kind: issue_to_pr\n"
        'title: "f should return 2"\n'
        'description: "f() returns the wrong value"\n'
        f"target_repo: {src}\n"
        "inputs:\n  context_query: f\n"
        'success:\n  - "pytest passes"\n'
    )
    return str(y)


class _FixedLLM(LLMClient):
    """Always returns the same body for m.py (and optionally a tampered test)."""

    def __init__(self, value: int, tamper: bool = False):
        self.value = value
        self.tamper = tamper

    def complete(self, system: str, prompt: str) -> str:
        return ""

    def propose_patch(self, step, bundle, workdir) -> Patch:
        fc = {"m.py": f"def f():\n    return {self.value}\n"}
        if self.tamper:
            fc["tests/test_m.py"] = "def test_f():\n    assert True\n"
        return Patch(step_id=step.step_id, file_contents=fc, touched_files=list(fc))


class _RepairLLM(LLMClient):
    """Wrong first attempt, correct on the repair — exercises the repair loop."""

    def __init__(self):
        self.n = 0

    def complete(self, system: str, prompt: str) -> str:
        return ""

    def propose_patch(self, step, bundle, workdir) -> Patch:
        self.n += 1
        value = 2 if self.n >= 2 else 3
        return Patch(
            step_id=step.step_id,
            file_contents={"m.py": f"def f():\n    return {value}\n"},
            touched_files=["m.py"],
        )


def _base(tmp_path: Path) -> Config:
    return Config(runs_dir=tmp_path / "runs", data_dir=tmp_path / "nodata")


def _run(tmp_path, llm, out="out"):
    src = _repo(tmp_path / "src")
    return run_ablation(
        _base(tmp_path), [_task(tmp_path, src)], llm="stub", reps=1,
        out_dir=tmp_path / out, llm_client=llm,
    )


def _by_cond(report) -> dict[str, RunRecord]:
    return {r.condition: r for r in report.records}


# --- patch sanitization (tamper-proofing) -----------------------------------
def test_sanitize_keeps_only_source():
    p = Patch(
        step_id="s",
        file_contents={
            "m.py": "x",
            "tests/test_m.py": "y",
            "conftest.py": "z",
            "pyproject.toml": "w",
        },
    )
    s = _sanitize(p)
    assert set(s.file_contents) == {"m.py"}
    assert s.touched_files == ["m.py"]


# --- the paired trust/gate paths --------------------------------------------
def test_trust_false_gate_refuses_on_wrong_fix(tmp_path):
    rec = _by_cond(_run(tmp_path, _FixedLLM(3)))  # always wrong
    assert rec["trust"].claimed_success and not rec["trust"].true_success
    assert rec["trust"].false_success  # silent wrong answer
    assert rec["gate"].status == "FAILED" and not rec["gate"].false_success  # refused
    assert rec["verify"].status == "FAILED"  # repair can't fix a stuck-wrong model


def test_correct_fix_true_everywhere(tmp_path):
    rec = _by_cond(_run(tmp_path, _FixedLLM(2)))  # correct
    for c in ("trust", "gate", "verify"):
        assert rec[c].true_success and not rec[c].false_success


def test_verify_repairs_to_success(tmp_path):
    rec = _by_cond(_run(tmp_path, _RepairLLM()))  # wrong, then correct
    # trust/gate score the same first (wrong) attempt: trust accepts it, gate refuses
    assert rec["trust"].false_success
    assert rec["gate"].status == "FAILED"
    # verify repairs the same attempt to a real success
    assert rec["verify"].true_success and rec["verify"].repairs >= 1


def test_tamper_proof_grading(tmp_path):
    # A wrong fix that also rewrites the test to pass trivially: the test rewrite is
    # stripped, so the canonical oracle still catches the wrong source.
    rec = _by_cond(_run(tmp_path, _FixedLLM(3, tamper=True)))
    assert rec["trust"].false_success  # canonical test caught it despite the tamper
    assert rec["gate"].status == "FAILED"


# --- transient errors are not cached / resumable ----------------------------
class _FailingLLM(LLMClient):
    name = "failing"

    def complete(self, system: str, prompt: str) -> str:
        raise RuntimeError("backend down")

    def propose_patch(self, step, bundle, workdir) -> Patch:
        raise RuntimeError("backend down")


def test_transient_errors_excluded_and_not_cached(tmp_path, monkeypatch):
    import lha.ablation as abl

    monkeypatch.setattr(abl, "_LLM_RETRIES", 1)
    monkeypatch.setattr(abl.time, "sleep", lambda *a: None)
    src = _repo(tmp_path / "src")
    out = tmp_path / "out"
    rep = run_ablation(
        _base(tmp_path), [_task(tmp_path, src)], reps=1, out_dir=out, llm_client=_FailingLLM()
    )
    assert all(r.status == "ERROR" for r in rep.records)
    # ERROR cells are NOT cached, so a later good run recomputes them.
    assert not (out / "results" / "task__r0.json").exists()


def test_resumable_caches_real_outcomes(tmp_path):
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    base = _base(tmp_path)
    run_ablation(base, [task], reps=1, out_dir=out, llm_client=_FixedLLM(2))
    assert (out / "results" / "task__r0.json").exists()

    # A cached cell must NOT re-invoke the LLM.
    rep2 = run_ablation(base, [task], reps=1, out_dir=out, llm_client=_FailingLLM())
    assert _by_cond(rep2)["trust"].true_success  # served from cache, no LLM call


# --- aggregation + report ---------------------------------------------------
def test_aggregate_and_markdown():
    records = [
        RunRecord("t1", "trust", 0, "DONE", True, False, True, 0),
        RunRecord("t1", "gate", 0, "FAILED", False, False, False, 0),
        RunRecord("t1", "verify", 0, "DONE", True, True, False, 1),
        RunRecord("t2", "trust", 0, "DONE", True, True, False, 0),
        RunRecord("t2", "gate", 0, "DONE", True, True, False, 0),
        RunRecord("t2", "verify", 0, "DONE", True, True, False, 0),
    ]
    stats = {s.condition: s for s in _aggregate(records)}
    assert stats["trust"].false_success_rate == 0.5
    assert stats["gate"].false_success_rate == 0.0
    assert stats["verify"].true_success_rate == 1.0
    report = AblationReport("stub", "", 1, ["t1", "t2"], records, list(stats.values()))
    md = report.to_markdown()
    assert "Verification ablation" in md and "false success" in md and "false-pass" in md


def test_errored_runs_excluded_from_rates():
    records = [
        RunRecord("t1", "trust", 0, "ERROR", False, False, False, 0),
        RunRecord("t2", "trust", 0, "DONE", True, False, True, 0),
    ]
    stats = {s.condition: s for s in _aggregate(records)}
    assert stats["trust"].n == 1 and stats["trust"].false_success_rate == 1.0
    assert isinstance(stats["trust"], ConditionStats)
    assert CONDITIONS[0][0] == "trust"
