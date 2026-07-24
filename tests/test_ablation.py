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


# --- P0-D: independent truth (prediction vs scorer) ---------------------------
def test_gate_rejected_correct_fix_is_scored_as_false_negative(tmp_path, monkeypatch):
    """The internal gate is a prediction, not truth: a correct fix the gate
    wrongly refuses must be graded by the scorer and counted as a false
    negative (measurable recall), not vanish."""
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    class _BrokenGateExec:
        """Agent-side backend whose pytest always 'fails' (e.g. broken local env)."""

        name = "broken-gate"

        def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
            return ProcResult(1, "", "simulated agent-env failure", 0.0)

        def python(self):
            return "python"

        def tool(self, name):
            return name

    monkeypatch.setattr(abl, "TrustedLocalBackend", _BrokenGateExec)
    report = _run(tmp_path, _FixedLLM(2))  # a CORRECT fix
    rec = _by_cond(report)
    # gate: claimed False (internal gate failed) but truth True (scorer passed)
    assert rec["gate"].claimed_success is False
    assert rec["gate"].true_success is True
    assert rec["gate"].false_success is False
    stats = {s.condition: s for s in report.stats}
    assert stats["gate"].fn == 1 and stats["gate"].recall == 0.0


def test_frozen_diff_excludes_oracle_and_junk(tmp_path):
    from lha.ablation import _frozen_diff

    src = _repo(tmp_path / "src")
    wd = tmp_path / "wd"
    import shutil as _sh

    _sh.copytree(src, wd)
    (wd / "m.py").write_text("def f():\n    return 2\n")  # source change
    (wd / "new_helper.py").write_text("x = 1\n")  # added file
    (wd / "tests" / "test_m.py").write_text("tampered")  # protected -> excluded
    (wd / "__pycache__").mkdir()
    (wd / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"junk")

    frozen = _frozen_diff(src, wd)
    assert set(frozen) == {"m.py", "new_helper.py"}


def test_frozen_diff_records_deletions(tmp_path):
    from lha.ablation import _frozen_diff

    src = _repo(tmp_path / "src")
    (src / "todelete.py").write_text("gone = 1\n")
    wd = tmp_path / "wd"
    import shutil as _sh

    _sh.copytree(src, wd)
    (wd / "todelete.py").unlink()
    frozen = _frozen_diff(src, wd)
    assert frozen == {"todelete.py": None}


def test_confusion_matrix_measures_false_positives():
    """A gate that passes a wrong fix (flaky oracle, dirty env) shows up as FP;
    nothing in the aggregation forces FP to zero."""
    records = [
        RunRecord("t1", "gate", 0, "DONE", True, False, True, 0, "", True, "sha1"),
        RunRecord("t2", "gate", 0, "DONE", True, True, False, 0, "", True, "sha2"),
        RunRecord("t3", "gate", 0, "FAILED", False, True, False, 0, "", False, "sha3"),
    ]
    stats = {s.condition: s for s in _aggregate(records)}
    g = stats["gate"]
    assert (g.tp, g.fp, g.tn, g.fn) == (1, 1, 0, 1)
    assert g.precision == 0.5 and g.recall == 0.5
    assert g.false_success_rate == 1 / 3  # the FP is a counted false success


def test_cache_busts_when_task_changes(tmp_path):
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    base = _base(tmp_path)
    run_ablation(base, [task], reps=1, out_dir=out, llm_client=_FixedLLM(2))
    assert (out / "results" / "task__r0.json").exists()

    # same cache dir, but the task definition changed -> fingerprint mismatch ->
    # the cell recomputes (and here the failing LLM makes that observable).
    Path(task).write_text(Path(task).read_text().replace("wrong value", "other value"))
    rep2 = run_ablation(base, [task], reps=1, out_dir=out, llm_client=_FailingLLM())
    assert all(r.status == "ERROR" for r in rep2.records)


def test_legacy_cache_format_is_recomputed(tmp_path):
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    (out / "results").mkdir(parents=True)
    # pre-fingerprint cache format: a bare list
    (out / "results" / "task__r0.json").write_text("[]")
    rep = run_ablation(_base(tmp_path), [task], reps=1, out_dir=out, llm_client=_FixedLLM(2))
    assert _by_cond(rep)["trust"].true_success  # recomputed, not served stale


def test_report_shows_gate_quality_and_scorer(tmp_path):
    report = _run(tmp_path, _FixedLLM(2))
    md = report.to_markdown()
    assert "final scorer" in md
    assert "precision" in md and "recall" in md and "FP=" in md
    assert report.scorer == "trusted-local"
    assert report.fingerprint
