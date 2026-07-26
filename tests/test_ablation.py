"""Deterministic tests for the paired verification ablation — no network.

The real ablation runs a live LLM, but every mechanism is pinned here with injected
fake backends: the paired trust/gate contrast, tamper-proof source-only patching,
the repair lift, transient-error handling, aggregation, and resumability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        self.calls = 0

    def complete(self, system: str, prompt: str) -> str:
        return ""

    def propose_patch(self, step, bundle, workdir) -> Patch:
        self.calls += 1
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


def test_error_records_are_never_reused_from_cache(tmp_path):
    import lha.ablation as abl

    cache = tmp_path / "cell.json"
    fingerprint = "f" * 64
    error = RunRecord("task", "trust", 0, "ERROR", False, False, False, 0)
    cache.write_text(
        json.dumps(
            {
                "schema_version": abl._CACHE_SCHEMA,
                "fingerprint": fingerprint,
                "records": [error.__dict__],
                "llm_calls": [],
            }
        )
    )
    assert abl._load_cached(cache, fingerprint) is None


def test_resumable_caches_real_outcomes(tmp_path):
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    base = _base(tmp_path)
    llm = _FixedLLM(2)
    rep1 = run_ablation(base, [task], reps=1, out_dir=out, llm_client=llm)
    assert (out / "results" / "task__r0.json").exists()
    assert rep1.llm_calls[0]["cache_hit"] is False
    calls_after_first_run = llm.calls

    # A cached cell must NOT re-invoke the LLM.
    rep2 = run_ablation(base, [task], reps=1, out_dir=out, llm_client=llm)
    assert _by_cond(rep2)["trust"].true_success  # served from cache, no LLM call
    assert llm.calls == calls_after_first_run
    assert rep2.llm_calls[0]["cache_hit"] is True


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


def test_boundary_rate_intervals_use_wilson_instead_of_collapsing():
    records = [
        RunRecord(f"t{i}", "verify", 0, "DONE", True, True, False, 0)
        for i in range(4)
    ]
    verify = {s.condition: s for s in _aggregate(records)}["verify"]
    assert verify.true_ci is not None and verify.true_ci[0] < 1.0
    assert verify.true_ci[1] == 1.0
    assert verify.false_ci is not None and verify.false_ci[0] == 0.0
    assert verify.false_ci[1] > 0.0


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


@pytest.mark.parametrize(
    "source_path",
    [
        "ablation.py",
        "verifiers/code/pytest_verifier.py",
        "tools/policy.py",
        "tools/patch.py",
        "sandbox/local.py",
        "bench/stats.py",
    ],
)
def test_cache_fingerprint_includes_all_outcome_source(tmp_path, source_path):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    source_files = abl._source_file_digests()
    assert source_path in source_files
    original = abl._fingerprint(
        task, src, "stub", None, source_files=source_files
    )
    changed_files = dict(source_files)
    changed_files[source_path] = "0" * 64
    changed = abl._fingerprint(
        task, src, "stub", None, source_files=changed_files
    )
    assert changed != original


def test_cache_fingerprint_binds_scorer_and_runtime(tmp_path):
    import lha.ablation as abl

    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    source_files = abl._source_file_digests()
    local = abl._fingerprint(
        task,
        src,
        "stub",
        None,
        "trusted-local",
        source_files=source_files,
        runtime={"scorer": {"actual": "trusted-local", "image_id": None}},
    )
    docker = abl._fingerprint(
        task,
        src,
        "stub",
        None,
        "docker",
        source_files=source_files,
        runtime={"scorer": {"actual": "docker", "image_id": "sha256:" + "a" * 64}},
    )
    assert docker != local


def test_legacy_cache_format_is_recomputed(tmp_path):
    import lha.ablation as abl

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    (out / "results").mkdir(parents=True)
    # pre-fingerprint cache format: a bare list
    (out / "results" / "task__r0.json").write_text("[]")
    assert abl._read_cache(out / "results" / "task__r0.json") == (None, [])
    rep = run_ablation(_base(tmp_path), [task], reps=1, out_dir=out, llm_client=_FixedLLM(2))
    assert _by_cond(rep)["trust"].true_success  # recomputed, not served stale


def test_report_shows_gate_quality_and_scorer(tmp_path):
    report = _run(tmp_path, _FixedLLM(2))
    md = report.to_markdown()
    assert "final scorer" in md
    assert "precision" in md and "recall" in md and "FP=" in md
    assert report.scorer == "trusted-local"
    assert report.fingerprint


def test_new_report_records_complete_secret_free_provenance(tmp_path):
    import lha.ablation as abl

    report = _run(tmp_path, _FixedLLM(2))
    raw = json.loads((tmp_path / "out" / "ablation_report.json").read_text())
    provenance = raw["provenance"]

    assert raw["schema_version"] == 2
    assert provenance["source_tree_sha256"] == report.provenance.source_tree_sha256
    assert provenance["source_files"]["ablation.py"]
    assert provenance["source_files"]["verifiers/code/pytest_verifier.py"]
    assert provenance["source_files"]["tools/policy.py"]
    assert provenance["source_files"]["tools/patch.py"]
    assert provenance["requested_llm_backend"] == "stub"
    assert provenance["actual_llm_backend"] == "base"
    assert provenance["model"] is None
    assert provenance["cli_version"] is None
    assert provenance["backend_library_version"] is None
    assert provenance["reasoning_effort"] is None
    assert provenance["scorer_requested"] == "trusted-local"
    assert provenance["scorer_backend"] == "trusted-local"
    assert provenance["task_files_sha256"]["task"]
    assert provenance["corpus_sha256"]["task"]
    assert provenance["task_paths"]["task"].endswith("task.yaml")
    assert provenance["corpus_paths"]["task"].endswith("src")
    assert provenance["configuration"]["repetitions"] == 1
    assert provenance["git_commit"] is None or len(provenance["git_commit"]) == 40
    assert provenance["git_dirty"] in (True, False, None)
    assert "auth" not in json.dumps(provenance).lower()
    assert raw["llm_calls"] == [
        {
            "task": "task",
            "rep": 0,
            "cache_hit": False,
            "label": "first",
            "status": "succeeded",
            "backend": "base",
        }
    ]

    loaded = abl.load_ablation_report(tmp_path / "out" / "ablation_report.json")
    assert loaded.schema_version == 2
    assert loaded.provenance is not None
    assert loaded.provenance.source_tree_sha256 == provenance["source_tree_sha256"]


def test_old_report_without_provenance_remains_readable(tmp_path):
    import lha.ablation as abl

    old = tmp_path / "old-report.json"
    old.write_text(
        json.dumps(
            {
                "llm": "claude_cli",
                "model": "old-model",
                "reps": 1,
                "tasks": ["old-task"],
                "scorer": "trusted-local",
                "records": [],
                "stats": [],
            }
        )
    )
    loaded = abl.load_ablation_report(old)
    assert loaded.schema_version == 1
    assert loaded.provenance is None
    assert loaded.tasks == ["old-task"]


def test_codex_runtime_and_call_audit_are_structured_and_secret_free():
    import lha.ablation as abl
    from lha.llm.codex_cli import CodexCLIClient

    client = CodexCLIClient(
        model="gpt-test-snapshot",
        reasoning_effort="high",
        no_tools=True,
    )
    client._version = "codex-cli 1.2.3"
    runtime = abl._client_runtime(
        "codex_cli",
        client,
        model="gpt-test-snapshot",
        cli_path="codex",
        backend_details="codex-cli 1.2.3",
    )
    assert runtime["actual_backend"] == "codex_cli"
    assert runtime["model"] == "gpt-test-snapshot"
    assert runtime["cli_version"] == "codex-cli 1.2.3"
    assert runtime["reasoning_effort"] == "high"

    client.last_call = {
        "status": "failed",
        "cli_version": "codex-cli 1.2.3",
        "model": "gpt-test-snapshot",
        "reasoning_effort": "high",
        "attempt_count": 1,
        "event_summary": {"total_events": 3, "events": {"turn.failed": 1}},
        "error_type": "CodexProtocolError",
        "error": "must-not-be-persisted",
        "attempts": [
            {
                "attempt": 1,
                "status": "failed",
                "event_summary": {"total_events": 3},
                "error": "must-not-be-persisted",
            }
        ],
    }
    client.last_usage = {
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "output_tokens": 3,
        "cost_usd": None,
        "model": "gpt-test-snapshot",
    }
    audit = abl._safe_call_audit(
        client,
        label="first",
        status="failed",
        error=RuntimeError("must-not-be-persisted"),
    )
    assert audit["event_summary"]["total_events"] == 3
    assert audit["status"] == "failed"
    assert audit["usage"]["output_tokens"] == 3
    assert "must-not-be-persisted" not in json.dumps(audit)
