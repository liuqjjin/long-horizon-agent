"""Deterministic tests for the paired verification ablation — no network.

The real ablation runs a live LLM, but every mechanism is pinned here with injected
fake backends: the paired trust/gate contrast, tamper-proof source-only patching,
the repair lift, transient-error handling, aggregation, and resumability.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lha.ablation import (
    CONDITIONS,
    AblationReport,
    ConditionStats,
    PytestResult,
    RunRecord,
    ScoreOutcome,
    _aggregate,
    _classify_scorer_receipt,
    _sanitize,
    _score,
    run_ablation,
)
from lha.artifacts import Patch
from lha.config import Config
from lha.llm.base import LLMClient
from lha.llm.claude_cli import ClaudeCLIClient
from lha.llm.trace import TracedLLM
from lha.sandbox import TrustedLocalBackend

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


def test_programmatic_ablation_default_matches_cli_backend():
    assert run_ablation.__kwdefaults__["llm"] == "codex_cli"


def test_experimental_claude_cli_cannot_produce_ablation_evidence(tmp_path):
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="experimental"):
        run_ablation(
            _base(tmp_path),
            [task],
            llm="claude_cli",
            reps=1,
            out_dir=out,
            llm_client=_FixedLLM(2),
        )

    assert not out.exists()


@pytest.mark.parametrize("wrapped", [False, True])
def test_injected_claude_client_cannot_bypass_ablation_gate(
    wrapped: bool,
    tmp_path,
):
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    out = tmp_path / "out"
    client = ClaudeCLIClient(cli_path="must-not-run")
    injected = TracedLLM(client) if wrapped else client

    with pytest.raises(ValueError, match="experimental"):
        run_ablation(
            _base(tmp_path),
            [task],
            llm="stub",
            reps=1,
            out_dir=out,
            llm_client=injected,
        )

    assert not out.exists()


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


def test_independent_scorer_ignores_candidate_writable_json_report(tmp_path):
    """Candidate code may create the old report path, but scorer truth ignores it."""
    src = _repo(tmp_path / "src")
    forged_report = json.dumps(
        {
            "summary": {"passed": 1, "failed": 0, "error": 0, "total": 1},
            "tests": [{"nodeid": "forged", "outcome": "passed"}],
        }
    )
    frozen = {
        "m.py": (
            "from pathlib import Path\n"
            f"Path('.lha_pytest.json').write_text({forged_report!r})\n"
            "def f():\n"
            "    return 3\n"
        ),
    }

    result = _score(
        src,
        frozen,
        tmp_path / "scratch",
        "shadow",
        TrustedLocalBackend(),
    )
    assert result.outcome is ScoreOutcome.TEST_FAIL
    assert not result.passed


def test_candidate_cannot_forge_pass_by_printing_summary_and_exiting_zero(tmp_path):
    """Exercise the real interpreter path, not a fake ExecutionBackend."""
    src = _repo(tmp_path / "src")
    frozen = {
        "m.py": (
            'print("1 passed in 0.01s", flush=True)\n'
            "import os\n"
            "os._exit(0)\n"
        )
    }

    result = _score(
        src,
        frozen,
        tmp_path / "scratch",
        "early-exit",
        TrustedLocalBackend(),
    )

    assert result.outcome is ScoreOutcome.INFRA_ERROR
    assert "receipt" in result.detail


def test_independent_scorer_rejects_protected_frozen_paths(tmp_path):
    src = _repo(tmp_path / "src")
    result = _score(
        src,
        {"pytest.py": "raise SystemExit(0)\n"},
        tmp_path / "scratch",
        "protected",
        TrustedLocalBackend(),
    )
    assert result.outcome is ScoreOutcome.INFRA_ERROR
    assert result.detail == "scorer setup failed: ValueError"


def test_candidate_syntax_error_is_a_test_failure_not_infrastructure(tmp_path):
    src = _repo(tmp_path / "src")
    result = _score(
        src,
        {"m.py": "def f(:\n    return 2\n"},
        tmp_path / "scratch",
        "syntax",
        TrustedLocalBackend(),
    )
    assert result.outcome is ScoreOutcome.TEST_FAIL


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
    error = RunRecord("task", "trust", 0, "ERROR", False, False, False, False, 0)
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


def test_cache_reader_rejects_oversized_file(tmp_path, monkeypatch):
    import lha.ablation as abl

    monkeypatch.setattr(abl, "_MAX_CACHE_BYTES", 128)
    cache = tmp_path / "cell.json"
    cache.write_bytes(b"x" * 129)

    assert abl._read_cache(cache) is None


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_bounded_evidence_reader_rejects_links(tmp_path, link_kind):
    import lha.ablation as abl

    backing = tmp_path / "backing.json"
    backing.write_text("{}")
    evidence = tmp_path / "evidence.json"
    if link_kind == "symlink":
        evidence.symlink_to(backing)
    else:
        os.link(backing, evidence)

    with pytest.raises(ValueError, match="regular file|hard links"):
        abl._read_bounded_bytes(evidence, max_bytes=1024)


def test_bounded_evidence_reader_rejects_file_changed_during_read(
    tmp_path,
    monkeypatch,
):
    import lha.ablation as abl

    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(b"{}")
    real_read = abl.os.read
    changed = False

    def mutate_after_read(descriptor, size):
        nonlocal changed
        chunk = real_read(descriptor, size)
        if chunk and not changed:
            changed = True
            with evidence.open("ab") as stream:
                stream.write(b" ")
        return chunk

    monkeypatch.setattr(abl.os, "read", mutate_after_read)

    with pytest.raises(ValueError, match="changed while"):
        abl._read_bounded_bytes(evidence, max_bytes=1024)


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


def test_cache_is_rejected_when_frozen_artifact_bytes_are_damaged(tmp_path):
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    base = _base(tmp_path)
    llm = _FixedLLM(2)
    first = run_ablation(base, [task], reps=1, out_dir=out, llm_client=llm)
    digest = _by_cond(first)["trust"].artifact_sha256
    artifact = out / "artifacts" / f"{digest}.json"
    artifact.write_text("{}")
    calls = llm.calls
    llm.value = 3

    second = run_ablation(base, [task], reps=1, out_dir=out, llm_client=llm)

    assert llm.calls > calls
    assert _by_cond(second)["trust"].false_success


def test_cache_rejects_valid_receipt_swapped_from_another_artifact(tmp_path):
    class DifferentPassingLLM(_FixedLLM):
        def propose_patch(self, step, bundle, workdir):
            self.calls += 1
            expression = "2" if self.calls % 2 else "1 + 1"
            return Patch(
                step_id=step.step_id,
                file_contents={"m.py": f"def f():\n    return {expression}\n"},
                touched_files=["m.py"],
            )

    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)
    llm = DifferentPassingLLM(2)
    first = run_ablation(
        _base(tmp_path),
        [task],
        reps=2,
        out_dir=out,
        llm_client=llm,
    )
    rep_artifacts = {
        rep: next(
            record.artifact_sha256
            for record in first.records
            if record.rep == rep and record.condition == "trust"
        )
        for rep in (0, 1)
    }
    assert rep_artifacts[0] != rep_artifacts[1]

    first_cache = out / "results" / "task__r0.json"
    second_cache = out / "results" / "task__r1.json"
    first_raw = json.loads(first_cache.read_text())
    second_raw = json.loads(second_cache.read_text())
    donor_digest = second_raw["records"][0]["scorer_evidence_sha256"]
    for record in first_raw["records"]:
        record["scorer_evidence_sha256"] = donor_digest
    first_cache.write_text(json.dumps(first_raw))
    calls = llm.calls

    run_ablation(
        _base(tmp_path),
        [task],
        reps=2,
        out_dir=out,
        llm_client=llm,
    )

    assert llm.calls == calls + 1


def test_live_corpus_change_then_restore_cannot_change_frozen_run_inputs(tmp_path):
    src = _repo(tmp_path / "src")
    original_test = (src / "tests" / "test_m.py").read_text()

    class _MutatingLLM(_FixedLLM):
        def propose_patch(self, step, bundle, workdir):
            if self.calls == 0:
                (src / "tests" / "test_m.py").write_text(original_test.replace("== 2", "== 3"))
            else:
                (src / "tests" / "test_m.py").write_text(original_test)
            return super().propose_patch(step, bundle, workdir)

    report = run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=2,
        out_dir=tmp_path / "out",
        llm_client=_MutatingLLM(2),
    )

    assert all(record.status != "ERROR" for record in report.records)
    assert all(record.artifact_correct for record in report.records)
    assert (src / "tests" / "test_m.py").read_text() == original_test


def test_cell_infrastructure_error_does_not_stop_later_cells(tmp_path, monkeypatch):
    import lha.ablation as abl

    original_evaluate = abl._evaluate
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated artifact store failure")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(abl, "_evaluate", fail_once)
    src = _repo(tmp_path / "src")
    out = tmp_path / "out"
    report = run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=2,
        out_dir=out,
        llm_client=_FixedLLM(2),
    )

    first = [record for record in report.records if record.rep == 0]
    second = [record for record in report.records if record.rep == 1]
    assert all(record.status == "ERROR" for record in first)
    assert all("infrastructure failure" in record.detail for record in first)
    assert all(record.status != "ERROR" for record in second)
    assert not (out / "results" / "task__r0.json").exists()
    assert (out / "results" / "task__r1.json").exists()


# --- aggregation + report ---------------------------------------------------
def test_aggregate_and_markdown():
    records = [
        RunRecord("t1", "trust", 0, "DONE", True, False, False, True, 0),
        RunRecord("t1", "gate", 0, "FAILED", False, False, False, False, 0),
        RunRecord("t1", "verify", 0, "DONE", True, True, True, False, 1),
        RunRecord("t2", "trust", 0, "DONE", True, True, True, False, 0),
        RunRecord("t2", "gate", 0, "DONE", True, True, True, False, 0),
        RunRecord("t2", "verify", 0, "DONE", True, True, True, False, 0),
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
        RunRecord("t1", "trust", 0, "ERROR", False, False, False, False, 0),
        RunRecord("t2", "trust", 0, "DONE", True, False, False, True, 0),
    ]
    stats = {s.condition: s for s in _aggregate(records)}
    assert stats["trust"].n == 1 and stats["trust"].false_success_rate == 1.0
    assert isinstance(stats["trust"], ConditionStats)
    assert CONDITIONS[0][0] == "trust"


def test_boundary_rate_intervals_use_wilson_instead_of_collapsing():
    records = [
        RunRecord(f"t{i}", "verify", 0, "DONE", True, True, True, False, 0)
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
            if input is not None:
                config = json.loads(input)
                nodeid = "tests/test_m.py::test_f"
                is_collect = config["mode"] == "collect"
                receipt = {
                    "schema_version": 1,
                    "nonce": config["nonce"],
                    "mode": config["mode"],
                    "pytest_exit_code": 0 if is_collect else 1,
                    "collected": [nodeid],
                    "collection_failures": 0,
                    "reports": (
                        []
                        if is_collect
                        else [
                            {
                                "nodeid": nodeid,
                                "when": "call",
                                "outcome": "failed",
                                "wasxfail": False,
                            }
                        ]
                    ),
                }
                payload = json.dumps(
                    receipt,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
                digest = hashlib.sha256(payload).hexdigest()
                (Path(cwd) / config["report_name"]).write_bytes(payload)
                marker = f"LHA_SCORER_RECEIPT {config['nonce']} {digest}\n"
                return ProcResult(receipt["pytest_exit_code"], marker, "", 0.0)
            report = Path(cwd) / ".lha_pytest.json"
            report.write_text(
                json.dumps(
                    {
                        "summary": {"passed": 0, "failed": 1, "error": 0, "total": 1},
                        "tests": [
                            {
                                "nodeid": "tests/test_m.py::test_m",
                                "outcome": "failed",
                                "call": {"longrepr": "E assert False"},
                            }
                        ],
                    }
                )
            )
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
    assert rec["gate"].artifact_correct is True
    assert rec["gate"].true_success is False
    assert rec["gate"].false_success is False
    stats = {s.condition: s for s in report.stats}
    assert stats["gate"].fn == 1 and stats["gate"].recall == 0.0
    assert stats["gate"].artifact_correct_rate == 1.0
    assert stats["gate"].true_success_rate == 0.0


def test_scorer_infrastructure_failure_is_not_a_wrong_patch(tmp_path, monkeypatch):
    import lha.ablation as abl

    monkeypatch.setattr(
        abl,
        "_score",
        lambda *args, **kwargs: PytestResult(
            ScoreOutcome.INFRA_ERROR,
            127,
            "scorer: Pytest infrastructure exit 127",
        ),
    )
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    report = run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=1,
        out_dir=out,
        llm_client=_FixedLLM(2),
    )

    assert all(record.status == "ERROR" for record in report.records)
    assert all(record.scorer_outcome == "INFRA_ERROR" for record in report.records)
    assert all(not record.true_success and not record.false_success for record in report.records)
    assert all(stat.n == 0 and stat.errors == 1 for stat in report.stats)
    assert not (out / "results" / "task__r0.json").exists()


@pytest.mark.parametrize(
    ("returncode", "call_outcome", "expected"),
    [
        (0, "passed", ScoreOutcome.PASS),
        (1, "failed", ScoreOutcome.TEST_FAIL),
        (2, "passed", ScoreOutcome.INFRA_ERROR),
        (3, "passed", ScoreOutcome.INFRA_ERROR),
        (5, "passed", ScoreOutcome.INFRA_ERROR),
        (124, "passed", ScoreOutcome.INFRA_ERROR),
        (127, "passed", ScoreOutcome.INFRA_ERROR),
    ],
)
def test_control_plane_scorer_classifies_cross_checked_receipt(
    returncode,
    call_outcome,
    expected,
):
    nodeid = "tests/test_m.py::test_f"
    receipt = {
        "schema_version": 1,
        "nonce": "n" * 48,
        "mode": "run",
        "pytest_exit_code": returncode,
        "collected": [nodeid],
        "collection_failures": 0,
        "reports": [
            {
                "nodeid": nodeid,
                "when": "call",
                "outcome": call_outcome,
                "wasxfail": False,
            }
        ],
    }
    outcome, _passed = _classify_scorer_receipt(
        process_returncode=returncode,
        receipt=receipt,
        expected_nodeids=(nodeid,),
    )
    assert outcome is expected


def test_docker_scorer_also_runs_internal_gate_in_docker(tmp_path, monkeypatch):
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    instances = []
    image_id = "sha256:" + "a" * 64
    requested_images = []

    class _RecordingDocker:
        name = "docker"

        def __init__(self, image):
            self.image = image
            self.calls = []
            instances.append(self)

        def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
            self.calls.append(list(cmd))
            if "--json-report" in cmd:
                (Path(cwd) / ".lha_pytest.json").write_text(
                    json.dumps(
                        {
                            "summary": {
                                "passed": 1,
                                "failed": 0,
                                "error": 0,
                                "total": 1,
                            },
                            "tests": [
                                {
                                    "nodeid": "tests/test_m.py::test_m",
                                    "outcome": "passed",
                                }
                            ],
                        }
                    )
                )
                return ProcResult(0, "1 passed in 0.01s\n", "", 0.01)
            config = json.loads(input)
            nodeid = "tests/test_m.py::test_f"
            reports = (
                []
                if config["mode"] == "collect"
                else [
                    {
                        "nodeid": nodeid,
                        "when": "call",
                        "outcome": "passed",
                        "wasxfail": False,
                    }
                ]
            )
            receipt = {
                "schema_version": 1,
                "nonce": config["nonce"],
                "mode": config["mode"],
                "pytest_exit_code": 0,
                "collected": [nodeid],
                "collection_failures": 0,
                "reports": reports,
            }
            payload = json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            digest = hashlib.sha256(payload).hexdigest()
            (Path(cwd) / config["report_name"]).write_bytes(payload)
            marker = f"LHA_SCORER_RECEIPT {config['nonce']} {digest}\n"
            return ProcResult(0, marker, "", 0.01)

        def python(self):
            return "python"

        def tool(self, name):
            return name

    def fake_backend(name, **kwargs):
        assert name == "docker"
        assert kwargs == {"image": image_id}
        return _RecordingDocker(kwargs["image"])

    monkeypatch.setattr(abl, "make_backend", fake_backend)
    monkeypatch.setattr(
        abl,
        "_resolve_docker_image_id",
        lambda image: requested_images.append(image) or image_id,
    )
    monkeypatch.setattr(
        abl,
        "TrustedLocalBackend",
        lambda: pytest.fail("Docker ablation must not construct a host gate"),
    )
    src = _repo(tmp_path / "src")
    report = run_ablation(
        _base(tmp_path),
        [_task(tmp_path, src)],
        llm="stub",
        reps=1,
        out_dir=tmp_path / "out",
        llm_client=_FixedLLM(2),
        scorer_backend="docker",
    )

    assert len(instances) == 2
    assert requested_images == [_base(tmp_path).exec_image]
    assert all(instance.image == image_id for instance in instances)
    assert sum("-I" in command and "-c" in command for command in instances[0].calls) >= 2
    assert all("--json-report" not in command for command in instances[0].calls)
    assert all("--json-report" not in command for command in instances[1].calls)
    assert report.provenance is not None
    assert report.provenance.agent_backend == "docker"
    assert report.provenance.scorer_backend == "docker"
    assert report.provenance.scorer_image == _base(tmp_path).exec_image
    assert report.provenance.scorer_image_id == image_id
    raw = json.loads((tmp_path / "out" / "ablation_report.json").read_text())
    for record in raw["records"]:
        evidence_path = (
            tmp_path
            / "out"
            / "scorer_evidence"
            / f"{record['scorer_evidence_sha256']}.json"
        )
        evidence = json.loads(evidence_path.read_text())
        assert evidence["binding"]["scorer_image_id"] == image_id


def test_docker_image_resolution_failure_precedes_any_model_call(tmp_path, monkeypatch):
    import lha.ablation as abl

    llm = _FixedLLM(2)
    monkeypatch.setattr(
        abl,
        "_resolve_docker_image_id",
        lambda _image: (_ for _ in ()).throw(RuntimeError("invalid image binding")),
    )

    with pytest.raises(RuntimeError, match="invalid image binding"):
        run_ablation(
            _base(tmp_path),
            [_task(tmp_path, _repo(tmp_path / "src"))],
            llm="stub",
            reps=1,
            out_dir=tmp_path / "out",
            llm_client=llm,
            scorer_backend="docker",
        )

    assert llm.calls == 0


def test_docker_image_resolution_keeps_only_required_host_configuration(
    monkeypatch,
):
    import lha.ablation as abl
    from lha.tools.shell import ProcResult

    image_id = "sha256:" + "b" * 64
    observed = {}
    monkeypatch.setenv("DOCKER_CONFIG", "/tmp/lha-test-docker-config")
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/lha-test-docker.sock")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-docker")

    def recording_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed.update(kwargs)
        return ProcResult(0, image_id + "\n", "", 0.0)

    monkeypatch.setattr(abl, "run", recording_run)

    assert abl._resolve_docker_image_id("lha:test") == image_id
    assert observed["env"]["DOCKER_CONFIG"] == "/tmp/lha-test-docker-config"
    assert observed["env"]["DOCKER_HOST"] == "unix:///tmp/lha-test-docker.sock"
    assert "OPENAI_API_KEY" not in observed["env"]


def test_pinned_docker_backend_argv_uses_image_id_not_mutable_tag(tmp_path):
    from lha.sandbox import DockerBackend

    image_id = "sha256:" + "b" * 64
    mutable_tag = "lha:release"
    backend = DockerBackend(image=image_id)
    argv = backend.build_argv(
        ["python", "-V"],
        cwd=tmp_path,
        name="lha-test",
    )

    assert image_id in argv
    assert mutable_tag not in argv


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
        RunRecord(
            "t1", "gate", 0, "DONE", True, False, False, True, 0, "", True, "sha1"
        ),
        RunRecord(
            "t2", "gate", 0, "DONE", True, True, True, False, 0, "", True, "sha2"
        ),
        RunRecord(
            "t3", "gate", 0, "FAILED", False, True, False, False, 0, "", False, "sha3"
        ),
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


def test_report_rejects_source_tree_drift_during_run(tmp_path, monkeypatch):
    import lha.ablation as abl

    initial = abl._source_file_digests()
    calls = 0

    def source_file_digests():
        nonlocal calls
        calls += 1
        if calls == 1:
            return initial
        changed = dict(initial)
        changed["ablation.py"] = "0" * 64
        return changed

    monkeypatch.setattr(abl, "_source_file_digests", source_file_digests)
    out = tmp_path / "out"
    src = _repo(tmp_path / "src")
    task = _task(tmp_path, src)

    with pytest.raises(RuntimeError, match="source tree changed during the ablation"):
        run_ablation(
            _base(tmp_path),
            [task],
            reps=1,
            out_dir=out,
            llm_client=_FixedLLM(2),
        )

    assert calls == 2
    assert not (out / "ablation_report.json").exists()
    assert not (out / "ablation_report.md").exists()


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

    assert raw["schema_version"] == 4
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
    assert provenance["input_snapshot_sha256"]["task"]
    assert provenance["task_paths"]["task"].endswith("task.yaml")
    assert provenance["corpus_paths"]["task"].endswith("src")
    assert provenance["configuration"]["repetitions"] == 1
    assert provenance["git_commit"] is None or len(provenance["git_commit"]) == 40
    assert provenance["git_dirty"] in (True, False, None)
    assert "auth" not in json.dumps(provenance).lower()
    assert raw["artifact_store"]["path"] == "artifacts"
    assert raw["scorer_evidence_store"]["path"] == "scorer_evidence"
    assert raw["scorer_evidence_store"]["schema_version"] == 2
    assert provenance["configuration"]["cache_schema"] == 7
    assert provenance["configuration"]["scorer_evidence_schema"] == 2
    digests = {record["artifact_sha256"] for record in raw["records"]}
    assert raw["artifact_store"]["count"] == len(digests)
    for digest in digests:
        artifact = tmp_path / "out" / "artifacts" / f"{digest}.json"
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    evidence_digests = {
        record["scorer_evidence_sha256"] for record in raw["records"]
    }
    assert raw["scorer_evidence_store"]["count"] == len(evidence_digests)
    for digest in evidence_digests:
        evidence = tmp_path / "out" / "scorer_evidence" / f"{digest}.json"
        assert evidence.is_file()
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == digest
        envelope = json.loads(evidence.read_text())
        assert envelope["schema_version"] == 2
        assert envelope["pytest_evidence"]["schema_version"] == 1
        assert envelope["binding"]["task"] == "task"
        assert envelope["binding"]["rep"] == 0
        assert envelope["binding"]["input_snapshot_sha256"] == provenance[
            "input_snapshot_sha256"
        ]["task"]
        assert envelope["binding"]["scorer_backend"] == "trusted-local"
        assert envelope["binding"]["scorer_image_id"] is None
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
    assert loaded.schema_version == 4
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
