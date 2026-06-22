"""Tests for the Phase-2 robustness/deployability hardening.

Covers the subsystems the audit flagged as zero-coverage single points of failure:
skill memory's gating, orchestrator spawn-failure isolation, eval per-case
isolation, the loop's mid-step error contract, and citation coverage of
experiment artifacts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lha import eval as lha_eval
from lha import orchestrator
from lha.artifacts import ExperimentResult, Patch, Step
from lha.config import Config
from lha.harness import Harness
from lha.memory import SkillMemory
from lha.tasks.spec import TaskSpec
from lha.verifiers import VerifyContext
from lha.verifiers.context import CitationVerifier
from lha.verifiers.verdict import Check, Verdict


def _cfg(tmp_path: Path, **over) -> Config:
    base = dict(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
    )
    base.update(over)
    return Config(**base)


# --- skill memory only records genuine, verified successes ------------------
def test_skillmemory_gates_on_verified_done(tmp_path):
    mem = SkillMemory(tmp_path / "skills")

    assert mem.record(SimpleNamespace(status="FAILED")) is None  # not DONE

    no_verify = tmp_path / "r1"
    no_verify.mkdir()
    assert mem.record(SimpleNamespace(status="DONE", run_dir=str(no_verify))) is None

    failed = tmp_path / "r2"
    failed.mkdir()
    (failed / "verify.json").write_text(Verdict(step_id="s", passed=False).model_dump_json())
    assert mem.record(SimpleNamespace(status="DONE", run_dir=str(failed))) is None


def test_skillmemory_writes_a_note_for_a_passed_run(tmp_path):
    rd = tmp_path / "r3"
    rd.mkdir()
    (rd / "verify.json").write_text(
        Verdict.from_checks(
            "s", [Check(name="pytest", family="code", passed=True, detail={"summary": "3 passed"})]
        ).model_dump_json()
    )
    (rd / "patch.json").write_text(
        Patch(step_id="s", rationale="fix the off-by-one", touched_files=["m.py"]).model_dump_json()
    )
    state = SimpleNamespace(
        status="DONE",
        run_dir=str(rd),
        run_id="run-x",
        task=SimpleNamespace(title="Fix avg", kind="issue_to_pr", description="desc"),
    )
    path = SkillMemory(tmp_path / "skills").record(state)
    assert path is not None and path.exists()
    body = path.read_text()
    assert "fix the off-by-one" in body and "pytest: 3 passed" in body and "m.py" in body


# --- orchestrator survives a per-task spawn failure -------------------------
def test_orchestrator_survives_spawn_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("EMFILE: too many open files")

    monkeypatch.setattr(orchestrator.subprocess, "run", boom)
    outs = orchestrator.run_tasks(["a.yaml", "b.yaml"], max_workers=2)
    assert len(outs) == 2  # one bad spawn did not discard the whole batch
    assert all(o.status == "ERROR" for o in outs)


# --- run_eval isolates a crashing case --------------------------------------
def _raising_case(base):
    raise RuntimeError("case crashed")


def test_eval_isolates_a_crashing_case(monkeypatch):
    monkeypatch.setattr(lha_eval, "_FAST", [_raising_case])
    report = lha_eval.run_eval(Config(), quick=True)
    assert len(report.results) == 1  # the crash became a failed result, not a lost report
    assert not report.results[0].passed
    assert "errored" in report.results[0].detail


# --- the loop fails closed on an unexpected mid-step fault -------------------
def test_loop_fails_closed_on_midstep_exception(tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("execute boom")

    monkeypatch.setattr(Harness, "_execute", boom)
    # run() must return a clean FAILED, not propagate the exception or wedge at RUNNING.
    result = Harness(_cfg(tmp_path)).run(TaskSpec.from_file("data/tasks/fix_average.yaml"))
    assert result.status == "FAILED"
    # the fault is ledgered as a fail (not silently dropped).
    ledger = (Path(result.state.run_dir) / "ledger.jsonl").read_text()
    assert '"phase":"fail"' in ledger and "execute boom" in ledger


def test_loop_reverts_applied_patch_on_midstep_fault(tmp_path, monkeypatch):
    from lha.agents import verifier_agent

    real = verifier_agent.VerifierAgent.verify

    def boom_on_edit(self, step, artifact, ctx):
        if step.action == "edit_code":  # fault AFTER the patch was applied
            raise RuntimeError("verify boom")
        return real(self, step, artifact, ctx)

    monkeypatch.setattr(verifier_agent.VerifierAgent, "verify", boom_on_edit)
    result = Harness(_cfg(tmp_path)).run(TaskSpec.from_file("data/tasks/fix_average.yaml"))
    assert result.status == "FAILED"
    # the in-flight patch must be reverted — the original bug is back in the sandbox.
    restored = (Path(result.state.run_dir) / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" in restored


# --- citations on an experiment artifact are now checked, not skipped -------
def test_citation_verifies_experiment_result(tmp_path):
    step = Step(
        step_id="s", kind="experiment", action="run_experiment", goal="g", verifiers=["citation"]
    )
    ctx = VerifyContext(workdir=tmp_path, step=step, bundle=None)  # no known locators

    unresolved = ExperimentResult(step_id="s", based_on_context=["paper:does-not-resolve"])
    assert not CitationVerifier().verify(unresolved, ctx).passed  # was silently passed before

    clean = ExperimentResult(step_id="s", based_on_context=[])
    assert CitationVerifier().verify(clean, ctx).passed
