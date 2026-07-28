"""Tests for the Phase-2 robustness/deployability hardening.

Covers the subsystems the audit flagged as zero-coverage single points of failure:
skill memory's gating, orchestrator spawn-failure isolation, eval per-case
isolation, the loop's mid-step error contract, and citation coverage of
experiment artifacts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import hermetic_task

from lha import eval as lha_eval
from lha import orchestrator
from lha.artifacts import ExperimentResult, Patch, Step
from lha.config import Config
from lha.harness import Harness
from lha.memory import SkillMemory
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


def test_docker_context_excludes_local_credentials_and_build_outputs():
    root = Path(__file__).resolve().parents[1]
    patterns = {
        line.strip()
        for line in (root / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {
        ".env",
        ".env.*",
        ".codex/",
        ".claude/",
        ".mcp.json",
        "auth.json",
        ".ssh/",
        ".aws/",
        ".config/gcloud/",
        ".netrc",
        ".pypirc",
        "dist/",
        "build/",
        ".coverage",
    } <= patterns
    assert "!.env.example" in patterns


def test_application_image_pins_and_bundles_the_context_model():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text()

    assert (
        "ghcr.io/astral-sh/uv:0.11.16"
        "@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d"
        in dockerfile
    )
    assert (
        "python:3.11-slim"
        "@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93"
        in dockerfile
    )
    assert (
        "EMBEDDER_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        in dockerfile
    )
    assert dockerfile.count("ARG EMBEDDER_REVISION") == 2
    assert dockerfile.count("ARG EMBEDDER_REPOSITORY") == 2
    assert "snapshot_download" in dockerfile
    assert "allow_patterns=" in dockerfile
    assert dockerfile.index("snapshot_download") < dockerfile.index("COPY . .")
    assert "COPY --from=builder --chown=lha:lha /opt/lha/models /opt/lha/models" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "COCOINDEX_DISABLE_USAGE_TRACKING=1" in dockerfile
    assert "LHA_EMBEDDER_MODEL=/opt/lha/models/all-MiniLM-L6-v2" in dockerfile


def test_cli_unexpected_error_has_a_stable_nonzero_exit_without_traceback(
    monkeypatch, capsys
):
    from lha import cli

    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["lha", "run", "/definitely/missing/task.yaml"],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()

    captured = capsys.readouterr()
    assert stopped.value.code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: FileNotFoundError:")
    assert "Traceback" not in captured.err


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
    # provenance: the note names the exact verdict bytes that justified it
    import hashlib

    expected = hashlib.sha256((rd / "verify.json").read_bytes()).hexdigest()
    assert f"verdict_sha256: {expected}" in body
    assert "harness_version:" in body


# --- orchestrator survives a per-task spawn failure -------------------------
def test_orchestrator_survives_spawn_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("EMFILE: too many open files")

    monkeypatch.setattr(orchestrator, "run_bounded_process", boom)
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
    result = Harness(_cfg(tmp_path)).run(hermetic_task("data/tasks/fix_average.yaml"))
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
    result = Harness(_cfg(tmp_path)).run(hermetic_task("data/tasks/fix_average.yaml"))
    assert result.status == "FAILED"
    # the in-flight patch must be reverted — the original bug is back in the sandbox.
    restored = (Path(result.state.run_dir) / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" in restored


# --- per-step artifacts preserve every step's provenance --------------------
def test_per_step_artifacts_written_and_finalizer_reads_the_edit_step(tmp_path):
    result = Harness(_cfg(tmp_path)).run(hermetic_task("data/tasks/fix_average.yaml"))
    assert result.status == "DONE"
    run_dir = Path(result.state.run_dir)
    # both steps keep their own verify.json; the edit step keeps its patch.json
    assert (run_dir / "steps" / "s1-context" / "verify.json").exists()
    assert (run_dir / "steps" / "s2-fix" / "patch.json").exists()
    assert (run_dir / "steps" / "s2-fix" / "verify.json").exists()
    # the PR summary reports the edit step's verdict (pytest/ruff), not the context step's
    pr = (run_dir / "pr_summary.md").read_text()
    assert "pytest" in pr


# --- per-step artifact paths can't escape the run dir (defense in depth) ----
def test_dump_keeps_artifacts_inside_the_run_dir(tmp_path):
    from lha.harness.loop import _dump

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="artifact identity"):
        _dump(run_dir, "../../escape", "verify.json", "{}")
    assert not (tmp_path / "escape").exists()
    assert not (run_dir / "steps").exists()


# --- a patch can never write outside the run sandbox ------------------------
def test_apply_patch_rejects_path_traversal(tmp_path):
    import pytest

    from lha.tools.patch import apply_patch

    workdir = tmp_path / "wd"
    workdir.mkdir()
    for bad_key in ("../escape.txt", "/tmp/lha_escape.txt", "a/../../escape.txt"):
        with pytest.raises(ValueError):
            apply_patch(Patch(step_id="s", file_contents={bad_key: "pwned"}), workdir)
    assert not (tmp_path / "escape.txt").exists()  # nothing landed outside the sandbox


# --- citations on an experiment artifact are now checked, not skipped -------
def test_citation_verifies_experiment_result(tmp_path):
    step = Step(
        step_id="s", kind="experiment", action="run_experiment", goal="g", verifiers=["citation"]
    )
    ctx = VerifyContext(workdir=tmp_path, step=step, bundle=None)  # no known locators

    unresolved = ExperimentResult(step_id="s", based_on_context=["paper:does-not-resolve"])
    assert not CitationVerifier().verify(unresolved, ctx).passed  # was silently passed before

    # zero citations: fails closed when the step requires context, passes only
    # under an explicit optional declaration
    clean = ExperimentResult(step_id="s", based_on_context=[])
    assert not CitationVerifier().verify(clean, ctx).passed
    opt_step = step.model_copy(update={"context_requirement": "optional"})
    opt_ctx = VerifyContext(workdir=tmp_path, step=opt_step, bundle=None)
    assert CitationVerifier().verify(clean, opt_ctx).passed
