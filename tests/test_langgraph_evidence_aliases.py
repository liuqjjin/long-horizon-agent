"""LangGraph grades immutable attempt evidence, never display aliases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lha.agents.experimenter import ExperimentEvidence, ExperimentIntent
from lha.artifacts import ExperimentResult, Plan, Step
from lha.clock import now
from lha.config import Config
from lha.harness import Harness
from lha.harness.checkpoint import append_ledger
from lha.harness.errors import CheckpointCorrupt
from lha.harness.manifest import sha256_bytes
from lha.harness.state import RunState, StepRecord
from lha.live_context.models import ContextBundle, Freshness
from lha.repo_adapter import (
    RepoCommandResult,
    RepoIntegrityResult,
    RepoStageEvidence,
    RepoStageIntent,
    RepoStageResult,
    repository_tree_sha256,
)
from lha.tasks.spec import TaskSpec

pytest.importorskip("langgraph")


def _envelope(model) -> bytes:
    payload = model.model_dump(mode="json")
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return json.dumps(
        {
            "schema_version": 1,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "payload": payload,
        },
        indent=2,
    ).encode()


def _state(tmp_path: Path, step: Step) -> RunState:
    run_dir = tmp_path / "runs" / step.step_id
    workdir = run_dir / "workdir"
    workdir.mkdir(parents=True)
    config = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
    )
    state = RunState.new(
        TaskSpec(
            kind="paper_to_experiment",
            title="evidence test",
            description="evidence test",
            context_requirement="optional",
        ),
        step.step_id,
        str(run_dir),
        str(workdir),
        config=config,
        runtime="langgraph",
    )
    state.plan = Plan(task_id="test", summary="test", steps=[step])
    attempt_id = state.attempt_id(step)
    attempt_dir = (
        run_dir / "steps" / step.step_id / "attempts" / attempt_id
    )
    attempt_dir.mkdir(parents=True)
    bundle = ContextBundle(
        query="test",
        freshness=Freshness(index_version="v1", indexed_at=now()),
        status="empty",
    )
    context = bundle.model_dump_json(indent=2).encode()
    context_ref = f"attempts/{attempt_id}/context_bundle.json"
    (attempt_dir / "context_bundle.json").write_bytes(context)
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id=step.step_id,
            phase="context",
            artifact_ref=context_ref,
            evidence_sha256=sha256_bytes(context),
            attempt_id=attempt_id,
            idempotency_key=f"{attempt_id}:context",
        ),
    )
    return state


def test_langgraph_experiment_alias_cannot_replace_scored_result(
    tmp_path: Path,
) -> None:
    from lha.runtime.langgraph_runner import _load_step_artifacts

    step = Step(
        step_id="experiment",
        kind="experiment",
        action="run_experiment",
        goal="run",
        context_requirement="optional",
    )
    state = _state(tmp_path, step)
    attempt_id = state.attempt_id(step)
    result = ExperimentResult(
        step_id=step.step_id,
        command=["python", "experiment.py"],
        returncode=1,
        stdout_tail="real failure",
    )
    evidence = ExperimentEvidence(
        intent=ExperimentIntent(
            step_id=step.step_id,
            attempt_id=attempt_id,
            command=tuple(result.command),
            params_sha256="a" * 64,
            context_sha256="b" * 64,
        ),
        result=result,
    )
    data = _envelope(evidence)
    evidence_ref = f"attempts/{attempt_id}/experiment_evidence.json"
    evidence_path = (
        Path(state.run_dir) / "steps" / step.step_id / evidence_ref
    )
    evidence_path.write_bytes(data)
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id=step.step_id,
            phase="execute",
            artifact_ref=evidence_ref,
            evidence_sha256=sha256_bytes(data),
            attempt_id=attempt_id,
            idempotency_key=f"{attempt_id}:execute",
        ),
    )
    alias = Path(state.run_dir) / "steps" / step.step_id / "experiment.json"
    alias.write_text(
        ExperimentResult(
            step_id=step.step_id,
            command=result.command,
            returncode=0,
            stdout_tail="forged pass",
        ).model_dump_json()
    )

    loaded, _bundle = _load_step_artifacts(state, step)
    assert loaded == result
    assert loaded.returncode == 1


def test_langgraph_repo_stage_alias_cannot_replace_scored_result(
    tmp_path: Path,
) -> None:
    from lha.runtime.langgraph_runner import _load_step_artifacts

    step = Step(
        step_id="stage",
        kind="code",
        action="repo_stage",
        goal="stage",
        context_requirement="optional",
    )
    state = _state(tmp_path, step)
    attempt_id = state.attempt_id(step)
    result = RepoStageResult(stage="setup", status="not_configured")
    evidence = RepoStageEvidence(
        intent=RepoStageIntent(
            step_id=step.step_id,
            attempt_id=attempt_id,
            stage="setup",
            spec_sha256="c" * 64,
        ),
        worktree_sha256=repository_tree_sha256(state.workdir),
        result=result,
    )
    data = _envelope(evidence)
    evidence_ref = f"attempts/{attempt_id}/repo_stage_evidence.json"
    evidence_path = (
        Path(state.run_dir) / "steps" / step.step_id / evidence_ref
    )
    evidence_path.write_bytes(data)
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id=step.step_id,
            phase="execute",
            artifact_ref=evidence_ref,
            evidence_sha256=sha256_bytes(data),
            attempt_id=attempt_id,
            idempotency_key=f"{attempt_id}:execute",
        ),
    )
    alias = Path(state.run_dir) / "steps" / step.step_id / "repo_stage.json"
    alias.write_text(
        RepoStageResult(
            stage="setup",
            status="failed",
            commands=(
                RepoCommandResult(
                    command_id="forged",
                    stage="setup",
                    argv=("false",),
                    cwd=".",
                    expected_returncodes=frozenset({0}),
                    returncode=1,
                    stdout="",
                    stderr="",
                    duration_s=0,
                    passed=False,
                ),
            ),
        ).model_dump_json()
    )

    loaded, _bundle = _load_step_artifacts(state, step)
    assert loaded == result
    assert loaded.status == "not_configured"


def test_langgraph_repo_integrity_alias_cannot_replace_scored_result(
    tmp_path: Path,
) -> None:
    from lha.runtime.langgraph_runner import _load_step_artifacts

    step = Step(
        step_id="integrity",
        kind="code",
        action="repo_integrity",
        goal="integrity",
        context_requirement="optional",
    )
    state = _state(tmp_path, step)
    attempt_id = state.attempt_id(step)
    result = RepoIntegrityResult(
        task_id="fixture",
        expected_repo_sha256="a" * 64,
        actual_repo_sha256="a" * 64,
        expected_task_sha256="b" * 64,
        actual_task_sha256="b" * 64,
        expected_adapter_sha256="c" * 64,
        actual_adapter_sha256="c" * 64,
        expected_reference_patch_sha256="d" * 64,
        actual_reference_patch_sha256="d" * 64,
        expected_reference_touched_files=(),
        actual_reference_touched_files=(),
        oracle_files=(),
        passed=True,
    )
    data = result.model_dump_json(indent=2).encode()
    evidence_ref = f"attempts/{attempt_id}/repo_integrity.json"
    evidence_path = (
        Path(state.run_dir) / "steps" / step.step_id / evidence_ref
    )
    evidence_path.write_bytes(data)
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id=step.step_id,
            phase="execute",
            artifact_ref=evidence_ref,
            evidence_sha256=sha256_bytes(data),
            attempt_id=attempt_id,
            idempotency_key=f"{attempt_id}:execute",
        ),
    )
    alias = Path(state.run_dir) / "steps" / step.step_id / "repo_integrity.json"
    alias.write_text("{\"forged\":true}")

    loaded, _bundle = _load_step_artifacts(state, step)
    assert loaded == result
    assert loaded.passed

    evidence_path.write_text("{\"changed\":true}")
    with pytest.raises(CheckpointCorrupt, match="immutable execute evidence changed"):
        _load_step_artifacts(state, step)


def test_loop_repo_integrity_resume_uses_the_bound_attempt_result(
    tmp_path: Path,
) -> None:
    step = Step(
        step_id="loop-integrity",
        kind="code",
        action="repo_integrity",
        goal="integrity",
        context_requirement="optional",
    )
    state = _state(tmp_path, step)
    attempt_id = state.attempt_id(step)
    result = RepoIntegrityResult(
        task_id="fixture",
        expected_repo_sha256="a" * 64,
        actual_repo_sha256=None,
        expected_task_sha256="b" * 64,
        actual_task_sha256=None,
        expected_adapter_sha256="c" * 64,
        actual_adapter_sha256=None,
        expected_reference_patch_sha256="d" * 64,
        actual_reference_patch_sha256=None,
        expected_reference_touched_files=(),
        actual_reference_touched_files=(),
        oracle_files=(),
        issues=("fixed input was unavailable",),
        passed=False,
    )
    data = result.model_dump_json(indent=2).encode()
    evidence_ref = f"attempts/{attempt_id}/repo_integrity.json"
    evidence_path = (
        Path(state.run_dir) / "steps" / step.step_id / evidence_ref
    )
    evidence_path.write_bytes(data)
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id=step.step_id,
            phase="execute",
            artifact_ref=evidence_ref,
            evidence_sha256=sha256_bytes(data),
            attempt_id=attempt_id,
            idempotency_key=f"{attempt_id}:execute",
        ),
    )

    loaded = Harness._committed_repo_integrity(
        state,
        step,
        attempt_id,
    )

    assert loaded == result
    assert not loaded.passed


def test_langgraph_repo_stage_replay_rejects_changed_worktree(
    tmp_path: Path,
) -> None:
    from lha.runtime.langgraph_runner import _load_step_artifacts

    step = Step(
        step_id="stage-drift",
        kind="code",
        action="repo_stage",
        goal="stage",
        context_requirement="optional",
    )
    state = _state(tmp_path, step)
    attempt_id = state.attempt_id(step)
    evidence = RepoStageEvidence(
        intent=RepoStageIntent(
            step_id=step.step_id,
            attempt_id=attempt_id,
            stage="setup",
            spec_sha256="c" * 64,
        ),
        worktree_sha256=repository_tree_sha256(state.workdir),
        result=RepoStageResult(stage="setup", status="not_configured"),
    )
    data = _envelope(evidence)
    evidence_ref = f"attempts/{attempt_id}/repo_stage_evidence.json"
    evidence_path = (
        Path(state.run_dir) / "steps" / step.step_id / evidence_ref
    )
    evidence_path.write_bytes(data)
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id=step.step_id,
            phase="execute",
            artifact_ref=evidence_ref,
            evidence_sha256=sha256_bytes(data),
            attempt_id=attempt_id,
            idempotency_key=f"{attempt_id}:execute",
        ),
    )
    (Path(state.workdir) / "drift.txt").write_text("changed after stage")

    with pytest.raises(CheckpointCorrupt, match="worktree changed after stage"):
        _load_step_artifacts(state, step)


@pytest.mark.parametrize("action", ["run_experiment", "repo_stage"])
def test_langgraph_rejects_immutable_evidence_changed_after_execute(
    action: str,
    tmp_path: Path,
) -> None:
    from lha.runtime.langgraph_runner import _load_step_artifacts

    step = Step(
        step_id="changed",
        kind="experiment" if action == "run_experiment" else "code",
        action=action,  # type: ignore[arg-type]
        goal="changed",
        context_requirement="optional",
    )
    state = _state(tmp_path, step)
    attempt_id = state.attempt_id(step)
    name = (
        "experiment_evidence.json"
        if action == "run_experiment"
        else "repo_stage_evidence.json"
    )
    ref = f"attempts/{attempt_id}/{name}"
    path = Path(state.run_dir) / "steps" / step.step_id / ref
    path.write_bytes(b"original")
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id=step.step_id,
            phase="execute",
            artifact_ref=ref,
            evidence_sha256=sha256_bytes(b"original"),
            attempt_id=attempt_id,
            idempotency_key=f"{attempt_id}:execute",
        ),
    )
    path.write_bytes(b"changed")

    with pytest.raises(Exception, match="immutable execute evidence changed"):
        _load_step_artifacts(state, step)
