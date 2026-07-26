"""The verification loop.

    context -> tool/execute -> verify -> repair -> checkpoint -> repeat

with max-steps, checkpoint/resume, and a human-approval gate. No LangGraph; the
state is a JSON checkpoint + a step ledger, shaped so LangGraph drops in later.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .. import live_context
from ..agents import (
    ContextEngineer,
    Implementer,
    Supervisor,
    VerifierAgent,
)
from ..agents.experimenter import execute_experiment_once
from ..artifacts import ExperimentResult, ExperimentSummary, Patch, Plan, PRSummary
from ..clock import now
from ..config import Config
from ..live_context.models import ContextBundle
from ..llm import get_llm
from ..repo_adapter import (
    RepoAdapterSpec,
    RepoReferenceManifest,
    RepoStageRequest,
    execute_repo_stage_once,
    inspect_repo_integrity,
)
from ..step_ids import canonical_artifact_segment
from ..tasks.spec import TaskSpec
from ..tools import policy
from ..tools.patch import (
    Backup,
    ResolvedPatch,
    apply_patch,
    backup_sha256,
    load_backup,
    render_review_diff,
    resolve_patch,
    revert_patch,
    save_backup,
    snapshot_paths,
)
from ..verifiers import VerifyContext
from ..verifiers.verdict import Check, Verdict
from .approval import (
    ApprovalDecision,
    HumanApprovalGate,
    validate_decision_binding,
)
from .budget import StepBudget
from .checkpoint import (
    append_ledger,
    load_state_by_id,
    read_ledger,
    run_lock,
    save_state,
    validate_run_id,
)
from .errors import (
    ApprovalPending,
    ApprovalRejected,
    BudgetExceeded,
    CheckpointCorrupt,
    PolicyViolation,
    TransactionCorrupt,
)
from .manifest import ArtifactManifest, build_manifest, saved_file_state, sha256_bytes
from .state import RUN_STATE_SCHEMA, LLMUsageState, RunState, StepRecord
from .transaction import (
    PatchTransaction,
    attempt_artifact_dir,
    build_transaction,
    durable_artifact_write,
    list_transactions,
    load_transaction,
    resolve_transaction_evidence,
    save_transaction,
    state_for_paths,
    validate_applied_state,
    validate_transaction_journals,
)

logger = logging.getLogger(__name__)

_IGNORE = shutil.ignore_patterns(
    ".cocoindex_code",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "*.pyc",
    ".coverage",
    ".lha_pytest.json",
    "runs",
)


@dataclass
class RunResult:
    state: RunState
    status: str
    message: str = ""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:32] or "task"


def _safe_seg(step_id: str) -> str:
    return canonical_artifact_segment(step_id)


def _dump(run_dir: Path, step_id: str, name: str, text: str) -> None:
    """Write an artifact both flat (back-compat / last-writer) and per-step under
    ``steps/<step_id>/`` so a multi-step plan keeps every step's provenance."""
    segment = _safe_seg(step_id)
    (run_dir / name).write_text(text)
    step_dir = run_dir / "steps" / segment
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / name).write_text(text)


def _write_immutable(path: Path, data: bytes) -> None:
    """Create one attempt artifact once; a replay may only present identical bytes."""
    if path.is_symlink():
        raise CheckpointCorrupt(f"immutable artifact path is a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise CheckpointCorrupt(f"immutable artifact changed: {path}")
        return
    durable_artifact_write(path, data)


def _verdict_ref(attempt_id: str) -> str:
    return (Path("attempts") / _safe_seg(attempt_id) / "verify.json").as_posix()


def _attempt_ref(attempt_id: str, name: str) -> str:
    return (Path("attempts") / _safe_seg(attempt_id) / name).as_posix()


def _initial_plan_ref() -> str:
    return (Path("plans") / "initial.json").as_posix()


def _repair_plan_ref(attempt_id: str) -> str:
    return (
        Path("plans") / f"{_safe_seg(attempt_id)}-repair.json"
    ).as_posix()


def _persist_verdict(
    run_dir: Path,
    step_id: str,
    attempt_id: str,
    verdict_json: str,
) -> str:
    reference = _verdict_ref(attempt_id)
    path = run_dir / "steps" / _safe_seg(step_id) / reference
    _write_immutable(path, verdict_json.encode("utf-8"))
    _dump(run_dir, step_id, "verify.json", verdict_json)
    return reference


def _gen_run_id(task: TaskSpec) -> str:
    return f"{now():%Y%m%d-%H%M%S}-{_slug(task.title)}-{uuid.uuid4().hex[:4]}"


def _claim_run_dir(runs_dir: str | Path, run_id: str) -> Path:
    """Atomically reserve a run id before copying or checkpointing anything."""
    validate_run_id(run_id)
    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    try:
        run_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"run already exists: {run_id}") from error
    return run_dir


def _policy_verdict(step_id: str, e: PolicyViolation) -> Verdict:
    """A failing verdict for a policy-refused patch (never applied)."""
    return Verdict.from_checks(
        step_id,
        [
            Check(
                name="oracle-policy",
                family="code",
                passed=False,
                detail={
                    "summary": (
                        "patch refused: it modifies protected oracle/config files "
                        f"({', '.join(e.violations)}); fix the source instead"
                    ),
                    "violations": e.violations,
                },
            )
        ],
    )


class Harness:
    def __init__(
        self,
        config: Config | None = None,
        *,
        auto_approve: bool = False,
        interactive_approval: bool = True,
    ):
        self.config = config or Config.from_env()
        self.auto_approve = auto_approve
        self.interactive_approval = interactive_approval
        from ..llm.trace import TracedLLM

        self.llm = TracedLLM(get_llm(self.config), max_calls=self.config.max_llm_calls)
        self.exec = self._make_exec_backend(self.config)
        self._backups: dict[str, Backup] = {}

    @staticmethod
    def _make_exec_backend(config: Config):
        from ..sandbox import make_backend

        if config.exec_backend == "docker":
            return make_backend("docker", image=config.exec_image)
        return make_backend(config.exec_backend)

    # --- public entry points ------------------------------------------------
    def run(self, task: TaskSpec, *, run_id: str | None = None) -> RunResult:
        run_id = run_id or _gen_run_id(task)
        run_dir = _claim_run_dir(self.config.runs_dir, run_id)
        with run_lock(run_dir):
            workdir = run_dir / "workdir"
            self._prepare_workdir(task, workdir)
            state = RunState.new(
                task,
                run_id,
                str(run_dir),
                str(workdir),
                config=self.config,
            )
            save_state(state)
            return self._drive(state)

    def resume(self, run_id: str) -> RunResult:
        validate_run_id(run_id)
        run_dir = Path(self.config.runs_dir).resolve() / run_id
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise CheckpointCorrupt(f"run directory is missing or unsafe: {run_dir}")
        with run_lock(run_dir):
            # Loading inside the lock is essential: a waiter must not resume a
            # stale pre-lock snapshot after another process has completed.
            state = load_state_by_id(self.config.runs_dir, run_id)
            if state.schema_version != RUN_STATE_SCHEMA:
                raise CheckpointCorrupt(
                    f"run {run_id} uses state schema {state.schema_version}; "
                    f"schema {RUN_STATE_SCHEMA} is required for safe resume"
                )
            limits = state.require_matching_budget_limits(self.config)
            if state.status in ("DONE", "FAILED"):
                return RunResult(state, state.status, "run already terminal")
            state.recover_active_elapsed()
            records = read_ledger(state.run_dir)
            if records:
                state.seq = max(state.seq, *(record.seq for record in records))
            try:
                validate_transaction_journals(Path(state.run_dir))
            except TransactionCorrupt as error:
                raise TransactionCorrupt(
                    f"run recovery evidence is invalid: {error}"
                ) from error
            records = self._reconcile_durable_ledger(state, records)
            if records:
                # A ledger append is durable before the following state save.
                # Preserve its sequence number after a crash so replayed work
                # cannot create a second event with an already-used sequence.
                state.seq = max(state.seq, *(record.seq for record in records))
            if state.is_terminal():
                save_state(state)
                return RunResult(
                    state,
                    state.status,
                    "terminal state recovered from durable ledger",
                )
            if (
                limits.deadline_s is not None
                and state.elapsed_s > limits.deadline_s
            ):
                state.status = "PAUSED"
                save_state(state)
                return RunResult(
                    state,
                    "PAUSED",
                    f"deadline {limits.deadline_s}s exceeded during interrupted activity",
                )
            state.status = "RUNNING"
            save_state(state)
            return self._drive(state)

    def _reconcile_durable_ledger(
        self,
        state: RunState,
        records: list[StepRecord],
    ) -> list[StepRecord]:
        """Consume committed ledger transitions missing from ``state.json``.

        The ledger append is deliberately durable before the checkpoint update.
        Recovery may therefore advance state only after revalidating the exact
        plan/verdict/artifact bytes named by that append.
        """
        run_dir = Path(state.run_dir)
        plan_path = run_dir / "plan.json"
        plan_events = [record for record in records if record.phase == "plan"]
        initial_path = run_dir / _initial_plan_ref()
        if not plan_events:
            if (
                state.plan is not None
                or plan_path.exists()
                or plan_path.is_symlink()
                or initial_path.exists()
                or initial_path.is_symlink()
            ):
                raise CheckpointCorrupt(
                    "plan evidence exists without a durable plan event"
                )
            return records
        if len(plan_events) != 1:
            raise CheckpointCorrupt("ledger contains multiple plan events")
        event = plan_events[0]
        if (
            event.artifact_ref != _initial_plan_ref()
            or event.evidence_sha256 is None
            or initial_path.is_symlink()
            or not initial_path.is_file()
            or sha256_bytes(initial_path.read_bytes()) != event.evidence_sha256
        ):
            raise CheckpointCorrupt(
                "durable plan event is not bound to the immutable initial plan"
            )
        try:
            initial_plan = Plan.model_validate_json(initial_path.read_bytes())
        except Exception as error:
            raise CheckpointCorrupt(
                f"durable initial plan is invalid: {error}"
            ) from error

        checkpoint_plan = state.plan.model_copy(deep=True) if state.plan else None
        checkpoint_cursor = state.cursor
        checkpoint_completed = list(state.completed_steps)
        checkpoint_failed = list(state.failed_steps)
        checkpoint_repairs = dict(state.repairs)
        checkpoint_attempts = dict(state.attempt_ids)

        # The immutable initial plan plus ledger-bound repair snapshots are the
        # source of truth. Replaying them prevents a consistently edited
        # state.json + plan.json pair from deleting unfinished work.
        state.plan = initial_plan
        state.cursor = 0
        state.completed_steps = []
        state.failed_steps = []
        state.repairs = {}
        state.attempt_ids = {}
        state.status = "RUNNING"
        valid_plan_snapshots = [initial_plan]

        while not state.is_terminal():
            step = state.next_step()
            if step is None:
                break
            attempt_id = f"{step.step_id}-r{state.repairs_for(step)}"
            state.attempt_ids[step.step_id] = attempt_id
            attempt_records = [
                record
                for record in records
                if record.step_id == step.step_id
                and record.attempt_id == attempt_id
            ]
            approval_records = [
                record
                for record in attempt_records
                if record.phase == "approval"
            ]
            verify_records = [
                record for record in attempt_records if record.phase == "verify"
            ]
            terminal_records = [
                record
                for record in attempt_records
                if record.phase in ("complete", "repair", "fail")
            ]
            if len(approval_records) > 1:
                raise CheckpointCorrupt(
                    f"attempt {attempt_id} has multiple approval decisions"
                )
            approval = None
            if approval_records:
                if not step.requires_approval:
                    raise CheckpointCorrupt(
                        f"attempt {attempt_id} has an unexpected approval decision"
                    )
                try:
                    approval = self._load_committed_approval(
                        state, step, attempt_id, approval_records[0]
                    )
                except Exception:
                    self._revert_step(step, Path(state.workdir))
                    raise
                if not approval.approved:
                    if verify_records:
                        raise CheckpointCorrupt(
                            f"rejected attempt {attempt_id} reached verification"
                        )
                    if any(
                        record.phase in ("complete", "repair")
                        for record in terminal_records
                    ):
                        raise CheckpointCorrupt(
                            f"rejected attempt {attempt_id} has an invalid terminal event"
                        )
                    self._revert_step(step, Path(state.workdir))
                    fail_records = [
                        record
                        for record in terminal_records
                        if record.phase == "fail"
                    ]
                    if len(fail_records) > 1:
                        raise CheckpointCorrupt(
                            f"rejected attempt {attempt_id} has multiple fail events"
                        )
                    if not fail_records:
                        append_ledger(
                            state,
                            StepRecord(
                                seq=state.next_seq(),
                                step_id=step.step_id,
                                phase="fail",
                                attempt_id=attempt_id,
                                idempotency_key=f"{attempt_id}:fail",
                                notes="recovered rejected approval",
                            ),
                        )
                        records = read_ledger(run_dir)
                    state.fail_current(step)
                    break
            if (
                step.requires_approval
                and verify_records
                and verify_records[0].idempotency_key != f"{attempt_id}:policy"
                and (approval is None or not approval.approved)
            ):
                raise CheckpointCorrupt(
                    f"attempt {attempt_id} was verified without approval"
                )
            if not verify_records:
                if terminal_records:
                    raise CheckpointCorrupt(
                        f"attempt {attempt_id} has a terminal event without verification"
                    )
                orphan_path = (
                    run_dir
                    / "steps"
                    / _safe_seg(step.step_id)
                    / _verdict_ref(attempt_id)
                )
                if orphan_path.exists() or orphan_path.is_symlink():
                    if orphan_path.is_symlink() or not orphan_path.is_file():
                        raise CheckpointCorrupt(
                            f"orphan verdict path is unsafe: {orphan_path}"
                        )
                    if not any(
                        record.phase == "context"
                        for record in attempt_records
                    ):
                        raise CheckpointCorrupt(
                            f"orphan verdict has no prepared attempt: {attempt_id}"
                        )
                    try:
                        orphan = Verdict.model_validate_json(
                            orphan_path.read_bytes()
                        )
                    except Exception as error:
                        raise CheckpointCorrupt(
                            f"orphan verdict is invalid for {attempt_id}: {error}"
                        ) from error
                    names = [check.name for check in orphan.checks]
                    is_policy = names == ["oracle-policy"]
                    if not is_policy and not any(
                        record.phase == "execute"
                        for record in attempt_records
                    ):
                        raise CheckpointCorrupt(
                            f"orphan verdict has no execute event: {attempt_id}"
                        )
                    if names not in (
                        list(step.verifiers),
                        ["oracle-policy"],
                    ):
                        raise CheckpointCorrupt(
                            f"orphan verdict checks do not match {attempt_id}"
                        )
                    verdict_bytes = orphan_path.read_bytes()
                    recovered_record = StepRecord(
                        seq=state.next_seq(),
                        step_id=step.step_id,
                        phase="verify",
                        verdict_ref=_verdict_ref(attempt_id),
                        evidence_sha256=sha256_bytes(verdict_bytes),
                        attempt_id=attempt_id,
                        idempotency_key=(
                            f"{attempt_id}:policy"
                            if is_policy
                            else f"{attempt_id}:verify"
                        ),
                        notes="recovered durable verdict after interrupted ledger append",
                    )
                    self._load_committed_verdict(
                        state, step, attempt_id, recovered_record
                    )
                    alias = (
                        run_dir
                        / "steps"
                        / _safe_seg(step.step_id)
                        / "verify.json"
                    )
                    if (
                        alias.is_symlink()
                        or not alias.is_file()
                        or alias.read_bytes() != verdict_bytes
                    ):
                        raise CheckpointCorrupt(
                            f"orphan verdict alias changed for {attempt_id}"
                        )
                    append_ledger(state, recovered_record)
                    records = read_ledger(run_dir)
                    continue
                break
            if len(verify_records) != 1 or len(terminal_records) > 1:
                raise CheckpointCorrupt(
                    f"attempt {attempt_id} has ambiguous durable transitions"
                )
            verdict = self._load_committed_verdict(
                state, step, attempt_id, verify_records[0]
            )
            terminal = terminal_records[0] if terminal_records else None
            if terminal is None:
                if verdict.passed:
                    self._mark_step_verified(state, step, Path(state.workdir))
                    state.complete_step(step)
                    append_ledger(
                        state,
                        StepRecord(
                            seq=state.next_seq(),
                            step_id=step.step_id,
                            phase="complete",
                            evidence_sha256=verify_records[0].evidence_sha256,
                            attempt_id=attempt_id,
                            idempotency_key=f"{attempt_id}:complete",
                        ),
                    )
                else:
                    self._repair_or_fail(
                        state,
                        step,
                        verdict,
                        StepBudget(
                            limits=state.require_matching_budget_limits(self.config)
                        ),
                        Path(state.workdir),
                    )
                records = read_ledger(run_dir)
                continue

            if terminal.phase == "complete":
                if (
                    not verdict.passed
                    or terminal.evidence_sha256
                    != verify_records[0].evidence_sha256
                ):
                    raise CheckpointCorrupt(
                        f"attempt {attempt_id} complete event is not bound to its verdict"
                    )
                self._mark_step_verified(state, step, Path(state.workdir))
                state.complete_step(step)
            elif terminal.phase == "repair":
                if verdict.passed:
                    raise CheckpointCorrupt(
                        f"passing attempt {attempt_id} cannot enter repair"
                    )
                state.plan = self._load_committed_repair_plan(
                    state, step, verdict, terminal
                )
                valid_plan_snapshots.append(state.plan.model_copy(deep=True))
                state.record_repair(step)
            else:
                if verdict.passed:
                    raise CheckpointCorrupt(
                        f"passing attempt {attempt_id} cannot fail the run"
                    )
                unsafe = [
                    transaction
                    for transaction in list_transactions(run_dir, step.step_id)
                    if transaction.status != "REVERTED"
                ]
                if unsafe:
                    raise CheckpointCorrupt(
                        f"failed attempt {attempt_id} has unreverted transactions"
                    )
                state.fail_current(step)

        if checkpoint_plan is not None and not any(
            checkpoint_plan == snapshot for snapshot in valid_plan_snapshots
        ):
            raise CheckpointCorrupt(
                "checkpoint plan is absent from the immutable plan history"
            )
        if (
            checkpoint_cursor > state.cursor
            or checkpoint_completed
            != state.completed_steps[: len(checkpoint_completed)]
            or (
                checkpoint_failed
                and checkpoint_failed != state.failed_steps
            )
            or any(
                count > state.repairs.get(step_id, 0)
                for step_id, count in checkpoint_repairs.items()
            )
        ):
            raise CheckpointCorrupt(
                "checkpoint progress is ahead of the durable ledger"
            )
        plan_step_ids = {step.step_id for step in initial_plan.steps}
        for step_id, attempt_id in checkpoint_attempts.items():
            if step_id not in plan_step_ids:
                raise CheckpointCorrupt(
                    f"checkpoint names an unknown attempt step: {step_id}"
                )
            count = checkpoint_repairs.get(step_id, 0)
            if attempt_id != f"{step_id}-r{count}":
                raise CheckpointCorrupt(
                    f"checkpoint attempt identity is invalid for {step_id}"
                )

        if plan_path.is_symlink() or not plan_path.is_file():
            raise CheckpointCorrupt("checksummed plan history has no safe plan.json")
        try:
            disk_plan = Plan.model_validate_json(plan_path.read_bytes())
        except Exception as error:
            raise CheckpointCorrupt(f"plan.json is invalid: {error}") from error
        if disk_plan != state.plan:
            raise CheckpointCorrupt(
                "plan.json does not match the reconciled plan history"
            )
        return read_ledger(run_dir)

    def _load_committed_approval(
        self,
        state: RunState,
        step,
        attempt_id: str,
        record: StepRecord,
    ) -> ApprovalDecision:
        gate = HumanApprovalGate(state.run_dir)
        try:
            request = gate.request_evidence(
                step.step_id, attempt_id, validate_alias=False
            )
            decision = gate.decision_evidence(
                step.step_id, attempt_id, validate_alias=False
            )
            if request is None or decision is None:
                raise ValueError("approval request or decision is missing")
            artifact_sha256 = None
            if step.action == "edit_code":
                patch_path = (
                    attempt_artifact_dir(
                        Path(state.run_dir), step.step_id, attempt_id
                    )
                    / "patch.json"
                )
                if patch_path.is_symlink() or not patch_path.is_file():
                    raise ValueError(
                        "reviewed attempt patch is missing or unsafe"
                    )
                artifact_sha256 = sha256_bytes(patch_path.read_bytes())
            validate_decision_binding(
                request=request,
                decision=decision,
                step_id=step.step_id,
                attempt_id=attempt_id,
                goal=step.goal,
                artifact_sha256=artifact_sha256,
            )
        except ValueError as error:
            raise CheckpointCorrupt(
                f"invalid approval evidence for {attempt_id}: {error}"
            ) from error
        if (
            record.artifact_ref != decision.reference
            or record.evidence_sha256 != decision.sha256
            or record.idempotency_key != f"{attempt_id}:approval"
        ):
            raise CheckpointCorrupt(
                f"approval ledger event is not bound to {attempt_id}"
            )
        return decision.value

    @staticmethod
    def _load_committed_verdict(
        state: RunState,
        step,
        attempt_id: str,
        record: StepRecord,
    ) -> Verdict:
        step_dir = Path(state.run_dir) / "steps" / _safe_seg(step.step_id)
        expected_ref = _verdict_ref(attempt_id)
        path = step_dir / expected_ref
        if (
            record.verdict_ref != expected_ref
            or record.evidence_sha256 is None
            or path.is_symlink()
            or not path.is_file()
            or sha256_bytes(path.read_bytes()) != record.evidence_sha256
        ):
            raise CheckpointCorrupt(
                f"durable verdict is missing or changed for {attempt_id}"
            )
        try:
            verdict = Verdict.model_validate_json(path.read_bytes())
        except Exception as error:
            raise CheckpointCorrupt(
                f"durable verdict is invalid for {attempt_id}: {error}"
            ) from error
        expected_checks = (
            ["oracle-policy"]
            if record.idempotency_key == f"{attempt_id}:policy"
            else list(step.verifiers)
        )
        if (
            verdict.step_id != step.step_id
            or verdict.attempt_id != attempt_id
            or [check.name for check in verdict.checks] != expected_checks
        ):
            raise CheckpointCorrupt(
                f"durable verdict identity does not match {attempt_id}"
            )
        expected_artifact_name = {
            "edit_code": "patch.json",
            "run_experiment": "experiment_evidence.json",
            "repo_stage": "repo_stage_evidence.json",
            "gather_context": "context_bundle.json",
            "answer_query": "context_bundle.json",
            "repo_integrity": "repo_integrity.json",
        }[step.action]
        expected_artifact_ref = _attempt_ref(
            attempt_id, expected_artifact_name
        )
        if verdict.artifact_ref != expected_artifact_ref:
            raise CheckpointCorrupt(
                f"durable verdict artifact reference is unsafe for {attempt_id}"
            )
        artifact_path = step_dir / expected_artifact_ref
        if (
            verdict.artifact_sha256 is None
            or artifact_path.is_symlink()
            or not artifact_path.is_file()
            or sha256_bytes(artifact_path.read_bytes())
            != verdict.artifact_sha256
        ):
            raise CheckpointCorrupt(
                f"durable verdict artifact changed for {attempt_id}"
            )
        return verdict

    @staticmethod
    def _load_committed_repair_plan(
        state: RunState,
        step,
        verdict: Verdict,
        record: StepRecord,
    ) -> Plan:
        expected_ref = _repair_plan_ref(record.attempt_id or "")
        path = Path(state.run_dir) / expected_ref
        if (
            record.artifact_ref != expected_ref
            or record.evidence_sha256 is None
            or path.is_symlink()
            or not path.is_file()
            or sha256_bytes(path.read_bytes()) != record.evidence_sha256
        ):
            raise CheckpointCorrupt(
                "repair event is not bound to its immutable plan for "
                f"{record.attempt_id}"
            )
        try:
            disk_plan = Plan.model_validate_json(path.read_bytes())
        except Exception as error:
            raise CheckpointCorrupt(f"repair plan is invalid: {error}") from error
        assert state.plan is not None
        expected = state.plan.model_copy(deep=True)
        expected.steps[state.cursor] = step.as_repair(verdict.failures)
        if disk_plan != expected:
            raise CheckpointCorrupt(
                f"repair plan does not match verdict for {record.attempt_id}"
            )
        return disk_plan

    # --- driver -------------------------------------------------------------
    def _drive(self, state: RunState) -> RunResult:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)
        from ..llm.trace import TracedLLM

        if isinstance(self.llm, TracedLLM):
            self.llm.bind(run_dir)  # per-call records land in llm_trace.jsonl
            self.llm.restore_totals(state.llm_usage)

        # Point code context at the stable target repo (indexed once, reused across
        # runs) rather than the ephemeral per-run workdir — the latter churns the
        # ccc daemon with throwaway projects. Edits/verification still hit workdir.
        code_root = state.task.target_repo or str(workdir)
        live_context.configure(code_root=code_root, config=self.config)
        try:
            live_context.index_code(code_root)
        except Exception:  # best-effort; loop still runs with empty context
            logger.debug("index_code(%s) failed", code_root, exc_info=True)
        if state.task.kind == "paper_to_experiment":
            try:
                live_context.index_docs()  # best-effort: paper/experiment context
            except Exception:
                logger.debug("index_docs() failed", exc_info=True)

        # Seed the budget from the checkpoint so max_steps/deadline bound the whole
        # run across pause/resume, not just this process.
        limits = state.require_matching_budget_limits(self.config)
        budget = StepBudget(
            limits=limits,
            steps_used=state.steps_used,
            prior_elapsed_s=state.elapsed_s,
        )

        in_flight = None  # the step currently executing, for revert on an unexpected fault
        try:
            # PLAN (once) — inside the fault boundary, so a plan-time
            # BudgetExceeded pauses the run instead of escaping as a traceback.
            if state.plan is None:
                budget.check_deadline()
                state.active_since = now()
                state.elapsed_s = budget.elapsed()
                self._save(state)
                if isinstance(self.llm, TracedLLM):
                    self.llm.set_call_context(
                        run_id=state.run_id,
                        attempt_id="plan",
                        task=state.task.model_dump(mode="json"),
                        config=self.config.model_dump(mode="json"),
                    )
                state.plan = Supervisor(self.config, self.llm).plan(state.task)
                plan_bytes = state.plan.model_dump_json(indent=2).encode("utf-8")
                initial_ref = _initial_plan_ref()
                _write_immutable(run_dir / initial_ref, plan_bytes)
                durable_artifact_write(run_dir / "plan.json", plan_bytes)
                (run_dir / "plan.md").write_text(self._plan_md(state))
                append_ledger(
                    state,
                    StepRecord(
                        seq=state.next_seq(),
                        step_id="-",
                        phase="plan",
                        artifact_ref=initial_ref,
                        evidence_sha256=sha256_bytes(plan_bytes),
                        idempotency_key=f"{state.run_id}:plan",
                    ),
                )
                state.active_since = None
                state.elapsed_s = budget.elapsed()
                self._save(state)

            while not state.is_terminal():
                step = state.next_step()
                if step is None:
                    break
                in_flight = step
                budget.check_deadline()
                if not state.attempt_is_budgeted(step):
                    budget.tick()
                    state.steps_used = budget.steps_used
                    state.mark_attempt_budgeted(step)
                    state.elapsed_s = budget.elapsed()
                    # The budget commit precedes every model/tool side effect.
                    # A crash-replay of this attempt therefore cannot buy
                    # another step by restoring an older counter.
                state.active_since = now()
                state.elapsed_s = budget.elapsed()
                self._save(state)
                self._run_step(state, step, budget)
                in_flight = None  # finished cleanly (cursor may have advanced)
                state.active_since = None
                state.elapsed_s = budget.elapsed()
                self._save(state)
        except BudgetExceeded as e:
            state.steps_used = budget.steps_used
            state.elapsed_s = budget.elapsed()
            state.active_since = None
            state.status = "PAUSED"
            self._save(state)
            return RunResult(state, "PAUSED", str(e))
        except ApprovalPending as e:
            state.elapsed_s = budget.elapsed()
            state.active_since = None
            self._save(state)
            return RunResult(state, "AWAITING_APPROVAL", str(e))
        except ApprovalRejected as e:
            state.elapsed_s = budget.elapsed()
            state.active_since = None
            state.status = "FAILED"
            self._save(state)
            return RunResult(state, "FAILED", str(e))
        except Exception as e:
            # An unexpected fault mid-step (unknown action, failed patch apply, a
            # crashing tool) must not leave the run wedged at RUNNING with a
            # half-applied sandbox: revert the in-flight step, fail closed, checkpoint.
            if in_flight is not None and in_flight.step_id not in state.completed_steps:
                try:
                    self._revert_step(in_flight, workdir)
                except Exception:  # a revert failure must not abort the fail-closed path
                    logger.exception("revert failed for step %s", in_flight.step_id)
                append_ledger(
                    state,
                    StepRecord(
                        seq=state.next_seq(),
                        step_id=in_flight.step_id,
                        phase="fail",
                        notes=f"error: {type(e).__name__}: {e}"[:300],
                    ),
                )
                state.fail_current(in_flight)
            else:
                # No step in flight, or the step itself completed and only its
                # bookkeeping failed — never revert verified work.
                state.status = "FAILED"
            state.elapsed_s = budget.elapsed()
            state.active_since = None
            self._save(state)
            return RunResult(state, "FAILED", f"{type(e).__name__}: {e}")

        if not state.is_terminal():
            state.status = "DONE"
        if state.status == "DONE" and state.task.kind == "issue_to_pr":
            self._finalize_pr(state)
        if state.status == "DONE" and state.task.kind == "paper_to_experiment":
            self._finalize_experiment(state)
        if state.status == "DONE" and self.config.use_skill_memory:
            try:
                from ..memory import SkillMemory

                if SkillMemory(Path(self.config.data_dir) / "skills").record(state) is not None:
                    # Re-index so the new skill is retrievable on the next run; otherwise
                    # the memory loop stays open for issue_to_pr runs.
                    live_context.index_docs(("skill",))
            except Exception:  # skill recording is best-effort
                logger.debug("skill recording failed", exc_info=True)
        self._save(state)
        return RunResult(state, state.status)

    def _save(self, state: RunState) -> None:
        from ..llm.trace import TracedLLM

        if isinstance(self.llm, TracedLLM):
            state.llm_usage = LLMUsageState.model_validate(asdict(self.llm.totals))
        save_state(state)

    # --- one step -----------------------------------------------------------
    def _run_step(self, state: RunState, step, budget: StepBudget) -> None:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)
        attempt_id = state.attempt_id(step)

        # 1. CONTEXT
        bundle = self._context_for_attempt(
            state, step, attempt_id, workdir
        )

        # 2/3. EXECUTE (dispatch by action)
        try:
            artifact, artifact_ref = self._execute(state, step, bundle)
        except PolicyViolation as e:
            # The patch never reached the sandbox. Record a failing verdict so
            # the repair loop is told exactly why, instead of aborting the run.
            verdict = _policy_verdict(step.step_id, e)
            verdict = self._bind_verdict(
                verdict,
                state,
                step,
                artifact_ref="patch.json",
                attempt_id=attempt_id,
            )
            verdict_json = verdict.model_dump_json(indent=2)
            verdict_sha = sha256_bytes(verdict_json.encode("utf-8"))
            verdict_ref = _persist_verdict(
                run_dir, step.step_id, attempt_id, verdict_json
            )
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id=step.step_id,
                    phase="verify",
                    verdict_ref=verdict_ref,
                    evidence_sha256=verdict_sha,
                    notes=str(e)[:300],
                    attempt_id=attempt_id,
                    idempotency_key=f"{attempt_id}:policy",
                ),
            )
            self._repair_or_fail(state, step, verdict, budget, workdir)
            self._save(state)
            return
        execute_ref, execute_sha = self._execution_ledger_binding(
            state, step, attempt_id, artifact_ref
        )
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id=step.step_id,
                phase="execute",
                artifact_ref=execute_ref,
                evidence_sha256=execute_sha,
                attempt_id=attempt_id,
                idempotency_key=f"{attempt_id}:execute",
            ),
        )

        # HUMAN-APPROVAL GATE (before verify / irreversible boundary)
        if step.requires_approval:
            self._approval_gate(state, step, artifact_ref)

        # 4. VERIFY
        verdict = VerifierAgent(parallel=self.config.parallel_verify).verify(
            step,
            artifact,
            VerifyContext(
                workdir=workdir,
                step=step,
                bundle=bundle,
                exec=self.exec,
                attempt_id=attempt_id,
            ),
        )
        verdict = self._bind_verdict(
            verdict,
            state,
            step,
            artifact_ref=artifact_ref,
            attempt_id=attempt_id,
        )
        verdict_json = verdict.model_dump_json(indent=2)
        verdict_sha = sha256_bytes(verdict_json.encode("utf-8"))
        verdict_ref = _persist_verdict(
            run_dir, step.step_id, attempt_id, verdict_json
        )
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id=step.step_id,
                phase="verify",
                verdict_ref=verdict_ref,
                evidence_sha256=verdict_sha,
                attempt_id=attempt_id,
                idempotency_key=f"{attempt_id}:verify",
            ),
        )

        # 5. REPAIR or ADVANCE
        if verdict.passed:
            self._mark_step_verified(state, step, workdir)
            state.complete_step(step)
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id=step.step_id,
                    phase="complete",
                    evidence_sha256=verdict_sha,
                    attempt_id=attempt_id,
                    idempotency_key=f"{attempt_id}:complete",
                ),
            )
        else:
            self._repair_or_fail(state, step, verdict, budget, workdir)

        # 6. CHECKPOINT
        self._save(state)

    def _context_for_attempt(
        self,
        state: RunState,
        step,
        attempt_id: str,
        workdir: Path,
    ) -> ContextBundle:
        run_dir = Path(state.run_dir)
        context_ref = _attempt_ref(attempt_id, "context_bundle.json")
        path = (
            run_dir / "steps" / _safe_seg(step.step_id) / context_ref
        )
        records = [
            record
            for record in read_ledger(run_dir)
            if record.phase == "context"
            and record.step_id == step.step_id
            and record.attempt_id == attempt_id
        ]
        if len(records) > 1:
            raise CheckpointCorrupt(
                f"attempt {attempt_id} has multiple context events"
            )
        if records or path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise CheckpointCorrupt(
                    f"context evidence is missing or unsafe: {path}"
                )
            data = path.read_bytes()
            digest = sha256_bytes(data)
            try:
                bundle = ContextBundle.model_validate_json(data)
            except Exception as error:
                raise CheckpointCorrupt(
                    f"context evidence is invalid for {attempt_id}: {error}"
                ) from error
            if records:
                record = records[0]
                if (
                    record.artifact_ref != context_ref
                    or record.evidence_sha256 != digest
                ):
                    raise CheckpointCorrupt(
                        f"context event is not bound to {attempt_id}"
                    )
            else:
                append_ledger(
                    state,
                    StepRecord(
                        seq=state.next_seq(),
                        step_id=step.step_id,
                        phase="context",
                        artifact_ref=context_ref,
                        evidence_sha256=digest,
                        attempt_id=attempt_id,
                        idempotency_key=f"{attempt_id}:context",
                        notes="recovered durable context after interrupted ledger append",
                    ),
                )
            _dump(
                run_dir,
                step.step_id,
                "context_bundle.json",
                data.decode("utf-8"),
            )
            return bundle

        bundle = ContextEngineer(self.config).gather(step, workdir=workdir)
        data = bundle.model_dump_json(indent=2).encode("utf-8")
        _write_immutable(path, data)
        _dump(
            run_dir,
            step.step_id,
            "context_bundle.json",
            data.decode("utf-8"),
        )
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id=step.step_id,
                phase="context",
                artifact_ref=context_ref,
                evidence_sha256=sha256_bytes(data),
                attempt_id=attempt_id,
                idempotency_key=f"{attempt_id}:context",
            ),
        )
        return bundle

    @staticmethod
    def _execution_ledger_binding(
        state: RunState,
        step,
        attempt_id: str,
        artifact_ref: str,
    ) -> tuple[str, str | None]:
        evidence_name = {
            "run_experiment": "experiment_evidence.json",
            "repo_stage": "repo_stage_evidence.json",
        }.get(step.action)
        if evidence_name is None:
            evidence_name = (
                "patch.json" if step.action == "edit_code" else artifact_ref
            )
        reference = _attempt_ref(attempt_id, evidence_name)
        path = (
            Path(state.run_dir)
            / "steps"
            / _safe_seg(step.step_id)
            / reference
        )
        if path.is_symlink() or not path.is_file():
            alias = (
                Path(state.run_dir)
                / "steps"
                / _safe_seg(step.step_id)
                / artifact_ref
            )
            if alias.is_symlink() or not alias.is_file():
                raise CheckpointCorrupt(
                    f"completed action evidence is missing: {path}"
                )
            _write_immutable(path, alias.read_bytes())
        return reference, sha256_bytes(path.read_bytes())

    @staticmethod
    def _bind_verdict(
        verdict: Verdict,
        state: RunState,
        step,
        *,
        artifact_ref: str,
        attempt_id: str,
    ) -> Verdict:
        stored_ref = {
            "edit_code": _attempt_ref(attempt_id, "patch.json"),
            "run_experiment": _attempt_ref(
                attempt_id, "experiment_evidence.json"
            ),
            "repo_stage": _attempt_ref(
                attempt_id, "repo_stage_evidence.json"
            ),
        }.get(step.action, _attempt_ref(attempt_id, artifact_ref))
        path = (
            Path(state.run_dir)
            / "steps"
            / _safe_seg(step.step_id)
            / stored_ref
        )
        if path.is_symlink() or not path.is_file():
            raise TransactionCorrupt(
                f"cannot bind verdict to missing artifact: {path}"
            )
        return verdict.model_copy(
            update={
                "artifact_ref": stored_ref,
                "artifact_sha256": sha256_bytes(path.read_bytes()),
                "attempt_id": attempt_id,
            }
        )

    def _repair_or_fail(self, state: RunState, step, verdict: Verdict, budget, workdir) -> None:
        """Re-issue the step as a repair with the verdict's failures, or fail the run."""
        attempt_id = state.attempt_id(step)
        non_retryable = any(check.detail.get("non_retryable") for check in verdict.checks)
        if not non_retryable and state.repairs_for(step) < budget.max_repairs:
            assert state.plan is not None  # set before the loop body runs
            state.plan.steps[state.cursor] = step.as_repair(verdict.failures)
            plan_bytes = state.plan.model_dump_json(indent=2).encode("utf-8")
            repair_ref = _repair_plan_ref(attempt_id)
            _write_immutable(
                Path(state.run_dir) / repair_ref,
                plan_bytes,
            )
            durable_artifact_write(
                Path(state.run_dir) / "plan.json",
                plan_bytes,
            )
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id=step.step_id,
                    phase="repair",
                    artifact_ref=repair_ref,
                    evidence_sha256=sha256_bytes(plan_bytes),
                    notes="; ".join(verdict.failures)[:300],
                    attempt_id=attempt_id,
                    idempotency_key=f"{attempt_id}:repair",
                ),
            )
            state.record_repair(step)
        else:
            self._revert_step(step, workdir)
            state.fail_current(step)
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id=step.step_id,
                    phase="fail",
                    attempt_id=attempt_id,
                    idempotency_key=f"{attempt_id}:fail",
                ),
            )

    # --- execute dispatch ---------------------------------------------------
    def _execute(self, state: RunState, step, bundle) -> tuple[Any, str]:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        if step.action in ("gather_context", "answer_query"):
            if bundle.answer:
                (run_dir / "answer.md").write_text(bundle.answer)
            return bundle, "context_bundle.json"

        if step.action == "edit_code":
            return self._execute_patch(state, step, bundle), "patch.diff"

        if step.action == "run_experiment":
            experiment: ExperimentResult = execute_experiment_once(
                step=step,
                bundle=bundle,
                workdir=workdir,
                run_dir=run_dir,
                attempt_id=state.attempt_id(step),
                backend=self.exec,
            )
            _dump(
                run_dir,
                step.step_id,
                "experiment.json",
                experiment.model_dump_json(indent=2),
            )
            return experiment, "experiment.json"

        if step.action == "repo_integrity":
            manifest = RepoReferenceManifest.from_file(
                str(step.params["reference_manifest_path"])
            )
            integrity = inspect_repo_integrity(
                workdir,
                manifest,
                str(step.params["reference_patch_path"]),
                task_path=str(step.params["task_path"]),
                adapter_path=str(step.params["repo_adapter_path"]),
            )
            _dump(
                run_dir,
                step.step_id,
                "repo_integrity.json",
                integrity.model_dump_json(indent=2),
            )
            return integrity, "repo_integrity.json"

        if step.action == "repo_stage":
            spec = RepoAdapterSpec.model_validate(step.params["repo_adapter_spec"])
            stage = RepoStageRequest(stage=step.params["repo_stage"]).stage
            stage_result = execute_repo_stage_once(
                worktree=workdir,
                run_dir=run_dir,
                step_id=step.step_id,
                attempt_id=state.attempt_id(step),
                spec=spec,
                backend=self.exec,
                stage=stage,
            )
            return stage_result, "repo_stage.json"

        raise ValueError(f"unknown action: {step.action}")

    def _execute_patch(self, state: RunState, step, bundle) -> Patch:
        """Apply or recover one durable patch attempt.

        A crash after PREPARED restores the persisted backup and reapplies the
        same bytes. A crash after APPLIED/VERIFIED validates the worktree and
        never asks the model for a different patch.
        """
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)
        attempt_id = state.attempt_id(step)
        artifact_dir = attempt_artifact_dir(run_dir, step.step_id, attempt_id)
        tx = load_transaction(run_dir, step.step_id, attempt_id)

        if tx is not None:
            data = self._patch_bytes(state, step)
            attempt_path = artifact_dir / "patch.json"
            try:
                attempt_data = attempt_path.read_bytes()
            except OSError as e:
                raise TransactionCorrupt(
                    f"persisted patch is missing for {step.step_id}/{attempt_id}"
                ) from e
            if data is None or sha256_bytes(data) != tx.patch_sha256:
                raise TransactionCorrupt(
                    f"patch hash mismatch for {step.step_id}/{attempt_id}"
                )
            if sha256_bytes(attempt_data) != tx.patch_sha256 or data != attempt_data:
                raise TransactionCorrupt(
                    f"attempt artifact hash mismatch for {step.step_id}/{attempt_id}"
                )
            try:
                patch = Patch.model_validate_json(data)
            except Exception as e:
                raise TransactionCorrupt(
                    f"persisted patch is invalid for {step.step_id}/{attempt_id}: {e}"
                ) from e
            if patch.step_id != step.step_id:
                raise TransactionCorrupt(
                    f"persisted patch step identity mismatch for "
                    f"{step.step_id}/{attempt_id}"
                )
            resolved = resolve_patch(patch, patch_bytes=data)
            if resolved.paths != tx.resolved_paths:
                raise TransactionCorrupt(
                    f"write set mismatch for {step.step_id}/{attempt_id}"
                )
            backup, degraded = self._transaction_backup(run_dir, tx)
            if degraded:
                # Restore from the redundant copy, but fail the attempt: damaged
                # recovery evidence must never be treated as a clean resume.
                revert_patch(backup, workdir)
                raise TransactionCorrupt(
                    f"primary backup is corrupt for {step.step_id}/{attempt_id}; "
                    "the mirror was used to restore the sandbox"
                )
            self._validate_attempt_manifest(
                state=state,
                step=step,
                tx=tx,
                artifact_dir=artifact_dir,
                patch_bytes=data,
                patch=patch,
                resolved=resolved,
                backup=backup,
            )
            if tx.status == "PREPARED":
                # PREPARED may mean the crash happened immediately before or
                # after the write. Restoring first makes both cases identical.
                revert_patch(backup, workdir)
                apply_patch(patch, workdir, resolved=resolved, backup=backup)
                tx = tx.transition("APPLIED", workdir=workdir)
                save_transaction(run_dir, tx)
            elif tx.status in ("APPLIED", "VERIFIED"):
                validate_applied_state(tx, workdir)
            else:
                raise TransactionCorrupt(
                    f"attempt {step.step_id}/{attempt_id} was already reverted"
                )
            return patch

        from ..llm.trace import TracedLLM

        if isinstance(self.llm, TracedLLM):
            self.llm.set_call_context(
                run_id=state.run_id,
                attempt_id=attempt_id,
                task=state.task.model_dump(mode="json"),
                config=self.config.model_dump(mode="json"),
            )
        patch = Implementer(self.llm).implement(step, bundle, workdir)
        patch_json = patch.model_dump_json(indent=2)
        patch_bytes = patch_json.encode("utf-8")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_immutable(artifact_dir / "patch.json", patch_bytes)
        _dump(run_dir, step.step_id, "patch.json", patch_json)
        if patch.step_id != step.step_id:
            raise PolicyViolation(
                step.step_id,
                [f"patch step_id {patch.step_id!r} does not match {step.step_id!r}"],
            )
        resolved = resolve_patch(patch, patch_bytes=patch_bytes)
        backup = snapshot_paths(resolved.paths, workdir)
        review_diff = render_review_diff(patch, resolved, backup)
        _write_immutable(
            artifact_dir / "review.diff", review_diff.encode("utf-8")
        )
        _dump(run_dir, step.step_id, "patch.diff", review_diff)
        violations = policy.check_resolved(resolved, state.task.allowed_protected_files)
        if violations:
            # Persist the refused proposal for audit, but do not create a
            # transaction because the sandbox was never touched.
            raise PolicyViolation(step.step_id, violations)

        backup_path = (
            run_dir / "backups" / _safe_seg(step.step_id) / f"{_safe_seg(attempt_id)}.json"
        )
        backup_digest = save_backup(backup, backup_path)
        tx = build_transaction(
            run_dir=run_dir,
            step_id=step.step_id,
            attempt_id=attempt_id,
            resolved=resolved,
            backup_sha256=backup_digest,
        )
        save_backup(backup, run_dir / tx.backup_mirror_ref)

        manifest = build_manifest(
            step=step,
            patch=patch,
            patch_json_bytes=patch_bytes,
            workdir=workdir,
            policy_overrides=list(state.task.allowed_protected_files),
            resolved=resolved,
        )
        manifest_json = manifest.model_dump_json(indent=2)
        durable_artifact_write(
            artifact_dir / "manifest.json", manifest_json.encode("utf-8")
        )
        _dump(run_dir, step.step_id, "manifest.json", manifest_json)

        # PREPARED is the point of no return: all bytes needed to replay,
        # approve, or revert now exist durably, and no target file changed yet.
        save_transaction(run_dir, tx)
        apply_patch(patch, workdir, resolved=resolved, backup=backup)
        tx = tx.transition("APPLIED", workdir=workdir)
        save_transaction(run_dir, tx)
        self._backups[attempt_id] = backup
        return patch

    @staticmethod
    def _validate_attempt_manifest(
        *,
        state: RunState,
        step,
        tx: PatchTransaction,
        artifact_dir: Path,
        patch_bytes: bytes,
        patch: Patch,
        resolved: ResolvedPatch,
        backup: Backup,
    ) -> None:
        """Bind recovery to the reviewed patch, base state, and policy."""
        path = artifact_dir / "manifest.json"
        if path.is_symlink() or not path.is_file():
            raise TransactionCorrupt(
                f"persisted manifest is missing for {tx.step_id}/{tx.attempt_id}"
            )
        try:
            manifest = ArtifactManifest.model_validate_json(path.read_bytes())
            expected_review = render_review_diff(
                patch, resolved, backup
            ).encode("utf-8")
            review_paths = [
                artifact_dir / "review.diff",
                artifact_dir.parents[1] / "patch.diff",
            ]
            for review_path in review_paths:
                if review_path.is_symlink() or not review_path.is_file():
                    raise ValueError(f"review artifact is missing: {review_path}")
                if review_path.read_bytes() != expected_review:
                    raise ValueError(
                        f"review artifact does not match executable patch: {review_path}"
                    )
            expected_base = {
                rel: saved_file_state(backup.originals[rel], backup.modes[rel])
                for rel in tx.resolved_paths
            }
        except Exception as error:
            raise TransactionCorrupt(
                f"persisted manifest is invalid for {tx.step_id}/{tx.attempt_id}: {error}"
            ) from error
        mismatches: list[str] = []
        if manifest.step_id != tx.step_id or manifest.step_id != step.step_id:
            mismatches.append("step")
        if (
            manifest.artifact_sha256 != tx.patch_sha256
            or sha256_bytes(patch_bytes) != manifest.artifact_sha256
        ):
            mismatches.append("patch")
        if manifest.touched_files != tx.resolved_paths:
            mismatches.append("write set")
        if manifest.base_state != expected_base:
            mismatches.append("base state")
        if manifest.verifiers != list(step.verifiers):
            mismatches.append("verifiers")
        if manifest.policy_overrides != list(state.task.allowed_protected_files):
            mismatches.append("policy")
        if mismatches:
            raise TransactionCorrupt(
                f"manifest mismatch for {tx.step_id}/{tx.attempt_id}: "
                + ", ".join(mismatches)
            )

    @staticmethod
    def _transaction_backup(run_dir: Path, tx: PatchTransaction) -> tuple[Backup, bool]:
        """Load a transaction backup, with a redundant copy for safe rollback."""
        primary_error: Exception | None = None
        try:
            backup = load_backup(
                resolve_transaction_evidence(run_dir, tx.backup_ref),
                required=True,
            )
            assert backup is not None
            if backup_sha256(backup) != tx.backup_sha256:
                raise ValueError("primary backup checksum mismatch")
            return backup, False
        except Exception as e:
            primary_error = e
        try:
            mirror = load_backup(
                resolve_transaction_evidence(run_dir, tx.backup_mirror_ref),
                required=True,
            )
            assert mirror is not None
            if backup_sha256(mirror) != tx.backup_sha256:
                raise ValueError("mirror backup checksum mismatch")
            return mirror, True
        except Exception as mirror_error:
            raise TransactionCorrupt(
                f"both backups are unusable for {tx.step_id}/{tx.attempt_id}: "
                f"primary={primary_error}; mirror={mirror_error}"
            ) from mirror_error

    def _mark_step_verified(self, state: RunState, step, workdir: Path) -> None:
        run_dir = Path(state.run_dir)
        transactions = [
            tx
            for tx in list_transactions(run_dir, step.step_id)
            if tx.status in ("APPLIED", "VERIFIED")
        ]
        # A repair may supersede a path written by an earlier attempt. Validate
        # the newest expected digest for every path exactly once.
        covered: set[str] = set()
        for tx in reversed(transactions):
            for rel in tx.resolved_paths:
                if rel in covered:
                    continue
                if state_for_paths(workdir, [rel])[rel] != tx.applied_state.get(rel):
                    raise TransactionCorrupt(
                        f"worktree drift for {step.step_id}: {rel} changed before verification"
                    )
                covered.add(rel)
        for tx in transactions:
            if tx.status == "APPLIED":
                save_transaction(run_dir, tx.transition("VERIFIED", workdir=workdir))

    def _revert_step(self, step, workdir: Path) -> None:
        run_dir = Path(workdir).parent
        transactions = self._revert_transactions(run_dir, step.step_id, workdir)

        # Legacy schema-1 runs remain traceable. Safe resume rejects them, but
        # this fallback preserves in-process rollback for an old caller.
        if not transactions:
            backup = self._backups.pop(step.step_id, None)
            if backup is None:
                backup = load_backup(run_dir / "backups" / f"{_safe_seg(step.step_id)}.json")
            if backup is not None:
                revert_patch(backup, workdir)

        rollback_step_id = getattr(step, "params", {}).get("rollback_step_id")
        if rollback_step_id and rollback_step_id != step.step_id:
            self._revert_transactions(run_dir, str(rollback_step_id), workdir)

    def _revert_transactions(self, run_dir: Path, step_id: str, workdir: Path) -> bool:
        transactions = list_transactions(run_dir, step_id)
        for tx in reversed(transactions):
            if tx.status == "REVERTED":
                continue
            backup = self._backups.pop(tx.attempt_id, None)
            if backup is None:
                backup, _degraded = self._transaction_backup(run_dir, tx)
            revert_patch(backup, workdir)
            save_transaction(run_dir, tx.transition("REVERTED"))
        return bool(transactions)

    def _patch_bytes(self, state: RunState, step) -> bytes | None:
        """The persisted patch.json bytes for this step (per-step file first)."""
        candidates = [
            Path(state.run_dir) / "steps" / _safe_seg(step.step_id) / "patch.json",
            Path(state.run_dir) / "patch.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return path.read_bytes()
                except OSError:
                    return None
        return None

    def _artifact_sha(self, state: RunState, step) -> str | None:
        """The reviewable artifact hash for an approval — patches only.

        Other actions regenerate their artifact on resume (a re-gathered bundle,
        a re-run experiment), so byte-binding them would livelock the gate;
        their approvals bind to the step id alone (``ApprovalDecision.binds``).
        """
        if step.action != "edit_code":
            return None
        data = self._patch_bytes(state, step)
        return sha256_bytes(data) if data is not None else None

    def _approved_patch(self, state: RunState, step) -> Patch | None:
        """The persisted patch a human already approved for this step, if any.

        Binds the approval to the artifact: on resume, an approval-gated step
        may only execute the exact bytes the human reviewed. A decision whose
        artifact hash no longer matches the persisted patch is tamper evidence
        and fails the run rather than executing an unreviewed change.
        """
        if not step.requires_approval:
            return None
        attempt_id = state.attempt_id(step)
        gate = HumanApprovalGate(state.run_dir)
        try:
            request = gate.request_evidence(step.step_id, attempt_id)
            decision_evidence = gate.decision_evidence(step.step_id, attempt_id)
        except ValueError as error:
            raise ApprovalRejected(
                f"invalid approval evidence for {step.step_id}: {error}"
            ) from error
        if request is None or decision_evidence is None:
            return None
        data = self._patch_bytes(state, step)
        if data is None:
            return None
        try:
            validate_decision_binding(
                request=request,
                decision=decision_evidence,
                step_id=step.step_id,
                attempt_id=attempt_id,
                goal=step.goal,
                artifact_sha256=sha256_bytes(data),
            )
        except ValueError as error:
            # The decision names this step but not these bytes: the artifact
            # changed after review. Never execute it; revert and fail closed.
            self._revert_step(step, Path(state.workdir))
            raise ApprovalRejected(
                f"approved artifact for step {step.step_id} does not match the "
                "persisted patch (hash mismatch) — refusing to execute"
            ) from error
        try:
            return Patch.model_validate_json(data)
        except Exception:
            return None

    # --- approval -----------------------------------------------------------
    def _approval_gate(self, state: RunState, step, artifact_ref: str) -> None:
        gate = HumanApprovalGate(state.run_dir)
        attempt_id = state.attempt_id(step)
        artifact_sha = self._artifact_sha(state, step)
        if step.action == "edit_code" and artifact_sha is None:
            # A patch step whose reviewed bytes are gone cannot be approved.
            self._revert_step(step, Path(state.workdir))
            raise ApprovalRejected(
                f"step {step.step_id}: the patch under review is missing — refusing"
            )
        try:
            gate.request(
                step,
                attempt_id,
                f"awaiting approval to apply {artifact_ref}",
                artifact_sha256=artifact_sha,
            )
            decision_evidence = gate.decision_evidence(
                step.step_id, attempt_id
            )
        except ValueError as error:
            self._revert_step(step, Path(state.workdir))
            raise ApprovalRejected(
                f"invalid approval evidence for step {step.step_id}: {error}"
            ) from error

        if decision_evidence is None and self.auto_approve:
            decision_evidence = gate.resolve(
                approved=True, note="automatically approved by run configuration"
            )
        elif (
            decision_evidence is None
            and self.interactive_approval
            and sys.stdin is not None
            and sys.stdin.isatty()
        ):
            ans = input(f"[approval] proceed with step {step.step_id} ({step.goal})? [y/N] ")
            approved = ans.strip().lower() in ("y", "yes")
            decision_evidence = gate.resolve(
                approved=approved,
                note="interactive approval" if approved else "interactive rejection",
            )

        if decision_evidence is not None:
            decision = self._consume_approval_decision(
                state, step, artifact_sha=artifact_sha
            )
            if not decision.approved:
                # A rejected change must not survive in the sandbox, same as a
                # change that fails verification.
                self._revert_step(step, Path(state.workdir))
                if step.step_id not in state.failed_steps:
                    state.fail_current(step)
                append_ledger(
                    state,
                    StepRecord(
                        seq=state.next_seq(),
                        step_id=step.step_id,
                        phase="fail",
                        attempt_id=attempt_id,
                        idempotency_key=f"{attempt_id}:fail",
                        notes="approval rejected",
                    ),
                )
                self._save(state)
                raise ApprovalRejected(
                    f"step {step.step_id} rejected: {decision.note}"
                )
            return

        # non-interactive: pause for async approval, binding the request to the
        # exact artifact bytes under review
        state.status = "AWAITING_APPROVAL"
        self._save(state)
        raise ApprovalPending(state.run_id, step.step_id)

    def _consume_approval_decision(
        self,
        state: RunState,
        step,
        *,
        artifact_sha: str | None,
    ) -> ApprovalDecision:
        """Validate and ledger-bind the immutable decision for one attempt."""
        gate = HumanApprovalGate(state.run_dir)
        attempt_id = state.attempt_id(step)
        try:
            request = gate.request_evidence(step.step_id, attempt_id)
            decision = gate.decision_evidence(step.step_id, attempt_id)
            if request is None or decision is None:
                raise ValueError("approval request or decision is missing")
            validate_decision_binding(
                request=request,
                decision=decision,
                step_id=step.step_id,
                attempt_id=attempt_id,
                goal=step.goal,
                artifact_sha256=artifact_sha,
            )
        except ValueError as error:
            raise ApprovalRejected(
                f"approval evidence for {step.step_id} is invalid: {error}"
            ) from error

        records = read_ledger(state.run_dir)
        if records:
            state.seq = max(state.seq, *(record.seq for record in records))
        note = decision.value.outcome or (
            "approved" if decision.value.approved else "rejected"
        )
        if decision.value.note:
            note = f"{note}: {decision.value.note}"
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id=step.step_id,
                phase="approval",
                artifact_ref=decision.reference,
                evidence_sha256=decision.sha256,
                attempt_id=attempt_id,
                idempotency_key=f"{attempt_id}:approval",
                notes=note,
            ),
        )
        gate.clear_transient()
        return decision.value

    # --- finalize -----------------------------------------------------------
    def _materializing_step_id(self, state: RunState, action: str) -> str | None:
        """The last completed step with this action — whose artifacts the finalizer
        should report. Robust to plan order rather than 'whichever file wrote last'."""
        if state.plan is None:
            return None
        done = set(state.completed_steps)
        ids = [s.step_id for s in state.plan.steps if s.action == action and s.step_id in done]
        return ids[-1] if ids else None

    def _load_artifact(self, state: RunState, step_id: str | None, name: str, model):
        """Load a per-step artifact (``steps/<id>/<name>``), falling back to the flat file."""
        run_dir = Path(state.run_dir)
        per_step = [run_dir / "steps" / _safe_seg(step_id) / name] if step_id else []
        candidates = per_step + [run_dir / name]
        for path in candidates:
            if path.exists():
                try:
                    return model.model_validate_json(path.read_text())
                except Exception:
                    continue
        return None

    def _finalize_pr(self, state: RunState) -> None:
        run_dir = Path(state.run_dir)
        sid = self._materializing_step_id(state, "edit_code")
        patch = self._load_artifact(state, sid, "patch.json", Patch) or Patch(step_id="-")
        verdict = self._load_artifact(state, sid, "verify.json", Verdict)
        manifest = self._load_artifact(state, sid, "manifest.json", ArtifactManifest)

        checks: list[str] = []
        if state.task.inputs.get("repo_adapter") and state.plan is not None:
            for step in state.plan.steps:
                if step.step_id not in state.completed_steps:
                    continue
                step_verdict = self._load_artifact(
                    state,
                    step.step_id,
                    "verify.json",
                    Verdict,
                )
                if step_verdict is None:
                    continue
                for check in step_verdict.checks:
                    checks.append(
                        f"{step.step_id}/{check.name}: "
                        f"{check.detail.get('summary', 'passed' if check.passed else 'failed')}"
                    )
        elif verdict:
            for check in verdict.checks:
                checks.append(
                    f"{check.name}: "
                    f"{check.detail.get('summary', 'passed' if check.passed else 'failed')}"
                )
        pr = PRSummary(
            title=state.task.title,
            task_id=state.run_id,
            rationale=patch.rationale or state.task.description or state.task.title,
            files_changed=manifest.touched_files if manifest is not None else patch.touched_files,
            checks=checks,
            provenance=patch.based_on_context,
        )
        path = run_dir / "pr_summary.md"
        path.write_text(pr.to_markdown())
        state.pr_summary_path = str(path)

    def _finalize_experiment(self, state: RunState) -> None:
        run_dir = Path(state.run_dir)
        sid = self._materializing_step_id(state, "run_experiment")
        result = self._load_artifact(state, sid, "experiment.json", ExperimentResult) or (
            ExperimentResult(step_id="-")
        )
        verdict = self._load_artifact(state, sid, "verify.json", Verdict)

        # Prefer verifier-recomputed metrics (authoritative) over self-reported.
        metrics: dict[str, float] = dict(result.metrics)
        checks: list[str] = []
        reproducible = False
        if verdict:
            for c in verdict.checks:
                checks.append(
                    f"{c.name}: {c.detail.get('summary', 'passed' if c.passed else 'failed')}"
                )
                if c.name in ("psnr", "ssim") and c.score is not None:
                    metrics[c.name] = round(c.score, 6)
                if c.name == "reproducibility":
                    reproducible = c.passed

        summary = ExperimentSummary(
            title=state.task.title,
            task_id=state.run_id,
            metrics=metrics,
            checks=checks,
            reproducible=reproducible,
            provenance=result.based_on_context,
        )
        path = run_dir / "experiment_summary.md"
        path.write_text(summary.to_markdown())
        state.pr_summary_path = str(path)

    # --- helpers ------------------------------------------------------------
    def _prepare_workdir(self, task: TaskSpec, workdir: Path) -> None:
        if workdir.exists():
            return  # resume / re-run reuse
        if task.target_repo:
            src = Path(task.target_repo)
            if not src.exists():
                raise FileNotFoundError(f"target_repo not found: {src}")
            self._reject_repo_symlinks(src)
            # Preserve links as links during the copy so a source-tree race can
            # never dereference an external target. The post-copy scan then
            # rejects the copied link and the run fails before any model call.
            shutil.copytree(src, workdir, ignore=_IGNORE, symlinks=True)
            try:
                self._reject_repo_symlinks(workdir)
            except Exception:
                shutil.rmtree(workdir)
                raise
        else:
            workdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _reject_repo_symlinks(root: Path) -> None:
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"target_repo must be a regular directory: {root}")
        for directory, names, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            for name in [*names, *files]:
                path = parent / name
                if path.is_symlink():
                    raise ValueError(
                        f"target_repo contains a symbolic link: "
                        f"{path.relative_to(root)}"
                    )

    @staticmethod
    def _plan_md(state: RunState) -> str:
        plan = state.plan
        assert plan is not None  # only called after planning
        lines = [f"# Plan — {plan.summary}", ""]
        for i, s in enumerate(plan.steps):
            lines.append(f"{i + 1}. **{s.step_id}** ({s.action}) — {s.goal}")
            if s.verifiers:
                lines.append(f"   - verifiers: {', '.join(s.verifiers)}")
        return "\n".join(lines)
