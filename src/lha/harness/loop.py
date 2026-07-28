"""The verification loop.

    context -> tool/execute -> verify -> repair -> checkpoint -> repeat

with max-steps, checkpoint/resume, and a human-approval gate. No LangGraph; the
state is a JSON checkpoint + a step ledger, shaped so LangGraph drops in later.
"""

from __future__ import annotations

import json
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
from ..durable_io import (
    anchored_atomic_replace_bytes,
    anchored_read_bytes,
    anchored_write_once_bytes,
    atomic_replace_text,
    durable_mkdir_chain,
    fsync_directory,
)
from ..live_context.models import ContextBundle
from ..llm import get_llm
from ..oracle_inventory import (
    OracleInventoryError,
    build_pytest_oracle_inventory,
    validate_pytest_oracle_inventory,
)
from ..oracle_models import PytestOracleInventory
from ..repo_adapter import (
    RepoAdapterSpec,
    RepoIntegrityResult,
    RepoReferenceManifest,
    RepoStageRequest,
    execute_repo_stage_once,
    inspect_repo_integrity,
)
from ..sandbox import ProcessCleanupUnconfirmed
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
from ..verifiers.verdict import (
    PROCESS_CLEANUP_UNCONFIRMED,
    Check,
    Verdict,
    verdict_requires_process_quarantine,
)
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
from .state import (
    RUN_STATE_SCHEMA,
    LLMUsageState,
    RunQuarantine,
    RunState,
    StepRecord,
)
from .transaction import (
    PatchTransaction,
    attempt_artifact_dir,
    build_transaction,
    list_transactions,
    load_transaction,
    recover_transaction_journals,
    resolve_transaction_evidence,
    save_transaction,
    state_for_paths,
    validate_applied_state,
    validate_terminal_transaction_state,
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
    step_dir = run_dir / "steps" / segment
    durable_mkdir_chain(step_dir, anchor=run_dir)
    atomic_replace_text(run_dir / name, text, anchor=run_dir)
    atomic_replace_text(step_dir / name, text, anchor=run_dir)


def _read_run_bytes(
    run_dir: Path,
    path: Path,
    *,
    label: str,
    missing_ok: bool = False,
) -> bytes | None:
    """Read run-owned evidence without following links or accepting hardlinks."""
    try:
        return anchored_read_bytes(
            path,
            anchor=run_dir,
            missing_ok=missing_ok,
        )
    except (OSError, ValueError) as error:
        raise CheckpointCorrupt(f"{label} is missing or unsafe: {path}") from error


def _write_immutable(
    path: Path,
    data: bytes,
    *,
    run_dir: Path,
) -> None:
    """Create one attempt artifact once; a replay may only present identical bytes."""
    try:
        anchored_write_once_bytes(path, data, anchor=run_dir)
    except (OSError, ValueError) as error:
        raise CheckpointCorrupt(
            f"immutable artifact changed or is unsafe: {path}"
        ) from error


def _validate_optional_aliases(
    run_dir: Path,
    authoritative: bytes,
    aliases: list[Path],
    *,
    label: str,
) -> None:
    """Treat compatibility files as diagnostics, never as recovery authority."""
    for alias in aliases:
        alias_bytes = _read_run_bytes(
            run_dir,
            alias,
            label=f"{label} alias",
            missing_ok=True,
        )
        if alias_bytes is None:
            continue
        if alias_bytes != authoritative:
            raise CheckpointCorrupt(
                f"{label} alias does not match immutable attempt evidence "
                f"(hash mismatch): {alias}"
            )


def _verdict_ref(attempt_id: str) -> str:
    return (Path("attempts") / _safe_seg(attempt_id) / "verify.json").as_posix()


def _attempt_ref(attempt_id: str, name: str) -> str:
    return (Path("attempts") / _safe_seg(attempt_id) / name).as_posix()


def _initial_plan_ref() -> str:
    return (Path("plans") / "initial.json").as_posix()


def _plan_failure_ref() -> str:
    return (Path("plans") / "failure.json").as_posix()


def _plan_failure_bytes(error_type: str) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "phase": "plan",
            "error_type": error_type,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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
    _write_immutable(path, verdict_json.encode("utf-8"), run_dir=run_dir)
    _dump(run_dir, step_id, "verify.json", verdict_json)
    return reference


def _gen_run_id(task: TaskSpec) -> str:
    return f"{now():%Y%m%d-%H%M%S}-{_slug(task.title)}-{uuid.uuid4().hex[:4]}"


def _claim_run_dir(runs_dir: str | Path, run_id: str) -> Path:
    """Atomically reserve a run id before copying or checkpointing anything."""
    validate_run_id(run_id)
    root = durable_mkdir_chain(runs_dir)
    run_dir = root / run_id
    try:
        run_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"run already exists: {run_id}") from error
    fsync_directory(run_dir)
    fsync_directory(root)
    return run_dir


def _discard_uninitialized_run_dir(run_dir: Path) -> None:
    """Remove a reservation that never reached its first durable checkpoint."""
    if (run_dir / "state.json").exists():
        return
    try:
        shutil.rmtree(run_dir)
    except FileNotFoundError:
        pass


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

    def _bind_execution_backend(self, run_dir: str | Path) -> None:
        """Bind subprocess recovery to the run, independent of command cwd."""
        expected = Path(run_dir).resolve()
        bound = self.exec.bind_operation_lease_dir(run_dir)
        if bound != expected:
            raise CheckpointCorrupt(
                "execution backend did not bind the requested operation lease directory"
            )

    @staticmethod
    def _validate_terminal_resume(
        state: RunState,
        records: list[StepRecord],
    ) -> None:
        """Fail closed before returning a terminal checkpoint to a caller."""
        from ..reporting import (
            ReportingError,
            validate_terminal_resume_evidence,
        )

        try:
            validate_terminal_resume_evidence(
                Path(state.run_dir),
                state,
                records,
            )
        except ReportingError as error:
            raise CheckpointCorrupt(
                f"terminal run evidence is invalid: {error}"
            ) from error

    def _resume_preflight(self, state: RunState) -> RunResult | None:
        """Reap old commands and validate baseline tests before recovery mutates state."""
        try:
            recovery_backend = self.exec
            recorded = state.runtime_contract
            if recorded is not None:
                recorded.verify_recorded_docker_control()
            if recorded is not None and (
                recorded.exec_backend == "docker"
                or recorded.exec_backend != getattr(self.exec, "name", None)
            ):
                from ..sandbox import make_backend

                if recorded.exec_backend == "docker":
                    recovery_backend = make_backend(
                        "docker",
                        image=recorded.exec_image_id or recorded.exec_image,
                    )
                else:
                    recovery_backend = make_backend(recorded.exec_backend)
            if (
                recorded is not None
                and recorded.exec_backend == "docker"
                and recorded.docker_cli is not None
            ):
                setattr(recovery_backend, "docker", recorded.docker_cli.path)
            recovery = recovery_backend.recover_active_operations(state.run_dir)
        except Exception as error:
            detail = (
                "active-operation recovery failed closed: "
                f"{type(error).__name__}: {error}"
            )
            state.quarantine = RunQuarantine(
                kind="active_operation_recovery_unconfirmed",
                step_id="-",
                attempt_id="resume",
                detail=detail,
            )
            state.status = "PAUSED"
            save_state(state)
            return RunResult(state, "PAUSED", detail)
        if recovery.requires_quarantine:
            detail = recovery.detail or (
                "one or more active operations could not be confirmed stopped"
            )
            state.quarantine = RunQuarantine(
                kind="active_operation_recovery_unconfirmed",
                step_id="-",
                attempt_id="resume",
                detail=detail[-1000:],
            )
            state.status = "PAUSED"
            save_state(state)
            return RunResult(state, "PAUSED", detail)
        try:
            self._validate_pytest_oracle_inventories(state)
        except (OSError, OracleInventoryError, CheckpointCorrupt) as error:
            detail = f"pytest oracle inventory is invalid: {error}"
            state.quarantine = RunQuarantine(
                kind="pytest_oracle_inventory_invalid",
                step_id="-",
                attempt_id="resume",
                detail=detail[-1000:],
            )
            state.status = "PAUSED"
            save_state(state)
            return RunResult(state, "PAUSED", detail)
        return None

    @staticmethod
    def _oracle_inventory_ref(step_id: str) -> Path:
        return Path("oracle_inventories") / f"{_safe_seg(step_id)}.json"

    def _validate_pytest_oracle_inventories(
        self,
        state: RunState,
    ) -> None:
        run_dir = Path(state.run_dir)
        inventory_dir = run_dir / "oracle_inventories"
        expected_names = {
            self._oracle_inventory_ref(step_id).name
            for step_id in state.pytest_oracle_inventories
        }
        if inventory_dir.is_symlink() or (
            inventory_dir.exists() and not inventory_dir.is_dir()
        ):
            raise CheckpointCorrupt(
                f"pytest oracle inventory directory is unsafe: {inventory_dir}"
            )
        if inventory_dir.exists():
            actual_names = {
                path.name
                for path in inventory_dir.iterdir()
                if path.is_file() or path.is_symlink()
            }
            if actual_names != expected_names:
                raise CheckpointCorrupt(
                    "pytest oracle inventory files do not match RunState"
                )
        elif expected_names:
            raise CheckpointCorrupt("pytest oracle inventory directory is missing")
        for step_id, inventory in state.pytest_oracle_inventories.items():
            path = run_dir / self._oracle_inventory_ref(step_id)
            expected = inventory.model_dump_json(indent=2).encode("utf-8")
            actual = _read_run_bytes(
                run_dir,
                path,
                label="pytest oracle inventory",
                missing_ok=True,
            )
            if actual != expected:
                raise CheckpointCorrupt(
                    f"immutable pytest oracle inventory changed for {step_id}"
                )
            validate_pytest_oracle_inventory(
                state.workdir,
                inventory,
                allowed_changes=tuple(
                    state.task.allowed_protected_files
                ),
            )

    def _pytest_oracle_inventory(
        self,
        state: RunState,
        step,
        *,
        allow_build: bool,
    ) -> PytestOracleInventory | None:
        if "pytest" not in step.verifiers:
            return None
        inventory = state.pytest_oracle_inventories.get(step.step_id)
        if inventory is None:
            if not allow_build:
                raise CheckpointCorrupt(
                    f"pytest oracle inventory is missing for {step.step_id}"
                )
            inventory = build_pytest_oracle_inventory(
                state.workdir,
                self.exec,
                timeout=float(step.params.get("timeout", 300.0)),
            )
            path = (
                Path(state.run_dir)
                / self._oracle_inventory_ref(step.step_id)
            )
            data = inventory.model_dump_json(indent=2).encode("utf-8")
            _write_immutable(path, data, run_dir=Path(state.run_dir))
            state.pytest_oracle_inventories[step.step_id] = inventory
            # This checkpoint precedes every implementation-model call.
            self._save(state)
        else:
            validate_pytest_oracle_inventory(
                state.workdir,
                inventory,
                allowed_changes=tuple(
                    state.task.allowed_protected_files
                ),
            )
            path = (
                Path(state.run_dir)
                / self._oracle_inventory_ref(step.step_id)
            )
            actual = _read_run_bytes(
                Path(state.run_dir),
                path,
                label="pytest oracle inventory",
                missing_ok=True,
            )
            if actual != inventory.model_dump_json(indent=2).encode("utf-8"):
                raise CheckpointCorrupt(
                    f"immutable pytest oracle inventory changed for {step.step_id}"
                )
        set_trusted_oracle_paths = getattr(
            self.llm,
            "set_trusted_oracle_paths",
            None,
        )
        if callable(set_trusted_oracle_paths):
            set_trusted_oracle_paths(inventory)
        return inventory

    # --- public entry points ------------------------------------------------
    def run(self, task: TaskSpec, *, run_id: str | None = None) -> RunResult:
        run_id = run_id or _gen_run_id(task)
        run_dir = _claim_run_dir(self.config.runs_dir, run_id)
        try:
            with run_lock(run_dir):
                self._bind_execution_backend(run_dir)
                workdir = run_dir / "workdir"
                self._prepare_workdir(task, workdir)
                state = RunState.new(
                    task,
                    run_id,
                    str(run_dir),
                    str(workdir),
                    config=self.config,
                    runtime="loop",
                    exec_backend=self.exec,
                    llm=self.llm,
                )
                save_state(state)
                return self._drive(state)
        except BaseException:
            _discard_uninitialized_run_dir(run_dir)
            raise

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
            self._bind_execution_backend(run_dir)
            blocked = self._resume_preflight(state)
            if blocked is not None:
                return blocked
            state.require_matching_runtime_contract(
                self.config,
                runtime="loop",
                exec_backend=self.exec,
                llm=self.llm,
            )
            limits = state.require_matching_budget_limits(self.config)
            if state.quarantine is not None:
                state.status = "PAUSED"
                save_state(state)
                return RunResult(
                    state,
                    "PAUSED",
                    state.quarantine.detail,
                )
            records = read_ledger(state.run_dir)
            try:
                recover_transaction_journals(Path(state.run_dir))
                validate_transaction_journals(Path(state.run_dir))
                if state.status in ("DONE", "FAILED"):
                    validate_terminal_transaction_state(
                        Path(state.run_dir),
                        Path(state.workdir),
                        state.status,
                    )
            except TransactionCorrupt as error:
                raise TransactionCorrupt(
                    f"run recovery evidence is invalid: {error}"
                ) from error
            if state.status in ("DONE", "FAILED"):
                self._validate_terminal_resume(state, records)
                return RunResult(state, state.status, "run already terminal")
            state.recover_active_elapsed()
            if records:
                state.seq = max(state.seq, *(record.seq for record in records))
            records = self._reconcile_durable_ledger(state, records)
            if records:
                # A ledger append is durable before the following state save.
                # Preserve its sequence number after a crash so replayed work
                # cannot create a second event with an already-used sequence.
                state.seq = max(state.seq, *(record.seq for record in records))
            if state.quarantine is not None:
                state.status = "PAUSED"
                save_state(state)
                return RunResult(
                    state,
                    "PAUSED",
                    state.quarantine.detail,
                )
            if state.is_terminal():
                self._validate_terminal_resume(state, records)
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

    def _record_plan_failure(
        self,
        state: RunState,
        error_type: str,
    ) -> None:
        run_dir = Path(state.run_dir)
        payload = _plan_failure_bytes(error_type)
        failure_ref = _plan_failure_ref()
        _write_immutable(
            run_dir / failure_ref,
            payload,
            run_dir=run_dir,
        )
        fail_key = f"{state.run_id}:plan-fail"
        if not any(
            record.idempotency_key == fail_key
            for record in read_ledger(run_dir)
        ):
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id="-",
                    phase="fail",
                    artifact_ref=failure_ref,
                    evidence_sha256=sha256_bytes(payload),
                    notes=f"error: {error_type}",
                    attempt_id="plan",
                    idempotency_key=fail_key,
                ),
            )
        state.status = "FAILED"

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
        plan_failure_events = [
            record
            for record in records
            if record.phase == "fail"
            and record.step_id == "-"
            and record.attempt_id == "plan"
        ]
        initial_path = run_dir / _initial_plan_ref()
        if not plan_events:
            if plan_failure_events:
                if len(plan_failure_events) != 1:
                    raise CheckpointCorrupt(
                        "ledger contains multiple plan failure events"
                    )
                event = plan_failure_events[0]
                failure_path = run_dir / _plan_failure_ref()
                failure_bytes = _read_run_bytes(
                    run_dir,
                    failure_path,
                    label="immutable plan failure",
                    missing_ok=True,
                )
                if (
                    event.artifact_ref != _plan_failure_ref()
                    or event.evidence_sha256 is None
                    or event.idempotency_key != f"{state.run_id}:plan-fail"
                    or failure_bytes is None
                    or sha256_bytes(failure_bytes) != event.evidence_sha256
                ):
                    raise CheckpointCorrupt(
                        "plan failure event is not bound to immutable evidence"
                    )
                try:
                    failure = json.loads(failure_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CheckpointCorrupt(
                        "immutable plan failure is invalid"
                    ) from error
                if (
                    not isinstance(failure, dict)
                    or set(failure) != {
                        "schema_version",
                        "phase",
                        "error_type",
                    }
                    or failure.get("schema_version") != 1
                    or failure.get("phase") != "plan"
                    or not isinstance(failure.get("error_type"), str)
                    or not failure["error_type"]
                    or event.notes != f"error: {failure['error_type']}"
                ):
                    raise CheckpointCorrupt(
                        "immutable plan failure identity is invalid"
                    )
                if (
                    state.plan is not None
                    or plan_path.exists()
                    or plan_path.is_symlink()
                    or initial_path.exists()
                    or initial_path.is_symlink()
                ):
                    raise CheckpointCorrupt(
                        "plan failure conflicts with successful plan evidence"
                    )
                state.status = "FAILED"
                state.active_since = None
                state.seq = max(state.seq, event.seq)
                return records
            orphan_failure_path = run_dir / _plan_failure_ref()
            orphan_failure_bytes = _read_run_bytes(
                run_dir,
                orphan_failure_path,
                label="orphan immutable plan failure",
                missing_ok=True,
            )
            if orphan_failure_bytes is not None:
                if (
                    records
                    or state.plan is not None
                    or plan_path.exists()
                    or plan_path.is_symlink()
                    or initial_path.exists()
                    or initial_path.is_symlink()
                ):
                    raise CheckpointCorrupt(
                        "orphan plan failure conflicts with other plan evidence"
                    )
                try:
                    failure = json.loads(orphan_failure_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CheckpointCorrupt(
                        "orphan plan failure is invalid"
                    ) from error
                if (
                    not isinstance(failure, dict)
                    or set(failure) != {
                        "schema_version",
                        "phase",
                        "error_type",
                    }
                    or failure.get("schema_version") != 1
                    or failure.get("phase") != "plan"
                    or not isinstance(failure.get("error_type"), str)
                    or not failure["error_type"]
                    or orphan_failure_bytes
                    != _plan_failure_bytes(failure["error_type"])
                ):
                    raise CheckpointCorrupt(
                        "orphan plan failure identity is invalid"
                    )
                self._record_plan_failure(
                    state,
                    failure["error_type"],
                )
                return self._reconcile_durable_ledger(
                    state,
                    read_ledger(run_dir),
                )
            initial_bytes = _read_run_bytes(
                run_dir,
                initial_path,
                label="orphan immutable initial plan",
                missing_ok=True,
            )
            if initial_bytes is not None:
                if records:
                    raise CheckpointCorrupt(
                        "orphan initial plan conflicts with other ledger events"
                    )
                try:
                    initial_plan = Plan.model_validate_json(initial_bytes)
                except Exception as error:
                    raise CheckpointCorrupt(
                        f"orphan initial plan is invalid: {error}"
                    ) from error
                if state.plan is not None and state.plan != initial_plan:
                    raise CheckpointCorrupt(
                        "checkpoint plan differs from orphan initial plan"
                    )
                alias_bytes = _read_run_bytes(
                    run_dir,
                    plan_path,
                    label="orphan initial plan alias",
                    missing_ok=True,
                )
                if alias_bytes is not None and alias_bytes != initial_bytes:
                    raise CheckpointCorrupt(
                        "orphan initial plan alias changed before recovery"
                    )
                if alias_bytes is None:
                    anchored_atomic_replace_bytes(
                        plan_path,
                        initial_bytes,
                        anchor=run_dir,
                    )
                state.plan = initial_plan
                atomic_replace_text(
                    run_dir / "plan.md",
                    self._plan_md(state),
                    anchor=run_dir,
                )
                append_ledger(
                    state,
                    StepRecord(
                        seq=state.next_seq(),
                        step_id="-",
                        phase="plan",
                        artifact_ref=_initial_plan_ref(),
                        evidence_sha256=sha256_bytes(initial_bytes),
                        idempotency_key=f"{state.run_id}:plan",
                    ),
                )
                return self._reconcile_durable_ledger(
                    state,
                    read_ledger(run_dir),
                )
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
        initial_bytes = _read_run_bytes(
            run_dir,
            initial_path,
            label="immutable initial plan",
            missing_ok=True,
        )
        if (
            event.artifact_ref != _initial_plan_ref()
            or event.evidence_sha256 is None
            or initial_bytes is None
            or sha256_bytes(initial_bytes) != event.evidence_sha256
        ):
            raise CheckpointCorrupt(
                "durable plan event is not bound to the immutable initial plan"
            )
        try:
            initial_plan = Plan.model_validate_json(initial_bytes)
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
        checkpoint_quarantine = state.quarantine

        # The immutable initial plan plus ledger-bound repair snapshots are the
        # source of truth. Replaying them prevents a consistently edited
        # state.json + plan.json pair from deleting unfinished work.
        state.plan = initial_plan
        state.cursor = 0
        state.completed_steps = []
        state.failed_steps = []
        state.repairs = {}
        state.attempt_ids = {}
        state.quarantine = None
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
                    verdict_bytes = _read_run_bytes(
                        run_dir,
                        orphan_path,
                        label="orphan verdict",
                    )
                    assert verdict_bytes is not None
                    if not any(
                        record.phase == "context"
                        for record in attempt_records
                    ):
                        raise CheckpointCorrupt(
                            f"orphan verdict has no prepared attempt: {attempt_id}"
                        )
                    try:
                        orphan = Verdict.model_validate_json(verdict_bytes)
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
                    alias_bytes = _read_run_bytes(
                        run_dir,
                        alias,
                        label="orphan verdict alias",
                        missing_ok=True,
                    )
                    if alias_bytes not in (None, verdict_bytes):
                        prior_records = sorted(
                            (
                                record
                                for record in records
                                if record.phase == "verify"
                                and record.step_id == step.step_id
                                and record.attempt_id != attempt_id
                                and record.verdict_ref is not None
                            ),
                            key=lambda record: record.seq,
                        )
                        prior_bytes = None
                        if prior_records:
                            prior = prior_records[-1]
                            prior_ref = prior.verdict_ref
                            assert prior_ref is not None
                            prior_path = (
                                run_dir
                                / "steps"
                                / _safe_seg(step.step_id)
                                / prior_ref
                            )
                            prior_bytes = _read_run_bytes(
                                run_dir,
                                prior_path,
                                label="prior immutable verdict",
                            )
                            if (
                                prior.evidence_sha256 is None
                                or prior_bytes is None
                                or sha256_bytes(prior_bytes)
                                != prior.evidence_sha256
                            ):
                                raise CheckpointCorrupt(
                                    "prior verdict is not bound to its ledger event"
                                )
                        if alias_bytes != prior_bytes:
                            raise CheckpointCorrupt(
                                f"orphan verdict alias changed for {attempt_id}"
                            )
                    anchored_atomic_replace_bytes(
                        alias,
                        verdict_bytes,
                        anchor=run_dir,
                    )
                    anchored_atomic_replace_bytes(
                        run_dir / "verify.json",
                        verdict_bytes,
                        anchor=run_dir,
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
            if verdict_requires_process_quarantine(verdict):
                if terminal is not None:
                    raise CheckpointCorrupt(
                        f"cleanup-failure attempt {attempt_id} has an unsafe "
                        f"{terminal.phase} transition"
                    )
                self._quarantine_process_failure(state, step, verdict)
                break
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
                    if state.quarantine is not None:
                        break
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

        if (
            checkpoint_quarantine is not None
            and checkpoint_quarantine != state.quarantine
        ):
            raise CheckpointCorrupt(
                "checkpoint quarantine is not bound to a cleanup-failure verdict"
            )

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

        plan_bytes = _read_run_bytes(
            run_dir,
            plan_path,
            label="plan history alias",
            missing_ok=True,
        )
        if plan_bytes is None:
            raise CheckpointCorrupt("checksummed plan history has no safe plan.json")
        try:
            disk_plan = Plan.model_validate_json(plan_bytes)
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
                patch_bytes = _read_run_bytes(
                    Path(state.run_dir),
                    patch_path,
                    label="reviewed attempt patch",
                    missing_ok=True,
                )
                if patch_bytes is None:
                    raise ValueError(
                        "reviewed attempt patch is missing or unsafe"
                    )
                artifact_sha256 = sha256_bytes(patch_bytes)
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
        run_dir = Path(state.run_dir)
        expected_ref = _verdict_ref(attempt_id)
        path = step_dir / expected_ref
        verdict_bytes = _read_run_bytes(
            run_dir,
            path,
            label="durable verdict",
            missing_ok=True,
        )
        if (
            record.verdict_ref != expected_ref
            or record.evidence_sha256 is None
            or verdict_bytes is None
            or sha256_bytes(verdict_bytes) != record.evidence_sha256
        ):
            raise CheckpointCorrupt(
                f"durable verdict is missing or changed for {attempt_id}"
            )
        try:
            verdict = Verdict.model_validate_json(verdict_bytes)
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
        artifact_bytes = _read_run_bytes(
            run_dir,
            artifact_path,
            label="durable verdict artifact",
            missing_ok=True,
        )
        if (
            verdict.artifact_sha256 is None
            or artifact_bytes is None
            or sha256_bytes(artifact_bytes) != verdict.artifact_sha256
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
        run_dir = Path(state.run_dir)
        path = run_dir / expected_ref
        plan_bytes = _read_run_bytes(
            run_dir,
            path,
            label="immutable repair plan",
            missing_ok=True,
        )
        if (
            record.artifact_ref != expected_ref
            or record.evidence_sha256 is None
            or plan_bytes is None
            or sha256_bytes(plan_bytes) != record.evidence_sha256
        ):
            raise CheckpointCorrupt(
                "repair event is not bound to its immutable plan for "
                f"{record.attempt_id}"
            )
        try:
            disk_plan = Plan.model_validate_json(plan_bytes)
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
        self._bind_execution_backend(run_dir)
        from ..llm.trace import TracedLLM

        if isinstance(self.llm, TracedLLM):
            self.llm.bind(run_dir)  # per-call records land in llm_trace.jsonl
            self.llm.restore_totals(state.llm_usage)

        # A run's context is isolated from later edits to the source repository.
        # Never point an index at the enclosing run directory: it also contains
        # approvals, model traces, backups, and other non-code evidence.
        code_root = str(workdir)
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
                _write_immutable(
                    run_dir / initial_ref,
                    plan_bytes,
                    run_dir=run_dir,
                )
                anchored_atomic_replace_bytes(
                    run_dir / "plan.json",
                    plan_bytes,
                    anchor=run_dir,
                )
                atomic_replace_text(
                    run_dir / "plan.md",
                    self._plan_md(state),
                    anchor=run_dir,
                )
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
                budget.check_deadline()
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
                if state.quarantine is not None:
                    state.active_since = None
                    state.elapsed_s = budget.elapsed()
                    state.status = "PAUSED"
                    self._save(state)
                    return RunResult(
                        state,
                        "PAUSED",
                        state.quarantine.detail,
                    )
                # A last step may consume the remaining wall-clock budget. Check
                # before committing a terminal status, not only before the next
                # iteration that may never exist.
                budget.check_deadline()
                state.active_since = None
                state.elapsed_s = budget.elapsed()
                self._save(state)
            budget.check_deadline()
        except BudgetExceeded as e:
            state.steps_used = budget.steps_used
            state.elapsed_s = budget.elapsed()
            state.active_since = None
            state.status = "PAUSED"
            self._save(state)
            return RunResult(state, "PAUSED", str(e))
        except ProcessCleanupUnconfirmed as error:
            state.steps_used = budget.steps_used
            state.elapsed_s = budget.elapsed()
            state.active_since = None
            step = in_flight or state.next_step()
            self._quarantine_cleanup_interruption(state, step, error)
            self._save(state)
            return RunResult(state, "PAUSED", str(error))
        except ApprovalPending as e:
            state.elapsed_s = budget.elapsed()
            state.active_since = None
            self._save(state)
            return RunResult(state, "AWAITING_APPROVAL", str(e))
        except ApprovalRejected as e:
            state.elapsed_s = budget.elapsed()
            state.active_since = None
            rollback_error: Exception | None = None
            if (
                in_flight is not None
                and in_flight.step_id not in state.completed_steps
            ):
                try:
                    self._revert_step(in_flight, workdir)
                except Exception as error:
                    rollback_error = error
                    logger.exception(
                        "revert failed for step %s", in_flight.step_id
                    )
            if rollback_error is not None:
                state.status = "PAUSED"
                self._save(state)
                return RunResult(
                    state,
                    "PAUSED",
                    f"{type(e).__name__}: {e}; rollback incomplete: "
                    f"{type(rollback_error).__name__}: {rollback_error}",
                )
            state.status = "FAILED"
            self._save(state)
            return RunResult(state, "FAILED", str(e))
        except Exception as e:
            # An unexpected fault mid-step (unknown action, failed patch apply, a
            # crashing tool) must not leave the run wedged at RUNNING with a
            # half-applied sandbox: revert the in-flight step, fail closed, checkpoint.
            if in_flight is not None and in_flight.step_id not in state.completed_steps:
                rollback_error: Exception | None = None
                try:
                    self._revert_step(in_flight, workdir)
                except Exception as error:
                    rollback_error = error
                    logger.exception("revert failed for step %s", in_flight.step_id)
                if rollback_error is not None:
                    state.elapsed_s = budget.elapsed()
                    state.active_since = None
                    state.status = "PAUSED"
                    self._save(state)
                    return RunResult(
                        state,
                        "PAUSED",
                        f"{type(e).__name__}: {e}; rollback incomplete: "
                        f"{type(rollback_error).__name__}: {rollback_error}",
                    )
                attempt_id = state.attempt_id(in_flight)
                append_ledger(
                    state,
                    StepRecord(
                        seq=state.next_seq(),
                        step_id=in_flight.step_id,
                        phase="fail",
                        notes=f"error: {type(e).__name__}",
                        attempt_id=attempt_id,
                        idempotency_key=f"{attempt_id}:fail",
                    ),
                )
                state.fail_current(in_flight)
            else:
                # No step in flight, or the step itself completed and only its
                # bookkeeping failed — never revert verified work.
                if state.plan is None:
                    self._record_plan_failure(state, type(e).__name__)
                else:
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
        committed_integrity = (
            self._committed_repo_integrity(
                state,
                step,
                attempt_id,
            )
            if step.action == "repo_integrity"
            else None
        )
        try:
            if committed_integrity is None:
                artifact, artifact_ref = self._execute(state, step, bundle)
            else:
                artifact = committed_integrity
                artifact_ref = "repo_integrity.json"
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
                require_transaction=False,
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
        replayed_execute = committed_integrity is not None
        if not replayed_execute:
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
                pytest_oracle_inventory=state.pytest_oracle_inventories.get(
                    step.step_id
                ),
                allowed_oracle_changes=tuple(
                    state.task.allowed_protected_files
                ),
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
        existing = _read_run_bytes(
            run_dir,
            path,
            label="context evidence",
            missing_ok=True,
        )
        if records or existing is not None:
            if existing is None:
                raise CheckpointCorrupt(
                    f"context evidence is missing or unsafe: {path}"
                )
            data = existing
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

        # Refresh the exact per-run code root for each new context attempt.
        # The enclosing runs/<id> directory is intentionally never indexed.
        live_context.configure(code_root=workdir, config=self.config)
        try:
            live_context.index_code(workdir)
        except Exception:
            logger.debug("index_code(%s) failed", workdir, exc_info=True)
        bundle = ContextEngineer(self.config).gather(step, workdir=workdir)
        data = bundle.model_dump_json(indent=2).encode("utf-8")
        _write_immutable(path, data, run_dir=run_dir)
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
    def _committed_repo_integrity(
        state: RunState,
        step,
        attempt_id: str,
    ) -> RepoIntegrityResult | None:
        run_dir = Path(state.run_dir)
        reference = _attempt_ref(attempt_id, "repo_integrity.json")
        path = (
            run_dir
            / "steps"
            / _safe_seg(step.step_id)
            / reference
        )
        records = [
            record
            for record in read_ledger(run_dir)
            if record.phase == "execute"
            and record.step_id == step.step_id
            and record.attempt_id == attempt_id
        ]
        if len(records) > 1:
            raise CheckpointCorrupt(
                f"attempt {attempt_id} has multiple execute events"
            )
        data = _read_run_bytes(
            run_dir,
            path,
            label="immutable repository integrity evidence",
            missing_ok=True,
        )
        if not records and data is None:
            return None
        if data is None:
            raise CheckpointCorrupt(
                f"repository integrity evidence is missing for {attempt_id}"
            )
        digest = sha256_bytes(data)
        if records:
            record = records[0]
            if (
                record.artifact_ref != reference
                or record.evidence_sha256 != digest
                or record.idempotency_key != f"{attempt_id}:execute"
            ):
                raise CheckpointCorrupt(
                    f"repository integrity execute event is invalid for {attempt_id}"
                )
        else:
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id=step.step_id,
                    phase="execute",
                    artifact_ref=reference,
                    evidence_sha256=digest,
                    attempt_id=attempt_id,
                    idempotency_key=f"{attempt_id}:execute",
                    notes=(
                        "recovered repository integrity evidence after "
                        "interrupted ledger append"
                    ),
                ),
            )
        try:
            return RepoIntegrityResult.model_validate_json(data)
        except Exception as error:
            raise CheckpointCorrupt(
                f"repository integrity evidence is invalid for {attempt_id}"
            ) from error

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
        run_dir = Path(state.run_dir)
        path_data = _read_run_bytes(
            run_dir,
            path,
            label="completed action evidence",
            missing_ok=True,
        )
        if path_data is None:
            alias = (
                run_dir
                / "steps"
                / _safe_seg(step.step_id)
                / artifact_ref
            )
            alias_data = _read_run_bytes(
                run_dir,
                alias,
                label="completed action alias",
                missing_ok=True,
            )
            if alias_data is None:
                raise CheckpointCorrupt(
                    f"completed action evidence is missing: {path}"
                )
            _write_immutable(path, alias_data, run_dir=run_dir)
            path_data = alias_data
        return reference, sha256_bytes(path_data)

    @staticmethod
    def _bind_verdict(
        verdict: Verdict,
        state: RunState,
        step,
        *,
        artifact_ref: str,
        attempt_id: str,
        require_transaction: bool = True,
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
        artifact_bytes = _read_run_bytes(
            Path(state.run_dir),
            path,
            label="verdict artifact",
            missing_ok=True,
        )
        if artifact_bytes is None:
            raise TransactionCorrupt(
                f"cannot bind verdict to missing artifact: {path}"
            )
        artifact_sha256 = sha256_bytes(artifact_bytes)
        if step.action == "edit_code" and require_transaction:
            tx = load_transaction(
                Path(state.run_dir), step.step_id, attempt_id
            )
            if tx is None:
                raise TransactionCorrupt(
                    f"cannot bind verdict without a patch transaction: "
                    f"{step.step_id}/{attempt_id}"
                )
            if artifact_sha256 != tx.patch_sha256:
                raise TransactionCorrupt(
                    f"verdict artifact hash does not match the patch transaction: "
                    f"{step.step_id}/{attempt_id}"
                )
        return verdict.model_copy(
            update={
                "artifact_ref": stored_ref,
                "artifact_sha256": artifact_sha256,
                "attempt_id": attempt_id,
            }
        )

    def _repair_or_fail(self, state: RunState, step, verdict: Verdict, budget, workdir) -> None:
        """Re-issue the step as a repair with the verdict's failures, or fail the run."""
        if verdict_requires_process_quarantine(verdict):
            self._quarantine_process_failure(state, step, verdict)
            return
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
                run_dir=Path(state.run_dir),
            )
            anchored_atomic_replace_bytes(
                Path(state.run_dir) / "plan.json",
                plan_bytes,
                anchor=Path(state.run_dir),
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

    @staticmethod
    def _quarantine_process_failure(
        state: RunState,
        step,
        verdict: Verdict,
    ) -> None:
        check = next(
            check
            for check in verdict.checks
            if check.detail.get(PROCESS_CLEANUP_UNCONFIRMED) is True
        )
        cleanup = check.detail.get("process_cleanup")
        cleanup_detail = (
            cleanup.get("detail") if isinstance(cleanup, dict) else None
        )
        detail = (
            cleanup_detail
            if isinstance(cleanup_detail, str)
            else str(
                check.detail.get("summary")
                or "process cleanup was not confirmed"
            )
        )
        state.quarantine = RunQuarantine(
            step_id=step.step_id,
            attempt_id=state.attempt_id(step),
            detail=detail,
        )
        state.status = "PAUSED"
        state.active_since = None

    @staticmethod
    def _quarantine_cleanup_interruption(
        state: RunState,
        step,
        error: ProcessCleanupUnconfirmed,
    ) -> None:
        step_id = step.step_id if step is not None else "-"
        attempt_id = state.attempt_id(step) if step is not None else "plan"
        state.quarantine = RunQuarantine(
            kind="process_cleanup_interrupted",
            step_id=step_id,
            attempt_id=attempt_id,
            detail=error.detail[-1000:],
        )
        state.status = "PAUSED"
        state.active_since = None

    # --- execute dispatch ---------------------------------------------------
    def _execute(self, state: RunState, step, bundle) -> tuple[Any, str]:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        if step.action in ("gather_context", "answer_query"):
            if bundle.answer:
                atomic_replace_text(
                    run_dir / "answer.md",
                    bundle.answer,
                    anchor=run_dir,
                )
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
        inventory = self._pytest_oracle_inventory(
            state,
            step,
            allow_build=(
                tx is None
                and not (artifact_dir / "patch.json").exists()
                and not (artifact_dir / "patch.json").is_symlink()
            ),
        )

        if tx is not None:
            data = self._patch_bytes(state, step)
            attempt_path = artifact_dir / "patch.json"
            try:
                attempt_data = _read_run_bytes(
                    run_dir,
                    attempt_path,
                    label="persisted patch",
                )
                assert attempt_data is not None
            except (OSError, CheckpointCorrupt) as e:
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
        durable_mkdir_chain(artifact_dir, anchor=run_dir)
        _write_immutable(
            artifact_dir / "patch.json",
            patch_bytes,
            run_dir=run_dir,
        )
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
            artifact_dir / "review.diff",
            review_diff.encode("utf-8"),
            run_dir=run_dir,
        )
        _dump(run_dir, step.step_id, "patch.diff", review_diff)
        violations = policy.check_resolved(
            resolved,
            state.task.allowed_protected_files,
            additional_protected_files=inventory or (),
        )
        if violations:
            # Persist the refused proposal for audit, but do not create a
            # transaction because the sandbox was never touched.
            raise PolicyViolation(step.step_id, violations)

        backup_path = (
            run_dir / "backups" / _safe_seg(step.step_id) / f"{_safe_seg(attempt_id)}.json"
        )
        backup_digest = save_backup(backup, backup_path, run_dir=run_dir)
        tx = build_transaction(
            run_dir=run_dir,
            step_id=step.step_id,
            attempt_id=attempt_id,
            resolved=resolved,
            backup_sha256=backup_digest,
        )
        save_backup(
            backup,
            run_dir / tx.backup_mirror_ref,
            run_dir=run_dir,
        )

        manifest = build_manifest(
            step=step,
            patch=patch,
            patch_json_bytes=patch_bytes,
            workdir=workdir,
            policy_overrides=list(state.task.allowed_protected_files),
            resolved=resolved,
        )
        manifest_json = manifest.model_dump_json(indent=2)
        _write_immutable(
            artifact_dir / "manifest.json",
            manifest_json.encode("utf-8"),
            run_dir=run_dir,
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
        run_dir = Path(state.run_dir)
        path = artifact_dir / "manifest.json"
        try:
            manifest_bytes = _read_run_bytes(
                run_dir,
                path,
                label="persisted manifest",
            )
            assert manifest_bytes is not None
            manifest = ArtifactManifest.model_validate_json(manifest_bytes)
            expected_review = render_review_diff(
                patch, resolved, backup
            ).encode("utf-8")
            review_path = artifact_dir / "review.diff"
            review_bytes = _read_run_bytes(
                run_dir,
                review_path,
                label="review artifact",
            )
            if review_bytes != expected_review:
                raise ValueError(
                    f"review artifact does not match executable patch: {review_path}"
                )
            _validate_optional_aliases(
                run_dir,
                expected_review,
                [
                    artifact_dir.parents[1] / "patch.diff",
                    artifact_dir.parents[3] / "patch.diff",
                ],
                label="review artifact",
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
                run_dir=run_dir,
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
                run_dir=run_dir,
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
                backup = load_backup(
                    run_dir / "backups" / f"{_safe_seg(step.step_id)}.json",
                    run_dir=run_dir,
                )
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
        """Read immutable attempt bytes and only cross-check compatibility aliases."""
        run_dir = Path(state.run_dir)
        attempt_id = state.attempt_id(step)
        authoritative_path = (
            attempt_artifact_dir(run_dir, step.step_id, attempt_id)
            / "patch.json"
        )
        data = _read_run_bytes(
            run_dir,
            authoritative_path,
            label="immutable patch artifact",
            missing_ok=True,
        )
        if data is None:
            return None
        _validate_optional_aliases(
            run_dir,
            data,
            [
                run_dir / "steps" / _safe_seg(step.step_id) / "patch.json",
                run_dir / "patch.json",
            ],
            label="patch",
        )
        return data

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
            data = _read_run_bytes(
                run_dir,
                path,
                label=f"{name} summary source",
                missing_ok=True,
            )
            if data is None:
                continue
            try:
                return model.model_validate_json(data)
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
        atomic_replace_text(path, pr.to_markdown(), anchor=run_dir)
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
        atomic_replace_text(path, summary.to_markdown(), anchor=run_dir)
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
            durable_mkdir_chain(workdir, anchor=workdir.parent)

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
