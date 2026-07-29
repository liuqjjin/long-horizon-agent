"""Run state + step ledger. The JSON checkpoint that makes runs resumable.

Shaped so LangGraph can drop in later: ``thread_id`` (== run_id) and a reserved
``channel_values`` field mirror LangGraph's checkpoint channels.
"""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from ..config import Config

from ..artifacts import Plan, Step
from ..clock import now
from ..oracle_models import PytestOracleInventory
from ..step_ids import validate_plan_step_ids
from ..tasks.spec import TaskSpec
from .budget import RunBudgetLimits
from .errors import CheckpointCorrupt

RunStatus = Literal["RUNNING", "AWAITING_APPROVAL", "DONE", "FAILED", "PAUSED"]
Phase = Literal["plan", "context", "execute", "approval", "verify", "repair", "complete", "fail"]
RuntimeKind = Literal["loop", "langgraph"]
RUN_STATE_SCHEMA = 2


def _event_id() -> str:
    return uuid.uuid4().hex


class StepRecord(BaseModel):
    """One append-only ledger entry.

    ``seq`` is the run's persisted monotonic event counter. Recovery first
    advances the checkpoint to the largest durable ledger sequence; the
    ``event_id`` identifies the physical append and ``idempotency_key`` prevents
    a replay from recording the same logical transition twice.
    """

    seq: int
    step_id: str
    phase: Phase
    artifact_ref: str | None = None
    verdict_ref: str | None = None
    evidence_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    timestamp: datetime = Field(default_factory=now)
    notes: str | None = None
    event_id: str = Field(default_factory=_event_id)
    # Hash of the preceding durable record (None only for the first record).
    # This makes deletion or reordering visible while still allowing a torn
    # final append to be dropped and the next sequence number to contain a gap.
    prev_event_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    attempt_id: str | None = None
    idempotency_key: str | None = None


class LLMUsageState(BaseModel):
    calls: int = Field(default=0, ge=0)
    wall_s: FiniteFloat = Field(default=0.0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: FiniteFloat = Field(default=0.0, ge=0)


class RunQuarantine(BaseModel):
    """Why an operator must inspect a paused run before any further mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "process_cleanup_unconfirmed",
        "process_cleanup_interrupted",
        "active_operation_recovery_unconfirmed",
        "pytest_oracle_inventory_invalid",
    ] = "process_cleanup_unconfirmed"
    step_id: str
    attempt_id: str
    returncode: Literal[126] = 126
    detail: str


class CLIIdentity(BaseModel):
    """Executable bytes and version used by a CLI model backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: str


class RunRuntimeContract(BaseModel):
    """Outcome-affecting runtime configuration fixed for a durable run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: RuntimeKind
    exec_backend: str
    exec_image: str | None = None
    exec_image_id: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    docker_cli: CLIIdentity | None = None
    llm_backend: str
    llm_model: str
    llm_orchestration_model: str | None = None
    llm_reasoning_effort: str | None = None
    llm_sandbox: str | None = None
    llm_external_sandbox: bool = False
    llm_cli: CLIIdentity | None = None
    # None is accepted only so an older checkpoint remains inspectable. CLI
    # recovery requires True because a CLI-default model can change in place.
    llm_model_pinned: bool | None = None
    codex_max_retries: int | None = Field(default=None, ge=0, le=10)
    codex_retry_backoff_s: FiniteFloat | None = Field(default=None, ge=0)
    code_backend: str
    # The concrete choice behind ``auto``. None marks an older checkpoint that
    # predates this contract field and is therefore unsafe to resume.
    resolved_code_backend: str | None = None
    embedder_model: str
    freshness_max_age_s: FiniteFloat = Field(ge=0)
    parallel_verify: bool
    dynamic_planning: bool
    use_skill_memory: bool
    data_dir: str

    @model_validator(mode="after")
    def _docker_identity_is_complete(self) -> "RunRuntimeContract":
        if self.exec_backend == "docker" and (
            self.exec_image is None
            or self.exec_image_id is None
        ):
            raise ValueError(
                "Docker runtime contract requires configured and immutable image identities"
            )
        if self.exec_backend != "docker" and (
            self.exec_image is not None
            or self.exec_image_id is not None
            or self.docker_cli is not None
        ):
            raise ValueError(
                "non-Docker runtime contract cannot carry a Docker image identity"
            )
        return self

    @classmethod
    def capture(
        cls,
        config: Config,
        *,
        runtime: RuntimeKind,
        exec_backend: object | None = None,
        llm: object | None = None,
        code_root: str | Path | None = None,
    ) -> "RunRuntimeContract":
        """Resolve mutable executable/image names before the first side effect."""
        image: str | None = None
        image_id: str | None = None
        docker_cli: CLIIdentity | None = None
        if config.exec_backend == "docker":
            image = config.exec_image
            binder = getattr(exec_backend, "bind_control_plane", None)
            bound_identity: dict[str, object] | None = None
            if callable(binder):
                try:
                    candidate = binder(verify_digest=True)
                except (OSError, RuntimeError, ValueError) as error:
                    raise CheckpointCorrupt(
                        f"cannot bind Docker control executable: {error}"
                    ) from error
                if not isinstance(candidate, dict):
                    raise CheckpointCorrupt(
                        "cannot bind Docker control executable: invalid identity"
                    )
                bound_identity = candidate
            docker_cli = _capture_cli_identity(
                str(getattr(exec_backend, "docker", "docker"))
            )
            docker = docker_cli.path
            if bound_identity is not None and (
                bound_identity.get("path") != docker_cli.path
                or bound_identity.get("sha256") != docker_cli.sha256
            ):
                raise CheckpointCorrupt(
                    "Docker executable changed while its runtime contract was captured"
                )
            if exec_backend is not None:
                setattr(exec_backend, "docker", docker)
            configured_image = getattr(exec_backend, "image", image)
            from ..tools.shell import run

            result = run(
                [
                    str(docker),
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    str(configured_image),
                ],
                timeout=30.0,
            )
            candidate = result.stdout.strip()
            if (
                result.returncode != 0
                or result.output_truncated
                or result.cleanup_unconfirmed
                or len(candidate) != 71
                or not candidate.startswith("sha256:")
                or any(ch not in "0123456789abcdef" for ch in candidate[7:])
            ):
                detail = (
                    result.cleanup_detail
                    or result.stderr.strip()
                    or "Docker image inspection returned no immutable image ID"
                )
                raise CheckpointCorrupt(
                    f"cannot bind Docker execution image {configured_image!r}: {detail}"
                )
            image_id = candidate
            if callable(binder):
                try:
                    after_identity = binder(verify_digest=True)
                except (OSError, RuntimeError, ValueError) as error:
                    raise CheckpointCorrupt(
                        "Docker executable changed during image inspection"
                    ) from error
                if (
                    not isinstance(after_identity, dict)
                    or after_identity.get("path") != docker_cli.path
                    or after_identity.get("sha256") != docker_cli.sha256
                ):
                    raise CheckpointCorrupt(
                        "Docker executable changed during image inspection"
                    )
            # Execute the image we inspected, even if the configured tag moves
            # between checkpoint creation and the first container start.
            if exec_backend is not None:
                setattr(exec_backend, "image", image_id)

        inner = getattr(llm, "inner", llm)
        cli: CLIIdentity | None = None
        cli_name: str | None = None
        if config.llm_backend == "codex_cli":
            cli_name = config.codex_cli_path
        elif config.llm_backend == "claude_cli":
            cli_name = config.claude_cli_path
        if cli_name is not None:
            if (
                config.llm_backend == "codex_cli"
                and inner is not None
                and hasattr(inner, "_resolved_cli_identity")
                and hasattr(inner, "_cli_version")
            ):
                identity = getattr(inner, "_resolved_cli_identity")()
                version = getattr(inner, "_cli_version")()
                if version == "unknown":
                    raise CheckpointCorrupt(
                        "cannot bind Codex CLI: version probe failed"
                    )
                cli = CLIIdentity(
                    path=identity[0],
                    sha256=identity[5],
                    version=version,
                )
            else:
                cli = _capture_cli_identity(cli_name)
            if inner is not None and config.llm_backend != "codex_cli":
                setattr(inner, "cli_path", cli.path)

        model = {
            "stub": "deterministic-stub",
            "codex_cli": config.codex_model or "cli-default",
            "claude_cli": config.claude_cli_model or "cli-default",
            "anthropic": config.anthropic_model_impl,
        }.get(config.llm_backend, "unknown")
        effort = (
            config.codex_reasoning_effort
            if config.llm_backend == "codex_cli"
            else ("high" if config.llm_backend == "anthropic" else None)
        )
        sandbox = (
            config.codex_sandbox
            if config.llm_backend == "codex_cli"
            else None
        )
        if config.code_backend == "auto" and code_root is None:
            raise CheckpointCorrupt(
                "cannot bind automatic code backend without the run's code root"
            )
        from ..live_context import resolve_code_backend_name

        resolved_code_backend = resolve_code_backend_name(
            code_root=code_root or Path.cwd(),
            config=config,
        )
        return cls(
            runtime=runtime,
            exec_backend=config.exec_backend,
            exec_image=image,
            exec_image_id=image_id,
            docker_cli=docker_cli,
            llm_backend=config.llm_backend,
            llm_model=model,
            llm_orchestration_model=(
                config.anthropic_model_orchestration
                if config.llm_backend == "anthropic"
                else None
            ),
            llm_reasoning_effort=effort,
            llm_sandbox=sandbox,
            llm_external_sandbox=(
                config.codex_external_sandbox
                if config.llm_backend == "codex_cli"
                else False
            ),
            llm_cli=cli,
            llm_model_pinned=(
                bool(config.codex_model.strip())
                if config.llm_backend == "codex_cli"
                else (
                    bool(config.claude_cli_model.strip())
                    if config.llm_backend == "claude_cli"
                    else True
                )
            ),
            codex_max_retries=(
                config.codex_max_retries
                if config.llm_backend == "codex_cli"
                else None
            ),
            codex_retry_backoff_s=(
                config.codex_retry_backoff_s
                if config.llm_backend == "codex_cli"
                else None
            ),
            code_backend=config.code_backend,
            resolved_code_backend=resolved_code_backend,
            embedder_model=config.embedder_model,
            freshness_max_age_s=config.freshness_max_age_s,
            parallel_verify=config.parallel_verify,
            dynamic_planning=config.dynamic_planning,
            use_skill_memory=config.use_skill_memory,
            data_dir=str(Path(config.data_dir).resolve()),
        )

    def verify_recorded_docker_control(self) -> None:
        """Check recorded Docker bytes before using them for orphan recovery."""
        if self.exec_backend != "docker":
            return
        recorded = self.docker_cli
        if recorded is None:
            raise CheckpointCorrupt(
                "Docker runtime contract has no recorded client identity"
            )
        from ..sandbox.docker import resolve_docker_executable

        try:
            current = resolve_docker_executable(recorded.path)
        except (OSError, RuntimeError, ValueError) as error:
            raise CheckpointCorrupt(
                "recorded Docker executable is unavailable"
            ) from error
        if current.path != recorded.path or current.sha256 != recorded.sha256:
            raise CheckpointCorrupt(
                "Docker executable bytes changed before operation recovery"
            )


def _capture_cli_identity(value: str) -> CLIIdentity:
    candidate = Path(_resolve_control_executable(value))
    try:
        resolved = candidate.resolve(strict=True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, RuntimeError) as error:
        raise CheckpointCorrupt(
            f"cannot bind model CLI {value!r}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size > 64 * 1024 * 1024
        ):
            raise CheckpointCorrupt(
                f"cannot bind model CLI {value!r}: executable is not a bounded regular file"
            )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise CheckpointCorrupt(
                    f"cannot bind model CLI {value!r}: executable changed while read"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            not stat.S_ISREG(after.st_mode)
            or any(
                getattr(before, field) != getattr(after, field)
                for field in stable
            )
        ):
            raise CheckpointCorrupt(
                f"cannot bind model CLI {value!r}: executable changed while read"
            )
    finally:
        os.close(descriptor)
    from ..tools.shell import run

    result = run([str(resolved), "--version"], timeout=30.0)
    version = (result.stdout.strip() or result.stderr.strip())
    if (
        result.returncode != 0
        or result.output_truncated
        or result.cleanup_unconfirmed
        or not version
        or len(version.encode("utf-8")) > 256
        or "\n" in version
        or "\r" in version
    ):
        detail = result.cleanup_detail or version or "version probe failed"
        raise CheckpointCorrupt(
            f"cannot bind model CLI {value!r}: {detail}"
        )
    return CLIIdentity(
        path=str(resolved),
        sha256=digest.hexdigest(),
        version=version,
    )


def _resolve_control_executable(value: str) -> str:
    from ..tools.shell import trusted_executable

    candidate = Path(value)
    if candidate.is_absolute():
        resolved = trusted_executable(
            candidate.name,
            path="",
            extra_dirs=(candidate.parent,),
            require_unwritable=False,
        )
    elif candidate.name == value:
        resolved = trusted_executable(value, require_unwritable=False)
    else:
        resolved = None
    if resolved is None:
        raise CheckpointCorrupt(
            f"cannot bind control executable {value!r} on a sanitized absolute PATH"
        )
    return resolved


class RunState(BaseModel):
    run_id: str
    task: TaskSpec
    status: RunStatus = "RUNNING"
    plan: Plan | None = None
    cursor: int = Field(default=0, ge=0)  # index of the next step -> the resume point
    completed_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    repairs: dict[str, int] = Field(default_factory=dict)
    run_dir: str = ""
    workdir: str = ""
    pr_summary_path: str | None = None
    seq: int = Field(default=0, ge=0)  # ledger sequence counter
    # Cumulative budget consumption, persisted so max_steps/deadline bound the
    # whole run across pause/resume cycles rather than resetting per process.
    steps_used: int = Field(default=0, ge=0)
    elapsed_s: FiniteFloat = Field(default=0.0, ge=0)
    # Set and fsynced before a model/tool side effect, cleared only after its
    # duration is settled. Resume conservatively charges an interrupted window.
    active_since: datetime | None = None
    schema_version: int = RUN_STATE_SCHEMA
    # The limits are part of the run's identity. A new process may resume only
    # with the exact contract that was recorded before the first side effect.
    budget_limits: RunBudgetLimits | None = None
    runtime_contract: RunRuntimeContract | None = None
    pytest_oracle_inventories: dict[str, PytestOracleInventory] = Field(
        default_factory=dict
    )
    attempt_ids: dict[str, str] = Field(default_factory=dict)
    # Attempt ids whose max-step budget unit was durably consumed before any
    # context/tool/model work began. Replaying the same attempt after a crash
    # must not consume a second unit.
    budgeted_attempts: list[str] = Field(default_factory=list)
    llm_usage: LLMUsageState = Field(default_factory=LLMUsageState)
    quarantine: RunQuarantine | None = None
    # --- LangGraph-shaped fields (unused in v1, present for drop-in) ---
    thread_id: str = ""
    channel_values: dict = Field(default_factory=dict)

    @field_validator("repairs")
    @classmethod
    def _non_negative_repairs(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("repair counters must be non-negative")
        return value

    @field_validator("active_since")
    @classmethod
    def _active_since_is_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError("active_since must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _plan_step_ids_are_canonical(self) -> "RunState":
        if self.plan is not None:
            validate_plan_step_ids(step.step_id for step in self.plan.steps)
        if self.schema_version >= RUN_STATE_SCHEMA and self.budget_limits is None:
            raise ValueError("schema-v2 run state is missing its budget limits")
        return self

    @classmethod
    def new(
        cls,
        task: TaskSpec,
        run_id: str,
        run_dir: str,
        workdir: str,
        *,
        config: Config,
        runtime: RuntimeKind = "loop",
        exec_backend: object | None = None,
        llm: object | None = None,
    ) -> "RunState":
        return cls(
            run_id=run_id,
            task=task,
            run_dir=run_dir,
            workdir=workdir,
            thread_id=run_id,
            budget_limits=RunBudgetLimits.from_config(config),
            runtime_contract=RunRuntimeContract.capture(
                config,
                runtime=runtime,
                exec_backend=exec_backend,
                llm=llm,
                code_root=workdir,
            ),
        )

    # --- queries ---
    def is_terminal(self) -> bool:
        return self.status in ("DONE", "FAILED")

    def next_step(self) -> Step | None:
        if self.plan and 0 <= self.cursor < len(self.plan.steps):
            return self.plan.steps[self.cursor]
        return None

    def repairs_for(self, step: Step) -> int:
        return self.repairs.get(step.step_id, 0)

    def require_matching_budget_limits(self, config: Config) -> RunBudgetLimits:
        """Reject resume when process configuration changes the recorded contract."""
        recorded = self.budget_limits
        if recorded is None:
            raise CheckpointCorrupt(
                f"run {self.run_id} has no persisted budget limits; refusing safe resume"
            )
        current = RunBudgetLimits.from_config(config)
        if current != recorded:
            changed = ", ".join(
                f"{field}: recorded={getattr(recorded, field)!r}, "
                f"current={getattr(current, field)!r}"
                for field in RunBudgetLimits.model_fields
                if getattr(recorded, field) != getattr(current, field)
            )
            raise CheckpointCorrupt(
                f"run {self.run_id} budget limits changed across resume ({changed})"
            )
        return recorded

    def require_matching_runtime_contract(
        self,
        config: Config,
        *,
        runtime: RuntimeKind,
        exec_backend: object | None = None,
        llm: object | None = None,
    ) -> RunRuntimeContract:
        """Reject legacy state and any runtime/backend drift before recovery."""
        recorded = self.runtime_contract
        if recorded is None:
            raise CheckpointCorrupt(
                f"run {self.run_id} has no persisted runtime contract; "
                "it may be inspected but cannot be resumed safely"
            )
        if (
            recorded.llm_backend in {"codex_cli", "claude_cli"}
            and recorded.llm_model_pinned is not True
        ):
            raise CheckpointCorrupt(
                f"run {self.run_id} did not record an explicitly pinned "
                f"{recorded.llm_backend} model; refusing safe resume"
            )
        current = RunRuntimeContract.capture(
            config,
            runtime=runtime,
            exec_backend=exec_backend,
            llm=llm,
            code_root=self.workdir,
        )
        if current != recorded:
            changed = ", ".join(
                f"{field}: recorded={getattr(recorded, field)!r}, "
                f"current={getattr(current, field)!r}"
                for field in RunRuntimeContract.model_fields
                if getattr(recorded, field) != getattr(current, field)
            )
            raise CheckpointCorrupt(
                f"run {self.run_id} runtime contract changed across resume ({changed})"
            )
        return recorded

    # --- transitions ---
    def record_repair(self, step: Step) -> None:
        self.repairs[step.step_id] = self.repairs.get(step.step_id, 0) + 1

    def complete_step(self, step: Step) -> None:
        if step.step_id not in self.completed_steps:
            self.completed_steps.append(step.step_id)
        self.cursor += 1

    def fail_current(self, step: Step) -> None:
        self.failed_steps.append(step.step_id)
        self.status = "FAILED"

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def attempt_id(self, step: Step) -> str:
        """Stable id for a step attempt across crashes and process resumes."""
        attempt = f"{step.step_id}-r{self.repairs_for(step)}"
        self.attempt_ids[step.step_id] = attempt
        return attempt

    def attempt_is_budgeted(self, step: Step) -> bool:
        return self.attempt_id(step) in self.budgeted_attempts

    def mark_attempt_budgeted(self, step: Step) -> None:
        attempt = self.attempt_id(step)
        if attempt not in self.budgeted_attempts:
            self.budgeted_attempts.append(attempt)

    def recover_active_elapsed(self) -> None:
        if self.active_since is None:
            return
        delta = (now() - self.active_since).total_seconds()
        if delta < 0:
            raise CheckpointCorrupt(
                "active_since is in the future; refusing to reset the deadline budget"
            )
        self.elapsed_s += delta
        self.active_since = None
