"""Run state + step ledger. The JSON checkpoint that makes runs resumable.

Shaped so LangGraph can drop in later: ``thread_id`` (== run_id) and a reserved
``channel_values`` field mirror LangGraph's checkpoint channels.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, FiniteFloat, field_validator, model_validator

if TYPE_CHECKING:
    from ..config import Config

from ..artifacts import Plan, Step
from ..clock import now
from ..step_ids import validate_plan_step_ids
from ..tasks.spec import TaskSpec
from .budget import RunBudgetLimits
from .errors import CheckpointCorrupt

RunStatus = Literal["RUNNING", "AWAITING_APPROVAL", "DONE", "FAILED", "PAUSED"]
Phase = Literal["plan", "context", "execute", "approval", "verify", "repair", "complete", "fail"]
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
    attempt_ids: dict[str, str] = Field(default_factory=dict)
    # Attempt ids whose max-step budget unit was durably consumed before any
    # context/tool/model work began. Replaying the same attempt after a crash
    # must not consume a second unit.
    budgeted_attempts: list[str] = Field(default_factory=list)
    llm_usage: LLMUsageState = Field(default_factory=LLMUsageState)
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
    ) -> "RunState":
        return cls(
            run_id=run_id,
            task=task,
            run_dir=run_dir,
            workdir=workdir,
            thread_id=run_id,
            budget_limits=RunBudgetLimits.from_config(config),
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
