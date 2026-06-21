"""Run state + step ledger. The JSON checkpoint that makes runs resumable.

Shaped so LangGraph can drop in later: ``thread_id`` (== run_id) and a reserved
``channel_values`` field mirror LangGraph's checkpoint channels.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..artifacts import Plan, Step
from ..clock import now
from ..tasks.spec import TaskSpec

RunStatus = Literal["RUNNING", "AWAITING_APPROVAL", "DONE", "FAILED", "PAUSED"]
Phase = Literal["plan", "context", "execute", "approval", "verify", "repair", "complete", "fail"]


class StepRecord(BaseModel):
    """One append-only ledger entry."""

    seq: int
    step_id: str
    phase: Phase
    artifact_ref: str | None = None
    verdict_ref: str | None = None
    timestamp: datetime = Field(default_factory=now)
    notes: str | None = None


class RunState(BaseModel):
    run_id: str
    task: TaskSpec
    status: RunStatus = "RUNNING"
    plan: Plan | None = None
    cursor: int = 0  # index of the next step -> the resume point
    completed_steps: list[str] = Field(default_factory=list)
    failed_steps: list[str] = Field(default_factory=list)
    repairs: dict[str, int] = Field(default_factory=dict)
    run_dir: str = ""
    workdir: str = ""
    pr_summary_path: str | None = None
    seq: int = 0  # ledger sequence counter
    schema_version: int = 1
    # --- LangGraph-shaped fields (unused in v1, present for drop-in) ---
    thread_id: str = ""
    channel_values: dict = Field(default_factory=dict)

    @classmethod
    def new(cls, task: TaskSpec, run_id: str, run_dir: str, workdir: str) -> "RunState":
        return cls(
            run_id=run_id,
            task=task,
            run_dir=run_dir,
            workdir=workdir,
            thread_id=run_id,
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
