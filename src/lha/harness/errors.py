"""Typed harness errors. All are catchable so a run can checkpoint and pause."""

from __future__ import annotations


class HarnessError(RuntimeError):
    pass


class BudgetExceeded(HarnessError):
    """Max steps / repairs / wall-clock exceeded. The run is checkpointed."""


class ApprovalPending(HarnessError):
    """A human-approval gate paused the run; resume after ``lha approve``."""

    def __init__(self, run_id: str, step_id: str):
        super().__init__(f"awaiting approval for run {run_id} step {step_id}")
        self.run_id = run_id
        self.step_id = step_id


class ApprovalRejected(HarnessError):
    pass
