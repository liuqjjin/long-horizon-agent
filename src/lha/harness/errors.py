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


class CheckpointCorrupt(HarnessError):
    """A checkpoint or ledger failed validation. Resuming from corrupt state
    would silently replay or skip work, so loading fails closed instead."""


class RunLocked(HarnessError):
    """Another process already owns this run.

    Concurrent resume is rejected because two writers cannot safely share a
    checkpoint, ledger, or sandbox.
    """


class TransactionCorrupt(HarnessError):
    """A persisted patch transaction or backup failed validation."""


class PolicyViolation(HarnessError):
    """A patch tried to touch protected oracle/config files. The patch is
    refused before it reaches the sandbox; the loop treats this as a failed
    verification so the repair loop gets the reason as feedback."""

    def __init__(self, step_id: str, violations: list[str]):
        super().__init__(
            f"patch for step {step_id} touches protected files: {', '.join(violations)}"
        )
        self.step_id = step_id
        self.violations = violations
