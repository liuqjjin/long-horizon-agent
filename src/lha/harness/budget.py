"""Loop budget: bound steps, repairs, and wall-clock so runs always terminate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .errors import BudgetExceeded


@dataclass
class StepBudget:
    max_steps: int = 20
    max_repairs: int = 3
    deadline_s: float | None = None
    steps_used: int = 0
    # Wall-clock already spent in prior processes (seeded from the checkpoint on
    # resume) so max_steps/deadline bound the whole RUN, not just this process.
    prior_elapsed_s: float = 0.0
    _start: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        """Total wall-clock for the run: this process plus all prior resumes."""
        return self.prior_elapsed_s + (time.monotonic() - self._start)

    def tick(self) -> None:
        # Check limits BEFORE consuming a step so a pause persists an accurate
        # steps_used (the failing tick must not be counted, or resume loses a step).
        if self.steps_used >= self.max_steps:
            raise BudgetExceeded(f"max_steps={self.max_steps} exceeded")
        if self.deadline_s is not None and self.elapsed() > self.deadline_s:
            raise BudgetExceeded(f"deadline {self.deadline_s}s exceeded")
        self.steps_used += 1
