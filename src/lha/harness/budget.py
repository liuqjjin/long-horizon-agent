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
    _start: float = field(default_factory=time.monotonic)

    def tick(self) -> None:
        self.steps_used += 1
        if self.steps_used > self.max_steps:
            raise BudgetExceeded(f"max_steps={self.max_steps} exceeded")
        if self.deadline_s is not None and (time.monotonic() - self._start) > self.deadline_s:
            raise BudgetExceeded(f"deadline {self.deadline_s}s exceeded")
