"""Loop budget: bound steps, repairs, and wall-clock so runs always terminate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

if TYPE_CHECKING:
    from ..config import Config

from .errors import BudgetExceeded


class RunBudgetLimits(BaseModel):
    """Immutable budget contract recorded when a run is created."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(ge=1)
    max_repairs: int = Field(ge=0)
    deadline_s: FiniteFloat | None = Field(default=None, ge=0)
    max_llm_calls: int | None = Field(default=None, ge=1)

    @classmethod
    def from_config(cls, config: Config) -> RunBudgetLimits:
        return cls(
            max_steps=config.max_steps,
            max_repairs=config.max_repairs,
            deadline_s=config.deadline_s,
            max_llm_calls=config.max_llm_calls,
        )


@dataclass
class StepBudget:
    limits: RunBudgetLimits
    steps_used: int = 0
    # Wall-clock already spent in prior processes (seeded from the checkpoint on
    # resume) so max_steps/deadline bound the whole RUN, not just this process.
    prior_elapsed_s: float = 0.0
    _start: float = field(default_factory=time.monotonic)

    @property
    def max_repairs(self) -> int:
        return self.limits.max_repairs

    def elapsed(self) -> float:
        """Total wall-clock for the run: this process plus all prior resumes."""
        return self.prior_elapsed_s + (time.monotonic() - self._start)

    def tick(self) -> None:
        # Check limits BEFORE consuming a step so a pause persists an accurate
        # steps_used (the failing tick must not be counted, or resume loses a step).
        if self.steps_used >= self.limits.max_steps:
            raise BudgetExceeded(f"max_steps={self.limits.max_steps} exceeded")
        self.check_deadline()
        self.steps_used += 1

    def check_deadline(self) -> None:
        """Check wall time without consuming another step."""
        deadline = self.limits.deadline_s
        if deadline is not None and self.elapsed() > deadline:
            raise BudgetExceeded(f"deadline {deadline}s exceeded")
