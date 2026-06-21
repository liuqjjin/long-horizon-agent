"""The verification verdict schema. ``verify.json`` is a serialized ``Verdict``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..clock import now

VerifierFamily = Literal["code", "experiment", "context"]


class Check(BaseModel):
    """The result of a single verifier."""

    name: str  # "pytest", "ruff", "psnr", "freshness", ...
    family: VerifierFamily
    passed: bool
    score: float | None = None
    threshold: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    duration_s: float = 0.0


class Verdict(BaseModel):
    """Aggregate verification result for one step's artifact."""

    step_id: str
    passed: bool
    checks: list[Check] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)  # fed back into the repair loop
    artifact_ref: str | None = None
    timestamp: datetime = Field(default_factory=now)
    env: dict[str, Any] = Field(default_factory=dict)  # reproducibility record

    @classmethod
    def from_checks(
        cls,
        step_id: str,
        checks: list[Check],
        *,
        artifact_ref: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> "Verdict":
        passed = all(c.passed for c in checks) if checks else True
        failures = [
            f"{c.name}: " + (str(c.detail.get("summary")) if c.detail.get("summary") else "failed")
            for c in checks
            if not c.passed
        ]
        return cls(
            step_id=step_id,
            passed=passed,
            checks=checks,
            failures=failures,
            artifact_ref=artifact_ref,
            env=env or {},
        )
