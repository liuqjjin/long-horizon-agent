"""The verification verdict schema. ``verify.json`` is a serialized ``Verdict``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, FiniteFloat, model_validator

from ..clock import now

VerifierFamily = Literal["code", "experiment", "context"]
PROCESS_CLEANUP_UNCONFIRMED = "process_cleanup_unconfirmed"


def process_cleanup_failure_detail(
    *,
    returncode: int,
    cleanup_unconfirmed: bool,
    detail: str = "",
) -> dict[str, Any]:
    """Structured verdict fields for a backend whose process may still run."""
    if not cleanup_unconfirmed:
        return {}
    return {
        "non_retryable": True,
        PROCESS_CLEANUP_UNCONFIRMED: True,
        "process_cleanup": {
            "returncode": returncode,
            "confirmed": False,
            "detail": detail or "backend process cleanup was not confirmed",
        },
    }


def verdict_requires_process_quarantine(verdict: "Verdict") -> bool:
    """Whether proceeding could race a process still mutating the worktree."""
    return any(
        check.detail.get(PROCESS_CLEANUP_UNCONFIRMED) is True
        for check in verdict.checks
    )


class Check(BaseModel):
    """The result of a single verifier."""

    name: str  # "pytest", "ruff", "psnr", "freshness", ...
    family: VerifierFamily
    passed: bool
    score: FiniteFloat | None = None
    threshold: FiniteFloat | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    duration_s: FiniteFloat = Field(default=0.0, ge=0)


class Verdict(BaseModel):
    """Aggregate verification result for one step's artifact."""

    step_id: str
    passed: bool
    checks: list[Check] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)  # fed back into the repair loop
    artifact_ref: str | None = None
    artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    attempt_id: str | None = None
    timestamp: datetime = Field(default_factory=now)
    env: dict[str, Any] = Field(default_factory=dict)  # reproducibility record

    @model_validator(mode="after")
    def _aggregate_is_not_forgeable(self) -> "Verdict":
        expected = bool(self.checks) and all(check.passed for check in self.checks)
        if self.passed != expected:
            raise ValueError(
                "passed must equal the non-empty conjunction of checks; "
                f"expected {expected}, got {self.passed}"
            )
        return self

    @classmethod
    def from_checks(
        cls,
        step_id: str,
        checks: list[Check],
        *,
        artifact_ref: str | None = None,
        artifact_sha256: str | None = None,
        attempt_id: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> "Verdict":
        # An empty check list verified nothing, so it must not pass. The rule
        # "a check that cannot run fails" has to hold for the aggregate too —
        # enforced here rather than only by convention at each call site.
        passed = bool(checks) and all(c.passed for c in checks)
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
            artifact_sha256=artifact_sha256,
            attempt_id=attempt_id,
            env=env or {},
        )
