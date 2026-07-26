"""Pluggable verifier interface. A verifier turns an artifact into one Check.

Families: code (pytest/ruff/type), experiment (PSNR/SSIM/repro), context
(freshness/citation). PSNR is just one member — the core has no PSNR-shaped
assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..artifacts import Step
from ..live_context.models import ContextBundle
from ..sandbox import ExecutionBackend, TrustedLocalBackend
from .verdict import Check, VerifierFamily


@dataclass
class VerifyContext:
    """Everything a verifier might need beyond the artifact itself."""

    workdir: Path
    step: Step
    bundle: ContextBundle | None = None
    # Where target code (pytest, experiments, re-runs) actually executes.
    exec: ExecutionBackend = field(default_factory=TrustedLocalBackend)
    # Stable across process crashes, different for each repair attempt.
    attempt_id: str | None = None


class Verifier(ABC):
    name: str = "base"
    family: VerifierFamily = "code"

    def applies_to(self, step: Step) -> bool:
        return self.name in step.verifiers

    @abstractmethod
    def verify(self, artifact: Any, ctx: VerifyContext) -> Check: ...
