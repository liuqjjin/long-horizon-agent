"""Verifier registry. ``select_verifiers(step)`` returns the verifiers a step
asked for. Adding a verifier = register it here (or via ``register``)."""

from __future__ import annotations

from ..artifacts import Step
from .base import Verifier
from .code import PytestVerifier, RuffVerifier
from .context import CitationVerifier, FreshnessVerifier
from .experiment import PSNRVerifier, ReproVerifier, SSIMVerifier

_REGISTRY: dict[str, Verifier] = {}


def register(verifier: Verifier) -> None:
    _REGISTRY[verifier.name] = verifier


def get(name: str) -> Verifier | None:
    return _REGISTRY.get(name)


def select_verifiers(step: Step) -> list[Verifier]:
    return [_REGISTRY[n] for n in step.verifiers if n in _REGISTRY]


def all_verifiers() -> list[Verifier]:
    return list(_REGISTRY.values())


# --- built-ins -------------------------------------------------------------
register(PytestVerifier())
register(RuffVerifier())
register(FreshnessVerifier())
register(CitationVerifier())
register(PSNRVerifier())
register(SSIMVerifier())
register(ReproVerifier())
