"""Shared test helpers."""

from __future__ import annotations

from lha.tasks.spec import TaskSpec


def hermetic_task(path: str) -> TaskSpec:
    """Load a bundled task for a backend-less test environment.

    The unit tests run hermetically (no code-search backend), so context
    gathering is declared optional explicitly — the default is the fail-closed
    ``required``, under which an empty/unavailable context fails the run.
    """
    return TaskSpec.from_file(path).model_copy(update={"context_requirement": "optional"})
