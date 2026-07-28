"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from lha.tasks.spec import TaskSpec


@pytest.fixture
def fake_docker_executable(tmp_path: Path) -> str:
    """A real executable path for Docker unit tests that mock daemon calls."""
    executable = tmp_path / "fake-docker-control"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    return str(executable)


def hermetic_task(path: str) -> TaskSpec:
    """Load a bundled task for a backend-less test environment.

    The unit tests run hermetically (no code-search backend), so context
    gathering is declared optional explicitly — the default is the fail-closed
    ``required``, under which an empty/unavailable context fails the run.
    """
    return TaskSpec.from_file(path).model_copy(update={"context_requirement": "optional"})
