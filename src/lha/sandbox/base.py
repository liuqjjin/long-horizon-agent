"""Execution backend interface + shared helpers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..tools.shell import ProcResult

# Environment variables that survive into target-code execution. Everything
# else — API keys, tokens, cloud credentials — is stripped: target code has no
# business reading the harness's secrets.
_KEEP_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TERM")


def scrub_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A minimal environment for running target code."""
    env = {k: v for k in _KEEP_ENV if (v := os.environ.get(k))}
    if extra:
        env.update(extra)
    return env


@dataclass
class ResourceLimits:
    """Bounds applied to target-code execution (best effort per backend).

    Defaults are all None: on the host, RLIMIT_NPROC counts the USER's whole
    process table (a low cap breaks fork for everything), so host limits are
    opt-in per call. The docker backend supplies its own strong defaults —
    container limits are scoped to the container.
    """

    cpu_s: int | None = None  # CPU seconds (RLIMIT_CPU)
    memory_mb: int | None = None
    pids: int | None = None


class ExecutionBackend(ABC):
    """Runs a command against a working directory, somewhere."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        timeout: float = 300.0,
        input: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> ProcResult:
        """Execute ``cmd`` with ``cwd`` as the working directory.

        Must terminate the entire process tree on timeout and return a
        ``ProcResult`` (returncode 124 on timeout) rather than raising.
        """

    @abstractmethod
    def python(self) -> str:
        """The Python interpreter argv[0] appropriate for this backend."""

    @abstractmethod
    def tool(self, name: str) -> str:
        """Resolve a console tool (e.g. ``ruff``) for this backend."""
