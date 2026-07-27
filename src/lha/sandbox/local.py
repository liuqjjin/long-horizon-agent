"""Trusted-local execution: the host interpreter, hardened.

Only for repositories you already trust (this repo's own tests and self-eval).
Still applies the cheap protections that cost nothing: a scrubbed environment
(no inherited secrets), best-effort POSIX resource limits, and process-group
kill on timeout so a runaway test can't orphan children.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from ..tools.shell import ProcResult, venv_tool
from .base import (
    ExecutionBackend,
    ResourceLimits,
    run_bounded_process,
    scrub_env,
)


def _limit_preexec(limits: ResourceLimits):
    """A preexec_fn applying rlimits in the child (best effort; POSIX only)."""

    def apply() -> None:
        try:
            import resource
        except ImportError:  # pragma: no cover - non-POSIX
            return
        pairs = []
        if limits.cpu_s:
            pairs.append((resource.RLIMIT_CPU, limits.cpu_s))
        if limits.memory_mb:
            # RLIMIT_AS is unreliable on macOS but harmless; enforced on Linux.
            pairs.append((resource.RLIMIT_AS, limits.memory_mb * 1024 * 1024))
        if limits.pids and hasattr(resource, "RLIMIT_NPROC"):
            pairs.append((resource.RLIMIT_NPROC, limits.pids))
        for res, val in pairs:
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass  # a limit the platform refuses is skipped, not fatal

    return apply


class TrustedLocalBackend(ExecutionBackend):
    name = "trusted-local"

    def __init__(self, limits: ResourceLimits | None = None):
        self.limits = limits or ResourceLimits()

    def python(self) -> str:
        return sys.executable

    def tool(self, name: str) -> str:
        return venv_tool(name)

    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        timeout: float = 300.0,
        input: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> ProcResult:
        limits = limits or self.limits

        def kill_tree(process) -> str:
            # The child owns a process group, so descendants that inherited its
            # output descriptors are stopped before the drain threads join.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                try:
                    process.kill()
                except OSError:
                    pass
            return "process tree killed"

        try:
            return run_bounded_process(
                cmd,
                cwd=str(cwd),
                env=scrub_env(),
                timeout=timeout,
                input=input,
                output_bytes=limits.output_bytes,
                start_new_session=True,  # own process group -> killable as a tree
                preexec_fn=_limit_preexec(limits),
                on_timeout=kill_tree,
            )
        except OSError as e:
            return ProcResult(127, "", f"failed to start {cmd[0]!r}: {e}", 0.0)
