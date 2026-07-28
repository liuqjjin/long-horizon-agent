"""Trusted-local execution: the host interpreter, hardened.

Only for repositories you already trust (this repo's own tests and self-eval).
Still applies the cheap protections that cost nothing: a scrubbed environment
(no inherited secrets), best-effort POSIX resource limits, and process-group
cleanup after every exit so a target cannot orphan background children.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..tools.shell import ProcResult, venv_tool
from .base import (
    PROCESS_CLEANUP_RETURN_CODE,
    ExecutionBackend,
    ResourceLimits,
    process_group_cleanup_supported,
    run_bounded_process,
    scrub_env,
    terminate_process_group,
)


def _limited_command(cmd: list[str], limits: ResourceLimits) -> list[str]:
    if not limits.has_process_limits:
        return cmd
    launcher = [sys.executable, "-m", "lha.sandbox.limit_exec"]
    if limits.cpu_s is not None:
        launcher.extend(["--cpu-s", str(limits.cpu_s)])
    if limits.memory_mb is not None:
        launcher.extend(["--memory-mb", str(limits.memory_mb)])
    if limits.pids is not None:
        launcher.extend(["--pids", str(limits.pids)])
    return [*launcher, "--", *cmd]


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
        if not process_group_cleanup_supported():
            return ProcResult(
                PROCESS_CLEANUP_RETURN_CODE,
                "",
                (
                    "trusted-local requires POSIX process-group cleanup; "
                    "use Linux, macOS, WSL2, or the Docker backend"
                ),
                0.0,
            )

        try:
            return run_bounded_process(
                _limited_command(cmd, limits),
                cwd=str(cwd),
                env=scrub_env(),
                timeout=timeout,
                input=input,
                output_bytes=limits.output_bytes,
                start_new_session=True,  # own process group -> killable as a tree
                # Always remove the original group. A background process can
                # close stdio before the leader exits, so pipe drainage alone
                # cannot prove that no descendant survived.
                on_exit=terminate_process_group,
            )
        except OSError as e:
            return ProcResult(127, "", f"failed to start {cmd[0]!r}: {e}", 0.0)
