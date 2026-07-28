"""Bounded helper for fixed control-plane commands."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ..process_result import ProcResult

_CONTROL_OUTPUT_BYTES = 4 * 1024 * 1024


def run(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: float = 300.0,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> ProcResult:
    """Run a fixed harness command without inheriting secrets or orphaning children."""
    from ..sandbox.base import (
        PROCESS_CLEANUP_RETURN_CODE,
        process_group_cleanup_supported,
        run_bounded_process,
        scrub_env,
        terminate_process_group,
    )

    if not process_group_cleanup_supported():
        return ProcResult(
            PROCESS_CLEANUP_RETURN_CODE,
            "",
            (
                "control command requires POSIX process-group cleanup; "
                "use Linux, macOS, or WSL2"
            ),
            0.0,
        )
    try:
        return run_bounded_process(
            cmd,
            cwd=cwd,
            timeout=timeout,
            output_bytes=_CONTROL_OUTPUT_BYTES,
            env=env if env is not None else scrub_env(),
            input=input,
            start_new_session=True,
            on_exit=terminate_process_group,
        )
    except OSError as error:
        return ProcResult(
            127,
            "",
            f"failed to execute {cmd[0] if cmd else '<empty>'}: {error}",
            0.0,
        )


def venv_tool(name: str) -> str:
    """Resolve a console-script tool from the running interpreter's venv first."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    return found or name
