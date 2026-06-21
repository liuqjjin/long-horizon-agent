"""Thin, capture-everything subprocess helper used by verifiers and tools."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: float = 300.0,
    env: dict[str, str] | None = None,
) -> ProcResult:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return ProcResult(proc.returncode, proc.stdout, proc.stderr, time.monotonic() - start)
    except subprocess.TimeoutExpired as e:
        return ProcResult(124, e.stdout or "", f"timeout after {timeout}s", time.monotonic() - start)


def venv_tool(name: str) -> str:
    """Resolve a console-script tool from the running interpreter's venv first."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    return found or name
