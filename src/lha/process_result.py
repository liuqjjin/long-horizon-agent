"""Result type shared by execution backends and fixed control commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProcResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    output_truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.output_truncated
