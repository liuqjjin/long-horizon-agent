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
    # ``None`` means the caller did not request a separately confirmed cleanup
    # boundary. A false value is never an ordinary command failure: code may
    # still be executing and mutating its working directory.
    cleanup_confirmed: bool | None = None
    cleanup_detail: str = ""

    def __post_init__(self) -> None:
        # The numeric status can also be a real target exit code (Docker
        # forwards it). Only the structured flag identifies cleanup failure.
        if self.cleanup_confirmed is False:
            self.returncode = 126
        if self.cleanup_confirmed is False and not self.cleanup_detail:
            self.cleanup_detail = self.stderr or "process cleanup was not confirmed"

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.output_truncated

    @property
    def cleanup_unconfirmed(self) -> bool:
        return self.cleanup_confirmed is False
