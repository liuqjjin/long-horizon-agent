"""Execution backend interface + shared helpers."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..tools.shell import ProcResult

# Environment variables that survive into target-code execution. Everything
# else — API keys, tokens, cloud credentials — is stripped: target code has no
# business reading the harness's secrets.
_KEEP_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TERM")
DEFAULT_OUTPUT_BYTES = 4 * 1024 * 1024
OUTPUT_LIMIT_RETURN_CODE = 125
_READ_CHUNK_BYTES = 64 * 1024
_DRAIN_GRACE_S = 1.0


def scrub_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A minimal environment for running target code."""
    env = {k: v for k in _KEEP_ENV if (v := os.environ.get(k))}
    if extra:
        env.update(extra)
    return env


@dataclass
class ResourceLimits:
    """Bounds applied to target-code execution (best effort per backend).

    Host CPU, memory, and process limits are opt-in because RLIMIT_NPROC counts
    the user's whole process table. Output capture is always bounded by default;
    ``output_bytes`` is the raw-byte limit for each of stdout and stderr. Both
    pipes continue to be drained after the limit so a noisy child cannot block.
    """

    cpu_s: int | None = None  # CPU seconds (RLIMIT_CPU)
    memory_mb: int | None = None
    pids: int | None = None
    output_bytes: int = DEFAULT_OUTPUT_BYTES

    def __post_init__(self) -> None:
        if type(self.output_bytes) is not int or self.output_bytes <= 0:
            raise ValueError("output_bytes must be a positive integer")


class _BoundedPipe:
    """Drain one pipe completely while retaining only a fixed prefix."""

    def __init__(self, limit: int):
        self.limit = limit
        self.data = bytearray()
        self.truncated = False
        self.read_failed = False

    def drain(self, stream) -> None:
        try:
            while True:
                chunk = os.read(stream.fileno(), _READ_CHUNK_BYTES)
                if not chunk:
                    return
                remaining = self.limit - len(self.data)
                if remaining > 0:
                    self.data.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        except OSError:
            # A read failure means the captured evidence is incomplete even if
            # fewer than ``limit`` bytes happened to be retained.
            self.read_failed = True


def _bounded_text(value: str, limit: int) -> str:
    return value.encode(errors="replace")[:limit].decode(errors="replace")


def _append_diagnostic(captured: str, diagnostic: str, limit: int) -> str:
    """Keep a harness diagnostic inside the same bound as child stderr."""
    separator = "\n" if captured and not captured.endswith("\n") else ""
    suffix = f"{separator}{diagnostic}"
    suffix_bytes = suffix.encode(errors="replace")
    if len(suffix_bytes) >= limit:
        return suffix_bytes[-limit:].decode(errors="replace")
    prefix_limit = limit - len(suffix_bytes)
    prefix = captured.encode(errors="replace")[:prefix_limit].decode(errors="replace")
    return f"{prefix}{suffix}"


def run_bounded_process(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: float,
    output_bytes: int,
    env: dict[str, str] | None = None,
    input: str | None = None,
    start_new_session: bool = False,
    preexec_fn: Callable[[], None] | None = None,
    on_timeout: Callable[[subprocess.Popen[bytes]], str] | None = None,
) -> ProcResult:
    """Run a process while continuously draining two bounded output pipes.

    A completed process whose output exceeds either capture bound returns 125
    and sets ``output_truncated``. This makes existing return-code gates fail
    closed while preserving an explicit reason instead of accepting a parsed
    prefix as complete evidence.
    """
    if type(output_bytes) is not int or output_bytes <= 0:
        raise ValueError("output_bytes must be a positive integer")
    start = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        start_new_session=start_new_session,
        preexec_fn=preexec_fn,
    )
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        process.kill()
        process.wait()
        raise RuntimeError("bounded process capture requires stdout and stderr pipes")

    stdout_capture = _BoundedPipe(output_bytes)
    stderr_capture = _BoundedPipe(output_bytes)
    readers = [
        threading.Thread(
            target=stdout_capture.drain,
            args=(process.stdout,),
            name="lha-stdout-drain",
            daemon=True,
        ),
        threading.Thread(
            target=stderr_capture.drain,
            args=(process.stderr,),
            name="lha-stderr-drain",
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    writer: threading.Thread | None = None
    if input is not None:
        payload = input.encode()

        def write_input() -> None:
            assert process.stdin is not None
            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        writer = threading.Thread(
            target=write_input,
            name="lha-stdin-writer",
            daemon=True,
        )
        writer.start()

    timed_out = False
    timeout_detail = ""

    def stop_process() -> str:
        if on_timeout is None:
            try:
                process.kill()
            except OSError:
                pass
            return "process killed"
        try:
            return on_timeout(process)
        except Exception:
            # Cleanup diagnostics must not strand the process whose output is
            # currently being drained.
            try:
                process.kill()
            except OSError:
                pass
            return "timeout cleanup failed"

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        timeout_detail = stop_process()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    if writer is not None:
        writer.join(timeout=1.0)
    writer_stalled = writer is not None and writer.is_alive()
    drain_deadline = time.monotonic() + _DRAIN_GRACE_S
    for reader in readers:
        reader.join(timeout=max(0.0, drain_deadline - time.monotonic()))
    drain_stalled = any(reader.is_alive() for reader in readers)
    if not timed_out and (writer_stalled or drain_stalled):
        # A child can exit after forking a descendant that keeps a standard-I/O
        # pipe open. Stop the process group/container instead of leaving the
        # descendant and pump threads behind.
        stop_process()
    if process.stdin is not None:
        try:
            if writer_stalled:
                os.close(process.stdin.fileno())
            else:
                process.stdin.close()
        except OSError:
            pass
    for stream in (process.stdout, process.stderr):
        try:
            stream.close()
        except OSError:
            pass
    for reader in readers:
        reader.join(timeout=1.0)

    stdout = bytes(stdout_capture.data).decode(errors="replace")
    stderr = bytes(stderr_capture.data).decode(errors="replace")
    limit_exceeded = stdout_capture.truncated or stderr_capture.truncated
    incomplete = (
        limit_exceeded
        or stdout_capture.read_failed
        or stderr_capture.read_failed
        or writer_stalled
        or drain_stalled
        or any(reader.is_alive() for reader in readers)
    )
    if timed_out:
        detail = f"timeout after {timeout}s"
        if timeout_detail:
            detail += f" ({timeout_detail})"
        stderr = _append_diagnostic(stderr, detail, output_bytes)
        return ProcResult(
            124,
            stdout,
            stderr,
            time.monotonic() - start,
            output_truncated=incomplete,
        )
    if incomplete:
        reason = (
            f"output exceeded the {output_bytes}-byte capture limit"
            if limit_exceeded
            else "output capture did not complete"
        )
        stderr = _append_diagnostic(
            stderr,
            reason,
            output_bytes,
        )
        return ProcResult(
            OUTPUT_LIMIT_RETURN_CODE,
            stdout,
            stderr,
            time.monotonic() - start,
            output_truncated=True,
        )
    return ProcResult(
        process.returncode,
        _bounded_text(stdout, output_bytes),
        _bounded_text(stderr, output_bytes),
        time.monotonic() - start,
    )


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
