"""Execution backend interface + shared helpers."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..process_result import ProcResult

# Environment variables that survive into target-code execution. Everything
# else — API keys, tokens, cloud credentials — is stripped: target code has no
# business reading the harness's secrets.
_KEEP_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TERM")
DEFAULT_OUTPUT_BYTES = 4 * 1024 * 1024
OUTPUT_LIMIT_RETURN_CODE = 125
PROCESS_CLEANUP_RETURN_CODE = 126
_READ_CHUNK_BYTES = 64 * 1024
_DRAIN_GRACE_S = 1.0
_PROCESS_GROUP_CLEANUP_S = 2.0


def process_group_cleanup_supported() -> bool:
    """Return whether the host provides POSIX process-group cleanup."""
    return os.name == "posix" and hasattr(os, "killpg")


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
        for name in ("cpu_s", "memory_mb", "pids"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer when set")
        if type(self.output_bytes) is not int or self.output_bytes <= 0:
            raise ValueError("output_bytes must be a positive integer")

    @property
    def has_process_limits(self) -> bool:
        return any(
            value is not None
            for value in (self.cpu_s, self.memory_mb, self.pids)
        )


@dataclass(frozen=True)
class ProcessCleanupResult:
    """Whether a process boundary was removed and independently confirmed."""

    confirmed: bool
    detail: str


def terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    confirmation_timeout_s: float = _PROCESS_GROUP_CLEANUP_S,
) -> ProcessCleanupResult:
    """Kill the session leader's original process group and confirm its absence.

    Waiting only for the leader is insufficient: a descendant can close stdout
    and stderr, keep running, and therefore leave no stalled pipe for the
    capture code to detect. ``start_new_session=True`` makes the leader PID the
    process-group ID, so the group remains addressable after the leader exits.
    """
    if not process_group_cleanup_supported():
        return ProcessCleanupResult(
            False,
            "POSIX process-group cleanup is unavailable on this platform",
        )

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return ProcessCleanupResult(True, "process group absent")
    except OSError as error:
        return ProcessCleanupResult(
            False,
            f"could not kill process group {process_group}: {error}",
        )

    deadline = time.monotonic() + confirmation_timeout_s
    while True:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return ProcessCleanupResult(True, "process group killed")
        except OSError as error:
            return ProcessCleanupResult(
                False,
                f"could not confirm process group {process_group} cleanup: {error}",
            )
        if time.monotonic() >= deadline:
            return ProcessCleanupResult(
                False,
                f"process group {process_group} still exists after cleanup",
            )
        time.sleep(0.01)


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
    on_timeout: Callable[[subprocess.Popen[bytes]], str] | None = None,
    on_exit: (
        Callable[[subprocess.Popen[bytes]], ProcessCleanupResult] | None
    ) = None,
) -> ProcResult:
    """Run a process while continuously draining two bounded output pipes.

    A completed process whose output exceeds either capture bound returns 125
    and sets ``output_truncated``. This makes existing return-code gates fail
    closed while preserving an explicit reason instead of accepting a parsed
    prefix as complete evidence.
    """
    if type(output_bytes) is not int or output_bytes <= 0:
        raise ValueError("output_bytes must be a positive integer")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    if on_exit is not None and not start_new_session:
        raise ValueError("on_exit cleanup requires start_new_session=True")
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
    )
    stdout_capture = _BoundedPipe(output_bytes)
    stderr_capture = _BoundedPipe(output_bytes)
    readers: list[threading.Thread] = []
    writer: threading.Thread | None = None
    writer_started = False
    timed_out = False
    timeout_detail = ""
    interrupted: BaseException | None = None

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

    def reap_leader() -> None:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except (ChildProcessError, OSError):
            pass

    try:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError(
                "bounded process capture requires stdout and stderr pipes"
            )
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

        if input is not None:
            if process.stdin is None:
                raise RuntimeError("bounded process input pipe was not created")
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
            writer_started = True

        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        timeout_detail = stop_process()
        reap_leader()
    except BaseException as error:
        # KeyboardInterrupt and cancellation must not leave the process tree
        # behind merely because no ProcResult will be returned to the caller.
        interrupted = error
        stop_process()
        reap_leader()

    cleanup = ProcessCleanupResult(
        process.poll() is not None,
        "process leader exited" if process.poll() is not None else "process leader remains",
    )
    if on_exit is not None:
        try:
            cleanup = on_exit(process)
        except BaseException as error:
            cleanup = ProcessCleanupResult(
                False,
                f"process cleanup raised {type(error).__name__}: {error}",
            )
        if not isinstance(cleanup, ProcessCleanupResult):
            cleanup = ProcessCleanupResult(
                False,
                "process cleanup returned an invalid result",
            )

    join_error: BaseException | None = None
    if writer is not None and writer_started:
        try:
            writer.join(timeout=1.0)
        except BaseException as error:
            join_error = error
    try:
        writer_stalled = (
            writer is not None and writer_started and writer.is_alive()
        )
    except BaseException as error:
        join_error = join_error or error
        writer_stalled = True
    drain_deadline = time.monotonic() + _DRAIN_GRACE_S
    for reader in readers:
        if reader.ident is None:
            continue
        try:
            reader.join(
                timeout=max(0.0, drain_deadline - time.monotonic())
            )
        except BaseException as error:
            join_error = join_error or error
    try:
        drain_stalled = any(
            reader.ident is not None and reader.is_alive()
            for reader in readers
        )
    except BaseException as error:
        join_error = join_error or error
        drain_stalled = True
    if process.stdin is not None:
        try:
            if writer_stalled:
                os.close(process.stdin.fileno())
            else:
                process.stdin.close()
        except OSError:
            pass
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass
    for reader in readers:
        if reader.ident is None:
            continue
        try:
            reader.join(timeout=1.0)
        except BaseException as error:
            join_error = join_error or error

    try:
        readers_alive = any(
            reader.ident is not None and reader.is_alive()
            for reader in readers
        )
    except BaseException as error:
        join_error = join_error or error
        readers_alive = True

    stdout = bytes(stdout_capture.data).decode(errors="replace")
    stderr = bytes(stderr_capture.data).decode(errors="replace")
    if not cleanup.confirmed:
        stderr = _append_diagnostic(
            stderr,
            f"process cleanup could not be confirmed: {cleanup.detail}",
            output_bytes,
        )
    limit_exceeded = stdout_capture.truncated or stderr_capture.truncated
    incomplete = (
        limit_exceeded
        or stdout_capture.read_failed
        or stderr_capture.read_failed
        or writer_stalled
        or drain_stalled
        or readers_alive
        or join_error is not None
    )
    if interrupted is not None:
        if not cleanup.confirmed:
            interrupted.add_note(
                f"process cleanup could not be confirmed: {cleanup.detail}"
            )
        raise interrupted
    if not cleanup.confirmed:
        return ProcResult(
            PROCESS_CLEANUP_RETURN_CODE,
            stdout,
            stderr,
            time.monotonic() - start,
            output_truncated=incomplete,
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
            else (
                f"output capture failed: {type(join_error).__name__}"
                if join_error is not None
                else "output capture did not complete"
            )
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
