"""LLM backend that shells out to the already-authenticated ``codex`` CLI.

``codex exec`` is the non-interactive entry point: the prompt goes in on stdin,
the final assistant message comes back via ``--output-last-message``, and
``--json`` emits one JSONL event per step so the call can be audited afterwards.

Two differences from the ``claude`` backend matter for the ablation.

**Isolation.** ``--ignore-user-config`` still loads the user's plugins and
skills, which cost ~6k input tokens here and could inject instructions into a
run that is supposed to measure the model. So the client builds a throwaway
``CODEX_HOME`` containing nothing but a copy of ``auth.json``: no config, no
plugins, no skills, no MCP servers, no notify hooks. Two runs on two machines
then see the same prompt.

**Leak-freedom is enforced by detection, not by a deny-list.** ``codex`` has no
``--disallowed-tools`` equivalent, and its most restrictive sandbox
(``-s read-only``) still permits reading the whole filesystem — so an agentic
model could in principle go and find the canonical test files the ablation
deliberately withheld. Instead of naming tools to forbid, the client audits every
event stream and **fails the call if the model used any tool at all**; ``no_tools``
also adds a prompt-only preamble to reduce the refusal rate. The audit is strictly
stronger than a deny-list, which a renamed tool in a future CLI would silently
defeat: here anything other than model-authored text is a failure, whatever it is
called.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..bench.codex_exec_events import (
    CodexEventError as StrictCodexEventError,
)
from ..bench.codex_exec_events import (
    CodexJsonlValidator,
)
from .base import LLMClient

# Event items that are a model *talking*. Anything else means the model reached
# outside the prompt, which breaks the ablation's leak-freedom guarantee.
_TEXT_ITEMS = frozenset({"agent_message", "reasoning", "todo_list"})
_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "item.started",
        "item.updated",
        "item.completed",
        "turn.completed",
        "turn.failed",
        "error",
    }
)
_TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection reset",
    "connection refused",
    "connection aborted",
    "network",
    "overloaded",
    "rate limit",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "try again",
)

# Passing the parent environment through to an agent process would expose every
# API key, cloud credential, and task-specific secret in that process. Codex
# needs executable lookup, locale, and (on some networks) an explicit proxy or
# custom CA bundle; authentication itself comes only from the copied auth.json.
_PASSTHROUGH_ENV = (
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)
_PROCESS_TERM_GRACE_S = 0.25
_PROCESS_KILL_GRACE_S = 2.0
_SUPPORTED_CLI_VERSION = "codex-cli 0.141.0"
_MAX_JSONL_LINE_BYTES = 2 * 1024 * 1024
_MAX_JSONL_BYTES = 16 * 1024 * 1024
_MAX_STDERR_BYTES = 1024 * 1024
_MAX_FINAL_MESSAGE_BYTES = 4 * 1024 * 1024
_OUTPUT_READ_CHUNK_BYTES = 64 * 1024

# Why this preamble exists, measured rather than assumed: the implementer prompt
# lists files as "### <path>", and codex reads that as "you are in a repository".
# On a sample of ablation tasks the model went looking for those paths — `find .
# -name brackets.py`, and in one case `find /private/var/.../T -name cipher.py`,
# which walks the very temp tree the experiment keeps its scratch copies in. The
# audit below catches that, but a run where a third of the cells are refused has
# no statistical power, so the model is also told plainly that there is nothing to
# look at. The instruction lowers the rate; the audit is what makes it a guarantee.
_NO_TOOLS_PREAMBLE = (
    "IMPORTANT: this is a text-only task. There is no repository and no filesystem "
    "here — the working directory is empty on purpose. Every file you need is "
    "reproduced verbatim in the message below. Do not run shell commands, do not "
    "search for or open files, and do not try to locate anything on disk: there is "
    "nothing to find, and any tool call causes this answer to be discarded. Reason "
    "from the message alone."
)


def _minimal_subprocess_env(*, codex_home: Path, temp_dir: Path) -> dict[str, str]:
    """Build the complete environment for a Codex subprocess.

    HOME and all XDG/temp roots point at attempt-local directories. In
    particular, this intentionally does not copy OPENAI_API_KEY, AWS_*, SSH
    agent sockets, NODE_OPTIONS, or arbitrary caller variables.
    """
    env = {"PATH": os.environ.get("PATH") or os.defpath}
    for name in _PASSTHROUGH_ENV:
        value = os.environ.get(name)
        if value:
            env[name] = value
    home = str(codex_home)
    temporary = str(temp_dir)
    env.update(
        {
            "HOME": home,
            "CODEX_HOME": home,
            "XDG_CONFIG_HOME": str(codex_home / "xdg-config"),
            "XDG_CACHE_HOME": str(codex_home / "xdg-cache"),
            "XDG_STATE_HOME": str(codex_home / "xdg-state"),
            "TMPDIR": temporary,
            "TMP": temporary,
            "TEMP": temporary,
        }
    )
    return env


def _process_group_exists(pgid: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It still exists; lack of permission must not be mistaken for cleanup.
        return True
    return True


def _terminate_process_group(proc: subprocess.Popen[Any]) -> bool:
    """Stop the process tree and confirm that its process group is absent."""
    if os.name == "posix":
        pgid = proc.pid  # start_new_session=True makes the child its group leader.
        if _process_group_exists(pgid):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.monotonic() + _PROCESS_TERM_GRACE_S
        while _process_group_exists(pgid) and time.monotonic() < deadline:
            # Reap an exited leader promptly. On macOS an unreaped group-leader
            # zombie still answers signal 0, then rejects SIGKILL with EPERM.
            proc.poll()
            time.sleep(0.01)

        # The CLI or one of its descendants may ignore SIGTERM. Kill the group,
        # including descendants left behind after the leader has already exited.
        proc.poll()
        if _process_group_exists(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=_PROCESS_KILL_GRACE_S)
        except (subprocess.TimeoutExpired, ChildProcessError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=_PROCESS_KILL_GRACE_S)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
        deadline = time.monotonic() + _PROCESS_KILL_GRACE_S
        while _process_group_exists(pgid) and time.monotonic() < deadline:
            proc.poll()
            time.sleep(0.01)
        return proc.poll() is not None and not _process_group_exists(pgid)

    # Windows has no killpg equivalent in the standard library. A new process
    # group still prevents signal sharing with the parent; terminate/kill is the
    # best standard-library fallback for the direct process.
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=_PROCESS_TERM_GRACE_S)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
                proc.wait(timeout=_PROCESS_KILL_GRACE_S)
            except (OSError, subprocess.TimeoutExpired):
                pass
    return proc.poll() is not None


def _attempt_process_group_cleanup(
    proc: subprocess.Popen[Any],
) -> tuple[bool, BaseException | None]:
    """Keep the process handle available even when the cleanup helper itself fails."""
    try:
        return _terminate_process_group(proc), None
    except BaseException as error:
        return False, error


class _ProcessLifecycle:
    """Own exactly one termination attempt for one newly spawned process."""

    def __init__(self, process: subprocess.Popen[Any]):
        self.process = process
        self.attempted = False
        self.cleaned = False
        self.error: BaseException | None = None

    def stop_once(self) -> tuple[bool, BaseException | None]:
        if not self.attempted:
            self.attempted = True
            self.cleaned, self.error = _attempt_process_group_cleanup(self.process)
        return self.cleaned, self.error


def _join_started_threads(
    threads: list[threading.Thread],
) -> BaseException | None:
    deadline = time.monotonic() + _PROCESS_KILL_GRACE_S
    first_error: BaseException | None = None
    for thread in threads:
        if thread.ident is None:
            continue
        try:
            thread.join(max(0.0, deadline - time.monotonic()))
        except BaseException as error:
            first_error = first_error or error
    try:
        if any(thread.is_alive() for thread in threads):
            first_error = first_error or RuntimeError("pipe thread did not stop")
    except BaseException as error:
        first_error = first_error or error
    return first_error


def _close_process_pipes(proc: subprocess.Popen[Any]) -> BaseException | None:
    """Close parent pipe endpoints even when no worker thread took ownership."""

    first_error: BaseException | None = None
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is None:
            continue
        try:
            pipe.close()
        except BaseException as error:
            first_error = first_error or error
    return first_error


class _ProcessOutputError(RuntimeError):
    """A child stream could not be captured within its registered boundary."""


def _read_bounded_pipe(
    pipe: Any,
    *,
    stream_name: str,
    maximum_bytes: int,
    maximum_line_bytes: int | None,
    sink: bytearray,
    failures: list[BaseException],
    failure_lock: threading.Lock,
    abort: threading.Event,
) -> None:
    """Drain one binary pipe without ever retaining more than its byte limit."""

    current_line_bytes = 0
    try:
        descriptor = pipe.fileno()
        while True:
            # BufferedReader.read(n) may wait for all n bytes even when a smaller
            # violating chunk is already in the pipe. os.read returns available
            # bytes, so the process is stopped as soon as a boundary is crossed.
            chunk = os.read(descriptor, _OUTPUT_READ_CHUNK_BYTES)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise _ProcessOutputError(
                    f"codex {stream_name} pipe did not produce binary output"
                )
            if len(sink) + len(chunk) > maximum_bytes:
                remaining = max(0, maximum_bytes - len(sink))
                sink.extend(chunk[:remaining])
                raise _ProcessOutputError(
                    f"codex {stream_name} exceeded the {maximum_bytes}-byte limit"
                )
            if maximum_line_bytes is not None:
                cursor = 0
                while cursor < len(chunk):
                    newline = chunk.find(b"\n", cursor)
                    if newline < 0:
                        current_line_bytes += len(chunk) - cursor
                        if current_line_bytes > maximum_line_bytes:
                            raise _ProcessOutputError(
                                "codex stdout JSONL line exceeded the "
                                f"{maximum_line_bytes}-byte limit"
                            )
                        break
                    current_line_bytes += newline - cursor + 1
                    if current_line_bytes > maximum_line_bytes:
                        raise _ProcessOutputError(
                            "codex stdout JSONL line exceeded the "
                            f"{maximum_line_bytes}-byte limit"
                        )
                    current_line_bytes = 0
                    cursor = newline + 1
            sink.extend(chunk)
    except BaseException as exc:
        with failure_lock:
            failures.append(exc)
        abort.set()
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _write_process_input(
    pipe: Any,
    payload: bytes,
    *,
    failures: list[BaseException],
    failure_lock: threading.Lock,
    abort: threading.Event,
) -> None:
    """Write stdin concurrently so output-first children cannot deadlock the parent."""

    try:
        if payload:
            pipe.write(payload)
            pipe.flush()
    except BrokenPipeError:
        # A process that exits before consuming stdin is accounted for by its
        # return code and audited output.
        pass
    except BaseException as exc:
        with failure_lock:
            failures.append(exc)
        abort.set()
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def _communicate_bounded(
    proc: subprocess.Popen[Any],
    *,
    argv: list[str],
    input_payload: bytes,
    timeout: float,
    lifecycle: _ProcessLifecycle,
) -> tuple[str, str]:
    """Capture both streams while a shared lifecycle owns process termination."""

    stdout_bytes = bytearray()
    stderr_bytes = bytearray()
    failures: list[BaseException] = []
    failure_lock: threading.Lock | None = None
    abort: threading.Event | None = None
    started_threads: list[threading.Thread] = []
    wait_error: BaseException | None = None
    try:
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise _ProcessOutputError("codex process pipes were not created")
        failure_lock = threading.Lock()
        abort = threading.Event()
        threads = [
            threading.Thread(
                target=_read_bounded_pipe,
                kwargs={
                    "pipe": proc.stdout,
                    "stream_name": "stdout",
                    "maximum_bytes": _MAX_JSONL_BYTES,
                    "maximum_line_bytes": _MAX_JSONL_LINE_BYTES,
                    "sink": stdout_bytes,
                    "failures": failures,
                    "failure_lock": failure_lock,
                    "abort": abort,
                },
                name="lha-codex-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=_read_bounded_pipe,
                kwargs={
                    "pipe": proc.stderr,
                    "stream_name": "stderr",
                    "maximum_bytes": _MAX_STDERR_BYTES,
                    "maximum_line_bytes": None,
                    "sink": stderr_bytes,
                    "failures": failures,
                    "failure_lock": failure_lock,
                    "abort": abort,
                },
                name="lha-codex-stderr",
                daemon=True,
            ),
            threading.Thread(
                target=_write_process_input,
                args=(proc.stdin, input_payload),
                kwargs={
                    "failures": failures,
                    "failure_lock": failure_lock,
                    "abort": abort,
                },
                name="lha-codex-stdin",
                daemon=True,
            ),
        ]
        for thread in threads:
            started_threads.append(thread)
            thread.start()
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                wait_error = subprocess.TimeoutExpired(argv, timeout)
                break
            if abort.wait(min(remaining, 0.05)):
                break
    except BaseException as exc:
        wait_error = exc
    finally:
        lifecycle.stop_once()
        try:
            join_error = _join_started_threads(started_threads)
        except BaseException as error:
            join_error = error
        pipe_close_error = _close_process_pipes(proc)

    if isinstance(wait_error, (KeyboardInterrupt, SystemExit)):
        raise wait_error
    if failure_lock is None:
        captured_failures = ()
    else:
        with failure_lock:
            captured_failures = tuple(failures)
    output_failure = next(
        (
            failure
            for failure in captured_failures
            if isinstance(failure, _ProcessOutputError)
        ),
        None,
    )
    if output_failure is not None:
        raise output_failure
    if wait_error is not None:
        raise wait_error
    if captured_failures:
        raise captured_failures[0]
    if join_error is not None:
        raise _ProcessOutputError(
            "codex output pipes remained open after process-group cleanup"
        ) from join_error
    if pipe_close_error is not None:
        raise _ProcessOutputError(
            "codex process pipes could not be closed after process-group cleanup"
        ) from pipe_close_error
    try:
        return bytes(stdout_bytes).decode("utf-8"), bytes(stderr_bytes).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _ProcessOutputError("codex process output is not UTF-8") from exc


def _run_isolated_process(
    argv: list[str],
    *,
    input: str | None,
    capture_output: bool,
    text: bool,
    timeout: float,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one process with a single, checked process-lifecycle owner."""
    if not capture_output or not text:
        raise ValueError("isolated Codex processes require captured text output")
    input_payload = (input or "").encode("utf-8")
    common: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": dict(env),
    }
    if os.name == "posix":
        proc = subprocess.Popen(argv, start_new_session=True, **common)
    else:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(argv, creationflags=creationflags, **common)
    lifecycle: _ProcessLifecycle | None = None
    try:
        lifecycle = _ProcessLifecycle(proc)
        stdout, stderr = _communicate_bounded(
            proc,
            argv=argv,
            input_payload=input_payload,
            timeout=timeout,
            lifecycle=lifecycle,
        )
    except BaseException as primary_error:
        cleaned, cleanup_error = (
            lifecycle.stop_once()
            if lifecycle is not None
            else _attempt_process_group_cleanup(proc)
        )
        if not cleaned:
            raise CodexProcessCleanupError(
                "Codex CLI process group could not be confirmed stopped",
                process=proc,
                primary_error=primary_error,
                cleanup_error=cleanup_error,
            ) from (cleanup_error or primary_error)
        raise
    else:
        cleaned, cleanup_error = lifecycle.stop_once()
        if not cleaned:
            raise CodexProcessCleanupError(
                "Codex CLI process group could not be confirmed stopped",
                process=proc,
                cleanup_error=cleanup_error,
            ) from cleanup_error
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


class CodexCLIError(RuntimeError):
    """Base error carrying whether repeating the same call can be safe."""

    retryable = False


class CodexProtocolError(CodexCLIError):
    """The CLI returned a malformed, incomplete, or unsafe event stream."""


class CodexTransientError(CodexCLIError):
    """A temporary transport or service failure that may succeed on retry."""

    retryable = True


class CodexInvocationError(CodexCLIError):
    """The local CLI could not be invoked correctly; retrying would not help."""


class CodexCleanupError(CodexCLIError):
    """Attempt-local files could not be removed and must be inspected or retried."""


class CodexProcessCleanupError(CodexCleanupError):
    """A process group remains unconfirmed, so attempt-local files must stay."""

    def __init__(
        self,
        message: str,
        *,
        process: subprocess.Popen[Any],
        primary_error: BaseException | None = None,
        cleanup_error: BaseException | None = None,
    ):
        super().__init__(message)
        self.process = process
        self.primary_error_type = (
            type(primary_error).__name__ if primary_error is not None else None
        )
        self.cleanup_error_type = (
            type(cleanup_error).__name__ if cleanup_error is not None else None
        )


class CodexToolUse(CodexProtocolError):
    """The model reached outside the supplied prompt, invalidating the result."""


class CodexCLIClient(LLMClient):
    name = "codex_cli"
    reserves_cli_attempts = True

    def __init__(
        self,
        cli_path: str = "codex",
        timeout: float = 300.0,
        model: str | None = None,
        reasoning_effort: str = "medium",
        no_tools: bool = False,
        sandbox_mode: str = "read-only",
        externally_sandboxed: bool = False,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        if (
            isinstance(retry_backoff_s, bool)
            or not isinstance(retry_backoff_s, (int, float))
            or not math.isfinite(retry_backoff_s)
            or retry_backoff_s < 0
        ):
            raise ValueError("retry_backoff_s must be a non-negative finite number")
        if not cli_path or "\x00" in cli_path:
            raise ValueError("cli_path must be a non-empty executable name")
        if model is not None and "\x00" in model:
            raise ValueError("model must not contain NUL bytes")
        if sandbox_mode not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError(f"unsupported Codex sandbox mode: {sandbox_mode!r}")
        if sandbox_mode == "danger-full-access" and not externally_sandboxed:
            raise ValueError(
                "danger-full-access requires externally_sandboxed=True; "
                "use it only inside a disposable outer sandbox"
            )
        self.cli_path = cli_path
        self.timeout = timeout
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.no_tools = no_tools
        self.sandbox_mode = sandbox_mode
        self.externally_sandboxed = externally_sandboxed
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        # usage of the most recent call (tokens), for the run trace
        self.last_usage: dict | None = None
        # commands the most recent call ran, if any — evidence for the audit
        self.last_tool_use: list[str] = []
        # Full audit metadata, including failures that do not report token usage.
        self.last_call: dict[str, Any] | None = None
        self.last_event_summary: dict[str, Any] = self._new_event_summary()
        self._home: Path | None = None
        self._workspace: Path | None = None
        self._output_dirs: set[Path] = set()
        self._pending_process: subprocess.Popen[Any] | None = None
        self.last_cleanup_failures: tuple[str, ...] = ()
        self._attempt_reserver: Callable[[], None] | None = None
        self._version: str | None = None

    # --- isolated environment ------------------------------------------------
    def _clean_home(self) -> Path:
        """A CODEX_HOME carrying only credentials, scoped to one attempt.

        ``--ignore-user-config`` skips ``config.toml`` but not plugins/skills, and
        the CLI reads auth from ``CODEX_HOME``, so the only way to get a clean
        environment while staying logged in is to point it at a directory holding
        just ``auth.json``.
        """
        if self._home is not None:
            return self._home
        source = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"
        if not source.exists():
            raise CodexInvocationError(
                "codex is not logged in; run `codex login` before starting LHA"
            )
        home = Path(tempfile.mkdtemp(prefix="lha_codex_home_"))
        self._home = home
        try:
            home.chmod(0o700)
            shutil.copy2(source, home / "auth.json")
            (home / "auth.json").chmod(0o600)
        except BaseException as setup_error:
            try:
                self.cleanup()
            except CodexCleanupError as cleanup_error:
                raise cleanup_error from setup_error
            raise
        return home

    def _empty_workspace(self) -> Path:
        """An empty working root for the agent.

        ``propose_patch`` passes the source in the prompt, so the model needs no
        workspace at all. Giving it an empty one removes the files it would
        otherwise be able to see without even reaching for a tool.
        """
        if self._workspace is None:
            self._workspace = Path(tempfile.mkdtemp(prefix="lha_codex_ws_"))
        return self._workspace

    def cleanup(self) -> None:
        """Remove attempt-local state; retain failed paths so cleanup can be retried."""
        pending_process = self._pending_process
        if pending_process is not None:
            try:
                leader_returncode = pending_process.poll()
            except BaseException as error:
                self.last_cleanup_failures = (
                    "temporary Codex state: process group recheck failed",
                )
                raise CodexProcessCleanupError(
                    "Codex CLI process group could not be rechecked; "
                    "attempt-local files remain retained",
                    process=pending_process,
                    cleanup_error=error,
                ) from error
            if os.name == "posix" and leader_returncode is not None:
                try:
                    cleaned = not _process_group_exists(pending_process.pid)
                except BaseException as error:
                    self.last_cleanup_failures = (
                        "temporary Codex state: process group recheck failed",
                    )
                    raise CodexProcessCleanupError(
                        "Codex CLI process group could not be rechecked; "
                        "attempt-local files remain retained",
                        process=pending_process,
                        cleanup_error=error,
                    ) from error
                cleanup_error = None
            else:
                cleaned, cleanup_error = _attempt_process_group_cleanup(pending_process)
            if cleanup_error is not None:
                self.last_cleanup_failures = (
                    "temporary Codex state: process group recheck failed",
                )
                raise CodexProcessCleanupError(
                    "Codex CLI process group could not be rechecked; "
                    "attempt-local files remain retained",
                    process=pending_process,
                    cleanup_error=cleanup_error,
                ) from cleanup_error
            if not cleaned:
                self.last_cleanup_failures = (
                    "temporary Codex state: process group still present",
                )
                raise CodexProcessCleanupError(
                    "Codex CLI process group is still present; "
                    "attempt-local files remain retained",
                    process=pending_process,
                )
            self._pending_process = None

        failures: list[str] = []

        def remove(path: Path | None, label: str) -> bool:
            if path is None:
                return True
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                return True
            except Exception as error:
                # A concurrent remover may have won after rmtree raised. Verify
                # absence before deciding that a credential-bearing path remains.
                if not os.path.lexists(path):
                    return True
                errno = getattr(error, "errno", None)
                detail = type(error).__name__
                if isinstance(errno, int):
                    detail = f"{detail} errno={errno}"
                failures.append(f"{label}: {detail}")
                return False
            return True

        if remove(self._home, "temporary Codex home"):
            self._home = None
        if remove(self._workspace, "temporary Codex workspace"):
            self._workspace = None
        for path in tuple(self._output_dirs):
            if remove(path, "temporary Codex output"):
                self._output_dirs.discard(path)

        self.last_cleanup_failures = tuple(failures)
        if failures:
            labels = ", ".join(failure.split(":", 1)[0] for failure in failures)
            raise CodexCleanupError(
                f"could not remove {labels}; paths remain retained for a cleanup retry"
            )

    @property
    def pending_cleanup_paths(self) -> tuple[Path, ...]:
        """Return local paths still awaiting cleanup; never serialize these paths."""
        paths = [path for path in (self._home, self._workspace) if path is not None]
        paths.extend(sorted(self._output_dirs))
        return tuple(paths)

    def set_attempt_reserver(self, reserver: Callable[[], None] | None) -> None:
        """Install the tracer's write-ahead reservation for each CLI process attempt."""
        self._attempt_reserver = reserver

    def _cli_version(self) -> str:
        if self._version is not None:
            return self._version
        scratch = Path(tempfile.mkdtemp(prefix="lha_codex_version_"))
        self._output_dirs.add(scratch)
        process_cleanup_failed = False
        try:
            res = _run_isolated_process(
                [self.cli_path, "--version"],
                input=None,
                capture_output=True,
                text=True,
                timeout=30,
                env=_minimal_subprocess_env(codex_home=scratch, temp_dir=scratch),
            )
        except CodexProcessCleanupError as error:
            self._pending_process = error.process
            process_cleanup_failed = True
            self.last_cleanup_failures = (
                "temporary Codex state: process group still present",
            )
            raise
        except (OSError, subprocess.TimeoutExpired, _ProcessOutputError):
            version = "unknown"
        else:
            version = (
                (res.stdout.strip() or res.stderr.strip()) if res.returncode == 0 else "unknown"
            )
        finally:
            if not process_cleanup_failed:
                self.cleanup()
        self._version = version or "unknown"
        return self._version

    def backend_provenance(self) -> str:
        """Everything about this client that can change an outcome.

        The driving CLI's version and the reasoning effort both do, and both were
        outside the ablation's cache fingerprint before: `docs/ABLATION.md` lists
        the un-fingerprinted CLI version as a known weakness of the earlier runs.
        Folding them in means a CLI upgrade or an effort change re-samples the
        cells instead of quietly mixing two generations of results.
        """
        return (
            f"{self._cli_version()} model={self.model or 'cli-default'} "
            f"effort={self.reasoning_effort} sandbox={self.sandbox_mode}"
        )

    # --- the call ------------------------------------------------------------
    def _argv(self, out_path: Path) -> list[str]:
        argv = [
            self.cli_path,
            "exec",
            "--ephemeral",  # 200+ ablation cells must not leave 200+ session files
            "--ignore-user-config",  # only the attempt-local auth file may influence the run
            "--skip-git-repo-check",  # cells run in temp dirs, not checkouts
            "--ignore-rules",  # no project/user execpolicy files
            "--sandbox",
            self.sandbox_mode,
            "-C",
            str(self._empty_workspace()),
            "--json",
            "-o",
            str(out_path),
        ]
        if self.model:
            argv += ["-m", self.model]
        if self.reasoning_effort:
            argv += ["-c", f"model_reasoning_effort={self.reasoning_effort!r}"]
        argv.append("-")  # read the prompt from stdin
        return argv

    def complete(self, system: str, prompt: str) -> str:
        # codex exec takes a single prompt; the system instructions are prepended
        # rather than passed separately (there is no --append-system-prompt here).
        self.last_usage = None
        self.last_tool_use = []
        self.last_call = None
        self.last_event_summary = self._new_event_summary()
        self.last_cleanup_failures = ()
        call_started = time.monotonic()
        try:
            self.cleanup()
        except Exception as error:
            self._finish_call(
                started=call_started,
                version=self._version or "unknown",
                attempts=[],
                status="failed",
                error=error,
            )
            raise
        if self.no_tools:
            system = f"{_NO_TOOLS_PREAMBLE}\n\n{system}"
        attempts: list[dict[str, Any]] = []
        try:
            version = self._cli_version()
        except Exception as error:
            self._finish_call(
                started=call_started,
                version=self._version or "unknown",
                attempts=attempts,
                status="failed",
                error=error,
            )
            raise
        if version != _SUPPORTED_CLI_VERSION:
            error = CodexProtocolError(
                "unsupported Codex CLI protocol version: "
                f"expected {_SUPPORTED_CLI_VERSION!r}, got {version!r}"
            )
            self._finish_call(
                started=call_started,
                version=version,
                attempts=attempts,
                status="failed",
                error=error,
            )
            raise error
        for attempt in range(self.max_retries + 1):
            if self._attempt_reserver is not None:
                try:
                    self._attempt_reserver()
                except Exception as exc:
                    self._finish_call(
                        started=call_started,
                        version=version,
                        attempts=attempts,
                        status="failed",
                        error=exc,
                    )
                    raise
            attempt_started = time.monotonic()
            try:
                answer = self._complete_once(system, prompt)
            except Exception as exc:
                attempt_record: dict[str, Any] = {
                    "attempt": attempt + 1,
                    "status": "failed",
                    "duration_s": round(time.monotonic() - attempt_started, 3),
                    "error_type": type(exc).__name__,
                    "retryable": bool(getattr(exc, "retryable", False)),
                    "event_summary": self.last_event_summary,
                }
                if isinstance(exc, CodexProcessCleanupError):
                    attempt_record["primary_error_type"] = exc.primary_error_type
                    attempt_record["cleanup_error_type"] = exc.cleanup_error_type
                attempts.append(attempt_record)
                if not isinstance(exc, CodexTransientError) or attempt >= self.max_retries:
                    self._finish_call(
                        started=call_started,
                        version=version,
                        attempts=attempts,
                        status="failed",
                        error=exc,
                    )
                    raise
                if self.retry_backoff_s:
                    time.sleep(self.retry_backoff_s * (2**attempt))
            else:
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "succeeded",
                        "duration_s": round(time.monotonic() - attempt_started, 3),
                        "event_summary": self.last_event_summary,
                    }
                )
                self._finish_call(
                    started=call_started,
                    version=version,
                    attempts=attempts,
                    status="succeeded",
                )
                return answer
        raise AssertionError("unreachable")

    def _complete_once(self, system: str, prompt: str) -> str:
        process_cleanup_failed = False
        try:
            out_dir = Path(tempfile.mkdtemp(prefix="lha_codex_out_"))
            self._output_dirs.add(out_dir)
            out_path = out_dir / "last_message.txt"
            env = _minimal_subprocess_env(
                codex_home=self._clean_home(),
                temp_dir=out_dir,
            )
            try:
                proc = _run_isolated_process(
                    self._argv(out_path),
                    input=f"{system}\n\n---\n\n{prompt}",
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                )
            except CodexProcessCleanupError as error:
                self._pending_process = error.process
                process_cleanup_failed = True
                self.last_cleanup_failures = (
                    "temporary Codex state: process group still present",
                )
                raise
            except subprocess.TimeoutExpired as e:
                raise CodexTransientError(f"codex CLI timed out after {self.timeout}s") from e
            except _ProcessOutputError as e:
                raise CodexProtocolError(str(e)) from e
            except OSError as e:
                raise CodexInvocationError(
                    f"could not execute codex CLI ({type(e).__name__})"
                ) from e
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout)[-400:]
                error_type = (
                    CodexTransientError
                    if proc.returncode == 124 or self._looks_transient(detail)
                    else CodexInvocationError
                )
                category = (
                    "transient service or transport failure"
                    if error_type is CodexTransientError
                    else "invocation failure"
                )
                raise error_type(
                    f"codex CLI failed ({proc.returncode}): {category}"
                )
            audited_answer = self._audit(proc.stdout)
            try:
                answer = self._read_final_message(out_path)
            except (OSError, UnicodeDecodeError, ValueError) as e:
                raise CodexProtocolError(
                    f"codex produced no readable final message ({type(e).__name__})"
                ) from e
            if not answer.strip():
                raise CodexProtocolError("codex produced an empty final message")
            if answer != audited_answer:
                raise CodexProtocolError(
                    "codex final-message file does not match the audited agent_message"
                )
            return answer
        finally:
            if not process_cleanup_failed:
                self.cleanup()

    @staticmethod
    def _read_final_message(path: Path) -> str:
        """Read the CLI result without following links or accepting unbounded output."""
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > _MAX_FINAL_MESSAGE_BYTES
            ):
                raise ValueError("final message is not a bounded standalone file")
            payload = bytearray()
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("final message changed while it was being read")
                payload.extend(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError("final message grew while it was being read")
            after = os.fstat(descriptor)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in stable):
                raise ValueError("final message changed while it was being read")
            return bytes(payload).decode("utf-8")
        finally:
            os.close(descriptor)

    def _finish_call(
        self,
        *,
        started: float,
        version: str,
        attempts: list[dict[str, Any]],
        status: str,
        error: Exception | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            "status": status,
            "cli_version": version,
            "model": self.model or "cli-default",
            "reasoning_effort": self.reasoning_effort,
            "sandbox_mode": self.sandbox_mode,
            "externally_sandboxed": self.externally_sandboxed,
            "retries": max(0, len(attempts) - 1),
            "attempt_count": len(attempts),
            "duration_s": round(time.monotonic() - started, 3),
            "event_summary": self.last_event_summary,
            "attempts": attempts,
        }
        if error is not None:
            metadata["error_type"] = type(error).__name__
            metadata["retryable"] = bool(getattr(error, "retryable", False))
            if isinstance(error, CodexProcessCleanupError):
                metadata["primary_error_type"] = error.primary_error_type
                metadata["cleanup_error_type"] = error.cleanup_error_type
        self.last_call = metadata
        if self.last_usage is None:
            # TracedLLM persists ``last_usage`` even on exceptions. Supplying the
            # audit fields with unknown token counts keeps failed calls observable
            # without pretending they were free.
            self.last_usage = {
                "input_tokens": None,
                "output_tokens": None,
                "cached_input_tokens": None,
                "cost_usd": None,
                "model": self.model or "cli-default",
            }
        self.last_usage.update(
            {
                "cli_version": version,
                "reasoning_effort": self.reasoning_effort,
                "retries": metadata["retries"],
                "duration_s": metadata["duration_s"],
                "event_summary": self.last_event_summary,
                "status": status,
            }
        )

    def _audit(self, event_stream: str) -> str:
        """Validate one exact Codex 0.141 turn and return its last agent message."""
        self.last_usage = None
        summary = self._new_event_summary()
        self.last_event_summary = summary
        self.last_tool_use = []
        validator = CodexJsonlValidator(
            max_tool_calls=1,
            max_line_bytes=_MAX_JSONL_LINE_BYTES,
            max_total_bytes=_MAX_JSONL_BYTES,
        )
        last_agent_message: str | None = None

        for line_number, raw_line in enumerate(
            event_stream.splitlines(keepends=True), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                summary["invalid_json_lines"] += 1
                raise CodexProtocolError(
                    f"codex JSONL line {line_number} is invalid JSON: {e.msg}"
                ) from e
            if not isinstance(event, dict):
                raise CodexProtocolError(
                    f"codex JSONL line {line_number} must be an object, got {type(event).__name__}"
                )
            is_error_path = self._find_is_error(event)
            if is_error_path is not None:
                raise CodexProtocolError(
                    f"codex event line {line_number} reported isError at {is_error_path}"
                )
            kind = event.get("type")
            if not isinstance(kind, str) or kind not in _EVENT_TYPES:
                raise CodexProtocolError(
                    f"unknown codex top-level event type at line {line_number}: {kind!r}"
                )
            summary["total_events"] += 1
            self._increment(summary["events"], kind)
            if kind in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item")
                if not isinstance(item, dict):
                    raise CodexProtocolError(f"{kind} must contain an object-valued item")
                item_type = item.get("type")
                if not isinstance(item_type, str) or not item_type:
                    raise CodexProtocolError(f"{kind} item has no valid type")
                self._increment(summary["items"], item_type)
                if item_type == "error":
                    raise self._error_event(item)
                if item_type not in _TEXT_ITEMS:
                    evidence = self._tool_evidence(item)
                    self.last_tool_use.append(evidence)
                    raise CodexToolUse(
                        f"codex used forbidden item {item_type!r}; refusing a result that may "
                        f"have reached outside the prompt. evidence={evidence}"
                    )
            elif kind in {"turn.failed", "error"}:
                raise self._error_event(event)
            try:
                validator.feed_line(raw_line)
            except StrictCodexEventError as exc:
                if "invalid JSON" in str(exc):
                    summary["invalid_json_lines"] += 1
                raise CodexProtocolError(str(exc)) from exc
            if kind == "item.completed":
                item = event["item"]
                if item.get("type") == "agent_message":
                    last_agent_message = item["text"]

        try:
            audit = validator.finish()
        except StrictCodexEventError as exc:
            raise CodexProtocolError(str(exc)) from exc
        if last_agent_message is None:
            # The strict validator checks this too; retain a local assertion so
            # the returned result cannot become optional if its audit model changes.
            raise CodexProtocolError(
                "codex turn completed without a completed agent_message"
            )
        self.last_usage = {
            "input_tokens": audit.input_tokens,
            "output_tokens": audit.output_tokens + audit.reasoning_output_tokens,
            "cached_input_tokens": audit.cached_input_tokens,
            "cost_usd": None,  # a ChatGPT subscription reports no per-call cost
            "model": self.model,
        }
        return last_agent_message

    @staticmethod
    def _new_event_summary() -> dict[str, Any]:
        return {
            "total_events": 0,
            "events": {},
            "items": {},
            "invalid_json_lines": 0,
        }

    @staticmethod
    def _increment(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    @classmethod
    def _find_is_error(cls, value: Any, path: str = "$") -> str | None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if key == "isError" and bool(child):
                    return child_path
                found = cls._find_is_error(child, child_path)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found = cls._find_is_error(child, f"{path}[{index}]")
                if found is not None:
                    return found
        return None

    @staticmethod
    def _tool_evidence(item: Mapping[str, Any]) -> str:
        item_type = str(item.get("type") or "unknown")
        details = []
        for key in ("command", "path", "file_path", "name", "tool", "server", "query"):
            if item.get(key) is not None:
                details.append(f"{key}={str(item[key])[:200]}")
        return f"{item_type}: {', '.join(details) or '(no details recorded)'}"

    @classmethod
    def _error_event(cls, event: Mapping[str, Any]) -> CodexCLIError:
        detail = str(
            event.get("message")
            or event.get("error")
            or event.get("detail")
            or event.get("status")
            or "no detail"
        )
        if cls._looks_transient(detail):
            return CodexTransientError("codex reported a transient failure")
        return CodexProtocolError("codex reported a non-transient failure")

    @staticmethod
    def _looks_transient(detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in _TRANSIENT_MARKERS)
