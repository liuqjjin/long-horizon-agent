"""Experimental prompt-only backend for the locally authenticated Claude CLI.

Claude Code does not publish a versioned, machine-checkable event contract that
this project can pin without installing and testing a real CLI release. This
backend is therefore available for ordinary local runs, but it is not accepted
as formal ablation or release evidence.

Each call runs in an empty workspace with a temporary HOME, a small environment
allow-list, no persisted session, no configured MCP servers, and an empty tool
set. The stream-json transcript is still audited: malformed or unknown events,
incomplete results, and any tool-use block fail the call.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import LLMClient

_PASSTHROUGH_ENV = (
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
_BARE_AUTH_ENV = "ANTHROPIC_API_KEY"
_MAX_SYSTEM_BYTES = 64 * 1024
_MAX_PROMPT_BYTES = 16 * 1024 * 1024
_MAX_STDOUT_BYTES = 16 * 1024 * 1024
_MAX_STDERR_BYTES = 2 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 2 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_PROCESS_TERM_GRACE_S = 0.25
_PROCESS_KILL_GRACE_S = 2.0
_TEXT_BLOCK_TYPES = frozenset({"text", "thinking", "redacted_thinking"})


class ClaudeCLIError(RuntimeError):
    """Base class for a failed experimental Claude CLI call."""


class ClaudeInvocationError(ClaudeCLIError):
    """The executable, authentication, or model invocation failed."""


class ClaudeTimeoutError(ClaudeCLIError):
    """The CLI exceeded its per-call wall-clock timeout."""


class ClaudeOutputLimitError(ClaudeCLIError):
    """The CLI exceeded a bounded stdout or stderr capture."""


class ClaudeProtocolError(ClaudeCLIError):
    """The CLI returned malformed, unknown, or incomplete stream-json."""


class ClaudeToolUse(ClaudeProtocolError):
    """The model attempted to use a tool in a prompt-only completion."""


class ClaudeCleanupError(ClaudeCLIError):
    """Attempt-local state could not be removed."""


class ClaudeProcessCleanupError(ClaudeCleanupError):
    """The CLI process group could not be confirmed absent."""

    def __init__(self, message: str, *, process: subprocess.Popen[Any]):
        super().__init__(message)
        self.process = process


class _DuplicateJSONKey(ValueError):
    """Internal marker converted to a public protocol error."""


def _minimal_subprocess_env(*, home: Path, temp_dir: Path) -> dict[str, str]:
    """Return the complete child environment, excluding unrelated credentials."""
    inherited_path = os.environ.get("PATH") or os.defpath
    safe_path = os.pathsep.join(
        item
        for item in inherited_path.split(os.pathsep)
        if item and Path(item).is_absolute()
    )
    env = {"PATH": safe_path or os.defpath}
    for name in _PASSTHROUGH_ENV:
        value = os.environ.get(name)
        if value:
            env[name] = value
    temporary = str(temp_dir)
    env.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / "xdg-config"),
            "XDG_CACHE_HOME": str(home / "xdg-cache"),
            "XDG_STATE_HOME": str(home / "xdg-state"),
            "TMPDIR": temporary,
            "TMP": temporary,
            "TEMP": temporary,
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_BUG_COMMAND": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
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
        return True
    return True


def _terminate_process_group(proc: subprocess.Popen[Any]) -> bool:
    """Stop the process leader and descendants before temporary state is removed."""
    if os.name == "posix":
        pgid = proc.pid
        if _process_group_exists(pgid):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + _PROCESS_TERM_GRACE_S
        while _process_group_exists(pgid) and time.monotonic() < deadline:
            proc.poll()
            time.sleep(0.01)
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
    """Never let a cleanup-helper failure lose the process handle."""
    try:
        return _terminate_process_group(proc), None
    except BaseException as error:
        return False, error


def _join_started_threads(
    threads: tuple[threading.Thread, ...],
) -> BaseException | None:
    deadline = time.monotonic() + _PROCESS_KILL_GRACE_S
    first_error: BaseException | None = None
    for thread in threads:
        if thread.ident is None:
            continue
        try:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as error:
            first_error = first_error or error
    try:
        if any(thread.ident is not None and thread.is_alive() for thread in threads):
            first_error = first_error or RuntimeError("pipe thread did not stop")
    except BaseException as error:
        first_error = first_error or error
    return first_error


def _run_isolated_process(
    argv: list[str],
    *,
    input_text: str,
    timeout: float,
    cwd: Path,
    env: Mapping[str, str],
    max_stdout_bytes: int = _MAX_STDOUT_BYTES,
    max_stderr_bytes: int = _MAX_STDERR_BYTES,
) -> subprocess.CompletedProcess[str]:
    """Capture a subprocess without allowing unbounded output or descendants."""
    common: dict[str, Any] = {
        "cwd": str(cwd),
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": dict(env),
    }
    try:
        if os.name == "posix":
            proc: subprocess.Popen[Any] = subprocess.Popen(
                argv, start_new_session=True, **common
            )
        else:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            proc = subprocess.Popen(argv, creationflags=creationflags, **common)
    except (OSError, ValueError) as error:
        raise ClaudeInvocationError("could not start the Claude CLI") from error

    threads: tuple[threading.Thread, ...] = ()
    timed_out = False
    try:
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        overflow = threading.Event()
        io_errors: list[BaseException] = []
        io_errors_lock = threading.Lock()

        def read_stream(
            stream,
            destination: bytearray,
            limit: int,
        ) -> None:
            try:
                while True:
                    chunk = os.read(stream.fileno(), _READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    remaining = max(0, limit - len(destination))
                    if remaining:
                        destination.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        overflow.set()
            except BaseException as error:
                with io_errors_lock:
                    io_errors.append(error)

        def write_input() -> None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(input_text.encode("utf-8"))
                proc.stdin.close()
            except BrokenPipeError:
                pass
            except BaseException as error:
                with io_errors_lock:
                    io_errors.append(error)

        if proc.stdout is None or proc.stderr is None:
            raise ClaudeInvocationError("Claude CLI pipes were not created")
        readers = (
            threading.Thread(
                target=read_stream,
                args=(proc.stdout, stdout_buffer, max_stdout_bytes),
                daemon=True,
            ),
            threading.Thread(
                target=read_stream,
                args=(proc.stderr, stderr_buffer, max_stderr_bytes),
                daemon=True,
            ),
        )
        writer = threading.Thread(target=write_input, daemon=True)
        threads = (*readers, writer)
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if overflow.is_set():
                break
            if io_errors:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.01)
    except BaseException as primary_error:
        cleaned, cleanup_error = _attempt_process_group_cleanup(proc)
        join_error = _join_started_threads(threads)
        if not cleaned:
            raise ClaudeProcessCleanupError(
                "Claude CLI process group could not be confirmed stopped",
                process=proc,
            ) from (cleanup_error or primary_error)
        if join_error is not None:
            raise ClaudeInvocationError(
                "Claude CLI pipes did not close after process cleanup"
            ) from join_error
        raise

    cleaned, cleanup_error = _attempt_process_group_cleanup(proc)
    join_error = _join_started_threads(threads)
    if not cleaned:
        raise ClaudeProcessCleanupError(
            "Claude CLI process group could not be confirmed stopped",
            process=proc,
        ) from cleanup_error
    if join_error is not None:
        raise ClaudeInvocationError(
            "Claude CLI pipes did not close after process cleanup"
        ) from join_error
    if proc.returncode is None:
        raise ClaudeInvocationError("Claude CLI did not terminate after process cleanup")
    if timed_out:
        raise ClaudeTimeoutError(f"Claude CLI timed out after {timeout:g}s")
    if overflow.is_set():
        raise ClaudeOutputLimitError("Claude CLI output exceeded the capture limit")
    if io_errors:
        raise ClaudeInvocationError("Claude CLI pipe handling failed") from io_errors[0]
    try:
        stdout = bytes(stdout_buffer).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ClaudeProtocolError("Claude CLI stdout is not valid UTF-8") from error
    stderr = bytes(stderr_buffer).decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ClaudeProtocolError(f"Claude result has invalid {field}")
    return value


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(key)
        value[key] = item
    return value


def _usage_counts(value: Any, source: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ClaudeProtocolError(f"Claude {source} usage is malformed")
    return (
        _non_negative_int(value.get("input_tokens"), f"{source} input_tokens"),
        _non_negative_int(value.get("output_tokens"), f"{source} output_tokens"),
    )


class ClaudeCLIClient(LLMClient):
    """Prompt-only local convenience backend; never formal benchmark evidence."""

    name = "claude_cli"
    experimental = True
    formal_evidence_supported = False

    def __init__(
        self,
        cli_path: str = "claude",
        timeout: float = 180.0,
        model: str | None = None,
        no_tools: bool = False,
    ):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or not math.isfinite(timeout)
        ):
            raise ValueError("timeout must be a positive finite number")
        if not cli_path or "\x00" in cli_path:
            raise ValueError("cli_path must be a non-empty executable name")
        if model is not None and "\x00" in model:
            raise ValueError("model must not contain NUL bytes")
        self.cli_path = cli_path
        self.timeout = timeout
        self.model = model
        self.no_tools = no_tools
        self.last_usage: dict[str, Any] | None = None
        self.last_call: dict[str, Any] | None = None
        self.last_tool_use: list[str] = []
        self.last_event_summary: dict[str, Any] = self._new_event_summary()
        self.last_cleanup_failures: tuple[str, ...] = ()
        self._attempt_root: Path | None = None
        self._pending_process: subprocess.Popen[Any] | None = None
        self._version: str | None = None

    @staticmethod
    def _new_event_summary() -> dict[str, Any]:
        return {
            "total_events": 0,
            "events": {},
            "content_blocks": {},
            "invalid_json_lines": 0,
        }

    @property
    def pending_cleanup_paths(self) -> tuple[Path, ...]:
        return (self._attempt_root,) if self._attempt_root is not None else ()

    def _prepare_attempt(self) -> tuple[Path, Path, Path, dict[str, str]]:
        if self._attempt_root is not None:
            self.cleanup()
        root = Path(tempfile.mkdtemp(prefix="lha_claude_"))
        self._attempt_root = root
        try:
            root.chmod(0o700)
            home = root / "home"
            workspace = root / "workspace"
            temporary = root / "tmp"
            for path in (home, workspace, temporary):
                path.mkdir(mode=0o700)
            env = _minimal_subprocess_env(home=home, temp_dir=temporary)
            return home, workspace, temporary, env
        except BaseException as setup_error:
            try:
                self.cleanup()
            except ClaudeCleanupError as cleanup_error:
                raise cleanup_error from setup_error
            raise

    def cleanup(self) -> None:
        root = self._attempt_root
        pending_process = self._pending_process
        if pending_process is not None:
            try:
                leader_returncode = pending_process.poll()
            except BaseException as error:
                self.last_cleanup_failures = (
                    "temporary Claude state: process group recheck failed",
                )
                raise ClaudeProcessCleanupError(
                    "Claude CLI process group could not be rechecked; "
                    "temporary state remains retained",
                    process=pending_process,
                ) from error
            if os.name == "posix" and leader_returncode is not None:
                try:
                    cleaned = not _process_group_exists(pending_process.pid)
                except BaseException as error:
                    self.last_cleanup_failures = (
                        "temporary Claude state: process group recheck failed",
                    )
                    raise ClaudeProcessCleanupError(
                        "Claude CLI process group could not be rechecked; "
                        "temporary state remains retained",
                        process=pending_process,
                    ) from error
            else:
                cleaned, cleanup_error = _attempt_process_group_cleanup(pending_process)
                if cleanup_error is not None:
                    self.last_cleanup_failures = (
                        "temporary Claude state: process group recheck failed",
                    )
                    raise ClaudeProcessCleanupError(
                        "Claude CLI process group could not be rechecked; "
                        "temporary state remains retained",
                        process=pending_process,
                    ) from cleanup_error
            if not cleaned:
                self.last_cleanup_failures = (
                    "temporary Claude state: process group still present",
                )
                raise ClaudeProcessCleanupError(
                    "Claude CLI process group is still present; "
                    "temporary state remains retained",
                    process=pending_process,
                )
            self._pending_process = None
        if root is None:
            self.last_cleanup_failures = ()
            return
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            self._attempt_root = None
            self.last_cleanup_failures = ()
            return
        except Exception as error:
            if not os.path.lexists(root):
                self._attempt_root = None
                self.last_cleanup_failures = ()
                return
            errno = getattr(error, "errno", None)
            detail = type(error).__name__
            if isinstance(errno, int):
                detail = f"{detail} errno={errno}"
            self.last_cleanup_failures = (f"temporary Claude state: {detail}",)
            raise ClaudeCleanupError(
                "could not remove temporary Claude state; "
                "the path is retained for a cleanup retry"
            ) from error
        self._attempt_root = None
        self.last_cleanup_failures = ()

    def _cli_version(
        self,
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> str:
        result = _run_isolated_process(
            [self.cli_path, "--version"],
            input_text="",
            timeout=min(self.timeout, 30.0),
            cwd=cwd,
            env=env,
            max_stdout_bytes=4096,
            max_stderr_bytes=4096,
        )
        value = result.stdout.strip()
        if result.returncode != 0 or not value or len(value.encode("utf-8")) > 256:
            raise ClaudeInvocationError("could not identify the Claude CLI version")
        if "\n" in value or "\r" in value:
            raise ClaudeProtocolError("Claude CLI version output is malformed")
        return value

    def _command(self, system: str, *, explicit_auth: bool) -> list[str]:
        command = [
            self.cli_path,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--strict-mcp-config",
            "--mcp-config",
            "{}",
            "--tools",
            "",
            "--permission-mode",
            "dontAsk",
            "--max-turns",
            "1",
            "--system-prompt",
            system,
            "--bare" if explicit_auth else "--safe-mode",
        ]
        if self.model:
            command += ["--model", self.model]
        return command

    def complete(self, system: str, prompt: str) -> str:
        self.last_usage = None
        self.last_call = None
        self.last_tool_use = []
        self.last_event_summary = self._new_event_summary()
        self._version = None
        started = time.monotonic()
        active_error: BaseException | None = None
        try:
            if "\x00" in system:
                raise ClaudeInvocationError("Claude system prompt contains a NUL byte")
            try:
                system_size = len(system.encode("utf-8"))
                prompt_size = len(prompt.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ClaudeInvocationError("Claude input is not valid UTF-8") from error
            if system_size > _MAX_SYSTEM_BYTES:
                raise ClaudeInvocationError("Claude system prompt exceeds the input limit")
            if prompt_size > _MAX_PROMPT_BYTES:
                raise ClaudeInvocationError("Claude prompt exceeds the input limit")
            _home, workspace, _temporary, env = self._prepare_attempt()
            explicit_auth = _BARE_AUTH_ENV in env
            self._version = self._cli_version(cwd=workspace, env=env)
            result = _run_isolated_process(
                self._command(system, explicit_auth=explicit_auth),
                input_text=prompt,
                timeout=self.timeout,
                cwd=workspace,
                env=env,
            )
            if result.returncode != 0:
                raise ClaudeInvocationError(
                    f"Claude CLI invocation failed with return code {result.returncode}"
                )
            answer = self._audit(result.stdout)
            duration = time.monotonic() - started
            self.last_call = {
                "status": "succeeded",
                "experimental": True,
                "formal_evidence_supported": False,
                "cli_version": self._version,
                "model": self.last_usage.get("model") if self.last_usage else self.model,
                "isolation_mode": "bare" if explicit_auth else "safe-mode-keychain",
                "event_summary": self.last_event_summary,
                "duration_s": duration,
            }
            return answer
        except BaseException as error:
            active_error = error
            if isinstance(error, ClaudeProcessCleanupError):
                self._pending_process = error.process
            self.last_call = {
                "status": "failed",
                "experimental": True,
                "formal_evidence_supported": False,
                "cli_version": self._version,
                "model": self.model,
                "event_summary": self.last_event_summary,
                "duration_s": time.monotonic() - started,
                "error_type": type(error).__name__,
                "primary_error_type": type(error).__name__,
            }
            raise
        finally:
            if isinstance(active_error, ClaudeProcessCleanupError):
                self.last_cleanup_failures = (
                    "temporary Claude state: process group still present",
                )
                self.last_call = {
                    **(self.last_call or {}),
                    "status": "cleanup_failed",
                    "cleanup_error_type": type(active_error).__name__,
                }
            else:
                try:
                    self.cleanup()
                except ClaudeCleanupError as cleanup_error:
                    self.last_call = {
                        **(self.last_call or {}),
                        "status": "cleanup_failed",
                        "error_type": (
                            type(active_error).__name__
                            if active_error is not None
                            else type(cleanup_error).__name__
                        ),
                        "cleanup_error_type": type(cleanup_error).__name__,
                    }
                    if self.last_usage is not None:
                        self.last_usage = {
                            **self.last_usage,
                            "status": "cleanup_failed",
                        }
                    if active_error is not None:
                        raise cleanup_error from active_error
                    raise

    def _audit(self, stream: str) -> str:
        """Validate one complete Claude stream-json transcript."""
        summary = self._new_event_summary()
        self.last_event_summary = summary
        if len(stream.encode("utf-8")) > _MAX_STDOUT_BYTES:
            raise ClaudeOutputLimitError("Claude CLI stdout exceeds the audit limit")

        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(stream.splitlines(), start=1):
            if not line:
                raise ClaudeProtocolError(
                    f"Claude stream contains a blank JSONL line at {line_number}"
                )
            if len(line.encode("utf-8")) > _MAX_JSONL_LINE_BYTES:
                raise ClaudeProtocolError(
                    f"Claude JSONL line {line_number} exceeds the protocol limit"
                )
            try:
                event = json.loads(
                    line,
                    object_pairs_hook=_json_object_without_duplicates,
                )
            except _DuplicateJSONKey as error:
                raise ClaudeProtocolError(
                    f"Claude JSONL line {line_number} contains a duplicate key"
                ) from error
            except json.JSONDecodeError as error:
                summary["invalid_json_lines"] += 1
                raise ClaudeProtocolError(
                    f"Claude stream contains invalid JSON at line {line_number}"
                ) from error
            if not isinstance(event, dict):
                raise ClaudeProtocolError(
                    f"Claude JSONL line {line_number} is not an object"
                )
            events.append(event)

        if not events:
            raise ClaudeProtocolError("Claude stream is empty")
        if events[0].get("type") != "system" or events[0].get("subtype") != "init":
            raise ClaudeProtocolError("Claude stream does not begin with system/init")
        if events[-1].get("type") != "result":
            raise ClaudeProtocolError("Claude stream does not end with result")

        session_id: str | None = None
        assistant_text: str | None = None
        assistant_model: str | None = None
        assistant_usage: dict[str, Any] | None = None
        result_event: dict[str, Any] | None = None
        seen_init = False
        for event in events:
            event_type = event.get("type")
            if not isinstance(event_type, str):
                raise ClaudeProtocolError("Claude event is missing a string type")
            summary["total_events"] += 1
            event_counts = summary["events"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            event_session = event.get("session_id")
            if not isinstance(event_session, str) or not event_session:
                raise ClaudeProtocolError("Claude event is missing session_id")
            if session_id is None:
                session_id = event_session
            elif event_session != session_id:
                raise ClaudeProtocolError("Claude session_id changes within one call")

            if event_type == "system":
                if seen_init or event.get("subtype") != "init":
                    raise ClaudeProtocolError("Claude stream contains an unknown system event")
                seen_init = True
                tools = event.get("tools")
                if not isinstance(tools, list) or tools:
                    raise ClaudeProtocolError(
                        "Claude prompt-only process did not expose an empty tool set"
                    )
                continue

            if event_type == "user":
                self.last_tool_use.append("tool_result")
                raise ClaudeToolUse("Claude emitted a user/tool-result event")

            if event_type == "assistant":
                if assistant_text is not None:
                    raise ClaudeProtocolError("Claude stream contains multiple assistant turns")
                message = event.get("message")
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    raise ClaudeProtocolError("Claude assistant event has an invalid message")
                content = message.get("content")
                if not isinstance(content, list):
                    raise ClaudeProtocolError("Claude assistant content is not a list")
                text_parts: list[str] = []
                for block in content:
                    if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                        raise ClaudeProtocolError("Claude content block is malformed")
                    block_type = block["type"]
                    block_counts = summary["content_blocks"]
                    block_counts[block_type] = block_counts.get(block_type, 0) + 1
                    if block_type not in _TEXT_BLOCK_TYPES:
                        self.last_tool_use.append(block_type)
                        if block_type.endswith("_use") or block_type == "tool_use":
                            raise ClaudeToolUse(
                                f"Claude attempted forbidden tool use: {block_type}"
                            )
                        raise ClaudeProtocolError(
                            f"Claude emitted unknown content block: {block_type}"
                        )
                    if block_type == "text":
                        text = block.get("text")
                        if not isinstance(text, str):
                            raise ClaudeProtocolError("Claude text block is malformed")
                        text_parts.append(text)
                assistant_text = "".join(text_parts)
                assistant_model = message.get("model")
                if not isinstance(assistant_model, str) or not assistant_model:
                    raise ClaudeProtocolError("Claude assistant model is missing")
                if message.get("stop_reason") != "end_turn":
                    raise ClaudeProtocolError("Claude assistant turn did not end normally")
                raw_usage = message.get("usage")
                if raw_usage is not None and not isinstance(raw_usage, dict):
                    raise ClaudeProtocolError("Claude assistant usage is malformed")
                assistant_usage = raw_usage
                continue

            if event_type == "result":
                if result_event is not None:
                    raise ClaudeProtocolError("Claude stream contains multiple result events")
                result_event = event
                continue

            raise ClaudeProtocolError(f"Claude stream contains unknown event type: {event_type}")

        if assistant_text is None or result_event is None:
            raise ClaudeProtocolError("Claude stream is missing an assistant or result event")
        if not assistant_text.strip():
            raise ClaudeProtocolError(
                "Claude assistant turn did not contain a non-empty text response"
            )
        if (
            result_event.get("subtype") != "success"
            or result_event.get("is_error") is not False
        ):
            raise ClaudeProtocolError("Claude result does not attest success")
        result_text = result_event.get("result")
        if not isinstance(result_text, str) or result_text != assistant_text:
            raise ClaudeProtocolError(
                "Claude result text does not match the audited assistant message"
            )
        num_turns = result_event.get("num_turns")
        if type(num_turns) is not int or num_turns != 1:
            raise ClaudeProtocolError("Claude result did not complete in exactly one turn")

        result_usage = result_event.get("usage")
        if result_usage is None and assistant_usage is None:
            raise ClaudeProtocolError("Claude result is missing usage")
        assistant_counts = (
            _usage_counts(assistant_usage, "assistant")
            if assistant_usage is not None
            else None
        )
        result_counts = (
            _usage_counts(result_usage, "result")
            if result_usage is not None
            else None
        )
        if (
            assistant_counts is not None
            and result_counts is not None
            and assistant_counts != result_counts
        ):
            raise ClaudeProtocolError(
                "Claude assistant and result usage do not match"
            )
        input_tokens, output_tokens = result_counts or assistant_counts or (0, 0)
        raw_cost = result_event.get("total_cost_usd")
        if raw_cost is not None and (
            isinstance(raw_cost, bool)
            or not isinstance(raw_cost, (int, float))
            or not math.isfinite(float(raw_cost))
            or float(raw_cost) < 0
        ):
            raise ClaudeProtocolError("Claude result has invalid total_cost_usd")
        self.last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": float(raw_cost) if raw_cost is not None else None,
            "model": assistant_model,
            "status": "succeeded",
            "cli_version": self._version,
            "experimental": True,
            "event_summary": summary,
        }
        return result_text
