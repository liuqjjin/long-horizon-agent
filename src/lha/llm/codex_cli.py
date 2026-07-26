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
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    """Stop the leader and every descendant before temporary secrets disappear."""
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
        return

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


def _run_isolated_process(
    argv: list[str],
    *,
    input: str | None,
    capture_output: bool,
    text: bool,
    timeout: float,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one process in its own group and reap its complete process tree."""
    if not capture_output or not text:
        raise ValueError("isolated Codex processes require captured text output")
    common: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": dict(env),
    }
    if os.name == "posix":
        proc = subprocess.Popen(argv, start_new_session=True, **common)
    else:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(argv, creationflags=creationflags, **common)
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except BaseException:
        _terminate_process_group(proc)
        raise
    else:
        # communicate() waits for the leader, but a detached worker can still
        # hold the process group open. Reap that worker before callers clean the
        # temporary CODEX_HOME.
        _terminate_process_group(proc)
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


class CodexToolUse(CodexProtocolError):
    """The model reached outside the supplied prompt, invalidating the result."""


class CodexCLIClient(LLMClient):
    name = "codex_cli"

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
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be >= 0")
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
                f"codex is not logged in ({source} is missing); run `codex login` first"
            )
        home = Path(tempfile.mkdtemp(prefix="lha_codex_home_"))
        try:
            home.chmod(0o700)
            shutil.copy2(source, home / "auth.json")
            (home / "auth.json").chmod(0o600)
        except Exception:
            shutil.rmtree(home, ignore_errors=True)
            raise
        self._home = home
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
        """Remove all attempt-local state, including the credential copy."""
        for path in (self._home, self._workspace):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)
        self._home = self._workspace = None

    def _cli_version(self) -> str:
        if self._version is not None:
            return self._version
        try:
            with tempfile.TemporaryDirectory(prefix="lha_codex_version_") as scratch:
                root = Path(scratch)
                res = _run_isolated_process(
                    [self.cli_path, "--version"],
                    input=None,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=_minimal_subprocess_env(codex_home=root, temp_dir=root),
                )
        except (OSError, subprocess.TimeoutExpired):
            version = "unknown"
        else:
            version = (
                (res.stdout.strip() or res.stderr.strip()) if res.returncode == 0 else "unknown"
            )
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
        self.cleanup()
        self.last_usage = None
        self.last_tool_use = []
        self.last_call = None
        self.last_event_summary = self._new_event_summary()
        if self.no_tools:
            system = f"{_NO_TOOLS_PREAMBLE}\n\n{system}"
        call_started = time.monotonic()
        version = self._cli_version()
        attempts: list[dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            attempt_started = time.monotonic()
            try:
                answer = self._complete_once(system, prompt)
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "failed",
                        "duration_s": round(time.monotonic() - attempt_started, 3),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:400],
                        "event_summary": self.last_event_summary,
                    }
                )
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
        out_dir: Path | None = None
        try:
            out_dir = Path(tempfile.mkdtemp(prefix="lha_codex_out_"))
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
            except subprocess.TimeoutExpired as e:
                raise CodexTransientError(f"codex CLI timed out after {self.timeout}s") from e
            except OSError as e:
                raise CodexInvocationError(f"could not execute codex CLI: {e}") from e
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout)[-400:]
                error_type = (
                    CodexTransientError
                    if proc.returncode == 124 or self._looks_transient(detail)
                    else CodexInvocationError
                )
                raise error_type(f"codex CLI failed ({proc.returncode}): {detail}")
            self._audit(proc.stdout)
            try:
                answer = out_path.read_text()
            except OSError as e:
                raise CodexProtocolError(f"codex produced no final message: {e}") from e
            if not answer.strip():
                raise CodexProtocolError("codex produced an empty final message")
            return answer
        finally:
            if out_dir is not None:
                shutil.rmtree(out_dir, ignore_errors=True)
            self.cleanup()

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
            metadata["error"] = str(error)[:400]
            metadata["retryable"] = bool(getattr(error, "retryable", False))
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

    def _audit(self, event_stream: str) -> None:
        """Validate the complete JSONL protocol and reject every external action."""
        self.last_usage = None
        summary = self._new_event_summary()
        self.last_event_summary = summary
        self.last_tool_use = []
        thread_started = False
        turn_started = False
        turn_completed = False
        saw_agent_message = False
        parsed_usage: dict[str, Any] | None = None

        for line_number, line in enumerate(event_stream.splitlines(), start=1):
            line = line.strip()
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
            if turn_completed:
                raise CodexProtocolError(f"codex emitted {kind!r} after turn.completed")

            if kind == "thread.started":
                if thread_started or turn_started or summary["total_events"] != 1:
                    raise CodexProtocolError(
                        "thread.started must be the first and only thread start"
                    )
                thread_started = True
            elif kind == "turn.started":
                if not thread_started or turn_started:
                    raise CodexProtocolError("turn.started must follow one thread.started")
                turn_started = True
            elif kind in {"item.started", "item.updated", "item.completed"}:
                if not turn_started:
                    raise CodexProtocolError(f"{kind} arrived before turn.started")
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
                if kind == "item.completed" and item_type == "agent_message":
                    saw_agent_message = True
            elif kind == "turn.completed":
                if not turn_started:
                    raise CodexProtocolError("turn.completed arrived before turn.started")
                usage = event.get("usage") or {}
                if not isinstance(usage, dict):
                    raise CodexProtocolError("turn.completed usage must be an object")
                parsed_usage = {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": (usage.get("output_tokens") or 0)
                    + (usage.get("reasoning_output_tokens") or 0),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "cost_usd": None,  # a ChatGPT subscription reports no per-call cost
                    "model": self.model,
                }
                turn_completed = True
            elif kind in {"turn.failed", "error"}:
                raise self._error_event(event)

        if not thread_started:
            raise CodexProtocolError("codex JSONL stream is empty or missing thread.started")
        if not turn_started:
            raise CodexProtocolError("codex JSONL stream is missing turn.started")
        if not turn_completed:
            raise CodexProtocolError("codex JSONL stream is missing turn.completed")
        if not saw_agent_message:
            raise CodexProtocolError("codex turn completed without a completed agent_message")
        self.last_usage = parsed_usage

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
            return CodexTransientError(f"codex reported a transient failure: {detail[:400]}")
        return CodexProtocolError(f"codex reported a non-transient failure: {detail[:400]}")

    @staticmethod
    def _looks_transient(detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in _TRANSIENT_MARKERS)
