"""Fail-closed checks for the experimental Claude CLI backend."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

import lha.llm.claude_cli as claude_backend
from lha.llm.claude_cli import (
    ClaudeCleanupError,
    ClaudeCLIClient,
    ClaudeInvocationError,
    ClaudeOutputLimitError,
    ClaudeProcessCleanupError,
    ClaudeProtocolError,
    ClaudeTimeoutError,
    ClaudeToolUse,
)
from lha.llm.trace import TracedLLM


def _stream(
    answer: str = "answer",
    *,
    content: list[dict] | None = None,
    result_updates: dict | None = None,
) -> str:
    session_id = "session-test"
    usage = {"input_tokens": 11, "output_tokens": 3}
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "session_id": session_id,
        "num_turns": 1,
        "result": answer,
        "usage": usage,
        "total_cost_usd": 0.02,
    }
    if result_updates:
        result.update(result_updates)
    return (
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": session_id,
                        "tools": [],
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "session_id": session_id,
                        "message": {
                            "role": "assistant",
                            "model": "claude-test-snapshot",
                            "content": content or [{"type": "text", "text": answer}],
                            "stop_reason": "end_turn",
                            "usage": usage,
                        },
                    }
                ),
                json.dumps(result),
            ]
        )
        + "\n"
    )


def _insert_before_result(event: dict) -> str:
    lines = _stream().splitlines()
    return "\n".join([*lines[:-1], json.dumps(event), lines[-1]]) + "\n"


def _update_assistant(**updates) -> str:
    lines = _stream().splitlines()
    event = json.loads(lines[1])
    event["message"].update(updates)
    lines[1] = json.dumps(event)
    return "\n".join(lines) + "\n"


def test_valid_stream_is_audited_and_records_usage() -> None:
    client = ClaudeCLIClient(model="claude-test")
    client._version = "2.1.219 (Claude Code)"

    assert client._audit(_stream()) == "answer"
    assert client.last_tool_use == []
    assert client.last_usage == {
        "input_tokens": 11,
        "output_tokens": 3,
        "cost_usd": 0.02,
        "model": "claude-test-snapshot",
        "status": "succeeded",
        "cli_version": "2.1.219 (Claude Code)",
        "experimental": True,
        "event_summary": client.last_event_summary,
    }


@pytest.mark.parametrize(
    ("stream", "message"),
    (
        ("plain text\n", "invalid JSON"),
        (
            _insert_before_result(
                {"type": "future_event", "session_id": "session-test"}
            ),
            "unknown event",
        ),
        ("\n".join(_stream().splitlines()[:-1]) + "\n", "does not end with result"),
        (_stream(result_updates={"is_error": True}), "does not attest success"),
        (_stream(result_updates={"result": "different"}), "does not match"),
        (_stream(result_updates={"num_turns": 2}), "exactly one turn"),
        (_stream(result_updates={"num_turns": True}), "exactly one turn"),
        (_stream(result_updates={"num_turns": 1.0}), "exactly one turn"),
        (
            _stream(
                answer="",
                content=[{"type": "thinking", "thinking": "private reasoning"}],
            ),
            "non-empty text response",
        ),
        (
            _update_assistant(
                usage={"input_tokens": 999, "output_tokens": 999}
            ),
            "usage do not match",
        ),
    ),
)
def test_malformed_or_incomplete_stream_fails_closed(
    stream: str,
    message: str,
) -> None:
    with pytest.raises(ClaudeProtocolError, match=message):
        ClaudeCLIClient()._audit(stream)


def test_duplicate_json_keys_fail_closed() -> None:
    lines = _stream().splitlines()
    lines[0] = (
        '{"type":"system","type":"system","subtype":"init",'
        '"session_id":"session-test","tools":[]}'
    )

    with pytest.raises(ClaudeProtocolError, match="duplicate key"):
        ClaudeCLIClient()._audit("\n".join(lines) + "\n")


@pytest.mark.parametrize("no_tools", [False, True])
def test_any_tool_use_fails_even_if_prompt_only_flag_was_not_requested(
    no_tools: bool,
) -> None:
    client = ClaudeCLIClient(no_tools=no_tools)
    with pytest.raises(ClaudeToolUse, match="tool use"):
        client._audit(
            _stream(
                content=[
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "/hidden/test.py"},
                    }
                ]
            )
        )
    assert client.last_tool_use == ["tool_use"]


def test_unknown_content_block_fails_closed() -> None:
    with pytest.raises(ClaudeProtocolError, match="unknown content block"):
        ClaudeCLIClient()._audit(
            _stream(content=[{"type": "future_action", "payload": {}}])
        )


def test_complete_uses_empty_workspace_and_minimal_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LHA_SENTINEL_SECRET", "must-not-reach-child")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-key")
    client = ClaudeCLIClient(cli_path="claude-test", model="claude-test")
    calls: list[tuple[list[str], Path, dict[str, str]]] = []
    attempt_paths: list[Path] = []

    def fake_run(
        argv,
        *,
        input_text,
        timeout,
        cwd,
        env,
        max_stdout_bytes=claude_backend._MAX_STDOUT_BYTES,
        max_stderr_bytes=claude_backend._MAX_STDERR_BYTES,
    ):
        del input_text, timeout, max_stdout_bytes, max_stderr_bytes
        calls.append((list(argv), Path(cwd), dict(env)))
        attempt_paths.extend([Path(cwd), Path(env["HOME"]), Path(env["TMPDIR"])])
        assert list(Path(cwd).iterdir()) == []
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "2.1.219 (Claude Code)\n", "")
        return subprocess.CompletedProcess(argv, 0, _stream(), "")

    monkeypatch.setattr(claude_backend, "_run_isolated_process", fake_run)

    assert client.complete("SYSTEM", "PROMPT") == "answer"
    assert len(calls) == 2
    command, cwd, env = calls[-1]
    assert cwd != Path.cwd()
    assert env["ANTHROPIC_API_KEY"] == "test-only-key"
    assert "LHA_SENTINEL_SECRET" not in env
    assert "--bare" in command
    assert command[command.index("--tools") + 1] == ""
    assert "--no-session-persistence" in command
    assert "--strict-mcp-config" in command
    assert all(not path.exists() for path in attempt_paths)
    assert client.pending_cleanup_paths == ()
    assert client.last_call is not None
    assert client.last_call["experimental"] is True
    assert client.last_call["formal_evidence_supported"] is False


def test_minimal_environment_drops_relative_path_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((".", "", "/safe/tools", "relative/bin", "/usr/bin")),
    )

    env = claude_backend._minimal_subprocess_env(
        home=tmp_path / "home",
        temp_dir=tmp_path / "tmp",
    )

    assert env["PATH"] == os.pathsep.join(("/safe/tools", "/usr/bin"))


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    (
        ("protocol", ClaudeProtocolError),
        ("exception", RuntimeError),
        ("interrupt", KeyboardInterrupt),
    ),
)
def test_temporary_state_is_removed_on_every_failed_exit(
    outcome: str,
    expected_error: type[BaseException],
    monkeypatch,
) -> None:
    client = ClaudeCLIClient()
    roots: list[Path] = []

    def fake_version(*, cwd, env):
        del env
        roots.append(Path(cwd).parent)
        return "2.1.219 (Claude Code)"

    def fake_run(*_args, **_kwargs):
        if outcome == "exception":
            raise RuntimeError("subprocess adapter exploded")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return subprocess.CompletedProcess([], 0, "not JSON\n", "")

    monkeypatch.setattr(client, "_cli_version", fake_version)
    monkeypatch.setattr(claude_backend, "_run_isolated_process", fake_run)

    with pytest.raises(expected_error):
        client.complete("SYSTEM", "PROMPT")

    assert roots and all(not root.exists() for root in roots)
    assert client.pending_cleanup_paths == ()
    assert client.last_usage is None


def test_cleanup_failure_is_fail_closed_and_retryable(monkeypatch) -> None:
    client = ClaudeCLIClient()
    root: Path | None = None
    real_rmtree = claude_backend.shutil.rmtree

    def fake_version(*, cwd, env):
        nonlocal root
        del env
        root = Path(cwd).parent
        return "2.1.219 (Claude Code)"

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, _stream(), "")

    def fail_cleanup(path, *args, **kwargs):
        if root is not None and Path(path) == root:
            raise PermissionError(13, "permission denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(client, "_cli_version", fake_version)
    monkeypatch.setattr(claude_backend, "_run_isolated_process", fake_run)
    monkeypatch.setattr(claude_backend.shutil, "rmtree", fail_cleanup)

    with pytest.raises(ClaudeCleanupError, match="retained") as error:
        client.complete("SYSTEM", "PROMPT")

    assert root is not None and root.exists()
    assert root in client.pending_cleanup_paths
    assert str(root) not in str(error.value)
    assert client.last_call is not None
    assert client.last_call["error_type"] == "ClaudeCleanupError"
    assert client.last_call["cleanup_error_type"] == "ClaudeCleanupError"
    assert client.last_call["status"] == "cleanup_failed"
    assert client.last_usage is not None
    assert client.last_usage["status"] == "cleanup_failed"

    monkeypatch.setattr(claude_backend.shutil, "rmtree", real_rmtree)
    client.cleanup()
    assert not root.exists()
    assert client.pending_cleanup_paths == ()


@pytest.mark.parametrize(
    ("outcome", "primary_error"),
    (
        ("protocol", "ClaudeProtocolError"),
        ("interrupt", "KeyboardInterrupt"),
    ),
)
def test_primary_and_cleanup_failures_are_both_recorded(
    outcome: str,
    primary_error: str,
    monkeypatch,
) -> None:
    client = ClaudeCLIClient()
    root: Path | None = None
    real_rmtree = claude_backend.shutil.rmtree

    def fake_version(*, cwd, env):
        nonlocal root
        del env
        root = Path(cwd).parent
        return "2.1.219 (Claude Code)"

    def fake_run(*_args, **_kwargs):
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return subprocess.CompletedProcess([], 0, "not JSON\n", "")

    def fail_cleanup(path, *args, **kwargs):
        if root is not None and Path(path) == root:
            raise PermissionError(13, "permission denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(client, "_cli_version", fake_version)
    monkeypatch.setattr(claude_backend, "_run_isolated_process", fake_run)
    monkeypatch.setattr(claude_backend.shutil, "rmtree", fail_cleanup)

    with pytest.raises(ClaudeCleanupError) as error:
        client.complete("SYSTEM", "PROMPT")

    assert isinstance(error.value.__cause__, (ClaudeProtocolError, KeyboardInterrupt))
    assert client.last_call is not None
    assert client.last_call["status"] == "cleanup_failed"
    assert client.last_call["error_type"] == primary_error
    assert client.last_call["primary_error_type"] == primary_error
    assert client.last_call["cleanup_error_type"] == "ClaudeCleanupError"
    assert root is not None and root.exists()

    monkeypatch.setattr(claude_backend.shutil, "rmtree", real_rmtree)
    client.cleanup()


def test_unconfirmed_process_group_retains_temporary_state(
    monkeypatch,
) -> None:
    class PendingProcess:
        pid = 424_242

        @staticmethod
        def poll() -> None:
            return None

    client = ClaudeCLIClient()
    root: Path | None = None
    pending_process = cast(subprocess.Popen[Any], PendingProcess())
    boundary_present = True
    process_checks = 0

    def fake_version(*, cwd, env):
        nonlocal root
        del env
        root = Path(cwd).parent
        return "2.1.219 (Claude Code)"

    def fail_process_cleanup(*_args, **_kwargs):
        raise ClaudeProcessCleanupError(
            "process group still present",
            process=pending_process,
        )

    def check_process_boundary(process):
        nonlocal process_checks
        assert process is pending_process
        process_checks += 1
        return not boundary_present

    monkeypatch.setattr(client, "_cli_version", fake_version)
    monkeypatch.setattr(
        claude_backend,
        "_run_isolated_process",
        fail_process_cleanup,
    )
    monkeypatch.setattr(
        claude_backend,
        "_terminate_process_group",
        check_process_boundary,
    )

    with pytest.raises(ClaudeProcessCleanupError):
        client.complete("SYSTEM", "PROMPT")

    assert root is not None and root.exists()
    assert root in client.pending_cleanup_paths
    assert client.last_call is not None
    assert client.last_call["status"] == "cleanup_failed"
    assert client.last_call["cleanup_error_type"] == "ClaudeProcessCleanupError"

    with pytest.raises(ClaudeProcessCleanupError, match="still present"):
        client.cleanup()
    assert root.exists()

    with pytest.raises(ClaudeProcessCleanupError):
        client.complete("SYSTEM", "PROMPT")
    assert process_checks == 2
    assert root.exists()

    boundary_present = False
    client.cleanup()
    assert not root.exists()
    assert client.pending_cleanup_paths == ()


def test_cleanup_retry_does_not_signal_after_process_leader_exited(
    tmp_path: Path,
    monkeypatch,
) -> None:
    if os.name != "posix":
        pytest.skip("PGID reuse protection requires POSIX process groups")

    class ExitedProcess:
        pid = 424_243

        @staticmethod
        def poll() -> int:
            return 1

    client = ClaudeCLIClient()
    root = tmp_path / "retained-claude-state"
    root.mkdir()
    pending_process = cast(subprocess.Popen[Any], ExitedProcess())
    client._attempt_root = root
    client._pending_process = pending_process
    group_present = True
    signals = 0

    def process_group_exists(pgid: int) -> bool:
        assert pgid == pending_process.pid
        return group_present

    def unexpected_signal(_process) -> bool:
        nonlocal signals
        signals += 1
        raise AssertionError("an exited leader must not be signalled during cleanup retry")

    monkeypatch.setattr(claude_backend, "_process_group_exists", process_group_exists)
    monkeypatch.setattr(claude_backend, "_terminate_process_group", unexpected_signal)

    with pytest.raises(ClaudeProcessCleanupError, match="still present"):
        client.cleanup()

    assert signals == 0
    assert root.exists()
    assert client._pending_process is pending_process

    group_present = False
    client.cleanup()

    assert signals == 0
    assert not root.exists()
    assert client.pending_cleanup_paths == ()


def test_input_rejection_clears_previous_usage_before_tracing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = ClaudeCLIClient()
    monkeypatch.setattr(
        client,
        "_cli_version",
        lambda **_kwargs: "2.1.219 (Claude Code)",
    )
    monkeypatch.setattr(
        claude_backend,
        "_run_isolated_process",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, _stream(), ""),
    )
    traced = TracedLLM(client).bind(tmp_path)

    assert traced.complete("SYSTEM", "PROMPT") == "answer"
    before = traced.totals
    before_tokens = (before.input_tokens, before.output_tokens, before.cost_usd)
    with pytest.raises(ClaudeInvocationError, match="NUL"):
        traced.complete("SYSTEM\x00", "PROMPT")

    assert client.last_usage is None
    assert client.last_call is not None
    assert client.last_call["status"] == "failed"
    assert (
        traced.totals.input_tokens,
        traced.totals.output_tokens,
        traced.totals.cost_usd,
    ) == before_tokens


def test_stderr_and_temporary_paths_do_not_enter_audit_metadata(
    monkeypatch,
) -> None:
    client = ClaudeCLIClient()
    protected = "secret-token-value"
    private_path = "/tmp/lha_claude_private/home"

    monkeypatch.setattr(
        client,
        "_cli_version",
        lambda **_kwargs: "2.1.219 (Claude Code)",
    )
    monkeypatch.setattr(
        claude_backend,
        "_run_isolated_process",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 2, "", f"authentication failed at {private_path}: {protected}"
        ),
    )

    with pytest.raises(ClaudeInvocationError, match="return code 2") as error:
        client.complete("SYSTEM", "PROMPT")

    durable = json.dumps(client.last_call)
    assert protected not in durable
    assert private_path not in durable
    assert protected not in str(error.value)
    assert private_path not in str(error.value)


def test_isolated_runner_bounds_stdout(tmp_path: Path) -> None:
    script = tmp_path / "noisy"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(b'x' * 100_000)\n"
        "sys.stdout.flush()\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    with pytest.raises(ClaudeOutputLimitError, match="capture limit"):
        claude_backend._run_isolated_process(
            [str(script)],
            input_text="prompt",
            timeout=5,
            cwd=tmp_path,
            env={"PATH": os.defpath},
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_limit_plus_one_then_hang_is_output_error_not_timeout(
    stream_name: str,
    tmp_path: Path,
) -> None:
    script = tmp_path / f"hang-after-{stream_name}"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "sys.stdin.buffer.read()\n"
        f"stream = sys.{stream_name}.buffer\n"
        "stream.write(b'x' * 1025)\n"
        "stream.flush()\n"
        "time.sleep(60)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    started = time.monotonic()

    with pytest.raises(ClaudeOutputLimitError, match="capture limit"):
        claude_backend._run_isolated_process(
            [str(script)],
            input_text="prompt",
            timeout=10,
            cwd=tmp_path,
            env={"PATH": os.defpath},
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )

    assert time.monotonic() - started < 5


@pytest.mark.parametrize("failed_start", [2, 3])
def test_thread_start_failure_reaps_process_group(
    failed_start: int,
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "wait-for-cleanup"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "sys.stdin.buffer.read()\n"
        "time.sleep(60)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    real_start = threading.Thread.start
    real_terminate = claude_backend._terminate_process_group
    starts = 0
    terminated: list[subprocess.Popen] = []

    def fail_selected_start(thread):
        nonlocal starts
        starts += 1
        if starts == failed_start:
            raise RuntimeError("thread resources exhausted")
        return real_start(thread)

    def record_termination(process):
        terminated.append(process)
        return real_terminate(process)

    monkeypatch.setattr(threading.Thread, "start", fail_selected_start)
    monkeypatch.setattr(
        claude_backend,
        "_terminate_process_group",
        record_termination,
    )

    with pytest.raises(RuntimeError, match="thread resources exhausted"):
        claude_backend._run_isolated_process(
            [str(script)],
            input_text="prompt",
            timeout=10,
            cwd=tmp_path,
            env={"PATH": os.defpath},
        )

    assert len(terminated) == 1
    assert terminated[0].poll() is not None
    if os.name == "posix":
        assert not claude_backend._process_group_exists(terminated[0].pid)


def test_post_popen_allocation_failure_reaps_process_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "wait-after-popen"
    script.write_text(
        f"#!{sys.executable}\n"
        "import time\n"
        "time.sleep(60)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    real_terminate = claude_backend._terminate_process_group
    terminated: list[subprocess.Popen] = []

    def fail_event_allocation():
        raise RuntimeError("event allocation failed")

    def record_termination(process):
        terminated.append(process)
        return real_terminate(process)

    monkeypatch.setattr(threading, "Event", fail_event_allocation)
    monkeypatch.setattr(
        claude_backend,
        "_terminate_process_group",
        record_termination,
    )

    with pytest.raises(RuntimeError, match="event allocation failed"):
        claude_backend._run_isolated_process(
            [str(script)],
            input_text="prompt",
            timeout=10,
            cwd=tmp_path,
            env={"PATH": os.defpath},
        )

    assert len(terminated) == 1
    assert terminated[0].poll() is not None
    if os.name == "posix":
        assert not claude_backend._process_group_exists(terminated[0].pid)


def test_termination_helper_failure_preserves_live_process_handle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "wait-for-manual-cleanup"
    script.write_text(
        f"#!{sys.executable}\n"
        "import time\n"
        "time.sleep(60)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    real_terminate = claude_backend._terminate_process_group

    def fail_termination(_process):
        raise PermissionError(1, "termination denied")

    monkeypatch.setattr(
        claude_backend,
        "_terminate_process_group",
        fail_termination,
    )

    with pytest.raises(ClaudeProcessCleanupError) as caught:
        claude_backend._run_isolated_process(
            [str(script)],
            input_text="prompt",
            timeout=0.1,
            cwd=tmp_path,
            env={"PATH": os.defpath},
        )

    process = caught.value.process
    assert isinstance(caught.value.__cause__, PermissionError)
    assert process.poll() is None

    monkeypatch.setattr(
        claude_backend,
        "_terminate_process_group",
        real_terminate,
    )
    assert real_terminate(process)
    assert process.poll() is not None
    if os.name == "posix":
        assert not claude_backend._process_group_exists(process.pid)


def test_timeout_kills_descendant_process_group(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("process-group lifecycle assertion requires POSIX")
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    script = tmp_path / "hanging"
    script.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import pathlib, subprocess, time",
                f"child = subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}])",
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))",
                "time.sleep(60)",
            ]
        )
        + "\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    with pytest.raises(ClaudeTimeoutError, match="timed out"):
        claude_backend._run_isolated_process(
            [str(script)],
            input_text="",
            timeout=2.0,
            cwd=tmp_path,
            env={"PATH": os.defpath},
        )

    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"descendant process {child_pid} survived Claude timeout")


def test_backend_is_explicitly_experimental() -> None:
    client = ClaudeCLIClient()
    assert client.experimental is True
    assert client.formal_evidence_supported is False
