"""The codex backend's leak audit, tested without touching the network.

The reason this file matters: `codex` has no `--disallowed-tools` flag, and its
most restrictive sandbox (`-s read-only`) still permits reading the whole
filesystem — verified by hand, the model will read a withheld test file if asked.
So leak-freedom for the ablation rests entirely on the audit below refusing any
answer that was produced with a tool. If that check silently stops firing, the
experiment silently stops being leak-free.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import lha.llm.codex_cli as codex_backend
from lha.config import Config
from lha.llm import get_llm
from lha.llm.codex_cli import (
    CodexCLIClient,
    CodexInvocationError,
    CodexProtocolError,
    CodexToolUse,
    CodexTransientError,
)
from lha.llm.trace import TracedLLM


def _events(*items: dict, usage: dict | None = None) -> str:
    """A codex `--json` event stream carrying the given completed items."""
    lines = [json.dumps({"type": "thread.started"}), json.dumps({"type": "turn.started"})]
    lines += [json.dumps({"type": "item.completed", "item": item}) for item in items]
    lines.append(json.dumps({"type": "turn.completed", "usage": usage or {}}))
    return "\n".join(lines) + "\n"


# --- the audit ---------------------------------------------------------------
def test_plain_answer_passes_the_audit():
    client = CodexCLIClient(no_tools=True)
    client._audit(_events({"type": "agent_message", "text": "ok"}))
    assert client.last_tool_use == []


def test_reasoning_items_are_not_tool_use():
    client = CodexCLIClient(no_tools=True)
    client._audit(
        _events({"type": "reasoning", "text": "thinking"}, {"type": "agent_message", "text": "ok"})
    )
    assert client.last_tool_use == []


def test_command_execution_is_refused_and_the_command_is_reported():
    client = CodexCLIClient(no_tools=True)
    stream = _events(
        {"type": "command_execution", "command": "find / -name test_lru.py", "exit_code": 0},
        {"type": "agent_message", "text": "here you go"},
    )
    with pytest.raises(CodexToolUse) as excinfo:
        client._audit(stream)
    # the evidence has to reach the operator, not just a boolean
    assert "command_execution" in str(excinfo.value)
    assert "test_lru.py" in str(excinfo.value)
    assert "test_lru.py" in client.last_tool_use[0]


def test_an_unknown_tool_type_is_also_refused():
    """Fail closed on tools that do not exist yet.

    A deny-list has to name what it forbids, so a renamed or newly added tool
    defeats it. This is an allow-list: anything that is not the model talking is
    treated as reaching outside the prompt.
    """
    client = CodexCLIClient(no_tools=True)
    with pytest.raises(CodexToolUse):
        client._audit(_events({"type": "some_future_tool_2027", "detail": "?"}))


@pytest.mark.parametrize(
    ("item_type", "detail"),
    [
        ("command_execution", {"command": "ls"}),
        ("file_read", {"path": "/tmp/withheld/tests/test_oracle.py"}),
        ("mcp_tool_call", {"server": "filesystem", "tool": "read_file"}),
    ],
)
def test_external_actions_are_always_refused_even_outside_prompt_only_mode(item_type, detail):
    """Read-only is not leak-free: every external action invalidates the answer."""
    client = CodexCLIClient(no_tools=False)
    with pytest.raises(CodexToolUse, match=item_type):
        client._audit(_events({"type": item_type, **detail}))
    assert client.last_tool_use


def test_usage_is_parsed_with_reasoning_tokens_counted_as_output():
    client = CodexCLIClient(no_tools=True, model="gpt-5.4-mini")
    client._audit(
        _events(
            {"type": "agent_message", "text": "ok"},
            usage={
                "input_tokens": 11346,
                "cached_input_tokens": 2432,
                "output_tokens": 20,
                "reasoning_output_tokens": 380,
            },
        )
    )
    assert client.last_usage is not None
    assert client.last_usage["input_tokens"] == 11346
    assert client.last_usage["output_tokens"] == 400  # 20 visible + 380 reasoning
    assert client.last_usage["model"] == "gpt-5.4-mini"
    assert client.last_event_summary["events"]["turn.completed"] == 1
    assert client.last_event_summary["items"]["agent_message"] == 1
    # A ChatGPT subscription bills no per-call amount; reporting 0.0 would be a lie.
    assert client.last_usage["cost_usd"] is None


def test_invalid_json_fails_closed():
    client = CodexCLIClient(no_tools=True)
    with pytest.raises(CodexProtocolError, match="invalid JSON"):
        client._audit("some warning line\n" + _events({"type": "agent_message", "text": "ok"}))
    assert client.last_event_summary["invalid_json_lines"] == 1


@pytest.mark.parametrize("event_type", ["future.magic", "item.failed"])
def test_unknown_top_level_event_fails_closed(event_type):
    client = CodexCLIClient()
    stream = "\n".join(
        [
            json.dumps({"type": "thread.started"}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": event_type}),
        ]
    )
    with pytest.raises(CodexProtocolError, match="unknown codex top-level event"):
        client._audit(stream)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "turn.started", "isError": True},
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "not safe",
                "result": {"content": [{"isError": True}]},
            },
        },
    ],
)
def test_is_error_anywhere_in_an_event_fails_closed(event):
    client = CodexCLIClient()
    lines = [json.dumps({"type": "thread.started"})]
    if event["type"] != "turn.started":
        lines.append(json.dumps({"type": "turn.started"}))
    lines.append(json.dumps(event))
    with pytest.raises(CodexProtocolError, match="isError"):
        client._audit("\n".join(lines))


@pytest.mark.parametrize(
    "stream",
    [
        "\n".join([json.dumps({"type": "thread.started"}), json.dumps({"type": "turn.started"})]),
        "\n".join(
            [
                json.dumps({"type": "thread.started"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "partial"},
                    }
                ),
            ]
        ),
    ],
)
def test_missing_turn_completed_fails_closed(stream):
    client = CodexCLIClient()
    with pytest.raises(CodexProtocolError, match="missing turn.completed"):
        client._audit(stream)


def test_started_item_without_a_completed_agent_message_fails_closed():
    client = CodexCLIClient()
    stream = "\n".join(
        [
            json.dumps({"type": "thread.started"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {"id": "item_1", "type": "agent_message", "text": ""},
                }
            ),
            json.dumps({"type": "turn.completed", "usage": {}}),
        ]
    )
    with pytest.raises(CodexProtocolError, match="without a completed agent_message"):
        client._audit(stream)


# --- invocation ---------------------------------------------------------------
def test_argv_isolates_the_run(tmp_path):
    client = CodexCLIClient(model="gpt-5.4-mini", reasoning_effort="low", no_tools=True)
    argv = client._argv(tmp_path / "last.txt")
    joined = " ".join(argv)
    assert "--ephemeral" in argv  # 200+ cells must not leave 200+ session files
    assert "--ignore-user-config" in argv  # no user config survives the temporary home
    assert "--skip-git-repo-check" in argv  # cells run in temp dirs
    assert "--ignore-rules" in argv  # no project execpolicy files
    assert "--sandbox read-only" in joined  # the model may not write
    assert argv[argv.index("-m") + 1] == "gpt-5.4-mini"
    assert "model_reasoning_effort='low'" in joined
    assert argv[-1] == "-"  # prompt arrives on stdin
    # the working root is an empty scratch dir, not the repo under test
    workspace = argv[argv.index("-C") + 1]
    assert not any(client._empty_workspace().iterdir())
    assert workspace == str(client._empty_workspace())
    client.cleanup()


def test_danger_full_access_requires_an_external_sandbox():
    with pytest.raises(ValueError, match="externally_sandboxed"):
        CodexCLIClient(sandbox_mode="danger-full-access")
    client = CodexCLIClient(
        sandbox_mode="danger-full-access",
        externally_sandboxed=True,
    )
    assert "--sandbox danger-full-access" in " ".join(client._argv(Path("/tmp/out")))
    client.cleanup()


def test_prompt_only_mode_tells_the_model_there_is_no_filesystem(monkeypatch):
    """The audit is the guarantee; the preamble is what keeps the refusal rate low."""
    client = CodexCLIClient(no_tools=True)
    seen: dict[str, str] = {}

    class _Proc:
        returncode = 0
        stdout = _events({"type": "agent_message", "text": "ok"})
        stderr = ""

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        seen["input"] = input
        out = argv[argv.index("-o") + 1]
        Path(out).write_text("answer")
        return _Proc()

    monkeypatch.setattr(CodexCLIClient, "_cli_version", lambda self: "codex-cli test")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    assert client.complete("SYSTEM", "PROMPT") == "answer"
    assert "no repository and no filesystem" in seen["input"]
    assert "SYSTEM" in seen["input"] and "PROMPT" in seen["input"]

    plain = CodexCLIClient(no_tools=False)
    monkeypatch.setattr(plain, "_clean_home", lambda: plain._empty_workspace())
    plain.complete("SYSTEM", "PROMPT")
    assert "no repository and no filesystem" not in seen["input"]  # agentic runs unaffected
    assert client._home is None and client._workspace is None
    assert plain._home is None and plain._workspace is None


def test_codex_process_receives_only_the_environment_allowlist(tmp_path, monkeypatch):
    source_home = tmp_path / "real_codex_home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token": "only-auth-source"}')
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    secrets = {
        "LHA_SENTINEL_SECRET": "do-not-inherit",
        "OPENAI_API_KEY": "openai-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secret/gcp.json",
        "SSH_AUTH_SOCK": "/secret/agent.sock",
        "NODE_OPTIONS": "--require=/secret/hook.js",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8080")
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(tmp_path / "company-ca.pem"))
    captured: dict[str, str] = {}

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        captured.update(env)
        copied_auth = Path(env["CODEX_HOME"]) / "auth.json"
        assert copied_auth.read_text() == '{"token": "only-auth-source"}'
        assert stat.S_IMODE(copied_auth.stat().st_mode) == 0o600
        Path(argv[argv.index("-o") + 1]).write_text("answer")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=_events({"type": "agent_message", "text": "answer"}),
            stderr="",
        )

    client = CodexCLIClient(max_retries=0)
    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    assert client.complete("SYSTEM", "PROMPT") == "answer"

    assert not secrets.keys() & captured.keys()
    assert captured["HOME"] == captured["CODEX_HOME"]
    assert captured["CODEX_HOME"] != str(source_home)
    assert captured["TMPDIR"] == captured["TMP"] == captured["TEMP"]
    assert captured["HTTPS_PROXY"] == "http://proxy.example.test:8080"
    assert captured["NODE_EXTRA_CA_CERTS"] == str(tmp_path / "company-ca.pem")
    assert "PATH" in captured


def test_protocol_errors_are_not_retried(monkeypatch):
    client = CodexCLIClient(max_retries=3, retry_backoff_s=0)
    calls = 0

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        nonlocal calls
        calls += 1
        Path(argv[argv.index("-o") + 1]).write_text("answer")
        return subprocess.CompletedProcess(argv, 0, stdout="not json\n", stderr="")

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.test")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    with pytest.raises(CodexProtocolError):
        client.complete("SYSTEM", "PROMPT")

    assert calls == 1
    assert client.last_call is not None
    assert client.last_call["retries"] == 0
    assert client.last_call["retryable"] is False
    assert client.last_call["error_type"] == "CodexProtocolError"
    assert client.last_usage is not None
    assert client.last_usage["status"] == "failed"
    assert client.last_usage["cli_version"] == "codex-cli 0.test"
    assert client.last_usage["event_summary"]["invalid_json_lines"] == 1


def test_failed_call_audit_metadata_is_persisted_by_the_standard_tracer(tmp_path, monkeypatch):
    client = CodexCLIClient(max_retries=0)

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        Path(argv[argv.index("-o") + 1]).write_text("answer")
        return subprocess.CompletedProcess(argv, 0, stdout="not json\n", stderr="")

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.test")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    traced = TracedLLM(client).bind(tmp_path)
    with pytest.raises(CodexProtocolError):
        traced.complete("SYSTEM", "PROMPT")

    record = json.loads((tmp_path / "llm_trace.jsonl").read_text().strip())
    assert record["usage"]["status"] == "failed"
    assert record["usage"]["cli_version"] == "codex-cli 0.test"
    assert record["usage"]["event_summary"]["invalid_json_lines"] == 1


def test_only_transient_failures_retry_and_record_audit_metadata(monkeypatch):
    client = CodexCLIClient(
        model="gpt-5.4-mini",
        reasoning_effort="high",
        max_retries=2,
        retry_backoff_s=0,
    )
    calls = 0

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="503 service temporarily unavailable"
            )
        Path(argv[argv.index("-o") + 1]).write_text("answer")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=_events(
                {"type": "agent_message", "text": "answer"},
                usage={"input_tokens": 10, "output_tokens": 3},
            ),
            stderr="",
        )

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.test")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    assert client.complete("SYSTEM", "PROMPT") == "answer"

    assert calls == 2
    assert client.last_call is not None
    assert client.last_call["cli_version"] == "codex-cli 0.test"
    assert client.last_call["model"] == "gpt-5.4-mini"
    assert client.last_call["reasoning_effort"] == "high"
    assert client.last_call["retries"] == 1
    assert client.last_call["attempt_count"] == 2
    assert client.last_call["duration_s"] >= 0
    assert client.last_call["event_summary"]["events"]["turn.completed"] == 1
    assert [attempt["status"] for attempt in client.last_call["attempts"]] == [
        "failed",
        "succeeded",
    ]
    assert client.last_usage is not None
    assert client.last_usage["cli_version"] == "codex-cli 0.test"
    assert client.last_usage["retries"] == 1


def test_non_transient_cli_failure_is_not_retried(monkeypatch):
    client = CodexCLIClient(max_retries=3, retry_backoff_s=0)
    calls = 0

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="invalid model name")

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.test")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    with pytest.raises(CodexInvocationError, match="invalid model"):
        client.complete("SYSTEM", "PROMPT")
    assert calls == 1


@pytest.mark.parametrize(
    ("outcome", "expected_error"),
    [
        ("success", None),
        ("protocol_failure", CodexProtocolError),
        ("unexpected_exception", RuntimeError),
        ("keyboard_interrupt", KeyboardInterrupt),
    ],
)
def test_all_temporary_state_is_cleaned_on_every_exit(
    outcome, expected_error, tmp_path, monkeypatch
):
    source_home = tmp_path / "real_codex_home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token": "secret"}')
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    client = CodexCLIClient(max_retries=0)
    paths_seen: list[Path] = []

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        codex_home = Path(env["CODEX_HOME"])
        workspace = Path(argv[argv.index("-C") + 1])
        output_dir = Path(argv[argv.index("-o") + 1]).parent
        paths_seen.extend([codex_home, workspace, output_dir])
        assert codex_home.exists() and (codex_home / "auth.json").exists()
        assert workspace.exists() and output_dir.exists()
        if outcome == "unexpected_exception":
            raise RuntimeError("subprocess hook exploded")
        if outcome == "keyboard_interrupt":
            raise KeyboardInterrupt
        Path(argv[argv.index("-o") + 1]).write_text("answer")
        stdout = (
            _events({"type": "agent_message", "text": "answer"})
            if outcome == "success"
            else "broken JSONL"
        )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.test")
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    if expected_error is None:
        assert client.complete("SYSTEM", "PROMPT") == "answer"
    else:
        with pytest.raises(expected_error):
            client.complete("SYSTEM", "PROMPT")

    assert paths_seen
    assert all(not path.exists() for path in paths_seen)
    assert (source_home / "auth.json").exists()
    assert client._home is None and client._workspace is None


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_isolated_runner_reaps_the_process_group_on_exception(error_type, tmp_path, monkeypatch):
    class ExplodingProcess:
        pid = 424242
        returncode = None

        def communicate(self, *, input, timeout):
            raise error_type("communicate interrupted")

    process = ExplodingProcess()
    popen_options: dict = {}
    reaped: list[object] = []

    def fake_popen(argv, **kwargs):
        popen_options.update(kwargs)
        return process

    monkeypatch.setattr(codex_backend.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        codex_backend,
        "_terminate_process_group",
        lambda proc: reaped.append(proc),
    )

    with pytest.raises(error_type):
        codex_backend._run_isolated_process(
            ["codex", "exec"],
            input="prompt",
            capture_output=True,
            text=True,
            timeout=1,
            env={"PATH": os.defpath},
        )

    assert reaped == [process]
    if os.name == "posix":
        assert popen_options["start_new_session"] is True


def test_timeout_kills_a_descendant_before_removing_credentials(tmp_path, monkeypatch):
    """A child that ignores SIGTERM must not outlive the copied auth.json."""
    if os.name != "posix":
        pytest.skip("process-group lifecycle assertion requires POSIX")

    source_home = tmp_path / "real_codex_home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token": "temporary-copy"}')
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setenv("LHA_SENTINEL_SECRET", "must-not-reach-descendants")

    record_path = tmp_path / "process.json"
    ready_path = tmp_path / "child-ready"
    child_code = (
        "import pathlib,signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        f"pathlib.Path({str(ready_path)!r}).write_text('ready');"
        "time.sleep(60)"
    )
    fake_codex = tmp_path / "codex-hangs"
    fake_codex.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import json, os, subprocess, time",
                f"child = subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}])",
                f"record = {str(record_path)!r}",
                f"ready = {str(ready_path)!r}",
                "while not os.path.exists(ready):",
                "    time.sleep(0.01)",
                "with open(record, 'w') as fh:",
                "    json.dump({",
                "        'child_pid': child.pid,",
                "        'codex_home': os.environ['CODEX_HOME'],",
                "        'temp_dir': os.environ['TMPDIR'],",
                "        'sentinel': os.environ.get('LHA_SENTINEL_SECRET'),",
                "    }, fh)",
                "time.sleep(60)",
            ]
        )
        + "\n"
    )
    fake_codex.chmod(0o755)

    client = CodexCLIClient(
        cli_path=str(fake_codex),
        timeout=2.0,
        max_retries=0,
    )
    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    with pytest.raises(CodexTransientError, match="timed out"):
        client.complete("SYSTEM", "PROMPT")

    record = json.loads(record_path.read_text())
    assert record["sentinel"] is None
    assert not Path(record["codex_home"]).exists()
    assert not Path(record["temp_dir"]).exists()
    assert (source_home / "auth.json").exists()

    child_pid = int(record["child_pid"])
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"descendant process {child_pid} survived the Codex timeout")


def test_missing_credentials_fail_with_a_usable_message(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nope"))
    client = CodexCLIClient()
    with pytest.raises(RuntimeError, match="codex login"):
        client._clean_home()
    assert client._home is None


def test_factory_builds_the_backend_from_config():
    client = get_llm(
        Config(
            llm_backend="codex_cli",
            codex_model="gpt-5.4-mini",
            codex_max_retries=4,
            codex_retry_backoff_s=0.25,
        )
    )
    assert client.name == "codex_cli"
    assert getattr(client, "model") == "gpt-5.4-mini"
    assert getattr(client, "no_tools") is False  # only the ablation demands prompt-only
    assert getattr(client, "sandbox_mode") == "read-only"
    assert getattr(client, "max_retries") == 4
    assert getattr(client, "retry_backoff_s") == 0.25


def test_factory_allows_danger_full_access_only_for_an_external_container():
    client = get_llm(
        Config(
            llm_backend="codex_cli",
            codex_sandbox="danger-full-access",
            codex_external_sandbox=True,
        )
    )
    assert getattr(client, "sandbox_mode") == "danger-full-access"
    with pytest.raises(ValueError, match="externally_sandboxed"):
        get_llm(Config(llm_backend="codex_cli", codex_sandbox="danger-full-access"))
