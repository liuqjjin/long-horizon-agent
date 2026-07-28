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
from typing import Any, cast

import pytest

import lha.llm.codex_cli as codex_backend
from lha.config import Config
from lha.harness.errors import BudgetExceeded
from lha.harness.state import LLMUsageState
from lha.llm import get_llm
from lha.llm.codex_cli import (
    CodexCleanupError,
    CodexCLIClient,
    CodexInvocationError,
    CodexProcessCleanupError,
    CodexProtocolError,
    CodexToolUse,
    CodexTransientError,
)
from lha.llm.trace import TracedLLM, load_usage_checkpoint


def _events(*items: dict, usage: dict | None = None) -> str:
    """A codex `--json` event stream carrying the given completed items."""
    normalized_items = []
    for index, item in enumerate(items, start=1):
        normalized = dict(item)
        normalized.setdefault("id", f"item-{index}")
        normalized_items.append(normalized)
    complete_usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    if usage is not None:
        complete_usage.update(usage)
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
        json.dumps({"type": "turn.started"}),
    ]
    lines += [
        json.dumps({"type": "item.completed", "item": item})
        for item in normalized_items
    ]
    lines.append(json.dumps({"type": "turn.completed", "usage": complete_usage}))
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
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
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
    lines = [json.dumps({"type": "thread.started", "thread_id": "thread-1"})]
    if event["type"] != "turn.started":
        lines.append(json.dumps({"type": "turn.started"}))
    lines.append(json.dumps(event))
    with pytest.raises(CodexProtocolError, match="isError"):
        client._audit("\n".join(lines))


@pytest.mark.parametrize(
    "stream",
    [
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                json.dumps({"type": "turn.started"}),
            ]
        ),
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "message-1",
                            "type": "agent_message",
                            "text": "partial",
                        },
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
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "id": "item_1",
                        "type": "todo_list",
                        "items": [],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_output_tokens": 0,
                    },
                }
            ),
        ]
    )
    with pytest.raises(CodexProtocolError, match="unfinished items"):
        client._audit(stream)


@pytest.mark.parametrize(
    "stream",
    [
        "\n".join(
            [
                json.dumps({"type": "thread.started"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "final",
                            "type": "agent_message",
                            "text": "done",
                        },
                    }
                ),
            ]
        ),
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "missing id"},
                    }
                ),
            ]
        ),
    ],
)
def test_malformed_required_fields_fail_closed(stream):
    with pytest.raises(CodexProtocolError, match="does not match 0.141"):
        CodexCLIClient()._audit(stream)


def test_duplicate_item_id_fails_closed():
    stream = _events(
        {"id": "same", "type": "agent_message", "text": "first"},
        {"id": "same", "type": "agent_message", "text": "second"},
    )
    with pytest.raises(CodexProtocolError, match="reused completed-only item"):
        CodexCLIClient()._audit(stream)


@pytest.mark.parametrize(
    "usage",
    [
        {
            "input_tokens": -1,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        },
        {
            "input_tokens": "1",
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        },
        {"input_tokens": 1, "output_tokens": 0, "reasoning_output_tokens": 0},
    ],
)
def test_invalid_or_incomplete_usage_fails_closed(usage):
    stream = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "final",
                        "type": "agent_message",
                        "text": "done",
                    },
                }
            ),
            json.dumps({"type": "turn.completed", "usage": usage}),
        ]
    )
    with pytest.raises(CodexProtocolError, match="does not match 0.141"):
        CodexCLIClient()._audit(stream)


def test_unfinished_tool_is_rejected_before_result_use():
    stream = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "id": "tool-1",
                        "type": "command_execution",
                        "command": "cat withheld-test.py",
                        "aggregated_output": "",
                        "exit_code": None,
                        "status": "in_progress",
                    },
                }
            ),
        ]
    )
    with pytest.raises(CodexToolUse, match="command_execution"):
        CodexCLIClient()._audit(stream)


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


def test_unknown_cli_version_is_rejected_before_execution(monkeypatch):
    client = CodexCLIClient(max_retries=0)
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsupported CLI must not execute")

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.142.0")
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    with pytest.raises(CodexProtocolError, match="unsupported Codex CLI protocol"):
        client.complete("SYSTEM", "PROMPT")
    assert called is False
    assert client.last_call is not None
    assert client.last_call["attempt_count"] == 0


def test_output_file_must_match_audited_agent_message(monkeypatch):
    client = CodexCLIClient(max_retries=0)

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        Path(argv[argv.index("-o") + 1]).write_text("different")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=_events({"type": "agent_message", "text": "audited"}),
            stderr="",
        )

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    with pytest.raises(CodexProtocolError, match="does not match"):
        client.complete("SYSTEM", "PROMPT")


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
        stdout = _events({"type": "agent_message", "text": "answer"})
        stderr = ""

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        seen["input"] = input
        out = argv[argv.index("-o") + 1]
        Path(out).write_text("answer")
        return _Proc()

    monkeypatch.setattr(
        CodexCLIClient, "_cli_version", lambda self: "codex-cli 0.141.0"
    )
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

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
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
    assert client.last_usage["cli_version"] == "codex-cli 0.141.0"
    assert client.last_usage["event_summary"]["invalid_json_lines"] == 1


def test_failed_call_audit_metadata_is_persisted_by_the_standard_tracer(tmp_path, monkeypatch):
    client = CodexCLIClient(max_retries=0)

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        Path(argv[argv.index("-o") + 1]).write_text("answer")
        return subprocess.CompletedProcess(argv, 0, stdout="not json\n", stderr="")

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    traced = TracedLLM(client).bind(tmp_path)
    with pytest.raises(CodexProtocolError):
        traced.complete("SYSTEM", "PROMPT")

    record = json.loads((tmp_path / "llm_trace.jsonl").read_text().strip())
    assert record["usage"]["status"] == "failed"
    assert record["usage"]["cli_version"] == "codex-cli 0.141.0"
    assert record["usage"]["event_summary"]["invalid_json_lines"] == 1


def test_codex_stderr_and_temporary_paths_never_enter_durable_trace(
    tmp_path,
    monkeypatch,
):
    client = CodexCLIClient(max_retries=0)
    protected = "token-super-secret"
    credential_path = "/tmp/lha_codex_home_private/auth.json"

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        return subprocess.CompletedProcess(
            argv,
            2,
            stdout="",
            stderr=f"authentication failed for {credential_path}: {protected}",
        )

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    traced = TracedLLM(client).bind(tmp_path)

    with pytest.raises(CodexInvocationError, match="invocation failure"):
        traced.complete("SYSTEM", "PROMPT")

    durable = (tmp_path / "llm_trace.jsonl").read_text()
    assert protected not in durable
    assert credential_path not in durable
    assert client.last_call is not None
    assert "error" not in client.last_call
    assert all("error" not in attempt for attempt in client.last_call["attempts"])


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

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    assert client.complete("SYSTEM", "PROMPT") == "answer"

    assert calls == 2
    assert client.last_call is not None
    assert client.last_call["cli_version"] == "codex-cli 0.141.0"
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
    assert client.last_usage["cli_version"] == "codex-cli 0.141.0"
    assert client.last_usage["retries"] == 1


def test_tracer_counts_each_real_codex_retry_against_the_budget(tmp_path, monkeypatch):
    client = CodexCLIClient(max_retries=2, retry_backoff_s=0)
    process_calls = 0

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        nonlocal process_calls
        process_calls += 1
        if process_calls == 1:
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr="503 service temporarily unavailable",
            )
        Path(argv[argv.index("-o") + 1]).write_text("answer")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=_events(
                {"type": "agent_message", "text": "answer"},
                usage={"input_tokens": 5, "output_tokens": 2},
            ),
            stderr="",
        )

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    traced = TracedLLM(client, max_calls=2).bind(tmp_path)
    traced.restore_totals(LLMUsageState())

    assert traced.complete("SYSTEM", "PROMPT") == "answer"
    assert process_calls == 2
    assert traced.totals.calls == 2
    durable = load_usage_checkpoint(tmp_path)
    assert durable is not None and durable.calls == 2
    record = json.loads((tmp_path / "llm_trace.jsonl").read_text())
    assert record["attempt_count"] == 2
    assert record["call"]["attempt_count"] == 2


def test_codex_retry_cannot_exceed_or_reset_the_durable_attempt_budget(
    tmp_path, monkeypatch
):
    process_calls = 0

    def make_client() -> CodexCLIClient:
        client = CodexCLIClient(max_retries=2, retry_backoff_s=0)

        def fake_run(argv, *, input, capture_output, text, timeout, env):
            nonlocal process_calls
            process_calls += 1
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr="503 service temporarily unavailable",
            )

        monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
        monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
        monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
        return client

    first = TracedLLM(make_client(), max_calls=1).bind(tmp_path)
    first.restore_totals(LLMUsageState())
    with pytest.raises(BudgetExceeded, match="max_llm_calls=1"):
        first.complete("SYSTEM", "PROMPT")
    assert process_calls == 1
    assert first.totals.calls == 1
    durable = load_usage_checkpoint(tmp_path)
    assert durable is not None and durable.calls == 1

    resumed = TracedLLM(make_client(), max_calls=1).bind(tmp_path)
    resumed.restore_totals(LLMUsageState())
    assert resumed.totals.calls == 1
    with pytest.raises(BudgetExceeded, match="max_llm_calls=1"):
        resumed.complete("SYSTEM", "PROMPT")
    assert process_calls == 1


def test_non_transient_cli_failure_is_not_retried(monkeypatch):
    client = CodexCLIClient(max_retries=3, retry_backoff_s=0)
    calls = 0

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr="invalid model name")

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    monkeypatch.setattr(client, "_clean_home", lambda: client._empty_workspace())
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    with pytest.raises(CodexInvocationError, match="invocation failure"):
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

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
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


def test_cleanup_failure_is_fail_closed_and_retains_the_credential_home(
    tmp_path, monkeypatch
):
    source_home = tmp_path / "real_codex_home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token": "secret"}')
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    client = CodexCLIClient(max_retries=0)
    real_rmtree = codex_backend.shutil.rmtree
    failed_home: Path | None = None

    def fake_run(argv, *, input, capture_output, text, timeout, env):
        nonlocal failed_home
        failed_home = Path(env["CODEX_HOME"])
        Path(argv[argv.index("-o") + 1]).write_text("answer")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=_events({"type": "agent_message", "text": "answer"}),
            stderr="",
        )

    def fail_credential_cleanup(path, *args, **kwargs):
        if failed_home is not None and Path(path) == failed_home:
            raise PermissionError(13, "permission denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    monkeypatch.setattr("lha.llm.codex_cli._run_isolated_process", fake_run)
    monkeypatch.setattr(codex_backend.shutil, "rmtree", fail_credential_cleanup)

    with pytest.raises(CodexCleanupError, match="paths remain retained") as error:
        client.complete("SYSTEM", "PROMPT")

    assert failed_home is not None and failed_home.exists()
    assert (failed_home / "auth.json").exists()
    assert failed_home in client.pending_cleanup_paths
    assert client.last_cleanup_failures == (
        "temporary Codex home: PermissionError errno=13",
    )
    assert str(failed_home) not in str(error.value)
    assert client.last_call is not None
    assert client.last_call["status"] == "failed"
    assert client.last_call["error_type"] == "CodexCleanupError"
    assert "error" not in client.last_call
    assert source_home.exists()

    monkeypatch.setattr(codex_backend.shutil, "rmtree", real_rmtree)
    client.cleanup()
    assert not failed_home.exists()
    assert client.pending_cleanup_paths == ()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_isolated_runner_reaps_the_process_group_on_exception(error_type, tmp_path, monkeypatch):
    class ExplodingProcess:
        pid = 424242
        returncode = None

    process = ExplodingProcess()
    popen_options: dict = {}
    reaped: list[object] = []

    def fake_popen(argv, **kwargs):
        popen_options.update(kwargs)
        return process

    def explode(*_args, **_kwargs):
        raise error_type("bounded communication interrupted")

    monkeypatch.setattr(codex_backend.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(codex_backend, "_communicate_bounded", explode)
    monkeypatch.setattr(
        codex_backend,
        "_terminate_process_group",
        lambda proc: not reaped.append(proc),
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


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_codex_client_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        CodexCLIClient(timeout=timeout)


@pytest.mark.parametrize("max_retries", [True, -1, 1.5])
def test_codex_client_rejects_invalid_retry_count(max_retries):
    with pytest.raises(ValueError, match="max_retries"):
        CodexCLIClient(max_retries=max_retries)


@pytest.mark.parametrize("retry_backoff_s", [True, -1, float("nan"), float("inf")])
def test_codex_client_rejects_invalid_retry_backoff(retry_backoff_s):
    with pytest.raises(ValueError, match="retry_backoff_s"):
        CodexCLIClient(retry_backoff_s=retry_backoff_s)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"cli_path": ""}, "cli_path"),
        ({"cli_path": "co\x00dex"}, "cli_path"),
        ({"model": "gpt\x00bad"}, "model"),
    ],
)
def test_codex_client_rejects_malformed_process_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        CodexCLIClient(**arguments)


def test_isolated_runner_terminates_each_process_group_once(
    tmp_path,
    monkeypatch,
):
    real_terminate = codex_backend._terminate_process_group
    terminated: list[subprocess.Popen] = []

    def record_termination(process):
        terminated.append(process)
        return real_terminate(process)

    monkeypatch.setattr(
        codex_backend,
        "_terminate_process_group",
        record_termination,
    )

    result = codex_backend._run_isolated_process(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdin.buffer.read(); sys.stdout.write('ok')",
        ],
        input="prompt",
        capture_output=True,
        text=True,
        timeout=5,
        env={"PATH": os.defpath},
    )

    assert result.returncode == 0
    assert result.stdout == "ok"
    assert len(terminated) == 1
    assert terminated[0].poll() is not None


@pytest.mark.parametrize(
    "failure_site",
    ["event", "thread_constructor", "second_thread_start", "thread_join"],
)
def test_post_popen_resource_failure_reaps_process_group_once(
    failure_site,
    tmp_path,
    monkeypatch,
):
    script = tmp_path / "codex-waits"
    sleep_seconds = "0.05" if failure_site == "thread_join" else "60"
    script.write_text(
        f"#!{sys.executable}\n"
        "import time\n"
        f"time.sleep({sleep_seconds})\n"
    )
    script.chmod(0o755)
    real_terminate = codex_backend._terminate_process_group
    real_start = codex_backend.threading.Thread.start
    terminated: list[subprocess.Popen] = []
    starts = 0

    def record_termination(process):
        terminated.append(process)
        return real_terminate(process)

    def fail_event():
        raise RuntimeError("event allocation failed")

    def fail_thread_constructor(*_args, **_kwargs):
        raise RuntimeError("thread construction failed")

    def fail_second_start(thread):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("thread start failed")
        return real_start(thread)

    def fail_join(_thread, _timeout=None):
        raise RuntimeError("thread join failed")

    monkeypatch.setattr(
        codex_backend,
        "_terminate_process_group",
        record_termination,
    )
    expected = "event allocation failed"
    expected_error: type[BaseException] = RuntimeError
    if failure_site == "event":
        monkeypatch.setattr(codex_backend.threading, "Event", fail_event)
    elif failure_site == "thread_constructor":
        expected = "thread construction failed"
        monkeypatch.setattr(
            codex_backend.threading,
            "Thread",
            fail_thread_constructor,
        )
    elif failure_site == "second_thread_start":
        expected = "thread start failed"
        monkeypatch.setattr(
            codex_backend.threading.Thread,
            "start",
            fail_second_start,
        )
    elif failure_site == "thread_join":
        expected = "output pipes remained open after process-group cleanup"
        expected_error = codex_backend._ProcessOutputError
        monkeypatch.setattr(
            codex_backend.threading.Thread,
            "join",
            fail_join,
        )

    with pytest.raises(expected_error, match=expected):
        codex_backend._run_isolated_process(
            [str(script)],
            input="prompt",
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": os.defpath},
        )

    assert len(terminated) == 1
    assert terminated[0].poll() is not None
    assert terminated[0].stdin is not None and terminated[0].stdin.closed
    assert terminated[0].stdout is not None and terminated[0].stdout.closed
    assert terminated[0].stderr is not None and terminated[0].stderr.closed
    if os.name == "posix":
        assert not codex_backend._process_group_exists(terminated[0].pid)


def test_process_cleanup_failure_retains_credentials_and_error_provenance(
    tmp_path,
    monkeypatch,
):
    source_home = tmp_path / "real_codex_home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token": "temporary-copy"}')
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    script = tmp_path / "codex-hangs-during-cleanup"
    script.write_text(
        f"#!{sys.executable}\n"
        "import time\n"
        "time.sleep(60)\n"
    )
    script.chmod(0o755)
    client = CodexCLIClient(
        cli_path=str(script),
        timeout=0.1,
        max_retries=0,
    )
    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    real_terminate = codex_backend._terminate_process_group
    cleanup_attempts = 0

    def fail_termination(_process):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        raise PermissionError(1, "termination denied")

    monkeypatch.setattr(
        codex_backend,
        "_terminate_process_group",
        fail_termination,
    )

    with pytest.raises(CodexProcessCleanupError) as caught:
        client.complete("SYSTEM", "PROMPT")

    assert cleanup_attempts == 1
    assert caught.value.primary_error_type == "TimeoutExpired"
    assert caught.value.cleanup_error_type == "PermissionError"
    assert caught.value.process.poll() is None
    pending = client.pending_cleanup_paths
    assert len(pending) == 3
    assert all(path.exists() for path in pending)
    copied_home = next(path for path in pending if path.name.startswith("lha_codex_home_"))
    assert (copied_home / "auth.json").read_text() == '{"token": "temporary-copy"}'
    assert client.last_call is not None
    assert client.last_call["error_type"] == "CodexProcessCleanupError"
    assert client.last_call["primary_error_type"] == "TimeoutExpired"
    assert client.last_call["cleanup_error_type"] == "PermissionError"
    assert client.last_call["attempts"][0]["primary_error_type"] == "TimeoutExpired"
    assert client.last_call["attempts"][0]["cleanup_error_type"] == "PermissionError"

    monkeypatch.setattr(
        codex_backend,
        "_terminate_process_group",
        real_terminate,
    )
    client.cleanup()

    assert client.pending_cleanup_paths == ()
    assert all(not path.exists() for path in pending)
    assert (source_home / "auth.json").exists()


def test_cleanup_retry_does_not_signal_after_process_leader_exited(
    tmp_path,
    monkeypatch,
):
    if os.name != "posix":
        pytest.skip("PGID reuse protection requires POSIX process groups")

    class ExitedProcess:
        pid = 424_244

        @staticmethod
        def poll() -> int:
            return 1

    client = CodexCLIClient()
    copied_home = tmp_path / "retained-codex-home"
    workspace = tmp_path / "retained-codex-workspace"
    output = tmp_path / "retained-codex-output"
    for path in (copied_home, workspace, output):
        path.mkdir()
    (copied_home / "auth.json").write_text('{"token": "retained"}')
    process = cast(subprocess.Popen[Any], ExitedProcess())
    client._home = copied_home
    client._workspace = workspace
    client._output_dirs.add(output)
    client._pending_process = process
    group_present = True
    signals = 0

    def process_group_exists(pgid: int) -> bool:
        assert pgid == process.pid
        return group_present

    def unexpected_signal(_process) -> bool:
        nonlocal signals
        signals += 1
        raise AssertionError("an exited leader must not be signalled on cleanup retry")

    monkeypatch.setattr(codex_backend, "_process_group_exists", process_group_exists)
    monkeypatch.setattr(codex_backend, "_terminate_process_group", unexpected_signal)

    with pytest.raises(CodexProcessCleanupError, match="still present"):
        client.cleanup()

    assert signals == 0
    assert (copied_home / "auth.json").exists()
    assert all(path.exists() for path in (copied_home, workspace, output))

    group_present = False
    client.cleanup()

    assert signals == 0
    assert client.pending_cleanup_paths == ()
    assert all(not path.exists() for path in (copied_home, workspace, output))


def _flooding_codex_client(tmp_path, monkeypatch, body):
    source_home = tmp_path / "real_codex_home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"token": "temporary-copy"}')
    monkeypatch.setenv("CODEX_HOME", str(source_home))
    executable = tmp_path / "codex-output-fixture"
    executable.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import os, pathlib, signal, subprocess, sys, time",
                *body,
            ]
        )
        + "\n"
    )
    executable.chmod(0o755)
    client = CodexCLIClient(
        cli_path=str(executable),
        timeout=5.0,
        max_retries=0,
    )
    monkeypatch.setattr(client, "_cli_version", lambda: "codex-cli 0.141.0")
    return client, source_home


def test_stdout_total_limit_stops_a_live_codex_process(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_backend, "_MAX_JSONL_BYTES", 4096)
    monkeypatch.setattr(codex_backend, "_MAX_JSONL_LINE_BYTES", 1024)
    client, source_home = _flooding_codex_client(
        tmp_path,
        monkeypatch,
        [
            "while True:",
            "    os.write(1, b'{}\\n' * 1024)",
            "    time.sleep(0.01)",
        ],
    )

    started = time.monotonic()
    with pytest.raises(CodexProtocolError, match="stdout exceeded the 4096-byte limit"):
        client.complete("SYSTEM", "PROMPT")
    assert time.monotonic() - started < 3
    assert client.pending_cleanup_paths == ()
    assert (source_home / "auth.json").exists()
    assert client.last_call is not None
    assert client.last_call["status"] == "failed"
    assert client.last_call["error_type"] == "CodexProtocolError"
    assert client.last_call["attempt_count"] == 1
    assert client.last_call["retryable"] is False
    assert client.last_usage is not None
    assert client.last_usage["status"] == "failed"


def test_stdout_long_line_stops_before_the_total_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_backend, "_MAX_JSONL_BYTES", 8192)
    monkeypatch.setattr(codex_backend, "_MAX_JSONL_LINE_BYTES", 1024)
    client, _source_home = _flooding_codex_client(
        tmp_path,
        monkeypatch,
        [
            "os.write(1, b'x' * 2048)",
            "time.sleep(60)",
        ],
    )

    started = time.monotonic()
    with pytest.raises(
        CodexProtocolError,
        match="stdout JSONL line exceeded the 1024-byte limit",
    ):
        client.complete("SYSTEM", "PROMPT")
    assert time.monotonic() - started < 3
    assert client.pending_cleanup_paths == ()


def test_stderr_limit_stops_a_flood_before_process_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_backend, "_MAX_STDERR_BYTES", 1024)
    client, _source_home = _flooding_codex_client(
        tmp_path,
        monkeypatch,
        [
            "os.write(2, b'e' * 4096)",
            "time.sleep(60)",
        ],
    )

    started = time.monotonic()
    with pytest.raises(CodexProtocolError, match="stderr exceeded the 1024-byte limit"):
        client.complete("SYSTEM", "PROMPT")
    assert time.monotonic() - started < 3
    assert client.pending_cleanup_paths == ()


def test_output_limit_kills_a_descendant_before_credential_cleanup(
    tmp_path,
    monkeypatch,
):
    if os.name != "posix":
        pytest.skip("process-group lifecycle assertion requires POSIX")
    monkeypatch.setattr(codex_backend, "_MAX_JSONL_BYTES", 4096)
    monkeypatch.setattr(codex_backend, "_MAX_JSONL_LINE_BYTES", 1024)
    record_path = tmp_path / "output-limit-process.json"
    ready_path = tmp_path / "output-limit-child-ready"
    child_code = (
        "import pathlib,signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        f"pathlib.Path({str(ready_path)!r}).write_text('ready');"
        "time.sleep(60)"
    )
    client, source_home = _flooding_codex_client(
        tmp_path,
        monkeypatch,
        [
            f"child = subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}])",
            f"ready = pathlib.Path({str(ready_path)!r})",
            "while not ready.exists():",
            "    time.sleep(0.01)",
            f"pathlib.Path({str(record_path)!r}).write_text(",
            "    str(child.pid) + '\\n' + os.environ['CODEX_HOME'] + '\\n'",
            "    + os.environ['TMPDIR']",
            ")",
            "while True:",
            "    os.write(1, b'{}\\n' * 1024)",
            "    time.sleep(0.01)",
        ],
    )

    with pytest.raises(CodexProtocolError, match="stdout exceeded"):
        client.complete("SYSTEM", "PROMPT")

    child_pid_text, copied_home, copied_temp = record_path.read_text().splitlines()
    assert not Path(copied_home).exists()
    assert not Path(copied_temp).exists()
    assert (source_home / "auth.json").exists()
    assert client.pending_cleanup_paths == ()

    child_pid = int(child_pid_text)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail(
            f"descendant process {child_pid} survived the Codex output boundary"
        )


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
