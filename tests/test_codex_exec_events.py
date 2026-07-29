from __future__ import annotations

import json

import pytest

from lha.bench.codex_exec_events import (
    CodexEventError,
    CodexJsonlValidator,
    CodexReportedError,
    CodexToolBudgetExceeded,
    audit_codex_0141_jsonl,
)


def _line(payload) -> str:
    return json.dumps(payload, separators=(",", ":")) + "\n"


def _complete_stream(*items: dict) -> str:
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        *items,
        {
            "type": "item.completed",
            "item": {
                "id": "answer-1",
                "type": "agent_message",
                "text": "Implemented and checked.",
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 11,
                "cached_input_tokens": 2,
                "output_tokens": 5,
                "reasoning_output_tokens": 3,
            },
        },
    ]
    return "".join(_line(event) for event in events)


def _command_pair(index: int = 1) -> tuple[dict, dict]:
    item_id = f"command-{index}"
    return (
        {
            "type": "item.started",
            "item": {
                "id": item_id,
                "type": "command_execution",
                "command": "pytest -q",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": item_id,
                "type": "command_execution",
                "command": "pytest -q",
                "aggregated_output": "1 passed\n",
                "exit_code": 0,
                "status": "completed",
            },
        },
    )


def _file_change_pair(index: int = 1) -> tuple[dict, dict]:
    item_id = f"file-{index}"
    changes = [{"path": "/app/convert_masks.py", "kind": "add"}]
    return (
        {
            "type": "item.started",
            "item": {
                "id": item_id,
                "type": "file_change",
                "changes": changes,
                "status": "in_progress",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": item_id,
                "type": "file_change",
                "changes": changes,
                "status": "completed",
            },
        },
    )


def _reconnect_event(message: str = "Reconnecting... 1/1 (stream interrupted)") -> dict:
    return {"type": "error", "message": message}


def test_official_0141_shape_is_accepted():
    audit = audit_codex_0141_jsonl(
        _complete_stream(*_command_pair()),
        max_tool_calls=20,
        max_line_bytes=1024 * 1024,
        max_total_bytes=16 * 1024 * 1024,
    )

    assert audit.tool_calls == 1
    assert audit.input_tokens == 11
    assert audit.reasoning_output_tokens == 3
    assert audit.item_counts == {"agent_message": 1, "command_execution": 2}
    assert audit.reconnect_notices == 0


def test_real_0141_file_change_started_then_completed_is_accepted():
    started, completed = _file_change_pair()
    audit = audit_codex_0141_jsonl(
        _complete_stream(_reconnect_event(), started, completed),
        max_tool_calls=20,
        max_line_bytes=4096,
        max_total_bytes=65536,
        max_reconnect_notices=1,
    )

    assert audit.tool_calls == 1
    assert audit.reconnect_notices == 1
    assert audit.item_counts == {"agent_message": 1, "file_change": 2}


def test_file_change_completed_without_start_is_rejected():
    _, completed = _file_change_pair()

    with pytest.raises(CodexEventError, match="completed unknown or changed"):
        audit_codex_0141_jsonl(
            _complete_stream(completed),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_file_change_duplicate_id_is_rejected_while_open_and_after_completion():
    started, completed = _file_change_pair()
    with pytest.raises(CodexEventError, match="reused item id"):
        audit_codex_0141_jsonl(
            _complete_stream(started, started),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )

    with pytest.raises(CodexEventError, match="reused item id"):
        audit_codex_0141_jsonl(
            _complete_stream(started, completed, started),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_file_change_type_drift_is_rejected():
    started, _ = _file_change_pair()
    _, command_completed = _command_pair()
    command_completed["item"]["id"] = started["item"]["id"]

    with pytest.raises(CodexEventError, match="completed unknown or changed"):
        audit_codex_0141_jsonl(
            _complete_stream(started, command_completed),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_file_change_must_reach_a_terminal_completion():
    started, completed = _file_change_pair()
    with pytest.raises(CodexEventError, match="unfinished items"):
        audit_codex_0141_jsonl(
            _complete_stream(started),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )

    completed["item"]["status"] = "in_progress"
    with pytest.raises(CodexEventError, match="completed file change is still in_progress"):
        audit_codex_0141_jsonl(
            _complete_stream(started, completed),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_file_change_started_status_and_updates_are_strict():
    started, completed = _file_change_pair()
    started["item"]["status"] = "completed"
    with pytest.raises(CodexEventError, match="started file change must be in_progress"):
        audit_codex_0141_jsonl(
            _complete_stream(started, completed),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )

    started["item"]["status"] = "in_progress"
    updated = {**started, "type": "item.updated"}
    with pytest.raises(CodexEventError, match="file_change cannot be emitted as item.updated"):
        audit_codex_0141_jsonl(
            _complete_stream(started, updated, completed),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


@pytest.mark.parametrize(
    "item",
    [
        {
            "id": "file-1",
            "type": "file_change",
            "status": "in_progress",
        },
        {
            "id": "file-1",
            "type": "file_change",
            "changes": [{"path": "/app/convert_masks.py"}],
            "status": "in_progress",
        },
        {
            "id": "file-1",
            "type": "file_change",
            "changes": [{"path": "/app/convert_masks.py", "kind": "rename"}],
            "status": "in_progress",
        },
    ],
)
def test_file_change_rejects_malformed_fields(item):
    with pytest.raises(CodexEventError, match="does not match 0.141"):
        audit_codex_0141_jsonl(
            _complete_stream({"type": "item.started", "item": item}),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_file_change_counts_against_tool_budget_at_start():
    command_started, command_completed = _command_pair()
    file_started, _ = _file_change_pair()

    with pytest.raises(CodexToolBudgetExceeded, match="tool call 2"):
        audit_codex_0141_jsonl(
            _complete_stream(command_started, command_completed, file_started),
            max_tool_calls=1,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_one_strict_reconnect_notice_can_finish_successfully():
    audit = audit_codex_0141_jsonl(
        _complete_stream(
            _reconnect_event(
                "Reconnecting... 1/1 (request failed for url (https://example.test))"
            )
        ),
        max_tool_calls=20,
        max_line_bytes=4096,
        max_total_bytes=65536,
        max_reconnect_notices=1,
    )

    assert audit.reconnect_notices == 1
    assert audit.event_counts["error"] == 1


def test_reconnect_notice_does_not_replace_turn_completion():
    validator = CodexJsonlValidator(
        max_tool_calls=20,
        max_line_bytes=4096,
        max_total_bytes=65536,
        max_reconnect_notices=1,
    )
    validator.feed_line(_line({"type": "thread.started", "thread_id": "thread-1"}))
    validator.feed_line(_line({"type": "turn.started"}))
    validator.feed_line(_line(_reconnect_event()))

    with pytest.raises(CodexEventError, match="missing turn.completed"):
        validator.finish()


def test_second_reconnect_notice_exceeds_the_fixed_budget():
    stream = _complete_stream(_reconnect_event(), _reconnect_event())

    with pytest.raises(CodexReportedError, match="reconnect notice budget"):
        audit_codex_0141_jsonl(
            stream,
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
            max_reconnect_notices=1,
        )


@pytest.mark.parametrize(
    "message",
    [
        "Reconnecting... 1/2 (stream interrupted)",
        "Reconnecting... 1/1 ()",
        "Reconnecting... 1/1 (line\nbreak)",
        "Reconnecting... 1/1 (snowman ☃)",
        "Reconnecting... 1/1 (stream interrupted) trailing",
        "reconnecting... 1/1 (stream interrupted)",
        "Reconnecting... 1/1 (" + "x" * 512 + ")",
        "ordinary provider error",
    ],
)
def test_non_strict_error_events_still_fail_immediately(message):
    stream = _complete_stream(_reconnect_event(message))

    with pytest.raises(CodexReportedError, match="Codex reported error"):
        audit_codex_0141_jsonl(
            stream,
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
            max_reconnect_notices=1,
        )


def test_reconnect_notice_before_turn_started_is_reported_error():
    validator = CodexJsonlValidator(
        max_tool_calls=20,
        max_line_bytes=4096,
        max_total_bytes=65536,
        max_reconnect_notices=1,
    )
    validator.feed_line(_line({"type": "thread.started", "thread_id": "thread-1"}))

    with pytest.raises(CodexReportedError, match="Codex reported error"):
        validator.feed_line(_line(_reconnect_event()))


def test_reconnect_notice_is_disabled_by_default():
    with pytest.raises(CodexReportedError, match="reconnect notice budget"):
        audit_codex_0141_jsonl(
            _complete_stream(_reconnect_event()),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


@pytest.mark.parametrize(
    "event",
    [
        {"type": "thread.started"},
        {
            "type": "item.completed",
            "item": {"id": "answer-1", "type": "agent_message"},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1,
                "cached_input_tokens": 0,
                "output_tokens": 1,
            },
        },
    ],
)
def test_codex_0141_rejects_missing_required_fields(event):
    validator = CodexJsonlValidator(
        max_tool_calls=20,
        max_line_bytes=4096,
        max_total_bytes=65536,
    )
    if event["type"] != "thread.started":
        validator.feed_line(_line({"type": "thread.started", "thread_id": "thread-1"}))
        validator.feed_line(_line({"type": "turn.started"}))
        if event["type"] == "turn.completed":
            validator.feed_line(
                _line(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "answer-1",
                            "type": "agent_message",
                            "text": "done",
                        },
                    }
                )
            )
    with pytest.raises(CodexEventError, match="does not match 0.141"):
        validator.feed_line(_line(event))


def test_codex_0141_rejects_unfinished_todo():
    stream = _complete_stream(
        {
            "type": "item.started",
            "item": {
                "id": "todo-1",
                "type": "todo_list",
                "items": [{"text": "run tests", "completed": False}],
            },
        }
    )
    with pytest.raises(CodexEventError, match="unfinished items"):
        audit_codex_0141_jsonl(
            stream,
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_codex_0141_rejects_nonterminal_completed_item():
    started, completed = _command_pair()
    completed["item"]["status"] = "in_progress"
    completed["item"]["exit_code"] = None
    with pytest.raises(CodexEventError, match="completed command is still in_progress"):
        audit_codex_0141_jsonl(
            _complete_stream(started, completed),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_codex_0141_requires_reasoning_usage():
    stream = _complete_stream()
    events = [json.loads(line) for line in stream.splitlines()]
    del events[-1]["usage"]["reasoning_output_tokens"]
    with pytest.raises(CodexEventError, match="does not match 0.141"):
        audit_codex_0141_jsonl(
            "".join(_line(event) for event in events),
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_21st_tool_call_stops_stream_before_completion():
    validator = CodexJsonlValidator(
        max_tool_calls=20,
        max_line_bytes=4096,
        max_total_bytes=1024 * 1024,
    )
    validator.feed_line(_line({"type": "thread.started", "thread_id": "thread-1"}))
    validator.feed_line(_line({"type": "turn.started"}))
    for index in range(1, 21):
        started, completed = _command_pair(index)
        validator.feed_line(_line(started))
        validator.feed_line(_line(completed))

    started, _ = _command_pair(21)
    with pytest.raises(CodexToolBudgetExceeded, match="tool call 21"):
        validator.feed_line(_line(started))


def test_collab_is_rejected_even_when_well_formed():
    stream = _complete_stream(
        {
            "type": "item.started",
            "item": {
                "id": "collab-1",
                "type": "collab_tool_call",
                "tool": "spawn_agent",
                "sender_thread_id": "thread-1",
                "receiver_thread_ids": ["thread-2"],
                "prompt": "help",
                "agents_states": {
                    "thread-2": {"status": "running", "message": None}
                },
                "status": "in_progress",
            },
        }
    )
    with pytest.raises(CodexEventError, match="disallowed collab_tool_call"):
        audit_codex_0141_jsonl(
            stream,
            max_tool_calls=20,
            max_line_bytes=4096,
            max_total_bytes=65536,
        )


def test_stream_limits_fail_before_unbounded_buffering():
    validator = CodexJsonlValidator(
        max_tool_calls=20,
        max_line_bytes=32,
        max_total_bytes=64,
    )
    with pytest.raises(CodexEventError, match="line exceeds"):
        validator.feed_line("x" * 33)

    validator.feed_line(" " * 32)
    validator.feed_line(" " * 32)
    with pytest.raises(CodexEventError, match="total byte limit"):
        validator.feed_line(" ")
