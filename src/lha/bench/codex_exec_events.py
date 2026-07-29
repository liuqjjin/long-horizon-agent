"""Strict parser for the JSONL contract emitted by Codex CLI 0.141.

The wire models in this module mirror ``codex-rs/exec/src/exec_events.rs`` at
the pinned ``rust-v0.141.0`` tag.  Parsing and lifecycle validation are kept
separate: a line must first have every required field, then it must be legal at
that point in the turn.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    TypeAdapter,
    model_validator,
)


class CodexEventError(RuntimeError):
    """The Codex stream does not match the pinned JSONL protocol."""


class CodexToolBudgetExceeded(CodexEventError):
    """The stream started more tool actions than the registered budget."""


class CodexReportedError(CodexEventError):
    """The pinned JSONL protocol carried an explicit Codex failure event."""


class _StrictModel(BaseModel):
    # JSON arrays deserialize as lists before Pydantic converts them to immutable
    # tuples.  Strict scalar types below prevent bool/int coercion where it
    # matters without rejecting the JSON representation of an array.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Usage(_StrictModel):
    input_tokens: StrictInt = Field(ge=0)
    cached_input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    reasoning_output_tokens: StrictInt = Field(ge=0)


class AgentMessageItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["agent_message"]
    text: str


class ReasoningItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["reasoning"]
    text: str


class CommandExecutionItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["command_execution"]
    command: str
    aggregated_output: str
    exit_code: StrictInt | None
    status: Literal["in_progress", "completed", "failed", "declined"]

    @model_validator(mode="after")
    def _exit_code_matches_status(self) -> CommandExecutionItem:
        if self.status == "in_progress" and self.exit_code is not None:
            raise ValueError("an in-progress command cannot have an exit code")
        if self.status == "completed" and self.exit_code is None:
            raise ValueError("a completed command must have an exit code")
        if self.status == "declined" and self.exit_code is not None:
            raise ValueError("a declined command cannot have an exit code")
        return self


class FileUpdateChange(_StrictModel):
    path: str = Field(min_length=1)
    kind: Literal["add", "delete", "update"]


class FileChangeItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["file_change"]
    changes: tuple[FileUpdateChange, ...]
    status: Literal["in_progress", "completed", "failed"]


class McpToolCallResult(_StrictModel):
    content: tuple[Any, ...]
    meta: Any | None = Field(alias="_meta")
    structured_content: Any | None


class McpToolCallError(_StrictModel):
    message: str


class McpToolCallItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["mcp_tool_call"]
    server: str
    tool: str
    arguments: Any = None
    result: McpToolCallResult | None
    error: McpToolCallError | None
    status: Literal["in_progress", "completed", "failed"]

    @model_validator(mode="after")
    def _result_matches_status(self) -> McpToolCallItem:
        if self.status == "in_progress" and (self.result is not None or self.error is not None):
            raise ValueError("an in-progress MCP call cannot have a result or error")
        if self.status == "completed" and (self.result is None or self.error is not None):
            raise ValueError("a completed MCP call requires exactly one result")
        if self.status == "failed" and (self.error is None or self.result is not None):
            raise ValueError("a failed MCP call requires exactly one error")
        return self


class CollabAgentState(_StrictModel):
    status: Literal[
        "pending_init",
        "running",
        "interrupted",
        "completed",
        "errored",
        "shutdown",
        "not_found",
    ]
    message: str | None


class CollabToolCallItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["collab_tool_call"]
    tool: Literal["spawn_agent", "send_input", "wait", "close_agent"]
    sender_thread_id: str
    receiver_thread_ids: tuple[str, ...]
    prompt: str | None
    agents_states: dict[str, CollabAgentState]
    status: Literal["in_progress", "completed", "failed"]


class WebSearchActionSearch(_StrictModel):
    type: Literal["search"]
    query: str | None = None
    queries: tuple[str, ...] | None = None


class WebSearchActionOpenPage(_StrictModel):
    type: Literal["open_page"]
    url: str | None = None


class WebSearchActionFindInPage(_StrictModel):
    type: Literal["find_in_page"]
    url: str | None = None
    pattern: str | None = None


class WebSearchActionOther(_StrictModel):
    type: Literal["other"]


WebSearchAction = Annotated[
    WebSearchActionSearch
    | WebSearchActionOpenPage
    | WebSearchActionFindInPage
    | WebSearchActionOther,
    Field(discriminator="type"),
]


class WebSearchItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["web_search"]
    query: str
    action: WebSearchAction


class TodoItem(_StrictModel):
    text: str
    completed: StrictBool


class TodoListItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["todo_list"]
    items: tuple[TodoItem, ...]


class ErrorItem(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["error"]
    message: str


ThreadItem = Annotated[
    AgentMessageItem
    | ReasoningItem
    | CommandExecutionItem
    | FileChangeItem
    | McpToolCallItem
    | CollabToolCallItem
    | WebSearchItem
    | TodoListItem
    | ErrorItem,
    Field(discriminator="type"),
]


class ThreadStartedEvent(_StrictModel):
    type: Literal["thread.started"]
    thread_id: str = Field(min_length=1)


class TurnStartedEvent(_StrictModel):
    type: Literal["turn.started"]


class TurnCompletedEvent(_StrictModel):
    type: Literal["turn.completed"]
    usage: Usage


class ThreadError(_StrictModel):
    message: str


class TurnFailedEvent(_StrictModel):
    type: Literal["turn.failed"]
    error: ThreadError


class ItemStartedEvent(_StrictModel):
    type: Literal["item.started"]
    item: ThreadItem


class ItemUpdatedEvent(_StrictModel):
    type: Literal["item.updated"]
    item: ThreadItem


class ItemCompletedEvent(_StrictModel):
    type: Literal["item.completed"]
    item: ThreadItem


class ErrorEvent(_StrictModel):
    type: Literal["error"]
    message: str


ThreadEvent = Annotated[
    ThreadStartedEvent
    | TurnStartedEvent
    | TurnCompletedEvent
    | TurnFailedEvent
    | ItemStartedEvent
    | ItemUpdatedEvent
    | ItemCompletedEvent
    | ErrorEvent,
    Field(discriminator="type"),
]

_EVENT_ADAPTER = TypeAdapter(ThreadEvent)
_PAIRED_TOOL_TYPES = frozenset(
    {"command_execution", "file_change", "mcp_tool_call", "web_search"}
)
_UPDATABLE_ITEM_TYPES = frozenset(
    {"command_execution", "mcp_tool_call", "todo_list", "web_search"}
)
_COMPLETED_ONLY_TYPES = frozenset({"agent_message", "reasoning"})
_RECONNECT_NOTICE_PREFIX = "Reconnecting... 1/1 ("
_MAX_RECONNECT_NOTICE_BYTES = 512


class CodexStreamAudit(_StrictModel):
    event_counts: dict[str, int]
    item_counts: dict[str, int]
    tool_calls: int = Field(ge=0)
    reconnect_notices: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_output_tokens: int = Field(ge=0)


class CodexJsonlValidator:
    """Incrementally validate one Codex 0.141 turn.

    ``feed_line`` can be called directly from Harbor's streamed-output callback.
    The tool budget exception is therefore raised while the process is still
    running, allowing the caller to kill the task container immediately.
    """

    def __init__(
        self,
        *,
        max_tool_calls: int,
        max_line_bytes: int,
        max_total_bytes: int,
        max_reconnect_notices: int = 0,
    ) -> None:
        if min(max_tool_calls, max_line_bytes, max_total_bytes) <= 0:
            raise ValueError("Codex stream limits must be positive")
        if type(max_reconnect_notices) is not int or max_reconnect_notices < 0:
            raise ValueError("the reconnect notice limit must be a non-negative integer")
        if max_line_bytes > max_total_bytes:
            raise ValueError("the line limit cannot exceed the total stream limit")
        self.max_tool_calls = max_tool_calls
        self.max_line_bytes = max_line_bytes
        self.max_total_bytes = max_total_bytes
        self.max_reconnect_notices = max_reconnect_notices
        self.total_bytes = 0
        self.event_counts: dict[str, int] = {}
        self.item_counts: dict[str, int] = {}
        self.open_items: dict[str, str] = {}
        self.completed_items: set[str] = set()
        self.tool_calls = 0
        self.reconnect_notices = 0
        self.thread_started = False
        self.turn_started = False
        self.turn_completed = False
        self.saw_agent_message = False
        self.usage: Usage | None = None

    def feed_line(self, raw_line: str | bytes) -> None:
        if isinstance(raw_line, bytes):
            encoded = raw_line
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CodexEventError("Codex JSONL is not UTF-8") from exc
        else:
            line = raw_line
            encoded = raw_line.encode("utf-8")
        if len(encoded) > self.max_line_bytes:
            raise CodexEventError("Codex JSONL line exceeds the registered byte limit")
        self.total_bytes += len(encoded)
        if self.total_bytes > self.max_total_bytes:
            raise CodexEventError("Codex JSONL exceeds the registered total byte limit")
        line = line.strip()
        if not line:
            return
        try:
            raw_event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexEventError("Codex JSONL contains invalid JSON") from exc
        if not isinstance(raw_event, dict):
            raise CodexEventError("Codex JSONL event is not an object")
        try:
            event = _EVENT_ADAPTER.validate_python(raw_event)
        except ValueError as exc:
            raise CodexEventError("Codex JSONL event does not match 0.141") from exc
        self._accept(event)

    def _accept(self, event: ThreadEvent) -> None:
        kind = event.type
        if self.turn_completed:
            raise CodexEventError(f"Codex emitted {kind!r} after turn.completed")
        self.event_counts[kind] = self.event_counts.get(kind, 0) + 1

        if isinstance(event, ThreadStartedEvent):
            if self.thread_started or self.turn_started or sum(self.event_counts.values()) != 1:
                raise CodexEventError("thread.started must be the first and only thread start")
            self.thread_started = True
            return
        if isinstance(event, TurnStartedEvent):
            if not self.thread_started or self.turn_started:
                raise CodexEventError("turn.started must follow one thread.started")
            self.turn_started = True
            return
        if isinstance(event, TurnFailedEvent):
            raise CodexReportedError(f"Codex reported {kind}")
        if isinstance(event, ErrorEvent):
            if not self.turn_started or not self._is_reconnect_notice(event.message):
                raise CodexReportedError(f"Codex reported {kind}")
            self.reconnect_notices += 1
            if self.reconnect_notices > self.max_reconnect_notices:
                raise CodexReportedError("Codex exceeded the reconnect notice budget")
            return
        if isinstance(event, TurnCompletedEvent):
            if not self.turn_started:
                raise CodexEventError("turn.completed arrived before turn.started")
            if self.open_items:
                raise CodexEventError(
                    f"Codex completed with unfinished items: {sorted(self.open_items)}"
                )
            if not self.saw_agent_message:
                raise CodexEventError("Codex completed without a final agent message")
            self.usage = event.usage
            self.turn_completed = True
            return
        if not self.turn_started:
            raise CodexEventError(f"{kind} arrived before turn.started")

        item = event.item
        item_type = item.type
        self.item_counts[item_type] = self.item_counts.get(item_type, 0) + 1
        if isinstance(item, (ErrorItem, CollabToolCallItem)):
            raise CodexEventError(f"Codex emitted disallowed {item_type}")
        if isinstance(event, ItemStartedEvent):
            self._start_item(item)
        elif isinstance(event, ItemUpdatedEvent):
            self._update_item(item)
        else:
            self._complete_item(item)

    def _start_item(self, item: ThreadItem) -> None:
        if item.id in self.open_items or item.id in self.completed_items:
            raise CodexEventError(f"Codex reused item id {item.id!r}")
        if item.type in _COMPLETED_ONLY_TYPES:
            raise CodexEventError(f"{item.type} must be emitted only as item.completed")
        if isinstance(item, CommandExecutionItem) and item.status != "in_progress":
            raise CodexEventError("a started command must be in_progress")
        if isinstance(item, McpToolCallItem) and item.status != "in_progress":
            raise CodexEventError("a started MCP call must be in_progress")
        if isinstance(item, FileChangeItem) and item.status != "in_progress":
            raise CodexEventError("a started file change must be in_progress")
        if item.type in _PAIRED_TOOL_TYPES:
            self._start_tool()
        self.open_items[item.id] = item.type

    def _update_item(self, item: ThreadItem) -> None:
        started_type = self.open_items.get(item.id)
        if started_type != item.type:
            raise CodexEventError(f"Codex updated unknown or changed item {item.id!r}")
        if item.type not in _UPDATABLE_ITEM_TYPES:
            raise CodexEventError(f"{item.type} cannot be emitted as item.updated")
        if isinstance(item, CommandExecutionItem) and item.status != "in_progress":
            raise CodexEventError("an updated command must remain in_progress")
        if isinstance(item, McpToolCallItem) and item.status != "in_progress":
            raise CodexEventError("an updated MCP call must remain in_progress")

    def _complete_item(self, item: ThreadItem) -> None:
        if item.type in _COMPLETED_ONLY_TYPES:
            if item.id in self.open_items or item.id in self.completed_items:
                raise CodexEventError(f"Codex reused completed-only item {item.id!r}")
            if isinstance(item, AgentMessageItem):
                self.saw_agent_message = True
            self.completed_items.add(item.id)
            return

        started_type = self.open_items.pop(item.id, None)
        if started_type != item.type:
            raise CodexEventError(f"Codex completed unknown or changed item {item.id!r}")
        if isinstance(item, CommandExecutionItem) and item.status == "in_progress":
            raise CodexEventError("a completed command is still in_progress")
        if isinstance(item, McpToolCallItem) and item.status == "in_progress":
            raise CodexEventError("a completed MCP call is still in_progress")
        if isinstance(item, FileChangeItem) and item.status == "in_progress":
            raise CodexEventError("a completed file change is still in_progress")
        self.completed_items.add(item.id)

    def _start_tool(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            raise CodexToolBudgetExceeded(
                f"Codex started tool call {self.tool_calls}; limit is {self.max_tool_calls}"
            )

    @staticmethod
    def _is_reconnect_notice(message: str) -> bool:
        """Recognize only Codex 0.141's bounded, printable retry notification."""
        encoded = message.encode("utf-8")
        if (
            len(encoded) > _MAX_RECONNECT_NOTICE_BYTES
            or not message.startswith(_RECONNECT_NOTICE_PREFIX)
            or not message.endswith(")")
        ):
            return False
        detail = message[len(_RECONNECT_NOTICE_PREFIX) : -1]
        return bool(detail) and all(
            0x20 <= ord(character) <= 0x7E for character in detail
        )

    def finish(self) -> CodexStreamAudit:
        if not self.thread_started:
            raise CodexEventError("Codex JSONL is missing thread.started")
        if not self.turn_started:
            raise CodexEventError("Codex JSONL is missing turn.started")
        if not self.turn_completed or self.usage is None:
            raise CodexEventError("Codex JSONL is missing turn.completed")
        return CodexStreamAudit(
            event_counts=dict(sorted(self.event_counts.items())),
            item_counts=dict(sorted(self.item_counts.items())),
            tool_calls=self.tool_calls,
            reconnect_notices=self.reconnect_notices,
            input_tokens=self.usage.input_tokens,
            cached_input_tokens=self.usage.cached_input_tokens,
            output_tokens=self.usage.output_tokens,
            reasoning_output_tokens=self.usage.reasoning_output_tokens,
        )


def audit_codex_0141_jsonl(
    event_stream: str,
    *,
    max_tool_calls: int,
    max_line_bytes: int,
    max_total_bytes: int,
    max_reconnect_notices: int = 0,
) -> CodexStreamAudit:
    """Validate a complete buffered stream using the incremental parser."""
    validator = CodexJsonlValidator(
        max_tool_calls=max_tool_calls,
        max_line_bytes=max_line_bytes,
        max_total_bytes=max_total_bytes,
        max_reconnect_notices=max_reconnect_notices,
    )
    for line in event_stream.splitlines(keepends=True):
        validator.feed_line(line)
    return validator.finish()
