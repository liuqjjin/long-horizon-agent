"""Append-only registration records for formal ablation attempts.

The registry prevents a formal run from moving to a fresh output directory
without leaving a committed record.  It does not claim that private forks or
deleted Git history never existed; it constrains the official runner and the
evidence accepted by the release check.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

FORMAL_ABLATION_ATTEMPTS_PATH = PurePosixPath(
    "benchmarks/formal_ablation_attempts.json"
)
FORMAL_ABLATION_ATTEMPTS_SCHEMA = 1
FORMAL_ABLATION_PROTOCOL_SCHEMA = 1
MAX_FORMAL_ABLATION_ATTEMPTS_BYTES = 1024 * 1024

_HEX_40_PATTERN = r"^[0-9a-f]{40}$"
_HEX_64_PATTERN = r"^[0-9a-f]{64}$"
_DOCKER_IMAGE_ID_PATTERN = r"^sha256:[0-9a-f]{64}$"
_WITNESS_REMOTE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_SCP_REMOTE_PATTERN = re.compile(
    r"^(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9.-]+:[A-Za-z0-9._~/-]+$"
)

FORMAL_ATTEMPT_WITNESS_REF_PREFIX = "refs/heads/formal-attempts"
FORMAL_ATTEMPT_WITNESS_SCHEMA = 1
FORMAL_ATTEMPT_WITNESS_IDENTITY = (
    "liuqjjin <156533013+liuqjjin@users.noreply.github.com>"
)
FORMAL_ATTEMPT_WITNESS_GIT_TIME = "946684800 +0000"


def _validate_timestamp(value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError("attempt timestamp must be a non-empty ISO-8601 value")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("attempt timestamp must be valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("attempt timestamp must include a UTC offset")
    return value


def _validate_output_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or value == "."
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("formal ablation output_path must be a fixed repository path")
    return value


def _validate_plain_text(value: str) -> str:
    if not value or value.strip() != value or "\x00" in value:
        raise ValueError("formal ablation text field is invalid")
    return value


def _validate_witness_remote_url(value: str) -> str:
    """Accept public, non-interactive Git URLs without embedded passwords."""
    if (
        not value
        or value.strip() != value
        or len(value.encode("utf-8")) > 2048
        or value.startswith("-")
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ValueError("formal ablation witness remote URL is invalid")
    if value.startswith("/"):
        path = PurePosixPath(value)
        if any(part in {"", ".", ".."} for part in path.parts[1:]):
            raise ValueError("formal ablation witness remote URL is invalid")
        return value
    if _SCP_REMOTE_PATTERN.fullmatch(value):
        return value
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"git", "https", "ssh"}
        or not parsed.hostname
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("formal ablation witness remote URL is invalid")
    if parsed.scheme in {"git", "https"} and parsed.username is not None:
        raise ValueError("formal ablation witness remote URL cannot contain credentials")
    return value


def formal_ablation_witness_ref(attempt_id: str) -> str:
    """Derive the only remote ref that may consume an attempt."""
    if re.fullmatch(_HEX_64_PATTERN, attempt_id) is None:
        raise ValueError("formal ablation witness attempt_id is invalid")
    return f"{FORMAL_ATTEMPT_WITNESS_REF_PREFIX}/{attempt_id}"


def formal_ablation_witness_message(
    *,
    attempt_id: str,
    registration_registry_sha256: str,
    protocol_sha256: str,
    outcome_key: str,
    run_header_sha256: str,
) -> bytes:
    """Return the canonical, secret-free commit message for one start witness."""
    fields = {
        "schema_version": FORMAL_ATTEMPT_WITNESS_SCHEMA,
        "formal_attempt_id": attempt_id,
        "registration_registry_sha256": registration_registry_sha256,
        "protocol_sha256": protocol_sha256,
        "outcome_key": outcome_key,
        "run_header_sha256": run_header_sha256,
    }
    for name, value in fields.items():
        if name == "schema_version":
            continue
        if not isinstance(value, str) or re.fullmatch(_HEX_64_PATTERN, value) is None:
            raise ValueError(f"formal ablation witness {name} is invalid")
    payload = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"LHA formal ablation attempt witness\n\n{payload}\n".encode("ascii")


def formal_ablation_witness_commit_bytes(
    *,
    tree: str,
    parent: str,
    message: bytes,
) -> bytes:
    """Build the exact deterministic Git commit object content for a witness."""
    if re.fullmatch(_HEX_40_PATTERN, tree) is None:
        raise ValueError("formal ablation witness tree is invalid")
    if re.fullmatch(_HEX_40_PATTERN, parent) is None:
        raise ValueError("formal ablation witness parent is invalid")
    if (
        not message
        or b"\x00" in message
        or not message.endswith(b"\n")
        or message != message.decode("ascii").encode("ascii")
    ):
        raise ValueError("formal ablation witness message is invalid")
    identity = FORMAL_ATTEMPT_WITNESS_IDENTITY
    timestamp = FORMAL_ATTEMPT_WITNESS_GIT_TIME
    header = (
        f"tree {tree}\n"
        f"parent {parent}\n"
        f"author {identity} {timestamp}\n"
        f"committer {identity} {timestamp}\n"
        "\n"
    ).encode("ascii")
    return header + message


def formal_ablation_witness_commit_oid(payload: bytes) -> str:
    """Compute the SHA-1 object ID used by the repository's 40-hex protocol."""
    object_bytes = f"commit {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(object_bytes, usedforsecurity=False).hexdigest()


class FormalCodexClientConfig(BaseModel):
    """Outcome-affecting Codex client settings fixed before registration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    no_tools: Literal[True] = True
    sandbox_mode: Literal["read-only"] = "read-only"
    permission_model: Literal["profile"] = "profile"
    permission_profile: Literal["lha-read"] = "lha-read"
    credential_barrier: Literal["verified"] = "verified"
    externally_sandboxed: Literal[False] = False
    max_retries: int = Field(ge=0, strict=True)
    timeout_s: float = Field(gt=0, allow_inf_nan=False, strict=True)
    retry_backoff_s: float = Field(
        ge=0,
        allow_inf_nan=False,
        strict=True,
    )


def formal_codex_client_sha256(config: FormalCodexClientConfig) -> str:
    """Hash the canonical outcome-affecting Codex client configuration."""
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FormalAblationProtocol(BaseModel):
    """Fields that select the code, corpus, model, and execution image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = FORMAL_ABLATION_PROTOCOL_SCHEMA
    source_commit: str = Field(pattern=_HEX_40_PATTERN)
    source_tree_sha256: str = Field(pattern=_HEX_64_PATTERN)
    manifest_sha256: str = Field(pattern=_HEX_64_PATTERN)
    model: str
    reasoning_effort: str
    docker_image_id: str = Field(pattern=_DOCKER_IMAGE_ID_PATTERN)
    codex_cli_version: str
    codex_cli_executable_sha256: str = Field(pattern=_HEX_64_PATTERN)
    codex_client: FormalCodexClientConfig
    codex_client_sha256: str = Field(pattern=_HEX_64_PATTERN)

    _model = field_validator("model")(_validate_plain_text)
    _effort = field_validator("reasoning_effort")(_validate_plain_text)
    _cli_version = field_validator("codex_cli_version")(_validate_plain_text)

    @model_validator(mode="after")
    def _client_digest_matches(self) -> "FormalAblationProtocol":
        if self.codex_client_sha256 != formal_codex_client_sha256(
            self.codex_client
        ):
            raise ValueError(
                "formal ablation codex_client_sha256 does not match codex_client"
            )
        return self


def formal_ablation_protocol_sha256(protocol: FormalAblationProtocol) -> str:
    """Hash the canonical execution choices, excluding attempt ID and timestamp."""
    payload = json.dumps(
        protocol.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def formal_ablation_selection_sha256(protocol: FormalAblationProtocol) -> str:
    """Hash outcome-affecting inputs while ignoring commit-only history changes.

    The protocol still records the exact source commit.  This second identity
    prevents an empty commit or a registry-only commit after ABANDONED from
    making the same code/model/corpus selection eligible for another attempt.
    """
    payload = protocol.model_dump(mode="json")
    payload.pop("source_commit")
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


class RegisteredAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["REGISTERED"] = "REGISTERED"
    attempt_id: str = Field(pattern=_HEX_64_PATTERN)
    protocol_sha256: str = Field(pattern=_HEX_64_PATTERN)
    source_commit: str = Field(pattern=_HEX_40_PATTERN)
    source_tree_sha256: str = Field(pattern=_HEX_64_PATTERN)
    manifest_sha256: str = Field(pattern=_HEX_64_PATTERN)
    output_path: str
    model: str
    reasoning_effort: str
    docker_image_id: str = Field(pattern=_DOCKER_IMAGE_ID_PATTERN)
    codex_cli_version: str
    codex_cli_executable_sha256: str = Field(pattern=_HEX_64_PATTERN)
    codex_client: FormalCodexClientConfig
    codex_client_sha256: str = Field(pattern=_HEX_64_PATTERN)
    witness_remote_name: str = Field(pattern=_WITNESS_REMOTE_NAME_PATTERN)
    witness_remote_url: str
    registered_at: str

    _output_path = field_validator("output_path")(_validate_output_path)
    _model = field_validator("model")(_validate_plain_text)
    _effort = field_validator("reasoning_effort")(_validate_plain_text)
    _cli_version = field_validator("codex_cli_version")(_validate_plain_text)
    _witness_remote_url = field_validator("witness_remote_url")(
        _validate_witness_remote_url
    )
    _registered_at = field_validator("registered_at")(_validate_timestamp)

    @property
    def witness_ref(self) -> str:
        return formal_ablation_witness_ref(self.attempt_id)

    def protocol(self) -> FormalAblationProtocol:
        return FormalAblationProtocol(
            source_commit=self.source_commit,
            source_tree_sha256=self.source_tree_sha256,
            manifest_sha256=self.manifest_sha256,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            docker_image_id=self.docker_image_id,
            codex_cli_version=self.codex_cli_version,
            codex_cli_executable_sha256=self.codex_cli_executable_sha256,
            codex_client=self.codex_client,
            codex_client_sha256=self.codex_client_sha256,
        )

    @model_validator(mode="after")
    def _protocol_digest_matches(self) -> "RegisteredAttempt":
        expected_output = f"runs/formal_ablation/{self.attempt_id}"
        if self.output_path != expected_output:
            raise ValueError(
                "REGISTERED output_path must equal runs/formal_ablation/<attempt_id>"
            )
        if self.protocol_sha256 != formal_ablation_protocol_sha256(self.protocol()):
            raise ValueError("REGISTERED protocol_sha256 does not match its fields")
        return self


class AbandonedAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["ABANDONED"] = "ABANDONED"
    attempt_id: str = Field(pattern=_HEX_64_PATTERN)
    recorded_at: str
    started_cells: int = Field(ge=0, le=204, strict=True)
    terminal_cells: int = Field(ge=0, le=204, strict=True)
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    reason: str
    report_sha256: str | None = Field(default=None, pattern=_HEX_64_PATTERN)
    report_fingerprint: str | None = Field(default=None, pattern=_HEX_64_PATTERN)

    _recorded_at = field_validator("recorded_at")(_validate_timestamp)
    _reason = field_validator("reason")(_validate_plain_text)

    @model_validator(mode="after")
    def _progress_and_report_are_consistent(self) -> "AbandonedAttempt":
        if self.terminal_cells > self.started_cells:
            raise ValueError("ABANDONED terminal_cells cannot exceed started_cells")
        if (self.report_sha256 is None) != (self.report_fingerprint is None):
            raise ValueError(
                "ABANDONED report_sha256 and report_fingerprint must appear together"
            )
        return self


class CompletedAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["COMPLETED"] = "COMPLETED"
    attempt_id: str = Field(pattern=_HEX_64_PATTERN)
    protocol_sha256: str = Field(pattern=_HEX_64_PATTERN)
    registration_registry_sha256: str = Field(pattern=_HEX_64_PATTERN)
    recorded_at: str
    report_sha256: str = Field(pattern=_HEX_64_PATTERN)
    report_fingerprint: str = Field(pattern=_HEX_64_PATTERN)

    _recorded_at = field_validator("recorded_at")(_validate_timestamp)


class UnregisteredRunRecorded(BaseModel):
    """A historical disclosure that never opens or authorizes an attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["UNREGISTERED_RUN_RECORDED"] = "UNREGISTERED_RUN_RECORDED"
    attempt_id: str = Field(pattern=_HEX_64_PATTERN)
    protocol_sha256: str = Field(pattern=_HEX_64_PATTERN)
    source_commit: str = Field(pattern=_HEX_40_PATTERN)
    source_tree_sha256: str = Field(pattern=_HEX_64_PATTERN)
    manifest_sha256: str = Field(pattern=_HEX_64_PATTERN)
    output_path: str
    model: str
    reasoning_effort: str
    docker_image_id: str = Field(pattern=_DOCKER_IMAGE_ID_PATTERN)
    codex_cli_version: str
    codex_cli_executable_sha256: str = Field(pattern=_HEX_64_PATTERN)
    codex_client: FormalCodexClientConfig
    codex_client_sha256: str = Field(pattern=_HEX_64_PATTERN)
    recorded_at: str
    reason: str
    published_report_path: str
    report_sha256: str = Field(pattern=_HEX_64_PATTERN)
    report_fingerprint: str = Field(pattern=_HEX_64_PATTERN)
    scheduled_cells: int = Field(ge=1, strict=True)
    usable_cells: int = Field(ge=0, strict=True)
    error_cells: int = Field(ge=0, strict=True)
    trust_delivered_correct: int = Field(ge=0, strict=True)
    trust_delivered_wrong: int = Field(ge=0, strict=True)
    gate_delivered_correct: int = Field(ge=0, strict=True)
    gate_delivered_wrong: int = Field(ge=0, strict=True)
    gate_intercepted_wrong: int = Field(ge=0, strict=True)
    gate_rejected_correct: int = Field(ge=0, strict=True)
    verify_delivered_correct: int = Field(ge=0, strict=True)
    verify_delivered_wrong: int = Field(ge=0, strict=True)
    verify_not_delivered: int = Field(ge=0, strict=True)

    _output_path = field_validator("output_path")(_validate_output_path)
    _model = field_validator("model")(_validate_plain_text)
    _effort = field_validator("reasoning_effort")(_validate_plain_text)
    _cli_version = field_validator("codex_cli_version")(_validate_plain_text)
    _recorded_at = field_validator("recorded_at")(_validate_timestamp)
    _reason = field_validator("reason")(_validate_plain_text)
    _published_report_path = field_validator("published_report_path")(
        _validate_output_path
    )

    def protocol(self) -> FormalAblationProtocol:
        return FormalAblationProtocol(
            source_commit=self.source_commit,
            source_tree_sha256=self.source_tree_sha256,
            manifest_sha256=self.manifest_sha256,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            docker_image_id=self.docker_image_id,
            codex_cli_version=self.codex_cli_version,
            codex_cli_executable_sha256=self.codex_cli_executable_sha256,
            codex_client=self.codex_client,
            codex_client_sha256=self.codex_client_sha256,
        )

    @model_validator(mode="after")
    def _protocol_digest_matches(self) -> "UnregisteredRunRecorded":
        expected_report_path = (
            "benchmarks/formal_ablation_history/"
            f"{self.source_commit}/ablation_report.json"
        )
        if self.published_report_path != expected_report_path:
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED published_report_path must be derived "
                "from source_commit"
            )
        if self.protocol_sha256 != formal_ablation_protocol_sha256(self.protocol()):
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED protocol_sha256 does not match its fields"
            )
        if self.usable_cells + self.error_cells != self.scheduled_cells:
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED cell totals are inconsistent"
            )
        if (
            self.trust_delivered_correct + self.trust_delivered_wrong
            != self.usable_cells
        ):
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED trust counts are inconsistent"
            )
        gate_total = (
            self.gate_delivered_correct
            + self.gate_delivered_wrong
            + self.gate_intercepted_wrong
            + self.gate_rejected_correct
        )
        verify_total = (
            self.verify_delivered_correct
            + self.verify_delivered_wrong
            + self.verify_not_delivered
        )
        if gate_total != self.usable_cells or verify_total != self.usable_cells:
            raise ValueError(
                "UNREGISTERED_RUN_RECORDED gate or verify counts are inconsistent"
            )
        return self


FormalAttemptEvent = Annotated[
    RegisteredAttempt
    | AbandonedAttempt
    | CompletedAttempt
    | UnregisteredRunRecorded,
    Field(discriminator="event"),
]


class FormalAblationAttemptRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = FORMAL_ABLATION_ATTEMPTS_SCHEMA
    events: tuple[FormalAttemptEvent, ...]

    @model_validator(mode="after")
    def _events_form_an_append_only_state_machine(
        self,
    ) -> "FormalAblationAttemptRegistry":
        registered: dict[str, RegisteredAttempt] = {}
        disclosed_ids: set[str] = set()
        disclosed_reports: set[str] = set()
        terminal_ids: set[str] = set()
        consumed_protocols: set[str] = set()
        consumed_selections: set[str] = set()
        open_attempt: str | None = None

        for event in self.events:
            if isinstance(event, RegisteredAttempt):
                if open_attempt is not None:
                    raise ValueError(
                        "an open REGISTERED attempt must end before a new attempt"
                    )
                if event.attempt_id in registered or event.attempt_id in disclosed_ids:
                    raise ValueError("formal ablation attempt_id must be unique")
                selection_sha256 = formal_ablation_selection_sha256(event.protocol())
                if (
                    event.protocol_sha256 in consumed_protocols
                    or selection_sha256 in consumed_selections
                ):
                    raise ValueError(
                        "a consumed formal ablation selection cannot be registered again"
                    )
                registered[event.attempt_id] = event
                consumed_protocols.add(event.protocol_sha256)
                consumed_selections.add(selection_sha256)
                open_attempt = event.attempt_id
                continue

            if isinstance(event, UnregisteredRunRecorded):
                if open_attempt is not None:
                    raise ValueError(
                        "an open REGISTERED attempt must end before another attempt record"
                    )
                if event.attempt_id in registered or event.attempt_id in disclosed_ids:
                    raise ValueError("formal ablation attempt_id must be unique")
                if event.published_report_path in disclosed_reports:
                    raise ValueError(
                        "formal ablation historical report path must be unique"
                    )
                selection_sha256 = formal_ablation_selection_sha256(event.protocol())
                if (
                    event.protocol_sha256 in consumed_protocols
                    or selection_sha256 in consumed_selections
                ):
                    raise ValueError(
                        "a consumed formal ablation selection cannot be recorded again"
                    )
                disclosed_ids.add(event.attempt_id)
                disclosed_reports.add(event.published_report_path)
                consumed_protocols.add(event.protocol_sha256)
                consumed_selections.add(selection_sha256)
                continue

            if event.attempt_id not in registered:
                raise ValueError(f"{event.event} must follow a REGISTERED attempt")
            if event.attempt_id in terminal_ids:
                raise ValueError("a terminal formal ablation attempt cannot change state")
            if open_attempt != event.attempt_id:
                raise ValueError(f"{event.event} does not terminate the open attempt")
            registration = registered[event.attempt_id]
            if isinstance(event, CompletedAttempt):
                if event.protocol_sha256 != registration.protocol_sha256:
                    raise ValueError(
                        "COMPLETED protocol_sha256 differs from its registration"
                    )
            terminal_ids.add(event.attempt_id)
            open_attempt = None
        return self

    def open_registration(self) -> RegisteredAttempt | None:
        open_attempt: RegisteredAttempt | None = None
        for event in self.events:
            if isinstance(event, RegisteredAttempt):
                open_attempt = event
            elif isinstance(event, (AbandonedAttempt, CompletedAttempt)):
                if open_attempt is not None and open_attempt.attempt_id == event.attempt_id:
                    open_attempt = None
        return open_attempt

    def registration(self, attempt_id: str) -> RegisteredAttempt | None:
        return next(
            (
                event
                for event in self.events
                if isinstance(event, RegisteredAttempt)
                and event.attempt_id == attempt_id
            ),
            None,
        )

    def completions(self) -> tuple[CompletedAttempt, ...]:
        return tuple(
            event for event in self.events if isinstance(event, CompletedAttempt)
        )

    def disclosures(self) -> tuple[UnregisteredRunRecorded, ...]:
        return tuple(
            event
            for event in self.events
            if isinstance(event, UnregisteredRunRecorded)
        )


def parse_formal_ablation_attempt_registry(
    payload: bytes,
) -> FormalAblationAttemptRegistry:
    if len(payload) > MAX_FORMAL_ABLATION_ATTEMPTS_BYTES:
        raise ValueError("formal ablation attempt registry is too large")
    try:
        raw: Any = json.loads(payload)
        return FormalAblationAttemptRegistry.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError("formal ablation attempt registry is invalid") from error


def formal_ablation_attempt_registry_bytes(
    registry: FormalAblationAttemptRegistry,
) -> bytes:
    """Return one stable representation suitable for a tracked registry file."""
    return (
        json.dumps(
            registry.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def registry_has_prefix(
    current: FormalAblationAttemptRegistry,
    historical: FormalAblationAttemptRegistry,
) -> bool:
    """Whether current retains every historical event in the original order."""
    if len(current.events) < len(historical.events):
        return False
    return current.events[: len(historical.events)] == historical.events
