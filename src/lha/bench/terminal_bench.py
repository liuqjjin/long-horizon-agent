"""Terminal-Bench 2.1 preregistration and Harbor adapter.

Harbor's verifier is the only source of task truth.  The adapter runs one real
``codex exec`` in Harbor's disposable task container and never uses LHA's
internal gate as a score.

The task container never receives a real Codex credential.  A short-lived
broker on the same private Docker network injects host-held ChatGPT
authorization upstream and gives the task only a bounded attempt capability.
Authoritative markers and event evidence live in a host-only control directory,
not in Harbor's model-writable agent log mount.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

from ..clock import now
from .codex_exec_events import (
    CodexEventError,
    CodexJsonlValidator,
    CodexReportedError,
    CodexToolBudgetExceeded,
    audit_codex_0141_jsonl,
)
from .terminal_control import (
    CommandEnvelope,
    CommandStartedMarker,
    ControlRecordExists,
    ControlStoreError,
    EvaluationRegistration,
    ModelStartedMarker,
    RegisteredAttempt,
    SecureDirectory,
    SmokeSeal,
    command_digest,
    evaluation_lock,
    initialize_control_store,
    open_attempt_store,
    open_control_store,
    terminal_attempt_id,
    terminal_control_root,  # noqa: F401 - kept as the adapter's public helper
    write_command_started,
    write_model_started,
)
from .terminal_proxy import (
    BROKER_MAX_BUFFERED_RESPONSE_BYTES,
    BROKER_MAX_OBSERVED_CONTENT_TYPE_CHARS,
    BROKER_MAX_OBSERVED_CONTENT_TYPES,
    BROKER_MAX_REQUESTS,
    BROKER_RECOVERABLE_STREAM_ERRORS,
    BROKER_RECOVERABLE_TRANSPORT_ERRORS,
    BROKER_REJECTION_REASONS,
    BROKER_STREAM_RETRY_LIMIT,
    BROKER_STREAM_RETRY_LIMIT_PER_REQUEST,
    CAPABILITY_ENV,
    BrokerSecrets,
    TerminalProxyController,
    TerminalProxyError,
    TerminalProxyHandle,
    content_types_are_sse,
)

DATASET = "terminal-bench/terminal-bench-2-1"
AGENT_IMPORT_PATH = "lha.bench.terminal_bench:LhaAgent"
HARBOR_VERSION = "0.20.0"

_CODEX_UPLOAD = "/tmp/.lha_codex_binary.upload"
_RUNTIME_STAGING_DIR = "/tmp/.lha_runtime_staging"
_CAPABILITY_STAGING = f"{_RUNTIME_STAGING_DIR}/capability.upload"
_CAPABILITY_UPLOAD = "/tmp/.lha_terminal_proxy_capability"
_TLS_CERT_STAGING = f"{_RUNTIME_STAGING_DIR}/ca.upload"
_TLS_CERT_PATH = "/tmp/.lha_terminal_proxy_ca.pem"
_CODEX_HOME = "/tmp/lha_codex_runtime"
_CODEX_STDERR_PATH = "/tmp/lha_codex_stderr.txt"
_CODEX_RUN_UID = 60000
_CODEX_RUN_USER = f"{_CODEX_RUN_UID}:{_CODEX_RUN_UID}"
_CODEX_PRIVILEGED_SHELL = "/usr/local/lib/lha/bash"
_CODEX_SUID_PROBE = "/usr/local/lib/lha/.suid-probe"
_AGENT_PROVENANCE = "terminal_bench_agent.json"
_CODEX_EVENTS = "codex_events.jsonl"
_CODEX_STDERR = "codex_stderr.txt"
_MODEL_STARTED = "MODEL_STARTED.json"
_TERMINAL_RECORD = "terminal.json"
_BROKER_RECEIPT = "broker_receipt.json"
_COMMAND_ENVELOPE = "command.json"
_COMMAND_STARTED = "COMMAND_STARTED.json"
_SMOKE_MANIFEST = "smoke_manifest.json"
_SMOKE_SEAL = "smoke_seal.json"
_LEGACY_MAX_JSONL_LINE_BYTES = 60 * 1024
_MAX_JSONL_LINE_BYTES = 2 * 1024 * 1024
_HARBOR_STREAM_LINE_HEADROOM_BYTES = 64 * 1024
_HOST_PROCESS_TERM_GRACE_S = 0.25
_HOST_PROCESS_KILL_GRACE_S = 5.0
_BROKER_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "type",
        "evaluation_id",
        "attempt_id",
        "source_container_id",
        "started_at",
        "stopped_at",
        "ttl_s",
        "max_requests",
        "max_buffered_response_bytes",
        "request_retry_limit",
        "stream_retry_limit",
        "stream_retry_limit_per_request",
        "downstream_accepted_requests",
        "rejected_requests",
        "rejection_reasons",
        "upstream_attempts",
        "upstream_statuses",
        "stream_retries_used",
        "stream_retried_requests",
        "max_stream_retries_on_request",
        "upstream_error",
        "upstream_transport_errors",
        "upstream_stream_errors",
        "observed_content_types",
        "revoked",
        "outcome",
    }
)
_CODEX_VERSION_RE = re.compile(r"^codex-cli ([0-9A-Za-z][0-9A-Za-z.+-]*)$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_VALUE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_REPO_DIGEST_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_BROKER_ERROR_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")
_BROKER_FINAL_STREAM_REASONS = frozenset(
    {
        "upstream_invalid_sse",
        "upstream_non_bytes_chunk",
        "upstream_response_too_large",
        "upstream_secret_in_body",
        "upstream_stream_failure",
        "upstream_timeout",
    }
)
_T = TypeVar("_T")
_CODEX_0141_REASONING = {
    "gpt-5.5": frozenset({"low", "medium", "high", "xhigh"}),
    "gpt-5.4": frozenset({"low", "medium", "high", "xhigh"}),
    "gpt-5.4-mini": frozenset({"low", "medium", "high", "xhigh"}),
    "gpt-5.3-codex": frozenset({"low", "medium", "high", "xhigh"}),
    "gpt-5.2": frozenset({"low", "medium", "high", "xhigh"}),
}
_CORPUS_RESOURCE = (
    Path(__file__).with_name("resources") / "terminal_bench_2_1_corpus.json"
)
_CORPUS_RESOURCE_SHA256 = (
    "d0cb0f28eea8d28ced31b9829d6975e08e5e8a1462649a545f96ff6c5595ea1e"
)


class CapabilityExposureError(CodexEventError):
    """The bounded broker capability appeared in model-controlled output."""


class _BoundedSecretStreamDetector:
    """Detect one secret across chunks without retaining the observed stream."""

    def __init__(self, secret: str, *, max_total_bytes: int) -> None:
        if (
            not 32 <= len(secret) <= 256
            or not secret.isascii()
            or any(character.isspace() for character in secret)
            or max_total_bytes <= 0
        ):
            raise ValueError("secret stream detector received an unsafe contract")
        self._secret = secret
        self._max_total_bytes = max_total_bytes
        self._observed_bytes = 0
        self._raw_tail = ""
        self._line_joined_tail = ""

    @staticmethod
    def _join_display_lines(text: str) -> str:
        # A capability printed in two lines is still exposed. JSON strings use
        # backslash escapes while streamed process output uses real line breaks.
        return (
            text.replace("\\r", "")
            .replace("\\n", "")
            .replace("\r", "")
            .replace("\n", "")
        )

    def feed(self, text: str) -> bool:
        """Scan one callback fragment and retain at most ``len(secret) - 1`` chars."""
        if not isinstance(text, str):
            raise CodexEventError("Harbor output callback returned a non-text fragment")
        encoded_size = len(text.encode("utf-8"))
        if self._observed_bytes + encoded_size > self._max_total_bytes:
            raise CodexEventError("Codex streamed output exceeded the registered byte limit")
        self._observed_bytes += encoded_size

        raw_candidate = self._raw_tail + text
        joined_candidate = self._line_joined_tail + self._join_display_lines(text)
        exposed = (
            self._secret in raw_candidate
            or self._secret in joined_candidate
        )
        tail_length = len(self._secret) - 1
        self._raw_tail = raw_candidate[-tail_length:]
        self._line_joined_tail = joined_candidate[-tail_length:]
        return exposed

    def contains_complete(self, text: str) -> bool:
        """Check one bounded final buffer independently of callback behavior."""
        if not isinstance(text, str):
            raise CodexEventError("Harbor returned non-text process output")
        if len(text.encode("utf-8")) > self._max_total_bytes:
            raise CodexEventError("Codex final output exceeded the registered byte limit")
        return self._secret in text or self._secret in self._join_display_lines(text)

    def clear(self) -> None:
        """Drop the secret and partial suffixes before durable evidence is written."""
        self._secret = ""
        self._raw_tail = ""
        self._line_joined_tail = ""
        self._observed_bytes = 0


class TerminalBenchBudgets(BaseModel):
    """Separate model, broker, and host-command limits for one Harbor trial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    codex_timeout_s: Literal[1800] = 1800
    host_command_timeout_s: Literal[12000] = 12000
    max_frozen_verifier_timeout_s: Literal[7200] = 7200
    # Codex may emit several parallel tool items in one model response.  This is
    # a runaway-output ceiling, not an LHA step budget.
    max_tool_calls: Literal[128] = 128
    max_model_requests: Literal[60] = 60
    request_max_retries: Literal[1] = 1
    stream_max_retries: Literal[0] = 0
    broker_stream_max_retries: Literal[12] = 12
    broker_stream_max_retries_per_request: Literal[4] = 4
    # Schema 13 used 60 KiB to stay below asyncio's default line reader. A real
    # Codex 0.141 event exceeded that transport limit. Schema 14 raises the
    # protocol boundary and explicitly configures Harbor's pinned stream reader.
    max_jsonl_line_bytes: Literal[61440, 2097152] = _MAX_JSONL_LINE_BYTES
    max_jsonl_bytes: Literal[16777216] = 16777216
    broker_ttl_s: Literal[2100] = 2100
    codex_exec_runs: Literal[1] = 1
    scored_runs_per_task: Literal[1] = 1
    # Harbor has already created the task container before this agent is
    # installed.  The host cannot prove that a setup failure happened before
    # container start, so formal trials never retry.
    infrastructure_retries: Literal[0] = 0
    task_retries: Literal[0] = 0
    harbor_agent_timeout_multiplier: Literal[4] = 4


class TerminalBenchCorpusTask(BaseModel):
    """One task resolved from the official immutable dataset snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str
    image_manifest_media_type: Literal[
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ]
    task_checksum: str
    task_content_digest: str
    task_image_digest: str
    task_toml_sha256: str
    environment_build_timeout_s: int = Field(gt=0)
    agent_timeout_s: int = Field(gt=0)
    verifier_timeout_s: int = Field(gt=0)

    @field_validator("image")
    @classmethod
    def _image_name_is_present(cls, value: str) -> str:
        if not value.strip() or any(character.isspace() for character in value):
            raise ValueError("official task image names must be non-empty")
        return value

    @field_validator("task_checksum", "task_toml_sha256")
    @classmethod
    def _task_file_digest_is_sha256(cls, value: str) -> str:
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("official task file digests must be SHA-256 hex")
        return value

    @field_validator("task_content_digest", "task_image_digest")
    @classmethod
    def _task_registry_digest_is_sha256(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("official task registry digests must be sha256 values")
        return value


class TerminalBenchCorpusManifest(BaseModel):
    """The complete 89-task Harbor resolution used for subset selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    dataset_version: str
    harbor_version: Literal["0.20.0"] = HARBOR_VERSION
    source_inputs_sha256: str
    resolution_failures: tuple[()] = ()
    tasks: dict[str, TerminalBenchCorpusTask]

    @field_validator("dataset_version")
    @classmethod
    def _dataset_digest_is_pinned(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("official dataset version must be a sha256 digest")
        return value

    @field_validator("source_inputs_sha256")
    @classmethod
    def _source_digest_is_sha256(cls, value: str) -> str:
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("official source input digest must be SHA-256 hex")
        return value

    @model_validator(mode="after")
    def _complete_corpus_is_present(self) -> "TerminalBenchCorpusManifest":
        if len(self.tasks) != 89:
            raise ValueError("Terminal-Bench 2.1 corpus must contain exactly 89 tasks")
        if any(
            not task_id.startswith("terminal-bench/")
            or task_id != task_id.strip()
            for task_id in self.tasks
        ):
            raise ValueError("official task IDs are malformed")
        return self


def load_terminal_bench_corpus() -> TerminalBenchCorpusManifest:
    """Load and authenticate the corpus resource shipped in this exact wheel."""
    try:
        payload = _CORPUS_RESOURCE.read_bytes()
    except OSError as exc:
        raise RuntimeError("the packaged Terminal-Bench corpus is unavailable") from exc
    if hashlib.sha256(payload).hexdigest() != _CORPUS_RESOURCE_SHA256:
        raise RuntimeError("the packaged Terminal-Bench corpus digest changed")
    try:
        return TerminalBenchCorpusManifest.model_validate_json(payload)
    except ValueError as exc:
        raise RuntimeError("the packaged Terminal-Bench corpus is invalid") from exc


class RegisteredSubset(BaseModel):
    """The deterministic task order and the two disjoint preregistered sets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["sha256(instance_id),hex-ascending"] = "sha256(instance_id),hex-ascending"
    corpus_size: int = Field(ge=23)
    corpus_digest: str
    scored_instance_ids: tuple[str, ...]
    smoke_instance_ids: tuple[str, ...]

    @field_validator("corpus_digest")
    @classmethod
    def _digest_is_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("corpus_digest must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("scored_instance_ids")
    @classmethod
    def _twenty_scored(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 20:
            raise ValueError("the scored subset must contain exactly 20 instances")
        return value

    @field_validator("smoke_instance_ids")
    @classmethod
    def _three_smoke(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 3:
            raise ValueError("the smoke subset must contain exactly 3 instances")
        return value

    @model_validator(mode="after")
    def _ids_are_unique_and_disjoint(self) -> "RegisteredSubset":
        scored = self.scored_instance_ids
        smoke = self.smoke_instance_ids
        if any(not instance_id.strip() for instance_id in (*scored, *smoke)):
            raise ValueError("registered instance ids may not be empty")
        if len(scored) != len(set(scored)):
            raise ValueError("scored instance ids must be unique")
        if len(smoke) != len(set(smoke)):
            raise ValueError("smoke instance ids must be unique")
        if set(scored) & set(smoke):
            raise ValueError("scored and smoke instance ids must be disjoint")
        return self


class TerminalBenchProtocol(BaseModel):
    """Public, secret-free provenance for one fixed-subset evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[13, 14] = 14
    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    evaluation_id: str
    output_root: str
    dataset_version: str
    corpus_manifest_sha256: Literal[
        "d0cb0f28eea8d28ced31b9829d6975e08e5e8a1462649a545f96ff6c5595ea1e"
    ] = _CORPUS_RESOURCE_SHA256
    corpus_instance_ids: tuple[str, ...]
    subset: RegisteredSubset
    model: str
    reasoning_effort: str
    harbor_version: str
    codex_cli_version: str
    codex_target: Literal["x86_64-unknown-linux-musl"]
    codex_binary_sha256: str
    broker_image_id: str
    task_content_digests: dict[str, str]
    task_checksums: dict[str, str]
    task_image_digests: dict[str, str]
    task_agent_timeout_s: dict[str, int]
    task_verifier_timeout_s: dict[str, int]
    task_environment_build_timeout_s: dict[str, int]
    wheel_sha256: str
    budgets: TerminalBenchBudgets = Field(default_factory=TerminalBenchBudgets)

    @field_validator("evaluation_id")
    @classmethod
    def _evaluation_id_is_random_hex(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{32}", value) is None:
            raise ValueError("evaluation_id must be 128-bit lowercase hex")
        return value

    @field_validator("output_root")
    @classmethod
    def _output_root_is_absolute_and_normalized(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError("output_root must be an absolute normalized path")
        return str(path)

    @field_validator("corpus_instance_ids")
    @classmethod
    def _corpus_ids_are_frozen(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) < 23 or len(value) != len(set(value)):
            raise ValueError("the frozen corpus must contain at least 23 unique tasks")
        if any(not item.strip() or item != item.strip() for item in value):
            raise ValueError("corpus instance IDs must be non-empty and normalized")
        return tuple(sorted(value))

    @field_validator("model", "reasoning_effort", "harbor_version")
    @classmethod
    def _must_be_pinned(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("protocol provenance fields may not be empty")
        return value.strip()

    @field_validator("codex_cli_version")
    @classmethod
    def _codex_version_is_exact(cls, value: str) -> str:
        value = value.strip()
        if _CODEX_VERSION_RE.fullmatch(value) is None:
            raise ValueError("codex_cli_version must be exact, for example 'codex-cli 0.141.0'")
        return value

    @field_validator("dataset_version")
    @classmethod
    def _dataset_version_is_immutable(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("dataset_version must be sha256:<64 lowercase hex>")
        return value

    @field_validator("task_content_digests", "task_image_digests")
    @classmethod
    def _task_digests_are_pinned(cls, value: dict[str, str]) -> dict[str, str]:
        checked: dict[str, str] = {}
        for instance_id, digest_value in value.items():
            if not instance_id.strip():
                raise ValueError("task digest keys may not be empty")
            if _SHA256_VALUE_RE.fullmatch(digest_value) is None:
                raise ValueError("task digests must be sha256:<64 lowercase hex>")
            checked[instance_id] = digest_value
        return dict(sorted(checked.items()))

    @field_validator("wheel_sha256", "codex_binary_sha256")
    @classmethod
    def _file_digest_is_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("file digests must be lowercase SHA-256 hex")
        return value

    @field_validator("broker_image_id")
    @classmethod
    def _broker_image_is_pinned(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("broker_image_id must be sha256:<64 lowercase hex>")
        return value

    @field_validator("task_checksums")
    @classmethod
    def _task_checksums_are_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        if any(_SHA256_HEX_RE.fullmatch(item) is None for item in value.values()):
            raise ValueError("task checksums must be lowercase SHA-256 hex")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def _digests_cover_the_registered_tasks(self) -> "TerminalBenchProtocol":
        corpus = load_terminal_bench_corpus()
        expected_line_limit = (
            _LEGACY_MAX_JSONL_LINE_BYTES
            if self.schema_version == 13
            else _MAX_JSONL_LINE_BYTES
        )
        if self.budgets.max_jsonl_line_bytes != expected_line_limit:
            raise ValueError(
                "Terminal-Bench protocol schema does not match its JSONL line limit"
            )
        if self.harbor_version != HARBOR_VERSION:
            raise ValueError(f"formal runs require Harbor {HARBOR_VERSION}")
        if (
            self.dataset_version != corpus.dataset_version
            or self.corpus_manifest_sha256 != _CORPUS_RESOURCE_SHA256
            or self.corpus_instance_ids != tuple(sorted(corpus.tasks))
        ):
            raise ValueError(
                "formal runs must use the packaged complete official corpus"
            )
        if self.codex_cli_version != "codex-cli 0.141.0":
            raise ValueError("the strict event parser is pinned to Codex CLI 0.141.0")
        efforts = _CODEX_0141_REASONING.get(self.model)
        if efforts is None or self.reasoning_effort not in efforts:
            raise ValueError(
                "the model and reasoning effort must exist in Codex 0.141's "
                "bundled model catalog"
            )
        recomputed = preregister_instances(self.corpus_instance_ids)
        if recomputed != self.subset:
            raise ValueError(
                "registered subsets must be recomputed from the frozen full corpus"
            )
        expected = set(self.subset.scored_instance_ids) | set(
            self.subset.smoke_instance_ids
        )
        if (
            set(self.task_content_digests) != expected
            or set(self.task_checksums) != expected
            or set(self.task_image_digests) != expected
            or set(self.task_agent_timeout_s) != expected
            or set(self.task_verifier_timeout_s) != expected
            or set(self.task_environment_build_timeout_s) != expected
        ):
            raise ValueError(
                "task digests must contain exactly the 20 scored and 3 smoke tasks"
            )
        for instance_id in expected:
            task = corpus.tasks[instance_id]
            if (
                self.task_content_digests[instance_id]
                != task.task_content_digest
                or self.task_checksums[instance_id] != task.task_checksum
                or self.task_image_digests[instance_id] != task.task_image_digest
                or self.task_agent_timeout_s[instance_id] != task.agent_timeout_s
                or self.task_verifier_timeout_s[instance_id]
                != task.verifier_timeout_s
                or self.task_environment_build_timeout_s[instance_id]
                != task.environment_build_timeout_s
            ):
                raise ValueError(
                    "selected task evidence differs from the official corpus"
                )
        if (
            max(self.task_verifier_timeout_s.values())
            != self.budgets.max_frozen_verifier_timeout_s
        ):
            raise ValueError("host timeout budget does not cover the selected verifier set")
        required_host_envelope = (
            max(self.task_environment_build_timeout_s.values())
            + 360
            + self.budgets.codex_timeout_s
            + self.budgets.max_frozen_verifier_timeout_s
            + 600
        )
        if self.budgets.host_command_timeout_s < required_host_envelope:
            raise ValueError(
                "host command timeout is shorter than the frozen trial stages"
            )
        return self


class CodexRunAudit(BaseModel):
    """Validated public JSONL summary for one tool-enabled Codex run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_counts: dict[str, int]
    item_counts: dict[str, int]
    tool_calls: int = Field(ge=0)
    reconnect_notices: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_output_tokens: int = Field(ge=0)


class DockerImageAttestation(BaseModel):
    """Image identity read from the live Harbor Docker container."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    container_id: str
    image_id: str
    configured_image: str
    repo_digests: tuple[str, ...]
    compose_project: str
    network_name: str
    container_ip: str

    @field_validator("container_id")
    @classmethod
    def _container_id_is_exact(cls, value: str) -> str:
        if _DOCKER_CONTAINER_ID_RE.fullmatch(value) is None:
            raise ValueError("container_id must be a full lowercase Docker container ID")
        return value

    @field_validator("image_id")
    @classmethod
    def _image_id_is_sha256(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("image_id must be a full lowercase Docker image ID")
        return value

    @field_validator("configured_image")
    @classmethod
    def _configured_image_is_present(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("configured_image may not be empty")
        return value

    @field_validator("compose_project", "network_name", "container_ip")
    @classmethod
    def _runtime_binding_is_present(cls, value: str) -> str:
        value = value.strip()
        if not value or any(character.isspace() for character in value):
            raise ValueError("Docker runtime bindings must be non-empty and contain no whitespace")
        return value

    @field_validator("repo_digests")
    @classmethod
    def _repo_digests_are_immutable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("the inspected image must expose at least one RepoDigest")
        if any(_DOCKER_REPO_DIGEST_RE.fullmatch(item) is None for item in value):
            raise ValueError("RepoDigests must use name@sha256:<64 lowercase hex>")
        if len(value) != len(set(value)):
            raise ValueError("RepoDigests may not contain duplicates")
        return tuple(sorted(value))

    def proves(self, expected_digest: str) -> bool:
        """Return whether runtime inspection binds this container to the digest."""
        pinned_suffix = f"@{expected_digest}"
        return self.configured_image.endswith(pinned_suffix) or any(
            item.endswith(pinned_suffix) for item in self.repo_digests
        )


class TerminalBenchAgentProvenance(BaseModel):
    """Authoritative, secret-free terminal evidence stored outside Harbor mounts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[6] = 6
    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    evaluation_id: str
    attempt_id: str
    dataset_version: str
    agent_import_path: Literal["lha.bench.terminal_bench:LhaAgent"] = AGENT_IMPORT_PATH
    lha_version: str
    run_kind: Literal["smoke", "scored"]
    instance_id: str
    model: str
    reasoning_effort: str
    harbor_version: str
    codex_cli_version: str
    observed_codex_cli_version: str | None = None
    codex_target: Literal["x86_64-unknown-linux-musl"]
    observed_codex_target: Literal["x86_64-unknown-linux-musl"] | None = None
    codex_binary_sha256: str
    observed_codex_binary_sha256: str | None = None
    broker_image_id: str
    task_content_digest: str
    task_image_digest: str
    image_attestation: DockerImageAttestation | None = None
    post_quiescence_attestation: DockerImageAttestation | None = None
    wheel_sha256: str
    protocol_sha256: str
    subset: RegisteredSubset
    budgets: TerminalBenchBudgets
    model_started: bool
    infrastructure_retries_used: Literal[0] = 0
    codex_outcome: Literal[
        "setup_error",
        "success",
        "process_error",
        "protocol_error",
        "execution_error",
    ]
    codex_return_code: int | None = None
    codex_failure_kind: Literal[
        "codex_nonzero_exit",
        "codex_reported_error",
        "codex_jsonl_invalid",
        "codex_tool_budget_exceeded",
        "codex_capability_exposed",
        "agent_setup_failed",
        "broker_start_failed",
        "codex_execution_exception",
        "codex_cancelled",
        "codex_runtime_cleanup_failed",
        "broker_cleanup_failed",
        "container_quiescence_failed",
    ] | None = None
    broker_cleanup_state: Literal[
        "not_started",
        "succeeded",
        "failed",
    ]
    container_quiescence: Literal["not_started", "restarted", "stopped", "failed"]
    smoke_seal_sha256: str | None = None
    codex_events_sha256: str | None = None
    codex_stderr_sha256: str | None = None
    broker_receipt_sha256: str | None = None
    broker_tls_certificate_sha256: str | None = None
    broker_accepted_requests: int | None = Field(default=None, ge=0, le=60)
    broker_revoked: bool | None = None
    codex_audit: CodexRunAudit | None = None

    @field_validator("lha_version", "model", "reasoning_effort", "harbor_version")
    @classmethod
    def _nonempty_provenance_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("provenance values may not be empty")
        return value

    @field_validator(
        "codex_binary_sha256",
        "wheel_sha256",
        "protocol_sha256",
        "observed_codex_binary_sha256",
        "smoke_seal_sha256",
        "codex_events_sha256",
        "codex_stderr_sha256",
        "broker_receipt_sha256",
        "broker_tls_certificate_sha256",
    )
    @classmethod
    def _provenance_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("provenance file digests must be lowercase SHA-256 hex")
        return value

    @field_validator("dataset_version", "task_content_digest", "task_image_digest")
    @classmethod
    def _provenance_pinned_digest(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("pinned digests must be sha256:<64 lowercase hex>")
        return value

    @field_validator("evaluation_id")
    @classmethod
    def _provenance_evaluation_id(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{32}", value) is None:
            raise ValueError("evaluation_id must be 128-bit lowercase hex")
        return value

    @field_validator("attempt_id")
    @classmethod
    def _provenance_attempt_id(cls, value: str) -> str:
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("attempt_id must be lowercase SHA-256 hex")
        return value

    @field_validator("broker_image_id")
    @classmethod
    def _provenance_broker_image_id(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("broker image must be pinned by full image ID")
        return value

    @model_validator(mode="after")
    def _codex_terminal_evidence_is_consistent(
        self,
    ) -> "TerminalBenchAgentProvenance":
        expected_attempt = terminal_attempt_id(
            self.evaluation_id,
            self.run_kind,
            self.instance_id,
        )
        if self.attempt_id != expected_attempt:
            raise ValueError("attempt_id does not match the registered trial")
        if self.run_kind == "smoke" and self.smoke_seal_sha256 is not None:
            raise ValueError("smoke attempts may not depend on their own smoke seal")
        if self.run_kind == "scored" and self.smoke_seal_sha256 is None:
            raise ValueError("scored attempts require the preregistered smoke seal")

        if self.codex_outcome == "setup_error":
            valid = (
                not self.model_started
                and self.codex_return_code is None
                and self.codex_failure_kind == "agent_setup_failed"
                and self.codex_events_sha256 is None
                and self.codex_audit is None
                and self.broker_cleanup_state == "not_started"
                and self.container_quiescence == "not_started"
            )
        elif self.codex_outcome == "success":
            valid = (
                self.model_started
                and self.codex_return_code == 0
                and self.codex_failure_kind is None
                and self.codex_events_sha256 is not None
                and self.codex_audit is not None
                and self.broker_cleanup_state == "succeeded"
                and self.broker_tls_certificate_sha256 is not None
                and self.container_quiescence == "restarted"
                and self.post_quiescence_attestation is not None
            )
        elif self.codex_outcome == "process_error":
            valid = (
                self.model_started
                and self.codex_return_code is not None
                and self.codex_return_code != 0
                and self.codex_failure_kind == "codex_nonzero_exit"
                and self.codex_events_sha256 is not None
                and self.codex_audit is None
                and self.container_quiescence in {"stopped", "failed"}
            )
        elif self.codex_outcome == "protocol_error":
            valid = (
                self.model_started
                and self.codex_failure_kind
                in {
                    "codex_reported_error",
                    "codex_jsonl_invalid",
                    "codex_tool_budget_exceeded",
                    "codex_capability_exposed",
                }
                and self.codex_events_sha256 is not None
                and self.codex_audit is None
                and self.container_quiescence in {"stopped", "failed"}
            )
        else:
            valid = (
                self.model_started
                and self.codex_failure_kind
                in {
                    "broker_start_failed",
                    "codex_execution_exception",
                    "codex_cancelled",
                    "codex_runtime_cleanup_failed",
                    "broker_cleanup_failed",
                    "container_quiescence_failed",
                }
                and self.codex_audit is None
                and self.container_quiescence in {"stopped", "failed"}
            )
        if not valid:
            raise ValueError("Codex outcome and terminal evidence are inconsistent")
        if self.model_started:
            if (
                self.image_attestation is None
                or self.observed_codex_cli_version is None
                or self.observed_codex_target is None
                or self.observed_codex_binary_sha256 is None
            ):
                raise ValueError("Codex execution evidence requires completed setup")
        if self.observed_codex_cli_version is not None:
            if self.observed_codex_cli_version != self.codex_cli_version:
                raise ValueError("observed Codex CLI version changed after setup")
        if (
            self.observed_codex_binary_sha256 is not None
            and self.observed_codex_binary_sha256 != self.codex_binary_sha256
        ):
            raise ValueError("observed Codex binary digest changed after setup")

        receipt_values = (
            self.broker_receipt_sha256,
            self.broker_accepted_requests,
            self.broker_revoked,
        )
        if self.broker_cleanup_state == "succeeded":
            if (
                any(value is None for value in receipt_values)
                or self.broker_revoked is not True
                or self.broker_tls_certificate_sha256 is None
            ):
                raise ValueError("successful broker cleanup requires a revoked receipt")
            if (
                self.codex_outcome == "success"
                and self.codex_audit is not None
                and self.broker_accepted_requests is not None
                and not (
                    1
                    <= self.broker_accepted_requests
                    <= min(
                        self.budgets.max_model_requests,
                        (self.codex_audit.tool_calls + 1)
                        * (self.budgets.request_max_retries + 1),
                    )
                )
            ):
                raise ValueError(
                    "successful Codex evidence has inconsistent broker request counts"
                )
        elif any(value is not None for value in receipt_values):
            raise ValueError("unverified broker cleanup may not claim a receipt")

        if self.post_quiescence_attestation is not None:
            if self.image_attestation is None:
                raise ValueError("post-quiescence evidence requires initial image evidence")
            before = self.image_attestation
            after = self.post_quiescence_attestation
            if before != after:
                raise ValueError("container identity changed during success quiescence")
        return self


class HarborRunCommand(BaseModel):
    """One exact one-instance Harbor invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: str
    attempt_id: str
    run_kind: Literal["smoke", "scored"]
    instance_id: str
    task_content_digest: str
    task_checksum: str
    task_image_digest: str
    argv: tuple[str, ...]
    command_sha256: str
    job_dir: str

    @model_validator(mode="after")
    def _command_binding_is_exact(self) -> "HarborRunCommand":
        if re.fullmatch(r"[0-9a-f]{32}", self.evaluation_id) is None:
            raise ValueError("Harbor command evaluation_id is invalid")
        if _SHA256_HEX_RE.fullmatch(self.attempt_id) is None:
            raise ValueError("Harbor command attempt_id is invalid")
        if self.attempt_id != terminal_attempt_id(
            self.evaluation_id,
            self.run_kind,
            self.instance_id,
        ):
            raise ValueError("Harbor command attempt_id does not match its task")
        if self.command_sha256 != command_digest(self.argv):
            raise ValueError("Harbor command digest does not match argv")
        return self


class HarborExecutionManifest(BaseModel):
    """Post-run evidence accounting for every registered command, including errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[5] = 5
    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    dataset_version: str
    run_kind: Literal["smoke", "scored"]
    protocol_sha256: str
    expected_instance_ids: tuple[str, ...]
    observed_instance_ids: tuple[str, ...]
    task_content_digests: dict[str, str]
    task_checksums: dict[str, str]
    task_image_digests: dict[str, str]
    codex_events_sha256: dict[str, str | None]
    container_image_ids: dict[str, str | None]
    command_started_sha256: dict[str, str | None]
    command_envelope_sha256: dict[str, str | None]
    terminal_record_sha256: dict[str, str | None]
    job_config_sha256: dict[str, str | None]
    job_lock_sha256: dict[str, str | None]
    job_result_sha256: dict[str, str | None]
    trial_result_sha256: dict[str, str | None]
    official_status: dict[str, Literal["PASS", "FAIL", "ERROR"]]
    protocol_errors: dict[str, str | None]
    job_dirs: tuple[str, ...]

    @field_validator("protocol_sha256")
    @classmethod
    def _protocol_digest_is_sha256(cls, value: str) -> str:
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("manifest protocol_sha256 must be lowercase SHA-256 hex")
        return value

    @field_validator("dataset_version")
    @classmethod
    def _manifest_dataset_version_is_immutable(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("manifest dataset_version must be sha256:<64 lowercase hex>")
        return value

    @field_validator("task_content_digests", "task_image_digests")
    @classmethod
    def _manifest_task_digests_are_immutable(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        if any(_SHA256_VALUE_RE.fullmatch(item) is None for item in value.values()):
            raise ValueError("manifest task digests must be sha256:<64 lowercase hex>")
        return dict(sorted(value.items()))

    @field_validator("task_checksums")
    @classmethod
    def _manifest_task_checksums_are_sha256(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        if any(_SHA256_HEX_RE.fullmatch(item) is None for item in value.values()):
            raise ValueError("manifest task checksums must be lowercase SHA-256 hex")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def _evidence_covers_exact_execution(self) -> "HarborExecutionManifest":
        expected = self.expected_instance_ids
        if len(expected) != len(set(expected)):
            raise ValueError("manifest expected instance ids must be unique")
        if self.observed_instance_ids != expected:
            raise ValueError("manifest observed instance order must match the expected order")
        expected_set = set(expected)
        evidence_maps = (
            self.task_content_digests,
            self.task_checksums,
            self.task_image_digests,
            self.codex_events_sha256,
            self.container_image_ids,
            self.command_started_sha256,
            self.command_envelope_sha256,
            self.terminal_record_sha256,
            self.job_config_sha256,
            self.job_lock_sha256,
            self.job_result_sha256,
            self.trial_result_sha256,
            self.official_status,
            self.protocol_errors,
        )
        if any(set(mapping) != expected_set for mapping in evidence_maps):
            raise ValueError("manifest evidence must cover exactly the executed instances")
        if any(
            digest is not None and _SHA256_HEX_RE.fullmatch(digest) is None
            for digest in self.codex_events_sha256.values()
        ) or any(
            digest is not None and _SHA256_HEX_RE.fullmatch(digest) is None
            for digest in self.terminal_record_sha256.values()
        ) or any(
            digest is not None and _SHA256_HEX_RE.fullmatch(digest) is None
            for digest in self.trial_result_sha256.values()
        ) or any(
            digest is not None and _SHA256_HEX_RE.fullmatch(digest) is None
            for mapping in (
                self.command_started_sha256,
                self.command_envelope_sha256,
                self.job_config_sha256,
                self.job_lock_sha256,
                self.job_result_sha256,
            )
            for digest in mapping.values()
        ):
            raise ValueError("manifest file digests must be lowercase SHA-256 hex")
        if any(
            image_id is not None and _SHA256_VALUE_RE.fullmatch(image_id) is None
            for image_id in self.container_image_ids.values()
        ):
            raise ValueError("manifest container image IDs must be sha256 digests")
        if len(self.job_dirs) != len(expected) or len(self.job_dirs) != len(
            set(self.job_dirs)
        ):
            raise ValueError("manifest must bind one unique job directory per instance")
        for instance_id in expected:
            status = self.official_status[instance_id]
            detail = self.protocol_errors[instance_id]
            if status == "ERROR" and (detail is None or not detail.strip()):
                raise ValueError("manifest ERROR rows require a stable explanation")
            if status != "ERROR" and detail is not None:
                raise ValueError("manifest PASS and FAIL rows may not claim protocol errors")
            if status != "ERROR" and self.trial_result_sha256[instance_id] is None:
                raise ValueError("manifest PASS and FAIL rows require an official result")
            if status != "ERROR" and any(
                mapping[instance_id] is None
                for mapping in (
                    self.command_started_sha256,
                    self.command_envelope_sha256,
                    self.terminal_record_sha256,
                    self.job_config_sha256,
                    self.job_lock_sha256,
                    self.job_result_sha256,
                )
            ):
                raise ValueError("manifest PASS and FAIL rows require complete evidence")
        return self


class TerminalBenchTaskRecord(BaseModel):
    """One scored-task result derived from an official Harbor trial.

    ``independent_correct`` is truth from Harbor's verifier reward.  The current
    Harbor agent runs one ``codex exec`` directly and never enters LHA's gate or
    repair loop, so those mechanism fields are explicitly unavailable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command_envelope_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    official_result_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    official_status: Literal["PASS", "FAIL", "ERROR"]
    independent_correct: bool | None = None
    gate_accepted: None = None
    repairs: None = None
    duration_s: FiniteFloat | None = Field(default=None, ge=0)
    protocol_error: str | None = None
    infrastructure_retries: Literal[0] | None = None
    task_runs: Literal[1] = 1

    @field_validator("independent_correct", mode="before")
    @classmethod
    def _optional_flags_are_strict_booleans(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, bool):
            raise ValueError("task record truth must be a boolean or null")
        return value

    @model_validator(mode="after")
    def _official_fields_are_consistent(self) -> "TerminalBenchTaskRecord":
        expected_correct = {"PASS": True, "FAIL": False, "ERROR": None}[
            self.official_status
        ]
        if self.independent_correct is not expected_correct:
            raise ValueError(
                "independent_correct must agree with the official Harbor status"
            )
        if self.official_status == "ERROR":
            if self.protocol_error is None or not self.protocol_error.strip():
                raise ValueError("ERROR records must explain the official protocol error")
        elif self.protocol_error is not None:
            raise ValueError("PASS and FAIL records may not carry protocol_error")
        elif self.official_result_sha256 is None:
            raise ValueError("PASS and FAIL records require an official Harbor result")
        elif self.command_envelope_sha256 is None:
            raise ValueError("PASS and FAIL records require a host command envelope")
        elif self.infrastructure_retries is None:
            raise ValueError("completed PASS and FAIL records must report retry usage")
        return self


class TerminalBenchRecordBatch(BaseModel):
    """Twenty records bound to one validated scored execution manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[TerminalBenchTaskRecord, ...]

    @model_validator(mode="after")
    def _records_share_one_binding(self) -> "TerminalBenchRecordBatch":
        if len(self.records) != 20:
            raise ValueError("a formal Terminal-Bench batch must contain exactly 20 records")
        instance_ids = [record.instance_id for record in self.records]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("formal Terminal-Bench records must have unique instance ids")
        if any(
            record.protocol_sha256 != self.protocol_sha256
            or record.execution_manifest_sha256 != self.execution_manifest_sha256
            for record in self.records
        ):
            raise ValueError("every task record must share the batch evidence binding")
        return self


class TerminalBenchSummary(BaseModel):
    """Official aggregate with unavailable LHA mechanism metrics kept as null."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    denominator: Literal[20] = 20
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    mechanism_metrics: Literal["unavailable"] = "unavailable"
    incorrect_deliveries: None = None
    intercepted_incorrect: None = None
    false_rejections: None = None
    repair_successes: None = None
    repair_attempts: None = None
    repair_success_rate: None = None
    p50_duration_s: float | None = Field(default=None, ge=0)
    p95_duration_s: float | None = Field(default=None, ge=0)
    protocol_errors: int = Field(ge=0)
    missing_instance_ids: tuple[str, ...] = ()

    def to_markdown(self) -> str:
        """Render without implying that a 20-task subset is a full leaderboard run."""
        return "\n".join(
            [
                "# Terminal-Bench 2.1 固定 20 题子集",
                "",
                f"- 通过：{self.passed}/20（{self.success_rate:.1%}）",
                f"- 失败：{self.failed}/20",
                f"- ERROR：{self.errors}/20（保留在分母中）",
                f"- P50 / P95 耗时（秒）：{self.p50_duration_s} / {self.p95_duration_s}",
                f"- 不可评分 ERROR：{self.protocol_errors}",
                "- 错误交付 / 拦截 / 错误拒绝：未测"
                "（当前 Harbor agent 未经过 LHA gate）",
                "- 修复成功率：不适用"
                "（当前协议只有一次 Codex 执行，没有 LHA repair 循环）",
                "",
                "该结果仅代表预注册的固定 20 题子集，不是完整排行榜成绩。",
            ]
        )


def preregister_instances(instance_ids: Sequence[str]) -> RegisteredSubset:
    """Hash-sort a frozen corpus, then select first 20 scored and next 3 smoke."""
    ids = [item.strip() for item in instance_ids]
    if any(not item for item in ids):
        raise ValueError("instance ids may not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("instance ids must be unique")
    if len(ids) < 23:
        raise ValueError("Terminal-Bench protocol needs at least 23 instance ids")
    ranked = sorted(ids, key=lambda item: (hashlib.sha256(item.encode()).hexdigest(), item))
    corpus_digest = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    return RegisteredSubset(
        corpus_size=len(ids),
        corpus_digest=corpus_digest,
        scored_instance_ids=tuple(ranked[:20]),
        smoke_instance_ids=tuple(ranked[20:23]),
    )


def sha256_file(path: str | Path) -> str:
    """Hash a wheel without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, payload: str) -> None:
    """Persist attempt evidence without exposing a partially written JSON file."""
    _durable_mkdir_chain(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_mkdir_chain(path: Path) -> None:
    """Create missing parents one at a time and persist every directory entry."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        if current.is_symlink():
            raise ValueError(f"directory path is a symbolic link: {current}")
        missing.append(current)
        if current.parent == current:
            raise ValueError(f"directory path has no existing ancestor: {path}")
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError(f"directory ancestor is unsafe: {current}")
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)


def _read_stable_regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one existing evidence file without following a replacement link."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("the existing Harbor execution manifest is unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError("the existing Harbor execution manifest is not a regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(
                    "the existing Harbor execution manifest ended before its stated size"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("the existing Harbor execution manifest grew while it was read")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("the existing Harbor execution manifest changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_execution_manifest_once(
    path: Path,
    manifest: HarborExecutionManifest,
) -> None:
    """Publish a canonical manifest once, or prove an identical one already exists."""
    payload = (manifest.model_dump_json(indent=2) + "\n").encode()
    _durable_mkdir_chain(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("manifest write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            existing = _read_stable_regular_bytes(
                path,
                maximum_bytes=max(len(payload), 1),
            )
            try:
                recorded = HarborExecutionManifest.model_validate_json(existing)
            except ValueError as exc:
                raise ValueError(
                    "the existing Harbor execution manifest is invalid"
                ) from exc
            if recorded != manifest or existing != payload:
                raise ValueError(
                    "the existing Harbor execution manifest conflicts with "
                    "current evidence"
                )
            return
        try:
            parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
        except OSError as exc:
            raise ValueError(
                "the Harbor execution manifest directory could not be synchronized"
            ) from exc
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise ValueError("the Harbor execution manifest could not be published") from exc
    finally:
        temporary.unlink(missing_ok=True)


async def _run_docker_inspect(*argv: str) -> Mapping[str, Any]:
    """Run one fixed-argument Docker inspection and return its sole object."""
    process = await asyncio.create_subprocess_exec(
        "docker",
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise RuntimeError("Docker inspection timed out") from exc
    except asyncio.CancelledError:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode != 0:
        detail = stderr.decode(errors="replace")[-400:]
        raise RuntimeError(f"Docker inspection failed with {process.returncode}: {detail}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Docker inspection returned invalid JSON") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], Mapping)
    ):
        raise RuntimeError("Docker inspection must return exactly one object")
    return payload[0]


async def _inspect_docker_container(container_id: str) -> DockerImageAttestation:
    """Read the live container and its image from Docker, not from the protocol."""
    container = await _run_docker_inspect("inspect", "--type", "container", container_id)
    actual_container_id = container.get("Id")
    image_id = container.get("Image")
    config = container.get("Config")
    configured_image = config.get("Image") if isinstance(config, Mapping) else None
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    state = container.get("State")
    host_config = container.get("HostConfig")
    network_settings = container.get("NetworkSettings")
    networks = (
        network_settings.get("Networks")
        if isinstance(network_settings, Mapping)
        else None
    )
    if (
        not isinstance(actual_container_id, str)
        or actual_container_id != container_id
        or not isinstance(image_id, str)
        or not isinstance(configured_image, str)
        or not isinstance(labels, Mapping)
        or labels.get("com.docker.compose.service") != "main"
        or not isinstance(state, Mapping)
        or state.get("Running") is not True
        or not isinstance(host_config, Mapping)
        or not isinstance(networks, Mapping)
        or len(networks) != 1
    ):
        raise RuntimeError("Docker container inspection omitted the required runtime binding")
    cap_add = host_config.get("CapAdd")
    security_options = host_config.get("SecurityOpt")
    if (
        host_config.get("Privileged") is True
        or host_config.get("PidMode") not in (None, "")
        or (
            isinstance(cap_add, list)
            and any(str(value).upper() in {"ALL", "SYS_PTRACE", "CAP_SYS_PTRACE"} for value in cap_add)
        )
        or (
            isinstance(security_options, list)
            and any("no-new-privileges" in str(value) for value in security_options)
        )
    ):
        raise RuntimeError(
            "the task container cannot enforce the registered process isolation"
        )
    compose_project = labels.get("com.docker.compose.project")
    network_name, network_binding = next(iter(networks.items()))
    container_ip = (
        network_binding.get("IPAddress")
        if isinstance(network_binding, Mapping)
        else None
    )
    if (
        not isinstance(compose_project, str)
        or not compose_project
        or not isinstance(network_name, str)
        or not network_name
        or not isinstance(container_ip, str)
        or not container_ip
    ):
        raise RuntimeError("Docker container inspection omitted compose network identity")

    network = await _run_docker_inspect("network", "inspect", network_name)
    network_labels = network.get("Labels")
    if (
        not isinstance(network_labels, Mapping)
        or network_labels.get("com.docker.compose.project") != compose_project
    ):
        raise RuntimeError("the task container is not attached to its Compose-owned network")

    image = await _run_docker_inspect("image", "inspect", image_id)
    inspected_image_id = image.get("Id")
    repo_digests = image.get("RepoDigests")
    if inspected_image_id != image_id:
        raise RuntimeError("Docker image inspection changed the container image ID")
    if not isinstance(repo_digests, list) or not all(
        isinstance(value, str) for value in repo_digests
    ):
        raise RuntimeError("Docker image inspection omitted RepoDigests")
    return DockerImageAttestation(
        container_id=actual_container_id,
        image_id=image_id,
        configured_image=configured_image,
        repo_digests=tuple(repo_digests),
        compose_project=compose_project,
        network_name=network_name,
        container_ip=container_ip,
    )


async def _attest_harbor_docker_image(environment) -> DockerImageAttestation:
    """Resolve Harbor's running ``main`` service to Docker's actual image."""
    compose = getattr(environment, "_run_docker_compose_command", None)
    if not callable(compose):
        raise RuntimeError(
            "formal Terminal-Bench runs require Harbor's auditable Docker environment"
        )
    compose_call = cast(Callable[[list[str]], Awaitable[Any]], compose)
    result = await compose_call(["ps", "--all", "--quiet", "main"])
    if getattr(result, "return_code", None) != 0:
        raise RuntimeError("Harbor could not resolve its running main container")
    container_ids = (getattr(result, "stdout", "") or "").splitlines()
    container_ids = [item.strip() for item in container_ids if item.strip()]
    if len(container_ids) != 1:
        raise RuntimeError("Harbor must expose exactly one running main container")
    container_id = container_ids[0]
    if _DOCKER_CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise RuntimeError("Harbor returned a non-canonical Docker container ID")
    return await _inspect_docker_container(container_id)


def _load_broker_secrets(
    path: Path,
    *,
    minimum_validity_s: int = 0,
) -> BrokerSecrets:
    """Read only the two ChatGPT fields needed by the host broker."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("the configured Codex authentication record is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not path.is_absolute()
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_mode & 0o077
            or info.st_size <= 0
            or info.st_size > 1024 * 1024
        ):
            raise RuntimeError("the configured Codex authentication record is unsafe")
        payload = bytearray()
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise RuntimeError("the configured Codex authentication record is truncated")
            payload.extend(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("the configured Codex authentication record is invalid") from exc
    finally:
        payload[:] = b"\0" * len(payload)
    tokens = raw.get("tokens") if isinstance(raw, Mapping) else None
    access_token = tokens.get("access_token") if isinstance(tokens, Mapping) else None
    account_id = tokens.get("account_id") if isinstance(tokens, Mapping) else None
    if not isinstance(access_token, str) or not isinstance(account_id, str):
        raise RuntimeError("the configured Codex authentication record lacks ChatGPT tokens")
    if minimum_validity_s:
        try:
            parts = access_token.split(".")
            if len(parts) != 3:
                raise ValueError
            encoded = parts[1] + "=" * (-len(parts[1]) % 4)
            claims = json.loads(base64.urlsafe_b64decode(encoded))
            expires_at = claims.get("exp") if isinstance(claims, Mapping) else None
        except (
            ValueError,
            UnicodeDecodeError,
            binascii.Error,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "the ChatGPT access token has no auditable expiry"
            ) from exc
        if (
            isinstance(expires_at, bool)
            or not isinstance(expires_at, (int, float))
            or not math.isfinite(float(expires_at))
            or float(expires_at) - time.time() < minimum_validity_s
        ):
            raise RuntimeError(
                "the ChatGPT access token lacks enough lifetime for one attempt"
            )
    return BrokerSecrets(access_token=access_token, account_id=account_id)


def _write_private_payload(payload: bytes, *, prefix: str) -> Path:
    """Create an owner-only transient file for Harbor's upload primitive."""
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix)
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("capability staging made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    return path


def _write_private_capability(value: str) -> Path:
    return _write_private_payload(
        value.encode(),
        prefix="lha-terminal-capability-",
    )


def _finalize_uploaded_file_command(
    *,
    staging_path: str,
    destination_path: str,
    owner: str,
    mode: int,
    expected_sha256: str,
) -> str:
    """Install one uploaded regular file and prove its exact container state."""
    if (
        Path(staging_path).parent.as_posix() != _RUNTIME_STAGING_DIR
        or not destination_path.startswith("/tmp/.lha_")
        or _SHA256_HEX_RE.fullmatch(expected_sha256) is None
        or re.fullmatch(r"[0-9]+:[0-9]+", owner) is None
        or mode not in {0o400, 0o600}
    ):
        raise ValueError("unsafe runtime-file installation contract")
    staging = shlex.quote(staging_path)
    destination = shlex.quote(destination_path)
    expected_state = shlex.quote(f"{owner}:{mode:o}:1:regular file")
    return (
        "set -eu; "
        f'[ "$(stat -c %u:%g:%a:%F {_RUNTIME_STAGING_DIR})" = '
        "'0:0:700:directory' ]; "
        f"test -f {staging}; test ! -L {staging}; "
        f'[ "$(stat -c %a:%h:%F {staging})" = '
        "'600:1:regular file' ]; "
        f"chown 0:0 {staging}; "
        f'[ "$(stat -c %u:%g:%a:%h:%F {staging})" = '
        "'0:0:600:1:regular file' ]; "
        f"rm -f {destination}; "
        f"install -o {owner.partition(':')[0]} -g {owner.partition(':')[2]} "
        f"-m {mode:o} {staging} {destination}; "
        f"rm -f {staging}; "
        f"test -f {destination}; test ! -L {destination}; "
        f'[ "$(stat -c %u:%g:%a:%h:%F {destination})" = {expected_state} ]; '
        f'[ "$(sha256sum {destination} | awk \'{{print $1}}\')" = '
        f"{expected_sha256} ]"
    )


def _control_registration(
    protocol: TerminalBenchProtocol,
    *,
    protocol_sha256: str,
) -> EvaluationRegistration:
    with open_control_store(protocol.output_root, protocol.evaluation_id) as store:
        registration = cast(
            EvaluationRegistration,
            store.read_json("registration.json", EvaluationRegistration),
        )
    if (
        registration.evaluation_id != protocol.evaluation_id
        or registration.protocol_sha256 != protocol_sha256
        or registration.output_root != protocol.output_root
    ):
        raise ControlStoreError("the host control registration does not match the protocol")
    expected = (
        *protocol.subset.smoke_instance_ids,
        *protocol.subset.scored_instance_ids,
    )
    if tuple(item.instance_id for item in registration.attempts) != expected:
        raise ControlStoreError("the host control registration changed the task order")
    return registration


def _registered_attempt(
    protocol: TerminalBenchProtocol,
    *,
    protocol_sha256: str,
    attempt_id: str,
    run_kind: Literal["smoke", "scored"],
    instance_id: str,
) -> RegisteredAttempt:
    registration = _control_registration(protocol, protocol_sha256=protocol_sha256)
    matches = [item for item in registration.attempts if item.attempt_id == attempt_id]
    if len(matches) != 1:
        raise ControlStoreError("the attempt is not present exactly once in registration")
    attempt = matches[0]
    if attempt.run_kind != run_kind or attempt.instance_id != instance_id:
        raise ControlStoreError("the attempt registration does not match the Harbor trial")
    return attempt


def _validated_smoke_seal(
    protocol: TerminalBenchProtocol,
    *,
    protocol_sha256: str,
) -> str:
    with open_control_store(protocol.output_root, protocol.evaluation_id) as store:
        try:
            payload = store.read(_SMOKE_SEAL)
        except ControlStoreError as exc:
            raise ControlStoreError("the smoke seal is unavailable") from exc
        try:
            seal = SmokeSeal.model_validate_json(payload)
        except ValueError as exc:
            raise ControlStoreError("the smoke seal is invalid") from exc
        if (
            seal.evaluation_id != protocol.evaluation_id
            or seal.protocol_sha256 != protocol_sha256
            or seal.smoke_instance_ids != protocol.subset.smoke_instance_ids
        ):
            raise ControlStoreError("the smoke seal does not match this evaluation")
        manifest_payload = store.read(_SMOKE_MANIFEST)
        if hashlib.sha256(manifest_payload).hexdigest() != seal.manifest_sha256:
            raise ControlStoreError("the sealed smoke manifest changed")
        try:
            manifest = HarborExecutionManifest.model_validate_json(manifest_payload)
        except ValueError as exc:
            raise ControlStoreError("the sealed smoke manifest is invalid") from exc
        if (
            manifest.run_kind != "smoke"
            or manifest.protocol_sha256 != protocol_sha256
            or manifest.expected_instance_ids != protocol.subset.smoke_instance_ids
            or manifest.observed_instance_ids != protocol.subset.smoke_instance_ids
            or any(status == "ERROR" for status in manifest.official_status.values())
        ):
            raise ControlStoreError("the sealed smoke manifest changed its binding")

        for index, instance_id in enumerate(protocol.subset.smoke_instance_ids):
            attempt_id = terminal_attempt_id(
                protocol.evaluation_id,
                "smoke",
                instance_id,
            )
            with store.open_directory(attempt_id) as attempt_store:
                started_payload = attempt_store.read(_COMMAND_STARTED)
                envelope_payload = attempt_store.read(_COMMAND_ENVELOPE)
                terminal_payload = attempt_store.read(_TERMINAL_RECORD)
                events_payload = attempt_store.read(_CODEX_EVENTS)
                receipt_payload = attempt_store.read(_BROKER_RECEIPT)
            if (
                hashlib.sha256(started_payload).hexdigest()
                != manifest.command_started_sha256[instance_id]
                or hashlib.sha256(envelope_payload).hexdigest()
                != manifest.command_envelope_sha256[instance_id]
            ):
                raise ControlStoreError("a smoke command record changed after sealing")
            if (
                hashlib.sha256(terminal_payload).hexdigest()
                != seal.terminal_record_sha256[instance_id]
                or seal.terminal_record_sha256[instance_id]
                != manifest.terminal_record_sha256[instance_id]
            ):
                raise ControlStoreError("a smoke terminal record changed after sealing")
            try:
                envelope = CommandEnvelope.model_validate_json(envelope_payload)
                provenance = TerminalBenchAgentProvenance.model_validate_json(
                    terminal_payload
                )
            except ValueError as exc:
                raise ControlStoreError("a sealed smoke control record is invalid") from exc
            if (
                envelope.outcome != "completed"
                or provenance.codex_outcome != "success"
                or hashlib.sha256(events_payload).hexdigest()
                != provenance.codex_events_sha256
                or hashlib.sha256(receipt_payload).hexdigest()
                != provenance.broker_receipt_sha256
                or manifest.codex_events_sha256[instance_id]
                != provenance.codex_events_sha256
            ):
                raise ControlStoreError("the smoke seal includes an unsuccessful run")

            job_dir = Path(manifest.job_dirs[index])
            if job_dir.parent != Path(protocol.output_root):
                raise ControlStoreError("a sealed smoke job moved outside the output root")
            for filename, digest_map in (
                ("config.json", manifest.job_config_sha256),
                ("lock.json", manifest.job_lock_sha256),
                ("result.json", manifest.job_result_sha256),
            ):
                path = job_dir / filename
                if (
                    not path.is_file()
                    or digest_map[instance_id] is None
                    or sha256_file(path) != digest_map[instance_id]
                ):
                    raise ControlStoreError("sealed Harbor job evidence changed")
            try:
                _trial_dir, _trial, result_digest = _read_single_trial_result(job_dir)
            except ValueError as exc:
                raise ControlStoreError("sealed official smoke result is unreadable") from exc
            if result_digest != manifest.trial_result_sha256[instance_id]:
                raise ControlStoreError("sealed official smoke result changed")
    return hashlib.sha256(payload).hexdigest()


async def _compose_control(
    environment,
    command: list[str],
    *,
    timeout_s: int = 30,
) -> Any:
    compose = getattr(environment, "_run_docker_compose_command", None)
    if not callable(compose):
        raise RuntimeError("formal runs require Harbor's Docker Compose environment")
    operation = cast(
        Awaitable[Any],
        compose(
            command,
            check=False,
            timeout_sec=timeout_s,
        ),
    )
    return await operation


async def _kill_and_confirm_main(
    environment,
    container_id: str,
) -> None:
    """SIGKILL the exact main container and prove that descendants cannot remain."""
    result = await _compose_control(
        environment,
        ["kill", "--signal", "SIGKILL", "main"],
    )
    if getattr(result, "return_code", None) not in (0, 1):
        raise RuntimeError("Docker Compose could not kill the benchmark container")
    deadline = time.monotonic() + 15
    while True:
        container = await _run_docker_inspect(
            "inspect",
            "--type",
            "container",
            container_id,
        )
        state = container.get("State")
        if (
            container.get("Id") == container_id
            and isinstance(state, Mapping)
            and state.get("Running") is False
            and state.get("Status") == "exited"
        ):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("the benchmark container did not reach a stopped state")
        await asyncio.sleep(0.1)


async def _restart_and_confirm_main(
    environment,
    before: DockerImageAttestation,
) -> DockerImageAttestation:
    """Restart in place, wait for readiness, and reject container replacement."""
    restarted = await _compose_control(
        environment,
        ["restart", "--no-deps", "--timeout", "0", "main"],
        timeout_s=60,
    )
    if getattr(restarted, "return_code", None) != 0:
        raise RuntimeError("Docker Compose could not restart the benchmark container")
    ready = await _compose_control(
        environment,
        [
            "up",
            "--detach",
            "--wait",
            "--wait-timeout",
            "60",
            "--no-recreate",
            "--no-deps",
            "main",
        ],
        timeout_s=70,
    )
    if getattr(ready, "return_code", None) != 0:
        raise RuntimeError("the restarted benchmark container did not become ready")
    after = await _attest_harbor_docker_image(environment)
    if after != before:
        raise RuntimeError("Docker Compose changed the benchmark container binding")
    return after


async def _finish_cleanup(
    operation: Awaitable[_T],
) -> tuple[_T, asyncio.CancelledError | None]:
    """Complete cleanup and return any cancellation that arrived while waiting."""
    task = asyncio.ensure_future(operation)
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    result = task.result()
    return result, cancellation


async def _cleanup_codex_runtime(environment) -> None:
    result = await environment.exec(
        command=(
            "set -eu; "
            f"rm -rf {_CODEX_HOME}; "
            f"rm -f {_CAPABILITY_UPLOAD} {_CAPABILITY_STAGING} "
            f"{_TLS_CERT_PATH} {_TLS_CERT_STAGING} {_CODEX_STDERR_PATH}"
        ),
        user="root",
    )
    if getattr(result, "return_code", None) != 0:
        raise RuntimeError("Codex temporary runtime cleanup failed")


async def _take_codex_stderr(environment) -> str:
    """Read one bounded stderr file, then remove it from the task container."""
    result = await environment.exec(
        command=(
            "set -eu; "
            f"test -f {_CODEX_STDERR_PATH}; test ! -L {_CODEX_STDERR_PATH}; "
            f'[ "$(stat -c %u:%g:%a:%h:%F {_CODEX_STDERR_PATH})" = '
            f"'{_CODEX_RUN_UID}:{_CODEX_RUN_UID}:600:1:regular file' ]; "
            f'[ "$(stat -c %s {_CODEX_STDERR_PATH})" -le 65536 ]; '
            f"cat {_CODEX_STDERR_PATH}; rm -f {_CODEX_STDERR_PATH}"
        ),
        user="root",
    )
    if getattr(result, "return_code", None) != 0:
        raise RuntimeError("Codex stderr could not be read and removed safely")
    return result.stdout or result.stderr or ""


async def _start_proxy_cancel_safe(
    controller: TerminalProxyController,
    **kwargs: Any,
) -> TerminalProxyHandle:
    """Never lose the handle to a broker whose blocking start outlived cancellation."""
    task = asyncio.create_task(asyncio.to_thread(controller.start, **kwargs))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    try:
        handle = task.result()
    except BaseException:
        if cancellation is not None:
            raise cancellation
        raise
    if cancellation is None:
        return handle
    try:
        _, cleanup_cancellation = await _finish_cleanup(
            asyncio.to_thread(controller.stop, handle)
        )
    except BaseException as cleanup_error:
        raise RuntimeError(
            "broker startup was cancelled and cleanup could not be confirmed"
        ) from cleanup_error
    if cleanup_cancellation is not None:
        cancellation = cleanup_cancellation
    raise cancellation


def create_protocol(
    *,
    evaluation_id: str,
    output_root: str | Path,
    model: str,
    reasoning_effort: str,
    codex_cli_version: str,
    codex_target: Literal["x86_64-unknown-linux-musl"],
    codex_binary_path: str | Path,
    broker_image_id: str,
    wheel_path: str | Path,
) -> TerminalBenchProtocol:
    """Build a preregistration only from the packaged official corpus."""
    corpus = load_terminal_bench_corpus()
    normalized_ids = tuple(sorted(corpus.tasks))
    subset = preregister_instances(normalized_ids)
    selected = (*subset.scored_instance_ids, *subset.smoke_instance_ids)
    return TerminalBenchProtocol(
        evaluation_id=evaluation_id,
        output_root=str(Path(output_root).resolve()),
        dataset_version=corpus.dataset_version,
        corpus_manifest_sha256=_CORPUS_RESOURCE_SHA256,
        corpus_instance_ids=normalized_ids,
        subset=subset,
        model=model,
        reasoning_effort=reasoning_effort,
        harbor_version=corpus.harbor_version,
        codex_cli_version=codex_cli_version,
        codex_target=codex_target,
        codex_binary_sha256=sha256_file(codex_binary_path),
        broker_image_id=broker_image_id,
        task_content_digests={
            item: corpus.tasks[item].task_content_digest for item in selected
        },
        task_checksums={
            item: corpus.tasks[item].task_checksum for item in selected
        },
        task_image_digests={
            item: corpus.tasks[item].task_image_digest for item in selected
        },
        task_agent_timeout_s={
            item: corpus.tasks[item].agent_timeout_s for item in selected
        },
        task_verifier_timeout_s={
            item: corpus.tasks[item].verifier_timeout_s for item in selected
        },
        task_environment_build_timeout_s={
            item: corpus.tasks[item].environment_build_timeout_s
            for item in selected
        },
        wheel_sha256=sha256_file(wheel_path),
    )


def write_protocol(protocol: TerminalBenchProtocol, path: str | Path) -> Path:
    """Write one deterministic, secret-free JSON protocol file."""
    target = Path(path)
    _atomic_write_text(target, protocol.model_dump_json(indent=2) + "\n")
    return target


def derive_terminal_bench_records(
    protocol: TerminalBenchProtocol,
    commands: Sequence[HarborRunCommand],
    *,
    protocol_path: str | Path,
    execution_manifest: HarborExecutionManifest,
    manifest_path: str | Path,
) -> TerminalBenchRecordBatch:
    """Derive scored rows only from a still-valid official Harbor execution."""
    manifest_digest = _load_execution_manifest(execution_manifest, manifest_path)
    protocol_digest = sha256_file(protocol_path)
    if (
        execution_manifest.run_kind != "scored"
        or execution_manifest.protocol_sha256 != protocol_digest
    ):
        raise ValueError("execution manifest is not bound to this scored protocol")

    current_manifest = validate_harbor_results(
        protocol,
        "scored",
        commands,
        protocol_path=protocol_path,
    )
    if current_manifest != execution_manifest:
        raise ValueError("Harbor evidence changed after the execution manifest was written")

    by_id = {command.instance_id: command for command in commands}
    if len(by_id) != len(commands):
        raise ValueError("Harbor commands contain duplicate instance ids")
    records: list[TerminalBenchTaskRecord] = []
    for instance_id in protocol.subset.scored_instance_ids:
        command = by_id.get(instance_id)
        if command is None:
            raise ValueError(f"missing scored Harbor command: {instance_id}")
        result_digest = execution_manifest.trial_result_sha256[instance_id]
        status = execution_manifest.official_status[instance_id]
        protocol_error = execution_manifest.protocol_errors[instance_id]
        independent_correct = {"PASS": True, "FAIL": False, "ERROR": None}[status]
        duration: float | None = None
        infrastructure_retries: int | None = None
        if result_digest is not None:
            _, trial_result, observed_digest = _read_single_trial_result(
                Path(command.job_dir)
            )
            if observed_digest != result_digest:
                raise ValueError(f"official Harbor result changed for {instance_id}")
            duration = _official_trial_duration(trial_result)
            if status != "ERROR":
                (
                    observed_status,
                    observed_correct,
                    observed_error,
                ) = _official_trial_outcome(trial_result)
                if (
                    observed_status != status
                    or observed_correct is not independent_correct
                    or observed_error is not None
                ):
                    raise ValueError("manifest status changed from the official verifier")
            infrastructure_retries = _official_infrastructure_retries(trial_result)
        records.append(
            TerminalBenchTaskRecord(
                instance_id=instance_id,
                protocol_sha256=protocol_digest,
                execution_manifest_sha256=manifest_digest,
                command_envelope_sha256=(
                    execution_manifest.command_envelope_sha256[instance_id]
                ),
                official_result_sha256=result_digest,
                official_status=status,
                independent_correct=independent_correct,
                duration_s=duration,
                protocol_error=protocol_error,
                infrastructure_retries=infrastructure_retries,
            )
        )
    return TerminalBenchRecordBatch(
        protocol_sha256=protocol_digest,
        execution_manifest_sha256=manifest_digest,
        records=tuple(records),
    )


def summarize_records(
    protocol: TerminalBenchProtocol,
    batch: TerminalBenchRecordBatch,
    *,
    commands: Sequence[HarborRunCommand],
    protocol_path: str | Path,
    execution_manifest: HarborExecutionManifest,
    manifest_path: str | Path,
) -> TerminalBenchSummary:
    """Summarize only the record batch re-derived from official Harbor evidence."""
    if not isinstance(batch, TerminalBenchRecordBatch):
        raise TypeError("summary requires a validated TerminalBenchRecordBatch")
    official_batch = derive_terminal_bench_records(
        protocol,
        commands,
        protocol_path=protocol_path,
        execution_manifest=execution_manifest,
        manifest_path=manifest_path,
    )
    if batch != official_batch:
        raise ValueError("task records do not match the bound official Harbor results")

    values = list(batch.records)
    passed = sum(row.official_status == "PASS" for row in values)
    failed = sum(row.official_status == "FAIL" for row in values)
    errors = sum(row.official_status == "ERROR" for row in values)
    durations = sorted(float(row.duration_s) for row in values if row.duration_s is not None)
    p50 = _percentile(durations, 0.50)
    p95 = _percentile(durations, 0.95)
    protocol_errors = sum(bool(row.protocol_error) for row in values)
    if passed + failed + errors != 20:
        raise ValueError("protocol summary must account for exactly 20 scored tasks")
    return TerminalBenchSummary(
        passed=passed,
        failed=failed,
        errors=errors,
        success_rate=passed / 20,
        p50_duration_s=p50,
        p95_duration_s=p95,
        protocol_errors=protocol_errors,
        missing_instance_ids=(),
    )


def _load_execution_manifest(
    execution_manifest: HarborExecutionManifest,
    manifest_path: str | Path,
) -> str:
    path = Path(manifest_path)
    try:
        raw = path.read_bytes()
        recorded = HarborExecutionManifest.model_validate_json(raw)
    except (OSError, ValueError) as exc:
        raise ValueError("the Harbor execution manifest is unreadable") from exc
    if recorded != execution_manifest:
        raise ValueError("manifest_path does not contain the supplied execution manifest")
    return hashlib.sha256(raw).hexdigest()


def _official_trial_outcome(
    trial_result: Mapping[str, Any],
) -> tuple[Literal["PASS", "FAIL", "ERROR"], bool | None, str | None]:
    exception = trial_result.get("exception_info")
    verifier = trial_result.get("verifier_result")
    if exception is not None:
        if not isinstance(exception, Mapping):
            raise ValueError("official Harbor exception_info must be an object")
        exception_type = exception.get("exception_type")
        if not isinstance(exception_type, str) or not exception_type.strip():
            raise ValueError("official Harbor exception_info omitted exception_type")
        if verifier is not None:
            raise ValueError(
                "official Harbor result cannot contain both an exception and a verifier result"
            )
        return "ERROR", None, f"Harbor trial exception: {exception_type.strip()}"

    if verifier is None:
        return "ERROR", None, "Harbor trial omitted verifier_result"
    if not isinstance(verifier, Mapping):
        raise ValueError("official Harbor verifier_result must be an object")
    rewards = verifier.get("rewards")
    if rewards is None:
        return "ERROR", None, "Harbor verifier omitted rewards"
    if not isinstance(rewards, Mapping):
        raise ValueError("official Harbor verifier rewards must be an object")
    if "reward" not in rewards:
        return "ERROR", None, "Harbor verifier omitted the official reward"
    reward = rewards["reward"]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or float(reward) not in (0.0, 1.0)
    ):
        raise ValueError("official Harbor reward must be the binary value 0 or 1")
    if float(reward) == 1.0:
        return "PASS", True, None
    return "FAIL", False, None


def _official_trial_duration(trial_result: Mapping[str, Any]) -> float | None:
    started = trial_result.get("started_at")
    finished = trial_result.get("finished_at")
    if started is None and finished is None:
        return None
    if not isinstance(started, str) or not isinstance(finished, str):
        raise ValueError("official Harbor duration requires both trial timestamps")
    try:
        duration = (
            datetime.fromisoformat(finished.replace("Z", "+00:00"))
            - datetime.fromisoformat(started.replace("Z", "+00:00"))
        ).total_seconds()
    except (TypeError, ValueError) as exc:
        raise ValueError("official Harbor trial timestamps are invalid") from exc
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("official Harbor trial duration must be finite and non-negative")
    return duration


def _official_infrastructure_retries(
    trial_result: Mapping[str, Any],
) -> Literal[0] | None:
    agent_result = trial_result.get("agent_result")
    metadata = (
        agent_result.get("metadata")
        if isinstance(agent_result, Mapping)
        else None
    )
    retries = (
        metadata.get("infrastructure_retries_used")
        if isinstance(metadata, Mapping)
        else None
    )
    if retries is None and trial_result.get("exception_info") is not None:
        return None
    if isinstance(retries, bool) or retries != 0:
        raise ValueError(
            "official Harbor metadata must prove that no infrastructure retry occurred"
        )
    return 0


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return round(float(values[index]), 3)


def selected_instance_ids(
    protocol: TerminalBenchProtocol,
    run_kind: Literal["smoke", "scored"],
) -> tuple[str, ...]:
    """Return the immutable instance order for one protocol phase."""
    if run_kind == "smoke":
        return protocol.subset.smoke_instance_ids
    return protocol.subset.scored_instance_ids


def install_commands(
    codex_cli_version: str,
    *,
    codex_binary_sha256: str,
    codex_target: Literal["x86_64-unknown-linux-musl"],
) -> list[str]:
    """Install the preregistered standalone binary without mutating the image."""
    match = _CODEX_VERSION_RE.fullmatch(codex_cli_version)
    if match is None:
        raise ValueError("invalid exact Codex CLI version")
    if _SHA256_HEX_RE.fullmatch(codex_binary_sha256) is None:
        raise ValueError("invalid Codex binary digest")
    if codex_target != "x86_64-unknown-linux-musl":
        raise ValueError("unsupported formal Codex target")
    expected = shlex.quote(codex_cli_version)
    passwd_entry = (
        f"lha_codex:x:{_CODEX_RUN_UID}:{_CODEX_RUN_UID}::/tmp:"
        f"{_CODEX_PRIVILEGED_SHELL}"
    )
    return [
        "set -eu; "
        f"install -o 0 -g 0 -m 4755 {_CODEX_UPLOAD} /usr/local/bin/codex; "
        f"rm -f {_CODEX_UPLOAD}; "
        '[ "$(uname -m)" = x86_64 ]; '
        f'[ "$(sha256sum /usr/local/bin/codex | awk \'{{print $1}}\')" = '
        f"{codex_binary_sha256} ]; "
        f'[ "$(/usr/local/bin/codex --version)" = {expected} ]; '
        '[ "$(stat -c %u:%g:%a /usr/local/bin/codex)" = 0:0:4755 ]',
        "set -eu; "
        "test -x /bin/bash; "
        '[ "$(cat /proc/sys/fs/suid_dumpable)" = 0 ]; '
        'no_new_privs=$(awk \'/^NoNewPrivs:/ {print $2}\' /proc/self/status); '
        '[ "$no_new_privs" = 0 ]; '
        'cap_bnd=$(awk \'/^CapBnd:/ {print $2}\' /proc/self/status); '
        '(( (16#$cap_bnd & 524288) == 0 )); '
        "mkdir -p /usr/local/lib/lha; "
        "printf '%s\\n' '#!/bin/bash -p' "
        "'if [ -e /proc/self/fd/3 ]; then exec 3<&-; fi' "
        "'exec /bin/bash -p \"$@\"' "
        "> /tmp/.lha-privileged-bash; "
        f"install -o 0 -g 0 -m 755 /tmp/.lha-privileged-bash "
        f"{_CODEX_PRIVILEGED_SHELL}; "
        "rm -f /tmp/.lha-privileged-bash; "
        f"install -o 0 -g 0 -m 4755 /bin/bash {_CODEX_SUID_PROBE}; "
        f"install -d -o 0 -g 0 -m 700 {_RUNTIME_STAGING_DIR}; "
        f"entry={shlex.quote(passwd_entry)}; "
        f"existing=$(awk -F: '$1==\"lha_codex\" || $3==\"{_CODEX_RUN_UID}\" "
        "{print}' /etc/passwd); "
        'if [ -n "$existing" ] && [ "$existing" != "$entry" ]; then exit 1; fi; '
        'if [ -z "$existing" ]; then printf \'%s\\n\' "$entry" >> /etc/passwd; fi; '
        f'[ "$(stat -c %u:%g:%a {_CODEX_PRIVILEGED_SHELL})" = 0:0:755 ]; '
        f'[ "$(stat -c %u:%g:%a {_CODEX_SUID_PROBE})" = 0:0:4755 ]',
    ]


def process_isolation_check_command() -> str:
    """Check the kernel setuid transition and the tool-shell descriptor boundary."""
    tool_check = (
        f'[ "$(id -ru)" = {_CODEX_RUN_UID} ] '
        '&& [ "$(id -u)" = 0 ] '
        "&& [ \"$(stat -Lc '%d:%i' /proc/self/fd/3 2>/dev/null || true)\" "
        '!= "$LHA_FD_PROBE_INODE" ] '
        "&& unset LHA_FD_PROBE_INODE "
        f"&& rm -f {_CODEX_SUID_PROBE}"
    )
    privileged_check = (
        "probe_file=$(mktemp /tmp/.lha-fd-probe.XXXXXX); "
        'chmod 600 "$probe_file"; '
        "LHA_FD_PROBE_INODE=$(stat -Lc '%d:%i' \"$probe_file\"); "
        'export LHA_FD_PROBE_INODE; exec 3<"$probe_file"; rm -f "$probe_file"; '
        f"exec {_CODEX_PRIVILEGED_SHELL} -c "
        + shlex.quote(tool_check)
    )
    return (
        "set -eu; "
        f'[ "$(id -ru)" = {_CODEX_RUN_UID} ]; '
        f"{_CODEX_SUID_PROBE} -p -c {shlex.quote(privileged_check)}; "
        f"test ! -e {_CODEX_SUID_PROBE}"
    )


def codex_exec_command(
    model: str,
    reasoning_effort: str,
    instruction: str,
    *,
    proxy_base_url: str = "https://127.0.0.1:8080",
    binding_headers: Mapping[str, str] | None = None,
    request_max_retries: int = 1,
    stream_max_retries: int = 0,
) -> str:
    """Build the one tool-enabled Codex invocation used inside Harbor.

    The command contains only the name of the ephemeral capability variable.
    Its value is passed by Docker exec and is never embedded in argv.
    """
    if (
        not proxy_base_url.startswith("https://")
        or any(character.isspace() for character in proxy_base_url)
        or "'" in proxy_base_url
    ):
        raise ValueError("proxy_base_url must be a simple private HTTPS endpoint")
    if request_max_retries != 1 or stream_max_retries != 0:
        raise ValueError(
            "formal Codex retries require one request retry and no client stream retry"
        )
    headers = dict(binding_headers or {})
    expected_headers = {
        "X-LHA-Evaluation-ID",
        "X-LHA-Attempt-ID",
        "X-LHA-Container-ID",
    }
    if set(headers) != expected_headers or any(
        not value
        or len(value) > 128
        or any(character.isspace() for character in value)
        for value in headers.values()
    ):
        raise ValueError("the broker binding headers are incomplete or unsafe")
    headers["version"] = "0.141.0"
    header_table = "{ " + ", ".join(
        f"{json.dumps(name)} = {json.dumps(value)}"
        for name, value in sorted(headers.items())
    ) + " }"
    provider = (
        "{ "
        "name = 'OpenAI', "
        f"base_url = '{proxy_base_url.rstrip('/')}', "
        f"env_key = '{CAPABILITY_ENV}', "
        "wire_api = 'responses', "
        f"request_max_retries = {request_max_retries}, "
        f"stream_max_retries = {stream_max_retries}, "
        "requires_openai_auth = false, "
        "supports_websockets = false, "
        f"http_headers = {header_table} "
        "}"
    )
    codex = " ".join(
        [
            "exec",
            "env",
            f"{CAPABILITY_ENV}=\"$lha_capability\"",
            "CODEX_CA_CERTIFICATE=/proc/self/fd/3",
            "SSL_CERT_FILE=/proc/self/fd/3",
            f"CODEX_HOME={_CODEX_HOME}",
            "/usr/local/bin/codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            "--disable",
            "multi_agent",
            "--disable",
            "multi_agent_v2",
            "--strict-config",
            "--json",
            "--model",
            shlex.quote(model),
            "-c",
            shlex.quote(f"model_reasoning_effort={reasoning_effort!r}"),
            "-c",
            shlex.quote("model_provider='lha_terminal_proxy'"),
            "-c",
            shlex.quote(f"model_providers.lha_terminal_proxy={provider}"),
            "-c",
            shlex.quote("shell_environment_policy.inherit='none'"),
            "-c",
            shlex.quote(
                "shell_environment_policy.set="
                "{ PATH='/usr/local/bin:/usr/bin:/bin', "
                f"HOME='{_CODEX_HOME}', USER='root', LOGNAME='root', "
                f"SHELL='{_CODEX_PRIVILEGED_SHELL}' }}"
            ),
            "-c",
            shlex.quote("allow_login_shell=false"),
            "--",
            shlex.quote(instruction),
            f"2>{_CODEX_STDERR_PATH}",
        ]
    )
    cleanup = (
        f"rm -f {_CAPABILITY_UPLOAD} {_CAPABILITY_STAGING} "
        f"{_TLS_CERT_PATH} {_TLS_CERT_STAGING}"
    )
    return (
        "set -eu; umask 077; "
        f"trap {shlex.quote(cleanup)} EXIT HUP INT TERM; "
        f"test ! -e {_CODEX_HOME}; mkdir -m 700 {_CODEX_HOME}; "
        f"test -f {_CAPABILITY_UPLOAD}; "
        f"test -f {_TLS_CERT_PATH}; "
        f"exec 3<{_TLS_CERT_PATH}; "
        f"rm -f {_TLS_CERT_PATH}; "
        f"lha_capability=$(cat {_CAPABILITY_UPLOAD}); "
        f"rm -f {_CAPABILITY_UPLOAD}; "
        f"test -n \"$lha_capability\"; {codex}"
    )


def audit_codex_jsonl(
    event_stream: str,
    *,
    max_tool_calls: int = TerminalBenchBudgets().max_tool_calls,
    max_reconnect_notices: int = TerminalBenchBudgets().stream_max_retries,
) -> CodexRunAudit:
    """Validate the exact pinned 0.141 event schema and its lifecycle."""
    budgets = TerminalBenchBudgets()
    try:
        strict = audit_codex_0141_jsonl(
            event_stream,
            max_tool_calls=max_tool_calls,
            max_line_bytes=budgets.max_jsonl_line_bytes,
            max_total_bytes=budgets.max_jsonl_bytes,
            max_reconnect_notices=max_reconnect_notices,
        )
    except CodexEventError as exc:
        raise RuntimeError(str(exc)) from exc
    return CodexRunAudit(
        event_counts=strict.event_counts,
        item_counts=strict.item_counts,
        tool_calls=strict.tool_calls,
        reconnect_notices=strict.reconnect_notices,
        input_tokens=strict.input_tokens,
        cached_input_tokens=strict.cached_input_tokens,
        output_tokens=strict.output_tokens,
        reasoning_output_tokens=strict.reasoning_output_tokens,
    )


def _harbor_argv(
    protocol: TerminalBenchProtocol,
    run_kind: Literal["smoke", "scored"],
    instance_id: str,
    *,
    protocol_file: Path,
    wheel: Path,
    codex_binary: Path,
    output_root: Path,
    job_name: str,
    attempt_id: str,
) -> tuple[str, ...]:
    return (
        "uvx",
        "--python",
        "3.12",
        "--with",
        f"harbor=={protocol.harbor_version}",
        "--with",
        str(wheel),
        "harbor",
        "run",
        "--dataset",
        f"{DATASET}@{protocol.dataset_version}",
        "--agent",
        AGENT_IMPORT_PATH,
        "--model",
        protocol.model,
        "--jobs-dir",
        str(output_root),
        "--job-name",
        job_name,
        "--n-attempts",
        "1",
        "--agent-timeout-multiplier",
        str(protocol.budgets.harbor_agent_timeout_multiplier),
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--include-task-name",
        instance_id,
        "--n-tasks",
        "1",
        "--yes",
        "--agent-kwarg",
        f"wheel_path={wheel}",
        "--agent-kwarg",
        f"codex_binary_path={codex_binary}",
        "--agent-kwarg",
        f"protocol_path={protocol_file}",
        "--agent-kwarg",
        f"reasoning_effort={protocol.reasoning_effort}",
        "--agent-kwarg",
        f"instance_id={instance_id}",
        "--agent-kwarg",
        f"run_kind={run_kind}",
        "--agent-kwarg",
        f"attempt_id={attempt_id}",
    )


def build_harbor_commands(
    protocol: TerminalBenchProtocol,
    run_kind: Literal["smoke", "scored"],
    *,
    protocol_path: str | Path,
    wheel_path: str | Path,
    codex_binary_path: str | Path,
) -> tuple[HarborRunCommand, ...]:
    """Generate one exact Harbor job per registered instance.

    A separate job lets the full instance ID be passed to the agent and checked
    against Harbor's trial name before Codex starts.  ``output_root`` and the
    job prefix come from the immutable protocol, so rebuilding commands cannot
    silently create a second legal evaluation.
    """
    protocol_file = Path(protocol_path).resolve()
    wheel = Path(wheel_path).resolve()
    codex_binary = Path(codex_binary_path).resolve()
    output_root = Path(protocol.output_root)
    if not protocol_file.is_file():
        raise ValueError("the protocol file does not exist")
    loaded = TerminalBenchProtocol.model_validate_json(protocol_file.read_text())
    if loaded != protocol:
        raise ValueError("protocol_path does not contain the supplied protocol")
    if not wheel.is_file() or sha256_file(wheel) != protocol.wheel_sha256:
        raise ValueError("wheel_path does not match the preregistration")
    if (
        not codex_binary.is_file()
        or sha256_file(codex_binary) != protocol.codex_binary_sha256
    ):
        raise ValueError("codex_binary_path does not match the preregistration")
    safe_prefix = f"lha-tbench-2-1-{protocol.evaluation_id[:12]}"

    commands: list[HarborRunCommand] = []
    for index, instance_id in enumerate(selected_instance_ids(protocol, run_kind), start=1):
        suffix = hashlib.sha256(instance_id.encode()).hexdigest()[:10]
        job_name = f"{safe_prefix}-{run_kind}-{index:02d}-{suffix}"
        attempt_id = terminal_attempt_id(
            protocol.evaluation_id,
            run_kind,
            instance_id,
        )
        argv = _harbor_argv(
            protocol,
            run_kind,
            instance_id,
            protocol_file=protocol_file,
            wheel=wheel,
            codex_binary=codex_binary,
            output_root=output_root,
            job_name=job_name,
            attempt_id=attempt_id,
        )
        commands.append(
            HarborRunCommand(
                evaluation_id=protocol.evaluation_id,
                attempt_id=attempt_id,
                run_kind=run_kind,
                instance_id=instance_id,
                task_content_digest=protocol.task_content_digests[instance_id],
                task_checksum=protocol.task_checksums[instance_id],
                task_image_digest=protocol.task_image_digests[instance_id],
                argv=argv,
                command_sha256=command_digest(argv),
                job_dir=str(output_root / job_name),
            )
        )
    return tuple(commands)


def initialize_terminal_evaluation(
    protocol: TerminalBenchProtocol,
    commands: Sequence[HarborRunCommand],
    *,
    protocol_path: str | Path,
) -> EvaluationRegistration:
    """Create the one legal jobs root and its unmounted control ledger."""
    protocol_file = Path(protocol_path).resolve()
    recorded = TerminalBenchProtocol.model_validate_json(protocol_file.read_text())
    if recorded != protocol:
        raise ValueError("protocol_path does not contain the supplied protocol")
    expected = (
        *selected_instance_ids(protocol, "smoke"),
        *selected_instance_ids(protocol, "scored"),
    )
    by_key = {(item.run_kind, item.instance_id): item for item in commands}
    expected_keys = {
        ("smoke", item) for item in selected_instance_ids(protocol, "smoke")
    } | {
        ("scored", item) for item in selected_instance_ids(protocol, "scored")
    }
    if len(commands) != 23 or set(by_key) != expected_keys:
        raise ValueError("evaluation initialization requires exactly the registered 23 commands")
    if any(
        item.evaluation_id != protocol.evaluation_id
        or Path(item.job_dir).parent != Path(protocol.output_root)
        for item in commands
    ):
        raise ValueError("a Harbor command is bound to another evaluation root")
    attempts: list[RegisteredAttempt] = []
    for kind, instance_ids in cast(
        tuple[
            tuple[Literal["smoke", "scored"], tuple[str, ...]],
            tuple[Literal["smoke", "scored"], tuple[str, ...]],
        ],
        (
        ("smoke", protocol.subset.smoke_instance_ids),
        ("scored", protocol.subset.scored_instance_ids),
        ),
    ):
        run_kind = kind
        for instance_id in instance_ids:
            item = by_key[(run_kind, instance_id)]
            attempts.append(
                RegisteredAttempt(
                    attempt_id=item.attempt_id,
                    run_kind=run_kind,
                    instance_id=instance_id,
                    command_sha256=item.command_sha256,
                )
            )
    if tuple(item.instance_id for item in attempts) != expected:
        raise AssertionError("registered command order changed during initialization")
    return initialize_control_store(
        evaluation_id=protocol.evaluation_id,
        protocol_sha256=sha256_file(protocol_file),
        output_root=protocol.output_root,
        attempts=tuple(attempts),
    )


def _formal_harbor_environment(auth_path: Path) -> dict[str, str]:
    """Build a small host environment without forwarding unrelated credentials."""
    allowed = (
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TERM",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "UV_CACHE_DIR",
        "XDG_CACHE_HOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    )
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment["LHA_CODEX_AUTH_FILE"] = str(auth_path)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _host_process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise RuntimeError("could not inspect the Harbor process group") from exc
    return True


def _wait_for_host_process_group(
    process: subprocess.Popen[bytes],
    process_group: int,
    *,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        process.poll()
        if not _host_process_group_exists(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _stop_host_process(process: subprocess.Popen[bytes]) -> None:
    """Stop Harbor and every child before publishing its command envelope."""
    process_group = getattr(process, "pid", None)
    if not isinstance(process_group, int) or isinstance(process_group, bool):
        if process.poll() is not None:
            return
        raise RuntimeError("the Harbor process has no valid process-group identity")

    if not _host_process_group_exists(process_group):
        try:
            process.wait(timeout=_HOST_PROCESS_KILL_GRACE_S)
        except (ChildProcessError, OSError, subprocess.TimeoutExpired) as exc:
            if process.poll() is None:
                raise RuntimeError("the Harbor process leader could not be reaped") from exc
        return

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not _wait_for_host_process_group(
        process,
        process_group,
        timeout_s=_HOST_PROCESS_TERM_GRACE_S,
    ):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=_HOST_PROCESS_KILL_GRACE_S)
    except (ChildProcessError, OSError, subprocess.TimeoutExpired) as exc:
        if process.poll() is None:
            raise RuntimeError("the Harbor process leader could not be reaped") from exc
    if not _wait_for_host_process_group(
        process,
        process_group,
        timeout_s=_HOST_PROCESS_KILL_GRACE_S,
    ):
        raise RuntimeError(
            "the Harbor process group is still present; "
            "refusing to publish mutable command evidence"
        )


def _command_failure_stage(
    store: SecureDirectory,
    *,
    return_code: int | None,
) -> Literal[
    "harbor_start",
    "environment_setup",
    "agent_setup",
    "model",
    "verification",
    "result_persistence",
    "unknown",
]:
    if return_code is None:
        return "harbor_start"
    if store.has(_TERMINAL_RECORD):
        try:
            provenance = TerminalBenchAgentProvenance.model_validate_json(
                store.read(_TERMINAL_RECORD)
            )
        except ValueError:
            return "result_persistence"
        if provenance.codex_outcome == "setup_error":
            return "agent_setup"
        if provenance.codex_outcome == "success":
            return "verification"
        return "model"
    if store.has(_MODEL_STARTED):
        return "model"
    return "environment_setup"


def run_harbor_command_once(
    protocol: TerminalBenchProtocol,
    command: HarborRunCommand,
    *,
    protocol_path: str | Path,
    auth_path: str | Path,
    _evaluation_lock_fd: int | None = None,
) -> CommandEnvelope:
    """Spend one registered command slot and always leave a host-side envelope."""
    if _evaluation_lock_fd is None:
        with evaluation_lock(
            protocol.output_root,
            protocol.evaluation_id,
        ) as lock_fd:
            return run_harbor_command_once(
                protocol,
                command,
                protocol_path=protocol_path,
                auth_path=auth_path,
                _evaluation_lock_fd=lock_fd,
            )
    try:
        lock_info = os.fstat(_evaluation_lock_fd)
    except OSError as exc:
        raise ControlStoreError("the evaluation lease descriptor is unavailable") from exc
    if not stat.S_ISREG(lock_info.st_mode):
        raise ControlStoreError("the evaluation lease descriptor is invalid")

    protocol_file = Path(protocol_path).resolve()
    if TerminalBenchProtocol.model_validate_json(protocol_file.read_text()) != protocol:
        raise ValueError("the protocol file changed before Harbor execution")
    _validate_command_contract(protocol, command, protocol_file=protocol_file)
    protocol_sha256 = sha256_file(protocol_file)
    registered = _registered_attempt(
        protocol,
        protocol_sha256=protocol_sha256,
        attempt_id=command.attempt_id,
        run_kind=command.run_kind,
        instance_id=command.instance_id,
    )
    if registered.command_sha256 != command.command_sha256:
        raise ControlStoreError("the registered Harbor command digest changed")

    auth = Path(auth_path)
    _load_broker_secrets(
        auth,
        minimum_validity_s=protocol.budgets.codex_timeout_s + 900,
    )
    process: subprocess.Popen[bytes] | None = None
    caught: BaseException | None = None
    return_code: int | None = None
    outcome: Literal["completed", "error", "interrupted"] = "error"
    failure_stage: Literal[
        "harbor_start",
        "environment_setup",
        "agent_setup",
        "model",
        "verification",
        "result_persistence",
        "unknown",
    ] | None = "harbor_start"
    started_at: str

    with open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        if any(
            store.has(name)
            for name in (
                _COMMAND_STARTED,
                _COMMAND_ENVELOPE,
                _MODEL_STARTED,
                _TERMINAL_RECORD,
            )
        ):
            raise ControlRecordExists("the registered Harbor command was already consumed")
        marker = write_command_started(
            store,
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            run_kind=command.run_kind,
            instance_id=command.instance_id,
            command_sha256=command.command_sha256,
        )
        started_at = marker.started_at

    try:
        process = subprocess.Popen(
            list(command.argv),
            env=_formal_harbor_environment(auth),
            cwd=Path.cwd(),
            start_new_session=True,
            pass_fds=(_evaluation_lock_fd,),
        )
        return_code = process.wait(timeout=protocol.budgets.host_command_timeout_s)
        if return_code == 0:
            outcome = "completed"
            failure_stage = None
        else:
            outcome = "error"
    except KeyboardInterrupt as exc:
        caught = exc
        outcome = "interrupted"
    except (OSError, subprocess.TimeoutExpired) as exc:
        caught = exc
        outcome = "error"
    finally:
        if process is not None:
            _stop_host_process(process)
            return_code = process.returncode
        with open_attempt_store(
            protocol.output_root,
            protocol.evaluation_id,
            command.attempt_id,
        ) as store:
            model_started = store.has(_MODEL_STARTED)
            if outcome != "completed":
                failure_stage = _command_failure_stage(
                    store,
                    return_code=return_code,
                )
            exception_sha256 = (
                hashlib.sha256(type(caught).__name__.encode()).hexdigest()
                if caught is not None
                else None
            )
            envelope = CommandEnvelope(
                evaluation_id=protocol.evaluation_id,
                attempt_id=command.attempt_id,
                run_kind=command.run_kind,
                instance_id=command.instance_id,
                command_sha256=command.command_sha256,
                started_at=started_at,
                finished_at=now().isoformat(),
                process_return_code=return_code,
                outcome=outcome,
                failure_stage=failure_stage,
                exception_sha256=exception_sha256,
                model_started=model_started,
            )
            store.write_json_once(_COMMAND_ENVELOPE, envelope)
    if caught is not None and isinstance(caught, KeyboardInterrupt):
        raise caught
    return envelope


def run_terminal_phase(
    protocol: TerminalBenchProtocol,
    run_kind: Literal["smoke", "scored"],
    commands: Sequence[HarborRunCommand],
    *,
    protocol_path: str | Path,
    auth_path: str | Path,
    _evaluation_lock_fd: int | None = None,
) -> tuple[CommandEnvelope, ...]:
    """Run or resume one registered phase without spending an attempt twice."""
    if _evaluation_lock_fd is None:
        with evaluation_lock(
            protocol.output_root,
            protocol.evaluation_id,
        ) as lock_fd:
            return run_terminal_phase(
                protocol,
                run_kind,
                commands,
                protocol_path=protocol_path,
                auth_path=auth_path,
                _evaluation_lock_fd=lock_fd,
            )
    expected = selected_instance_ids(protocol, run_kind)
    if tuple(command.instance_id for command in commands) != expected:
        raise ValueError("Harbor phase commands are not in preregistered order")
    protocol_file = Path(protocol_path).resolve()
    try:
        recorded = TerminalBenchProtocol.model_validate_json(protocol_file.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("the preregistration file is unreadable") from exc
    if recorded != protocol:
        raise ValueError("protocol_path does not contain the supplied protocol")
    protocol_sha256 = sha256_file(protocol_file)
    _control_registration(protocol, protocol_sha256=protocol_sha256)
    states: list[Literal["pending", "enveloped", "partial"]] = []
    for command in commands:
        if command.run_kind != run_kind:
            raise ValueError("a Harbor command is outside this protocol phase")
        _validate_command_contract(protocol, command, protocol_file=protocol_file)
        registered = _registered_attempt(
            protocol,
            protocol_sha256=protocol_sha256,
            attempt_id=command.attempt_id,
            run_kind=run_kind,
            instance_id=command.instance_id,
        )
        if registered.command_sha256 != command.command_sha256:
            raise ControlStoreError("the registered Harbor command digest changed")
        state = _attempt_control_state(
            protocol,
            command,
            protocol_sha256=protocol_sha256,
        )
        if state == "partial":
            _finalize_abandoned_attempt(
                protocol,
                command,
                protocol_sha256=protocol_sha256,
            )
            state = "enveloped"
        states.append(state)
    if run_kind == "scored":
        _validated_smoke_seal(
            protocol,
            protocol_sha256=protocol_sha256,
        )

    envelopes: list[CommandEnvelope] = []
    first_pending: int | None = None
    for index, (command, state) in enumerate(zip(commands, states, strict=True)):
        if state == "pending":
            first_pending = index
            break
        envelope, *_rest = _load_attempt_control(
            protocol,
            command,
            protocol_sha256=protocol_sha256,
        )
        envelopes.append(envelope)
        if run_kind == "smoke" and envelope.outcome != "completed":
            return tuple(envelopes)
        if run_kind == "smoke" and not _smoke_result_allows_continuation(
            protocol,
            commands,
            protocol_path=protocol_file,
            instance_id=command.instance_id,
        ):
            return tuple(envelopes)

    if first_pending is None:
        return tuple(envelopes)
    if any(state != "pending" for state in states[first_pending:]):
        raise ControlStoreError(
            "phase control records are out of order after the first unconsumed command"
        )

    for command in commands[first_pending:]:
        envelope = run_harbor_command_once(
            protocol,
            command,
            protocol_path=protocol_path,
            auth_path=auth_path,
            _evaluation_lock_fd=_evaluation_lock_fd,
        )
        envelopes.append(envelope)
        if run_kind == "smoke" and envelope.outcome != "completed":
            break
        if run_kind == "smoke" and not _smoke_result_allows_continuation(
            protocol,
            commands,
            protocol_path=protocol_file,
            instance_id=command.instance_id,
        ):
            break
    return tuple(envelopes)


def _smoke_result_allows_continuation(
    protocol: TerminalBenchProtocol,
    commands: Sequence[HarborRunCommand],
    *,
    protocol_path: str | Path,
    instance_id: str,
) -> bool:
    """Fail closed until the completed smoke row has all formal evidence."""
    manifest = validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
    )
    if (
        manifest.official_status[instance_id] == "ERROR"
        or manifest.protocol_errors[instance_id] is not None
    ):
        return False
    evidence = (
        manifest.codex_events_sha256,
        manifest.container_image_ids,
        manifest.command_started_sha256,
        manifest.command_envelope_sha256,
        manifest.terminal_record_sha256,
        manifest.job_config_sha256,
        manifest.job_lock_sha256,
        manifest.job_result_sha256,
        manifest.trial_result_sha256,
    )
    return all(values[instance_id] is not None for values in evidence)


def _finalize_abandoned_attempt(
    protocol: TerminalBenchProtocol,
    command: HarborRunCommand,
    *,
    protocol_sha256: str,
) -> CommandEnvelope:
    """Convert a controller-crash gap into one immutable ERROR envelope."""
    with open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        if not store.has(_COMMAND_STARTED) or store.has(_COMMAND_ENVELOPE):
            raise ControlStoreError(
                "only a started command without an envelope can be abandoned"
            )
        try:
            started = CommandStartedMarker.model_validate_json(
                store.read(_COMMAND_STARTED)
            )
        except ValueError as exc:
            raise ControlStoreError("the abandoned command marker is invalid") from exc
        expected = (
            protocol.evaluation_id,
            command.attempt_id,
            command.run_kind,
            command.instance_id,
            command.command_sha256,
        )
        if (
            started.evaluation_id,
            started.attempt_id,
            started.run_kind,
            started.instance_id,
            started.command_sha256,
        ) != expected:
            raise ControlStoreError("the abandoned command changed its binding")

        if not store.has(_MODEL_STARTED):
            raise ControlStoreError(
                "abandoned task cleanup cannot be proven without a model-start marker"
            )
        try:
            marker = ModelStartedMarker.model_validate_json(
                store.read(_MODEL_STARTED)
            )
        except ValueError as exc:
            raise ControlStoreError(
                "the abandoned model-start marker is invalid"
            ) from exc
        if (
            marker.evaluation_id != protocol.evaluation_id
            or marker.attempt_id != command.attempt_id
            or marker.protocol_sha256 != protocol_sha256
            or marker.run_kind != command.run_kind
            or marker.instance_id != command.instance_id
        ):
            raise ControlStoreError(
                "the abandoned model-start marker changed its binding"
            )
        try:
            TerminalProxyController(
                image_id=protocol.broker_image_id
            ).cleanup_abandoned(
                evaluation_id=protocol.evaluation_id,
                attempt_id=command.attempt_id,
                source_container_id=marker.container_id,
                expected_task_image_digest=protocol.task_image_digests[
                    command.instance_id
                ],
            )
        except (TerminalProxyError, ValueError) as exc:
            raise ControlStoreError(
                "abandoned broker and task cleanup could not be proven"
            ) from exc

        envelope = CommandEnvelope(
            evaluation_id=protocol.evaluation_id,
            attempt_id=command.attempt_id,
            run_kind=command.run_kind,
            instance_id=command.instance_id,
            command_sha256=command.command_sha256,
            started_at=started.started_at,
            finished_at=now().isoformat(),
            process_return_code=None,
            outcome="interrupted",
            failure_stage="model",
            exception_sha256=hashlib.sha256(
                b"controller exited before writing the command envelope"
            ).hexdigest(),
            model_started=True,
        )
        store.write_json_once(_COMMAND_ENVELOPE, envelope)
    return envelope


def _attempt_control_state(
    protocol: TerminalBenchProtocol,
    command: HarborRunCommand,
    *,
    protocol_sha256: str,
) -> Literal["pending", "enveloped", "partial"]:
    with open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        present = {
            name: store.has(name)
            for name in (
                _COMMAND_STARTED,
                _COMMAND_ENVELOPE,
                _MODEL_STARTED,
                _TERMINAL_RECORD,
            )
        }
    if not any(present.values()):
        return "pending"
    if present[_COMMAND_STARTED] and present[_COMMAND_ENVELOPE]:
        _load_attempt_control(
            protocol,
            command,
            protocol_sha256=protocol_sha256,
        )
        return "enveloped"
    return "partial"


def _load_attempt_control(
    protocol: TerminalBenchProtocol,
    command: HarborRunCommand,
    *,
    protocol_sha256: str,
) -> tuple[
    CommandEnvelope,
    str,
    str,
    ModelStartedMarker | None,
    bytes | None,
    str | None,
]:
    with open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        try:
            started_payload = store.read(_COMMAND_STARTED)
            envelope_payload = store.read(_COMMAND_ENVELOPE)
            started = CommandStartedMarker.model_validate_json(started_payload)
            envelope = CommandEnvelope.model_validate_json(envelope_payload)
        except (ControlStoreError, ValueError) as exc:
            raise ValueError("the host command control record is missing or invalid") from exc
        expected = (
            protocol.evaluation_id,
            command.attempt_id,
            command.run_kind,
            command.instance_id,
            command.command_sha256,
        )
        if (
            started.evaluation_id,
            started.attempt_id,
            started.run_kind,
            started.instance_id,
            started.command_sha256,
        ) != expected or (
            envelope.evaluation_id,
            envelope.attempt_id,
            envelope.run_kind,
            envelope.instance_id,
            envelope.command_sha256,
        ) != expected:
            raise ValueError("the host command control record changed its binding")
        if envelope.started_at != started.started_at:
            raise ValueError("the command envelope changed its immutable start time")

        marker: ModelStartedMarker | None = None
        if store.has(_MODEL_STARTED):
            try:
                marker = ModelStartedMarker.model_validate_json(store.read(_MODEL_STARTED))
            except ValueError as exc:
                raise ValueError("the model-start marker is invalid") from exc
            if (
                marker.evaluation_id != protocol.evaluation_id
                or marker.attempt_id != command.attempt_id
                or marker.protocol_sha256 != protocol_sha256
                or marker.run_kind != command.run_kind
                or marker.instance_id != command.instance_id
            ):
                raise ValueError("the model-start marker changed its trial binding")
        if envelope.model_started != (marker is not None):
            raise ValueError("the command envelope disagrees with the model-start marker")

        terminal_payload = store.read(_TERMINAL_RECORD) if store.has(_TERMINAL_RECORD) else None
    return (
        envelope,
        hashlib.sha256(started_payload).hexdigest(),
        hashlib.sha256(envelope_payload).hexdigest(),
        marker,
        terminal_payload,
        (
            hashlib.sha256(terminal_payload).hexdigest()
            if terminal_payload is not None
            else None
        ),
    )


def validate_harbor_results(
    protocol: TerminalBenchProtocol,
    run_kind: Literal["smoke", "scored"],
    commands: Sequence[HarborRunCommand],
    *,
    protocol_path: str | Path,
    manifest_path: str | Path | None = None,
) -> HarborExecutionManifest:
    """Validate every registered row without dropping pre-verifier failures."""
    protocol_file = Path(protocol_path).resolve()
    try:
        recorded_protocol = TerminalBenchProtocol.model_validate_json(
            protocol_file.read_text()
        )
    except (OSError, ValueError) as exc:
        raise ValueError("the preregistration file is unreadable") from exc
    if recorded_protocol != protocol:
        raise ValueError("protocol_path does not contain the supplied protocol")
    expected = selected_instance_ids(protocol, run_kind)
    if tuple(command.instance_id for command in commands) != expected:
        raise ValueError("the Harbor commands do not match preregistered order")
    protocol_sha256 = sha256_file(protocol_file)
    _control_registration(protocol, protocol_sha256=protocol_sha256)

    observed: list[str] = []
    job_dirs: list[str] = []
    event_digests: dict[str, str | None] = {}
    image_ids: dict[str, str | None] = {}
    started_digests: dict[str, str | None] = {}
    envelope_digests: dict[str, str | None] = {}
    terminal_digests: dict[str, str | None] = {}
    config_digests: dict[str, str | None] = {}
    lock_digests: dict[str, str | None] = {}
    job_result_digests: dict[str, str | None] = {}
    result_digests: dict[str, str | None] = {}
    statuses: dict[str, Literal["PASS", "FAIL", "ERROR"]] = {}
    protocol_errors: dict[str, str | None] = {}

    for command in commands:
        if command.run_kind != run_kind:
            raise ValueError("a Harbor command is outside this protocol phase")
        if (
            command.task_content_digest
            != protocol.task_content_digests[command.instance_id]
        ):
            raise ValueError("a Harbor command has an unregistered task-content digest")
        if command.task_checksum != protocol.task_checksums[command.instance_id]:
            raise ValueError("a Harbor command has an unregistered task checksum")
        if command.task_image_digest != protocol.task_image_digests[command.instance_id]:
            raise ValueError("a Harbor command has an unregistered task-image digest")
        expected_kwargs = _validate_command_contract(
            protocol,
            command,
            protocol_file=protocol_file,
        )
        registered = _registered_attempt(
            protocol,
            protocol_sha256=protocol_sha256,
            attempt_id=command.attempt_id,
            run_kind=run_kind,
            instance_id=command.instance_id,
        )
        if registered.command_sha256 != command.command_sha256:
            raise ValueError("the host registration contains a different command")
        control_state = _attempt_control_state(
            protocol,
            command,
            protocol_sha256=protocol_sha256,
        )

        job_dir = Path(command.job_dir)
        config_path = job_dir / "config.json"
        lock_path = job_dir / "lock.json"
        result_path = job_dir / "result.json"
        control_started_digest: str | None = None
        envelope_digest: str | None = None
        model_marker: ModelStartedMarker | None = None
        terminal_payload: bytes | None = None
        terminal_digest: str | None = None

        if control_state != "enveloped":
            with open_attempt_store(
                protocol.output_root,
                protocol.evaluation_id,
                command.attempt_id,
            ) as store:
                if store.has(_COMMAND_STARTED):
                    started_payload = store.read(_COMMAND_STARTED)
                    try:
                        started = CommandStartedMarker.model_validate_json(
                            started_payload
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "a partial command-start marker is invalid"
                        ) from exc
                    if (
                        started.evaluation_id != protocol.evaluation_id
                        or started.attempt_id != command.attempt_id
                        or started.run_kind != run_kind
                        or started.instance_id != command.instance_id
                        or started.command_sha256 != command.command_sha256
                    ):
                        raise ValueError(
                            "a partial command-start marker changed its binding"
                        )
                    control_started_digest = hashlib.sha256(
                        started_payload
                    ).hexdigest()
                if store.has(_TERMINAL_RECORD):
                    terminal_payload = store.read(_TERMINAL_RECORD)
                    terminal_digest = hashlib.sha256(terminal_payload).hexdigest()

            detail = (
                "registered Harbor command was not started"
                if control_state == "pending"
                else "registered Harbor command started without a complete envelope"
            )
            observed.append(command.instance_id)
            event_digests[command.instance_id] = None
            image_ids[command.instance_id] = None
            started_digests[command.instance_id] = control_started_digest
            envelope_digests[command.instance_id] = None
            terminal_digests[command.instance_id] = terminal_digest
            config_digests[command.instance_id] = (
                sha256_file(config_path) if config_path.is_file() else None
            )
            lock_digests[command.instance_id] = (
                sha256_file(lock_path) if lock_path.is_file() else None
            )
            job_result_digests[command.instance_id] = (
                sha256_file(result_path) if result_path.is_file() else None
            )
            result_digests[command.instance_id] = None
            statuses[command.instance_id] = "ERROR"
            protocol_errors[command.instance_id] = detail
            job_dirs.append(str(job_dir.resolve()))
            continue

        (
            envelope,
            control_started_digest,
            envelope_digest,
            model_marker,
            terminal_payload,
            terminal_digest,
        ) = _load_attempt_control(
            protocol,
            command,
            protocol_sha256=protocol_sha256,
        )

        trial_result: Mapping[str, Any] | None = None
        trial_result_digest: str | None = None
        status: Literal["PASS", "FAIL", "ERROR"] = "ERROR"
        protocol_error: str | None
        if not all(path.is_file() for path in (config_path, lock_path, result_path)):
            config_digest = sha256_file(config_path) if config_path.is_file() else None
            lock_digest = sha256_file(lock_path) if lock_path.is_file() else None
            job_result_digest = (
                sha256_file(result_path) if result_path.is_file() else None
            )
            stage = envelope.failure_stage or "result_persistence"
            protocol_error = (
                f"Harbor command {envelope.outcome} before complete job evidence "
                f"(stage={stage})"
            )
        else:
            try:
                config_payload = config_path.read_bytes()
                lock_payload = lock_path.read_bytes()
                job_result_payload = result_path.read_bytes()
                config = json.loads(config_payload)
                job_lock = json.loads(lock_payload)
                job_result = json.loads(job_result_payload)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Harbor job output is unreadable: {job_dir}") from exc
            config_digest = hashlib.sha256(config_payload).hexdigest()
            lock_digest = hashlib.sha256(lock_payload).hexdigest()
            job_result_digest = hashlib.sha256(job_result_payload).hexdigest()
            if not all(
                isinstance(value, Mapping) for value in (config, job_lock, job_result)
            ):
                raise ValueError("Harbor job config, lock, and result must be JSON objects")
            configured = _configured_task_names(config)
            if configured != [command.instance_id]:
                raise ValueError(
                    f"Harbor config selected {configured!r}, "
                    f"expected {command.instance_id!r}"
                )
            _validate_job_config(
                config,
                protocol=protocol,
                expected_kwargs=expected_kwargs,
            )
            _validate_job_lock(
                job_lock,
                protocol=protocol,
                command=command,
                expected_kwargs=expected_kwargs,
            )
            if config.get("n_attempts", 1) != 1:
                raise ValueError("Harbor config changed n_attempts")
            retry = config.get("retry") or {}
            if not isinstance(retry, Mapping) or retry.get("max_retries", 0) != 0:
                raise ValueError("Harbor-level trial retries must remain disabled")
            if job_result.get("n_total_trials") != 1:
                raise ValueError("each Harbor job must report exactly one trial")
            stats = job_result.get("stats")
            if not isinstance(stats, Mapping) or stats.get("n_retries") != 0:
                raise ValueError("Harbor result recorded an unexpected trial retry")
            if (
                stats.get("n_completed_trials") != 1
                or stats.get("n_running_trials") != 0
                or stats.get("n_pending_trials") != 0
            ):
                protocol_error = "Harbor job did not reach a completed one-trial state"
            else:
                try:
                    trial_dir, loaded_trial, trial_result_digest = (
                        _read_single_trial_result(job_dir)
                    )
                except ValueError:
                    protocol_error = "Harbor job omitted its single official trial result"
                else:
                    trial_result = loaded_trial
                    embedded_trials = job_result.get("trial_results")
                    if embedded_trials is not None and (
                        not isinstance(embedded_trials, list)
                        or len(embedded_trials) != 1
                        or embedded_trials[0] != trial_result
                    ):
                        raise ValueError("Harbor job and nested trial results disagree")
                    task_name = trial_result.get("task_name")
                    if task_name != command.instance_id:
                        raise ValueError(
                            f"Harbor ran {task_name!r}, expected {command.instance_id!r}"
                        )
                    status, _, protocol_error = _official_trial_outcome(trial_result)
                    if envelope.outcome != "completed":
                        status = "ERROR"
                        stage = envelope.failure_stage or "unknown"
                        protocol_error = (
                            f"Harbor host command {envelope.outcome} (stage={stage})"
                        )

        event_digest, image_id = _validate_trial_evidence(
            trial_result,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            command=command,
            expected_kwargs=expected_kwargs,
            official_status=status,
            model_marker=model_marker,
            terminal_payload=terminal_payload,
            terminal_sha256=terminal_digest,
        )

        observed.append(command.instance_id)
        event_digests[command.instance_id] = event_digest
        image_ids[command.instance_id] = image_id
        started_digests[command.instance_id] = control_started_digest
        envelope_digests[command.instance_id] = envelope_digest
        terminal_digests[command.instance_id] = terminal_digest
        config_digests[command.instance_id] = config_digest
        lock_digests[command.instance_id] = lock_digest
        job_result_digests[command.instance_id] = job_result_digest
        result_digests[command.instance_id] = trial_result_digest
        statuses[command.instance_id] = status
        protocol_errors[command.instance_id] = protocol_error
        job_dirs.append(str(job_dir.resolve()))

    manifest = HarborExecutionManifest(
        dataset_version=protocol.dataset_version,
        run_kind=run_kind,
        protocol_sha256=protocol_sha256,
        expected_instance_ids=expected,
        observed_instance_ids=tuple(observed),
        task_content_digests={
            item: protocol.task_content_digests[item] for item in expected
        },
        task_checksums={item: protocol.task_checksums[item] for item in expected},
        task_image_digests={item: protocol.task_image_digests[item] for item in expected},
        codex_events_sha256={item: event_digests[item] for item in expected},
        container_image_ids={item: image_ids[item] for item in expected},
        command_started_sha256={item: started_digests[item] for item in expected},
        command_envelope_sha256={item: envelope_digests[item] for item in expected},
        terminal_record_sha256={item: terminal_digests[item] for item in expected},
        job_config_sha256={item: config_digests[item] for item in expected},
        job_lock_sha256={item: lock_digests[item] for item in expected},
        job_result_sha256={item: job_result_digests[item] for item in expected},
        trial_result_sha256={item: result_digests[item] for item in expected},
        official_status={item: statuses[item] for item in expected},
        protocol_errors={item: protocol_errors[item] for item in expected},
        job_dirs=tuple(job_dirs),
    )
    if manifest_path is not None:
        target = Path(manifest_path)
        _write_execution_manifest_once(target, manifest)
    return manifest


def seal_smoke_phase(
    protocol: TerminalBenchProtocol,
    commands: Sequence[HarborRunCommand],
    *,
    protocol_path: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[SmokeSeal, HarborExecutionManifest]:
    """Seal three infrastructure-valid smoke runs before any scored command."""
    with evaluation_lock(protocol.output_root, protocol.evaluation_id):
        return _seal_smoke_phase_locked(
            protocol,
            commands,
            protocol_path=protocol_path,
            manifest_path=manifest_path,
        )


def _seal_smoke_phase_locked(
    protocol: TerminalBenchProtocol,
    commands: Sequence[HarborRunCommand],
    *,
    protocol_path: str | Path,
    manifest_path: str | Path | None,
) -> tuple[SmokeSeal, HarborExecutionManifest]:
    manifest = validate_harbor_results(
        protocol,
        "smoke",
        commands,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
    )
    if any(status == "ERROR" for status in manifest.official_status.values()):
        raise ValueError("the smoke phase contains an ERROR and cannot be sealed")
    terminal_digests = manifest.terminal_record_sha256
    if any(value is None for value in terminal_digests.values()):
        raise ValueError("the smoke phase omitted host-only terminal evidence")
    smoke_ids = protocol.subset.smoke_instance_ids
    if len(smoke_ids) != 3:
        raise ValueError("the smoke phase must contain exactly three registered tasks")
    terminal_records = {
        instance_id: cast(str, terminal_digests[instance_id])
        for instance_id in smoke_ids
    }
    with open_control_store(protocol.output_root, protocol.evaluation_id) as store:
        if store.has(_SMOKE_MANIFEST):
            manifest_payload = store.read(_SMOKE_MANIFEST)
            try:
                recorded_manifest = HarborExecutionManifest.model_validate_json(
                    manifest_payload
                )
            except ValueError as exc:
                raise ControlStoreError("the smoke manifest is invalid") from exc
            if recorded_manifest != manifest:
                raise ControlStoreError(
                    "the existing smoke manifest differs from current evidence"
                )
            manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        else:
            manifest_sha256 = store.write_json_once(_SMOKE_MANIFEST, manifest)

        if store.has(_SMOKE_SEAL):
            try:
                recorded_seal = SmokeSeal.model_validate_json(store.read(_SMOKE_SEAL))
            except ValueError as exc:
                raise ControlStoreError("the smoke seal is invalid") from exc
            if (
                recorded_seal.evaluation_id != protocol.evaluation_id
                or recorded_seal.protocol_sha256 != manifest.protocol_sha256
                or recorded_seal.manifest_sha256 != manifest_sha256
                or recorded_seal.smoke_instance_ids != smoke_ids
                or recorded_seal.terminal_record_sha256 != terminal_records
            ):
                raise ControlStoreError(
                    "the existing smoke seal differs from current evidence"
                )
            return recorded_seal, manifest

        seal = SmokeSeal(
            evaluation_id=protocol.evaluation_id,
            protocol_sha256=manifest.protocol_sha256,
            manifest_sha256=manifest_sha256,
            smoke_instance_ids=(smoke_ids[0], smoke_ids[1], smoke_ids[2]),
            terminal_record_sha256=terminal_records,
            sealed_at=now().isoformat(),
        )
        store.write_json_once(_SMOKE_SEAL, seal)
    return seal, manifest


def _validate_command_contract(
    protocol: TerminalBenchProtocol,
    command: HarborRunCommand,
    *,
    protocol_file: Path,
) -> dict[str, Any]:
    """Rebuild the exact command and bind its local artifacts to the protocol."""
    raw_kwargs = _argv_key_values(command.argv, "--agent-kwarg")
    expected_keys = {
        "wheel_path",
        "codex_binary_path",
        "protocol_path",
        "reasoning_effort",
        "instance_id",
        "run_kind",
        "attempt_id",
    }
    if set(raw_kwargs) != expected_keys:
        raise ValueError("Harbor command changed the registered agent kwargs")
    wheel = Path(raw_kwargs["wheel_path"])
    codex_binary = Path(raw_kwargs["codex_binary_path"])
    configured_protocol = Path(raw_kwargs["protocol_path"])
    if not all(path.is_absolute() for path in (wheel, codex_binary, configured_protocol)):
        raise ValueError("Harbor artifact paths must be absolute")
    if configured_protocol != protocol_file:
        raise ValueError("Harbor command points at a different protocol")
    if not wheel.is_file() or sha256_file(wheel) != protocol.wheel_sha256:
        raise ValueError("Harbor command wheel no longer matches the protocol")
    if (
        not codex_binary.is_file()
        or sha256_file(codex_binary) != protocol.codex_binary_sha256
    ):
        raise ValueError("Harbor command Codex binary no longer matches the protocol")

    jobs_dir = Path(_single_argv_value(command.argv, "--jobs-dir"))
    job_name = _single_argv_value(command.argv, "--job-name")
    if (
        not jobs_dir.is_absolute()
        or jobs_dir != Path(protocol.output_root)
        or Path(command.job_dir) != jobs_dir / job_name
        or command.evaluation_id != protocol.evaluation_id
        or raw_kwargs["attempt_id"] != command.attempt_id
    ):
        raise ValueError("Harbor command job directory is inconsistent")
    expected_argv = _harbor_argv(
        protocol,
        command.run_kind,
        command.instance_id,
        protocol_file=protocol_file,
        wheel=wheel,
        codex_binary=codex_binary,
        output_root=jobs_dir,
        job_name=job_name,
        attempt_id=command.attempt_id,
    )
    if command.argv != expected_argv:
        raise ValueError("Harbor command differs from the release protocol")
    return {
        "wheel_path": str(wheel),
        "codex_binary_path": str(codex_binary),
        "protocol_path": str(protocol_file),
        "reasoning_effort": protocol.reasoning_effort,
        "instance_id": command.instance_id,
        "run_kind": command.run_kind,
        "attempt_id": command.attempt_id,
    }


def _single_argv_value(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"Harbor command must contain exactly one {flag}")
    return argv[positions[0] + 1]


def _argv_key_values(argv: Sequence[str], flag: str) -> dict[str, str]:
    values: dict[str, str] = {}
    positions = [index for index, value in enumerate(argv) if value == flag]
    for position in positions:
        if position + 1 >= len(argv) or "=" not in argv[position + 1]:
            raise ValueError(f"Harbor command contains a malformed {flag}")
        key, value = argv[position + 1].split("=", 1)
        if not key or key in values:
            raise ValueError(f"Harbor command contains a duplicate {flag}")
        values[key] = value
    return values


def _validate_job_config(
    config: Mapping[str, Any],
    *,
    protocol: TerminalBenchProtocol,
    expected_kwargs: Mapping[str, Any],
) -> None:
    if config.get("n_concurrent_trials") != 1:
        raise ValueError("Harbor config changed trial concurrency")
    if (
        config.get("agent_timeout_multiplier")
        != protocol.budgets.harbor_agent_timeout_multiplier
    ):
        raise ValueError("Harbor config changed the agent timeout envelope")
    datasets = config.get("datasets")
    dataset = datasets[0] if isinstance(datasets, list) and len(datasets) == 1 else None
    if not isinstance(dataset, Mapping) or dataset.get("name") != DATASET:
        raise ValueError("Harbor config changed the official dataset")
    if dataset.get("ref") != protocol.dataset_version:
        raise ValueError("Harbor config changed the immutable dataset version")
    if dataset.get("n_tasks") != 1:
        raise ValueError("Harbor config changed the one-task job limit")
    agents = config.get("agents")
    if not isinstance(agents, list) or len(agents) != 1:
        raise ValueError("Harbor config must contain exactly one agent")
    _validate_agent_config(
        agents[0],
        protocol=protocol,
        expected_kwargs=expected_kwargs,
    )
    if "environment" in config:
        _validate_docker_environment(config["environment"])


def _validate_job_lock(
    job_lock: Mapping[str, Any],
    *,
    protocol: TerminalBenchProtocol,
    command: HarborRunCommand,
    expected_kwargs: Mapping[str, Any],
) -> None:
    if job_lock.get("schema_version") != 2:
        raise ValueError("Harbor lock schema is not the pinned 0.20 contract")
    harbor = job_lock.get("harbor")
    if not isinstance(harbor, Mapping) or harbor.get("version") != protocol.harbor_version:
        raise ValueError("Harbor lock does not prove the preregistered Harbor version")
    if job_lock.get("n_concurrent_trials") != 1:
        raise ValueError("Harbor lock changed trial concurrency")
    retry = job_lock.get("retry")
    if not isinstance(retry, Mapping) or retry.get("max_retries") != 0:
        raise ValueError("Harbor lock changed the retry policy")
    trials = job_lock.get("trials")
    if not isinstance(trials, list) or len(trials) != 1:
        raise ValueError("Harbor lock must contain exactly one resolved trial")
    trial = trials[0]
    if not isinstance(trial, Mapping):
        raise ValueError("Harbor trial lock must be a JSON object")
    if (
        trial.get("agent_timeout_multiplier")
        != protocol.budgets.harbor_agent_timeout_multiplier
    ):
        raise ValueError("Harbor lock changed the agent timeout envelope")
    task = trial.get("task")
    if (
        not isinstance(task, Mapping)
        or task.get("name") != command.instance_id
        or task.get("digest") != command.task_content_digest
    ):
        raise ValueError("Harbor lock does not bind the registered task")
    _validate_agent_config(
        trial.get("agent"),
        protocol=protocol,
        expected_kwargs=expected_kwargs,
    )
    _validate_docker_environment(trial.get("environment"))


def _validate_agent_config(
    raw_agent: Any,
    *,
    protocol: TerminalBenchProtocol,
    expected_kwargs: Mapping[str, Any],
) -> None:
    if not isinstance(raw_agent, Mapping):
        raise ValueError("Harbor agent config must be a JSON object")
    if (
        raw_agent.get("name") != AGENT_IMPORT_PATH
        or raw_agent.get("import_path") not in (None, "")
    ):
        raise ValueError("Harbor config changed the LHA agent import")
    if raw_agent.get("model_name") != protocol.model:
        raise ValueError("Harbor config changed the preregistered model")
    if raw_agent.get("kwargs") != dict(expected_kwargs):
        raise ValueError("Harbor config changed critical agent kwargs")
    for field in (
        "skills",
        "extra_allowed_hosts",
        "include_logs",
        "exclude_logs",
        "env",
        "mcp_servers",
    ):
        if raw_agent.get(field) not in (None, [], {}):
            raise ValueError(f"Harbor config unexpectedly set agent {field}")
    for field in (
        "override_timeout_sec",
        "override_setup_timeout_sec",
        "max_timeout_sec",
        "load_trajectory",
    ):
        if raw_agent.get(field) is not None:
            raise ValueError(f"Harbor config unexpectedly set agent {field}")
    if raw_agent.get("resume_trajectory") not in (None, False):
        raise ValueError("Harbor config enabled trajectory resume")


def _broker_receipt_proves_clean_success(
    receipt: Mapping[str, Any],
    accepted_requests: object,
    reconnect_notices: object,
) -> bool:
    """Allow only preregistered in-process recovery within the fixed retry budget."""
    if not _broker_receipt_diagnostics_are_valid(receipt):
        return False
    transport_errors = cast(Mapping[str, int], receipt["upstream_transport_errors"])
    stream_errors = cast(Mapping[str, int], receipt["upstream_stream_errors"])
    transport_count = sum(transport_errors.values())
    stream_count = sum(stream_errors.values())
    upstream_attempts = cast(int, receipt["upstream_attempts"])
    stream_retries_used = cast(int, receipt["stream_retries_used"])
    observed_content_types = cast(list[str], receipt["observed_content_types"])
    reasons = cast(Mapping[str, int], receipt["rejection_reasons"])
    return (
        type(accepted_requests) is int
        and accepted_requests >= 1
        and accepted_requests == receipt.get("downstream_accepted_requests")
        and transport_count <= cast(int, receipt["request_retry_limit"])
        and stream_retries_used <= cast(int, receipt["stream_retry_limit"])
        and stream_count == stream_retries_used
        and type(reconnect_notices) is int
        and reconnect_notices == 0
        and set(transport_errors) <= BROKER_RECOVERABLE_TRANSPORT_ERRORS
        and set(stream_errors) <= BROKER_RECOVERABLE_STREAM_ERRORS
        and receipt.get("rejected_requests") == transport_count
        and reasons
        in (
            {},
            {"upstream_transport_exception": transport_count},
        )
        and receipt.get("upstream_statuses")
        == {"200": upstream_attempts - transport_count}
        and (
            not observed_content_types
            or content_types_are_sse(observed_content_types)
        )
        and receipt.get("upstream_error") is None
    )


def _broker_receipt_diagnostics_are_valid(receipt: Mapping[str, Any]) -> bool:
    rejected = receipt.get("rejected_requests")
    reasons = receipt.get("rejection_reasons")
    accepted = receipt.get("downstream_accepted_requests")
    upstream_attempts = receipt.get("upstream_attempts")
    statuses = receipt.get("upstream_statuses")
    stream_retries_used = receipt.get("stream_retries_used")
    stream_retried_requests = receipt.get("stream_retried_requests")
    max_stream_retries_on_request = receipt.get(
        "max_stream_retries_on_request"
    )
    request_retry_limit = receipt.get("request_retry_limit")
    stream_retry_limit = receipt.get("stream_retry_limit")
    stream_retry_limit_per_request = receipt.get(
        "stream_retry_limit_per_request"
    )
    transport_errors = receipt.get("upstream_transport_errors")
    stream_errors = receipt.get("upstream_stream_errors")
    observed_content_types = receipt.get("observed_content_types")
    if (
        type(accepted) is not int
        or not 0 <= accepted <= BROKER_MAX_REQUESTS
        or type(rejected) is not int
        or rejected < 0
        or type(upstream_attempts) is not int
        or not 0
        <= upstream_attempts
        <= BROKER_MAX_REQUESTS + BROKER_STREAM_RETRY_LIMIT
        or type(stream_retries_used) is not int
        or not 0 <= stream_retries_used <= BROKER_STREAM_RETRY_LIMIT
        or type(stream_retried_requests) is not int
        or not 0 <= stream_retried_requests <= accepted
        or type(max_stream_retries_on_request) is not int
        or not 0
        <= max_stream_retries_on_request
        <= BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
        or request_retry_limit != 1
        or stream_retry_limit != BROKER_STREAM_RETRY_LIMIT
        or stream_retry_limit_per_request
        != BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
        or receipt.get("max_buffered_response_bytes")
        != BROKER_MAX_BUFFERED_RESPONSE_BYTES
        or not isinstance(observed_content_types, list)
        or len(observed_content_types) > BROKER_MAX_OBSERVED_CONTENT_TYPES
        or any(
            not isinstance(item, str)
            or not 0 < len(item) <= BROKER_MAX_OBSERVED_CONTENT_TYPE_CHARS
            or not item.isascii()
            or not item.isprintable()
            for item in observed_content_types
        )
        or not isinstance(reasons, Mapping)
        or any(
            not isinstance(reason, str)
            or reason not in BROKER_REJECTION_REASONS
            or type(count) is not int
            or count < 1
            for reason, count in reasons.items()
        )
        or sum(cast(int, count) for count in reasons.values()) != rejected
        or not isinstance(statuses, Mapping)
        or any(
            not isinstance(status, str)
            or not status.isdecimal()
            or not 100 <= int(status) <= 599
            or type(count) is not int
            or count < 1
            for status, count in statuses.items()
        )
    ):
        return False
    if (
        (stream_retries_used == 0)
        != (
            stream_retried_requests == 0
            and max_stream_retries_on_request == 0
        )
        or stream_retries_used < stream_retried_requests
        or max_stream_retries_on_request > stream_retries_used
        or stream_retries_used
        > stream_retried_requests * max_stream_retries_on_request
    ):
        return False
    error_maps = (transport_errors, stream_errors)
    if any(
        not isinstance(errors, Mapping)
        or any(
            not isinstance(error_type, str)
            or _BROKER_ERROR_TYPE_RE.fullmatch(error_type) is None
            or type(count) is not int
            or count < 1
            for error_type, count in errors.items()
        )
        or sum(cast(int, count) for count in errors.values())
        > BROKER_MAX_REQUESTS + BROKER_STREAM_RETRY_LIMIT
        for errors in error_maps
    ):
        return False
    assert isinstance(transport_errors, Mapping)
    assert isinstance(stream_errors, Mapping)
    transport_rejections = reasons.get("upstream_transport_exception", 0)
    if sum(cast(int, count) for count in transport_errors.values()) != (
        transport_rejections
    ):
        return False
    transport_error_count = sum(
        cast(int, count) for count in transport_errors.values()
    )
    invalid_statuses = cast(int, reasons.get("upstream_invalid_status", 0))
    attempt_timeouts = cast(int, reasons.get("upstream_attempt_timeout", 0))
    stream_error_count = sum(
        cast(int, count) for count in stream_errors.values()
    )
    final_stream_failures = sum(
        cast(int, reasons.get(reason, 0))
        for reason in _BROKER_FINAL_STREAM_REASONS
    )
    return (
        upstream_attempts
        == sum(cast(int, count) for count in statuses.values())
        + transport_error_count
        + invalid_statuses
        and accepted + stream_retries_used == upstream_attempts + attempt_timeouts
        and stream_error_count == stream_retries_used + final_stream_failures
        and stream_error_count
        <= sum(
            cast(int, count)
            for status, count in statuses.items()
            if 200 <= int(status) < 400
        )
    )


def _validate_docker_environment(raw_environment: Any) -> None:
    if not isinstance(raw_environment, Mapping):
        raise ValueError("Harbor environment config must be a JSON object")
    if (
        raw_environment.get("type") != "docker"
        or raw_environment.get("import_path") not in (None, "")
        or raw_environment.get("kwargs") not in (None, {})
    ):
        raise ValueError("formal results require Harbor's built-in Docker environment")


def _validate_trial_evidence(
    trial_result: Mapping[str, Any] | None,
    *,
    protocol: TerminalBenchProtocol,
    protocol_sha256: str,
    command: HarborRunCommand,
    expected_kwargs: Mapping[str, Any],
    official_status: Literal["PASS", "FAIL", "ERROR"],
    model_marker: ModelStartedMarker | None,
    terminal_payload: bytes | None,
    terminal_sha256: str | None,
) -> tuple[str | None, str | None]:
    if terminal_payload is None:
        if official_status != "ERROR":
            raise ValueError("PASS and FAIL rows require host-only terminal evidence")
        return None, None
    if terminal_sha256 is None:
        raise ValueError("terminal evidence is present without its digest")
    try:
        provenance = TerminalBenchAgentProvenance.model_validate_json(terminal_payload)
    except ValueError as exc:
        raise ValueError("host-only terminal provenance is invalid") from exc

    expected_task_digest = protocol.task_content_digests[command.instance_id]
    expected_digest = protocol.task_image_digests[command.instance_id]
    if (
        provenance.evaluation_id != protocol.evaluation_id
        or provenance.attempt_id != command.attempt_id
        or provenance.run_kind != command.run_kind
        or provenance.instance_id != command.instance_id
        or provenance.dataset_version != protocol.dataset_version
        or provenance.model != protocol.model
        or provenance.reasoning_effort != protocol.reasoning_effort
        or provenance.harbor_version != protocol.harbor_version
        or provenance.codex_cli_version != protocol.codex_cli_version
        or provenance.codex_target != protocol.codex_target
        or provenance.codex_binary_sha256 != protocol.codex_binary_sha256
        or provenance.broker_image_id != protocol.broker_image_id
        or provenance.task_content_digest != expected_task_digest
        or provenance.task_image_digest != expected_digest
        or provenance.wheel_sha256 != protocol.wheel_sha256
        or provenance.protocol_sha256 != protocol_sha256
        or provenance.subset != protocol.subset
        or provenance.budgets != protocol.budgets
    ):
        raise ValueError("host-only terminal provenance does not match the protocol")
    expected_smoke_seal = (
        _validated_smoke_seal(protocol, protocol_sha256=protocol_sha256)
        if command.run_kind == "scored"
        else None
    )
    if provenance.smoke_seal_sha256 != expected_smoke_seal:
        raise ValueError("terminal provenance does not match the sealed smoke phase")
    if provenance.model_started != (model_marker is not None):
        raise ValueError("terminal provenance disagrees with the model-start marker")
    if model_marker is not None:
        if (
            provenance.image_attestation is None
            or model_marker.protocol_sha256 != protocol_sha256
            or model_marker.container_id
            != provenance.image_attestation.container_id
        ):
            raise ValueError("model-start evidence does not match runtime provenance")
    if (
        provenance.image_attestation is not None
        and not provenance.image_attestation.proves(expected_digest)
    ):
        raise ValueError("runtime Docker evidence does not prove the registered image")

    events_sha256: str | None = None
    event_stream: str | None = None
    receipt: Mapping[str, Any] | None = None
    with open_attempt_store(
        protocol.output_root,
        protocol.evaluation_id,
        command.attempt_id,
    ) as store:
        if provenance.codex_events_sha256 is not None:
            try:
                event_payload = store.read(_CODEX_EVENTS)
                event_stream = event_payload.decode("utf-8")
            except (ControlStoreError, UnicodeDecodeError) as exc:
                raise ValueError("host-only Codex JSONL is unreadable") from exc
            events_sha256 = hashlib.sha256(event_payload).hexdigest()
            if events_sha256 != provenance.codex_events_sha256:
                raise ValueError("host-only Codex JSONL digest changed")
        elif store.has(_CODEX_EVENTS):
            raise ValueError("the attempt contains unbound Codex JSONL evidence")

        if provenance.broker_receipt_sha256 is not None:
            try:
                receipt_payload = store.read(_BROKER_RECEIPT)
                receipt = json.loads(receipt_payload)
            except (ControlStoreError, json.JSONDecodeError) as exc:
                raise ValueError("host-only broker receipt is unreadable") from exc
            if (
                hashlib.sha256(receipt_payload).hexdigest()
                != provenance.broker_receipt_sha256
            ):
                raise ValueError("host-only broker receipt digest changed")
            if (
                not isinstance(receipt, Mapping)
                or set(receipt) != _BROKER_RECEIPT_KEYS
                or receipt.get("schema_version") != 5
                or receipt.get("type") != "terminal_proxy_receipt"
                or receipt.get("evaluation_id") != protocol.evaluation_id
                or receipt.get("attempt_id") != command.attempt_id
                or receipt.get("source_container_id")
                != (
                    provenance.image_attestation.container_id
                    if provenance.image_attestation is not None
                    else None
                )
                or receipt.get("ttl_s") != protocol.budgets.broker_ttl_s
                or receipt.get("max_requests")
                != protocol.budgets.max_model_requests
                or receipt.get("max_buffered_response_bytes")
                != BROKER_MAX_BUFFERED_RESPONSE_BYTES
                or receipt.get("request_retry_limit")
                != protocol.budgets.request_max_retries
                or receipt.get("stream_retry_limit")
                != protocol.budgets.broker_stream_max_retries
                or receipt.get("stream_retry_limit_per_request")
                != protocol.budgets.broker_stream_max_retries_per_request
                or receipt.get("downstream_accepted_requests")
                != provenance.broker_accepted_requests
                or receipt.get("revoked") is not True
                or receipt.get("outcome") != "sigterm"
                or not _broker_receipt_diagnostics_are_valid(receipt)
            ):
                raise ValueError("host-only broker receipt changed its binding")
        elif store.has(_BROKER_RECEIPT):
            raise ValueError("the attempt contains an unbound broker receipt")

    audit: CodexRunAudit | None = None
    if provenance.codex_outcome == "success":
        if event_stream is None:
            raise ValueError("successful terminal evidence omitted Codex JSONL")
        try:
            audit = audit_codex_jsonl(
                event_stream,
                max_tool_calls=protocol.budgets.max_tool_calls,
                max_reconnect_notices=protocol.budgets.stream_max_retries,
            )
        except RuntimeError as exc:
            raise ValueError("Harbor trial Codex JSONL failed audit") from exc
        if provenance.codex_audit != audit:
            raise ValueError("host-only Codex audit changed")
        if (
            receipt is None
            or not _broker_receipt_proves_clean_success(
                receipt,
                receipt.get("downstream_accepted_requests"),
                audit.reconnect_notices,
            )
        ):
            raise ValueError(
                "successful Codex evidence does not match its registered "
                "upstream recovery record"
            )
    elif provenance.codex_outcome == "protocol_error":
        if event_stream is None:
            raise ValueError("protocol-error evidence omitted the rejected JSONL")
        try:
            audit_codex_jsonl(
                event_stream,
                max_tool_calls=protocol.budgets.max_tool_calls,
                max_reconnect_notices=protocol.budgets.stream_max_retries,
            )
        except RuntimeError:
            pass
        else:
            if provenance.codex_failure_kind == "codex_capability_exposed":
                pass
            else:
                raise ValueError("Codex JSONL was valid despite protocol_error provenance")

    if official_status != "ERROR" and (
        provenance.codex_outcome != "success"
        or provenance.broker_cleanup_state != "succeeded"
        or provenance.container_quiescence != "restarted"
    ):
        raise ValueError(
            "PASS and FAIL rows require a successful audited run, revoked broker, "
            "and in-place container restart"
        )

    if trial_result is None:
        return (
            events_sha256,
            (
                provenance.image_attestation.image_id
                if provenance.image_attestation is not None
                else None
            ),
        )

    config = trial_result.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Harbor trial result omitted its resolved config")
    _validate_agent_config(
        config.get("agent"),
        protocol=protocol,
        expected_kwargs=expected_kwargs,
    )
    _validate_docker_environment(config.get("environment"))
    if trial_result.get("task_checksum") != protocol.task_checksums[command.instance_id]:
        raise ValueError("Harbor trial result changed the registered task content")

    agent_info = trial_result.get("agent_info")
    if isinstance(agent_info, Mapping):
        if agent_info.get("name") != "lha":
            raise ValueError("Harbor trial result was produced by a different agent")
        model_info = agent_info.get("model_info")
        expected_provider: str | None = None
        expected_model_name = protocol.model
        if "/" in protocol.model:
            expected_provider, expected_model_name = protocol.model.split("/", 1)
        if (
            not isinstance(model_info, Mapping)
            or model_info.get("name") != expected_model_name
            or model_info.get("provider") != expected_provider
        ):
            raise ValueError("Harbor trial result was produced by a different model")
        if agent_info.get("version") != provenance.lha_version:
            raise ValueError("Harbor agent version and wheel provenance disagree")
    elif official_status != "ERROR":
        raise ValueError("Harbor trial result omitted completed agent identity")

    agent_result = trial_result.get("agent_result")
    if provenance.codex_outcome == "success":
        assert audit is not None
        assert events_sha256 is not None
        assert provenance.image_attestation is not None
        if not isinstance(agent_result, Mapping):
            if official_status != "ERROR":
                raise ValueError("Harbor trial omitted the completed agent result")
        else:
            metadata = agent_result.get("metadata")
            expected_metadata = _agent_metadata(
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                instance_id=command.instance_id,
                run_kind=command.run_kind,
                audit=audit,
                codex_events_sha256=events_sha256,
                image_attestation=provenance.image_attestation,
                terminal_record_sha256=terminal_sha256,
                provenance=provenance,
            )
            if metadata != expected_metadata:
                raise ValueError("Harbor agent metadata does not match host-only evidence")
            for field, value in (
                ("n_input_tokens", audit.input_tokens),
                ("n_cache_tokens", audit.cached_input_tokens),
                ("n_output_tokens", audit.output_tokens),
            ):
                if agent_result.get(field) != value:
                    raise ValueError("Harbor token usage does not match the Codex audit")
    elif official_status != "ERROR":
        raise ValueError("Harbor trial has no successful Codex evidence")

    image_id = (
        provenance.image_attestation.image_id
        if provenance.image_attestation is not None
        else None
    )
    return events_sha256, image_id


def _agent_metadata(
    *,
    protocol: TerminalBenchProtocol,
    protocol_sha256: str,
    instance_id: str,
    run_kind: Literal["smoke", "scored"],
    audit: CodexRunAudit,
    codex_events_sha256: str,
    image_attestation: DockerImageAttestation,
    terminal_record_sha256: str,
    provenance: TerminalBenchAgentProvenance,
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "evaluation_id": protocol.evaluation_id,
        "attempt_id": provenance.attempt_id,
        "dataset_version": protocol.dataset_version,
        "agent_import_path": AGENT_IMPORT_PATH,
        "instance_id": instance_id,
        "run_kind": run_kind,
        "model": protocol.model,
        "reasoning_effort": protocol.reasoning_effort,
        "harbor_version": protocol.harbor_version,
        "codex_cli_version": protocol.codex_cli_version,
        "codex_binary_sha256": protocol.codex_binary_sha256,
        "wheel_sha256": protocol.wheel_sha256,
        "protocol_sha256": protocol_sha256,
        "task_content_digest": protocol.task_content_digests[instance_id],
        "task_image_digest": protocol.task_image_digests[instance_id],
        "container_id": image_attestation.container_id,
        "container_image_id": image_attestation.image_id,
        "container_configured_image": image_attestation.configured_image,
        "container_repo_digests": list(image_attestation.repo_digests),
        "container_compose_project": image_attestation.compose_project,
        "container_network_name": image_attestation.network_name,
        "post_quiescence_container_id": (
            provenance.post_quiescence_attestation.container_id
            if provenance.post_quiescence_attestation is not None
            else None
        ),
        "broker_image_id": provenance.broker_image_id,
        "broker_receipt_sha256": provenance.broker_receipt_sha256,
        "broker_accepted_requests": provenance.broker_accepted_requests,
        "broker_revoked": provenance.broker_revoked,
        "terminal_record_sha256": terminal_record_sha256,
        "codex_events_sha256": codex_events_sha256,
        "codex_event_counts": audit.event_counts,
        "codex_item_counts": audit.item_counts,
        "codex_tool_calls": audit.tool_calls,
        "codex_reconnect_notices": audit.reconnect_notices,
        "codex_reasoning_output_tokens": audit.reasoning_output_tokens,
        "infrastructure_retries_used": 0,
        "smoke_seal_sha256": provenance.smoke_seal_sha256,
    }


def _read_single_trial_result(
    job_dir: Path,
) -> tuple[Path, dict[str, Any], str]:
    """Load Harbor 0.20's one nested ``<trial>/result.json`` record."""
    paths = sorted(
        path
        for path in job_dir.glob("*/result.json")
        if path.parent != job_dir
    )
    if len(paths) != 1:
        raise ValueError("each Harbor job must contain exactly one trial result")
    try:
        raw = paths[0].read_bytes()
        result = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Harbor trial output is unreadable: {paths[0]}") from exc
    if not isinstance(result, dict):
        raise ValueError("Harbor trial result must be a JSON object")
    return paths[0].parent, result, hashlib.sha256(raw).hexdigest()


def _configured_task_names(config: Mapping[str, Any]) -> list[str]:
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("Harbor config must contain exactly one dataset")
    task_names = datasets[0].get("task_names")
    if not isinstance(task_names, list) or not all(
        isinstance(value, str) for value in task_names
    ):
        raise ValueError("Harbor config omitted the exact task filter")
    return task_names


def _require_harbor_docker_environment(environment) -> None:
    """Reject providers whose process and network lifecycle cannot be attested."""
    try:
        from harbor.environments.docker.docker import (  # pyright: ignore[reportMissingImports]
            DockerEnvironment,
        )
    except ImportError as exc:
        raise RuntimeError("Harbor's Docker environment implementation is unavailable") from exc
    if not isinstance(environment, DockerEnvironment):
        raise RuntimeError("formal Terminal-Bench runs require Harbor's Docker environment")


def _contains_exception(
    error: BaseException,
    expected: type[BaseException],
    *,
    _seen: set[int] | None = None,
) -> bool:
    seen = _seen if _seen is not None else set()
    if id(error) in seen:
        return False
    seen.add(id(error))
    if isinstance(error, expected):
        return True
    nested = getattr(error, "exceptions", ())
    if isinstance(nested, tuple) and any(
        isinstance(item, BaseException)
        and _contains_exception(item, expected, _seen=seen)
        for item in nested
    ):
        return True
    return any(
        isinstance(item, BaseException)
        and _contains_exception(item, expected, _seen=seen)
        for item in (error.__cause__, error.__context__)
    )


@contextmanager
def _harbor_stream_line_limit(
    environment: Any,
    *,
    maximum_line_bytes: int,
) -> Iterator[None]:
    """Raise Harbor 0.20's private asyncio line limit within one Codex exec.

    Harbor streams Docker output with ``StreamReader.readline()`` but leaves
    asyncio's 64 KiB default in place. Codex JSONL has its own checked line and
    total byte limits, so the transport must allow one registered line to reach
    that validator. The wrapper is scoped to one environment instance and is
    restored before any cleanup command runs.
    """
    collector = getattr(environment, "_collect_streamed_output", None)
    if collector is None or not callable(collector):
        # Lightweight test doubles do not implement Harbor's private collector.
        yield
        return
    async_collector = cast(Callable[..., Awaitable[Any]], collector)
    instance_values = getattr(environment, "__dict__", None)
    if not isinstance(instance_values, dict):
        raise RuntimeError("Harbor environment cannot scope its stream reader limit")
    had_instance_value = "_collect_streamed_output" in instance_values
    previous_instance_value = instance_values.get("_collect_streamed_output")
    transport_limit = maximum_line_bytes + _HARBOR_STREAM_LINE_HEADROOM_BYTES

    async def collect_with_registered_limit(process, **kwargs):
        reader = getattr(process, "stdout", None)
        if not isinstance(reader, asyncio.StreamReader):
            raise RuntimeError("Harbor streaming process omitted its asyncio reader")
        reader_state = cast(Any, reader)
        previous_limit = reader_state._limit
        reader_state._limit = transport_limit
        try:
            return await async_collector(process, **kwargs)
        finally:
            reader_state._limit = previous_limit

    setattr(environment, "_collect_streamed_output", collect_with_registered_limit)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(
                environment,
                "_collect_streamed_output",
                previous_instance_value,
            )
        else:
            delattr(environment, "_collect_streamed_output")


def build_agent():  # -> type[BaseInstalledAgent]
    """Create the Harbor agent class.  ``harbor`` itself requires Python 3.12+."""
    try:
        from harbor.agents.installed.base import (  # pyright: ignore[reportMissingImports]
            BaseInstalledAgent,
        )
    except ImportError as e:  # pragma: no cover - exercised through a stub in tests
        raise ImportError(
            "harbor is not installed (it needs Python >= 3.12). Run with "
            f"`uvx --python 3.12 --with harbor harbor run -d {DATASET} ...`"
        ) from e

    from .. import __version__

    class LhaAgent(BaseInstalledAgent):
        """One protocol-bound Codex execution inside Harbor's task container."""

        @staticmethod
        def name() -> str:
            return "lha"

        def version(self) -> str | None:
            return __version__

        def __init__(
            self,
            logs_dir,
            wheel_path: str | None = None,
            codex_binary_path: str | None = None,
            auth_path: str | None = None,
            model_name: str | None = None,
            model: str | None = None,
            reasoning_effort: str | None = None,
            protocol_path: str | None = None,
            instance_id: str | None = None,
            run_kind: Literal["smoke", "scored"] | None = None,
            attempt_id: str | None = None,
            *args,
            **kwargs,
        ):
            self._logs_dir = Path(logs_dir)
            self._wheel = wheel_path or os.environ.get("LHA_WHEEL")
            self._codex_binary = codex_binary_path or os.environ.get(
                "LHA_CODEX_BINARY_FILE"
            )
            self._auth_source = auth_path or os.environ.get("LHA_CODEX_AUTH_FILE")
            self._model = model or model_name or os.environ.get("LHA_CODEX_MODEL")
            self._reasoning_effort = reasoning_effort or os.environ.get("LHA_CODEX_EFFORT")
            self._protocol_path = protocol_path or os.environ.get("LHA_TBENCH_PROTOCOL")
            self._instance_id = instance_id
            self._run_kind = run_kind
            self._attempt_id = attempt_id
            self._protocol: TerminalBenchProtocol | None = None
            self._codex_version: str | None = None
            self._observed_codex_target: Literal[
                "x86_64-unknown-linux-musl"
            ] | None = None
            self._observed_codex_binary_sha256: str | None = None
            self._image_attestation: DockerImageAttestation | None = None
            self._post_quiescence_attestation: DockerImageAttestation | None = None
            self._validated_wheel: Path | None = None
            self._protocol_sha256: str | None = None
            self._smoke_seal_sha256: str | None = None
            self._broker_tls_certificate_sha256: str | None = None
            self._setup_complete = False
            self._agent_started = False
            self._task_run_consumed = False
            super().__init__(logs_dir, model_name=model_name, *args, **kwargs)

        def _load_protocol(self) -> TerminalBenchProtocol:
            if self._protocol is not None:
                return self._protocol
            if not self._protocol_path:
                raise RuntimeError(
                    "a preregistration JSON is required via protocol_path or "
                    "LHA_TBENCH_PROTOCOL"
                )
            try:
                protocol = TerminalBenchProtocol.model_validate_json(
                    Path(self._protocol_path).read_text()
                )
            except Exception as exc:
                raise RuntimeError("the Terminal-Bench preregistration is invalid") from exc
            for supplied, recorded, name in (
                (self._model, protocol.model, "model"),
                (self._reasoning_effort, protocol.reasoning_effort, "reasoning effort"),
            ):
                if supplied is not None and supplied != recorded:
                    raise RuntimeError(f"{name} does not match the preregistration")
            self._model = protocol.model
            self._reasoning_effort = protocol.reasoning_effort
            self._protocol = protocol
            return protocol

        def _validate_inputs(
            self,
        ) -> tuple[
            Path,
            Path,
            Path,
            TerminalBenchProtocol,
            str,
            Literal["smoke", "scored"],
            str,
        ]:
            if not self._wheel:
                raise RuntimeError(
                    "no lha wheel: build with `uv build` and pass wheel_path or set LHA_WHEEL"
                )
            wheel = Path(self._wheel)
            if not wheel.is_file():
                raise RuntimeError("the configured lha wheel does not exist")
            protocol = self._load_protocol()
            if sha256_file(wheel) != protocol.wheel_sha256:
                raise RuntimeError("the lha wheel does not match the preregistration")
            if not self._codex_binary:
                raise RuntimeError(
                    "no Codex binary: pass codex_binary_path or set "
                    "LHA_CODEX_BINARY_FILE"
                )
            codex_binary = Path(self._codex_binary)
            if not codex_binary.is_file():
                raise RuntimeError("the configured Codex binary does not exist")
            if sha256_file(codex_binary) != protocol.codex_binary_sha256:
                raise RuntimeError("the Codex binary does not match the preregistration")
            if not self._auth_source:
                raise RuntimeError(
                    "Codex auth requires an explicit host-only "
                    "auth_path/LHA_CODEX_AUTH_FILE"
                )
            auth = Path(self._auth_source)
            if not auth.is_absolute() or not auth.is_file():
                raise RuntimeError("the configured Codex auth file does not exist")
            if self._run_kind == "smoke":
                run_kind: Literal["smoke", "scored"] = "smoke"
            elif self._run_kind == "scored":
                run_kind = "scored"
            else:
                raise RuntimeError("run_kind must be exactly 'smoke' or 'scored'")
            if not self._instance_id:
                raise RuntimeError("instance_id is required for a protocol-bound trial")
            allowed = selected_instance_ids(protocol, run_kind)
            if self._instance_id not in allowed:
                raise RuntimeError(
                    f"instance_id {self._instance_id!r} is outside the {run_kind} set"
                )
            self._validate_harbor_trial_name(self._instance_id)
            if not self._attempt_id:
                raise RuntimeError("attempt_id is required for a protocol-bound trial")
            expected_attempt_id = terminal_attempt_id(
                protocol.evaluation_id,
                run_kind,
                self._instance_id,
            )
            if self._attempt_id != expected_attempt_id:
                raise RuntimeError("attempt_id does not match the preregistration")
            protocol_sha256 = sha256_file(Path(self._protocol_path or ""))
            _registered_attempt(
                protocol,
                protocol_sha256=protocol_sha256,
                attempt_id=self._attempt_id,
                run_kind=run_kind,
                instance_id=self._instance_id,
            )
            with open_attempt_store(
                protocol.output_root,
                protocol.evaluation_id,
                self._attempt_id,
            ) as store:
                if store.has(_MODEL_STARTED) or store.has(_TERMINAL_RECORD):
                    raise RuntimeError("this registered attempt was already consumed")
            smoke_seal = (
                _validated_smoke_seal(protocol, protocol_sha256=protocol_sha256)
                if run_kind == "scored"
                else None
            )
            self._protocol_sha256 = protocol_sha256
            self._smoke_seal_sha256 = smoke_seal
            return (
                wheel,
                codex_binary,
                auth,
                protocol,
                self._instance_id,
                run_kind,
                self._attempt_id,
            )

        def _validate_harbor_trial_name(self, expected_instance_id: str) -> None:
            session_id = getattr(self, "session_id", None)
            if not isinstance(session_id, str) or not session_id.endswith("__agent"):
                raise RuntimeError("Harbor did not bind a valid trial session to the agent")
            parts = session_id.rsplit("__", 2)
            if len(parts) != 3:
                raise RuntimeError("Harbor trial session does not contain a task prefix")
            actual_prefix = parts[0]
            short_name = expected_instance_id.rsplit("/", 1)[-1]
            expected_prefix = short_name[:32].rstrip("_-")
            if actual_prefix != expected_prefix:
                raise RuntimeError(
                    f"Harbor trial prefix {actual_prefix!r} does not match the registered "
                    f"instance {expected_instance_id!r}"
                )

        async def install(self, environment) -> None:
            _require_harbor_docker_environment(environment)
            (
                wheel,
                codex_binary,
                _auth,
                protocol,
                instance_id,
                _run_kind,
                _attempt_id,
            ) = self._validate_inputs()
            self._validated_wheel = wheel
            try:
                image_attestation = await _attest_harbor_docker_image(environment)
                expected_image_digest = protocol.task_image_digests[instance_id]
                if not image_attestation.proves(expected_image_digest):
                    raise RuntimeError(
                        "the running Harbor container does not match the preregistered "
                        "task-image digest"
                    )
                self._image_attestation = image_attestation
                await environment.upload_file(str(codex_binary), _CODEX_UPLOAD)
                for command in install_commands(
                    codex_cli_version=protocol.codex_cli_version,
                    codex_binary_sha256=protocol.codex_binary_sha256,
                    codex_target=protocol.codex_target,
                ):
                    result = await environment.exec(
                        command=command,
                        user="root",
                    )
                    if result.return_code != 0:
                        raise RuntimeError(
                            "Codex CLI installation returned a non-zero status"
                        )
                isolation = await environment.exec(
                    command=process_isolation_check_command(),
                    user=_CODEX_RUN_USER,
                )
                if isolation.return_code != 0:
                    raise RuntimeError(
                        "Codex process isolation check returned a non-zero status"
                    )
                version = await environment.exec(
                    command=(
                        "set -eu; "
                        "uname -m; "
                        "sha256sum /usr/local/bin/codex | awk '{print $1}'; "
                        "/usr/local/bin/codex --version 2>/dev/null"
                    ),
                    user=_CODEX_RUN_USER,
                )
                if version.return_code != 0:
                    raise RuntimeError(
                        "Codex CLI version check returned a non-zero status"
                    )
                proof = (version.stdout or version.stderr or "").splitlines()
                if len(proof) != 3:
                    raise RuntimeError("Codex CLI installation proof is incomplete")
                architecture, observed_digest, self._codex_version = (
                    item.strip() for item in proof
                )
                if architecture != "x86_64":
                    raise RuntimeError("installed Codex CLI architecture changed")
                self._observed_codex_target = "x86_64-unknown-linux-musl"
                self._observed_codex_binary_sha256 = observed_digest
                if self._codex_version != protocol.codex_cli_version:
                    raise RuntimeError(
                        "installed Codex CLI does not match the preregistration"
                    )
                if observed_digest != protocol.codex_binary_sha256:
                    raise RuntimeError(
                        "installed Codex CLI digest does not match the preregistration"
                    )
            except BaseException:
                self._write_terminal_record(
                    codex_outcome="setup_error",
                    codex_failure_kind="agent_setup_failed",
                    model_started=False,
                    broker_cleanup_state="not_started",
                    container_quiescence="not_started",
                )
                raise
            self._setup_complete = True

        def _write_terminal_record(
            self,
            *,
            codex_outcome: Literal[
                "setup_error",
                "success",
                "process_error",
                "protocol_error",
                "execution_error",
            ],
            codex_failure_kind: Literal[
                "codex_nonzero_exit",
                "codex_reported_error",
                "codex_jsonl_invalid",
                "codex_tool_budget_exceeded",
                "codex_capability_exposed",
                "agent_setup_failed",
                "broker_start_failed",
                "codex_execution_exception",
                "codex_cancelled",
                "codex_runtime_cleanup_failed",
                "broker_cleanup_failed",
                "container_quiescence_failed",
            ]
            | None,
            model_started: bool,
            broker_cleanup_state: Literal["not_started", "succeeded", "failed"],
            container_quiescence: Literal[
                "not_started", "restarted", "stopped", "failed"
            ],
            codex_return_code: int | None = None,
            audit: CodexRunAudit | None = None,
            event_stream: str | None = None,
            stderr_stream: str | None = None,
            broker_receipt: Mapping[str, Any] | None = None,
        ) -> tuple[TerminalBenchAgentProvenance, str]:
            assert self._model is not None
            assert self._reasoning_effort is not None
            assert self._protocol is not None
            assert self._run_kind in ("smoke", "scored")
            assert self._instance_id is not None
            assert self._attempt_id is not None
            assert self._validated_wheel is not None
            assert self._protocol_sha256 is not None

            with open_attempt_store(
                self._protocol.output_root,
                self._protocol.evaluation_id,
                self._attempt_id,
            ) as store:
                events_sha256 = (
                    store.write_once(_CODEX_EVENTS, event_stream.encode())
                    if event_stream is not None
                    else None
                )
                stderr_sha256 = (
                    store.write_once(_CODEX_STDERR, stderr_stream.encode())
                    if stderr_stream is not None
                    else None
                )
                receipt_sha256 = (
                    store.write_json_once(_BROKER_RECEIPT, broker_receipt)
                    if broker_receipt is not None
                    else None
                )
                accepted = (
                    broker_receipt.get("downstream_accepted_requests")
                    if broker_receipt is not None
                    else None
                )
                revoked = (
                    broker_receipt.get("revoked")
                    if broker_receipt is not None
                    else None
                )
                record = TerminalBenchAgentProvenance(
                    evaluation_id=self._protocol.evaluation_id,
                    attempt_id=self._attempt_id,
                    lha_version=__version__,
                    run_kind=self._run_kind,
                    instance_id=self._instance_id,
                    dataset_version=self._protocol.dataset_version,
                    model=self._model,
                    reasoning_effort=self._reasoning_effort,
                    harbor_version=self._protocol.harbor_version,
                    codex_cli_version=self._protocol.codex_cli_version,
                    observed_codex_cli_version=self._codex_version,
                    codex_target=self._protocol.codex_target,
                    observed_codex_target=self._observed_codex_target,
                    codex_binary_sha256=self._protocol.codex_binary_sha256,
                    observed_codex_binary_sha256=self._observed_codex_binary_sha256,
                    broker_image_id=self._protocol.broker_image_id,
                    task_content_digest=self._protocol.task_content_digests[
                        self._instance_id
                    ],
                    task_image_digest=self._protocol.task_image_digests[
                        self._instance_id
                    ],
                    image_attestation=self._image_attestation,
                    post_quiescence_attestation=self._post_quiescence_attestation,
                    wheel_sha256=sha256_file(self._validated_wheel),
                    protocol_sha256=self._protocol_sha256,
                    subset=self._protocol.subset,
                    budgets=self._protocol.budgets,
                    model_started=model_started,
                    infrastructure_retries_used=0,
                    codex_outcome=codex_outcome,
                    codex_return_code=codex_return_code,
                    codex_failure_kind=codex_failure_kind,
                    broker_cleanup_state=broker_cleanup_state,
                    container_quiescence=container_quiescence,
                    smoke_seal_sha256=self._smoke_seal_sha256,
                    codex_events_sha256=events_sha256,
                    codex_stderr_sha256=stderr_sha256,
                    broker_receipt_sha256=receipt_sha256,
                    broker_tls_certificate_sha256=(
                        self._broker_tls_certificate_sha256
                    ),
                    broker_accepted_requests=(
                        accepted if isinstance(accepted, int) else None
                    ),
                    broker_revoked=revoked if isinstance(revoked, bool) else None,
                    codex_audit=audit,
                )
                terminal_sha256 = store.write_json_once(_TERMINAL_RECORD, record)
            return record, terminal_sha256

        async def run(self, instruction: str, environment, context) -> None:
            if self._agent_started or self._task_run_consumed:
                raise RuntimeError("this Harbor agent permits exactly one Codex run")
            if (
                not self._setup_complete
                or self._validated_wheel is None
                or self._image_attestation is None
                or self._codex_version is None
            ):
                raise RuntimeError("Harbor agent install and image attestation must run first")
            _require_harbor_docker_environment(environment)
            (
                _wheel,
                _codex_binary,
                auth,
                protocol,
                instance_id,
                run_kind,
                attempt_id,
            ) = self._validate_inputs()
            self._task_run_consumed = True
            assert self._protocol_sha256 is not None
            with open_attempt_store(
                protocol.output_root,
                protocol.evaluation_id,
                attempt_id,
            ) as attempt_store:
                write_model_started(
                    attempt_store,
                    evaluation_id=protocol.evaluation_id,
                    attempt_id=attempt_id,
                    protocol_sha256=self._protocol_sha256,
                    run_kind=run_kind,
                    instance_id=instance_id,
                    container_id=self._image_attestation.container_id,
                )
            self._agent_started = True

            outcome: Literal[
                "success",
                "process_error",
                "protocol_error",
                "execution_error",
            ] = "execution_error"
            return_code: int | None = None
            failure_kind: Literal[
                "codex_nonzero_exit",
                "codex_reported_error",
                "codex_jsonl_invalid",
                "codex_tool_budget_exceeded",
                "codex_capability_exposed",
                "broker_start_failed",
                "codex_execution_exception",
                "codex_cancelled",
                "codex_runtime_cleanup_failed",
                "broker_cleanup_failed",
                "container_quiescence_failed",
            ] | None = "broker_start_failed"
            audit: CodexRunAudit | None = None
            event_parts: list[str] = []
            stream_validator = CodexJsonlValidator(
                max_tool_calls=protocol.budgets.max_tool_calls,
                max_line_bytes=protocol.budgets.max_jsonl_line_bytes,
                max_total_bytes=protocol.budgets.max_jsonl_bytes,
                max_reconnect_notices=protocol.budgets.stream_max_retries,
            )
            controller = TerminalProxyController(image_id=protocol.broker_image_id)
            handle: TerminalProxyHandle | None = None
            receipt: Mapping[str, Any] | None = None
            broker_cleanup_state: Literal[
                "not_started", "succeeded", "failed"
            ] = "not_started"
            container_quiescence: Literal[
                "not_started", "restarted", "stopped", "failed"
            ] = "not_started"
            run_error: BaseException | None = None
            capability_value = ""
            capability_detector: _BoundedSecretStreamDetector | None = None
            stderr_stream: str | None = None
            stream_error: CodexEventError | None = None
            captured_event_bytes = 0

            async def on_output(text: str, stream: str) -> None:
                nonlocal captured_event_bytes, stream_error
                detector = capability_detector
                if detector is None:
                    raise CodexEventError(
                        "Harbor emitted model output before capability registration"
                    )
                try:
                    capability_exposed = detector.feed(text)
                except CodexEventError as exc:
                    if stream_error is None:
                        stream_error = exc
                    # Harbor buffers streamed lines internally. Stop collection
                    # once the registered total is exceeded.
                    raise
                if capability_exposed:
                    event_parts.clear()
                    captured_event_bytes = 0
                    raise CapabilityExposureError(
                        "Codex output exposed its bounded capability"
                    )
                if stream != "stdout":
                    if stream_error is None:
                        stream_error = CodexEventError(
                            "Harbor changed the Codex output stream"
                        )
                    return
                if stream_error is not None:
                    return
                encoded = text.encode("utf-8")
                within_capture_limit = (
                    len(encoded) <= protocol.budgets.max_jsonl_line_bytes
                    and captured_event_bytes + len(encoded)
                    <= protocol.budgets.max_jsonl_bytes
                )
                if within_capture_limit:
                    event_parts.append(text)
                    captured_event_bytes += len(encoded)
                try:
                    stream_validator.feed_line(text)
                except CodexToolBudgetExceeded:
                    raise
                except CodexEventError as exc:
                    # Preserve the first failing event and bounded stderr. If the
                    # callback raises here, Harbor cancels collection before the
                    # adapter can store either diagnostic.
                    stream_error = exc

            try:
                assert self._model is not None
                assert self._reasoning_effort is not None
                credentials = _load_broker_secrets(
                    auth,
                    minimum_validity_s=protocol.budgets.codex_timeout_s + 300,
                )
                handle = await _start_proxy_cancel_safe(
                    controller,
                    evaluation_id=protocol.evaluation_id,
                    attempt_id=attempt_id,
                    source_container_id=self._image_attestation.container_id,
                    network=self._image_attestation.network_name,
                    model=self._model,
                    reasoning_effort=self._reasoning_effort,
                    credentials=credentials,
                )
                self._broker_tls_certificate_sha256 = (
                    handle.tls_certificate_sha256
                )
                failure_kind = "codex_execution_exception"
                capability_value = handle.capability_environment()[CAPABILITY_ENV]
                capability_detector = _BoundedSecretStreamDetector(
                    capability_value,
                    max_total_bytes=protocol.budgets.max_jsonl_bytes,
                )
                _, cleanup_cancellation = await _finish_cleanup(
                    _cleanup_codex_runtime(environment)
                )
                if cleanup_cancellation is not None:
                    raise cleanup_cancellation
                capability_file = _write_private_capability(capability_value)
                try:
                    certificate_file = _write_private_payload(
                        handle.tls_certificate_pem,
                        prefix="lha-terminal-proxy-ca-",
                    )
                    try:
                        await environment.upload_file(
                            str(capability_file),
                            _CAPABILITY_STAGING,
                        )
                        await environment.upload_file(
                            str(certificate_file),
                            _TLS_CERT_STAGING,
                        )
                        for label, command in (
                            (
                                "capability",
                                _finalize_uploaded_file_command(
                                    staging_path=_CAPABILITY_STAGING,
                                    destination_path=_CAPABILITY_UPLOAD,
                                    owner=_CODEX_RUN_USER,
                                    mode=0o600,
                                    expected_sha256=hashlib.sha256(
                                        capability_value.encode()
                                    ).hexdigest(),
                                ),
                            ),
                            (
                                "TLS certificate",
                                _finalize_uploaded_file_command(
                                    staging_path=_TLS_CERT_STAGING,
                                    destination_path=_TLS_CERT_PATH,
                                    owner=_CODEX_RUN_USER,
                                    mode=0o600,
                                    expected_sha256=handle.tls_certificate_sha256,
                                ),
                            ),
                        ):
                            installed = await environment.exec(
                                command=command,
                                user="root",
                            )
                            if installed.return_code != 0:
                                raise RuntimeError(
                                    f"Codex {label} installation failed"
                                )
                    finally:
                        certificate_file.unlink(missing_ok=True)
                finally:
                    capability_file.unlink(missing_ok=True)
                with (
                    _harbor_stream_line_limit(
                        environment,
                        maximum_line_bytes=protocol.budgets.max_jsonl_line_bytes,
                    ),
                    environment.scoped_output_callback(on_output),
                ):
                    result = await environment.exec(
                        command=codex_exec_command(
                            self._model,
                            self._reasoning_effort,
                            instruction,
                            proxy_base_url=handle.base_url,
                            binding_headers=handle.binding_headers(),
                            request_max_retries=(
                                protocol.budgets.request_max_retries
                            ),
                            stream_max_retries=(
                                protocol.budgets.stream_max_retries
                            ),
                        ),
                        timeout_sec=protocol.budgets.codex_timeout_s,
                        user=_CODEX_RUN_USER,
                    )
                result_stdout = result.stdout or ""
                if capability_detector.contains_complete(result_stdout):
                    event_parts.clear()
                    captured_event_bytes = 0
                    raise CapabilityExposureError(
                        "Codex final output exposed its bounded capability"
                    )
                if (
                    stream_error is None
                    and result_stdout != "".join(event_parts)
                ):
                    raise CodexEventError(
                        "Harbor output callbacks did not cover the final Codex output"
                    )
                stderr_stream = await _take_codex_stderr(environment)
                if capability_detector.contains_complete(stderr_stream):
                    event_parts.clear()
                    captured_event_bytes = 0
                    stderr_stream = None
                    raise CapabilityExposureError(
                        "Codex stderr exposed its bounded capability"
                    )
                return_code = result.return_code
                if stream_error is not None:
                    raise stream_error
                if result.return_code != 0:
                    outcome = "process_error"
                    failure_kind = "codex_nonzero_exit"
                    raise RuntimeError(
                        f"the single Codex run exited {result.return_code}"
                    )
                try:
                    strict = stream_validator.finish()
                    audit = CodexRunAudit(
                        event_counts=strict.event_counts,
                        item_counts=strict.item_counts,
                        tool_calls=strict.tool_calls,
                        reconnect_notices=strict.reconnect_notices,
                        input_tokens=strict.input_tokens,
                        cached_input_tokens=strict.cached_input_tokens,
                        output_tokens=strict.output_tokens,
                        reasoning_output_tokens=strict.reasoning_output_tokens,
                    )
                except CodexEventError as exc:
                    outcome = "protocol_error"
                    return_code = 0
                    failure_kind = "codex_jsonl_invalid"
                    raise RuntimeError("Codex JSONL failed protocol audit") from exc
                outcome = "success"
                return_code = 0
                failure_kind = None
            except BaseException as exc:
                run_error = exc
                if isinstance(exc, asyncio.CancelledError):
                    outcome = "execution_error"
                    failure_kind = "codex_cancelled"
                elif _contains_exception(exc, CodexToolBudgetExceeded):
                    outcome = "protocol_error"
                    failure_kind = "codex_tool_budget_exceeded"
                elif _contains_exception(exc, CapabilityExposureError):
                    outcome = "protocol_error"
                    failure_kind = "codex_capability_exposed"
                elif _contains_exception(exc, CodexEventError):
                    outcome = "protocol_error"
                    failure_kind = (
                        "codex_reported_error"
                        if _contains_exception(exc, CodexReportedError)
                        else "codex_jsonl_invalid"
                    )
                elif _contains_exception(exc, asyncio.LimitOverrunError):
                    outcome = "protocol_error"
                    failure_kind = "codex_jsonl_invalid"
                    line_error = CodexEventError(
                        "Codex JSONL line exceeded the registered transport limit"
                    )
                    line_error.__cause__ = exc
                    run_error = line_error
            finally:
                try:
                    _, cleanup_cancellation = await _finish_cleanup(
                        _cleanup_codex_runtime(environment)
                    )
                    if cleanup_cancellation is not None:
                        outcome = "execution_error"
                        failure_kind = "codex_cancelled"
                        if run_error is None:
                            run_error = cleanup_cancellation
                except BaseException as exc:
                    if run_error is None:
                        outcome = "execution_error"
                        failure_kind = "codex_runtime_cleanup_failed"
                        run_error = exc
                if handle is not None:
                    try:
                        receipt, cleanup_cancellation = await _finish_cleanup(
                            asyncio.to_thread(controller.stop, handle)
                        )
                        broker_cleanup_state = "succeeded"
                        if cleanup_cancellation is not None and run_error is None:
                            outcome = "execution_error"
                            failure_kind = "codex_cancelled"
                            run_error = cleanup_cancellation
                    except BaseException as exc:
                        broker_cleanup_state = "failed"
                        if run_error is None:
                            outcome = "execution_error"
                            failure_kind = "broker_cleanup_failed"
                            run_error = exc
                capability_value = ""
                if capability_detector is not None:
                    capability_detector.clear()

                if outcome == "success" and broker_cleanup_state == "succeeded":
                    try:
                        (
                            self._post_quiescence_attestation,
                            cleanup_cancellation,
                        ) = await _finish_cleanup(
                            _restart_and_confirm_main(
                                environment,
                                self._image_attestation,
                            )
                        )
                        container_quiescence = "restarted"
                        if cleanup_cancellation is not None and run_error is None:
                            outcome = "execution_error"
                            failure_kind = "codex_cancelled"
                            run_error = cleanup_cancellation
                    except BaseException as exc:
                        outcome = "execution_error"
                        failure_kind = "container_quiescence_failed"
                        if run_error is None:
                            run_error = exc
                if outcome != "success":
                    try:
                        _, cleanup_cancellation = await _finish_cleanup(
                            _kill_and_confirm_main(
                                environment,
                                self._image_attestation.container_id,
                            )
                        )
                        container_quiescence = "stopped"
                        if cleanup_cancellation is not None and run_error is None:
                            failure_kind = "codex_cancelled"
                            run_error = cleanup_cancellation
                    except BaseException as exc:
                        container_quiescence = "failed"
                        if run_error is None:
                            outcome = "execution_error"
                            failure_kind = "container_quiescence_failed"
                            run_error = exc

            event_stream = "".join(event_parts)
            record, terminal_sha256 = self._write_terminal_record(
                codex_outcome=outcome,
                codex_failure_kind=failure_kind,
                model_started=True,
                broker_cleanup_state=broker_cleanup_state,
                container_quiescence=container_quiescence,
                codex_return_code=return_code,
                audit=audit if outcome == "success" else None,
                event_stream=event_stream,
                stderr_stream=stderr_stream,
                broker_receipt=receipt if broker_cleanup_state == "succeeded" else None,
            )
            self._agent_started = False
            if outcome != "success":
                if run_error is not None:
                    raise run_error
                raise RuntimeError(f"Terminal-Bench agent failed: {failure_kind}")
            assert audit is not None
            self._fill_usage(
                context,
                audit,
                instance_id,
                record=record,
                terminal_sha256=terminal_sha256,
            )

        def _fill_usage(
            self,
            context,
            audit: CodexRunAudit,
            instance_id: str,
            *,
            record: TerminalBenchAgentProvenance,
            terminal_sha256: str,
        ) -> None:
            """Copy audited Codex usage; Harbor's verifier still supplies truth."""
            for dst, value in (
                ("n_input_tokens", audit.input_tokens),
                ("n_cache_tokens", audit.cached_input_tokens),
                ("n_output_tokens", audit.output_tokens),
            ):
                if value is not None and hasattr(context, dst):
                    setattr(context, dst, value)
            if hasattr(context, "metadata"):
                assert self._protocol is not None
                assert self._protocol_path is not None
                assert self._image_attestation is not None
                assert self._run_kind in ("smoke", "scored")
                context.metadata = _agent_metadata(
                    protocol=self._protocol,
                    protocol_sha256=sha256_file(Path(self._protocol_path)),
                    instance_id=instance_id,
                    run_kind=self._run_kind,
                    audit=audit,
                    codex_events_sha256=cast(str, record.codex_events_sha256),
                    image_attestation=self._image_attestation,
                    terminal_record_sha256=terminal_sha256,
                    provenance=record,
                )

    return LhaAgent


try:  # Harbor import-path loading imports this module before looking up the class.
    LhaAgent = build_agent()
except ImportError:  # Core lha remains importable on Python 3.11 without the bench extra.
    class LhaAgent:  # type: ignore[no-redef]
        """Placeholder replaced by the real Harbor subclass when Harbor is installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("LhaAgent requires harbor>=0.20 on Python >=3.12")
