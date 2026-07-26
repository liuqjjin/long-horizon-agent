"""Terminal-Bench 2.1 preregistration and Harbor adapter.

Harbor's verifier is the only source of task truth.  The adapter runs one real
``codex exec`` in Harbor's disposable task container and never uses LHA's
text-only ablation patch generator or an internal test result as a score.

ChatGPT-backed Codex credentials are accepted only as an explicit trusted-local
input.  The host file is uploaded for one trial, installed into a temporary
``CODEX_HOME`` with mode 0600, and removed in a ``finally`` block.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shlex
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

DATASET = "terminal-bench/terminal-bench-2-1"
AGENT_IMPORT_PATH = "lha.bench.terminal_bench:LhaAgent"
HARBOR_VERSION = "0.20.0"

_AUTH_UPLOAD = "/tmp/.lha_codex_auth.upload"
_CODEX_UPLOAD = "/tmp/.lha_codex_binary.upload"
_CODEX_HOME = "/tmp/lha_codex_home"
_AGENT_PROVENANCE = "terminal_bench_agent.json"
_CODEX_EVENTS = "codex_events.jsonl"
_CODEX_STDERR = "codex_stderr.txt"
_CODEX_VERSION_RE = re.compile(r"^codex-cli ([0-9A-Za-z][0-9A-Za-z.+-]*)$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_VALUE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_REPO_DIGEST_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_TOP_LEVEL_EVENTS = frozenset(
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
_TEXT_ITEMS = frozenset({"agent_message", "reasoning", "todo_list"})
_PAIRED_TOOL_ITEMS = frozenset(
    {"command_execution", "mcp_tool_call", "collab_tool_call", "web_search"}
)
# Codex 0.141 emits file changes only after the patch attempt reaches a
# terminal status; there is deliberately no matching item.started event.
_COMPLETED_ONLY_TOOL_ITEMS = frozenset({"file_change"})
_ITEM_TYPES = (
    _TEXT_ITEMS | _PAIRED_TOOL_ITEMS | _COMPLETED_ONLY_TOOL_ITEMS | {"error"}
)
_T = TypeVar("_T")


class TerminalBenchBudgets(BaseModel):
    """Limits that this adapter can enforce for one Harbor trial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_s: Literal[1800] = 1800
    max_tool_calls: Literal[20] = 20
    codex_exec_runs: Literal[1] = 1
    scored_runs_per_task: Literal[1] = 1
    infrastructure_retries: Literal[1] = 1
    task_retries: Literal[0] = 0


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

    schema_version: Literal[2] = 2
    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    subset: RegisteredSubset
    model: str
    reasoning_effort: str
    harbor_version: str
    codex_cli_version: str
    codex_target: Literal["x86_64-unknown-linux-musl"]
    codex_binary_sha256: str
    task_image_digests: dict[str, str]
    wheel_sha256: str
    budgets: TerminalBenchBudgets = Field(default_factory=TerminalBenchBudgets)

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

    @field_validator("task_image_digests")
    @classmethod
    def _image_digests_are_pinned(cls, value: dict[str, str]) -> dict[str, str]:
        checked: dict[str, str] = {}
        for instance_id, image_digest in value.items():
            if not instance_id.strip():
                raise ValueError("task image digest keys may not be empty")
            if not image_digest.startswith("sha256:"):
                raise ValueError("task image digests must start with sha256:")
            digest = image_digest.removeprefix("sha256:")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(
                    "task image digests must contain a lowercase SHA-256 digest"
                )
            checked[instance_id] = image_digest
        return dict(sorted(checked.items()))

    @field_validator("wheel_sha256", "codex_binary_sha256")
    @classmethod
    def _file_digest_is_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("file digests must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def _images_cover_the_registered_tasks(self) -> "TerminalBenchProtocol":
        expected = set(self.subset.scored_instance_ids) | set(
            self.subset.smoke_instance_ids
        )
        if set(self.task_image_digests) != expected:
            raise ValueError(
                "task_image_digests must contain exactly the 20 scored and 3 smoke tasks"
            )
        return self


class CodexRunAudit(BaseModel):
    """Validated public JSONL summary for one tool-enabled Codex run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_counts: dict[str, int]
    item_counts: dict[str, int]
    tool_calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class DockerImageAttestation(BaseModel):
    """Image identity read from the live Harbor Docker container."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    container_id: str
    image_id: str
    configured_image: str
    repo_digests: tuple[str, ...]

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
    """Secret-free adapter and audit evidence retained in Harbor's trial logs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    agent_import_path: Literal["lha.bench.terminal_bench:LhaAgent"] = AGENT_IMPORT_PATH
    lha_version: str
    run_kind: Literal["smoke", "scored"]
    instance_id: str
    model: str
    reasoning_effort: str
    harbor_version: str
    codex_cli_version: str
    codex_target: Literal["x86_64-unknown-linux-musl"]
    codex_binary_sha256: str
    task_image_digest: str
    image_attestation: DockerImageAttestation
    wheel_sha256: str
    protocol_sha256: str
    subset: RegisteredSubset
    budgets: TerminalBenchBudgets
    infrastructure_retries_used: int = Field(ge=0, le=1)
    codex_events_sha256: str | None = None
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
        "codex_events_sha256",
    )
    @classmethod
    def _provenance_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("provenance file digests must be lowercase SHA-256 hex")
        return value

    @field_validator("task_image_digest")
    @classmethod
    def _provenance_image_digest(cls, value: str) -> str:
        if _SHA256_VALUE_RE.fullmatch(value) is None:
            raise ValueError("task_image_digest must be sha256:<64 lowercase hex>")
        return value

    @model_validator(mode="after")
    def _audit_fields_move_together(self) -> "TerminalBenchAgentProvenance":
        if (self.codex_events_sha256 is None) != (self.codex_audit is None):
            raise ValueError("Codex event digest and audit must both be present or absent")
        return self


class HarborRunCommand(BaseModel):
    """One exact one-instance Harbor invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_kind: Literal["smoke", "scored"]
    instance_id: str
    task_image_digest: str
    argv: tuple[str, ...]
    job_dir: str


class HarborExecutionManifest(BaseModel):
    """Post-run evidence that Harbor executed exactly the registered set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    run_kind: Literal["smoke", "scored"]
    protocol_sha256: str
    expected_instance_ids: tuple[str, ...]
    observed_instance_ids: tuple[str, ...]
    task_image_digests: dict[str, str]
    codex_events_sha256: dict[str, str]
    container_image_ids: dict[str, str]
    trial_result_sha256: dict[str, str]
    job_dirs: tuple[str, ...]

    @field_validator("protocol_sha256")
    @classmethod
    def _protocol_digest_is_sha256(cls, value: str) -> str:
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("manifest protocol_sha256 must be lowercase SHA-256 hex")
        return value

    @model_validator(mode="after")
    def _evidence_covers_exact_execution(self) -> "HarborExecutionManifest":
        expected = self.expected_instance_ids
        if len(expected) != len(set(expected)):
            raise ValueError("manifest expected instance ids must be unique")
        if self.observed_instance_ids != expected:
            raise ValueError("manifest observed instance order must match the expected order")
        expected_set = set(expected)
        evidence_maps = (
            self.task_image_digests,
            self.codex_events_sha256,
            self.container_image_ids,
            self.trial_result_sha256,
        )
        if any(set(mapping) != expected_set for mapping in evidence_maps):
            raise ValueError("manifest evidence must cover exactly the executed instances")
        if any(
            _SHA256_HEX_RE.fullmatch(digest) is None
            for digest in (*self.codex_events_sha256.values(), *self.trial_result_sha256.values())
        ):
            raise ValueError("manifest file digests must be lowercase SHA-256 hex")
        if len(self.job_dirs) != len(expected) or len(self.job_dirs) != len(
            set(self.job_dirs)
        ):
            raise ValueError("manifest must bind one unique job directory per instance")
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
    official_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    official_status: Literal["PASS", "FAIL", "ERROR"]
    independent_correct: bool | None = None
    gate_accepted: None = None
    repairs: None = None
    duration_s: FiniteFloat | None = Field(default=None, ge=0)
    protocol_error: str | None = None
    infrastructure_retries: int = Field(default=0, ge=0, le=1)
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
                f"- 协议错误：{self.protocol_errors}",
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
    if (
        not isinstance(actual_container_id, str)
        or not actual_container_id.startswith(container_id)
        or not isinstance(image_id, str)
        or not isinstance(configured_image, str)
    ):
        raise RuntimeError("Docker container inspection omitted image identity")

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
    )


async def _attest_harbor_docker_image(environment) -> DockerImageAttestation:
    """Resolve Harbor's running ``main`` service to Docker's actual image."""
    compose = getattr(environment, "_run_docker_compose_command", None)
    if not callable(compose):
        raise RuntimeError(
            "formal Terminal-Bench runs require Harbor's auditable Docker environment"
        )
    compose_call = cast(Callable[[list[str]], Awaitable[Any]], compose)
    result = await compose_call(["ps", "-q", "main"])
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


def create_protocol(
    instance_ids: Sequence[str],
    *,
    model: str,
    reasoning_effort: str,
    harbor_version: str,
    codex_cli_version: str,
    codex_target: Literal["x86_64-unknown-linux-musl"],
    codex_binary_path: str | Path,
    task_image_digests: Mapping[str, str],
    wheel_path: str | Path,
) -> TerminalBenchProtocol:
    """Build the complete preregistration record from measured inputs."""
    normalized_ids = [item.strip() for item in instance_ids]
    if set(task_image_digests) != set(normalized_ids):
        raise ValueError("task_image_digests must cover the complete frozen corpus")
    subset = preregister_instances(normalized_ids)
    selected = (*subset.scored_instance_ids, *subset.smoke_instance_ids)
    return TerminalBenchProtocol(
        subset=subset,
        model=model,
        reasoning_effort=reasoning_effort,
        harbor_version=harbor_version,
        codex_cli_version=codex_cli_version,
        codex_target=codex_target,
        codex_binary_sha256=sha256_file(codex_binary_path),
        task_image_digests={item: task_image_digests[item] for item in selected},
        wheel_sha256=sha256_file(wheel_path),
    )


def write_protocol(protocol: TerminalBenchProtocol, path: str | Path) -> Path:
    """Write one deterministic, secret-free JSON protocol file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(protocol.model_dump_json(indent=2) + "\n")
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
        _, trial_result, result_digest = _read_single_trial_result(
            Path(command.job_dir)
        )
        if result_digest != execution_manifest.trial_result_sha256[instance_id]:
            raise ValueError(f"official Harbor result changed for {instance_id}")
        status, independent_correct, protocol_error = _official_trial_outcome(
            trial_result
        )
        records.append(
            TerminalBenchTaskRecord(
                instance_id=instance_id,
                protocol_sha256=protocol_digest,
                execution_manifest_sha256=manifest_digest,
                official_result_sha256=result_digest,
                official_status=status,
                independent_correct=independent_correct,
                duration_s=_official_trial_duration(trial_result),
                protocol_error=protocol_error,
                infrastructure_retries=_official_infrastructure_retries(
                    trial_result
                ),
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


def _official_infrastructure_retries(trial_result: Mapping[str, Any]) -> int:
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
    if isinstance(retries, bool) or not isinstance(retries, int) or retries not in (0, 1):
        raise ValueError("official Harbor metadata omitted infrastructure retry count")
    return retries


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


def install_commands(codex_cli_version: str) -> list[str]:
    """Install the preregistered standalone binary without mutating the image."""
    match = _CODEX_VERSION_RE.fullmatch(codex_cli_version)
    if match is None:
        raise ValueError("invalid exact Codex CLI version")
    expected = shlex.quote(codex_cli_version)
    return [
        "set -eu; "
        f"install -m 755 {_CODEX_UPLOAD} /usr/local/bin/codex; "
        f"rm -f {_CODEX_UPLOAD}; "
        f'[ "$(codex --version)" = {expected} ]',
    ]


def codex_exec_command(
    model: str,
    reasoning_effort: str,
    instruction: str,
) -> str:
    """Build the one tool-enabled Codex invocation used inside Harbor."""
    return " ".join(
        [
            f"CODEX_HOME={_CODEX_HOME}",
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            "--json",
            "--model",
            shlex.quote(model),
            "-c",
            shlex.quote(f"model_reasoning_effort={reasoning_effort!r}"),
            "--",
            shlex.quote(instruction),
        ]
    )


def audit_codex_jsonl(
    event_stream: str,
    *,
    max_tool_calls: int = TerminalBenchBudgets().max_tool_calls,
) -> CodexRunAudit:
    """Validate a complete Codex stream while allowing finished tool actions."""
    event_counts: dict[str, int] = {}
    item_counts: dict[str, int] = {}
    open_tools: dict[str, str] = {}
    thread_started = False
    turn_started = False
    turn_completed = False
    saw_agent_message = False
    usage: dict[str, Any] = {}
    tool_calls = 0

    for line_number, raw_line in enumerate(event_stream.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Codex JSONL line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(event, dict):
            raise RuntimeError(f"Codex JSONL line {line_number} is not an object")
        error_path = _find_true_is_error(event)
        if error_path is not None:
            raise RuntimeError(f"Codex event reported isError at {error_path}")
        kind = event.get("type")
        if not isinstance(kind, str) or kind not in _TOP_LEVEL_EVENTS:
            raise RuntimeError(f"unknown Codex event type at line {line_number}: {kind!r}")
        event_counts[kind] = event_counts.get(kind, 0) + 1
        if turn_completed:
            raise RuntimeError(f"Codex emitted {kind!r} after turn.completed")

        if kind == "thread.started":
            if thread_started or turn_started or sum(event_counts.values()) != 1:
                raise RuntimeError("thread.started must be the first event")
            thread_started = True
        elif kind == "turn.started":
            if not thread_started or turn_started:
                raise RuntimeError("turn.started must follow one thread.started")
            turn_started = True
        elif kind in {"item.started", "item.updated", "item.completed"}:
            if not turn_started:
                raise RuntimeError(f"{kind} arrived before turn.started")
            item = event.get("item")
            if not isinstance(item, dict):
                raise RuntimeError(f"{kind} has no object-valued item")
            item_id = item.get("id")
            item_type = item.get("type")
            if not isinstance(item_id, str) or not item_id:
                raise RuntimeError(f"{kind} item has no stable id")
            if not isinstance(item_type, str) or not item_type:
                raise RuntimeError(f"{kind} item has no type")
            if item_type not in _ITEM_TYPES:
                raise RuntimeError(
                    f"unknown Codex item type at line {line_number}: {item_type!r}"
                )
            item_counts[item_type] = item_counts.get(item_type, 0) + 1
            if item_type == "error":
                raise RuntimeError("Codex emitted an error item")
            if item_type in _COMPLETED_ONLY_TOOL_ITEMS:
                if kind != "item.completed":
                    raise RuntimeError(
                        f"Codex {item_type} item {item_id!r} must be completed-only"
                    )
                tool_calls += 1
                if tool_calls > max_tool_calls:
                    raise RuntimeError(
                        f"Codex exceeded the {max_tool_calls}-tool-call protocol limit"
                    )
            elif kind == "item.started" and item_type in _PAIRED_TOOL_ITEMS:
                if item_id in open_tools:
                    raise RuntimeError(f"Codex started tool item {item_id!r} twice")
                open_tools[item_id] = item_type
                tool_calls += 1
                if tool_calls > max_tool_calls:
                    raise RuntimeError(
                        f"Codex exceeded the {max_tool_calls}-tool-call protocol limit"
                    )
            elif (
                kind == "item.updated"
                and item_type in _PAIRED_TOOL_ITEMS
                and item_id not in open_tools
            ):
                raise RuntimeError(f"Codex updated tool item {item_id!r} before it started")
            elif kind == "item.completed":
                if item_type in _PAIRED_TOOL_ITEMS:
                    started_type = open_tools.pop(item_id, None)
                    if started_type is None:
                        raise RuntimeError(
                            f"Codex completed tool item {item_id!r} before it started"
                        )
                    if started_type != item_type:
                        raise RuntimeError(
                            f"Codex changed tool item {item_id!r} from "
                            f"{started_type!r} to {item_type!r}"
                        )
                elif item_type == "agent_message":
                    saw_agent_message = True
        elif kind == "turn.completed":
            if not turn_started:
                raise RuntimeError("turn.completed arrived before turn.started")
            if open_tools:
                unfinished = ", ".join(sorted(open_tools))
                raise RuntimeError(f"Codex turn completed with unfinished tools: {unfinished}")
            raw_usage = event.get("usage") or {}
            if not isinstance(raw_usage, dict):
                raise RuntimeError("turn.completed usage is not an object")
            usage = raw_usage
            turn_completed = True
        elif kind in {"turn.failed", "error"}:
            detail = event.get("message") or event.get("error") or "no detail"
            raise RuntimeError(f"Codex reported {kind}: {str(detail)[:400]}")

    if not thread_started:
        raise RuntimeError("Codex JSONL stream is empty or missing thread.started")
    if not turn_started:
        raise RuntimeError("Codex JSONL stream is missing turn.started")
    if not turn_completed:
        unfinished = f"; unfinished tools: {sorted(open_tools)}" if open_tools else ""
        raise RuntimeError(f"Codex JSONL stream is missing turn.completed{unfinished}")
    if not saw_agent_message:
        raise RuntimeError("Codex turn completed without a final agent message")
    return CodexRunAudit(
        event_counts=event_counts,
        item_counts=item_counts,
        tool_calls=tool_calls,
        input_tokens=_optional_nonnegative_int(usage.get("input_tokens")),
        cached_input_tokens=_optional_nonnegative_int(usage.get("cached_input_tokens")),
        output_tokens=_optional_nonnegative_int(usage.get("output_tokens")),
    )


def _find_true_is_error(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "isError" and bool(child):
                return child_path
            found = _find_true_is_error(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_true_is_error(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Codex usage value is not a non-negative integer: {value!r}")
    return value


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
        DATASET,
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
        "trusted_local_auth=true",
        "--agent-kwarg",
        "externally_sandboxed=true",
    )


def build_harbor_commands(
    protocol: TerminalBenchProtocol,
    run_kind: Literal["smoke", "scored"],
    *,
    protocol_path: str | Path,
    wheel_path: str | Path,
    codex_binary_path: str | Path,
    auth_path: str | Path,
    jobs_dir: str | Path,
    job_name_prefix: str = "lha-tbench-2-1",
) -> tuple[HarborRunCommand, ...]:
    """Generate one exact Harbor job per registered instance.

    A separate job lets the full instance ID be passed to the agent and checked
    against Harbor's trial name before Codex starts.  The host credential path
    is deliberately not placed in argv; callers provide it through
    ``LHA_CODEX_AUTH_FILE``.
    """
    protocol_file = Path(protocol_path).resolve()
    wheel = Path(wheel_path).resolve()
    codex_binary = Path(codex_binary_path).resolve()
    auth = Path(auth_path).resolve()
    output_root = Path(jobs_dir).resolve()
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
    if not auth.is_file():
        raise ValueError("auth_path does not exist")
    safe_prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "-", job_name_prefix).strip("-")
    if not safe_prefix:
        raise ValueError("job_name_prefix has no safe characters")

    commands: list[HarborRunCommand] = []
    for index, instance_id in enumerate(selected_instance_ids(protocol, run_kind), start=1):
        suffix = hashlib.sha256(instance_id.encode()).hexdigest()[:10]
        job_name = f"{safe_prefix}-{run_kind}-{index:02d}-{suffix}"
        argv = _harbor_argv(
            protocol,
            run_kind,
            instance_id,
            protocol_file=protocol_file,
            wheel=wheel,
            codex_binary=codex_binary,
            output_root=output_root,
            job_name=job_name,
        )
        commands.append(
            HarborRunCommand(
                run_kind=run_kind,
                instance_id=instance_id,
                task_image_digest=protocol.task_image_digests[instance_id],
                argv=argv,
                job_dir=str(output_root / job_name),
            )
        )
    return tuple(commands)


def validate_harbor_results(
    protocol: TerminalBenchProtocol,
    run_kind: Literal["smoke", "scored"],
    commands: Sequence[HarborRunCommand],
    *,
    protocol_path: str | Path,
    manifest_path: str | Path | None = None,
) -> HarborExecutionManifest:
    """Validate Harbor config, runtime provenance, and each Codex event stream."""
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
    expected_set = set(expected)
    if len(commands) != len(expected):
        raise ValueError("the Harbor command count does not match the protocol")
    observed: list[str] = []
    job_dirs: list[str] = []
    event_digests: dict[str, str] = {}
    image_ids: dict[str, str] = {}
    result_digests: dict[str, str] = {}
    for command in commands:
        if command.run_kind != run_kind or command.instance_id not in expected_set:
            raise ValueError("a Harbor command is outside this protocol phase")
        if command.task_image_digest != protocol.task_image_digests[command.instance_id]:
            raise ValueError("a Harbor command has an unregistered task-image digest")
        expected_kwargs = _validate_command_contract(
            protocol,
            command,
            protocol_file=protocol_file,
        )
        job_dir = Path(command.job_dir)
        config_path = job_dir / "config.json"
        lock_path = job_dir / "lock.json"
        result_path = job_dir / "result.json"
        try:
            config = json.loads(config_path.read_text())
            job_lock = json.loads(lock_path.read_text())
            job_result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Harbor job output is unreadable: {job_dir}") from exc
        if not all(isinstance(value, Mapping) for value in (config, job_lock, job_result)):
            raise ValueError("Harbor job config, lock, and result must be JSON objects")
        configured = _configured_task_names(config)
        if configured != [command.instance_id]:
            raise ValueError(
                f"Harbor config selected {configured!r}, expected {command.instance_id!r}"
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
        # Harbor persists JobConfig with exclude_defaults=True, so explicit
        # CLI values equal to a default are absent from config.json.
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
            raise ValueError("Harbor result is not a completed one-trial job")
        trial_dir, trial_result, trial_result_digest = _read_single_trial_result(
            job_dir
        )
        embedded_trials = job_result.get("trial_results")
        # Harbor 0.20 intentionally excludes trial_results from the persisted
        # job result. If another producer includes them, they cannot disagree.
        if embedded_trials is not None and (
            not isinstance(embedded_trials, list)
            or len(embedded_trials) != 1
            or embedded_trials[0] != trial_result
        ):
            raise ValueError("Harbor job and nested trial results disagree")
        task_name = trial_result.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            raise ValueError("Harbor trial result omitted task_name")
        if task_name != command.instance_id:
            raise ValueError(
                f"Harbor ran {task_name!r}, expected {command.instance_id!r}"
            )
        event_digest, image_id = _validate_trial_evidence(
            trial_dir,
            trial_result,
            protocol=protocol,
            protocol_sha256=sha256_file(protocol_file),
            command=command,
            expected_kwargs=expected_kwargs,
        )
        observed.append(task_name)
        event_digests[task_name] = event_digest
        image_ids[task_name] = image_id
        result_digests[task_name] = trial_result_digest
        job_dirs.append(str(job_dir.resolve()))
    if len(observed) != len(set(observed)) or set(observed) != expected_set:
        raise ValueError("Harbor results do not exactly match the registered instance set")

    manifest = HarborExecutionManifest(
        run_kind=run_kind,
        protocol_sha256=sha256_file(protocol_file),
        expected_instance_ids=expected,
        observed_instance_ids=tuple(observed),
        task_image_digests={item: protocol.task_image_digests[item] for item in expected},
        codex_events_sha256={item: event_digests[item] for item in expected},
        container_image_ids={item: image_ids[item] for item in expected},
        trial_result_sha256={item: result_digests[item] for item in expected},
        job_dirs=tuple(job_dirs),
    )
    if manifest_path is not None:
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


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
        "trusted_local_auth",
        "externally_sandboxed",
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
    if not jobs_dir.is_absolute() or Path(command.job_dir) != jobs_dir / job_name:
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
        "trusted_local_auth": True,
        "externally_sandboxed": True,
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
    datasets = config.get("datasets")
    dataset = datasets[0] if isinstance(datasets, list) and len(datasets) == 1 else None
    if not isinstance(dataset, Mapping) or dataset.get("name") != DATASET:
        raise ValueError("Harbor config changed the official dataset")
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
    task = trial.get("task")
    if (
        not isinstance(task, Mapping)
        or task.get("name") != command.instance_id
        or _SHA256_VALUE_RE.fullmatch(str(task.get("digest", ""))) is None
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
    trial_dir: Path,
    trial_result: Mapping[str, Any],
    *,
    protocol: TerminalBenchProtocol,
    protocol_sha256: str,
    command: HarborRunCommand,
    expected_kwargs: Mapping[str, Any],
) -> tuple[str, str]:
    config = trial_result.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Harbor trial result omitted its resolved config")
    _validate_agent_config(
        config.get("agent"),
        protocol=protocol,
        expected_kwargs=expected_kwargs,
    )
    _validate_docker_environment(config.get("environment"))

    task_checksum = trial_result.get("task_checksum")
    if not isinstance(task_checksum, str) or _SHA256_HEX_RE.fullmatch(task_checksum) is None:
        raise ValueError("Harbor trial result omitted its task checksum")
    agent_info = trial_result.get("agent_info")
    if not isinstance(agent_info, Mapping) or agent_info.get("name") != "lha":
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

    agent_dir = trial_dir / "agent"
    provenance_path = agent_dir / _AGENT_PROVENANCE
    events_path = agent_dir / _CODEX_EVENTS
    try:
        provenance = TerminalBenchAgentProvenance.model_validate_json(
            provenance_path.read_text()
        )
        event_stream = events_path.read_text()
    except (OSError, ValueError) as exc:
        raise ValueError("Harbor trial provenance or Codex JSONL is unreadable") from exc
    try:
        audit = audit_codex_jsonl(
            event_stream,
            max_tool_calls=protocol.budgets.max_tool_calls,
        )
    except RuntimeError as exc:
        raise ValueError("Harbor trial Codex JSONL failed audit") from exc
    events_sha256 = sha256_file(events_path)
    expected_digest = protocol.task_image_digests[command.instance_id]
    if (
        provenance.run_kind != command.run_kind
        or provenance.instance_id != command.instance_id
        or provenance.model != protocol.model
        or provenance.reasoning_effort != protocol.reasoning_effort
        or provenance.harbor_version != protocol.harbor_version
        or provenance.codex_cli_version != protocol.codex_cli_version
        or provenance.codex_target != protocol.codex_target
        or provenance.codex_binary_sha256 != protocol.codex_binary_sha256
        or provenance.task_image_digest != expected_digest
        or provenance.wheel_sha256 != protocol.wheel_sha256
        or provenance.protocol_sha256 != protocol_sha256
        or provenance.subset != protocol.subset
        or provenance.budgets != protocol.budgets
        or provenance.codex_events_sha256 != events_sha256
        or provenance.codex_audit != audit
    ):
        raise ValueError("Harbor trial provenance does not match the protocol")
    if not provenance.image_attestation.proves(expected_digest):
        raise ValueError("runtime Docker evidence does not prove the registered image")
    if agent_info.get("version") != provenance.lha_version:
        raise ValueError("Harbor agent version and wheel provenance disagree")

    agent_result = trial_result.get("agent_result")
    if not isinstance(agent_result, Mapping):
        raise ValueError("Harbor trial result omitted the completed agent result")
    metadata = agent_result.get("metadata")
    expected_metadata = _agent_metadata(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        instance_id=command.instance_id,
        run_kind=command.run_kind,
        audit=audit,
        codex_events_sha256=events_sha256,
        image_attestation=provenance.image_attestation,
        infrastructure_retries_used=provenance.infrastructure_retries_used,
    )
    if metadata != expected_metadata:
        raise ValueError("Harbor agent metadata does not match the audited trial")
    for field, value in (
        ("n_input_tokens", audit.input_tokens),
        ("n_cache_tokens", audit.cached_input_tokens),
        ("n_output_tokens", audit.output_tokens),
    ):
        if agent_result.get(field) != value:
            raise ValueError("Harbor token usage does not match the Codex audit")
    return events_sha256, provenance.image_attestation.image_id


def _agent_metadata(
    *,
    protocol: TerminalBenchProtocol,
    protocol_sha256: str,
    instance_id: str,
    run_kind: Literal["smoke", "scored"],
    audit: CodexRunAudit,
    codex_events_sha256: str,
    image_attestation: DockerImageAttestation,
    infrastructure_retries_used: int,
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
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
        "task_image_digest": protocol.task_image_digests[instance_id],
        "container_id": image_attestation.container_id,
        "container_image_id": image_attestation.image_id,
        "container_configured_image": image_attestation.configured_image,
        "container_repo_digests": list(image_attestation.repo_digests),
        "codex_events_sha256": codex_events_sha256,
        "codex_event_counts": audit.event_counts,
        "codex_item_counts": audit.item_counts,
        "codex_tool_calls": audit.tool_calls,
        "infrastructure_retries_used": infrastructure_retries_used,
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
        """LHA driven by Codex inside Harbor's disposable task container."""

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
            trusted_local_auth: bool = False,
            externally_sandboxed: bool = False,
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
            self._protocol: TerminalBenchProtocol | None = None
            self._trusted_local_auth = trusted_local_auth or (
                os.environ.get("LHA_TBENCH_TRUSTED_LOCAL", "0")
                not in ("0", "false", "False")
            )
            self._externally_sandboxed = externally_sandboxed
            self._codex_version: str | None = None
            self._image_attestation: DockerImageAttestation | None = None
            self._validated_wheel: Path | None = None
            self._agent_started = False
            self._task_run_consumed = False
            self._infrastructure_retries_used = 0
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
        ) -> tuple[Path, Path, Path, TerminalBenchProtocol, str]:
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
            if (
                not self._auth_source
                or not self._trusted_local_auth
                or not self._externally_sandboxed
            ):
                raise RuntimeError(
                    "Codex auth requires an explicit auth_path/LHA_CODEX_AUTH_FILE and "
                    "trusted_local_auth=True, and danger-full-access requires "
                    "externally_sandboxed=True"
                )
            auth = Path(self._auth_source)
            if not auth.is_file():
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
            return wheel, codex_binary, auth, protocol, self._instance_id

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

        async def _before_agent_operation(
            self,
            label: str,
            operation: Callable[[], Awaitable[_T]],
            *,
            succeeded: Callable[[_T], bool] | None = None,
        ) -> _T:
            """Share one retry across every setup operation in this trial."""
            if self._agent_started:
                raise RuntimeError("infrastructure retry attempted after agent start")
            while True:
                try:
                    value = await operation()
                    if succeeded is not None and not succeeded(value):
                        raise RuntimeError(f"{label} returned a non-zero status")
                    return value
                except Exception as exc:
                    if self._infrastructure_retries_used >= 1:
                        raise RuntimeError(
                            f"{label} failed; the one shared infrastructure retry "
                            "is already exhausted"
                        ) from exc
                    self._infrastructure_retries_used += 1

        async def install(self, environment) -> None:
            wheel, codex_binary, _auth, protocol, instance_id = self._validate_inputs()
            image_attestation = await self._before_agent_operation(
                "Harbor Docker image attestation",
                lambda: _attest_harbor_docker_image(environment),
            )
            expected_image_digest = protocol.task_image_digests[instance_id]
            if not image_attestation.proves(expected_image_digest):
                raise RuntimeError(
                    "the running Harbor container does not match the preregistered "
                    "task-image digest"
                )
            self._image_attestation = image_attestation
            self._validated_wheel = wheel
            await self._before_agent_operation(
                "Codex binary upload",
                lambda: environment.upload_file(str(codex_binary), _CODEX_UPLOAD),
            )
            for command in install_commands(codex_cli_version=protocol.codex_cli_version):
                await self._before_agent_operation(
                    "Codex CLI installation",
                    lambda command=command: environment.exec(command=command, user="root"),
                    succeeded=lambda result: result.return_code == 0,
                )
            version = await self._before_agent_operation(
                "Codex CLI version check",
                lambda: environment.exec(command="codex --version"),
                succeeded=lambda result: result.return_code == 0,
            )
            self._codex_version = (version.stdout or version.stderr or "").strip()
            if not self._codex_version:
                raise RuntimeError("Codex CLI returned an empty version")
            if self._codex_version != protocol.codex_cli_version:
                raise RuntimeError("installed Codex CLI does not match the preregistration")
            self._write_provenance(wheel, instance_id)

        def _write_provenance(
            self,
            wheel: Path,
            instance_id: str,
            *,
            audit: CodexRunAudit | None = None,
            codex_events_sha256: str | None = None,
        ) -> None:
            assert self._model is not None
            assert self._reasoning_effort is not None
            assert self._codex_version is not None
            assert self._protocol is not None
            assert self._image_attestation is not None
            assert self._run_kind in ("smoke", "scored")
            record = TerminalBenchAgentProvenance(
                lha_version=__version__,
                run_kind=self._run_kind,
                instance_id=instance_id,
                model=self._model,
                reasoning_effort=self._reasoning_effort,
                harbor_version=self._protocol.harbor_version,
                codex_cli_version=self._codex_version,
                codex_target=self._protocol.codex_target,
                codex_binary_sha256=self._protocol.codex_binary_sha256,
                task_image_digest=self._protocol.task_image_digests[instance_id],
                image_attestation=self._image_attestation,
                wheel_sha256=sha256_file(wheel),
                protocol_sha256=sha256_file(Path(self._protocol_path or "")),
                subset=self._protocol.subset,
                budgets=self._protocol.budgets,
                infrastructure_retries_used=self._infrastructure_retries_used,
                codex_events_sha256=codex_events_sha256,
                codex_audit=audit,
            )
            self._logs_dir.mkdir(parents=True, exist_ok=True)
            (self._logs_dir / _AGENT_PROVENANCE).write_text(
                record.model_dump_json(indent=2) + "\n"
            )

        async def _prepare_auth(self, environment, auth: Path) -> None:
            async def stage():
                try:
                    await environment.upload_file(str(auth), _AUTH_UPLOAD)
                    staged = await environment.exec(
                        command=(
                            "umask 077 && "
                            f"rm -rf {_CODEX_HOME} && mkdir -m 700 {_CODEX_HOME} && "
                            f"install -m 600 {_AUTH_UPLOAD} {_CODEX_HOME}/auth.json && "
                            f"rm -f {_AUTH_UPLOAD}"
                        )
                    )
                    if staged.return_code != 0:
                        raise RuntimeError("credential staging command failed")
                    return staged
                except Exception:
                    await environment.exec(
                        command=f"rm -rf {_CODEX_HOME} && rm -f {_AUTH_UPLOAD}"
                    )
                    raise

            await self._before_agent_operation("Codex credential staging", stage)

        @staticmethod
        async def _cleanup_auth(environment) -> None:
            """Finish credential deletion even if the surrounding task is cancelled."""
            import asyncio

            cleanup = asyncio.create_task(
                environment.exec(command=f"rm -rf {_CODEX_HOME} && rm -f {_AUTH_UPLOAD}")
            )
            try:
                result = await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # Shield keeps the environment operation alive; wait for it before
                # propagating cancellation so auth does not outlive this attempt.
                await cleanup
                raise
            if result.return_code != 0:
                raise RuntimeError("failed to remove task-local Codex credentials")

        async def run(self, instruction: str, environment, context) -> None:
            if self._agent_started or self._task_run_consumed:
                raise RuntimeError("this Harbor agent permits exactly one Codex run")
            if (
                self._validated_wheel is None
                or self._image_attestation is None
                or self._codex_version is None
            ):
                raise RuntimeError("Harbor agent install and image attestation must run first")
            _wheel, _codex_binary, auth, protocol, instance_id = self._validate_inputs()
            self._task_run_consumed = True
            try:
                await self._prepare_auth(environment, auth)
                self._agent_started = True
                assert self._model is not None
                assert self._reasoning_effort is not None
                result = await environment.exec(
                    command=codex_exec_command(
                        self._model,
                        self._reasoning_effort,
                        instruction,
                    ),
                    timeout_sec=protocol.budgets.timeout_s,
                )
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                self._logs_dir.mkdir(parents=True, exist_ok=True)
                (self._logs_dir / _CODEX_EVENTS).write_text(stdout)
                (self._logs_dir / _CODEX_STDERR).write_text(stderr)
                if result.return_code != 0:
                    raise RuntimeError(
                        f"the single Codex run exited {result.return_code}: {stderr[-400:]}"
                    )
                audit = audit_codex_jsonl(
                    stdout,
                    max_tool_calls=protocol.budgets.max_tool_calls,
                )
                assert self._validated_wheel is not None
                events_sha256 = sha256_file(self._logs_dir / _CODEX_EVENTS)
                self._write_provenance(
                    self._validated_wheel,
                    instance_id,
                    audit=audit,
                    codex_events_sha256=events_sha256,
                )
                self._fill_usage(context, audit, instance_id)
            finally:
                # Cleanup is attempted for success, failure, cancellation, and
                # exceptions.  A fixed container path keeps the host secret path
                # out of shell commands and agent logs.
                try:
                    await self._cleanup_auth(environment)
                finally:
                    self._agent_started = False

        def _fill_usage(
            self,
            context,
            audit: CodexRunAudit,
            instance_id: str,
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
                    codex_events_sha256=sha256_file(self._logs_dir / _CODEX_EVENTS),
                    image_attestation=self._image_attestation,
                    infrastructure_retries_used=self._infrastructure_retries_used,
                )

    return LhaAgent


try:  # Harbor import-path loading imports this module before looking up the class.
    LhaAgent = build_agent()
except ImportError:  # Core lha remains importable on Python 3.11 without the bench extra.
    class LhaAgent:  # type: ignore[no-redef]
        """Placeholder replaced by the real Harbor subclass when Harbor is installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("LhaAgent requires harbor>=0.20 on Python >=3.12")
