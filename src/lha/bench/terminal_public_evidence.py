"""Secret-free Terminal-Bench evidence that can be checked without run directories.

The live adapter validates considerably more evidence than is suitable for a
public repository: control records, Codex JSONL, broker receipts, and Harbor
logs stay in the private run directory.  For the committed fixed-20 run, the
16 PASS/FAIL files retain the official Harbor ``result.json`` bytes.  The four
ERROR files are deterministic redacted projections bound to the official
source SHA-256 values.  Those commitments detect substitution; the public
package alone cannot reproduce or disclose the private tracebacks.

The offline validator does not replace live validation.  It checks that the
published files are the output of that validation, re-derives every scored row
from the public trial evidence, and recomputes the fixed-20 summary.  Schema 4
also binds the evaluated Git commit, Git tree, package version, and wheel hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .terminal_bench import (
    AGENT_IMPORT_PATH,
    DATASET,
    HarborExecutionManifest,
    HarborRunCommand,
    TerminalBenchProtocol,
    TerminalBenchRecordBatch,
    TerminalBenchSummary,
    TerminalBenchTaskRecord,
    _official_infrastructure_retries,
    _official_trial_duration,
    _official_trial_outcome,
    _percentile,
    _read_single_trial_result,
    derive_terminal_bench_records,
    summarize_records,
)
from .terminal_control import SmokeSeal, terminal_attempt_id

_INDEX_FILE = "evidence.json"
_PROTOCOL_FILE = "protocol.json"
_SMOKE_MANIFEST_FILE = "smoke_manifest.json"
_SMOKE_SEAL_FILE = "smoke_seal.json"
_SCORED_MANIFEST_FILE = "scored_manifest.json"
_RECORDS_FILE = "records.json"
_SUMMARY_FILE = "summary.json"
_SUMMARY_MARKDOWN_FILE = "summary.md"
_SOURCE_ATTESTATION_FILE = "source_attestation.json"
_FIXED_FILES = frozenset(
    {
        _INDEX_FILE,
        _PROTOCOL_FILE,
        _SMOKE_MANIFEST_FILE,
        _SMOKE_SEAL_FILE,
        _SCORED_MANIFEST_FILE,
        _RECORDS_FILE,
        _SUMMARY_FILE,
        _SUMMARY_MARKDOWN_FILE,
    }
)
_SCHEMA4_FIXED_FILES = _FIXED_FILES | {_SOURCE_ATTESTATION_FILE}
_TRIAL_PATH_RE = re.compile(r"^trials/[0-9]{2}-[0-9a-f]{12}\.json$")
_EXCEPTION_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_PACKAGE_VERSION_RE = re.compile(r"^[0-9A-Za-z](?:[0-9A-Za-z.!+_-]{0,126})$")
_WHEEL_FILENAME_RE = re.compile(
    r"^lha-[0-9A-Za-z.!+_]+-(?:[0-9][0-9A-Za-z.]*-)?"
    r"[0-9A-Za-z_.]+-[0-9A-Za-z_.]+-[0-9A-Za-z_.]+\.whl$"
)
_MAX_INDEX_BYTES = 256 * 1024
_MAX_PROTOCOL_BYTES = 256 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_RECORDS_BYTES = 512 * 1024
_MAX_SUMMARY_BYTES = 64 * 1024
_MAX_TRIAL_BYTES = 2 * 1024 * 1024
_MAX_PACKAGE_BYTES = 64 * 1024 * 1024
_FORBIDDEN_KEYS = frozenset(
    {
        "access_token",
        "account_id",
        "api_key",
        "auth",
        "auth_path",
        "authorization",
        "capability",
        "capability_token",
        "client_secret",
        "cookie",
        "credentials",
        "id_token",
        "password",
        "refresh_token",
        "secret",
        "session_cookie",
        "session_token",
        "tokens",
    }
)
_FULL_LOG_KEYS = frozenset(
    {
        "codex_events",
        "event_stream",
        "exception_traceback",
        "logs",
        "stderr",
        "stdout",
        "trajectory",
    }
)
_FORBIDDEN_TEXT = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\bsk-(?:proj-)?[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)LHA_CODEX_AUTH_FILE"),
    re.compile(r"(?i)(?:^|[/\\])\.codex[/\\](?:auth|credentials)(?:\.json)?"),
    re.compile(
        r"(?i)(?:^|[/\\])(?:auth|authentication|credentials?|"
        r"codex[-_.]?auth)(?:[-_.][A-Za-z0-9]+)*\.json(?:$|[\s'\"?,])"
    ),
    re.compile(r"(?i)capability_[A-Za-z0-9_-]{12,}"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b"
    ),
)
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class PublicHarborTrial(BaseModel):
    """Digest and derived fields for one exported official trial result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_kind: Literal["official", "redacted_error"]
    raw_status: Literal["PASS", "FAIL", "ERROR"]
    raw_correct: bool | None
    raw_protocol_error: str | None
    exception_type: str | None
    duration_s: FiniteFloat | None = Field(default=None, ge=0)
    infrastructure_retries: Literal[0] | None = None

    @model_validator(mode="after")
    def _raw_outcome_is_consistent(self) -> "PublicHarborTrial":
        expected_correct = {"PASS": True, "FAIL": False, "ERROR": None}[
            self.raw_status
        ]
        if self.raw_correct is not expected_correct:
            raise ValueError("raw_correct does not match the official raw status")
        if self.raw_status == "ERROR":
            if self.raw_protocol_error is None or not self.raw_protocol_error.strip():
                raise ValueError("raw ERROR evidence requires a stable explanation")
            if self.payload_kind != "redacted_error":
                raise ValueError("raw ERROR evidence must use the redacted projection")
        elif self.raw_protocol_error is not None:
            raise ValueError("raw PASS and FAIL evidence may not claim a protocol error")
        elif self.payload_kind != "official":
            raise ValueError("raw PASS and FAIL evidence must keep the official payload")
        if self.payload_kind == "official" and self.payload_sha256 != self.source_sha256:
            raise ValueError("an official public payload must equal its source bytes")
        if self.exception_type is not None:
            if (
                self.raw_status != "ERROR"
                or _EXCEPTION_TYPE_RE.fullmatch(self.exception_type) is None
                or self.raw_protocol_error
                != f"Harbor trial exception: {self.exception_type}"
            ):
                raise ValueError("public exception type is inconsistent with the raw outcome")
        elif self.raw_protocol_error is not None and self.raw_protocol_error.startswith(
            "Harbor trial exception:"
        ):
            raise ValueError("public exception evidence omitted its exception type")
        if _TRIAL_PATH_RE.fullmatch(self.path) is None:
            raise ValueError("trial evidence path is not a fixed safe relative path")
        return self


class PublicHarborErrorProjection(BaseModel):
    """Minimal deterministic projection of an official Harbor ERROR result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["terminal-bench-error-redaction"] = "terminal-bench-error-redaction"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_name: str = Field(min_length=1)
    task_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_status: Literal["ERROR"] = "ERROR"
    raw_correct: None = None
    raw_protocol_error: str = Field(min_length=1)
    exception_type: str | None = None
    duration_s: FiniteFloat | None = Field(default=None, ge=0)
    infrastructure_retries: Literal[0] | None = None

    @model_validator(mode="after")
    def _error_fields_are_safe_and_consistent(
        self,
    ) -> "PublicHarborErrorProjection":
        if self.exception_type is not None:
            if (
                _EXCEPTION_TYPE_RE.fullmatch(self.exception_type) is None
                or self.raw_protocol_error
                != f"Harbor trial exception: {self.exception_type}"
            ):
                raise ValueError("redacted exception type is unsafe or inconsistent")
        elif self.raw_protocol_error not in {
            "Harbor trial omitted verifier_result",
            "Harbor verifier omitted rewards",
            "Harbor verifier omitted the official reward",
        }:
            raise ValueError("redacted ERROR reason is not an official stable outcome")
        _assert_public_string(self.task_name, label="redacted ERROR task name")
        _assert_public_string(
            self.raw_protocol_error,
            label="redacted ERROR explanation",
        )
        return self


class SourceAttestation(BaseModel):
    """Public identity of the source tree and wheel used by Harbor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    repository_url: str = Field(min_length=1, max_length=512)
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    tree_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    package_version: str = Field(min_length=1, max_length=127)
    wheel_filename: str = Field(min_length=1, max_length=255)
    wheel_size_bytes: int = Field(gt=0, le=_MAX_PACKAGE_BYTES)
    wheel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reproducible_build_command: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _public_source_identity_is_safe(self) -> "SourceAttestation":
        parsed = urlsplit(self.repository_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("source repository URL has an invalid port") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or port not in (None, 443)
            or parsed.query
            or parsed.fragment
            or not parsed.path.strip("/")
            or self.repository_url != self.repository_url.strip()
        ):
            raise ValueError("source repository URL must be a public HTTPS repository")
        if _PACKAGE_VERSION_RE.fullmatch(self.package_version) is None:
            raise ValueError("source package version is malformed")
        if (
            _WHEEL_FILENAME_RE.fullmatch(self.wheel_filename) is None
            or not self.wheel_filename.startswith(f"lha-{self.package_version}-")
        ):
            raise ValueError("source wheel filename does not match the LHA package version")
        if (
            self.reproducible_build_command != self.reproducible_build_command.strip()
            or any(
                character in self.reproducible_build_command
                for character in ("\0", "\r", "\n")
            )
        ):
            raise ValueError("reproducible build command must be one non-empty line")
        for label, value in (
            ("source repository URL", self.repository_url),
            ("source package version", self.package_version),
            ("source wheel filename", self.wheel_filename),
            ("reproducible build command", self.reproducible_build_command),
        ):
            _assert_public_string(value, label=label)
        return self


class TerminalBenchPublicEvidenceIndex(BaseModel):
    """Content-addressed index for one public fixed-subset result package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3, 4] = 3
    dataset: Literal["terminal-bench/terminal-bench-2-1"] = DATASET
    evaluation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    public_path_root: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    smoke_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    smoke_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scored_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_markdown_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_attestation_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    trials: tuple[PublicHarborTrial, ...]

    @model_validator(mode="after")
    def _trial_index_is_unique(self) -> "TerminalBenchPublicEvidenceIndex":
        instance_ids = [trial.instance_id for trial in self.trials]
        paths = [trial.path for trial in self.trials]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("public trial evidence contains duplicate instance ids")
        if len(paths) != len(set(paths)):
            raise ValueError("public trial evidence contains duplicate paths")
        if (self.schema_version == 4) is (self.source_attestation_sha256 is None):
            raise ValueError("schema 4 requires exactly one source attestation digest")
        _normalized_public_root(self.public_path_root)
        return self


class TerminalBenchPublicEvidenceValidation(BaseModel):
    """Offline validation result and the digest to anchor in a Git commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scored_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluated_commit_sha: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
    )
    evaluated_tree_sha: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
    )
    evaluated_wheel_filename: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )
    evaluated_wheel_size_bytes: int | None = Field(
        default=None,
        gt=0,
        le=_MAX_PACKAGE_BYTES,
    )
    evaluated_wheel_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    model: str = Field(min_length=1)
    reasoning_effort: str = Field(min_length=1)
    harbor_version: str = Field(min_length=1)
    denominator: Literal[20] = 20
    passed: int = Field(ge=0, le=20)
    failed: int = Field(ge=0, le=20)
    errors: int = Field(ge=0, le=20)

    @model_validator(mode="after")
    def _all_rows_are_accounted_for(
        self,
    ) -> "TerminalBenchPublicEvidenceValidation":
        if self.passed + self.failed + self.errors != 20:
            raise ValueError("public evidence must account for exactly 20 tasks")
        source_identity = (
            self.evaluated_commit_sha,
            self.evaluated_tree_sha,
            self.evaluated_wheel_filename,
            self.evaluated_wheel_size_bytes,
            self.evaluated_wheel_sha256,
        )
        if any(value is None for value in source_identity) and not all(
            value is None for value in source_identity
        ):
            raise ValueError(
                "evaluated commit, tree, and wheel identity must be reported together"
            )
        return self


def export_terminal_bench_public_evidence(
    protocol: TerminalBenchProtocol,
    *,
    protocol_path: str | Path,
    smoke_manifest: HarborExecutionManifest,
    smoke_manifest_path: str | Path,
    smoke_seal: SmokeSeal,
    smoke_seal_path: str | Path,
    scored_manifest: HarborExecutionManifest,
    scored_manifest_path: str | Path,
    records: TerminalBenchRecordBatch,
    summary: TerminalBenchSummary,
    scored_commands: Sequence[HarborRunCommand],
    output_dir: str | Path,
    public_path_root: str | Path,
    auth_parent: str | Path,
) -> TerminalBenchPublicEvidenceValidation:
    """Export a validated, minimal package without private model or broker logs."""
    public_root = _normalized_public_root(public_path_root)
    private_roots = (
        Path.home().resolve(),
        Path(__file__).resolve().parents[3],
        Path(auth_parent).resolve(),
    )
    if any(
        public_root == private_root or public_root.is_relative_to(private_root)
        for private_root in private_roots
    ):
        raise ValueError("public path root overlaps a private path root")
    protocol_bytes = _load_exact_model(
        protocol_path,
        protocol,
        maximum_bytes=_MAX_PROTOCOL_BYTES,
        label="protocol",
    )
    smoke_bytes = _read_regular_file(
        Path(smoke_manifest_path).resolve(),
        maximum_bytes=_MAX_MANIFEST_BYTES,
        label="smoke manifest",
    )
    if _load_model_bytes(
        smoke_bytes,
        HarborExecutionManifest,
        label="smoke manifest",
    ) != smoke_manifest:
        raise ValueError("smoke manifest path does not contain the supplied model")
    smoke_seal_bytes = _read_regular_file(
        Path(smoke_seal_path).resolve(),
        maximum_bytes=_MAX_SUMMARY_BYTES,
        label="smoke seal",
    )
    if _load_model_bytes(
        smoke_seal_bytes,
        SmokeSeal,
        label="smoke seal",
    ) != smoke_seal:
        raise ValueError("smoke seal path does not contain the supplied model")
    scored_bytes = _load_exact_model(
        scored_manifest_path,
        scored_manifest,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        label="scored manifest",
    )
    protocol_digest = hashlib.sha256(protocol_bytes).hexdigest()
    smoke_digest = hashlib.sha256(smoke_bytes).hexdigest()
    smoke_seal_digest = hashlib.sha256(smoke_seal_bytes).hexdigest()
    scored_digest = hashlib.sha256(scored_bytes).hexdigest()
    _validate_manifest_binding(
        smoke_manifest,
        protocol=protocol,
        protocol_sha256=protocol_digest,
        run_kind="smoke",
    )
    _validate_smoke_seal_binding(
        smoke_seal,
        protocol=protocol,
        protocol_sha256=protocol_digest,
        smoke_manifest=smoke_manifest,
        smoke_manifest_sha256=smoke_digest,
    )
    _validate_manifest_binding(
        scored_manifest,
        protocol=protocol,
        protocol_sha256=protocol_digest,
        run_kind="scored",
    )

    official_records = derive_terminal_bench_records(
        protocol,
        scored_commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    if records != official_records:
        raise ValueError("records differ from the current validated Harbor results")
    official_summary = summarize_records(
        protocol,
        official_records,
        commands=scored_commands,
        protocol_path=protocol_path,
        execution_manifest=scored_manifest,
        manifest_path=scored_manifest_path,
    )
    if summary != official_summary:
        raise ValueError("summary differs from the current validated Harbor results")

    records_bytes = _canonical_model_bytes(records)
    summary_bytes = _canonical_model_bytes(summary)
    summary_markdown_bytes = _summary_markdown_bytes(summary, schema_version=3)
    command_by_id = {command.instance_id: command for command in scored_commands}
    if (
        len(command_by_id) != len(scored_commands)
        or tuple(command_by_id) != protocol.subset.scored_instance_ids
    ):
        raise ValueError("scored commands do not match the registered task order")

    trial_payloads: dict[str, bytes] = {}
    trial_index: list[PublicHarborTrial] = []
    for number, instance_id in enumerate(
        protocol.subset.scored_instance_ids,
        start=1,
    ):
        expected_digest = scored_manifest.trial_result_sha256[instance_id]
        if expected_digest is None:
            continue
        command = command_by_id[instance_id]
        _trial_dir, trial_result, observed_digest = _read_single_trial_result(
            Path(command.job_dir)
        )
        if observed_digest != expected_digest:
            raise ValueError(f"official Harbor result changed for {instance_id}")
        trial_path, raw = _read_live_trial_bytes(Path(command.job_dir))
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ValueError(f"official Harbor result bytes changed for {instance_id}")
        if trial_path.parent.parent != Path(command.job_dir):
            raise ValueError("official Harbor result moved outside its registered job")
        raw_status, raw_correct, raw_error = _official_trial_outcome(trial_result)
        duration = _official_trial_duration(trial_result)
        retries = _official_infrastructure_retries(trial_result)
        exception_type = _official_exception_type(trial_result)
        if raw_status == "ERROR":
            if raw_error is None:
                raise ValueError("official Harbor ERROR omitted its stable explanation")
            _validate_raw_trial_binding(
                trial_result,
                protocol=protocol,
                protocol_sha256=protocol_digest,
                instance_id=instance_id,
                smoke_seal_sha256=smoke_seal_digest,
                public_path_root=public_root,
            )
            projection = PublicHarborErrorProjection(
                source_sha256=expected_digest,
                task_name=instance_id,
                task_checksum=protocol.task_checksums[instance_id],
                raw_protocol_error=raw_error,
                exception_type=exception_type,
                duration_s=duration,
                infrastructure_retries=retries,
            )
            public_trial_bytes = _canonical_model_bytes(projection)
            payload_kind: Literal["official", "redacted_error"] = "redacted_error"
        else:
            _assert_public_payload(trial_result, label=f"trial {instance_id}")
            public_trial_bytes = raw
            payload_kind = "official"
        relative_path = (
            f"trials/{number:02d}-"
            f"{hashlib.sha256(instance_id.encode()).hexdigest()[:12]}.json"
        )
        trial_payloads[relative_path] = public_trial_bytes
        trial_index.append(
            PublicHarborTrial(
                instance_id=instance_id,
                path=relative_path,
                source_sha256=expected_digest,
                payload_sha256=hashlib.sha256(public_trial_bytes).hexdigest(),
                payload_kind=payload_kind,
                raw_status=raw_status,
                raw_correct=raw_correct,
                raw_protocol_error=raw_error,
                exception_type=exception_type,
                duration_s=duration,
                infrastructure_retries=retries,
            )
        )

    index = TerminalBenchPublicEvidenceIndex(
        evaluation_id=protocol.evaluation_id,
        public_path_root=str(public_root),
        protocol_sha256=protocol_digest,
        smoke_manifest_sha256=smoke_digest,
        smoke_seal_sha256=smoke_seal_digest,
        scored_manifest_sha256=scored_digest,
        records_sha256=hashlib.sha256(records_bytes).hexdigest(),
        summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        summary_markdown_sha256=hashlib.sha256(summary_markdown_bytes).hexdigest(),
        trials=tuple(trial_index),
    )
    public_payloads = {
        _PROTOCOL_FILE: protocol_bytes,
        _SMOKE_MANIFEST_FILE: smoke_bytes,
        _SMOKE_SEAL_FILE: smoke_seal_bytes,
        _SCORED_MANIFEST_FILE: scored_bytes,
        _RECORDS_FILE: records_bytes,
        _SUMMARY_FILE: summary_bytes,
        _SUMMARY_MARKDOWN_FILE: summary_markdown_bytes,
        **trial_payloads,
        _INDEX_FILE: _canonical_model_bytes(index),
    }
    _assert_public_package_payloads(public_payloads)
    _assert_private_roots_absent(public_payloads, private_roots)

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"public evidence directory already exists: {target}")
    _durable_mkdir_chain(target.parent)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            dir=target.parent,
        )
    )
    try:
        for relative_path, payload in public_payloads.items():
            _write_new_file(temporary / relative_path, payload)
        validation = validate_terminal_bench_public_evidence(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validation


def upgrade_terminal_bench_public_evidence(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    source_attestation: SourceAttestation,
) -> TerminalBenchPublicEvidenceValidation:
    """Copy a validated schema-3 package into an attested schema-4 package.

    The source is never changed.  Files are copied through bounded, no-follow
    reads into a sibling temporary directory, the new package is validated
    offline, and only then is the complete directory published.  The upgrade
    also regenerates ``summary.md`` with the current wording while preserving
    the measured JSON summary.
    """
    validate_terminal_bench_public_evidence(source_dir)
    supplied_source = Path(source_dir)
    if supplied_source.is_symlink():
        raise ValueError("schema-3 evidence root may not be a symlink")
    source_root = supplied_source.resolve()

    index_bytes = _read_regular_file(
        source_root / _INDEX_FILE,
        maximum_bytes=_MAX_INDEX_BYTES,
        label="schema-3 evidence index",
    )
    source_index = _load_model_bytes(
        index_bytes,
        TerminalBenchPublicEvidenceIndex,
        label="schema-3 evidence index",
    )
    if source_index.schema_version != 3:
        raise ValueError("only a validated schema-3 package can be upgraded")
    source_files = {
        *_FIXED_FILES,
        *(trial.path for trial in source_index.trials),
    }
    if _list_package_files(source_root) != source_files:
        raise ValueError("schema-3 evidence changed after validation")

    protocol_bytes = _read_regular_file(
        source_root / _PROTOCOL_FILE,
        maximum_bytes=_MAX_PROTOCOL_BYTES,
        label="schema-3 protocol",
    )
    protocol = _load_model_bytes(
        protocol_bytes,
        TerminalBenchProtocol,
        label="schema-3 protocol",
    )
    if source_attestation.wheel_sha256 != protocol.wheel_sha256:
        raise ValueError("source attestation wheel does not match the frozen protocol")

    payloads = {
        relative: _read_regular_file(
            source_root / relative,
            maximum_bytes=_maximum_for_relative_path(relative),
            label=f"schema-3 {relative}",
        )
        for relative in source_files
        if relative != _INDEX_FILE
    }
    summary = _load_model_bytes(
        payloads[_SUMMARY_FILE],
        TerminalBenchSummary,
        label="schema-3 summary",
    )
    summary_markdown_bytes = _summary_markdown_bytes(summary, schema_version=4)
    attestation_bytes = _canonical_model_bytes(source_attestation)
    upgraded_index = TerminalBenchPublicEvidenceIndex.model_validate(
        {
            **source_index.model_dump(mode="json"),
            "schema_version": 4,
            "summary_markdown_sha256": hashlib.sha256(
                summary_markdown_bytes
            ).hexdigest(),
            "source_attestation_sha256": hashlib.sha256(
                attestation_bytes
            ).hexdigest(),
        }
    )
    payloads[_SUMMARY_MARKDOWN_FILE] = summary_markdown_bytes
    payloads[_SOURCE_ATTESTATION_FILE] = attestation_bytes
    payloads[_INDEX_FILE] = _canonical_model_bytes(upgraded_index)
    _assert_public_package_payloads(payloads)

    supplied_target = Path(output_dir)
    if supplied_target.is_symlink() or supplied_target.exists():
        raise FileExistsError(
            f"attested public evidence directory already exists: {supplied_target}"
        )
    target = supplied_target.resolve()
    if (
        target == source_root
        or target.is_relative_to(source_root)
        or source_root.is_relative_to(target)
    ):
        raise ValueError("source and attested evidence directories may not overlap")
    _durable_mkdir_chain(target.parent)
    lock_path = target.parent / f".{target.name}.attestation.lock"
    lock_descriptor = _open_upgrade_lock(lock_path)
    temporary: Path | None = None
    try:
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"attested public evidence directory already exists: {target}"
            )
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.",
                dir=target.parent,
            )
        )
        for relative_path, payload in payloads.items():
            _write_new_file(temporary / relative_path, payload)
        validation = validate_terminal_bench_public_evidence(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"attested public evidence directory already exists: {target}"
            )
        os.replace(temporary, target)
        temporary = None
        _fsync_directory(target.parent)
        return validation
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        os.close(lock_descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(target.parent)


def validate_terminal_bench_public_evidence(
    package_dir: str | Path,
) -> TerminalBenchPublicEvidenceValidation:
    """Recompute a public result using only files inside ``package_dir``.

    Schema 3 remains readable for historical evidence.  A schema-3 result has
    no evaluated commit in the return value because its index did not bind a
    source attestation.
    """
    supplied_root = Path(package_dir)
    if supplied_root.is_symlink():
        raise ValueError("public evidence root may not be a symlink")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise ValueError("public evidence root must be a real directory")

    index_bytes = _read_regular_file(
        root / _INDEX_FILE,
        maximum_bytes=_MAX_INDEX_BYTES,
        label="evidence index",
    )
    index = _load_model_bytes(
        index_bytes,
        TerminalBenchPublicEvidenceIndex,
        label="evidence index",
    )
    if index_bytes != _canonical_model_bytes(index):
        raise ValueError("evidence index is not in canonical exported form")

    fixed_files = (
        _SCHEMA4_FIXED_FILES if index.schema_version == 4 else _FIXED_FILES
    )
    expected_files = {
        *fixed_files,
        *(trial.path for trial in index.trials),
    }
    observed_files = _list_package_files(root)
    if observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        unexpected = sorted(observed_files - expected_files)
        raise ValueError(
            "public evidence file set changed "
            f"(missing={missing!r}, unexpected={unexpected!r})"
        )

    protocol_bytes = _read_regular_file(
        root / _PROTOCOL_FILE,
        maximum_bytes=_MAX_PROTOCOL_BYTES,
        label="protocol",
    )
    protocol = _load_model_bytes(
        protocol_bytes,
        TerminalBenchProtocol,
        label="protocol",
    )
    if protocol_bytes != _canonical_model_bytes(protocol):
        raise ValueError("protocol is not in canonical exported form")
    protocol_digest = hashlib.sha256(protocol_bytes).hexdigest()
    if (
        protocol.evaluation_id != index.evaluation_id
        or protocol_digest != index.protocol_sha256
    ):
        raise ValueError("public protocol does not match the evidence index")
    public_path_root = _normalized_public_root(index.public_path_root)
    if not _path_is_within(protocol.output_root, public_path_root):
        raise ValueError("public protocol output root is outside its neutral path root")

    source_attestation: SourceAttestation | None = None
    if index.schema_version == 4:
        attestation_bytes = _read_regular_file(
            root / _SOURCE_ATTESTATION_FILE,
            maximum_bytes=_MAX_SUMMARY_BYTES,
            label="source attestation",
        )
        source_attestation = _load_model_bytes(
            attestation_bytes,
            SourceAttestation,
            label="source attestation",
        )
        if (
            attestation_bytes != _canonical_model_bytes(source_attestation)
            or hashlib.sha256(attestation_bytes).hexdigest()
            != index.source_attestation_sha256
        ):
            raise ValueError("public source attestation changed")
        if source_attestation.wheel_sha256 != protocol.wheel_sha256:
            raise ValueError("evaluated source wheel does not match the frozen protocol")

    smoke_bytes = _read_regular_file(
        root / _SMOKE_MANIFEST_FILE,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        label="smoke manifest",
    )
    smoke_manifest = _load_model_bytes(
        smoke_bytes,
        HarborExecutionManifest,
        label="smoke manifest",
    )
    if hashlib.sha256(smoke_bytes).hexdigest() != index.smoke_manifest_sha256:
        raise ValueError("public smoke manifest changed")
    _validate_manifest_binding(
        smoke_manifest,
        protocol=protocol,
        protocol_sha256=protocol_digest,
        run_kind="smoke",
    )
    smoke_seal_bytes = _read_regular_file(
        root / _SMOKE_SEAL_FILE,
        maximum_bytes=_MAX_SUMMARY_BYTES,
        label="smoke seal",
    )
    smoke_seal = _load_model_bytes(
        smoke_seal_bytes,
        SmokeSeal,
        label="smoke seal",
    )
    smoke_seal_digest = hashlib.sha256(smoke_seal_bytes).hexdigest()
    if smoke_seal_digest != index.smoke_seal_sha256:
        raise ValueError("public smoke seal changed")
    _validate_smoke_seal_binding(
        smoke_seal,
        protocol=protocol,
        protocol_sha256=protocol_digest,
        smoke_manifest=smoke_manifest,
        smoke_manifest_sha256=index.smoke_manifest_sha256,
    )

    scored_bytes = _read_regular_file(
        root / _SCORED_MANIFEST_FILE,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        label="scored manifest",
    )
    scored_manifest = _load_model_bytes(
        scored_bytes,
        HarborExecutionManifest,
        label="scored manifest",
    )
    scored_digest = hashlib.sha256(scored_bytes).hexdigest()
    if (
        scored_bytes != _canonical_model_bytes(scored_manifest)
        or scored_digest != index.scored_manifest_sha256
    ):
        raise ValueError("public scored manifest changed")
    _validate_manifest_binding(
        scored_manifest,
        protocol=protocol,
        protocol_sha256=protocol_digest,
        run_kind="scored",
    )

    indexed_trials = {trial.instance_id: trial for trial in index.trials}
    expected_trial_ids = tuple(
        instance_id
        for instance_id in protocol.subset.scored_instance_ids
        if scored_manifest.trial_result_sha256[instance_id] is not None
    )
    if tuple(trial.instance_id for trial in index.trials) != expected_trial_ids:
        raise ValueError("public trial index does not match the scored manifest")

    derived_records: list[TerminalBenchTaskRecord] = []
    for instance_id in protocol.subset.scored_instance_ids:
        trial_digest = scored_manifest.trial_result_sha256[instance_id]
        has_public_trial = False
        raw_duration: float | None = None
        raw_retries: Literal[0] | None = None
        raw_status: Literal["PASS", "FAIL", "ERROR"] | None = None
        raw_correct: bool | None = None
        raw_error: str | None = None
        exception_type: str | None = None
        if trial_digest is not None:
            trial_index = indexed_trials.get(instance_id)
            if trial_index is None or trial_index.source_sha256 != trial_digest:
                raise ValueError(f"public trial digest is missing for {instance_id}")
            raw_bytes = _read_indexed_trial(root, trial_index)
            if trial_index.payload_kind == "official":
                raw_trial = _load_json_object(
                    raw_bytes,
                    label=f"trial {instance_id}",
                )
                _assert_public_payload(raw_trial, label=f"trial {instance_id}")
                _validate_raw_trial_binding(
                    raw_trial,
                    protocol=protocol,
                    protocol_sha256=protocol_digest,
                    instance_id=instance_id,
                    smoke_seal_sha256=smoke_seal_digest,
                    public_path_root=public_path_root,
                )
                raw_status, raw_correct, raw_error = _official_trial_outcome(raw_trial)
                raw_duration = _official_trial_duration(raw_trial)
                raw_retries = _official_infrastructure_retries(raw_trial)
                exception_type = _official_exception_type(raw_trial)
            else:
                projection = _load_model_bytes(
                    raw_bytes,
                    PublicHarborErrorProjection,
                    label=f"trial {instance_id}",
                )
                if raw_bytes != _canonical_model_bytes(projection):
                    raise ValueError(
                        f"redacted ERROR projection is not canonical for {instance_id}"
                    )
                _validate_error_projection_binding(
                    projection,
                    protocol=protocol,
                    instance_id=instance_id,
                    source_sha256=trial_digest,
                )
                raw_status = projection.raw_status
                raw_correct = projection.raw_correct
                raw_error = projection.raw_protocol_error
                raw_duration = projection.duration_s
                raw_retries = projection.infrastructure_retries
                exception_type = projection.exception_type
            has_public_trial = True
            observed_trial = PublicHarborTrial(
                instance_id=instance_id,
                path=trial_index.path,
                source_sha256=trial_digest,
                payload_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                payload_kind=trial_index.payload_kind,
                raw_status=raw_status,
                raw_correct=raw_correct,
                raw_protocol_error=raw_error,
                exception_type=exception_type,
                duration_s=raw_duration,
                infrastructure_retries=raw_retries,
            )
            if observed_trial != trial_index:
                raise ValueError(f"public trial summary changed for {instance_id}")

        manifest_status = scored_manifest.official_status[instance_id]
        manifest_error = scored_manifest.protocol_errors[instance_id]
        if has_public_trial:
            if (
                raw_status != manifest_status
                or raw_correct
                is not {
                    "PASS": True,
                    "FAIL": False,
                    "ERROR": None,
                }[manifest_status]
            ):
                raise ValueError(
                    f"official raw result disagrees with {instance_id} status"
                )
            if raw_error != manifest_error:
                if raw_status == manifest_status == "ERROR":
                    raise ValueError(
                        f"official raw ERROR explanation changed for {instance_id}"
                    )
                raise ValueError(
                    f"official raw result disagrees with {instance_id} status"
                )
        elif manifest_status != "ERROR":
            raise ValueError(
                f"official raw result is missing for {instance_id} status"
            )

        derived_records.append(
            TerminalBenchTaskRecord(
                instance_id=instance_id,
                protocol_sha256=protocol_digest,
                execution_manifest_sha256=scored_digest,
                command_envelope_sha256=(
                    scored_manifest.command_envelope_sha256[instance_id]
                ),
                official_result_sha256=trial_digest,
                official_status=manifest_status,
                independent_correct={
                    "PASS": True,
                    "FAIL": False,
                    "ERROR": None,
                }[manifest_status],
                duration_s=raw_duration,
                protocol_error=manifest_error,
                infrastructure_retries=raw_retries,
            )
        )
    derived_batch = TerminalBenchRecordBatch(
        protocol_sha256=protocol_digest,
        execution_manifest_sha256=scored_digest,
        records=tuple(derived_records),
    )

    records_bytes = _read_regular_file(
        root / _RECORDS_FILE,
        maximum_bytes=_MAX_RECORDS_BYTES,
        label="records",
    )
    records = _load_model_bytes(
        records_bytes,
        TerminalBenchRecordBatch,
        label="records",
    )
    if (
        records_bytes != _canonical_model_bytes(records)
        or hashlib.sha256(records_bytes).hexdigest() != index.records_sha256
        or records != derived_batch
    ):
        raise ValueError("public records do not match the official raw results")

    derived_summary = _summarize_offline(derived_batch)
    summary_bytes = _read_regular_file(
        root / _SUMMARY_FILE,
        maximum_bytes=_MAX_SUMMARY_BYTES,
        label="summary",
    )
    summary = _load_model_bytes(
        summary_bytes,
        TerminalBenchSummary,
        label="summary",
    )
    if (
        summary_bytes != _canonical_model_bytes(summary)
        or hashlib.sha256(summary_bytes).hexdigest() != index.summary_sha256
        or summary != derived_summary
    ):
        raise ValueError("public summary does not match the recomputed 20-task result")

    markdown_bytes = _read_regular_file(
        root / _SUMMARY_MARKDOWN_FILE,
        maximum_bytes=_MAX_SUMMARY_BYTES,
        label="summary markdown",
    )
    expected_markdown = _summary_markdown_bytes(
        derived_summary,
        schema_version=index.schema_version,
    )
    if (
        hashlib.sha256(markdown_bytes).hexdigest()
        != index.summary_markdown_sha256
        or markdown_bytes != expected_markdown
    ):
        raise ValueError("public summary markdown does not match the JSON summary")

    _assert_public_package_payloads(
        {
            relative: _read_regular_file(
                root / relative,
                maximum_bytes=_maximum_for_relative_path(relative),
                label=relative,
            )
            for relative in expected_files
        }
    )
    return TerminalBenchPublicEvidenceValidation(
        evidence_tree_sha256=_evidence_tree_sha256(root, expected_files),
        evaluation_id=protocol.evaluation_id,
        protocol_sha256=protocol_digest,
        scored_manifest_sha256=scored_digest,
        records_sha256=hashlib.sha256(records_bytes).hexdigest(),
        evaluated_commit_sha=(
            source_attestation.commit_sha
            if source_attestation is not None
            else None
        ),
        evaluated_tree_sha=(
            source_attestation.tree_sha
            if source_attestation is not None
            else None
        ),
        evaluated_wheel_filename=(
            source_attestation.wheel_filename
            if source_attestation is not None
            else None
        ),
        evaluated_wheel_size_bytes=(
            source_attestation.wheel_size_bytes
            if source_attestation is not None
            else None
        ),
        evaluated_wheel_sha256=(
            source_attestation.wheel_sha256
            if source_attestation is not None
            else None
        ),
        model=protocol.model,
        reasoning_effort=protocol.reasoning_effort,
        harbor_version=protocol.harbor_version,
        passed=derived_summary.passed,
        failed=derived_summary.failed,
        errors=derived_summary.errors,
    )


def _validate_manifest_binding(
    manifest: HarborExecutionManifest,
    *,
    protocol: TerminalBenchProtocol,
    protocol_sha256: str,
    run_kind: Literal["smoke", "scored"],
) -> None:
    expected = (
        protocol.subset.smoke_instance_ids
        if run_kind == "smoke"
        else protocol.subset.scored_instance_ids
    )
    if (
        manifest.run_kind != run_kind
        or manifest.dataset_version != protocol.dataset_version
        or manifest.protocol_sha256 != protocol_sha256
        or manifest.expected_instance_ids != expected
        or manifest.observed_instance_ids != expected
        or manifest.task_content_digests
        != {item: protocol.task_content_digests[item] for item in expected}
        or manifest.task_checksums
        != {item: protocol.task_checksums[item] for item in expected}
        or manifest.task_image_digests
        != {item: protocol.task_image_digests[item] for item in expected}
    ):
        raise ValueError(f"public {run_kind} manifest changed its protocol binding")
    output_root = Path(protocol.output_root)
    if any(
        not Path(job_dir).is_absolute()
        or Path(job_dir) != Path(job_dir).resolve()
        or Path(job_dir).parent != output_root
        for job_dir in manifest.job_dirs
    ):
        raise ValueError(f"public {run_kind} manifest contains an unbound job path")
    if run_kind == "smoke":
        if any(status == "ERROR" for status in manifest.official_status.values()):
            raise ValueError("public smoke evidence contains an unsealed ERROR")
        required_maps = (
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
        if any(
            value is None
            for evidence in required_maps
            for value in evidence.values()
        ):
            raise ValueError("public smoke evidence is incomplete")


def _validate_smoke_seal_binding(
    seal: SmokeSeal,
    *,
    protocol: TerminalBenchProtocol,
    protocol_sha256: str,
    smoke_manifest: HarborExecutionManifest,
    smoke_manifest_sha256: str,
) -> None:
    expected_records = {
        instance_id: smoke_manifest.terminal_record_sha256[instance_id]
        for instance_id in protocol.subset.smoke_instance_ids
    }
    if (
        seal.evaluation_id != protocol.evaluation_id
        or seal.protocol_sha256 != protocol_sha256
        or seal.manifest_sha256 != smoke_manifest_sha256
        or seal.smoke_instance_ids != protocol.subset.smoke_instance_ids
        or seal.terminal_record_sha256 != expected_records
        or any(value is None for value in expected_records.values())
    ):
        raise ValueError("public smoke seal changed its manifest binding")


def _validate_raw_trial_binding(
    trial: Mapping[str, Any],
    *,
    protocol: TerminalBenchProtocol,
    protocol_sha256: str,
    instance_id: str,
    smoke_seal_sha256: str,
    public_path_root: Path,
) -> None:
    if (
        trial.get("task_name") != instance_id
        or trial.get("task_checksum") != protocol.task_checksums[instance_id]
    ):
        raise ValueError("official raw trial changed its registered task binding")
    status, _correct, _error = _official_trial_outcome(trial)
    config = trial.get("config")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    environment = config.get("environment") if isinstance(config, Mapping) else None
    if not isinstance(agent, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("official raw trial omitted its resolved configuration")
    kwargs = agent.get("kwargs")
    path_kwargs = (
        {
            key: kwargs.get(key)
            for key in ("wheel_path", "codex_binary_path", "protocol_path")
        }
        if isinstance(kwargs, Mapping)
        else {}
    )
    if (
        agent.get("name") != AGENT_IMPORT_PATH
        or agent.get("model_name") != protocol.model
        or not isinstance(kwargs, Mapping)
        or set(kwargs)
        != {
            "wheel_path",
            "codex_binary_path",
            "protocol_path",
            "reasoning_effort",
            "instance_id",
            "run_kind",
            "attempt_id",
        }
        or kwargs.get("reasoning_effort") != protocol.reasoning_effort
        or kwargs.get("instance_id") != instance_id
        or kwargs.get("run_kind") != "scored"
        or kwargs.get("attempt_id")
        != terminal_attempt_id(protocol.evaluation_id, "scored", instance_id)
        or any(
            not isinstance(path, str)
            or not _path_is_within(path, public_path_root)
            for path in path_kwargs.values()
        )
        or environment.get("type") != "docker"
        or environment.get("import_path") not in (None, "")
        or environment.get("kwargs") not in (None, {})
    ):
        raise ValueError("official raw trial changed its model or agent binding")

    agent_info = trial.get("agent_info")
    if isinstance(agent_info, Mapping):
        model_info = agent_info.get("model_info")
        expected_provider: str | None = None
        expected_model = protocol.model
        if "/" in protocol.model:
            expected_provider, expected_model = protocol.model.split("/", 1)
        if (
            agent_info.get("name") != "lha"
            or not isinstance(model_info, Mapping)
            or model_info.get("name") != expected_model
            or model_info.get("provider") != expected_provider
        ):
            raise ValueError("official raw trial changed its agent identity")
    elif status != "ERROR":
        raise ValueError("completed official raw trial omitted its agent identity")

    agent_result = trial.get("agent_result")
    metadata = (
        agent_result.get("metadata")
        if isinstance(agent_result, Mapping)
        else None
    )
    if isinstance(metadata, Mapping):
        expected_metadata = {
            "dataset": DATASET,
            "evaluation_id": protocol.evaluation_id,
            "attempt_id": terminal_attempt_id(
                protocol.evaluation_id,
                "scored",
                instance_id,
            ),
            "dataset_version": protocol.dataset_version,
            "agent_import_path": AGENT_IMPORT_PATH,
            "instance_id": instance_id,
            "run_kind": "scored",
            "model": protocol.model,
            "reasoning_effort": protocol.reasoning_effort,
            "harbor_version": protocol.harbor_version,
            "codex_cli_version": protocol.codex_cli_version,
            "codex_binary_sha256": protocol.codex_binary_sha256,
            "wheel_sha256": protocol.wheel_sha256,
            "protocol_sha256": protocol_sha256,
            "task_content_digest": protocol.task_content_digests[instance_id],
            "task_image_digest": protocol.task_image_digests[instance_id],
            "broker_image_id": protocol.broker_image_id,
            "smoke_seal_sha256": smoke_seal_sha256,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("official raw trial metadata changed its protocol binding")
    elif status != "ERROR":
        raise ValueError("completed official raw trial omitted its agent metadata")


def _official_exception_type(trial: Mapping[str, Any]) -> str | None:
    exception = trial.get("exception_info")
    if exception is None:
        return None
    if not isinstance(exception, Mapping):
        raise ValueError("official Harbor exception_info must be an object")
    value = exception.get("exception_type")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("official Harbor exception_info omitted exception_type")
    normalized = value.strip()
    if _EXCEPTION_TYPE_RE.fullmatch(normalized) is None:
        raise ValueError("official Harbor exception_type is unsafe for public evidence")
    _assert_public_string(normalized, label="official Harbor exception_type")
    return normalized


def _validate_error_projection_binding(
    projection: PublicHarborErrorProjection,
    *,
    protocol: TerminalBenchProtocol,
    instance_id: str,
    source_sha256: str,
) -> None:
    if (
        projection.source_sha256 != source_sha256
        or projection.task_name != instance_id
        or projection.task_checksum != protocol.task_checksums[instance_id]
    ):
        raise ValueError("redacted ERROR projection changed its official source binding")


def _summarize_offline(batch: TerminalBenchRecordBatch) -> TerminalBenchSummary:
    values = list(batch.records)
    if len(values) != 20:
        raise ValueError("offline Terminal-Bench summary requires exactly 20 rows")
    passed = sum(row.official_status == "PASS" for row in values)
    failed = sum(row.official_status == "FAIL" for row in values)
    errors = sum(row.official_status == "ERROR" for row in values)
    durations = sorted(
        float(row.duration_s)
        for row in values
        if row.duration_s is not None
    )
    return TerminalBenchSummary(
        passed=passed,
        failed=failed,
        errors=errors,
        success_rate=passed / 20,
        p50_duration_s=_percentile(durations, 0.50),
        p95_duration_s=_percentile(durations, 0.95),
        protocol_errors=sum(bool(row.protocol_error) for row in values),
        missing_instance_ids=(),
    )


def _normalized_public_root(value: str | Path) -> Path:
    supplied = Path(value)
    normalized = supplied.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    home_root = Path.home().resolve()
    if (
        not supplied.is_absolute()
        or supplied != normalized
        or normalized == Path(normalized.anchor)
        or normalized == repository_root
        or normalized.is_relative_to(repository_root)
        or normalized == home_root
        or normalized.is_relative_to(home_root)
        or (
            len(normalized.parts) >= 3
            and normalized.parts[1].casefold() in {"home", "users"}
        )
    ):
        raise ValueError(
            "public path root must be absolute, normalized, and outside user homes"
        )
    return normalized


def _path_is_within(value: str | Path, root: Path) -> bool:
    supplied = Path(value)
    normalized = supplied.resolve()
    return (
        supplied.is_absolute()
        and supplied == normalized
        and normalized != root
        and normalized.is_relative_to(root)
    )


def _assert_private_roots_absent(
    payloads: Mapping[str, bytes],
    private_roots: Sequence[Path],
) -> None:
    rendered = b"\n".join(payloads.values()).decode("utf-8")
    for root in private_roots:
        root_text = str(root)
        if root_text and root_text in rendered:
            raise ValueError("public evidence contains a private path root")


def _load_exact_model(
    path: str | Path,
    supplied: BaseModel,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    payload = _read_regular_file(
        Path(path).resolve(),
        maximum_bytes=maximum_bytes,
        label=label,
    )
    recorded = _load_model_bytes(payload, type(supplied), label=label)
    if recorded != supplied:
        raise ValueError(f"{label} path does not contain the supplied model")
    if payload != _canonical_model_bytes(recorded):
        raise ValueError(f"{label} is not in canonical exported form")
    return payload


def _load_model_bytes(
    payload: bytes,
    model: type[_ModelT],
    *,
    label: str,
) -> _ModelT:
    raw = _load_json_object(payload, label=label)
    try:
        return model.model_validate(raw)
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc


def _load_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{label} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_live_trial_bytes(job_dir: Path) -> tuple[Path, bytes]:
    paths = sorted(
        path
        for path in job_dir.glob("*/result.json")
        if path.parent != job_dir
    )
    if len(paths) != 1:
        raise ValueError("each Harbor job must contain exactly one trial result")
    return (
        paths[0],
        _read_regular_file(
            paths[0],
            maximum_bytes=_MAX_TRIAL_BYTES,
            label="official Harbor trial result",
        ),
    )


def _read_indexed_trial(root: Path, trial: PublicHarborTrial) -> bytes:
    relative = PurePosixPath(trial.path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or _TRIAL_PATH_RE.fullmatch(trial.path) is None
    ):
        raise ValueError("public trial path is unsafe")
    payload = _read_regular_file(
        root.joinpath(*relative.parts),
        maximum_bytes=_MAX_TRIAL_BYTES,
        label=f"trial {trial.instance_id}",
    )
    if hashlib.sha256(payload).hexdigest() != trial.payload_sha256:
        raise ValueError(f"public trial bytes changed for {trial.instance_id}")
    return payload


def _read_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} is not a bounded standalone regular file")
        payload = bytearray()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"{label} changed while it was being read")
            payload.extend(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew while it was being read")
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"{label} changed while it was being read")
        return bytes(payload)
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    finally:
        os.close(descriptor)


def _list_package_files(root: Path) -> set[str]:
    files: set[str] = set()
    directories: set[str] = set()
    total_size = 0
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in directory_names:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError("public evidence may not contain symlink directories")
            directories.add(path.relative_to(root).as_posix())
        for name in file_names:
            path = directory_path / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("public evidence may contain only standalone files")
            total_size += info.st_size
            if total_size > _MAX_PACKAGE_BYTES:
                raise ValueError("public evidence package exceeds the size limit")
            files.add(path.relative_to(root).as_posix())
    allowed_directories = {"trials"} if any(
        relative.startswith("trials/") for relative in files
    ) else set()
    if directories != allowed_directories:
        raise ValueError("public evidence directory set changed")
    return files


def _assert_public_package_payloads(payloads: Mapping[str, bytes]) -> None:
    if sum(len(payload) for payload in payloads.values()) > _MAX_PACKAGE_BYTES:
        raise ValueError("public evidence package exceeds the size limit")
    for relative_path, payload in payloads.items():
        if relative_path == _SUMMARY_MARKDOWN_FILE:
            text = payload.decode("utf-8")
            _assert_public_string(text, label=relative_path)
            continue
        raw = _load_json_object(payload, label=relative_path)
        _assert_public_payload(raw, label=relative_path)


def _assert_public_payload(value: Any, *, label: str, key: str | None = None) -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            normalized = child_key.casefold()
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"{label} contains forbidden credential field {child_key!r}")
            if normalized in _FULL_LOG_KEYS and child not in (None, "", [], {}):
                raise ValueError(f"{label} contains a full private log field {child_key!r}")
            _assert_public_payload(child, label=label, key=normalized)
        return
    if isinstance(value, list):
        for child in value:
            _assert_public_payload(child, label=label, key=key)
        return
    if isinstance(value, str):
        _assert_public_string(value, label=label)
        if key in _FULL_LOG_KEYS and value:
            raise ValueError(f"{label} contains a private log payload")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")


def _assert_public_string(value: str, *, label: str) -> None:
    for pattern in _FORBIDDEN_TEXT:
        if pattern.search(value):
            raise ValueError(f"{label} contains credential-like or private data")


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return (model.model_dump_json(indent=2) + "\n").encode()


def _summary_markdown_bytes(
    summary: TerminalBenchSummary,
    *,
    schema_version: Literal[3, 4],
) -> bytes:
    if schema_version == 3:
        rendered = "\n".join(
            [
                "# Terminal-Bench 2.1 固定 20 题子集",
                "",
                f"- 通过：{summary.passed}/20（{summary.success_rate:.1%}）",
                f"- 失败：{summary.failed}/20",
                f"- ERROR：{summary.errors}/20（保留在分母中）",
                f"- P50 / P95 耗时（秒）："
                f"{summary.p50_duration_s} / {summary.p95_duration_s}",
                f"- 协议错误：{summary.protocol_errors}",
                "- 错误交付 / 拦截 / 错误拒绝：未测"
                "（当前 Harbor agent 未经过 LHA gate）",
                "- 修复成功率：不适用"
                "（当前协议只有一次 Codex 执行，没有 LHA repair 循环）",
                "",
                "该结果仅代表预注册的固定 20 题子集，不是完整排行榜成绩。",
            ]
        )
    else:
        rendered = summary.to_markdown()
    return (rendered + "\n").encode()


def _open_upgrade_lock(path: Path) -> int:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"another source-attestation upgrade is active: {path}"
        ) from exc
    try:
        os.fsync(descriptor)
        _fsync_directory(path.parent)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return descriptor


def _write_new_file(path: Path, payload: bytes) -> None:
    _durable_mkdir_chain(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("public evidence write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


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
    """Create missing parents and persist each new name in its parent."""
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


def _maximum_for_relative_path(relative_path: str) -> int:
    if relative_path == _INDEX_FILE:
        return _MAX_INDEX_BYTES
    if relative_path == _PROTOCOL_FILE:
        return _MAX_PROTOCOL_BYTES
    if relative_path in {_SMOKE_MANIFEST_FILE, _SCORED_MANIFEST_FILE}:
        return _MAX_MANIFEST_BYTES
    if relative_path == _RECORDS_FILE:
        return _MAX_RECORDS_BYTES
    if relative_path in {
        _SMOKE_SEAL_FILE,
        _SUMMARY_FILE,
        _SUMMARY_MARKDOWN_FILE,
        _SOURCE_ATTESTATION_FILE,
    }:
        return _MAX_SUMMARY_BYTES
    if _TRIAL_PATH_RE.fullmatch(relative_path):
        return _MAX_TRIAL_BYTES
    raise ValueError(f"unexpected public evidence path: {relative_path}")


def _evidence_tree_sha256(root: Path, relative_paths: set[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths):
        payload = _read_regular_file(
            root / relative_path,
            maximum_bytes=_maximum_for_relative_path(relative_path),
            label=relative_path,
        )
        payload_digest = hashlib.sha256(payload).hexdigest()
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(payload_digest.encode())
        digest.update(b"\n")
    return digest.hexdigest()
