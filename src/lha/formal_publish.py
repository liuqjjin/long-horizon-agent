"""Crash-recoverable publication of validated formal-ablation evidence.

The benchmark runner writes evidence below ``runs/``.  Publishing that evidence
is a separate transaction because a process can stop after replacing one file
but before appending the final ``COMPLETED`` registry event.

Before creating transaction state, this module repeats the complete formal
output validation and freezes the exact bytes that passed it.  Recovery does
not rerun that repository-cleanliness-sensitive validator; it accepts only the
semantically validated byte mapping bound into the durable journal.  The
transaction then:

* freezes the exact evidence whitelist and a seed-0 horizon rendering;
* records the final ``CompletedAttempt`` and exact registry before/after bytes
  in a durable Git-directory journal before touching ``benchmarks/``;
* stages payloads and backups on the same filesystem as ``benchmarks/``;
* installs every file with digest checks and atomic replacement; and
* keeps the backup until the caller proves whether its registry CAS happened.

An uncertain registry append is never answered by rolling evidence back.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .ablation import (
    _DIFF_IGNORE,
    _FORMAL_CORPUS_MANIFEST_PATH,
    _FORMAL_OUTPUT_LOCK_NAME,
    _FORMAL_REPETITIONS,
    _FORMAL_RUN_HEADER_NAME,
    _FORMAL_TASK_COUNT,
    _git_control_env,
    _input_snapshot_digest,
    _load_formal_corpus_manifest,
    _read_bounded_bytes,
    _repo_digest,
    _source_file_digests,
    _source_tree_digest,
    _trusted_control_executable,
    _validate_formal_head_checkout,
)
from .ablation_attempts import (
    FORMAL_ABLATION_ATTEMPTS_PATH,
    CompletedAttempt,
    FormalAblationAttemptRegistry,
    RegisteredAttempt,
    _formal_git_directory,
    formal_ablation_attempt_registry_bytes,
    parse_formal_ablation_attempt_registry,
)
from .clock import now
from .durable_io import (
    anchored_read_bytes,
    anchored_replace_bytes_if_current,
    anchored_unlink_file_if_bytes,
    anchored_write_once_bytes,
    durable_mkdir_chain,
)

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_TASK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_JOURNAL_SCHEMA = 1
_SNAPSHOT_SCHEMA = 1
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_JOURNAL_BYTES = 16 * 1024 * 1024
_JOURNAL_DIRECTORY = "lha-formal-publications"
_WORKSPACE_DIRECTORY = ".formal-publish"

_RAW_FILES = (
    "ablation_report.json",
    "ablation_report.md",
    _FORMAL_RUN_HEADER_NAME,
)
_EVIDENCE_DIRECTORIES = (
    "input_snapshots",
    "artifacts",
    "scorer_evidence",
    "llm_call_receipts",
    "results",
)
_EMPTY_RUNTIME_DIRECTORIES = (
    "active-operations",
    "active-container-ids",
)
_HORIZON_FILES = (
    "horizon_report.json",
    "horizon_report.md",
    "horizon_curve.svg",
)
_OUTPUT_ENTRIES = frozenset(
    (
        *_RAW_FILES,
        *_EVIDENCE_DIRECTORIES,
        *_EMPTY_RUNTIME_DIRECTORIES,
        _FORMAL_OUTPUT_LOCK_NAME,
    )
)
_PUBLICATION_PREFIXES = tuple(f"benchmarks/{name}/" for name in _EVIDENCE_DIRECTORIES)
_PUBLICATION_FILES = frozenset(f"benchmarks/{name}" for name in (*_RAW_FILES, *_HORIZON_FILES))

FaultInjector = Callable[[str], None]
TransactionState = Literal["PREPARED", "INSTALLING", "INSTALLED"]
InspectionState = Literal["CLEAN", "RECOVERY_REQUIRED", "QUARANTINED"]
DirectoryIdentity = tuple[int, int, int, int]


class FormalPublishError(RuntimeError):
    """Formal evidence could not be installed or recovered safely."""


class FormalPublishUncertainError(FormalPublishError):
    """The registry append cannot be classified as definitely absent or present."""


class _FileDigest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0, strict=True)


class _PublicationJournal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = _JOURNAL_SCHEMA
    semantically_validated: Literal[True]
    state: TransactionState
    attempt_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: str
    registration: dict[str, Any]
    completion: dict[str, Any]
    registry_before_text: str
    registry_before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_after_text: str
    registry_after_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    desired: dict[str, _FileDigest]
    prior: dict[str, _FileDigest | None]
    installed: tuple[str, ...] = ()

    @field_validator("output_path")
    @classmethod
    def _fixed_output_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != value
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("publication output_path is unsafe")
        return value


@dataclass(frozen=True)
class FormalPublishSummary:
    """Stable handoff from evidence installation to the registry CAS."""

    state: Literal["INSTALLED"]
    attempt_id: str
    completion: CompletedAttempt
    registry_before_bytes: bytes
    registry_after_bytes: bytes
    registry_before_sha256: str
    registry_after_sha256: str
    report_sha256: str
    report_fingerprint: str
    evidence_files: int
    evidence_bytes: int
    horizon_files: int
    horizon_bytes: int
    registry_already_appended: bool


@dataclass(frozen=True)
class FormalPublishFinalizeSummary:
    """Result of resolving the registry append and retiring the transaction."""

    attempt_id: str
    action: Literal["COMMITTED_AND_CLEANED", "ROLLED_BACK_AND_CLEANED"]


@dataclass(frozen=True)
class FormalPublishInspection:
    """Read-only status used by lifecycle commands before another state change."""

    status: InspectionState
    attempt_id: str | None = None
    transaction_state: TransactionState | None = None
    reason: str = ""


def _fault(injector: FaultInjector | None, phase: str) -> None:
    if injector is not None:
        injector(phase)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(payload: bytes) -> _FileDigest:
    return _FileDigest(sha256=_sha256(payload), size=len(payload))


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _tree_digest(files: Mapping[str, bytes]) -> str:
    payload = {
        name: {"sha256": _sha256(data), "size": len(data)} for name, data in sorted(files.items())
    }
    return _sha256(_canonical_json_bytes(payload))


def _digest_records(files: Mapping[str, _FileDigest]) -> str:
    payload = {name: value.model_dump(mode="json") for name, value in sorted(files.items())}
    return _sha256(_canonical_json_bytes(payload))


def _safe_repository(repository: str | Path) -> Path:
    candidate = Path(repository).resolve(strict=True)
    if not (candidate / "benchmarks").is_dir() or not (candidate / "src" / "lha").is_dir():
        raise FormalPublishError("formal publication requires the LHA repository root")
    try:
        _formal_git_directory(candidate)
    except OSError as error:
        raise FormalPublishError("formal publication Git metadata is unsafe") from error
    return candidate


def _safe_relative_repository_path(
    repository: Path,
    value: object,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str):
        raise FormalPublishError(f"{label} path is missing")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise FormalPublishError(f"{label} path is unsafe")
    candidate = repository.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise FormalPublishError(f"{label} path is unavailable") from error
    if not resolved.is_relative_to(repository):
        raise FormalPublishError(f"{label} path escapes the repository")
    current = repository
    for part in relative.parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise FormalPublishError(f"{label} path contains a symbolic link")
    return resolved


def _read_regular(path: Path, *, max_bytes: int = _MAX_FILE_BYTES) -> bytes:
    try:
        return _read_bounded_bytes(path, max_bytes=max_bytes)
    except (OSError, ValueError) as error:
        raise FormalPublishError(f"unsafe or unstable evidence file: {path}") from error


def _directory_entries(path: Path) -> dict[str, Path]:
    try:
        root = path.lstat()
    except OSError as error:
        raise FormalPublishError(f"evidence directory is unavailable: {path}") from error
    if stat.S_ISLNK(root.st_mode) or not stat.S_ISDIR(root.st_mode):
        raise FormalPublishError(f"evidence directory is unsafe: {path}")
    try:
        entries = list(path.iterdir())
    except OSError as error:
        raise FormalPublishError(f"evidence directory cannot be enumerated: {path}") from error
    return {entry.name: entry for entry in entries}


def _flat_store(
    directory: Path,
    *,
    expected: set[str],
    label: str,
) -> dict[str, bytes]:
    entries = _directory_entries(directory)
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise FormalPublishError(
            f"{label} store differs from its exact references "
            f"(missing={missing[:1]!r}, extra={extra[:1]!r})"
        )
    result: dict[str, bytes] = {}
    for name, path in sorted(entries.items()):
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise FormalPublishError(f"{label} store contains an unsafe entry: {name}")
        result[name] = _read_regular(path)
    return result


def _tree_shape(
    root: Path,
    *,
    reject_ignored_entries: bool,
) -> tuple[dict[str, bytes], set[str]]:
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise FormalPublishError(f"snapshot corpus root is unsafe: {root}")
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    for entry in sorted(root.rglob("*")):
        relative = entry.relative_to(root)
        ignored = any(part in _DIFF_IGNORE for part in relative.parts)
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise FormalPublishError(f"snapshot corpus contains a link: {relative}")
        if ignored:
            if reject_ignored_entries:
                raise FormalPublishError(
                    f"snapshot corpus contains an unpublishable entry: {relative}"
                )
            continue
        if stat.S_ISDIR(metadata.st_mode):
            directories.add(relative.as_posix())
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise FormalPublishError(f"snapshot corpus contains an unsafe entry: {relative}")
        files[relative.as_posix()] = _read_regular(entry)
    return files, directories


def _snapshot_files(
    *,
    repository: Path,
    output: Path,
    raw_report: Mapping[str, Any],
) -> dict[str, bytes]:
    tasks = raw_report.get("tasks")
    provenance = raw_report.get("provenance")
    if (
        not isinstance(tasks, list)
        or len(tasks) != _FORMAL_TASK_COUNT
        or not all(isinstance(task, str) and _TASK_NAME.fullmatch(task) for task in tasks)
        or len(set(tasks)) != _FORMAL_TASK_COUNT
        or not isinstance(provenance, dict)
    ):
        raise FormalPublishError("formal report does not describe exactly 17 safe task names")
    mappings: dict[str, dict[str, Any]] = {}
    for field in (
        "task_paths",
        "corpus_paths",
        "task_files_sha256",
        "corpus_sha256",
        "input_snapshot_sha256",
    ):
        value = provenance.get(field)
        if not isinstance(value, dict) or set(value) != set(tasks):
            raise FormalPublishError(f"formal report provenance {field!r} is incomplete")
        mappings[field] = value

    snapshot_root = output / "input_snapshots"
    entries = _directory_entries(snapshot_root)
    digest_values = [mappings["input_snapshot_sha256"][task] for task in tasks]
    if (
        not all(isinstance(value, str) and _HEX_64.fullmatch(value) for value in digest_values)
        or len(set(digest_values)) != _FORMAL_TASK_COUNT
    ):
        raise FormalPublishError("formal output does not contain 17 valid snapshot references")
    expected_digests = set(digest_values)
    if set(entries) != expected_digests:
        raise FormalPublishError("formal output does not contain exactly 17 referenced snapshots")

    published: dict[str, bytes] = {}
    for task in sorted(tasks):
        task_path = _safe_relative_repository_path(
            repository,
            mappings["task_paths"][task],
            label=f"{task} task",
        )
        corpus_path = _safe_relative_repository_path(
            repository,
            mappings["corpus_paths"][task],
            label=f"{task} corpus",
        )
        if not task_path.is_file() or not corpus_path.is_dir():
            raise FormalPublishError(f"{task} source has the wrong filesystem type")
        task_bytes = _read_regular(task_path)
        task_digest = _sha256(task_bytes)
        try:
            corpus_digest = _repo_digest(corpus_path)
        except (OSError, ValueError) as error:
            raise FormalPublishError(f"{task} corpus cannot be hashed safely") from error
        snapshot_digest = _input_snapshot_digest(task_digest, corpus_digest)
        if (
            mappings["task_files_sha256"][task] != task_digest
            or mappings["corpus_sha256"][task] != corpus_digest
            or mappings["input_snapshot_sha256"][task] != snapshot_digest
        ):
            raise FormalPublishError(f"{task} source changed after the validated report")

        snapshot = entries[snapshot_digest]
        snapshot_entries = _directory_entries(snapshot)
        if set(snapshot_entries) != {"snapshot.json", "task.yaml", "repo"}:
            raise FormalPublishError(f"{task} snapshot contains an extra or missing entry")
        metadata_bytes = _read_regular(snapshot_entries["snapshot.json"])
        try:
            metadata = json.loads(metadata_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FormalPublishError(f"{task} snapshot metadata is invalid") from error
        expected_metadata = {
            "schema_version": _SNAPSHOT_SCHEMA,
            "task": task,
            "task_sha256": task_digest,
            "corpus_sha256": corpus_digest,
            "snapshot_sha256": snapshot_digest,
        }
        if metadata != expected_metadata:
            raise FormalPublishError(f"{task} snapshot metadata is stale")
        snapshot_task = _read_regular(snapshot_entries["task.yaml"])
        if snapshot_task != task_bytes:
            raise FormalPublishError(f"{task} snapshot task bytes differ from the source")
        snapshot_repo = snapshot_entries["repo"]
        source_files, source_directories = _tree_shape(
            corpus_path,
            reject_ignored_entries=False,
        )
        frozen_files, frozen_directories = _tree_shape(
            snapshot_repo,
            reject_ignored_entries=True,
        )
        if (
            source_files != frozen_files
            or source_directories != frozen_directories
            or _repo_digest(snapshot_repo) != corpus_digest
        ):
            raise FormalPublishError(f"{task} snapshot corpus differs from the fixed source")

        prefix = f"input_snapshots/{snapshot_digest}"
        published[f"{prefix}/snapshot.json"] = metadata_bytes
        published[f"{prefix}/task.yaml"] = snapshot_task
        for relative, payload in frozen_files.items():
            published[f"{prefix}/repo/{relative}"] = payload
    return published


def _digest_references(raw_report: Mapping[str, Any], field: str) -> set[str]:
    records = raw_report.get("records")
    if not isinstance(records, list):
        raise FormalPublishError("formal report records are missing")
    values: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise FormalPublishError("formal report contains an invalid record")
        value = record.get(field)
        if value is None or value == "":
            continue
        if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
            raise FormalPublishError(f"formal report has an invalid {field}")
        values.add(value)
    return values


def _receipt_references(raw_report: Mapping[str, Any]) -> set[str]:
    calls = raw_report.get("llm_calls")
    if not isinstance(calls, list):
        raise FormalPublishError("formal report llm_calls are missing")
    values: set[str] = set()
    for call in calls:
        if not isinstance(call, dict):
            raise FormalPublishError("formal report contains an invalid LLM call reference")
        value = call.get("receipt_sha256")
        if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
            raise FormalPublishError("formal report has an invalid receipt reference")
        if value in values:
            raise FormalPublishError("formal report repeats an LLM receipt reference")
        values.add(value)
    return values


def _require_store_metadata(
    raw_report: Mapping[str, Any],
    *,
    field: str,
    path: str,
    count: int,
) -> None:
    store = raw_report.get(field)
    if not isinstance(store, dict) or store.get("path") != path or store.get("count") != count:
        raise FormalPublishError(f"formal report {field} does not match its references")


def _collect_output_evidence(
    *,
    repository: Path,
    output: Path,
    raw_report: Mapping[str, Any],
) -> dict[str, bytes]:
    try:
        output_metadata = output.lstat()
    except OSError as error:
        raise FormalPublishError("formal output directory is unavailable") from error
    if stat.S_ISLNK(output_metadata.st_mode) or not stat.S_ISDIR(output_metadata.st_mode):
        raise FormalPublishError("formal output directory is unsafe")
    top_entries = _directory_entries(output)
    if set(top_entries) != _OUTPUT_ENTRIES:
        extra = sorted(set(top_entries) - _OUTPUT_ENTRIES)
        missing = sorted(_OUTPUT_ENTRIES - set(top_entries))
        raise FormalPublishError(
            "formal output contains an extra or missing top-level entry "
            f"(missing={missing[:1]!r}, extra={extra[:1]!r})"
        )
    lock_metadata = top_entries[_FORMAL_OUTPUT_LOCK_NAME].lstat()
    if (
        stat.S_ISLNK(lock_metadata.st_mode)
        or not stat.S_ISREG(lock_metadata.st_mode)
        or lock_metadata.st_nlink != 1
    ):
        raise FormalPublishError("formal output lock is unsafe")
    for directory_name in _EMPTY_RUNTIME_DIRECTORIES:
        runtime_entries = _directory_entries(top_entries[directory_name])
        if runtime_entries:
            raise FormalPublishError(
                f"formal output runtime directory is not empty: {directory_name}"
            )

    collected = {name: _read_regular(top_entries[name]) for name in _RAW_FILES}
    try:
        decoded_report = json.loads(collected["ablation_report.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPublishError("formal output report JSON is invalid") from error
    if decoded_report != dict(raw_report):
        raise FormalPublishError("validated raw report differs from output bytes")

    provenance = raw_report.get("provenance")
    if not isinstance(provenance, dict):
        raise FormalPublishError("formal report provenance is missing")
    run_header_digest = provenance.get("formal_run_header_sha256")
    if run_header_digest != _sha256(collected[_FORMAL_RUN_HEADER_NAME]):
        raise FormalPublishError("formal run header differs from report provenance")

    artifacts = _digest_references(raw_report, "artifact_sha256")
    scorer_evidence = _digest_references(raw_report, "scorer_evidence_sha256")
    receipts = _receipt_references(raw_report)
    _require_store_metadata(
        raw_report,
        field="artifact_store",
        path="artifacts",
        count=len(artifacts),
    )
    _require_store_metadata(
        raw_report,
        field="scorer_evidence_store",
        path="scorer_evidence",
        count=len(scorer_evidence),
    )
    _require_store_metadata(
        raw_report,
        field="llm_call_receipt_store",
        path="llm_call_receipts",
        count=len(receipts),
    )
    flat_specs = (
        ("artifacts", artifacts, "artifact"),
        ("scorer_evidence", scorer_evidence, "scorer evidence"),
        ("llm_call_receipts", receipts, "LLM receipt"),
    )
    for directory_name, digests, label in flat_specs:
        files = _flat_store(
            output / directory_name,
            expected={f"{digest}.json" for digest in digests},
            label=label,
        )
        for name, payload in files.items():
            if _sha256(payload) != name.removesuffix(".json"):
                raise FormalPublishError(f"{label} bytes do not match their filename digest")
            collected[f"{directory_name}/{name}"] = payload

    tasks = raw_report.get("tasks")
    reps = raw_report.get("reps")
    if (
        not isinstance(tasks, list)
        or not all(isinstance(task, str) and _TASK_NAME.fullmatch(task) for task in tasks)
        or not isinstance(reps, int)
        or isinstance(reps, bool)
        or reps != _FORMAL_REPETITIONS
    ):
        raise FormalPublishError("formal report has an invalid task/repetition schedule")
    result_names = {
        name
        for task in tasks
        for rep in range(reps)
        for name in (f"{task}__r{rep}.started.json", f"{task}__r{rep}.json")
    }
    result_files = _flat_store(
        output / "results",
        expected=result_names,
        label="formal cell result",
    )
    collected.update({f"results/{name}": payload for name, payload in result_files.items()})
    collected.update(
        _snapshot_files(
            repository=repository,
            output=output,
            raw_report=raw_report,
        )
    )
    total = sum(len(payload) for payload in collected.values())
    if total <= 0 or total > _MAX_TOTAL_BYTES:
        raise FormalPublishError("formal evidence exceeds its publication size boundary")
    return collected


def _render_horizon(output: Path) -> dict[str, bytes]:
    from .horizon import _svg, build_report, load_cells

    source = "benchmarks/ablation_report.json"
    try:
        report = build_report(
            load_cells(output / "ablation_report.json", source_label=source),
            seed=0,
        )
    except (OSError, TypeError, ValueError) as error:
        raise FormalPublishError("formal horizon cannot be generated from the report") from error
    if report.source != source:
        raise FormalPublishError("formal horizon source label is not fixed to benchmarks")
    return {
        "horizon_report.json": report.to_json().encode("utf-8"),
        "horizon_report.md": report.to_markdown().encode("utf-8"),
        "horizon_curve.svg": _svg(report).encode("utf-8"),
    }


def _desired_publication(
    *,
    repository: Path,
    output: Path,
    raw_report: Mapping[str, Any],
) -> tuple[dict[str, bytes], int]:
    evidence = _collect_output_evidence(
        repository=repository,
        output=output,
        raw_report=raw_report,
    )
    horizon = _render_horizon(output)
    desired = {**evidence, **horizon}
    total = sum(len(payload) for payload in desired.values())
    if total > _MAX_TOTAL_BYTES:
        raise FormalPublishError("formal publication exceeds its total size boundary")
    return desired, len(evidence)


def _revalidate_formal_output(
    *,
    repository: Path,
    output: Path,
    raw_report: Mapping[str, Any],
) -> None:
    from .release_claims import validate_formal_ablation_output

    try:
        validated = validate_formal_ablation_output(
            output,
            repo_root=repository,
        )
    except (OSError, TypeError, ValueError) as error:
        raise FormalPublishError(
            "formal output failed its publication-time schema-4 validation"
        ) from error
    if validated != dict(raw_report):
        raise FormalPublishError(
            "publication-time validation returned different formal report data"
        )


def _git(
    repository: Path,
    arguments: list[str],
    *,
    label: str,
) -> bytes:
    try:
        executable = str(_trusted_control_executable("git")["path"])
        result = subprocess.run(
            [executable, *arguments],
            cwd=repository,
            env=_git_control_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        raise FormalPublishError(f"cannot {label}") from error
    if result.returncode != 0:
        raise FormalPublishError(f"cannot {label}")
    return result.stdout


def _git_dirty_paths(repository: Path) -> set[str]:
    payload = _git(
        repository,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        label="inspect publication worktree state",
    )
    paths: set[str] = set()
    entries = payload.split(b"\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2:3] != b" ":
            raise FormalPublishError("Git returned an unparseable worktree status")
        status_code = entry[:2]
        try:
            path = entry[3:].decode("utf-8")
        except UnicodeDecodeError as error:
            raise FormalPublishError("Git returned a non-UTF-8 worktree path") from error
        paths.add(path)
        if b"R" in status_code or b"C" in status_code:
            if index >= len(entries) or not entries[index]:
                raise FormalPublishError("Git returned an incomplete rename status")
            try:
                paths.add(entries[index].decode("utf-8"))
            except UnicodeDecodeError as error:
                raise FormalPublishError("Git returned a non-UTF-8 rename path") from error
            index += 1
    return paths


def _require_clean_worktree(repository: Path) -> None:
    dirty = sorted(_git_dirty_paths(repository))
    if dirty:
        raise FormalPublishError(f"formal publication requires a clean worktree: {dirty[0]}")


def _require_transaction_dirty_whitelist(
    repository: Path,
    journal: _PublicationJournal,
    *,
    allow_registry: bool,
) -> None:
    exact_targets = {f"benchmarks/{relative}" for relative in journal.desired}
    workspace_prefix = f"benchmarks/{_WORKSPACE_DIRECTORY}/{journal.attempt_id}/"
    unexpected = sorted(
        path
        for path in _git_dirty_paths(repository)
        if (
            path not in exact_targets
            and not path.startswith(workspace_prefix)
            and not (allow_registry and path == FORMAL_ABLATION_ATTEMPTS_PATH.as_posix())
        )
    )
    if unexpected:
        raise FormalPublishError(
            f"formal publication refuses a dirty path outside its whitelist: {unexpected[0]}"
        )


def _head(repository: Path) -> str:
    value = _git(repository, ["rev-parse", "--verify", "HEAD"], label="resolve HEAD")
    try:
        text = value.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise FormalPublishError("Git HEAD is invalid") from error
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise FormalPublishError("Git HEAD is invalid")
    return text


def _validate_trusted_formal_checkout(
    repository: Path,
    registration: RegisteredAttempt,
) -> None:
    """Rebind outcome-affecting inputs to blobs from the current trusted HEAD.

    Git status and index-based commands can hide paths marked assume-unchanged
    or skip-worktree.  The formal validator reads blobs by commit ID and then
    compares every source, control, task, and corpus byte with the worktree.
    """
    try:
        manifest, manifest_sha256 = _load_formal_corpus_manifest(
            repository / _FORMAL_CORPUS_MANIFEST_PATH,
            repository,
        )
        if manifest_sha256 != registration.manifest_sha256:
            raise FormalPublishError(
                "formal corpus manifest differs from its registration"
            )
        git_path = str(_trusted_control_executable("git")["path"])
        source_files = _validate_formal_head_checkout(
            repository,
            git_path=git_path,
            head=_head(repository),
            manifest=manifest,
            manifest_sha256=manifest_sha256,
        )
    except FormalPublishError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise FormalPublishError(
            "formal checkout differs from the trusted HEAD inputs"
        ) from error
    if _source_tree_digest(source_files) != registration.source_tree_sha256:
        raise FormalPublishError(
            "trusted HEAD source digest differs from the formal registration"
        )


def _tracked_publication_bytes(
    repository: Path,
    desired_names: set[str],
) -> dict[str, bytes]:
    raw_names = _git(
        repository,
        ["ls-tree", "-r", "-z", "--name-only", "HEAD", "--", "benchmarks"],
        label="inspect tracked publication baseline",
    )
    tracked: set[str] = set()
    for raw in raw_names.split(b"\0"):
        if not raw:
            continue
        try:
            name = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise FormalPublishError("tracked publication path is not UTF-8") from error
        relative = name.removeprefix("benchmarks/")
        if name in _PUBLICATION_FILES or any(
            name.startswith(prefix) for prefix in _PUBLICATION_PREFIXES
        ):
            if relative not in desired_names:
                raise FormalPublishError(
                    f"tracked publication target is outside the evidence whitelist: {name}"
                )
            tracked.add(relative)
    baseline: dict[str, bytes] = {}
    for relative in sorted(tracked):
        baseline[relative] = _git(
            repository,
            ["show", f"HEAD:benchmarks/{relative}"],
            label=f"read tracked baseline for {relative}",
        )
    return baseline


def _head_registry_bytes(repository: Path) -> bytes:
    return _git(
        repository,
        [
            "show",
            f"HEAD:{FORMAL_ABLATION_ATTEMPTS_PATH.as_posix()}",
        ],
        label="read the committed formal attempt registry",
    )


def _existing_publication_files(
    benchmarks: Path,
    *,
    expected: set[str] | None = None,
) -> set[str]:
    existing: set[str] = set()
    for name in (*_RAW_FILES, *_HORIZON_FILES):
        path = benchmarks / name
        if not path.exists() and not path.is_symlink():
            continue
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise FormalPublishError(f"publication target is unsafe: {name}")
        existing.add(name)
    for directory_name in _EVIDENCE_DIRECTORIES:
        directory = benchmarks / directory_name
        if not directory.exists() and not directory.is_symlink():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise FormalPublishError(f"publication target is unsafe: {directory_name}")
        for entry in sorted(directory.rglob("*")):
            metadata = entry.lstat()
            relative = entry.relative_to(benchmarks).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise FormalPublishError(f"publication target contains a link: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                if expected is not None and not any(
                    name.startswith(f"{relative}/") for name in expected
                ):
                    raise FormalPublishError(
                        f"publication target contains an extra directory: {relative}"
                    )
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise FormalPublishError(f"publication target contains an unsafe entry: {relative}")
            existing.add(relative)
    return existing


def _journal_directory(repository: Path) -> Path:
    return _formal_git_directory(repository) / _JOURNAL_DIRECTORY


def _journal_path(repository: Path, attempt_id: str) -> Path:
    return _journal_directory(repository) / f"{attempt_id}.json"


def _workspace(repository: Path, attempt_id: str) -> Path:
    return repository / "benchmarks" / _WORKSPACE_DIRECTORY / attempt_id


def _require_single_journal(repository: Path, attempt_id: str) -> None:
    directory = _journal_directory(repository)
    try:
        directory_metadata = directory.lstat()
    except OSError as error:
        raise FormalPublishError(
            "formal publication journal directory is unavailable"
        ) from error
    if (
        stat.S_ISLNK(directory_metadata.st_mode)
        or not stat.S_ISDIR(directory_metadata.st_mode)
        or directory_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(directory_metadata.st_mode) & 0o077
    ):
        raise FormalPublishError("formal publication journal directory is unsafe")
    try:
        entries = list(directory.iterdir())
    except OSError as error:
        raise FormalPublishError(
            "formal publication journal directory cannot be enumerated"
        ) from error
    expected = f"{attempt_id}.json"
    if len(entries) != 1 or entries[0].name != expected:
        raise FormalPublishError("formal publication requires exactly its one durable journal")
    metadata = entries[0].lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise FormalPublishError("formal publication journal entry is unsafe")


def _journal_bytes(journal: _PublicationJournal) -> bytes:
    return _canonical_json_bytes(journal.model_dump(mode="json"))


def _write_journal(
    repository: Path,
    journal: _PublicationJournal,
    *,
    expected_previous: bytes | None,
) -> None:
    directory = _journal_directory(repository)
    git_directory = _formal_git_directory(repository)
    durable_mkdir_chain(directory, anchor=git_directory, mode=0o700)
    _require_private_directory(
        directory,
        label="formal publication journal directory",
    )
    path = _journal_path(repository, journal.attempt_id)
    payload = _journal_bytes(journal)
    try:
        if expected_previous is None:
            anchored_write_once_bytes(
                path,
                payload,
                anchor=git_directory,
                mode=0o600,
            )
        else:
            _require_single_journal(repository, journal.attempt_id)
            anchored_replace_bytes_if_current(
                path,
                payload,
                anchor=git_directory,
                expected_current=(expected_previous,),
                mode=0o600,
            )
        _require_single_journal(repository, journal.attempt_id)
        written = anchored_read_bytes(path, anchor=git_directory)
    except (OSError, ValueError) as error:
        raise FormalPublishError("formal publication journal is not durable") from error
    if written != payload:
        raise FormalPublishError("formal publication journal changed after its durable write")


def _read_journal(repository: Path, attempt_id: str) -> _PublicationJournal:
    _require_single_journal(repository, attempt_id)
    path = _journal_path(repository, attempt_id)
    try:
        payload = anchored_read_bytes(path, anchor=_formal_git_directory(repository))
    except OSError as error:
        raise FormalPublishError("formal publication journal is unsafe") from error
    if payload is None or len(payload) > _MAX_JOURNAL_BYTES:
        raise FormalPublishError("formal publication journal is unavailable or too large")
    try:
        journal = _PublicationJournal.model_validate_json(payload)
    except ValidationError as error:
        raise FormalPublishError("formal publication journal is invalid") from error
    _validate_journal_bytes(journal)
    return journal


def _validate_journal_bytes(journal: _PublicationJournal) -> None:
    try:
        before = journal.registry_before_text.encode("utf-8")
        after = journal.registry_after_text.encode("utf-8")
        registration = RegisteredAttempt.model_validate(journal.registration)
        completion = CompletedAttempt.model_validate(journal.completion)
        before_registry = parse_formal_ablation_attempt_registry(before)
        after_registry = parse_formal_ablation_attempt_registry(after)
    except (UnicodeEncodeError, ValidationError, ValueError) as error:
        raise FormalPublishError("formal publication journal binding is invalid") from error
    names = (*journal.desired, *journal.prior, *journal.installed)
    for name in names:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or any(part in {"", ".", ".."} for part in path.parts)
            or (
                name not in {*_RAW_FILES, *_HORIZON_FILES}
                and (len(path.parts) < 2 or path.parts[0] not in _EVIDENCE_DIRECTORIES)
            )
        ):
            raise FormalPublishError(
                f"formal publication journal contains an unsafe target: {name}"
            )
    report = journal.desired.get("ablation_report.json")
    source_records = {
        name: value for name, value in journal.desired.items() if name not in _HORIZON_FILES
    }
    if (
        _sha256(before) != journal.registry_before_sha256
        or _sha256(after) != journal.registry_after_sha256
        or registration.attempt_id != journal.attempt_id
        or registration.output_path != journal.output_path
        or registration.source_tree_sha256 != journal.source_tree_sha256
        or completion.attempt_id != journal.attempt_id
        or completion.protocol_sha256 != registration.protocol_sha256
        or completion.registration_registry_sha256 != journal.registry_before_sha256
        or completion.report_sha256 != journal.report_sha256
        or completion.report_fingerprint != journal.report_fingerprint
        or report is None
        or report.sha256 != journal.report_sha256
        or _digest_records(source_records) != journal.source_evidence_sha256
        or before_registry.open_registration() != registration
        or tuple(after_registry.events) != (*before_registry.events, completion)
        or formal_ablation_attempt_registry_bytes(before_registry) != before
        or formal_ablation_attempt_registry_bytes(after_registry) != after
        or set(journal.prior) != set(journal.desired)
        or not set(journal.installed) <= set(journal.desired)
        or (journal.state == "PREPARED" and journal.installed)
        or (
            journal.state == "INSTALLED"
            and set(journal.installed) != set(journal.desired)
        )
    ):
        raise FormalPublishError("formal publication journal hashes or transition are invalid")


def _registry_on_disk(repository: Path) -> bytes:
    try:
        payload = anchored_read_bytes(
            repository / FORMAL_ABLATION_ATTEMPTS_PATH,
            anchor=repository,
        )
    except OSError as error:
        raise FormalPublishError("formal attempt registry cannot be read safely") from error
    if payload is None:
        raise FormalPublishError("formal attempt registry is missing")
    return payload


def _write_workspace_file(
    repository: Path,
    path: Path,
    payload: bytes,
) -> None:
    try:
        anchored_replace_bytes_if_current(
            path,
            payload,
            anchor=repository,
            expected_current=(payload,),
            expected_missing=True,
            mode=0o600,
        )
    except (OSError, ValueError) as error:
        raise FormalPublishError(f"cannot stage formal publication file {path.name}") from error


def _stage_path(repository: Path, attempt_id: str, relative: str) -> Path:
    return _workspace(repository, attempt_id) / "stage" / Path(relative)


def _backup_path(repository: Path, attempt_id: str, relative: str) -> Path:
    return _workspace(repository, attempt_id) / "backup" / Path(relative)


def _prepare_workspace(
    *,
    repository: Path,
    journal: _PublicationJournal,
    desired: Mapping[str, bytes],
    baseline: Mapping[str, bytes],
) -> None:
    workspace = _workspace(repository, journal.attempt_id)
    durable_mkdir_chain(workspace / "stage", anchor=repository, mode=0o700)
    durable_mkdir_chain(workspace / "backup", anchor=repository, mode=0o700)
    for relative, payload in sorted(desired.items()):
        expected = journal.desired[relative]
        if _digest(payload) != expected:
            raise FormalPublishError(f"staged source digest drifted for {relative}")
        _write_workspace_file(
            repository,
            _stage_path(repository, journal.attempt_id, relative),
            payload,
        )
    for relative, payload in sorted(baseline.items()):
        expected = journal.prior[relative]
        if expected is None or _digest(payload) != expected:
            raise FormalPublishError(f"backup source digest drifted for {relative}")
        _write_workspace_file(
            repository,
            _backup_path(repository, journal.attempt_id, relative),
            payload,
        )
    _verify_workspace(repository, journal)


def _verify_payload(path: Path, expected: _FileDigest) -> bytes:
    payload = _read_regular(path)
    if _digest(payload) != expected:
        raise FormalPublishError(f"publication payload digest is stale: {path}")
    return payload


def _verify_workspace(repository: Path, journal: _PublicationJournal) -> None:
    _verify_workspace_no_extras(repository, journal)
    for relative, expected in journal.desired.items():
        _verify_payload(
            _stage_path(repository, journal.attempt_id, relative),
            expected,
        )
    for relative, expected in journal.prior.items():
        path = _backup_path(repository, journal.attempt_id, relative)
        if expected is None:
            if path.exists() or path.is_symlink():
                raise FormalPublishError(f"unexpected backup exists for {relative}")
            continue
        _verify_payload(path, expected)


def _verify_backups(repository: Path, journal: _PublicationJournal) -> None:
    _verify_workspace_no_extras(repository, journal)
    for relative, expected in journal.prior.items():
        path = _backup_path(repository, journal.attempt_id, relative)
        if expected is None:
            if path.exists() or path.is_symlink():
                raise FormalPublishError(f"unexpected backup exists for {relative}")
            continue
        _verify_payload(path, expected)


def _workspace_expected_records(
    journal: _PublicationJournal,
) -> dict[str, _FileDigest]:
    records = {
        f"stage/{relative}": expected
        for relative, expected in journal.desired.items()
    }
    records.update(
        {
            f"backup/{relative}": expected
            for relative, expected in journal.prior.items()
            if expected is not None
        }
    )
    return records


def _workspace_allowed_directories(expected_files: set[str]) -> set[str]:
    allowed = {"stage", "backup"}
    for name in expected_files:
        path = PurePosixPath(name)
        for index in range(1, len(path.parts)):
            allowed.add(PurePosixPath(*path.parts[:index]).as_posix())
    return allowed


def _require_private_directory(path: Path, *, label: str) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise FormalPublishError(f"{label} is unsafe")


def _verify_workspace_no_extras(
    repository: Path,
    journal: _PublicationJournal,
) -> None:
    workspace = _workspace(repository, journal.attempt_id)
    workspace_root = workspace.parent
    if workspace_root.exists() or workspace_root.is_symlink():
        _require_private_directory(
            workspace_root,
            label="formal publication workspace root",
        )
        root_entries = _directory_entries(workspace_root)
        unexpected_roots = set(root_entries) - {journal.attempt_id}
        if unexpected_roots:
            raise FormalPublishError(
                "formal publication workspace root contains another entry: "
                f"{sorted(unexpected_roots)[0]}"
            )
    if not workspace.exists() and not workspace.is_symlink():
        return
    _require_private_directory(workspace, label="formal publication workspace")
    expected_records = _workspace_expected_records(journal)
    expected_files = set(expected_records)
    allowed_directories = _workspace_allowed_directories(expected_files)
    for entry in workspace.rglob("*"):
        item = entry.lstat()
        relative = entry.relative_to(workspace).as_posix()
        if stat.S_ISLNK(item.st_mode):
            raise FormalPublishError(f"formal publication workspace contains a link: {relative}")
        if stat.S_ISDIR(item.st_mode):
            if (
                relative not in allowed_directories
                or item.st_uid != os.geteuid()
                or stat.S_IMODE(item.st_mode) & 0o077
            ):
                raise FormalPublishError(
                    f"formal publication workspace contains an unsafe directory: {relative}"
                )
            continue
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_nlink != 1
            or item.st_uid != os.geteuid()
            or stat.S_IMODE(item.st_mode) & 0o077
            or relative not in expected_records
        ):
            raise FormalPublishError(
                f"formal publication workspace contains an extra or unsafe file: {relative}"
            )
        if _digest(_read_regular(entry)) != expected_records[relative]:
            raise FormalPublishError(
                f"formal publication workspace file changed: {relative}"
            )


def _workspace_directory_identities(
    repository: Path,
    journal: _PublicationJournal,
) -> dict[str, DirectoryIdentity]:
    _verify_workspace_no_extras(repository, journal)
    workspace = _workspace(repository, journal.attempt_id)
    workspace_root = workspace.parent
    candidates: dict[str, tuple[Path, DirectoryIdentity]] = {}
    for path in (workspace_root, workspace):
        if not path.exists() and not path.is_symlink():
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FormalPublishError(
                f"formal publication workspace directory is unsafe: {path}"
            )
        relative = path.relative_to(repository).as_posix()
        candidates[relative] = (path, _directory_identity(metadata))
    if workspace.exists() and not workspace.is_symlink():
        for entry in workspace.rglob("*"):
            metadata = entry.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                continue
            relative = entry.relative_to(repository).as_posix()
            candidates[relative] = (entry, _directory_identity(metadata))

    identities: dict[str, DirectoryIdentity] = {}
    for relative, (path, expected_identity) in sorted(candidates.items()):
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or _directory_identity(metadata) != expected_identity
        ):
            raise FormalPublishError(
                f"formal publication workspace directory changed: {relative}"
            )
        identities[relative] = expected_identity
    return identities


def _ensure_backups(repository: Path, journal: _PublicationJournal) -> None:
    durable_mkdir_chain(
        _workspace(repository, journal.attempt_id) / "backup",
        anchor=repository,
        mode=0o700,
    )
    for relative, expected in journal.prior.items():
        if expected is None:
            path = _backup_path(repository, journal.attempt_id, relative)
            if path.exists() or path.is_symlink():
                raise FormalPublishError(f"unexpected backup exists for {relative}")
            continue
        payload = _git(
            repository,
            ["show", f"HEAD:benchmarks/{relative}"],
            label=f"recover tracked baseline for {relative}",
        )
        if _digest(payload) != expected:
            raise FormalPublishError(f"tracked rollback baseline drifted for {relative}")
        _write_workspace_file(
            repository,
            _backup_path(repository, journal.attempt_id, relative),
            payload,
        )
    _verify_backups(repository, journal)


def _verify_repository_binding(
    repository: Path,
    registration: RegisteredAttempt,
    raw_report: Mapping[str, Any],
) -> None:
    if _source_tree_digest(_source_file_digests(repository / "src" / "lha")) != (
        registration.source_tree_sha256
    ):
        raise FormalPublishError("LHA source changed after formal registration")
    provenance = raw_report.get("provenance")
    if not isinstance(provenance, dict):
        raise FormalPublishError("formal report provenance is missing")
    registration_commit = provenance.get("formal_attempt_registration_commit")
    if registration_commit != _head(repository):
        raise FormalPublishError("formal registration commit is no longer HEAD")


def _binding_from_inputs(
    *,
    registration: RegisteredAttempt,
    registration_registry_bytes: bytes,
    raw_report: Mapping[str, Any],
    report_bytes: bytes,
) -> None:
    try:
        registry = parse_formal_ablation_attempt_registry(registration_registry_bytes)
    except ValueError as error:
        raise FormalPublishError("provided formal registry bytes are invalid") from error
    provenance = raw_report.get("provenance")
    fingerprint = raw_report.get("fingerprint")
    if (
        not isinstance(provenance, dict)
        or not isinstance(fingerprint, str)
        or _HEX_64.fullmatch(fingerprint) is None
        or provenance.get("formal_attempt_id") != registration.attempt_id
        or provenance.get("formal_attempt_protocol_sha256") != registration.protocol_sha256
        or provenance.get("formal_attempt_registry_sha256") != _sha256(registration_registry_bytes)
        or registry.open_registration() != registration
        or _sha256(report_bytes) == ""
    ):
        raise FormalPublishError("validated report differs from its registration")


def _new_journal(
    *,
    registration: RegisteredAttempt,
    registry_before: bytes,
    raw_report: Mapping[str, Any],
    desired: Mapping[str, bytes],
    baseline: Mapping[str, bytes],
    evidence: Mapping[str, bytes],
) -> _PublicationJournal:
    report_bytes = desired["ablation_report.json"]
    fingerprint = raw_report["fingerprint"]
    assert isinstance(fingerprint, str)
    completion = CompletedAttempt(
        attempt_id=registration.attempt_id,
        protocol_sha256=registration.protocol_sha256,
        registration_registry_sha256=_sha256(registry_before),
        recorded_at=now().isoformat(),
        report_sha256=_sha256(report_bytes),
        report_fingerprint=fingerprint,
    )
    try:
        before_registry = parse_formal_ablation_attempt_registry(registry_before)
        after_registry = FormalAblationAttemptRegistry(events=(*before_registry.events, completion))
    except (ValidationError, ValueError) as error:
        raise FormalPublishError("COMPLETED transition cannot be fixed for publication") from error
    registry_after = formal_ablation_attempt_registry_bytes(after_registry)
    return _PublicationJournal(
        semantically_validated=True,
        state="PREPARED",
        attempt_id=registration.attempt_id,
        output_path=registration.output_path,
        registration=registration.model_dump(mode="json"),
        completion=completion.model_dump(mode="json"),
        registry_before_text=registry_before.decode("utf-8"),
        registry_before_sha256=_sha256(registry_before),
        registry_after_text=registry_after.decode("utf-8"),
        registry_after_sha256=_sha256(registry_after),
        report_sha256=completion.report_sha256,
        report_fingerprint=completion.report_fingerprint,
        source_tree_sha256=registration.source_tree_sha256,
        source_evidence_sha256=_tree_digest(evidence),
        desired={name: _digest(payload) for name, payload in desired.items()},
        prior={name: (_digest(baseline[name]) if name in baseline else None) for name in desired},
    )


def _journal_with(
    journal: _PublicationJournal,
    *,
    state: TransactionState | None = None,
    installed: tuple[str, ...] | None = None,
) -> _PublicationJournal:
    values = journal.model_dump(mode="python")
    if state is not None:
        values["state"] = state
    if installed is not None:
        values["installed"] = installed
    return _PublicationJournal.model_validate(values)


def _install_files(
    *,
    repository: Path,
    journal: _PublicationJournal,
    fault_injector: FaultInjector | None,
) -> _PublicationJournal:
    if journal.state == "PREPARED":
        previous = journal
        journal = _journal_with(previous, state="INSTALLING")
        _write_journal(
            repository,
            journal,
            expected_previous=_journal_bytes(previous),
        )
        _fault(fault_injector, "after_installing")
    installed = list(journal.installed)
    for relative, expected in sorted(journal.desired.items()):
        _fault(fault_injector, f"before_install:{relative}")
        staged = _verify_payload(
            _stage_path(repository, journal.attempt_id, relative),
            expected,
        )
        target = repository / "benchmarks" / Path(relative)
        allowed = [staged]
        prior = journal.prior[relative]
        if prior is not None:
            allowed.append(
                _verify_payload(
                    _backup_path(repository, journal.attempt_id, relative),
                    prior,
                )
            )
        try:
            anchored_replace_bytes_if_current(
                target,
                staged,
                anchor=repository,
                expected_current=tuple(allowed),
                expected_missing=prior is None,
                mode=0o644,
            )
        except (OSError, ValueError) as error:
            raise FormalPublishError(
                f"cannot install formal publication file {relative}"
            ) from error
        _verify_payload(target, expected)
        if relative not in installed:
            installed.append(relative)
            previous = journal
            journal = _journal_with(previous, installed=tuple(installed))
            _write_journal(
                repository,
                journal,
                expected_previous=_journal_bytes(previous),
            )
        _fault(fault_injector, f"after_install:{relative}")
    _verify_installed_targets(repository, journal)
    _fault(fault_injector, "before_installed")
    previous = journal
    journal = _journal_with(
        previous,
        state="INSTALLED",
        installed=tuple(sorted(journal.desired)),
    )
    _write_journal(
        repository,
        journal,
        expected_previous=_journal_bytes(previous),
    )
    _fault(fault_injector, "after_installed")
    return journal


def _verify_installed_targets(repository: Path, journal: _PublicationJournal) -> None:
    existing = _existing_publication_files(
        repository / "benchmarks",
        expected=set(journal.desired),
    )
    desired = set(journal.desired)
    unexpected = existing - desired
    missing = desired - existing
    if unexpected or missing:
        raise FormalPublishError(
            "installed publication differs from its exact whitelist "
            f"(missing={sorted(missing)[:1]!r}, extra={sorted(unexpected)[:1]!r})"
        )
    for relative, expected in journal.desired.items():
        _verify_payload(repository / "benchmarks" / Path(relative), expected)


def _staged_payload_if_present(
    repository: Path,
    journal: _PublicationJournal,
    relative: str,
) -> bytes | None:
    path = _stage_path(repository, journal.attempt_id, relative)
    if not path.exists() and not path.is_symlink():
        return None
    return _verify_payload(path, journal.desired[relative])


def _safe_remove_file(
    repository: Path,
    path: Path,
    *,
    expected_current: tuple[bytes, ...],
) -> None:
    try:
        anchored_unlink_file_if_bytes(
            path,
            anchor=repository,
            expected_current=expected_current,
            missing_ok=True,
        )
    except FileNotFoundError:
        # ``missing_ok`` applies to the final name.  A parent that never
        # existed also proves that this target is absent; do not create the
        # directory merely to remove a file from it.
        if path.exists() or path.is_symlink():
            raise FormalPublishError(f"rollback target appeared during removal: {path}") from None
    except OSError as error:
        raise FormalPublishError(f"rollback target is unsafe: {path}") from error


def _publication_directories(names: set[str]) -> set[str]:
    directories: set[str] = set()
    for name in names:
        path = PurePosixPath(name)
        if not path.parts or path.parts[0] not in _EVIDENCE_DIRECTORIES:
            continue
        for index in range(1, len(path.parts)):
            directories.add(PurePosixPath(*path.parts[:index]).as_posix())
    return directories


def _directory_identity(metadata: os.stat_result) -> DirectoryIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
    )


def _publication_directory_identities(
    repository: Path,
    *,
    allowed: set[str],
) -> dict[str, DirectoryIdentity]:
    benchmarks = repository / "benchmarks"
    candidates: dict[str, tuple[Path, DirectoryIdentity]] = {}
    for directory_name in _EVIDENCE_DIRECTORIES:
        root = benchmarks / directory_name
        if not root.exists() and not root.is_symlink():
            continue
        try:
            metadata = root.lstat()
        except OSError as error:
            raise FormalPublishError(
                f"publication directory cannot be inspected: {directory_name}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FormalPublishError(
                f"publication directory is unsafe: {directory_name}"
            )
        candidates[directory_name] = (root, _directory_identity(metadata))
        try:
            entries = sorted(root.rglob("*"))
        except OSError as error:
            raise FormalPublishError(
                f"publication directory cannot be enumerated: {directory_name}"
            ) from error
        for entry in entries:
            metadata = entry.lstat()
            relative = entry.relative_to(benchmarks).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise FormalPublishError(
                    f"publication directory contains a link: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                candidates[relative] = (entry, _directory_identity(metadata))

    unexpected = set(candidates) - allowed
    if unexpected:
        raise FormalPublishError(
            "publication target contains an extra directory: "
            f"{sorted(unexpected)[0]}"
        )
    identities: dict[str, DirectoryIdentity] = {}
    for relative, (path, expected_identity) in sorted(candidates.items()):
        try:
            metadata = path.lstat()
        except OSError as error:
            raise FormalPublishError(
                f"publication directory changed during validation: {relative}"
            ) from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or _directory_identity(metadata) != expected_identity
        ):
            raise FormalPublishError(
                f"publication directory changed during validation: {relative}"
            )
        identities[relative] = expected_identity
    return identities


def _preflight_rollback_targets(
    repository: Path,
    journal: _PublicationJournal,
) -> dict[str, DirectoryIdentity]:
    existing = _existing_publication_files(
        repository / "benchmarks",
        expected=set(journal.desired),
    )
    unexpected = existing - set(journal.desired)
    if unexpected:
        raise FormalPublishError(
            f"rollback found a publication path outside the journal: {sorted(unexpected)[0]}"
        )
    for relative, desired in journal.desired.items():
        target = repository / "benchmarks" / Path(relative)
        prior = journal.prior[relative]
        if not target.exists() and not target.is_symlink():
            if prior is None:
                continue
            raise FormalPublishError(f"rollback cannot classify the missing target {relative}")
        metadata = target.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise FormalPublishError(f"rollback cannot classify the unsafe target {relative}")
        current = _digest(_read_regular(target))
        if current != desired and (prior is None or current != prior):
            raise FormalPublishError(f"rollback refuses user or third-party bytes at {relative}")
    return _publication_directory_identities(
        repository,
        allowed=_publication_directories(set(journal.desired)),
    )


def _targets_match_exact_prior(
    repository: Path,
    journal: _PublicationJournal,
) -> bool:
    expected_prior = {
        relative
        for relative, value in journal.prior.items()
        if value is not None
    }
    existing = _existing_publication_files(
        repository / "benchmarks",
        expected=set(journal.desired),
    )
    if existing != expected_prior:
        return False
    _publication_directory_identities(
        repository,
        allowed=_publication_directories(set(journal.desired)),
    )
    for relative in expected_prior:
        prior = journal.prior[relative]
        assert prior is not None
        if _digest(
            _read_regular(repository / "benchmarks" / Path(relative))
        ) != prior:
            return False
    return True


def _rollback(repository: Path, journal: _PublicationJournal) -> None:
    _verify_backups(repository, journal)
    # Classify every target before changing any of them.  A single unknown byte
    # sequence quarantines the whole transaction instead of partially
    # overwriting work that may have been created after the crash.
    directory_identities = _preflight_rollback_targets(repository, journal)
    for relative, prior in sorted(journal.prior.items()):
        target = repository / "benchmarks" / Path(relative)
        staged = _staged_payload_if_present(repository, journal, relative)
        if prior is None:
            _safe_remove_file(
                repository,
                target,
                expected_current=((staged,) if staged is not None else ()),
            )
            continue
        backup = _verify_payload(
            _backup_path(repository, journal.attempt_id, relative),
            prior,
        )
        allowed = (backup,) if staged is None else (backup, staged)
        try:
            anchored_replace_bytes_if_current(
                target,
                backup,
                anchor=repository,
                expected_current=allowed,
                mode=0o644,
            )
        except (OSError, ValueError) as error:
            raise FormalPublishError(f"cannot roll back publication file {relative}") from error
        _verify_payload(target, prior)
    final_directories = _publication_directory_identities(
        repository,
        allowed=_publication_directories(set(journal.desired)),
    )
    if final_directories != directory_identities:
        raise FormalPublishError(
            "publication directory identity changed during rollback"
        )
    expected_prior = {
        relative
        for relative, value in journal.prior.items()
        if value is not None
    }
    existing = _existing_publication_files(
        repository / "benchmarks",
        expected=set(journal.desired),
    )
    if existing != expected_prior:
        raise FormalPublishError("rollback did not restore the exact tracked file set")
    for relative in expected_prior:
        prior = journal.prior[relative]
        assert prior is not None
        _verify_payload(repository / "benchmarks" / Path(relative), prior)


def _remove_workspace_payload(
    repository: Path,
    path: Path,
    expected: _FileDigest,
) -> None:
    try:
        current = anchored_read_bytes(
            path,
            anchor=repository,
            missing_ok=True,
        )
    except OSError as error:
        raise FormalPublishError(
            f"formal publication workspace cannot be read safely: {path}"
        ) from error
    if current is None:
        return
    if _digest(current) != expected:
        raise FormalPublishError(
            f"formal publication workspace changed before cleanup: {path}"
        )
    try:
        anchored_unlink_file_if_bytes(
            path,
            anchor=repository,
            expected_current=(current,),
            missing_ok=True,
        )
    except OSError as error:
        raise FormalPublishError(
            f"formal publication workspace changed during cleanup: {path}"
        ) from error


def _remove_workspace(repository: Path, journal: _PublicationJournal) -> None:
    workspace = _workspace(repository, journal.attempt_id)
    directory_identities = _workspace_directory_identities(
        repository,
        journal,
    )
    for relative, expected in sorted(_workspace_expected_records(journal).items()):
        _remove_workspace_payload(
            repository,
            workspace / Path(relative),
            expected,
        )
    final_identities = _workspace_directory_identities(
        repository,
        journal,
    )
    if final_identities != directory_identities:
        raise FormalPublishError(
            "formal publication workspace directory identity changed during cleanup"
        )


def _remove_journal(repository: Path, journal: _PublicationJournal) -> None:
    git_directory = _formal_git_directory(repository)
    path = _journal_path(repository, journal.attempt_id)
    _require_single_journal(repository, journal.attempt_id)
    try:
        anchored_unlink_file_if_bytes(
            path,
            anchor=git_directory,
            expected_current=(_journal_bytes(journal),),
            missing_ok=True,
        )
    except OSError as error:
        raise FormalPublishError("formal publication journal cannot be removed safely") from error


def _cleanup_transaction(
    repository: Path,
    journal: _PublicationJournal,
) -> None:
    _remove_workspace(repository, journal)
    _remove_journal(repository, journal)


def _private_directory_is_absent_or_empty(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return True
    try:
        entries = (path, *path.rglob("*"))
    except OSError as error:
        raise FormalPublishError(
            f"formal publication residue cannot be inspected: {path}"
        ) from error
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError as error:
            raise FormalPublishError(
                f"formal publication residue cannot be inspected: {entry}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise FormalPublishError(
                f"formal publication residue is unsafe: {entry}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            if entry == path:
                raise FormalPublishError(
                    f"formal publication residue is unsafe: {entry}"
                )
            return False
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise FormalPublishError(
                f"formal publication residue is unsafe: {entry}"
            )
    return True


def _summary(
    repository: Path,
    journal: _PublicationJournal,
    *,
    evidence_files: int | None = None,
) -> FormalPublishSummary:
    completion = CompletedAttempt.model_validate(journal.completion)
    before = journal.registry_before_text.encode("utf-8")
    after = journal.registry_after_text.encode("utf-8")
    horizon_names = set(_HORIZON_FILES)
    evidence_names = set(journal.desired) - horizon_names
    measured_evidence_files = len(evidence_names)
    if evidence_files is not None and evidence_files != measured_evidence_files:
        raise FormalPublishError("publication evidence count changed during preparation")
    current = _registry_on_disk(repository)
    if current not in {before, after}:
        raise FormalPublishUncertainError(
            "formal registry differs from both journaled before and after bytes"
        )
    return FormalPublishSummary(
        state="INSTALLED",
        attempt_id=journal.attempt_id,
        completion=completion,
        registry_before_bytes=before,
        registry_after_bytes=after,
        registry_before_sha256=journal.registry_before_sha256,
        registry_after_sha256=journal.registry_after_sha256,
        report_sha256=journal.report_sha256,
        report_fingerprint=journal.report_fingerprint,
        evidence_files=measured_evidence_files,
        evidence_bytes=sum(
            value.size for name, value in journal.desired.items() if name in evidence_names
        ),
        horizon_files=len(horizon_names),
        horizon_bytes=sum(journal.desired[name].size for name in horizon_names),
        registry_already_appended=current == after,
    )


def _validate_journal_inputs(
    *,
    repository: Path,
    registration: RegisteredAttempt,
    registration_registry_bytes: bytes,
    raw_report: Mapping[str, Any],
    journal: _PublicationJournal,
) -> None:
    if RegisteredAttempt.model_validate(journal.registration) != registration:
        raise FormalPublishError("publication journal registration differs from the caller")
    current = _registry_on_disk(repository)
    before = journal.registry_before_text.encode("utf-8")
    after = journal.registry_after_text.encode("utf-8")
    if registration_registry_bytes != before:
        raise FormalPublishError(
            "recovery requires the exact journaled registration registry bytes"
        )
    if current not in {before, after}:
        raise FormalPublishUncertainError(
            "formal registry differs from both publication transaction states"
        )
    if journal.state != "INSTALLED" and current == after:
        raise FormalPublishUncertainError(
            "COMPLETED registry bytes appeared before evidence reached INSTALLED"
        )
    provenance = raw_report.get("provenance")
    if (
        not isinstance(provenance, dict)
        or raw_report.get("fingerprint") != journal.report_fingerprint
        or provenance.get("formal_attempt_id") != journal.attempt_id
        or provenance.get("formal_attempt_protocol_sha256") != registration.protocol_sha256
        or provenance.get("formal_attempt_registry_sha256") != journal.registry_before_sha256
    ):
        raise FormalPublishError("validated report input differs from the publication journal")
    _verify_repository_binding(repository, registration, raw_report)


def _validate_recovery_source(
    *,
    repository: Path,
    output: Path,
    registration: RegisteredAttempt,
    registration_registry_bytes: bytes,
    raw_report: Mapping[str, Any],
    journal: _PublicationJournal,
) -> tuple[dict[str, bytes], int]:
    _validate_journal_inputs(
        repository=repository,
        registration=registration,
        registration_registry_bytes=registration_registry_bytes,
        raw_report=raw_report,
        journal=journal,
    )
    desired, evidence_files = _desired_publication(
        repository=repository,
        output=output,
        raw_report=raw_report,
    )
    _require_desired_matches_journal(desired=desired, journal=journal)
    return desired, evidence_files


def _require_desired_matches_journal(
    *,
    desired: Mapping[str, bytes],
    journal: _PublicationJournal,
) -> None:
    if (
        set(desired) != set(journal.desired)
        or any(_digest(payload) != journal.desired[name] for name, payload in desired.items())
        or _tree_digest(
            {name: payload for name, payload in desired.items() if name not in _HORIZON_FILES}
        )
        != journal.source_evidence_sha256
    ):
        raise FormalPublishError("formal publication source changed after PREPARED")


def _registered_output(
    *,
    repository: Path,
    journal: _PublicationJournal,
    output: str | Path | None,
) -> Path:
    expected = _safe_relative_repository_path(
        repository,
        journal.output_path,
        label="formal publication output",
    )
    if not expected.is_dir():
        raise FormalPublishError("formal publication output is not a directory")
    if output is None:
        return expected
    try:
        supplied = Path(output).resolve(strict=True)
    except OSError as error:
        raise FormalPublishError("formal publication output is unavailable") from error
    if supplied != expected:
        raise FormalPublishError("formal publication output differs from registration")
    return expected


def _validated_source_desired(
    *,
    repository: Path,
    output: Path,
    raw_report: Mapping[str, Any],
    journal: _PublicationJournal,
) -> dict[str, bytes]:
    desired, _evidence_files = _desired_publication(
        repository=repository,
        output=output,
        raw_report=raw_report,
    )
    _require_desired_matches_journal(desired=desired, journal=journal)
    _verify_workspace_no_extras(repository, journal)
    for relative, payload in desired.items():
        staged = _verify_payload(
            _stage_path(repository, journal.attempt_id, relative),
            journal.desired[relative],
        )
        if staged != payload:
            raise FormalPublishError(
                f"staged publication differs from formal output at {relative}"
            )
    return desired


def _verify_installed_against_source(
    *,
    repository: Path,
    output: Path,
    raw_report: Mapping[str, Any],
    journal: _PublicationJournal,
) -> None:
    desired = _validated_source_desired(
        repository=repository,
        output=output,
        raw_report=raw_report,
        journal=journal,
    )
    for relative, payload in desired.items():
        installed = _verify_payload(
            repository / "benchmarks" / Path(relative),
            journal.desired[relative],
        )
        if installed != payload:
            raise FormalPublishError(
                f"installed publication differs from formal output at {relative}"
            )


def _validate_installed_inputs(
    *,
    repository: Path,
    registration: RegisteredAttempt,
    registration_registry_bytes: bytes,
    raw_report: Mapping[str, Any],
    journal: _PublicationJournal,
    output: str | Path | None,
) -> None:
    _validate_journal_inputs(
        repository=repository,
        registration=registration,
        registration_registry_bytes=registration_registry_bytes,
        raw_report=raw_report,
        journal=journal,
    )
    _verify_installed_targets(repository, journal)
    published_report = _read_regular(repository / "benchmarks" / "ablation_report.json")
    if _sha256(published_report) != journal.report_sha256:
        raise FormalPublishError("installed formal report differs from the journal")
    try:
        decoded = json.loads(published_report)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormalPublishError("installed formal report is not valid JSON") from error
    if decoded != dict(raw_report):
        raise FormalPublishError("installed formal report differs from validated raw data")
    current = _registry_on_disk(repository)
    before = journal.registry_before_text.encode("utf-8")
    if current == before:
        source = _registered_output(
            repository=repository,
            journal=journal,
            output=output,
        )
        _verify_installed_against_source(
            repository=repository,
            output=source,
            raw_report=raw_report,
            journal=journal,
        )


def install_formal_publication(
    *,
    repository: str | Path,
    output: str | Path,
    registration: RegisteredAttempt,
    registration_registry_bytes: bytes,
    raw_report: Mapping[str, Any],
    fault_injector: FaultInjector | None = None,
) -> FormalPublishSummary:
    """Prepare, recover, and install one validated evidence publication.

    The caller must hold the formal-attempt lifecycle lock.  This function is
    idempotent for the exact journaled registration and report.
    """
    repo = _safe_repository(repository)

    path = _journal_path(repo, registration.attempt_id)
    if path.exists() or path.is_symlink():
        return recover_formal_publication(
            repository=repo,
            output=output,
            registration=registration,
            registration_registry_bytes=registration_registry_bytes,
            raw_report=raw_report,
            fault_injector=fault_injector,
        )
    output_path = Path(output).resolve(strict=True)
    expected_output = repo / registration.output_path
    if output_path != expected_output.resolve(strict=True):
        raise FormalPublishError("formal publication output differs from registration")
    inspection = inspect_formal_publication(repository=repo)
    if inspection.status != "CLEAN":
        raise FormalPublishError(
            f"formal publication refuses an existing or malformed transaction: {inspection.reason}"
        )
    actual_registry = _registry_on_disk(repo)
    if (
        actual_registry != registration_registry_bytes
        or _head_registry_bytes(repo) != registration_registry_bytes
    ):
        raise FormalPublishError(
            "fresh publication requires registry bytes identical to committed HEAD"
        )
    _require_clean_worktree(repo)
    _validate_trusted_formal_checkout(repo, registration)

    desired_first, evidence_files = _desired_publication(
        repository=repo,
        output=output_path,
        raw_report=raw_report,
    )
    _revalidate_formal_output(
        repository=repo,
        output=output_path,
        raw_report=raw_report,
    )
    desired_second, second_evidence_files = _desired_publication(
        repository=repo,
        output=output_path,
        raw_report=raw_report,
    )
    if desired_second != desired_first or second_evidence_files != evidence_files:
        raise FormalPublishError(
            "formal publication source changed during publication-time validation"
        )
    _require_clean_worktree(repo)
    _validate_trusted_formal_checkout(repo, registration)
    report_bytes = desired_first["ablation_report.json"]
    _binding_from_inputs(
        registration=registration,
        registration_registry_bytes=registration_registry_bytes,
        raw_report=raw_report,
        report_bytes=report_bytes,
    )
    _verify_repository_binding(repo, registration, raw_report)
    existing = _existing_publication_files(
        repo / "benchmarks",
        expected=set(desired_first),
    )
    if not existing <= set(desired_first):
        raise FormalPublishError(
            f"publication target contains an extra file: {sorted(existing - set(desired_first))[0]}"
        )
    baseline = _tracked_publication_bytes(repo, set(desired_first))
    journal = _new_journal(
        registration=registration,
        registry_before=registration_registry_bytes,
        raw_report=raw_report,
        desired=desired_first,
        baseline=baseline,
        evidence={
            name: payload for name, payload in desired_first.items() if name not in _HORIZON_FILES
        },
    )
    _write_journal(repo, journal, expected_previous=None)
    _fault(fault_injector, "after_prepared")
    _prepare_workspace(
        repository=repo,
        journal=journal,
        desired=desired_first,
        baseline=baseline,
    )
    _fault(fault_injector, "after_stage")
    desired_third, third_evidence_files = _desired_publication(
        repository=repo,
        output=output_path,
        raw_report=raw_report,
    )
    if desired_third != desired_first or third_evidence_files != evidence_files:
        raise FormalPublishError("formal publication source changed while staging")
    _validate_trusted_formal_checkout(repo, registration)
    journal = _install_files(
        repository=repo,
        journal=journal,
        fault_injector=fault_injector,
    )
    return _summary(repo, journal, evidence_files=evidence_files)


def recover_formal_publication(
    *,
    repository: str | Path,
    output: str | Path,
    registration: RegisteredAttempt,
    registration_registry_bytes: bytes,
    raw_report: Mapping[str, Any],
    fault_injector: FaultInjector | None = None,
) -> FormalPublishSummary:
    """Roll an interrupted PREPARED/INSTALLING transaction forward.

    If the frozen source no longer matches, a PREPARED or INSTALLING
    transaction is rolled back to its tracked baseline.  INSTALLED is never
    rolled back here because its registry CAS may already have happened.
    """
    repo = _safe_repository(repository)
    journal = _read_journal(repo, registration.attempt_id)
    current_registry = _registry_on_disk(repo)
    _require_transaction_dirty_whitelist(
        repo,
        journal,
        allow_registry=(current_registry == journal.registry_after_text.encode("utf-8")),
    )
    _validate_trusted_formal_checkout(repo, registration)
    if journal.state == "INSTALLED":
        _validate_installed_inputs(
            repository=repo,
            registration=registration,
            registration_registry_bytes=registration_registry_bytes,
            raw_report=raw_report,
            journal=journal,
            output=output,
        )
        return _summary(repo, journal)

    output_path = Path(output).resolve(strict=True)
    expected_output = (repo / registration.output_path).resolve(strict=True)
    if output_path != expected_output:
        raise FormalPublishError("formal publication output differs from registration")
    try:
        desired, evidence_files = _validate_recovery_source(
            repository=repo,
            output=output_path,
            registration=registration,
            registration_registry_bytes=registration_registry_bytes,
            raw_report=raw_report,
            journal=journal,
        )
        baseline = {
            relative: _git(
                repo,
                ["show", f"HEAD:benchmarks/{relative}"],
                label=f"read tracked baseline for {relative}",
            )
            for relative, prior in journal.prior.items()
            if prior is not None
        }
        _prepare_workspace(
            repository=repo,
            journal=journal,
            desired=desired,
            baseline=baseline,
        )
        desired_after_stage, evidence_files_after_stage = _validate_recovery_source(
            repository=repo,
            output=output_path,
            registration=registration,
            registration_registry_bytes=registration_registry_bytes,
            raw_report=raw_report,
            journal=journal,
        )
        if desired_after_stage != desired or evidence_files_after_stage != evidence_files:
            raise FormalPublishError("formal publication source changed while rebuilding its stage")
        _validate_trusted_formal_checkout(repo, registration)
    except FormalPublishUncertainError:
        raise
    except FormalPublishError as error:
        _ensure_backups(repo, journal)
        _rollback(repo, journal)
        _cleanup_transaction(repo, journal)
        raise FormalPublishError(
            "formal publication source or stage changed; transaction was rolled back"
        ) from error
    journal = _install_files(
        repository=repo,
        journal=journal,
        fault_injector=fault_injector,
    )
    return _summary(repo, journal, evidence_files=evidence_files)


def verify_installed_publication(
    *,
    repository: str | Path,
    registration: RegisteredAttempt,
    registration_registry_bytes: bytes,
    raw_report: Mapping[str, Any],
) -> FormalPublishSummary:
    """Revalidate INSTALLED immediately before or after the exact registry CAS."""
    repo = _safe_repository(repository)
    journal = _read_journal(repo, registration.attempt_id)
    if journal.state != "INSTALLED":
        raise FormalPublishError("formal publication has not reached INSTALLED")
    _validate_trusted_formal_checkout(repo, registration)
    _validate_installed_inputs(
        repository=repo,
        registration=registration,
        registration_registry_bytes=registration_registry_bytes,
        raw_report=raw_report,
        journal=journal,
        output=repo / registration.output_path,
    )
    _verify_workspace(repo, journal)
    _require_transaction_dirty_whitelist(
        repo,
        journal,
        allow_registry=(_registry_on_disk(repo) == journal.registry_after_text.encode("utf-8")),
    )
    _validate_trusted_formal_checkout(repo, registration)
    return _summary(repo, journal)


def cleanup_formal_publication(
    *,
    repository: str | Path,
    attempt_id: str,
) -> FormalPublishFinalizeSummary:
    """Remove a transaction only after the exact registry-after bytes are present."""
    repo = _safe_repository(repository)
    journal = _read_journal(repo, attempt_id)
    if journal.state != "INSTALLED":
        raise FormalPublishError("only an INSTALLED publication can be cleaned")
    current = _registry_on_disk(repo)
    after = journal.registry_after_text.encode("utf-8")
    if current != after:
        raise FormalPublishUncertainError(
            "cannot clean publication before exact COMPLETED registry bytes are present"
        )
    _verify_installed_targets(repo, journal)
    _verify_workspace_no_extras(repo, journal)
    _cleanup_transaction(repo, journal)
    return FormalPublishFinalizeSummary(
        attempt_id=attempt_id,
        action="COMMITTED_AND_CLEANED",
    )


def finalize_formal_publication(
    *,
    repository: str | Path,
    attempt_id: str,
    observed_registry_bytes: bytes | None,
    fault_injector: FaultInjector | None = None,
) -> FormalPublishFinalizeSummary:
    """Resolve the CAS result without guessing after an ambiguous append."""
    repo = _safe_repository(repository)
    journal = _read_journal(repo, attempt_id)
    current = _registry_on_disk(repo)
    before = journal.registry_before_text.encode("utf-8")
    after = journal.registry_after_text.encode("utf-8")
    if observed_registry_bytes is None or current != observed_registry_bytes:
        raise FormalPublishUncertainError(
            "registry append outcome is uncertain; publication was preserved"
        )
    if current == after:
        _fault(fault_injector, "before_cleanup")
        return cleanup_formal_publication(repository=repo, attempt_id=attempt_id)
    if current == before:
        _fault(fault_injector, "before_rollback")
        if not _targets_match_exact_prior(repo, journal):
            _ensure_backups(repo, journal)
            _rollback(repo, journal)
        _fault(fault_injector, "after_rollback")
        _cleanup_transaction(repo, journal)
        return FormalPublishFinalizeSummary(
            attempt_id=attempt_id,
            action="ROLLED_BACK_AND_CLEANED",
        )
    raise FormalPublishUncertainError(
        "registry append outcome is uncertain; publication was preserved"
    )


def inspect_formal_publication(
    *,
    repository: str | Path,
) -> FormalPublishInspection:
    """Inspect journals without mutating evidence, registry, or transaction state."""
    try:
        repo = _safe_repository(repository)
        directory = _journal_directory(repo)
        workspace_root = repo / "benchmarks" / _WORKSPACE_DIRECTORY
        journal_root_clean = _private_directory_is_absent_or_empty(directory)
        if journal_root_clean:
            if _private_directory_is_absent_or_empty(workspace_root):
                return FormalPublishInspection(status="CLEAN")
            return FormalPublishInspection(
                status="QUARANTINED",
                reason="publication workspace exists without a durable journal",
            )
        if not directory.exists() and not directory.is_symlink():
            return FormalPublishInspection(status="CLEAN")
        if directory.is_symlink() or not directory.is_dir():
            return FormalPublishInspection(
                status="QUARANTINED",
                reason="publication journal directory is unsafe",
            )
        entries = list(directory.iterdir())
        if len(entries) != 1:
            return FormalPublishInspection(
                status="QUARANTINED",
                reason="expected exactly one active publication journal",
            )
        entry = entries[0]
        match = re.fullmatch(r"([0-9a-f]{64})\.json", entry.name)
        if match is None:
            return FormalPublishInspection(
                status="QUARANTINED",
                reason="publication journal has an unexpected name",
            )
        attempt_id = match.group(1)
        journal = _read_journal(repo, attempt_id)
        current = _registry_on_disk(repo)
        before = journal.registry_before_text.encode("utf-8")
        after = journal.registry_after_text.encode("utf-8")
        if current not in {before, after}:
            return FormalPublishInspection(
                status="QUARANTINED",
                attempt_id=attempt_id,
                transaction_state=journal.state,
                reason="registry differs from both journaled states",
            )
        if journal.state != "INSTALLED" and current == after:
            return FormalPublishInspection(
                status="QUARANTINED",
                attempt_id=attempt_id,
                transaction_state=journal.state,
                reason="registry completed before evidence installation",
            )
        _require_transaction_dirty_whitelist(
            repo,
            journal,
            allow_registry=current == after,
        )
        _verify_workspace_no_extras(repo, journal)
        if journal.state == "INSTALLED" and current == after:
            _verify_installed_targets(repo, journal)
        elif journal.state == "INSTALLED":
            if not _targets_match_exact_prior(repo, journal):
                registration = RegisteredAttempt.model_validate(
                    journal.registration
                )
                source = _registered_output(
                    repository=repo,
                    journal=journal,
                    output=None,
                )
                try:
                    raw_report = json.loads(
                        _read_regular(source / "ablation_report.json")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise FormalPublishError(
                        "formal publication output report is invalid"
                    ) from error
                if not isinstance(raw_report, dict):
                    raise FormalPublishError(
                        "formal publication output report is not an object"
                    )
                _validate_journal_inputs(
                    repository=repo,
                    registration=registration,
                    registration_registry_bytes=before,
                    raw_report=raw_report,
                    journal=journal,
                )
                _validated_source_desired(
                    repository=repo,
                    output=source,
                    raw_report=raw_report,
                    journal=journal,
                )
                try:
                    _verify_installed_targets(repo, journal)
                except FormalPublishError:
                    _preflight_rollback_targets(repo, journal)
        else:
            # A crash may occur after rollback restored some or all prior
            # bytes but before the transaction metadata was removed.  Any
            # mixture of exact prior/desired bytes is resumable; a third value
            # remains quarantined.
            _preflight_rollback_targets(repo, journal)
        return FormalPublishInspection(
            status="RECOVERY_REQUIRED",
            attempt_id=attempt_id,
            transaction_state=journal.state,
            reason=(
                "registry append is complete; transaction cleanup is required"
                if current == after
                else "formal publication transaction must be resumed"
            ),
        )
    except (FormalPublishError, OSError, ValueError) as error:
        return FormalPublishInspection(
            status="QUARANTINED",
            reason=str(error),
        )
