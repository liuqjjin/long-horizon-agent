"""Validate public benchmark claims against committed machine-readable evidence.

The release check intentionally reads only the public project overview and the
generated benchmark reports.  Resume text lives outside the repository and is
not parsed here.

Reports through schema 3 predate delivery-correctness separation, immutable
input snapshots, and nonce-bound scorer receipts. They remain readable as
historical evidence, but only schema 4 can support current formal claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from .ablation import (
    _MAX_ARTIFACT_BYTES,
    _MAX_REPORT_BYTES,
    _MAX_SCORER_EVIDENCE_BYTES,
    _MAX_TASK_BYTES,
    CONDITIONS,
    AblationReport,
    RunRecord,
    ScoreOutcome,
    ScorerEvidenceBinding,
    _ablation_report_from_raw,
    _frozen_artifact_bytes,
    _input_snapshot_digest,
    _read_bounded_bytes,
    _read_bounded_text,
    _record_from_raw,
    _repo_digest,
    _scorer_evidence_bytes,
    _source_file_digests,
    _source_tree_digest,
    _validate_scorer_evidence,
)
from .bench.stats import mcnemar_exact, wilson_interval
from .bench.terminal_public_evidence import (
    TerminalBenchPublicEvidenceValidation,
    validate_terminal_bench_public_evidence,
)
from .horizon import Cells, build_report

_CONDITION_NAMES = tuple(name for name, _blurb in CONDITIONS)
_LEGACY_README_MARKER = "历史报告"
_FORMAL_README_MARKER = "正式报告"
_LEGACY_ABLATION_MARKER = "legacy snapshot"
_LEGACY_HORIZON_MARKER = "legacy snapshot"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RELEASE_TEXT_BYTES = 8 * 1024 * 1024
_TERMINAL_EVIDENCE_DIR = "terminal_bench_2_1"
_TERMINAL_SUBSET_LABEL = "Terminal-Bench 2.1 固定 20 题子集"
_TERMINAL_NUMERIC_CLAIM = re.compile(
    r"(?:\d+\s*/\s*20|"
    r"(?:passed|pass|failed|fail|error|通过|失败|正确|错误|得分|成绩)"
    r"\s*(?:为|[:：=])?\s*\d+)",
    re.IGNORECASE,
)


class ReleaseClaimsError(ValueError):
    """A public claim is missing, stale, or unsupported by committed evidence."""


@dataclass(frozen=True)
class ReleaseClaimsSummary:
    status: str
    tasks: int
    repetitions: int
    cells: int
    model: str
    trust_successes: int
    trust_false_successes: int
    gate_successes: int
    gate_interceptions: int
    verify_successes: int
    headline_mcnemar_p: float
    terminal_bench: TerminalBenchPublicEvidenceValidation | None


@dataclass(frozen=True)
class _AblationFacts:
    report: AblationReport
    raw: dict[str, Any]
    records: tuple[RunRecord, ...]
    status: str
    cells: int
    trust_successes: int
    trust_false_successes: int
    gate_successes: int
    gate_interceptions: int
    verify_successes: int
    trust_gate_p: float


def _fail(message: str) -> NoReturn:
    raise ReleaseClaimsError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_bounded_text(path, max_bytes=_MAX_REPORT_BYTES)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path} must contain a JSON object")
    return value


def _read_text(path: Path) -> str:
    try:
        return _read_bounded_text(path, max_bytes=_MAX_RELEASE_TEXT_BYTES)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        _fail(f"cannot read {path}: {exc}")


def _is_close(left: Any, right: float) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and math.isclose(
        float(left), right, rel_tol=1e-12, abs_tol=1e-12
    )


def _require_rate(stat: dict[str, Any], field: str, expected: float, condition: str) -> None:
    if not _is_close(stat.get(field), expected):
        _fail(
            f"ablation stats for {condition!r} have stale {field}: "
            f"expected {expected}, got {stat.get(field)!r}"
        )


def _record_bool(record: dict[str, Any], field: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        _fail(f"ablation record field {field!r} must be boolean")
    return value


def _validate_record_grid(raw: dict[str, Any]) -> tuple[list[str], int, tuple[RunRecord, ...]]:
    schema_raw = raw.get("schema_version", 1)
    if not isinstance(schema_raw, int) or isinstance(schema_raw, bool) or schema_raw < 1:
        _fail("ablation schema_version must be a positive integer")
    schema_version = schema_raw
    tasks = raw.get("tasks")
    reps = raw.get("reps")
    records_raw = raw.get("records")
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(task, str) and task for task in tasks)
        or len(tasks) != len(set(tasks))
    ):
        _fail("ablation tasks must be a non-empty unique string list")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps <= 0:
        _fail("ablation reps must be a positive integer")
    if not isinstance(records_raw, list):
        _fail("ablation records must be a list")

    task_names = cast(list[str], tasks)
    rep_count = cast(int, reps)
    records_list = cast(list[Any], records_raw)
    seen: set[tuple[str, str, int]] = set()
    records: list[RunRecord] = []
    for record_raw in records_list:
        if not isinstance(record_raw, dict):
            _fail("every ablation record must be an object")
        task = record_raw.get("task")
        condition = record_raw.get("condition")
        rep = record_raw.get("rep")
        status = record_raw.get("status")
        if (
            not isinstance(task, str)
            or task not in task_names
            or not isinstance(condition, str)
            or condition not in _CONDITION_NAMES
        ):
            _fail(f"ablation record has unknown task or condition: {task!r}/{condition!r}")
        if not isinstance(rep, int) or isinstance(rep, bool) or not 0 <= rep < rep_count:
            _fail(f"ablation record has invalid repetition: {rep!r}")
        if status not in {"DONE", "FAILED", "ERROR"}:
            _fail(f"ablation record has invalid status: {status!r}")
        key = (task, condition, rep)
        if key in seen:
            _fail(f"duplicate ablation record: {key!r}")
        seen.add(key)

        claimed = _record_bool(record_raw, "claimed_success")
        true_success = _record_bool(record_raw, "true_success")
        artifact_correct = (
            _record_bool(record_raw, "artifact_correct")
            if schema_version >= 4
            else true_success
        )
        false_success = _record_bool(record_raw, "false_success")
        if false_success != (claimed and not artifact_correct):
            _fail(f"ablation false_success is inconsistent for {key!r}")
        if schema_version >= 4:
            if true_success != (claimed and artifact_correct):
                _fail(f"ablation true_success is inconsistent for {key!r}")
            scorer_outcome = record_raw.get("scorer_outcome")
            expected_outcome = (
                ScoreOutcome.INFRA_ERROR.value
                if status == "ERROR"
                else (
                    ScoreOutcome.PASS.value
                    if artifact_correct
                    else ScoreOutcome.TEST_FAIL.value
                )
            )
            if scorer_outcome != expected_outcome:
                _fail(f"ablation scorer_outcome is inconsistent for {key!r}")
            if status != "ERROR" and status != ("DONE" if claimed else "FAILED"):
                _fail(f"ablation delivery status is inconsistent for {key!r}")
            for field in ("scorer_expected_tests", "scorer_passed_tests"):
                value = record_raw.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    _fail(f"ablation record field {field!r} is invalid for {key!r}")
            if status != "ERROR" and record_raw["scorer_expected_tests"] <= 0:
                _fail(f"ablation scorer collected no expected tests for {key!r}")
        try:
            records.append(_record_from_raw(record_raw, schema_version=schema_version))
        except TypeError as exc:
            _fail(f"invalid ablation record {key!r}: {exc}")

    expected = {
        (task, condition, rep)
        for task in task_names
        for condition in _CONDITION_NAMES
        for rep in range(rep_count)
    }
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        _fail(f"ablation record grid is incomplete (missing={missing[:3]}, extra={extra[:3]})")
    return task_names, rep_count, tuple(records)


def _validate_record_artifacts(records: tuple[RunRecord, ...]) -> None:
    """Bind each formal cell to a valid artifact and preserve the paired design."""
    by_cell: dict[tuple[str, int], dict[str, str]] = {}
    evidence_by_cell: dict[tuple[str, int], dict[str, str]] = {}
    for record in records:
        if not _HEX_64.fullmatch(record.artifact_sha256):
            _fail(
                "formal ablation record has an invalid artifact_sha256 "
                f"for {(record.task, record.condition, record.rep)!r}"
            )
        by_cell.setdefault((record.task, record.rep), {})[
            record.condition
        ] = record.artifact_sha256
        evidence_by_cell.setdefault((record.task, record.rep), {})[
            record.condition
        ] = record.scorer_evidence_sha256
    for key, conditions in by_cell.items():
        if conditions["trust"] != conditions["gate"]:
            _fail(
                "formal ablation trust/gate records do not reference the same "
                f"first-attempt artifact for {key!r}"
            )
        evidence = evidence_by_cell[key]
        if evidence["trust"] != evidence["gate"]:
            _fail(
                "formal ablation trust/gate records do not reference the same "
                f"scorer receipt for {key!r}"
            )


def _validate_artifact_store(
    raw: dict[str, Any],
    records: tuple[RunRecord, ...],
    report_dir: Path,
) -> None:
    store = raw.get("artifact_store")
    expected_digests = {record.artifact_sha256 for record in records}
    expected_store = {
        "schema_version": 1,
        "path": "artifacts",
        "encoding": "canonical-json",
        "count": len(expected_digests),
    }
    if store != expected_store:
        _fail("schema-4 ablation report has missing or stale artifact_store metadata")

    artifact_dir = report_dir / "artifacts"
    for digest in sorted(expected_digests):
        path = artifact_dir / f"{digest}.json"
        try:
            payload = _read_bounded_bytes(
                path,
                max_bytes=_MAX_ARTIFACT_BYTES,
            )
        except (OSError, ValueError) as exc:
            _fail(f"cannot read schema-4 ablation artifact {digest}: {exc}")
        if hashlib.sha256(payload).hexdigest() != digest:
            _fail(f"schema-4 ablation artifact digest does not match {digest}")
        try:
            artifact = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(f"schema-4 ablation artifact {digest} is invalid JSON: {exc}")
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"schema_version", "files"}
            or artifact.get("schema_version") != 1
            or not isinstance(artifact.get("files"), dict)
        ):
            _fail(f"schema-4 ablation artifact {digest} has an invalid envelope")
        files = cast(dict[Any, Any], artifact["files"])
        if not all(
            isinstance(rel, str) and (content is None or isinstance(content, str))
            for rel, content in files.items()
        ):
            _fail(f"schema-4 ablation artifact {digest} has invalid file entries")
        try:
            canonical = _frozen_artifact_bytes(cast(dict[str, str | None], files))
        except ValueError as exc:
            _fail(f"schema-4 ablation artifact {digest} is unsafe: {exc}")
        if payload != canonical:
            _fail(f"schema-4 ablation artifact {digest} is not canonical")


def _validate_scorer_evidence_store(
    raw: dict[str, Any],
    records: tuple[RunRecord, ...],
    report_dir: Path,
) -> None:
    expected_digests = {
        record.scorer_evidence_sha256
        for record in records
        if record.status != "ERROR"
    }
    expected_store = {
        "schema_version": 2,
        "path": "scorer_evidence",
        "encoding": "canonical-json",
        "count": len(expected_digests),
    }
    if raw.get("scorer_evidence_store") != expected_store:
        _fail("schema-4 ablation report has stale scorer_evidence_store metadata")

    evidence_dir = report_dir / "scorer_evidence"
    cache: dict[str, dict[str, Any]] = {}
    for digest in sorted(expected_digests):
        if not _HEX_64.fullmatch(digest):
            _fail(f"schema-4 ablation scorer evidence has an invalid digest: {digest!r}")
        path = evidence_dir / f"{digest}.json"
        try:
            payload = _read_bounded_bytes(
                path,
                max_bytes=_MAX_SCORER_EVIDENCE_BYTES,
            )
        except (OSError, ValueError) as exc:
            _fail(f"cannot read schema-4 scorer evidence {digest}: {exc}")
        if hashlib.sha256(payload).hexdigest() != digest:
            _fail(f"schema-4 scorer evidence digest does not match {digest}")
        try:
            evidence = json.loads(payload)
            if payload != _scorer_evidence_bytes(evidence):
                _fail(f"schema-4 scorer evidence {digest} is not canonical")
            _validate_scorer_evidence(evidence)
            cache[digest] = evidence
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            _fail(f"schema-4 scorer evidence {digest} is invalid: {exc}")

    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        _fail("schema-4 scorer evidence is missing provenance bindings")
    snapshots = provenance.get("input_snapshot_sha256")
    scorer_backend = provenance.get("scorer_backend")
    scorer_image_id = provenance.get("scorer_image_id")
    if (
        not isinstance(snapshots, dict)
        or not isinstance(scorer_backend, str)
        or not scorer_backend
    ):
        _fail("schema-4 scorer evidence has invalid provenance bindings")
    for record in records:
        if record.status == "ERROR":
            continue
        snapshot_digest = snapshots.get(record.task)
        if not isinstance(snapshot_digest, str):
            _fail(f"schema-4 scorer evidence has no input snapshot for {record.task!r}")
        expected_binding = ScorerEvidenceBinding(
            task=record.task,
            rep=record.rep,
            artifact_sha256=record.artifact_sha256,
            input_snapshot_sha256=snapshot_digest,
            scorer_backend=scorer_backend,
            scorer_image_id=scorer_image_id if isinstance(scorer_image_id, str) else None,
        )
        try:
            outcome, expected, passed = _validate_scorer_evidence(
                cache[record.scorer_evidence_sha256],
                expected_binding=expected_binding,
            )
        except (TypeError, ValueError) as exc:
            _fail(
                "schema-4 scorer evidence binding disagrees with record "
                f"{(record.task, record.condition, record.rep)!r}: {exc}"
            )
        if (
            record.scorer_outcome != outcome.value
            or record.scorer_expected_tests != expected
            or record.scorer_passed_tests != passed
            or record.artifact_correct != (outcome is ScoreOutcome.PASS)
        ):
            _fail(
                "schema-4 scorer evidence disagrees with record "
                f"{(record.task, record.condition, record.rep)!r}"
            )


def _boundary_interval_problem(
    stat: dict[str, Any],
    *,
    field: str,
    successes: int,
    total: int,
) -> str | None:
    interval_name = {
        "artifact_correct_rate": "artifact_ci",
        "true_success_rate": "true_ci",
        "false_success_rate": "false_ci",
    }[field]
    interval = stat.get(interval_name)
    if successes not in (0, total):
        return None
    if not isinstance(interval, list) or len(interval) != 2:
        return f"{stat.get('condition')} {interval_name} is missing"
    expected = wilson_interval(successes, total)
    if not all(_is_close(actual, wanted) for actual, wanted in zip(interval, expected)):
        return (
            f"{stat.get('condition')} {interval_name} must use Wilson "
            f"{expected}, got {interval!r}"
        )
    return None


def _validate_condition_stats(
    raw: dict[str, Any], records: tuple[RunRecord, ...]
) -> list[str]:
    schema_version = raw.get("schema_version", 1)
    current_schema = isinstance(schema_version, int) and schema_version >= 4
    stats_raw = raw.get("stats")
    if not isinstance(stats_raw, list):
        _fail("ablation stats must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for stat in cast(list[Any], stats_raw):
        if not isinstance(stat, dict) or stat.get("condition") not in _CONDITION_NAMES:
            _fail("ablation stats contain an unknown condition")
        name = str(stat["condition"])
        if name in by_name:
            _fail(f"duplicate ablation stats for {name!r}")
        by_name[name] = stat
    if set(by_name) != set(_CONDITION_NAMES):
        _fail("ablation stats do not cover trust, gate, and verify exactly once")

    boundary_problems: list[str] = []
    for condition in _CONDITION_NAMES:
        condition_records = [record for record in records if record.condition == condition]
        usable = [record for record in condition_records if record.status != "ERROR"]
        stat = by_name[condition]
        errors = len(condition_records) - len(usable)
        n = len(usable)
        if stat.get("n") != n or stat.get("errors") != errors:
            _fail(f"ablation stats for {condition!r} have stale n/errors")
        if not n:
            _fail(f"ablation condition {condition!r} has no usable measured cells")

        claimed = sum(record.claimed_success for record in usable)
        artifacts_correct = sum(record.artifact_correct for record in usable)
        delivered_correct = sum(record.true_success for record in usable)
        false_successes = sum(record.false_success for record in usable)
        repairs = sum(record.repairs for record in usable)
        _require_rate(stat, "claimed_success_rate", claimed / n, condition)
        if current_schema:
            _require_rate(
                stat,
                "artifact_correct_rate",
                artifacts_correct / n,
                condition,
            )
            _require_rate(stat, "true_success_rate", delivered_correct / n, condition)
        else:
            _require_rate(stat, "true_success_rate", artifacts_correct / n, condition)
        _require_rate(stat, "false_success_rate", false_successes / n, condition)
        _require_rate(stat, "mean_repairs", repairs / n, condition)

        predictions = [record for record in usable if record.gate_prediction is not None]
        if predictions:
            confusion = {
                "tp": sum(
                    bool(record.gate_prediction) and record.artifact_correct
                    for record in predictions
                ),
                "fp": sum(
                    bool(record.gate_prediction) and not record.artifact_correct
                    for record in predictions
                ),
                "tn": sum(
                    not record.gate_prediction and not record.artifact_correct
                    for record in predictions
                ),
                "fn": sum(
                    not record.gate_prediction and record.artifact_correct
                    for record in predictions
                ),
            }
            for field, expected in confusion.items():
                if stat.get(field) != expected:
                    _fail(f"ablation stats for {condition!r} have stale {field}")

        rate_fields = (
            (
                ("artifact_correct_rate", artifacts_correct),
                ("true_success_rate", delivered_correct),
                ("false_success_rate", false_successes),
            )
            if current_schema
            else (
                ("true_success_rate", artifacts_correct),
                ("false_success_rate", false_successes),
            )
        )
        for field, successes in rate_fields:
            problem = _boundary_interval_problem(
                stat, field=field, successes=successes, total=n
            )
            if problem:
                boundary_problems.append(problem)
    return boundary_problems


def _paired_p(
    records: tuple[RunRecord, ...], left: str, right: str, metric: str
) -> float:
    def outcomes(condition: str) -> dict[tuple[str, int], bool]:
        return {
            (record.task, record.rep): bool(getattr(record, metric))
            for record in records
            if record.condition == condition and record.status != "ERROR"
        }

    left_outcomes = outcomes(left)
    right_outcomes = outcomes(right)
    pairs = set(left_outcomes) & set(right_outcomes)
    if not pairs:
        _fail(f"no paired cells for {left}/{right}")
    only_left = sum(left_outcomes[key] and not right_outcomes[key] for key in pairs)
    only_right = sum(right_outcomes[key] and not left_outcomes[key] for key in pairs)
    return mcnemar_exact(only_left, only_right)


def _repo_evidence_path(repo_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"formal ablation provenance has an invalid {label} path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"formal ablation provenance {label} path escapes the repository")
    path = repo_root / relative
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repo_root.resolve())
    except (OSError, ValueError) as exc:
        _fail(f"formal ablation provenance {label} path is unavailable: {exc}")
    if path.is_symlink():
        _fail(f"formal ablation provenance {label} path must not be a symlink")
    return path


def _validate_provenance(
    raw: dict[str, Any],
    tasks: list[str],
    repo_root: Path,
) -> None:
    provenance_raw = raw.get("provenance")
    if not isinstance(provenance_raw, dict):
        _fail("formal ablation report is missing provenance")
    provenance = cast(dict[str, Any], provenance_raw)
    required_strings = (
        "generated_at",
        "harness_version",
        "git_commit",
        "source_tree_sha256",
        "requested_llm_backend",
        "actual_llm_backend",
        "model",
        "agent_backend",
        "scorer_requested",
        "scorer_backend",
        "platform",
        "python_version",
        "pytest_version",
    )
    for field in required_strings:
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            _fail(f"formal ablation provenance is missing {field!r}")
    if provenance.get("git_dirty") is not False:
        _fail("formal ablation provenance must come from a clean git worktree")
    if provenance["model"] != raw.get("model"):
        _fail("formal ablation model differs from provenance")
    if provenance["requested_llm_backend"] != raw.get("llm"):
        _fail("formal ablation LLM backend differs from provenance")
    if (
        provenance["requested_llm_backend"] != "codex_cli"
        or provenance["actual_llm_backend"] != "codex_cli"
    ):
        _fail("formal ablation evidence requires the hardened Codex CLI backend")
    if (
        provenance.get("agent_backend") != "docker"
        or provenance.get("scorer_requested") != "docker"
        or provenance.get("scorer_backend") != "docker"
    ):
        _fail("formal ablation evidence requires Docker for gate and independent scoring")
    scorer_image_id = provenance.get("scorer_image_id")
    if (
        not isinstance(scorer_image_id, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", scorer_image_id) is None
    ):
        _fail("formal ablation provenance is missing an immutable scorer image ID")
    if provenance["scorer_backend"] != raw.get("scorer"):
        _fail("formal ablation scorer differs from provenance")
    if provenance["harness_version"] != raw.get("harness_version"):
        _fail("formal ablation harness version differs from provenance")
    if not _HEX_64.fullmatch(provenance["source_tree_sha256"]):
        _fail("formal ablation provenance has an invalid source_tree_sha256")
    source_root = repo_root / "src" / "lha"
    if not source_root.is_dir():
        _fail("formal ablation checkout is missing src/lha")
    measured_source_files = _source_file_digests(source_root)
    if provenance.get("source_files") != measured_source_files:
        _fail("formal ablation source_files disagree with the current checkout")
    if provenance["source_tree_sha256"] != _source_tree_digest(measured_source_files):
        _fail("formal ablation source_tree_sha256 disagrees with the current checkout")
    if not _HEX_64.fullmatch(str(raw.get("fingerprint", ""))):
        _fail("formal ablation report has an invalid fingerprint")
    if provenance["actual_llm_backend"].endswith("_cli") and not provenance.get("cli_version"):
        _fail("formal CLI-backed ablation provenance is missing cli_version")
    for field in (
        "task_paths",
        "corpus_paths",
        "task_files_sha256",
        "corpus_sha256",
        "input_snapshot_sha256",
    ):
        values = provenance.get(field)
        if not isinstance(values, dict) or set(values) != set(tasks):
            _fail(f"formal ablation provenance {field!r} does not cover every task")
    for task in tasks:
        task_path = _repo_evidence_path(
            repo_root,
            provenance["task_paths"][task],
            label=f"{task} task",
        )
        corpus_path = _repo_evidence_path(
            repo_root,
            provenance["corpus_paths"][task],
            label=f"{task} corpus",
        )
        if not task_path.is_file() or not corpus_path.is_dir():
            _fail(f"formal ablation task/corpus evidence is not the expected type for {task!r}")
        try:
            task_bytes = _read_bounded_bytes(
                task_path,
                max_bytes=_MAX_TASK_BYTES,
            )
        except (OSError, ValueError) as exc:
            _fail(f"cannot read formal task evidence for {task!r}: {exc}")
        task_digest = hashlib.sha256(task_bytes).hexdigest()
        corpus_digest = _repo_digest(corpus_path)
        if provenance["task_files_sha256"][task] != task_digest:
            _fail(f"formal ablation task digest disagrees with committed bytes for {task!r}")
        if provenance["corpus_sha256"][task] != corpus_digest:
            _fail(f"formal ablation corpus digest disagrees with committed bytes for {task!r}")
        expected_snapshot = _input_snapshot_digest(task_digest, corpus_digest)
        if provenance["input_snapshot_sha256"][task] != expected_snapshot:
            _fail(f"formal ablation input snapshot digest is stale for {task!r}")
    configuration = provenance.get("configuration")
    expected_configuration = {
        "repetitions": raw.get("reps"),
        "task_count": len(tasks),
        "conditions": list(_CONDITION_NAMES),
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(field) != expected
        for field, expected in expected_configuration.items()
    ):
        _fail("formal ablation provenance configuration differs from the report")
    required_protocol = {
        "cache_schema": 7,
        "report_schema": 4,
        "frozen_artifact_schema": 1,
        "input_snapshot_schema": 1,
        "scorer_evidence_schema": 2,
        "scorer_result_source": "nonce-bound-pytest-hook-receipt",
    }
    if any(configuration.get(field) != expected for field, expected in required_protocol.items()):
        _fail("formal ablation provenance does not use the schema-4 evidence protocol")


def _format_percent(value: float) -> str:
    return f"{100 * value:.0f}%"


def _validate_legacy_ablation_markdown(
    markdown: str, report: AblationReport, records: tuple[RunRecord, ...]
) -> None:
    if _LEGACY_ABLATION_MARKER not in markdown.lower():
        _fail("historical ablation Markdown must be labelled 'legacy snapshot'")
    regenerated = report.to_markdown().replace(
        "# Verification ablation",
        "# Verification ablation — legacy snapshot",
        1,
    )
    if markdown.strip() == regenerated.strip():
        return
    header = re.search(
        r"implementer: `([^`]+)`(?: \([^)]*\))? · model: `([^`]+)` · "
        r"tasks: (\d+) · repetitions: (\d+) .* final scorer: `([^`]+)`",
        markdown,
    )
    expected_header = (
        report.llm,
        report.model or "(backend default)",
        str(len(report.tasks)),
        str(report.reps),
        report.scorer,
    )
    if header is None or header.groups() != expected_header:
        _fail("legacy ablation Markdown header differs from its JSON report")

    stats = {stat.condition: stat for stat in report.stats}
    for condition in _CONDITION_NAMES:
        stat = stats[condition]
        row = re.search(
            rf"^\| `{condition}` \| (\d+) \| ([0-9]+%) \| "
            r"([0-9]+%)(?: \([^)]*\))? \| ([0-9]+%)(?: \([^)]*\))? \| "
            r"([0-9]+\.[0-9]+) \| (\d+) \|$",
            markdown,
            re.MULTILINE,
        )
        expected = (
            str(stat.n),
            _format_percent(stat.claimed_success_rate),
            _format_percent(stat.true_success_rate),
            _format_percent(stat.false_success_rate),
            f"{stat.mean_repairs:.2f}",
            str(stat.errors),
        )
        if row is None or row.groups() != expected:
            _fail(f"legacy ablation Markdown row for {condition!r} is stale")

    expected_p_values = (
        _paired_p(records, "trust", "gate", "false_success"),
        _paired_p(records, "gate", "verify", "true_success"),
    )
    for expected in expected_p_values:
        if f"exact McNemar p = {expected:.2f}" not in markdown:
            _fail("legacy ablation Markdown has a stale McNemar p-value")


def _validate_ablation(ablation_json: Path, ablation_md: Path) -> _AblationFacts:
    raw = _load_json(ablation_json)
    tasks, reps, records = _validate_record_grid(raw)
    boundary_problems = _validate_condition_stats(raw, records)
    report = _ablation_report_from_raw(raw)
    if report.tasks != tasks or report.reps != reps:
        _fail("ablation loader disagrees with the report's task/repetition fields")
    if not report.model:
        _fail("ablation report must name the evaluated model")

    schema_version = raw.get("schema_version", 1)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        _fail("ablation schema_version must be an integer")
    if schema_version > 4:
        _fail(f"unsupported ablation schema_version {schema_version}")
    status = "formal" if schema_version == 4 else "legacy"
    errors = sum(record.status == "ERROR" for record in records)
    markdown = _read_text(ablation_md)
    if status == "formal":
        _validate_record_artifacts(records)
        _validate_artifact_store(raw, records, ablation_json.parent)
        _validate_scorer_evidence_store(raw, records, ablation_json.parent)
        if errors:
            _fail(f"formal ablation report contains {errors} ERROR cells")
        _validate_provenance(raw, tasks, ablation_json.parent.parent)
        if boundary_problems:
            _fail("formal ablation report violates the Wilson contract: " + boundary_problems[0])
        if _LEGACY_ABLATION_MARKER in markdown.lower():
            _fail("formal ablation Markdown is still labelled as a legacy snapshot")
        if markdown.strip() != report.to_markdown().strip():
            _fail("formal ablation Markdown was not generated from the committed JSON")
    else:
        _validate_legacy_ablation_markdown(markdown, report, records)

    usable = [record for record in records if record.status != "ERROR"]
    trust = [record for record in usable if record.condition == "trust"]
    gate = [record for record in usable if record.condition == "gate"]
    verify = [record for record in usable if record.condition == "verify"]
    return _AblationFacts(
        report=report,
        raw=raw,
        records=records,
        status=status,
        cells=len(trust),
        trust_successes=sum(record.true_success for record in trust),
        trust_false_successes=sum(record.false_success for record in trust),
        gate_successes=sum(record.true_success for record in gate),
        gate_interceptions=sum(
            not record.claimed_success and not record.artifact_correct for record in gate
        ),
        verify_successes=sum(record.true_success for record in verify),
        trust_gate_p=_paired_p(records, "trust", "gate", "false_success"),
    )


def _validate_horizon(
    horizon_json: Path,
    horizon_md: Path,
    ablation_json: Path,
    ablation: _AblationFacts,
) -> float:
    raw = _load_json(horizon_json)
    expected_source = "benchmarks/ablation_report.json"
    if raw.get("source") != expected_source:
        _fail(
            "committed horizon source must be 'benchmarks/ablation_report.json'; "
            "regenerate it after committing the ablation snapshot"
        )
    cells = Cells(
        tasks=sorted(ablation.report.tasks),
        reps=sorted(
            {
                record.rep
                for record in ablation.records
                if record.status != "ERROR"
            }
        ),
        outcome={
            (record.condition, record.task, record.rep): record.true_success
            for record in ablation.records
            if record.status != "ERROR"
        },
        model=ablation.report.model,
        source=expected_source,
    )
    expected = build_report(cells)
    expected_raw = json.loads(expected.to_json())
    if raw != expected_raw:
        _fail("horizon JSON cannot be reproduced from the committed ablation cells")

    markdown = _read_text(horizon_md)
    expected_markdown = expected.to_markdown()
    if ablation.status == "legacy":
        if _LEGACY_HORIZON_MARKER not in markdown.lower():
            _fail("horizon Markdown derived from a legacy report must say 'legacy snapshot'")
        expected_markdown = expected_markdown.replace(
            "# Error compounding over a horizon",
            "# Error compounding over a horizon — legacy snapshot",
            1,
        )
    elif _LEGACY_HORIZON_MARKER in markdown.lower():
        _fail("formal horizon Markdown is still labelled as a legacy snapshot")
    if markdown.strip() != expected_markdown.strip():
        _fail("horizon Markdown was not generated from the committed horizon JSON")

    estimands = raw["estimands"]
    if "mcnemar_p" in estimands["composition"]:
        _fail("horizon composition must not report a McNemar p-value")
    if estimands["composition"]["independent_samples_added"] != 0:
        _fail("horizon composition must add zero independent samples")
    return float(estimands["cell"]["mcnemar_p"])


def _result_section(readme: str) -> str:
    match = re.search(
        r"^## 已提交的实测结果\s*$\n(?P<body>.*?)(?=^## |\Z)",
        readme,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        _fail("README is missing the '已提交的实测结果' section")
    return match.group("body")


def _require_readme_match(section: str, pattern: str, expected: tuple[str, ...], label: str) -> None:
    match = re.search(pattern, section)
    if match is None or match.groups() != expected:
        _fail(f"README {label} claim differs from the committed reports")


def _validate_terminal_evidence(
    evidence_dir: Path,
) -> TerminalBenchPublicEvidenceValidation | None:
    if not evidence_dir.exists() and not evidence_dir.is_symlink():
        return None
    try:
        validation = validate_terminal_bench_public_evidence(evidence_dir)
    except (OSError, ValueError) as exc:
        _fail(f"cannot validate committed Terminal-Bench evidence: {exc}")
    source_identity = (
        validation.evaluated_commit_sha,
        validation.evaluated_tree_sha,
        validation.evaluated_wheel_filename,
        validation.evaluated_wheel_size_bytes,
        validation.evaluated_wheel_sha256,
    )
    if any(value is None for value in source_identity):
        _fail(
            "committed Terminal-Bench evidence must use schema 4 with a complete "
            "source attestation"
        )
    return validation


def _terminal_claim_value(
    claim: str,
    pattern: str,
    *,
    label: str,
) -> str:
    match = re.search(pattern, claim, re.IGNORECASE)
    if match is None:
        _fail(f"README Terminal-Bench claim is missing {label}")
    value = next((group for group in match.groups() if group is not None), None)
    if value is None:
        _fail(f"README Terminal-Bench claim is missing {label}")
    return value


def _validate_terminal_readme(
    readme: str,
    section: str,
    terminal: TerminalBenchPublicEvidenceValidation | None,
) -> None:
    if terminal is None:
        for mention in re.finditer(r"Terminal[- ]Bench", readme, re.IGNORECASE):
            paragraph_end = readme.find("\n\n", mention.start())
            if paragraph_end < 0:
                paragraph_end = len(readme)
            claim_window = readme[mention.start() : paragraph_end]
            if _TERMINAL_NUMERIC_CLAIM.search(claim_window):
                _fail(
                    "README cannot publish a Terminal-Bench numeric result "
                    "without committed public evidence"
                )
        return

    marker = section.find(_TERMINAL_SUBSET_LABEL)
    if marker < 0:
        _fail(
            "README measured-results section must name "
            f"'{_TERMINAL_SUBSET_LABEL}'"
        )
    claim = section[marker : marker + 1600]
    passed = _terminal_claim_value(
        claim,
        r"(?:passed|pass|通过)\s*(?:为|[:：=])?\s*`?(\d+)\s*/\s*20`?"
        r"|`?(\d+)\s*/\s*20`?\s*(?:passed|pass|通过)",
        label="passed/20",
    )
    failed = _terminal_claim_value(
        claim,
        r"(?:failed|fail|失败)\s*(?:为|[:：=])?\s*`?(\d+)`?",
        label="failed",
    )
    errors = _terminal_claim_value(
        claim,
        r"ERROR\s*(?:为|[:：=])?\s*`?(\d+)`?",
        label="ERROR",
    )
    if (
        passed != str(terminal.passed)
        or failed != str(terminal.failed)
        or errors != str(terminal.errors)
    ):
        _fail("README Terminal-Bench counts differ from the committed evidence")

    expected_labels = (
        (
            r"(?:模型|model)\s*(?:为|[:：=])?\s*`?([A-Za-z0-9._/+:-]+)`?",
            terminal.model,
            "model",
        ),
        (
            r"(?:推理强度|reasoning[_ -]?effort)\s*(?:为|[:：=])?\s*"
            r"`?([A-Za-z0-9._+-]+)`?",
            terminal.reasoning_effort,
            "reasoning effort",
        ),
        (
            r"Harbor(?:\s*版本|\s+version)?\s*(?:为|[:：=])?\s*"
            r"`?([A-Za-z0-9.+-]+)`?",
            terminal.harbor_version,
            "Harbor version",
        ),
    )
    for pattern, expected, label in expected_labels:
        actual = _terminal_claim_value(claim, pattern, label=label)
        if actual != expected:
            _fail(
                f"README Terminal-Bench {label} differs from the committed evidence"
            )


def _validate_readme(
    readme_path: Path,
    ablation: _AblationFacts,
    horizon_cell_p: float,
    terminal: TerminalBenchPublicEvidenceValidation | None,
) -> None:
    readme = _read_text(readme_path)
    section = _result_section(readme)
    _validate_terminal_readme(readme, section, terminal)
    pending = bool(
        re.search(r"204.{0,160}(?:未完成|尚未|只有在|后才)", section, re.DOTALL)
    )
    if ablation.status == "legacy":
        if _LEGACY_README_MARKER not in section:
            _fail("README must label historical evidence as '历史报告'")
        if not pending:
            _fail("README must state that the formal 204-cell rerun is still pending")
        if _FORMAL_README_MARKER in section:
            _fail("README cannot label a legacy benchmark as formal")
    else:
        if _FORMAL_README_MARKER not in section:
            _fail("README must label schema-4 evidence as '正式报告'")
        if _LEGACY_README_MARKER in section or pending:
            _fail("README still describes the committed formal benchmark as pending/legacy")
        if ablation.report.model not in section:
            _fail("README formal result section must name the evaluated model")

    _require_readme_match(
        section,
        r"(\d+)\s*个预设 Python 缺陷，每个任务重复\s*(\d+)\s*次，共\s*(\d+)\s*组",
        (
            str(len(ablation.report.tasks)),
            str(ablation.report.reps),
            str(ablation.cells),
        ),
        "task/repetition/cell",
    )
    _require_readme_match(
        section,
        r"`trust`[^|\n]*\|[^|\n]*\|\s*(\d+)\s*个正确，\s*(\d+)\s*个错误仍被接受",
        (str(ablation.trust_successes), str(ablation.trust_false_successes)),
        "trust outcome",
    )
    _require_readme_match(
        section,
        r"`gate`[^|\n]*\|[^|\n]*\|\s*接受\s*(\d+)\s*个正确补丁，"
        r"\s*拦截\s*(\d+)\s*个错误补丁",
        (str(ablation.gate_successes), str(ablation.gate_interceptions)),
        "gate outcome",
    )
    _require_readme_match(
        section,
        r"`verify`[^|\n]*\|[^|\n]*\|\s*(\d+)/(\d+)\s*通过独立评分",
        (str(ablation.verify_successes), str(ablation.cells)),
        "verify outcome",
    )

    published_p = re.findall(r"\bp\s*=\s*([0-9]+(?:\.[0-9]+)?)", section)
    for expected in {ablation.trust_gate_p, horizon_cell_p}:
        if not any(
            len(value.partition(".")[2]) >= 2
            and
            math.isclose(
                float(value),
                expected,
                rel_tol=1e-9,
                abs_tol=0.5 * 10 ** -len(value.partition(".")[2]) + 1e-12,
            )
            for value in published_p
        ):
            _fail(f"README is missing the measured McNemar p-value {expected:.4f}")
    if not re.search(r"(?:不增加|没有增加).{0,12}(?:样本|观测)", section):
        _fail("README must state that horizon composition adds no samples")


def validate_release_claims(root: str | Path = ".") -> ReleaseClaimsSummary:
    """Validate the committed README and benchmark report set.

    ``root`` is the repository root.  The function is reusable from tests, CI,
    and release tooling; it performs no writes.
    """
    repo = Path(root).resolve()
    benchmarks = repo / "benchmarks"
    ablation = _validate_ablation(
        benchmarks / "ablation_report.json",
        benchmarks / "ablation_report.md",
    )
    horizon_p = _validate_horizon(
        benchmarks / "horizon_report.json",
        benchmarks / "horizon_report.md",
        benchmarks / "ablation_report.json",
        ablation,
    )
    terminal = _validate_terminal_evidence(
        benchmarks / _TERMINAL_EVIDENCE_DIR
    )
    _validate_readme(repo / "README.md", ablation, horizon_p, terminal)
    return ReleaseClaimsSummary(
        status=ablation.status,
        tasks=len(ablation.report.tasks),
        repetitions=ablation.report.reps,
        cells=ablation.cells,
        model=ablation.report.model,
        trust_successes=ablation.trust_successes,
        trust_false_successes=ablation.trust_false_successes,
        gate_successes=ablation.gate_successes,
        gate_interceptions=ablation.gate_interceptions,
        verify_successes=ablation.verify_successes,
        headline_mcnemar_p=horizon_p,
        terminal_bench=terminal,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="check README benchmark claims against committed JSON/Markdown reports"
    )
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    args = parser.parse_args(argv)
    try:
        summary = validate_release_claims(args.root)
    except ReleaseClaimsError as exc:
        print(f"release claims: FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "release claims: ok "
        f"({summary.status}; tasks={summary.tasks}; reps={summary.repetitions}; "
        f"cells={summary.cells}; model={summary.model})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
