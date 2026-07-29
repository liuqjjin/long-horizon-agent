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
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from .ablation import (
    _CACHE_SCHEMA,
    _CELL_ATTEMPT_SCHEMA,
    _FORMAL_CORPUS_MANIFEST_PATH,
    _FORMAL_OUTPUT_LOCK_NAME,
    _FORMAL_REPETITIONS,
    _FORMAL_RUN_HEADER_NAME,
    _FORMAL_RUN_HEADER_SCHEMA,
    _FORMAL_TASK_COUNT,
    _LLM_CALL_RECEIPT_SCHEMA,
    _LLM_RETRIES,
    _MAX_ARTIFACT_BYTES,
    _MAX_CACHE_BYTES,
    _MAX_CELL_ATTEMPT_BYTES,
    _MAX_FORMAL_MANIFEST_BYTES,
    _MAX_FORMAL_RUN_HEADER_BYTES,
    _MAX_REPAIRS,
    _MAX_REPORT_BYTES,
    _MAX_SCORER_EVIDENCE_BYTES,
    _MAX_TASK_BYTES,
    CONDITIONS,
    AblationReport,
    RunRecord,
    ScoreOutcome,
    ScorerEvidenceBinding,
    _ablation_report_from_raw,
    _aggregate,
    _canonical_json_object_bytes,
    _frozen_artifact_bytes,
    _git_control_env,
    _input_snapshot_digest,
    _load_formal_corpus_manifest,
    _read_bounded_bytes,
    _read_bounded_text,
    _read_llm_call_receipt,
    _record_from_raw,
    _repo_digest,
    _report_fingerprint,
    _scorer_evidence_bytes,
    _scorer_runtime_digest,
    _source_file_digests,
    _source_tree_digest,
    _trusted_control_executable,
    _validate_cell_call_sequence,
    _validate_scorer_evidence,
)
from .ablation_attempts import (
    FORMAL_ABLATION_ATTEMPTS_PATH,
    MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
    CompletedAttempt,
    FormalAblationProtocol,
    FormalCodexClientConfig,
    RegisteredAttempt,
    UnregisteredRunRecorded,
    formal_ablation_protocol_sha256,
    formal_ablation_witness_commit_bytes,
    formal_ablation_witness_commit_oid,
    formal_ablation_witness_message,
    formal_codex_client_sha256,
    parse_formal_ablation_attempt_registry,
    registry_has_prefix,
)
from .bench.stats import (
    mcnemar_exact,
    paired_cluster_sign_flip_exact,
    wilson_interval,
)
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
    scheduled_cells: int
    usable_cells: int
    error_cells: int
    # Compatibility alias for callers that treated ``cells`` as the number of
    # measured rate observations. It is deliberately the usable count, not the
    # scheduled denominator.
    cells: int
    model: str
    trust_successes: int
    trust_false_successes: int
    gate_successes: int
    gate_interceptions: int
    verify_successes: int
    # Compatibility name. For schema-v4 this is the task-cluster exact paired
    # sign-flip p-value; historical schemas retain their cell McNemar value.
    headline_mcnemar_p: float
    terminal_bench: TerminalBenchPublicEvidenceValidation | None


@dataclass(frozen=True)
class _AblationFacts:
    report: AblationReport
    raw: dict[str, Any]
    records: tuple[RunRecord, ...]
    status: str
    scheduled_cells: int
    usable_cells: int
    error_cells: int
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
        value = json.loads(_read_bounded_text(path, max_bytes=_MAX_REPORT_BYTES))
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
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isclose(float(left), right, rel_tol=1e-12, abs_tol=1e-12)
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
            _record_bool(record_raw, "artifact_correct") if schema_version >= 4 else true_success
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
                else (ScoreOutcome.PASS.value if artifact_correct else ScoreOutcome.TEST_FAIL.value)
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


def _validate_formal_error_cells(records: tuple[RunRecord, ...]) -> set[tuple[str, int]]:
    """Return terminal ERROR cells after validating their empty measurement envelope.

    A formal ERROR is a cell-level infrastructure outcome, not three unrelated
    condition results. All conditions therefore carry the same absence of a
    delivery, artifact, and scorer measurement. The call receipts are checked
    separately because they are content-addressed files rather than record
    fields.
    """
    by_cell: dict[tuple[str, int], list[RunRecord]] = {}
    for record in records:
        by_cell.setdefault((record.task, record.rep), []).append(record)

    error_cells: set[tuple[str, int]] = set()
    for cell, cell_records in by_cell.items():
        if not any(record.status == "ERROR" for record in cell_records):
            continue
        if (
            len(cell_records) != len(_CONDITION_NAMES)
            or {record.condition for record in cell_records} != set(_CONDITION_NAMES)
            or any(record.status != "ERROR" for record in cell_records)
        ):
            _fail(
                "formal ablation ERROR must cover trust, gate, and verify "
                f"consistently for {cell!r}"
            )
        if len({record.repairs for record in cell_records}) != 1 or len(
            {record.detail for record in cell_records}
        ) != 1:
            _fail(f"formal ablation ERROR records disagree within cell {cell!r}")
        for record in cell_records:
            key = (record.task, record.condition, record.rep)
            if (
                record.scorer_outcome != ScoreOutcome.INFRA_ERROR.value
                or record.claimed_success
                or record.artifact_correct
                or record.true_success
                or record.false_success
                or record.gate_prediction is not None
                or record.artifact_sha256
                or record.scorer_evidence_sha256
                or record.scorer_expected_tests != 0
                or record.scorer_passed_tests != 0
                or record.repairs != 0
            ):
                _fail(f"formal ablation ERROR record is not empty and typed for {key!r}")
        error_cells.add(cell)
    return error_cells


def _validate_record_artifacts(records: tuple[RunRecord, ...]) -> None:
    """Bind each formal cell to a valid artifact and preserve the paired design."""
    by_cell: dict[tuple[str, int], dict[str, str]] = {}
    evidence_by_cell: dict[tuple[str, int], dict[str, str]] = {}
    for record in records:
        if record.status == "ERROR":
            continue
        if not _HEX_64.fullmatch(record.artifact_sha256):
            _fail(
                "formal ablation record has an invalid artifact_sha256 "
                f"for {(record.task, record.condition, record.rep)!r}"
            )
        by_cell.setdefault((record.task, record.rep), {})[record.condition] = record.artifact_sha256
        evidence_by_cell.setdefault((record.task, record.rep), {})[record.condition] = (
            record.scorer_evidence_sha256
        )
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
    expected_digests = {
        record.artifact_sha256 for record in records if record.status != "ERROR"
    }
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
        record.scorer_evidence_sha256 for record in records if record.status != "ERROR"
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
    if not isinstance(snapshots, dict) or not isinstance(scorer_backend, str) or not scorer_backend:
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
            f"{stat.get('condition')} {interval_name} must use Wilson {expected}, got {interval!r}"
        )
    return None


def _derived_stat_value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, float):
        return _is_close(actual, expected)
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, str):
        return actual == expected
    if isinstance(expected, tuple):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(
                _derived_stat_value_matches(observed, wanted)
                for observed, wanted in zip(actual, expected)
            )
        )
    return actual == expected


def _validate_current_derived_stats(
    by_name: dict[str, dict[str, Any]],
    records: tuple[RunRecord, ...],
) -> None:
    """Recompute every schema-4 ConditionStats field from immutable cell rows."""
    expected_by_name = {
        stat.condition: asdict(stat) for stat in _aggregate(list(records))
    }
    for condition in _CONDITION_NAMES:
        observed = by_name[condition]
        expected = expected_by_name[condition]
        if set(observed) != set(expected):
            _fail(
                f"ablation stats for {condition!r} do not contain exactly "
                "the generated ConditionStats fields"
            )
        for field, wanted in expected.items():
            if not _derived_stat_value_matches(observed.get(field), wanted):
                _fail(
                    f"ablation stats for {condition!r} have stale {field}: "
                    f"expected {wanted!r}, got {observed.get(field)!r}"
                )


def _validate_condition_stats(raw: dict[str, Any], records: tuple[RunRecord, ...]) -> list[str]:
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

    tasks = raw.get("tasks")
    reps = raw.get("reps")
    if not isinstance(tasks, list) or not isinstance(reps, int) or isinstance(reps, bool):
        _fail("ablation schedule is invalid while validating condition stats")
    scheduled_per_condition = len(tasks) * reps

    boundary_problems: list[str] = []
    for condition in _CONDITION_NAMES:
        condition_records = [record for record in records if record.condition == condition]
        usable = [record for record in condition_records if record.status != "ERROR"]
        stat = by_name[condition]
        errors = len(condition_records) - len(usable)
        n = len(usable)
        if stat.get("n") != n or stat.get("errors") != errors:
            _fail(f"ablation stats for {condition!r} have stale n/errors")
        if n + errors != scheduled_per_condition:
            _fail(
                f"ablation stats for {condition!r} do not retain the scheduled denominator"
            )
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
                    not record.gate_prediction and record.artifact_correct for record in predictions
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
            problem = _boundary_interval_problem(stat, field=field, successes=successes, total=n)
            if problem:
                boundary_problems.append(problem)
    if current_schema and not boundary_problems:
        _validate_current_derived_stats(by_name, records)
    return boundary_problems


def _paired_p(
    records: tuple[RunRecord, ...],
    left: str,
    right: str,
    metric: str,
    *,
    task_cluster_inference: bool = False,
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
    if task_cluster_inference:
        pairs_by_task: dict[str, list[tuple[bool, bool]]] = {}
        for task, rep in sorted(pairs):
            pairs_by_task.setdefault(task, []).append(
                (
                    left_outcomes[(task, rep)],
                    right_outcomes[(task, rep)],
                )
            )
        return paired_cluster_sign_flip_exact(pairs_by_task).p_value
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


def _git_bytes(
    repository_root: Path,
    arguments: list[str],
    *,
    git_executable: str,
    timeout: float = 15,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [git_executable, *arguments],
            cwd=repository_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=_git_control_env(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        _fail(f"cannot inspect formal ablation Git evidence: {type(error).__name__}")


def _git_success(
    repository_root: Path,
    arguments: list[str],
    *,
    git_executable: str,
    label: str,
) -> bytes:
    result = _git_bytes(
        repository_root,
        arguments,
        git_executable=git_executable,
    )
    if result.returncode != 0:
        _fail(f"formal ablation Git evidence failed {label}")
    return result.stdout


def _git_blob_bytes(
    repository_root: Path,
    *,
    commit: str,
    path: str,
    git_executable: str,
    max_bytes: int,
    label: str,
) -> bytes:
    object_name = f"{commit}:{path}"
    size_raw = _git_success(
        repository_root,
        ["cat-file", "-s", object_name],
        git_executable=git_executable,
        label=f"{label} size",
    )
    try:
        size = int(size_raw.decode("ascii", errors="strict").strip())
    except (UnicodeDecodeError, ValueError) as error:
        _fail(f"formal ablation Git evidence has an invalid {label} size: {error}")
    if size < 0 or size > max_bytes:
        _fail(f"formal ablation Git evidence {label} is too large")
    payload = _git_success(
        repository_root,
        ["show", object_name],
        git_executable=git_executable,
        label=label,
    )
    if len(payload) != size:
        _fail(f"formal ablation Git evidence {label} changed while reading")
    return payload


def _disclosed_report_counts(
    raw: dict[str, Any],
) -> tuple[dict[str, int], tuple[dict[str, Any], ...]]:
    tasks = raw.get("tasks")
    reps = raw.get("reps")
    if (
        raw.get("schema_version") != 4
        or not isinstance(tasks, list)
        or len(tasks) != _FORMAL_TASK_COUNT
        or len(set(tasks)) != len(tasks)
        or not all(isinstance(task, str) and task for task in tasks)
        or type(reps) is not int
        or reps != _FORMAL_REPETITIONS
    ):
        _fail("disclosed formal ablation report is not the fixed schema-4 grid")
    records_raw = raw.get("records")
    if not isinstance(records_raw, list):
        _fail("disclosed formal ablation report has no record grid")
    records: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    boolean_fields = (
        "claimed_success",
        "artifact_correct",
        "true_success",
        "false_success",
    )
    for value in records_raw:
        if not isinstance(value, dict):
            _fail("disclosed formal ablation report has a non-object record")
        task = value.get("task")
        rep = value.get("rep")
        condition = value.get("condition")
        status = value.get("status")
        key = (task, rep, condition)
        if (
            not isinstance(task, str)
            or task not in tasks
            or type(rep) is not int
            or rep < 0
            or rep >= reps
            or condition not in _CONDITION_NAMES
            or not isinstance(status, str)
            or not status
            or any(type(value.get(field)) is not bool for field in boolean_fields)
            or key in by_key
        ):
            _fail("disclosed formal ablation report has an invalid record key")
        claimed = cast(bool, value["claimed_success"])
        correct = cast(bool, value["artifact_correct"])
        if (
            value["true_success"] != (claimed and correct)
            or value["false_success"] != (claimed and not correct)
            or (
                status == "ERROR"
                and any(cast(bool, value[field]) for field in boolean_fields)
            )
        ):
            _fail("disclosed formal ablation report has inconsistent outcomes")
        by_key[cast(tuple[str, int, str], key)] = value
        records.append(value)
    expected_keys = {
        (task, rep, condition)
        for task in tasks
        for rep in range(reps)
        for condition in _CONDITION_NAMES
    }
    if set(by_key) != expected_keys:
        _fail("disclosed formal ablation report does not cover its fixed grid")
    error_keys: set[tuple[str, int]] = set()
    for task in cast(list[str], tasks):
        for rep in range(reps):
            statuses = {
                by_key[(task, rep, condition)]["status"]
                for condition in _CONDITION_NAMES
            }
            contains_error = "ERROR" in statuses
            if contains_error and statuses != {"ERROR"}:
                _fail(
                    "disclosed formal ablation report has a condition-local ERROR"
                )
            if contains_error:
                error_keys.add((task, rep))
    usable = [
        record
        for record in records
        if (cast(str, record["task"]), cast(int, record["rep"])) not in error_keys
    ]
    by_condition = {
        condition: [
            record for record in usable if record["condition"] == condition
        ]
        for condition in _CONDITION_NAMES
    }
    trust = by_condition["trust"]
    gate = by_condition["gate"]
    verify = by_condition["verify"]
    counts = {
        "scheduled_cells": len(tasks) * reps,
        "usable_cells": len(trust),
        "error_cells": len(error_keys),
        "trust_delivered_correct": sum(record["true_success"] for record in trust),
        "trust_delivered_wrong": sum(record["false_success"] for record in trust),
        "gate_delivered_correct": sum(record["true_success"] for record in gate),
        "gate_delivered_wrong": sum(record["false_success"] for record in gate),
        "gate_intercepted_wrong": sum(
            not record["claimed_success"] and not record["artifact_correct"]
            for record in gate
        ),
        "gate_rejected_correct": sum(
            not record["claimed_success"] and record["artifact_correct"]
            for record in gate
        ),
        "verify_delivered_correct": sum(record["true_success"] for record in verify),
        "verify_delivered_wrong": sum(record["false_success"] for record in verify),
        "verify_not_delivered": sum(
            not record["claimed_success"] for record in verify
        ),
    }
    return counts, tuple(records)


def _formal_codex_client_from_provenance(
    provenance: dict[str, Any],
    *,
    label: str,
) -> FormalCodexClientConfig:
    configuration = provenance.get("configuration")
    client = (
        configuration.get("client")
        if isinstance(configuration, dict)
        else None
    )
    if not isinstance(client, dict):
        _fail(f"{label} has no Codex client configuration")
    fixed_values = {
        "no_tools": True,
        "sandbox_mode": "read-only",
        "permission_model": "profile",
        "permission_profile": "lha-read",
        "credential_barrier": "verified",
        "externally_sandboxed": False,
    }
    if any(
        client.get(field) != expected
        for field, expected in fixed_values.items()
    ):
        _fail(f"{label} has an invalid fixed Codex permission boundary")
    timeout = client.get("timeout")
    retry_backoff = client.get("retry_backoff_s")
    max_retries = client.get("max_retries")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or isinstance(retry_backoff, bool)
        or not isinstance(retry_backoff, (int, float))
        or type(max_retries) is not int
        or max_retries < 0
    ):
        _fail(f"{label} has an invalid Codex timeout or retry budget")
    try:
        return FormalCodexClientConfig(
            no_tools=True,
            sandbox_mode="read-only",
            permission_model="profile",
            permission_profile="lha-read",
            credential_barrier="verified",
            externally_sandboxed=False,
            max_retries=cast(int, max_retries),
            timeout_s=float(timeout),
            retry_backoff_s=float(retry_backoff),
        )
    except ValueError as error:
        _fail(f"{label} has an invalid fixed Codex client configuration: {error}")


def _codex_executable_sha256_from_provenance(
    provenance: dict[str, Any],
    *,
    label: str,
    allow_backend_details: bool,
) -> str:
    recorded = provenance.get("cli_executable_sha256")
    if isinstance(recorded, str) and _HEX_64.fullmatch(recorded):
        return recorded
    if allow_backend_details:
        details = provenance.get("backend_details")
        if isinstance(details, str):
            matches = re.findall(
                r"(?:^|\s)cli_sha256=([0-9a-f]{64})(?=\s|$)",
                details,
            )
            if len(matches) == 1:
                return matches[0]
    _fail(f"{label} has no unambiguous Codex executable digest")


def _validate_disclosed_formal_report(
    disclosure: UnregisteredRunRecorded,
    repo_root: Path,
    *,
    git_executable: str,
    head: str,
) -> None:
    _git_success(
        repo_root,
        ["cat-file", "-e", f"{disclosure.source_commit}^{{commit}}"],
        git_executable=git_executable,
        label="disclosed formal source commit",
    )
    report_relative = disclosure.published_report_path
    report_path = _repo_evidence_path(
        repo_root,
        report_relative,
        label="disclosed formal report",
    )
    try:
        current_bytes = _read_bounded_bytes(
            report_path,
            max_bytes=_MAX_REPORT_BYTES,
        )
    except (OSError, ValueError) as error:
        _fail(f"disclosed formal ablation report is unavailable: {error}")
    committed_bytes = _git_blob_bytes(
        repo_root,
        commit=head,
        path=report_relative,
        git_executable=git_executable,
        max_bytes=_MAX_REPORT_BYTES,
        label="disclosed formal report",
    )
    if (
        current_bytes != committed_bytes
        or hashlib.sha256(current_bytes).hexdigest()
        != disclosure.report_sha256
    ):
        _fail("disclosed formal ablation report digest is stale")
    try:
        raw = json.loads(current_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"disclosed formal ablation report is invalid JSON: {error}")
    if not isinstance(raw, dict):
        _fail("disclosed formal ablation report is not an object")
    if (
        raw.get("fingerprint") != disclosure.report_fingerprint
        or raw.get("fingerprint") != _report_fingerprint(raw)
    ):
        _fail("disclosed formal ablation report fingerprint is stale")
    provenance = raw.get("provenance")
    if (
        raw.get("model") != disclosure.model
        or not isinstance(provenance, dict)
    ):
        _fail("disclosed formal ablation report differs from its recorded protocol")
    provenance = cast(dict[str, Any], provenance)
    cli_version = provenance.get("cli_version")
    if not isinstance(cli_version, str) or not cli_version:
        _fail("disclosed formal ablation report has no Codex CLI version")
    client = _formal_codex_client_from_provenance(
        provenance,
        label="disclosed formal ablation report",
    )
    try:
        protocol = FormalAblationProtocol(
            schema_version=1,
            source_commit=cast(str, provenance.get("git_commit")),
            source_tree_sha256=cast(str, provenance.get("source_tree_sha256")),
            manifest_sha256=cast(
                str,
                provenance.get("formal_corpus_manifest_sha256"),
            ),
            model=cast(str, provenance.get("model")),
            reasoning_effort=cast(str, provenance.get("reasoning_effort")),
            docker_image_id=cast(str, provenance.get("scorer_image_id")),
            codex_cli_version=cli_version,
            codex_cli_executable_sha256=(
                _codex_executable_sha256_from_provenance(
                    provenance,
                    label="disclosed formal ablation report",
                    allow_backend_details=True,
                )
            ),
            codex_client=client,
            codex_client_sha256=formal_codex_client_sha256(client),
        )
    except ValueError as error:
        _fail(f"disclosed formal ablation protocol is invalid: {error}")
    if (
        protocol != disclosure.protocol()
        or disclosure.protocol_sha256
        != formal_ablation_protocol_sha256(protocol)
    ):
        _fail("disclosed formal ablation report differs from its recorded protocol")
    measured_counts, _records = _disclosed_report_counts(raw)
    expected_counts = {
        field: getattr(disclosure, field) for field in measured_counts
    }
    if measured_counts != expected_counts:
        _fail("disclosed formal ablation report counts are stale")


def _validate_formal_ablation_disclosures(repo_root: Path) -> None:
    registry_path = repo_root / FORMAL_ABLATION_ATTEMPTS_PATH
    history_root = repo_root / "benchmarks" / "formal_ablation_history"
    registry_present = registry_path.exists() or registry_path.is_symlink()
    history_present = history_root.exists() or history_root.is_symlink()
    if not registry_present and not history_present:
        return
    git_executable = str(_trusted_control_executable("git")["path"])
    head = (
        _git_success(
            repo_root,
            ["rev-parse", "--verify", "HEAD"],
            git_executable=git_executable,
            label="disclosure HEAD resolution",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    history_lines = (
        _git_success(
            repo_root,
            [
                "ls-tree",
                "-r",
                "--name-only",
                head,
                "--",
                "benchmarks/formal_ablation_history",
            ],
            git_executable=git_executable,
            label="formal ablation history paths",
        )
        .decode("utf-8", errors="strict")
        .splitlines()
    )
    tracked_reports = {
        path
        for path in history_lines
        if path.endswith("/ablation_report.json")
    }
    if not registry_present:
        if tracked_reports:
            _fail(
                "tracked formal ablation history reports have no registry disclosures"
            )
        return
    current_path = _repo_evidence_path(
        repo_root,
        FORMAL_ABLATION_ATTEMPTS_PATH.as_posix(),
        label="attempt registry",
    )
    try:
        current_bytes = _read_bounded_bytes(
            current_path,
            max_bytes=MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
        )
    except (OSError, ValueError) as error:
        _fail(f"formal ablation attempt registry is unsafe: {error}")
    committed_bytes = _git_blob_bytes(
        repo_root,
        commit=head,
        path=FORMAL_ABLATION_ATTEMPTS_PATH.as_posix(),
        git_executable=git_executable,
        max_bytes=MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
        label="current attempt registry",
    )
    if current_bytes != committed_bytes:
        _fail("formal ablation current attempt registry differs from HEAD")
    try:
        registry = parse_formal_ablation_attempt_registry(current_bytes)
    except ValueError as error:
        _fail(f"formal ablation current attempt registry is invalid: {error}")
    disclosed_reports = {
        disclosure.published_report_path
        for disclosure in registry.disclosures()
    }
    if tracked_reports != disclosed_reports:
        _fail(
            "tracked formal ablation history reports and registry disclosures differ"
        )
    for disclosure in registry.disclosures():
        _validate_disclosed_formal_report(
            disclosure,
            repo_root,
            git_executable=git_executable,
            head=head,
        )


def _validate_formal_attempt_provenance(
    raw: dict[str, Any],
    repo_root: Path,
    *,
    git_executable: str,
    head: str,
    require_completion: bool = True,
) -> None:
    provenance = cast(dict[str, Any], raw["provenance"])
    attempt_id = cast(str, provenance["formal_attempt_id"])
    registry_relative = cast(str, provenance["formal_attempt_registry_path"])
    registration_sha256 = cast(
        str,
        provenance["formal_attempt_registry_sha256"],
    )
    protocol_sha256 = cast(str, provenance["formal_attempt_protocol_sha256"])
    registration_commit = cast(
        str,
        provenance["formal_attempt_registration_commit"],
    )

    registration_bytes = _git_blob_bytes(
        repo_root,
        commit=registration_commit,
        path=registry_relative,
        git_executable=git_executable,
        max_bytes=MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
        label="attempt registry at registration",
    )
    if hashlib.sha256(registration_bytes).hexdigest() != registration_sha256:
        _fail("formal ablation registration registry digest is stale")
    try:
        registration_registry = parse_formal_ablation_attempt_registry(
            registration_bytes
        )
    except ValueError as error:
        _fail(f"formal ablation registration registry is invalid: {error}")
    registration = registration_registry.open_registration()
    if (
        not isinstance(registration, RegisteredAttempt)
        or registration.attempt_id != attempt_id
    ):
        _fail("formal ablation attempt was not open in the registration commit")

    parents_raw = _git_success(
        repo_root,
        ["rev-list", "--parents", "-n", "1", registration_commit],
        git_executable=git_executable,
        label="attempt registration parents",
    )
    parents = parents_raw.decode("ascii", errors="strict").strip().split()
    if parents != [registration_commit, registration.source_commit]:
        _fail("formal ablation registration commit does not directly follow its source")
    changed_paths = (
        _git_success(
            repo_root,
            [
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                registration_commit,
            ],
            git_executable=git_executable,
            label="attempt registration changes",
        )
        .decode("utf-8", errors="strict")
        .splitlines()
    )
    if changed_paths != [registry_relative]:
        _fail("formal ablation registration commit changed files other than the registry")

    effort = provenance.get("reasoning_effort")
    if not isinstance(effort, str) or not effort:
        _fail("formal ablation attempt registration has no reasoning effort")
    cli_version = provenance.get("cli_version")
    if not isinstance(cli_version, str) or not cli_version:
        _fail("formal ablation attempt registration has no Codex CLI version")
    codex_client = _formal_codex_client_from_provenance(
        provenance,
        label="formal ablation report",
    )
    try:
        scorer_runtime_sha256 = (
            _scorer_runtime_digest(
                repo_root,
                git_path=git_executable,
                commit=registration.source_commit,
            )
            if registration.scorer_runtime_sha256 is not None
            else None
        )
    except RuntimeError as error:
        _fail(f"formal ablation scorer runtime is invalid: {error}")
    try:
        protocol = FormalAblationProtocol(
            schema_version=(
                2 if registration.scorer_runtime_sha256 is not None else 1
            ),
            source_commit=registration.source_commit,
            source_tree_sha256=cast(str, provenance["source_tree_sha256"]),
            manifest_sha256=cast(
                str,
                provenance["formal_corpus_manifest_sha256"],
            ),
            model=cast(str, provenance["model"]),
            reasoning_effort=effort,
            docker_image_id=cast(str, provenance["scorer_image_id"]),
            scorer_runtime_sha256=scorer_runtime_sha256,
            codex_cli_version=cli_version,
            codex_cli_executable_sha256=(
                _codex_executable_sha256_from_provenance(
                    provenance,
                    label="formal ablation report",
                    allow_backend_details=False,
                )
            ),
            codex_client=codex_client,
            codex_client_sha256=formal_codex_client_sha256(codex_client),
            witness_credential_helper=registration.witness_credential_helper,
        )
    except ValueError as error:
        _fail(f"formal ablation attempt protocol is invalid: {error}")
    measured_protocol_sha256 = formal_ablation_protocol_sha256(protocol)
    expected_registration = {
        "source_tree_sha256": protocol.source_tree_sha256,
        "manifest_sha256": protocol.manifest_sha256,
        "model": protocol.model,
        "reasoning_effort": protocol.reasoning_effort,
        "docker_image_id": protocol.docker_image_id,
        "scorer_runtime_sha256": protocol.scorer_runtime_sha256,
        "codex_cli_version": protocol.codex_cli_version,
        "codex_cli_executable_sha256": protocol.codex_cli_executable_sha256,
        "codex_client": protocol.codex_client,
        "codex_client_sha256": protocol.codex_client_sha256,
        "output_path": f"runs/formal_ablation/{attempt_id}",
        "protocol_sha256": measured_protocol_sha256,
    }
    if (
        protocol_sha256 != measured_protocol_sha256
        or provenance.get("scorer_runtime_sha256")
        != registration.scorer_runtime_sha256
        or any(
            getattr(registration, field) != expected
            for field, expected in expected_registration.items()
        )
    ):
        _fail("formal ablation attempt registration differs from report provenance")
    _validate_formal_start_witness(
        registration,
        provenance,
        repo_root,
        git_executable=git_executable,
        registration_commit=registration_commit,
    )

    current_path = _repo_evidence_path(
        repo_root,
        registry_relative,
        label="attempt registry",
    )
    try:
        current_bytes = _read_bounded_bytes(
            current_path,
            max_bytes=MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
        )
    except (OSError, ValueError) as error:
        _fail(f"formal ablation current attempt registry is unsafe: {error}")
    head_bytes = _git_blob_bytes(
        repo_root,
        commit=head,
        path=registry_relative,
        git_executable=git_executable,
        max_bytes=MAX_FORMAL_ABLATION_ATTEMPTS_BYTES,
        label="current attempt registry",
    )
    if current_bytes != head_bytes:
        _fail("formal ablation current attempt registry differs from HEAD")
    try:
        current_registry = parse_formal_ablation_attempt_registry(current_bytes)
    except ValueError as error:
        _fail(f"formal ablation current attempt registry is invalid: {error}")
    if not registry_has_prefix(current_registry, registration_registry):
        _fail("formal ablation current attempt registry rewrites historical events")
    if not require_completion:
        if (
            current_bytes != registration_bytes
            or current_registry.open_registration() != registration
        ):
            _fail(
                "formal ablation output does not match the still-open "
                "registration at HEAD"
            )
        return
    if current_registry.open_registration() is not None:
        _fail("formal ablation release has an open attempt")

    completions = [
        event
        for event in current_registry.completions()
        if event.attempt_id == attempt_id
    ]
    if len(completions) != 1:
        _fail("formal ablation report does not have one matching COMPLETED event")
    completion = completions[0]
    if not isinstance(completion, CompletedAttempt):
        _fail("formal ablation completion event is invalid")
    report_path = repo_root / "benchmarks" / "ablation_report.json"
    try:
        report_bytes = _read_bounded_bytes(
            report_path,
            max_bytes=_MAX_REPORT_BYTES,
        )
    except (OSError, ValueError) as error:
        _fail(f"formal ablation report bytes are unavailable: {error}")
    if (
        completion.protocol_sha256 != protocol_sha256
        or completion.registration_registry_sha256 != registration_sha256
        or completion.report_sha256 != hashlib.sha256(report_bytes).hexdigest()
        or completion.report_fingerprint != raw.get("fingerprint")
    ):
        _fail("formal ablation COMPLETED event differs from the published report")


def _validate_formal_start_witness(
    registration: RegisteredAttempt,
    provenance: dict[str, Any],
    repo_root: Path,
    *,
    git_executable: str,
    registration_commit: str,
) -> None:
    """Verify the external, create-only ref that consumed this registration."""
    witness_ref = provenance.get("formal_attempt_witness_ref")
    witness_commit = provenance.get("formal_attempt_witness_commit")
    expected_fields = {
        "formal_attempt_witness_remote_name": registration.witness_remote_name,
        "formal_attempt_witness_remote_url": registration.witness_remote_url,
        "formal_attempt_witness_ref": registration.witness_ref,
    }
    if any(provenance.get(field) != value for field, value in expected_fields.items()):
        _fail("formal ablation witness provenance differs from its registration")
    if (
        not isinstance(witness_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", witness_commit) is None
        or not isinstance(witness_ref, str)
    ):
        _fail("formal ablation witness provenance is invalid")

    tree = (
        _git_success(
            repo_root,
            ["rev-parse", f"{registration_commit}^{{tree}}"],
            git_executable=git_executable,
            label="formal ablation witness tree",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    try:
        message = formal_ablation_witness_message(
            attempt_id=registration.attempt_id,
            registration_registry_sha256=cast(
                str,
                provenance["formal_attempt_registry_sha256"],
            ),
            protocol_sha256=registration.protocol_sha256,
            outcome_key=cast(str, provenance["formal_outcome_key"]),
            run_header_sha256=cast(
                str,
                provenance["formal_run_header_sha256"],
            ),
        )
        expected_commit_bytes = formal_ablation_witness_commit_bytes(
            tree=tree,
            parent=registration_commit,
            message=message,
        )
    except (KeyError, ValueError) as error:
        _fail(f"formal ablation witness binding is invalid: {error}")
    if formal_ablation_witness_commit_oid(expected_commit_bytes) != witness_commit:
        _fail("formal ablation witness commit does not bind the recorded run")
    witness_bytes = _git_success(
        repo_root,
        ["cat-file", "commit", witness_commit],
        git_executable=git_executable,
        label="formal ablation witness commit",
    )
    if witness_bytes != expected_commit_bytes:
        _fail(
            "formal ablation witness commit has the wrong tree, parent, or message"
        )

    remote_ref = (
        _git_success(
            repo_root,
            [
                "ls-remote",
                "--refs",
                registration.witness_remote_url,
                registration.witness_ref,
            ],
            git_executable=git_executable,
            label="formal ablation witness remote ref",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    if remote_ref != f"{witness_commit}\t{registration.witness_ref}":
        _fail("formal ablation witness remote ref is missing or changed")


def _recorded_control_executable(
    provenance: dict[str, Any],
    field: str,
    *,
    require_trusted_install: bool,
) -> dict[str, Any]:
    raw = provenance.get(field)
    required = {"path", "sha256", "size_bytes", "trusted_install"}
    expected_name = {
        "git_executable": "git",
        "docker_executable": "docker",
    }.get(field)
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or not isinstance(raw.get("path"), str)
        or not Path(raw["path"]).is_absolute()
        or (expected_name is not None and Path(raw["path"]).name != expected_name)
        or not isinstance(raw.get("sha256"), str)
        or _HEX_64.fullmatch(raw["sha256"]) is None
        or not isinstance(raw.get("size_bytes"), int)
        or isinstance(raw.get("size_bytes"), bool)
        or raw["size_bytes"] <= 0
        or type(raw.get("trusted_install")) is not bool
        or (require_trusted_install and raw["trusted_install"] is not True)
    ):
        _fail(f"formal ablation provenance has an invalid {field} identity")
    return cast(dict[str, Any], raw)


def _validate_git_executable_provenance(provenance: dict[str, Any]) -> str:
    """Validate historical identity, then choose this host's trusted Git for checks."""
    _recorded_control_executable(
        provenance,
        "git_executable",
        require_trusted_install=True,
    )
    try:
        current = _trusted_control_executable("git")
    except (OSError, RuntimeError, ValueError) as error:
        _fail(
            "a trusted local Git executable is required to validate commit evidence: "
            f"{type(error).__name__}"
        )
    return cast(str, current["path"])


def _validate_docker_executable_provenance(
    provenance: dict[str, Any],
) -> None:
    """Validate portable run provenance without requiring Docker on the reader host."""
    _recorded_control_executable(
        provenance,
        "docker_executable",
        require_trusted_install=False,
    )


def _validate_docker_image_probe(
    provenance: dict[str, Any],
    configuration: dict[str, Any],
) -> None:
    probe = configuration.get("docker_image_probe")
    required = {
        "schema_version",
        "image_id",
        "network",
        "minimal_pytest",
        "python_version",
        "pytest_version",
        "pytest_json_report_version",
    }
    if (
        not isinstance(probe, dict)
        or set(probe) != required
        or probe.get("schema_version") != 1
        or probe.get("image_id") != provenance.get("scorer_image_id")
        or probe.get("network") != "none"
        or probe.get("minimal_pytest") != "passed"
        or any(
            not isinstance(probe.get(field), str) or not probe[field]
            for field in (
                "python_version",
                "pytest_version",
                "pytest_json_report_version",
            )
        )
    ):
        _fail("formal ablation has no valid pinned Docker image capability probe")


def _validate_operation_lease_store(
    report_dir: Path,
    configuration: dict[str, Any],
) -> None:
    if (
        configuration.get("codex_operation_lease_store") != "."
        or configuration.get("docker_operation_lease_store") != "."
        or configuration.get("docker_container_absence_filter") != "label=lha.operation_id"
        or not isinstance(
            configuration.get("docker_operations_recovered_before_run"),
            int,
        )
        or isinstance(
            configuration.get("docker_operations_recovered_before_run"),
            bool,
        )
        or configuration["docker_operations_recovered_before_run"] != 0
        or configuration.get("docker_operations_recovered_at_completion") != 0
    ):
        _fail("formal ablation has no complete operation-lease attestation")
    for name in ("active-operations", "active-container-ids"):
        directory = report_dir / name
        if not directory.exists() and not directory.is_symlink():
            continue
        if directory.is_symlink() or not directory.is_dir():
            _fail(f"formal ablation operation store is unsafe: {name}")
        if any(directory.iterdir()):
            _fail(f"formal ablation operation store is not empty: {name}")


def _validate_formal_manifest_provenance(
    raw: dict[str, Any],
    tasks: list[str],
    repo_root: Path,
    *,
    require_completion: bool = True,
) -> None:
    provenance = cast(dict[str, Any], raw["provenance"])
    git_executable = _validate_git_executable_provenance(provenance)
    manifest_relative = provenance.get("formal_corpus_manifest_path")
    manifest_sha256 = provenance.get("formal_corpus_manifest_sha256")
    preregistration_commit = provenance.get("preregistration_commit")
    evaluated_commit = provenance.get("git_commit")
    if (
        manifest_relative != _FORMAL_CORPUS_MANIFEST_PATH.as_posix()
        or not isinstance(manifest_sha256, str)
        or _HEX_64.fullmatch(manifest_sha256) is None
        or not isinstance(preregistration_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", preregistration_commit) is None
        or not isinstance(evaluated_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", evaluated_commit) is None
    ):
        _fail("formal ablation has no valid preregistered corpus binding")
    manifest_relative = cast(str, manifest_relative)
    preregistration_commit = cast(str, preregistration_commit)
    evaluated_commit = cast(str, evaluated_commit)
    manifest_path = _repo_evidence_path(
        repo_root,
        manifest_relative,
        label="formal corpus manifest",
    )
    try:
        manifest, measured_manifest_sha256 = _load_formal_corpus_manifest(
            manifest_path,
            repo_root,
        )
    except (OSError, TypeError, ValueError) as error:
        _fail(f"formal corpus manifest is invalid: {error}")
    if measured_manifest_sha256 != manifest_sha256:
        _fail("formal corpus manifest digest disagrees with current bytes")
    if [entry["name"] for entry in manifest["tasks"]] != tasks:
        _fail("formal ablation tasks differ from the preregistered manifest")

    status = _git_success(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=normal"],
        git_executable=git_executable,
        label="worktree status",
    )
    if status:
        _fail("formal ablation release validation requires a clean Git worktree")
    head = (
        _git_success(
            repo_root,
            ["rev-parse", "--verify", "HEAD"],
            git_executable=git_executable,
            label="HEAD resolution",
        )
        .decode("ascii", errors="strict")
        .strip()
    )
    for commit, label in (
        (manifest["corpus_commit"], "corpus commit"),
        (preregistration_commit, "preregistration commit"),
        (evaluated_commit, "evaluated commit"),
    ):
        _git_success(
            repo_root,
            ["cat-file", "-e", f"{commit}^{{commit}}"],
            git_executable=git_executable,
            label=label,
        )
    for earlier, later, label in (
        (
            manifest["corpus_commit"],
            preregistration_commit,
            "corpus/preregistration ancestry",
        ),
        (
            preregistration_commit,
            evaluated_commit,
            "preregistration/evaluation ancestry",
        ),
        (evaluated_commit, head, "evaluation/HEAD ancestry"),
    ):
        _git_success(
            repo_root,
            ["merge-base", "--is-ancestor", earlier, later],
            git_executable=git_executable,
            label=label,
        )

    _validate_formal_attempt_provenance(
        raw,
        repo_root,
        git_executable=git_executable,
        head=head,
        require_completion=require_completion,
    )

    manifest_bytes = _git_success(
        repo_root,
        ["show", f"{preregistration_commit}:{manifest_relative}"],
        git_executable=git_executable,
        label="manifest blob",
    )
    if (
        len(manifest_bytes) > _MAX_FORMAL_MANIFEST_BYTES
        or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
        or manifest_bytes
        != _read_bounded_bytes(
            manifest_path,
            max_bytes=_MAX_FORMAL_MANIFEST_BYTES,
        )
    ):
        _fail("formal corpus manifest was not fixed before model execution")

    registered_inputs = [
        str(value)
        for entry in manifest["tasks"]
        for value in (entry["task_path"], entry["corpus_path"])
    ]
    comparisons: tuple[tuple[str, str, list[str], str], ...] = (
        (
            str(manifest["corpus_commit"]),
            evaluated_commit,
            registered_inputs,
            "registered task/corpus bytes",
        ),
        (
            preregistration_commit,
            head,
            [manifest_relative],
            "preregistered manifest bytes",
        ),
        (
            evaluated_commit,
            head,
            ["src/lha", *registered_inputs],
            "evaluated source and inputs",
        ),
    )
    for left, right, paths, label in comparisons:
        _git_success(
            repo_root,
            [
                "diff",
                "--quiet",
                "--no-ext-diff",
                "--no-textconv",
                left,
                right,
                "--",
                *paths,
            ],
            git_executable=git_executable,
            label=label,
        )


def _validate_provenance(
    raw: dict[str, Any],
    tasks: list[str],
    repo_root: Path,
    *,
    require_completion: bool = True,
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
        "cli_version",
        "cli_executable_sha256",
        "agent_backend",
        "scorer_requested",
        "scorer_backend",
        "platform",
        "python_version",
        "pytest_version",
        "formal_attempt_id",
        "formal_attempt_registry_path",
        "formal_attempt_registry_sha256",
        "formal_attempt_protocol_sha256",
        "formal_attempt_registration_commit",
        "formal_attempt_witness_remote_name",
        "formal_attempt_witness_remote_url",
        "formal_attempt_witness_ref",
        "formal_attempt_witness_commit",
        "formal_run_header_path",
        "formal_run_header_sha256",
        "formal_outcome_key",
    )
    for field in required_strings:
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            _fail(f"formal ablation provenance is missing {field!r}")
    if provenance.get("git_dirty") is not False:
        _fail("formal ablation provenance must come from a clean git worktree")
    if re.fullmatch(r"[0-9a-f]{40}", provenance["git_commit"]) is None:
        _fail("formal ablation provenance has an invalid git_commit")
    if (
        _HEX_64.fullmatch(provenance["formal_attempt_id"]) is None
        or provenance["formal_attempt_registry_path"]
        != FORMAL_ABLATION_ATTEMPTS_PATH.as_posix()
        or _HEX_64.fullmatch(provenance["formal_attempt_registry_sha256"]) is None
        or _HEX_64.fullmatch(provenance["formal_attempt_protocol_sha256"]) is None
        or re.fullmatch(
            r"[0-9a-f]{40}",
            provenance["formal_attempt_registration_commit"],
        )
        is None
        or provenance["formal_attempt_registration_commit"]
        != provenance["git_commit"]
        or provenance["formal_attempt_witness_ref"]
        != (
            "refs/heads/formal-attempts/"
            f"{provenance['formal_attempt_id']}"
        )
        or re.fullmatch(
            r"[0-9a-f]{40}",
            provenance["formal_attempt_witness_commit"],
        )
        is None
        or provenance["formal_run_header_path"] != _FORMAL_RUN_HEADER_NAME
        or _HEX_64.fullmatch(provenance["formal_run_header_sha256"]) is None
        or _HEX_64.fullmatch(provenance["formal_outcome_key"]) is None
    ):
        _fail("formal ablation provenance has an invalid attempt registration")
    if provenance["model"] != raw.get("model"):
        _fail("formal ablation model differs from provenance")
    if provenance["requested_llm_backend"] != raw.get("llm"):
        _fail("formal ablation LLM backend differs from provenance")
    if (
        provenance["requested_llm_backend"] != "codex_cli"
        or provenance["actual_llm_backend"] != "codex_cli"
    ):
        _fail("formal ablation evidence requires Codex CLI protocol validation")
    if (
        provenance.get("agent_backend") != "docker"
        or provenance.get("scorer_requested") != "docker"
        or provenance.get("scorer_backend") != "docker"
    ):
        _fail("formal ablation evidence requires Docker for gate and independent scoring")
    _validate_docker_executable_provenance(provenance)
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
    if _HEX_64.fullmatch(provenance["cli_executable_sha256"]) is None:
        _fail("formal ablation provenance has an invalid Codex executable digest")
    for field in (
        "task_paths",
        "corpus_paths",
        "task_files_sha256",
        "corpus_sha256",
        "input_snapshot_sha256",
        "cell_fingerprints",
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
        configuration.get(field) != expected for field, expected in expected_configuration.items()
    ):
        _fail("formal ablation provenance configuration differs from the report")
    _validate_docker_image_probe(provenance, configuration)
    if configuration.get("run_control_executables") != {
        "git": provenance.get("git_executable"),
        "docker": provenance.get("docker_executable"),
    }:
        _fail("formal ablation control-executable provenance is not internally bound")
    required_protocol = {
        "max_repairs": _MAX_REPAIRS,
        "llm_retries": _LLM_RETRIES,
        "cache_schema": _CACHE_SCHEMA,
        "report_schema": 4,
        "frozen_artifact_schema": 1,
        "input_snapshot_schema": 1,
        "scorer_evidence_schema": 2,
        "llm_call_receipt_schema": _LLM_CALL_RECEIPT_SCHEMA,
        "cell_attempt_schema": _CELL_ATTEMPT_SCHEMA,
        "formal_output_lock": {
            "protocol": "flock-exclusive-nonblocking",
            "path": _FORMAL_OUTPUT_LOCK_NAME,
            "lifetime": "full-run",
        },
        "formal_fresh_run": {
            "run_header_schema": _FORMAL_RUN_HEADER_SCHEMA,
            "run_header_path": _FORMAL_RUN_HEADER_NAME,
            "resume": False,
            "cache_reads": False,
            "expected_cell_starts": len(tasks) * cast(int, raw.get("reps")),
            "expected_terminal_cells": len(tasks) * cast(int, raw.get("reps")),
        },
        "scorer_result_source": "nonce-bound-pytest-hook-receipt",
    }
    if any(configuration.get(field) != expected for field, expected in required_protocol.items()):
        _fail("formal ablation provenance does not use the schema-4 evidence protocol")
    if any(
        not isinstance(provenance["cell_fingerprints"][task], str)
        or _HEX_64.fullmatch(provenance["cell_fingerprints"][task]) is None
        for task in tasks
    ):
        _fail("formal ablation provenance has an invalid cell fingerprint")
    if require_completion:
        _validate_formal_manifest_provenance(raw, tasks, repo_root)
    else:
        _validate_formal_manifest_provenance(
            raw,
            tasks,
            repo_root,
            require_completion=False,
        )


def _validate_llm_call_audits(
    raw: dict[str, Any],
    tasks: list[str],
    reps: int,
    report_dir: Path,
) -> None:
    """Load content-addressed receipts and bind every call to one measured cell."""
    provenance = cast(dict[str, Any], raw["provenance"])
    configuration = provenance.get("configuration")
    client = configuration.get("client") if isinstance(configuration, dict) else None
    fixed_client = _formal_codex_client_from_provenance(
        provenance,
        label="formal ablation report",
    )
    required_client = {
        "no_tools": True,
        "sandbox_mode": "read-only",
        "permission_model": "profile",
        "permission_profile": "lha-read",
        "credential_barrier": "verified",
        "externally_sandboxed": False,
    }
    if not isinstance(client, dict) or any(
        client.get(field) != expected for field, expected in required_client.items()
    ):
        _fail(
            "formal ablation provenance is missing the verified prompt-only "
            "Codex permission boundary"
        )
    configuration = cast(dict[str, Any], configuration)
    max_inner_retries = fixed_client.max_retries

    calls = raw.get("llm_calls")
    if not isinstance(calls, list) or not calls:
        _fail("formal ablation report is missing Codex call audits")
    store = raw.get("llm_call_receipt_store")
    if not isinstance(store, dict) or store != {
        "schema_version": _LLM_CALL_RECEIPT_SCHEMA,
        "path": "llm_call_receipts",
        "encoding": "canonical-json",
        "count": len(calls),
    }:
        _fail("formal ablation has an invalid LLM call receipt store")
    receipt_dir = report_dir / "llm_call_receipts"
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        _fail("formal ablation LLM call receipt store is unavailable")

    cells = {(task, rep) for task in tasks for rep in range(reps)}
    calls_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {cell: [] for cell in cells}
    executable_digests: set[str] = set()
    receipt_digests: set[str] = set()
    cache_modes: dict[tuple[str, int], set[bool]] = {cell: set() for cell in cells}
    for reference in calls:
        if not isinstance(reference, dict) or set(reference) != {
            "task",
            "rep",
            "ordinal",
            "receipt_sha256",
            "cache_hit",
        }:
            _fail("formal ablation Codex call reference must be an exact object")
        task = reference.get("task")
        rep = reference.get("rep")
        ordinal = reference.get("ordinal")
        digest = reference.get("receipt_sha256")
        cell = (task, rep)
        if (
            not isinstance(task, str)
            or not isinstance(rep, int)
            or isinstance(rep, bool)
            or cell not in cells
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal < 0
            or not isinstance(digest, str)
            or _HEX_64.fullmatch(digest) is None
            or type(reference.get("cache_hit")) is not bool
        ):
            _fail("formal ablation Codex call audit has an invalid cell binding")
        if reference["cache_hit"] is not False:
            _fail("formal ablation must not reuse any cached cell")
        if digest in receipt_digests:
            _fail("formal ablation reuses one LLM call receipt")
        receipt_digests.add(digest)
        try:
            receipt = _read_llm_call_receipt(
                receipt_dir / f"{digest}.json",
                digest,
                expected_binding={
                    "task": task,
                    "rep": rep,
                    "ordinal": ordinal,
                    "cell_fingerprint": provenance["cell_fingerprints"][task],
                    "input_snapshot_sha256": provenance["input_snapshot_sha256"][task],
                    "formal_attempt_id": provenance["formal_attempt_id"],
                    "formal_registration_registry_sha256": provenance[
                        "formal_attempt_registry_sha256"
                    ],
                    "formal_protocol_sha256": provenance[
                        "formal_attempt_protocol_sha256"
                    ],
                    "formal_outcome_key": provenance["formal_outcome_key"],
                },
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            _fail(f"formal ablation LLM call receipt {digest} is invalid: {error}")
        call = receipt["call"]
        if (
            call["cli_version"] != provenance.get("cli_version")
            or call["cli_executable_sha256"]
            != provenance.get("cli_executable_sha256")
            or call["model"] != provenance.get("model")
            or call["reasoning_effort"] != provenance.get("reasoning_effort")
        ):
            _fail("formal ablation Codex call audit disagrees with its protocol")
        executable_digests.add(call["cli_executable_sha256"])
        calls_by_cell[cast(tuple[str, int], cell)].append(receipt)
        cache_modes[cast(tuple[str, int], cell)].add(reference["cache_hit"])
    if executable_digests != {provenance["cli_executable_sha256"]}:
        _fail("formal ablation mixed Codex executable bytes")

    records_by_key = {
        (record["task"], record["rep"], record["condition"]): record
        for record in cast(list[dict[str, Any]], raw["records"])
    }
    for cell, cell_calls in calls_by_cell.items():
        verify = records_by_key[(*cell, "verify")]
        terminal_error = verify["status"] == "ERROR"
        if terminal_error:
            if not cell_calls:
                _fail(f"formal ablation ERROR cell {cell!r} has no LLM call receipt")
            if len(cache_modes[cell]) != 1:
                _fail(f"formal ablation ERROR cell {cell!r} mixes cache provenance")
            if any(
                receipt["binding"]["label"] != "first"
                or receipt["call"]["status"] != "failed"
                for receipt in cell_calls
            ):
                _fail(
                    "formal ablation ERROR cell must contain only failed first-call "
                    f"receipts for {cell!r}"
                )
        elif len(cache_modes[cell]) != 1:
            _fail(f"formal ablation cell {cell!r} mixes cache provenance")
        try:
            _validate_cell_call_sequence(
                cell_calls,
                repairs=verify["repairs"],
                max_outer_attempts=configuration["llm_retries"],
                max_inner_attempts=max_inner_retries + 1,
                terminal_error=terminal_error,
            )
        except (KeyError, TypeError, ValueError) as error:
            _fail(f"formal ablation cell {cell!r} call sequence is invalid: {error}")
        if terminal_error:
            continue
        successful = [receipt for receipt in cell_calls if receipt["call"]["status"] == "succeeded"]
        first = successful[0]["binding"]["result_artifact_sha256"]
        trust = records_by_key[(*cell, "trust")]["artifact_sha256"]
        gate = records_by_key[(*cell, "gate")]["artifact_sha256"]
        final = successful[-1]["binding"]["result_artifact_sha256"]
        if first != trust or first != gate or final != verify["artifact_sha256"]:
            _fail(f"formal ablation cell {cell!r} call outputs do not bind its artifacts")

    try:
        stored_receipts = {path.name for path in receipt_dir.iterdir()}
    except OSError as error:
        _fail(f"cannot enumerate formal ablation LLM call receipts: {error}")
    expected_receipts = {f"{digest}.json" for digest in receipt_digests}
    if stored_receipts != expected_receipts:
        _fail("formal ablation LLM call receipt store contains unreferenced or missing entries")


def _validate_formal_fresh_cells(
    raw: dict[str, Any],
    tasks: list[str],
    reps: int,
    report_dir: Path,
) -> None:
    """Require one new start and terminal seal for every scheduled formal cell."""
    provenance = cast(dict[str, Any], raw["provenance"])
    attempt_id = cast(str, provenance["formal_attempt_id"])
    registration_sha256 = cast(
        str,
        provenance["formal_attempt_registry_sha256"],
    )
    protocol_sha256 = cast(str, provenance["formal_attempt_protocol_sha256"])
    outcome_key = cast(str, provenance["formal_outcome_key"])
    header_path = report_dir / cast(str, provenance["formal_run_header_path"])
    try:
        header_bytes = _read_bounded_bytes(
            header_path,
            max_bytes=_MAX_FORMAL_RUN_HEADER_BYTES,
        )
    except (OSError, ValueError) as error:
        _fail(f"formal ablation run header is unavailable: {error}")
    expected_header = _canonical_json_object_bytes(
        {
            "schema_version": _FORMAL_RUN_HEADER_SCHEMA,
            "formal_attempt_id": attempt_id,
            "registration_registry_sha256": registration_sha256,
            "protocol_sha256": protocol_sha256,
            "outcome_key": outcome_key,
        }
    )
    if (
        header_bytes != expected_header
        or hashlib.sha256(header_bytes).hexdigest()
        != provenance["formal_run_header_sha256"]
    ):
        _fail("formal ablation run header differs from report provenance")

    results_dir = report_dir / "results"
    if results_dir.is_symlink() or not results_dir.is_dir():
        _fail("formal ablation fresh-cell store is unavailable")
    expected_names = {
        name
        for task in tasks
        for rep in range(reps)
        for name in (f"{task}__r{rep}.started.json", f"{task}__r{rep}.json")
    }
    try:
        actual_names = {entry.name for entry in results_dir.iterdir()}
    except OSError as error:
        _fail(f"cannot enumerate formal ablation fresh-cell store: {error}")
    if actual_names != expected_names:
        _fail(
            "formal ablation must contain exactly one new start and terminal "
            "seal for every scheduled cell"
        )

    records = cast(list[dict[str, Any]], raw["records"])
    references = cast(list[dict[str, Any]], raw["llm_calls"])
    condition_order = {name: index for index, name in enumerate(_CONDITION_NAMES)}
    formal_fields = {
        "formal_attempt_id": attempt_id,
        "formal_registration_registry_sha256": registration_sha256,
        "formal_protocol_sha256": protocol_sha256,
        "formal_outcome_key": outcome_key,
    }
    cache_keys = {
        "schema_version",
        "fingerprint",
        "terminal_error",
        "records",
        "llm_call_receipts",
        *formal_fields,
    }
    for task in tasks:
        for rep in range(reps):
            marker_path = results_dir / f"{task}__r{rep}.started.json"
            cache_path = results_dir / f"{task}__r{rep}.json"
            expected_marker = _canonical_json_object_bytes(
                {
                    "schema_version": _CELL_ATTEMPT_SCHEMA,
                    "task": task,
                    "rep": rep,
                    "cell_fingerprint": provenance["cell_fingerprints"][task],
                    "input_snapshot_sha256": (
                        provenance["input_snapshot_sha256"][task]
                    ),
                    **formal_fields,
                }
            )
            try:
                marker_bytes = _read_bounded_bytes(
                    marker_path,
                    max_bytes=_MAX_CELL_ATTEMPT_BYTES,
                )
            except (OSError, ValueError) as error:
                _fail(f"formal ablation cell-start seal is invalid: {error}")
            if marker_bytes != expected_marker:
                _fail(
                    f"formal ablation cell-start seal is stale for {(task, rep)!r}"
                )

            try:
                cache_bytes = _read_bounded_bytes(
                    cache_path,
                    max_bytes=_MAX_CACHE_BYTES,
                )
                cache = json.loads(cache_bytes)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as error:
                _fail(f"formal ablation terminal seal is invalid: {error}")
            if not isinstance(cache, dict) or set(cache) != cache_keys:
                _fail(
                    f"formal ablation terminal seal has an invalid shape for "
                    f"{(task, rep)!r}"
                )
            expected_records = sorted(
                [
                    record
                    for record in records
                    if record.get("task") == task and record.get("rep") == rep
                ],
                key=lambda record: condition_order.get(
                    cast(str, record.get("condition")),
                    len(condition_order),
                ),
            )
            expected_receipts = [
                cast(str, reference["receipt_sha256"])
                for reference in sorted(
                    (
                        reference
                        for reference in references
                        if reference.get("task") == task
                        and reference.get("rep") == rep
                    ),
                    key=lambda reference: cast(int, reference["ordinal"]),
                )
            ]
            terminal_error = any(
                record.get("status") == "ERROR" for record in expected_records
            )
            if (
                len(expected_records) != len(_CONDITION_NAMES)
                or cache.get("schema_version") != _CACHE_SCHEMA
                or cache.get("fingerprint")
                != provenance["cell_fingerprints"][task]
                or cache.get("terminal_error") is not terminal_error
                or cache.get("records") != expected_records
                or cache.get("llm_call_receipts") != expected_receipts
                or any(cache.get(field) != value for field, value in formal_fields.items())
            ):
                _fail(
                    f"formal ablation terminal seal differs from report cell "
                    f"{(task, rep)!r}"
                )


def _expected_formal_tasks(repository_root: Path) -> list[str]:
    manifest, _digest = _load_formal_corpus_manifest(
        repository_root / _FORMAL_CORPUS_MANIFEST_PATH,
        repository_root,
    )
    return [entry["name"] for entry in manifest["tasks"]]


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


def validate_formal_ablation_output(
    report_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate a finished formal run before its ``COMPLETED`` event exists.

    This is deliberately the same evidence gate used for a published schema-4
    report, except that the attempt registry must still contain the matching
    open registration.  A formal command may return non-zero because one or
    more cells ended in ``ERROR``; complete cell seals and failed-call receipts
    are still a valid, denominator-preserving formal result.
    """
    repository = Path(repo_root).resolve()
    directory = Path(report_dir)
    if not directory.is_absolute():
        directory = repository / directory
    try:
        resolved_directory = directory.resolve(strict=True)
        resolved_directory.relative_to(repository)
    except (OSError, ValueError) as error:
        _fail(f"formal ablation output directory is unsafe: {error}")
    if directory.is_symlink() or not resolved_directory.is_dir():
        _fail("formal ablation output directory must be a real directory")

    ablation_json = resolved_directory / "ablation_report.json"
    ablation_md = resolved_directory / "ablation_report.md"
    raw = _load_json(ablation_json)
    tasks, reps, records = _validate_record_grid(raw)
    boundary_problems = _validate_condition_stats(raw, records)
    report = _ablation_report_from_raw(raw)
    if (
        raw.get("schema_version") != 4
        or len(tasks) != _FORMAL_TASK_COUNT
        or reps != _FORMAL_REPETITIONS
        or report.tasks != tasks
        or report.reps != reps
        or not report.model
    ):
        _fail(
            "completed formal ablation output must contain the fixed "
            f"{_FORMAL_TASK_COUNT}-task x {_FORMAL_REPETITIONS}-repetition grid"
        )
    try:
        expected_tasks = _expected_formal_tasks(repository)
    except (OSError, TypeError, ValueError) as error:
        _fail(f"formal corpus manifest is invalid: {error}")
    if tasks != expected_tasks:
        _fail("formal ablation tasks differ from the fixed committed corpus")

    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        _fail("formal ablation report is missing provenance")
    attempt_id = provenance.get("formal_attempt_id")
    if (
        not isinstance(attempt_id, str)
        or resolved_directory
        != repository / "runs" / "formal_ablation" / attempt_id
    ):
        _fail("formal ablation report is not in its registered output directory")

    _validate_formal_error_cells(records)
    _validate_record_artifacts(records)
    _validate_artifact_store(raw, records, resolved_directory)
    _validate_scorer_evidence_store(raw, records, resolved_directory)
    _validate_provenance(
        raw,
        tasks,
        repository,
        require_completion=False,
    )
    _validate_operation_lease_store(
        resolved_directory,
        cast(dict[str, Any], provenance["configuration"]),
    )
    _validate_llm_call_audits(raw, tasks, reps, resolved_directory)
    _validate_formal_fresh_cells(raw, tasks, reps, resolved_directory)
    if boundary_problems:
        _fail(
            "formal ablation report violates the Wilson contract: "
            + boundary_problems[0]
        )
    if raw.get("fingerprint") != _report_fingerprint(raw):
        _fail("formal ablation report fingerprint does not match its contents")
    markdown = _read_text(ablation_md)
    if _LEGACY_ABLATION_MARKER in markdown.lower():
        _fail("formal ablation Markdown is still labelled as a legacy snapshot")
    if markdown.strip() != report.to_markdown().strip():
        _fail("formal ablation Markdown was not generated from its JSON report")
    return raw


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
    markdown = _read_text(ablation_md)
    if status == "formal":
        if len(tasks) != _FORMAL_TASK_COUNT or reps != _FORMAL_REPETITIONS:
            _fail(
                "formal ablation must contain the fixed "
                f"{_FORMAL_TASK_COUNT}-task x {_FORMAL_REPETITIONS}-repetition grid"
            )
        try:
            expected_tasks = _expected_formal_tasks(ablation_json.parent.parent)
        except (OSError, TypeError, ValueError) as error:
            _fail(f"formal corpus manifest is invalid: {error}")
        if tasks != expected_tasks:
            _fail("formal ablation tasks differ from the fixed committed corpus")
        _validate_formal_error_cells(records)
        _validate_record_artifacts(records)
        _validate_artifact_store(raw, records, ablation_json.parent)
        _validate_scorer_evidence_store(raw, records, ablation_json.parent)
        _validate_provenance(raw, tasks, ablation_json.parent.parent)
        _validate_operation_lease_store(
            ablation_json.parent,
            cast(dict[str, Any], raw["provenance"]["configuration"]),
        )
        _validate_llm_call_audits(raw, tasks, reps, ablation_json.parent)
        _validate_formal_fresh_cells(raw, tasks, reps, ablation_json.parent)
        if boundary_problems:
            _fail("formal ablation report violates the Wilson contract: " + boundary_problems[0])
        if raw.get("fingerprint") != _report_fingerprint(raw):
            _fail("formal ablation report fingerprint does not match its contents")
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
    scheduled_cell_keys = {
        (task, rep)
        for task in tasks
        for rep in range(reps)
    }
    error_cell_keys = {
        (record.task, record.rep) for record in records if record.status == "ERROR"
    }
    usable_cell_keys = scheduled_cell_keys - error_cell_keys
    return _AblationFacts(
        report=report,
        raw=raw,
        records=records,
        status=status,
        scheduled_cells=len(scheduled_cell_keys),
        usable_cells=len(usable_cell_keys),
        error_cells=len(error_cell_keys),
        trust_successes=sum(record.true_success for record in trust),
        trust_false_successes=sum(record.false_success for record in trust),
        gate_successes=sum(record.true_success for record in gate),
        gate_interceptions=sum(
            not record.claimed_success and not record.artifact_correct for record in gate
        ),
        verify_successes=sum(record.true_success for record in verify),
        trust_gate_p=_paired_p(
            records,
            "trust",
            "gate",
            "false_success",
            task_cluster_inference=status == "formal",
        ),
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
        reps=list(range(ablation.report.reps)),
        outcome={
            (record.condition, record.task, record.rep): record.true_success
            for record in ablation.records
            if record.status != "ERROR"
        },
        model=ablation.report.model,
        source=expected_source,
        source_schema_version=ablation.report.schema_version,
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
    cell_estimand = estimands["cell"]
    if "mcnemar_p" in estimands["composition"]:
        _fail("horizon composition must not report a McNemar p-value")
    if estimands["composition"]["independent_samples_added"] != 0:
        _fail("horizon composition must add zero independent samples")
    if ablation.status == "formal":
        if "mcnemar_p" in cell_estimand:
            _fail("formal horizon cells must not report a cell-level McNemar p-value")
        cluster_p = cell_estimand.get("task_cluster_sign_flip_p")
        expected_cluster_p = _paired_p(
            ablation.records,
            "trust",
            "verify",
            "true_success",
            task_cluster_inference=True,
        )
        if not _is_close(cluster_p, expected_cluster_p):
            _fail("formal horizon has a stale task-cluster paired sign-flip p-value")
        return expected_cluster_p
    if "task_cluster_sign_flip_p" in cell_estimand:
        _fail("legacy horizon unexpectedly changes its historical cell estimand")
    mcnemar_p = cell_estimand.get("mcnemar_p")
    if not isinstance(mcnemar_p, (int, float)) or isinstance(mcnemar_p, bool):
        _fail("legacy horizon cell McNemar p-value is invalid")
    return float(mcnemar_p)


def _result_section(readme: str) -> str:
    match = re.search(
        r"^## (?:已验证结果|已提交的实测结果)\s*$\n(?P<body>.*?)(?=^## |\Z)",
        readme,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        _fail("README is missing the measured-results section")
    return match.group("body")


def _require_readme_match(
    section: str, pattern: str, expected: tuple[str, ...], label: str
) -> None:
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
            "committed Terminal-Bench evidence must use schema 4 with a complete source attestation"
        )
    return validation


def _terminal_claim_scope(text: str) -> str | None:
    """Return the Terminal-Bench subsection without borrowing nearby claims."""
    heading = re.search(
        rf"^(?P<marks>#{{2,6}})[^\n]*{re.escape(_TERMINAL_SUBSET_LABEL)}[^\n]*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if heading is not None:
        level = len(heading.group("marks"))
        end = len(text)
        for candidate in re.finditer(r"^(?P<marks>#{2,6})\s+", text[heading.end() :], re.MULTILINE):
            if len(candidate.group("marks")) <= level:
                end = heading.end() + candidate.start()
                break
        return text[heading.start() : end]

    marker = text.find(_TERMINAL_SUBSET_LABEL)
    if marker < 0:
        return None
    return text[marker:]


def _claim_values(text: str, patterns: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = next((group for group in match.groups() if group is not None), None)
            if value is not None:
                values.append(value)
    return values


def _require_claim_values(
    values: list[str],
    *,
    expected: str,
    label: str,
) -> None:
    if not values:
        _fail(f"Terminal-Bench claim is missing {label}")
    if any(value != expected for value in values):
        if label in {"passed/20", "PASS count", "FAIL count", "ERROR count"}:
            _fail("Terminal-Bench counts differ from the committed evidence")
        _fail(f"Terminal-Bench {label} differs from the committed evidence")


def _validate_terminal_counts(
    claim: str,
    terminal: TerminalBenchPublicEvidenceValidation,
) -> None:
    ratios = _claim_values(claim, (r"(?<!\d)(\d+)\s*/\s*20(?!\d)",))
    pass_counts = _claim_values(
        claim,
        (
            r"(?:passed|pass|通过)\s*(?:为|[:：=|])?\s*`?(\d+)`?",
            r"(?<!\d)(\d+)\s*(?:个)?\s*(?:"
            r"`(?:passed|pass|通过)`(?!/)|(?:passed|pass|通过)(?![/A-Za-z]))",
        ),
    )
    failed_counts = _claim_values(
        claim,
        (
            r"(?:failed|fail|失败)\s*(?:为|[:：=|])?\s*`?(\d+)`?",
            r"(?<!\d)(\d+)\s*(?:个)?\s*`?(?:failed|fail|失败)`?",
        ),
    )
    error_counts = _claim_values(
        claim,
        (
            r"ERROR\s*(?:为|[:：=|])?\s*`?(\d+)`?",
            r"(?<!\d)(\d+)\s*(?:个)?\s*`?ERROR`?",
        ),
    )
    _require_claim_values(ratios, expected=str(terminal.passed), label="passed/20")
    if pass_counts:
        _require_claim_values(
            pass_counts,
            expected=str(terminal.passed),
            label="PASS count",
        )
    _require_claim_values(
        failed_counts,
        expected=str(terminal.failed),
        label="FAIL count",
    )
    _require_claim_values(
        error_counts,
        expected=str(terminal.errors),
        label="ERROR count",
    )


def _terminal_paragraphs(text: str) -> tuple[str, ...]:
    return tuple(
        paragraph
        for paragraph in re.split(r"\n[ \t]*\n", text)
        if re.search(r"Terminal[- ]Bench", paragraph, re.IGNORECASE)
    )


def _validate_terminal_runtime_claims(
    scopes: tuple[str, ...],
    terminal: TerminalBenchPublicEvidenceValidation,
) -> None:
    expected_labels = (
        (
            (r"(?:模型|model)\s*(?:为|[:：=])?\s*`?([A-Za-z0-9][A-Za-z0-9._/+:-]*)`?",),
            terminal.model,
            "model",
        ),
        (
            (
                r"(?:推理强度|reasoning[_ -]?effort)\s*(?:为|[:：=])?\s*"
                r"`?([A-Za-z0-9._+-]+)`?",
            ),
            terminal.reasoning_effort,
            "reasoning effort",
        ),
        (
            (
                r"\bHarbor\b(?:\s*版本|\s+version)?\s*(?:为|[:：=])?\s*"
                r"`?([A-Za-z0-9.+-]+)`?",
            ),
            terminal.harbor_version,
            "Harbor version",
        ),
    )
    for patterns, expected, label in expected_labels:
        values: list[str] = []
        for scope in scopes:
            values.extend(_claim_values(scope, patterns))
        if not values:
            _fail(f"Terminal-Bench public documentation is missing {label}")
        if any(value != expected for value in values):
            _fail(f"Terminal-Bench {label} differs from the committed evidence")


def _validate_terminal_readme(
    readme: str,
    section: str,
    terminal: TerminalBenchPublicEvidenceValidation | None,
    *,
    benchmark_docs: str,
) -> None:
    if terminal is None:
        for document in (readme, benchmark_docs):
            for claim_window in _terminal_paragraphs(document):
                if not _TERMINAL_NUMERIC_CLAIM.search(claim_window):
                    continue
                _fail(
                    "public documentation cannot publish a Terminal-Bench numeric result "
                    "without committed public evidence"
                )
        return

    claim = _terminal_claim_scope(section)
    if claim is None:
        _fail(f"README measured-results section must name '{_TERMINAL_SUBSET_LABEL}'")
    _validate_terminal_counts(claim, terminal)

    docs_claim = _terminal_claim_scope(benchmark_docs)
    if docs_claim is not None and _TERMINAL_NUMERIC_CLAIM.search(docs_claim):
        _validate_terminal_counts(docs_claim, terminal)
    runtime_scopes = (
        claim,
        *((docs_claim,) if docs_claim is not None else ()),
        *_terminal_paragraphs(readme),
        *_terminal_paragraphs(benchmark_docs),
    )
    _validate_terminal_runtime_claims(runtime_scopes, terminal)


def _require_readme_count(
    section: str,
    patterns: tuple[str, ...],
    *,
    expected: int,
    label: str,
) -> None:
    for pattern in patterns:
        match = re.search(pattern, section, re.IGNORECASE)
        if match is not None:
            if match.group(1) == str(expected):
                return
            break
    _fail(f"README {label} claim differs from the committed reports")


def _validate_formal_readme_coverage(section: str, ablation: _AblationFacts) -> None:
    """Require explicit schedule, availability, ERROR, and rate-denominator claims."""
    # Terminal-Bench also reports ERROR. Restrict these patterns to the
    # ablation subsection so one benchmark cannot accidentally satisfy another.
    ablation_section = re.split(r"Terminal[- ]Bench", section, maxsplit=1, flags=re.IGNORECASE)[
        0
    ]
    _require_readme_count(
        ablation_section,
        (
            r"(?:计划(?:执行|安排)?|预定)\s*`?(\d+)`?\s*组",
            r"(?:计划(?:执行|安排)?|预定)[^。\n]{0,32}?`?(\d+)`?\s*组",
        ),
        expected=ablation.scheduled_cells,
        label="formal ablation scheduled-cell",
    )
    _require_readme_count(
        ablation_section,
        (
            r"`?(\d+)`?\s*组(?:结果|单元|数据)?\s*可用",
            r"可用(?:结果|单元|数据)?\s*(?:为|[:：=])?\s*`?(\d+)`?\s*组",
        ),
        expected=ablation.usable_cells,
        label="formal ablation usable-cell",
    )
    _require_readme_count(
        ablation_section,
        (
            r"ERROR\s*(?:为|[:：=])?\s*`?(\d+)`?\s*组",
            r"`?(\d+)`?\s*组(?:结果|单元|数据)?\s*(?:为|是)?\s*ERROR",
        ),
        expected=ablation.error_cells,
        label="formal ablation ERROR-cell",
    )
    _require_readme_count(
        ablation_section,
        (
            r"(?:比例|成功率|统计率)[^。\n]{0,64}?"
            r"(?:以|按)\s*`?(\d+)`?\s*组(?:可用(?:结果|单元|数据)?)?"
            r"\s*(?:为|作|作为)\s*分母",
            r"(?:统计)?分母\s*(?:为|是|[:：=])\s*`?(\d+)`?\s*组"
            r"(?:可用(?:结果|单元|数据)?)?",
        ),
        expected=ablation.usable_cells,
        label="formal ablation rate denominator",
    )


def _validate_ablation_schedule_claims(
    section: str,
    ablation: _AblationFacts,
) -> None:
    """Validate concise schedule claims even when legacy outcomes are omitted."""
    for match in re.finditer(
        r"(?<!\d)(\d+)\s*(?:个\s*)?(?:(?:固定|预设)\s*)?"
        r"(?:Python\s*)?(?:缺陷|任务)\s*[×xX*]\s*(\d+)\s*次",
        section,
        re.IGNORECASE,
    ):
        if match.groups() != (
            str(len(ablation.report.tasks)),
            str(ablation.report.reps),
        ):
            _fail("README formal ablation task/repetition schedule differs")


def _has_legacy_outcome_number(section: str) -> bool:
    number = r"(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?\s*%|\d+)"
    outcome = (
        r"(?:trust|gate|verify|正确|错误(?:交付|补丁)?|拦截|误拒|"
        r"修复成功|成功率|通过率|交付率|独立评分|success|wrong|intercept|repair)"
    )
    return bool(
        re.search(
            rf"(?:{number})[^。\n]{{0,48}}{outcome}|"
            rf"{outcome}[^。\n]{{0,48}}(?:{number})|"
            r"\bp\s*=\s*\d+(?:\.\d+)?",
            section,
            re.IGNORECASE,
        )
    )


def _validate_readme(
    readme_path: Path,
    ablation: _AblationFacts,
    horizon_cell_p: float,
    terminal: TerminalBenchPublicEvidenceValidation | None,
) -> None:
    readme = _read_text(readme_path)
    benchmark_docs_path = readme_path.parent / "docs" / "BENCHMARKS.md"
    benchmark_docs = (
        _read_text(benchmark_docs_path)
        if benchmark_docs_path.exists() and not benchmark_docs_path.is_symlink()
        else ""
    )
    section = _result_section(readme)
    _validate_terminal_readme(
        readme,
        section,
        terminal,
        benchmark_docs=benchmark_docs,
    )
    pending = bool(
        re.search(
            r"(?:正式[^。\n]{0,80}(?:尚未|未完成|没有)|"
            r"(?:尚未|未完成)[^。\n]{0,80}(?:正式|COMPLETED)|"
            r"204.{0,160}(?:未完成|尚未|只有在|后才))",
            section,
            re.DOTALL,
        )
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
        _validate_formal_readme_coverage(section, ablation)

    # A concise README may omit legacy outcome counts entirely. If it chooses
    # to publish them, every count and p-value remains bound to the committed
    # reports. Formal evidence always requires the complete headline set.
    ablation_section = re.split(
        r"Terminal[- ]Bench",
        section,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    _validate_ablation_schedule_claims(ablation_section, ablation)
    detailed_legacy_claim = bool(
        re.search(r"`(?:trust|gate|verify)`|\bp\s*=", ablation_section)
    )
    if (
        ablation.status == "legacy"
        and _has_legacy_outcome_number(ablation_section)
        and not detailed_legacy_claim
    ):
        _fail("README cannot publish unbound legacy ablation outcome numbers")
    publishes_ablation_numbers = ablation.status != "legacy" or detailed_legacy_claim
    if not publishes_ablation_numbers:
        return

    _require_readme_match(
        section,
        r"(\d+)\s*个预设 Python 缺陷，每个任务重复\s*(\d+)\s*次，共\s*(\d+)\s*组",
        (
            str(len(ablation.report.tasks)),
            str(ablation.report.reps),
            str(ablation.scheduled_cells),
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
        (str(ablation.verify_successes), str(ablation.usable_cells)),
        "verify outcome",
    )

    published_p = re.findall(r"\bp\s*=\s*([0-9]+(?:\.[0-9]+)?)", section)
    expected_p_values = {ablation.trust_gate_p, horizon_cell_p}
    for expected in expected_p_values:
        if not any(
            len(value.partition(".")[2]) >= 2
            and math.isclose(
                float(value),
                expected,
                rel_tol=1e-9,
                abs_tol=0.5 * 10 ** -len(value.partition(".")[2]) + 1e-12,
            )
            for value in published_p
        ):
            label = (
                "task-cluster paired sign-flip"
                if ablation.status == "formal"
                else "historical McNemar"
            )
            _fail(f"README is missing the measured {label} p-value {expected:.4f}")
    if ablation.status == "formal" and not (
        re.search(r"(?:按任务聚类|任务聚类)", section)
        and re.search(r"(?:符号翻转|sign-flip)", section, re.IGNORECASE)
    ):
        _fail("README must identify schema-v4 paired inference as task-cluster sign-flip")
    if not re.search(r"(?:不增加|没有增加).{0,12}(?:样本|观测)", section):
        _fail("README must state that horizon composition adds no samples")


def validate_release_claims(root: str | Path = ".") -> ReleaseClaimsSummary:
    """Validate the committed README and benchmark report set.

    ``root`` is the repository root.  The function is reusable from tests, CI,
    and release tooling; it performs no writes.
    """
    repo = Path(root).resolve()
    benchmarks = repo / "benchmarks"
    _validate_formal_ablation_disclosures(repo)
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
    terminal = _validate_terminal_evidence(benchmarks / _TERMINAL_EVIDENCE_DIR)
    _validate_readme(repo / "README.md", ablation, horizon_p, terminal)
    return ReleaseClaimsSummary(
        status=ablation.status,
        tasks=len(ablation.report.tasks),
        repetitions=ablation.report.reps,
        scheduled_cells=ablation.scheduled_cells,
        usable_cells=ablation.usable_cells,
        error_cells=ablation.error_cells,
        cells=ablation.usable_cells,
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
        f"scheduled={summary.scheduled_cells}; usable={summary.usable_cells}; "
        f"errors={summary.error_cells}; model={summary.model})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
