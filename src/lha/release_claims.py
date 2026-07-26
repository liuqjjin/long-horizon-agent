"""Validate public benchmark claims against committed machine-readable evidence.

The release check intentionally reads only the public project overview and the
generated benchmark reports.  Resume text lives outside the repository and is
not parsed here.

Schema-1 ablation reports predate complete provenance and Wilson boundary
intervals.  They may remain available as historical evidence, but every public
surface must label them as a legacy snapshot.  Schema-2 reports receive no such
exemption: their provenance, statistics, generated Markdown, and derived
horizon report must all validate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, cast

from .ablation import CONDITIONS, AblationReport, RunRecord, load_ablation_report
from .bench.stats import mcnemar_exact, wilson_interval
from .horizon import build_report, load_cells

_CONDITION_NAMES = tuple(name for name, _blurb in CONDITIONS)
_LEGACY_README_MARKER = "历史快照（legacy）"
_FORMAL_README_MARKER = "正式报告（formal）"
_LEGACY_ABLATION_MARKER = "legacy snapshot"
_LEGACY_HORIZON_MARKER = "legacy snapshot"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


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
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path} must contain a JSON object")
    return value


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
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
        truth = _record_bool(record_raw, "true_success")
        false_success = _record_bool(record_raw, "false_success")
        if false_success != (claimed and not truth):
            _fail(f"ablation false_success is inconsistent for {key!r}")
        try:
            records.append(RunRecord(**record_raw))
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


def _boundary_interval_problem(
    stat: dict[str, Any],
    *,
    field: str,
    successes: int,
    total: int,
) -> str | None:
    interval_name = "true_ci" if field == "true_success_rate" else "false_ci"
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
        truths = sum(record.true_success for record in usable)
        false_successes = sum(record.false_success for record in usable)
        repairs = sum(record.repairs for record in usable)
        _require_rate(stat, "claimed_success_rate", claimed / n, condition)
        _require_rate(stat, "true_success_rate", truths / n, condition)
        _require_rate(stat, "false_success_rate", false_successes / n, condition)
        _require_rate(stat, "mean_repairs", repairs / n, condition)

        predictions = [record for record in usable if record.gate_prediction is not None]
        if predictions:
            confusion = {
                "tp": sum(bool(record.gate_prediction) and record.true_success for record in predictions),
                "fp": sum(
                    bool(record.gate_prediction) and not record.true_success
                    for record in predictions
                ),
                "tn": sum(
                    not record.gate_prediction and not record.true_success
                    for record in predictions
                ),
                "fn": sum(
                    not record.gate_prediction and record.true_success
                    for record in predictions
                ),
            }
            for field, expected in confusion.items():
                if stat.get(field) != expected:
                    _fail(f"ablation stats for {condition!r} have stale {field}")

        for field, successes in (
            ("true_success_rate", truths),
            ("false_success_rate", false_successes),
        ):
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


def _validate_provenance(raw: dict[str, Any], tasks: list[str]) -> None:
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
    if provenance["scorer_backend"] != raw.get("scorer"):
        _fail("formal ablation scorer differs from provenance")
    if provenance["harness_version"] != raw.get("harness_version"):
        _fail("formal ablation harness version differs from provenance")
    if not _HEX_64.fullmatch(provenance["source_tree_sha256"]):
        _fail("formal ablation provenance has an invalid source_tree_sha256")
    if not _HEX_64.fullmatch(str(raw.get("fingerprint", ""))):
        _fail("formal ablation report has an invalid fingerprint")
    if provenance["actual_llm_backend"].endswith("_cli") and not provenance.get("cli_version"):
        _fail("formal CLI-backed ablation provenance is missing cli_version")
    for field in ("task_paths", "task_files_sha256", "corpus_sha256"):
        values = provenance.get(field)
        if not isinstance(values, dict) or set(values) != set(tasks):
            _fail(f"formal ablation provenance {field!r} does not cover every task")
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


def _format_percent(value: float) -> str:
    return f"{100 * value:.0f}%"


def _validate_legacy_ablation_markdown(
    markdown: str, report: AblationReport, records: tuple[RunRecord, ...]
) -> None:
    if _LEGACY_ABLATION_MARKER not in markdown.lower():
        _fail("schema-1 ablation Markdown must be labelled 'legacy snapshot'")
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
    report = load_ablation_report(ablation_json)
    if report.tasks != tasks or report.reps != reps:
        _fail("ablation loader disagrees with the report's task/repetition fields")
    if not report.model:
        _fail("ablation report must name the evaluated model")

    schema_version = raw.get("schema_version", 1)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        _fail("ablation schema_version must be an integer")
    status = "formal" if schema_version >= 2 else "legacy"
    errors = sum(record.status == "ERROR" for record in records)
    markdown = _read_text(ablation_md)
    if status == "formal":
        if errors:
            _fail(f"formal ablation report contains {errors} ERROR cells")
        _validate_provenance(raw, tasks)
        if boundary_problems:
            _fail("formal ablation report violates the Wilson contract: " + boundary_problems[0])
        if _LEGACY_ABLATION_MARKER in markdown.lower():
            _fail("formal ablation Markdown is still labelled as a legacy snapshot")
        if markdown.strip() != report.to_markdown().strip():
            _fail("formal ablation Markdown was not generated from the committed JSON")
    else:
        if not boundary_problems and raw.get("provenance"):
            _fail("schema-1 report has no legacy deficiency; regenerate it as schema 2")
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
        gate_successes=sum(record.claimed_success and record.true_success for record in gate),
        gate_interceptions=sum(
            not record.claimed_success and not record.true_success for record in gate
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
    cells = replace(load_cells(ablation_json), source=expected_source)
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


def _validate_readme(
    readme_path: Path,
    ablation: _AblationFacts,
    horizon_cell_p: float,
) -> None:
    section = _result_section(_read_text(readme_path))
    pending = bool(
        re.search(r"204.{0,160}(?:未完成|尚未|只有在|后才)", section, re.DOTALL)
    )
    if ablation.status == "legacy":
        if _LEGACY_README_MARKER not in section:
            _fail("README must label a schema-1 benchmark as '历史快照（legacy）'")
        if not pending:
            _fail("README must state that the formal 204-cell rerun is still pending")
        if _FORMAL_README_MARKER in section:
            _fail("README cannot label a legacy benchmark as formal")
    else:
        if _FORMAL_README_MARKER not in section:
            _fail("README must label schema-2 evidence as '正式报告（formal）'")
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
    _validate_readme(repo / "README.md", ablation, horizon_p)
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
