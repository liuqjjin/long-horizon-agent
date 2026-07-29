"""The horizon analysis keeps cells, repetition aggregates, and composition distinct.

Complete-corpus repetition aggregates are built after execution from measured
cells; they are not executed shared-state long tasks. Current cell inference
clusters repetitions by task, while the aggregate comparison pairs complete
repetitions. Composition remains descriptive and contributes zero independent
samples.
"""

from __future__ import annotations

import json
from math import comb

import pytest

from lha.bench.stats import mcnemar_exact, wilson_interval
from lha.horizon import (
    HorizonDataError,
    build_report,
    compounding_curve,
    episodes_for,
    load_cells,
    per_task_p,
    run_horizon,
)


def _report(tmp_path, outcomes: dict[str, list[tuple[bool, bool]]], model="m"):
    """Write a minimal ablation report. ``outcomes[task] = [(trust, verify), ...]``."""
    records = []
    for task, reps in outcomes.items():
        for rep, (trust, verify) in enumerate(reps):
            records.append(
                {
                    "task": task,
                    "condition": "trust",
                    "rep": rep,
                    "status": "DONE",
                    "claimed_success": True,
                    "artifact_correct": trust,
                    "true_success": trust,
                }
            )
            records.append(
                {
                    "task": task,
                    "condition": "verify",
                    "rep": rep,
                    "status": "DONE",
                    "claimed_success": verify,
                    "artifact_correct": verify,
                    "true_success": verify,
                }
            )
    path = tmp_path / "ablation_report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "tasks": list(outcomes),
                "model": model,
                "records": records,
            }
        )
    )
    return path


# --- the compounding model is exact, not sampled -----------------------------
def test_compounding_curve_matches_p_to_the_n_for_uniform_p():
    p = 0.9
    curve = compounding_curve([p] * 6)
    for k, value in enumerate(curve, start=1):
        assert value == pytest.approx(p**k)


def test_compounding_curve_is_the_symmetric_polynomial_over_random_orderings():
    # Two tasks at 2/3 among five perfect ones: P(first k all succeed) is the
    # degree-k elementary symmetric polynomial over C(n, k) subsets.
    probs = [1.0, 1.0, 1.0, 2 / 3, 2 / 3]
    curve = compounding_curve(probs)
    n = len(probs)
    for k in (1, 3, 5):
        expected = sum(
            (2 / 3) ** j * comb(2, j) * comb(3, k - j) for j in range(0, min(2, k) + 1)
        ) / comb(n, k)
        assert curve[k - 1] == pytest.approx(expected)
    assert curve[-1] == pytest.approx((2 / 3) ** 2)  # terminal = the plain product


def test_compounding_curve_is_monotone_and_starts_at_the_mean():
    probs = [1.0, 0.5, 0.8, 1.0]
    curve = compounding_curve(probs)
    assert curve[0] == pytest.approx(sum(probs) / len(probs))
    assert all(a >= b - 1e-12 for a, b in zip(curve, curve[1:]))


def test_compounding_curve_handles_the_empty_corpus():
    assert compounding_curve([]) == []


# --- loading ------------------------------------------------------------------
def test_load_cells_drops_error_records(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)]})
    raw = json.loads(path.read_text())
    raw["records"].append(
        {
            "task": "a",
            "condition": "trust",
            "rep": 1,
            "status": "ERROR",
            "claimed_success": False,
            "artifact_correct": False,
            "true_success": False,
        }
    )
    path.write_text(json.dumps(raw))
    cells = load_cells(path)
    assert ("trust", "a", 1) not in cells.outcome  # an ERROR is not a measurement
    assert cells.reps == [0, 1]  # but it remains in the scheduled coverage
    assert cells.complete_reps("trust") == [0]


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_load_cells_rejects_non_boolean_truth_values(tmp_path, value):
    path = _report(tmp_path, {"a": [(True, True)]})
    raw = json.loads(path.read_text())
    raw["records"][0]["true_success"] = value
    path.write_text(json.dumps(raw))

    with pytest.raises(HorizonDataError, match="'true_success' must be boolean"):
        load_cells(path)


def test_load_cells_validates_error_records_before_dropping_them(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)]})
    raw = json.loads(path.read_text())
    raw["records"].append(
        {
            "task": "a",
            "condition": "trust",
            "rep": 1,
            "status": "ERROR",
            "claimed_success": False,
            "artifact_correct": False,
            "true_success": "false",
        }
    )
    path.write_text(json.dumps(raw))

    with pytest.raises(HorizonDataError, match="'true_success' must be boolean"):
        load_cells(path)


def test_load_cells_rejects_duplicate_condition_task_rep_cells(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)]})
    raw = json.loads(path.read_text())
    duplicate = dict(raw["records"][0])
    duplicate["claimed_success"] = False
    duplicate["artifact_correct"] = False
    duplicate["true_success"] = False
    raw["records"].append(duplicate)
    path.write_text(json.dumps(raw))

    with pytest.raises(HorizonDataError, match="duplicate measured cell"):
        load_cells(path)


def test_load_cells_rejects_boolean_repetition_values(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)]})
    raw = json.loads(path.read_text())
    raw["records"][0]["rep"] = True
    path.write_text(json.dumps(raw))

    with pytest.raises(HorizonDataError, match="invalid repetition"):
        load_cells(path)


def test_schema_four_chain_uses_delivery_not_artifact_correctness(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)]})
    raw = json.loads(path.read_text())
    verify = next(record for record in raw["records"] if record["condition"] == "verify")
    verify["claimed_success"] = False
    verify["artifact_correct"] = True
    verify["true_success"] = False
    verify["status"] = "FAILED"
    path.write_text(json.dumps(raw))

    cells = load_cells(path)

    assert cells.truth("verify", "a", 0) is False


def test_schema_four_rejects_inconsistent_delivery_truth(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)]})
    raw = json.loads(path.read_text())
    raw["records"][0]["claimed_success"] = False
    path.write_text(json.dumps(raw))

    with pytest.raises(HorizonDataError, match="inconsistent delivered correctness"):
        load_cells(path)


def test_load_cells_rejects_an_empty_report(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"tasks": [], "records": []}))
    with pytest.raises(HorizonDataError):
        load_cells(path)


def test_incomplete_rep_is_not_a_shorter_complete_corpus_aggregate(tmp_path):
    # rep 1 is missing task "b": it must be excluded, not scored over one step.
    path = _report(tmp_path, {"a": [(True, True), (True, True)], "b": [(True, True)]})
    cells = load_cells(path)
    assert cells.complete_reps("trust") == [0]
    assert [e.rep for e in episodes_for(cells, "trust-chain", "trust")] == [0]
    report = build_report(cells)
    assert report.independent_episode_count == 1
    assert report.episode_estimand.pairs == 1
    assert {episode.rep for episode in report.episodes} == {0}


# --- complete-corpus repetition aggregates -----------------------------------
def test_episode_compat_record_aggregates_one_complete_corpus_repetition(tmp_path):
    cells = load_cells(
        _report(
            tmp_path,
            {
                "a": [(True, True), (True, True)],
                "b": [(False, True), (True, True)],  # rep 0 wrong under trust
            },
        )
    )
    eps = {e.rep: e for e in episodes_for(cells, "trust-chain", "trust")}
    assert eps[0].end_to_end is False and eps[0].failing_tasks == ["b"]
    assert eps[0].steps_correct == 1 and eps[0].n_steps == 2
    assert eps[1].end_to_end is True and eps[1].failing_tasks == []


# --- the three estimands ------------------------------------------------------
def test_report_separates_cells_repetition_aggregates_and_composition(tmp_path):
    outcomes = {f"t{i:02d}": [(True, True)] * 3 for i in range(17)}
    outcomes["t05"][0] = (False, True)  # wrong in rep 0, repaired by verify
    outcomes["t11"][1] = (False, True)  # wrong in rep 1, repaired by verify
    report = build_report(load_cells(_report(tmp_path, outcomes)))

    assert report.n_steps == 17
    # The compatibility field counts aggregates, not independent executed episodes.
    assert report.independent_episode_count == 3
    assert report.coverage.scheduled_paired_cells == 51
    assert report.coverage.usable_paired_cells == 51
    assert report.coverage.unavailable_or_error_cells == 0
    assert report.coverage.scheduled_repetitions == 3
    assert report.coverage.complete_paired_repetitions == 3

    assert report.cell_estimand.pairs == 51
    assert report.cell_estimand.discordant == (2, 0)
    assert report.cell_estimand.task_clusters == 17
    assert report.cell_estimand.nonzero_task_clusters == 2
    assert report.cell_estimand.task_cluster_sign_flip_p == pytest.approx(0.5)

    assert report.episode_estimand.pairs == 3
    assert report.episode_estimand.discordant == (2, 0)
    assert report.episode_estimand.mcnemar_p == pytest.approx(mcnemar_exact(2, 0))

    composition = report.composition_estimand
    assert composition.independent_samples_added == 0
    trust = next(c for c in composition.curves if c.condition == "trust-chain")
    verify = next(c for c in composition.curves if c.condition == "verify-chain")
    assert trust.rate[0] == pytest.approx(49 / 51)  # per-step, as measured
    assert trust.rate[-1] == pytest.approx((2 / 3) ** 2)  # 17 steps: 44.4%
    assert verify.rate[-1] == pytest.approx(1.0)

    md = report.to_markdown()
    assert "Estimand 1" in md and "Estimand 2" in md and "Estimand 3" in md
    assert "not significant" in md
    assert "adds no observations" in md
    assert "Independent samples added by composition: **0**" in md
    assert "complete-corpus repetition aggregates" in md
    assert "not an executed shared-state long task" in md
    assert "independent observed episodes" not in md


def test_error_cell_is_disclosed_without_becoming_an_observation(tmp_path):
    outcomes = {f"t{i:02d}": [(False, True)] * 12 for i in range(17)}
    path = _report(tmp_path, outcomes)
    raw = json.loads(path.read_text())
    raw["reps"] = 12
    error = next(
        record
        for record in raw["records"]
        if record["task"] == "t00"
        and record["condition"] == "trust"
        and record["rep"] == 0
    )
    error.update(
        status="ERROR",
        claimed_success=False,
        artifact_correct=False,
        true_success=False,
    )
    path.write_text(json.dumps(raw))

    cells = load_cells(path)
    assert cells.reps == list(range(12))

    report = build_report(cells)
    assert report.coverage.scheduled_paired_cells == 204
    assert report.coverage.usable_paired_cells == 203
    assert report.coverage.unavailable_or_error_cells == 1
    assert report.coverage.scheduled_repetitions == 12
    assert report.coverage.complete_paired_repetitions == 11
    assert report.independent_episode_count == 11
    assert report.cell_estimand.pairs == 203
    assert report.episode_estimand.pairs == 11

    composition = report.composition_estimand
    assert composition.independent_samples_added == 0
    assert composition.per_task_n["trust-chain"]["t00"] == 11
    assert composition.per_task_n["verify-chain"]["t00"] == 12
    assert composition.per_task_n["trust-chain"]["t01"] == 12

    payload = json.loads(report.to_json())
    assert payload["coverage"] == {
        "scheduled_paired_cells": 204,
        "usable_paired_cells": 203,
        "unavailable_or_error_cells": 1,
        "scheduled_repetitions": 12,
        "complete_paired_repetitions": 11,
    }
    assert payload["estimands"]["composition"]["independent_samples_added"] == 0
    assert payload["estimands"]["composition"]["per_task_n"]["trust-chain"]["t00"] == 11

    markdown = report.to_markdown()
    assert "scheduled paired cells **204**" in markdown
    assert "usable paired cells **203**" in markdown
    assert "unavailable/error cells **1**" in markdown
    assert "complete paired repetitions **11**" in markdown
    assert "`t00` | 0% (n=11) | 100% (n=12)" in markdown
    cell_line = next(
        line for line in markdown.splitlines() if line.startswith("Task-cluster inference")
    )
    assert "0.0000" not in cell_line
    assert "e-" in cell_line


def test_task_cluster_and_repetition_aggregate_p_can_differ(tmp_path):
    """Eight task effects in one rep collapse into one aggregate disagreement."""
    outcomes = {f"t{i:02d}": [(False, True), (True, True)] for i in range(8)}
    report = build_report(load_cells(_report(tmp_path, outcomes)))

    assert report.cell_estimand.pairs == 16
    assert report.cell_estimand.discordant == (8, 0)
    assert report.cell_estimand.task_clusters == 8
    assert report.cell_estimand.nonzero_task_clusters == 8
    assert report.cell_estimand.task_cluster_sign_flip_p == pytest.approx(0.0078125)

    assert report.independent_episode_count == 2
    assert report.episode_estimand.pairs == 2
    assert report.episode_estimand.discordant == (1, 0)
    assert report.episode_estimand.mcnemar_p == pytest.approx(mcnemar_exact(1, 0))
    assert report.episode_estimand.mcnemar_p == pytest.approx(1.0)
    assert (
        report.cell_estimand.task_cluster_sign_flip_p
        != report.episode_estimand.mcnemar_p
    )


def test_cell_inference_does_not_treat_repetitions_of_one_task_as_independent(
    tmp_path,
):
    outcomes = {
        "repeated_failure": [(False, True)] * 12,
        "unchanged": [(True, True)] * 12,
    }
    report = build_report(load_cells(_report(tmp_path, outcomes)))

    assert report.cell_estimand.pairs == 24
    assert report.cell_estimand.discordant == (12, 0)
    assert mcnemar_exact(12, 0) == pytest.approx(0.00048828125)
    assert report.cell_estimand.task_clusters == 2
    assert report.cell_estimand.nonzero_task_clusters == 1
    assert report.cell_estimand.task_cluster_sign_flip_p == pytest.approx(1.0)

    payload = json.loads(report.to_json())
    cell = payload["estimands"]["cell"]
    assert "mcnemar_p" not in cell
    assert cell["task_cluster_sign_flip_p"] == pytest.approx(1.0)
    assert "no cell-level inferential p" not in report.to_markdown()
    assert "do not receive separate inferential weight" in report.to_markdown()


def test_per_task_p_averages_over_reps(tmp_path):
    cells = load_cells(_report(tmp_path, {"a": [(True, True), (False, True), (True, True)]}))
    assert per_task_p(cells, "trust")["a"] == pytest.approx(2 / 3)
    assert per_task_p(cells, "verify")["a"] == pytest.approx(1.0)


def test_task_bootstrap_interval_brackets_the_projection_and_is_seeded(tmp_path):
    outcomes = {f"t{i:02d}": [(True, True)] * 3 for i in range(17)}
    outcomes["t05"][0] = (False, True)
    path = _report(tmp_path, outcomes)
    a = build_report(load_cells(path), seed=7)
    b = build_report(load_cells(path), seed=7)
    trust_a = next(c for c in a.composition_estimand.curves if c.condition == "trust-chain")
    trust_b = next(c for c in b.composition_estimand.curves if c.condition == "trust-chain")
    assert trust_a.task_bootstrap_lo == trust_b.task_bootstrap_lo
    assert trust_a.task_bootstrap_hi == trust_b.task_bootstrap_hi
    for k in range(a.n_steps):
        assert trust_a.task_bootstrap_lo[k] <= trust_a.rate[k]
        assert trust_a.rate[k] <= trust_a.task_bootstrap_hi[k]


def test_run_horizon_writes_all_artifacts_and_explicit_estimands(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)], "b": [(False, True)]})
    report = run_horizon(path, tmp_path / "out")
    assert (tmp_path / "out" / "horizon_report.md").exists()
    assert (tmp_path / "out" / "horizon_report.json").exists()
    assert (tmp_path / "out" / "horizon_curve.svg").exists()
    reloaded = json.loads((tmp_path / "out" / "horizon_report.json").read_text())
    assert reloaded["n_steps"] == 2
    assert reloaded["independent_episode_count"] == 1
    assert reloaded["coverage"] == {
        "scheduled_paired_cells": 2,
        "usable_paired_cells": 2,
        "unavailable_or_error_cells": 0,
        "scheduled_repetitions": 1,
        "complete_paired_repetitions": 1,
    }
    assert reloaded["estimands"]["cell"]["discordant"] == [1, 0]
    assert reloaded["estimands"]["cell"]["task_clusters"] == 2
    assert reloaded["estimands"]["cell"]["nonzero_task_clusters"] == 1
    assert reloaded["estimands"]["cell"]["task_cluster_sign_flip_p"] == pytest.approx(1.0)
    assert "mcnemar_p" not in reloaded["estimands"]["cell"]
    # Legacy JSON names remain stable, but their records are repetition aggregates.
    assert reloaded["estimands"]["episode"]["discordant"] == [1, 0]
    assert reloaded["estimands"]["composition"]["independent_samples_added"] == 0
    assert reloaded["estimands"]["composition"]["per_task_n"] == {
        "trust-chain": {"a": 1, "b": 1},
        "verify-chain": {"a": 1, "b": 1},
    }
    assert "mcnemar_p" not in reloaded["estimands"]["composition"]
    assert report.episode_estimand.mcnemar_p == pytest.approx(1.0)
    markdown = (tmp_path / "out" / "horizon_report.md").read_text()
    assert "not an executed shared-state long task" in markdown
    assert "independent observed episodes" not in markdown
    svg = (tmp_path / "out" / "horizon_curve.svg").read_text()
    assert "0 added observations" in svg
    assert "added episodes" not in svg


def test_legacy_schema_keeps_historical_episode_rendering_without_new_json_fields(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)], "b": [(False, True)]})
    raw = json.loads(path.read_text())
    raw["schema_version"] = 2
    path.write_text(json.dumps(raw))

    report = run_horizon(path, tmp_path / "legacy")

    assert report.source_schema_version == 2
    payload = json.loads((tmp_path / "legacy" / "horizon_report.json").read_text())
    assert "source_schema_version" not in payload
    assert payload["independent_episode_count"] == 1
    assert payload["estimands"]["cell"]["mcnemar_p"] == pytest.approx(1.0)
    assert "task_cluster_sign_flip_p" not in payload["estimands"]["cell"]
    markdown = (tmp_path / "legacy" / "horizon_report.md").read_text()
    assert "independent observed episodes" in markdown
    assert "complete-corpus repetition aggregates" not in markdown
    svg = (tmp_path / "legacy" / "horizon_curve.svg").read_text()
    assert "0 added episodes" in svg
    assert "0 added observations" not in svg


# --- Wilson intervals ---------------------------------------------------------
def test_wilson_does_not_collapse_at_the_boundaries():
    # 0/51 -> 0%-7.0%: the same order as the rule of three (3/51 ~ 5.9%), and
    # nothing like the 0%-0% a percentile bootstrap reports here.
    lo, hi = wilson_interval(0, 51)
    assert lo == 0.0 and hi == pytest.approx(0.0700, abs=5e-4)
    lo, hi = wilson_interval(51, 51)
    assert hi == 1.0 and lo == pytest.approx(1 - 0.0700, abs=5e-4)  # symmetric at the other end


def test_wilson_is_centred_and_validated():
    lo, hi = wilson_interval(25, 50)
    assert lo < 0.5 < hi
    assert (0.5 - lo) == pytest.approx(hi - 0.5, abs=1e-9)  # symmetric at p = 0.5
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
