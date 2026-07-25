"""The horizon analysis: exact compounding, honest episodes, no invented power.

The point of these tests is the last one. Composing measured cells into a
longer horizon must re-express the evidence without inflating it: the paired
test at the terminal step has to return exactly what the same cells return at
the step level, no matter how many orderings the curve averages over.
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
                {"task": task, "condition": "trust", "rep": rep, "status": "DONE",
                 "true_success": trust}
            )
            records.append(
                {"task": task, "condition": "verify", "rep": rep, "status": "DONE",
                 "true_success": verify}
            )
    path = tmp_path / "ablation_report.json"
    path.write_text(json.dumps({"tasks": list(outcomes), "model": model, "records": records}))
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
        {"task": "a", "condition": "trust", "rep": 1, "status": "ERROR", "true_success": False}
    )
    path.write_text(json.dumps(raw))
    cells = load_cells(path)
    assert ("trust", "a", 1) not in cells.outcome  # an ERROR is not a measurement
    assert cells.complete_reps("trust") == [0]


def test_load_cells_rejects_an_empty_report(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"tasks": [], "records": []}))
    with pytest.raises(HorizonDataError):
        load_cells(path)


def test_incomplete_rep_is_not_a_shorter_episode(tmp_path):
    # rep 1 is missing task "b": it must be excluded, not scored over one step.
    path = _report(tmp_path, {"a": [(True, True), (True, True)], "b": [(True, True)]})
    cells = load_cells(path)
    assert cells.complete_reps("trust") == [0]
    assert [e.rep for e in episodes_for(cells, "trust-chain", "trust")] == [0]


# --- episodes -----------------------------------------------------------------
def test_episode_is_one_repetition_of_the_whole_corpus(tmp_path):
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


# --- the property that matters: composition must not invent power -------------
def test_composition_does_not_manufacture_significance(tmp_path):
    """A longer horizon re-expresses the same events; it must not add evidence.

    Three repetitions of a 17-task corpus with two wrong first attempts give
    b=2, c=0 at the terminal step — the same discordance, and so the same exact
    p-value, that the underlying per-cell comparison gives.
    """
    outcomes = {f"t{i:02d}": [(True, True)] * 3 for i in range(17)}
    outcomes["t05"][0] = (False, True)  # wrong in rep 0, repaired by verify
    outcomes["t11"][1] = (False, True)  # wrong in rep 1, repaired by verify
    report = build_report(load_cells(_report(tmp_path, outcomes)))

    assert report.n_steps == 17 and report.reps == 3
    assert report.discordant == (2, 0)
    assert report.mcnemar_p == pytest.approx(mcnemar_exact(2, 0))
    assert report.mcnemar_p == pytest.approx(0.5)  # unchanged by the horizon framing
    # ...while the effect SIZE is legitimately much larger at the terminal step.
    trust = next(c for c in report.curves if c.condition == "trust-chain")
    verify = next(c for c in report.curves if c.condition == "verify-chain")
    assert trust.rate[0] == pytest.approx(49 / 51)  # per-step, as measured
    assert trust.rate[-1] == pytest.approx((2 / 3) ** 2)  # 17 steps: 44.4%
    assert verify.rate[-1] == pytest.approx(1.0)


def test_report_states_how_many_reps_reach_significance(tmp_path):
    outcomes = {f"t{i:02d}": [(True, True)] * 3 for i in range(17)}
    outcomes["t05"][0] = (False, True)
    outcomes["t11"][1] = (False, True)
    report = build_report(load_cells(_report(tmp_path, outcomes)))
    assert report.reps_for_alpha is not None and report.reps_for_alpha > report.reps
    md = report.to_markdown()
    assert "not significant" in md
    assert "repetitions" in md
    assert "cannot create information" in md  # the honesty note is not optional


def test_no_extrapolation_needed_when_already_significant(tmp_path):
    outcomes = {f"t{i:02d}": [(True, True)] * 8 for i in range(4)}
    for rep in range(7):  # 7 of 8 episodes discordant, all one direction
        outcomes["t00"][rep] = (False, True)
    report = build_report(load_cells(_report(tmp_path, outcomes)))
    assert report.mcnemar_p < 0.05
    assert report.reps_for_alpha is None


def test_per_task_p_averages_over_reps(tmp_path):
    cells = load_cells(_report(tmp_path, {"a": [(True, True), (False, True), (True, True)]}))
    assert per_task_p(cells, "trust")["a"] == pytest.approx(2 / 3)
    assert per_task_p(cells, "verify")["a"] == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_estimate_and_is_seeded(tmp_path):
    outcomes = {f"t{i:02d}": [(True, True)] * 3 for i in range(17)}
    outcomes["t05"][0] = (False, True)
    path = _report(tmp_path, outcomes)
    a = build_report(load_cells(path), seed=7)
    b = build_report(load_cells(path), seed=7)
    trust_a = next(c for c in a.curves if c.condition == "trust-chain")
    trust_b = next(c for c in b.curves if c.condition == "trust-chain")
    assert trust_a.ci_lo == trust_b.ci_lo and trust_a.ci_hi == trust_b.ci_hi  # deterministic
    for k in range(a.n_steps):
        assert trust_a.ci_lo[k] <= trust_a.rate[k] <= trust_a.ci_hi[k]


def test_run_horizon_writes_both_artifacts(tmp_path):
    path = _report(tmp_path, {"a": [(True, True)], "b": [(False, True)]})
    report = run_horizon(path, tmp_path / "out")
    assert (tmp_path / "out" / "horizon_report.md").exists()
    assert (tmp_path / "out" / "horizon_report.json").exists()
    reloaded = json.loads((tmp_path / "out" / "horizon_report.json").read_text())
    assert reloaded["n_steps"] == 2
    assert reloaded["discordant"] == [1, 0]
    assert report.mcnemar_p == pytest.approx(1.0)  # a single discordant pair proves nothing


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
