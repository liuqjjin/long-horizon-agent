"""Describe error compounding without conflating three statistical questions.

``lha ablate`` produces paired delivered-correctness labels for each
``(task, repetition)`` cell. This module reports three estimands from those
measurements:

1. **Cell level** — how often the same task attempt succeeds under ``trust`` and
   ``verify``. Its paired unit is one ``(task, repetition)`` cell.
2. **Episode level** — how often an entire corpus repetition succeeds end to
   end. Its paired unit is one complete repetition, so ``R`` repetitions are
   exactly ``R`` independent observed episodes.
3. **Composition** — the survival curve obtained by inserting empirical
   per-task rates into an independent-step model. It is a descriptive
   projection, not another experiment, and adds zero independent samples.

Cell and episode McNemar tests answer different questions and need not have the
same p-value: several discordant cells can collapse into one discordant episode.
The composition has no McNemar test at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import comb
from pathlib import Path
from random import Random

from .bench.stats import mcnemar_exact, wilson_interval

# horizon condition -> the ablation condition whose truth it reads
CONDITIONS: list[tuple[str, str, str]] = [
    ("trust-chain", "trust", "no gate: a wrong step is accepted silently"),
    ("verify-chain", "verify", "gate plus repair at every step"),
]

_BOOTSTRAP_N = 2_000
_TARGET_ALPHA = 0.05
_ABLATION_CONDITIONS = frozenset({"trust", "gate", "verify"})
_RECORD_STATUSES = frozenset({"DONE", "FAILED", "ERROR"})


class HorizonDataError(ValueError):
    """The measured cells cannot support a horizon analysis."""


@dataclass(frozen=True)
class Cells:
    """Scheduled cells and their usable delivered-correctness measurements."""

    tasks: list[str]
    # Scheduled repetitions include ERROR-only and declared-but-missing cells.
    # Keeping them here prevents coverage loss from disappearing during load.
    reps: list[int]
    outcome: dict[tuple[str, str, int], bool]
    model: str
    source: str

    def truth(self, condition: str, task: str, rep: int) -> bool:
        return self.outcome[(condition, task, rep)]

    def complete_reps(self, condition: str) -> list[int]:
        """Reps with a measured outcome for every task — i.e. usable episodes.

        A rep missing any task is not a shorter episode, it is an incomplete
        one; counting it would quietly change the horizon length.
        """
        return [
            rep
            for rep in self.reps
            if all((condition, task, rep) in self.outcome for task in self.tasks)
        ]


@dataclass
class Episode:
    """One repetition of the whole corpus, run under one condition."""

    condition: str
    rep: int
    n_steps: int
    steps_correct: int
    end_to_end: bool
    failing_tasks: list[str] = field(default_factory=list)


@dataclass
class Curve:
    """Descriptive P(chain still correct through step k), k = 1..n."""

    condition: str
    rate: list[float]
    task_bootstrap_lo: list[float]
    task_bootstrap_hi: list[float]


@dataclass(frozen=True)
class PairedEstimand:
    """A paired binary comparison at one explicitly named unit of analysis."""

    unit: str
    pairs: int
    trust_successes: int
    verify_successes: int
    # (verify succeeds / trust fails, trust succeeds / verify fails)
    discordant: tuple[int, int]
    mcnemar_p: float


@dataclass
class CompositionEstimand:
    """A model-derived curve; it contributes no additional observations."""

    unit: str
    independent_samples_added: int
    per_task_p: dict[str, dict[str, float]]
    per_task_n: dict[str, dict[str, int]]
    curves: list[Curve]


@dataclass(frozen=True)
class HorizonCoverage:
    """Scheduled analysis units and the subset with usable paired evidence."""

    scheduled_paired_cells: int
    usable_paired_cells: int
    unavailable_or_error_cells: int
    scheduled_repetitions: int
    complete_paired_repetitions: int


@dataclass
class HorizonReport:
    tasks: list[str]
    n_steps: int
    independent_episode_count: int
    model: str
    source: str
    coverage: HorizonCoverage
    cell_estimand: PairedEstimand
    episode_estimand: PairedEstimand
    composition_estimand: CompositionEstimand
    episodes: list[Episode]
    alpha: float = _TARGET_ALPHA

    # --- rendering ---------------------------------------------------------
    def to_markdown(self) -> str:
        by_cond = {c.condition: c for c in self.composition_estimand.curves}
        trust, verify = by_cond["trust-chain"], by_cond["verify-chain"]
        n = self.n_steps
        cell = self.cell_estimand
        episode = self.episode_estimand
        cell_b, cell_c = cell.discordant
        episode_b, episode_c = episode.discordant
        lines = [
            "# Error compounding over a horizon",
            "",
            f"corpus: {n} independent subtasks · model: `{self.model or '(backend default)'}` · "
            f"complete paired repetitions: {self.independent_episode_count} → "
            f"**{self.independent_episode_count} independent observed episodes** · "
            f"per-step truth from `{self.source}`",
            "",
            "Coverage: "
            f"scheduled paired cells **{self.coverage.scheduled_paired_cells}** · "
            f"usable paired cells **{self.coverage.usable_paired_cells}** · "
            "unavailable/error cells "
            f"**{self.coverage.unavailable_or_error_cells}** · "
            f"scheduled repetitions **{self.coverage.scheduled_repetitions}** · "
            "complete paired repetitions "
            f"**{self.coverage.complete_paired_repetitions}**.",
            "",
            "This report keeps three estimands separate. The cell and episode tests use "
            "different paired units; the composition is a descriptive model projection "
            "and adds no observations.",
            "",
            "## Estimand 1 — paired cells",
            "",
            f"Unit: `{cell.unit}` · pairs: **{cell.pairs}** · "
            f"`trust` true success: {cell.trust_successes}/{cell.pairs} · "
            f"`verify` true success: {cell.verify_successes}/{cell.pairs}.",
            "",
            f"Discordant cells (verify-only / trust-only): {cell_b}/{cell_c} · "
            f"exact McNemar p = {_format_p(cell.mcnemar_p)}"
            + ("" if cell.mcnemar_p < self.alpha else " — **not significant**"),
            "",
            "## Estimand 2 — observed episodes",
            "",
            "An episode is one complete corpus repetition and is correct only if every "
            "subtask in that repetition truly succeeded. Multiple failed cells in the "
            "same repetition still make one failed episode.",
            "",
            "| condition | end-to-end correct | first failing subtask(s) |",
            "|---|---|---|",
        ]
        for name, _src, _blurb in CONDITIONS:
            eps = [e for e in self.episodes if e.condition == name]
            ok = sum(e.end_to_end for e in eps)
            failing = sorted({t for e in eps for t in e.failing_tasks})
            lo, hi = wilson_interval(ok, len(eps)) if eps else (0.0, 0.0)
            lines.append(
                f"| `{name}` | {ok}/{len(eps)} ({_pct(lo)}–{_pct(hi)}, Wilson) "
                f"| {', '.join(f'`{t}`' for t in failing) or '—'} |"
            )
        lines += [
            "",
            f"Discordant episodes (verify-only / trust-only): {episode_b}/{episode_c} of "
            f"{episode.pairs} paired episodes · exact McNemar p = "
            f"{_format_p(episode.mcnemar_p)}"
            + ("" if episode.mcnemar_p < self.alpha else " — **not significant**"),
            "",
            "The cell- and episode-level p-values may coincide for a particular dataset, "
            "but equality is not a statistical contract: aggregation changes the paired "
            "unit and can collapse many cell disagreements into one episode disagreement.",
            "",
            "## Estimand 3 — descriptive composition",
            "",
            "The curve inserts empirical per-task success rates into an independent-step, "
            "uniform-random-order model. Its task-bootstrap interval describes sensitivity "
            "to the observed task mix; it is not an episode confidence interval and has no "
            "McNemar p-value.",
            "",
            f"Independent samples added by composition: "
            f"**{self.composition_estimand.independent_samples_added}**.",
            "",
            "Composition uses every available measurement for each task. Per-task "
            "sample sizes may differ after ERROR or missing cells; these measurements "
            "are reused descriptively and do not add observations.",
            "",
            "| task | `trust-chain` measured rate | `verify-chain` measured rate |",
            "|---|---:|---:|",
        ]
        for task in self.tasks:
            trust_p = self.composition_estimand.per_task_p["trust-chain"][task]
            verify_p = self.composition_estimand.per_task_p["verify-chain"][task]
            trust_n = self.composition_estimand.per_task_n["trust-chain"][task]
            verify_n = self.composition_estimand.per_task_n["verify-chain"][task]
            lines.append(
                f"| `{task}` | {_pct(trust_p)} (n={trust_n}) "
                f"| {_pct(verify_p)} (n={verify_n}) |"
            )
        lines += [
            "",
            "| k | `trust-chain` (95% task-bootstrap interval) "
            "| `verify-chain` (95% task-bootstrap interval) | gap |",
            "|---:|---|---|---:|",
        ]
        for k in _milestones(n):
            i = k - 1
            gap = 100 * (verify.rate[i] - trust.rate[i])
            lines.append(
                f"| {k} | {_pct(trust.rate[i])} "
                f"({_pct(trust.task_bootstrap_lo[i])}–"
                f"{_pct(trust.task_bootstrap_hi[i])}) "
                f"| {_pct(verify.rate[i])} "
                f"({_pct(verify.task_bootstrap_lo[i])}–"
                f"{_pct(verify.task_bootstrap_hi[i])}) "
                f"| {gap:+.1f} pp |"
            )
        lines += [
            "",
            "Conditions:",
        ]
        for name, _src, blurb in CONDITIONS:
            lines.append(f"- `{name}` — {blurb}.")
        lines += [
            "",
            "Only new complete repetitions increase the episode sample count. Reordering "
            "or composing the existing cells changes the projected effect size, not the "
            "number of independent observed episodes.",
            "",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "tasks": self.tasks,
                "n_steps": self.n_steps,
                "independent_episode_count": self.independent_episode_count,
                "model": self.model,
                "source": self.source,
                "coverage": asdict(self.coverage),
                "estimands": {
                    "cell": asdict(self.cell_estimand),
                    "episode": asdict(self.episode_estimand),
                    "composition": asdict(self.composition_estimand),
                },
                "episodes": [asdict(e) for e in self.episodes],
                "alpha": self.alpha,
            },
            indent=2,
        )


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _format_p(value: float) -> str:
    """Keep ordinary p-values readable without rounding a non-zero tail to zero."""
    if 0 < value < 0.0001:
        return f"{value:.4e}"
    return f"{value:.4f}"


def _milestones(n: int) -> list[int]:
    """A readable subset of step indices, always including 1 and n."""
    wanted = {1, n}
    wanted.update(k for k in (2, 4, 6, 8, 10, 12, 14, 16, 20) if k < n)
    return sorted(wanted)


# --- loading -----------------------------------------------------------------
def load_cells(report_path: str | Path) -> Cells:
    """Read measured per-cell truth out of an ``ablation_report.json``.

    ``ERROR`` cells carry no measurement, but their scheduled repetition remains
    visible for coverage accounting. A repetition that loses any task is
    excluded from the episode analysis by ``complete_reps`` rather than silently
    shortening the horizon.
    """
    path = Path(report_path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise HorizonDataError(f"unreadable ablation report {path}: {e}") from e

    if not isinstance(raw, dict):
        raise HorizonDataError(f"{path} must contain a JSON object")
    schema_raw = raw.get("schema_version", 1)
    if not isinstance(schema_raw, int) or isinstance(schema_raw, bool) or schema_raw < 1:
        raise HorizonDataError(f"{path} has an invalid schema_version")
    schema_version = schema_raw
    tasks_raw = raw.get("tasks")
    if (
        not isinstance(tasks_raw, list)
        or not tasks_raw
        or not all(isinstance(task, str) and task for task in tasks_raw)
        or len(tasks_raw) != len(set(tasks_raw))
    ):
        raise HorizonDataError(f"{path} has an invalid task list")
    records = raw.get("records")
    if not isinstance(records, list):
        raise HorizonDataError(f"{path} has an invalid records list")

    tasks = sorted(tasks_raw)
    declared_tasks = set(tasks)
    outcome: dict[tuple[str, str, int], bool] = {}
    observed_reps: set[int] = set()
    declared_reps_raw = raw.get("reps")
    if declared_reps_raw is None:
        declared_reps: int | None = None
    elif (
        not isinstance(declared_reps_raw, int)
        or isinstance(declared_reps_raw, bool)
        or declared_reps_raw <= 0
    ):
        raise HorizonDataError(f"{path} has an invalid repetition count")
    else:
        declared_reps = declared_reps_raw
    seen: set[tuple[str, str, int]] = set()
    for index, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise HorizonDataError(f"{path} record {index} must be a JSON object")
        condition = rec.get("condition")
        task = rec.get("task")
        rep = rec.get("rep")
        status = rec.get("status")
        true_success = rec.get("true_success")
        if not isinstance(condition, str) or condition not in _ABLATION_CONDITIONS:
            raise HorizonDataError(f"{path} record {index} has invalid condition {condition!r}")
        if not isinstance(task, str) or task not in declared_tasks:
            raise HorizonDataError(f"{path} record {index} has invalid task {task!r}")
        if not isinstance(rep, int) or isinstance(rep, bool) or rep < 0:
            raise HorizonDataError(f"{path} record {index} has invalid repetition {rep!r}")
        if declared_reps is not None and rep >= declared_reps:
            raise HorizonDataError(
                f"{path} record {index} repetition {rep} exceeds the declared schedule"
            )
        if not isinstance(status, str) or status not in _RECORD_STATUSES:
            raise HorizonDataError(f"{path} record {index} has invalid status {status!r}")
        if not isinstance(true_success, bool):
            raise HorizonDataError(
                f"{path} record {index} field 'true_success' must be boolean"
            )
        if schema_version >= 4:
            claimed_success = rec.get("claimed_success")
            artifact_correct = rec.get("artifact_correct")
            if not isinstance(claimed_success, bool):
                raise HorizonDataError(
                    f"{path} record {index} field 'claimed_success' must be boolean"
                )
            if not isinstance(artifact_correct, bool):
                raise HorizonDataError(
                    f"{path} record {index} field 'artifact_correct' must be boolean"
                )
            if true_success != (claimed_success and artifact_correct):
                raise HorizonDataError(
                    f"{path} record {index} has inconsistent delivered correctness"
                )
        else:
            # Historical schemas used true_success for artifact correctness.
            # If the delivery decision is present, recover the chain outcome;
            # otherwise keep the historical value for read-only compatibility.
            claimed_success = rec.get("claimed_success")
            if isinstance(claimed_success, bool):
                true_success = claimed_success and true_success

        key = (condition, task, rep)
        if key in seen:
            raise HorizonDataError(f"{path} contains duplicate measured cell {key!r}")
        seen.add(key)
        observed_reps.add(rep)
        if status == "ERROR":
            continue
        outcome[key] = true_success
    if not tasks or not outcome:
        raise HorizonDataError(f"{path} contains no usable measured cells")
    reps = list(range(declared_reps)) if declared_reps is not None else sorted(observed_reps)
    model = raw.get("model", "")
    if not isinstance(model, str):
        raise HorizonDataError(f"{path} field 'model' must be a string")
    return Cells(
        tasks=tasks,
        reps=reps,
        outcome=outcome,
        model=model,
        source=str(path),
    )


# --- the compounding model ----------------------------------------------------
def per_task_p(cells: Cells, ablation_condition: str) -> dict[str, float]:
    """Measured first-attempt truth rate per task, over that task's repetitions."""
    out: dict[str, float] = {}
    for task in cells.tasks:
        vals = [
            cells.truth(ablation_condition, task, rep)
            for rep in cells.reps
            if (ablation_condition, task, rep) in cells.outcome
        ]
        if not vals:
            raise HorizonDataError(f"no measured cells for {task!r} under {ablation_condition!r}")
        out[task] = sum(vals) / len(vals)
    return out


def per_task_n(cells: Cells, ablation_condition: str) -> dict[str, int]:
    """Number of usable measurements contributing to each per-task rate."""
    return {
        task: sum((ablation_condition, task, rep) in cells.outcome for rep in cells.reps)
        for task in cells.tasks
    }


def compounding_curve(probabilities: list[float]) -> list[float]:
    """P(all of the first k steps succeed), k = 1..n, over a uniformly random order.

    The first k of a uniformly random permutation is a uniformly random
    k-subset, so the answer is the degree-k elementary symmetric polynomial of
    the per-step probabilities divided by the number of k-subsets. Computed
    exactly by the standard DP — no sampling, so the curve carries no Monte
    Carlo noise of its own.
    """
    n = len(probabilities)
    if n == 0:
        return []
    e = [0.0] * (n + 1)
    e[0] = 1.0
    for p in probabilities:
        for j in range(n, 0, -1):
            e[j] += p * e[j - 1]
    return [e[k] / comb(n, k) for k in range(1, n + 1)]


def _task_bootstrap_interval(
    probabilities_by_task: dict[str, float], *, n: int = _BOOTSTRAP_N, seed: int = 0
) -> tuple[list[float], list[float]]:
    """Per-k sensitivity interval from resampling tasks with replacement.

    Tasks are the exchangeable unit: repetitions of one task are correlated, and
    here the whole effect is carried by however many tasks the model gets wrong,
    so this describes how the projection changes with the observed task mix. It
    does not add episodes or supply an episode-level confidence interval.
    """
    tasks = list(probabilities_by_task)
    steps = len(tasks)
    rng = Random(seed)
    draws: list[list[float]] = []
    for _ in range(n):
        sample = [probabilities_by_task[tasks[rng.randrange(steps)]] for _ in range(steps)]
        draws.append(compounding_curve(sample))
    lo_idx, hi_idx = int(0.025 * (n - 1)), int(0.975 * (n - 1))
    lo: list[float] = []
    hi: list[float] = []
    for k in range(steps):
        column = sorted(d[k] for d in draws)
        lo.append(column[lo_idx])
        hi.append(column[hi_idx])
    return lo, hi


# --- observed episodes --------------------------------------------------------
def episodes_for(cells: Cells, condition: str, ablation_condition: str) -> list[Episode]:
    """One episode per complete repetition of the corpus."""
    out: list[Episode] = []
    n = len(cells.tasks)
    for rep in cells.complete_reps(ablation_condition):
        failing = [t for t in cells.tasks if not cells.truth(ablation_condition, t, rep)]
        out.append(
            Episode(
                condition=condition,
                rep=rep,
                n_steps=n,
                steps_correct=n - len(failing),
                end_to_end=not failing,
                failing_tasks=failing,
            )
        )
    return out


def _paired_estimand(unit: str, pairs: list[tuple[bool, bool]]) -> PairedEstimand:
    """Summarize paired ``(trust, verify)`` outcomes at one unit of analysis."""
    trust_successes = sum(trust for trust, _verify in pairs)
    verify_successes = sum(verify for _trust, verify in pairs)
    verify_only = sum(verify and not trust for trust, verify in pairs)
    trust_only = sum(trust and not verify for trust, verify in pairs)
    return PairedEstimand(
        unit=unit,
        pairs=len(pairs),
        trust_successes=trust_successes,
        verify_successes=verify_successes,
        discordant=(verify_only, trust_only),
        mcnemar_p=mcnemar_exact(verify_only, trust_only),
    )


def build_report(cells: Cells, *, seed: int = 0, alpha: float = _TARGET_ALPHA) -> HorizonReport:
    curves: list[Curve] = []
    episodes: list[Episode] = []
    probabilities: dict[str, dict[str, float]] = {}
    sample_sizes: dict[str, dict[str, int]] = {}
    for name, source, _blurb in CONDITIONS:
        probs = per_task_p(cells, source)
        probabilities[name] = probs
        sample_sizes[name] = per_task_n(cells, source)
        rate = compounding_curve(list(probs.values()))
        lo, hi = _task_bootstrap_interval(probs, seed=seed)
        curves.append(
            Curve(
                condition=name,
                rate=rate,
                task_bootstrap_lo=lo,
                task_bootstrap_hi=hi,
            )
        )
        episodes.extend(episodes_for(cells, name, source))

    cell_pairs = [
        (cells.truth("trust", task, rep), cells.truth("verify", task, rep))
        for task in cells.tasks
        for rep in cells.reps
        if ("trust", task, rep) in cells.outcome and ("verify", task, rep) in cells.outcome
    ]
    if not cell_pairs:
        raise HorizonDataError("no paired trust/verify cells")

    trust_eps = {e.rep: e.end_to_end for e in episodes if e.condition == "trust-chain"}
    verify_eps = {e.rep: e.end_to_end for e in episodes if e.condition == "verify-chain"}
    paired = sorted(set(trust_eps) & set(verify_eps))
    if not paired:
        raise HorizonDataError("no complete repetition is paired under trust and verify")
    episode_pairs = [(trust_eps[rep], verify_eps[rep]) for rep in paired]
    paired_episodes = [episode for episode in episodes if episode.rep in paired]
    scheduled_cell_pairs = len(cells.tasks) * len(cells.reps)
    usable_cell_pairs = len(cell_pairs)

    return HorizonReport(
        tasks=cells.tasks,
        n_steps=len(cells.tasks),
        independent_episode_count=len(paired),
        model=cells.model,
        source=cells.source,
        coverage=HorizonCoverage(
            scheduled_paired_cells=scheduled_cell_pairs,
            usable_paired_cells=usable_cell_pairs,
            unavailable_or_error_cells=scheduled_cell_pairs - usable_cell_pairs,
            scheduled_repetitions=len(cells.reps),
            complete_paired_repetitions=len(paired),
        ),
        cell_estimand=_paired_estimand("task × repetition cell", cell_pairs),
        episode_estimand=_paired_estimand("complete corpus repetition", episode_pairs),
        composition_estimand=CompositionEstimand(
            unit="independent-step projection over empirical per-task rates",
            independent_samples_added=0,
            per_task_p=probabilities,
            per_task_n=sample_sizes,
            curves=curves,
        ),
        episodes=paired_episodes,
        alpha=alpha,
    )


# --- figure ------------------------------------------------------------------
_W, _H = 720, 380
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 62, 130, 26, 46
_INK = "#3d444d"
_TRUST = "#c9432f"
_VERIFY = "#1f7a4d"


def _svg(report: HorizonReport) -> str:
    """A dependency-free line chart of the two curves.

    Written by hand rather than through a plotting library so the figure is
    regenerable from the committed report by anyone who clones the repo, with no
    extra install. Explicit light panel so it reads on both GitHub themes.
    """
    by = {c.condition: c for c in report.composition_estimand.curves}
    trust, verify = by["trust-chain"], by["verify-chain"]
    n = report.n_steps
    plot_w = _W - _PAD_L - _PAD_R
    plot_h = _H - _PAD_T - _PAD_B

    def x(k: int) -> float:
        return _PAD_L + (plot_w * (k - 1) / max(n - 1, 1))

    def y(v: float) -> float:
        return _PAD_T + plot_h * (1.0 - v)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{_W}" height="{_H}" fill="#ffffff" rx="6"/>',
    ]
    # gridlines + y axis labels
    for pct in range(0, 101, 20):
        gy = y(pct / 100)
        parts.append(
            f'<line x1="{_PAD_L}" y1="{gy:.1f}" x2="{_PAD_L + plot_w}" y2="{gy:.1f}" '
            f'stroke="#e6e8eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{_PAD_L - 10}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="{_INK}">{pct}%</text>'
        )
    # x axis labels
    for k in _milestones(n):
        parts.append(
            f'<text x="{x(k):.1f}" y="{_PAD_T + plot_h + 18:.1f}" text-anchor="middle" '
            f'font-size="11" fill="{_INK}">{k}</text>'
        )
    # trust-chain confidence band
    band = " ".join(f"{x(k):.1f},{y(trust.task_bootstrap_hi[k - 1]):.1f}" for k in range(1, n + 1))
    band += " " + " ".join(
        f"{x(k):.1f},{y(trust.task_bootstrap_lo[k - 1]):.1f}" for k in range(n, 0, -1)
    )
    parts.append(f'<polygon points="{band}" fill="{_TRUST}" fill-opacity="0.10"/>')
    # the two curves
    for curve, colour, dash in ((verify, _VERIFY, ""), (trust, _TRUST, "")):
        pts = " ".join(f"{x(k):.1f},{y(curve.rate[k - 1]):.1f}" for k in range(1, n + 1))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2.4"'
            f'{extra} stroke-linejoin="round"/>'
        )
    # terminal markers + labels
    for curve, colour, label in (
        (verify, _VERIFY, "verify-chain"),
        (trust, _TRUST, "trust-chain"),
    ):
        vy, terminal = y(curve.rate[-1]), curve.rate[-1]
        parts.append(f'<circle cx="{x(n):.1f}" cy="{vy:.1f}" r="3.6" fill="{colour}"/>')
        parts.append(
            f'<text x="{x(n) + 10:.1f}" y="{vy + 4:.1f}" font-size="12" fill="{colour}" '
            f'font-weight="600">{label} {100 * terminal:.0f}%</text>'
        )
    # axes + titles
    parts.append(
        f'<line x1="{_PAD_L}" y1="{_PAD_T + plot_h}" x2="{_PAD_L + plot_w}" '
        f'y2="{_PAD_T + plot_h}" stroke="{_INK}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{_PAD_L}" y1="{_PAD_T}" x2="{_PAD_L}" y2="{_PAD_T + plot_h}" '
        f'stroke="{_INK}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{_PAD_L + plot_w / 2:.1f}" y="{_H - 10}" text-anchor="middle" '
        f'font-size="12" fill="{_INK}">steps completed (k)</text>'
    )
    parts.append(
        f'<text x="14" y="{_PAD_T + plot_h / 2:.1f}" font-size="12" fill="{_INK}" '
        f'transform="rotate(-90 14 {_PAD_T + plot_h / 2:.1f})" text-anchor="middle">'
        "end-to-end still correct</text>"
    )
    parts.append(
        f'<text x="{_PAD_L}" y="{_PAD_T - 10}" font-size="11" fill="#8b949e">'
        f"{n} subtasks · descriptive composition · 0 added episodes</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def run_horizon(report_path: str | Path, out_dir: str | Path, *, seed: int = 0) -> HorizonReport:
    """Compose the horizon analysis from a measured ablation report and write it out."""
    report = build_report(load_cells(report_path), seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "horizon_report.json").write_text(report.to_json())
    (out / "horizon_report.md").write_text(report.to_markdown())
    (out / "horizon_curve.svg").write_text(_svg(report))
    return report
