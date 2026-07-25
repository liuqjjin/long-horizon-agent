"""Error compounding over a horizon: what the per-step gate buys across n steps.

``docs/VERIFICATION_FIRST.md`` models a task as ``n`` sequential steps, each
succeeding independently with probability ``p`` and with no recovery, so
end-to-end success is ``p**n``. ``lha ablate`` measures ``p`` at ``n = 1``. This
module measures the other half of that claim — what the same per-step effect
compounds to over a horizon — and checks it against the model's own prediction.

An **episode** is an ordered sequence of ``k`` independent subtasks: each is its
own repository, its own model call, sharing no state with the others. The
episode is still correct through step ``k`` only if every one of steps ``1..k``
truly succeeded, as judged by the ablation's independent scorer. Because the
subtasks share no state, the attempt drawn at step ``k`` cannot depend on what
happened at steps ``1..k-1`` — so one measured attempt scores under both
conditions and the pairing is exact at every ``k``, not only at ``k = 1``.

Two conditions, both read off the same measurements:

- ``trust-chain``  — no gate. A wrong step is accepted silently; the episode is
  already lost and nothing reports it.
- ``verify-chain`` — gate plus repair at every step.

What this is, stated precisely because the distinction is easy to blur:

- The **curve** is the compounding model evaluated at the measured per-task
  ``p``. It is exact given independence, which holds by construction. It
  re-expresses the measured per-step effect on the axis the thesis argues
  about; it is not a second experiment.
- The **episodes** are the direct observations. One repetition of the whole
  corpus is one independent episode, so ``R`` repetitions give exactly ``R``
  episodes. Composing cells into more orderings does not create information:
  with ``R = 3`` the paired test at the terminal step returns the same p-value
  as the single-step test on the same cells. Raising ``--reps`` on ``lha
  ablate`` is the only thing that changes it.

See ``docs/HORIZON.md`` for the registered prediction and the stopping rule.
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


class HorizonDataError(ValueError):
    """The measured cells cannot support a horizon analysis."""


@dataclass(frozen=True)
class Cells:
    """Per-``(condition, task, rep)`` truth, as measured by the ablation scorer."""

    tasks: list[str]
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
    """P(episode still correct through step k), k = 1..n."""

    condition: str
    rate: list[float]
    ci_lo: list[float]
    ci_hi: list[float]


@dataclass
class HorizonReport:
    tasks: list[str]
    n_steps: int
    reps: int
    model: str
    source: str
    per_task_p: dict[str, float]
    curves: list[Curve]
    episodes: list[Episode]
    discordant: tuple[int, int]
    mcnemar_p: float
    reps_for_alpha: int | None
    alpha: float = _TARGET_ALPHA

    # --- rendering ---------------------------------------------------------
    def to_markdown(self) -> str:
        by_cond = {c.condition: c for c in self.curves}
        trust, verify = by_cond["trust-chain"], by_cond["verify-chain"]
        n = self.n_steps
        lines = [
            "# Error compounding over a horizon",
            "",
            f"corpus: {n} independent subtasks · model: `{self.model or '(backend default)'}` · "
            f"repetitions: {self.reps} → **{self.reps} independent episodes** · "
            f"per-step truth from `{self.source}`",
            "",
            "An episode is correct through step k only if every one of steps 1..k truly "
            "succeeded, as graded by the ablation's independent scorer.",
            "",
            "| k | `trust-chain` (95% CI) | `verify-chain` (95% CI) | gap |",
            "|---:|---|---|---:|",
        ]
        for k in _milestones(n):
            i = k - 1
            gap = 100 * (verify.rate[i] - trust.rate[i])
            lines.append(
                f"| {k} | {_pct(trust.rate[i])} ({_pct(trust.ci_lo[i])}–{_pct(trust.ci_hi[i])}) "
                f"| {_pct(verify.rate[i])} ({_pct(verify.ci_lo[i])}–{_pct(verify.ci_hi[i])}) "
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
            "## Observed episodes",
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
        b, c = self.discordant
        lines += [
            "",
            f"Paired at the terminal step: discordant {b}/{c} of {self.reps} episodes · "
            f"exact McNemar p = {self.mcnemar_p:.4f}"
            + ("" if self.mcnemar_p < self.alpha else " — **not significant**"),
            "",
            "## What this does and does not show",
            "",
            "The curve is the compounding model evaluated at the measured per-task p. "
            "Independence holds by construction (each subtask is its own repository and "
            "its own model call), so the composition is exact — but it is a "
            "re-expression of the per-step measurement, not a second experiment. "
            "Composing cells into more orderings cannot create information.",
            "",
            f"The evidence is the {self.reps} observed episodes. "
            + (
                f"To reach p < {self.alpha} at the observed discordance rate, "
                f"re-run `lha ablate` with about **{self.reps_for_alpha} repetitions** "
                "and regenerate this report."
                if self.reps_for_alpha is not None
                else "The observed discordance already clears the target alpha."
            ),
            "",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "tasks": self.tasks,
                "n_steps": self.n_steps,
                "reps": self.reps,
                "model": self.model,
                "source": self.source,
                "per_task_p": self.per_task_p,
                "curves": [asdict(c) for c in self.curves],
                "episodes": [asdict(e) for e in self.episodes],
                "discordant": list(self.discordant),
                "mcnemar_p": self.mcnemar_p,
                "reps_for_alpha": self.reps_for_alpha,
                "alpha": self.alpha,
            },
            indent=2,
        )


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _milestones(n: int) -> list[int]:
    """A readable subset of step indices, always including 1 and n."""
    wanted = {1, n}
    wanted.update(k for k in (2, 4, 6, 8, 10, 12, 14, 16, 20) if k < n)
    return sorted(wanted)


# --- loading -----------------------------------------------------------------
def load_cells(report_path: str | Path) -> Cells:
    """Read measured per-cell truth out of an ``ablation_report.json``.

    ``ERROR`` cells carry no measurement and are dropped here; a repetition that
    loses any task is excluded from the episode analysis by ``complete_reps``
    rather than silently shortening the horizon.
    """
    path = Path(report_path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise HorizonDataError(f"unreadable ablation report {path}: {e}") from e

    outcome: dict[tuple[str, str, int], bool] = {}
    reps: set[int] = set()
    for rec in raw.get("records", []):
        if rec.get("status") == "ERROR":
            continue
        outcome[(rec["condition"], rec["task"], int(rec["rep"]))] = bool(rec["true_success"])
        reps.add(int(rec["rep"]))
    tasks = sorted(raw.get("tasks", []))
    if not tasks or not outcome:
        raise HorizonDataError(f"{path} contains no usable measured cells")
    return Cells(
        tasks=tasks,
        reps=sorted(reps),
        outcome=outcome,
        model=raw.get("model", ""),
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


def _bootstrap_ci(
    probabilities_by_task: dict[str, float], *, n: int = _BOOTSTRAP_N, seed: int = 0
) -> tuple[list[float], list[float]]:
    """Per-k 95% CI by resampling TASKS with replacement.

    Tasks are the exchangeable unit: repetitions of one task are correlated, and
    here the whole effect is carried by however many tasks the model gets wrong,
    so between-task variation is the uncertainty that matters.
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


def _reps_for_alpha(b: int, c: int, reps: int, alpha: float) -> int | None:
    """Repetitions needed to reach ``alpha``, extrapolating the observed rate.

    Returns ``None`` when the current data already clears it. Assumes the
    discordance rate holds and stays one-directional — an optimistic estimate,
    so treat it as a floor on the sample size rather than a promise.
    """
    if mcnemar_exact(b, c) < alpha:
        return None
    total = b + c
    if reps <= 0 or total == 0:
        return None
    rate = total / reps
    for candidate in range(reps + 1, 200):
        projected = round(candidate * rate)
        if projected and mcnemar_exact(projected, 0) < alpha:
            return candidate
    return None


def build_report(cells: Cells, *, seed: int = 0, alpha: float = _TARGET_ALPHA) -> HorizonReport:
    curves: list[Curve] = []
    episodes: list[Episode] = []
    p_trust: dict[str, float] = {}
    for name, source, _blurb in CONDITIONS:
        probs = per_task_p(cells, source)
        if name == "trust-chain":
            p_trust = probs
        rate = compounding_curve(list(probs.values()))
        lo, hi = _bootstrap_ci(probs, seed=seed)
        curves.append(Curve(condition=name, rate=rate, ci_lo=lo, ci_hi=hi))
        episodes.extend(episodes_for(cells, name, source))

    trust_eps = {e.rep: e.end_to_end for e in episodes if e.condition == "trust-chain"}
    verify_eps = {e.rep: e.end_to_end for e in episodes if e.condition == "verify-chain"}
    paired = sorted(set(trust_eps) & set(verify_eps))
    b = sum(1 for r in paired if verify_eps[r] and not trust_eps[r])
    c = sum(1 for r in paired if trust_eps[r] and not verify_eps[r])
    p_value = mcnemar_exact(b, c)

    return HorizonReport(
        tasks=cells.tasks,
        n_steps=len(cells.tasks),
        reps=len(paired),
        model=cells.model,
        source=cells.source,
        per_task_p=p_trust,
        curves=curves,
        episodes=episodes,
        discordant=(b, c),
        mcnemar_p=p_value,
        reps_for_alpha=_reps_for_alpha(b, c, len(paired), alpha),
        alpha=alpha,
    )


def run_horizon(report_path: str | Path, out_dir: str | Path, *, seed: int = 0) -> HorizonReport:
    """Compose the horizon analysis from a measured ablation report and write it out."""
    report = build_report(load_cells(report_path), seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "horizon_report.json").write_text(report.to_json())
    (out / "horizon_report.md").write_text(report.to_markdown())
    return report
