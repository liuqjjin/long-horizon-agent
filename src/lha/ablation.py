"""Verification ablation: measure whether the verifier loop makes a real LLM more reliable.

For each bug-fix task, draw one first attempt from the LLM and score that same attempt
under three conditions (a paired design, so only the verification varies):

  - ``trust``  — apply the first attempt and accept it; no gate, no repair.
  - ``gate``   — apply it, run the internal test gate, refuse (revert, FAIL) on failure.
  - ``verify`` — gate, then repair (re-prompt with the failing-test feedback) up to a budget.

Prediction and truth are produced by different mechanisms:

  - The **internal gate** runs pytest in the agent's working copy. Its outcome is the
    condition's *claim* (a prediction).
  - The **final scorer** freezes the effective source diff (SHA-256 recorded), applies
    it to a fresh copy of the canonical repo — canonical tests restored regardless of
    what happened in the working copy — and runs pytest there through a separate
    execution backend. Its outcome is *truth*.

Every condition is scored by the final scorer, including attempts the gate refused,
so the gate's precision/recall/FPR/FNR and a full confusion matrix are measured
rather than assumed. (With a shared oracle, prediction==truth would hold by
construction; here divergence — stale caches, dirty workdirs, flaky tests — is
observable.)

Integrity properties:
  - Leak-free: the implementer is a single-shot completion with file tools denied
    (``no_tools``) and sees only non-test source, so it cannot read the oracle.
  - Tamper-proof: a patch may only edit source (tools.policy); the test oracle and
    config stay canonical, and the frozen diff excludes protected paths.
  - Transient backend errors are retried and, if they persist, recorded as ``ERROR``
    (never cached; excluded from rates but counted and reported, never silently
    dropped).
  - Cached cells carry a provenance fingerprint (task bytes, repo digest, backend,
    model, prompt source, harness version, repair budget); any change recomputes.

A weaker implementer ``model`` calibrates difficulty, so first-attempt success lands
in a range where the gate has errors to catch. This runner isolates the gate
mechanism (single-step fix, no planning/context retrieval); it is not the full
harness loop.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__
from .agents.implementer import Implementer
from .artifacts import Patch, Step
from .clock import now
from .config import Config
from .live_context.models import ContextBundle, Freshness
from .llm.base import LLMClient
from .sandbox import ExecutionBackend, TrustedLocalBackend, make_backend
from .tasks.spec import TaskSpec
from .tools import policy
from .tools.patch import apply_patch
from .verifiers import VerifyContext
from .verifiers.code.pytest_verifier import PytestVerifier

logger = logging.getLogger(__name__)

CONDITIONS = [
    ("trust", "apply the first attempt and accept it; no gate, no repair"),
    ("gate", "apply it, run the internal test gate, refuse on failure"),
    ("verify", "gate plus repair loop"),
]

_MAX_REPAIRS = 3
_LLM_RETRIES = 3
_CACHE_SCHEMA = 2
_BOOTSTRAP_N = 10_000


class _Transient(Exception):
    """A backend error that should be retried / excluded, not counted as a result."""


@dataclass
class RunRecord:
    task: str
    condition: str
    rep: int
    status: str  # DONE | FAILED | ERROR
    claimed_success: bool  # the condition's own decision (internal gate for gate/verify)
    true_success: bool  # the independent final scorer's verdict on the frozen artifact
    false_success: bool  # claimed and not true
    repairs: int
    detail: str = ""
    # internal-gate prediction (None for trust, which runs no gate)
    gate_prediction: bool | None = None
    artifact_sha256: str = ""


@dataclass
class ConditionStats:
    condition: str
    blurb: str
    n: int
    claimed_success_rate: float
    true_success_rate: float
    false_success_rate: float
    mean_repairs: float
    errors: int = 0
    # gate-vs-truth confusion (None for trust — no prediction is made)
    tp: int | None = None
    fp: int | None = None
    tn: int | None = None
    fn: int | None = None
    precision: float | None = None
    recall: float | None = None
    fpr: float | None = None
    fnr: float | None = None
    # task-cluster bootstrap 95% CIs
    true_ci: tuple[float, float] | None = None
    false_ci: tuple[float, float] | None = None


@dataclass
class AblationReport:
    llm: str
    model: str
    reps: int
    tasks: list[str]
    records: list[RunRecord] = field(default_factory=list)
    stats: list[ConditionStats] = field(default_factory=list)
    scorer: str = "trusted-local"
    fingerprint: str = ""

    def to_markdown(self) -> str:
        model = self.model or "(backend default)"
        lines = [
            "# Verification ablation",
            "",
            f"implementer: `{self.llm}` · model: `{model}` · tasks: {len(self.tasks)} · "
            f"repetitions: {self.reps} · paired (trust/gate score the same attempt) · "
            f"final scorer: `{self.scorer}` (fresh copy, canonical tests, independent of "
            "the internal gate)",
            "",
            "| condition | n | claimed | true success (95% CI) | false success (95% CI) "
            "| mean repairs | errors |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in self.stats:
            lines.append(
                f"| `{s.condition}` | {s.n} | {_pct(s.claimed_success_rate)} | "
                f"{_pct(s.true_success_rate)}{_ci(s.true_ci)} | "
                f"{_pct(s.false_success_rate)}{_ci(s.false_ci)} | "
                f"{s.mean_repairs:.2f} | {s.errors} |"
            )
        lines += ["", "Conditions:"]
        for name, blurb in CONDITIONS:
            lines.append(f"- `{name}` — {blurb}.")
        gate_lines = _gate_quality_lines(self.stats)
        if gate_lines:
            lines += ["", "Internal gate vs final scorer (per attempt):", *gate_lines]
        mcnemar_lines = _paired_mcnemar_lines(self.records)
        if mcnemar_lines:
            lines += ["", "Paired contrasts:", *mcnemar_lines]
        summary = _summary(self.stats)
        if summary:
            lines += ["", summary]
        lines += ["", "## Per-task outcomes", ""]
        lines += ["| task | trust | gate | verify |", "|---|---|---|---|"]
        for task in self.tasks:
            cells = [_task_cell(self.records, task, c) for c in ("trust", "gate", "verify")]
            lines.append(f"| `{task}` | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines += [
            "",
            "Legend: pass = true success · fail = refused · false-pass = claimed but wrong. "
            "Modal outcome across repetitions; outcomes are the final scorer's.",
            "",
        ]
        return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _ci(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return ""
    return f" ({_pct(ci[0])}–{_pct(ci[1])})"


def _task_cell(records: list[RunRecord], task: str, cond: str) -> str:
    recs = [r for r in records if r.task == task and r.condition == cond and r.status != "ERROR"]
    if not recs:
        return "—"

    def sym(r: RunRecord) -> str:
        return "false-pass" if r.false_success else ("pass" if r.true_success else "fail")

    syms = [sym(r) for r in recs]
    return max(set(syms), key=syms.count)


def _gate_quality_lines(stats: list[ConditionStats]) -> list[str]:
    out = []
    for s in stats:
        if s.tp is None:
            continue
        out.append(
            f"- `{s.condition}`: TP={s.tp} FP={s.fp} TN={s.tn} FN={s.fn} · "
            f"precision={_opt_pct(s.precision)} recall={_opt_pct(s.recall)} "
            f"FPR={_opt_pct(s.fpr)} FNR={_opt_pct(s.fnr)}"
        )
    return out


def _opt_pct(x: float | None) -> str:
    return "n/a" if x is None else _pct(x)


def _summary(stats: list[ConditionStats]) -> str:
    by = {s.condition: s for s in stats}
    trust, gate, verify = by.get("trust"), by.get("gate"), by.get("verify")
    if not (trust and gate and verify):
        return ""
    return (
        f"Without the gate, {_pct(trust.false_success_rate)} of accepted fixes are wrong "
        f"(scorer-graded); the gate (same attempts) reduces that to "
        f"{_pct(gate.false_success_rate)}, and the repair loop raises true success from "
        f"{_pct(gate.true_success_rate)} to {_pct(verify.true_success_rate)}. "
        f"The gate discarded {gate.fn or 0} correct fix(es) (false negatives)."
    )


# --- core mechanics ---------------------------------------------------------
_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", ".lha_pytest.json", ".cocoindex_code", ".git"
)
_DIFF_IGNORE = {"__pycache__", ".pytest_cache", ".lha_pytest.json", ".cocoindex_code", ".git"}


def _sanitize(patch: Patch) -> Patch:
    """Keep only source edits, so a patch can never rewrite the test oracle or config.

    Uses the same protected-file policy the run loop enforces (tools.policy);
    the ablation strips rather than rejects because the paired design scores
    whatever source edits the first attempt made.
    """
    return policy.strip_protected(patch)


def _empty_bundle() -> ContextBundle:
    return ContextBundle(query="", freshness=Freshness(index_version="none", indexed_at=now()))


def _fix_step(task: TaskSpec) -> Step:
    return Step(
        step_id="s2-fix",
        kind="code",
        action="edit_code",
        goal=task.description or task.title,
        verifiers=["pytest"],
    )


def _copy_repo(source: Path, dst: Path, *, include_tests: bool) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(source, dst, ignore=_IGNORE)
    if not include_tests:
        shutil.rmtree(dst / "tests", ignore_errors=True)


def _retry(fn, label: str):
    last: Exception | None = None
    for i in range(_LLM_RETRIES):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - any backend error is retryable here
            last = e
            logger.warning("transient LLM error [%s] %d/%d: %s", label, i + 1, _LLM_RETRIES, e)
            time.sleep(2 * (i + 1))
    raise _Transient(f"{label}: {last}")


def _pytest(workdir: Path, exec_backend: ExecutionBackend) -> tuple[bool, list[str]]:
    """Run pytest via the given backend; return (passed, failing-assertion messages)."""
    step = Step(step_id="grade", kind="code", action="edit_code", goal="grade", verifiers=["pytest"])
    check = PytestVerifier().verify(
        None, VerifyContext(workdir=workdir, step=step, exec=exec_backend)
    )
    return check.passed, list(check.detail.get("messages", []))


def _first_attempt(llm: LLMClient, source: Path, task: TaskSpec, scratch: Path) -> Patch:
    """One leak-free first attempt: implement against source with NO tests present."""
    wd = scratch / "attempt"
    _copy_repo(source, wd, include_tests=False)
    patch = _retry(lambda: Implementer(llm).implement(_fix_step(task), _empty_bundle(), wd), "first")
    return _sanitize(patch)


# --- frozen artifact + independent scorer ------------------------------------
def _iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _DIFF_IGNORE for part in rel.parts) or rel.name in _DIFF_IGNORE:
            continue
        yield rel.as_posix(), p


def _frozen_diff(source: Path, wd: Path) -> dict[str, str | None]:
    """The effective source change: workdir files that differ from the pristine repo.

    Protected (oracle/config) paths are excluded — the scorer always grades
    against canonical tests, whatever happened in the working copy. ``None``
    marks a deletion.
    """
    src_files = dict(_iter_files(source))
    wd_files = dict(_iter_files(wd))
    frozen: dict[str, str | None] = {}
    for rel in sorted(set(src_files) | set(wd_files)):
        if policy.is_protected(rel):
            continue
        s, w = src_files.get(rel), wd_files.get(rel)
        if w is None:
            frozen[rel] = None
        else:
            text = w.read_text(errors="replace")
            if s is None or s.read_text(errors="replace") != text:
                frozen[rel] = text
    return frozen


def _artifact_digest(frozen: dict[str, str | None]) -> str:
    h = hashlib.sha256()
    for rel in sorted(frozen):
        h.update(rel.encode())
        h.update(b"\0")
        content = frozen[rel]
        h.update(("\0<deleted>" if content is None else content).encode())
        h.update(b"\0")
    return h.hexdigest()


def _score(
    source: Path,
    frozen: dict[str, str | None],
    scratch: Path,
    label: str,
    scorer: ExecutionBackend,
) -> bool:
    """Final truth: canonical repo + frozen diff, graded in a fresh copy.

    Shares no state with the internal gate: fresh directory, canonical tests,
    its own execution backend.
    """
    wd = scratch / f"score_{label}"
    _copy_repo(source, wd, include_tests=True)
    for rel, content in frozen.items():
        target = wd / rel
        if content is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    ok, _ = _pytest(wd, scorer)
    return ok


def _evaluate(
    llm: LLMClient,
    source: Path,
    task: TaskSpec,
    patch: Patch,
    scratch: Path,
    rep: int,
    agent_exec: ExecutionBackend,
    scorer: ExecutionBackend,
) -> list[RunRecord]:
    name = task.inputs.get("_name", task.title)
    out: list[RunRecord] = []

    # ONE working copy for the first attempt: trust and gate score the same
    # artifact (paired). The internal gate predicts; the scorer decides truth.
    wd = scratch / "first"
    _copy_repo(source, wd, include_tests=True)
    if patch.file_contents:
        apply_patch(patch, wd)
    gate_pred, _ = _pytest(wd, agent_exec)
    frozen = _frozen_diff(source, wd)
    sha = _artifact_digest(frozen)
    truth = _score(source, frozen, scratch, "first", scorer)

    out.append(
        RunRecord(
            name, "trust", rep, "DONE", True, truth, not truth, 0,
            _truth_detail(truth), None, sha,
        )
    )
    # The scorer grades the SAME artifact even when the gate refused it, so a
    # refusal of a correct fix is counted (false negative), not invisible.
    out.append(
        RunRecord(
            name, "gate", rep, "DONE" if gate_pred else "FAILED",
            gate_pred, truth, gate_pred and not truth, 0,
            _truth_detail(truth), gate_pred, sha,
        )
    )

    # verify: same first attempt, then repair with failing-test feedback.
    wd2 = scratch / "verify"
    _copy_repo(source, wd2, include_tests=True)
    if patch.file_contents:
        apply_patch(patch, wd2)
    ok, failures = _pytest(wd2, agent_exec)
    repairs = 0
    while not ok and repairs < _MAX_REPAIRS:
        repairs += 1
        step = _fix_step(task).as_repair(failures)
        repair = _retry(
            lambda s=step: _sanitize(Implementer(llm).implement(s, _empty_bundle(), wd2)), "repair"
        )
        if not repair.file_contents:
            break  # nothing new to try
        apply_patch(repair, wd2)
        ok, failures = _pytest(wd2, agent_exec)
    frozen2 = _frozen_diff(source, wd2)
    sha2 = _artifact_digest(frozen2)
    truth2 = _score(source, frozen2, scratch, "verify", scorer)
    out.append(
        RunRecord(
            name, "verify", rep, "DONE" if ok else "FAILED",
            ok, truth2, ok and not truth2, repairs,
            _truth_detail(truth2), ok, sha2,
        )
    )
    return out


def _truth_detail(ok: bool) -> str:
    return "scorer: tests pass" if ok else "scorer: tests fail"


def _make_llm(llm: str, model: str | None, *, cli_path: str = "claude") -> LLMClient:
    if llm == "stub":
        from .llm.stub import DeterministicStub

        return DeterministicStub()
    if llm == "claude_cli":
        from .llm.claude_cli import ClaudeCLIClient

        # no_tools => single-shot completion; the implementer cannot read the oracle.
        return ClaudeCLIClient(cli_path=cli_path, model=model, no_tools=True)
    if llm == "anthropic":
        from .llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model or "claude-opus-4-8")
    raise ValueError(f"unknown llm backend: {llm!r}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --- provenance fingerprint ---------------------------------------------------
def _repo_digest(source: Path) -> str:
    h = hashlib.sha256()
    for rel, p in sorted(_iter_files(source)):
        h.update(rel.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _fingerprint(
    task_path: str, source: Path, llm: str, model: str | None, scorer: str = "trusted-local"
) -> str:
    """Everything that determines a cell's outcome. Any change busts the cache."""
    from .llm import base as llm_base

    h = hashlib.sha256()
    h.update(f"schema={_CACHE_SCHEMA}|harness={__version__}".encode())
    h.update(Path(task_path).read_bytes())
    h.update(_repo_digest(source).encode())
    h.update(f"|llm={llm}|model={model or ''}|repairs={_MAX_REPAIRS}".encode())
    # The truth labels belong to a specific scorer: cached cells must never be
    # relabeled as another backend's verdicts on a --scorer-backend change.
    h.update(f"|scorer={scorer}".encode())
    # The prompt/parsing logic IS part of the experiment configuration.
    h.update(inspect.getsource(llm_base).encode())
    return h.hexdigest()


def _load_cached(path: Path, fingerprint: str) -> list[RunRecord] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("fingerprint") != fingerprint:
            return None  # stale schema or changed provenance -> recompute
        return [RunRecord(**r) for r in data["records"]]
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return None  # corrupt/partial cache -> recompute


def _run_cell(
    llm: LLMClient,
    source: Path,
    task: TaskSpec,
    rep: int,
    out_dir: Path,
    fingerprint: str,
    agent_exec: ExecutionBackend,
    scorer: ExecutionBackend,
) -> list[RunRecord]:
    name = task.inputs.get("_name", task.title)
    cache = out_dir / "results" / f"{name}__r{rep}.json"
    cached = _load_cached(cache, fingerprint)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory(prefix="lha_abl_") as tmp:
        scratch = Path(tmp)
        try:
            patch = _first_attempt(llm, source, task, scratch)
            records = _evaluate(llm, source, task, patch, scratch, rep, agent_exec, scorer)
        except _Transient as e:
            logger.error("transient failure on %s rep %d: %s — not caching", name, rep, e)
            return [
                RunRecord(name, c, rep, "ERROR", False, False, False, 0, str(e)[:200])
                for c, _ in CONDITIONS
            ]
    _atomic_write(
        cache,
        json.dumps(
            {"fingerprint": fingerprint, "records": [asdict(r) for r in records]}, indent=2
        ),
    )
    return records


# --- aggregation ---------------------------------------------------------------
def _bootstrap_ci(
    records: list[RunRecord], metric: str, *, n: int = _BOOTSTRAP_N, seed: int = 0
) -> tuple[float, float] | None:
    """Task-cluster bootstrap 95% CI: tasks are resampled with replacement and
    each task carries all its repetitions (reps are nested, not independent).
    The resampling itself lives in ``bench.stats`` — one implementation."""
    by_task: dict[str, list[float]] = {}
    for r in records:
        by_task.setdefault(r.task, []).append(float(getattr(r, metric)))
    if len(by_task) < 2:
        return None  # a single task cannot express between-task variation
    from .bench.stats import cluster_bootstrap_ci

    return cluster_bootstrap_ci(by_task, n=n, seed=seed)


def _paired_mcnemar_lines(records: list[RunRecord]) -> list[str]:
    """Exact McNemar on the paired (task, rep) cells for the headline contrasts."""
    from .bench.stats import mcnemar_exact

    def outcomes(cond: str, metric: str) -> dict[tuple[str, int], bool]:
        return {
            (r.task, r.rep): bool(getattr(r, metric))
            for r in records
            if r.condition == cond and r.status != "ERROR"
        }

    lines: list[str] = []
    for a, b, metric in (
        ("trust", "gate", "false_success"),
        ("gate", "verify", "true_success"),
    ):
        oa, ob = outcomes(a, metric), outcomes(b, metric)
        pairs = sorted(set(oa) & set(ob))
        if not pairs:
            continue
        only_a = sum(oa[k] and not ob[k] for k in pairs)
        only_b = sum(ob[k] and not oa[k] for k in pairs)
        p = mcnemar_exact(only_a, only_b)
        lines.append(
            f"- `{a}` vs `{b}` on {metric.replace('_', ' ')}: "
            f"discordant {only_a}/{only_b} of {len(pairs)} pairs · exact McNemar p = {p:.2f}"
        )
    return lines


def _aggregate(records: list[RunRecord]) -> list[ConditionStats]:
    stats: list[ConditionStats] = []
    for name, blurb in CONDITIONS:
        all_recs = [r for r in records if r.condition == name]
        recs = [r for r in all_recs if r.status != "ERROR"]
        errors = len(all_recs) - len(recs)
        n = len(recs)
        if n == 0:
            stats.append(ConditionStats(name, blurb, 0, 0.0, 0.0, 0.0, 0.0, errors=errors))
            continue
        s = ConditionStats(
            condition=name,
            blurb=blurb,
            n=n,
            claimed_success_rate=sum(r.claimed_success for r in recs) / n,
            true_success_rate=sum(r.true_success for r in recs) / n,
            false_success_rate=sum(r.false_success for r in recs) / n,
            mean_repairs=sum(r.repairs for r in recs) / n,
            errors=errors,
            true_ci=_bootstrap_ci(recs, "true_success"),
            false_ci=_bootstrap_ci(recs, "false_success"),
        )
        preds = [r for r in recs if r.gate_prediction is not None]
        if preds:
            tp = sum(bool(r.gate_prediction) and r.true_success for r in preds)
            fp = sum(bool(r.gate_prediction) and not r.true_success for r in preds)
            tn = sum(not r.gate_prediction and not r.true_success for r in preds)
            fn = sum(not r.gate_prediction and r.true_success for r in preds)
            s.tp, s.fp, s.tn, s.fn = tp, fp, tn, fn
            s.precision = tp / (tp + fp) if (tp + fp) else None
            s.recall = tp / (tp + fn) if (tp + fn) else None
            s.fpr = fp / (fp + tn) if (fp + tn) else None
            s.fnr = fn / (fn + tp) if (fn + tp) else None
        stats.append(s)
    return stats


def run_ablation(
    base: Config,
    task_paths: list[str],
    *,
    llm: str = "claude_cli",
    model: str | None = None,
    reps: int = 1,
    out_dir: Path | None = None,
    llm_client: LLMClient | None = None,
    scorer_backend: str = "trusted-local",
) -> AblationReport:
    out = Path(out_dir) if out_dir else (Path(base.runs_dir) / "ablation")
    out.mkdir(parents=True, exist_ok=True)
    # LHA_CLAUDE_MODEL / LHA_CLAUDE_CLI apply here exactly as in `lha run`; an
    # explicit --model wins. The resolved name feeds the provenance fingerprint.
    model = model or (base.claude_cli_model or None)
    client = llm_client or _make_llm(llm, model, cli_path=base.claude_cli_path)
    # The agent-side gate and the final scorer never share an execution backend
    # instance; the scorer can be a container while the gate stays local.
    agent_exec: ExecutionBackend = TrustedLocalBackend()
    scorer: ExecutionBackend = (
        make_backend("docker", image=base.exec_image)
        if scorer_backend == "docker"
        else make_backend(scorer_backend)
    )

    tasks: list[tuple[str, TaskSpec, Path, str]] = []
    for tp in task_paths:
        spec = TaskSpec.from_file(tp)
        spec.inputs["_name"] = Path(tp).stem
        source = Path(spec.target_repo or ".")
        tasks.append(
            (Path(tp).stem, spec, source, _fingerprint(tp, source, llm, model, scorer_backend))
        )

    records: list[RunRecord] = []
    total = len(tasks) * reps
    i = 0
    for rep in range(reps):
        for name, spec, source, fp in tasks:
            i += 1
            logger.info("ablation %d/%d: %s (rep %d)", i, total, name, rep)
            records.extend(_run_cell(client, source, spec, rep, out, fp, agent_exec, scorer))

    combined = hashlib.sha256("".join(t[3] for t in tasks).encode()).hexdigest()
    report = AblationReport(
        llm=llm,
        model=model or "",
        reps=reps,
        tasks=[t[0] for t in tasks],
        records=records,
        stats=_aggregate(records),
        scorer=scorer.name,
        fingerprint=combined,
    )
    _atomic_write(
        out / "ablation_report.json",
        json.dumps(
            {
                "llm": report.llm,
                "model": report.model,
                "reps": report.reps,
                "tasks": report.tasks,
                "scorer": report.scorer,
                "fingerprint": report.fingerprint,
                "harness_version": __version__,
                "stats": [asdict(s) for s in report.stats],
                "records": [asdict(r) for r in report.records],
            },
            indent=2,
        ),
    )
    _atomic_write(out / "ablation_report.md", report.to_markdown())
    return report
