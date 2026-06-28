"""Verification ablation: does the objective verifier loop make a real LLM more reliable?

This is the project's central, *measured* claim. For each bug-fix task we draw ONE
first attempt from a real LLM and evaluate that SAME attempt under three conditions —
a genuinely *paired* design, so the only thing that varies is the verification:

  - ``trust``  — apply the first attempt and accept it. No objective gate. This is how
                 an "orchestrate-and-trust" agent operates.
  - ``gate``   — apply it, run the objective test gate; refuse (revert, FAIL) if it
                 doesn't pass. Catch, don't fix.
  - ``verify`` — the full harness: gate, then repair (re-prompt with the failing-test
                 feedback) up to a budget.

Because ``trust`` and ``gate`` score the identical first attempt, the contrast is
exact, not statistical:

  - **claimed success** — the condition declared success;
  - **true success**    — the canonical test suite actually passes;
  - **false success**   — ``claimed and not true``: a *silently wrong* answer;
  - **mean repairs**    — repair rounds spent (verify only).

By construction ``true_success(trust) == true_success(gate)`` (same attempt, same
oracle); the gate's whole contribution is driving **false success to zero**, and the
repair loop's is lifting **true success** above the first-try rate.

Integrity properties (so the headline survives scrutiny):
  - **Leak-free.** The implementer is a single-shot completion with file tools denied
    (``no_tools``) and is shown only non-test source — it cannot read the oracle.
  - **Tamper-proof.** A patch may only edit source; test files, ``conftest.py`` and
    ``pyproject.toml`` are stripped from it, so the oracle the gate and the grade run
    is always the canonical one. Difficulty is calibrated with a weaker ``model``.
  - **Honest.** A transient backend error is retried and, if it persists, recorded as
    ``ERROR`` (never cached, never counted) so infra flakiness can't pollute the rates.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agents.implementer import Implementer
from .artifacts import Patch, Step
from .clock import now
from .config import Config
from .live_context.models import ContextBundle, Freshness
from .llm.base import LLMClient
from .tasks.spec import TaskSpec
from .tools.patch import apply_patch
from .verifiers import VerifyContext
from .verifiers.code.pytest_verifier import PytestVerifier

logger = logging.getLogger(__name__)

CONDITIONS = [
    ("trust", "apply the first attempt and accept it — no objective gate, no repair"),
    ("gate", "apply it, run the test gate, refuse on failure — catch but don't fix"),
    ("verify", "the full harness: gate + repair loop"),
]

_MAX_REPAIRS = 3
_LLM_RETRIES = 3


class _Transient(Exception):
    """A backend error that should be retried / excluded, not counted as a result."""


@dataclass
class RunRecord:
    task: str
    condition: str
    rep: int
    status: str  # DONE | FAILED | ERROR
    claimed_success: bool
    true_success: bool
    false_success: bool
    repairs: int
    detail: str = ""


@dataclass
class ConditionStats:
    condition: str
    blurb: str
    n: int
    claimed_success_rate: float
    true_success_rate: float
    false_success_rate: float
    mean_repairs: float


@dataclass
class AblationReport:
    llm: str
    model: str
    reps: int
    tasks: list[str]
    records: list[RunRecord] = field(default_factory=list)
    stats: list[ConditionStats] = field(default_factory=list)

    def to_markdown(self) -> str:
        model = self.model or "(backend default)"
        lines = [
            "# Verification Ablation",
            "",
            f"implementer: `{self.llm}` · model: `{model}` · tasks: {len(self.tasks)} · "
            f"repetitions: {self.reps} · paired (trust/gate score the same attempt)",
            "",
            "| condition | claimed | **true success** | **false success** | mean repairs |",
            "|---|---|---|---|---|",
        ]
        for s in self.stats:
            lines.append(
                f"| `{s.condition}` | {_pct(s.claimed_success_rate)} | "
                f"**{_pct(s.true_success_rate)}** | **{_pct(s.false_success_rate)}** | "
                f"{s.mean_repairs:.2f} |"
            )
        lines += ["", "_condition legend:_"]
        for name, blurb in CONDITIONS:
            lines.append(f"- `{name}` — {blurb}")
        lines += ["", _headline(self.stats), "", "## Per-task outcomes", ""]
        lines += ["| task | trust | gate | verify |", "|---|---|---|---|"]
        for task in self.tasks:
            cells = [_task_cell(self.records, task, c) for c in ("trust", "gate", "verify")]
            lines.append(f"| `{task}` | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines += [
            "",
            "Legend: ✓ true success · ✗ failed (refused) · ⚠ **false success** "
            "(claimed but wrong). Modal outcome across repetitions.",
            "",
        ]
        return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _task_cell(records: list[RunRecord], task: str, cond: str) -> str:
    recs = [r for r in records if r.task == task and r.condition == cond and r.status != "ERROR"]
    if not recs:
        return "—"

    def sym(r: RunRecord) -> str:
        return "⚠" if r.false_success else ("✓" if r.true_success else "✗")

    syms = [sym(r) for r in recs]
    return max(set(syms), key=syms.count)


def _headline(stats: list[ConditionStats]) -> str:
    by = {s.condition: s for s in stats}
    trust, gate, verify = by.get("trust"), by.get("gate"), by.get("verify")
    if not (trust and gate and verify):
        return ""
    return (
        f"**Headline.** On the same first attempts, orchestrate-and-trust ships "
        f"**{_pct(trust.false_success_rate)}** silently-wrong fixes; the objective gate "
        f"drives that to **{_pct(gate.false_success_rate)}**. The repair loop then lifts "
        f"true success from **{_pct(gate.true_success_rate)}** (first try) to "
        f"**{_pct(verify.true_success_rate)}** (full harness)."
    )


# --- core mechanics ---------------------------------------------------------
_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", ".lha_pytest.json", ".cocoindex_code", ".git"
)


def _is_protected(rel: str) -> bool:
    """Files a source fix may NOT touch — the oracle and its config stay canonical."""
    p = Path(rel)
    name = p.name
    return (
        "tests" in p.parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or name == "pyproject.toml"
    )


def _sanitize(patch: Patch) -> Patch:
    """Keep only source edits, so a patch can never rewrite the test oracle or config."""
    fc = {k: v for k, v in patch.file_contents.items() if not _is_protected(k)}
    return patch.model_copy(
        update={"file_contents": fc, "touched_files": list(fc), "unified_diff": ""}
    )


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


def _pytest(workdir: Path) -> tuple[bool, list[str]]:
    """Run the canonical test oracle; return (passed, failing-assertion messages)."""
    step = Step(step_id="grade", kind="code", action="edit_code", goal="grade", verifiers=["pytest"])
    check = PytestVerifier().verify(None, VerifyContext(workdir=workdir, step=step))
    return check.passed, list(check.detail.get("messages", []))


def _first_attempt(llm: LLMClient, source: Path, task: TaskSpec, scratch: Path) -> Patch:
    """One leak-free first attempt: implement against source with NO tests present."""
    wd = scratch / "attempt"
    _copy_repo(source, wd, include_tests=False)
    patch = _retry(lambda: Implementer(llm).implement(_fix_step(task), _empty_bundle(), wd), "first")
    return _sanitize(patch)


def _evaluate(
    llm: LLMClient, source: Path, task: TaskSpec, patch: Patch, scratch: Path, rep: int
) -> list[RunRecord]:
    t = task.title  # human label; the corpus key is the file stem (set by caller)
    name = task.inputs.get("_name", t)
    out: list[RunRecord] = []

    # trust + gate score the SAME first attempt (paired).
    wd = scratch / "trust"
    _copy_repo(source, wd, include_tests=True)
    if patch.file_contents:
        apply_patch(patch, wd)
    ok, _ = _pytest(wd)
    out.append(RunRecord(name, "trust", rep, "DONE", True, ok, not ok, 0, _ok_detail(ok)))
    out.append(
        RunRecord(name, "gate", rep, "DONE" if ok else "FAILED", ok, ok, False, 0, _ok_detail(ok))
    )

    # verify: same attempt, then repair with failing-test feedback.
    wd = scratch / "verify"
    _copy_repo(source, wd, include_tests=True)
    if patch.file_contents:
        apply_patch(patch, wd)
    ok, failures = _pytest(wd)
    repairs = 0
    while not ok and repairs < _MAX_REPAIRS:
        repairs += 1
        step = _fix_step(task).as_repair(failures)
        repair = _retry(
            lambda s=step: _sanitize(Implementer(llm).implement(s, _empty_bundle(), wd)), "repair"
        )
        if not repair.file_contents:
            break  # nothing new to try
        apply_patch(repair, wd)
        ok, failures = _pytest(wd)
    out.append(
        RunRecord(
            name, "verify", rep, "DONE" if ok else "FAILED", ok, ok, False, repairs, _ok_detail(ok)
        )
    )
    return out


def _ok_detail(ok: bool) -> str:
    return "tests pass" if ok else "tests fail"


def _make_llm(llm: str, model: str | None) -> LLMClient:
    if llm == "stub":
        from .llm.stub import DeterministicStub

        return DeterministicStub()
    if llm == "claude_cli":
        from .llm.claude_cli import ClaudeCLIClient

        # no_tools => single-shot completion; the implementer cannot read the oracle.
        return ClaudeCLIClient(model=model, no_tools=True)
    if llm == "anthropic":
        from .llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model or "claude-opus-4-8")
    raise ValueError(f"unknown llm backend: {llm!r}")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _load_cached(path: Path) -> list[RunRecord] | None:
    if not path.exists():
        return None
    try:
        return [RunRecord(**r) for r in json.loads(path.read_text())]
    except (OSError, json.JSONDecodeError, TypeError):
        return None  # corrupt/partial cache -> recompute


def _run_cell(
    llm: LLMClient, source: Path, task: TaskSpec, rep: int, out_dir: Path
) -> list[RunRecord]:
    name = task.inputs.get("_name", task.title)
    cache = out_dir / "results" / f"{name}__r{rep}.json"
    cached = _load_cached(cache)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory(prefix="lha_abl_") as tmp:
        scratch = Path(tmp)
        try:
            patch = _first_attempt(llm, source, task, scratch)
            records = _evaluate(llm, source, task, patch, scratch, rep)
        except _Transient as e:
            logger.error("transient failure on %s rep %d: %s — not caching", name, rep, e)
            return [
                RunRecord(name, c, rep, "ERROR", False, False, False, 0, str(e)[:200])
                for c, _ in CONDITIONS
            ]
    _atomic_write(cache, json.dumps([asdict(r) for r in records], indent=2))
    return records


def _aggregate(records: list[RunRecord]) -> list[ConditionStats]:
    stats: list[ConditionStats] = []
    for name, blurb in CONDITIONS:
        recs = [r for r in records if r.condition == name and r.status != "ERROR"]
        n = len(recs)
        if n == 0:
            stats.append(ConditionStats(name, blurb, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        stats.append(
            ConditionStats(
                condition=name,
                blurb=blurb,
                n=n,
                claimed_success_rate=sum(r.claimed_success for r in recs) / n,
                true_success_rate=sum(r.true_success for r in recs) / n,
                false_success_rate=sum(r.false_success for r in recs) / n,
                mean_repairs=sum(r.repairs for r in recs) / n,
            )
        )
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
) -> AblationReport:
    out = Path(out_dir) if out_dir else (Path(base.runs_dir) / "ablation")
    out.mkdir(parents=True, exist_ok=True)
    client = llm_client or _make_llm(llm, model)

    tasks: list[tuple[str, TaskSpec, Path]] = []
    for tp in task_paths:
        spec = TaskSpec.from_file(tp)
        spec.inputs["_name"] = Path(tp).stem
        tasks.append((Path(tp).stem, spec, Path(spec.target_repo or ".")))

    records: list[RunRecord] = []
    total = len(tasks) * reps
    i = 0
    for rep in range(reps):
        for name, spec, source in tasks:
            i += 1
            logger.info("ablation %d/%d: %s (rep %d)", i, total, name, rep)
            records.extend(_run_cell(client, source, spec, rep, out))

    report = AblationReport(
        llm=llm,
        model=model or "",
        reps=reps,
        tasks=[t[0] for t in tasks],
        records=records,
        stats=_aggregate(records),
    )
    _atomic_write(
        out / "ablation_report.json",
        json.dumps(
            {
                "llm": report.llm,
                "model": report.model,
                "reps": report.reps,
                "tasks": report.tasks,
                "stats": [asdict(s) for s in report.stats],
                "records": [asdict(r) for r in report.records],
            },
            indent=2,
        ),
    )
    _atomic_write(out / "ablation_report.md", report.to_markdown())
    return report
