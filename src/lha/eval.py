"""ResearchAgentBench-Lite: the harness measuring itself across its workflows.

Five dimensions:
  issue-to-PR · paper-to-experiment · resume · freshness · verification-ablation

Each case runs in-process (sequential, so the singleton facade isn't raced) under
its own runs dir, and asserts the *expected* outcome — including that the harness
correctly FAILS a result that doesn't meet the bar (verification is load-bearing).
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import live_context
from .config import Config
from .harness import Harness
from .tasks.spec import TaskSpec
from .verifiers.verdict import Verdict


@dataclass
class EvalResult:
    name: str
    dimension: str
    passed: bool
    detail: str = ""


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def score(self) -> str:
        n = sum(r.passed for r in self.results)
        return f"{n}/{len(self.results)}"

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_markdown(self) -> str:
        lines = [f"# ResearchAgentBench-Lite — {self.score}", ""]
        lines += ["| dimension | case | result | detail |", "|---|---|---|---|"]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"| {r.dimension} | {r.name} | {mark} | {r.detail} |")
        return "\n".join(lines) + "\n"


def _cfg(base: Config, sub: str, **over) -> Config:
    return base.model_copy(update={"runs_dir": Path(base.runs_dir) / "eval" / sub, **over})


def _verified(result) -> bool:
    vj = Path(result.state.run_dir) / "verify.json"
    if not vj.exists():
        return False
    return Verdict.model_validate_json(vj.read_text()).passed


def _case_issue_to_pr(base: Config) -> EvalResult:
    r = Harness(_cfg(base, "issue_to_pr")).run(TaskSpec.from_file("data/tasks/fix_average.yaml"))
    ok = r.status == "DONE" and _verified(r)
    return EvalResult(
        "fix_average", "issue-to-PR", ok, f"status={r.status} verified={_verified(r)}"
    )


def _case_resume(base: Config) -> EvalResult:
    paused = Harness(_cfg(base, "resume", max_steps=1)).run(
        TaskSpec.from_file("data/tasks/fix_average.yaml")
    )
    resumed = Harness(_cfg(base, "resume")).resume(paused.state.run_id)
    ok = paused.status == "PAUSED" and resumed.status == "DONE" and _verified(resumed)
    return EvalResult(
        "pause_resume", "resume", ok, f"first={paused.status} resumed={resumed.status}"
    )


def _case_freshness(base: Config) -> EvalResult:
    # Use the paper corpus (CocoFlowBackend — no background daemon) for a
    # deterministic freshness test, isolated in a temp data dir so the committed
    # sample notes are never mutated.
    data = Path(base.runs_dir) / "eval" / "fresh_data"
    if data.exists():
        shutil.rmtree(data)
    shutil.copytree("data/papers", data / "papers")
    fcfg = base.model_copy(update={"data_dir": data})
    live_context.configure(config=fcfg)
    live_context.index_docs(("paper",))

    q = "super resolution PSNR data_range"
    cap = live_context.get_fresh_context(q, kinds=("paper",), k=2)
    initial_fresh = (not cap.freshness.is_stale()) and len(cap.items) > 0

    note = sorted((data / "papers").glob("*.md"))[0]
    note.write_text(note.read_text() + "\n\nEVAL_FRESHNESS_PROBE: acceptance SSIM 0.90\n")
    time.sleep(1.1)

    # Re-search without reindex: the JSON index now lags the edited note, so the
    # bundle is detected stale; reject_stale incrementally rebuilds it.
    b2 = live_context.get_fresh_context(q, kinds=("paper",), k=2)
    stale_after_edit = b2.freshness.is_stale()
    b3 = live_context.reject_stale(b2) if stale_after_edit else b2
    fresh_after_reject = not b3.freshness.is_stale()

    ok = initial_fresh and stale_after_edit and fresh_after_reject
    return EvalResult(
        "edit_reindex",
        "freshness",
        ok,
        f"initial_fresh={initial_fresh} stale_after_edit={stale_after_edit} "
        f"fresh_after_reject={fresh_after_reject}",
    )


def _case_paper_to_experiment(base: Config) -> EvalResult:
    r = Harness(_cfg(base, "paper_to_experiment")).run(
        TaskSpec.from_file("data/tasks/run_sr_experiment.yaml")
    )
    ok = r.status == "DONE" and _verified(r)
    return EvalResult(
        "bicubic_sr", "paper-to-experiment", ok, f"status={r.status} verified={_verified(r)}"
    )


def _case_verification_ablation(base: Config) -> EvalResult:
    # An unreachable PSNR bar: a correct harness must FAIL (verifier catches it).
    r = Harness(_cfg(base, "ablation", max_repairs=0)).run(
        TaskSpec.from_file("data/tasks/run_sr_experiment_strict.yaml")
    )
    psnr_failed = False
    reached_psnr = False
    vj = Path(r.state.run_dir) / "verify.json"
    if vj.exists():
        v = Verdict.model_validate_json(vj.read_text())
        reached_psnr = any(c.name == "psnr" for c in v.checks)
        psnr_failed = any(c.name == "psnr" and not c.passed for c in v.checks)
    ok = r.status == "FAILED" and psnr_failed
    return EvalResult(
        "strict_threshold_caught",
        "verification-ablation",
        ok,
        f"status={r.status} psnr_correctly_rejected={psnr_failed} reached_psnr_step={reached_psnr}",
    )


_FAST = [_case_issue_to_pr, _case_resume, _case_freshness]
_SLOW = [_case_paper_to_experiment, _case_verification_ablation]


def run_eval(base: Config, *, quick: bool = False) -> EvalReport:
    cases = _FAST + ([] if quick else _SLOW)
    return EvalReport(results=[case(base) for case in cases])
