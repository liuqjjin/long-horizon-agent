"""Self-eval: run the harness across its own workflows and check each outcome.

Six workflows: issue-to-PR, resume, freshness, fail-closed context,
paper-to-experiment, and verification-ablation. Each case runs in-process
(sequential, so the singleton facade isn't raced) under its own runs dir and
asserts the expected outcome — including the two cases that pass only when the
harness reports FAILED, because refusing to claim an unverifiable result is the
behaviour under test.

Every case is environment-independent by construction: the loop cases declare
retrieval optional and are graded by a real ``pytest`` run, and the fail-closed
case forces the code backend dark instead of depending on whether ``ccc`` is
installed. The score means the same thing on a laptop and on a CI runner.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import timedelta
from importlib import resources
from pathlib import Path

from . import live_context
from .clock import now
from .config import Config
from .harness import Harness
from .harness.approval import HumanApprovalGate
from .live_context import freshness as context_freshness
from .live_context.models import ContextItem, Provenance
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
        lines = [f"# Self-eval — {self.score}", ""]
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


def _task(base: Config, name: str) -> TaskSpec:
    """Load one fixed eval task and bind its repository to the active fixture root."""
    task = TaskSpec.from_file(Path(base.data_dir) / "tasks" / name)
    if task.target_repo is None:
        return task
    target = Path(task.target_repo)
    if not target.is_absolute():
        parts = target.parts[1:] if target.parts[:1] == ("data",) else target.parts
        target = Path(base.data_dir).joinpath(*parts)
    return task.model_copy(update={"target_repo": str(target.resolve())})


def _loop_task(base: Config, name: str) -> TaskSpec:
    """A bundled task loaded for the loop-focused cases.

    Their oracle is the real ``pytest`` run on the patched sandbox, and that
    oracle is available with or without a code-search backend. Retrieval is
    declared optional here so the case measures the loop rather than whether
    ``ccc`` happens to be installed on this machine — the same judgement, and the
    same explicit declaration, that ``tests/conftest.py`` makes for the unit
    suite. Fail-closed context is not thereby untested: it is a case of its own
    (``_case_context_fail_closed``), which forces the backend dark in every
    environment instead of relying on one.
    """
    return _task(base, name).model_copy(update={"context_requirement": "optional"})


def _case_issue_to_pr(base: Config) -> EvalResult:
    r = Harness(_cfg(base, "issue_to_pr")).run(_loop_task(base, "fix_average.yaml"))
    ok = r.status == "DONE" and _verified(r)
    return EvalResult(
        "fix_average", "issue-to-PR", ok, f"status={r.status} verified={_verified(r)}"
    )


def _case_resume(base: Config) -> EvalResult:
    config = _cfg(base, "resume")
    task = _loop_task(base, "fix_average.yaml")
    task = task.model_copy(
        update={
            "inputs": {
                **task.inputs,
                "require_approval": True,
            }
        }
    )
    paused = Harness(config, interactive_approval=False).run(task)
    HumanApprovalGate(paused.state.run_dir).resolve(
        approved=True,
        note="self-eval approval",
    )
    resumed = Harness(config, interactive_approval=False).resume(paused.state.run_id)
    ok = (
        paused.status == "AWAITING_APPROVAL"
        and resumed.status == "DONE"
        and _verified(resumed)
    )
    return EvalResult(
        "pause_resume", "resume", ok, f"first={paused.status} resumed={resumed.status}"
    )


def _case_context_fail_closed(base: Config) -> EvalResult:
    """A step that requires context and cannot get it must fail, and say why.

    The backend is forced dark (``code_backend="null"``) rather than left to the
    machine, so this asserts the same thing on a laptop with ``ccc`` installed
    and on a CI runner without it. Passing here means the run failed *for the
    right reason*: the verdict has to name the unavailable context, not merely
    be a failure of some other kind.
    """
    cfg = _cfg(base, "context_fail_closed", code_backend="null", max_repairs=0)
    task = _task(base, "fix_average.yaml").model_copy(
        update={"context_requirement": "required"}
    )
    r = Harness(cfg).run(task)

    named_the_reason = False
    vj = Path(r.state.run_dir) / "verify.json"
    if vj.exists():
        verdict = Verdict.model_validate_json(vj.read_text())
        named_the_reason = any(
            c.name == "freshness"
            and not c.passed
            and (
                "unavailable" in str(c.detail.get("summary", ""))
                or "no context found" in str(c.detail.get("summary", ""))
            )
            for c in verdict.checks
        )
    ok = r.status == "FAILED" and named_the_reason
    return EvalResult(
        "required_context_unavailable",
        "fail-closed context",
        ok,
        f"status={r.status} verdict_named_the_reason={named_the_reason}",
    )


def _case_freshness(base: Config) -> EvalResult:
    checkout_data = (Path.cwd() / "data").resolve()
    if Path(base.data_dir).resolve() == checkout_data:
        return _case_index_freshness(base)
    return _case_packaged_freshness(base)


def _case_index_freshness(base: Config) -> EvalResult:
    """Exercise the optional CocoIndex refresh path in a source checkout."""
    data = Path(base.runs_dir) / "eval" / "fresh_data"
    if data.exists():
        shutil.rmtree(data)
    shutil.copytree(Path(base.data_dir) / "papers", data / "papers")
    config = base.model_copy(update={"data_dir": data})
    live_context.configure(config=config)
    live_context.index_docs(("paper",))

    query = "super resolution PSNR data_range"
    first = live_context.get_fresh_context(query, kinds=("paper",), k=2)
    initial_fresh = (not first.freshness.is_stale()) and len(first.items) > 0

    note = sorted((data / "papers").glob("*.md"))[0]
    note.write_text(note.read_text() + "\n\nEVAL_FRESHNESS_PROBE: acceptance SSIM 0.90\n")
    time.sleep(1.1)

    changed = live_context.get_fresh_context(query, kinds=("paper",), k=2)
    stale_after_edit = changed.freshness.is_stale()
    refreshed = live_context.reject_stale(changed) if stale_after_edit else changed
    fresh_after_reject = not refreshed.freshness.is_stale()
    return EvalResult(
        "edit_reindex",
        "freshness",
        initial_fresh and stale_after_edit and fresh_after_reject,
        f"initial_fresh={initial_fresh} stale_after_edit={stale_after_edit} "
        f"fresh_after_reject={fresh_after_reject}",
    )


def _case_packaged_freshness(base: Config) -> EvalResult:
    """Exercise source-digest freshness without the optional context extra."""
    data = Path(base.runs_dir) / "eval" / "fresh_data"
    if data.exists():
        shutil.rmtree(data)
    shutil.copytree(Path(base.data_dir) / "papers", data / "papers")
    note = sorted((data / "papers").glob("*.md"))[0].resolve()
    indexed_at = now() + timedelta(seconds=5)
    original_sha256 = context_freshness.file_sha256(note)
    item = ContextItem(
        text=note.read_text(),
        provenance=Provenance(
            source_kind="paper",
            locator=str(note),
            indexed_at=indexed_at,
            content_hash=context_freshness.content_hash(note.read_text()),
            source_sha256=original_sha256,
        ),
    )
    initial = context_freshness.assess(
        [item], index_version="eval-source-v1", indexed_at=indexed_at
    )
    initial_fresh = not initial.is_stale()

    stat_before = note.stat()
    note.write_bytes(note.read_bytes() + b"\nEVAL_FRESHNESS_PROBE: acceptance SSIM 0.90\n")
    os.utime(note, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    changed = context_freshness.assess(
        [item], index_version="eval-source-v1", indexed_at=indexed_at
    )
    stale_after_edit = changed.is_stale()

    item.provenance.source_sha256 = context_freshness.file_sha256(note)
    item.text = note.read_text()
    item.provenance.content_hash = context_freshness.content_hash(item.text)
    refreshed = context_freshness.assess(
        [item], index_version="eval-source-v2", indexed_at=now() + timedelta(seconds=5)
    )
    fresh_after_reject = not refreshed.is_stale()

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
        _task(base, "run_sr_experiment.yaml")
    )
    ok = r.status == "DONE" and _verified(r)
    return EvalResult(
        "bicubic_sr", "paper-to-experiment", ok, f"status={r.status} verified={_verified(r)}"
    )


def _case_verification_ablation(base: Config) -> EvalResult:
    # An unreachable PSNR bar: a correct harness must FAIL (verifier catches it).
    r = Harness(_cfg(base, "ablation", max_repairs=0)).run(
        _task(base, "run_sr_experiment_strict.yaml")
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


_FAST = [_case_issue_to_pr, _case_resume, _case_freshness, _case_context_fail_closed]
_SLOW = [_case_paper_to_experiment, _case_verification_ablation]


def _copy_resource_tree(source, destination: Path) -> None:
    if destination.is_symlink():
        raise RuntimeError(f"refusing symlinked eval resource path: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "__pycache__" or child.name.endswith(".pyc"):
            continue
        target = destination / child.name
        if target.is_symlink():
            raise RuntimeError(f"refusing symlinked eval resource target: {target}")
        if child.is_dir():
            _copy_resource_tree(child, target)
        else:
            target.write_bytes(child.read_bytes())


def _clear_generated_fixture_state(destination: Path) -> None:
    """Remove only indexes generated by a previous installed-package self-eval."""
    if not destination.exists():
        return
    for path in destination.rglob(".cocoindex_code"):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"unsafe generated eval state: {path}")
        try:
            path.resolve().relative_to(destination.resolve())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"generated eval state escapes the fixture directory: {path}"
            ) from error
        shutil.rmtree(path)


def _eval_data_root(base: Config, *, quick: bool) -> Path:
    """Use checkout fixtures when present, otherwise materialize packaged ones."""
    checkout = Path.cwd() / "data"
    required = [
        checkout / "tasks" / "fix_average.yaml",
        checkout / "sample_repo" / "mathutils.py",
        checkout / "papers" / "note_srgan.md",
    ]
    if not quick:
        required.extend(
            [
                checkout / "tasks" / "run_sr_experiment.yaml",
                checkout / "tasks" / "run_sr_experiment_strict.yaml",
                checkout / "sample_experiment" / "experiment.py",
            ]
        )
    if all(path.is_file() for path in required):
        return checkout.resolve()
    if not quick:
        raise FileNotFoundError(
            "the full self-eval requires a source checkout; installed packages support --quick"
        )

    destination = Path(base.runs_dir).absolute() / "eval" / "_fixtures"
    if destination.is_symlink():
        raise RuntimeError(f"refusing symlinked eval fixture directory: {destination}")
    if destination.exists() and not destination.is_dir():
        raise RuntimeError(f"eval fixture path is not a directory: {destination}")
    _clear_generated_fixture_state(destination)
    source = resources.files("lha.resources").joinpath("eval")
    _copy_resource_tree(source, destination)
    return destination.resolve()


def run_eval(base: Config, *, quick: bool = False) -> EvalReport:
    try:
        data_root = _eval_data_root(base, quick=quick)
    except Exception as error:
        return EvalReport(
            [EvalResult("fixtures", "packaging", False, f"errored: {error!r}")]
        )
    base = base.model_copy(update={"data_dir": data_root})
    cases = _FAST + ([] if quick else _SLOW)
    results: list[EvalResult] = []
    for case in cases:
        try:
            results.append(case(base))
        except Exception as e:  # one case crashing must not zero out the whole report
            results.append(EvalResult(case.__name__, "?", False, f"errored: {e!r}"))
    return EvalReport(results=results)
