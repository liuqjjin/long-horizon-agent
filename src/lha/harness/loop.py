"""The verification loop — the spine of the harness.

    context -> tool/execute -> verify -> repair -> checkpoint -> repeat

with max-steps, checkpoint/resume, and a human-approval gate. No LangGraph; the
state is a JSON checkpoint + a step ledger, shaped so LangGraph drops in later.
"""

from __future__ import annotations

import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import live_context
from ..agents import (
    ContextEngineer,
    Experimenter,
    Implementer,
    Supervisor,
    VerifierAgent,
)
from ..artifacts import ExperimentResult, ExperimentSummary, Patch, PRSummary
from ..clock import now
from ..config import Config
from ..llm import get_llm
from ..tasks.spec import TaskSpec
from ..tools.patch import Backup, apply_patch, load_backup, revert_patch, save_backup
from ..verifiers import VerifyContext
from ..verifiers.verdict import Verdict
from .approval import HumanApprovalGate
from .budget import StepBudget
from .checkpoint import append_ledger, load_state_by_id, save_state
from .errors import ApprovalPending, ApprovalRejected, BudgetExceeded
from .state import RunState, StepRecord

_IGNORE = shutil.ignore_patterns(
    ".cocoindex_code", ".git", "__pycache__", "*.pyc", ".lha_pytest.json", "runs"
)


@dataclass
class RunResult:
    state: RunState
    status: str
    message: str = ""


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:32] or "task"


def _gen_run_id(task: TaskSpec) -> str:
    return f"{now():%Y%m%d-%H%M%S}-{_slug(task.title)}-{uuid.uuid4().hex[:4]}"


class Harness:
    def __init__(self, config: Config | None = None, *, auto_approve: bool = False):
        self.config = config or Config.from_env()
        self.auto_approve = auto_approve
        self.llm = get_llm(self.config)
        self._backups: dict[str, Backup] = {}

    # --- public entry points ------------------------------------------------
    def run(self, task: TaskSpec, *, run_id: str | None = None) -> RunResult:
        run_id = run_id or _gen_run_id(task)
        run_dir = self.config.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        workdir = run_dir / "workdir"
        self._prepare_workdir(task, workdir)
        state = RunState.new(task, run_id, str(run_dir), str(workdir))
        save_state(state)
        return self._drive(state)

    def resume(self, run_id: str) -> RunResult:
        state = load_state_by_id(self.config.runs_dir, run_id)
        if state.status in ("DONE", "FAILED"):
            return RunResult(state, state.status, "run already terminal")
        state.status = "RUNNING"
        return self._drive(state)

    # --- driver -------------------------------------------------------------
    def _drive(self, state: RunState) -> RunResult:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        # Point code context at the stable target repo (indexed once, reused across
        # runs) rather than the ephemeral per-run workdir — the latter churns the
        # ccc daemon with throwaway projects. Edits/verification still hit workdir.
        code_root = state.task.target_repo or str(workdir)
        live_context.configure(code_root=code_root, config=self.config)
        try:
            live_context.index_code(code_root)
        except Exception:
            pass  # best-effort; loop still runs with empty context
        if state.task.kind == "paper_to_experiment":
            try:
                live_context.index_docs()  # best-effort: paper/experiment context
            except Exception:
                pass

        # Seed the budget from the checkpoint so max_steps/deadline bound the whole
        # run across pause/resume, not just this process.
        budget = StepBudget(
            max_steps=self.config.max_steps,
            max_repairs=self.config.max_repairs,
            deadline_s=self.config.deadline_s,
            steps_used=state.steps_used,
            prior_elapsed_s=state.elapsed_s,
        )

        # PLAN (once)
        if state.plan is None:
            state.plan = Supervisor(self.config).plan(state.task)
            (run_dir / "plan.json").write_text(state.plan.model_dump_json(indent=2))
            (run_dir / "plan.md").write_text(self._plan_md(state))
            append_ledger(state, StepRecord(seq=state.next_seq(), step_id="-", phase="plan"))
            save_state(state)

        try:
            while not state.is_terminal():
                step = state.next_step()
                if step is None:
                    break
                budget.tick()
                state.steps_used = budget.steps_used  # persist even if the step pauses
                self._run_step(state, step, budget)
                state.elapsed_s = budget.elapsed()
                save_state(state)
        except BudgetExceeded as e:
            state.steps_used = budget.steps_used
            state.elapsed_s = budget.elapsed()
            state.status = "PAUSED"
            save_state(state)
            return RunResult(state, "PAUSED", str(e))
        except ApprovalPending as e:
            state.elapsed_s = budget.elapsed()
            save_state(state)
            return RunResult(state, "AWAITING_APPROVAL", str(e))
        except ApprovalRejected as e:
            state.elapsed_s = budget.elapsed()
            state.status = "FAILED"
            save_state(state)
            return RunResult(state, "FAILED", str(e))

        if not state.is_terminal():
            state.status = "DONE"
        if state.status == "DONE" and state.task.kind == "issue_to_pr":
            self._finalize_pr(state)
        if state.status == "DONE" and state.task.kind == "paper_to_experiment":
            self._finalize_experiment(state)
        if state.status == "DONE" and self.config.use_skill_memory:
            try:
                from ..memory import SkillMemory

                SkillMemory(Path(self.config.data_dir) / "skills").record(state)
            except Exception:
                pass  # skill recording is best-effort
        save_state(state)
        return RunResult(state, state.status)

    # --- one step -----------------------------------------------------------
    def _run_step(self, state: RunState, step, budget: StepBudget) -> None:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        # 1. CONTEXT
        bundle = ContextEngineer(self.config).gather(step)
        (run_dir / "context_bundle.json").write_text(bundle.model_dump_json(indent=2))
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id=step.step_id,
                phase="context",
                artifact_ref="context_bundle.json",
            ),
        )

        # 2/3. EXECUTE (dispatch by action)
        artifact, artifact_ref = self._execute(state, step, bundle)
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id=step.step_id,
                phase="execute",
                artifact_ref=artifact_ref,
            ),
        )

        # HUMAN-APPROVAL GATE (before verify / irreversible boundary)
        if step.requires_approval:
            self._approval_gate(state, step, artifact_ref)

        # 4. VERIFY
        verdict = VerifierAgent(parallel=self.config.parallel_verify).verify(
            step, artifact, VerifyContext(workdir=workdir, step=step, bundle=bundle)
        )
        (run_dir / "verify.json").write_text(verdict.model_dump_json(indent=2))
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id=step.step_id,
                phase="verify",
                verdict_ref="verify.json",
            ),
        )

        # 5. REPAIR or ADVANCE
        if verdict.passed:
            state.complete_step(step)
            append_ledger(
                state, StepRecord(seq=state.next_seq(), step_id=step.step_id, phase="complete")
            )
        elif state.repairs_for(step) < budget.max_repairs:
            state.record_repair(step)
            assert state.plan is not None  # set before the loop body runs
            state.plan.steps[state.cursor] = step.as_repair(verdict.failures)
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id=step.step_id,
                    phase="repair",
                    notes="; ".join(verdict.failures)[:300],
                ),
            )
        else:
            self._revert_step(step, workdir)
            state.fail_current(step)
            append_ledger(
                state, StepRecord(seq=state.next_seq(), step_id=step.step_id, phase="fail")
            )

        # 6. CHECKPOINT
        save_state(state)

    # --- execute dispatch ---------------------------------------------------
    def _execute(self, state: RunState, step, bundle) -> tuple[Any, str]:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        if step.action in ("gather_context", "answer_query"):
            if bundle.answer:
                (run_dir / "answer.md").write_text(bundle.answer)
            return bundle, "context_bundle.json"

        if step.action == "edit_code":
            patch: Patch = Implementer(self.llm).implement(step, bundle, workdir)
            (run_dir / "patch.diff").write_text(patch.unified_diff or "(no diff)\n")
            (run_dir / "patch.json").write_text(patch.model_dump_json(indent=2))
            # Only touch the sandbox if there is a new patch. On a repair pass an
            # idempotent implementer may return an empty patch (already fixed) —
            # keep the prior fix rather than reverting it.
            if not patch.is_empty():
                self._revert_step(step, workdir)  # undo a prior attempt for this step
                _touched, backup = apply_patch(patch, workdir)
                self._backups[step.step_id] = backup
                save_backup(backup, run_dir / "backups" / f"{step.step_id}.json")
            return patch, "patch.diff"

        if step.action == "run_experiment":
            result: ExperimentResult = Experimenter().run(step, bundle, workdir)
            (run_dir / "experiment.json").write_text(result.model_dump_json(indent=2))
            return result, "experiment.json"

        raise ValueError(f"unknown action: {step.action}")

    def _revert_step(self, step, workdir: Path) -> None:
        backup = self._backups.pop(step.step_id, None)
        if backup is None:  # not in memory (e.g. resumed in a fresh process) -> disk
            backup = load_backup(Path(workdir).parent / "backups" / f"{step.step_id}.json")
        if backup is not None:
            revert_patch(backup, workdir)

    # --- approval -----------------------------------------------------------
    def _approval_gate(self, state: RunState, step, artifact_ref: str) -> None:
        gate = HumanApprovalGate(state.run_dir)
        decision = gate.decision()
        if decision is not None:
            gate.clear()
            if not decision.approved:
                # A rejected change must not survive in the sandbox, same as a
                # change that fails verification.
                self._revert_step(step, Path(state.workdir))
                raise ApprovalRejected(f"step {step.step_id} rejected: {decision.note}")
            return
        if self.auto_approve:
            return
        if sys.stdin is not None and sys.stdin.isatty():
            ans = input(f"[approval] proceed with step {step.step_id} ({step.goal})? [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                self._revert_step(step, Path(state.workdir))
                raise ApprovalRejected(f"step {step.step_id} rejected interactively")
            return
        # non-interactive: pause for async approval
        gate.request(step, f"awaiting approval to apply {artifact_ref}")
        state.status = "AWAITING_APPROVAL"
        append_ledger(
            state, StepRecord(seq=state.next_seq(), step_id=step.step_id, phase="approval")
        )
        save_state(state)
        raise ApprovalPending(state.run_id, step.step_id)

    # --- finalize -----------------------------------------------------------
    def _finalize_pr(self, state: RunState) -> None:
        run_dir = Path(state.run_dir)
        patch = Patch(step_id="-")
        patch_json = run_dir / "patch.json"
        if patch_json.exists():
            patch = Patch.model_validate_json(patch_json.read_text())
        verdict = None
        verify_json = run_dir / "verify.json"
        if verify_json.exists():
            verdict = Verdict.model_validate_json(verify_json.read_text())

        checks = []
        if verdict:
            for c in verdict.checks:
                checks.append(
                    f"{c.name}: {c.detail.get('summary', 'passed' if c.passed else 'failed')}"
                )
        pr = PRSummary(
            title=state.task.title,
            task_id=state.run_id,
            rationale=patch.rationale or state.task.description or state.task.title,
            files_changed=patch.touched_files,
            checks=checks,
            provenance=patch.based_on_context,
        )
        path = run_dir / "pr_summary.md"
        path.write_text(pr.to_markdown())
        state.pr_summary_path = str(path)

    def _finalize_experiment(self, state: RunState) -> None:
        run_dir = Path(state.run_dir)
        result = ExperimentResult(step_id="-")
        exp_json = run_dir / "experiment.json"
        if exp_json.exists():
            result = ExperimentResult.model_validate_json(exp_json.read_text())
        verdict = None
        verify_json = run_dir / "verify.json"
        if verify_json.exists():
            verdict = Verdict.model_validate_json(verify_json.read_text())

        # Prefer verifier-recomputed metrics (authoritative) over self-reported.
        metrics: dict[str, float] = dict(result.metrics)
        checks: list[str] = []
        reproducible = False
        if verdict:
            for c in verdict.checks:
                checks.append(
                    f"{c.name}: {c.detail.get('summary', 'passed' if c.passed else 'failed')}"
                )
                if c.name in ("psnr", "ssim") and c.score is not None:
                    metrics[c.name] = round(c.score, 6)
                if c.name == "reproducibility":
                    reproducible = c.passed

        summary = ExperimentSummary(
            title=state.task.title,
            task_id=state.run_id,
            metrics=metrics,
            checks=checks,
            reproducible=reproducible,
            provenance=result.based_on_context,
        )
        path = run_dir / "experiment_summary.md"
        path.write_text(summary.to_markdown())
        state.pr_summary_path = str(path)

    # --- helpers ------------------------------------------------------------
    def _prepare_workdir(self, task: TaskSpec, workdir: Path) -> None:
        if workdir.exists():
            return  # resume / re-run reuse
        if task.target_repo:
            src = Path(task.target_repo)
            if not src.exists():
                raise FileNotFoundError(f"target_repo not found: {src}")
            shutil.copytree(src, workdir, ignore=_IGNORE)
        else:
            workdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _plan_md(state: RunState) -> str:
        plan = state.plan
        assert plan is not None  # only called after planning
        lines = [f"# Plan — {plan.summary}", ""]
        for i, s in enumerate(plan.steps):
            lines.append(f"{i + 1}. **{s.step_id}** ({s.action}) — {s.goal}")
            if s.verifiers:
                lines.append(f"   - verifiers: {', '.join(s.verifiers)}")
        return "\n".join(lines)
