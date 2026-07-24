"""The verification loop.

    context -> tool/execute -> verify -> repair -> checkpoint -> repeat

with max-steps, checkpoint/resume, and a human-approval gate. No LangGraph; the
state is a JSON checkpoint + a step ledger, shaped so LangGraph drops in later.
"""

from __future__ import annotations

import logging
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
from ..tools import policy
from ..tools.patch import Backup, apply_patch, load_backup, revert_patch, save_backup
from ..verifiers import VerifyContext
from ..verifiers.verdict import Check, Verdict
from .approval import HumanApprovalGate
from .budget import StepBudget
from .checkpoint import append_ledger, load_state_by_id, save_state
from .errors import ApprovalPending, ApprovalRejected, BudgetExceeded, PolicyViolation
from .manifest import build_manifest, sha256_bytes
from .state import RunState, StepRecord

logger = logging.getLogger(__name__)

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


def _safe_seg(step_id: str) -> str:
    """A single, path-safe filesystem segment for a step_id.

    A dynamic (LLM-generated) plan could carry a step_id like ``../../escape``;
    collapsing path separators and leading/trailing dots keeps it inside the run dir.
    """
    seg = re.sub(r"[^A-Za-z0-9_.-]", "_", step_id).strip(".")
    return seg or "step"


def _dump(run_dir: Path, step_id: str, name: str, text: str) -> None:
    """Write an artifact both flat (back-compat / last-writer) and per-step under
    ``steps/<step_id>/`` so a multi-step plan keeps every step's provenance."""
    (run_dir / name).write_text(text)
    step_dir = run_dir / "steps" / _safe_seg(step_id)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / name).write_text(text)


def _gen_run_id(task: TaskSpec) -> str:
    return f"{now():%Y%m%d-%H%M%S}-{_slug(task.title)}-{uuid.uuid4().hex[:4]}"


def _policy_verdict(step_id: str, e: PolicyViolation) -> Verdict:
    """A failing verdict for a policy-refused patch (never applied)."""
    return Verdict.from_checks(
        step_id,
        [
            Check(
                name="oracle-policy",
                family="code",
                passed=False,
                detail={
                    "summary": (
                        "patch refused: it modifies protected oracle/config files "
                        f"({', '.join(e.violations)}); fix the source instead"
                    ),
                    "violations": e.violations,
                },
            )
        ],
    )


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
        except Exception:  # best-effort; loop still runs with empty context
            logger.debug("index_code(%s) failed", code_root, exc_info=True)
        if state.task.kind == "paper_to_experiment":
            try:
                live_context.index_docs()  # best-effort: paper/experiment context
            except Exception:
                logger.debug("index_docs() failed", exc_info=True)

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
            state.plan = Supervisor(self.config, self.llm).plan(state.task)
            (run_dir / "plan.json").write_text(state.plan.model_dump_json(indent=2))
            (run_dir / "plan.md").write_text(self._plan_md(state))
            append_ledger(state, StepRecord(seq=state.next_seq(), step_id="-", phase="plan"))
            save_state(state)

        in_flight = None  # the step currently executing, for revert on an unexpected fault
        try:
            while not state.is_terminal():
                step = state.next_step()
                if step is None:
                    break
                budget.tick()
                state.steps_used = budget.steps_used  # persist even if the step pauses
                in_flight = step
                self._run_step(state, step, budget)
                in_flight = None  # finished cleanly (cursor may have advanced)
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
        except Exception as e:
            # An unexpected fault mid-step (unknown action, failed patch apply, a
            # crashing tool) must not leave the run wedged at RUNNING with a
            # half-applied sandbox: revert the in-flight step, fail closed, checkpoint.
            if in_flight is not None:
                try:
                    self._revert_step(in_flight, workdir)
                except Exception:  # a revert failure must not abort the fail-closed path
                    logger.exception("revert failed for step %s", in_flight.step_id)
                append_ledger(
                    state,
                    StepRecord(
                        seq=state.next_seq(),
                        step_id=in_flight.step_id,
                        phase="fail",
                        notes=f"error: {type(e).__name__}: {e}"[:300],
                    ),
                )
                state.fail_current(in_flight)
            else:
                state.status = "FAILED"
            state.elapsed_s = budget.elapsed()
            save_state(state)
            return RunResult(state, "FAILED", f"{type(e).__name__}: {e}")

        if not state.is_terminal():
            state.status = "DONE"
        if state.status == "DONE" and state.task.kind == "issue_to_pr":
            self._finalize_pr(state)
        if state.status == "DONE" and state.task.kind == "paper_to_experiment":
            self._finalize_experiment(state)
        if state.status == "DONE" and self.config.use_skill_memory:
            try:
                from ..memory import SkillMemory

                if SkillMemory(Path(self.config.data_dir) / "skills").record(state) is not None:
                    # Re-index so the new skill is retrievable on the next run; otherwise
                    # the memory loop stays open for issue_to_pr runs.
                    live_context.index_docs(("skill",))
            except Exception:  # skill recording is best-effort
                logger.debug("skill recording failed", exc_info=True)
        save_state(state)
        return RunResult(state, state.status)

    # --- one step -----------------------------------------------------------
    def _run_step(self, state: RunState, step, budget: StepBudget) -> None:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        # 1. CONTEXT
        bundle = ContextEngineer(self.config).gather(step)
        _dump(run_dir, step.step_id, "context_bundle.json", bundle.model_dump_json(indent=2))
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
        try:
            artifact, artifact_ref = self._execute(state, step, bundle)
        except PolicyViolation as e:
            # The patch never reached the sandbox. Record a failing verdict so
            # the repair loop is told exactly why, instead of aborting the run.
            verdict = _policy_verdict(step.step_id, e)
            _dump(run_dir, step.step_id, "verify.json", verdict.model_dump_json(indent=2))
            append_ledger(
                state,
                StepRecord(
                    seq=state.next_seq(),
                    step_id=step.step_id,
                    phase="verify",
                    verdict_ref="verify.json",
                    notes=str(e)[:300],
                ),
            )
            self._repair_or_fail(state, step, verdict, budget, workdir)
            save_state(state)
            return
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
        _dump(run_dir, step.step_id, "verify.json", verdict.model_dump_json(indent=2))
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
        else:
            self._repair_or_fail(state, step, verdict, budget, workdir)

        # 6. CHECKPOINT
        save_state(state)

    def _repair_or_fail(self, state: RunState, step, verdict: Verdict, budget, workdir) -> None:
        """Re-issue the step as a repair with the verdict's failures, or fail the run."""
        if state.repairs_for(step) < budget.max_repairs:
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

    # --- execute dispatch ---------------------------------------------------
    def _execute(self, state: RunState, step, bundle) -> tuple[Any, str]:
        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        if step.action in ("gather_context", "answer_query"):
            if bundle.answer:
                (run_dir / "answer.md").write_text(bundle.answer)
            return bundle, "context_bundle.json"

        if step.action == "edit_code":
            # On resume after approval, reuse the exact patch the human reviewed
            # rather than regenerating it (a non-deterministic backend could differ).
            patch = self._approved_patch(state, step)
            if patch is None:
                patch = Implementer(self.llm).implement(step, bundle, workdir)
            (run_dir / "patch.diff").write_text(patch.unified_diff or "(no diff)\n")
            patch_json = patch.model_dump_json(indent=2)
            _dump(run_dir, step.step_id, "patch.json", patch_json)
            # Oracle protection: a patch that touches the test suite or the
            # verifier/build config is refused BEFORE it can reach the sandbox,
            # unless the task manifest explicitly authorizes those exact paths.
            violations = policy.check_patch(patch, state.task.allowed_protected_files)
            if violations:
                raise PolicyViolation(step.step_id, violations)
            # Immutable artifact manifest: what an approval binds to (exact
            # bytes, touched files, pre-apply base state, verifier/policy config).
            manifest = build_manifest(
                step=step,
                patch=patch,
                patch_json_bytes=patch_json.encode("utf-8"),
                workdir=workdir,
                policy_overrides=list(state.task.allowed_protected_files),
            )
            _dump(run_dir, step.step_id, "manifest.json", manifest.model_dump_json(indent=2))
            # Only touch the sandbox if there is a new patch. On a repair pass an
            # idempotent implementer may return an empty patch (already fixed) —
            # keep the prior fix rather than reverting it.
            if not patch.is_empty():
                self._revert_step(step, workdir)  # undo a prior attempt for this step
                _touched, backup = apply_patch(patch, workdir)
                self._backups[step.step_id] = backup
                save_backup(backup, run_dir / "backups" / f"{_safe_seg(step.step_id)}.json")
            return patch, "patch.diff"

        if step.action == "run_experiment":
            result: ExperimentResult = Experimenter().run(step, bundle, workdir)
            _dump(run_dir, step.step_id, "experiment.json", result.model_dump_json(indent=2))
            return result, "experiment.json"

        raise ValueError(f"unknown action: {step.action}")

    def _revert_step(self, step, workdir: Path) -> None:
        backup = self._backups.pop(step.step_id, None)
        if backup is None:  # not in memory (e.g. resumed in a fresh process) -> disk
            backup = load_backup(
                Path(workdir).parent / "backups" / f"{_safe_seg(step.step_id)}.json"
            )
        if backup is not None:
            revert_patch(backup, workdir)

    def _patch_bytes(self, state: RunState, step) -> bytes | None:
        """The persisted patch.json bytes for this step (per-step file first)."""
        candidates = [
            Path(state.run_dir) / "steps" / _safe_seg(step.step_id) / "patch.json",
            Path(state.run_dir) / "patch.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return path.read_bytes()
                except OSError:
                    return None
        return None

    def _artifact_sha(self, state: RunState, step) -> str | None:
        data = self._patch_bytes(state, step)
        return sha256_bytes(data) if data is not None else None

    def _approved_patch(self, state: RunState, step) -> Patch | None:
        """The persisted patch a human already approved for this step, if any.

        Binds the approval to the artifact: on resume, an approval-gated step
        may only execute the exact bytes the human reviewed. A decision whose
        artifact hash no longer matches the persisted patch is tamper evidence
        and fails the run rather than executing an unreviewed change.
        """
        if not step.requires_approval:
            return None
        decision = HumanApprovalGate(state.run_dir).decision()
        if decision is None or decision.step_id != step.step_id:
            return None  # no decision, or one for a different step (gate clears it)
        data = self._patch_bytes(state, step)
        if data is None:
            return None
        if not decision.binds(step.step_id, sha256_bytes(data)):
            # The decision names this step but not these bytes: the artifact
            # changed after review. Never execute it; revert and fail closed.
            self._revert_step(step, Path(state.workdir))
            raise ApprovalRejected(
                f"approved artifact for step {step.step_id} does not match the "
                "persisted patch (hash mismatch) — refusing to execute"
            )
        try:
            return Patch.model_validate_json(data)
        except Exception:
            return None

    # --- approval -----------------------------------------------------------
    def _approval_gate(self, state: RunState, step, artifact_ref: str) -> None:
        gate = HumanApprovalGate(state.run_dir)
        decision = gate.decision()
        artifact_sha = self._artifact_sha(state, step)
        if decision is not None and not decision.binds(step.step_id, artifact_sha):
            # A decision for a different step, a different artifact, or one that
            # names neither must not approve this change. Clear it and re-request.
            gate.clear()
            decision = None
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
        # non-interactive: pause for async approval, binding the request to the
        # exact artifact bytes under review
        gate.request(
            step,
            f"awaiting approval to apply {artifact_ref}",
            artifact_sha256=artifact_sha,
        )
        state.status = "AWAITING_APPROVAL"
        append_ledger(
            state, StepRecord(seq=state.next_seq(), step_id=step.step_id, phase="approval")
        )
        save_state(state)
        raise ApprovalPending(state.run_id, step.step_id)

    # --- finalize -----------------------------------------------------------
    def _materializing_step_id(self, state: RunState, action: str) -> str | None:
        """The last completed step with this action — whose artifacts the finalizer
        should report. Robust to plan order rather than 'whichever file wrote last'."""
        if state.plan is None:
            return None
        done = set(state.completed_steps)
        ids = [s.step_id for s in state.plan.steps if s.action == action and s.step_id in done]
        return ids[-1] if ids else None

    def _load_artifact(self, state: RunState, step_id: str | None, name: str, model):
        """Load a per-step artifact (``steps/<id>/<name>``), falling back to the flat file."""
        run_dir = Path(state.run_dir)
        per_step = [run_dir / "steps" / _safe_seg(step_id) / name] if step_id else []
        candidates = per_step + [run_dir / name]
        for path in candidates:
            if path.exists():
                try:
                    return model.model_validate_json(path.read_text())
                except Exception:
                    continue
        return None

    def _finalize_pr(self, state: RunState) -> None:
        run_dir = Path(state.run_dir)
        sid = self._materializing_step_id(state, "edit_code")
        patch = self._load_artifact(state, sid, "patch.json", Patch) or Patch(step_id="-")
        verdict = self._load_artifact(state, sid, "verify.json", Verdict)

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
        sid = self._materializing_step_id(state, "run_experiment")
        result = self._load_artifact(state, sid, "experiment.json", ExperimentResult) or (
            ExperimentResult(step_id="-")
        )
        verdict = self._load_artifact(state, sid, "verify.json", Verdict)

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
