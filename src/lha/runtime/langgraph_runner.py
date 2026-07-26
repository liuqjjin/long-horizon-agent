"""Opt-in LangGraph durable runtime.

Runs the same plan/agents/verifiers as the default loop, but drives them through a
``StateGraph`` checkpointed by ``SqliteSaver`` (durable, resumable by thread_id)
with ``interrupt()`` for the human-approval gate. It reuses the Harness's
execute/finalize helpers, so there is one implementation of the actual work.

The graph is split into three nodes precisely so approval cannot re-execute
work: ``prepare`` (context + execute, checkpointed BEFORE the interrupt),
``gate`` (only the interrupt), and ``verify``. A LangGraph interrupt replays
its node from the top on resume — with a single node that replay would call the
implementer again and could apply a patch the human never saw. With the split,
resume replays only ``gate``, and ``verify`` grades the artifact persisted by
``prepare``, whose SHA-256 the approval decision is bound to.

Enable with ``lha run --runtime langgraph`` (and ``lha resume --runtime langgraph``).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Hashable
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from .. import live_context
from ..agents import Supervisor, VerifierAgent
from ..artifacts import ExperimentResult, Patch
from ..clock import now
from ..config import Config
from ..harness.approval import (
    HumanApprovalGate,
    validate_decision_binding,
)
from ..harness.checkpoint import (
    append_ledger,
    load_state,
    load_state_by_id,
    read_ledger,
    run_lock,
    save_state,
    validate_run_id,
)
from ..harness.errors import (
    ApprovalRejected,
    BudgetExceeded,
    CheckpointCorrupt,
    PolicyViolation,
    TransactionCorrupt,
)
from ..harness.loop import (
    Harness,
    RunResult,
    _claim_run_dir,
    _gen_run_id,
    _initial_plan_ref,
    _persist_verdict,
    _policy_verdict,
    _repair_plan_ref,
    _safe_seg,
    _write_immutable,
)
from ..harness.manifest import sha256_bytes
from ..harness.state import RUN_STATE_SCHEMA, Phase, RunState, StepRecord
from ..harness.transaction import validate_transaction_journals
from ..live_context.models import ContextBundle
from ..repo_adapter import RepoIntegrityResult, RepoStageResult
from ..verifiers import VerifyContext

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    rs: dict  # RunState.model_dump(mode="json")
    # routing hint set by each node: "gate" | "verify" | "continue" | "done"
    next: NotRequired[str]


class LangGraphHarness:
    """Durable runtime sharing all logic with the default Harness."""

    def __init__(self, config: Config | None = None, *, auto_approve: bool = False):
        self.config = config or Config.from_env()
        self.auto_approve = auto_approve
        self._h = Harness(self.config, auto_approve=auto_approve)  # reuse execute/finalize

    # --- public entry points ----------------------------------------------
    def run(self, task, *, run_id: str | None = None) -> RunResult:
        run_id = run_id or _gen_run_id(task)
        run_dir = _claim_run_dir(self.config.runs_dir, run_id)
        with run_lock(run_dir):
            workdir = run_dir / "workdir"
            self._h._prepare_workdir(task, workdir)
            state = RunState.new(
                task,
                run_id,
                str(run_dir),
                str(workdir),
                config=self.config,
            )
            save_state(state)
            return self._drive(state)

    def resume(self, run_id: str) -> RunResult:
        validate_run_id(run_id)
        run_dir = Path(self.config.runs_dir).resolve() / run_id
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise CheckpointCorrupt(f"run directory is missing or unsafe: {run_dir}")
        with run_lock(run_dir):
            state = load_state_by_id(self.config.runs_dir, run_id)
            if state.schema_version != RUN_STATE_SCHEMA:
                raise CheckpointCorrupt(
                    f"run {run_id} uses state schema {state.schema_version}; "
                    f"schema {RUN_STATE_SCHEMA} is required for safe resume"
                )
            limits = state.require_matching_budget_limits(self.config)
            if state.is_terminal():
                return RunResult(state, state.status, "run already terminal")
            state.recover_active_elapsed()
            records = read_ledger(state.run_dir)
            if records:
                state.seq = max(state.seq, *(record.seq for record in records))
            try:
                validate_transaction_journals(Path(state.run_dir))
            except TransactionCorrupt as error:
                raise TransactionCorrupt(
                    f"run recovery evidence is invalid: {error}"
                ) from error
            records = self._h._reconcile_durable_ledger(state, records)
            if records:
                state.seq = max(state.seq, *(record.seq for record in records))
            if state.is_terminal():
                save_state(state)
                return RunResult(
                    state,
                    state.status,
                    "terminal state recovered from durable ledger",
                )
            if (
                limits.deadline_s is not None
                and state.elapsed_s > limits.deadline_s
            ):
                state.status = "PAUSED"
                save_state(state)
                return RunResult(
                    state,
                    "PAUSED",
                    f"deadline {limits.deadline_s}s exceeded during interrupted activity",
                )
            state.status = "RUNNING"
            save_state(state)
            return self._drive(state)

    # --- driver ------------------------------------------------------------
    def _drive(self, state: RunState) -> RunResult:
        from langgraph.types import Command

        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)
        # Wall-clock budget across nodes: prior_elapsed (from earlier processes) plus
        # this process's monotonic delta gives the run's total elapsed for deadline_s.
        self._t0 = time.monotonic()
        self._prior_elapsed = state.elapsed_s
        from ..llm.trace import TracedLLM

        if isinstance(self._h.llm, TracedLLM):
            self._h.llm.bind(run_dir)
            self._h.llm.restore_totals(state.llm_usage)
        code_root = state.task.target_repo or str(workdir)
        live_context.configure(code_root=code_root, config=self.config)
        try:
            live_context.index_code(code_root)
        except Exception:  # best-effort; loop still runs with empty context
            logger.debug("index_code(%s) failed", code_root, exc_info=True)
        if state.task.kind == "paper_to_experiment":
            try:
                live_context.index_docs()
            except Exception:
                logger.debug("index_docs() failed", exc_info=True)

        if state.plan is None:
            state.active_since = now()
            self._h._save(state)
            try:
                if isinstance(self._h.llm, TracedLLM):
                    self._h.llm.set_call_context(
                        run_id=state.run_id,
                        attempt_id="plan",
                        task=state.task.model_dump(mode="json"),
                        config=self.config.model_dump(mode="json"),
                    )
                state.plan = Supervisor(self.config, self._h.llm).plan(
                    state.task
                )
                from ..harness.transaction import durable_artifact_write

                plan_bytes = state.plan.model_dump_json(indent=2).encode(
                    "utf-8"
                )
                initial_ref = _initial_plan_ref()
                _write_immutable(run_dir / initial_ref, plan_bytes)
                durable_artifact_write(run_dir / "plan.json", plan_bytes)
                append_ledger(
                    state,
                    StepRecord(
                        seq=state.next_seq(),
                        step_id="-",
                        phase="plan",
                        artifact_ref=initial_ref,
                        evidence_sha256=sha256_bytes(plan_bytes),
                        idempotency_key=f"{state.run_id}:plan",
                    ),
                )
                state.elapsed_s = self._prior_elapsed + (
                    time.monotonic() - self._t0
                )
                state.active_since = None
                self._h._save(state)
            except BudgetExceeded as error:
                state.elapsed_s = self._prior_elapsed + (
                    time.monotonic() - self._t0
                )
                state.active_since = None
                state.status = "PAUSED"
                self._h._save(state)
                return RunResult(state, "PAUSED", str(error))
            except Exception as error:
                state.elapsed_s = self._prior_elapsed + (
                    time.monotonic() - self._t0
                )
                state.active_since = None
                state.status = "FAILED"
                self._h._save(state)
                return RunResult(
                    state,
                    "FAILED",
                    f"{type(error).__name__}: {error}",
                )

        conn = sqlite3.connect(str(run_dir / "graph.sqlite"), check_same_thread=False)
        try:
            graph = self._compile(conn)
            gcfg: Any = {"configurable": {"thread_id": state.run_id}}  # LangGraph RunnableConfig
            snap = graph.get_state(gcfg)
            has_interrupt = bool(snap.next) and bool(snap.interrupts or [])

            if snap.next and not has_interrupt:
                # A previous process died mid-node: the checkpoint stopped inside
                # the graph with no interrupt pending. Invoking with no input
                # resumes from the checkpoint and re-runs the pending node —
                # this is a crash recovery, not an approval wait.
                resume_input: Any = None
            elif has_interrupt:  # paused at a real interrupt (awaiting approval)
                gate = HumanApprovalGate(run_dir)
                interrupted_step = None
                interrupted_attempt = None
                for intr in snap.interrupts or []:
                    payload = getattr(intr, "value", {}) or {}
                    interrupted_step = payload.get("step_id")
                    interrupted_attempt = payload.get("attempt_id")
                current_step = state.next_step()
                if (
                    current_step is None
                    or not isinstance(interrupted_step, str)
                    or not isinstance(interrupted_attempt, str)
                    or current_step.step_id != interrupted_step
                    or state.attempt_id(current_step) != interrupted_attempt
                ):
                    final = load_state_by_id(
                        self.config.runs_dir, state.run_id
                    )
                    final.status = "FAILED"
                    self._h._save(final)
                    return RunResult(
                        final,
                        "FAILED",
                        "approval interrupt does not match the current attempt",
                    )
                try:
                    request = gate.request_evidence(
                        interrupted_step, interrupted_attempt
                    )
                    decision_evidence = gate.decision_evidence(
                        interrupted_step, interrupted_attempt
                    )
                except ValueError as error:
                    return self._fail_invalid_approval(
                        state,
                        current_step,
                        f"invalid approval evidence: {error}",
                    )
                if decision_evidence is None:
                    return RunResult(
                        load_state_by_id(self.config.runs_dir, state.run_id),
                        "AWAITING_APPROVAL",
                        "approval pending",
                    )
                # The decision must bind the exact persisted artifact bytes. A
                # mismatch means the artifact changed after review — tamper
                # evidence, so the run fails closed with the change reverted.
                current_sha = _current_artifact_sha(
                    run_dir, interrupted_step, current_step.action
                )
                try:
                    if request is None:
                        raise ValueError("immutable approval request is missing")
                    validate_decision_binding(
                        request=request,
                        decision=decision_evidence,
                        step_id=interrupted_step,
                        attempt_id=interrupted_attempt,
                        goal=current_step.goal,
                        artifact_sha256=current_sha,
                    )
                except ValueError as error:
                    return self._fail_invalid_approval(
                        state,
                        current_step,
                        "approved artifact does not match the persisted patch "
                        f"or request: {error}",
                    )
                resume_input = Command(
                    resume={
                        "approved": decision_evidence.value.approved,
                        "decision_sha256": decision_evidence.sha256,
                    }
                )
            else:
                resume_input = {"rs": state.model_dump(mode="json")}

            try:
                graph.invoke(resume_input, gcfg)
            except BudgetExceeded as e:
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.recover_active_elapsed()
                final.status = "PAUSED"
                self._h._save(final)
                return RunResult(final, "PAUSED", str(e))
            except ApprovalRejected as e:
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.recover_active_elapsed()
                final.status = "FAILED"
                self._h._save(final)
                return RunResult(final, "FAILED", str(e))
            except Exception as e:
                # Same fail-closed contract as the default loop: a mid-node fault
                # must not leave the run wedged at RUNNING with an unverified
                # patch in the sandbox.
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.recover_active_elapsed()
                step = final.next_step()
                if step is not None and step.step_id not in final.completed_steps:
                    try:
                        self._h._revert_step(step, workdir)
                    except Exception:
                        logger.exception("revert failed for step %s", step.step_id)
                    final.fail_current(step)
                else:
                    final.status = "FAILED"
                self._h._save(final)
                return RunResult(final, "FAILED", f"{type(e).__name__}: {e}")

            # paused on a fresh interrupt?
            snap = graph.get_state(gcfg)
            if snap.next and (snap.interrupts or []):
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.status = "AWAITING_APPROVAL"
                self._h._save(final)
                self._request_approval(snap, final)
                return RunResult(final, "AWAITING_APPROVAL", "awaiting approval")
            if snap.next:  # stopped mid-graph without an interrupt: wedged, fail closed
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.status = "FAILED"
                self._h._save(final)
                return RunResult(final, "FAILED", "graph stopped mid-run without an interrupt")
        finally:
            conn.close()

        final = load_state_by_id(self.config.runs_dir, state.run_id)
        final.recover_active_elapsed()
        if final.status == "PAUSED":  # budget exhausted mid-graph
            return RunResult(final, "PAUSED", "budget exceeded")
        if not final.is_terminal():
            final.status = "DONE"
        if final.status == "DONE":
            self._finalize(final)
        self._h._save(final)
        return RunResult(final, final.status)

    # --- graph -------------------------------------------------------------
    def _compile(self, conn: sqlite3.Connection):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, StateGraph

        saver = SqliteSaver(conn)
        saver.setup()
        builder = StateGraph(GraphState)
        builder.add_node("prepare", self._prepare_node)  # type: ignore[arg-type]
        builder.add_node("gate", self._gate_node)  # type: ignore[arg-type]
        builder.add_node("verify", self._verify_node)  # type: ignore[arg-type]
        builder.add_edge(START, "prepare")
        routes: dict[Hashable, str] = {
            "gate": "gate",
            "verify": "verify",
            "continue": "prepare",
            "done": END,
        }
        builder.add_conditional_edges("prepare", _route_next, routes)
        builder.add_conditional_edges("gate", _route_next, routes)
        builder.add_conditional_edges("verify", _route_next, routes)
        return builder.compile(checkpointer=saver)

    def _ledger(self, state: RunState, step, phase: Phase, **kw) -> None:
        attempt_id = kw.pop("attempt_id", None) or state.attempt_id(step)
        idempotency_key = kw.pop("idempotency_key", None) or f"{attempt_id}:{phase}"
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id=step.step_id,
                phase=phase,
                attempt_id=attempt_id,
                idempotency_key=idempotency_key,
                **kw,
            ),
        )

    def _repair_or_fail(self, state: RunState, step, verdict, workdir: Path) -> None:
        non_retryable = any(check.detail.get("non_retryable") for check in verdict.checks)
        limits = state.require_matching_budget_limits(self.config)
        if not non_retryable and state.repairs_for(step) < limits.max_repairs:
            assert state.plan is not None  # plan set before stepping
            state.plan.steps[state.cursor] = step.as_repair(verdict.failures)
            from ..harness.transaction import durable_artifact_write

            plan_bytes = state.plan.model_dump_json(indent=2).encode("utf-8")
            attempt_id = state.attempt_id(step)
            repair_ref = _repair_plan_ref(attempt_id)
            _write_immutable(Path(state.run_dir) / repair_ref, plan_bytes)
            durable_artifact_write(
                Path(state.run_dir) / "plan.json",
                plan_bytes,
            )
            self._ledger(
                state,
                step,
                "repair",
                artifact_ref=repair_ref,
                evidence_sha256=sha256_bytes(plan_bytes),
                notes="; ".join(verdict.failures)[:300],
            )
            state.record_repair(step)
        else:
            # A step that exhausts its repairs is reverted so an unverified
            # change never survives in the sandbox (same as the default loop).
            self._h._revert_step(step, workdir)
            state.fail_current(step)
            self._ledger(state, step, "fail")

    def _prepare_node(self, gstate: GraphState) -> GraphState:
        """Context + execute. Checkpointed BEFORE any approval interrupt, so a
        resume can never re-run the implementer."""
        state = RunState.model_validate(gstate["rs"])
        # A LangGraph node checkpoint is written only after the node returns.
        # The file checkpoint records the budget before side effects, so a
        # process death inside this node must reload that newer counter.
        disk_state = load_state(state.run_dir)
        if (
            disk_state.run_id == state.run_id
            and disk_state.cursor == state.cursor
            and disk_state.seq >= state.seq
        ):
            state = disk_state
        step = state.next_step()
        if step is None or state.is_terminal():
            return {"rs": state.model_dump(mode="json"), "next": "done"}
        if state.active_since is not None:
            state.recover_active_elapsed()
        state.elapsed_s = max(state.elapsed_s, self._prior_elapsed)

        # Bound the run: max_steps caps prepare executions and deadline_s caps
        # wall-clock, both persisted so they survive resume.
        elapsed = self._prior_elapsed + (time.monotonic() - self._t0)
        limits = state.require_matching_budget_limits(self.config)
        deadline = limits.deadline_s
        if deadline is not None and elapsed > deadline:
            state.status = "PAUSED"
            state.elapsed_s = elapsed
            state.active_since = None
            self._h._save(state)
            return {"rs": state.model_dump(mode="json"), "next": "done"}
        if not state.attempt_is_budgeted(step):
            if state.steps_used >= limits.max_steps:
                state.status = "PAUSED"
                state.elapsed_s = elapsed
                self._h._save(state)
                return {"rs": state.model_dump(mode="json"), "next": "done"}
            state.steps_used += 1
            state.mark_attempt_budgeted(step)
        state.elapsed_s = elapsed
        state.active_since = now()
        self._h._save(state)

        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        attempt_id = state.attempt_id(step)
        bundle = self._h._context_for_attempt(
            state, step, attempt_id, workdir
        )
        try:
            _artifact, ref = self._h._execute(state, step, bundle)
        except PolicyViolation as e:
            # Same oracle protection as the default loop: the patch was refused
            # before it reached the sandbox; feed the reason to the repair loop.
            verdict = _policy_verdict(step.step_id, e)
            attempt_id = state.attempt_id(step)
            verdict = self._h._bind_verdict(
                verdict,
                state,
                step,
                artifact_ref="patch.json",
                attempt_id=attempt_id,
            )
            verdict_json = verdict.model_dump_json(indent=2)
            verdict_sha = sha256_bytes(verdict_json.encode("utf-8"))
            verdict_ref = _persist_verdict(
                run_dir, step.step_id, attempt_id, verdict_json
            )
            self._ledger(
                state,
                step,
                "verify",
                verdict_ref=verdict_ref,
                evidence_sha256=verdict_sha,
                notes=str(e)[:300],
            )
            self._repair_or_fail(state, step, verdict, workdir)
            state.elapsed_s = self._prior_elapsed + (
                time.monotonic() - self._t0
            )
            state.active_since = None
            self._h._save(state)
            nxt = "done" if state.is_terminal() else "continue"
            return {"rs": state.model_dump(mode="json"), "next": nxt}
        attempt_id = state.attempt_id(step)
        execute_ref, execute_sha = self._h._execution_ledger_binding(
            state, step, attempt_id, ref
        )
        self._ledger(
            state,
            step,
            "execute",
            artifact_ref=execute_ref,
            evidence_sha256=execute_sha,
            attempt_id=attempt_id,
        )
        state.elapsed_s = self._prior_elapsed + (time.monotonic() - self._t0)
        state.active_since = None
        self._h._save(state)

        if step.requires_approval and self.auto_approve:
            self._h._approval_gate(state, step, ref)
            self._h._save(state)
        needs_gate = step.requires_approval and not self.auto_approve
        return {"rs": state.model_dump(mode="json"), "next": "gate" if needs_gate else "verify"}

    def _gate_node(self, gstate: GraphState) -> GraphState:
        """Only the approval interrupt lives here. On resume this node replays
        from the top, which re-raises the interrupt — nothing else re-runs."""
        from langgraph.types import interrupt

        state = RunState.model_validate(gstate["rs"])
        step = state.next_step()
        if step is None or state.is_terminal():
            return {"rs": state.model_dump(mode="json"), "next": "done"}

        run_dir = Path(state.run_dir)
        sha = _current_artifact_sha(run_dir, step.step_id, step.action)
        attempt_id = state.attempt_id(step)
        resumed = interrupt(
            {
                "step_id": step.step_id,
                "attempt_id": attempt_id,
                "goal": step.goal,
                "artifact_sha256": sha,
            }
        )
        gate = HumanApprovalGate(run_dir)
        evidence = gate.decision_evidence(step.step_id, attempt_id)
        if (
            evidence is None
            or not isinstance(resumed, dict)
            or resumed.get("approved") is not evidence.value.approved
            or resumed.get("decision_sha256") != evidence.sha256
        ):
            raise ApprovalRejected(
                f"approval resume payload is not bound to {attempt_id}"
            )
        decision = self._h._consume_approval_decision(
            state, step, artifact_sha=sha
        )
        if not decision.approved:
            self._h._revert_step(step, Path(state.workdir))  # rejected change must not survive
            state.fail_current(step)
            self._ledger(
                state,
                step,
                "fail",
                notes="approval rejected",
            )
            self._h._save(state)
            return {"rs": state.model_dump(mode="json"), "next": "done"}
        self._h._save(state)
        return {"rs": state.model_dump(mode="json"), "next": "verify"}

    def _verify_node(self, gstate: GraphState) -> GraphState:
        """Grade the artifact ``prepare`` persisted — never a regenerated one."""
        state = RunState.model_validate(gstate["rs"])
        disk_state = load_state(state.run_dir)
        if (
            disk_state.seq > state.seq
            and (
                disk_state.cursor != state.cursor
                or disk_state.repairs != state.repairs
                or disk_state.is_terminal()
            )
        ):
            next_route = (
                "done"
                if disk_state.is_terminal() or disk_state.next_step() is None
                else "continue"
            )
            return {
                "rs": disk_state.model_dump(mode="json"),
                "next": next_route,
            }
        step = state.next_step()
        if step is None or state.is_terminal():
            return {"rs": state.model_dump(mode="json"), "next": "done"}

        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)
        state.active_since = now()
        self._h._save(state)
        artifact, bundle = _load_step_artifacts(run_dir, step)

        verdict = VerifierAgent(parallel=self.config.parallel_verify).verify(
            step,
            artifact,
            VerifyContext(
                workdir=workdir,
                step=step,
                bundle=bundle,
                exec=self._h.exec,
                attempt_id=state.attempt_id(step),
            ),
        )
        attempt_id = state.attempt_id(step)
        artifact_ref = "patch.diff" if step.action == "edit_code" else {
            "gather_context": "context_bundle.json",
            "answer_query": "context_bundle.json",
            "run_experiment": "experiment.json",
            "repo_integrity": "repo_integrity.json",
            "repo_stage": "repo_stage.json",
        }[step.action]
        verdict = self._h._bind_verdict(
            verdict,
            state,
            step,
            artifact_ref=artifact_ref,
            attempt_id=attempt_id,
        )
        verdict_json = verdict.model_dump_json(indent=2)
        verdict_sha = sha256_bytes(verdict_json.encode("utf-8"))
        verdict_ref = _persist_verdict(
            run_dir, step.step_id, attempt_id, verdict_json
        )
        self._ledger(
            state,
            step,
            "verify",
            verdict_ref=verdict_ref,
            evidence_sha256=verdict_sha,
        )

        if verdict.passed:
            self._h._mark_step_verified(state, step, workdir)
            state.complete_step(step)
            self._ledger(
                state,
                step,
                "complete",
                attempt_id=attempt_id,
                evidence_sha256=verdict_sha,
                idempotency_key=f"{state.attempt_id(step)}:complete",
            )
        else:
            self._repair_or_fail(state, step, verdict, workdir)
        state.elapsed_s = self._prior_elapsed + (time.monotonic() - self._t0)
        state.active_since = None
        self._h._save(state)
        nxt = "done" if (state.is_terminal() or state.next_step() is None) else "continue"
        return {"rs": state.model_dump(mode="json"), "next": nxt}

    # --- helpers -----------------------------------------------------------
    def _request_approval(self, snap, state: RunState) -> None:
        payload: dict = {}
        for intr in snap.interrupts or []:
            payload = getattr(intr, "value", {}) or {}
        step = state.next_step()
        if (
            step is None
            or payload.get("step_id") != step.step_id
            or payload.get("attempt_id") != state.attempt_id(step)
            or payload.get("goal") != step.goal
        ):
            raise CheckpointCorrupt(
                "LangGraph approval interrupt does not match the current attempt"
            )
        HumanApprovalGate(state.run_dir).request(
            step,
            state.attempt_id(step),
            "awaiting approval (langgraph interrupt)",
            artifact_sha256=payload.get("artifact_sha256"),
        )

    def _fail_invalid_approval(
        self,
        state: RunState,
        step,
        message: str,
    ) -> RunResult:
        final = load_state_by_id(self.config.runs_dir, state.run_id)
        try:
            self._h._revert_step(step, Path(final.workdir))
        except Exception:
            logger.exception("revert failed for step %s", step.step_id)
        final.status = "FAILED"
        self._h._save(final)
        return RunResult(final, "FAILED", message)

    def _finalize(self, state: RunState) -> None:
        if state.task.kind == "issue_to_pr":
            self._h._finalize_pr(state)
        elif state.task.kind == "paper_to_experiment":
            self._h._finalize_experiment(state)
        if self.config.use_skill_memory:
            try:
                from ..memory import SkillMemory

                if SkillMemory(Path(self.config.data_dir) / "skills").record(state) is not None:
                    live_context.index_docs(("skill",))  # keep the skill retrievable next run
            except Exception:
                pass


def _route_next(gstate: GraphState) -> str:
    return gstate.get("next", "done")


def _current_artifact_sha(run_dir: Path, step_id: str | None, action: str) -> str | None:
    """SHA-256 of the persisted patch.json for a patch step (per-step file first).

    Non-patch actions regenerate their artifact on resume, so their approvals
    bind by step id alone (None here; see ``ApprovalDecision.binds``).
    """
    if not step_id or action != "edit_code":
        return None
    candidates = [run_dir / "steps" / _safe_seg(step_id) / "patch.json", run_dir / "patch.json"]
    for path in candidates:
        if path.exists():
            try:
                return sha256_bytes(path.read_bytes())
            except OSError:
                return None
    return None


def _load_step_artifacts(run_dir: Path, step) -> tuple[Any, ContextBundle | None]:
    """Reload the artifacts ``prepare`` persisted for this step.

    The verify node grades exactly what was executed (and, for gated steps,
    approved) — a missing artifact fails closed rather than passing vacuously.
    """
    sdir = run_dir / "steps" / _safe_seg(step.step_id)
    bundle: ContextBundle | None = None
    bpath = sdir / "context_bundle.json"
    if bpath.exists():
        try:
            bundle = ContextBundle.model_validate_json(bpath.read_text())
        except Exception:
            bundle = None
    if step.action == "edit_code":
        ppath = sdir / "patch.json"
        if ppath.exists():
            try:
                return Patch.model_validate_json(ppath.read_text()), bundle
            except Exception:
                pass
        return Patch(step_id=step.step_id), bundle
    if step.action == "run_experiment":
        epath = sdir / "experiment.json"
        if epath.exists():
            try:
                return ExperimentResult.model_validate_json(epath.read_text()), bundle
            except Exception:
                pass
        return (
            ExperimentResult(step_id=step.step_id, returncode=1, stdout_tail="artifact missing"),
            bundle,
        )
    if step.action == "repo_integrity":
        ipath = sdir / "repo_integrity.json"
        if ipath.exists():
            try:
                return RepoIntegrityResult.model_validate_json(ipath.read_text()), bundle
            except Exception:
                pass
        return None, bundle
    if step.action == "repo_stage":
        rpath = sdir / "repo_stage.json"
        if rpath.exists():
            try:
                return RepoStageResult.model_validate_json(rpath.read_text()), bundle
            except Exception:
                pass
        return None, bundle
    return bundle, bundle  # gather_context / answer_query: the bundle IS the artifact
