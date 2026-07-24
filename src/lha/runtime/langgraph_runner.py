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
from ..agents import ContextEngineer, Supervisor, VerifierAgent
from ..artifacts import ExperimentResult, Patch
from ..config import Config
from ..harness.approval import HumanApprovalGate
from ..harness.checkpoint import append_ledger, load_state_by_id, save_state
from ..harness.errors import ApprovalRejected, BudgetExceeded, PolicyViolation
from ..harness.loop import Harness, RunResult, _dump, _gen_run_id, _policy_verdict, _safe_seg
from ..harness.manifest import sha256_bytes
from ..harness.state import Phase, RunState, StepRecord
from ..live_context.models import ContextBundle
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
        run_dir = self.config.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        workdir = run_dir / "workdir"
        self._h._prepare_workdir(task, workdir)
        state = RunState.new(task, run_id, str(run_dir), str(workdir))
        save_state(state)
        return self._drive(state)

    def resume(self, run_id: str) -> RunResult:
        state = load_state_by_id(self.config.runs_dir, run_id)
        if state.is_terminal():
            return RunResult(state, state.status, "run already terminal")
        state.status = "RUNNING"
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
            state.plan = Supervisor(self.config, self._h.llm).plan(state.task)
            (run_dir / "plan.json").write_text(state.plan.model_dump_json(indent=2))
            save_state(state)

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
                decision = gate.decision()
                interrupted_step = None
                for intr in snap.interrupts or []:
                    interrupted_step = (getattr(intr, "value", {}) or {}).get("step_id")
                if decision is not None and decision.step_id != interrupted_step:
                    gate.clear()  # a decision for a different step must not resume this one
                    decision = None
                if decision is None:
                    return RunResult(
                        load_state_by_id(self.config.runs_dir, state.run_id),
                        "AWAITING_APPROVAL",
                        "approval pending",
                    )
                # The decision must bind the exact persisted artifact bytes. A
                # mismatch means the artifact changed after review — tamper
                # evidence, so the run fails closed with the change reverted.
                action = next(
                    (
                        s.action
                        for s in (state.plan.steps if state.plan else [])
                        if s.step_id == interrupted_step
                    ),
                    "edit_code",  # unknown step: demand the strictest binding
                )
                current_sha = _current_artifact_sha(run_dir, interrupted_step, action)
                if not decision.binds(interrupted_step or "", current_sha):
                    from types import SimpleNamespace

                    gate.clear()
                    final = load_state_by_id(self.config.runs_dir, state.run_id)
                    step = final.next_step() or SimpleNamespace(step_id=interrupted_step or "?")
                    try:
                        self._h._revert_step(step, workdir)
                    except Exception:
                        logger.exception("revert failed for step %s", interrupted_step)
                    final.status = "FAILED"
                    save_state(final)
                    return RunResult(
                        final,
                        "FAILED",
                        "approved artifact does not match the persisted patch "
                        "(hash mismatch) — refusing to execute",
                    )
                gate.clear()
                resume_input = Command(resume={"approved": decision.approved})
            else:
                resume_input = {"rs": state.model_dump(mode="json")}

            try:
                graph.invoke(resume_input, gcfg)
            except BudgetExceeded as e:
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.status = "PAUSED"
                save_state(final)
                return RunResult(final, "PAUSED", str(e))
            except ApprovalRejected as e:
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.status = "FAILED"
                save_state(final)
                return RunResult(final, "FAILED", str(e))
            except Exception as e:
                # Same fail-closed contract as the default loop: a mid-node fault
                # must not leave the run wedged at RUNNING with an unverified
                # patch in the sandbox.
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                step = final.next_step()
                if step is not None and step.step_id not in final.completed_steps:
                    try:
                        self._h._revert_step(step, workdir)
                    except Exception:
                        logger.exception("revert failed for step %s", step.step_id)
                    final.fail_current(step)
                else:
                    final.status = "FAILED"
                save_state(final)
                return RunResult(final, "FAILED", f"{type(e).__name__}: {e}")

            # paused on a fresh interrupt?
            snap = graph.get_state(gcfg)
            if snap.next and (snap.interrupts or []):
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.status = "AWAITING_APPROVAL"
                save_state(final)
                self._request_approval(snap, final)
                return RunResult(final, "AWAITING_APPROVAL", "awaiting approval")
            if snap.next:  # stopped mid-graph without an interrupt: wedged, fail closed
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.status = "FAILED"
                save_state(final)
                return RunResult(final, "FAILED", "graph stopped mid-run without an interrupt")
        finally:
            conn.close()

        final = load_state_by_id(self.config.runs_dir, state.run_id)
        if final.status == "PAUSED":  # budget exhausted mid-graph
            return RunResult(final, "PAUSED", "budget exceeded")
        if not final.is_terminal():
            final.status = "DONE"
        if final.status == "DONE":
            self._finalize(final)
        save_state(final)
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
        append_ledger(
            state, StepRecord(seq=state.next_seq(), step_id=step.step_id, phase=phase, **kw)
        )

    def _repair_or_fail(self, state: RunState, step, verdict, workdir: Path) -> None:
        if state.repairs_for(step) < self.config.max_repairs:
            state.record_repair(step)
            assert state.plan is not None  # plan set before stepping
            state.plan.steps[state.cursor] = step.as_repair(verdict.failures)
            self._ledger(state, step, "repair", notes="; ".join(verdict.failures)[:300])
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
        step = state.next_step()
        if step is None or state.is_terminal():
            return {"rs": state.model_dump(mode="json"), "next": "done"}

        # Bound the run: max_steps caps prepare executions and deadline_s caps
        # wall-clock, both persisted so they survive resume.
        elapsed = self._prior_elapsed + (time.monotonic() - self._t0)
        deadline = self.config.deadline_s
        if state.steps_used >= self.config.max_steps or (
            deadline is not None and elapsed > deadline
        ):
            state.status = "PAUSED"
            state.elapsed_s = elapsed
            save_state(state)
            return {"rs": state.model_dump(mode="json"), "next": "done"}
        state.steps_used += 1
        state.elapsed_s = elapsed

        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        bundle = ContextEngineer(self.config).gather(step, workdir=workdir)
        _dump(run_dir, step.step_id, "context_bundle.json", bundle.model_dump_json(indent=2))
        self._ledger(state, step, "context", artifact_ref="context_bundle.json")
        try:
            _artifact, ref = self._h._execute(state, step, bundle)
        except PolicyViolation as e:
            # Same oracle protection as the default loop: the patch was refused
            # before it reached the sandbox; feed the reason to the repair loop.
            verdict = _policy_verdict(step.step_id, e)
            _dump(run_dir, step.step_id, "verify.json", verdict.model_dump_json(indent=2))
            self._ledger(state, step, "verify", verdict_ref="verify.json", notes=str(e)[:300])
            self._repair_or_fail(state, step, verdict, workdir)
            save_state(state)
            nxt = "done" if state.is_terminal() else "continue"
            return {"rs": state.model_dump(mode="json"), "next": nxt}
        self._ledger(state, step, "execute", artifact_ref=ref)
        save_state(state)

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
        decision = interrupt({"step_id": step.step_id, "goal": step.goal, "artifact_sha256": sha})
        if not (isinstance(decision, dict) and decision.get("approved")):
            self._ledger(state, step, "approval", notes="rejected")
            self._h._revert_step(step, Path(state.workdir))  # rejected change must not survive
            state.fail_current(step)
            self._ledger(state, step, "fail")
            save_state(state)
            return {"rs": state.model_dump(mode="json"), "next": "done"}
        self._ledger(state, step, "approval", notes="approved")
        save_state(state)
        return {"rs": state.model_dump(mode="json"), "next": "verify"}

    def _verify_node(self, gstate: GraphState) -> GraphState:
        """Grade the artifact ``prepare`` persisted — never a regenerated one."""
        state = RunState.model_validate(gstate["rs"])
        step = state.next_step()
        if step is None or state.is_terminal():
            return {"rs": state.model_dump(mode="json"), "next": "done"}

        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)
        artifact, bundle = _load_step_artifacts(run_dir, step)

        verdict = VerifierAgent(parallel=self.config.parallel_verify).verify(
            step,
            artifact,
            VerifyContext(workdir=workdir, step=step, bundle=bundle, exec=self._h.exec),
        )
        _dump(run_dir, step.step_id, "verify.json", verdict.model_dump_json(indent=2))
        self._ledger(state, step, "verify", verdict_ref="verify.json")

        if verdict.passed:
            state.complete_step(step)
            self._ledger(state, step, "complete")
        else:
            self._repair_or_fail(state, step, verdict, workdir)
        save_state(state)
        nxt = "done" if (state.is_terminal() or state.next_step() is None) else "continue"
        return {"rs": state.model_dump(mode="json"), "next": nxt}

    # --- helpers -----------------------------------------------------------
    def _request_approval(self, snap, state: RunState) -> None:
        from types import SimpleNamespace

        payload: dict = {}
        for intr in snap.interrupts or []:
            payload = getattr(intr, "value", {}) or {}
        step = SimpleNamespace(step_id=payload.get("step_id", "?"), goal=payload.get("goal", ""))
        HumanApprovalGate(state.run_dir).request(
            step,
            "awaiting approval (langgraph interrupt)",
            artifact_sha256=payload.get("artifact_sha256"),
        )

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
    return bundle, bundle  # gather_context / answer_query: the bundle IS the artifact
