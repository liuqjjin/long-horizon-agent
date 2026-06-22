"""Opt-in LangGraph durable runtime.

Runs the same plan/agents/verifiers as the default loop, but drives them through a
``StateGraph`` checkpointed by ``SqliteSaver`` (durable, resumable by thread_id)
with ``interrupt()`` for the human-approval gate. It reuses the Harness's
execute/finalize helpers, so there is one implementation of the actual work.

Enable with ``lha run --runtime langgraph`` (and ``lha resume --runtime langgraph``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from .. import live_context
from ..agents import ContextEngineer, Supervisor, VerifierAgent
from ..config import Config
from ..harness.approval import HumanApprovalGate
from ..harness.checkpoint import append_ledger, load_state_by_id, save_state
from ..harness.loop import Harness, RunResult, _gen_run_id
from ..harness.state import Phase, RunState, StepRecord
from ..verifiers import VerifyContext


class GraphState(TypedDict):
    rs: dict  # RunState.model_dump(mode="json")


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
        code_root = state.task.target_repo or str(workdir)
        live_context.configure(code_root=code_root, config=self.config)
        try:
            live_context.index_code(code_root)
        except Exception:
            pass
        if state.task.kind == "paper_to_experiment":
            try:
                live_context.index_docs()
            except Exception:
                pass

        if state.plan is None:
            state.plan = Supervisor(self.config).plan(state.task)
            (run_dir / "plan.json").write_text(state.plan.model_dump_json(indent=2))
            save_state(state)

        conn = sqlite3.connect(str(run_dir / "graph.sqlite"), check_same_thread=False)
        try:
            graph = self._compile(conn)
            gcfg: Any = {"configurable": {"thread_id": state.run_id}}  # LangGraph RunnableConfig
            snap = graph.get_state(gcfg)

            if snap.next:  # graph is paused at an interrupt (awaiting approval)
                decision = HumanApprovalGate(run_dir).decision()
                if decision is None:
                    return RunResult(
                        load_state_by_id(self.config.runs_dir, state.run_id),
                        "AWAITING_APPROVAL",
                        "approval pending",
                    )
                HumanApprovalGate(run_dir).clear()
                graph.invoke(Command(resume={"approved": decision.approved}), gcfg)
            else:
                graph.invoke({"rs": state.model_dump(mode="json")}, gcfg)

            # paused on a fresh interrupt?
            snap = graph.get_state(gcfg)
            if snap.next:
                final = load_state_by_id(self.config.runs_dir, state.run_id)
                final.status = "AWAITING_APPROVAL"
                save_state(final)
                self._request_approval(snap, final)
                return RunResult(final, "AWAITING_APPROVAL", "awaiting approval")
        finally:
            conn.close()

        final = load_state_by_id(self.config.runs_dir, state.run_id)
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
        builder.add_node("step", self._step_node)  # type: ignore[arg-type]
        builder.add_edge(START, "step")
        builder.add_conditional_edges("step", _route, {"continue": "step", "done": END})
        return builder.compile(checkpointer=saver)

    def _step_node(self, gstate: GraphState) -> GraphState:
        from langgraph.types import interrupt

        state = RunState.model_validate(gstate["rs"])
        step = state.next_step()
        if step is None or state.is_terminal():
            return {"rs": state.model_dump(mode="json")}

        run_dir = Path(state.run_dir)
        workdir = Path(state.workdir)

        def ledger(phase: Phase, **kw):
            append_ledger(
                state, StepRecord(seq=state.next_seq(), step_id=step.step_id, phase=phase, **kw)
            )

        bundle = ContextEngineer(self.config).gather(step)
        (run_dir / "context_bundle.json").write_text(bundle.model_dump_json(indent=2))
        ledger("context", artifact_ref="context_bundle.json")
        artifact, ref = self._h._execute(state, step, bundle)
        ledger("execute", artifact_ref=ref)

        if step.requires_approval and not self.auto_approve:
            decision = interrupt({"step_id": step.step_id, "goal": step.goal})
            if not (isinstance(decision, dict) and decision.get("approved")):
                ledger("approval", notes="rejected")
                self._h._revert_step(step, workdir)  # rejected change must not survive
                state.fail_current(step)
                ledger("fail")
                save_state(state)
                return {"rs": state.model_dump(mode="json")}
            ledger("approval", notes="approved")

        verdict = VerifierAgent(parallel=self.config.parallel_verify).verify(
            step, artifact, VerifyContext(workdir=workdir, step=step, bundle=bundle)
        )
        (run_dir / "verify.json").write_text(verdict.model_dump_json(indent=2))
        ledger("verify", verdict_ref="verify.json")

        if verdict.passed:
            state.complete_step(step)
            ledger("complete")
        elif state.repairs_for(step) < self.config.max_repairs:
            state.record_repair(step)
            assert state.plan is not None  # plan set before stepping
            state.plan.steps[state.cursor] = step.as_repair(verdict.failures)
            ledger("repair", notes="; ".join(verdict.failures)[:300])
        else:
            # Match the default loop: a step that exhausts its repairs is reverted
            # so an unverified change never survives in the sandbox (loop.py:_revert_step).
            self._h._revert_step(step, workdir)
            state.fail_current(step)
            ledger("fail")
        save_state(state)
        return {"rs": state.model_dump(mode="json")}

    # --- helpers -----------------------------------------------------------
    def _request_approval(self, snap, state: RunState) -> None:
        from types import SimpleNamespace

        payload: dict = {}
        for intr in snap.interrupts or []:
            payload = getattr(intr, "value", {}) or {}
        step = SimpleNamespace(step_id=payload.get("step_id", "?"), goal=payload.get("goal", ""))
        HumanApprovalGate(state.run_dir).request(step, "awaiting approval (langgraph interrupt)")

    def _finalize(self, state: RunState) -> None:
        if state.task.kind == "issue_to_pr":
            self._h._finalize_pr(state)
        elif state.task.kind == "paper_to_experiment":
            self._h._finalize_experiment(state)
        if self.config.use_skill_memory:
            try:
                from ..memory import SkillMemory

                SkillMemory(Path(self.config.data_dir) / "skills").record(state)
            except Exception:
                pass


def _route(gstate: GraphState) -> str:
    state = RunState.model_validate(gstate["rs"])
    if state.is_terminal() or state.next_step() is None:
        return "done"
    return "continue"
