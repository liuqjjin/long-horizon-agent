"""Supervisor: decompose a task into a verifiable Plan (structured artifact)."""

from __future__ import annotations

import re

from ..artifacts import Plan, Step
from ..config import Config
from ..tasks.spec import TaskSpec
from ..verifiers.registry import get as _get_verifier

_SAFE_STEP_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


class Supervisor:
    def __init__(self, config: Config, llm=None):
        self.config = config
        self.llm = llm

    def plan(self, task: TaskSpec) -> Plan:
        """A verifiable Plan: the deterministic template, optionally replaced by an
        LLM-generated plan when dynamic_planning is on and the candidate is valid."""
        template = self._template(task)
        if self.config.dynamic_planning and self.llm is not None:
            try:
                candidate = self.llm.plan(task, template)
            except Exception:
                candidate = None
            if candidate is not None and self._valid(candidate):
                return candidate
        return template

    @staticmethod
    def _valid(plan: Plan) -> bool:
        """A usable LLM plan: non-empty; every step has a path-safe step_id and declares
        only registered verifiers (so it can't inject a path or an unrunnable check)."""
        if not plan.steps:
            return False
        return all(
            _SAFE_STEP_ID.fullmatch(s.step_id) is not None
            and s.step_id not in (".", "..")
            and bool(s.verifiers)
            and all(_get_verifier(v) is not None for v in s.verifiers)
            for s in plan.steps
        )

    def _template(self, task: TaskSpec) -> Plan:
        cq = task.inputs.get("context_query") or task.title
        if task.kind == "issue_to_pr":
            steps = [
                Step(
                    step_id="s1-context",
                    kind="code",
                    action="gather_context",
                    goal=f"Gather code context for: {task.title}",
                    context_query=cq,
                    verifiers=["freshness", "citation"],
                    success_criteria=["context retrieved with provenance"],
                ),
                Step(
                    step_id="s2-fix",
                    kind="code",
                    action="edit_code",
                    goal=task.description or task.title,
                    context_query=cq,
                    verifiers=["pytest", "ruff"],
                    requires_approval=bool(task.inputs.get("require_approval", False)),
                    success_criteria=task.success or ["pytest passes", "ruff clean"],
                ),
            ]
            return Plan(
                task_id=task.title,
                summary=f"Fix issue: {task.title}",
                steps=steps,
                overall_success=task.success or ["pytest passes", "ruff clean"],
            )

        if task.kind == "freshness":
            steps = [
                Step(
                    step_id="s1-answer",
                    kind="context",
                    action="answer_query",
                    goal=task.title,
                    context_query=cq,
                    verifiers=["freshness", "citation"],
                    success_criteria=["answer cites fresh sources"],
                )
            ]
            return Plan(
                task_id=task.title,
                summary=f"Answer with fresh, cited context: {task.title}",
                steps=steps,
                overall_success=["answer cites fresh sources"],
            )

        if task.kind == "paper_to_experiment":
            params: dict = dict(task.inputs.get("thresholds", {}))
            params["experiment_script"] = task.inputs.get("experiment_script", "experiment.py")
            params["experiment_args"] = task.inputs.get("experiment_args", [])
            params["out_dir"] = task.inputs.get("out_dir", "out")
            steps = [
                Step(
                    step_id="s1-context",
                    kind="experiment",
                    action="gather_context",
                    goal=f"Gather paper/experiment context for: {task.title}",
                    context_query=cq,
                    verifiers=["freshness", "citation"],
                    success_criteria=["context retrieved with provenance"],
                ),
                Step(
                    step_id="s2-run",
                    kind="experiment",
                    action="run_experiment",
                    goal=task.description or task.title,
                    context_query=cq,
                    verifiers=["psnr", "ssim", "reproducibility"],
                    params=params,
                    success_criteria=task.success or ["metrics meet thresholds", "reproducible"],
                ),
            ]
            return Plan(
                task_id=task.title,
                summary=f"Run + verify experiment: {task.title}",
                steps=steps,
                overall_success=task.success or ["metrics meet thresholds", "reproducible"],
            )

        raise ValueError(f"unknown task kind: {task.kind}")
