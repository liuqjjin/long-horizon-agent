"""Supervisor: decompose a task into a verifiable Plan (structured artifact)."""

from __future__ import annotations

from ..artifacts import Plan, Step
from ..config import Config
from ..tasks.spec import TaskSpec


class Supervisor:
    def __init__(self, config: Config):
        self.config = config

    def plan(self, task: TaskSpec) -> Plan:
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
