"""Supervisor: decompose a task into a verifiable Plan (structured artifact)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import Plan, Step
from ..config import Config
from ..repo_adapter import (
    RepoAdapterSpec,
    RepoReferenceManifest,
    RepoStage,
    resolve_repo_task_assets,
)
from ..step_ids import validate_plan_step_ids
from ..tasks.spec import TaskSpec
from ..verifiers.registry import get as _get_verifier

_REQUIRED_EXPERIMENT_VERIFIERS = frozenset({"psnr", "ssim", "reproducibility"})
_ACTION_FAMILIES = {
    "gather_context": {"context"},
    "answer_query": {"context"},
    "edit_code": {"code"},
    "run_experiment": {"experiment"},
    "repo_integrity": {"code"},
    "repo_stage": {"code"},
}


class Supervisor:
    def __init__(self, config: Config, llm=None):
        self.config = config
        self.llm = llm

    def plan(self, task: TaskSpec) -> Plan:
        """A verifiable Plan: the deterministic template, optionally replaced by an
        LLM-generated plan when dynamic_planning is on and the candidate is valid."""
        template = self._template(task)
        # Long-task plans encode a fixed measurement protocol. Letting a model
        # omit the baseline or final gates would change the experiment itself.
        if (
            not self._is_long_task(task)
            and self.config.dynamic_planning
            and self.llm is not None
        ):
            # Backends return ``None`` for a syntactically invalid candidate.
            # Transport, protocol, budget, and durable-evidence exceptions must
            # propagate; substituting a template would hide a failed paid call.
            candidate = self.llm.plan(task, template)
            if candidate is not None and self._valid(
                candidate, task=task, template=template
            ):
                return candidate
        return template

    @staticmethod
    def _valid(plan: Plan, *, task: TaskSpec, template: Plan) -> bool:
        """A usable LLM plan: non-empty; every step has a path-safe step_id and declares
        only registered verifiers (so it can't inject a path or an unrunnable check)."""
        if not plan.steps:
            return False
        try:
            validate_plan_step_ids(step.step_id for step in plan.steps)
        except ValueError:
            return False
        for step in plan.steps:
            verifiers = [_get_verifier(name) for name in step.verifiers]
            if (
                not verifiers
                or any(verifier is None for verifier in verifiers)
            ):
                return False
            allowed = _ACTION_FAMILIES[step.action]
            if any(verifier is not None and verifier.family not in allowed for verifier in verifiers):
                return False
        if task.kind == "paper_to_experiment":
            return _valid_experiment_protocol(plan, task, template)
        return _valid_template_protocol(plan, template)

    def _template(self, task: TaskSpec) -> Plan:
        cq = task.inputs.get("context_query") or task.title
        creq = task.context_requirement
        if task.kind == "issue_to_pr":
            if self._has_partial_long_task_config(task):
                raise ValueError(
                    "long tasks require both inputs.repo_adapter and "
                    "inputs.reference_manifest"
                )
            if self._is_long_task(task):
                return self._long_task_template(task, cq)
            steps = [
                Step(
                    step_id="s1-context",
                    kind="code",
                    action="gather_context",
                    goal=f"Gather code context for: {task.title}",
                    context_query=cq,
                    verifiers=["freshness", "citation"],
                    success_criteria=["context retrieved with provenance"],
                    context_requirement=creq,
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
                    context_requirement=creq,
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
                    context_requirement=creq,
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
                    context_requirement=creq,
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
                    context_requirement=creq,
                ),
            ]
            return Plan(
                task_id=task.title,
                summary=f"Run + verify experiment: {task.title}",
                steps=steps,
                overall_success=task.success or ["metrics meet thresholds", "reproducible"],
            )

        raise ValueError(f"unknown task kind: {task.kind}")

    @staticmethod
    def _is_long_task(task: TaskSpec) -> bool:
        return bool(task.inputs.get("repo_adapter")) and bool(
            task.inputs.get("reference_manifest")
        )

    @staticmethod
    def _has_partial_long_task_config(task: TaskSpec) -> bool:
        return bool(task.inputs.get("repo_adapter")) != bool(
            task.inputs.get("reference_manifest")
        )

    @staticmethod
    def _long_task_template(task: TaskSpec, context_query: str) -> Plan:
        if task.target_repo is None:
            raise ValueError("long tasks require target_repo")
        assets = resolve_repo_task_assets(
            target_repo=task.target_repo,
            adapter_path=str(task.inputs["repo_adapter"]),
            manifest_path=str(task.inputs["reference_manifest"]),
        )
        spec = RepoAdapterSpec.from_file(assets.adapter_path)
        manifest = RepoReferenceManifest.from_file(assets.manifest_path)
        if manifest.task_id != assets.task_root.name:
            raise ValueError(
                "reference manifest task_id does not match the long-task directory"
            )
        required_stages = ("setup", "baseline", "reproduce", "targeted", "full", "lint", "build")
        missing = [stage for stage in required_stages if not spec.commands_for(stage)]
        if missing:
            raise ValueError(f"long-task adapter has unconfigured required stages: {missing}")

        spec_payload = spec.model_dump(mode="json")
        integrity_params = {
            "task_path": str(assets.task_path),
            "repo_adapter_path": str(assets.adapter_path),
            "reference_manifest_path": str(assets.manifest_path),
            "reference_patch_path": str(assets.reference_patch_path),
        }

        def stage(step_id: str, stage_name: RepoStage, goal: str, **params) -> Step:
            return Step(
                step_id=step_id,
                kind="code",
                action="repo_stage",
                goal=goal,
                verifiers=["repo-stage"],
                params={
                    "repo_adapter_spec": spec_payload,
                    "repo_stage": stage_name,
                    **params,
                },
                context_requirement="optional",
            )

        steps = [
            Step(
                step_id="s01-integrity",
                kind="code",
                action="repo_integrity",
                goal="Validate the fixed repository, oracle, and reference metadata",
                verifiers=["repo-integrity"],
                params=integrity_params,
                context_requirement="optional",
            ),
            stage("s02-setup", "setup", "Check the repository execution environment"),
            stage(
                "s03-baseline",
                "baseline",
                "Confirm the complete baseline suite reproduces the known failure",
                expected_failure=True,
                expected_returncode=manifest.expected_baseline_returncode,
            ),
            stage(
                "s04-reproduce",
                "reproduce",
                "Reproduce the task's focused failing case",
                expected_failure=True,
            ),
            Step(
                step_id="s05-context",
                kind="code",
                action="gather_context",
                goal=f"Gather code context for: {task.title}",
                context_query=context_query,
                verifiers=["freshness", "citation"],
                success_criteria=["context retrieved with provenance"],
                context_requirement=task.context_requirement,
            ),
            Step(
                step_id="s06-edit",
                kind="code",
                action="edit_code",
                goal=task.description or task.title,
                context_query=context_query,
                verifiers=["repo-targeted"],
                params={
                    "repo_adapter_spec": spec_payload,
                    "repo_stage": "targeted",
                },
                requires_approval=True,
                success_criteria=["the focused repository gate passes"],
                context_requirement=task.context_requirement,
            ),
            stage(
                "s07-targeted",
                "targeted",
                "Repeat the focused gate on the approved patch",
                rollback_step_id="s06-edit",
            ),
            stage(
                "s08-full",
                "full",
                "Run the complete repository test suite",
                expected_test_count=manifest.expected_patched_test_count,
                rollback_step_id="s06-edit",
            ),
            stage(
                "s09-lint",
                "lint",
                "Run the repository lint gate",
                rollback_step_id="s06-edit",
            ),
            stage(
                "s10-build",
                "build",
                "Build or compile the patched repository",
                rollback_step_id="s06-edit",
            ),
        ]
        return Plan(
            task_id=task.title,
            summary=f"Run the fixed ten-stage repository protocol: {task.title}",
            steps=steps,
            overall_success=task.success
            or ["focused tests, full tests, lint, and build all pass"],
        )


def _valid_experiment_protocol(
    candidate: Plan, task: TaskSpec, template: Plan
) -> bool:
    """Keep a model-generated experiment plan on the task's fixed protocol."""
    runs = [step for step in candidate.steps if step.action == "run_experiment"]
    template_runs = [step for step in template.steps if step.action == "run_experiment"]
    if not runs or len(runs) != len(template_runs):
        return False
    if any(
        step.action not in {"gather_context", "run_experiment"}
        for step in candidate.steps
    ):
        return False
    for step, expected in zip(runs, template_runs, strict=True):
        if not _REQUIRED_EXPERIMENT_VERIFIERS.issubset(step.verifiers):
            return False
        # Commands, thresholds, scoring parameters, and the output location are
        # one measurement protocol. A planner may reorder safe orchestration,
        # but it may not silently substitute or weaken this protocol.
        if step.params != expected.params:
            return False
        if not _runnable_experiment_params(step.params, task):
            return False
    return True


def _valid_template_protocol(candidate: Plan, template: Plan) -> bool:
    """A planner may refine wording, but cannot remove a required gate."""
    if len(candidate.steps) != len(template.steps):
        return False
    for step, expected in zip(candidate.steps, template.steps, strict=True):
        if step.action != expected.action or step.kind != expected.kind:
            return False
        if not set(expected.verifiers).issubset(step.verifiers):
            return False
        if expected.requires_approval and not step.requires_approval:
            return False
        if (
            expected.context_requirement == "required"
            and step.context_requirement != "required"
        ):
            return False
        if step.params != expected.params:
            return False
    return True


def _runnable_experiment_params(params: dict[str, Any], task: TaskSpec) -> bool:
    out_dir = params.get("out_dir")
    if not isinstance(out_dir, str):
        return False
    rel_out = Path(out_dir)
    if not rel_out.parts or rel_out.is_absolute() or ".." in rel_out.parts:
        return False

    script = params.get("experiment_script")
    args = params.get("experiment_args")
    if not isinstance(script, str) or not script.strip() or not isinstance(args, list):
        return False
    script_rel = Path(script)
    if script_rel.is_absolute() or ".." in script_rel.parts or task.target_repo is None:
        return False
    repo = Path(task.target_repo).resolve()
    script_path = (repo / script_rel).resolve(strict=False)
    try:
        script_path.relative_to(repo)
    except ValueError:
        return False
    return script_path.is_file() and not script_path.is_symlink()
