"""Structured artifacts emitted by the agents.

Each role emits one of these (never free-form chat):
  Supervisor       -> Plan
  Context Engineer -> ContextBundle   (see live_context.models)
  Implementer      -> Patch
  Verifier         -> Verdict          (see verifiers.verdict)
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, FiniteFloat, field_validator, model_validator

from .step_ids import validate_plan_step_ids, validate_step_id

StepKind = Literal["code", "experiment", "context"]
StepAction = Literal[
    "edit_code",
    "run_experiment",
    "gather_context",
    "answer_query",
    "repo_integrity",
    "repo_stage",
]


class Step(BaseModel):
    """One node of a Plan."""

    step_id: str
    kind: StepKind
    action: StepAction
    goal: str
    context_query: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    verifiers: list[str] = Field(default_factory=list)  # verifier names to run
    params: dict[str, Any] = Field(default_factory=dict)  # verifier/executor params
    requires_approval: bool = False
    # "required": missing/unavailable context fails the context checks (default —
    # fail closed). "optional": a task that genuinely needs no retrieval declares
    # it, instead of passing on an empty result.
    context_requirement: Literal["required", "optional"] = "required"
    # populated when a step is re-issued as a repair
    repair_of: str | None = None
    prior_failures: list[str] = Field(default_factory=list)

    @field_validator("step_id")
    @classmethod
    def _canonical_step_id(cls, value: str) -> str:
        return validate_step_id(value)

    @field_validator("params")
    @classmethod
    def _finite_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_non_finite(value, "params")
        return value

    def as_repair(self, failures: list[str]) -> "Step":
        return self.model_copy(update={"repair_of": self.step_id, "prior_failures": failures})


class Plan(BaseModel):
    """The Supervisor's artifact."""

    task_id: str
    summary: str
    steps: list[Step] = Field(default_factory=list)
    overall_success: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_nonempty_steps(self) -> "Plan":
        if not self.steps:
            raise ValueError("plan must contain at least one step")
        validate_plan_step_ids(step.step_id for step in self.steps)
        return self


class Patch(BaseModel):
    """The Implementer's artifact: a code change."""

    step_id: str
    unified_diff: str = ""
    touched_files: list[str] = Field(default_factory=list)
    # explicit new file contents {relpath: text}; preferred over diff when present
    file_contents: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""
    based_on_context: list[str] = Field(default_factory=list)  # provenance locators used

    @model_validator(mode="after")
    def _one_executable_payload(self) -> "Patch":
        if self.unified_diff.strip() and self.file_contents:
            raise ValueError(
                "patch must use either unified_diff or file_contents, not both"
            )
        return self

    def is_empty(self) -> bool:
        return not self.unified_diff and not self.file_contents


class PRSummary(BaseModel):
    """Final human-facing summary for an issue_to_pr run."""

    title: str
    task_id: str
    rationale: str
    files_changed: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)  # human lines, e.g. "pytest: 3 passed"
    provenance: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines += [self.rationale, ""]
        if self.files_changed:
            lines += ["## Files changed", ""]
            lines += [f"- `{f}`" for f in self.files_changed]
            lines += [""]
        if self.checks:
            lines += ["## Verification", ""]
            lines += [f"- {c}" for c in self.checks]
            lines += [""]
        if self.provenance:
            lines += ["## Provenance", ""]
            lines += [f"- {p}" for p in self.provenance]
            lines += [""]
        lines += [f"_task: {self.task_id}_"]
        return "\n".join(lines)


class ExperimentResult(BaseModel):
    """The Experimenter's artifact: the outcome of running an experiment."""

    step_id: str
    out_dir: str = "out"  # relative to the run sandbox (workdir)
    command: list[str] = Field(default_factory=list)  # re-runnable
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)  # self-reported
    reference_path: str | None = None  # relative to workdir (e.g. out/reference.npy)
    prediction_path: str | None = None
    repro: dict[str, Any] = Field(default_factory=dict)  # seed, versions, git_commit, ...
    returncode: int = 0
    stdout_tail: str = ""
    output_truncated: bool = False
    cleanup_unconfirmed: bool = False
    cleanup_detail: str = ""
    based_on_context: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cleanup_status_matches_returncode(self) -> "ExperimentResult":
        if self.cleanup_unconfirmed and self.returncode != 126:
            raise ValueError(
                "cleanup_unconfirmed requires the process cleanup return code"
            )
        return self


class ExperimentSummary(BaseModel):
    """Final human-facing summary for a paper_to_experiment run."""

    title: str
    task_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    checks: list[str] = Field(default_factory=list)
    reproducible: bool = False
    provenance: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.metrics:
            lines += ["## Metrics (verifier-recomputed)", ""]
            lines += [f"- **{k}**: {v}" for k, v in self.metrics.items()]
            lines += [""]
        lines += [f"Reproducible: {'yes' if self.reproducible else 'no'}", ""]
        if self.checks:
            lines += ["## Verification", ""]
            lines += [f"- {c}" for c in self.checks]
            lines += [""]
        if self.provenance:
            lines += ["## Provenance", ""]
            lines += [f"- {p}" for p in self.provenance]
            lines += [""]
        lines += [f"_task: {self.task_id}_"]
        return "\n".join(lines)


def _reject_non_finite(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must not contain NaN or infinity")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")
