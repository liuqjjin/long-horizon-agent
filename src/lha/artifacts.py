"""Structured artifacts emitted by the agents.

Each role emits one of these (never free-form chat):
  Supervisor       -> Plan
  Context Engineer -> ContextBundle   (see live_context.models)
  Implementer      -> Patch
  Verifier         -> Verdict          (see verifiers.verdict)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StepKind = Literal["code", "experiment", "context"]
StepAction = Literal["edit_code", "run_experiment", "gather_context", "answer_query"]


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

    def as_repair(self, failures: list[str]) -> "Step":
        return self.model_copy(update={"repair_of": self.step_id, "prior_failures": failures})


class Plan(BaseModel):
    """The Supervisor's artifact."""

    task_id: str
    summary: str
    steps: list[Step] = Field(default_factory=list)
    overall_success: list[str] = Field(default_factory=list)


class Patch(BaseModel):
    """The Implementer's artifact: a code change."""

    step_id: str
    unified_diff: str = ""
    touched_files: list[str] = Field(default_factory=list)
    # explicit new file contents {relpath: text}; preferred over diff when present
    file_contents: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""
    based_on_context: list[str] = Field(default_factory=list)  # provenance locators used

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
    metrics: dict[str, float] = Field(default_factory=dict)  # self-reported
    reference_path: str | None = None  # relative to workdir (e.g. out/reference.npy)
    prediction_path: str | None = None
    repro: dict[str, Any] = Field(default_factory=dict)  # seed, versions, git_commit, ...
    returncode: int = 0
    stdout_tail: str = ""
    based_on_context: list[str] = Field(default_factory=list)


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
