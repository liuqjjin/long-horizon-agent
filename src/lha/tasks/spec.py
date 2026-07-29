"""Task specifications — the 'what to do' fed to the harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..artifacts import _reject_non_finite

TaskKind = Literal["issue_to_pr", "paper_to_experiment", "freshness"]


class TaskSpec(BaseModel):
    """A unit of work for the harness."""

    model_config = ConfigDict(extra="forbid")

    kind: TaskKind
    title: str
    description: str = ""
    # Repo the task operates on (relative to cwd). Copied into the run sandbox.
    target_repo: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    success: list[str] = Field(default_factory=list)
    # Whether the plan's context-gathering must produce verifiable context
    # ("required", the fail-closed default) or the task can proceed without any
    # ("optional" — an explicit declaration, never an implicit fallback).
    context_requirement: Literal["required", "optional"] = "required"
    # Files the task explicitly authorizes the agent to modify even though the
    # oracle-protection policy would refuse them (e.g. a task whose goal IS to
    # fix a test). Relative paths, exact matches against patch entries.
    allowed_protected_files: list[str] = Field(default_factory=list)

    @field_validator("inputs")
    @classmethod
    def _finite_inputs(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_non_finite(value, "inputs")
        return value

    @classmethod
    def from_file(cls, path: str | Path) -> "TaskSpec":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
