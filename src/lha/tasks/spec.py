"""Task specifications — the 'what to do' fed to the harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

TaskKind = Literal["issue_to_pr", "paper_to_experiment", "freshness"]


class TaskSpec(BaseModel):
    """A unit of work for the harness."""

    kind: TaskKind
    title: str
    description: str = ""
    # Repo the task operates on (relative to cwd). Copied into the run sandbox.
    target_repo: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    success: list[str] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "TaskSpec":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
