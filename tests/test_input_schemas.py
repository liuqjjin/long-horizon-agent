"""Strict parsing checks for user-supplied task and repository YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lha.repo_adapter import RepoAdapterSpec
from lha.tasks.spec import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
TASK_PATHS = tuple(sorted((ROOT / "data" / "tasks").glob("*.yaml"))) + tuple(
    sorted((ROOT / "data" / "long_tasks").glob("*/task.yaml"))
)
ADAPTER_PATHS = tuple(
    sorted((ROOT / "data" / "long_tasks").glob("*/adapter.yaml"))
)


@pytest.mark.parametrize("path", TASK_PATHS, ids=lambda path: path.stem)
def test_checked_in_task_yaml_matches_strict_schema(path: Path) -> None:
    TaskSpec.from_file(path)


@pytest.mark.parametrize(
    "path",
    ADAPTER_PATHS,
    ids=lambda path: path.parent.name,
)
def test_checked_in_repo_adapter_yaml_matches_strict_schema(path: Path) -> None:
    RepoAdapterSpec.from_file(path)


def test_task_yaml_rejects_misspelled_field(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        "kind: issue_to_pr\n"
        "title: strict input\n"
        "succes:\n"
        "  - pytest passes\n"
    )

    with pytest.raises(ValidationError, match="succes"):
        TaskSpec.from_file(path)


@pytest.mark.parametrize(
    ("yaml_text", "misspelled_field"),
    (
        (
            "schema_version: 1\n"
            "allowd_tools:\n"
            "  - python\n",
            "allowd_tools",
        ),
        (
            "schema_version: 1\n"
            "allowed_tools:\n"
            "  - python\n"
            "setup:\n"
            "  - id: probe\n"
            "    tool: python\n"
            "    timeot_s: 10\n",
            "timeot_s",
        ),
    ),
)
def test_repo_adapter_yaml_rejects_misspelled_fields(
    yaml_text: str,
    misspelled_field: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "adapter.yaml"
    path.write_text(yaml_text)

    with pytest.raises(ValidationError, match=misspelled_field):
        RepoAdapterSpec.from_file(path)
