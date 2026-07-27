"""Executable evidence for the pre-fixed, multi-file long-task corpus."""

from __future__ import annotations

import shutil
import stat
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from lha.repo_adapter import (
    RepoAdapter,
    RepoAdapterSpec,
    RepoCommand,
    RepoReferenceManifest,
    RepoStageRequest,
    RepoStageResult,
    inspect_repo_integrity,
    repository_tree_sha256,
)
from lha.sandbox import ExecutionBackend, TrustedLocalBackend
from lha.tasks.spec import TaskSpec
from lha.tools.shell import ProcResult

ROOT = Path(__file__).resolve().parents[1]
LONG_TASKS = ROOT / "data" / "long_tasks"
TASK_IDS = (
    "config_parser",
    "sqlite_migration",
    "concurrency_failure",
    "cli_contract",
    "experiment_repro",
)


def _patch_touched_files(patch: Path) -> tuple[str, ...]:
    return tuple(
        line.removeprefix("+++ b/")
        for line in patch.read_text().splitlines()
        if line.startswith("+++ b/")
    )


def _assert_stage_passed(result: RepoStageResult) -> None:
    details = "\n".join(
        f"{command.command_id}: rc={command.returncode}\n"
        f"stdout:\n{command.stdout}\nstderr:\n{command.stderr}"
        for command in result.commands
    )
    assert result.passed, details


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_long_task_reference_patch_passes_every_declared_gate(task_id: str, tmp_path: Path):
    task_dir = LONG_TASKS / task_id
    repo = task_dir / "repo"
    patch = task_dir / "reference.patch"
    manifest = RepoReferenceManifest.from_file(task_dir / "reference_manifest.json")
    task = TaskSpec.from_file(task_dir / "task.yaml")
    spec = RepoAdapterSpec.from_file(task_dir / "adapter.yaml")

    assert manifest.task_id == task_id
    assert manifest.repo_sha256 == repository_tree_sha256(repo)
    assert manifest.task_sha256 == sha256((task_dir / "task.yaml").read_bytes()).hexdigest()
    assert manifest.adapter_sha256 == sha256(
        (task_dir / "adapter.yaml").read_bytes()
    ).hexdigest()
    assert manifest.reference_patch_sha256 == sha256(patch.read_bytes()).hexdigest()
    assert manifest.reference_touched_files == _patch_touched_files(patch)
    assert len(manifest.reference_touched_files) >= 2
    assert all(not path.startswith("tests/") for path in manifest.reference_touched_files)
    assert all((repo / path).is_file() for path in manifest.oracle_files)
    assert all(path.startswith("tests/") for path in manifest.oracle_files)

    source_modules = [
        path
        for path in (repo / "src").rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    ]
    assert len(source_modules) >= 2
    assert task.target_repo == f"data/long_tasks/{task_id}/repo"
    assert task.context_requirement == "optional"
    assert task.inputs["repo_adapter"] == f"data/long_tasks/{task_id}/adapter.yaml"
    assert task.inputs["reference_manifest"] == f"data/long_tasks/{task_id}/reference_manifest.json"

    worktree = tmp_path / task_id
    shutil.copytree(
        repo,
        worktree,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )
    backend = TrustedLocalBackend()
    adapter = RepoAdapter(worktree, spec, backend)

    _assert_stage_passed(adapter.run_stage(RepoStageRequest(stage="setup")))
    baseline = adapter.run_stage(RepoStageRequest(stage="baseline"))
    _assert_stage_passed(baseline)
    assert baseline.commands[-1].returncode == manifest.expected_baseline_returncode
    _assert_stage_passed(adapter.run_stage(RepoStageRequest(stage="reproduce")))

    git = backend.tool("git")
    patch_check = backend.run([git, "apply", "--check", str(patch)], cwd=worktree)
    assert patch_check.returncode == 0, patch_check.stderr
    patch_apply = backend.run([git, "apply", str(patch)], cwd=worktree)
    assert patch_apply.returncode == 0, patch_apply.stderr

    _assert_stage_passed(adapter.run_stage(RepoStageRequest(stage="targeted")))
    full = adapter.run_stage(RepoStageRequest(stage="full"))
    _assert_stage_passed(full)
    assert f"{manifest.expected_patched_test_count} passed" in full.commands[-1].stdout
    _assert_stage_passed(adapter.run_stage(RepoStageRequest(stage="lint")))
    _assert_stage_passed(adapter.run_stage(RepoStageRequest(stage="build")))

    cleanup = adapter.run_stage(RepoStageRequest(stage="cleanup"))
    assert cleanup.status == "not_configured"
    assert not cleanup.commands


def test_repo_adapter_rejects_undeclared_tools_and_escaping_cwds(tmp_path: Path):
    with pytest.raises(ValidationError, match="repository root"):
        RepoCommand(id="escape", tool="python", cwd="../outside")

    with pytest.raises(ValidationError, match="may never pass"):
        RepoCommand(id="timeout", tool="python", expected_returncodes={124})

    with pytest.raises(ValidationError, match="may never pass"):
        RepoCommand(id="truncated", tool="python", expected_returncodes={125})

    with pytest.raises(ValidationError, match="shell tools"):
        RepoAdapterSpec(allowed_tools={"sh"})

    with pytest.raises(ValidationError, match="non-allow-listed"):
        RepoAdapterSpec(
            allowed_tools={"python"},
            setup=(RepoCommand(id="undeclared", tool="pytest"),),
        )

    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    spec = RepoAdapterSpec(
        allowed_tools={"python"},
        setup=(RepoCommand(id="symlink-escape", tool="python", cwd="escape"),),
    )
    result = RepoAdapter(repo, spec, TrustedLocalBackend()).run_stage(
        RepoStageRequest(stage="setup")
    )
    assert result.status == "failed"
    assert result.commands[0].returncode == 126
    assert "outside the repository" in result.commands[0].stderr


class _TruncatedBackend(ExecutionBackend):
    name = "truncated"

    def run(self, cmd, *, cwd, timeout=300.0, input=None, limits=None):
        return ProcResult(
            125,
            "partial",
            "output exceeded capture limit",
            0.01,
            output_truncated=True,
        )

    def python(self) -> str:
        return "python"

    def tool(self, name: str) -> str:
        return name


def test_repo_stage_records_and_rejects_truncated_output(tmp_path: Path) -> None:
    spec = RepoAdapterSpec(
        allowed_tools={"python"},
        setup=(RepoCommand(id="noisy", tool="python"),),
    )

    result = RepoAdapter(tmp_path, spec, _TruncatedBackend()).run_stage(
        RepoStageRequest(stage="setup")
    )

    assert result.status == "failed"
    assert result.commands[0].output_truncated is True
    assert result.commands[0].passed is False


@pytest.mark.parametrize(
    ("metadata_name", "expected_issue"),
    (
        ("task.yaml", "task specification digest does not match"),
        ("adapter.yaml", "repository adapter digest does not match"),
    ),
)
def test_long_task_integrity_rejects_protocol_metadata_tampering(
    metadata_name: str,
    expected_issue: str,
    tmp_path: Path,
):
    source = LONG_TASKS / "config_parser"
    task_root = tmp_path / "config_parser"
    shutil.copytree(source, task_root)
    manifest = RepoReferenceManifest.from_file(task_root / "reference_manifest.json")

    original = inspect_repo_integrity(
        task_root / "repo",
        manifest,
        task_root / "reference.patch",
        task_path=task_root / "task.yaml",
        adapter_path=task_root / "adapter.yaml",
    )
    assert original.passed, original.issues

    metadata = task_root / metadata_name
    metadata.write_bytes(metadata.read_bytes() + b"\n# tampered\n")
    tampered = inspect_repo_integrity(
        task_root / "repo",
        manifest,
        task_root / "reference.patch",
        task_path=task_root / "task.yaml",
        adapter_path=task_root / "adapter.yaml",
    )

    assert not tampered.passed
    assert any(expected_issue in issue for issue in tampered.issues)


def test_long_task_integrity_binds_repository_file_modes(tmp_path: Path):
    source = LONG_TASKS / "config_parser"
    task_root = tmp_path / "config_parser"
    shutil.copytree(source, task_root)
    manifest = RepoReferenceManifest.from_file(task_root / "reference_manifest.json")
    target = task_root / "repo" / "src" / "config_service" / "loader.py"
    target.chmod(stat.S_IMODE(target.stat().st_mode) | stat.S_IXUSR)

    tampered = inspect_repo_integrity(
        task_root / "repo",
        manifest,
        task_root / "reference.patch",
        task_path=task_root / "task.yaml",
        adapter_path=task_root / "adapter.yaml",
    )

    assert not tampered.passed
    assert "worktree digest does not match the fixed repository" in tampered.issues
