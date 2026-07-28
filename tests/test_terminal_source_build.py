from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from lha.bench.terminal_public_evidence import SourceAttestation


def _load_tool() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools/verify_terminal_source_build.py"
    spec = importlib.util.spec_from_file_location("verify_terminal_source_build", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _repository_and_attestation(
    tmp_path: Path,
) -> tuple[Path, SourceAttestation, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "source.txt").write_text("evaluated source\n")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-qm", "evaluated")
    evaluated_commit = _git(repo, "rev-parse", "HEAD")
    evaluated_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    (repo / "later.txt").write_text("release checks\n")
    _git(repo, "add", "later.txt")
    _git(repo, "commit", "-qm", "later")

    wheel_payload = b"byte-identical-wheel"
    wheel_filename = "lha-0.4.2.dev0-py3-none-any.whl"
    fake_uv = tmp_path / "uv"
    _write_executable(
        fake_uv,
        (
            "#!/bin/sh\n"
            'test "$1" = build\n'
            'test "$2" = --clear\n'
            "mkdir -p dist\n"
            f"printf %s byte-identical-wheel > dist/{wheel_filename}\n"
        ),
    )
    attestation = SourceAttestation(
        repository_url="https://example.com/owner/repository",
        commit_sha=evaluated_commit,
        tree_sha=evaluated_tree,
        package_version="0.4.2.dev0",
        wheel_filename=wheel_filename,
        wheel_size_bytes=len(wheel_payload),
        wheel_sha256=hashlib.sha256(wheel_payload).hexdigest(),
        reproducible_build_command="uv build --clear",
    )
    return repo, attestation, fake_uv


def test_attested_source_build_checks_tree_wheel_and_cleans_worktree(
    tmp_path: Path,
) -> None:
    repo, attestation, fake_uv = _repository_and_attestation(tmp_path)

    result = TOOL.verify_attested_source_build(
        repo,
        attestation,
        uv_executable=str(fake_uv),
    )

    assert result == attestation.wheel_filename
    worktree_lines = _git(repo, "worktree", "list", "--porcelain").splitlines()
    assert [line for line in worktree_lines if line.startswith("worktree ")] == [
        f"worktree {repo}"
    ]


def test_attested_source_build_rejects_a_reachable_but_non_ancestor_commit(
    tmp_path: Path,
) -> None:
    repo, attestation, fake_uv = _repository_and_attestation(tmp_path)
    old_commit = attestation.commit_sha
    env = {**os.environ, "GIT_AUTHOR_DATE": "2001-01-01T00:00:00Z"}
    subprocess.run(
        ["git", "checkout", "--orphan", "squashed-main"],
        cwd=repo,
        check=True,
        capture_output=True,
        env=env,
    )
    for path in repo.iterdir():
        if path.name != ".git" and path.is_file():
            path.unlink()
    (repo / "squashed.txt").write_text("squashed release\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "squashed")
    assert _git(repo, "cat-file", "-t", old_commit) == "commit"

    with pytest.raises(
        TOOL.SourceBuildVerificationError,
        match="squash merge cannot preserve",
    ):
        TOOL.verify_attested_source_build(
            repo,
            attestation,
            uv_executable=str(fake_uv),
        )


def test_attested_source_build_rejects_wheel_size_drift(tmp_path: Path) -> None:
    repo, attestation, fake_uv = _repository_and_attestation(tmp_path)
    drifted = attestation.model_copy(
        update={"wheel_size_bytes": attestation.wheel_size_bytes + 1}
    )

    with pytest.raises(
        TOOL.SourceBuildVerificationError,
        match="wheel size differs",
    ):
        TOOL.verify_attested_source_build(
            repo,
            drifted,
            uv_executable=str(fake_uv),
        )
