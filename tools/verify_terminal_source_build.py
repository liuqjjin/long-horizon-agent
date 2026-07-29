#!/usr/bin/env python3
"""Rebuild and verify the source package recorded by Terminal-Bench evidence."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from lha.bench.terminal_public_evidence import (
    SourceAttestation,
    TerminalBenchPublicEvidenceIndex,
    TerminalBenchPublicEvidenceValidation,
    validate_terminal_bench_public_evidence,
)
from lha.tools.shell import run, venv_tool

_INDEX_FILE = "evidence.json"
_SOURCE_ATTESTATION_FILE = "source_attestation.json"
_MAX_ATTESTATION_BYTES = 64 * 1024
_BUILD_COMMAND = "uv build --clear"


class SourceBuildVerificationError(ValueError):
    """The recorded evaluation source cannot reproduce the attested wheel."""


def _load_model_file(
    path: Path,
    model_type: type[SourceAttestation] | type[TerminalBenchPublicEvidenceIndex],
) -> SourceAttestation | TerminalBenchPublicEvidenceIndex:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not (
            0 < metadata.st_size <= _MAX_ATTESTATION_BYTES
        ):
            raise SourceBuildVerificationError(
                f"{path.name} must be a bounded regular file"
            )
        payload = b""
        while len(payload) <= _MAX_ATTESTATION_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _MAX_ATTESTATION_BYTES + 1 - len(payload)),
            )
            if not chunk:
                break
            payload += chunk
        if len(payload) != metadata.st_size:
            raise SourceBuildVerificationError(
                f"{path.name} changed while it was read"
            )
        return model_type.model_validate_json(payload)
    except SourceBuildVerificationError:
        raise
    except (OSError, ValueError) as exc:
        raise SourceBuildVerificationError(
            f"cannot read {path.name}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def load_attested_source(
    evidence_dir: str | Path,
) -> tuple[SourceAttestation, TerminalBenchPublicEvidenceValidation]:
    """Load a schema-4 attestation after validating the complete evidence package."""
    root = Path(evidence_dir)
    index = _load_model_file(root / _INDEX_FILE, TerminalBenchPublicEvidenceIndex)
    if not isinstance(index, TerminalBenchPublicEvidenceIndex):
        raise AssertionError("unexpected evidence index type")
    if index.schema_version != 4:
        raise SourceBuildVerificationError(
            "Terminal-Bench source reproduction requires evidence schema 4"
        )
    attestation = _load_model_file(
        root / _SOURCE_ATTESTATION_FILE,
        SourceAttestation,
    )
    if not isinstance(attestation, SourceAttestation):
        raise AssertionError("unexpected source attestation type")
    try:
        validation = validate_terminal_bench_public_evidence(root)
    except (OSError, ValueError) as exc:
        raise SourceBuildVerificationError(
            f"cannot validate Terminal-Bench evidence: {exc}"
        ) from exc
    expected_identity = (
        validation.evaluated_commit_sha,
        validation.evaluated_tree_sha,
        validation.evaluated_wheel_filename,
        validation.evaluated_wheel_size_bytes,
        validation.evaluated_wheel_sha256,
    )
    observed_identity = (
        attestation.commit_sha,
        attestation.tree_sha,
        attestation.wheel_filename,
        attestation.wheel_size_bytes,
        attestation.wheel_sha256,
    )
    if expected_identity != observed_identity:
        raise SourceBuildVerificationError(
            "validated evidence does not return the complete source attestation"
        )
    return attestation, validation


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    label: str,
) -> str:
    result = run(command, cwd=cwd, timeout=timeout)
    if not result.ok:
        detail = (result.stderr or result.stdout).strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        raise SourceBuildVerificationError(
            f"{label} failed with exit code {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    return result.stdout.strip()


def verify_attested_source_build(
    repository_root: str | Path,
    attestation: SourceAttestation,
    *,
    git_executable: str | None = None,
    uv_executable: str | None = None,
) -> str:
    """Check ancestry and tree identity, then reproduce the attested wheel."""
    root = Path(repository_root).resolve()
    git = git_executable or venv_tool("git")
    uv = uv_executable or venv_tool("uv")

    commit = _run_checked(
        [git, "rev-parse", "--verify", f"{attestation.commit_sha}^{{commit}}"],
        cwd=root,
        timeout=60,
        label="evaluated commit lookup",
    )
    if commit != attestation.commit_sha:
        raise SourceBuildVerificationError(
            "evaluated commit does not resolve to the attested full SHA"
        )
    ancestry = run(
        [git, "merge-base", "--is-ancestor", attestation.commit_sha, "HEAD"],
        cwd=root,
        timeout=60,
    )
    if ancestry.returncode == 1:
        raise SourceBuildVerificationError(
            "evaluated commit is not an ancestor of HEAD; a squash merge cannot "
            "preserve this source attestation"
        )
    if not ancestry.ok:
        raise SourceBuildVerificationError(
            "cannot prove that the evaluated commit is reachable from HEAD"
        )
    tree = _run_checked(
        [git, "rev-parse", f"{attestation.commit_sha}^{{tree}}"],
        cwd=root,
        timeout=60,
        label="evaluated tree lookup",
    )
    if tree != attestation.tree_sha:
        raise SourceBuildVerificationError(
            f"evaluated Git tree differs: expected {attestation.tree_sha}, got {tree}"
        )
    if attestation.reproducible_build_command != _BUILD_COMMAND:
        raise SourceBuildVerificationError(
            "unsupported source build command; expected exactly 'uv build --clear'"
        )

    temporary_root = Path(tempfile.mkdtemp(prefix="lha-source-rebuild-"))
    worktree = temporary_root / "checkout"
    registered = False
    primary_error: BaseException | None = None
    reproduced_wheel: str | None = None
    try:
        _run_checked(
            [
                git,
                "worktree",
                "add",
                "--detach",
                str(worktree),
                attestation.commit_sha,
            ],
            cwd=root,
            timeout=120,
            label="isolated source checkout",
        )
        registered = True
        _run_checked(
            [uv, "build", "--clear"],
            cwd=worktree,
            timeout=900,
            label="attested wheel build",
        )
        wheel = worktree / "dist" / attestation.wheel_filename
        try:
            metadata = wheel.lstat()
        except OSError as exc:
            raise SourceBuildVerificationError(
                f"attested wheel was not produced: {wheel.name}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SourceBuildVerificationError(
                "attested wheel output must be a regular file"
            )
        if metadata.st_size != attestation.wheel_size_bytes:
            raise SourceBuildVerificationError(
                "reproduced wheel size differs: "
                f"expected {attestation.wheel_size_bytes}, got {metadata.st_size}"
            )
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest != attestation.wheel_sha256:
            raise SourceBuildVerificationError(
                "reproduced wheel SHA-256 differs: "
                f"expected {attestation.wheel_sha256}, got {digest}"
            )
        reproduced_wheel = wheel.name
    except BaseException as exc:
        primary_error = exc
    finally:
        cleanup_error: SourceBuildVerificationError | None = None
        if registered:
            result = run(
                [git, "worktree", "remove", "--force", str(worktree)],
                cwd=root,
                timeout=120,
            )
            if not result.ok:
                cleanup_error = SourceBuildVerificationError(
                    "failed to remove the isolated source worktree"
                )
        shutil.rmtree(temporary_root, ignore_errors=True)
        if cleanup_error is not None:
            if primary_error is not None:
                raise cleanup_error from primary_error
            raise cleanup_error
    if primary_error is not None:
        raise primary_error
    if reproduced_wheel is None:
        raise AssertionError("source build completed without a wheel result")
    return reproduced_wheel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="rebuild the wheel bound to Terminal-Bench schema-4 evidence"
    )
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--evidence",
        default="benchmarks/terminal_bench_2_1",
        help="Terminal-Bench public evidence directory",
    )
    args = parser.parse_args(argv)
    try:
        attestation, validation = load_attested_source(args.evidence)
        wheel = verify_attested_source_build(args.root, attestation)
    except SourceBuildVerificationError as exc:
        print(f"Terminal-Bench source reproduction: FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "Terminal-Bench source reproduction: ok "
        f"(commit={validation.evaluated_commit_sha}; "
        f"tree={validation.evaluated_tree_sha}; wheel={wheel}; "
        f"sha256={validation.evaluated_wheel_sha256})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
