"""Objective checks for the typed repository lifecycle adapter."""

from __future__ import annotations

import re
from typing import Any

from ...repo_adapter import (
    RepoAdapterSpec,
    RepoIntegrityResult,
    RepoReferenceManifest,
    RepoStageAmbiguous,
    RepoStageResult,
    execute_repo_stage_once,
    inspect_repo_integrity,
)
from ..base import Verifier, VerifyContext
from ..verdict import Check, process_cleanup_failure_detail


class RepoIntegrityVerifier(Verifier):
    name = "repo-integrity"
    family = "code"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        if not isinstance(artifact, RepoIntegrityResult):
            return _failed(self.name, "repository integrity artifact is missing or invalid")
        try:
            manifest = RepoReferenceManifest.from_file(
                str(ctx.step.params["reference_manifest_path"])
            )
            current = inspect_repo_integrity(
                ctx.workdir,
                manifest,
                str(ctx.step.params["reference_patch_path"]),
                task_path=str(ctx.step.params["task_path"]),
                adapter_path=str(ctx.step.params["repo_adapter_path"]),
            )
        except Exception as error:
            return _failed(
                self.name,
                f"repository integrity could not be recomputed: {type(error).__name__}: {error}",
            )
        if current != artifact:
            return _failed(
                self.name,
                "repository integrity evidence changed between execution and verification",
                expected=artifact.model_dump(mode="json"),
                actual=current.model_dump(mode="json"),
            )
        summary = (
            "repository, oracle files, and reference metadata match"
            if current.passed
            else "; ".join(current.issues)
        )
        return Check(
            name=self.name,
            family=self.family,
            passed=current.passed,
            detail={
                "summary": summary,
                "repo_sha256": current.actual_repo_sha256,
                "task_sha256": current.actual_task_sha256,
                "adapter_sha256": current.actual_adapter_sha256,
                "reference_patch_sha256": current.actual_reference_patch_sha256,
                "oracle_files": list(current.oracle_files),
            },
        )


class RepoStageVerifier(Verifier):
    name = "repo-stage"
    family = "code"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        if not isinstance(artifact, RepoStageResult):
            return _failed(self.name, "repository stage artifact is missing or invalid")
        expected_stage = ctx.step.params.get("repo_stage")
        if artifact.stage != expected_stage:
            return _failed(
                self.name,
                f"repository stage mismatch: expected {expected_stage!r}, got {artifact.stage!r}",
            )
        if not artifact.commands:
            return _failed(self.name, f"repository stage {artifact.stage!r} is not configured")
        if not artifact.passed:
            return _stage_check(self.name, artifact, passed=False)

        if ctx.step.params.get("expected_failure"):
            if not any(command.returncode != 0 for command in artifact.commands):
                return _stage_check(
                    self.name,
                    artifact,
                    passed=False,
                    summary=f"{artifact.stage} did not reproduce the expected failure",
                )
        expected_returncode = ctx.step.params.get("expected_returncode")
        if expected_returncode is not None and (
            artifact.commands[-1].returncode != int(expected_returncode)
        ):
            return _stage_check(
                self.name,
                artifact,
                passed=False,
                summary=(
                    f"{artifact.stage} returned {artifact.commands[-1].returncode}; "
                    f"expected {expected_returncode}"
                ),
            )

        expected_test_count = ctx.step.params.get("expected_test_count")
        if expected_test_count is not None:
            output = "\n".join(
                f"{command.stdout}\n{command.stderr}" for command in artifact.commands
            )
            if re.search(rf"\b{int(expected_test_count)} passed\b", output) is None:
                return _stage_check(
                    self.name,
                    artifact,
                    passed=False,
                    summary=(
                        f"{artifact.stage} did not report the expected "
                        f"{expected_test_count} passing tests"
                    ),
                )
        return _stage_check(self.name, artifact, passed=True)


class RepoTargetedVerifier(Verifier):
    """Run the adapter's focused gate against the just-applied model patch."""

    name = "repo-targeted"
    family = "code"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        try:
            spec = RepoAdapterSpec.model_validate(ctx.step.params["repo_adapter_spec"])
            if ctx.step.params.get("repo_stage") != "targeted":
                raise ValueError("repo-targeted verifier requires the targeted stage")
            result = execute_repo_stage_once(
                worktree=ctx.workdir,
                run_dir=ctx.workdir.parent,
                step_id=ctx.step.step_id,
                attempt_id=ctx.attempt_id or f"{ctx.step.step_id}-standalone",
                spec=spec,
                backend=ctx.exec,
                stage="targeted",
            )
        except RepoStageAmbiguous as error:
            return _failed(
                self.name,
                str(error),
                non_retryable=True,
            )
        except Exception as error:
            return _failed(
                self.name,
                f"targeted repository gate could not run: {type(error).__name__}: {error}",
            )
        return _stage_check(self.name, result, passed=result.passed)


def _stage_check(
    name: str,
    result: RepoStageResult,
    *,
    passed: bool,
    summary: str | None = None,
) -> Check:
    cleanup_failures = [
        command for command in result.commands if command.cleanup_unconfirmed
    ]
    if cleanup_failures:
        passed = False
    if summary is None:
        summary = (
            f"{result.stage}: {len(result.commands)} command(s) passed"
            if passed
            else _stage_failure_summary(result)
        )
    detail: dict[str, Any] = {
        "summary": summary,
        "stage": result.stage,
        "result": result.model_dump(mode="json"),
    }
    if cleanup_failures:
        command = cleanup_failures[0]
        detail.update(
            process_cleanup_failure_detail(
                returncode=command.returncode,
                cleanup_unconfirmed=command.cleanup_unconfirmed,
                detail=command.cleanup_detail or command.stderr[-500:],
            )
        )
    return Check(
        name=name,
        family="code",
        passed=passed,
        score=float(sum(command.passed for command in result.commands)),
        threshold=float(len(result.commands)),
        detail=detail,
        duration_s=sum(command.duration_s for command in result.commands),
    )


def _stage_failure_summary(result: RepoStageResult) -> str:
    failures = [
        f"{command.command_id} rc={command.returncode}: "
        f"{(command.stderr or command.stdout).strip()[-240:]}"
        for command in result.commands
        if not command.passed
    ]
    return "; ".join(failures) or f"{result.stage} did not produce passing evidence"


def _failed(name: str, summary: str, **detail: Any) -> Check:
    return Check(
        name=name,
        family="code",
        passed=False,
        detail={"summary": summary, **detail},
    )
