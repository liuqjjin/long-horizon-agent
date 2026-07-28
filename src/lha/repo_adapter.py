"""Typed, shell-free commands for repository lifecycle checks.

A trusted fixture declares a finite command plan, the adapter resolves only
allow-listed tools, and every request/result crossing the execution boundary is
a Pydantic model. Target-controlled strings are passed as argv elements, never
through a shell.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .sandbox.base import ExecutionBackend
from .step_ids import canonical_artifact_segment

RepoStage = Literal[
    "setup",
    "baseline",
    "reproduce",
    "targeted",
    "full",
    "lint",
    "build",
    "cleanup",
]
StageStatus = Literal["passed", "failed", "not_configured"]

_STAGES: tuple[RepoStage, ...] = (
    "setup",
    "baseline",
    "reproduce",
    "targeted",
    "full",
    "lint",
    "build",
    "cleanup",
)
_INFRASTRUCTURE_FAILURE_CODES = frozenset({124, 125, 126, 127})
_SHELL_TOOLS = frozenset({"bash", "cmd", "powershell", "pwsh", "sh", "zsh"})


class RepoCommand(BaseModel):
    """One fixed argv command in a repository lifecycle stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    tool: str = Field(pattern=r"^[a-zA-Z0-9_.+-]+$")
    args: tuple[str, ...] = ()
    cwd: str = "."
    timeout_s: float = Field(default=300.0, gt=0.0, le=3600.0)
    expected_returncodes: frozenset[int] = Field(default_factory=lambda: frozenset({0}))
    stdin: str | None = None

    @field_validator("args")
    @classmethod
    def _safe_args(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any("\x00" in arg for arg in value):
            raise ValueError("argv elements may not contain NUL bytes")
        return value

    @field_validator("cwd")
    @classmethod
    def _relative_cwd(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError("cwd must stay below the repository root")
        return value

    @field_validator("expected_returncodes")
    @classmethod
    def _non_empty_returncodes(cls, value: frozenset[int]) -> frozenset[int]:
        if not value:
            raise ValueError("expected_returncodes must not be empty")
        reserved = value & _INFRASTRUCTURE_FAILURE_CODES
        if reserved:
            raise ValueError(
                f"infrastructure failure return codes may never pass: {sorted(reserved)}"
            )
        return value


class RepoAdapterSpec(BaseModel):
    """The complete, immutable command surface for one repository."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    allowed_tools: frozenset[str] = Field(min_length=1)
    setup: tuple[RepoCommand, ...] = ()
    baseline: tuple[RepoCommand, ...] = ()
    reproduce: tuple[RepoCommand, ...] = ()
    targeted: tuple[RepoCommand, ...] = ()
    full: tuple[RepoCommand, ...] = ()
    lint: tuple[RepoCommand, ...] = ()
    build: tuple[RepoCommand, ...] = ()
    cleanup: tuple[RepoCommand, ...] = ()

    @model_validator(mode="after")
    def _commands_are_unique_and_allowed(self) -> Self:
        forbidden_shells = self.allowed_tools & _SHELL_TOOLS
        if forbidden_shells:
            raise ValueError(f"shell tools are not supported: {sorted(forbidden_shells)}")
        seen: set[str] = set()
        for stage in _STAGES:
            for command in self.commands_for(stage):
                if command.id in seen:
                    raise ValueError(f"duplicate command id: {command.id}")
                seen.add(command.id)
                if command.tool not in self.allowed_tools:
                    raise ValueError(
                        f"command {command.id!r} uses non-allow-listed tool {command.tool!r}"
                    )
        return self

    def commands_for(self, stage: RepoStage) -> tuple[RepoCommand, ...]:
        return getattr(self, stage)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class RepoReferenceManifest(BaseModel):
    """Hashes and oracle metadata that bind one long-task fixture."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[2] = 2
    task_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    repo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_touched_files: tuple[str, ...] = Field(min_length=1)
    oracle_files: tuple[str, ...] = Field(min_length=1)
    expected_baseline_returncode: int
    expected_patched_test_count: int = Field(gt=0)

    @field_validator("reference_touched_files", "oracle_files")
    @classmethod
    def _safe_manifest_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("manifest paths must be unique")
        for item in value:
            path = PurePosixPath(item)
            if not item or path.is_absolute() or ".." in path.parts:
                raise ValueError("manifest paths must stay below the repository root")
        return value

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.model_validate(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class RepoTaskAssets:
    """Trusted files beside a long-task repository, resolved below one task root."""

    task_root: Path
    task_path: Path
    adapter_path: Path
    manifest_path: Path
    reference_patch_path: Path


class RepoIntegrityResult(BaseModel):
    """Evidence that a copied long-task worktree still matches its fixed corpus."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[2] = 2
    task_id: str
    expected_repo_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_repo_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_task_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_adapter_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_reference_patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_reference_patch_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    expected_reference_touched_files: tuple[str, ...]
    actual_reference_touched_files: tuple[str, ...] = ()
    oracle_files: tuple[str, ...]
    issues: tuple[str, ...] = ()
    passed: bool

    @model_validator(mode="after")
    def _passed_matches_evidence(self) -> Self:
        expected = (
            not self.issues
            and self.actual_repo_sha256 == self.expected_repo_sha256
            and self.actual_task_sha256 == self.expected_task_sha256
            and self.actual_adapter_sha256 == self.expected_adapter_sha256
            and self.actual_reference_patch_sha256 == self.expected_reference_patch_sha256
            and self.actual_reference_touched_files
            == self.expected_reference_touched_files
        )
        if self.passed != expected:
            raise ValueError("passed must match repository integrity evidence")
        return self


class RepoStageRequest(BaseModel):
    """A request to execute one declared stage."""

    model_config = ConfigDict(frozen=True)

    stage: RepoStage
    stop_on_failure: bool = True


class RepoCommandResult(BaseModel):
    """Captured result of one command, including the resolved argv."""

    model_config = ConfigDict(frozen=True)

    command_id: str
    stage: RepoStage
    argv: tuple[str, ...]
    cwd: str
    expected_returncodes: frozenset[int]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float = Field(ge=0.0)
    output_truncated: bool = False
    passed: bool

    @model_validator(mode="after")
    def _passed_matches_returncode(self) -> Self:
        expected = (
            self.returncode in self.expected_returncodes
            and not self.output_truncated
        )
        if self.passed != expected:
            raise ValueError(
                "passed must match expected_returncodes and complete output"
            )
        return self


class RepoStageResult(BaseModel):
    """Typed output for one stage."""

    model_config = ConfigDict(frozen=True)

    stage: RepoStage
    status: StageStatus
    commands: tuple[RepoCommandResult, ...] = ()

    @model_validator(mode="after")
    def _status_matches_commands(self) -> Self:
        if not self.commands and self.status != "not_configured":
            raise ValueError("an empty stage must be not_configured")
        if self.commands:
            expected = "passed" if all(result.passed for result in self.commands) else "failed"
            if self.status != expected:
                raise ValueError("stage status does not match command results")
        return self

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class RepoStageIntent(BaseModel):
    """Durable declaration written before a stage command can have side effects."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    step_id: str
    attempt_id: str
    stage: RepoStage
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RepoStageEvidence(BaseModel):
    """A completed stage bound to its command spec and resulting worktree."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    intent: RepoStageIntent
    worktree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: RepoStageResult

    @model_validator(mode="after")
    def _stage_matches_intent(self) -> Self:
        if self.result.stage != self.intent.stage:
            raise ValueError("stage evidence does not match its prepared intent")
        return self


class RepoStageAmbiguous(RuntimeError):
    """A stage may have run, but no durable result proves its outcome."""


_EnvelopeModel = TypeVar("_EnvelopeModel", bound=BaseModel)


class RepoAdapter:
    """Execute a trusted :class:`RepoAdapterSpec` through an execution backend."""

    def __init__(
        self,
        root: str | Path,
        spec: RepoAdapterSpec,
        backend: ExecutionBackend,
    ):
        resolved = Path(root).resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"repository root is not a directory: {resolved}")
        self.root = resolved
        self.spec = spec
        self.backend = backend

    def run_stage(self, request: RepoStageRequest) -> RepoStageResult:
        commands = self.spec.commands_for(request.stage)
        if not commands:
            return RepoStageResult(stage=request.stage, status="not_configured")

        results: list[RepoCommandResult] = []
        for command in commands:
            result = self._run_command(request.stage, command)
            results.append(result)
            if request.stop_on_failure and not result.passed:
                break
        status: StageStatus = "passed" if all(result.passed for result in results) else "failed"
        return RepoStageResult(stage=request.stage, status=status, commands=tuple(results))

    def _run_command(self, stage: RepoStage, command: RepoCommand) -> RepoCommandResult:
        try:
            cwd = self._safe_cwd(command.cwd)
        except (OSError, ValueError) as error:
            return RepoCommandResult(
                command_id=command.id,
                stage=stage,
                argv=(),
                cwd=command.cwd,
                expected_returncodes=command.expected_returncodes,
                returncode=126,
                stdout="",
                stderr=f"unsafe repository cwd: {error}",
                duration_s=0.0,
                passed=False,
            )

        executable = (
            self.backend.python() if command.tool == "python" else self.backend.tool(command.tool)
        )
        argv = (executable, *command.args)
        proc = self.backend.run(
            list(argv),
            cwd=cwd,
            timeout=command.timeout_s,
            input=command.stdin,
        )
        return RepoCommandResult(
            command_id=command.id,
            stage=stage,
            argv=argv,
            cwd=str(cwd.relative_to(self.root)) or ".",
            expected_returncodes=command.expected_returncodes,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=proc.duration_s,
            output_truncated=proc.output_truncated,
            passed=(
                proc.returncode in command.expected_returncodes
                and not proc.output_truncated
            ),
        )

    def _safe_cwd(self, relative: str) -> Path:
        candidate = self.root / relative
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir() or not resolved.is_relative_to(self.root):
            raise ValueError(f"{relative!r} resolves outside the repository")
        return resolved


def resolve_repo_task_assets(
    *,
    target_repo: str | Path,
    adapter_path: str | Path,
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> RepoTaskAssets:
    """Resolve long-task metadata without allowing an input to escape its task root."""

    base = Path(base_dir or Path.cwd()).resolve()
    target = _resolve_from(base, target_repo)
    if not target.is_dir():
        raise ValueError(f"target repository is not a directory: {target}")
    task_root = target.parent
    if target.name != "repo":
        raise ValueError("long-task target_repo must be the task's repo/ directory")

    task = (task_root / "task.yaml").resolve(strict=True)
    adapter = _resolve_from(base, adapter_path)
    manifest = _resolve_from(base, manifest_path)
    reference_patch = manifest.with_name("reference.patch").resolve(strict=True)
    for label, path in (
        ("task spec", task),
        ("repo adapter", adapter),
        ("reference manifest", manifest),
        ("reference patch", reference_patch),
    ):
        if not path.is_file() or path.parent != task_root:
            raise ValueError(f"{label} must be a regular file beside target_repo")
    return RepoTaskAssets(
        task_root=task_root,
        task_path=task,
        adapter_path=adapter,
        manifest_path=manifest,
        reference_patch_path=reference_patch,
    )


def repository_tree_sha256(root: str | Path) -> str:
    """Hash stable path/content pairs, excluding interpreter and test caches."""

    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        raise ValueError(f"repository root is not a directory: {base}")
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"repository contains a symbolic link: {path.relative_to(base)}")
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or ".ruff_cache" in path.parts
            or path.name in {".lha_pytest.json", ".coverage"}
            or path.suffix == ".pyc"
        ):
            continue
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        digest.update(f"{mode:04o}".encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def inspect_repo_integrity(
    root: str | Path,
    manifest: RepoReferenceManifest,
    reference_patch_path: str | Path,
    *,
    task_path: str | Path,
    adapter_path: str | Path,
) -> RepoIntegrityResult:
    """Recompute all fixed-corpus evidence without returning reference patch bytes."""

    worktree = Path(root).resolve(strict=True)
    reference_patch = Path(reference_patch_path).resolve(strict=True)
    issues: list[str] = []
    actual_repo_sha256: str | None = None
    actual_task_sha256: str | None = None
    actual_adapter_sha256: str | None = None
    actual_patch_sha256: str | None = None
    actual_touched: tuple[str, ...] = ()

    try:
        actual_repo_sha256 = repository_tree_sha256(worktree)
        if actual_repo_sha256 != manifest.repo_sha256:
            issues.append("worktree digest does not match the fixed repository")
    except (OSError, ValueError) as error:
        issues.append(f"worktree could not be hashed: {error}")

    actual_task_sha256 = _checked_metadata_digest(
        task_path,
        expected=manifest.task_sha256,
        label="task specification",
        issues=issues,
    )
    actual_adapter_sha256 = _checked_metadata_digest(
        adapter_path,
        expected=manifest.adapter_sha256,
        label="repository adapter",
        issues=issues,
    )

    try:
        patch_bytes = reference_patch.read_bytes()
        actual_patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
        if actual_patch_sha256 != manifest.reference_patch_sha256:
            issues.append("reference patch digest does not match the manifest")
        from .tools.policy import diff_paths

        actual_touched = tuple(diff_paths(patch_bytes.decode("utf-8")))
        if actual_touched != manifest.reference_touched_files:
            issues.append("reference patch write set does not match the manifest")
        overlap = sorted(set(actual_touched) & set(manifest.oracle_files))
        if overlap:
            issues.append(f"reference patch touches oracle files: {overlap}")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        issues.append(f"reference patch could not be checked: {error}")

    for relative in manifest.oracle_files:
        candidate = worktree / relative
        try:
            resolved = candidate.resolve(strict=True)
            if (
                candidate.is_symlink()
                or not resolved.is_file()
                or not resolved.is_relative_to(worktree)
            ):
                raise ValueError("not a regular in-repository file")
        except (OSError, ValueError) as error:
            issues.append(f"oracle file {relative!r} is unavailable: {error}")

    return RepoIntegrityResult(
        task_id=manifest.task_id,
        expected_repo_sha256=manifest.repo_sha256,
        actual_repo_sha256=actual_repo_sha256,
        expected_task_sha256=manifest.task_sha256,
        actual_task_sha256=actual_task_sha256,
        expected_adapter_sha256=manifest.adapter_sha256,
        actual_adapter_sha256=actual_adapter_sha256,
        expected_reference_patch_sha256=manifest.reference_patch_sha256,
        actual_reference_patch_sha256=actual_patch_sha256,
        expected_reference_touched_files=manifest.reference_touched_files,
        actual_reference_touched_files=actual_touched,
        oracle_files=manifest.oracle_files,
        issues=tuple(issues),
        passed=not issues,
    )


def _checked_metadata_digest(
    path: str | Path,
    *,
    expected: str,
    label: str,
    issues: list[str],
) -> str | None:
    candidate = Path(path)
    try:
        if candidate.is_symlink():
            raise ValueError("symbolic links are not accepted")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("not a regular file")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        issues.append(f"{label} could not be hashed: {error}")
        return None
    if actual != expected:
        issues.append(f"{label} digest does not match the manifest")
    return actual


def execute_repo_stage_once(
    *,
    worktree: str | Path,
    run_dir: str | Path,
    step_id: str,
    attempt_id: str,
    spec: RepoAdapterSpec,
    backend: ExecutionBackend,
    stage: RepoStage,
) -> RepoStageResult:
    """Execute one stage at most once per attempt and durably bind its result.

    If a process dies after the command starts but before evidence is persisted,
    recovery fails closed. Replaying an unknown side effect would be less safe
    than reporting that the attempt needs operator inspection.
    """

    root = Path(worktree).resolve(strict=True)
    run_root = Path(run_dir).resolve(strict=True)
    safe_step = _safe_segment(step_id)
    safe_attempt = _safe_segment(attempt_id)
    attempt_dir = run_root / "steps" / safe_step / "attempts" / safe_attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    intent_path = attempt_dir / "repo_stage_intent.json"
    evidence_path = attempt_dir / "repo_stage_evidence.json"
    spec_payload = json.dumps(
        spec.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    intent = RepoStageIntent(
        step_id=step_id,
        attempt_id=attempt_id,
        stage=stage,
        spec_sha256=hashlib.sha256(spec_payload).hexdigest(),
    )

    if evidence_path.exists():
        try:
            persisted_intent = _read_checksummed_model(intent_path, RepoStageIntent)
            evidence = _read_checksummed_model(evidence_path, RepoStageEvidence)
        except Exception as error:
            raise RepoStageAmbiguous(
                f"persisted stage evidence is invalid for {step_id}/{attempt_id}: {error}"
            ) from error
        if persisted_intent != intent or evidence.intent != persisted_intent:
            raise RepoStageAmbiguous(
                f"stage intent changed for {step_id}/{attempt_id}; refusing replay"
            )
        current_tree = repository_tree_sha256(root)
        if current_tree != evidence.worktree_sha256:
            raise RepoStageAmbiguous(
                f"worktree changed after stage {stage!r}; refusing stale evidence"
            )
        _publish_stage_result(run_root, safe_step, evidence.result)
        return evidence.result

    if intent_path.exists():
        try:
            persisted_intent = _read_checksummed_model(intent_path, RepoStageIntent)
        except Exception as error:
            raise RepoStageAmbiguous(
                f"prepared stage intent is invalid for {step_id}/{attempt_id}: {error}"
            ) from error
        if persisted_intent != intent:
            raise RepoStageAmbiguous(
                f"prepared stage intent changed for {step_id}/{attempt_id}"
            )
        raise RepoStageAmbiguous(
            f"stage {stage!r} may already have executed for {step_id}/{attempt_id}; "
            "refusing to duplicate its side effects"
        )

    _write_checksummed_model(intent_path, intent)
    result = RepoAdapter(root, spec, backend).run_stage(RepoStageRequest(stage=stage))
    evidence = RepoStageEvidence(
        intent=intent,
        worktree_sha256=repository_tree_sha256(root),
        result=result,
    )
    _write_checksummed_model(evidence_path, evidence)
    _publish_stage_result(run_root, safe_step, result)
    return result


def _resolve_from(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


def _safe_segment(value: str) -> str:
    return canonical_artifact_segment(value)


def _publish_stage_result(run_dir: Path, safe_step: str, result: RepoStageResult) -> None:
    payload = result.model_dump_json(indent=2)
    step_path = run_dir / "steps" / safe_step / "repo_stage.json"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    _durable_replace(step_path, payload)
    _durable_replace(run_dir / "repo_stage.json", payload)


def _canonical_payload(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _write_checksummed_model(path: Path, model: BaseModel) -> None:
    payload = model.model_dump(mode="json")
    envelope = {
        "schema_version": 1,
        "sha256": hashlib.sha256(_canonical_payload(payload)).hexdigest(),
        "payload": payload,
    }
    _durable_replace(path, json.dumps(envelope, indent=2))


def _read_checksummed_model(
    path: Path,
    model_type: type[_EnvelopeModel],
) -> _EnvelopeModel:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact path is missing or unsafe: {path}")
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"artifact is unreadable: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "sha256", "payload"}:
        raise ValueError("artifact is not a checksummed envelope")
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported artifact envelope version: {raw['schema_version']!r}")
    payload = raw["payload"]
    stored = raw["sha256"]
    if not isinstance(payload, dict) or not isinstance(stored, str):
        raise ValueError("artifact envelope fields have invalid types")
    actual = hashlib.sha256(_canonical_payload(payload)).hexdigest()
    if actual != stored:
        raise ValueError(
            "artifact failed its integrity check "
            f"(stored {stored[:12]}…, computed {actual[:12]}…)"
        )
    return model_type.model_validate(payload)


def _durable_replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"artifact path is unsafe: {path}")
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:  # pragma: no cover - platforms without directory fsync
        pass


__all__ = [
    "RepoAdapter",
    "RepoAdapterSpec",
    "RepoCommand",
    "RepoCommandResult",
    "RepoIntegrityResult",
    "RepoReferenceManifest",
    "RepoStageAmbiguous",
    "RepoStageEvidence",
    "RepoStageIntent",
    "RepoStage",
    "RepoStageRequest",
    "RepoStageResult",
    "RepoTaskAssets",
    "StageStatus",
    "execute_repo_stage_once",
    "inspect_repo_integrity",
    "repository_tree_sha256",
    "resolve_repo_task_assets",
]
