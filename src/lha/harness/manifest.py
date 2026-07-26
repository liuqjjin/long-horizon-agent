"""Artifact manifest: the immutable record an approval binds to.

A human approves *a specific change*, not "whatever patch.json contains when
the run resumes". The manifest pins the artifact bytes (SHA-256), the files it
touches, the pre-apply state of those files, the step, and the policy/verifier
configuration in force — so a resume can only ever execute the exact artifact
that was reviewed, and any drift is detected instead of silently applied.
"""

from __future__ import annotations

import hashlib
import stat
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..artifacts import Patch, Step
from ..clock import now
from ..tools.patch import ResolvedPatch, resolve_patch


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str | None:
    """SHA-256 of a regular file, or None if it does not exist."""
    if path.is_symlink():
        raise ValueError(f"refusing to hash a symbolic link: {path}")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"refusing to hash a non-regular file: {path}")
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise ValueError(f"could not hash file {path}: {error}") from error


class FileState(BaseModel):
    """Byte and permission identity for one regular path."""

    kind: Literal["missing", "file"]
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mode: int | None = Field(default=None, ge=0, le=0o7777)

    @model_validator(mode="after")
    def _fields_match_kind(self) -> "FileState":
        if self.kind == "missing" and (self.sha256 is not None or self.mode is not None):
            raise ValueError("missing path cannot carry a hash or mode")
        if self.kind == "file" and (self.sha256 is None or self.mode is None):
            raise ValueError("regular file must carry both a hash and mode")
        return self


def file_state(path: Path) -> FileState:
    """Capture regular-file bytes and mode without following symbolic links."""
    if path.is_symlink():
        raise ValueError(f"refusing to inspect a symbolic link: {path}")
    if not path.exists():
        return FileState(kind="missing")
    if not path.is_file():
        raise ValueError(f"refusing to inspect a non-regular file: {path}")
    try:
        metadata = path.stat()
        return FileState(
            kind="file",
            sha256=sha256_bytes(path.read_bytes()),
            mode=stat.S_IMODE(metadata.st_mode),
        )
    except OSError as error:
        raise ValueError(f"could not inspect file {path}: {error}") from error


def saved_file_state(original: bytes | None, mode: int | None) -> FileState:
    """Reconstruct a pre-apply state from a byte-exact rollback snapshot."""
    if original is None:
        if mode is not None:
            raise ValueError("missing backup path cannot carry a mode")
        return FileState(kind="missing")
    if mode is None:
        raise ValueError("existing backup path is missing its mode")
    return FileState(kind="file", sha256=sha256_bytes(original), mode=mode)


class ArtifactManifest(BaseModel):
    """Immutable description of one step's patch artifact at execute time."""

    step_id: str
    artifact_sha256: str  # hash of the persisted patch.json bytes
    touched_files: list[str] = Field(default_factory=list)
    # Byte hash and permission bits of each touched path before apply.
    base_state: dict[str, FileState] = Field(default_factory=dict)
    verifiers: list[str] = Field(default_factory=list)
    policy_overrides: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)


def build_manifest(
    *,
    step: Step,
    patch: Patch,
    patch_json_bytes: bytes,
    workdir: Path,
    policy_overrides: list[str],
    resolved: ResolvedPatch | None = None,
) -> ArtifactManifest:
    touched = (resolved or resolve_patch(patch, patch_bytes=patch_json_bytes)).paths
    return ArtifactManifest(
        step_id=step.step_id,
        artifact_sha256=sha256_bytes(patch_json_bytes),
        touched_files=touched,
        base_state={rel: file_state(workdir / rel) for rel in touched},
        verifiers=list(step.verifiers),
        policy_overrides=list(policy_overrides),
    )
