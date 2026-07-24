"""Artifact manifest: the immutable record an approval binds to.

A human approves *a specific change*, not "whatever patch.json contains when
the run resumes". The manifest pins the artifact bytes (SHA-256), the files it
touches, the pre-apply state of those files, the step, and the policy/verifier
configuration in force — so a resume can only ever execute the exact artifact
that was reviewed, and any drift is detected instead of silently applied.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ..artifacts import Patch, Step
from ..clock import now


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str | None:
    """SHA-256 of a file's bytes, or None if it does not exist."""
    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


class ArtifactManifest(BaseModel):
    """Immutable description of one step's patch artifact at execute time."""

    step_id: str
    artifact_sha256: str  # hash of the persisted patch.json bytes
    touched_files: list[str] = Field(default_factory=list)
    # sha256 of each touched file BEFORE the patch was applied (None = absent),
    # i.e. the base the reviewed diff applies to.
    base_state: dict[str, str | None] = Field(default_factory=dict)
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
) -> ArtifactManifest:
    touched = sorted(set(patch.file_contents) | set(patch.touched_files))
    return ArtifactManifest(
        step_id=step.step_id,
        artifact_sha256=sha256_bytes(patch_json_bytes),
        touched_files=touched,
        base_state={rel: file_sha256(workdir / rel) for rel in touched},
        verifiers=list(step.verifiers),
        policy_overrides=list(policy_overrides),
    )
