"""Apply / revert Implementer patches inside a run sandbox.

A Patch may carry explicit ``file_contents`` (preferred — robust) or a
``unified_diff`` (applied with ``git apply``, which works on a plain directory).
Originals are snapshotted so a failed verification can be reverted.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..artifacts import Patch
from .shell import run


def _safe_target(workdir: Path, rel: str) -> Path:
    """Resolve ``rel`` inside the sandbox, refusing anything that escapes it.

    A patch is model/LLM output; an absolute or ``../`` key must never let a write
    land outside the run sandbox. (``git apply`` already rejects out-of-tree paths;
    this guards the direct ``file_contents`` write path.)
    """
    root = workdir.resolve()
    target = (root / rel).resolve()
    if Path(rel).is_absolute() or not target.is_relative_to(root):
        raise ValueError(f"refusing to write outside the sandbox: {rel!r}")
    return target


def make_unified_diff(original: str, updated: str, relpath: str) -> str:
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{relpath}",
        tofile=f"b/{relpath}",
    )
    return "".join(diff)


@dataclass
class Backup:
    """Snapshot of files touched by a patch, for revert."""

    # relpath -> original text, or None if the file did not exist before
    originals: dict[str, str | None] = field(default_factory=dict)


def apply_patch(patch: Patch, workdir: str | Path) -> tuple[list[str], Backup]:
    workdir = Path(workdir)
    backup = Backup()
    touched: list[str] = []

    if patch.file_contents:
        # Snapshot-as-we-go, and on any mid-write failure revert what was already
        # written so a multi-file patch can never leave a half-applied sandbox.
        try:
            for rel, content in patch.file_contents.items():
                target = _safe_target(workdir, rel)
                backup.originals[rel] = target.read_text() if target.exists() else None
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                touched.append(rel)
        except Exception:
            revert_patch(backup, workdir)
            raise
        return touched, backup

    if patch.unified_diff.strip():
        for rel in patch.touched_files:
            target = workdir / rel
            backup.originals[rel] = target.read_text() if target.exists() else None
        # Pipe the diff via stdin (no temp file to leak). Idempotent: if it is
        # already applied (e.g. on resume), skip rather than failing.
        already = run(
            ["git", "apply", "--reverse", "--check", "-p1", "-"],
            cwd=workdir,
            input=patch.unified_diff,
        )
        if already.ok:
            return list(patch.touched_files), backup
        res = run(
            ["git", "apply", "--whitespace=nowarn", "-p1", "-"],
            cwd=workdir,
            input=patch.unified_diff,
        )
        if not res.ok:
            raise RuntimeError(f"git apply failed: {res.stderr or res.stdout}")
        return list(patch.touched_files), backup

    return touched, backup


def revert_patch(backup: Backup, workdir: str | Path) -> None:
    workdir = Path(workdir)
    for rel, original in backup.originals.items():
        target = workdir / rel
        if original is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(original)


def save_backup(backup: Backup, path: str | Path) -> None:
    """Persist a backup so revert survives a cross-process resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(backup.originals))


def load_backup(path: str | Path) -> Backup | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return Backup(originals=json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        return None
