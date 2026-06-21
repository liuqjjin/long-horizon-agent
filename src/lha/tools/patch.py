"""Apply / revert Implementer patches inside a run sandbox.

A Patch may carry explicit ``file_contents`` (preferred — robust) or a
``unified_diff`` (applied with ``git apply``, which works on a plain directory).
Originals are snapshotted so a failed verification can be reverted.
"""

from __future__ import annotations

import difflib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..artifacts import Patch
from .shell import run


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
        for rel, content in patch.file_contents.items():
            target = workdir / rel
            backup.originals[rel] = target.read_text() if target.exists() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            touched.append(rel)
        return touched, backup

    if patch.unified_diff.strip():
        for rel in patch.touched_files:
            target = workdir / rel
            backup.originals[rel] = target.read_text() if target.exists() else None
        with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
            f.write(patch.unified_diff)
            diff_path = f.name
        # Idempotent: if the diff is already applied (e.g. on resume), skip it
        # rather than failing with "patch already applied".
        already = run(
            ["git", "apply", "--reverse", "--check", "-p1", diff_path], cwd=workdir
        )
        if already.ok:
            return list(patch.touched_files), backup
        res = run(
            ["git", "apply", "--whitespace=nowarn", "-p1", diff_path],
            cwd=workdir,
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
