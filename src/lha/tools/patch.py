"""Apply / revert Implementer patches inside a run sandbox.

A Patch may carry explicit ``file_contents`` (preferred — robust) or a
``unified_diff`` (applied with ``git apply``, which works on a plain directory).
Originals are snapshotted so a failed verification can be reverted.
"""

from __future__ import annotations

import base64
import binascii
import difflib
import hashlib
import json
import os
import stat
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..artifacts import Patch
from .shell import ProcResult

_PATCH_APPLY_TIMEOUT_S = 60.0
_PATCH_APPLY_OUTPUT_BYTES = 1024 * 1024


class ResolvedPatch(BaseModel):
    """The write set derived from the patch bytes that will actually execute.

    ``Patch.touched_files`` is model-authored metadata and is intentionally not
    trusted. Whole-file patches write exactly ``file_contents``; unified diffs
    mutate exactly the paths parsed from their headers, including both sides of
    a deletion or rename.
    """

    step_id: str
    mode: Literal["contents", "diff", "empty"]
    paths: list[str] = Field(default_factory=list)
    patch_sha256: str


def _canonical_relpath(rel: str) -> str:
    """Return one portable relative path, rejecting aliases and traversal."""
    if not rel or "\\" in rel or "\x00" in rel:
        raise ValueError(f"unsafe patch path: {rel!r}")
    normalized = unicodedata.normalize("NFC", rel)
    p = Path(normalized)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        raise ValueError(f"unsafe patch path: {rel!r}")
    return p.as_posix()


def resolve_patch(patch: Patch, *, patch_bytes: bytes | None = None) -> ResolvedPatch:
    """Resolve the only paths an apply operation can mutate."""
    if patch.file_contents:
        mode: Literal["contents", "diff", "empty"] = "contents"
        raw_paths = list(patch.file_contents)
    elif patch.unified_diff.strip():
        from .policy import diff_paths

        mode = "diff"
        raw_paths = diff_paths(patch.unified_diff)
        if not raw_paths:
            raise ValueError("unified diff has no parseable file paths")
    else:
        mode = "empty"
        raw_paths = []

    paths: list[str] = []
    seen: dict[str, str] = {}
    for raw in raw_paths:
        rel = _canonical_relpath(raw)
        alias = unicodedata.normalize("NFC", rel).casefold()
        if alias in seen and seen[alias] != rel:
            raise ValueError(f"patch contains path aliases: {seen[alias]!r} and {rel!r}")
        seen[alias] = rel
        if rel not in paths:
            paths.append(rel)
    paths.sort()
    ordered_aliases = sorted((alias, rel) for alias, rel in seen.items())
    for index, (parent_alias, parent_rel) in enumerate(ordered_aliases):
        prefix = f"{parent_alias}/"
        for child_alias, child_rel in ordered_aliases[index + 1 :]:
            if child_alias.startswith(prefix):
                raise ValueError(
                    f"patch contains parent/child paths: {parent_rel!r} and {child_rel!r}"
                )
            if child_alias > prefix and not child_alias.startswith(parent_alias):
                break

    encoded = patch_bytes if patch_bytes is not None else patch.model_dump_json().encode("utf-8")
    return ResolvedPatch(
        step_id=patch.step_id,
        mode=mode,
        paths=paths,
        patch_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _safe_target(
    workdir: Path,
    rel: str,
    *,
    allow_leaf_symlink: bool = False,
) -> Path:
    """Resolve ``rel`` inside the sandbox, refusing anything that escapes it.

    A patch is model/LLM output; an absolute or ``../`` key must never let a write
    land outside the run sandbox. (``git apply`` already rejects out-of-tree paths;
    this guards the direct ``file_contents`` write path.) A symlink anywhere on
    the path is refused too: a link created by an earlier write could otherwise
    redirect this one into a protected path that policy checks by name.
    """
    rel = _canonical_relpath(rel)
    root = workdir.resolve()
    target = root / rel
    # `_canonical_relpath` already rejects absolute paths and `..`, so this
    # lexical target is below root without following the final component.
    # Inspect every parent explicitly: resolving the leaf first would follow a
    # malicious symlink outside the sandbox before rollback gets a chance to
    # unlink it.
    probe = target.parent
    while probe != root:
        if probe.is_symlink():
            raise ValueError(f"refusing to write through a symlink: {rel!r}")
        probe = probe.parent
    if target.is_symlink() and not allow_leaf_symlink:
        raise ValueError(f"refusing to write through a symlink: {rel!r}")
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

    # relpath -> original bytes, or None if the file did not exist before
    originals: dict[str, bytes | None] = field(default_factory=dict)
    # relpath -> original permission bits, or None for a newly created file
    modes: dict[str, int | None] = field(default_factory=dict)
    # Parent directories that did not exist before the patch.
    created_dirs: list[str] = field(default_factory=list)


def snapshot_paths(paths: list[str], workdir: str | Path) -> Backup:
    """Capture the exact pre-apply state of a resolved write set."""
    root = Path(workdir)
    backup = Backup()
    created_dirs: set[str] = set()
    for rel in paths:
        target = _safe_target(root, rel)
        try:
            if target.exists():
                if not target.is_file():
                    raise ValueError("target is not a regular file")
                backup.originals[rel] = target.read_bytes()
                backup.modes[rel] = stat.S_IMODE(target.stat().st_mode)
            else:
                backup.originals[rel] = None
                backup.modes[rel] = None
                parent = target.parent
                resolved_root = root.resolve()
                while parent != resolved_root and not parent.exists():
                    created_dirs.add(parent.relative_to(resolved_root).as_posix())
                    parent = parent.parent
        except OSError as e:
            raise ValueError(f"cannot snapshot patch target {rel!r}: {e}") from e
    backup.created_dirs = sorted(
        created_dirs,
        key=lambda value: (len(Path(value).parts), value),
    )
    return backup


def render_review_diff(
    patch: Patch,
    resolved: ResolvedPatch,
    backup: Backup,
) -> str:
    """Render exactly the payload that ``apply_patch`` will execute."""
    if resolved.mode == "diff":
        return patch.unified_diff
    if resolved.mode == "empty":
        return "(no diff)\n"
    contents = {
        _canonical_relpath(path): value
        for path, value in patch.file_contents.items()
    }
    rendered: list[str] = []
    for rel in resolved.paths:
        original = backup.originals[rel]
        try:
            before = original.decode("utf-8") if original is not None else ""
        except UnicodeDecodeError as error:
            raise ValueError(
                f"cannot render a byte-exact review diff for non-UTF-8 target {rel!r}"
            ) from error
        rendered.append(make_unified_diff(before, contents[rel], rel))
    return "".join(rendered) or "(no diff)\n"


def _run_git_apply(diff: str, workdir: Path):
    """Run the fixed control-plane command without importing sandbox eagerly."""
    from ..sandbox.base import (
        PROCESS_CLEANUP_RETURN_CODE,
        process_group_cleanup_supported,
        run_bounded_process,
        scrub_env,
        terminate_process_group,
    )

    if not process_group_cleanup_supported():
        return ProcResult(
            PROCESS_CLEANUP_RETURN_CODE,
            "",
            (
                "git apply requires POSIX process-group cleanup; "
                "use Linux, macOS, or WSL2"
            ),
            0.0,
        )
    environment = scrub_env(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return run_bounded_process(
        ["git", "apply", "--whitespace=nowarn", "-p1", "-"],
        cwd=workdir,
        timeout=_PATCH_APPLY_TIMEOUT_S,
        output_bytes=_PATCH_APPLY_OUTPUT_BYTES,
        env=environment,
        input=diff,
        start_new_session=True,
        on_exit=terminate_process_group,
    )


def apply_patch(
    patch: Patch,
    workdir: str | Path,
    *,
    resolved: ResolvedPatch | None = None,
    backup: Backup | None = None,
) -> tuple[list[str], Backup]:
    workdir = Path(workdir)
    resolved = resolved or resolve_patch(patch)
    backup = backup or snapshot_paths(resolved.paths, workdir)
    touched: list[str] = []

    if resolved.mode == "contents":
        contents_by_path = {_canonical_relpath(rel): value for rel, value in patch.file_contents.items()}
        # Snapshot-as-we-go, and on any mid-write failure revert what was already
        # written so a multi-file patch can never leave a half-applied sandbox.
        try:
            for rel in resolved.paths:
                content = contents_by_path[rel]
                target = _safe_target(workdir, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                touched.append(rel)
        except Exception:
            revert_patch(backup, workdir)
            raise
        return touched, backup

    if resolved.mode == "diff":
        # Pipe the diff via stdin (no temp file to leak). Duplicate application
        # is an error: the transaction journal, not a heuristic reverse-check,
        # decides whether a persisted attempt should be replayed. A reverse
        # check can misclassify mode-only patches as already applied.
        res = _run_git_apply(patch.unified_diff, workdir)
        if not res.ok:
            raise RuntimeError(f"git apply failed: {res.stderr or res.stdout}")
        try:
            for rel in resolved.paths:
                target = _safe_target(workdir, rel)
                # Missing is a valid terminal state for a deletion or the source
                # side of a rename. Symlinks and other file types remain invalid.
                if target.exists() and not target.is_file():
                    raise ValueError(f"patch produced a non-regular file: {rel!r}")
        except Exception:
            revert_patch(backup, workdir)
            raise
        return list(resolved.paths), backup

    return touched, backup


def revert_patch(backup: Backup, workdir: str | Path) -> None:
    workdir = Path(workdir)
    for rel, original in backup.originals.items():
        target = _safe_target(workdir, rel, allow_leaf_symlink=True)
        if original is None:
            target.unlink(missing_ok=True)
        else:
            if target.is_symlink():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original)
            mode = backup.modes.get(rel)
            if mode is not None:
                target.chmod(mode)
    for rel in sorted(
        backup.created_dirs,
        key=lambda value: (len(Path(value).parts), value),
        reverse=True,
    ):
        directory = _safe_target(workdir, rel)
        if directory.exists():
            if not directory.is_dir():
                raise ValueError(f"cannot remove created directory path {rel!r}")
            directory.rmdir()


def _backup_payload(backup: Backup) -> str:
    return json.dumps(
        {
            "schema_version": 4,
            "originals_b64": {
                path: (
                    base64.b64encode(value).decode("ascii")
                    if value is not None
                    else None
                )
                for path, value in backup.originals.items()
            },
            "modes": backup.modes,
            "created_dirs": backup.created_dirs,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def backup_sha256(backup: Backup) -> str:
    return hashlib.sha256(_backup_payload(backup).encode("utf-8")).hexdigest()


def _validated_backup(
    originals: object,
    modes: object,
    created_dirs: object,
) -> Backup:
    if not isinstance(originals, dict) or not isinstance(modes, dict):
        raise ValueError("backup originals and modes must be objects")
    if not isinstance(created_dirs, list):
        raise ValueError("backup created_dirs must be an array")

    checked_originals: dict[str, bytes | None] = {}
    for raw_path, value in originals.items():
        if not isinstance(raw_path, str) or (
            value is not None and not isinstance(value, bytes)
        ):
            raise ValueError("backup originals contain an invalid path or value")
        path = _canonical_relpath(raw_path)
        if path != raw_path:
            raise ValueError(f"backup path is not canonical: {raw_path!r}")
        checked_originals[path] = value

    if set(modes) != set(checked_originals):
        raise ValueError("backup modes must name exactly the original paths")
    checked_modes: dict[str, int | None] = {}
    for raw_path, value in modes.items():
        if not isinstance(raw_path, str):
            raise ValueError("backup mode path must be a string")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 0o7777
        ):
            raise ValueError(f"backup mode is invalid for {raw_path!r}")
        if checked_originals[raw_path] is None and value is not None:
            raise ValueError(f"new backup path {raw_path!r} cannot have an original mode")
        if checked_originals[raw_path] is not None and value is None:
            raise ValueError(f"existing backup path {raw_path!r} is missing its mode")
        checked_modes[raw_path] = value

    checked_dirs: list[str] = []
    for raw_path in created_dirs:
        if not isinstance(raw_path, str):
            raise ValueError("backup directory path must be a string")
        path = _canonical_relpath(raw_path)
        if path != raw_path or path in checked_dirs:
            raise ValueError(f"backup directory path is invalid: {raw_path!r}")
        checked_dirs.append(path)
    return Backup(
        originals=checked_originals,
        modes=checked_modes,
        created_dirs=checked_dirs,
    )


def save_backup(backup: Backup, path: str | Path) -> str:
    """Persist a checksummed backup so revert survives a cross-process resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = _validated_backup(
        backup.originals,
        backup.modes,
        backup.created_dirs,
    )
    digest = backup_sha256(backup)
    envelope = {
        "schema_version": 4,
        "sha256": digest,
        "originals_b64": {
            key: (
                base64.b64encode(value).decode("ascii")
                if value is not None
                else None
            )
            for key, value in backup.originals.items()
        },
        "modes": backup.modes,
        "created_dirs": backup.created_dirs,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(envelope, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:  # pragma: no cover - platform without directory fsync
        pass
    return digest


def load_backup(path: str | Path, *, required: bool = False) -> Backup | None:
    path = Path(path)
    if not path.exists():
        if required:
            raise ValueError(f"required backup is missing: {path}")
        return None
    try:
        raw = json.loads(path.read_text())
        if (
            isinstance(raw, dict)
            and raw.get("schema_version") == 4
            and "originals_b64" in raw
            and "modes" in raw
            and "created_dirs" in raw
            and "sha256" in raw
        ):
            encoded = raw["originals_b64"]
            if not isinstance(encoded, dict):
                raise ValueError(f"backup originals are invalid: {path}")
            originals: dict[str, bytes | None] = {}
            for key, value in encoded.items():
                if not isinstance(key, str) or (
                    value is not None and not isinstance(value, str)
                ):
                    raise ValueError(f"backup originals are invalid: {path}")
                originals[key] = (
                    base64.b64decode(value, validate=True)
                    if value is not None
                    else None
                )
            backup = _validated_backup(originals, raw["modes"], raw["created_dirs"])
            if backup_sha256(backup) != raw["sha256"]:
                raise ValueError(f"backup checksum mismatch: {path}")
            return backup
        if (
            isinstance(raw, dict)
            and raw.get("schema_version") == 3
            and "originals" in raw
            and "modes" in raw
            and "created_dirs" in raw
            and "sha256" in raw
        ):
            old_payload = json.dumps(
                {
                    "originals": raw["originals"],
                    "modes": raw["modes"],
                    "created_dirs": raw["created_dirs"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if hashlib.sha256(old_payload.encode("utf-8")).hexdigest() != raw["sha256"]:
                raise ValueError(f"backup checksum mismatch: {path}")
            originals = {
                key: value.encode("utf-8") if value is not None else None
                for key, value in raw["originals"].items()
            }
            return _validated_backup(originals, raw["modes"], raw["created_dirs"])
        if isinstance(raw, dict) and "originals" in raw and "sha256" in raw:
            # Schema 2 lacked mode/directory evidence. It can be inspected but
            # its digest will not satisfy a schema-2 PatchTransaction, so safe
            # resume fails instead of pretending rollback would be exact.
            old_payload = json.dumps(
                raw["originals"], sort_keys=True, separators=(",", ":")
            )
            if hashlib.sha256(old_payload.encode("utf-8")).hexdigest() != raw["sha256"]:
                raise ValueError(f"backup checksum mismatch: {path}")
            return Backup(
                originals={
                    key: value.encode("utf-8") if value is not None else None
                    for key, value in raw["originals"].items()
                }
            )
        # Schema 1 backups remain readable for trace/recovery of pre-v2 runs.
        if isinstance(raw, dict) and all(v is None or isinstance(v, str) for v in raw.values()):
            return Backup(
                originals={
                    key: value.encode("utf-8") if value is not None else None
                    for key, value in raw.items()
                }
            )
        raise ValueError(f"invalid backup schema: {path}")
    except (OSError, json.JSONDecodeError, TypeError, ValueError, binascii.Error) as e:
        if required:
            raise ValueError(f"required backup is corrupt: {path}: {e}") from e
        return None
