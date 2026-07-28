"""Strict filesystem durability helpers for run-owned state and worktrees.

An ``fsync`` on a newly created child directory does not persist the directory
entry that names it.  Callers that create nested evidence therefore use
``durable_mkdir_chain`` so every new entry is followed by a sync of its parent.
File helpers also reject links and inode replacement while establishing a
durability barrier.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def _absolute(path: str | Path) -> Path:
    """Return a lexical absolute path without following a symbolic link."""
    return Path(os.path.abspath(os.fspath(path)))


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def fsync_directory(path: str | Path) -> None:
    """Synchronize one real directory, failing if its identity is unsafe."""
    directory = Path(path)
    descriptor = os.open(directory, _directory_flags())
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError(f"directory durability target is not a directory: {directory}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_directory(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"durable directory path is unsafe: {path}")
    return metadata


def _anchored_directory_target(target: Path, anchor: str | Path) -> tuple[Path, Path]:
    """Map a target onto an existing anchor's real directory identity.

    macOS exposes some temporary directories through aliases such as ``/var``
    and ``/private/var``.  The anchor may use the alias while a previously
    resolved patch target uses the real path.  Only the pre-anchor spelling is
    normalized here; components below the anchor remain lexical and are still
    checked one by one by ``durable_mkdir_chain``.
    """
    lexical_base = _absolute(anchor)
    lexical_metadata = _validate_directory(lexical_base)
    real_base = lexical_base.resolve(strict=True)
    real_metadata = _validate_directory(real_base)
    if (
        lexical_metadata.st_dev,
        lexical_metadata.st_ino,
    ) != (
        real_metadata.st_dev,
        real_metadata.st_ino,
    ):
        raise OSError(f"durable directory anchor identity changed: {lexical_base}")

    if target.is_relative_to(lexical_base):
        relative = target.relative_to(lexical_base)
    elif target.is_relative_to(real_base):
        relative = target.relative_to(real_base)
    else:
        raise ValueError(f"durable directory escapes its anchor: {target}")
    return real_base / relative, real_base


def durable_mkdir_chain(
    path: str | Path,
    *,
    anchor: str | Path | None = None,
    mode: int = 0o777,
) -> Path:
    """Create missing directories one at a time and sync every new parent entry.

    When ``anchor`` is supplied, every created component is required to remain
    lexically below that existing real directory.  This is used for run evidence
    and patch targets, where following an in-tree link would cross a trust
    boundary.
    """
    target = _absolute(path)
    if anchor is not None:
        target, base = _anchored_directory_target(target, anchor)
        current = base
        parts = target.relative_to(base).parts
    else:
        missing: list[str] = []
        current = target
        while True:
            try:
                _validate_directory(current)
            except FileNotFoundError:
                if current.parent == current:
                    raise
                missing.append(current.name)
                current = current.parent
                continue
            break
        parts = tuple(reversed(missing))

    for part in parts:
        candidate = current / part
        created = False
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            try:
                os.mkdir(candidate, mode)
                created = True
            except FileExistsError:
                metadata = candidate.lstat()
            else:
                metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError(f"durable directory path is unsafe: {candidate}")
        if created:
            # Sync the child inode first, then the parent entry that makes the
            # child reachable after a crash.
            fsync_directory(candidate)
            fsync_directory(current)
        current = candidate
    return target


def _regular_file_metadata(path: Path) -> os.stat_result | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise OSError(f"durable file path is unsafe: {path}")
    return metadata


def sync_regular_file(
    path: str | Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> os.stat_result:
    """Sync one non-linked regular file and prove the named inode was synced."""
    target = Path(path)
    named_before = _regular_file_metadata(target)
    if named_before is None:
        raise FileNotFoundError(target)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(target, flags)
    try:
        opened_before = os.fstat(descriptor)
        identity = (opened_before.st_dev, opened_before.st_ino)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or identity != (named_before.st_dev, named_before.st_ino)
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise OSError(f"durable file identity changed before sync: {target}")
        os.fsync(descriptor)
        opened_after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_after.st_mode)
            or opened_after.st_nlink != 1
            or (opened_after.st_dev, opened_after.st_ino) != identity
        ):
            raise OSError(f"durable file identity changed during sync: {target}")
    finally:
        os.close(descriptor)

    named_after = _regular_file_metadata(target)
    if (
        named_after is None
        or (named_after.st_dev, named_after.st_ino) != identity
        or named_after.st_nlink != 1
    ):
        raise OSError(f"durable file identity changed after sync: {target}")
    return named_after


def atomic_replace_bytes(
    path: str | Path,
    data: bytes,
    *,
    anchor: str | Path | None = None,
    mode: int | None = None,
) -> None:
    """Replace one regular file with synced bytes and a synced directory entry."""
    target = _absolute(path)
    parent = durable_mkdir_chain(target.parent, anchor=anchor)
    previous = _regular_file_metadata(target)
    selected_mode = (
        mode
        if mode is not None
        else stat.S_IMODE(previous.st_mode) if previous is not None else 0o600
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    identity: tuple[int, int] | None = None
    try:
        os.fchmod(descriptor, selected_mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError(f"temporary durable file is unsafe: {temporary}")
            identity = (metadata.st_dev, metadata.st_ino)
        os.replace(temporary, target)
        fsync_directory(parent)
        assert identity is not None
        sync_regular_file(target, expected_identity=identity)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def atomic_replace_text(
    path: str | Path,
    text: str,
    *,
    anchor: str | Path | None = None,
    mode: int | None = None,
) -> None:
    """UTF-8 text variant of ``atomic_replace_bytes``."""
    atomic_replace_bytes(
        path,
        text.encode("utf-8"),
        anchor=anchor,
        mode=mode,
    )
