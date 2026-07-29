"""Strict filesystem durability helpers for run-owned state and worktrees.

An ``fsync`` on a newly created child directory does not persist the directory
entry that names it.  Callers that create nested evidence therefore use
``durable_mkdir_chain`` so every new entry is followed by a sync of its parent.
File helpers also reject links and inode replacement while establishing a
durability barrier.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

_ATOMIC_REPLACE_TEMP = re.compile(
    r"\.(?P<target>.+)\.(?P<nonce>[0-9a-f]{32})\.tmp\Z"
)


@dataclass(frozen=True)
class AnchoredAtomicReplaceTemp:
    """A validated, run-owned atomic-replace file that has not been committed."""

    path: Path
    target_name: str
    data: bytes
    metadata: os.stat_result


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


@contextmanager
def _anchored_directory_target(
    target: Path,
    anchor: str | Path,
) -> Iterator[tuple[Path, Path, int]]:
    """Open an anchor and map a target onto that exact directory identity.

    macOS exposes some temporary directories through aliases such as ``/var``
    and ``/private/var``.  The anchor may use the alias while a previously
    resolved patch target uses the real path.  Only the pre-anchor spelling is
    normalized here. The named anchor is checked before and after opening, and
    again when the caller finishes, so replacing it with another same-owner
    directory cannot redirect any descriptor-relative operation.
    """
    lexical_base = _absolute(anchor)
    lexical_metadata = _validate_directory(lexical_base)
    real_base = lexical_base.resolve(strict=True)
    real_metadata = _validate_directory(real_base)
    if not _same_inode(lexical_metadata, real_metadata):
        raise OSError(f"durable directory anchor identity changed: {lexical_base}")

    if target.is_relative_to(lexical_base):
        relative = target.relative_to(lexical_base)
    elif target.is_relative_to(real_base):
        relative = target.relative_to(real_base)
    else:
        raise ValueError(f"durable directory escapes its anchor: {target}")

    descriptor = os.open(lexical_base, _directory_flags())
    try:
        opened = os.fstat(descriptor)
        _require_directory_identity(lexical_metadata, opened, lexical_base)
        _require_directory_identity(real_metadata, opened, real_base)
        _require_anchor_names(
            lexical_base,
            real_base,
            opened,
        )
        yield real_base / relative, real_base, descriptor
        _require_anchor_names(
            lexical_base,
            real_base,
            opened,
        )
    finally:
        os.close(descriptor)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _same_file_version(left: os.stat_result, right: os.stat_result) -> bool:
    stable = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, field) == getattr(right, field) for field in stable)


def _require_directory_identity(
    named: os.stat_result,
    opened: os.stat_result,
    path: Path,
) -> None:
    if (
        stat.S_ISLNK(named.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or not _same_inode(named, opened)
    ):
        raise OSError(f"durable directory identity is unsafe: {path}")


def _require_anchor_names(
    lexical_base: Path,
    real_base: Path,
    opened: os.stat_result,
) -> None:
    """Prove both accepted spellings still name the opened anchor."""
    try:
        lexical_named = lexical_base.lstat()
        real_named = real_base.lstat()
    except FileNotFoundError as error:
        raise OSError(
            f"durable directory anchor disappeared: {lexical_base}"
        ) from error
    _require_directory_identity(lexical_named, opened, lexical_base)
    _require_directory_identity(real_named, opened, real_base)


def _require_regular_identity(
    named: os.stat_result,
    opened: os.stat_result,
    path: Path,
) -> None:
    if (
        stat.S_ISLNK(named.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not _same_inode(named, opened)
    ):
        raise OSError(f"durable file identity is unsafe: {path}")


def _named_stat(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _validate_named_regular(
    directory_fd: int,
    name: str,
    path: Path,
) -> os.stat_result | None:
    metadata = _named_stat(directory_fd, name)
    if metadata is None:
        return None
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise OSError(f"durable file path is unsafe: {path}")
    return metadata


def _sync_created_directory(
    child_fd: int,
    parent_fd: int,
    child_path: Path,
    parent_path: Path,
) -> None:
    """Persist a new directory inode and the parent entry that names it."""
    del child_path, parent_path  # Paths are diagnostic inputs for fault-injection tests.
    os.fsync(child_fd)
    os.fsync(parent_fd)


def _open_or_create_directory_at(
    parent_fd: int,
    component: str,
    parent_path: Path,
    *,
    create: bool,
    mode: int,
) -> tuple[int, os.stat_result]:
    """Open one child using only its already-validated parent descriptor."""
    child_path = parent_path / component
    named = _named_stat(parent_fd, component)
    created = False
    if named is None:
        if not create:
            raise FileNotFoundError(child_path)
        try:
            os.mkdir(component, mode, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        named = _named_stat(parent_fd, component)
    assert named is not None
    if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
        raise OSError(f"durable directory path is unsafe: {child_path}")

    child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
    try:
        opened = os.fstat(child_fd)
        _require_directory_identity(named, opened, child_path)
        named_after = _named_stat(parent_fd, component)
        if named_after is None:
            raise OSError(f"durable directory disappeared: {child_path}")
        _require_directory_identity(named_after, opened, child_path)
        if created:
            _sync_created_directory(
                child_fd,
                parent_fd,
                child_path,
                parent_path,
            )
    except BaseException:
        os.close(child_fd)
        raise
    return child_fd, opened


def _revalidate_directory_chain(
    anchor_fd: int,
    base: Path,
    identities: list[tuple[str, os.stat_result]],
) -> None:
    """Rewalk the named chain and reject replacement after a child was opened."""
    current_fd = os.dup(anchor_fd)
    current_path = base
    try:
        for component, expected in identities:
            named = _named_stat(current_fd, component)
            child_path = current_path / component
            if named is None:
                raise OSError(f"durable directory disappeared: {child_path}")
            child_fd = os.open(
                component,
                _directory_flags(),
                dir_fd=current_fd,
            )
            try:
                opened = os.fstat(child_fd)
                _require_directory_identity(named, opened, child_path)
                if not _same_inode(expected, opened):
                    raise OSError(
                        f"durable directory identity changed: {child_path}"
                    )
                named_after = _named_stat(current_fd, component)
                if named_after is None:
                    raise OSError(f"durable directory disappeared: {child_path}")
                _require_directory_identity(named_after, opened, child_path)
            except BaseException:
                os.close(child_fd)
                raise
            os.close(current_fd)
            current_fd = child_fd
            current_path = child_path
    finally:
        os.close(current_fd)


@contextmanager
def _anchored_directory_descriptor(
    anchor_fd: int,
    base: Path,
    components: tuple[str, ...],
    *,
    create: bool,
    mode: int,
) -> Iterator[int]:
    """Walk a directory chain with openat and retain every expected identity."""
    current_fd = os.dup(anchor_fd)
    current_path = base
    identities: list[tuple[str, os.stat_result]] = []
    try:
        for component in components:
            if component in ("", ".", ".."):
                raise ValueError(
                    f"durable path has an unsafe component: "
                    f"{base.joinpath(*components)}"
                )
            child_fd, opened = _open_or_create_directory_at(
                current_fd,
                component,
                current_path,
                create=create,
                mode=mode,
            )
            identities.append((component, opened))
            os.close(current_fd)
            current_fd = child_fd
            current_path /= component
        try:
            yield current_fd
        finally:
            _revalidate_directory_chain(anchor_fd, base, identities)
    finally:
        os.close(current_fd)


@contextmanager
def _anchored_parent_descriptor(
    target: str | Path,
    *,
    anchor: str | Path,
    create: bool,
) -> Iterator[tuple[int, str, Path]]:
    """Open a target's parent one component at a time below an opened anchor."""
    with _anchored_directory_target(_absolute(target), anchor) as (
        target_path,
        base,
        anchor_fd,
    ):
        relative = target_path.relative_to(base)
        if not relative.parts:
            raise ValueError(f"durable file target names a directory: {target_path}")
        with _anchored_directory_descriptor(
            anchor_fd,
            base,
            relative.parts[:-1],
            create=create,
            mode=0o700,
        ) as parent_fd:
            yield parent_fd, relative.parts[-1], target_path


def _read_regular_at(
    directory_fd: int,
    name: str,
    path: Path,
    *,
    missing_ok: bool,
) -> tuple[bytes | None, os.stat_result | None]:
    named_before = _validate_named_regular(directory_fd, name, path)
    if named_before is None:
        if missing_ok:
            return None, None
        raise FileNotFoundError(path)
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        opened_before = os.fstat(descriptor)
        _require_regular_identity(named_before, opened_before, path)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            not stat.S_ISREG(opened_after.st_mode)
            or opened_after.st_nlink != 1
            or any(
                getattr(opened_before, field) != getattr(opened_after, field)
                for field in stable
            )
        ):
            raise OSError(f"durable file changed while reading: {path}")
        named_after = _validate_named_regular(directory_fd, name, path)
        if named_after is None:
            raise OSError(f"durable file disappeared while reading: {path}")
        _require_regular_identity(named_after, opened_after, path)
        if any(
            getattr(opened_after, field) != getattr(named_after, field)
            for field in stable
        ):
            raise OSError(f"durable file name changed while reading: {path}")
        return b"".join(chunks), opened_after
    finally:
        os.close(descriptor)


def anchored_read_bytes(
    path: str | Path,
    *,
    anchor: str | Path,
    missing_ok: bool = False,
) -> bytes | None:
    """Read one unlinked regular file through a directory-fd anchored path."""
    try:
        with _anchored_parent_descriptor(path, anchor=anchor, create=False) as (
            directory_fd,
            name,
            target,
        ):
            data, _metadata = _read_regular_at(
                directory_fd,
                name,
                target,
                missing_ok=missing_ok,
            )
            return data
    except FileNotFoundError:
        if missing_ok:
            return None
        raise


def anchored_file_exists(path: str | Path, *, anchor: str | Path) -> bool:
    """Return existence only after rejecting links and non-regular targets."""
    return anchored_read_bytes(path, anchor=anchor, missing_ok=True) is not None


def anchored_unlink_file(
    path: str | Path,
    *,
    anchor: str | Path,
    missing_ok: bool = False,
) -> bool:
    """Unlink one anchored regular file without following or deleting links.

    Returns ``True`` when a file was removed and ``False`` only when the final
    name was already absent and ``missing_ok`` is set. Missing or replaced
    parent directories remain errors.
    """
    with _anchored_parent_descriptor(path, anchor=anchor, create=False) as (
        directory_fd,
        name,
        target,
    ):
        named = _validate_named_regular(directory_fd, name, target)
        if named is None:
            if missing_ok:
                return False
            raise FileNotFoundError(target)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            opened = os.fstat(descriptor)
            _require_regular_identity(named, opened, target)
            named_after = _validate_named_regular(directory_fd, name, target)
            if named_after is None or not _same_file_version(named_after, opened):
                raise OSError(f"durable file identity changed before unlink: {target}")
            os.unlink(name, dir_fd=directory_fd)
            unlinked = os.fstat(descriptor)
            if (
                not stat.S_ISREG(unlinked.st_mode)
                or unlinked.st_nlink != 0
                or not _same_inode(unlinked, opened)
            ):
                raise OSError(f"durable file unlink was not isolated: {target}")
            os.fsync(directory_fd)
            if _named_stat(directory_fd, name) is not None:
                raise OSError(f"durable file name survived unlink: {target}")
        finally:
            os.close(descriptor)
        return True


def anchored_unlink_file_if_bytes(
    path: str | Path,
    *,
    anchor: str | Path,
    expected_current: tuple[bytes, ...],
    missing_ok: bool = False,
) -> bool:
    """Unlink only when the current complete bytes match an allowed value.

    The content read, file-version comparison, and unlink all use one anchored
    parent-directory descriptor. This closes the gap between a caller's
    earlier validation and the destructive operation: a file replaced or
    edited after that validation is preserved and reported as an error.
    """
    with _anchored_parent_descriptor(path, anchor=anchor, create=False) as (
        directory_fd,
        name,
        target,
    ):
        current, expected = _read_regular_at(
            directory_fd,
            name,
            target,
            missing_ok=True,
        )
        if current is None:
            if missing_ok:
                return False
            raise FileNotFoundError(target)
        if not any(current == allowed for allowed in expected_current):
            raise OSError(f"durable file content changed before unlink: {target}")
        assert expected is not None
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            opened = os.fstat(descriptor)
            _require_regular_identity(expected, opened, target)
            if not _same_file_version(expected, opened):
                raise OSError(
                    f"durable file version changed before content-CAS unlink: {target}"
                )
            named_after = _validate_named_regular(directory_fd, name, target)
            if named_after is None or not _same_file_version(named_after, expected):
                raise OSError(
                    f"durable file identity changed before content-CAS unlink: {target}"
                )
            os.unlink(name, dir_fd=directory_fd)
            unlinked = os.fstat(descriptor)
            if (
                not stat.S_ISREG(unlinked.st_mode)
                or unlinked.st_nlink != 0
                or not _same_inode(unlinked, opened)
            ):
                raise OSError(f"durable file unlink was not isolated: {target}")
            os.fsync(directory_fd)
            if _named_stat(directory_fd, name) is not None:
                raise OSError(f"durable file name survived unlink: {target}")
        finally:
            os.close(descriptor)
        return True


def atomic_replace_temp_target_name(name: str) -> str | None:
    """Return the bound target for an exact atomic-replace temporary name."""
    match = _ATOMIC_REPLACE_TEMP.fullmatch(name)
    if match is None:
        return None
    target = match.group("target")
    if target in ("", ".", "..") or Path(target).name != target or "\x00" in target:
        return None
    return target


def inspect_anchored_atomic_replace_temp(
    path: str | Path,
    *,
    anchor: str | Path,
    expected_target_name: str,
    owner_uid: int,
    mode: int = 0o600,
) -> AnchoredAtomicReplaceTemp:
    """Validate a crash-left temporary file without changing it.

    The caller must separately prove that ``expected_target_name`` is a valid
    evidence target.  This helper binds the exact temporary name to that target
    and rejects links, ownership drift, permissive modes, and in-place writes.
    """
    with _anchored_parent_descriptor(path, anchor=anchor, create=False) as (
        directory_fd,
        name,
        target,
    ):
        if atomic_replace_temp_target_name(name) != expected_target_name:
            raise OSError(f"atomic-replace temporary name is invalid: {target}")
        data, metadata = _read_regular_at(
            directory_fd,
            name,
            target,
            missing_ok=False,
        )
        assert data is not None and metadata is not None
        if metadata.st_uid != owner_uid:
            raise OSError(f"atomic-replace temporary owner is unsafe: {target}")
        if stat.S_IMODE(metadata.st_mode) != mode:
            raise OSError(f"atomic-replace temporary mode is unsafe: {target}")
        return AnchoredAtomicReplaceTemp(
            path=target,
            target_name=expected_target_name,
            data=data,
            metadata=metadata,
        )


def remove_anchored_atomic_replace_temp(
    record: AnchoredAtomicReplaceTemp,
    *,
    anchor: str | Path,
    owner_uid: int,
    mode: int = 0o600,
) -> None:
    """Remove one previously validated temporary file and sync its directory.

    Recovery callers serialize this operation with their run lock.  Re-reading
    the file and comparing its complete version prevents a changed candidate
    from being reclassified between the validation and removal phases.
    """
    current = inspect_anchored_atomic_replace_temp(
        record.path,
        anchor=anchor,
        expected_target_name=record.target_name,
        owner_uid=owner_uid,
        mode=mode,
    )
    if current.data != record.data or not _same_file_version(
        current.metadata,
        record.metadata,
    ):
        raise OSError(
            f"atomic-replace temporary file changed before removal: {record.path}"
        )

    with _anchored_parent_descriptor(
        record.path,
        anchor=anchor,
        create=False,
    ) as (directory_fd, name, target):
        named = _validate_named_regular(directory_fd, name, target)
        if named is None or not _same_file_version(named, record.metadata):
            raise OSError(
                f"atomic-replace temporary identity changed before removal: {target}"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            opened = os.fstat(descriptor)
            _require_regular_identity(named, opened, target)
            if not _same_file_version(opened, record.metadata):
                raise OSError(
                    f"atomic-replace temporary version changed before removal: {target}"
                )
            os.unlink(name, dir_fd=directory_fd)
            unlinked = os.fstat(descriptor)
            if (
                not stat.S_ISREG(unlinked.st_mode)
                or unlinked.st_nlink != 0
                or not _same_inode(unlinked, opened)
            ):
                raise OSError(
                    f"atomic-replace temporary unlink was not isolated: {target}"
                )
            os.fsync(directory_fd)
            if _named_stat(directory_fd, name) is not None:
                raise OSError(
                    f"atomic-replace temporary name survived removal: {target}"
                )
        finally:
            os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("durable write made no progress")
        written += count


def _replace_at(
    directory_fd: int,
    name: str,
    path: Path,
    data: bytes,
    *,
    expected: os.stat_result | None,
    mode: int | None,
) -> None:
    selected_mode = (
        mode
        if mode is not None
        else stat.S_IMODE(expected.st_mode) if expected is not None else 0o600
    )
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        selected_mode,
        dir_fd=directory_fd,
    )
    temporary_exists = True
    try:
        os.fchmod(descriptor, selected_mode)
        _write_all(descriptor, data)
        os.fsync(descriptor)
        temporary_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(temporary_metadata.st_mode)
            or temporary_metadata.st_nlink != 1
        ):
            raise OSError(f"temporary durable file is unsafe: {path}")

        current = _validate_named_regular(directory_fd, name, path)
        if (expected is None) != (current is None) or (
            expected is not None
            and current is not None
            and not _same_file_version(expected, current)
        ):
            raise OSError(f"durable file identity changed before replace: {path}")

        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_exists = False
        os.fsync(directory_fd)
        named_after = _validate_named_regular(directory_fd, name, path)
        if named_after is None or not _same_inode(named_after, temporary_metadata):
            raise OSError(f"durable file identity changed after replace: {path}")
        verify_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            _require_regular_identity(named_after, os.fstat(verify_fd), path)
            os.fsync(verify_fd)
        finally:
            os.close(verify_fd)
    finally:
        os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            else:
                os.fsync(directory_fd)


def anchored_atomic_replace_bytes(
    path: str | Path,
    data: bytes,
    *,
    anchor: str | Path,
    mode: int | None = None,
) -> None:
    """Atomically replace one anchored file without touching its old inode."""
    with _anchored_parent_descriptor(path, anchor=anchor, create=True) as (
        directory_fd,
        name,
        target,
    ):
        expected = _validate_named_regular(directory_fd, name, target)
        _replace_at(
            directory_fd,
            name,
            target,
            data,
            expected=expected,
            mode=mode,
        )


def anchored_replace_bytes_if_current(
    path: str | Path,
    data: bytes,
    *,
    anchor: str | Path,
    expected_current: tuple[bytes, ...],
    expected_missing: bool = False,
    mode: int | None = None,
) -> None:
    """Replace only when current bytes, or permitted absence, match.

    ``_read_regular_at`` returns the exact file version associated with the
    accepted bytes. ``_replace_at`` then compares that version again in the
    same parent-directory descriptor immediately before ``os.replace``. When
    absence is accepted, exclusive parent traversal plus the same version check
    prevents a file raced into the missing name from being overwritten.
    """
    with _anchored_parent_descriptor(
        path,
        anchor=anchor,
        create=expected_missing,
    ) as (
        directory_fd,
        name,
        target,
    ):
        current, expected = _read_regular_at(
            directory_fd,
            name,
            target,
            missing_ok=expected_missing,
        )
        if current is None:
            if not expected_missing:
                raise FileNotFoundError(target)
        elif not any(current == allowed for allowed in expected_current):
            raise OSError(f"durable file content changed before replace: {target}")
        _replace_at(
            directory_fd,
            name,
            target,
            data,
            expected=expected,
            mode=mode,
        )


def anchored_update_bytes(
    path: str | Path,
    update: Callable[[bytes | None], bytes],
    *,
    anchor: str | Path,
    mode: int | None = None,
) -> None:
    """Read/transform/replace one file while retaining its directory identity.

    Authoritative logs still serialize writers with their run lock. The
    pre-replace file-version check is an additional fail-closed barrier against
    an in-place writer that ignores that lock.
    """
    with _anchored_parent_descriptor(path, anchor=anchor, create=True) as (
        directory_fd,
        name,
        target,
    ):
        current, expected = _read_regular_at(
            directory_fd,
            name,
            target,
            missing_ok=True,
        )
        _replace_at(
            directory_fd,
            name,
            target,
            update(current),
            expected=expected,
            mode=mode,
        )


def anchored_write_once_bytes(
    path: str | Path,
    data: bytes,
    *,
    anchor: str | Path,
    mode: int = 0o600,
) -> bool:
    """Create immutable evidence once, or verify the exact existing bytes."""
    with _anchored_parent_descriptor(path, anchor=anchor, create=True) as (
        directory_fd,
        name,
        target,
    ):
        current, expected = _read_regular_at(
            directory_fd,
            name,
            target,
            missing_ok=True,
        )
        if current is not None:
            if current != data:
                raise OSError(f"durable write-once file changed: {target}")
            return False
        assert expected is None
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            raced, _metadata = _read_regular_at(
                directory_fd,
                name,
                target,
                missing_ok=False,
            )
            if raced != data:
                raise OSError(f"durable write-once file changed: {target}") from None
            return False
        try:
            os.fchmod(descriptor, mode)
            _write_all(descriptor, data)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            named = _validate_named_regular(directory_fd, name, target)
            if named is None:
                raise OSError(f"durable write-once file disappeared: {target}")
            _require_regular_identity(named, opened, target)
            os.fsync(directory_fd)
        finally:
            os.close(descriptor)
        return True


def anchored_open_lock_file(
    path: str | Path,
    *,
    anchor: str | Path,
    mode: int = 0o600,
) -> int:
    """Open/create a link-free regular lock inode without writing its contents."""
    with _anchored_parent_descriptor(path, anchor=anchor, create=True) as (
        directory_fd,
        name,
        target,
    ):
        named = _validate_named_regular(directory_fd, name, target)
        created = False
        if named is None:
            try:
                descriptor = os.open(
                    name,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    mode,
                    dir_fd=directory_fd,
                )
                created = True
            except FileExistsError:
                named = _validate_named_regular(directory_fd, name, target)
                if named is None:
                    raise OSError(f"durable lock identity is unstable: {target}")
                descriptor = os.open(
                    name,
                    os.O_RDWR
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
        else:
            descriptor = os.open(
                name,
                os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        try:
            opened = os.fstat(descriptor)
            named_after = _validate_named_regular(directory_fd, name, target)
            if named_after is None:
                raise OSError(f"durable lock disappeared while opening: {target}")
            _require_regular_identity(named_after, opened, target)
            if named is not None and not _same_inode(named, opened):
                raise OSError(f"durable lock identity changed while opening: {target}")
            if created:
                os.fsync(descriptor)
                os.fsync(directory_fd)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor


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
        with _anchored_directory_target(target, anchor) as (
            anchored_target,
            base,
            anchor_fd,
        ):
            parts = anchored_target.relative_to(base).parts
            with _anchored_directory_descriptor(
                anchor_fd,
                base,
                parts,
                create=True,
                mode=mode,
            ):
                pass
        return anchored_target
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
    if anchor is not None:
        anchored_atomic_replace_bytes(
            target,
            data,
            anchor=anchor,
            mode=mode,
        )
        return
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
