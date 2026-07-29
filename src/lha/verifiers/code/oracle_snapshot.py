"""Byte-exact snapshots of the files that determine code-verifier outcomes."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_IGNORED_FILES = frozenset({".coverage", ".lha_pytest.json"})
_READ_SIZE = 1024 * 1024


class OracleSnapshotError(ValueError):
    """The protected test/configuration surface could not be inspected safely."""


@dataclass(frozen=True)
class OracleFile:
    sha256: str
    mode: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class OracleSnapshot:
    files: dict[str, OracleFile]
    sha256: str

    def difference(self, other: "OracleSnapshot") -> str | None:
        before = set(self.files)
        after = set(other.files)
        removed = sorted(before - after)
        added = sorted(after - before)
        changed = sorted(
            path for path in before & after if self.files[path] != other.files[path]
        )
        parts: list[str] = []
        if removed:
            parts.append(f"removed: {', '.join(removed[:5])}")
        if added:
            parts.append(f"added: {', '.join(added[:5])}")
        if changed:
            parts.append(f"changed: {', '.join(changed[:5])}")
        return "; ".join(parts) or None


def capture_oracle_snapshot(workdir: Path) -> OracleSnapshot:
    """Hash every pre-existing repository file without following aliases.

    Pytest imports repository code while collecting tests. That code must not be
    able to shrink or rewrite a custom-collected oracle before the execution
    phase. Looking only for ``test_*.py`` is insufficient when a repository uses
    custom collection rules. Expected cache/build directories are excluded;
    every other add, remove, rewrite, inode swap, or metadata change is rejected.
    """
    root = workdir
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise OracleSnapshotError("pytest workdir cannot be inspected") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise OracleSnapshotError("pytest workdir must be a real directory")

    files: dict[str, OracleFile] = {}

    def walk_error(error: OSError) -> None:
        raise OracleSnapshotError("repository tree cannot be inspected") from error

    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=walk_error,
    ):
        current = Path(directory)
        directory_names[:] = sorted(
            name for name in directory_names if name not in _IGNORED_DIRECTORIES
        )
        for name in tuple(directory_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            try:
                metadata = path.lstat()
            except OSError as error:
                raise OracleSnapshotError(
                    f"repository directory changed during inspection: {relative}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise OracleSnapshotError(
                    f"repository directory is a symbolic link: {relative}"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise OracleSnapshotError(
                    f"repository directory is not a directory: {relative}"
                )
        for name in sorted(file_names):
            if (
                name in _IGNORED_FILES
                or name.startswith(".coverage.")
                or (
                    name.startswith(".lha-scorer-")
                    and name.endswith(".json")
                )
            ):
                continue
            path = current / name
            relative = path.relative_to(root).as_posix()
            files[relative] = _read_oracle_file(path, relative=relative)

    payload = json.dumps(
        [
            {
                "path": path,
                "sha256": files[path].sha256,
                "mode": files[path].mode,
                "device": files[path].device,
                "inode": files[path].inode,
                "mtime_ns": files[path].mtime_ns,
                "ctime_ns": files[path].ctime_ns,
            }
            for path in sorted(files)
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return OracleSnapshot(files=files, sha256=hashlib.sha256(payload).hexdigest())


def _read_oracle_file(path: Path, *, relative: str) -> OracleFile:
    try:
        path_metadata = path.lstat()
    except OSError as error:
        raise OracleSnapshotError(
            f"protected file changed during inspection: {relative}"
        ) from error
    if stat.S_ISLNK(path_metadata.st_mode):
        raise OracleSnapshotError(f"protected file is a symbolic link: {relative}")
    if not stat.S_ISREG(path_metadata.st_mode):
        raise OracleSnapshotError(f"protected path is not a regular file: {relative}")
    if path_metadata.st_nlink != 1:
        raise OracleSnapshotError(f"protected file has a hard-link alias: {relative}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OracleSnapshotError(f"protected file cannot be opened: {relative}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise OracleSnapshotError(
                f"protected file changed during inspection: {relative}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _READ_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            after.st_nlink != 1
            or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        ):
            raise OracleSnapshotError(
                f"protected file changed during inspection: {relative}"
            )
    except OSError as error:
        raise OracleSnapshotError(f"protected file cannot be read: {relative}") from error
    finally:
        os.close(descriptor)
    return OracleFile(
        sha256=digest.hexdigest(),
        mode=stat.S_IMODE(after.st_mode),
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )
