"""Trusted baseline inventory for repository-local Pytest oracles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .oracle_models import OracleInventoryFile, PytestOracleInventory
from .pytest_evidence import InventoryResult, collect_inventory
from .sandbox import ExecutionBackend, ProcessCleanupUnconfirmed
from .tools.policy import PytestCollectionConfig, discover_pytest_configuration

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


class OracleInventoryError(ValueError):
    """A baseline test inventory could not be established safely."""


@dataclass(frozen=True)
class _TreeFile:
    sha256: str
    mode: int
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _DisposableCollection:
    canonical_files: dict[str, _TreeFile]
    config: PytestCollectionConfig
    inventory: InventoryResult


def build_pytest_oracle_inventory(
    workdir: str | Path,
    backend: ExecutionBackend,
    *,
    timeout: float = 300.0,
) -> PytestOracleInventory:
    """Collect pristine tests and bind their complete local support tree.

    Collection uses the same isolated driver as the code verifier, with plugin
    autoload disabled. In addition to files named by actual node IDs, every
    pre-existing file below configured ``testpaths`` is protected. The latter
    includes helpers, fixtures, and data that do not themselves match
    ``python_files``. A whole-tree snapshot around collection rejects import
    side effects that try to change the baseline before it is recorded.
    """
    baseline = _collect_from_disposable_copy(
        Path(workdir),
        backend,
        timeout=timeout,
    )
    before = baseline.canonical_files
    config = baseline.config
    collected = baseline.inventory
    if collected.driver.cleanup_unconfirmed:
        raise ProcessCleanupUnconfirmed(
            collected.driver.cleanup_detail
            or "pytest baseline collection cleanup could not be confirmed"
        )
    if not collected.ready:
        detail = collected.driver.detail or "pytest baseline collection failed"
        raise OracleInventoryError(detail)

    selected: set[str] = set()
    support_roots: set[str] = set()
    for nodeid in collected.expected_nodeids:
        relative = _nodeid_path(nodeid)
        if relative not in before:
            raise OracleInventoryError(
                f"pytest collected a path outside the pristine file inventory: {relative}"
            )
        selected.add(relative)
        if not any(_within(relative, testpath) for testpath in config.testpaths):
            support_roots.add(PurePosixPath(relative).parent.as_posix())

    protected_roots = tuple(
        dict.fromkeys((*config.testpaths, *sorted(support_roots)))
    )
    for relative in before:
        if any(_within(relative, root) for root in protected_roots):
            selected.add(relative)

    files = tuple(
        OracleInventoryFile(path=relative, sha256=before[relative].sha256)
        for relative in sorted(selected)
    )
    payload = {
        "files": [item.model_dump(mode="json") for item in files],
        "nodeids": list(collected.expected_nodeids),
        "configured_testpaths": list(config.testpaths),
        "support_roots": sorted(support_roots),
        "collection_receipt_sha256": collected.driver.receipt_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PytestOracleInventory(
        files=files,
        nodeids=collected.expected_nodeids,
        configured_testpaths=config.testpaths,
        support_roots=tuple(sorted(support_roots)),
        collection_receipt_sha256=collected.driver.receipt_sha256,
        sha256=digest,
    )


def collect_pytest_inventory_disposable(
    workdir: str | Path,
    backend: ExecutionBackend,
    *,
    timeout: float = 300.0,
) -> InventoryResult:
    """Collect Pytest node IDs without importing target code in the caller's tree."""
    return _collect_from_disposable_copy(
        Path(workdir),
        backend,
        timeout=timeout,
    ).inventory


def validate_pytest_oracle_inventory(
    workdir: str | Path,
    inventory: PytestOracleInventory,
    *,
    allowed_changes: tuple[str, ...] = (),
) -> None:
    """Require the saved path/content bindings to match the current worktree."""
    try:
        current = _snapshot_tree(Path(workdir))
    except OracleInventoryError as error:
        raise OracleInventoryError(str(error)) from error
    allowed = set(allowed_changes)
    expected = {
        item.path: item.sha256
        for item in inventory.files
        if item.path not in allowed
    }
    selected = {
        relative: record.sha256
        for relative, record in current.items()
        if relative not in allowed
        and (
            relative in expected
            or any(
                _within(relative, protected_root)
                for protected_root in inventory.protected_roots
            )
        )
    }
    if selected != expected:
        missing = sorted(set(expected) - set(selected))
        added = sorted(set(selected) - set(expected))
        changed = sorted(
            path
            for path in set(expected) & set(selected)
            if expected[path] != selected[path]
        )
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing[:5])}")
        if added:
            detail.append(f"added: {', '.join(added[:5])}")
        if changed:
            detail.append(f"changed: {', '.join(changed[:5])}")
        raise OracleInventoryError(
            "pytest oracle inventory no longer matches the worktree"
            + (f" ({'; '.join(detail)})" if detail else "")
        )


def _collect_from_disposable_copy(
    root: Path,
    backend: ExecutionBackend,
    *,
    timeout: float,
) -> _DisposableCollection:
    """Collect in a copy while binding the result to canonical pre-collection bytes."""
    try:
        canonical_before = _snapshot_tree(root)
        config = discover_pytest_configuration(root)
    except (OracleInventoryError, ValueError) as error:
        raise OracleInventoryError(str(error)) from error

    scratch = Path(tempfile.mkdtemp(prefix="lha-pytest-inventory-"))
    disposable = scratch / "workdir"
    cleanup_confirmed = True
    try:
        shutil.copytree(
            root,
            disposable,
            symlinks=True,
            ignore=_copy_ignore,
        )
        canonical_after_copy = _snapshot_tree(root)
        if difference := _snapshot_difference(
            canonical_before,
            canonical_after_copy,
        ):
            raise OracleInventoryError(
                f"pytest worktree changed while its baseline was copied ({difference})"
            )
        disposable_before = _snapshot_tree(disposable)
        if difference := _logical_snapshot_difference(
            canonical_before,
            disposable_before,
        ):
            raise OracleInventoryError(
                f"pytest disposable baseline differs from the worktree ({difference})"
            )

        # An exception at this boundary could leave an unconfirmed descendant.
        # Retain only the disposable copy; the canonical run worktree stays clean.
        cleanup_confirmed = False
        collected = collect_inventory(
            disposable,
            backend,
            timeout=timeout,
            autoload_plugins=False,
        )
        cleanup_confirmed = not collected.driver.cleanup_unconfirmed
        if collected.driver.cleanup_unconfirmed:
            raise ProcessCleanupUnconfirmed(
                collected.driver.cleanup_detail
                or "pytest baseline collection cleanup could not be confirmed"
            )

        disposable_after = _snapshot_tree(disposable)
        if difference := _snapshot_difference(
            disposable_before,
            disposable_after,
        ):
            raise OracleInventoryError(
                f"pytest baseline collection changed repository files ({difference})"
            )
        canonical_after_collection = _snapshot_tree(root)
        if difference := _snapshot_difference(
            canonical_before,
            canonical_after_collection,
        ):
            raise OracleInventoryError(
                f"canonical pytest worktree changed during collection ({difference})"
            )
        return _DisposableCollection(
            canonical_files=canonical_before,
            config=config,
            inventory=collected,
        )
    finally:
        if cleanup_confirmed:
            try:
                shutil.rmtree(scratch)
            except OSError as error:
                raise OracleInventoryError(
                    "pytest disposable collection worktree could not be removed"
                ) from error


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if (
            name in _IGNORED_DIRECTORIES
            or name in _IGNORED_FILES
            or name.startswith(".coverage.")
            or (name.startswith(".lha-scorer-") and name.endswith(".json"))
        ):
            ignored.add(name)
    return ignored


def _logical_snapshot_difference(
    before: dict[str, _TreeFile],
    after: dict[str, _TreeFile],
) -> str | None:
    before_paths = set(before)
    after_paths = set(after)
    removed = sorted(before_paths - after_paths)
    added = sorted(after_paths - before_paths)
    changed = sorted(
        path
        for path in before_paths & after_paths
        if (
            before[path].sha256,
            before[path].mode,
            before[path].size,
        )
        != (
            after[path].sha256,
            after[path].mode,
            after[path].size,
        )
    )
    detail = []
    if removed:
        detail.append(f"removed: {', '.join(removed[:5])}")
    if added:
        detail.append(f"added: {', '.join(added[:5])}")
    if changed:
        detail.append(f"changed: {', '.join(changed[:5])}")
    return "; ".join(detail) or None


def _within(relative: str, root: str) -> bool:
    return (
        root == "."
        or relative == root
        or relative.startswith(f"{root}/")
    )


def _snapshot_tree(root: Path) -> dict[str, _TreeFile]:
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise OracleInventoryError("pytest worktree cannot be inspected") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise OracleInventoryError("pytest worktree must be a real directory")

    files: dict[str, _TreeFile] = {}

    def walk_error(error: OSError) -> None:
        raise OracleInventoryError("repository tree cannot be inspected") from error

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
                raise OracleInventoryError(
                    f"repository directory changed during inspection: {relative}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise OracleInventoryError(
                    f"repository directory is unsafe: {relative}"
                )
        for name in sorted(file_names):
            if (
                name in _IGNORED_FILES
                or name.startswith(".coverage.")
                or (name.startswith(".lha-scorer-") and name.endswith(".json"))
            ):
                continue
            path = current / name
            relative = path.relative_to(root).as_posix()
            files[relative] = _read_tree_file(path, relative)
    return files


def _read_tree_file(path: Path, relative: str) -> _TreeFile:
    try:
        path_metadata = path.lstat()
    except OSError as error:
        raise OracleInventoryError(
            f"repository file changed during inspection: {relative}"
        ) from error
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or not stat.S_ISREG(path_metadata.st_mode)
        or path_metadata.st_nlink != 1
    ):
        raise OracleInventoryError(f"repository file is unsafe: {relative}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OracleInventoryError(
            f"repository file cannot be opened: {relative}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise OracleInventoryError(
                f"repository file changed during inspection: {relative}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _READ_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            after.st_nlink != 1
            or any(getattr(before, field) != getattr(after, field) for field in stable)
        ):
            raise OracleInventoryError(
                f"repository file changed during inspection: {relative}"
            )
    except OSError as error:
        raise OracleInventoryError(
            f"repository file cannot be read: {relative}"
        ) from error
    finally:
        os.close(descriptor)
    return _TreeFile(
        sha256=digest.hexdigest(),
        mode=stat.S_IMODE(after.st_mode),
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def _snapshot_difference(
    before: dict[str, _TreeFile],
    after: dict[str, _TreeFile],
) -> str | None:
    before_paths = set(before)
    after_paths = set(after)
    removed = sorted(before_paths - after_paths)
    added = sorted(after_paths - before_paths)
    changed = sorted(
        path for path in before_paths & after_paths if before[path] != after[path]
    )
    detail = []
    if removed:
        detail.append(f"removed: {', '.join(removed[:5])}")
    if added:
        detail.append(f"added: {', '.join(added[:5])}")
    if changed:
        detail.append(f"changed: {', '.join(changed[:5])}")
    return "; ".join(detail) or None


def _nodeid_path(nodeid: str) -> str:
    raw = nodeid.split("::", 1)[0]
    if not raw or "\\" in raw or "\x00" in raw:
        raise OracleInventoryError(f"pytest returned an unsafe node ID: {nodeid!r}")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw
    ):
        raise OracleInventoryError(f"pytest returned an unsafe node ID: {nodeid!r}")
    return path.as_posix()
