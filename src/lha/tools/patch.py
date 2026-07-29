"""Apply or revert patches inside a run worktree.

A patch may carry complete ``file_contents`` or a ``unified_diff`` applied with
``git apply``. Originals are snapshotted so a failed verification can be
reverted.
"""

from __future__ import annotations

import base64
import binascii
import difflib
import hashlib
import json
import os
import stat
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..artifacts import Patch
from ..durable_io import (
    anchored_atomic_replace_bytes,
    anchored_read_bytes,
    atomic_replace_bytes,
    fsync_directory,
    sync_regular_file,
)
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
        try:
            # Keep the stricter redundant-header checks for ordinary paths.
            # The legacy parser cannot tokenize Git's valid unquoted spaces,
            # so exact machine output remains authoritative for those headers.
            diff_paths(patch.unified_diff)
        except ValueError as error:
            message = str(error)
            space_tokenization_errors = (
                "malformed diff --git header:",
                "malformed rename from header:",
                "malformed rename to header:",
                "malformed copy from header:",
                "malformed copy to header:",
            )
            if not message.startswith(space_tokenization_errors):
                raise
        raw_paths = _git_machine_diff_paths(patch.unified_diff)
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


def _decode_git_c_path(value: str) -> str:
    """Decode one whole Git pathname without splitting on human whitespace."""
    if not value:
        raise ValueError("empty Git path")
    if not value.startswith('"'):
        return value
    if len(value) < 2 or not value.endswith('"'):
        raise ValueError(f"malformed quoted Git path: {value!r}")
    encoded = value[1:-1]
    decoded = bytearray()
    escapes = {
        "a": b"\a",
        "b": b"\b",
        "t": b"\t",
        "n": b"\n",
        "v": b"\v",
        "f": b"\f",
        "r": b"\r",
        '"': b'"',
        "\\": b"\\",
    }
    index = 0
    while index < len(encoded):
        char = encoded[index]
        if char != "\\":
            decoded.extend(char.encode("utf-8"))
            index += 1
            continue
        index += 1
        if index >= len(encoded):
            raise ValueError("trailing backslash in a quoted Git path")
        escaped = encoded[index]
        if escaped in escapes:
            decoded.extend(escapes[escaped])
            index += 1
            continue
        if escaped in "01234567":
            end = index + 1
            while end < min(index + 3, len(encoded)) and encoded[end] in "01234567":
                end += 1
            decoded.append(int(encoded[index:end], 8))
            index = end
            continue
        raise ValueError(f"unsupported escape in a quoted Git path: \\{escaped}")
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("quoted Git path is not valid UTF-8") from error


def _extended_git_paths(unified_diff: str, machine_paths: set[str]) -> set[str]:
    """Add rename/copy sources that ``git apply --numstat`` does not emit."""
    paths: set[str] = set()
    current: dict[str, str] | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        rename = ("rename_from", "rename_to")
        copy = ("copy_from", "copy_to")
        for source_key, destination_key in (rename, copy):
            source = current.get(source_key)
            destination = current.get(destination_key)
            if (source is None) != (destination is None):
                raise ValueError("unified diff has an incomplete rename/copy header pair")
            if source is None or destination is None:
                continue
            if destination not in machine_paths:
                raise ValueError(
                    "Git machine output disagrees with rename/copy destination: "
                    f"{destination!r}"
                )
            paths.update((source, destination))
        if all(key in current for key in (*rename, *copy)):
            raise ValueError("unified diff cannot mix rename and copy headers")
        current = None

    prefixes = (
        ("rename from ", "rename_from"),
        ("rename to ", "rename_to"),
        ("copy from ", "copy_from"),
        ("copy to ", "copy_to"),
    )
    for line in unified_diff.splitlines():
        if line.startswith("diff --git "):
            finish()
            current = {}
            continue
        for prefix, key in prefixes:
            if not line.startswith(prefix):
                continue
            if current is None:
                raise ValueError(f"{prefix.rstrip()} appears outside diff --git")
            if key in current:
                raise ValueError(f"duplicate {prefix.rstrip()} header")
            decoded = _decode_git_c_path(line[len(prefix) :])
            if decoded == "/dev/null":
                raise ValueError(f"{prefix.rstrip()} cannot name /dev/null")
            current[key] = decoded
            break
    finish()
    return paths


def _parse_git_numstat(output: str) -> list[str]:
    if "\ufffd" in output or not output.endswith("\x00"):
        raise ValueError("Git numstat output is not lossless NUL-delimited UTF-8")
    paths: list[str] = []
    for record in output[:-1].split("\x00"):
        added, separator, remainder = record.partition("\t")
        removed, second_separator, path = remainder.partition("\t")
        if (
            not separator
            or not second_separator
            or not path
            or not (added == "-" or added.isdecimal())
            or not (removed == "-" or removed.isdecimal())
        ):
            raise ValueError("malformed Git numstat record")
        paths.append(path)
    return paths


def _git_machine_diff_paths(unified_diff: str) -> list[str]:
    """Ask Git for exact NUL-delimited target names, then add operation sources."""
    with tempfile.TemporaryDirectory(prefix="lha-diff-paths-") as temporary:
        result = _run_git_control(
            unified_diff,
            Path(temporary),
            options=("-p1", "--numstat", "-z"),
        )
    if not result.ok:
        raise ValueError(
            "Git could not parse unified diff paths: "
            f"{result.stderr or result.stdout}"
        )
    primary = _parse_git_numstat(result.stdout)
    paths = set(primary)
    paths.update(_extended_git_paths(unified_diff, paths))
    return sorted(paths)


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


def _worktree_inode_aliases(
    workdir: Path,
    *,
    device: int,
    inode: int,
) -> list[str]:
    """Find regular-file names for one inode without following directory links."""
    root = workdir.resolve()
    aliases: list[str] = []

    def ignore_error(_error: OSError) -> None:
        # The target's link count is already sufficient to reject the patch.
        # Alias discovery only makes the diagnostic more useful.
        return None

    for directory, dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=ignore_error,
    ):
        base = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if not (base / name).is_symlink()
        ]
        for name in filenames:
            candidate = base / name
            try:
                info = candidate.lstat()
            except OSError:
                continue
            if (
                stat.S_ISREG(info.st_mode)
                and info.st_dev == device
                and info.st_ino == inode
            ):
                aliases.append(candidate.relative_to(root).as_posix())
    return sorted(aliases)


def _validate_patch_targets(paths: list[str], workdir: Path) -> None:
    """Reject links and non-files before any patch bytes can be applied."""
    root = workdir.resolve()
    write_set = set(paths)
    for rel in paths:
        target = _safe_target(root, rel)
        try:
            info = target.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect patch target {rel!r}: {error}") from error
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"patch target is not a regular file: {rel!r}")
        if info.st_nlink == 1:
            continue

        aliases = _worktree_inode_aliases(
            root,
            device=info.st_dev,
            inode=info.st_ino,
        )
        unauthorized = [alias for alias in aliases if alias not in write_set]
        detail = (
            f"; same-inode path outside the write set: {unauthorized[0]!r}"
            if unauthorized
            else ""
        )
        raise ValueError(
            f"refusing patch target with {info.st_nlink} hard links: {rel!r}{detail}"
        )


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
    _validate_patch_targets(paths, root)
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


def _run_git_control(
    diff: str,
    workdir: Path,
    *,
    options: tuple[str, ...],
):
    """Run one fixed Git control-plane command without shell interpretation."""
    from ..sandbox.base import (
        PROCESS_CLEANUP_RETURN_CODE,
        ProcessCleanupUnconfirmed,
        process_group_cleanup_supported,
        run_bounded_process,
        scrub_env,
        terminate_process_group,
    )
    from .shell import sanitized_absolute_path, trusted_executable

    if not process_group_cleanup_supported():
        result = ProcResult(
            PROCESS_CLEANUP_RETURN_CODE,
            "",
            (
                "git apply requires POSIX process-group cleanup; "
                "use Linux, macOS, or WSL2"
            ),
            0.0,
            cleanup_confirmed=False,
            cleanup_detail=(
                "git apply requires POSIX process-group cleanup; "
                "use Linux, macOS, or WSL2"
            ),
        )
        raise ProcessCleanupUnconfirmed(result.cleanup_detail)
    git = trusted_executable("git", require_unwritable=True)
    if git is None:
        return ProcResult(
            127,
            "",
            "no trusted absolute git executable is available",
            0.0,
        )
    safe_path = sanitized_absolute_path(require_unwritable=True)
    environment = scrub_env(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
            "PATH": safe_path,
        }
    )
    argv = [git, "apply", "--whitespace=nowarn", *options]
    argv.append("-")
    result = run_bounded_process(
        argv,
        cwd=workdir,
        timeout=_PATCH_APPLY_TIMEOUT_S,
        output_bytes=_PATCH_APPLY_OUTPUT_BYTES,
        env=environment,
        input=diff,
        start_new_session=True,
        on_exit=terminate_process_group,
    )
    if result.cleanup_unconfirmed:
        raise ProcessCleanupUnconfirmed(
            result.cleanup_detail or result.stderr[-1000:]
        )
    return result


def _run_git_apply(
    diff: str,
    workdir: Path,
    *,
    check: bool = False,
    reverse: bool = False,
):
    options = ["-p1"]
    if check:
        options.append("--check")
    if reverse:
        options.append("--reverse")
    return _run_git_control(diff, workdir, options=tuple(options))


def _worktree_file_state(workdir: Path) -> dict[str, tuple[str, int, str]]:
    """Hash non-directory entries without following links."""
    root = workdir.resolve()
    result: dict[str, tuple[str, int, str]] = {}

    def record(path: Path) -> None:
        try:
            info = path.lstat()
            rel = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISREG(info.st_mode):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                result[rel] = ("file", mode, digest.hexdigest())
            elif stat.S_ISLNK(info.st_mode):
                result[rel] = ("symlink", mode, os.readlink(path))
            else:
                result[rel] = ("other", mode, str(stat.S_IFMT(info.st_mode)))
        except OSError as error:
            raise ValueError(f"cannot inspect worktree entry {path}: {error}") from error

    def walk_error(error: OSError) -> None:
        raise ValueError(f"cannot inspect worktree: {error}") from error

    for directory, dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=walk_error,
    ):
        base = Path(directory)
        linked_dirs = [name for name in dirnames if (base / name).is_symlink()]
        dirnames[:] = [name for name in dirnames if name not in linked_dirs]
        for name in linked_dirs:
            record(base / name)
        for name in filenames:
            record(base / name)
    return result


def _changed_paths(
    before: dict[str, tuple[str, int, str]],
    after: dict[str, tuple[str, int, str]],
) -> set[str]:
    return {
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    }


def _expected_diff_state(
    *,
    patch: Patch,
    resolved: ResolvedPatch,
    backup: Backup,
) -> dict[str, tuple[str, int, str]]:
    """Apply once to a minimal copy to derive the exact expected end state."""
    if set(backup.originals) != set(resolved.paths):
        raise ValueError("patch backup does not match the resolved write set")
    with tempfile.TemporaryDirectory(prefix="lha-patch-check-") as temporary:
        root = Path(temporary)
        for rel in resolved.paths:
            original = backup.originals[rel]
            if original is None:
                continue
            target = _safe_target(root, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original)
            mode = backup.modes.get(rel)
            if mode is None:
                raise ValueError(f"existing backup target has no mode: {rel!r}")
            target.chmod(mode)
        before = _worktree_file_state(root)
        result = _run_git_apply(patch.unified_diff, root)
        if not result.ok:
            raise RuntimeError(
                "git apply failed against snapshotted inputs: "
                f"{result.stderr or result.stdout}"
            )
        after = _worktree_file_state(root)
        unexpected = sorted(_changed_paths(before, after) - set(resolved.paths))
        if unexpected:
            raise ValueError(
                "git apply would change a path outside the resolved write set: "
                f"{unexpected[0]!r}"
            )
        return after


def _restore_failed_diff(
    *,
    patch: Patch,
    backup: Backup,
    workdir: Path,
    before: dict[str, tuple[str, int, str]],
) -> None:
    """Best-effort reverse plus authoritative backups, then prove restoration."""
    _run_git_apply(patch.unified_diff, workdir, reverse=True)
    revert_patch(backup, workdir)
    restored = _worktree_file_state(workdir)
    if restored != before:
        remaining = sorted(_changed_paths(before, restored))
        preview = ", ".join(repr(path) for path in remaining[:3])
        raise RuntimeError(f"failed patch rollback left changed paths: {preview}")


def _sync_patch_paths(workdir: Path, paths: list[str]) -> None:
    """Persist the complete file and directory state named by a resolved patch."""
    root = workdir.resolve()
    directories: set[Path] = {root}
    for rel in paths:
        target = _safe_target(root, rel)
        try:
            metadata = target.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None:
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ValueError(f"patch durability target is unsafe: {rel!r}")
            sync_regular_file(target)

        parent = target.parent
        while True:
            directories.add(parent)
            if parent == root:
                break
            parent = parent.parent

    # Deepest-first persists file entries before the directory entries that
    # make their containing directories reachable.
    for directory in sorted(
        directories,
        key=lambda value: len(value.relative_to(root).parts),
        reverse=True,
    ):
        try:
            metadata = directory.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"patch durability directory is unsafe: {directory}")
        fsync_directory(directory)


def apply_patch(
    patch: Patch,
    workdir: str | Path,
    *,
    resolved: ResolvedPatch | None = None,
    backup: Backup | None = None,
) -> tuple[list[str], Backup]:
    workdir = Path(workdir)
    actual = resolve_patch(patch)
    if resolved is None:
        resolved = actual
    elif resolved.mode != actual.mode or resolved.paths != actual.paths:
        raise ValueError("resolved patch does not match the executable payload")
    _validate_patch_targets(resolved.paths, workdir)
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
                encoded = content.encode("utf-8")
                mode = backup.modes.get(rel)
                atomic_replace_bytes(
                    target,
                    encoded,
                    anchor=workdir,
                    mode=mode if mode is not None else 0o644,
                )
                if target.read_bytes() != encoded:
                    raise ValueError(f"whole-file patch content mismatch: {rel!r}")
                info = target.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError(f"whole-file patch produced an unsafe target: {rel!r}")
                touched.append(rel)
            _sync_patch_paths(workdir, resolved.paths)
        except Exception:
            revert_patch(backup, workdir)
            raise
        return touched, backup

    if resolved.mode == "diff":
        before = _worktree_file_state(workdir)
        expected = _expected_diff_state(
            patch=patch,
            resolved=resolved,
            backup=backup,
        )
        # Pipe the diff via stdin (no temp file to leak). Duplicate application
        # is an error: the transaction journal, not a heuristic reverse-check,
        # decides whether a persisted attempt should be replayed. A reverse
        # check can misclassify mode-only patches as already applied.
        res = _run_git_apply(patch.unified_diff, workdir)
        if not res.ok:
            raise RuntimeError(f"git apply failed: {res.stderr or res.stdout}")
        try:
            after = _worktree_file_state(workdir)
            unexpected = sorted(_changed_paths(before, after) - set(resolved.paths))
            if unexpected:
                raise ValueError(
                    f"git apply changed a path outside the resolved write set: "
                    f"{unexpected[0]!r}"
                )
            mismatched = [
                rel
                for rel in resolved.paths
                if after.get(rel) != expected.get(rel)
            ]
            if mismatched:
                raise ValueError(
                    "applied file contents do not match the unified diff: "
                    f"{mismatched[0]!r}"
                )
            for rel in resolved.paths:
                target = _safe_target(workdir, rel)
                # Missing is a valid terminal state for a deletion or the source
                # side of a rename. Symlinks and other file types remain invalid.
                try:
                    info = target.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError(f"patch produced an unsafe file: {rel!r}")
            _sync_patch_paths(workdir, resolved.paths)
        except Exception:
            _restore_failed_diff(
                patch=patch,
                backup=backup,
                workdir=workdir,
                before=before,
            )
            raise
        return list(resolved.paths), backup

    return touched, backup


def revert_patch(backup: Backup, workdir: str | Path) -> None:
    workdir = Path(workdir).resolve()
    for rel, original in backup.originals.items():
        target = _safe_target(workdir, rel, allow_leaf_symlink=True)
        if original is None:
            try:
                target.lstat()
            except FileNotFoundError:
                pass
            else:
                target.unlink()
        else:
            if target.is_symlink():
                target.unlink()
            mode = backup.modes.get(rel)
            if mode is None:
                raise ValueError(f"existing backup target has no mode: {rel!r}")
            atomic_replace_bytes(
                target,
                original,
                anchor=workdir,
                mode=mode,
            )
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
    _sync_patch_paths(workdir, sorted(backup.originals))


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


def save_backup(
    backup: Backup,
    path: str | Path,
    *,
    run_dir: str | Path,
) -> str:
    """Persist a checksummed backup below the run-owned directory anchor."""
    path = Path(path)
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
    anchored_atomic_replace_bytes(
        path,
        json.dumps(envelope, sort_keys=True).encode("utf-8"),
        anchor=run_dir,
        mode=0o600,
    )
    return digest


def load_backup(
    path: str | Path,
    *,
    run_dir: str | Path,
    required: bool = False,
) -> Backup | None:
    """Load one backup through its run-owned directory anchor."""
    path = Path(path)
    try:
        encoded = anchored_read_bytes(
            path,
            anchor=run_dir,
            missing_ok=True,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"backup path is corrupt: {path}: {error}") from error
    if encoded is None:
        if required:
            raise ValueError(f"required backup is missing: {path}")
        return None
    try:
        raw = json.loads(encoded)
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
