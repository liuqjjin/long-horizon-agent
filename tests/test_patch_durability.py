"""Crash barriers for patch bytes, directory entries, and recovery aliases."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import hermetic_task

import lha.durable_io as durable_io
import lha.tools.patch as patch_module
from lha.artifacts import Patch
from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import HumanApprovalGate
from lha.harness.transaction import (
    build_transaction,
    load_transaction,
    save_transaction,
)
from lha.tools.patch import (
    apply_patch,
    resolve_patch,
    revert_patch,
    snapshot_paths,
)


def _cfg(tmp_path: Path) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
    )


def _paused(tmp_path: Path):
    result = Harness(_cfg(tmp_path)).run(
        hermetic_task("data/tasks/fix_average_approval.yaml")
    )
    assert result.status == "AWAITING_APPROVAL"
    return result


def test_nested_directory_creation_syncs_each_child_and_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def record(
        child_fd: int,
        parent_fd: int,
        child_path: Path,
        parent_path: Path,
    ) -> None:
        del child_fd, parent_fd
        calls.extend((child_path, parent_path))

    monkeypatch.setattr(durable_io, "_sync_created_directory", record)
    one = tmp_path / "one"
    two = one / "two"
    three = two / "three"

    durable_io.durable_mkdir_chain(three, anchor=tmp_path)

    assert calls == [one, tmp_path, two, one, three, two]


def test_nested_directory_creation_propagates_parent_sync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_parent = tmp_path / "one"

    def fail(
        child_fd: int,
        parent_fd: int,
        child_path: Path,
        parent_path: Path,
    ) -> None:
        del child_fd, parent_fd, child_path
        if parent_path == failed_parent:
            raise OSError("simulated nested-parent fsync failure")

    monkeypatch.setattr(durable_io, "_sync_created_directory", fail)

    with pytest.raises(OSError, match="nested-parent fsync failure"):
        durable_io.durable_mkdir_chain(
            tmp_path / "one" / "two",
            anchor=tmp_path,
        )


@pytest.mark.parametrize("use_real_target", [False, True])
def test_anchor_with_symlinked_ancestor_accepts_the_same_real_directory(
    tmp_path: Path,
    use_real_target: bool,
) -> None:
    real_parent = tmp_path / "real"
    real_anchor = real_parent / "run"
    real_anchor.mkdir(parents=True)
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    alias_anchor = alias_parent / "run"
    relative = Path("steps") / "s1" / "attempts" / "s1-r0"
    target = (
        real_anchor / relative
        if use_real_target
        else alias_anchor / relative
    )

    created = durable_io.durable_mkdir_chain(target, anchor=alias_anchor)

    assert created == real_anchor / relative
    assert (real_anchor / relative).is_dir()


def test_real_anchor_mapping_still_rejects_a_child_symlink(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_anchor = real_parent / "run"
    outside = tmp_path / "outside"
    real_anchor.mkdir(parents=True)
    outside.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    (real_anchor / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="durable directory path is unsafe"):
        durable_io.durable_mkdir_chain(
            real_anchor / "linked" / "evidence",
            anchor=alias_parent / "run",
        )

    assert not (outside / "evidence").exists()


def test_real_anchor_mapping_rejects_a_true_escape(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_anchor = real_parent / "run"
    real_anchor.mkdir(parents=True)
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes its anchor"):
        durable_io.durable_mkdir_chain(
            tmp_path / "outside" / "evidence",
            anchor=alias_parent / "run",
        )


def test_replaced_anchor_never_redirects_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "anchor"
    replacement = tmp_path / "replacement"
    detached = tmp_path / "detached-anchor"
    anchor.mkdir()
    replacement.mkdir()
    real_mkdir = durable_io.os.mkdir
    raced = False

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal raced
        if Path(path).name == "child" and not raced:
            raced = True
            anchor.rename(detached)
            replacement.rename(anchor)
        if dir_fd is None:
            return real_mkdir(path, mode)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(durable_io.os, "mkdir", racing_mkdir)

    with pytest.raises(OSError, match="identity|anchor"):
        durable_io.durable_mkdir_chain(anchor / "child", anchor=anchor)

    assert raced
    assert not (anchor / "child").exists()
    assert (detached / "child").is_dir()


def test_replaced_parent_symlink_never_redirects_child_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "anchor"
    parent = anchor / "parent"
    detached = anchor / "detached-parent"
    outside = tmp_path / "outside"
    parent.mkdir(parents=True)
    outside.mkdir()
    real_mkdir = durable_io.os.mkdir
    raced = False

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal raced
        if Path(path).name == "child" and not raced:
            raced = True
            parent.rename(detached)
            parent.symlink_to(outside, target_is_directory=True)
        if dir_fd is None:
            return real_mkdir(path, mode)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(durable_io.os, "mkdir", racing_mkdir)

    with pytest.raises(OSError):
        durable_io.durable_mkdir_chain(
            parent / "child",
            anchor=anchor,
        )

    assert raced
    assert not (outside / "child").exists()
    assert (detached / "child").is_dir()


def test_anchored_atomic_replace_never_follows_a_replaced_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "anchor"
    parent = anchor / "parent"
    detached = anchor / "detached-parent"
    outside = tmp_path / "outside"
    parent.mkdir(parents=True)
    outside.mkdir()
    external_target = outside / "state.json"
    external_target.write_bytes(b"outside")
    real_open = durable_io.os.open
    raced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal raced
        if Path(path).name.startswith(".state.json.") and not raced:
            raced = True
            parent.rename(detached)
            parent.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(durable_io.os, "open", racing_open)

    with pytest.raises(OSError):
        durable_io.atomic_replace_bytes(
            parent / "state.json",
            b"inside",
            anchor=anchor,
        )

    assert raced
    assert external_target.read_bytes() == b"outside"
    assert (detached / "state.json").read_bytes() == b"inside"


def _record_patch_barrier(monkeypatch: pytest.MonkeyPatch):
    synced_files: list[Path] = []
    synced_directories: list[Path] = []
    real_file_sync = patch_module.sync_regular_file
    real_directory_sync = patch_module.fsync_directory

    def record_file(path: str | Path, **kwargs):
        synced_files.append(Path(path))
        return real_file_sync(path, **kwargs)

    def record_directory(path: str | Path) -> None:
        synced_directories.append(Path(path))
        real_directory_sync(path)

    monkeypatch.setattr(patch_module, "sync_regular_file", record_file)
    monkeypatch.setattr(patch_module, "fsync_directory", record_directory)
    return synced_files, synced_directories


def test_whole_file_patch_syncs_target_and_nested_parent_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    synced_files, synced_directories = _record_patch_barrier(monkeypatch)
    patch = Patch(
        step_id="s",
        file_contents={"src/nested/app.py": "answer = 42\n"},
    )

    apply_patch(patch, workdir)

    target = workdir / "src" / "nested" / "app.py"
    assert target in synced_files
    assert {
        workdir,
        workdir / "src",
        workdir / "src" / "nested",
    }.issubset(set(synced_directories))


@pytest.mark.parametrize("operation", ["delete", "rename", "chmod"])
def test_unified_diff_patch_syncs_every_changed_entry_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    workdir = tmp_path / "workdir"
    source_dir = workdir / "source"
    destination_dir = workdir / "destination"
    source_dir.mkdir(parents=True)
    destination_dir.mkdir()
    source = source_dir / "app.py"
    source.write_text("answer = 41\n")
    if operation == "delete":
        diff = (
            "diff --git a/source/app.py b/source/app.py\n"
            "deleted file mode 100644\n"
            "--- a/source/app.py\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-answer = 41\n"
        )
    elif operation == "rename":
        diff = (
            "diff --git a/source/app.py b/destination/app.py\n"
            "similarity index 100%\n"
            "rename from source/app.py\n"
            "rename to destination/app.py\n"
        )
    else:
        diff = (
            "diff --git a/source/app.py b/source/app.py\n"
            "old mode 100644\n"
            "new mode 100755\n"
        )
    synced_files, synced_directories = _record_patch_barrier(monkeypatch)

    apply_patch(Patch(step_id="s", unified_diff=diff), workdir)

    assert workdir in synced_directories
    # Git may remove an empty source directory. In that case its deletion is
    # made durable by syncing the nearest surviving parent (the worktree).
    assert source_dir in synced_directories or not source_dir.exists()
    if operation == "delete":
        assert not source.exists()
    elif operation == "rename":
        destination = destination_dir / "app.py"
        assert destination in synced_files
        assert destination_dir in synced_directories
        assert not source.exists()
    else:
        assert source in synced_files
        assert source.stat().st_mode & 0o111


def test_revert_syncs_restored_target_and_removed_directory_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    patch = Patch(
        step_id="s",
        file_contents={"created/nested/app.py": "answer = 42\n"},
    )
    _paths, backup = apply_patch(patch, workdir)
    synced_files, synced_directories = _record_patch_barrier(monkeypatch)

    revert_patch(backup, workdir)

    assert not (workdir / "created").exists()
    assert synced_files == []
    assert workdir in synced_directories


def test_apply_sync_failure_does_not_advance_prepared_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    workdir = run_dir / "workdir"
    workdir.mkdir(parents=True)
    target = workdir / "app.py"
    target.write_text("answer = 41\n")
    patch = Patch(step_id="s", file_contents={"app.py": "answer = 42\n"})
    resolved = resolve_patch(patch)
    backup = snapshot_paths(resolved.paths, workdir)
    transaction = build_transaction(
        run_dir=run_dir,
        step_id="s",
        attempt_id="s-r0",
        resolved=resolved,
        backup_sha256="0" * 64,
    )
    save_transaction(run_dir, transaction)
    real_barrier = patch_module._sync_patch_paths
    calls = 0

    def fail_first_barrier(root: Path, paths: list[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated target fsync failure")
        real_barrier(root, paths)

    monkeypatch.setattr(patch_module, "_sync_patch_paths", fail_first_barrier)

    with pytest.raises(OSError, match="target fsync failure"):
        apply_patch(patch, workdir, resolved=resolved, backup=backup)

    persisted = load_transaction(run_dir, "s", "s-r0")
    assert persisted is not None and persisted.status == "PREPARED"
    assert target.read_text() == "answer = 41\n"


def test_revert_sync_failure_does_not_advance_applied_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    workdir = run_dir / "workdir"
    workdir.mkdir(parents=True)
    target = workdir / "app.py"
    target.write_text("answer = 41\n")
    patch = Patch(step_id="s", file_contents={"app.py": "answer = 42\n"})
    resolved = resolve_patch(patch)
    backup = snapshot_paths(resolved.paths, workdir)
    transaction = build_transaction(
        run_dir=run_dir,
        step_id="s",
        attempt_id="s-r0",
        resolved=resolved,
        backup_sha256="0" * 64,
    )
    save_transaction(run_dir, transaction)
    apply_patch(patch, workdir, resolved=resolved, backup=backup)
    applied = transaction.transition("APPLIED", workdir=workdir)
    save_transaction(run_dir, applied)

    def fail_barrier(_root: Path, _paths: list[str]) -> None:
        raise OSError("simulated revert fsync failure")

    monkeypatch.setattr(patch_module, "_sync_patch_paths", fail_barrier)

    with pytest.raises(OSError, match="revert fsync failure"):
        revert_patch(backup, workdir)

    persisted = load_transaction(run_dir, "s", "s-r0")
    assert persisted is not None and persisted.status == "APPLIED"
    assert target.read_text() == "answer = 41\n"


def test_resume_uses_attempt_patch_when_compatibility_aliases_are_missing(
    tmp_path: Path,
) -> None:
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    for relative in (
        "patch.json",
        "patch.diff",
        "steps/s2-fix/patch.json",
        "steps/s2-fix/patch.diff",
    ):
        (run_dir / relative).unlink()
    HumanApprovalGate(run_dir).resolve(
        approved=True,
        note="reviewed immutable attempt evidence",
    )

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)

    assert resumed.status == "DONE"
    assert "len(values) - 1" not in (
        run_dir / "workdir" / "mathutils.py"
    ).read_text()


@pytest.mark.parametrize(
    ("relative", "replacement"),
    [
        ("patch.json", b"{}\n"),
        ("steps/s2-fix/patch.json", b"{}\n"),
        ("patch.diff", b"misleading review\n"),
        ("steps/s2-fix/patch.diff", b"misleading review\n"),
    ],
)
def test_resume_rejects_mismatched_compatibility_alias(
    tmp_path: Path,
    relative: str,
    replacement: bytes,
) -> None:
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    (run_dir / relative).write_bytes(replacement)
    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")

    resumed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)

    assert resumed.status == "FAILED"
    assert "alias does not match immutable attempt evidence" in resumed.message
    assert "len(values) - 1" in (
        run_dir / "workdir" / "mathutils.py"
    ).read_text()
