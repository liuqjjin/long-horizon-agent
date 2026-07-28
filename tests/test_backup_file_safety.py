"""Adversarial checks for run-owned transaction backup files."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import lha.durable_io as durable_io
from lha.tools.patch import Backup, load_backup, save_backup


def _backup() -> Backup:
    return Backup(
        originals={"module.py": b"value = 1\n"},
        modes={"module.py": 0o640},
    )


def test_backup_save_refuses_external_hardlink_without_overwriting(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    backup_dir = run_dir / "backups"
    backup_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside-owned")
    target = backup_dir / "primary.json"
    os.link(outside, target)

    with pytest.raises(OSError, match="unsafe"):
        save_backup(_backup(), target, run_dir=run_dir)

    assert outside.read_bytes() == b"outside-owned"
    assert target.read_bytes() == b"outside-owned"


def test_backup_load_refuses_an_external_hardlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    backup_dir = run_dir / "backups"
    backup_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    save_backup(_backup(), outside, run_dir=tmp_path)
    target = backup_dir / "primary.json"
    os.link(outside, target)

    with pytest.raises(ValueError, match="backup path is corrupt"):
        load_backup(target, run_dir=run_dir, required=True)
    with pytest.raises(ValueError, match="backup path is corrupt"):
        load_backup(target, run_dir=run_dir)


def test_backup_save_and_load_refuse_a_symlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    backup_dir = run_dir / "backups"
    backup_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside-owned")
    target = backup_dir / "primary.json"
    target.symlink_to(outside)

    with pytest.raises(OSError, match="unsafe"):
        save_backup(_backup(), target, run_dir=run_dir)
    with pytest.raises(ValueError, match="backup path is corrupt"):
        load_backup(target, run_dir=run_dir, required=True)

    assert outside.read_bytes() == b"outside-owned"


def test_backup_save_refuses_a_symlinked_parent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_dir / "backups").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="unsafe"):
        save_backup(
            _backup(),
            run_dir / "backups" / "primary.json",
            run_dir=run_dir,
        )

    assert not (outside / "primary.json").exists()


def test_backup_save_detects_parent_replacement_without_redirecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    backup_dir = run_dir / "backups"
    detached = run_dir / "detached-backups"
    outside = tmp_path / "outside"
    backup_dir.mkdir(parents=True)
    outside.mkdir()
    (outside / "primary.json").write_bytes(b"outside-owned")
    real_replace = durable_io._replace_at
    raced = False

    def racing_replace(*args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            backup_dir.rename(detached)
            backup_dir.symlink_to(outside, target_is_directory=True)
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(durable_io, "_replace_at", racing_replace)

    with pytest.raises(OSError):
        save_backup(
            _backup(),
            backup_dir / "primary.json",
            run_dir=run_dir,
        )

    assert raced
    assert (outside / "primary.json").read_bytes() == b"outside-owned"


def test_backup_load_detects_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    backup_dir = run_dir / "backups"
    detached = run_dir / "detached-backups"
    outside = tmp_path / "outside"
    backup_dir.mkdir(parents=True)
    outside.mkdir()
    target = backup_dir / "primary.json"
    save_backup(_backup(), target, run_dir=run_dir)
    (outside / "primary.json").write_bytes(b"outside-owned")
    real_read = durable_io._read_regular_at
    raced = False

    def racing_read(*args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            backup_dir.rename(detached)
            backup_dir.symlink_to(outside, target_is_directory=True)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(durable_io, "_read_regular_at", racing_read)

    with pytest.raises(ValueError, match="backup path is corrupt"):
        load_backup(target, run_dir=run_dir, required=True)

    assert raced
    assert (outside / "primary.json").read_bytes() == b"outside-owned"
