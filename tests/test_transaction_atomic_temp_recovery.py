"""Crash recovery tests for transaction atomic-replace temporary files."""

from __future__ import annotations

import signal
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from lha.harness.checkpoint import run_lock
from lha.harness.errors import TransactionCorrupt
from lha.harness.transaction import (
    PatchTransaction,
    recover_transaction_journals,
    save_transaction,
    transaction_log_path,
    transaction_path,
    validate_transaction_journals,
)


def _transaction_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    (run_dir / "workdir").mkdir(parents=True)
    transaction = PatchTransaction(
        sequence=1,
        step_id="step",
        attempt_id="step-r0",
        patch_sha256="a" * 64,
        resolved_paths=[],
        backup_ref="backups/step/step-r0.json",
        backup_mirror_ref="steps/step/attempts/step-r0/backup.json",
        backup_sha256="b" * 64,
    )
    with run_lock(run_dir):
        save_transaction(run_dir, transaction)
    return run_dir, transaction_path(run_dir, "step", "step-r0")


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")


@pytest.mark.parametrize("evidence", ["state", "events"])
def test_sigkill_before_atomic_replace_is_recovered_under_run_lock(
    tmp_path: Path,
    evidence: str,
) -> None:
    run_dir, state_target = _transaction_run(tmp_path)
    target = (
        state_target
        if evidence == "state"
        else transaction_log_path(run_dir, "step", "step-r0")
    )
    before = target.read_bytes()
    child = """
import os
import signal
import sys
from pathlib import Path

import lha.durable_io as durable_io
from lha.harness.checkpoint import run_lock

run_dir = Path(sys.argv[1])
target = Path(sys.argv[2])

def die_before_replace(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)

with run_lock(run_dir):
    durable_io.os.replace = die_before_replace
    durable_io.anchored_atomic_replace_bytes(
        target,
        target.read_bytes(),
        anchor=run_dir,
    )
"""
    completed = subprocess.run(
        [sys.executable, "-c", child, str(run_dir), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == -signal.SIGKILL
    assert target.read_bytes() == before
    leftovers = list(target.parent.glob(f".{target.name}.*.tmp"))
    assert len(leftovers) == 1
    with pytest.raises(TransactionCorrupt, match="unknown transaction evidence"):
        validate_transaction_journals(run_dir)
    assert leftovers[0].exists()

    with run_lock(run_dir):
        recover_transaction_journals(run_dir)

    assert not leftovers[0].exists()
    validate_transaction_journals(run_dir)
    assert target.read_bytes() == before


def test_forged_exact_temp_name_with_invalid_payload_is_not_removed(
    tmp_path: Path,
) -> None:
    run_dir, target = _transaction_run(tmp_path)
    temporary = _temporary_path(target)
    temporary.write_bytes(b'{"looks": "similar"}')
    temporary.chmod(0o600)

    with run_lock(run_dir):
        with pytest.raises(
            TransactionCorrupt,
            match="invalid transaction atomic-replace temporary file",
        ):
            recover_transaction_journals(run_dir)

    assert temporary.read_bytes() == b'{"looks": "similar"}'


def test_hardlinked_atomic_temp_is_rejected_without_touching_victim(
    tmp_path: Path,
) -> None:
    run_dir, target = _transaction_run(tmp_path)
    victim = tmp_path / "outside-transaction-temp"
    victim.write_bytes(target.read_bytes())
    victim.chmod(0o600)
    temporary = _temporary_path(target)
    temporary.hardlink_to(victim)
    before = victim.read_bytes()

    with run_lock(run_dir):
        with pytest.raises(TransactionCorrupt, match="temporary file is unsafe"):
            recover_transaction_journals(run_dir)

    assert temporary.exists()
    assert victim.read_bytes() == before
    assert victim.stat().st_nlink == 2


def test_symlinked_atomic_temp_is_rejected_without_touching_victim(
    tmp_path: Path,
) -> None:
    run_dir, target = _transaction_run(tmp_path)
    victim = tmp_path / "outside-transaction-temp"
    victim.write_bytes(target.read_bytes())
    temporary = _temporary_path(target)
    temporary.symlink_to(victim)
    before = victim.read_bytes()

    with run_lock(run_dir):
        with pytest.raises(TransactionCorrupt, match="refusing symlink"):
            recover_transaction_journals(run_dir)

    assert temporary.is_symlink()
    assert victim.read_bytes() == before


def test_malformed_atomic_temp_name_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    run_dir, target = _transaction_run(tmp_path)
    temporary = target.with_name(f".{target.name}.not-a-uuid.tmp")
    temporary.write_bytes(target.read_bytes())
    temporary.chmod(0o600)

    with run_lock(run_dir):
        with pytest.raises(TransactionCorrupt, match="unknown transaction temporary"):
            recover_transaction_journals(run_dir)

    assert temporary.exists()
