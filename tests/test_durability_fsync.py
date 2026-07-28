"""Directory-sync failures must stop durable state transitions."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Callable

import pytest

from lha.agents.experimenter import _durable_replace as replace_experiment_artifact
from lha.bench import terminal_bench, terminal_public_evidence
from lha.harness import approval, checkpoint, transaction
from lha.llm.base import LLMClient
from lha.llm.trace import LLMUsageTotals, TracedLLM, _save_usage_checkpoint
from lha.repo_adapter import _durable_replace as replace_repo_artifact
from lha.tools.patch import Backup, save_backup


class _CountingLLM(LLMClient):
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, prompt: str) -> str:
        self.calls += 1
        return "{}"


def _reject_directory_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = os.fsync

    def fail_directory(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory)


@pytest.mark.parametrize(
    "write",
    [
        lambda path: checkpoint._fsync_write(path, "{}"),
        lambda path: transaction._fsync_replace(path, "{}"),
        lambda path: transaction.durable_artifact_write(path, b"{}"),
        lambda path: replace_repo_artifact(
            path,
            "{}",
            run_dir=path.parent,
        ),
        lambda path: replace_experiment_artifact(
            path,
            b"{}",
            run_dir=path.parent,
        ),
        lambda path: save_backup(Backup(), path, run_dir=path.parent),
    ],
)
def test_atomic_replacements_reject_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write: Callable[[Path], object],
) -> None:
    _reject_directory_sync(monkeypatch)

    with pytest.raises(OSError, match="directory fsync failure"):
        write(tmp_path / "artifact.json")


def test_approval_directory_sync_failure_is_not_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reject_directory_sync(monkeypatch)

    gate = approval.HumanApprovalGate(tmp_path)
    with pytest.raises(ValueError, match="unsafe approval path") as caught:
        gate._atomic_write(tmp_path / "pending_approval.json", "{}")
    assert isinstance(caught.value.__cause__, OSError)
    assert "directory fsync failure" in str(caught.value.__cause__)


def test_llm_usage_rejects_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reject_directory_sync(monkeypatch)

    with pytest.raises(OSError, match="directory fsync failure"):
        _save_usage_checkpoint(tmp_path, LLMUsageTotals())


def test_llm_backend_is_not_called_when_usage_reservation_is_not_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _CountingLLM()
    traced = TracedLLM(backend).bind(tmp_path)
    _reject_directory_sync(monkeypatch)

    with pytest.raises(OSError, match="directory fsync failure"):
        traced.complete("system", "prompt")

    assert backend.calls == 0
    assert traced.totals.calls == 0


def test_terminal_protocol_rejects_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reject_directory_sync(monkeypatch)

    with pytest.raises(OSError, match="directory fsync failure"):
        terminal_bench._atomic_write_text(tmp_path / "protocol.json", "{}")


def test_terminal_publication_rejects_directory_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open

    def fail_directory_open(path, flags, *args, **kwargs):
        if Path(path) == tmp_path:
            raise OSError("simulated directory open failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_directory_open)

    with pytest.raises(OSError, match="directory open failure"):
        terminal_public_evidence._fsync_directory(tmp_path)


@pytest.mark.parametrize(
    "mkdir_chain",
    [
        terminal_bench._durable_mkdir_chain,
        terminal_public_evidence._durable_mkdir_chain,
    ],
)
def test_terminal_nested_directory_chain_syncs_each_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mkdir_chain: Callable[[Path], None],
) -> None:
    real_fsync = os.fsync
    synced_directories: list[tuple[int, int]] = []

    def record_sync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            synced_directories.append((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_sync)
    destination = tmp_path / "one" / "two" / "three"

    mkdir_chain(destination)

    expected = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in (tmp_path, tmp_path / "one", tmp_path / "one" / "two")
    }
    assert expected.issubset(set(synced_directories))
