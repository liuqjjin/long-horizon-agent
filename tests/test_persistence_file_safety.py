"""Adversarial link tests for run-owned persistence files."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha import durable_io
from lha.config import Config
from lha.durable_io import anchored_update_bytes, anchored_write_once_bytes
from lha.harness.checkpoint import (
    append_ledger,
    load_state,
    run_lock,
    save_state,
)
from lha.harness.errors import CheckpointCorrupt, TransactionCorrupt
from lha.harness.state import RunState, StepRecord
from lha.harness.transaction import (
    PatchTransaction,
    save_transaction,
    transaction_log_path,
)
from lha.llm.base import LLMClient
from lha.llm.trace import (
    LLMUsageTotals,
    TracedLLM,
    _require_exact_file,
    _save_usage_checkpoint,
    _write_once,
    load_usage_checkpoint,
)
from lha.runtime.langgraph_runner import _graph_database_connection


class _AnsweringLLM(LLMClient):
    name = "answering"

    def complete(self, system: str, prompt: str) -> str:
        return "{}"


def _state(tmp_path: Path) -> RunState:
    run_dir = tmp_path / "run"
    workdir = run_dir / "workdir"
    workdir.mkdir(parents=True)
    config = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        use_skill_memory=False,
    )
    return RunState.new(
        hermetic_task("data/tasks/fix_average.yaml"),
        "run",
        str(run_dir),
        str(workdir),
        config=config,
    )


def _replace_with_hardlink(path: Path, victim: Path) -> None:
    victim.write_bytes(path.read_bytes())
    path.unlink()
    path.hardlink_to(victim)


def test_checkpoint_reader_rejects_a_hardlink(tmp_path: Path) -> None:
    state = _state(tmp_path)
    save_state(state)
    checkpoint = Path(state.run_dir) / "state.json"
    victim = tmp_path / "outside-state.json"
    _replace_with_hardlink(checkpoint, victim)

    with pytest.raises(CheckpointCorrupt, match="unsafe"):
        load_state(state.run_dir)


def test_checkpoint_save_does_not_modify_a_hardlinked_external_file(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    save_state(state)
    checkpoint = Path(state.run_dir) / "state.json"
    victim = tmp_path / "outside-state.json"
    _replace_with_hardlink(checkpoint, victim)
    before = victim.read_bytes()

    with pytest.raises(OSError, match="unsafe"):
        save_state(state.model_copy(update={"seq": 9}))
    assert victim.read_bytes() == before


def test_checkpoint_reader_rejects_inode_replacement_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(tmp_path)
    save_state(state)
    checkpoint = Path(state.run_dir) / "state.json"
    replacement = Path(state.run_dir) / "replacement.json"
    replacement.write_bytes(checkpoint.read_bytes())
    real_open = durable_io.os.open
    raced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal raced
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "state.json" and not raced:
            raced = True
            replacement.replace(checkpoint)
        return descriptor

    monkeypatch.setattr(durable_io.os, "open", racing_open)
    with pytest.raises(CheckpointCorrupt, match="unsafe|changed while reading"):
        load_state(state.run_dir)


def test_ledger_append_does_not_modify_a_hardlinked_external_file(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    first = StepRecord(seq=state.next_seq(), step_id="-", phase="plan")
    append_ledger(state, first)
    ledger = Path(state.run_dir) / "ledger.jsonl"
    victim = tmp_path / "outside-ledger.jsonl"
    _replace_with_hardlink(ledger, victim)
    before = victim.read_bytes()

    with pytest.raises(CheckpointCorrupt, match="unsafe"):
        append_ledger(
            state,
            StepRecord(seq=state.next_seq(), step_id="s", phase="context"),
        )
    assert victim.read_bytes() == before


def test_run_lock_does_not_modify_a_hardlinked_external_file(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    victim = tmp_path / "outside-lock"
    victim.write_text("keep\n")
    (run_dir / ".run.lock").hardlink_to(victim)
    before = victim.read_bytes()

    with pytest.raises(CheckpointCorrupt, match="unsafe"):
        with run_lock(run_dir):
            pass
    assert victim.read_bytes() == before


def test_transaction_event_append_does_not_modify_a_hardlink(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "workdir").mkdir(parents=True)
    transaction = PatchTransaction(
        sequence=1,
        step_id="s",
        attempt_id="s-r0",
        patch_sha256="a" * 64,
        resolved_paths=[],
        backup_ref="backups/s/s-r0.json",
        backup_mirror_ref="steps/s/attempts/s-r0/backup.json",
        backup_sha256="b" * 64,
    )
    save_transaction(run_dir, transaction)
    log = transaction_log_path(run_dir, "s", "s-r0")
    victim = tmp_path / "outside-events.jsonl"
    _replace_with_hardlink(log, victim)
    before = victim.read_bytes()

    with pytest.raises(TransactionCorrupt, match="unsafe"):
        save_transaction(
            run_dir,
            transaction.transition("APPLIED", workdir=run_dir / "workdir"),
        )
    assert victim.read_bytes() == before


def test_trace_append_does_not_modify_a_hardlinked_external_file(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "llm_trace.jsonl"
    victim = tmp_path / "outside-trace.jsonl"
    victim.write_text(json.dumps({"keep": True}) + "\n")
    trace.hardlink_to(victim)
    before = victim.read_bytes()

    traced = TracedLLM(_AnsweringLLM()).bind(tmp_path)
    assert traced.complete("system", "prompt") == "{}"
    assert victim.read_bytes() == before


def test_usage_checkpoint_rejects_a_hardlink_for_read_and_write(
    tmp_path: Path,
) -> None:
    _save_usage_checkpoint(tmp_path, LLMUsageTotals(calls=1))
    checkpoint = tmp_path / "llm_usage.json"
    victim = tmp_path / "outside-usage.json"
    _replace_with_hardlink(checkpoint, victim)
    before = victim.read_bytes()

    with pytest.raises(CheckpointCorrupt, match="invalid LLM usage checkpoint"):
        load_usage_checkpoint(tmp_path)
    with pytest.raises(OSError, match="unsafe"):
        _save_usage_checkpoint(tmp_path, LLMUsageTotals(calls=2))
    assert victim.read_bytes() == before


@pytest.mark.parametrize("operation", [_write_once, _require_exact_file])
def test_llm_attempt_artifacts_reject_hardlinks(
    tmp_path: Path,
    operation,
) -> None:
    victim = tmp_path / "outside.json"
    victim.write_bytes(b"{}")
    artifact = tmp_path / "artifact.json"
    artifact.hardlink_to(victim)

    with pytest.raises(CheckpointCorrupt, match="unsafe|hardlink"):
        if operation is _require_exact_file:
            operation(artifact, b"{}", "artifact")
        else:
            operation(artifact, b"{}")


def test_write_once_has_exactly_one_concurrent_creator(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    payload = b"x" * (2 * 1024 * 1024)
    barrier = threading.Barrier(2)
    results: list[bool | BaseException] = []

    def write() -> None:
        barrier.wait()
        try:
            results.append(
                anchored_write_once_bytes(path, payload, anchor=tmp_path)
            )
        except BaseException as error:
            results.append(error)

    threads = [threading.Thread(target=write) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(result is True for result in results) == 1
    assert all(
        result is True or result is False or isinstance(result, OSError)
        for result in results
    )
    assert path.read_bytes() == payload
    assert path.stat().st_nlink == 1


def test_anchored_update_rejects_same_inode_content_race(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b"first\n")

    def racing_update(current: bytes | None) -> bytes:
        assert current == b"first\n"
        path.write_bytes(b"attacker\n")
        return current + b"second\n"

    with pytest.raises(OSError, match="identity changed before replace"):
        anchored_update_bytes(path, racing_update, anchor=tmp_path)
    assert path.read_bytes() == b"attacker\n"


def _sqlite_bytes(path: Path) -> bytes:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE durable(value TEXT)")
        connection.execute("INSERT INTO durable VALUES ('keep')")
        connection.commit()
    finally:
        connection.close()
    return path.read_bytes()


def test_graph_database_rejects_a_hardlinked_main_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    victim = tmp_path / "outside.sqlite"
    _sqlite_bytes(victim)
    (run_dir / "graph.sqlite").hardlink_to(victim)
    before = victim.read_bytes()

    with pytest.raises(CheckpointCorrupt, match="SQLite path is unsafe"):
        _graph_database_connection(run_dir)
    assert victim.read_bytes() == before


def test_graph_database_rejects_a_symlinked_main_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    victim = tmp_path / "outside.sqlite"
    _sqlite_bytes(victim)
    (run_dir / "graph.sqlite").symlink_to(victim)
    before = victim.read_bytes()

    with pytest.raises(CheckpointCorrupt, match="SQLite path is unsafe"):
        _graph_database_connection(run_dir)
    assert victim.read_bytes() == before


@pytest.mark.parametrize("sidecar", ["graph.sqlite-journal", "graph.sqlite-wal", "graph.sqlite-shm"])
def test_graph_database_rejects_hardlinked_sidecars(
    tmp_path: Path,
    sidecar: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _sqlite_bytes(run_dir / "graph.sqlite")
    victim = tmp_path / f"outside-{sidecar}"
    victim.write_bytes(b"outside")
    (run_dir / sidecar).hardlink_to(victim)
    before = victim.read_bytes()

    with pytest.raises(CheckpointCorrupt, match="SQLite path is unsafe"):
        _graph_database_connection(run_dir)
    assert victim.read_bytes() == before


def test_graph_database_never_opens_the_run_owned_path_with_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lha.runtime import langgraph_runner

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    victim = tmp_path / "outside.sqlite"
    _sqlite_bytes(victim)
    real_connect = sqlite3.connect
    opened: list[str] = []

    def guarded_connect(database, *args, **kwargs):
        opened.append(str(database))
        if str(database) == str(run_dir / "graph.sqlite"):
            (run_dir / "graph.sqlite").symlink_to(victim)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(langgraph_runner.sqlite3, "connect", guarded_connect)
    connection, persist = _graph_database_connection(run_dir)
    try:
        connection.execute("CREATE TABLE durable(value TEXT)")
        connection.execute("INSERT INTO durable VALUES ('inside')")
        persist()
    finally:
        connection.close()

    assert str(run_dir / "graph.sqlite") not in opened
    check = real_connect(run_dir / "graph.sqlite")
    try:
        assert check.execute("SELECT value FROM durable").fetchone() == ("inside",)
    finally:
        check.close()
    outside = real_connect(victim)
    try:
        assert outside.execute("SELECT value FROM durable").fetchone() == ("keep",)
    finally:
        outside.close()


def test_graph_database_fails_closed_on_regular_wal_sidecars(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    source_connection = sqlite3.connect(source)
    source_connection.execute("PRAGMA journal_mode=WAL")
    source_connection.execute("PRAGMA wal_autocheckpoint=0")
    source_connection.execute("CREATE TABLE durable(value TEXT)")
    source_connection.execute("INSERT INTO durable VALUES ('from-wal')")
    source_connection.commit()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for suffix in ("", "-wal", "-shm"):
        source_path = Path(f"{source}{suffix}")
        assert source_path.is_file()
        (run_dir / f"graph.sqlite{suffix}").write_bytes(source_path.read_bytes())
    source_connection.close()
    sentinel = tmp_path / "outside-sentinel"
    sentinel.write_bytes(b"keep")
    before = {
        path.name: path.read_bytes()
        for path in run_dir.iterdir()
    }

    with pytest.raises(CheckpointCorrupt, match="offline recovery"):
        _graph_database_connection(run_dir)
    assert {
        path.name: path.read_bytes()
        for path in run_dir.iterdir()
    } == before
    assert sentinel.read_bytes() == b"keep"
