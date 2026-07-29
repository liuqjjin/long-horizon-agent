"""Regression tests for read-only transaction inspection."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.config import Config
from lha.harness import Harness
from lha.harness.checkpoint import run_lock
from lha.harness.errors import TransactionCorrupt
from lha.harness.transaction import (
    list_transactions,
    load_transaction,
    read_transaction_events,
    recover_transaction_journals,
    save_transaction,
    transaction_log_path,
    validate_transaction_journals,
)
from lha.reporting import ReportingError, collect_run, discover_runs


def _config(tmp_path: Path) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
    )


def _paused_run(tmp_path: Path):
    return Harness(_config(tmp_path)).run(
        hermetic_task("data/tasks/fix_average_approval.yaml")
    )


def test_missing_event_is_repaired_only_by_explicit_locked_recovery(tmp_path):
    paused = _paused_run(tmp_path)
    run_dir = Path(paused.state.run_dir)
    transaction = list_transactions(run_dir, "s2-fix")[0]
    log_path = transaction_log_path(
        run_dir, transaction.step_id, transaction.attempt_id
    )
    prepared_line = log_path.read_bytes().splitlines(keepends=True)[0]
    log_path.write_bytes(prepared_line)
    before = log_path.read_bytes()

    assert [event.status for event in read_transaction_events(
        run_dir, transaction.step_id, transaction.attempt_id
    )] == ["PREPARED"]
    assert log_path.read_bytes() == before
    with pytest.raises(TransactionCorrupt, match="does not end"):
        load_transaction(run_dir, transaction.step_id, transaction.attempt_id)
    assert log_path.read_bytes() == before
    with pytest.raises(TransactionCorrupt, match="does not end"):
        list_transactions(run_dir, transaction.step_id)
    assert log_path.read_bytes() == before

    with run_lock(run_dir):
        recover_transaction_journals(run_dir)

    validate_transaction_journals(run_dir)
    assert [event.status for event in read_transaction_events(
        run_dir, transaction.step_id, transaction.attempt_id
    )] == ["PREPARED", "APPLIED"]


def test_torn_event_tail_is_never_truncated_by_a_reader(tmp_path):
    paused = _paused_run(tmp_path)
    run_dir = Path(paused.state.run_dir)
    transaction = list_transactions(run_dir, "s2-fix")[0]
    log_path = transaction_log_path(
        run_dir, transaction.step_id, transaction.attempt_id
    )
    complete = log_path.read_bytes()
    log_path.write_bytes(complete + b'{"incomplete":')
    damaged = log_path.read_bytes()

    with pytest.raises(TransactionCorrupt, match="torn final append"):
        read_transaction_events(
            run_dir, transaction.step_id, transaction.attempt_id
        )
    with pytest.raises(ReportingError, match="torn final append"):
        collect_run(run_dir.parent, paused.state.run_id)
    assert log_path.read_bytes() == damaged

    with run_lock(run_dir):
        recover_transaction_journals(run_dir)

    assert log_path.read_bytes() == complete
    validate_transaction_journals(run_dir)


def test_reporting_never_repairs_a_transaction_during_concurrent_save(
    tmp_path,
    monkeypatch,
):
    import lha.harness.transaction as transaction_module

    paused = _paused_run(tmp_path)
    run_dir = Path(paused.state.run_dir)
    transaction = list_transactions(run_dir, "s2-fix")[0]
    log_path = transaction_log_path(
        run_dir, transaction.step_id, transaction.attempt_id
    )
    before = log_path.read_bytes()
    updated = transaction.model_copy(
        update={"updated_at": "2099-01-01T00:00:00+00:00"}
    )

    writer_reached_append = threading.Event()
    release_writer = threading.Event()
    writer_errors: list[BaseException] = []
    writer_thread: threading.Thread | None = None
    real_append = transaction_module._append_transaction_event

    def delayed_append(*args, **kwargs):
        if threading.current_thread() is writer_thread:
            writer_reached_append.set()
            if not release_writer.wait(timeout=10):
                raise AssertionError("test did not release transaction writer")
        return real_append(*args, **kwargs)

    def write_transaction() -> None:
        try:
            with run_lock(run_dir):
                save_transaction(run_dir, updated)
        except BaseException as error:
            writer_errors.append(error)

    monkeypatch.setattr(
        transaction_module,
        "_append_transaction_event",
        delayed_append,
    )
    writer_thread = threading.Thread(target=write_transaction)
    writer_thread.start()
    assert writer_reached_append.wait(timeout=10)

    try:
        with pytest.raises(ReportingError, match="does not end"):
            collect_run(run_dir.parent, paused.state.run_id)
        summaries = {
            summary.run_id: summary
            for summary in discover_runs(run_dir.parent)
        }
        assert summaries[paused.state.run_id].status == "CORRUPT"
        assert log_path.read_bytes() == before
    finally:
        release_writer.set()
        writer_thread.join(timeout=10)

    assert not writer_thread.is_alive()
    assert writer_errors == []
    validate_transaction_journals(run_dir)
    assert len(read_transaction_events(
        run_dir, transaction.step_id, transaction.attempt_id
    )) == 3
    collect_run(run_dir.parent, paused.state.run_id)
