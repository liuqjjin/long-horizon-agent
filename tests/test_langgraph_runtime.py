"""The LangGraph durable runtime reaches the same verified result as the loop.

Hermetic: stub LLM, null code backend, isolated data dir (no ccc / no model).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.config import Config
from lha.runtime.langgraph_runner import (
    _durable_sqlite_saver,
    _graph_database_connection,
)
from lha.verifiers.verdict import Verdict

pytest.importorskip("langgraph")


def test_langgraph_runtime_reaches_verified_done(tmp_path):
    from lha.runtime.langgraph_runner import LangGraphHarness

    cfg = Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
    )
    result = LangGraphHarness(cfg).run(hermetic_task("data/tasks/fix_average.yaml"))
    assert result.status == "DONE"

    run_dir = Path(result.state.run_dir)
    assert (run_dir / "graph.sqlite").exists()  # durable checkpoint
    verdict = Verdict.model_validate_json((run_dir / "verify.json").read_text())
    assert verdict.passed
    fixed = (run_dir / "workdir" / "mathutils.py").read_text()
    assert "len(values) - 1" not in fixed


def test_durable_sqlite_saver_concurrent_get_and_put_share_one_rlock(
    tmp_path: Path,
) -> None:
    from langgraph.checkpoint.base import empty_checkpoint

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    connection, persistence = _graph_database_connection(run_dir)
    saver = _durable_sqlite_saver(connection, persistence)

    # This identity is the regression barrier: the old implementation used a
    # base cursor lock and a second persistence lock in opposite orders.
    assert saver.lock is persistence.lock
    assert saver.lock.acquire(timeout=1)
    try:
        assert saver.lock.acquire(timeout=1)
        saver.lock.release()
    finally:
        saver.lock.release()

    base_config = {
        "configurable": {
            "thread_id": "concurrent",
            "checkpoint_ns": "",
        }
    }
    start = threading.Barrier(3)
    writer_done = threading.Event()
    failures: list[BaseException] = []
    reads = 0

    def write_checkpoints() -> None:
        config = base_config
        try:
            start.wait()
            for index in range(60):
                checkpoint = empty_checkpoint()
                checkpoint["id"] = f"{index:032d}"
                config = saver.put(
                    config,
                    checkpoint,
                    {"source": "loop", "step": index},
                    {},
                )
        except BaseException as error:
            failures.append(error)
        finally:
            writer_done.set()

    def read_checkpoints() -> None:
        nonlocal reads
        try:
            start.wait()
            while not writer_done.is_set():
                saver.get_tuple(base_config)
                reads += 1
            saver.get_tuple(base_config)
            reads += 1
        except BaseException as error:
            failures.append(error)

    writer = threading.Thread(target=write_checkpoints, daemon=True)
    reader = threading.Thread(target=read_checkpoints, daemon=True)
    writer.start()
    reader.start()
    start.wait()
    writer.join(timeout=10)
    reader.join(timeout=10)

    assert not writer.is_alive(), "concurrent checkpoint writer deadlocked"
    assert not reader.is_alive(), "concurrent checkpoint reader deadlocked"
    assert not failures
    assert reads > 0
    connection.close()

    reopened, reopened_persistence = _graph_database_connection(run_dir)
    try:
        durable = _durable_sqlite_saver(reopened, reopened_persistence)
        latest = durable.get_tuple(base_config)
        assert latest is not None
        assert latest.checkpoint["id"] == f"{59:032d}"
    finally:
        reopened.close()
