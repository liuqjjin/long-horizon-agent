from __future__ import annotations

from threading import Barrier

import pytest

from concurrent_counter import CounterStore, run_workers


def test_concurrent_increment_is_atomic():
    workers = 8
    barrier = Barrier(workers)
    store = CounterStore(before_write=barrier.wait)
    assert run_workers(store, workers=workers, increments=1) == workers


def test_worker_exception_is_propagated():
    def fail() -> None:
        raise RuntimeError("worker failed")

    store = CounterStore(before_write=fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        run_workers(store, workers=2, increments=1)

