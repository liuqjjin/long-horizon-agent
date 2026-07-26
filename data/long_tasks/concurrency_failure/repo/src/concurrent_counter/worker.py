"""Thread-pool orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .store import CounterStore


def _increment_many(store: CounterStore, count: int) -> None:
    for _ in range(count):
        store.increment()


def run_workers(store: CounterStore, workers: int, increments: int) -> int:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers):
            pool.submit(_increment_many, store, increments)
    return store.value

