"""Shared counter state."""

from __future__ import annotations

from collections.abc import Callable


class CounterStore:
    def __init__(self, before_write: Callable[[], object] | None = None):
        self._value = 0
        self._before_write = before_write

    @property
    def value(self) -> int:
        return self._value

    def increment(self) -> None:
        current = self._value
        if self._before_write is not None:
            self._before_write()
        self._value = current + 1

