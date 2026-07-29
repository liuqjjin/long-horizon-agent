"""Compute the median of a sequence of numbers (pure stdlib)."""

from __future__ import annotations

from collections.abc import Sequence


def median(values: Sequence[float]) -> float:
    """Return the median of ``values``.

    The input does not need to be pre-sorted; a sorted copy is used.
    """
    if not values:
        raise ValueError("median() arg is an empty sequence")

    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return ordered[mid - 1]
