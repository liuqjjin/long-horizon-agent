"""Metrics recomputed from raw values."""

from __future__ import annotations


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)

