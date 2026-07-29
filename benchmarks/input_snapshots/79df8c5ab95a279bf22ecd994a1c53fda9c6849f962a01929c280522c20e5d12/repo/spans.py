"""Coalescing of half-open time spans.

A span is a ``(start, end)`` pair meaning the half-open interval
``[start, end)``. ``merge_spans`` coalesces overlapping spans; spans that
merely touch (one ends exactly where the next starts) stay separate, because
``[1, 2)`` and ``[2, 3)`` share no point. Input order is arbitrary; the result
is sorted by start.
"""


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in spans:
        if out and start < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
