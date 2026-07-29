"""Interval merging utilities."""


def merge(intervals: list[list[int]]) -> list[list[int]]:
    """Combine overlapping intervals from an unsorted list.

    Each interval is a ``[start, end]`` pair with ``start <= end``. The result
    is returned sorted by start position with no two output intervals sharing
    any region.
    """
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda pair: (pair[0], pair[1]))
    merged: list[list[int]] = [list(ordered[0])]

    for start, end in ordered[1:]:
        last_end = merged[-1][1]
        if start < last_end:
            if end > last_end:
                merged[-1][1] = end
        else:
            merged.append([start, end])

    return merged
