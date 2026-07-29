"""Flatten arbitrarily nested sequences into a single flat list."""

from collections.abc import Iterable


def flatten(seq):
    """Return a flat list containing every leaf element of ``seq``.

    Nested sequences are expanded recursively so that the result has no
    nested structure left.
    """
    result = []
    for item in seq:
        if isinstance(item, Iterable):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
