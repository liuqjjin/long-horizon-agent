"""Utilities for checking bracket sequences."""

_PAIRS = {")": "(", "]": "[", "}": "{"}
_OPENERS = frozenset(_PAIRS.values())


def is_balanced(s: str) -> bool:
    """Return whether the brackets in ``s`` are balanced.

    Recognizes the three bracket kinds ``()``, ``[]`` and ``{}``.
    Characters that are not brackets are ignored.
    """
    stack = []
    for ch in s:
        if ch in _OPENERS:
            stack.append(ch)
        elif ch in _PAIRS:
            if not stack:
                return False
            stack.pop()
    return True
