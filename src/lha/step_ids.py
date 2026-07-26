"""Canonical step identifiers used by plans, checkpoints, and artifact paths."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_STEP_ID = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?"
)
_ARTIFACT_SEGMENT = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?"
)


def step_id_alias_key(value: str) -> str:
    """Return the comparison key used by case-insensitive Unicode filesystems."""
    return unicodedata.normalize("NFC", value).casefold()


def validate_step_id(value: str) -> str:
    """Require a canonical 1–64 character step ID and return it unchanged."""
    if not isinstance(value, str) or _STEP_ID.fullmatch(value) is None:
        raise ValueError(
            "step_id must be 1-64 ASCII characters, start and end with an "
            "alphanumeric character, and contain only letters, digits, '_', '.', or '-'"
        )
    return value


def validate_plan_step_ids(values: Iterable[str]) -> tuple[str, ...]:
    """Reject filesystem aliases across a plan, then validate every identifier."""
    ids = tuple(values)
    aliases: dict[str, str] = {}
    for value in ids:
        key = step_id_alias_key(value)
        previous = aliases.get(key)
        if previous is not None:
            raise ValueError(
                f"plan step_id values alias on case-insensitive Unicode filesystems: "
                f"{previous!r} and {value!r}"
            )
        aliases[key] = value
    for value in ids:
        validate_step_id(value)
    return ids


def canonical_artifact_segment(value: str) -> str:
    """Validate an artifact identity segment and return the original bytes as text."""
    if not isinstance(value, str) or _ARTIFACT_SEGMENT.fullmatch(value) is None:
        raise ValueError(
            "artifact identity must be 1-128 ASCII characters, start and end with "
            "an alphanumeric character, and contain only letters, digits, '_', '.', or '-'"
        )
    return value
