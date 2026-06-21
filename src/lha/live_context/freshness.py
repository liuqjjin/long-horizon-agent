"""Freshness tracking: decide whether indexed context still reflects the source.

Three signals, cheap -> strong:
  1. file mtime vs. the chunk's ``indexed_at``  (cheap)
  2. content hash of the source vs. ``content_hash`` recorded at index time
  3. backend index_version / generation counter
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from ..clock import now
from .models import ContextItem, Freshness

_LINES_SUFFIX = re.compile(r":\d+(?:-\d+)?$")


def content_hash(text: str) -> str:
    """sha256 of a chunk of text (hex)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def path_from_locator(locator: str) -> str:
    """Strip a trailing ``:start-end`` line range from a locator to get a path."""
    return _LINES_SUFFIX.sub("", locator)


def file_mtime(path: Path) -> datetime | None:
    try:
        ts = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def assess(
    items: list[ContextItem],
    *,
    index_version: str,
    indexed_at: datetime,
    base_dir: Path = Path("."),
) -> Freshness:
    """Build a Freshness verdict for a set of context items."""
    reasons: list[str] = []
    mtimes: list[datetime] = []
    stale = False

    for item in items:
        rel = path_from_locator(item.provenance.locator)
        p = Path(rel)
        path = p if p.is_absolute() else (base_dir / rel)
        mtime = file_mtime(path)
        if mtime is None:
            continue
        mtimes.append(mtime)
        if mtime > item.provenance.indexed_at:
            stale = True
            reasons.append(f"{rel} modified after it was indexed")

    return Freshness(
        index_version=index_version,
        indexed_at=indexed_at,
        source_mtime_max=max(mtimes) if mtimes else None,
        is_stale_flag=stale,
        reasons=reasons,
    )


def fresh_now(index_version: str) -> Freshness:
    """A trivially-fresh verdict (used when there are no items to assess)."""
    return Freshness(index_version=index_version, indexed_at=now())
