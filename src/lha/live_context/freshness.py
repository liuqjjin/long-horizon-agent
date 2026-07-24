"""Freshness tracking: decide whether indexed context still reflects the source.

Signals actually checked by ``assess`` (cheap -> strong):
  1. file mtime vs. the chunk's ``indexed_at``;
  2. for code chunks (verbatim file slices): chunk text still present in the
     current source — content drift that mtime alone can miss;
  3. a source file that no longer exists is stale, not silently skipped.

The backend index generation (``index_version``) is carried on the verdict for
comparison across reindexes; ``content_hash`` on provenance records what was
indexed.
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
            # A source that disappeared is stale context, not ignorable context.
            stale = True
            reasons.append(f"{rel} no longer exists (or is unreadable)")
            continue
        mtimes.append(mtime)
        if mtime > item.provenance.indexed_at:
            stale = True
            reasons.append(f"{rel} modified after it was indexed")
        elif (
            item.provenance.source_kind == "code"
            and item.text
            and not _chunk_in_source(item.text, path)
        ):
            # Code chunks are verbatim slices of the file, so absence means the
            # content drifted. Doc chunks may be reflowed by their chunker, so
            # the probe would false-positive there; mtime/existence still apply.
            stale = True
            reasons.append(f"{rel} content no longer contains the indexed chunk")

    return Freshness(
        index_version=index_version,
        indexed_at=indexed_at,
        source_mtime_max=max(mtimes) if mtimes else None,
        is_stale_flag=stale,
        reasons=reasons,
    )


_CHUNK_PROBE_BYTES = 2_000_000  # skip the content probe on very large files


def _chunk_in_source(chunk: str, path: Path) -> bool:
    """Whether the indexed chunk text still appears in the source file.

    Whitespace-normalized so chunkers that reflow lines don't false-positive.
    Errs open (returns True) when the file can't be read as text or is too
    large — the mtime and existence signals still apply.
    """
    try:
        if path.stat().st_size > _CHUNK_PROBE_BYTES:
            return True
        source = path.read_text(errors="strict")
    except (OSError, UnicodeDecodeError):
        return True
    norm = " ".join(chunk.split())
    if not norm:
        return True
    return norm in " ".join(source.split())


def fresh_now(index_version: str) -> Freshness:
    """A trivially-fresh verdict (used when there are no items to assess)."""
    return Freshness(index_version=index_version, indexed_at=now())
