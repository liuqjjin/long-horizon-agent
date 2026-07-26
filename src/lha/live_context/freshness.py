"""Freshness tracking: decide whether indexed context still reflects the source.

Signals actually checked by ``assess`` (cheap -> strong):
  1. file mtime vs. the chunk's ``indexed_at``;
  2. the complete source-file digest recorded by document indexes;
  3. for code chunks (verbatim file slices): chunk text still present in the
     current source — content drift that mtime alone can miss;
  4. a source file that no longer exists is stale, not silently skipped.

The backend index generation (``index_version``) is carried on the verdict for
comparison across reindexes; ``content_hash`` on provenance records what was
indexed.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

from ..clock import now
from .models import ContextItem, Freshness

_LINES_SUFFIX = re.compile(r":\d+(?:-\d+)?$")


def content_hash(text: str) -> str:
    """sha256 of a chunk of text (hex)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strict_file_sha256(path: Path, *, root: Path | None = None) -> str:
    """Hash exact bytes without following a source-file symlink.

    When ``root`` is supplied, every path component below that canonical root
    is checked before opening the file.  This is the provenance path used by
    the code backend: a locator may not escape the indexed repository through
    ``..`` or an intermediate symlink.
    """
    candidate = Path(path)
    if root is not None:
        canonical_root = Path(root).resolve(strict=True)
        if candidate.is_absolute():
            try:
                relative = candidate.relative_to(canonical_root)
            except ValueError as error:
                raise OSError(f"source is outside indexed root: {candidate}") from error
        else:
            relative = candidate
        if relative.is_absolute() or ".." in relative.parts:
            raise OSError(f"unsafe source path: {candidate}")
        candidate = canonical_root
        for part in relative.parts:
            if part in ("", "."):
                continue
            candidate = candidate / part
            if stat.S_ISLNK(candidate.lstat().st_mode):
                raise OSError(f"source path contains a symlink: {candidate}")
    elif candidate.is_symlink():
        raise OSError(f"source is a symlink: {candidate}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(candidate, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError(f"source is not a regular file: {candidate}")
        digest = hashlib.sha256()
        while block := os.read(fd, 1024 * 1024):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(fd)


def file_sha256(path: Path, *, root: Path | None = None) -> str | None:
    """Return the SHA-256 of exact source bytes, or ``None`` when unsafe/unreadable."""
    try:
        return strict_file_sha256(path, root=root)
    except (OSError, RuntimeError):
        return None


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
        source_root = item.provenance.source_root
        if p.is_absolute():
            path = p
        elif source_root:
            path = Path(source_root) / rel
        else:
            path = base_dir / rel
        if (
            item.provenance.content_hash is not None
            and content_hash(item.text) != item.provenance.content_hash
        ):
            stale = True
            reasons.append(f"{rel} indexed chunk does not match its recorded content hash")
        mtime = file_mtime(path)
        if mtime is None:
            # A source that disappeared is stale context, not ignorable context.
            stale = True
            reasons.append(f"{rel} no longer exists (or is unreadable)")
            continue
        mtimes.append(mtime)
        source_sha256 = item.provenance.source_sha256
        if source_sha256 is not None:
            current_sha256 = file_sha256(
                path,
                root=Path(source_root) if source_root is not None else None,
            )
            if current_sha256 is None:
                stale = True
                reasons.append(f"{rel} source cannot be read safely for SHA-256 verification")
            elif current_sha256 != source_sha256:
                stale = True
                reasons.append(f"{rel} source bytes do not match the indexed SHA-256")
        if mtime > item.provenance.indexed_at:
            stale = True
            reasons.append(f"{rel} modified after it was indexed")
        elif (
            item.provenance.source_kind == "code"
            and source_sha256 is None
            and item.text
        ):
            chunk_present, probe_error = _chunk_in_source(item.text, path)
            if not chunk_present:
                # Legacy code hits without a complete source digest still use a
                # content probe.  Inability to perform that weaker check is a
                # stale verdict, never an assumed match.
                stale = True
                reasons.append(
                    f"{rel} {probe_error or 'content no longer contains the indexed chunk'}"
                )

    return Freshness(
        index_version=index_version,
        indexed_at=indexed_at,
        source_mtime_max=max(mtimes) if mtimes else None,
        is_stale_flag=stale,
        reasons=reasons,
    )


_CHUNK_PROBE_BYTES = 2_000_000  # skip the content probe on very large files


def _chunk_in_source(chunk: str, path: Path) -> tuple[bool, str | None]:
    """Whether the indexed chunk text still appears in the source file.

    Whitespace-normalized so chunkers that reflow lines don't false-positive.
    This is only a compatibility path for legacy code hits that lack a complete
    source digest, so an unreadable, symlinked, or oversized source fails closed.
    """
    try:
        if path.is_symlink():
            return False, "source is a symlink and cannot be probed safely"
        if path.stat().st_size > _CHUNK_PROBE_BYTES:
            return False, "source is too large to verify without a full SHA-256"
        source = path.read_text(errors="strict")
    except (OSError, UnicodeDecodeError):
        return False, "source cannot be read for content verification"
    norm = " ".join(chunk.split())
    if not norm:
        return True, None
    return norm in " ".join(source.split()), None


def fresh_now(index_version: str) -> Freshness:
    """A trivially-fresh verdict (used when there are no items to assess)."""
    return Freshness(index_version=index_version, indexed_at=now())
