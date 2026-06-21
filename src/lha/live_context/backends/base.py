"""Backend abstraction. Nothing outside ``live_context`` imports these.

A backend knows how to (a) search a corpus and (b) report/refresh its index.
The facade composes one or more backends; swapping CocoIndex internals or the
``ccc`` access path means changing only a backend, never a caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..models import Hit, SourceKind


class SearchBackend(ABC):
    name: str = "base"
    kind: SourceKind = "code"

    def available(self) -> bool:
        """Whether this backend can serve requests right now."""
        return True

    @abstractmethod
    def search(self, query: str, *, k: int = 8, **filters) -> list[Hit]: ...

    @abstractmethod
    def index_meta(self) -> tuple[str, datetime]:
        """Return ``(index_version, indexed_at)`` for freshness accounting."""

    def reindex(self, paths: list[str] | None = None) -> None:
        """Incrementally refresh the index (default: no-op)."""
        return None
