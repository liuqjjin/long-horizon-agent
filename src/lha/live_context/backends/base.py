"""Backend abstraction. Nothing outside ``live_context`` imports these.

A backend knows how to (a) search a corpus and (b) report/refresh its index.
The facade composes one or more backends; swapping CocoIndex internals or the
``ccc`` access path means changing only a backend, never a caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..models import Hit, ReindexResult, SourceKind


class BackendUnavailable(RuntimeError):
    """The backend could not serve the request (process/index failure).

    Distinct from an empty result: a caller must be able to tell "searched and
    found nothing" from "could not search at all", or empty context silently
    reads as verified context.
    """


class SearchBackend(ABC):
    name: str = "base"
    kind: SourceKind = "code"

    def available(self) -> bool:
        """Whether this backend can serve requests right now."""
        return True

    @abstractmethod
    def search(self, query: str, *, k: int = 8, **filters) -> list[Hit]:
        """Search the corpus. Raises ``BackendUnavailable`` on backend failure
        (never returns ``[]`` for a failure)."""

    @abstractmethod
    def index_meta(self) -> tuple[str, datetime]:
        """Return ``(index_version, indexed_at)`` for freshness accounting."""

    def reindex(self, paths: list[str] | None = None) -> ReindexResult:
        """Refresh the index and report whether it actually succeeded.

        The default is an honest "cannot reindex": callers that need a fresh
        index (``reject_stale``) fail closed on ``ok=False`` instead of assuming
        the refresh happened.
        """
        return ReindexResult(kind=self.kind, ok=False, detail=f"{self.name} cannot reindex")
