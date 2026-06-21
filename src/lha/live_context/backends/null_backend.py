"""A backend that returns nothing.

Used for hermetic tests and as a graceful fallback when a real backend (e.g.
``ccc``) is not available, so the loop still runs (just with empty context).
"""

from __future__ import annotations

from datetime import datetime

from ...clock import now
from ..models import Hit, SourceKind
from .base import SearchBackend


class NullBackend(SearchBackend):
    def __init__(self, kind: SourceKind = "code"):
        self.name = f"null:{kind}"
        self.kind = kind
        self._created = now()

    def available(self) -> bool:
        return True

    def search(self, query: str, *, k: int = 8, **filters) -> list[Hit]:
        return []

    def index_meta(self) -> tuple[str, datetime]:
        return ("null-0", self._created)
