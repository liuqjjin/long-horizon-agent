"""Public data models for the live-context facade.

These carry *provenance* and *freshness* so every downstream claim is
traceable to a source and a point in time. Nothing here knows about CocoIndex.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..clock import now

SourceKind = Literal["code", "paper", "experiment", "skill"]


class Provenance(BaseModel):
    """Where a piece of context came from and when it was indexed."""

    source_kind: SourceKind
    locator: str  # e.g. "src/mathutils.py:10-14" or "papers/note_srgan.md" or "exp_0001"
    score: float = 0.0  # similarity score from the search backend
    indexed_at: datetime = Field(default_factory=now)
    content_hash: str | None = None  # sha256 of the chunk at index time


class Hit(BaseModel):
    """A single search result."""

    text: str
    provenance: Provenance
    score: float = 0.0


class CodeHit(Hit):
    language: str | None = None
    line_start: int | None = None
    line_end: int | None = None


class PaperHit(Hit):
    title: str | None = None
    paper_id: str | None = None


class ExperimentHit(Hit):
    run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class SkillHit(Hit):
    """A retrieved past success/skill (episodic memory)."""

    title: str | None = None
    skill_id: str | None = None


class Freshness(BaseModel):
    """A freshness verdict for a bundle of context items."""

    index_version: str
    indexed_at: datetime
    source_mtime_max: datetime | None = None
    is_stale_flag: bool = False
    reasons: list[str] = Field(default_factory=list)

    def is_stale(self, max_age_s: float | None = None) -> bool:
        if self.is_stale_flag:
            return True
        if max_age_s is not None:
            age = (now() - self.indexed_at).total_seconds()
            if age > max_age_s:
                return True
        return False


class ContextItem(BaseModel):
    """One unit of context the rest of the system reasons over."""

    text: str
    provenance: Provenance

    @classmethod
    def from_hit(cls, hit: Hit) -> "ContextItem":
        return cls(text=hit.text, provenance=hit.provenance)


class ContextBundle(BaseModel):
    """The Context Engineer's structured artifact: items + freshness + answer."""

    query: str
    items: list[ContextItem] = Field(default_factory=list)
    freshness: Freshness
    answer: str | None = None

    def locators(self) -> list[str]:
        return [i.provenance.locator for i in self.items]
