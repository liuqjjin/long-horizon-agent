"""Live-context facade: the only public door to code/paper/experiment search.

Import from here, never from ``cocoindex`` / ``cocoindex_code`` / ``ccc``.
"""

from .backends.base import BackendUnavailable
from .facade import (
    StaleContextError,
    configure,
    get_fresh_context,
    index_code,
    index_docs,
    reject_stale,
    search_code,
    search_experiments,
    search_papers,
    search_skills,
)
from .models import (
    CodeHit,
    ContextBundle,
    ContextItem,
    ContextStatus,
    ExperimentHit,
    Freshness,
    PaperHit,
    Provenance,
    ReindexResult,
    SkillHit,
)

__all__ = [
    "search_code",
    "search_papers",
    "search_experiments",
    "search_skills",
    "get_fresh_context",
    "reject_stale",
    "configure",
    "index_code",
    "index_docs",
    "StaleContextError",
    "BackendUnavailable",
    "ContextStatus",
    "ReindexResult",
    "CodeHit",
    "PaperHit",
    "ExperimentHit",
    "SkillHit",
    "ContextItem",
    "ContextBundle",
    "Provenance",
    "Freshness",
]
