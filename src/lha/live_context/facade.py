"""The live-context facade — the ONLY public door to code/paper/experiment search.

The rest of the system imports exactly these five functions (plus the models and
``configure``). It must never import CocoIndex or ``ccc`` directly.
"""

from __future__ import annotations

from pathlib import Path

from ..clock import now
from ..config import Config
from . import freshness as _freshness
from .backends.base import SearchBackend
from .backends.ccc_backend import CccBackend
from .backends.coco_flow import CocoFlowBackend
from .backends.null_backend import NullBackend
from .models import (
    CodeHit,
    ContextBundle,
    ContextItem,
    ExperimentHit,
    PaperHit,
    SkillHit,
    SourceKind,
)


class StaleContextError(RuntimeError):
    """Raised by ``reject_stale`` when context is stale and reindex is disabled."""


class _FacadeState:
    def __init__(self) -> None:
        self.config = Config()
        self.code_root = Path.cwd()
        self._cache: dict[str, SearchBackend] = {}

    def reset_cache(self) -> None:
        self._cache.clear()


_state = _FacadeState()


def configure(*, code_root: str | Path | None = None, config: Config | None = None) -> None:
    """Point the facade at a code root (the repo being searched) and/or config."""
    if config is not None:
        _state.config = config
    if code_root is not None:
        _state.code_root = Path(code_root)
    _state.reset_cache()


def _code_backend() -> SearchBackend:
    key = f"code:{_state.code_root}"
    if key in _state._cache:
        return _state._cache[key]
    mode = _state.config.code_backend
    backend: SearchBackend
    if mode in ("auto", "ccc"):
        candidate = CccBackend(_state.code_root)
        backend = candidate if candidate.available() else NullBackend("code")
        if mode == "ccc":  # explicit request: use it even if not yet indexed
            backend = candidate
    else:
        backend = NullBackend("code")
    _state._cache[key] = backend
    return backend


def index_code(path: str | Path) -> None:
    """Build/refresh the code index for ``path`` and point the facade at it.

    Used to prime a run sandbox before the Context Engineer searches it.
    """
    configure(code_root=path)
    _code_backend().reindex()
    _state.reset_cache()


def index_docs(kinds: tuple[SourceKind, ...] = ("paper", "experiment", "skill")) -> None:
    """Run the CocoIndex BUILD flows to (re)index paper/experiment/skill notes."""
    for kind in kinds:
        sourcedir = _state.config.data_dir / f"{kind}s"
        if not sourcedir.exists():
            continue  # nothing to index for this kind yet
        CocoFlowBackend(kind, _state.config.data_dir).reindex()
    _state.reset_cache()


def _backend_for(kind: SourceKind) -> SearchBackend:
    if kind == "code":
        return _code_backend()
    key = f"{kind}"
    if key not in _state._cache:
        be = CocoFlowBackend(kind, _state.config.data_dir)
        _state._cache[key] = be if be.available() else NullBackend(kind)
    return _state._cache[key]


# --- the five public functions ---------------------------------------------
def search_code(
    query: str,
    *,
    k: int = 8,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[CodeHit]:
    hits = _backend_for("code").search(query, k=k, languages=languages, paths=paths)
    return [h for h in hits if isinstance(h, CodeHit)]


def search_papers(query: str, *, k: int = 5) -> list[PaperHit]:
    hits = _backend_for("paper").search(query, k=k)
    return [h if isinstance(h, PaperHit) else PaperHit(**h.model_dump()) for h in hits]


def search_experiments(
    query: str, *, k: int = 5, metric: str | None = None
) -> list[ExperimentHit]:
    hits = _backend_for("experiment").search(query, k=k, metric=metric)
    return [
        h if isinstance(h, ExperimentHit) else ExperimentHit(**h.model_dump()) for h in hits
    ]


def search_skills(query: str, *, k: int = 5) -> list[SkillHit]:
    """Retrieve past successes/skills (episodic memory)."""
    hits = _backend_for("skill").search(query, k=k)
    return [h if isinstance(h, SkillHit) else SkillHit(**h.model_dump()) for h in hits]


def get_fresh_context(
    query: str,
    *,
    kinds: tuple[SourceKind, ...] = ("code", "paper", "experiment"),
    k: int = 8,
    max_age_s: float | None = None,
) -> ContextBundle:
    """Multi-source search + provenance + freshness in one bundle."""
    items: list[ContextItem] = []
    versions: list[str] = []
    indexed_ats = []
    for kind in kinds:
        be = _backend_for(kind)
        hits = be.search(query, k=k)
        # Only backends that actually contributed context affect freshness, so an
        # empty/uninitialized backend can't skew the bundle's indexed_at.
        if hits:
            iv, ia = be.index_meta()
            versions.append(iv)
            indexed_ats.append(ia)
        items.extend(ContextItem.from_hit(h) for h in hits)

    indexed_at = min(indexed_ats) if indexed_ats else now()
    freshness = _freshness.assess(
        items,
        index_version=";".join(versions) or "empty",
        indexed_at=indexed_at,
        base_dir=Path.cwd(),
    )
    if max_age_s is not None and freshness.is_stale(max_age_s) and not freshness.is_stale_flag:
        freshness.is_stale_flag = True
        freshness.reasons.append(f"older than max_age_s={max_age_s}")
    return ContextBundle(query=query, items=items, freshness=freshness)


def reject_stale(bundle: ContextBundle, *, reindex: bool = True) -> ContextBundle:
    """Refresh a stale bundle: incrementally re-index its sources and re-search."""
    if not bundle.freshness.is_stale():
        return bundle
    if not reindex:
        raise StaleContextError(
            f"context for {bundle.query!r} is stale: {bundle.freshness.reasons}"
        )
    kinds = tuple(dict.fromkeys(i.provenance.source_kind for i in bundle.items)) or ("code",)
    for kind in kinds:
        _backend_for(kind).reindex()
    _state.reset_cache()
    k = max(len(bundle.items), 1)
    refreshed = get_fresh_context(bundle.query, kinds=kinds, k=k)
    # We just reindexed, so the bundle is fresh by construction; mtime lag (e.g. a
    # touched-but-unchanged source that memoization skipped) must not re-flag it.
    refreshed.freshness.is_stale_flag = False
    refreshed.freshness.reasons = []
    refreshed.freshness.indexed_at = now()
    return refreshed
