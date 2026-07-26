"""The live-context facade — the ONLY public door to code/paper/experiment search.

The rest of the system imports exactly these five functions (plus the models and
``configure``). It must never import CocoIndex or ``ccc`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from ..clock import now
from ..config import Config
from . import freshness as _freshness
from .backends.base import BackendUnavailable, SearchBackend
from .backends.ccc_backend import CccBackend
from .backends.coco_flow import CocoFlowBackend
from .backends.null_backend import NullBackend
from .models import (
    CodeHit,
    ContextBundle,
    ContextItem,
    ContextStatus,
    ExperimentHit,
    PaperHit,
    ReindexResult,
    SkillHit,
    SourceKind,
)


class StaleContextError(RuntimeError):
    """Raised by ``reject_stale`` when stale context cannot be refreshed —
    either reindexing is disabled or the reindex itself failed."""


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


def index_code(path: str | Path) -> ReindexResult:
    """Build/refresh the code index for ``path`` and point the facade at it.

    Used to prime a run sandbox before the Context Engineer searches it.
    Returns the structured reindex outcome; callers that require fresh context
    must check ``.ok`` rather than assume success.
    """
    configure(code_root=path)
    result = _code_backend().reindex()
    _state.reset_cache()
    return result


def index_docs(kinds: tuple[SourceKind, ...] = ("paper", "experiment", "skill")) -> list[ReindexResult]:
    """Run the CocoIndex BUILD flows to (re)index paper/experiment/skill notes."""
    results: list[ReindexResult] = []
    for kind in kinds:
        sourcedir = _state.config.data_dir / f"{kind}s"
        if not sourcedir.exists():
            continue  # nothing to index for this kind yet
        results.append(
            CocoFlowBackend(
                kind,
                _state.config.data_dir,
                embedder_model=_state.config.embedder_model,
            ).reindex()
        )
    _state.reset_cache()
    return results


def _backend_for(kind: SourceKind) -> SearchBackend:
    if kind == "code":
        return _code_backend()
    key = f"{kind}"
    if key not in _state._cache:
        be = CocoFlowBackend(
            kind,
            _state.config.data_dir,
            embedder_model=_state.config.embedder_model,
        )
        # Persisted-but-incomplete/corrupt generations must reach ``search`` so
        # they are classified as ``index_failed``.  Only a genuinely
        # uninitialized optional index maps to an unavailable null backend.
        _state._cache[key] = be if be.has_index_state() else NullBackend(kind)
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


def search_experiments(query: str, *, k: int = 5, metric: str | None = None) -> list[ExperimentHit]:
    hits = _backend_for("experiment").search(query, k=k, metric=metric)
    return [h if isinstance(h, ExperimentHit) else ExperimentHit(**h.model_dump()) for h in hits]


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
    """Multi-source search + provenance + freshness + availability in one bundle.

    The bundle's ``status`` distinguishes "searched and found nothing" (``empty``)
    from "could not search" (``backend_unavailable``) — the two must never be
    conflated, or missing infrastructure reads as verified-empty context.
    """
    requested_kinds = tuple(dict.fromkeys(kinds))
    items: list[ContextItem] = []
    versions: list[str] = []
    indexed_ats = []
    notes: list[str] = []
    unavailable_kinds: list[SourceKind] = []
    unavailable_reasons: dict[SourceKind, str] = {}
    failed_kinds: list[SourceKind] = []
    failure_reasons: dict[SourceKind, str] = {}
    for kind in requested_kinds:
        be = _backend_for(kind)
        if isinstance(be, NullBackend):
            unavailable_kinds.append(kind)
            unavailable_reasons[kind] = "no backend available"
            notes.append(f"{kind}: {unavailable_reasons[kind]}")
            continue
        try:
            hits = be.search(query, k=k)
        except BackendUnavailable as e:
            unavailable_kinds.append(kind)
            unavailable_reasons[kind] = str(e)
            notes.append(f"{kind}: {unavailable_reasons[kind]}")
            continue
        except Exception as e:
            failed_kinds.append(kind)
            failure_reasons[kind] = f"{type(e).__name__}: {e}"
            notes.append(f"{kind} index query failed: {failure_reasons[kind]}")
            continue
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
    if failed_kinds:
        freshness.is_stale_flag = True
        freshness.reasons.extend(
            f"{kind} index cannot be matched to its current sources"
            for kind in failed_kinds
        )
    if max_age_s is not None and freshness.is_stale(max_age_s) and not freshness.is_stale_flag:
        freshness.is_stale_flag = True
        freshness.reasons.append(f"older than max_age_s={max_age_s}")
    status: ContextStatus = "ok"
    if failed_kinds:
        status = "index_failed"
    elif not items:
        status = "backend_unavailable" if unavailable_kinds else "empty"
    return ContextBundle(
        query=query,
        items=items,
        freshness=freshness,
        status=status,
        status_notes=notes,
        unavailable_kinds=unavailable_kinds,
        unavailable_reasons=unavailable_reasons,
        failed_kinds=failed_kinds,
        failure_reasons=failure_reasons,
        requested_kinds=list(requested_kinds),
    )


def reject_stale(bundle: ContextBundle, *, reindex: bool = True) -> ContextBundle:
    """Refresh a stale bundle: incrementally re-index its sources and re-search.

    Fails closed: if any source's reindex does not verifiably succeed, this
    raises ``StaleContextError`` and the bundle stays stale — a failed refresh
    must never clear the stale flag.
    """
    if not bundle.freshness.is_stale():
        return bundle
    if not reindex:
        raise StaleContextError(
            f"context for {bundle.query!r} is stale: {bundle.freshness.reasons}"
        )
    inferred: list[SourceKind] = [i.provenance.source_kind for i in bundle.items]
    inferred.extend(
        cast(SourceKind, k)
        for k in bundle.unavailable_kinds
        if k in ("code", "paper", "experiment", "skill")
    )
    kinds: tuple[SourceKind, ...] = tuple(
        dict.fromkeys(bundle.requested_kinds or inferred)
    ) or ("code",)
    results: list[ReindexResult] = []
    for kind in kinds:
        try:
            results.append(_backend_for(kind).reindex())
        except Exception as error:
            results.append(
                ReindexResult(
                    kind=kind,
                    ok=False,
                    detail=f"{type(error).__name__}: {error}",
                )
            )
    # Skill memory is an optional augmentation (matching FreshnessVerifier).
    # Still attempt its refresh, but a dark memory index must not block otherwise
    # valid task context.
    failed = [r for r in results if not r.ok and r.kind != "skill"]
    if failed:
        raise StaleContextError(
            "stale context could not be refreshed: "
            + "; ".join(f"{r.kind}: {r.detail}" for r in failed)
        )
    _state.reset_cache()
    k = max(len(bundle.items), 1)
    refreshed = get_fresh_context(bundle.query, kinds=kinds, k=k)
    blocking_unavailable = [k for k in refreshed.unavailable_kinds if k != "skill"]
    if blocking_unavailable or refreshed.status == "index_failed":
        detail = "; ".join(refreshed.status_notes) or ", ".join(blocking_unavailable)
        raise StaleContextError(f"refreshed context is still unavailable: {detail}")
    if refreshed.freshness.is_stale():
        raise StaleContextError(
            f"refreshed context is still stale: {refreshed.freshness.reasons}"
        )
    return refreshed
