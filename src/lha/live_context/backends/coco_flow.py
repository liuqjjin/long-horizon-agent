"""Paper / experiment context backed by CocoIndex BUILD flows.

- ``reindex()`` runs the CocoIndex App (incremental + memoized) to (re)build the
  embedded JSON chunk records under ``data/.lha_index/<kind>s/``.
- ``search()`` reads those records and does cosine vector search (no CocoIndex at
  query time).

This module + ``ccc_backend`` are the ONLY places allowed to touch CocoIndex.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ...clock import now
from ..freshness import content_hash
from ..models import (
    ExperimentHit,
    Hit,
    PaperHit,
    Provenance,
    ReindexResult,
    SkillHit,
    SourceKind,
)
from . import vector_query
from .base import SearchBackend


class CocoFlowBackend(SearchBackend):
    def __init__(self, kind: SourceKind, data_dir: str | Path):
        self.name = f"coco:{kind}"
        self.kind = kind
        self.data_dir = Path(data_dir)
        self.sourcedir = self.data_dir / f"{kind}s"
        self.index_dir = self.data_dir / ".lha_index" / f"{kind}s"

    def available(self) -> bool:
        return self.index_dir.exists() and any(self.index_dir.glob("*.json"))

    # --- query path (no CocoIndex) -----------------------------------------
    def search(self, query: str, *, k: int = 5, **filters) -> list[Hit]:
        metric = filters.get("metric")
        _iv, indexed_at = self.index_meta()
        try:
            rows = vector_query.search(self.index_dir, query, k, metric=metric)
        except Exception as e:  # embedder/IO failure is not an empty result
            from .base import BackendUnavailable

            raise BackendUnavailable(f"{self.name} search failed: {type(e).__name__}: {e}") from e
        # Drop malformed records with no resolvable source, so freshness never
        # silently passes on an unknown locator.
        rows = [r for r in rows if r.get("source_path") or r.get("locator")]
        return [self._to_hit(r, indexed_at) for r in rows]

    def _to_hit(self, rec: dict, indexed_at: datetime) -> Hit:
        loc = rec.get("source_path", rec.get("locator", "?"))
        score = float(rec.get("score", 0.0))
        meta = rec.get("metadata", {}) or {}
        prov = Provenance(
            source_kind=self.kind,
            locator=loc,
            score=score,
            indexed_at=indexed_at,
            content_hash=content_hash(rec.get("text", "")),
        )
        if self.kind == "paper":
            return PaperHit(
                text=rec.get("text", ""),
                score=score,
                provenance=prov,
                title=meta.get("title"),
                paper_id=loc,
            )
        if self.kind == "skill":
            return SkillHit(
                text=rec.get("text", ""),
                score=score,
                provenance=prov,
                title=meta.get("title"),
                skill_id=meta.get("skill_id") or loc,  # the originating run_id
            )
        metrics_raw = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
        metrics = {
            str(k): float(v) for k, v in (metrics_raw or {}).items() if isinstance(v, (int, float))
        }
        return ExperimentHit(
            text=rec.get("text", ""),
            score=score,
            provenance=prov,
            run_id=meta.get("run_id"),
            metrics=metrics,
        )

    def index_meta(self) -> tuple[str, datetime]:
        if self.available():
            mtime = max(p.stat().st_mtime for p in self.index_dir.glob("*.json"))
            return (
                f"coco:{self.kind}@{int(mtime)}",
                datetime.fromtimestamp(mtime, tz=timezone.utc),
            )
        return (f"coco:{self.kind}@uninitialized", now())

    # --- build path (runs CocoIndex) ---------------------------------------
    def reindex(self, paths: list[str] | None = None) -> ReindexResult:
        version_before, _ = self.index_meta()
        try:
            import cocoindex as coco

            state_dir = (self.data_dir / ".lha_index").resolve()
            state_dir.mkdir(parents=True, exist_ok=True)
            os.environ["COCOINDEX_DB"] = str(state_dir / "state.db")

            app = self._build_app()
            with coco.runtime():
                app.update_blocking(report_to_stdout=False)
        except ImportError as e:  # optional dependency not installed
            return ReindexResult(
                kind=self.kind, ok=False, version_before=version_before,
                detail=f"cocoindex is not installed (pip install 'lha[context]'): {e}",
            )
        except Exception as e:  # flow error, IO — all mean "not refreshed"
            return ReindexResult(
                kind=self.kind, ok=False, version_before=version_before,
                detail=f"cocoindex flow failed: {type(e).__name__}: {e}",
            )
        version_after, _ = self.index_meta()
        return ReindexResult(
            kind=self.kind, ok=True, version_before=version_before, version_after=version_after
        )

    def _build_app(self):
        _ensure_flows_importable()
        if self.kind == "paper":
            from flows.papers_flow import build
        elif self.kind == "skill":
            from flows.skills_flow import build
        else:
            from flows.experiments_flow import build
        return build(str(self.sourcedir), str(self.index_dir))


def _ensure_flows_importable() -> None:
    """The top-level ``flows/`` package isn't installed; add the repo root to
    sys.path (computed from this file, with cwd as fallback)."""
    here = Path(__file__).resolve()
    candidates = []
    if len(here.parents) >= 5:
        candidates.append(here.parents[4])  # backends/live_context/lha/src/<root>
    candidates.append(Path.cwd())
    for root in candidates:
        if (root / "flows" / "__init__.py").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return
