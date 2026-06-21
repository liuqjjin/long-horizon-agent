"""Read the CocoIndex-built JSON chunk records and do cosine vector search.

This is the query path for paper/experiment context. It is deliberately
dependency-light (numpy + the shared embedder) and never runs CocoIndex.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import embedder as _embedder


def load_records(index_dir: Path) -> list[dict]:
    records: list[dict] = []
    for p in sorted(Path(index_dir).glob("*.json")):
        try:
            records.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def search(index_dir: str | Path, query: str, k: int, *, metric: str | None = None) -> list[dict]:
    records = load_records(Path(index_dir))
    if metric:
        filtered = [r for r in records if _has_metric(r, metric)]
        if filtered:
            records = filtered
    if not records:
        return []

    matrix = np.asarray([r["embedding"] for r in records], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-8, None)
    q = _embedder.embed(query)
    sims = matrix @ q

    order = np.argsort(-sims)[: max(k, 0)]
    out: list[dict] = []
    for i in order:
        rec = dict(records[int(i)])
        rec["score"] = float(sims[int(i)])
        out.append(rec)
    return out


def _has_metric(record: dict, metric: str) -> bool:
    meta = record.get("metadata", {}) or {}
    metrics = meta.get("metrics")
    keys = set()
    if isinstance(metrics, dict):
        keys = {str(x).lower() for x in metrics.keys()}
    elif isinstance(metrics, list):
        keys = {str(x).lower() for x in metrics}
    return metric.lower() in keys
