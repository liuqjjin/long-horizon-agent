"""Read the CocoIndex-built JSON chunk records and do cosine vector search.

This is the query path for paper/experiment context. It is deliberately
dependency-light (numpy + the shared embedder) and never runs CocoIndex.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

import numpy as np

from . import embedder as _embedder


class IndexRecordError(RuntimeError):
    """The index exists but cannot be trusted as a complete search corpus."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{path} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def load_records(
    index_dir: Path,
    *,
    expected_records: dict[str, str] | None = None,
) -> list[dict]:
    records: list[dict] = []
    bad: list[str] = []
    paths = sorted(Path(index_dir).glob("*.json"))
    if expected_records is not None and {path.name for path in paths} != set(expected_records):
        raise IndexRecordError("index record set does not match its completion manifest")
    for p in paths:
        try:
            raw = _read_regular_file(p)
            if (
                expected_records is not None
                and hashlib.sha256(raw).hexdigest() != expected_records[p.name]
            ):
                bad.append(f"{p.name}: digest mismatch")
                continue
            record = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            bad.append(f"{p.name}: {type(e).__name__}")
            continue
        if (
            not isinstance(record, dict)
            or not (record.get("source_path") or record.get("locator"))
            or not isinstance(record.get("text"), str)
            or not record.get("text")
            or not isinstance(record.get("embedding"), list)
            or not record.get("embedding")
            or not isinstance(record.get("source_sha256"), str)
            or _SHA256.fullmatch(record["source_sha256"]) is None
            or not isinstance(record.get("chunk_sha256"), str)
            or _SHA256.fullmatch(record["chunk_sha256"]) is None
        ):
            bad.append(f"{p.name}: malformed record")
            continue
        records.append(record)
    if bad:
        raise IndexRecordError("corrupt index record(s): " + "; ".join(bad[:5]))
    return records


def search(
    index_dir: str | Path,
    query: str,
    k: int,
    *,
    metric: str | None = None,
    embedder_model: str = _embedder.DEFAULT_MODEL,
    expected_records: dict[str, str] | None = None,
) -> list[dict]:
    records = load_records(Path(index_dir), expected_records=expected_records)
    if metric:
        filtered = [r for r in records if _has_metric(r, metric)]
        records = filtered
    if not records:
        return []

    record_models = {str(record.get("embedder_model", "legacy")) for record in records}
    if record_models != {embedder_model}:
        raise IndexRecordError(
            f"index model {sorted(record_models)} does not match query model {embedder_model!r}"
        )
    if any(record.get("schema_version") != 3 for record in records):
        raise IndexRecordError("index schema is incompatible; rebuild the index")
    matrix = np.asarray([r["embedding"] for r in records], dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise IndexRecordError("index embeddings are ragged or non-finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-8, None)
    q = _embedder.embed(query, model=embedder_model)
    if q.ndim != 1 or q.shape[0] != matrix.shape[1] or not np.isfinite(q).all():
        raise IndexRecordError(
            f"query embedding shape {q.shape} does not match index width {matrix.shape[1]}"
        )
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
