"""Paper/experiment query path (vector search over CocoIndex-built records).

Hermetic: stubs the embedder and uses synthetic records, so no model load / no
CocoIndex run is needed.
"""

from __future__ import annotations

import json

import numpy as np

from lha.live_context.backends import vector_query
from lha.live_context.backends.coco_flow import CocoFlowBackend


def test_vector_query_ranks_by_cosine(monkeypatch, tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"kind": "paper", "source_path": "p/a.md", "text": "alpha",
                    "embedding": [1.0, 0.0], "metadata": {"title": "A"}})
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"kind": "paper", "source_path": "p/b.md", "text": "beta",
                    "embedding": [0.0, 1.0], "metadata": {"title": "B"}})
    )
    monkeypatch.setattr(
        vector_query._embedder, "embed", lambda text, model=None: np.array([1.0, 0.0], dtype=np.float32)
    )
    rows = vector_query.search(tmp_path, "q", k=2)
    assert [r["source_path"] for r in rows] == ["p/a.md", "p/b.md"]
    assert rows[0]["score"] > rows[1]["score"]


def test_cocoflow_to_hit_carries_provenance(monkeypatch, tmp_path):
    (tmp_path / ".lha_index" / "papers").mkdir(parents=True)
    rec = {"kind": "paper", "source_path": "data/papers/x.md", "text": "hello",
           "embedding": [1.0, 0.0], "metadata": {"title": "X"}}
    (tmp_path / ".lha_index" / "papers" / "p__x__0.json").write_text(json.dumps(rec))

    monkeypatch.setattr(
        vector_query._embedder, "embed", lambda text, model=None: np.array([1.0, 0.0], dtype=np.float32)
    )
    be = CocoFlowBackend("paper", tmp_path)
    assert be.available()
    hits = be.search("hello", k=1)
    assert len(hits) == 1
    assert hits[0].provenance.locator == "data/papers/x.md"
    assert hits[0].provenance.source_kind == "paper"
    assert hits[0].title == "X"


def test_cocoflow_empty_is_graceful(tmp_path):
    be = CocoFlowBackend("experiment", tmp_path)
    assert not be.available()
    assert be.search("anything") == []
