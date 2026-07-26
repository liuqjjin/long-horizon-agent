"""Paper/experiment query path (vector search over CocoIndex-built records).

Hermetic: stubs the embedder and uses synthetic records, so no model load / no
CocoIndex run is needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from lha.agents.context_engineer import ContextEngineer
from lha.clock import now
from lha.live_context import freshness as fr
from lha.live_context.backends import vector_query
from lha.live_context.backends.base import BackendUnavailable
from lha.live_context.backends.coco_flow import CocoFlowBackend
from lha.live_context.flows._coco_impl import chunk_evidence
from lha.live_context.models import ContextBundle, ContextItem, Provenance, ReindexResult

_SOURCE_SHA = "a" * 64


def test_vector_query_ranks_by_cosine(monkeypatch, tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "kind": "paper",
                "source_path": "p/a.md",
                "text": "alpha",
                "schema_version": 3,
                "embedder_model": vector_query._embedder.DEFAULT_MODEL,
                "embedding_dimension": 2,
                "embedding": [1.0, 0.0],
                "source_sha256": _SOURCE_SHA,
                "chunk_sha256": fr.content_hash("alpha"),
                "metadata": {"title": "A"},
            }
        )
    )
    (tmp_path / "b.json").write_text(
        json.dumps(
            {
                "kind": "paper",
                "source_path": "p/b.md",
                "text": "beta",
                "schema_version": 3,
                "embedder_model": vector_query._embedder.DEFAULT_MODEL,
                "embedding_dimension": 2,
                "embedding": [0.0, 1.0],
                "source_sha256": _SOURCE_SHA,
                "chunk_sha256": fr.content_hash("beta"),
                "metadata": {"title": "B"},
            }
        )
    )
    monkeypatch.setattr(
        vector_query._embedder,
        "embed",
        lambda text, model=None: np.array([1.0, 0.0], dtype=np.float32),
    )
    rows = vector_query.search(tmp_path, "q", k=2)
    assert [r["source_path"] for r in rows] == ["p/a.md", "p/b.md"]
    assert rows[0]["score"] > rows[1]["score"]


def test_vector_query_metric_filter_does_not_fall_back_to_unrelated_records(
    monkeypatch, tmp_path
):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "kind": "experiment",
                "source_path": "experiments/a.md",
                "text": "only SSIM",
                "schema_version": 3,
                "embedder_model": vector_query._embedder.DEFAULT_MODEL,
                "embedding_dimension": 2,
                "embedding": [1.0, 0.0],
                "source_sha256": _SOURCE_SHA,
                "chunk_sha256": fr.content_hash("only SSIM"),
                "metadata": {"metrics": {"ssim": 0.9}},
            }
        )
    )
    monkeypatch.setattr(
        vector_query._embedder,
        "embed",
        lambda text, model=None: np.array([1.0, 0.0], dtype=np.float32),
    )
    assert vector_query.search(tmp_path, "q", k=2, metric="psnr") == []


def test_cocoflow_to_hit_carries_provenance(monkeypatch, tmp_path):
    source_dir = tmp_path / "papers"
    source_dir.mkdir()
    source = source_dir / "x.md"
    source.write_text("hello")
    (tmp_path / ".lha_index" / "papers").mkdir(parents=True)
    chunk = chunk_evidence(b"hello")[0]
    rec = {
        "kind": "paper",
        "source_path": "x.md",
        "text": chunk["text"],
        "schema_version": 3,
        "embedder_model": vector_query._embedder.DEFAULT_MODEL,
        "embedding_dimension": 2,
        "embedding": [1.0, 0.0],
        "source_sha256": fr.file_sha256(source),
        "chunk_sha256": chunk["chunk_sha256"],
        "chunk_index": chunk["chunk_index"],
        "chunk_start": chunk["chunk_start"],
        "chunk_end": chunk["chunk_end"],
        "metadata": {"title": "X"},
    }
    (tmp_path / ".lha_index" / "papers" / "p__x__0.json").write_text(json.dumps(rec))

    monkeypatch.setattr(
        vector_query._embedder,
        "embed",
        lambda text, model=None: np.array([1.0, 0.0], dtype=np.float32),
    )
    be = CocoFlowBackend("paper", tmp_path)
    be._seal_generation(be.index_dir, generation_id="test-generation")
    assert be.available()
    hits = be.search("hello", k=1)
    assert len(hits) == 1
    assert hits[0].provenance.locator == "x.md"
    assert hits[0].provenance.source_kind == "paper"
    assert hits[0].provenance.source_sha256 == fr.file_sha256(source)
    assert hits[0].title == "X"


def test_cocoflow_empty_is_graceful(tmp_path):
    be = CocoFlowBackend("experiment", tmp_path)
    assert not be.available()
    with pytest.raises(BackendUnavailable, match="no completed"):
        be.search("anything")

    be.sourcedir.mkdir()
    be.index_dir.mkdir(parents=True)
    be._seal_generation(be.index_dir, generation_id="empty-generation")
    assert be.available()
    assert be.search("anything") == []


def _write_complete_generation(tmp_path, *, records: int = 2) -> CocoFlowBackend:
    backend = CocoFlowBackend("paper", tmp_path)
    backend.sourcedir.mkdir(parents=True)
    backend.index_dir.mkdir(parents=True)
    source = backend.sourcedir / "x.md"
    raw = ("word " * 250).encode()
    source.write_bytes(raw)
    source_sha = fr.file_sha256(source)
    chunks = chunk_evidence(raw)
    assert len(chunks) == records
    for chunk in chunks:
        index = int(chunk["chunk_index"])
        text = str(chunk["text"])
        record = {
            "kind": "paper",
            "source_path": "x.md",
            "text": text,
            "schema_version": 3,
            "embedder_model": vector_query._embedder.DEFAULT_MODEL,
            "embedding_dimension": 2,
            "embedding": [1.0, 0.0],
            "source_sha256": source_sha,
            "chunk_sha256": chunk["chunk_sha256"],
            "chunk_index": index,
            "chunk_start": chunk["chunk_start"],
            "chunk_end": chunk["chunk_end"],
            "metadata": {},
        }
        (backend.index_dir / f"paper__x__{index}.json").write_text(json.dumps(record))
    backend._seal_generation(backend.index_dir, generation_id="complete-generation")
    assert backend.available()
    return backend


def test_cocoflow_seal_rejects_a_missing_source_generation(tmp_path):
    backend = CocoFlowBackend("paper", tmp_path)
    backend.sourcedir.mkdir(parents=True)
    backend.index_dir.mkdir(parents=True)
    first = backend.sourcedir / "a.md"
    second = backend.sourcedir / "b.md"
    first.write_text("alpha")
    second.write_text("beta")
    chunk = chunk_evidence(first.read_bytes())[0]
    record = {
        "kind": "paper",
        "source_path": "a.md",
        "text": chunk["text"],
        "schema_version": 3,
        "embedder_model": vector_query._embedder.DEFAULT_MODEL,
        "embedding_dimension": 2,
        "embedding": [1.0, 0.0],
        "source_sha256": fr.file_sha256(first),
        "chunk_sha256": chunk["chunk_sha256"],
        "chunk_index": chunk["chunk_index"],
        "chunk_start": chunk["chunk_start"],
        "chunk_end": chunk["chunk_end"],
        "metadata": {},
    }
    (backend.index_dir / "paper__a__0.json").write_text(json.dumps(record))

    with pytest.raises(vector_query.IndexRecordError, match="partial source index"):
        backend._seal_generation(backend.index_dir, generation_id="partial")
    assert not (backend.index_dir / ".complete").exists()


def test_cocoflow_seal_rejects_nonfinite_embedding(tmp_path):
    backend = _write_complete_generation(tmp_path)
    (backend.index_dir / ".complete").unlink()
    record_path = backend.index_dir / "paper__x__0.json"
    record = json.loads(record_path.read_text())
    record["embedding"] = [float("nan"), 0.0]
    record_path.write_text(json.dumps(record))
    with pytest.raises(vector_query.IndexRecordError, match="schema, model"):
        backend._seal_generation(backend.index_dir, generation_id="nonfinite")


def test_cocoflow_seal_rejects_discontinuous_chunk_indexes(tmp_path):
    backend = _write_complete_generation(tmp_path)
    (backend.index_dir / ".complete").unlink()
    record_path = backend.index_dir / "paper__x__1.json"
    record = json.loads(record_path.read_text())
    record["chunk_index"] = 2
    record_path.write_text(json.dumps(record))
    with pytest.raises(vector_query.IndexRecordError, match="not continuous"):
        backend._seal_generation(backend.index_dir, generation_id="gap")


def test_cocoflow_seal_rejects_fabricated_chunk_text(tmp_path):
    backend = _write_complete_generation(tmp_path)
    (backend.index_dir / ".complete").unlink()
    record_path = backend.index_dir / "paper__x__0.json"
    record = json.loads(record_path.read_text())
    record["text"] = "fabricated"
    record["chunk_sha256"] = fr.content_hash("fabricated")
    record_path.write_text(json.dumps(record))
    with pytest.raises(vector_query.IndexRecordError, match="do not match the source"):
        backend._seal_generation(backend.index_dir, generation_id="fabricated")


def test_cocoflow_reindex_lock_is_cross_process_and_released_on_exit(
    tmp_path, monkeypatch
):
    backend = CocoFlowBackend("paper", tmp_path)
    backend.lock_path.parent.mkdir(parents=True)
    script = (
        "import fcntl, sys\n"
        "handle = open(sys.argv[1], 'a+')\n"
        "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
        "print('locked', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(backend.lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        busy = backend.reindex()
        assert not busy.ok
        assert "busy in another process" in busy.detail
    finally:
        if process.stdin is not None:
            process.stdin.write("\n")
            process.stdin.flush()
        process.wait(timeout=5)

    monkeypatch.setattr(
        backend,
        "_reindex_locked",
        lambda paths, *, version_before: ReindexResult(
            kind="paper",
            ok=True,
            version_before=version_before,
            version_after="test",
        ),
    )
    assert backend.reindex().ok


def test_cocoflow_building_sentinel_hides_an_otherwise_complete_generation(tmp_path):
    backend = _write_complete_generation(tmp_path)
    backend.building_path.write_text("building")
    assert not backend.available()
    with pytest.raises(vector_query.IndexRecordError, match="unfinished build"):
        backend.search("alpha")


def test_cocoflow_missing_record_is_not_a_complete_generation(tmp_path):
    backend = _write_complete_generation(tmp_path)
    (backend.index_dir / "paper__x__1.json").unlink()
    assert not backend.available()
    with pytest.raises(vector_query.IndexRecordError, match="chunk count"):
        backend.search("alpha")


def test_cocoflow_rejects_record_replacement_with_same_count(tmp_path):
    backend = _write_complete_generation(tmp_path)
    record_path = backend.index_dir / "paper__x__0.json"
    record = json.loads(record_path.read_text())
    record["text"] = "replacement"
    record["chunk_sha256"] = fr.content_hash("replacement")
    record_path.write_text(json.dumps(record))
    assert not backend.available()
    with pytest.raises(vector_query.IndexRecordError, match="record set"):
        backend.search("alpha")


def test_cocoflow_rejects_tampered_completion_manifest(tmp_path):
    backend = _write_complete_generation(tmp_path)
    complete = backend.index_dir / ".complete"
    envelope = json.loads(complete.read_text())
    envelope["payload"]["record_count"] = 999
    complete.write_text(json.dumps(envelope))
    assert not backend.available()
    with pytest.raises(vector_query.IndexRecordError, match="checksum mismatch"):
        backend.search("alpha")


@pytest.mark.parametrize("change", ["add", "delete", "tamper"])
def test_cocoflow_source_set_or_digest_tamper_invalidates_generation(tmp_path, change):
    backend = _write_complete_generation(tmp_path)
    source = backend.sourcedir / "x.md"
    if change == "add":
        (backend.sourcedir / "new.md").write_text("new source")
    elif change == "delete":
        source.unlink()
    else:
        source.write_text("changed source")
    assert not backend.available()
    with pytest.raises(vector_query.IndexRecordError, match="source set"):
        backend.search("alpha")


def test_cocoflow_hard_exit_leaves_building_sentinel_and_never_exposes_staging(
    tmp_path, monkeypatch
):
    backend = _write_complete_generation(tmp_path)

    class App:
        def update_blocking(self, **kwargs):
            raise SystemExit(17)

    monkeypatch.setitem(
        sys.modules,
        "cocoindex",
        SimpleNamespace(runtime=lambda: nullcontext()),
    )
    monkeypatch.setattr(backend, "_build_app", lambda outdir: App())
    with pytest.raises(SystemExit, match="17"):
        backend.reindex()

    assert backend.building_path.exists()
    assert not backend.available()
    with pytest.raises(vector_query.IndexRecordError, match="unfinished build"):
        backend.search("alpha")


@pytest.mark.parametrize("kind", ["paper", "experiment", "skill"])
def test_document_source_digest_detects_same_mtime_tamper(kind, tmp_path):
    source = tmp_path / f"{kind}.md"
    source.write_bytes(b"original bytes\n")
    original_stat = source.stat()
    item = ContextItem(
        text="original bytes",
        provenance=Provenance(
            source_kind=kind,
            locator=source.name,
            indexed_at=now() + timedelta(seconds=5),
            content_hash=fr.content_hash("original bytes"),
            source_sha256=fr.file_sha256(source),
        ),
    )
    initial = fr.assess([item], index_version="v3", indexed_at=now(), base_dir=tmp_path)
    assert not initial.is_stale()

    source.write_bytes(b"tampered bytes\n")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    verdict = fr.assess([item], index_version="v3", indexed_at=now(), base_dir=tmp_path)
    assert verdict.is_stale()
    assert "indexed SHA-256" in verdict.reasons[-1]


def test_vector_query_rejects_an_index_built_with_another_model(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "embedder_model": "model-a",
                "embedding_dimension": 2,
                "kind": "paper",
                "source_path": "p/a.md",
                "text": "alpha",
                "embedding": [1.0, 0.0],
                "source_sha256": _SOURCE_SHA,
                "chunk_sha256": fr.content_hash("alpha"),
                "metadata": {},
            }
        )
    )
    with pytest.raises(vector_query.IndexRecordError, match="does not match query model"):
        vector_query.search(tmp_path, "q", 1, embedder_model="model-b")


def test_vector_query_rejects_record_without_source_digest(tmp_path):
    (tmp_path / "legacy.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "embedder_model": vector_query._embedder.DEFAULT_MODEL,
                "kind": "paper",
                "source_path": "p/a.md",
                "text": "alpha",
                "embedding": [1.0, 0.0],
                "chunk_sha256": fr.content_hash("alpha"),
                "metadata": {},
            }
        )
    )
    with pytest.raises(vector_query.IndexRecordError, match="malformed record"):
        vector_query.load_records(tmp_path)


def test_repair_overlay_reads_current_sandbox_not_original_root(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "a.py").write_text("value = 'old'\n")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "a.py").write_text("value = 'current'\n")
    bundle = ContextBundle(
        query="q",
        items=[
            ContextItem(
                text="value = 'old'",
                provenance=Provenance(
                    source_kind="code",
                    locator="a.py:1",
                    source_root=str(original),
                ),
            )
        ],
        freshness=fr.fresh_now("v"),
    )

    ContextEngineer._overlay_workdir(bundle, workdir)

    assert bundle.items[0].text == "value = 'current'"
    assert bundle.items[0].provenance.source_root == str(workdir.resolve())
    assert bundle.items[0].provenance.content_hash == fr.content_hash("value = 'current'")
    assert bundle.items[0].provenance.source_sha256 == fr.file_sha256(workdir / "a.py")


def test_repair_overlay_drops_legacy_absolute_locator(tmp_path):
    original = tmp_path / "original.py"
    original.write_text("value = 'outside'\n")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    bundle = ContextBundle(
        query="q",
        items=[
            ContextItem(
                text="stale",
                provenance=Provenance(source_kind="code", locator=str(original)),
            )
        ],
        freshness=fr.fresh_now("v"),
    )

    ContextEngineer._overlay_workdir(bundle, workdir)

    assert bundle.items == []
    assert bundle.status == "empty"
    assert "dropped" in bundle.status_notes[-1]
