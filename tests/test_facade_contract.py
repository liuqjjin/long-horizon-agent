"""Facade contract: provenance, freshness, and graceful no-backend behavior."""

from __future__ import annotations

from datetime import timedelta

from lha.clock import now
from lha.config import Config
from lha.live_context import (
    ContextBundle,
    ContextItem,
    Provenance,
    configure,
    get_fresh_context,
    reject_stale,
    search_code,
)
from lha.live_context import freshness as fr


def test_null_backend_is_graceful(tmp_path):
    configure(code_root=str(tmp_path), config=Config(code_backend="null"))
    assert search_code("anything") == []
    bundle = get_fresh_context("anything", kinds=("code",), k=3)
    assert isinstance(bundle, ContextBundle)
    assert bundle.items == []
    assert not bundle.freshness.is_stale()


def test_content_hash_is_stable_and_sensitive():
    assert fr.content_hash("abc") == fr.content_hash("abc")
    assert fr.content_hash("abc") != fr.content_hash("abd")


def test_freshness_detects_edit_after_index(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("hello")
    indexed_long_ago = now() - timedelta(hours=1)
    item = ContextItem(
        text="hello",
        provenance=Provenance(source_kind="code", locator="a.py", indexed_at=indexed_long_ago),
    )
    verdict = fr.assess([item], index_version="v1", indexed_at=indexed_long_ago, base_dir=tmp_path)
    assert verdict.is_stale_flag
    assert "a.py" in verdict.reasons[0]


def test_path_from_locator_strips_line_range():
    assert fr.path_from_locator("src/m.py:10-14") == "src/m.py"
    assert fr.path_from_locator("notes.md") == "notes.md"


def test_reject_stale_is_noop_when_fresh():
    bundle = ContextBundle(query="q", items=[], freshness=fr.fresh_now("v1"))
    assert reject_stale(bundle) is bundle
