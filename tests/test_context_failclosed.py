"""Fail-closed context: empty/unavailable/failed-index context must never verify.

Covers the invariants:
  - a step that requires context fails when no bundle / empty / backend-unavailable;
  - a step may declare context optional explicitly (and only then proceed);
  - reject_stale fails closed when a reindex does not verifiably succeed;
  - ccc reindex checks subprocess return codes;
  - a deleted source file makes context stale, not silently fresh;
  - a code chunk that vanished from its source makes context stale.
"""

from __future__ import annotations

import subprocess
from datetime import timedelta

import pytest

from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.live_context import (
    ContextBundle,
    ContextItem,
    Provenance,
    ReindexResult,
    StaleContextError,
    configure,
    get_fresh_context,
    reject_stale,
)
from lha.live_context import freshness as fr
from lha.live_context.backends.ccc_backend import CccBackend
from lha.verifiers import VerifyContext
from lha.verifiers.context.citation_verifier import CitationVerifier
from lha.verifiers.context.freshness_verifier import FreshnessVerifier


def _step(requirement: str = "required", verifiers: list[str] | None = None) -> Step:
    return Step(
        step_id="s1",
        kind="context",
        action="gather_context",
        goal="g",
        verifiers=verifiers or ["freshness"],
        context_requirement=requirement,  # type: ignore[arg-type]
    )


def _ctx(tmp_path, bundle, requirement="required"):
    return VerifyContext(workdir=tmp_path, step=_step(requirement), bundle=bundle)


def _bundle(items=None, status="ok", notes=None) -> ContextBundle:
    return ContextBundle(
        query="q",
        items=items or [],
        freshness=fr.fresh_now("v1"),
        status=status,
        status_notes=notes or [],
    )


# --- FreshnessVerifier -------------------------------------------------------
def test_freshness_fails_without_bundle(tmp_path):
    check = FreshnessVerifier().verify(None, _ctx(tmp_path, bundle=None))
    assert not check.passed


def test_freshness_fails_on_backend_unavailable(tmp_path):
    bundle = _bundle(status="backend_unavailable", notes=["code: no backend available"])
    check = FreshnessVerifier().verify(None, _ctx(tmp_path, bundle))
    assert not check.passed
    assert "backend_unavailable" in check.detail["summary"]


def test_freshness_fails_on_index_failed(tmp_path):
    bundle = _bundle(status="index_failed", notes=["reindex failed"])
    check = FreshnessVerifier().verify(None, _ctx(tmp_path, bundle))
    assert not check.passed


def test_freshness_fails_on_empty_when_required(tmp_path):
    check = FreshnessVerifier().verify(None, _ctx(tmp_path, _bundle(status="empty")))
    assert not check.passed


def test_freshness_passes_on_empty_when_declared_optional(tmp_path):
    check = FreshnessVerifier().verify(
        None, _ctx(tmp_path, _bundle(status="empty"), requirement="optional")
    )
    assert check.passed
    assert "optional" in check.detail["summary"]


def test_freshness_optional_does_not_excuse_staleness(tmp_path):
    bundle = _bundle()
    bundle.freshness.is_stale_flag = True
    bundle.freshness.reasons = ["edited after index"]
    check = FreshnessVerifier().verify(
        None, _ctx(tmp_path, bundle, requirement="optional")
    )
    assert not check.passed


# --- CitationVerifier --------------------------------------------------------
class _Cited:
    def __init__(self, cites):
        self.based_on_context = cites


def test_citation_fails_on_zero_citations_when_required(tmp_path):
    check = CitationVerifier().verify(_Cited([]), _ctx(tmp_path, _bundle()))
    assert not check.passed


def test_citation_passes_zero_citations_when_optional(tmp_path):
    check = CitationVerifier().verify(
        _Cited([]), _ctx(tmp_path, _bundle(), requirement="optional")
    )
    assert check.passed


def test_citation_fails_on_empty_required_bundle_artifact(tmp_path):
    bundle = _bundle(status="empty")
    check = CitationVerifier().verify(bundle, _ctx(tmp_path, bundle))
    assert not check.passed


def test_citation_fails_on_unknown_artifact(tmp_path):
    check = CitationVerifier().verify(object(), _ctx(tmp_path, _bundle()))
    assert not check.passed


def test_citation_still_resolves_real_citations(tmp_path):
    item = ContextItem(
        text="t", provenance=Provenance(source_kind="code", locator="src/a.py:1-2")
    )
    bundle = _bundle(items=[item])
    good = CitationVerifier().verify(_Cited(["src/a.py:1-2"]), _ctx(tmp_path, bundle))
    assert good.passed
    bad = CitationVerifier().verify(_Cited(["src/other.py:9"]), _ctx(tmp_path, bundle))
    assert not bad.passed


# --- bundle status from the facade ------------------------------------------
def test_get_fresh_context_reports_backend_unavailable(tmp_path):
    configure(code_root=str(tmp_path), config=Config(code_backend="null"))
    bundle = get_fresh_context("anything", kinds=("code",), k=3)
    assert bundle.items == []
    assert bundle.status == "backend_unavailable"
    assert bundle.status_notes


# --- reject_stale fails closed ----------------------------------------------
class _StubBackend:
    """Minimal SearchBackend double with a scriptable reindex outcome."""

    kind = "code"
    name = "stub"

    def __init__(self, reindex_ok: bool):
        self._ok = reindex_ok

    def search(self, query, *, k=8, **filters):
        return []

    def index_meta(self):
        return ("stub@1", now())

    def reindex(self, paths=None):
        return ReindexResult(kind="code", ok=self._ok, detail="" if self._ok else "boom")


def _stale_bundle() -> ContextBundle:
    b = ContextBundle(query="q", items=[], freshness=fr.fresh_now("v1"))
    b.freshness.is_stale_flag = True
    b.freshness.reasons = ["test-forced stale"]
    return b


def test_reject_stale_raises_when_reindex_fails(monkeypatch):
    from lha.live_context import facade

    monkeypatch.setattr(facade, "_backend_for", lambda kind: _StubBackend(reindex_ok=False))
    bundle = _stale_bundle()
    with pytest.raises(StaleContextError, match="boom"):
        reject_stale(bundle)
    # the original bundle must still read as stale — the flag was not cleared
    assert bundle.freshness.is_stale()


def test_reject_stale_clears_flag_only_after_verified_reindex(monkeypatch):
    from lha.live_context import facade

    monkeypatch.setattr(facade, "_backend_for", lambda kind: _StubBackend(reindex_ok=True))
    refreshed = reject_stale(_stale_bundle())
    assert not refreshed.freshness.is_stale()


# --- ccc reindex checks subprocess results ------------------------------------
class _Proc:
    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_ccc_reindex_fails_on_nonzero_exit(tmp_path, monkeypatch):
    (tmp_path / ".cocoindex_code").mkdir()
    (tmp_path / ".cocoindex_code" / "settings.yml").write_text("x: 1")
    backend = CccBackend(tmp_path)
    backend._ccc = "/fake/ccc"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(2, "index blew up"))
    result = backend.reindex()
    assert not result.ok
    assert "exit 2" in result.detail


def test_ccc_reindex_fails_when_binary_missing(tmp_path):
    backend = CccBackend(tmp_path)
    backend._ccc = None
    result = backend.reindex()
    assert not result.ok


def test_ccc_reindex_ok_on_zero_exit(tmp_path, monkeypatch):
    (tmp_path / ".cocoindex_code").mkdir()
    (tmp_path / ".cocoindex_code" / "settings.yml").write_text("x: 1")
    backend = CccBackend(tmp_path)
    backend._ccc = "/fake/ccc"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc(0))
    result = backend.reindex()
    assert result.ok


# --- freshness signals --------------------------------------------------------
def test_missing_source_file_is_stale(tmp_path):
    item = ContextItem(
        text="gone",
        provenance=Provenance(source_kind="code", locator="deleted.py", indexed_at=now()),
    )
    verdict = fr.assess([item], index_version="v1", indexed_at=now(), base_dir=tmp_path)
    assert verdict.is_stale_flag
    assert "no longer exists" in verdict.reasons[0]


def test_vanished_code_chunk_is_stale(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def other():\n    return 2\n")
    # indexed_at AFTER the file's mtime, so the mtime signal alone would pass
    later = now() + timedelta(seconds=5)
    item = ContextItem(
        text="def removed():\n    return 1\n",
        provenance=Provenance(source_kind="code", locator="a.py", indexed_at=later),
    )
    verdict = fr.assess([item], index_version="v1", indexed_at=later, base_dir=tmp_path)
    assert verdict.is_stale_flag
    assert "chunk" in verdict.reasons[0]


def test_present_code_chunk_stays_fresh(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def keep():\n    return 1\n")
    later = now() + timedelta(seconds=5)
    item = ContextItem(
        text="def keep():\n    return 1",
        provenance=Provenance(source_kind="code", locator="a.py", indexed_at=later),
    )
    verdict = fr.assess([item], index_version="v1", indexed_at=later, base_dir=tmp_path)
    assert not verdict.is_stale_flag
