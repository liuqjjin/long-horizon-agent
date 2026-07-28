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

import asyncio
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from lha.artifacts import Step
from lha.clock import now
from lha.config import Config
from lha.live_context import (
    BackendUnavailable,
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
from lha.live_context.backends import ccc_backend
from lha.live_context.backends.ccc_backend import (
    CccBackend,
    _await_mcp,
    _checked_results,
    _env_with_local_bin,
)
from lha.live_context.backends.coco_flow import CocoFlowBackend
from lha.live_context.backends.vector_query import IndexRecordError
from lha.verifiers import VerifyContext
from lha.verifiers.context.citation_verifier import CitationVerifier
from lha.verifiers.context.freshness_verifier import FreshnessVerifier


@pytest.fixture(autouse=True)
def _restore_facade_state():
    """``configure`` mutates the module singleton; put it back so ordering
    between test files stays irrelevant."""
    from lha.live_context import facade

    saved_root, saved_config = facade._state.code_root, facade._state.config
    yield
    facade._state.code_root = saved_root
    facade._state.config = saved_config
    facade._state.reset_cache()


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


def test_optional_context_does_not_excuse_a_failed_stale_index(tmp_path):
    bundle = _bundle(status="index_failed", notes=["refresh failed"])
    bundle.freshness.is_stale_flag = True
    bundle.freshness.reasons = ["source changed"]
    check = FreshnessVerifier().verify(None, _ctx(tmp_path, bundle, requirement="optional"))
    assert not check.passed
    assert "stale" in check.detail["summary"]


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
    assert bundle.requested_kinds == ["code"]
    assert bundle.unavailable_reasons == {"code": "no backend available"}


def test_get_fresh_context_distinguishes_a_broken_index(monkeypatch):
    from lha.live_context import facade

    class BrokenBackend:
        def search(self, query, *, k=8, **filters):
            raise ValueError("index schema is corrupt")

    monkeypatch.setattr(facade, "_backend_for", lambda kind: BrokenBackend())
    bundle = get_fresh_context("q", kinds=("code",))

    assert bundle.status == "index_failed"
    assert bundle.failed_kinds == ["code"]
    assert "index schema is corrupt" in bundle.failure_reasons["code"]
    assert bundle.unavailable_kinds == []


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


def test_reject_stale_wraps_a_reindex_exception_and_stays_stale(monkeypatch):
    from lha.live_context import facade

    class RaisingBackend(_StubBackend):
        def reindex(self, paths=None):
            raise RuntimeError("damaged index")

    monkeypatch.setattr(
        facade, "_backend_for", lambda kind: RaisingBackend(reindex_ok=True)
    )
    bundle = _stale_bundle()
    with pytest.raises(StaleContextError, match="damaged index"):
        reject_stale(bundle)
    assert bundle.freshness.is_stale()


def test_reject_stale_clears_flag_only_after_verified_reindex(monkeypatch):
    from lha.live_context import facade

    monkeypatch.setattr(facade, "_backend_for", lambda kind: _StubBackend(reindex_ok=True))
    refreshed = reject_stale(_stale_bundle())
    assert not refreshed.freshness.is_stale()


def test_reject_stale_refreshes_every_requested_kind(monkeypatch, tmp_path):
    from lha.live_context import facade
    from lha.live_context.models import PaperHit

    source = tmp_path / "paper.md"
    source.write_text("paper")
    calls: list[tuple[str, str]] = []

    class Backend:
        def __init__(self, kind):
            self.kind = kind
            self.name = f"stub-{kind}"

        def search(self, query, *, k=8, **filters):
            calls.append(("search", self.kind))
            if self.kind == "paper":
                return [
                    PaperHit(
                        text="paper",
                        provenance=Provenance(
                            source_kind="paper", locator=str(source), indexed_at=now()
                        ),
                    )
                ]
            return []

        def index_meta(self):
            return ("v", now())

        def reindex(self, paths=None):
            calls.append(("reindex", self.kind))
            return ReindexResult(kind=self.kind, ok=True)

    backends = {"paper": Backend("paper"), "code": Backend("code")}
    monkeypatch.setattr(facade, "_backend_for", lambda kind: backends[kind])
    bundle = ContextBundle(
        query="q",
        items=[
            ContextItem(
                text="paper",
                provenance=Provenance(
                    source_kind="paper", locator=str(source), indexed_at=now()
                ),
            )
        ],
        freshness=fr.fresh_now("v"),
        unavailable_kinds=["code"],
        requested_kinds=["paper", "code"],
    )
    bundle.freshness.is_stale_flag = True
    bundle.freshness.reasons = ["forced"]

    refreshed = reject_stale(bundle)
    assert ("reindex", "paper") in calls and ("reindex", "code") in calls
    assert refreshed.requested_kinds == ["paper", "code"]
    assert refreshed.unavailable_kinds == []


def test_reject_stale_does_not_clear_newly_detected_staleness(monkeypatch, tmp_path):
    from lha.live_context import facade
    from lha.live_context.models import PaperHit

    missing = tmp_path / "missing.md"

    class BadBackend(_StubBackend):
        kind = "paper"

        def search(self, query, *, k=8, **filters):
            return [
                PaperHit(
                    text="ghost",
                    provenance=Provenance(
                        source_kind="paper", locator=str(missing), indexed_at=now()
                    ),
                )
            ]

        def index_meta(self):
            return ("v", now())

        def reindex(self, paths=None):
            return ReindexResult(kind="paper", ok=True)

    monkeypatch.setattr(facade, "_backend_for", lambda kind: BadBackend(reindex_ok=True))
    bundle = ContextBundle(
        query="q",
        items=[
            ContextItem(
                text="old",
                provenance=Provenance(source_kind="paper", locator=str(missing)),
            )
        ],
        freshness=fr.fresh_now("v"),
        requested_kinds=["paper"],
    )
    bundle.freshness.is_stale_flag = True
    with pytest.raises(StaleContextError, match="still stale"):
        reject_stale(bundle)


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
    monkeypatch.setattr(
        ccc_backend,
        "_run_ccc_control",
        lambda *a, **k: _Proc(2, "index blew up"),
    )
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
    monkeypatch.setattr(
        ccc_backend,
        "_run_ccc_control",
        lambda *a, **k: _Proc(0),
    )
    result = backend.reindex()
    assert result.ok


def test_ccc_mcp_error_is_not_an_empty_result():
    result = SimpleNamespace(
        isError=True,
        structuredContent=None,
        content=[SimpleNamespace(text='{"error": "boom"}')],
    )
    with pytest.raises(BackendUnavailable, match="returned an error"):
        _checked_results(result)


def test_ccc_unsuccessful_structured_result_is_not_empty():
    result = SimpleNamespace(
        isError=False,
        structuredContent={"success": False, "results": [], "message": "broken index"},
        content=[],
    )
    with pytest.raises(BackendUnavailable, match="unsuccessful"):
        _checked_results(result)


def test_ccc_unsuccessful_text_result_is_not_empty():
    result = SimpleNamespace(
        isError=False,
        structuredContent=None,
        content=[
            SimpleNamespace(
                text='{"success": false, "results": [], "message": "broken index"}'
            )
        ],
    )
    with pytest.raises(BackendUnavailable, match="unsuccessful"):
        _checked_results(result)


def test_ccc_subprocess_environment_excludes_host_credentials(tmp_path, monkeypatch):
    absolute_bin = tmp_path / "custom" / "bin"
    absolute_bin.mkdir(parents=True)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((".", "relative/bin", str(absolute_bin))),
    )
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("CODEX_HOME", "/tmp/codex")

    env = _env_with_local_bin()

    assert env["LANG"] == "en_US.UTF-8"
    assert str(absolute_bin) in env["PATH"].split(os.pathsep)
    assert "" not in env["PATH"].split(os.pathsep)
    assert "." not in env["PATH"].split(os.pathsep)
    assert "relative/bin" not in env["PATH"].split(os.pathsep)
    assert all(
        Path(component).is_absolute()
        for component in env["PATH"].split(os.pathsep)
    )
    assert set(env) <= {"PATH", "LANG", "LC_ALL", "LC_CTYPE"}
    assert not {
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "SSH_AUTH_SOCK",
        "CODEX_HOME",
    } & env.keys()


def test_ccc_discovery_ignores_a_relative_worktree_path(tmp_path, monkeypatch):
    fake = tmp_path / "ccc"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", ".")

    found = ccc_backend.find_ccc()

    assert found is None or Path(found) != fake
    assert found is None or Path(found).is_absolute()


def test_ccc_control_command_uses_bounded_process_group(tmp_path, monkeypatch):
    observed = {}

    def recording_runner(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        return _Proc(0)

    monkeypatch.setattr(ccc_backend, "run_bounded_process", recording_runner)

    result = ccc_backend._run_ccc_control(
        ["/fake/ccc", "index"],
        root=tmp_path,
        env={"PATH": "/bin"},
        timeout_s=600,
    )

    assert result.returncode == 0
    assert observed["argv"] == ["/fake/ccc", "index"]
    assert observed["cwd"] == tmp_path
    assert observed["output_bytes"] == 1024 * 1024
    assert observed["start_new_session"] is True
    assert observed["on_exit"] is ccc_backend.terminate_process_group


def test_ccc_control_rejects_unsupported_process_groups_before_spawn(
    tmp_path, monkeypatch
):
    def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("unsupported hosts must fail before spawning ccc")

    monkeypatch.setattr(
        ccc_backend,
        "process_group_cleanup_supported",
        lambda: False,
    )
    monkeypatch.setattr(ccc_backend, "run_bounded_process", unexpected_spawn)

    result = ccc_backend._run_ccc_control(
        ["/fake/ccc", "index"],
        root=tmp_path,
        env={"PATH": "/bin"},
        timeout_s=600,
    )

    assert result.returncode == 126
    assert "requires POSIX process-group cleanup" in result.stderr


def test_ccc_mcp_operation_timeout_fails_as_backend_unavailable():
    async def wait_forever():
        await asyncio.Event().wait()

    with pytest.raises(BackendUnavailable, match="initialization timed out"):
        asyncio.run(
            _await_mcp(
                wait_forever(),
                timeout_s=0.001,
                operation="initialization",
            )
        )


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_ccc_rejects_invalid_mcp_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite positive"):
        CccBackend(tmp_path, mcp_search_timeout_s=timeout)


def test_corrupt_vector_index_is_reported_as_index_failed(tmp_path, monkeypatch):
    from lha.live_context import facade

    index = tmp_path / ".lha_index" / "papers"
    index.mkdir(parents=True)
    (index / "bad.json").write_text("{not json")
    backend = CocoFlowBackend("paper", tmp_path)
    assert not backend.available()
    with pytest.raises(IndexRecordError, match="completion manifest"):
        backend.search("anything")
    monkeypatch.setattr(facade, "_backend_for", lambda kind: backend)
    bundle = get_fresh_context("anything", kinds=("paper",))
    assert bundle.status == "index_failed"
    assert bundle.failed_kinds == ["paper"]
    assert bundle.unavailable_kinds == []


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


def test_changed_indexed_chunk_hash_is_stale(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("value = 1\n")
    later = now() + timedelta(seconds=5)
    item = ContextItem(
        text="value = 1",
        provenance=Provenance(
            source_kind="code",
            locator="a.py",
            indexed_at=later,
            content_hash=fr.content_hash("different indexed bytes"),
        ),
    )
    verdict = fr.assess([item], index_version="v1", indexed_at=later, base_dir=tmp_path)
    assert verdict.is_stale_flag
    assert "content hash" in verdict.reasons[0]


def test_ccc_hit_binds_full_source_digest_and_detects_same_mtime_tamper(
    tmp_path, monkeypatch
):
    source = tmp_path / "a.py"
    source.write_bytes(b"value = 1\n")
    original_stat = source.stat()
    backend = CccBackend(tmp_path)
    backend._ccc = "/fake/ccc"

    async def search(*args, **kwargs):
        return [
            {
                "path": "a.py",
                "code": "value = 1",
                "line_start": 1,
                "line_end": 1,
            }
        ]

    monkeypatch.setattr(backend, "_search_async", search)
    hit = backend.search("value", refresh=False)[0]
    assert hit.provenance.source_sha256 == fr.file_sha256(source)
    initial = fr.assess(
        [ContextItem.from_hit(hit)],
        index_version="ccc@test",
        indexed_at=hit.provenance.indexed_at,
    )
    assert not initial.is_stale()

    source.write_bytes(b"value = 2\n")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    changed = fr.assess(
        [ContextItem.from_hit(hit)],
        index_version="ccc@test",
        indexed_at=hit.provenance.indexed_at,
    )
    assert changed.is_stale()
    assert any("indexed SHA-256" in reason for reason in changed.reasons)


def test_ccc_rejects_snippet_that_does_not_match_claimed_lines(tmp_path, monkeypatch):
    source = tmp_path / "a.py"
    source.write_text("first = 1\nsecond = 2\n")
    backend = CccBackend(tmp_path)
    backend._ccc = "/fake/ccc"

    async def search(*args, **kwargs):
        return [
            {
                "path": "a.py",
                "content": "second = 2",
                "start_line": 1,
                "end_line": 1,
            }
        ]

    monkeypatch.setattr(backend, "_search_async", search)
    with pytest.raises(BackendUnavailable, match="snippet does not match"):
        backend.search("second", refresh=False)


@pytest.mark.parametrize(
    ("line_start", "line_end"),
    [(0, 1), (2, 1), (1, 3), ("1", 1), (True, 1)],
)
def test_ccc_rejects_invalid_or_out_of_bounds_line_range(
    tmp_path, monkeypatch, line_start, line_end
):
    source = tmp_path / "a.py"
    source.write_text("value = 1\n")
    backend = CccBackend(tmp_path)
    backend._ccc = "/fake/ccc"

    async def search(*args, **kwargs):
        return [
            {
                "path": "a.py",
                "code": "value = 1",
                "line_start": line_start,
                "line_end": line_end,
            }
        ]

    monkeypatch.setattr(backend, "_search_async", search)
    with pytest.raises(BackendUnavailable, match="source provenance"):
        backend.search("value", refresh=False)


def test_ccc_accepts_trimmed_multiline_snippet_with_exact_line_range(
    tmp_path, monkeypatch
):
    source = tmp_path / "a.py"
    source.write_text("  first = 1\nsecond = 2  \nthird = 3\n")
    backend = CccBackend(tmp_path)
    backend._ccc = "/fake/ccc"

    async def search(*args, **kwargs):
        return [
            {
                "file_path": "a.py",
                "content": "first = 1\nsecond = 2",
                "start_line": 1,
                "end_line": 2,
            }
        ]

    monkeypatch.setattr(backend, "_search_async", search)
    hit = backend.search("value", refresh=False)[0]

    assert hit.text == "first = 1\nsecond = 2"
    assert hit.line_start == 1
    assert hit.line_end == 2
    assert hit.provenance.locator == "a.py:1-2"


def test_ccc_rejects_symlink_source_provenance(tmp_path, monkeypatch):
    target = tmp_path / "real.py"
    target.write_text("value = 1\n")
    (tmp_path / "alias.py").symlink_to(target)
    backend = CccBackend(tmp_path)
    backend._ccc = "/fake/ccc"

    async def search(*args, **kwargs):
        return [{"path": "alias.py", "code": "value = 1"}]

    monkeypatch.setattr(backend, "_search_async", search)
    with pytest.raises(BackendUnavailable, match="symlink"):
        backend.search("value", refresh=False)


def test_ccc_rejects_source_outside_indexed_root(tmp_path, monkeypatch):
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n")
    root = tmp_path / "repo"
    root.mkdir()
    backend = CccBackend(root)
    backend._ccc = "/fake/ccc"

    async def search(*args, **kwargs):
        return [{"path": "../outside.py", "code": "value = 1"}]

    monkeypatch.setattr(backend, "_search_async", search)
    with pytest.raises(BackendUnavailable, match="unsafe source path"):
        backend.search("value", refresh=False)


def test_code_freshness_fails_closed_when_legacy_probe_is_too_large(tmp_path):
    source = tmp_path / "large.py"
    source.write_bytes(b"value = 1\n" + b"#" * (2_000_000 + 1))
    later = now() + timedelta(seconds=5)
    legacy = ContextItem(
        text="value = 1",
        provenance=Provenance(source_kind="code", locator="large.py", indexed_at=later),
    )
    verdict = fr.assess([legacy], index_version="legacy", indexed_at=later, base_dir=tmp_path)
    assert verdict.is_stale()
    assert any("too large" in reason for reason in verdict.reasons)

    bound = legacy.model_copy(deep=True)
    bound.provenance.source_sha256 = fr.file_sha256(source)
    digest_verdict = fr.assess(
        [bound], index_version="digest", indexed_at=later, base_dir=tmp_path
    )
    assert not digest_verdict.is_stale()


def test_code_freshness_rejects_symlink_even_when_target_digest_matches(tmp_path):
    target = tmp_path / "real.py"
    target.write_text("value = 1\n")
    alias = tmp_path / "alias.py"
    alias.symlink_to(target)
    later = now() + timedelta(seconds=5)
    item = ContextItem(
        text="value = 1",
        provenance=Provenance(
            source_kind="code",
            locator="alias.py",
            indexed_at=later,
            source_sha256=fr.file_sha256(target),
        ),
    )
    verdict = fr.assess([item], index_version="digest", indexed_at=later, base_dir=tmp_path)
    assert verdict.is_stale()
    assert any("read safely" in reason for reason in verdict.reasons)


def test_code_freshness_rejects_unreadable_digest_source(tmp_path, monkeypatch):
    source = tmp_path / "a.py"
    source.write_text("value = 1\n")
    later = now() + timedelta(seconds=5)
    item = ContextItem(
        text="value = 1",
        provenance=Provenance(
            source_kind="code",
            locator="a.py",
            indexed_at=later,
            source_sha256=fr.file_sha256(source),
        ),
    )

    def unreadable(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(fr, "strict_file_sha256", unreadable)
    verdict = fr.assess([item], index_version="digest", indexed_at=later, base_dir=tmp_path)
    assert verdict.is_stale()
    assert any("read safely" in reason for reason in verdict.reasons)
