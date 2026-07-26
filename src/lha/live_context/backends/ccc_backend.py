"""Code-search backend backed by ``cocoindex-code`` (the ``ccc`` tool).

Access path (decided in the plan): the harness talks to the *structured* MCP
``search`` tool exposed by ``ccc mcp`` over stdio. ``ccc search`` has no JSON
output and there is no Python API, so the MCP tool is the only structured
surface. Index refresh / status go through the ``ccc`` CLI.

This module is the ONLY place that knows about ``ccc``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...clock import now
from ..freshness import content_hash, strict_file_sha256
from ..models import CodeHit, Hit, Provenance, ReindexResult
from .base import BackendUnavailable, SearchBackend


def find_ccc() -> str | None:
    """Locate the ``ccc`` executable, including the pipx default bin dir."""
    found = shutil.which("ccc")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "ccc"
    return str(candidate) if candidate.exists() else None


def _env_with_local_bin() -> dict[str, str]:
    env = dict(os.environ)
    extra = f"{Path.home()}/.local/bin:/opt/homebrew/bin"
    env["PATH"] = extra + ":" + env.get("PATH", "")
    return env


# Flexible field extraction — we do not hard-code ccc's exact JSON keys.
def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _result_to_codehit(d: dict[str, Any], root: Path, indexed_at: datetime) -> CodeHit:
    raw_path = _first(d, "path", "file", "filename", "file_path", default="")
    # Express the locator relative to the indexed root so a repair can overlay the
    # same path in its copied workdir. Provenance retains the original root for
    # source-freshness checks.
    validated_rel = d.get("_lha_source_rel")
    validated_root = d.get("_lha_source_root")
    source_sha256 = d.get("_lha_source_sha256")
    if isinstance(validated_rel, str) and isinstance(validated_root, str):
        rel = validated_rel
        source_root: str | None = validated_root
    else:
        # Keep the pure parser tolerant for diagnostics. ``search`` always
        # supplies validated evidence before this conversion path is reached.
        p = Path(str(raw_path))
        abs_path = p if p.is_absolute() else (root / p)
        resolved_root = root.resolve()
        try:
            rel = str(abs_path.resolve().relative_to(resolved_root))
            source_root = str(resolved_root)
        except ValueError:
            rel = str(abs_path.resolve())
            source_root = None
    line_start = _first(d, "line_start", "start_line", "start", "lineStart")
    line_end = _first(d, "line_end", "end_line", "end", "lineEnd")
    code = str(_first(d, "code", "content", "text", "snippet", default="") or "")
    language = _first(d, "language", "lang")
    score = float(_first(d, "score", "similarity", "distance_score", default=0.0) or 0.0)

    locator = rel
    if line_start is not None:
        locator = f"{rel}:{line_start}" + (f"-{line_end}" if line_end is not None else "")

    return CodeHit(
        text=code,
        score=score,
        language=language,
        line_start=line_start,
        line_end=line_end,
        provenance=Provenance(
            source_kind="code",
            locator=locator,
            score=score,
            indexed_at=indexed_at,  # the index generation time, so edits read as stale
            content_hash=content_hash(code) if code else None,
            source_sha256=source_sha256 if isinstance(source_sha256, str) else None,
            source_root=source_root,
        ),
    )


def _bind_source_evidence(row: dict[str, Any], root: Path) -> dict[str, Any]:
    """Bind a CCC row to one exact, regular source file inside ``root``."""
    raw = str(_first(row, "path", "file", "filename", "file_path", default=""))
    if not raw:
        raise OSError("ccc result has no source path")
    canonical_root = root.resolve(strict=True)
    raw_path = Path(raw)
    if raw_path.is_absolute():
        try:
            relative = raw_path.relative_to(canonical_root)
        except ValueError as error:
            raise OSError(f"ccc returned path outside indexed root: {raw_path}") from error
    else:
        relative = raw_path
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise OSError(f"ccc returned an unsafe source path: {raw_path}")

    source = canonical_root / relative
    digest = strict_file_sha256(source, root=canonical_root)
    resolved = source.resolve(strict=True)
    try:
        canonical_relative = resolved.relative_to(canonical_root)
    except ValueError as error:
        raise OSError(f"ccc returned path outside indexed root: {raw_path}") from error
    bound = dict(row)
    bound["_lha_source_rel"] = canonical_relative.as_posix()
    bound["_lha_source_root"] = str(canonical_root)
    bound["_lha_source_sha256"] = digest
    return bound


def _extract_results(call_result: Any) -> list[dict]:
    """Pull a list of result dicts out of an MCP CallToolResult, defensively."""
    # 1) structured content (preferred)
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        for key in ("results", "items", "hits", "matches"):
            if isinstance(structured.get(key), list):
                return structured[key]
        # a bare list wrapped under some single key
        for v in structured.values():
            if isinstance(v, list):
                return v
    if isinstance(structured, list):
        return structured

    # 2) text content blocks, each possibly JSON
    out: list[dict] = []
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            out.extend(x for x in parsed if isinstance(x, dict))
        elif isinstance(parsed, dict):
            for key in ("results", "items", "hits", "matches"):
                if isinstance(parsed.get(key), list):
                    out.extend(parsed[key])
                    break
            else:
                out.append(parsed)
    return out


def _explicit_empty_result(call_result: Any) -> bool:
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        for key in ("results", "items", "hits", "matches"):
            if structured.get(key) == []:
                return True
    elif structured == []:
        return True
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if parsed == []:
            return True
        if isinstance(parsed, dict) and any(
            parsed.get(key) == [] for key in ("results", "items", "hits", "matches")
        ):
            return True
    return False


def _checked_results(call_result: Any) -> list[dict]:
    """A tool failure or malformed response is not an empty code search."""
    if bool(getattr(call_result, "isError", False)):
        raise BackendUnavailable("ccc MCP search tool returned an error")
    rows = _extract_results(call_result)
    if not rows:
        if _explicit_empty_result(call_result):
            return []
        raise BackendUnavailable("ccc MCP search returned no parseable result envelope")
    for row in rows:
        path = _first(row, "path", "file", "filename", "file_path")
        text = _first(row, "code", "content", "text", "snippet")
        if not path or not text:
            raise BackendUnavailable("ccc MCP search returned a malformed result without path/text")
    return rows


class CccBackend(SearchBackend):
    name = "ccc"
    kind = "code"

    def __init__(self, root: Path):
        self.root = Path(root)
        self._ccc = find_ccc()

    def available(self) -> bool:
        return self._ccc is not None and self.root.exists()

    # --- search via MCP -----------------------------------------------------
    async def _search_async(
        self,
        query: str,
        k: int,
        languages: list[str] | None,
        paths: list[str] | None,
        refresh: bool,
    ) -> list[dict]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._ccc or "ccc",  # available() guarantees non-None; satisfies the type
            args=["mcp"],
            cwd=str(self.root),
            env=_env_with_local_bin(),
        )
        args: dict[str, Any] = {"query": query, "limit": k, "refresh_index": refresh}
        if languages:
            args["languages"] = languages
        if paths:
            args["paths"] = paths

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("search", args)
        return _checked_results(result)

    def search(self, query: str, *, k: int = 8, **filters) -> list[Hit]:
        if not self.available():
            raise BackendUnavailable(f"ccc backend unavailable (ccc={self._ccc}, root={self.root})")
        languages = filters.get("languages")
        paths = filters.get("paths")
        # Default to refreshing: ccc builds its vector target lazily on the first
        # refresh search, and its daemon auto-watches sources anyway, so code
        # context is kept fresh here. Capture-time staleness for un-watched corpora
        # (papers/experiments/skills) is handled by their backends + freshness.assess.
        refresh = bool(filters.get("refresh", True))
        _iv, indexed_at = self.index_meta()
        try:
            rows = asyncio.run(self._search_async(query, k, languages, paths, refresh))
        except Exception as e:
            # A failed search is NOT an empty result — the caller must know the
            # difference or missing context silently reads as "nothing relevant".
            raise BackendUnavailable(f"ccc search failed: {type(e).__name__}: {e}") from e
        try:
            bound_rows = [_bind_source_evidence(row, self.root) for row in rows]
        except (OSError, RuntimeError) as error:
            raise BackendUnavailable(
                f"ccc returned unverifiable source provenance: {type(error).__name__}: {error}"
            ) from error
        hits: list[Hit] = [
            _result_to_codehit(row, self.root, indexed_at) for row in bound_rows
        ]
        return hits[:k]

    # --- index management via CLI ------------------------------------------
    def index_meta(self) -> tuple[str, datetime]:
        idx_dir = self.root / ".cocoindex_code"
        if idx_dir.exists():
            mtime = max(
                (p.stat().st_mtime for p in idx_dir.glob("*") if p.is_file()),
                default=idx_dir.stat().st_mtime,
            )
            ts = datetime.fromtimestamp(mtime, tz=timezone.utc)
            return (f"ccc@{int(mtime)}", ts)
        return ("ccc@uninitialized", now())

    def reindex(self, paths: list[str] | None = None) -> ReindexResult:
        version_before, _ = self.index_meta()
        if not self._ccc:
            return ReindexResult(
                kind=self.kind, ok=False, version_before=version_before,
                detail="ccc executable not found",
            )
        env = _env_with_local_bin()
        try:
            # Auto-init a fresh project (e.g. a run sandbox) before indexing.
            if not (self.root / ".cocoindex_code" / "settings.yml").exists():
                init = subprocess.run(
                    [self._ccc, "init", "-f"],
                    cwd=str(self.root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if init.returncode != 0:
                    return ReindexResult(
                        kind=self.kind, ok=False, version_before=version_before,
                        detail=f"ccc init failed (exit {init.returncode}): {init.stderr[-300:]}",
                    )
            proc = subprocess.run(
                [self._ccc, "index"],
                cwd=str(self.root),
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return ReindexResult(
                kind=self.kind, ok=False, version_before=version_before,
                detail=f"ccc index did not run: {type(e).__name__}: {e}",
            )
        version_after, _ = self.index_meta()
        if proc.returncode != 0:
            return ReindexResult(
                kind=self.kind, ok=False,
                version_before=version_before, version_after=version_after,
                detail=f"ccc index failed (exit {proc.returncode}): {proc.stderr[-300:]}",
            )
        if not (self.root / ".cocoindex_code").exists():
            return ReindexResult(
                kind=self.kind, ok=False,
                version_before=version_before, version_after=version_after,
                detail="ccc index reported success but no index directory exists",
            )
        return ReindexResult(
            kind=self.kind, ok=True, version_before=version_before, version_after=version_after
        )
