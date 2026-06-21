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
from ..freshness import content_hash
from ..models import CodeHit, Hit, Provenance
from .base import SearchBackend


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
    # Express the locator relative to cwd so freshness checks resolve uniformly.
    p = Path(raw_path)
    abs_path = p if p.is_absolute() else (root / p)
    try:
        rel = str(abs_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        # Outside cwd (e.g. an absolute root): keep an absolute, resolvable
        # locator so freshness can still stat it.
        rel = str(abs_path.resolve())
    line_start = _first(d, "line_start", "start_line", "start", "lineStart")
    line_end = _first(d, "line_end", "end_line", "end", "lineEnd")
    code = _first(d, "code", "content", "text", "snippet", default="")
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
        ),
    )


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
            command=self._ccc,
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
        return _extract_results(result)

    def search(self, query: str, *, k: int = 8, **filters) -> list[Hit]:
        if not self.available():
            return []
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
        except Exception:
            return []
        return [_result_to_codehit(r, self.root, indexed_at) for r in rows][:k]

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

    def reindex(self, paths: list[str] | None = None) -> None:
        if not self._ccc:
            return
        env = _env_with_local_bin()
        # Auto-init a fresh project (e.g. a run sandbox) before indexing.
        if not (self.root / ".cocoindex_code" / "settings.yml").exists():
            subprocess.run(
                [self._ccc, "init", "-f"],
                cwd=str(self.root),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        subprocess.run(
            [self._ccc, "index"],
            cwd=str(self.root),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
