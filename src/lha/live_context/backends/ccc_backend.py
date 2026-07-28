"""Code-search backend backed by ``cocoindex-code`` (the ``ccc`` tool).

Access path (decided in the plan): the harness talks to the *structured* MCP
``search`` tool exposed by ``ccc mcp`` over stdio. ``ccc search`` has no JSON
output and there is no Python API, so the MCP tool is the only structured
surface. Index refresh / status go through the ``ccc`` CLI.

This module is the ONLY place that knows about ``ccc``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from bisect import bisect_right
from collections.abc import Awaitable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from ...clock import now
from ...sandbox.base import (
    PROCESS_CLEANUP_RETURN_CODE,
    process_group_cleanup_supported,
    run_bounded_process,
    terminate_process_group,
)
from ...tools.shell import (
    ProcResult,
    sanitized_absolute_path,
    trusted_executable,
)
from ..freshness import content_hash, strict_file_sha256
from ..models import CodeHit, Hit, Provenance, ReindexResult
from .base import BackendUnavailable, SearchBackend

_MCP_INITIALIZE_TIMEOUT_S = 30.0
_MCP_SEARCH_TIMEOUT_S = 180.0
_CCC_CONTROL_OUTPUT_BYTES = 1024 * 1024
_T = TypeVar("_T")


def find_ccc() -> str | None:
    """Locate the ``ccc`` executable, including the pipx default bin dir."""
    return trusted_executable(
        "ccc",
        extra_dirs=(
            Path.home() / ".local" / "bin",
            Path("/opt/homebrew/bin"),
        ),
        require_unwritable=False,
    )


def _env_with_local_bin() -> dict[str, str]:
    """Build the narrow environment shared with the trusted CCC subprocess.

    Passing ``os.environ`` would also pass model, cloud, GitHub, and SSH
    credentials to a process started inside a target repository. CCC needs an
    executable path and locale settings here; everything else is deliberately
    absent.
    """
    env = {
        "PATH": sanitized_absolute_path(
            extra_dirs=(
                Path.home() / ".local" / "bin",
                Path("/opt/homebrew/bin"),
            ),
        ),
    }
    for key in ("LANG", "LC_ALL", "LC_CTYPE"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _checked_timeout(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return timeout


def _run_ccc_control(
    argv: list[str],
    *,
    root: Path,
    env: dict[str, str],
    timeout_s: float,
) -> ProcResult:
    """Run a fixed CCC control command with bounded output and tree cleanup."""
    if not process_group_cleanup_supported():
        return ProcResult(
            PROCESS_CLEANUP_RETURN_CODE,
            "",
            (
                "ccc control command requires POSIX process-group cleanup; "
                "use Linux, macOS, or WSL2"
                ),
                0.0,
                cleanup_confirmed=False,
                cleanup_detail=(
                    "POSIX process-group cleanup is unavailable"
                ),
            )
    return run_bounded_process(
        argv,
        cwd=root,
        env=env,
        timeout=timeout_s,
        output_bytes=_CCC_CONTROL_OUTPUT_BYTES,
        start_new_session=True,
        on_exit=terminate_process_group,
    )


async def _await_mcp(
    awaitable: Awaitable[_T],
    *,
    timeout_s: float,
    operation: str,
) -> _T:
    """Await one MCP operation with a fail-closed wall-clock deadline."""
    try:
        async with asyncio.timeout(timeout_s):
            return await awaitable
    except TimeoutError as error:
        raise BackendUnavailable(
            f"ccc MCP {operation} timed out after {timeout_s:g}s"
        ) from error


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
    """Bind a CCC row to exact bytes and an exact line span inside ``root``."""
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

    line_start = _first(row, "line_start", "start_line", "start", "lineStart")
    line_end = _first(row, "line_end", "end_line", "end", "lineEnd")
    if (
        isinstance(line_start, bool)
        or not isinstance(line_start, int)
        or isinstance(line_end, bool)
        or not isinstance(line_end, int)
    ):
        raise OSError("ccc result has missing or non-integer source line numbers")
    if line_start < 1 or line_end < line_start:
        raise OSError(f"ccc returned an invalid source line range: {line_start}-{line_end}")

    snippet = str(_first(row, "code", "content", "text", "snippet", default="") or "")
    if not snippet:
        raise OSError("ccc result has no source snippet")
    try:
        source_bytes = source.read_bytes()
        source_text = source_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise OSError(f"ccc source cannot be read as UTF-8: {canonical_relative}") from error
    if hashlib.sha256(source_bytes).hexdigest() != digest:
        raise OSError(f"ccc source changed while validating: {canonical_relative}")

    lines = source_text.splitlines(keepends=True)
    if line_end > len(lines):
        raise OSError(
            f"ccc source line range {line_start}-{line_end} exceeds "
            f"{canonical_relative} ({len(lines)} lines)"
        )
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    span_start = line_offsets[line_start - 1]
    span_end = span_start + sum(len(line) for line in lines[line_start - 1 : line_end])

    occurrence = source_text.find(snippet, span_start, span_end)
    matched_range = False
    while occurrence != -1 and occurrence + len(snippet) <= span_end:
        actual_start = bisect_right(line_offsets, occurrence)
        actual_end = bisect_right(line_offsets, occurrence + len(snippet) - 1)
        if actual_start == line_start and actual_end == line_end:
            matched_range = True
            break
        occurrence = source_text.find(snippet, occurrence + 1, span_end)
    if not matched_range:
        raise OSError(
            f"ccc snippet does not match {canonical_relative}:{line_start}-{line_end}"
        )

    bound = dict(row)
    # Canonical keys prevent conflicting aliases in an untrusted response from
    # changing what the converter persists after the checks above.
    bound["path"] = canonical_relative.as_posix()
    bound["line_start"] = line_start
    bound["line_end"] = line_end
    bound["code"] = snippet
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


def _reported_unsuccessful(call_result: Any) -> bool:
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict) and structured.get("success") is False:
        return True
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("success") is False:
            return True
    return False


def _checked_results(call_result: Any) -> list[dict]:
    """A tool failure or malformed response is not an empty code search."""
    if bool(getattr(call_result, "isError", False)):
        raise BackendUnavailable("ccc MCP search tool returned an error")
    if _reported_unsuccessful(call_result):
        raise BackendUnavailable("ccc MCP search reported an unsuccessful result")
    rows = _extract_results(call_result)
    if not rows:
        if _explicit_empty_result(call_result):
            return []
        raise BackendUnavailable("ccc MCP search returned no parseable result envelope")
    for row in rows:
        path = _first(row, "path", "file", "filename", "file_path")
        text = _first(row, "code", "content", "text", "snippet")
        line_start = _first(row, "line_start", "start_line", "start", "lineStart")
        line_end = _first(row, "line_end", "end_line", "end", "lineEnd")
        if not path or not text or line_start is None or line_end is None:
            raise BackendUnavailable(
                "ccc MCP search returned a malformed result without path/text/line range"
            )
    return rows


class CccBackend(SearchBackend):
    name = "ccc"
    kind = "code"

    def __init__(
        self,
        root: Path,
        *,
        mcp_initialize_timeout_s: float = _MCP_INITIALIZE_TIMEOUT_S,
        mcp_search_timeout_s: float = _MCP_SEARCH_TIMEOUT_S,
    ):
        self.root = Path(root)
        self._ccc = find_ccc()
        self._mcp_initialize_timeout_s = _checked_timeout(
            mcp_initialize_timeout_s,
            name="mcp_initialize_timeout_s",
        )
        self._mcp_search_timeout_s = _checked_timeout(
            mcp_search_timeout_s,
            name="mcp_search_timeout_s",
        )

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
                await _await_mcp(
                    session.initialize(),
                    timeout_s=self._mcp_initialize_timeout_s,
                    operation="initialization",
                )
                result = await _await_mcp(
                    session.call_tool("search", args),
                    timeout_s=self._mcp_search_timeout_s,
                    operation="search",
                )
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
                init = _run_ccc_control(
                    [self._ccc, "init", "-f"],
                    root=self.root,
                    env=env,
                    timeout_s=120,
                )
                if init.returncode != 0:
                    return ReindexResult(
                        kind=self.kind, ok=False, version_before=version_before,
                        detail=f"ccc init failed (exit {init.returncode}): {init.stderr[-300:]}",
                    )
            proc = _run_ccc_control(
                [self._ccc, "index"],
                root=self.root,
                env=env,
                timeout_s=600,
            )
        except OSError as e:
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
