"""Paper / experiment context backed by CocoIndex BUILD flows.

- ``reindex()`` runs the CocoIndex App (incremental + memoized) to (re)build the
  embedded JSON chunk records under ``data/.lha_index/<kind>s/``.
- ``search()`` reads those records and does cosine vector search (no CocoIndex at
  query time).

This module + ``ccc_backend`` are the ONLY places allowed to touch CocoIndex.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...clock import now
from ..freshness import content_hash, strict_file_sha256
from ..models import (
    ExperimentHit,
    Hit,
    PaperHit,
    Provenance,
    ReindexResult,
    SkillHit,
    SourceKind,
)
from . import vector_query
from .base import SearchBackend
from .embedder import DEFAULT_MODEL

_MANIFEST_SCHEMA = 1
_RECORD_SCHEMA = 3
_COMPLETE_FILE = ".complete"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(path.parent)
    finally:
        temp.unlink(missing_ok=True)


class CocoFlowBackend(SearchBackend):
    def __init__(
        self,
        kind: SourceKind,
        data_dir: str | Path,
        embedder_model: str = DEFAULT_MODEL,
    ):
        self.name = f"coco:{kind}"
        self.kind = kind
        self.data_dir = Path(data_dir)
        self.sourcedir = self.data_dir / f"{kind}s"
        self.index_dir = self.data_dir / ".lha_index" / f"{kind}s"
        self.embedder_model = embedder_model

    @property
    def building_path(self) -> Path:
        return self.index_dir.parent / f".{self.index_dir.name}.building"

    @property
    def lock_path(self) -> Path:
        return self.index_dir.parent / f".{self.index_dir.name}.lock"

    def has_index_state(self) -> bool:
        """Whether there is persisted state that a query must validate or reject."""
        if self.building_path.exists() or self.building_path.is_symlink():
            return True
        if not self.index_dir.exists():
            return False
        try:
            return any(self.index_dir.iterdir())
        except OSError:
            return True

    def available(self) -> bool:
        try:
            self._validate_generation()
        except vector_query.IndexRecordError:
            return False
        return True

    # --- query path (no CocoIndex) -----------------------------------------
    def search(self, query: str, *, k: int = 5, **filters) -> list[Hit]:
        metric = filters.get("metric")
        if not self.has_index_state():
            from .base import BackendUnavailable

            raise BackendUnavailable(f"{self.name} has no completed index generation")
        try:
            manifest = self._validate_generation()
            _iv, indexed_at = self._meta_from_manifest(manifest)
            expected_records = {
                str(record["path"]): str(record["sha256"])
                for record in manifest["record_files"]
            }
            rows = vector_query.search(
                self.index_dir,
                query,
                k,
                metric=metric,
                embedder_model=self.embedder_model,
                expected_records=expected_records,
            )
            return [self._to_hit(r, indexed_at) for r in rows]
        except vector_query.IndexRecordError:
            # The backend is present, but its persisted corpus is corrupt or
            # incompatible. Let the facade classify this as ``index_failed``;
            # reporting it as an unavailable optional dependency hides damage.
            raise
        except Exception as e:  # embedder/IO failure is not an empty result
            from .base import BackendUnavailable

            raise BackendUnavailable(f"{self.name} search failed: {type(e).__name__}: {e}") from e

    def _to_hit(self, rec: dict, indexed_at: datetime) -> Hit:
        loc = rec.get("source_path", rec.get("locator", "?"))
        score = float(rec.get("score", 0.0))
        meta = rec.get("metadata", {}) or {}
        prov = Provenance(
            source_kind=self.kind,
            locator=loc,
            score=score,
            indexed_at=indexed_at,
            content_hash=rec.get("chunk_sha256") or content_hash(rec.get("text", "")),
            source_sha256=rec.get("source_sha256"),
            source_root=str(self.sourcedir.resolve()),
        )
        if self.kind == "paper":
            return PaperHit(
                text=rec.get("text", ""),
                score=score,
                provenance=prov,
                title=meta.get("title"),
                paper_id=loc,
            )
        if self.kind == "skill":
            return SkillHit(
                text=rec.get("text", ""),
                score=score,
                provenance=prov,
                title=meta.get("title"),
                skill_id=meta.get("skill_id") or loc,  # the originating run_id
            )
        metrics_raw = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
        metrics = {
            str(k): float(v) for k, v in (metrics_raw or {}).items() if isinstance(v, (int, float))
        }
        return ExperimentHit(
            text=rec.get("text", ""),
            score=score,
            provenance=prov,
            run_id=meta.get("run_id"),
            metrics=metrics,
        )

    def index_meta(self) -> tuple[str, datetime]:
        try:
            return self._meta_from_manifest(self._validate_generation())
        except vector_query.IndexRecordError:
            if self.has_index_state():
                return (f"coco:{self.kind}@invalid", now())
        return (f"coco:{self.kind}@uninitialized", now())

    # --- build path (runs CocoIndex) ---------------------------------------
    def reindex(self, paths: list[str] | None = None) -> ReindexResult:
        version_before, _ = self.index_meta()
        parent = self.index_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            lock_fd = os.open(self.lock_path, flags, 0o600)
        except OSError as error:
            return ReindexResult(
                kind=self.kind,
                ok=False,
                version_before=version_before,
                detail=f"context index lock could not be opened: {type(error).__name__}: {error}",
            )
        acquired = False
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                return ReindexResult(
                    kind=self.kind,
                    ok=False,
                    version_before=version_before,
                    detail=f"{self.kind} context index is busy in another process",
                )
            except OSError as error:
                return ReindexResult(
                    kind=self.kind,
                    ok=False,
                    version_before=version_before,
                    detail=f"context index lock failed: {type(error).__name__}: {error}",
                )
            return self._reindex_locked(paths, version_before=version_before)
        finally:
            try:
                if acquired:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    def _reindex_locked(
        self,
        paths: list[str] | None,
        *,
        version_before: str,
    ) -> ReindexResult:
        generation_id = uuid.uuid4().hex
        parent = self.index_dir.parent
        staging = parent / f".{self.index_dir.name}.staging-{generation_id}"
        backup = parent / f".{self.index_dir.name}.previous-{generation_id}"
        sentinel_created = False
        old_moved = False
        new_installed = False
        try:
            import cocoindex as coco

            parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                self.building_path,
                _canonical_json(
                    {
                        "generation_id": generation_id,
                        "kind": self.kind,
                        "model": self.embedder_model,
                        "started_at": now().isoformat(),
                    }
                ),
            )
            sentinel_created = True
            staging.mkdir()

            state_dir = (parent / f"state-{self.kind}.db").resolve()
            previous_db = os.environ.get("COCOINDEX_DB")
            state_dir.mkdir(parents=True, exist_ok=True)
            os.environ["COCOINDEX_DB"] = str(state_dir)
            try:
                app = self._build_app(staging)
                with coco.runtime():
                    app.update_blocking(report_to_stdout=False)
            finally:
                if previous_db is None:
                    os.environ.pop("COCOINDEX_DB", None)
                else:
                    os.environ["COCOINDEX_DB"] = previous_db

            self._seal_generation(staging, generation_id=generation_id)
            self._validate_generation(directory=staging, ignore_building=True)
            if self.index_dir.exists() or self.index_dir.is_symlink():
                os.replace(self.index_dir, backup)
                old_moved = True
            os.replace(staging, self.index_dir)
            new_installed = True
            _fsync_dir(parent)
            if backup.exists():
                shutil.rmtree(backup)
            self.building_path.unlink()
            _fsync_dir(parent)
            sentinel_created = False
        except ImportError as e:  # optional dependency not installed
            self._recover_failed_build(
                staging, backup, old_moved=old_moved, new_installed=new_installed
            )
            if sentinel_created and not new_installed:
                self.building_path.unlink(missing_ok=True)
                _fsync_dir(parent)
            return ReindexResult(
                kind=self.kind, ok=False, version_before=version_before,
                detail=f"cocoindex is not installed (pip install 'lha[context]'): {e}",
            )
        except Exception as e:  # flow error, IO — all mean "not refreshed"
            self._recover_failed_build(
                staging, backup, old_moved=old_moved, new_installed=new_installed
            )
            # A failure before the directory switch leaves the previous
            # generation untouched.  Once a new directory was installed, retain
            # the sentinel so readers fail closed until another reindex repairs it.
            if sentinel_created and not new_installed:
                self.building_path.unlink(missing_ok=True)
                _fsync_dir(parent)
            return ReindexResult(
                kind=self.kind, ok=False, version_before=version_before,
                detail=f"cocoindex flow failed: {type(e).__name__}: {e}",
            )
        version_after, _ = self.index_meta()
        if not self.available():
            return ReindexResult(
                kind=self.kind,
                ok=False,
                version_before=version_before,
                version_after=version_after,
                detail="cocoindex flow produced no validated completion generation",
            )
        return ReindexResult(
            kind=self.kind, ok=True, version_before=version_before, version_after=version_after
        )

    def _build_app(self, outdir: Path):
        if self.kind == "paper":
            from ..flows.papers_flow import build
        elif self.kind == "skill":
            from ..flows.skills_flow import build
        else:
            from ..flows.experiments_flow import build
        return build(str(self.sourcedir), str(outdir), self.embedder_model)

    def _source_snapshot(self) -> list[dict[str, str]]:
        if not self.sourcedir.exists():
            return []
        if self.sourcedir.is_symlink() or not self.sourcedir.is_dir():
            raise vector_query.IndexRecordError("context source root is not a safe directory")
        root = self.sourcedir.resolve(strict=True)
        sources: list[dict[str, str]] = []
        for path in sorted(self.sourcedir.rglob("*")):
            if path.is_symlink():
                raise vector_query.IndexRecordError(
                    f"context source tree contains a symlink: {path}"
                )
            if not path.is_file() or path.suffix != ".md":
                continue
            try:
                relative = path.relative_to(self.sourcedir).as_posix()
                digest = strict_file_sha256(root / relative, root=root)
            except (OSError, RuntimeError, ValueError) as error:
                raise vector_query.IndexRecordError(
                    f"cannot bind context source {path}: {type(error).__name__}: {error}"
                ) from error
            sources.append({"path": relative, "sha256": digest})
        return sources

    def _record_snapshot(
        self,
        directory: Path,
        *,
        sources: list[dict[str, str]],
        expected_chunks: dict[str, list[dict[str, int | str]]] | None = None,
        declared_chunk_counts: dict[str, int] | None = None,
    ) -> tuple[list[dict[str, str]], int, dict[str, int]]:
        paths = sorted(directory.glob("*.json"))
        source_digests = {entry["path"]: entry["sha256"] for entry in sources}
        records = vector_query.load_records(directory)
        if len(records) != len(paths):
            raise vector_query.IndexRecordError("record count changed while validating generation")
        dimensions: set[int] = set()
        identities: set[tuple[str, int]] = set()
        records_by_source: dict[str, list[dict[str, Any]]] = {
            source_path: [] for source_path in source_digests
        }
        files: list[dict[str, str]] = []
        for path, record in zip(paths, records, strict=True):
            source_path = record.get("source_path")
            source_rel = Path(str(source_path))
            if (
                not isinstance(source_path, str)
                or source_rel.is_absolute()
                or ".." in source_rel.parts
                or source_path not in source_digests
                or record.get("source_sha256") != source_digests.get(source_path)
            ):
                raise vector_query.IndexRecordError(
                    f"{path.name}: source evidence does not match the indexed source set"
                )
            embedding = record.get("embedding")
            dimension = record.get("embedding_dimension")
            chunk_index = record.get("chunk_index")
            chunk_start = record.get("chunk_start")
            chunk_end = record.get("chunk_end")
            if (
                record.get("schema_version") != _RECORD_SCHEMA
                or record.get("embedder_model") != self.embedder_model
                or record.get("kind") != self.kind
                or not isinstance(embedding, list)
                or not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                or dimension != len(embedding)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in embedding
                )
                or not isinstance(chunk_index, int)
                or isinstance(chunk_index, bool)
                or chunk_index < 0
                or not isinstance(chunk_start, int)
                or not isinstance(chunk_end, int)
                or isinstance(chunk_start, bool)
                or isinstance(chunk_end, bool)
                or chunk_start < 0
                or chunk_end <= chunk_start
                or record.get("chunk_sha256") != content_hash(str(record.get("text", "")))
            ):
                raise vector_query.IndexRecordError(
                    f"{path.name}: schema, model, or chunk evidence is inconsistent"
                )
            identity = (source_path, chunk_index)
            if identity in identities:
                raise vector_query.IndexRecordError(
                    f"{path.name}: duplicate source/chunk identity {identity!r}"
                )
            identities.add(identity)
            dimensions.add(dimension)
            records_by_source[source_path].append(record)
            try:
                digest = strict_file_sha256(Path(path.name), root=directory)
            except (OSError, RuntimeError) as error:
                raise vector_query.IndexRecordError(
                    f"{path.name}: record cannot be read safely"
                ) from error
            files.append({"path": path.name, "sha256": digest})
        if len(dimensions) > 1:
            raise vector_query.IndexRecordError("index contains mixed embedding dimensions")

        chunk_counts: dict[str, int] = {}
        for source_path, source_records in records_by_source.items():
            ordered = sorted(source_records, key=lambda record: int(record["chunk_index"]))
            indices = [int(record["chunk_index"]) for record in ordered]
            if indices != list(range(len(ordered))):
                raise vector_query.IndexRecordError(
                    f"{source_path}: chunk indexes are not continuous from zero"
                )
            chunk_counts[source_path] = len(ordered)
            if declared_chunk_counts is not None:
                declared = declared_chunk_counts.get(source_path)
                if declared is None or declared != len(ordered):
                    raise vector_query.IndexRecordError(
                        f"{source_path}: chunk count does not match the completion manifest"
                    )
            if expected_chunks is not None:
                expected = expected_chunks.get(source_path)
                if expected is None or len(expected) != len(ordered):
                    raise vector_query.IndexRecordError(
                        f"{source_path}: partial source index; expected "
                        f"{len(expected or [])} chunk(s), found {len(ordered)}"
                    )
                for actual, wanted in zip(ordered, expected, strict=True):
                    if any(
                        actual.get(field) != wanted[field]
                        for field in (
                            "chunk_index",
                            "chunk_start",
                            "chunk_end",
                            "text",
                            "chunk_sha256",
                        )
                    ):
                        raise vector_query.IndexRecordError(
                            f"{source_path}: chunk text or offsets do not match the source"
                        )
        if declared_chunk_counts is not None and set(declared_chunk_counts) != set(
            records_by_source
        ):
            raise vector_query.IndexRecordError(
                "completion manifest chunk counts do not match the source set"
            )
        return files, next(iter(dimensions), 0), chunk_counts

    def _expected_chunk_evidence(
        self,
        sources: list[dict[str, str]],
    ) -> dict[str, list[dict[str, int | str]]]:
        # Loaded only during a build/seal. Query validation remains usable in a
        # core installation that does not have the CocoIndex extra.
        from ..flows._coco_impl import chunk_evidence

        root = self.sourcedir.resolve(strict=True)
        expected: dict[str, list[dict[str, int | str]]] = {}
        for source in sources:
            path = root / source["path"]
            try:
                raw = vector_query._read_regular_file(path)
            except OSError as error:
                raise vector_query.IndexRecordError(
                    f"cannot read source while sealing generation: {source['path']}"
                ) from error
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise vector_query.IndexRecordError(
                    f"source changed while sealing generation: {source['path']}"
                )
            try:
                expected[source["path"]] = chunk_evidence(raw)
            except (UnicodeDecodeError, ValueError) as error:
                raise vector_query.IndexRecordError(
                    f"cannot deterministically split source: {source['path']}"
                ) from error
        return expected

    def _seal_generation(self, directory: Path, *, generation_id: str) -> dict[str, Any]:
        sources = self._source_snapshot()
        expected_chunks = self._expected_chunk_evidence(sources)
        record_files, dimension, chunk_counts = self._record_snapshot(
            directory,
            sources=sources,
            expected_chunks=expected_chunks,
        )
        manifest_sources = [
            {
                "path": source["path"],
                "sha256": source["sha256"],
                "chunk_count": chunk_counts[source["path"]],
            }
            for source in sources
        ]
        payload: dict[str, Any] = {
            "manifest_schema": _MANIFEST_SCHEMA,
            "record_schema": _RECORD_SCHEMA,
            "generation_id": generation_id,
            "kind": self.kind,
            "embedder_model": self.embedder_model,
            "embedding_dimension": dimension,
            "source_files": manifest_sources,
            "record_files": record_files,
            "record_count": len(record_files),
            "completed_at": now().isoformat(),
        }
        envelope = {
            "schema_version": _MANIFEST_SCHEMA,
            "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
            "payload": payload,
        }
        _atomic_write(directory / _COMPLETE_FILE, _canonical_json(envelope))
        return payload

    def _validate_generation(
        self,
        *,
        directory: Path | None = None,
        ignore_building: bool = False,
    ) -> dict[str, Any]:
        target = directory or self.index_dir
        if not ignore_building and (
            self.building_path.exists() or self.building_path.is_symlink()
        ):
            raise vector_query.IndexRecordError("context index has an unfinished build")
        if target.is_symlink() or not target.is_dir():
            raise vector_query.IndexRecordError("context index has no safe generation directory")
        complete = target / _COMPLETE_FILE
        if complete.is_symlink() or not complete.is_file():
            raise vector_query.IndexRecordError("context index has no completion manifest")
        try:
            envelope = json.loads(vector_query._read_regular_file(complete))
        except (OSError, json.JSONDecodeError) as error:
            raise vector_query.IndexRecordError(
                f"context completion manifest is unreadable: {type(error).__name__}"
            ) from error
        if not isinstance(envelope, dict) or envelope.get("schema_version") != _MANIFEST_SCHEMA:
            raise vector_query.IndexRecordError("context completion manifest schema is invalid")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise vector_query.IndexRecordError("context completion manifest payload is invalid")
        expected_checksum = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if envelope.get("sha256") != expected_checksum:
            raise vector_query.IndexRecordError("context completion manifest checksum mismatch")
        if (
            payload.get("manifest_schema") != _MANIFEST_SCHEMA
            or payload.get("record_schema") != _RECORD_SCHEMA
            or payload.get("kind") != self.kind
            or payload.get("embedder_model") != self.embedder_model
            or not isinstance(payload.get("generation_id"), str)
            or not isinstance(payload.get("completed_at"), str)
            or not isinstance(payload.get("source_files"), list)
            or not isinstance(payload.get("record_files"), list)
            or not isinstance(payload.get("record_count"), int)
            or not isinstance(payload.get("embedding_dimension"), int)
        ):
            raise vector_query.IndexRecordError(
                "context completion manifest does not match backend configuration"
            )
        sources = self._source_snapshot()
        try:
            manifest_sources = [
                {"path": entry["path"], "sha256": entry["sha256"]}
                for entry in payload["source_files"]
                if isinstance(entry, dict)
            ]
            declared_chunk_counts = {
                str(entry["path"]): int(entry["chunk_count"])
                for entry in payload["source_files"]
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("path"), str)
                    and isinstance(entry.get("chunk_count"), int)
                    and not isinstance(entry.get("chunk_count"), bool)
                    and entry["chunk_count"] >= 0
                )
            }
        except (KeyError, TypeError, ValueError) as error:
            raise vector_query.IndexRecordError(
                "completion manifest source evidence is malformed"
            ) from error
        if (
            len(manifest_sources) != len(payload["source_files"])
            or len(declared_chunk_counts) != len(payload["source_files"])
            or manifest_sources != sources
        ):
            raise vector_query.IndexRecordError(
                "current context source set does not match the completed generation"
            )
        record_files, dimension, chunk_counts = self._record_snapshot(
            target,
            sources=sources,
            declared_chunk_counts=declared_chunk_counts,
        )
        if (
            payload["record_files"] != record_files
            or payload["record_count"] != len(record_files)
            or payload["embedding_dimension"] != dimension
            or chunk_counts != declared_chunk_counts
        ):
            raise vector_query.IndexRecordError(
                "context record set does not match the completion manifest"
            )
        return payload

    def _meta_from_manifest(self, manifest: dict[str, Any]) -> tuple[str, datetime]:
        try:
            completed_at = datetime.fromisoformat(str(manifest["completed_at"]))
            if completed_at.tzinfo is None:
                raise ValueError("timestamp has no timezone")
            completed_at = completed_at.astimezone(timezone.utc)
        except (KeyError, TypeError, ValueError) as error:
            raise vector_query.IndexRecordError(
                "context completion manifest has an invalid timestamp"
            ) from error
        generation = str(manifest["generation_id"])[:12]
        return (
            f"coco:{self.kind}:v{_RECORD_SCHEMA}:{self.embedder_model}:"
            f"{manifest['embedding_dimension']}@{generation}",
            completed_at,
        )

    def _recover_failed_build(
        self,
        staging: Path,
        backup: Path,
        *,
        old_moved: bool,
        new_installed: bool,
    ) -> None:
        if not new_installed and staging.exists():
            shutil.rmtree(staging)
        if old_moved and not new_installed and backup.exists() and not self.index_dir.exists():
            os.replace(backup, self.index_dir)
            _fsync_dir(self.index_dir.parent)
