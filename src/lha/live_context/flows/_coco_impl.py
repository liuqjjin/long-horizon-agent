"""CocoIndex implementation loaded only when a document index is built."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import AsyncIterator

import cocoindex as coco
import frontmatter
from cocoindex.connectors import localfs
from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder
from cocoindex.ops.text import RecursiveSplitter
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from .common import DEFAULT_MODEL, _sanitize

_active_model = DEFAULT_MODEL
EMBEDDER = coco.ContextKey[SentenceTransformerEmbedder]("lha_embedder")
_splitter = RecursiveSplitter()


def chunk_evidence(raw_bytes: bytes) -> list[dict[str, int | str]]:
    """Return the deterministic chunk sequence used by both build and sealing."""
    raw = raw_bytes.decode("utf-8")
    post = frontmatter.loads(raw)
    body = post.content or raw
    chunks = _splitter.split(body, chunk_size=1200, chunk_overlap=200)
    return [
        {
            "chunk_index": index,
            "chunk_start": chunk.start.char_offset,
            "chunk_end": chunk.end.char_offset,
            "text": chunk.text,
            "chunk_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        }
        for index, chunk in enumerate(chunks)
    ]


@coco.lifespan
async def _lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    builder.provide(EMBEDDER, SentenceTransformerEmbedder(_active_model))
    yield


@coco.fn(memo=True)
async def process_note(
    file: FileLike,
    outdir: pathlib.Path,
    kind: str,
    sourcedir: str,
    embedder_model: str,
) -> None:
    raw_bytes = await file.read()
    raw = raw_bytes.decode("utf-8")
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    post = frontmatter.loads(raw)
    meta = dict(post.metadata)
    raw_path = pathlib.Path(str(file.file_path.path))
    source_root = pathlib.Path(sourcedir).resolve()
    if raw_path.is_absolute():
        source_path = raw_path.resolve()
    else:
        cwd_path = raw_path.resolve()
        source_path = (
            cwd_path
            if cwd_path.is_relative_to(source_root)
            else (source_root / raw_path).resolve()
        )
    rel = source_path.relative_to(source_root).as_posix()
    stem = _sanitize(rel)

    chunks = chunk_evidence(raw_bytes)
    embedder = coco.use_context(EMBEDDER)
    for chunk in chunks:
        text = str(chunk["text"])
        embedding = await embedder.embed(text)
        vector = [float(value) for value in embedding.tolist()]
        record = {
            "schema_version": 3,
            "embedder_model": embedder_model,
            "embedding_dimension": len(vector),
            "kind": kind,
            "sourcedir": sourcedir,
            "source_path": rel,
            "source_sha256": source_sha256,
            "chunk_sha256": chunk["chunk_sha256"],
            "chunk_index": chunk["chunk_index"],
            "chunk_start": chunk["chunk_start"],
            "chunk_end": chunk["chunk_end"],
            "text": text,
            "embedding": vector,
            "metadata": meta,
        }
        localfs.declare_file(
            outdir / f"{kind}__{stem}__{chunk['chunk_index']}.json",
            json.dumps(record, default=str),
            create_parent_dirs=True,
        )


@coco.fn
async def app_main(
    sourcedir: pathlib.Path,
    outdir: pathlib.Path,
    kind: str,
    embedder_model: str,
) -> None:
    files = localfs.walk_dir(
        sourcedir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(included_patterns=["**/*.md"]),
    )
    await coco.mount_each(
        process_note,
        files.items(),
        outdir,
        kind,
        str(sourcedir),
        embedder_model,
    )


def make_app(
    name: str,
    sourcedir: str | pathlib.Path,
    outdir: str | pathlib.Path,
    kind: str,
    embedder_model: str = DEFAULT_MODEL,
) -> coco.App:
    global _active_model
    _active_model = embedder_model
    model_suffix = _sanitize(embedder_model)[-40:]
    return coco.App(
        coco.AppConfig(name=f"{name}_{model_suffix}"),
        app_main,
        sourcedir=pathlib.Path(sourcedir),
        outdir=pathlib.Path(outdir),
        kind=kind,
        embedder_model=embedder_model,
    )
