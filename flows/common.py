"""Shared CocoIndex flow: walk markdown notes -> chunk -> embed -> JSON target.

One App per kind (papers / experiments). The per-file function is memoized, so an
unchanged note is skipped on re-index; the LocalFS target is declarative, so
chunks of deleted/shrunken files are cleaned up automatically. The embedded JSON
records are read back at query time by ``live_context.backends.vector_query``.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import AsyncIterator

import cocoindex as coco
import frontmatter
from cocoindex.connectors import localfs
from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder
from cocoindex.ops.text import RecursiveSplitter
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER = coco.ContextKey[SentenceTransformerEmbedder]("lha_embedder")
_splitter = RecursiveSplitter()


@coco.lifespan
async def _lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    builder.provide(EMBEDDER, SentenceTransformerEmbedder(MODEL))
    yield


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "doc"


@coco.fn(memo=True)
async def process_note(file: FileLike, outdir: pathlib.Path, kind: str, sourcedir: str) -> None:
    raw = await file.read_text()
    post = frontmatter.loads(raw)
    meta = dict(post.metadata)
    body = post.content or raw
    rel = str(file.file_path.path)
    stem = _sanitize(rel)

    chunks = _splitter.split(body, chunk_size=1200, chunk_overlap=200)
    embedder = coco.use_context(EMBEDDER)
    for idx, chunk in enumerate(chunks):
        emb = await embedder.embed(chunk.text)
        record = {
            "kind": kind,
            "sourcedir": sourcedir,
            "source_path": rel,
            "chunk_index": idx,
            "chunk_start": chunk.start.char_offset,
            "chunk_end": chunk.end.char_offset,
            "text": chunk.text,
            "embedding": [float(x) for x in emb.tolist()],
            "metadata": meta,
        }
        localfs.declare_file(
            outdir / f"{kind}__{stem}__{idx}.json",
            json.dumps(record, default=str),  # frontmatter may contain dates etc.
            create_parent_dirs=True,
        )


@coco.fn
async def app_main(sourcedir: pathlib.Path, outdir: pathlib.Path, kind: str) -> None:
    files = localfs.walk_dir(
        sourcedir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(included_patterns=["**/*.md"]),
    )
    await coco.mount_each(process_note, files.items(), outdir, kind, str(sourcedir))


def make_app(
    name: str, sourcedir: str | pathlib.Path, outdir: str | pathlib.Path, kind: str
) -> coco.App:
    return coco.App(
        coco.AppConfig(name=name),
        app_main,
        sourcedir=pathlib.Path(sourcedir),
        outdir=pathlib.Path(outdir),
        kind=kind,
    )
