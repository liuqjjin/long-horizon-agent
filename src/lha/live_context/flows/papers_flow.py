"""Build the paper-note context index."""

from __future__ import annotations

from .common import DEFAULT_MODEL, make_app


def build(
    sourcedir: str = "data/papers",
    outdir: str = "data/.lha_index/papers",
    embedder_model: str = DEFAULT_MODEL,
):
    return make_app("LhaPapers", sourcedir, outdir, "paper", embedder_model)
