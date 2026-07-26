"""Build the experiment-note context index."""

from __future__ import annotations

from .common import DEFAULT_MODEL, make_app


def build(
    sourcedir: str = "data/experiments",
    outdir: str = "data/.lha_index/experiments",
    embedder_model: str = DEFAULT_MODEL,
):
    return make_app("LhaExperiments", sourcedir, outdir, "experiment", embedder_model)
