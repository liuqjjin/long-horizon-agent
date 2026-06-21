"""CocoIndex flow that indexes paper notes (data/papers/*.md)."""

from __future__ import annotations

from .common import make_app


def build(sourcedir: str = "data/papers", outdir: str = "data/.lha_index/papers"):
    return make_app("LhaPapers", sourcedir, outdir, "paper")
