"""CocoIndex flow that indexes experiment logs (data/experiments/*.md)."""

from __future__ import annotations

from .common import make_app


def build(sourcedir: str = "data/experiments", outdir: str = "data/.lha_index/experiments"):
    return make_app("LhaExperiments", sourcedir, outdir, "experiment")
