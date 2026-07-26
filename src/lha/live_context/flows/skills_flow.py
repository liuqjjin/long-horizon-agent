"""Build the verified-skill context index."""

from __future__ import annotations

from .common import DEFAULT_MODEL, make_app


def build(
    sourcedir: str = "data/skills",
    outdir: str = "data/.lha_index/skills",
    embedder_model: str = DEFAULT_MODEL,
):
    return make_app("LhaSkills", sourcedir, outdir, "skill", embedder_model)
