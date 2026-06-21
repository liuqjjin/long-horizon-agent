"""CocoIndex flow that indexes skill notes (data/skills/*.md) — episodic memory."""

from __future__ import annotations

from .common import make_app


def build(sourcedir: str = "data/skills", outdir: str = "data/.lha_index/skills"):
    return make_app("LhaSkills", sourcedir, outdir, "skill")
