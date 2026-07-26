"""Dependency-light entry point for the optional CocoIndex document flows.

Importing the public flow modules is part of the core package smoke test.  The
actual flow implementation is imported only when ``build`` is called, so a
plain ``lha`` installation can inspect these modules without installing the
``context`` extra.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "doc"


def make_app(
    name: str,
    sourcedir: str | pathlib.Path,
    outdir: str | pathlib.Path,
    kind: str,
    embedder_model: str = DEFAULT_MODEL,
) -> Any:
    """Build a CocoIndex app, loading optional dependencies on demand."""
    from ._coco_impl import make_app as make_coco_app

    return make_coco_app(name, sourcedir, outdir, kind, embedder_model)
