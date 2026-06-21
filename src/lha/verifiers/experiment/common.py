"""Shared helpers for experiment verifiers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..verdict import Check


def is_finite(x: Any) -> bool:
    """True only for a real, finite number (rejects None/NaN/inf/garbage)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def precheck(artifact: Any, name: str, family: str = "experiment") -> Check | None:
    """Fail fast if the experiment that produced the artifact did not succeed.

    A nonzero exit means the metrics/arrays can't be trusted, so no experiment
    verifier should pass regardless of what files were left behind.
    """
    rc = getattr(artifact, "returncode", 0)
    if rc != 0:
        return Check(
            name=name,
            family=family,
            passed=False,
            detail={"summary": f"experiment command failed (returncode={rc})"},
        )
    return None


def load_arrays(artifact: Any, workdir: str | Path):
    """Load (reference, prediction) numpy arrays from an ExperimentResult, or (None, None)."""
    import numpy as np

    workdir = Path(workdir)
    rp = getattr(artifact, "reference_path", None)
    pp = getattr(artifact, "prediction_path", None)
    if not rp or not pp:
        return None, None
    try:
        return np.load(workdir / rp), np.load(workdir / pp)
    except (OSError, ValueError):
        return None, None
