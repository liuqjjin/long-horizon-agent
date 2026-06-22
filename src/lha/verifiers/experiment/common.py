"""Shared helpers for experiment verifiers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..verdict import Check, VerifierFamily


def is_finite(x: Any) -> bool:
    """True only for a real, finite number (rejects None/NaN/inf/garbage)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def metric_param(artifact: Any, step: Any, key: str, default: Any) -> tuple[Any, bool]:
    """Resolve a metric param (e.g. ``data_range``), preferring what the experiment recorded.

    The experiment writes the value it actually used into ``repro.json``; the task may
    pin an override in ``step.params``. Returns ``(value, conflict)``: when both are
    present and disagree, ``conflict`` is True — a mismatched ``data_range``/``channel_axis``
    makes the recomputed metric (and its consistency check vs the self-reported value)
    meaningless, so the caller must fail rather than silently rescale.
    """
    recorded = (getattr(artifact, "repro", {}) or {}).get(key)
    override = step.params.get(key)
    if override is not None and recorded is not None and override != recorded:
        return override, True
    if override is not None:
        return override, False
    if recorded is not None:
        return recorded, False
    return default, False


def precheck(artifact: Any, name: str, family: VerifierFamily = "experiment") -> Check | None:
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
        # allow_pickle=False (numpy's default since 2.x) keeps a malicious .npy from
        # executing pickle on load — make it explicit so it can't drift.
        return (
            np.load(workdir / rp, allow_pickle=False),
            np.load(workdir / pp, allow_pickle=False),
        )
    except (OSError, ValueError):
        return None, None
