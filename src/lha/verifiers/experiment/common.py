"""Shared helpers for experiment verifiers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ...array_evidence import load_bounded_npy, safe_artifact_path
from ..verdict import Check, VerifierFamily


def is_finite(x: Any) -> bool:
    """True only for a real, finite number (rejects None/NaN/inf/garbage)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def recompute_image_metrics(
    artifact: Any,
    step: Any,
    reference: Any,
    prediction: Any,
) -> dict[str, float]:
    """Recompute the experiment family's metrics from arrays, never metrics.json."""
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    ref = np.asarray(reference)
    pred = np.asarray(prediction)
    if ref.shape != pred.shape:
        raise ValueError(f"reference/prediction shape mismatch: {ref.shape} != {pred.shape}")
    if ref.size == 0 or not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise ValueError("reference/prediction arrays are empty or non-finite")

    data_range_v, dr_conflict = metric_param(artifact, step, "data_range", 1.0)
    if dr_conflict:
        raise ValueError("task data_range conflicts with experiment evidence")
    if not is_finite(data_range_v) or float(data_range_v) <= 0:
        raise ValueError(f"invalid data_range: {data_range_v!r}")
    data_range = float(data_range_v)

    channel_axis, ca_conflict = metric_param(
        artifact, step, "channel_axis", -1 if ref.ndim == 3 else None
    )
    if ca_conflict:
        raise ValueError("task channel_axis conflicts with experiment evidence")
    if channel_axis is not None and not isinstance(channel_axis, int):
        raise ValueError(f"invalid channel_axis: {channel_axis!r}")

    kwargs: dict[str, Any] = {"data_range": data_range, "channel_axis": channel_axis}
    if "win_size" in step.params:
        kwargs["win_size"] = int(step.params["win_size"])
    metrics = {
        "psnr": float(peak_signal_noise_ratio(ref, pred, data_range=data_range)),
        "ssim": float(structural_similarity(ref, pred, **kwargs)),  # type: ignore[arg-type]
    }
    bad = {name: value for name, value in metrics.items() if not is_finite(value)}
    if bad:
        raise ValueError(f"recomputed metrics are non-finite: {bad}")
    return metrics


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
    if getattr(artifact, "output_truncated", False):
        return Check(
            name=name,
            family=family,
            passed=False,
            detail={
                "summary": (
                    "experiment command output exceeded the capture limit; "
                    "evidence is incomplete"
                )
            },
        )
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
    rp = getattr(artifact, "reference_path", None)
    pp = getattr(artifact, "prediction_path", None)
    if not rp or not pp:
        return None, None
    try:
        return (
            load_bounded_npy(safe_artifact_path(workdir, rp)),
            load_bounded_npy(safe_artifact_path(workdir, pp)),
        )
    except (MemoryError, OSError, OverflowError, ValueError):
        return None, None
