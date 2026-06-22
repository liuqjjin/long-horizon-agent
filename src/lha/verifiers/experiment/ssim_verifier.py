"""Experiment verifier: independently recompute SSIM and gate on a threshold.

A non-finite result or a failed experiment is a verification FAILURE.
"""

from __future__ import annotations

from typing import Any

from ..base import Verifier, VerifyContext
from ..verdict import Check
from .common import is_finite, load_arrays, metric_param, precheck


class SSIMVerifier(Verifier):
    name = "ssim"
    family = "experiment"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        failed = precheck(artifact, self.name, self.family)
        if failed is not None:
            return failed

        ref, pred = load_arrays(artifact, ctx.workdir)
        if ref is None:
            return Check(
                name=self.name,
                family=self.family,
                passed=False,
                detail={"summary": "no reference/prediction arrays to score"},
            )

        from skimage.metrics import structural_similarity

        data_range_v, dr_conflict = metric_param(artifact, ctx.step, "data_range", 1.0)
        data_range = float(data_range_v)
        channel_axis, ca_conflict = metric_param(
            artifact, ctx.step, "channel_axis", -1 if ref.ndim == 3 else None
        )
        kwargs: dict[str, Any] = {"data_range": data_range, "channel_axis": channel_axis}
        if "win_size" in ctx.step.params:
            kwargs["win_size"] = int(ctx.step.params["win_size"])

        # ssim returns a float here; with **kwargs the stub widens to a union, so cast.
        value = float(structural_similarity(ref, pred, **kwargs))  # type: ignore[arg-type]
        threshold = ctx.step.params.get("ssim_min")
        reported = getattr(artifact, "metrics", {}).get("ssim")

        repro = getattr(artifact, "repro", {}) or {}
        reasons: list[str] = []
        if dr_conflict:
            reasons.append(
                f"data_range mismatch: task={ctx.step.params.get('data_range')} vs "
                f"experiment-recorded={repro.get('data_range')}"
            )
        if ca_conflict:
            reasons.append(
                f"channel_axis mismatch: task={ctx.step.params.get('channel_axis')} vs "
                f"experiment-recorded={repro.get('channel_axis')}"
            )
        if not is_finite(value):
            reasons.append(f"non-finite SSIM ({value}) — degenerate result")
        else:
            if threshold is not None and value < float(threshold):
                reasons.append(f"SSIM {value:.4f} < threshold {threshold}")
            if reported is not None:
                if not is_finite(reported):
                    reasons.append(f"self-reported SSIM is non-finite ({reported})")
                elif abs(value - float(reported)) > 0.01:
                    reasons.append(
                        f"recomputed {value:.4f} inconsistent with reported {float(reported):.4f}"
                    )

        score = value if is_finite(value) else None
        shown = f"{value:.4f}" if is_finite(value) else str(value)
        return Check(
            name=self.name,
            family=self.family,
            passed=not reasons,
            score=score,
            threshold=float(threshold) if threshold is not None else None,
            detail={
                "summary": f"SSIM={shown} (reported={reported}, threshold={threshold})",
                "reasons": reasons,
            },
        )
