"""Experiment verifier: independently recompute PSNR and gate on a threshold.

Recomputing from the saved arrays, rather than trusting the experiment's
self-reported number, catches a fabricated metric. A non-finite result (NaN/inf,
e.g. a degenerate or diverged run) or a failed experiment is a verification failure,
never a silent pass.
"""

from __future__ import annotations

from typing import Any

from ..base import Verifier, VerifyContext
from ..verdict import Check
from .common import is_finite, load_arrays, metric_param, precheck


class PSNRVerifier(Verifier):
    name = "psnr"
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

        from skimage.metrics import peak_signal_noise_ratio

        data_range_v, dr_conflict = metric_param(artifact, ctx.step, "data_range", 1.0)
        data_range = float(data_range_v)
        value = float(peak_signal_noise_ratio(ref, pred, data_range=data_range))
        threshold = ctx.step.params.get("psnr_min")
        reported = getattr(artifact, "metrics", {}).get("psnr")

        reasons: list[str] = []
        if dr_conflict:
            reasons.append(
                f"data_range mismatch: task={ctx.step.params.get('data_range')} vs "
                f"experiment-recorded={(getattr(artifact, 'repro', {}) or {}).get('data_range')}"
            )
        if not is_finite(value):
            reasons.append(f"non-finite PSNR ({value}) — degenerate result")
        else:
            if threshold is not None and value < float(threshold):
                reasons.append(f"PSNR {value:.3f} < threshold {threshold}")
            if reported is not None:
                if not is_finite(reported):
                    reasons.append(f"self-reported PSNR is non-finite ({reported})")
                elif abs(value - float(reported)) > 1.0:
                    reasons.append(
                        f"recomputed {value:.3f} dB inconsistent with reported {float(reported):.3f} dB"
                    )

        score = value if is_finite(value) else None
        shown = f"{value:.3f} dB" if is_finite(value) else str(value)
        return Check(
            name=self.name,
            family=self.family,
            passed=not reasons,
            score=score,
            threshold=float(threshold) if threshold is not None else None,
            detail={
                "summary": f"PSNR={shown} (reported={reported}, threshold={threshold})",
                "reasons": reasons,
            },
        )
