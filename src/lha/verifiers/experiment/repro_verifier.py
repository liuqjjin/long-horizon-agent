"""Experiment verifier: reproducibility.

Passes only if (a) a seed and library versions were recorded, and (b) re-running
the experiment reproduces the same metrics within tolerance. The re-run writes to
a separate output dir so the original artifact stays intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...tools.shell import run
from ..base import Verifier, VerifyContext
from ..verdict import Check
from .common import is_finite


class ReproVerifier(Verifier):
    name = "reproducibility"
    family = "experiment"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        repro = getattr(artifact, "repro", {}) or {}
        has_seed = repro.get("seed") is not None
        has_versions = bool(repro.get("versions"))
        has_commit = bool(repro.get("git_commit"))

        deterministic, rerun_detail = self._determinism(artifact, ctx)

        reasons: list[str] = []
        rc = getattr(artifact, "returncode", 0)
        if rc != 0:
            reasons.append(f"original experiment failed (returncode={rc})")
        if not has_seed:
            reasons.append("no seed recorded")
        if not has_versions:
            reasons.append("no library versions recorded")
        if not deterministic:
            reasons.append(f"not deterministic ({rerun_detail})")

        return Check(
            name=self.name,
            family=self.family,
            passed=not reasons,
            detail={
                "summary": (
                    f"seed={has_seed} versions={has_versions} "
                    f"git_commit={has_commit} deterministic={deterministic}"
                ),
                "reasons": reasons,
                "rerun": rerun_detail,
            },
        )

    def _determinism(self, artifact: Any, ctx: VerifyContext) -> tuple[bool, str]:
        command = list(getattr(artifact, "command", []) or [])
        metrics = getattr(artifact, "metrics", {}) or {}
        if not command or not metrics:
            return False, "no command or metrics to re-run"

        repro_out = (getattr(artifact, "out_dir", "out") or "out") + "_repro"
        res = run(command + ["--out", repro_out], cwd=ctx.workdir, timeout=600)
        if res.returncode != 0:
            return False, f"re-run exit {res.returncode}: {res.stderr[-200:]}"
        try:
            new = json.loads((Path(ctx.workdir) / repro_out / "metrics.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            return False, f"re-run metrics unreadable: {e}"

        for key, value in metrics.items():
            rerun_value = new.get(key)
            if not (is_finite(value) and is_finite(rerun_value)):
                return False, f"{key}: non-finite metric (orig={value}, re-run={rerun_value})"
            if abs(float(rerun_value) - float(value)) > 1e-6:
                return False, f"{key}: re-run {rerun_value} != {value}"
        return True, f"re-run matches within 1e-6 ({new})"
