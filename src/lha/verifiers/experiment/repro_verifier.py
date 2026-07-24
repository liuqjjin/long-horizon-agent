"""Experiment verifier: reproducibility.

Passes only if (a) a seed and library versions were recorded, and (b) re-running
the experiment reproduces the same metrics within tolerance. The re-run writes to
a separate output dir so the original artifact stays intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        has_input = bool(repro.get("input_sha256") or repro.get("inputs"))

        deterministic, rerun_detail = self._determinism(artifact, ctx)

        reasons: list[str] = []
        rc = getattr(artifact, "returncode", 0)
        if rc != 0:
            reasons.append(f"original experiment failed (returncode={rc})")
        if not has_seed:
            reasons.append("no seed recorded")
        if not has_versions:
            reasons.append("no library versions recorded")
        if not has_input:
            reasons.append("no input digest recorded (input_sha256/inputs)")
        # A commit is only demandable when the experiment ran in a git checkout;
        # run sandboxes are copied without .git.
        if not has_commit and (Path(ctx.workdir) / ".git").exists():
            reasons.append("workdir is a git checkout but no commit was recorded")
        if not deterministic:
            reasons.append(f"not deterministic ({rerun_detail})")

        return Check(
            name=self.name,
            family=self.family,
            passed=not reasons,
            detail={
                "summary": (
                    f"seed={has_seed} versions={has_versions} input={has_input} "
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
        res = ctx.exec.run(command + ["--out", repro_out], cwd=ctx.workdir, timeout=600)
        if res.returncode != 0:
            return False, f"re-run exit {res.returncode}: {res.stderr[-200:]}"
        try:
            new = json.loads((Path(ctx.workdir) / repro_out / "metrics.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            return False, f"re-run metrics unreadable: {e}"

        # Matching metrics on DIFFERENT inputs is not a reproduction.
        orig_input = (getattr(artifact, "repro", {}) or {}).get("input_sha256")
        try:
            rerun_repro = json.loads((Path(ctx.workdir) / repro_out / "repro.json").read_text())
        except (OSError, json.JSONDecodeError):
            rerun_repro = {}
        rerun_input = rerun_repro.get("input_sha256")
        if orig_input and rerun_input and orig_input != rerun_input:
            return False, "re-run used a different input (input_sha256 mismatch)"

        for key, value in metrics.items():
            rerun_value = new.get(key)
            if not (is_finite(value) and is_finite(rerun_value)):
                return False, f"{key}: non-finite metric (orig={value}, re-run={rerun_value})"
            if abs(float(rerun_value) - float(value)) > 1e-6:
                return False, f"{key}: re-run {rerun_value} != {value}"
        return True, f"re-run matches within 1e-6 ({new})"
