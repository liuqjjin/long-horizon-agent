"""Experimenter: run an experiment in the sandbox and emit an ExperimentResult.

Deterministic by design — it just executes the experiment command and collects
its outputs. The objective judgement (do PSNR/SSIM meet thresholds? is it
reproducible?) is the verifiers' job, not the runner's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..artifacts import ExperimentResult, Step
from ..live_context.models import ContextBundle
from ..tools.shell import run


class Experimenter:
    def run(self, step: Step, bundle: ContextBundle, workdir: str | Path) -> ExperimentResult:
        workdir = Path(workdir)
        out_dir = step.params.get("out_dir", "out")
        cmd = build_cmd(step)
        res = run(cmd, cwd=workdir, timeout=float(step.params.get("timeout", 600)))

        out_path = workdir / out_dir
        metrics_raw = _read_json(out_path / "metrics.json")
        repro = _read_json(out_path / "repro.json")
        ref = f"{out_dir}/reference.npy" if (out_path / "reference.npy").exists() else None
        pred = f"{out_dir}/prediction.npy" if (out_path / "prediction.npy").exists() else None

        return ExperimentResult(
            step_id=step.step_id,
            out_dir=out_dir,
            command=cmd,
            metrics={
                k: float(v) for k, v in (metrics_raw or {}).items() if isinstance(v, (int, float))
            },
            reference_path=ref,
            prediction_path=pred,
            repro=repro or {},
            returncode=res.returncode,
            stdout_tail=(res.stdout or res.stderr)[-1000:],
            based_on_context=bundle.locators(),
        )


def build_cmd(step: Step) -> list[str]:
    """Resolve the experiment command, pinning python to the current interpreter."""
    script = step.params.get("experiment_script")
    if script:
        args = step.params.get("experiment_args", [])
        return [sys.executable, str(script), *[str(a) for a in args]]
    cmd = [str(c) for c in step.params.get("experiment_cmd", [])]
    if cmd and cmd[0] in ("python", "python3"):
        cmd[0] = sys.executable
    return cmd


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
