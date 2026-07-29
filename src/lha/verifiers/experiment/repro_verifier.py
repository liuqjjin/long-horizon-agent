"""Experiment verifier: independently reproduce arrays and metrics.

The experiment's metrics file is never evidence for determinism. A re-run writes
to a newly-created empty directory; the verifier checks the input digest, loads
both output arrays, recomputes PSNR/SSIM, and compares an output-array digest.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ...array_evidence import (
    array_summary,
    output_sha256,
    raw_array_sha256,
    safe_artifact_path,
)
from ..base import Verifier, VerifyContext
from ..verdict import Check, process_cleanup_failure_detail
from .common import (
    is_finite,
    load_arrays,
    recompute_image_metrics,
)

_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


class ReproVerifier(Verifier):
    name = "reproducibility"
    family = "experiment"

    def verify(self, artifact: Any, ctx: VerifyContext) -> Check:
        repro = getattr(artifact, "repro", {}) or {}
        has_seed = repro.get("seed") is not None
        has_versions = isinstance(repro.get("versions"), dict) and bool(repro["versions"])
        original_input = repro.get("input_sha256")
        has_input = isinstance(original_input, str) and _SHA256.fullmatch(original_input) is not None
        collected = repro.get("collected")
        has_array_evidence = isinstance(collected, dict) and all(
            key in collected
            for key in ("input_sha256", "output_sha256", "reference", "prediction")
        )
        has_commit = bool(repro.get("git_commit"))

        rc = getattr(artifact, "returncode", 0)
        cleanup_unconfirmed = bool(
            getattr(artifact, "cleanup_unconfirmed", False)
        )
        if cleanup_unconfirmed:
            detail = {
                "summary": "experiment process cleanup could not be confirmed",
            }
            detail.update(
                process_cleanup_failure_detail(
                    returncode=rc,
                    cleanup_unconfirmed=cleanup_unconfirmed,
                    detail=str(getattr(artifact, "stdout_tail", ""))[-500:],
                )
            )
            return Check(
                name=self.name,
                family=self.family,
                passed=False,
                detail=detail,
            )
        if rc == 0:
            deterministic, rerun_detail, evidence = self._determinism(artifact, ctx)
        else:
            deterministic = False
            rerun_detail = f"original experiment failed (returncode={rc}); re-run not attempted"
            evidence = {}

        reasons: list[str] = []
        if rc != 0:
            reasons.append(f"original experiment failed (returncode={rc})")
        if not has_seed:
            reasons.append("no seed recorded")
        if not has_versions:
            reasons.append("no library versions recorded")
        if not has_input:
            reasons.append("no valid SHA-256 input digest recorded")
        if not has_array_evidence:
            reasons.append("no collected input/output array evidence recorded")
        # A commit is only demandable when the experiment ran in a git checkout;
        # run sandboxes are copied without .git.
        if not has_commit and (Path(ctx.workdir) / ".git").exists():
            reasons.append("workdir is a git checkout but no commit was recorded")
        if not deterministic:
            reasons.append(f"not deterministic ({rerun_detail})")

        detail: dict[str, Any] = {
            "summary": (
                f"seed={has_seed} versions={has_versions} input={has_input} "
                f"arrays={has_array_evidence} git_commit={has_commit} "
                f"deterministic={deterministic}"
            ),
            "reasons": reasons,
            "rerun": rerun_detail,
        }
        detail.update(evidence)
        return Check(
            name=self.name,
            family=self.family,
            passed=not reasons,
            detail=detail,
        )

    def _determinism(
        self, artifact: Any, ctx: VerifyContext
    ) -> tuple[bool, str, dict[str, Any]]:
        command = list(getattr(artifact, "command", []) or [])
        if not command:
            return False, "no command to re-run", {}

        root = Path(ctx.workdir).resolve()
        try:
            safe_artifact_path(root, getattr(artifact, "out_dir", "out") or "out")
        except ValueError as e:
            return False, str(e), {}

        reference, prediction = load_arrays(artifact, root)
        if reference is None or prediction is None:
            return False, "original reference/prediction arrays are missing or unsafe", {}

        repro = getattr(artifact, "repro", {}) or {}
        original_input = repro.get("input_sha256")
        if not isinstance(original_input, str) or _SHA256.fullmatch(original_input) is None:
            return False, "original input_sha256 is missing or malformed", {}
        actual_input = raw_array_sha256(reference)
        if actual_input != original_input.lower():
            return False, "original input_sha256 does not match the reference array", {}
        collected = repro.get("collected")
        expected_collected = {
            "input_sha256": actual_input,
            "output_sha256": output_sha256(reference, prediction),
            "reference": array_summary(reference),
            "prediction": array_summary(prediction),
        }
        if not isinstance(collected, dict) or collected != expected_collected:
            return False, "collected array summaries do not match the original arrays", {
                "expected_collected": expected_collected,
                "recorded_collected": collected,
            }

        try:
            original_metrics = recompute_image_metrics(
                artifact, ctx.step, reference, prediction
            )
        except (TypeError, ValueError) as e:
            return False, f"original arrays could not be scored: {e}", {}

        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", ctx.step.step_id)[:32] or "step"
        try:
            rerun = _isolated_rerun(
                artifact=artifact,
                ctx=ctx,
                root=root,
                command=command,
                prefix=prefix,
            )
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return False, f"re-run reproducibility evidence unreadable: {e}", {
                "rerun_dir": ".lha-repro-output",
                "rerun_isolation": "ephemeral-worktree-copy",
            }
        rerun_rel = Path(rerun["rerun_dir"])
        res = rerun["process"]
        if res.returncode != 0:
            evidence = {
                "rerun_dir": rerun_rel.as_posix(),
                "rerun_isolation": "ephemeral-worktree-copy",
            }
            evidence.update(
                process_cleanup_failure_detail(
                    returncode=res.returncode,
                    cleanup_unconfirmed=res.cleanup_unconfirmed,
                    detail=res.cleanup_detail or res.stderr[-500:],
                )
            )
            return False, f"re-run exit {res.returncode}: {res.stderr[-200:]}", evidence
        rerun_repro = rerun["repro"]

        rerun_input = rerun_repro.get("input_sha256")
        if not isinstance(rerun_input, str) or _SHA256.fullmatch(rerun_input) is None:
            return False, "re-run input_sha256 is missing or malformed", {
                "rerun_dir": rerun_rel.as_posix()
            }
        if original_input.lower() != rerun_input.lower():
            return False, "re-run used a different input (input_sha256 mismatch)", {
                "rerun_dir": rerun_rel.as_posix()
            }
        for key in ("seed", "versions", "data_range", "channel_axis", "scale"):
            if key in repro and rerun_repro.get(key) != repro.get(key):
                return False, f"re-run changed recorded parameter {key}", {
                    "rerun_dir": rerun_rel.as_posix()
                }

        rerun_artifact = artifact.model_copy(update={"repro": rerun_repro})
        rerun_reference = rerun["reference"]
        rerun_prediction = rerun["prediction"]
        if rerun_reference is None or rerun_prediction is None:
            return False, "re-run reference/prediction arrays are missing or unsafe", {
                "rerun_dir": rerun_rel.as_posix()
            }
        if raw_array_sha256(rerun_reference) != rerun_input.lower():
            return False, "re-run input_sha256 does not match the reference array", {
                "rerun_dir": rerun_rel.as_posix()
            }

        try:
            rerun_metrics = recompute_image_metrics(
                rerun_artifact, ctx.step, rerun_reference, rerun_prediction
            )
        except (TypeError, ValueError) as e:
            return False, f"re-run arrays could not be scored: {e}", {
                "rerun_dir": rerun_rel.as_posix()
            }

        tolerance_v = ctx.step.params.get("repro_tolerance", 1e-6)
        if not is_finite(tolerance_v) or float(tolerance_v) < 0:
            return False, f"invalid reproduction tolerance: {tolerance_v!r}", {
                "rerun_dir": rerun_rel.as_posix()
            }
        tolerance = float(tolerance_v)
        for key, original_value in original_metrics.items():
            rerun_value = rerun_metrics[key]
            if abs(rerun_value - original_value) > tolerance:
                return False, f"{key}: re-run {rerun_value} != {original_value}", {
                    "rerun_dir": rerun_rel.as_posix()
                }

        original_output = output_sha256(reference, prediction)
        rerun_output = output_sha256(rerun_reference, rerun_prediction)
        evidence = {
            "rerun_dir": rerun_rel.as_posix(),
            "rerun_isolation": "ephemeral-worktree-copy",
            "input_sha256": original_input.lower(),
            "original_output_sha256": original_output,
            "rerun_output_sha256": rerun_output,
            "original_reference": array_summary(reference),
            "original_prediction": array_summary(prediction),
            "rerun_reference": array_summary(rerun_reference),
            "rerun_prediction": array_summary(rerun_prediction),
            "original_metrics_recomputed": original_metrics,
            "rerun_metrics_recomputed": rerun_metrics,
        }
        if original_output != rerun_output:
            return False, "re-run output arrays differ (output SHA-256 mismatch)", evidence
        return True, f"re-run arrays and recomputed metrics match within {tolerance:g}", evidence


def _isolated_rerun(
    *,
    artifact: Any,
    ctx: VerifyContext,
    root: Path,
    command: list[str],
    prefix: str,
) -> dict[str, Any]:
    """Run target code in a disposable copy so retries cannot mutate workdir."""
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                "worktree contains a symlink and cannot be copied safely: "
                f"{path.relative_to(root)}"
            )

    with tempfile.TemporaryDirectory(
        prefix=f".lha-repro-worktree-{prefix}-",
        dir=root.parent,
    ) as temporary:
        copy_root = Path(temporary) / "worktree"
        shutil.copytree(
            root,
            copy_root,
            symlinks=False,
            ignore=shutil.ignore_patterns(
                ".git",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                "*.pyc",
            ),
        )
        output_path = Path(
            tempfile.mkdtemp(prefix=".lha-repro-output-", dir=copy_root)
        )
        rerun_rel = output_path.relative_to(copy_root)
        remapped = [
            _remap_worktree_argument(value, source=root, target=copy_root)
            for value in command
        ]
        process = ctx.exec.run(
            remapped + ["--out", rerun_rel.as_posix()],
            cwd=copy_root,
            timeout=600,
        )
        if process.returncode != 0:
            return {
                "rerun_dir": rerun_rel.as_posix(),
                "process": process,
                "repro": {},
                "reference": None,
                "prediction": None,
            }
        rerun_repro_path = safe_artifact_path(
            copy_root, rerun_rel / "repro.json"
        )
        rerun_repro = json.loads(rerun_repro_path.read_text())
        if not isinstance(rerun_repro, dict):
            raise ValueError("re-run repro.json is not an object")
        rerun_artifact = artifact.model_copy(
            update={
                "reference_path": (rerun_rel / "reference.npy").as_posix(),
                "prediction_path": (rerun_rel / "prediction.npy").as_posix(),
                "repro": rerun_repro,
            }
        )
        reference, prediction = load_arrays(rerun_artifact, copy_root)
        return {
            "rerun_dir": rerun_rel.as_posix(),
            "process": process,
            "repro": rerun_repro,
            "reference": reference,
            "prediction": prediction,
        }


def _remap_worktree_argument(value: str, *, source: Path, target: Path) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        relative = candidate.relative_to(source)
    except ValueError:
        return value
    return str(target / relative)
