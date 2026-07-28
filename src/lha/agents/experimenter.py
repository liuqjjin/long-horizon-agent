"""Experimenter: run an experiment in the sandbox and emit an ExperimentResult.

Deterministic by design — it just executes the experiment command and collects
its outputs. The objective judgement (do PSNR/SSIM meet thresholds? is it
reproducible?) is the verifiers' job, not the runner's.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..array_evidence import (
    array_summary,
    load_bounded_npy,
    output_sha256,
    raw_array_sha256,
    safe_artifact_path,
)
from ..artifacts import ExperimentResult, Step
from ..durable_io import (
    atomic_replace_bytes,
    atomic_replace_text,
    durable_mkdir_chain,
    fsync_directory,
)
from ..live_context.models import ContextBundle
from ..sandbox import ExecutionBackend, TrustedLocalBackend

_OUTPUT_OWNER = ".lha-experiment-output"
_LEGACY_OUTPUT_FILES = {
    "metrics.json",
    "prediction.npy",
    "reference.npy",
    "repro.json",
}

class ExperimentIntent(BaseModel):
    """Durable declaration written before an experiment command can run."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    step_id: str
    attempt_id: str
    command: tuple[str, ...]
    params_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperimentEvidence(BaseModel):
    """A completed experiment bound to its prepared attempt."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    intent: ExperimentIntent
    result: ExperimentResult

    @model_validator(mode="after")
    def _identity_matches(self) -> Self:
        if self.result.step_id != self.intent.step_id:
            raise ValueError("experiment result does not match its intent")
        if tuple(self.result.command) != self.intent.command:
            raise ValueError("experiment command does not match its intent")
        return self


class ExperimentAmbiguous(RuntimeError):
    """The command may have run, but no durable completed result exists."""


class Experimenter:
    def __init__(self, exec_backend: ExecutionBackend | None = None):
        self.exec = exec_backend or TrustedLocalBackend()

    def run(self, step: Step, bundle: ContextBundle, workdir: str | Path) -> ExperimentResult:
        workdir = Path(workdir).resolve()
        out_dir = str(step.params.get("out_dir", "out"))
        out_path = _prepare_output_dir(workdir, out_dir)
        cmd = build_cmd(step)
        if not cmd:
            return ExperimentResult(
                step_id=step.step_id,
                out_dir=out_dir,
                command=[],
                returncode=127,
                stdout_tail="experiment command is empty",
                based_on_context=bundle.locators(),
            )
        res = self.exec.run(cmd, cwd=workdir, timeout=float(step.params.get("timeout", 600)))
        if res.cleanup_unconfirmed:
            # Do not inspect output paths while a surviving process may still
            # be replacing them. The verifier will quarantine this attempt.
            return ExperimentResult(
                step_id=step.step_id,
                out_dir=out_dir,
                command=cmd,
                returncode=res.returncode,
                stdout_tail=(res.cleanup_detail or res.stderr or res.stdout)[
                    -1000:
                ],
                output_truncated=res.output_truncated,
                cleanup_unconfirmed=True,
                cleanup_detail=res.cleanup_detail[-1000:],
                based_on_context=bundle.locators(),
            )

        # Re-resolve after target code exits. A command that replaced the output
        # directory or one of its files with a symlink must not supply evidence.
        checked_out_path = safe_artifact_path(workdir, out_dir)
        if checked_out_path != out_path or not checked_out_path.is_dir():
            raise ValueError("experiment output directory was removed or replaced")
        metrics_path = _regular_artifact_file(workdir, out_dir, "metrics.json")
        repro_path = _regular_artifact_file(workdir, out_dir, "repro.json")
        reference_path = _regular_artifact_file(workdir, out_dir, "reference.npy")
        prediction_path = _regular_artifact_file(workdir, out_dir, "prediction.npy")
        metrics_raw = _read_json(metrics_path)
        repro = _read_json(repro_path)
        ref = (
            (Path(out_dir) / "reference.npy").as_posix()
            if reference_path is not None
            else None
        )
        pred = (
            (Path(out_dir) / "prediction.npy").as_posix()
            if prediction_path is not None
            else None
        )
        collected = _collect_array_evidence(reference_path, prediction_path)
        if collected is not None:
            repro = dict(repro or {})
            repro["collected"] = collected

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
            output_truncated=res.output_truncated,
            based_on_context=bundle.locators(),
        )


def execute_experiment_once(
    *,
    step: Step,
    bundle: ContextBundle,
    workdir: str | Path,
    run_dir: str | Path,
    attempt_id: str,
    backend: ExecutionBackend,
) -> ExperimentResult:
    """Execute at most once, reusing only a checksummed completed result."""
    root = Path(workdir).resolve(strict=True)
    run_root = Path(run_dir).resolve(strict=True)
    attempt_dir = (
        run_root
        / "steps"
        / _safe_segment(step.step_id)
        / "attempts"
        / _safe_segment(attempt_id)
    )
    if attempt_dir.is_symlink() or (
        attempt_dir.exists() and not attempt_dir.is_dir()
    ):
        raise ExperimentAmbiguous(
            f"experiment attempt path is unsafe: {attempt_dir}"
        )
    durable_mkdir_chain(attempt_dir, anchor=run_root)
    intent_path = attempt_dir / "experiment_intent.json"
    evidence_path = attempt_dir / "experiment_evidence.json"
    params = json.dumps(
        step.params, sort_keys=True, separators=(",", ":")
    ).encode()
    context = _semantic_context(bundle)
    intent = ExperimentIntent(
        step_id=step.step_id,
        attempt_id=attempt_id,
        command=tuple(build_cmd(step)),
        params_sha256=hashlib.sha256(params).hexdigest(),
        context_sha256=hashlib.sha256(context).hexdigest(),
    )

    if evidence_path.exists() or evidence_path.is_symlink():
        try:
            persisted_intent = _read_checksummed_model(
                intent_path, ExperimentIntent
            )
            evidence = _read_checksummed_model(
                evidence_path, ExperimentEvidence
            )
        except Exception as error:
            raise ExperimentAmbiguous(
                "persisted experiment evidence is invalid for "
                f"{step.step_id}/{attempt_id}: {error}"
            ) from error
        if persisted_intent != intent or evidence.intent != persisted_intent:
            raise ExperimentAmbiguous(
                f"experiment intent changed for {step.step_id}/{attempt_id}"
            )
        validate_experiment_result(root, evidence.result)
        return evidence.result

    if intent_path.exists() or intent_path.is_symlink():
        try:
            persisted_intent = _read_checksummed_model(
                intent_path, ExperimentIntent
            )
        except Exception as error:
            raise ExperimentAmbiguous(
                "prepared experiment intent is invalid for "
                f"{step.step_id}/{attempt_id}: {error}"
            ) from error
        if persisted_intent != intent:
            raise ExperimentAmbiguous(
                f"prepared experiment intent changed for {step.step_id}/{attempt_id}"
            )
        raise ExperimentAmbiguous(
            f"experiment may already have executed for {step.step_id}/{attempt_id}; "
            "refusing to duplicate its side effects"
        )

    _write_checksummed_model(intent_path, intent)
    result = Experimenter(backend).run(step, bundle, root)
    validate_experiment_result(root, result)
    _write_checksummed_model(
        evidence_path,
        ExperimentEvidence(intent=intent, result=result),
    )
    return result


def validate_experiment_result(
    workdir: str | Path,
    result: ExperimentResult,
) -> None:
    """Recompute the array evidence named by an ExperimentResult."""
    root = Path(workdir).resolve(strict=True)
    if result.returncode != 0:
        return
    reference = (
        safe_artifact_path(root, result.reference_path)
        if result.reference_path
        else None
    )
    prediction = (
        safe_artifact_path(root, result.prediction_path)
        if result.prediction_path
        else None
    )
    recorded = result.repro.get("collected")
    if (
        reference is None
        and prediction is None
        and recorded is None
    ):
        return
    actual = _collect_array_evidence(reference, prediction)
    if actual is None or not isinstance(recorded, dict) or actual != recorded:
        raise ValueError(
            f"experiment arrays do not match persisted evidence for {result.step_id}"
        )
    input_sha256 = result.repro.get("input_sha256")
    if (
        not isinstance(input_sha256, str)
        or input_sha256.lower() != actual["input_sha256"]
    ):
        raise ValueError(
            f"experiment input digest does not match for {result.step_id}"
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


def _prepare_output_dir(workdir: Path, out_dir: str) -> Path:
    """Create an empty output directory for exactly one invocation."""
    out_path = safe_artifact_path(workdir, out_dir)
    if out_path.exists():
        if not out_path.is_dir():
            raise ValueError(f"experiment output path is not a directory: {out_dir}")
        if not _owned_or_artifact_only(out_path):
            raise ValueError(
                f"refusing to clear an output directory not owned by LHA: {out_dir}"
            )
        # ``safe_artifact_path`` rejected a symlink at every path component,
        # and the content check ruled out a mistyped source directory.
        shutil.rmtree(out_path)
        fsync_directory(out_path.parent)
    durable_mkdir_chain(out_path, anchor=workdir)
    atomic_replace_text(
        out_path / _OUTPUT_OWNER,
        "LHA experiment output\n",
        anchor=workdir,
    )
    checked = safe_artifact_path(workdir, out_dir)
    if checked != out_path or not checked.is_dir():
        raise ValueError(f"experiment output directory could not be prepared: {out_dir}")
    return out_path


def _owned_or_artifact_only(out_path: Path) -> bool:
    """Protect a mistyped source directory while still clearing legacy outputs."""
    marker = out_path / _OUTPUT_OWNER
    try:
        if stat.S_ISREG(marker.lstat().st_mode):
            return marker.read_text() == "LHA experiment output\n"
    except OSError:
        pass
    try:
        entries = list(out_path.iterdir())
    except OSError:
        return False
    return all(
        entry.name in _LEGACY_OUTPUT_FILES
        and not entry.is_symlink()
        and entry.is_file()
        for entry in entries
    )


def _regular_artifact_file(workdir: Path, out_dir: str, name: str) -> Path | None:
    """Return one non-symlink regular file below the invocation output."""
    try:
        path = safe_artifact_path(workdir, Path(out_dir) / name)
        mode = path.lstat().st_mode
    except (OSError, ValueError):
        return None
    return path if stat.S_ISREG(mode) else None


def _read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _collect_array_evidence(
    reference_path: Path | None, prediction_path: Path | None
) -> dict | None:
    """Collect structural evidence without trusting metrics.json."""
    import numpy as np

    if reference_path is None or prediction_path is None:
        return None
    try:
        reference = load_bounded_npy(reference_path)
        prediction = load_bounded_npy(prediction_path)
    except (MemoryError, OSError, OverflowError, ValueError):
        return None
    if (
        reference.size == 0
        or prediction.size == 0
        or reference.shape != prediction.shape
        or not np.issubdtype(reference.dtype, np.number)
        or not np.issubdtype(prediction.dtype, np.number)
        or not np.isfinite(reference).all()
        or not np.isfinite(prediction).all()
    ):
        return None
    return {
        "input_sha256": raw_array_sha256(reference),
        "output_sha256": output_sha256(reference, prediction),
        "reference": array_summary(reference),
        "prediction": array_summary(prediction),
    }


def _semantic_context(bundle: ContextBundle) -> bytes:
    value = bundle.model_dump(mode="json")
    freshness = dict(value.get("freshness") or {})
    freshness.pop("indexed_at", None)
    freshness.pop("source_mtime_max", None)
    value["freshness"] = freshness
    for item in value.get("items", []):
        provenance = item.get("provenance")
        if isinstance(provenance, dict):
            provenance.pop("indexed_at", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _safe_segment(value: str) -> str:
    import re

    segment = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip(".")
    return segment or "item"


def _canonical_payload(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _write_checksummed_model(path: Path, model: BaseModel) -> None:
    payload = model.model_dump(mode="json")
    envelope = {
        "schema_version": 1,
        "sha256": hashlib.sha256(_canonical_payload(payload)).hexdigest(),
        "payload": payload,
    }
    _durable_replace(path, json.dumps(envelope, sort_keys=True).encode())


def _read_checksummed_model(path: Path, model_type):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact path is missing or unsafe: {path}")
    raw = json.loads(path.read_text())
    payload = raw["payload"]
    if (
        raw.get("schema_version") != 1
        or raw.get("sha256")
        != hashlib.sha256(_canonical_payload(payload)).hexdigest()
    ):
        raise ValueError("artifact checksum mismatch")
    return model_type.model_validate(payload)


def _durable_replace(path: Path, data: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"artifact path is unsafe: {path}")
    atomic_replace_bytes(path, data)
