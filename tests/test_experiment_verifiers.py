"""Experiment verifier family: PSNR/SSIM recomputation + reproducibility."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from lha.array_evidence import array_summary, output_sha256, raw_array_sha256
from lha.artifacts import ExperimentResult, Step
from lha.verifiers import VerifyContext
from lha.verifiers.experiment import PSNRVerifier, ReproVerifier, SSIMVerifier


def _write_pair(workdir: Path, ref, pred) -> ExperimentResult:
    out = workdir / "out"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "reference.npy", ref)
    np.save(out / "prediction.npy", pred)
    return ExperimentResult(
        step_id="s",
        out_dir="out",
        reference_path="out/reference.npy",
        prediction_path="out/prediction.npy",
    )


def _step(verifier: str, params: dict) -> Step:
    return Step(
        step_id="s",
        kind="experiment",
        action="run_experiment",
        goal="g",
        verifiers=[verifier],
        params=params,
    )


def test_psnr_passes_above_threshold(tmp_path):
    rng = np.random.default_rng(0)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    check = PSNRVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 30.0}))
    )
    assert check.passed and check.score > 30


def test_psnr_fails_below_threshold(tmp_path):
    rng = np.random.default_rng(1)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = np.clip(ref + 0.3, 0, 1).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    check = PSNRVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 40.0}))
    )
    assert not check.passed


def test_psnr_catches_fabricated_metric(tmp_path):
    rng = np.random.default_rng(2)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = np.clip(ref + 0.3, 0, 1).astype(np.float32)  # really low PSNR
    art = _write_pair(tmp_path, ref, pred)
    art.metrics = {"psnr": 99.0}  # a lie
    check = PSNRVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 5.0}))
    )
    assert not check.passed  # recomputed value is inconsistent with the claim


def test_ssim_passes_above_threshold(tmp_path):
    rng = np.random.default_rng(3)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    check = SSIMVerifier().verify(
        art,
        VerifyContext(workdir=tmp_path, step=_step("ssim", {"ssim_min": 0.9, "channel_axis": -1})),
    )
    assert check.passed and check.score > 0.9


def _repro_arrays(delta: float = 0.01):
    ref = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    return ref, np.clip(ref + delta, 0.0, 1.0).astype(np.float32)


def _script(*, delta: float = 0.01, write_repro: bool = True) -> str:
    repro = (
        "json.dump({'seed': 1, 'versions': {'numpy': np.__version__}, "
        "'input_sha256': hashlib.sha256(ref.tobytes()).hexdigest(), "
        "'data_range': 1.0, 'channel_axis': None}, "
        "open(os.path.join(a.out, 'repro.json'), 'w'))\n"
        if write_repro
        else ""
    )
    return (
        "import argparse, hashlib, json, os\n"
        "import numpy as np\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--out', default='out')\n"
        "a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)\n"
        "ref = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)\n"
        f"pred = np.clip(ref + {delta!r}, 0.0, 1.0).astype(np.float32)\n"
        "np.save(os.path.join(a.out, 'reference.npy'), ref)\n"
        "np.save(os.path.join(a.out, 'prediction.npy'), pred)\n"
        # Deliberately fabricated: ReproVerifier must ignore metrics.json and
        # recompute both metrics from the arrays.
        "json.dump({'psnr': -999.0, 'ssim': -999.0}, "
        "open(os.path.join(a.out, 'metrics.json'), 'w'))\n"
        + repro
    )


def _repro_artifact(tmp_path: Path, *, script_body: str | None = None) -> ExperimentResult:
    script = tmp_path / "exp.py"
    script.write_text(script_body if script_body is not None else _script())
    ref, pred = _repro_arrays()
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    np.save(out / "reference.npy", ref)
    np.save(out / "prediction.npy", pred)
    return ExperimentResult(
        step_id="s",
        out_dir="out",
        command=[sys.executable, str(script)],
        reference_path="out/reference.npy",
        prediction_path="out/prediction.npy",
        repro={
            "seed": 1,
            "versions": {"numpy": np.__version__},
            "git_commit": "abc",
            "input_sha256": hashlib.sha256(ref.tobytes()).hexdigest(),
            "data_range": 1.0,
            "channel_axis": None,
            "collected": {
                "input_sha256": raw_array_sha256(ref),
                "output_sha256": output_sha256(ref, pred),
                "reference": array_summary(ref),
                "prediction": array_summary(pred),
            },
        },
    )


def test_repro_passes_when_deterministic_and_recorded(tmp_path):
    art = _repro_artifact(tmp_path)
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert check.passed
    assert check.detail["original_output_sha256"] == check.detail["rerun_output_sha256"]
    assert check.detail["original_metrics_recomputed"] == check.detail["rerun_metrics_recomputed"]


def test_repro_fails_without_input_digest(tmp_path):
    art = _repro_artifact(tmp_path)
    art.repro.pop("input_sha256")
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert not check.passed
    assert any("input" in r for r in check.detail["reasons"])


def test_repro_fails_without_seed(tmp_path):
    art = _repro_artifact(tmp_path)
    art.repro.pop("seed")
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert not check.passed


def test_repro_fails_when_recorded_array_summary_is_tampered(tmp_path):
    art = _repro_artifact(tmp_path)
    art.repro["collected"]["prediction"]["shape"] = [1]
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert not check.passed
    assert "summaries" in check.detail["rerun"]


def test_repro_fails_when_rerun_omits_input_evidence(tmp_path):
    art = _repro_artifact(tmp_path, script_body=_script(write_repro=False))
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert not check.passed
    assert "evidence unreadable" in check.detail["rerun"]


def test_repro_does_not_reuse_a_stale_output_directory(tmp_path):
    old = tmp_path / "out_repro"
    old.mkdir()
    (old / "metrics.json").write_text('{"psnr": 1, "ssim": 1}')
    (old / "repro.json").write_text('{"input_sha256": "stale"}')
    art = _repro_artifact(tmp_path, script_body="pass\n")
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert not check.passed
    assert check.detail["rerun_dir"] != "out_repro"


def test_repro_fails_on_same_reported_metrics_but_different_arrays(tmp_path):
    art = _repro_artifact(tmp_path, script_body=_script(delta=0.02))
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert not check.passed
    assert "output" in check.detail["rerun"] or "psnr" in check.detail["rerun"]


def test_repro_rejects_unsafe_original_artifact_path(tmp_path):
    art = _repro_artifact(tmp_path)
    art.reference_path = "../reference.npy"
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert not check.passed
    assert "unsafe" in check.detail["rerun"]


# --- the verifier scores with the params the experiment recorded -----------
def test_psnr_fails_on_data_range_mismatch(tmp_path):
    rng = np.random.default_rng(6)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    art.repro = {"data_range": 255.0}  # experiment used 255; task pins 1.0 -> meaningless
    check = PSNRVerifier().verify(
        art,
        VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 1.0, "data_range": 1.0})),
    )
    assert not check.passed
    assert any("data_range mismatch" in r for r in check.detail["reasons"])


def test_ssim_uses_recorded_channel_axis_without_override(tmp_path):
    rng = np.random.default_rng(7)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    art.repro = {"data_range": 1.0, "channel_axis": -1}  # recorded; task gives no override
    check = SSIMVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("ssim", {"ssim_min": 0.9}))
    )
    assert check.passed and check.score > 0.9


# --- review regressions: non-finite metrics + failed experiments -----------
def test_psnr_fails_when_experiment_failed(tmp_path):
    rng = np.random.default_rng(4)
    ref = rng.random((16, 16, 3)).astype(np.float32)
    art = _write_pair(tmp_path, ref, ref.copy())  # identical -> PSNR inf, but...
    art.returncode = 1  # ...the experiment failed, so it must not pass
    check = PSNRVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 1.0}))
    )
    assert not check.passed


def test_psnr_fails_on_nonfinite_recomputed(tmp_path):
    ref = np.full((16, 16, 3), np.nan, dtype=np.float32)
    pred = np.zeros((16, 16, 3), dtype=np.float32)
    art = _write_pair(tmp_path, ref, pred)
    check = PSNRVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 1.0}))
    )
    assert not check.passed
    assert check.score is None


def test_psnr_fails_on_nonfinite_threshold(tmp_path):
    rng = np.random.default_rng(8)
    ref = rng.random((16, 16, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    step = _step("psnr", {"psnr_min": 30.0})
    # Simulate a corrupted/tampered in-memory model so the verifier remains
    # fail-closed even if boundary validation was bypassed.
    step.params["psnr_min"] = float("nan")
    check = PSNRVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=step)
    )
    assert not check.passed
    assert check.threshold is None


def test_psnr_fails_on_nonfinite_reported(tmp_path):
    rng = np.random.default_rng(5)
    ref = rng.random((16, 16, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    art.metrics = {"psnr": float("nan")}  # self-reported garbage
    check = PSNRVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 5.0}))
    )
    assert not check.passed


def test_repro_fails_when_experiment_failed(tmp_path):
    art = _repro_artifact(tmp_path)
    art.returncode = 1
    check = ReproVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {}))
    )
    assert not check.passed
