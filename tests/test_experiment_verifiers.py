"""Experiment verifier family: PSNR/SSIM recomputation + reproducibility."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
    return Step(step_id="s", kind="experiment", action="run_experiment", goal="g",
                verifiers=[verifier], params=params)


def test_psnr_passes_above_threshold(tmp_path):
    rng = np.random.default_rng(0)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    check = PSNRVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 30.0})))
    assert check.passed and check.score > 30


def test_psnr_fails_below_threshold(tmp_path):
    rng = np.random.default_rng(1)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = np.clip(ref + 0.3, 0, 1).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    check = PSNRVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 40.0})))
    assert not check.passed


def test_psnr_catches_fabricated_metric(tmp_path):
    rng = np.random.default_rng(2)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = np.clip(ref + 0.3, 0, 1).astype(np.float32)  # really low PSNR
    art = _write_pair(tmp_path, ref, pred)
    art.metrics = {"psnr": 99.0}  # a lie
    check = PSNRVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 5.0})))
    assert not check.passed  # recomputed value is inconsistent with the claim


def test_ssim_passes_above_threshold(tmp_path):
    rng = np.random.default_rng(3)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    check = SSIMVerifier().verify(
        art, VerifyContext(workdir=tmp_path, step=_step("ssim", {"ssim_min": 0.9, "channel_axis": -1}))
    )
    assert check.passed and check.score > 0.9


_DUMMY = (
    "import argparse, json, os\n"
    "ap = argparse.ArgumentParser(); ap.add_argument('--out', default='out')\n"
    "a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)\n"
    "json.dump({'psnr': 30.0, 'ssim': 0.9}, open(os.path.join(a.out, 'metrics.json'), 'w'))\n"
)


def test_repro_passes_when_deterministic_and_recorded(tmp_path):
    script = tmp_path / "exp.py"
    script.write_text(_DUMMY)
    art = ExperimentResult(
        step_id="s",
        out_dir="out",
        command=[sys.executable, str(script), "--out", "out"],
        metrics={"psnr": 30.0, "ssim": 0.9},
        repro={"seed": 1, "versions": {"numpy": "x"}, "git_commit": "abc"},
    )
    check = ReproVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {})))
    assert check.passed


def test_repro_fails_without_seed(tmp_path):
    script = tmp_path / "exp.py"
    script.write_text(_DUMMY)
    art = ExperimentResult(
        step_id="s",
        out_dir="out",
        command=[sys.executable, str(script), "--out", "out"],
        metrics={"psnr": 30.0},
        repro={"versions": {"numpy": "x"}},  # no seed
    )
    check = ReproVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {})))
    assert not check.passed


# --- review regressions: non-finite metrics + failed experiments -----------
def test_psnr_fails_when_experiment_failed(tmp_path):
    rng = np.random.default_rng(4)
    ref = rng.random((16, 16, 3)).astype(np.float32)
    art = _write_pair(tmp_path, ref, ref.copy())  # identical -> PSNR inf, but...
    art.returncode = 1  # ...the experiment failed, so it must not pass
    check = PSNRVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 1.0})))
    assert not check.passed


def test_psnr_fails_on_nonfinite_recomputed(tmp_path):
    ref = np.full((16, 16, 3), np.nan, dtype=np.float32)
    pred = np.zeros((16, 16, 3), dtype=np.float32)
    art = _write_pair(tmp_path, ref, pred)
    check = PSNRVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 1.0})))
    assert not check.passed
    assert check.score is None


def test_psnr_fails_on_nonfinite_reported(tmp_path):
    rng = np.random.default_rng(5)
    ref = rng.random((16, 16, 3)).astype(np.float32)
    pred = (ref + rng.normal(0, 0.001, ref.shape)).astype(np.float32)
    art = _write_pair(tmp_path, ref, pred)
    art.metrics = {"psnr": float("nan")}  # self-reported garbage
    check = PSNRVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("psnr", {"psnr_min": 5.0})))
    assert not check.passed


def test_repro_fails_when_experiment_failed(tmp_path):
    script = tmp_path / "exp.py"
    script.write_text(_DUMMY)
    art = ExperimentResult(
        step_id="s",
        out_dir="out",
        command=[sys.executable, str(script), "--out", "out"],
        metrics={"psnr": 30.0},
        repro={"seed": 1, "versions": {"numpy": "x"}, "git_commit": "abc"},
        returncode=1,
    )
    check = ReproVerifier().verify(art, VerifyContext(workdir=tmp_path, step=_step("reproducibility", {})))
    assert not check.passed
