#!/usr/bin/env python
"""Deterministic bicubic 4x super-resolution baseline experiment.

Downscales then bicubic-upscales an image and reports PSNR/SSIM against the
original (data_range=1.0, the convention from the SRGAN paper note). Writes
``reference.npy``, ``prediction.npy``, ``metrics.json`` and ``repro.json`` into
the --out directory so a verifier can INDEPENDENTLY recompute the metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess

import numpy as np
import skimage
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def git_commit() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    data_range = 1.0

    ref = (data.astronaut() / 255.0).astype(np.float32)  # (512, 512, 3) in [0, 1]
    h, w = ref.shape[:2]
    low = resize(ref, (h // args.scale, w // args.scale), anti_aliasing=True, order=1)
    pred = np.clip(resize(low, (h, w), order=3), 0.0, 1.0).astype(np.float32)  # bicubic up

    p = float(psnr(ref, pred, data_range=data_range))
    s = float(ssim(ref, pred, data_range=data_range, channel_axis=-1))

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "reference.npy"), ref)
    np.save(os.path.join(args.out, "prediction.npy"), pred)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"psnr": p, "ssim": s}, f, indent=2)
    with open(os.path.join(args.out, "repro.json"), "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "scale": args.scale,
                "data_range": data_range,
                "channel_axis": -1,
                # digest of the exact input array, so a re-run on different data
                # cannot masquerade as a reproduction
                "input_sha256": hashlib.sha256(ref.tobytes()).hexdigest(),
                "git_commit": git_commit(),
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scikit-image": skimage.__version__,
                },
            },
            f,
            indent=2,
        )
    print(f"PSNR={p:.3f} dB  SSIM={s:.4f}")


if __name__ == "__main__":
    main()
