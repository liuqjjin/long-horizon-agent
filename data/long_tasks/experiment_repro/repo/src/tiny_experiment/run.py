"""Deterministic sampling experiment."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .artifacts import write_artifacts


def run_experiment(output_dir: str | Path, seed: int, count: int) -> dict[str, Any]:
    generator = random.Random()
    values = [generator.random() for _ in range(count)]
    return write_artifacts(output_dir, values, {"seed": seed, "count": count})

