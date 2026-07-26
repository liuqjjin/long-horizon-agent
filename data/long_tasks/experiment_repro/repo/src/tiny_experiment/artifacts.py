"""Experiment artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import mean


def write_artifacts(
    output_dir: str | Path,
    values: list[float],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "values.json").write_text(json.dumps(values))
    manifest = {
        "seed": parameters["seed"],
        "count": parameters["count"],
        "mean": mean(values),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest

