from __future__ import annotations

import hashlib
import json

from tiny_experiment import run_experiment


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_reproduces_raw_values_and_manifest(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_a = run_experiment(first, seed=7, count=12)
    manifest_b = run_experiment(second, seed=7, count=12)

    assert (first / "values.json").read_bytes() == (second / "values.json").read_bytes()
    assert manifest_a == manifest_b


def test_manifest_binds_inputs_and_raw_output(tmp_path):
    output = tmp_path / "run"
    manifest = run_experiment(output, seed=11, count=5)
    parameters = json.dumps(
        {"algorithm": "python-random-v1", "count": 5, "seed": 11},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert manifest["input_sha256"] == hashlib.sha256(parameters).hexdigest()
    assert manifest["output_sha256"] == _sha256(output / "values.json")
    assert manifest["count"] == 5

