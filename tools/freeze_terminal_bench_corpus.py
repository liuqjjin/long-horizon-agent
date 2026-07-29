#!/usr/bin/env python3
"""Freeze Harbor's resolved Terminal-Bench 2.1 metadata into package data.

The input JSON is the result of resolving the pinned dataset with Harbor 0.20.
The metadata directory is produced by:

    uvx --python 3.12 --from harbor==0.20.0 harbor download \
      terminal-bench/terminal-bench-2-1@sha256:7d7bdc1c... \
      --output-dir <DIR> --overwrite --export

This command does not contact a registry. It joins the already resolved remote
digests with task.toml timeouts, validates all 89 tasks, and writes canonical
JSON. Use ``--check`` in CI to compare existing files without changing them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DATASET = "terminal-bench/terminal-bench-2-1"
DATASET_VERSION = (
    "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
)
HARBOR_VERSION = "0.20.0"
TASK_COUNT = 89
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
SHA256_VALUE = re.compile(r"^sha256:[0-9a-f]{64}$")
RAW_KEYS = {
    "dataset",
    "dataset_version",
    "harbor_version",
    "resolution_failures",
    "tasks",
}
RAW_TASK_KEYS = {
    "image",
    "image_manifest_media_type",
    "task_checksum",
    "task_content_digest",
    "task_image_digest",
}


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON input is not an object: {path}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _positive_seconds(value: Any, *, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        or int(value) != value
    ):
        raise ValueError(f"{label} must be a positive whole number of seconds")
    return int(value)


def _task_metadata(metadata_root: Path, instance_id: str) -> tuple[dict[str, Any], bytes]:
    short_name = instance_id.removeprefix("terminal-bench/")
    if not short_name or "/" in short_name:
        raise ValueError(f"invalid Terminal-Bench task ID: {instance_id}")
    task_path = metadata_root / short_name / "task.toml"
    try:
        payload = task_path.read_bytes()
        metadata = tomllib.loads(payload.decode())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read official metadata: {task_path}") from exc
    return metadata, payload


def build_snapshot(
    resolved_input: Path,
    metadata_root: Path,
) -> dict[str, Any]:
    """Validate and combine one complete Harbor resolution."""
    raw = _read_json_object(resolved_input)
    if set(raw) != RAW_KEYS:
        raise ValueError("resolved input has unexpected top-level fields")
    if (
        raw["dataset"] != DATASET
        or raw["dataset_version"] != DATASET_VERSION
        or raw["harbor_version"] != HARBOR_VERSION
        or raw["resolution_failures"] != []
    ):
        raise ValueError("resolved input is not the pinned successful Harbor resolution")
    raw_tasks = raw["tasks"]
    if not isinstance(raw_tasks, dict) or len(raw_tasks) != TASK_COUNT:
        raise ValueError(f"resolved input must contain exactly {TASK_COUNT} tasks")

    tasks: dict[str, dict[str, Any]] = {}
    for instance_id in sorted(raw_tasks):
        remote = raw_tasks[instance_id]
        if not isinstance(remote, dict) or set(remote) != RAW_TASK_KEYS:
            raise ValueError(f"resolved task fields changed: {instance_id}")
        if (
            not isinstance(remote["image"], str)
            or not remote["image"]
            or any(character.isspace() for character in remote["image"])
            or SHA256_HEX.fullmatch(str(remote["task_checksum"])) is None
            or SHA256_VALUE.fullmatch(str(remote["task_content_digest"])) is None
            or SHA256_VALUE.fullmatch(str(remote["task_image_digest"])) is None
        ):
            raise ValueError(f"resolved task digests are invalid: {instance_id}")

        metadata, task_toml = _task_metadata(metadata_root, instance_id)
        task_section = metadata.get("task")
        agent = metadata.get("agent")
        verifier = metadata.get("verifier")
        environment = metadata.get("environment")
        if not all(
            isinstance(section, dict)
            for section in (task_section, agent, verifier, environment)
        ):
            raise ValueError(f"task.toml sections are incomplete: {instance_id}")
        assert isinstance(task_section, dict)
        assert isinstance(agent, dict)
        assert isinstance(verifier, dict)
        assert isinstance(environment, dict)
        if task_section.get("name") != instance_id:
            raise ValueError(f"task.toml name changed: {instance_id}")
        if environment.get("docker_image") != remote["image"]:
            raise ValueError(f"task image differs between Harbor records: {instance_id}")

        tasks[instance_id] = {
            **remote,
            "task_toml_sha256": hashlib.sha256(task_toml).hexdigest(),
            "environment_build_timeout_s": _positive_seconds(
                environment.get("build_timeout_sec"),
                label=f"{instance_id} environment timeout",
            ),
            "agent_timeout_s": _positive_seconds(
                agent.get("timeout_sec"),
                label=f"{instance_id} agent timeout",
            ),
            "verifier_timeout_s": _positive_seconds(
                verifier.get("timeout_sec"),
                label=f"{instance_id} verifier timeout",
            ),
        }
    return {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "harbor_version": HARBOR_VERSION,
        "resolution_failures": [],
        "tasks": tasks,
    }


def build_resource(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Bind package data to the exact public resolution snapshot."""
    payload = _canonical_json(snapshot)
    return {
        **snapshot,
        "source_inputs_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_or_check(path: Path, payload: bytes, *, check: bool) -> None:
    if check:
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"expected generated file is missing: {path}") from exc
        if current != payload:
            raise ValueError(f"generated file is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-input", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--snapshot-out", type=Path, required=True)
    parser.add_argument("--resource-out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    snapshot = build_snapshot(args.resolved_input, args.metadata_root)
    resource = build_resource(snapshot)
    _write_or_check(
        args.snapshot_out,
        _canonical_json(snapshot),
        check=args.check,
    )
    _write_or_check(
        args.resource_out,
        _canonical_json(resource),
        check=args.check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
