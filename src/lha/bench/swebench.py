"""SWE-bench Verified adapter: predictions file + official-harness invocation.

Separation of prediction and truth, same as ``lha ablate``:

  - The harness (this repo) produces patches. Its internal gate may run the
    target repo's own tests, but it never sees the evaluation oracle —
    SWE-bench applies its held-out FAIL_TO_PASS tests itself, inside a fresh
    per-instance container, from the frozen predictions file.
  - This module only formats predictions (``write_predictions``), builds the
    exact official command (``eval_command``), and parses the official report
    (``parse_report``). Instances that ERROR in evaluation stay in the
    denominator; they are reported, never dropped.

Requires the ``bench`` extra (``pip install 'lha[bench]'``) plus Docker for a
real evaluation run. On arm64 hosts pass ``namespace=""`` so images build
locally (upstream images are x86_64).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

DATASET = "SWE-bench/SWE-bench_Verified"  # 500 instances, frozen
SPLIT = "test"


@dataclass(frozen=True)
class Prediction:
    """One row of the predictions JSONL, in the official field names."""

    instance_id: str
    model_patch: str
    model_name_or_path: str

    def to_json(self) -> dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


def prediction_from_run(run_dir: str | Path, instance_id: str, model_name: str) -> Prediction:
    """The frozen patch a finished ``lha run`` produced, as a prediction.

    An absent or placeholder patch becomes an empty ``model_patch``, which the
    official harness buckets as ``empty_patch`` — a visible zero, not a crash.
    """
    diff_path = Path(run_dir) / "patch.diff"
    patch = diff_path.read_text() if diff_path.exists() else ""
    if patch.strip() == "(no diff)":
        patch = ""
    return Prediction(instance_id=instance_id, model_patch=patch, model_name_or_path=model_name)


def write_predictions(preds: list[Prediction], path: str | Path) -> Path:
    """Write the predictions JSONL the official harness consumes."""
    seen: set[str] = set()
    for p in preds:
        if p.instance_id in seen:
            raise ValueError(f"duplicate prediction for {p.instance_id}")
        seen.add(p.instance_id)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p.to_json()) + "\n" for p in preds))
    return out


def eval_command(
    predictions_path: str | Path,
    run_id: str,
    *,
    dataset: str = DATASET,
    split: str = SPLIT,
    max_workers: int = 8,
    namespace: str | None = None,
) -> list[str]:
    """The exact official evaluation invocation (swebench >= 4.1)."""
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if namespace is not None:  # "" -> build images locally (arm64)
        cmd += ["--namespace", namespace]
    return cmd


@dataclass
class SWEBenchSummary:
    """The official report's buckets, with errors kept in the denominator."""

    total: int
    submitted: int
    resolved: int
    unresolved: int
    empty_patch: int
    error: int
    incomplete: int
    resolved_ids: list[str] = field(default_factory=list)
    error_ids: list[str] = field(default_factory=list)

    @property
    def resolved_rate(self) -> float:
        """Resolved over ALL submitted instances — errors count against it."""
        return self.resolved / self.submitted if self.submitted else 0.0

    @property
    def error_rate(self) -> float:
        return self.error / self.submitted if self.submitted else 0.0

    def to_markdown(self) -> str:
        return (
            f"| {self.resolved}/{self.submitted} resolved "
            f"({self.resolved_rate:.1%}) | {self.unresolved} unresolved | "
            f"{self.empty_patch} empty | {self.error} error | "
            f"{self.incomplete} incomplete |"
        )


def parse_report(path: str | Path) -> SWEBenchSummary:
    """Parse the official ``<model>.<run_id>.json`` report (schema_version 2)."""
    raw = json.loads(Path(path).read_text())
    ids = {k: list(raw.get(k, [])) for k in ("resolved_ids", "error_ids")}
    return SWEBenchSummary(
        total=int(raw["total_instances"]),
        submitted=int(raw["submitted_instances"]),
        resolved=int(raw["resolved_instances"]),
        unresolved=int(raw["unresolved_instances"]),
        empty_patch=int(raw["empty_patch_instances"]),
        error=int(raw["error_instances"]),
        incomplete=len(raw.get("incomplete_ids", [])),
        resolved_ids=ids["resolved_ids"],
        error_ids=ids["error_ids"],
    )
