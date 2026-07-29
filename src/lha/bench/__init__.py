"""Adapters for independent public evaluation.

The point of this package is that the numbers come from someone else's
harness: predictions are frozen to a file, the official evaluator runs them
in fresh containers, and this code only formats inputs and parses outputs.
Nothing here executes model patches itself.
"""

from .stats import (
    ClusterSignFlipResult,
    cluster_bootstrap_ci,
    mcnemar_exact,
    paired_cluster_sign_flip_exact,
)
from .swebench import Prediction, SWEBenchSummary, eval_command, parse_report, write_predictions

__all__ = [
    "Prediction",
    "SWEBenchSummary",
    "ClusterSignFlipResult",
    "cluster_bootstrap_ci",
    "eval_command",
    "mcnemar_exact",
    "paired_cluster_sign_flip_exact",
    "parse_report",
    "write_predictions",
]
