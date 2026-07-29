"""Statistics for paired benchmark comparisons.

The paired tests are exact, the bootstrap is deterministic for a fixed seed
and resamples task clusters, and the Wilson interval covers boundary
proportions where a percentile bootstrap degenerates. Repetitions of one task
are correlated, so current cell-level inference treats tasks, not cells, as the
exchangeable unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, sqrt
from random import Random

# Two-sided normal quantile for 95% coverage.
_Z95 = 1.959963984540054


@dataclass(frozen=True)
class ClusterSignFlipResult:
    """Exact paired sign-flip result with clusters as the exchangeable unit."""

    clusters: int
    nonzero_clusters: int
    mean_difference: float
    p_value: float


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    ``b`` = pairs where only condition A succeeded, ``c`` = pairs where only
    condition B succeeded. Concordant pairs carry no information about the
    difference and are not passed in. With no discordant pairs the test is
    undefined and the conventional p = 1.0 is returned.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_cluster_sign_flip_exact(
    pairs_by_cluster: dict[str, list[tuple[bool, bool]]],
) -> ClusterSignFlipResult:
    """Compare paired binary outcomes without treating repeated cells as independent.

    Each cluster contributes its mean paired difference, so a task with twelve
    repetitions has the same inferential weight as every other task. Under the
    paired randomization null, independently flipping the sign of every non-zero
    task effect gives the exact two-sided reference distribution.
    """
    if not pairs_by_cluster:
        raise ValueError("paired cluster test requires at least one cluster")

    effects: list[Fraction] = []
    for cluster, pairs in sorted(pairs_by_cluster.items()):
        if not cluster or not pairs:
            raise ValueError("paired cluster test requires named, non-empty clusters")
        if any(type(left) is not bool or type(right) is not bool for left, right in pairs):
            raise ValueError("paired cluster outcomes must be booleans")
        effects.append(
            Fraction(
                sum(int(right) - int(left) for left, right in pairs),
                len(pairs),
            )
        )

    nonzero = [effect for effect in effects if effect]
    observed = abs(sum(nonzero))
    if not nonzero:
        p_value = 1.0
    else:
        # The formal corpus has 17 task clusters. Exact rational sums prevent
        # a floating-point tie from changing whether a permutation is extreme.
        distribution = [Fraction(0)]
        for effect in nonzero:
            distribution = [
                *(value + effect for value in distribution),
                *(value - effect for value in distribution),
            ]
        extreme = sum(abs(value) >= observed for value in distribution)
        p_value = extreme / len(distribution)

    return ClusterSignFlipResult(
        clusters=len(effects),
        nonzero_clusters=len(nonzero),
        mean_difference=float(sum(effects) / len(effects)),
        p_value=p_value,
    )


def cluster_bootstrap_ci(
    values_by_cluster: dict[str, list[float]],
    *,
    n: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile CI for the pooled mean, resampling clusters with replacement.

    Returns None when there are no clusters or no values.
    """
    # A fixed seed must describe the data, not the insertion order used by a
    # JSON producer. Sorting cluster names makes the finite bootstrap exactly
    # reproducible after an otherwise harmless record reorder.
    clusters = [values_by_cluster[name] for name in sorted(values_by_cluster)]
    clusters = [values for values in clusters if values]
    if not clusters:
        return None
    rng = Random(seed)
    means: list[float] = []
    for _ in range(n):
        pooled: list[float] = []
        for _ in range(len(clusters)):
            pooled.extend(clusters[rng.randrange(len(clusters))])
        means.append(sum(pooled) / len(pooled))
    means.sort()
    lo = means[int((alpha / 2) * (n - 1))]
    hi = means[int((1 - alpha / 2) * (n - 1))]
    return (lo, hi)


def wilson_interval(successes: int, n: int, *, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Use this wherever a rate can sit on a boundary. A percentile bootstrap of
    ``0/n`` or ``n/n`` resamples an all-identical sample every time and reports
    a zero-width interval (``0%–0%``, ``100%–100%``) — that is an artifact of
    the method, not a measurement of certainty. Wilson stays honest there:
    ``0/51`` gives ``0%–7.0%``, the same order as the rule of three (≈5.9%).
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))
