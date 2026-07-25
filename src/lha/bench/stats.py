"""Statistics for paired benchmark comparisons.

All three helpers are exact or deterministic: McNemar uses the exact binomial
(no chi-square approximation), the bootstrap is seeded and resamples task
clusters — repetitions of one task are correlated, so tasks, not cells, are the
exchangeable unit — and the Wilson interval covers the boundary proportions
where a percentile bootstrap degenerates.
"""

from __future__ import annotations

from math import comb, sqrt
from random import Random

# Two-sided normal quantile for 95% coverage.
_Z95 = 1.959963984540054


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
    clusters = [v for v in values_by_cluster.values() if v]
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
