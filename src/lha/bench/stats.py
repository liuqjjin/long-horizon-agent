"""Statistics for paired benchmark comparisons.

Both helpers are exact/deterministic: McNemar uses the exact binomial (no
chi-square approximation), and the bootstrap is seeded and resamples task
clusters — repetitions of one task are correlated, so tasks, not cells, are
the exchangeable unit.
"""

from __future__ import annotations

from math import comb
from random import Random


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
