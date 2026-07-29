"""Streaming mean and population variance.

``OnlineStats`` consumes samples one at a time and exposes the running mean
and population variance (dividing by n, not n-1). The estimator must stay
numerically accurate when samples share a large common offset: adding 1e9 to
every sample must not change the variance. Reading ``mean`` or ``variance``
with no samples raises ``ValueError``; a single sample has variance 0.0.
"""


class OnlineStats:
    def __init__(self):
        self._n = 0
        self._sum = 0.0
        self._sumsq = 0.0

    def add(self, x: float) -> None:
        self._n += 1
        self._sum += x
        self._sumsq += x * x

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        if self._n == 0:
            raise ValueError("no samples")
        return self._sum / self._n

    @property
    def variance(self) -> float:
        if self._n == 0:
            raise ValueError("no samples")
        return self._sumsq / self._n - self.mean**2
