"""Small statistical helpers used across the package and its tests.

Kept deliberately thin: the estimators live in their pillars. What belongs
here is shared plumbing — Monte-Carlo standard errors, exact binomial
acceptance regions (review B5: every rate criterion states N and a region),
and the z-score comparison the ``statistical`` golden fixtures use (B6).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats as _st

from axiom.core.spec import Spec

__all__ = [
    "AcceptanceRegion",
    "clopper_pearson",
    "effective_sample_size",
    "mc_standard_error",
    "z_score",
]


class AcceptanceRegion(Spec):
    """An exact binomial acceptance region for a rate criterion.

    ``lower``/``upper`` bound the *count* of successes out of ``n`` that is
    consistent with ``p`` at level ``alpha`` (two-sided). A test that counts
    ``k`` successes passes iff ``lower <= k <= upper``.
    """

    n: int
    p: float
    alpha: float
    lower: int
    upper: int

    def accepts(self, k: int) -> bool:
        return self.lower <= k <= self.upper

    @property
    def rate_bounds(self) -> tuple[float, float]:
        return self.lower / self.n, self.upper / self.n


def clopper_pearson(n: int, p: float, alpha: float = 0.001) -> AcceptanceRegion:
    """Exact two-sided binomial acceptance region for ``k ~ Binomial(n, p)``."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if n < 1:
        raise ValueError("n must be positive")
    lo = int(_st.binom.ppf(alpha / 2, n, p))
    hi = int(_st.binom.ppf(1 - alpha / 2, n, p))
    return AcceptanceRegion(n=n, p=p, alpha=alpha, lower=lo, upper=hi)


def effective_sample_size(draws: npt.ArrayLike) -> float:
    """Bulk ESS via initial positive sequence of autocorrelations (Geyer 1992).

    ``draws`` is ``(chain, draw)`` or ``(draw,)``. Good enough for the
    diagnostics in core; ``infer.diagnostics`` wraps arviz when present.
    """
    x = np.asarray(draws, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    c, n = x.shape
    if n < 4:
        return float(c * n)
    x = x - x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, ddof=1).mean()
    if var == 0:
        return float(c * n)
    acov = np.zeros(n)
    for chain in x:
        f = np.fft.rfft(chain, 2 * n)
        acov += np.fft.irfft(f * np.conjugate(f))[:n] / n
    rho = acov / (c * var)
    # Geyer's initial positive sequence over pairs of autocorrelations.
    tau = -1.0
    for t in range(0, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    tau = max(tau, 1.0)
    return float(c * n / tau)


def mc_standard_error(draws: npt.ArrayLike) -> float:
    """Monte-Carlo SE of the mean, ``sd / sqrt(ESS)``."""
    x = np.asarray(draws, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(effective_sample_size(x)))


def z_score(value: float, reference: float, reference_se: float) -> float:
    """``(value - reference) / se``: how the statistical golden fixtures compare."""
    if reference_se <= 0:
        raise ValueError("reference_se must be positive")
    return (value - reference) / reference_se
