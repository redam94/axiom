"""Expected value of information for a two-action decision on one parameter.

Ported from the parent's ``planning/evoi.py`` (golden
``planning.evoi::preposterior_sd_ratio``; ``compute_evpi`` ported by
specification). The decision is ``DecisionSpec``: *act* pays
``value_per_outcome_unit · (theta − threshold)`` in the numeraire, *hold*
pays nothing, so the decision-maker acts iff the expected ``theta`` exceeds
the threshold.

* ``preposterior_sd_ratio`` — ``sqrt(prior_sd² / (prior_sd² + experiment_se²))``:
  the standard deviation of the posterior *mean* the experiment will produce,
  relative to the prior sd. ``1`` is perfect information, ``0`` none.
* ``evpi_gaussian`` / ``evsi_gaussian`` / ``evoi_gaussian`` — closed forms
  for a Gaussian prior: ``value · E[max(theta − threshold, 0)] − value ·
  max(E theta − threshold, 0)`` with ``E[max(·, 0)]`` the standard normal
  partial expectation ``sd · (φ(d) + d · Φ(d))``. EVSI replaces the prior
  sd by the preposterior sd of the mean.
* ``evpi`` / ``evsi`` — Monte-Carlo versions for a prior given as draws and
  an optional non-linear ``value_fn``. ``evsi`` simulates the experiment and
  updates the empirical prior by importance weights under the Gaussian
  likelihood.

All values are in the decision's numeraire; the ``EVOIResult`` carries it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import field_validator
from scipy.special import logsumexp, ndtr

from axiom.core import NonEmptyStr, Spec

__all__ = [
    "DecisionSpec",
    "EVOIResult",
    "ValueFn",
    "evoi_gaussian",
    "evpi",
    "evpi_gaussian",
    "evsi",
    "evsi_gaussian",
    "preposterior_sd",
    "preposterior_sd_ratio",
]

Array = npt.NDArray[np.float64]
ValueFn = Callable[[Array], Array]
"""Maps parameter draws ``theta`` (shape ``(n,)``) to the payoff of *acting* at each."""


class DecisionSpec(Spec):
    """A two-action decision with value linear in the parameter.

    Acting pays ``value_per_outcome_unit · (theta − threshold)``; holding
    pays ``0``. ``numeraire`` names the currency the value is denominated in
    (rule 4: every currency-valued number carries its numeraire).
    """

    name: NonEmptyStr = "act"
    threshold: float = 0.0
    value_per_outcome_unit: float = 1.0
    numeraire: str = ""

    @field_validator("threshold")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"threshold must be finite, got {v}")
        return v

    @field_validator("value_per_outcome_unit")
    @classmethod
    def _positive_value(cls, v: float) -> float:
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"value_per_outcome_unit must be finite and positive, got {v}")
        return v

    def payoff(self, theta: npt.ArrayLike) -> Array:
        """The linear payoff of acting at each ``theta``."""
        x = np.asarray(theta, dtype=np.float64)
        out: Array = self.value_per_outcome_unit * (x - self.threshold)
        return out


class EVOIResult(Spec):
    """EVPI and EVSI of one experiment against one decision, in the numeraire.

    ``preposterior_sd`` is the standard deviation of the posterior mean the
    experiment will produce; ``evsi_se`` the Monte-Carlo standard error of
    ``evsi`` (``0`` for the closed form); ``n_sims``/``seed`` the simulation
    budget (``0``/``None`` for the closed form).
    """

    evpi: float
    evsi: float
    preposterior_sd: float
    decision_threshold: float
    n_sims: int
    seed: int | None = None
    method: Literal["gaussian", "monte_carlo"]
    evsi_se: float = 0.0
    experiment_se: float
    prior_mean: float
    prior_sd: float
    value_per_outcome_unit: float
    numeraire: str = ""


# -- Gaussian closed forms -------------------------------------------------------


def _positive(name: str, v: float) -> float:
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and positive, got {v}")
    return float(v)


def _finite(name: str, v: float) -> float:
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return float(v)


def preposterior_sd_ratio(prior_sd: float, experiment_se: float) -> float:
    """``sqrt(prior_sd² / (prior_sd² + experiment_se²))``.

    The sd of the posterior mean the experiment will produce, divided by the
    prior sd — the fraction of prior uncertainty the experiment resolves on
    the sd scale. Equivalently ``sqrt(1 − posterior_var / prior_var)``.
    """
    s = _positive("prior_sd", prior_sd)
    e = _positive("experiment_se", experiment_se)
    return math.sqrt(s * s / (s * s + e * e))


def preposterior_sd(prior_sd: float, experiment_se: float) -> float:
    """``prior_sd · preposterior_sd_ratio``: the sd of the posterior mean to come."""
    return prior_sd * preposterior_sd_ratio(prior_sd, experiment_se)


def _expected_positive_part(mean: float, sd: float) -> float:
    """``E[max(X, 0)]`` for ``X ~ N(mean, sd²)``: ``sd · φ(d) + mean · Φ(d)``, ``d = mean / sd``."""
    if sd == 0.0:
        return max(mean, 0.0)
    d = mean / sd
    phi = math.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
    return sd * phi + mean * float(ndtr(d))


def evpi_gaussian(decision: DecisionSpec, prior_mean: float, prior_sd: float) -> float:
    """Expected value of perfect information under ``theta ~ N(prior_mean, prior_sd²)``.

    ``value · (E[max(theta − threshold, 0)] − max(prior_mean − threshold, 0))``:
    what a clairvoyant would gain over acting on the prior mean alone.
    """
    m = _finite("prior_mean", prior_mean)
    s = _positive("prior_sd", prior_sd)
    gap = m - decision.threshold
    v = decision.value_per_outcome_unit
    return v * (_expected_positive_part(gap, s) - max(gap, 0.0))


def evsi_gaussian(
    decision: DecisionSpec, prior_mean: float, prior_sd: float, experiment_se: float
) -> float:
    """Expected value of the sample information an experiment with ``experiment_se`` yields.

    The posterior mean is preposterior-distributed ``N(prior_mean,
    preposterior_sd²)``; EVSI is the EVPI formula with that sd in place of
    the prior sd. It tends to ``evpi_gaussian`` as ``experiment_se → 0`` and
    to ``0`` as it grows.
    """
    m = _finite("prior_mean", prior_mean)
    s_pre = preposterior_sd(prior_sd, experiment_se)
    gap = m - decision.threshold
    v = decision.value_per_outcome_unit
    return v * (_expected_positive_part(gap, s_pre) - max(gap, 0.0))


def evoi_gaussian(
    decision: DecisionSpec, prior_mean: float, prior_sd: float, experiment_se: float
) -> EVOIResult:
    """``evpi_gaussian`` and ``evsi_gaussian`` together, as an ``EVOIResult``."""
    m = _finite("prior_mean", prior_mean)
    s = _positive("prior_sd", prior_sd)
    e = _positive("experiment_se", experiment_se)
    return EVOIResult(
        evpi=evpi_gaussian(decision, m, s),
        evsi=evsi_gaussian(decision, m, s, e),
        preposterior_sd=preposterior_sd(s, e),
        decision_threshold=decision.threshold,
        n_sims=0,
        seed=None,
        method="gaussian",
        evsi_se=0.0,
        experiment_se=e,
        prior_mean=m,
        prior_sd=s,
        value_per_outcome_unit=decision.value_per_outcome_unit,
        numeraire=decision.numeraire,
    )


# -- Monte Carlo -----------------------------------------------------------------


def _prior_draws_1d(draws: npt.ArrayLike) -> Array:
    x = np.asarray(draws, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least two finite prior draws")
    return np.asarray(x, dtype=np.float64)


def _payoffs(decision: DecisionSpec, theta: Array, value_fn: ValueFn | None) -> Array:
    v = np.asarray(decision.payoff(theta) if value_fn is None else value_fn(theta), dtype=float)
    if v.shape != theta.shape:
        raise ValueError(f"value_fn must return shape {theta.shape}, got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("value_fn returned non-finite payoffs")
    return np.asarray(v, dtype=np.float64)


def evpi(
    decision: DecisionSpec, prior_draws: npt.ArrayLike, *, value_fn: ValueFn | None = None
) -> float:
    """Monte-Carlo EVPI: ``mean(max(payoff, 0)) − max(mean(payoff), 0)`` over the prior draws.

    ``value_fn`` overrides the decision's linear payoff of acting; holding
    always pays zero. Deterministic given the draws.
    """
    theta = _prior_draws_1d(prior_draws)
    v = _payoffs(decision, theta, value_fn)
    return float(np.maximum(v, 0.0).mean() - max(float(v.mean()), 0.0))


def evsi(
    decision: DecisionSpec,
    prior_draws: npt.ArrayLike,
    experiment_se: float,
    *,
    n_sims: int = 1000,
    seed: int | None = None,
    value_fn: ValueFn | None = None,
    chunk: int = 256,
) -> EVOIResult:
    """Monte-Carlo EVSI of observing ``y = theta + N(0, experiment_se²)``.

    For each simulated outcome ``y_i`` (``theta_i`` resampled from the prior
    draws), the posterior over the empirical prior is the importance-weighted
    draw set with weights ``∝ N(y_i; theta_j, experiment_se²)``; the
    decision-maker acts iff the posterior expected payoff is positive. EVSI
    is the mean of the best posterior payoff minus the best prior payoff,
    truncated at zero (EVSI is non-negative by construction; ``evsi_se`` is
    the Monte-Carlo standard error of the untruncated estimator).
    ``preposterior_sd`` is the sd of the simulated posterior means.
    """
    theta = _prior_draws_1d(prior_draws)
    e = _positive("experiment_se", experiment_se)
    if n_sims < 2:
        raise ValueError(f"n_sims must be at least 2, got {n_sims}")
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    v = _payoffs(decision, theta, value_fn)
    prior_best = max(float(v.mean()), 0.0)
    rng = np.random.default_rng(seed)
    m = theta.size
    outer = theta[rng.integers(0, m, size=n_sims)]
    y = outer + e * rng.standard_normal(n_sims)
    best = np.empty(n_sims, dtype=np.float64)
    post_mean = np.empty(n_sims, dtype=np.float64)
    for start in range(0, n_sims, chunk):
        stop = min(start + chunk, n_sims)
        ll = -0.5 * ((y[start:stop, None] - theta[None, :]) / e) ** 2
        log_w = ll - np.asarray(logsumexp(ll, axis=1), dtype=np.float64)[:, None]
        w = np.exp(log_w)
        best[start:stop] = np.maximum(w @ v, 0.0)
        post_mean[start:stop] = w @ theta
    raw = float(best.mean()) - prior_best
    se = float(best.std(ddof=1) / math.sqrt(n_sims))
    return EVOIResult(
        evpi=evpi(decision, theta, value_fn=value_fn),
        evsi=max(raw, 0.0),
        preposterior_sd=float(post_mean.std(ddof=1)),
        decision_threshold=decision.threshold,
        n_sims=int(n_sims),
        seed=seed,
        method="monte_carlo",
        evsi_se=se,
        experiment_se=e,
        prior_mean=float(theta.mean()),
        prior_sd=float(theta.std(ddof=1)),
        value_per_outcome_unit=decision.value_per_outcome_unit,
        numeraire=decision.numeraire,
    )
