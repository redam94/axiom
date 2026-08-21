"""Expected information gain of a one-parameter experiment, and how it decays.

Ported from the parent's ``planning/eig.py`` (golden ``planning.eig::*``,
bit-stable). Everything here concerns one scalar parameter ``theta`` — the
value of an estimand — with a Gaussian prior of standard deviation
``prior_sd`` (or a set of prior draws) and an experiment that measures
``theta`` with standard error ``experiment_se``:

* ``eig_gaussian`` — the closed form ``0.5 · ln(1 + prior_sd² / experiment_se²)``
  nats: the mutual information between ``theta`` and the experiment's estimate.
* ``eig_monte_carlo`` — the nested Monte-Carlo estimator of the same quantity
  for an arbitrary prior given as draws, with its Monte-Carlo standard error.
* ``experiment_se_for_design`` — a design kind's relative standard error on
  the estimand, times a reference value. The table is documented below.
* ``decayed_sd`` — the posterior standard deviation after ``t`` periods under
  the parent's decay model: the posterior *variance* doubles every half-life.
* ``information_half_life`` — how long until half the information an
  experiment bought (in nats) has decayed away.
* ``time_to_re_experiment`` — the first period at which repeating an
  experiment would be worth at least ``min_eig`` nats again.

Pure numpy and scipy; no sampler, no fitted model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import field_validator
from scipy.special import logsumexp

from axiom.core import NonEmptyStr, Spec, Unsupported

__all__ = [
    "DESIGN_RELATIVE_SE",
    "EIGEstimate",
    "ReExperimentTiming",
    "decayed_sd",
    "eig_gaussian",
    "eig_monte_carlo",
    "experiment_se_for_design",
    "information_half_life",
    "time_to_re_experiment",
]

Array = npt.NDArray[np.float64]

DESIGN_RELATIVE_SE: Mapping[str, float] = MappingProxyType(
    {
        # Ported from the parent (golden ``planning.eig::sigma_exp_for_design``):
        # a cluster holdout measures the estimand to 10 % of its reference value,
        # a ghost (would-have-been-exposed control) design to 25 %.
        "cluster_holdout": 0.10,
        "ghost": 0.25,
    }
)
"""Relative standard error of the estimand by design kind.

Only kinds with a recorded provenance are listed. Any other kind needs an
explicit ``relative_se`` — ``experiment_se_for_design`` returns
``Unsupported`` rather than guess.
"""


class EIGEstimate(Spec):
    """Expected information gain in nats, with the error of its estimator.

    ``se`` is the Monte-Carlo standard error of ``eig`` (``0`` for the closed
    form). ``n_sims`` is the number of outer simulations and ``n_prior_draws``
    the size of the prior sample used as the inner mixture.
    """

    eig: float
    se: float
    n_sims: int
    seed: int | None = None
    method: Literal["gaussian", "nested_monte_carlo"]
    n_prior_draws: int = 0
    experiment_se: float
    unit: Literal["nats"] = "nats"


class ReExperimentTiming(Spec):
    """When an experiment with ``experiment_se`` is worth ``min_eig`` nats again.

    ``sd_at_threshold`` is the posterior standard deviation at which the
    Gaussian EIG of the experiment equals ``min_eig``; ``periods`` is the
    first time the decayed posterior reaches it (``0`` if it already has).
    ``eig_now`` is the EIG of running the experiment today.
    """

    posterior_sd: float
    half_life_periods: float
    experiment_se: float
    min_eig: float
    sd_at_threshold: float
    periods: float
    eig_now: float
    design_kind: NonEmptyStr = "unspecified"

    @field_validator("periods")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"periods must be non-negative, got {v}")
        return v


# -- closed forms -------------------------------------------------------------


def _positive(name: str, v: float) -> float:
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and positive, got {v}")
    return float(v)


def eig_gaussian(prior_sd: float, experiment_se: float) -> float:
    """``0.5 · ln(1 + prior_sd² / experiment_se²)`` nats.

    The mutual information between ``theta ~ N(m, prior_sd²)`` and an
    estimate ``y = theta + N(0, experiment_se²)``; equivalently the entropy
    drop from prior to posterior, ``ln(prior_sd / posterior_sd)``.
    """
    s = _positive("prior_sd", prior_sd)
    e = _positive("experiment_se", experiment_se)
    return 0.5 * math.log(1.0 + s * s / (e * e))


def experiment_se_for_design(
    design_kind: str, reference_value: float, *, relative_se: float | None = None
) -> float | Unsupported:
    """``relative_se · |reference_value|``, with ``relative_se`` looked up by design kind.

    ``reference_value`` is the scale the experiment is measured against — the
    prior median of the estimand, say. Pass ``relative_se`` to override the
    table or to supply a kind the table does not list; a kind it does not
    list and no override is ``Unsupported``, not a default.
    """
    if not math.isfinite(reference_value) or reference_value == 0:
        raise ValueError(f"reference_value must be finite and non-zero, got {reference_value}")
    if relative_se is None:
        if design_kind not in DESIGN_RELATIVE_SE:
            return Unsupported(
                reason=(
                    f"no relative standard error is recorded for design kind {design_kind!r}; "
                    "pass relative_se explicitly"
                ),
                missing=(design_kind,),
                detail={"known_kinds": ", ".join(sorted(DESIGN_RELATIVE_SE))},
            )
        relative_se = DESIGN_RELATIVE_SE[design_kind]
    rel = _positive("relative_se", relative_se)
    return rel * abs(reference_value)


def decayed_sd(posterior_sd: float, periods_elapsed: float, half_life_periods: float) -> float:
    """``posterior_sd · 2^(t / (2 · half_life))``: the posterior variance doubles every half-life.

    The parent's decay model for information that ages — the world drifts,
    so what an experiment established loses precision at a fixed rate. It is
    unbounded by design: there is no floor at the prior.
    """
    s = _positive("posterior_sd", posterior_sd)
    hl = _positive("half_life_periods", half_life_periods)
    if not math.isfinite(periods_elapsed) or periods_elapsed < 0:
        raise ValueError(f"periods_elapsed must be finite and non-negative, got {periods_elapsed}")
    return float(s * 2.0 ** (periods_elapsed / (2.0 * hl)))


def information_half_life(prior_sd: float, posterior_sd: float, half_life_periods: float) -> float:
    """Periods until half the information the experiment bought has decayed.

    The experiment took the standard deviation from ``prior_sd`` to
    ``posterior_sd``, a gain of ``ln(prior_sd / posterior_sd)`` nats. Under
    ``decayed_sd`` that gain is halved when the decayed sd reaches
    ``sqrt(prior_sd · posterior_sd)``, i.e. after
    ``half_life_periods · log2(prior_sd / posterior_sd)`` periods. Zero when
    nothing was learned.
    """
    p = _positive("prior_sd", prior_sd)
    s = _positive("posterior_sd", posterior_sd)
    hl = _positive("half_life_periods", half_life_periods)
    if s > p:
        raise ValueError(
            f"posterior_sd {s} exceeds prior_sd {p}; an experiment cannot lose information"
        )
    return hl * math.log2(p / s)


def time_to_re_experiment(
    posterior_sd: float,
    half_life_periods: float,
    experiment_se: float,
    min_eig: float,
    *,
    design_kind: str = "unspecified",
) -> ReExperimentTiming:
    """First period at which re-running the experiment is worth ``min_eig`` nats.

    The Gaussian EIG of an experiment with ``experiment_se`` against a
    posterior of sd ``s`` is ``0.5 · ln(1 + s² / experiment_se²)``; it equals
    ``min_eig`` at ``s* = experiment_se · sqrt(exp(2 · min_eig) − 1)``. Under
    ``decayed_sd`` the posterior reaches ``s*`` after
    ``2 · half_life · log2(s* / posterior_sd)`` periods, or immediately if it
    is already there.
    """
    s = _positive("posterior_sd", posterior_sd)
    hl = _positive("half_life_periods", half_life_periods)
    e = _positive("experiment_se", experiment_se)
    m = _positive("min_eig", min_eig)
    sd_star = e * math.sqrt(math.expm1(2.0 * m))
    periods = max(0.0, 2.0 * hl * math.log2(sd_star / s))
    return ReExperimentTiming(
        posterior_sd=s,
        half_life_periods=hl,
        experiment_se=e,
        min_eig=m,
        sd_at_threshold=sd_star,
        periods=periods,
        eig_now=eig_gaussian(s, e),
        design_kind=design_kind,
    )


# -- Monte Carlo --------------------------------------------------------------


def _prior_draws_1d(draws: npt.ArrayLike) -> Array:
    x = np.asarray(draws, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least two finite prior draws")
    return np.asarray(x, dtype=np.float64)


def eig_monte_carlo(
    prior_draws: npt.ArrayLike,
    experiment_se: float,
    n_sims: int = 2000,
    seed: int | None = None,
    *,
    chunk: int = 256,
) -> EIGEstimate:
    """Nested Monte-Carlo EIG of ``y = theta + N(0, experiment_se²)`` under an empirical prior.

    Outer loop: ``theta_i`` is resampled from ``prior_draws`` and ``y_i``
    simulated. Inner expectation: the marginal ``p(y_i)`` is the mixture of
    the Gaussian likelihood over *all* prior draws. The estimator is
    ``mean_i [log p(y_i | theta_i) − log p(y_i)]`` with ``se`` its
    Monte-Carlo standard error. It is consistent as the number of prior draws
    grows; its bias is ``O(1 / n_prior_draws)``. For a Gaussian prior it
    agrees with ``eig_gaussian`` to within ``se`` (tested).
    """
    theta = _prior_draws_1d(prior_draws)
    e = _positive("experiment_se", experiment_se)
    if n_sims < 2:
        raise ValueError(f"n_sims must be at least 2, got {n_sims}")
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    rng = np.random.default_rng(seed)
    m = theta.size
    outer = theta[rng.integers(0, m, size=n_sims)]
    y = outer + e * rng.standard_normal(n_sims)
    terms = np.empty(n_sims, dtype=np.float64)
    log_m = math.log(m)
    for start in range(0, n_sims, chunk):
        stop = min(start + chunk, n_sims)
        ys = y[start:stop]
        ll = -0.5 * ((ys[:, None] - theta[None, :]) / e) ** 2
        log_marginal = np.asarray(logsumexp(ll, axis=1), dtype=np.float64) - log_m
        ll_true = -0.5 * ((ys - outer[start:stop]) / e) ** 2
        terms[start:stop] = ll_true - log_marginal
    eig = float(terms.mean())
    se = float(terms.std(ddof=1) / math.sqrt(n_sims))
    return EIGEstimate(
        eig=eig,
        se=se,
        n_sims=int(n_sims),
        seed=seed,
        method="nested_monte_carlo",
        n_prior_draws=int(m),
        experiment_se=e,
    )
