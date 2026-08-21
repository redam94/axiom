"""Posterior predictive checks: is the observed panel a plausible draw from the fitted model?

For each test statistic ``T`` — a function of the ``(n_units, n_periods)``
outcome grid — the check compares ``T(y_obs)`` with the distribution of
``T(y_rep)`` over replicated outcomes, each ``y_rep`` drawn from the
likelihood at one posterior draw (``surface.forward.predict`` with
``noise=True``: one ``forward`` per draw, the likelihood family's noise on
top). Reported per statistic: the observed value, the equal-tailed
``mass`` predictive interval, and the posterior predictive p-value
``P(T(y_rep) >= T(y_obs))`` (Gelman, Meng & Stern 1996) with its two-sided
companion ``min(1, 2 · min(P(T_rep >= T_obs), P(T_rep <= T_obs)))`` — the
two tails are counted separately so ties (a replicate equal to the
observed value) count on both sides; ``extreme`` flags a two-sided p below
``alpha``. The number of replicates used is on every statistic.

Default statistics: ``mean``, ``sd``, ``min``, ``max``, ``lag1_autocorr``
(within-unit lag-1 autocorrelation, averaged over units), and ``unit_sd``
(the sd of the per-unit means — a check on the intercept structure).

Ported by specification from the parent's
``validation/posterior_predictive.py`` (ledger row ``diagnose/ppc.py``,
PORT).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import Interval, Posterior, Spec, Unsupported, Unverified, eti
from axiom.surface import FitResult
from axiom.surface.forward import predict

__all__ = [
    "DEFAULT_STATISTICS",
    "PPCResult",
    "Statistic",
    "StatisticCheck",
    "posterior_predictive",
]

Array = npt.NDArray[np.float64]
Statistic = Callable[[Array], float]
"""``(n_units, n_periods) outcome grid -> float``: a test statistic for the check."""


def _lag1(y: Array) -> float:
    if y.shape[1] < 3:
        return float("nan")
    centred = y - y.mean(axis=1, keepdims=True)
    num = np.sum(centred[:, 1:] * centred[:, :-1], axis=1)
    den = np.sum(centred**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(r)) if np.any(np.isfinite(r)) else float("nan")


DEFAULT_STATISTICS: dict[str, Statistic] = {
    "mean": lambda y: float(np.mean(y)),
    "sd": lambda y: float(np.std(y, ddof=1)) if y.size > 1 else float("nan"),
    "min": lambda y: float(np.min(y)),
    "max": lambda y: float(np.max(y)),
    "lag1_autocorr": _lag1,
    "unit_sd": lambda y: float(np.std(y.mean(axis=1), ddof=1)) if y.shape[0] > 1 else float("nan"),
}


class StatisticCheck(Spec):
    """One statistic: observed value, predictive interval, and the tail p-values over ``n``
    replicates. ``p_value`` is ``P(T_rep >= T_obs)``; ``p_two_sided`` is
    ``min(1, 2 · min(P(T_rep >= T_obs), P(T_rep <= T_obs)))``."""

    name: str
    observed: float
    interval: Interval
    p_value: float = Field(ge=0, le=1)
    p_two_sided: float = Field(ge=0, le=1)
    n: int = Field(ge=1)
    alpha: float = Field(gt=0, lt=1)
    extreme: bool

    @model_validator(mode="after")
    def _consistent(self) -> StatisticCheck:
        if self.extreme != (self.p_two_sided < self.alpha):
            raise ValueError(f"{self.name}: extreme must equal p_two_sided < alpha")
        return self


class PPCResult(Spec):
    """Every statistic's check plus the run's bookkeeping. ``skipped`` names statistics whose
    observed value was not finite (a lag-1 autocorrelation on a one-period panel, say) —
    reported, never silently dropped."""

    n_draws: int = Field(ge=1)
    seed: int | None
    mass: float = Field(gt=0, lt=1)
    alpha: float = Field(gt=0, lt=1)
    statistics: tuple[StatisticCheck, ...]
    skipped: tuple[str, ...] = ()
    extreme_statistics: tuple[str, ...]
    provenance: dict[str, Any] = {}

    @model_validator(mode="after")
    def _consistent(self) -> PPCResult:
        expected = tuple(s.name for s in self.statistics if s.extreme)
        if self.extreme_statistics != expected:
            raise ValueError("extreme_statistics must list exactly the extreme checks")
        return self


def _subsample(posterior: Posterior, n: int, rng: np.random.Generator) -> Posterior:
    """``n`` draws chosen without replacement (all of them when ``n`` covers the posterior)."""
    total = posterior.n_draws()
    if n >= total:
        idx = np.arange(total)
    else:
        idx = np.sort(rng.choice(total, size=n, replace=False))
    draws = {name: posterior.flat(name)[idx][None, ...] for name in sorted(posterior.names())}
    return Posterior(draws, coords=posterior.coords(), provenance=posterior.provenance)


def posterior_predictive(
    result: FitResult,
    *,
    statistics: Mapping[str, Statistic] | None = None,
    n_draws: int = 200,
    seed: int | None = 0,
    mass: float = 0.9,
    alpha: float = 0.05,
) -> PPCResult | Unsupported:
    """Posterior predictive checks of ``result`` on its own fitted panel.

    ``statistics`` maps names to functions of the outcome grid (default:
    ``DEFAULT_STATISTICS``); ``n_draws`` posterior draws are used (a
    seeded subsample without replacement when the posterior has more), each
    pushed through ``forward`` with likelihood noise. ``Unsupported`` when
    the fit has no posterior.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")
    stats = dict(statistics) if statistics is not None else dict(DEFAULT_STATISTICS)
    if not stats:
        raise ValueError("at least one statistic is needed")
    if isinstance(result.posterior, Unverified | Unsupported):
        return Unsupported(
            reason=f"posterior predictive check needs a posterior; the fit has "
            f"{type(result.posterior).__name__}: {result.posterior.reason}",
            missing=("posterior",),
        )
    rng = np.random.default_rng(seed)
    sub = _subsample(result.posterior, n_draws, rng)
    outcome = result.surface.spec.outcome.name
    observed = np.asarray(result.data[outcome], dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError(
            f"the fitted outcome must be an (n_units, n_periods) grid, got {observed.shape}"
        )
    rep = predict(
        result.surface, sub, result.data, seed=int(rng.integers(0, 2**31 - 1)), noise=True
    )
    grids = np.asarray(rep.values, dtype=np.float64).reshape(-1, *observed.shape)
    n = int(grids.shape[0])
    checks: list[StatisticCheck] = []
    skipped: list[str] = []
    for name, fn in stats.items():
        obs = float(fn(observed))
        if not np.isfinite(obs):
            skipped.append(name)
            continue
        reps = np.asarray([float(fn(g)) for g in grids], dtype=np.float64)
        finite = reps[np.isfinite(reps)]
        if finite.size == 0:
            skipped.append(name)
            continue
        p = float(np.mean(finite >= obs))
        p_lower = float(np.mean(finite <= obs))
        two = float(min(1.0, 2.0 * min(p, p_lower)))
        checks.append(
            StatisticCheck(
                name=name,
                observed=obs,
                interval=eti(finite, mass),
                p_value=p,
                p_two_sided=two,
                n=int(finite.size),
                alpha=alpha,
                extreme=two < alpha,
            )
        )
    return PPCResult(
        n_draws=n,
        seed=seed,
        mass=mass,
        alpha=alpha,
        statistics=tuple(checks),
        skipped=tuple(skipped),
        extreme_statistics=tuple(c.name for c in checks if c.extreme),
        provenance={
            "spec_hash": result.surface.spec.content_hash(),
            "model_hash": result.provenance.get("model_hash", result.surface.model.content_hash()),
            "panel_hash": result.provenance.get("panel_hash"),
            "n_posterior_draws": result.posterior.n_draws(),
        },
    )
