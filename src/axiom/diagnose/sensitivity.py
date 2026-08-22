"""Sensitivity to unobserved confounding: one engine, three views.

Consolidates the parent's ``diagnostics/bias_sensitivity.py``,
``validation/confounding_sensitivity.py`` and
``validation/sensitivity_unobserved.py`` (porting ledger: three views of one
engine). Everything here is closed form; nothing refits.

View (a) — partial-R² benchmarking (Cinelli & Hazlett 2020, *JRSS-B* 82(1)).
For a linear estimate ``tau_hat`` with standard error ``se`` on ``df``
residual degrees of freedom, an omitted confounder ``Z`` is described by two
partial R² values: ``r2_dz_x`` (how much of the treatment's residual
variance ``Z`` explains) and ``r2_yz_dx`` (how much of the outcome's
residual variance ``Z`` explains, given the treatment). Then

    |bias|  = se · sqrt(df) · sqrt(r2_yz_dx · r2_dz_x / (1 − r2_dz_x))
    se_adj  = se · sqrt((1 − r2_yz_dx) / (1 − r2_dz_x)) · sqrt(df / (df − 1))

The **robustness value** ``RV_q`` is the strength (equal for both partial
R²s) a confounder needs to reduce the estimate by a fraction ``q``; with
``f = q · |t| / sqrt(df)`` it is ``RV_q = ½ (sqrt(f⁴ + 4 f²) − f²)``.
``RV_{q,α}`` is the strength needed to bring the estimate to the edge of
statistical significance at level ``α``: the same formula with
``f_{q,α} = q · |t| / sqrt(df) − t_{α/2, df−1} / sqrt(df − 1)`` (zero when
that is negative). ``benchmark`` bounds the two partial R²s by stating
that ``Z`` is at most ``k_d`` (``k_y``) times as strong as an observed
covariate ``X_j`` whose own partial R²s are known (sensemakr's
``ovb_bounds``).

View (b) — a bias parameter applied to existing draws. ``shift_posterior``
subtracts a bias draw from each estimate draw, so a Gaussian bias prior
turns a posterior into its convolution with that prior without a refit;
the result carries its own interval.

View (c) — decision-scale tipping points. ``tipping_point`` walks a grid of
hypothesised biases and reports the smallest at which the decision —
``P(effect − bias > threshold) >= certainty`` — flips, with the interval at
that bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import stats as _st

from axiom.core import Interval, Posterior, Spec, Summary, Unsupported, summarize, wald
from axiom.core.intervals import IntervalDefinition

__all__ = [
    "Benchmark",
    "BiasBounds",
    "RobustnessValue",
    "ShiftedPosterior",
    "TippingPoint",
    "benchmark",
    "bias_bounds",
    "partial_r2",
    "robustness_value",
    "shift_posterior",
    "tipping_point",
]

Array = npt.NDArray[np.float64]


# -- view (a): partial-R² benchmarking --------------------------------------------------


def partial_r2(t_statistic: float, df: int) -> float:
    """Partial R² of a regressor from its t statistic: ``t² / (t² + df)``."""
    if df < 1:
        raise ValueError(f"df must be positive, got {df}")
    t2 = float(t_statistic) ** 2
    return t2 / (t2 + float(df))


def _rv(f: float) -> float:
    """``½ (sqrt(f⁴ + 4 f²) − f²)`` — the root of ``RV² / (1 − RV) = f²``."""
    f2 = f * f
    return 0.5 * (float(np.sqrt(f2 * f2 + 4.0 * f2)) - f2)


class RobustnessValue(Spec):
    """Cinelli–Hazlett robustness values for one linear estimate.

    ``rv`` is ``RV_q``; ``rv_alpha`` is ``RV_{q,α}``; ``r2_yd_x`` is the
    treatment's own partial R² with the outcome (``t² / (t² + df)``), the
    strength a confounder would need to explain away the estimate if it
    were as associated with the outcome as the treatment is.
    """

    estimate: float
    se: float = Field(gt=0)
    df: int = Field(ge=2)
    q: float = Field(gt=0)
    alpha: float = Field(gt=0, lt=1)
    t: float
    r2_yd_x: float = Field(ge=0, le=1)
    rv: float = Field(ge=0, le=1)
    rv_alpha: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self) -> RobustnessValue:
        if self.rv_alpha > self.rv + 1e-12:
            raise ValueError("RV_{q,alpha} cannot exceed RV_q")
        return self


def robustness_value(
    estimate: float, se: float, df: int, *, q: float = 1.0, alpha: float = 0.05
) -> RobustnessValue:
    """``RV_q`` and ``RV_{q,α}`` for a linear estimate (Cinelli & Hazlett 2020, §4.2)."""
    if se <= 0:
        raise ValueError(f"se must be positive, got {se}")
    if df < 2:
        raise ValueError(f"df must be at least 2, got {df}")
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    t = float(estimate) / float(se)
    f = float(q) * abs(t) / float(np.sqrt(df))
    # the significance-adjusted f uses df − 1 because the adjusted model has one more regressor
    f_alpha = f - float(_st.t.ppf(1.0 - alpha / 2.0, df - 1)) / float(np.sqrt(df - 1))
    return RobustnessValue(
        estimate=float(estimate),
        se=float(se),
        df=int(df),
        q=float(q),
        alpha=float(alpha),
        t=t,
        r2_yd_x=partial_r2(t, df),
        rv=_rv(f),
        rv_alpha=_rv(f_alpha) if f_alpha > 0.0 else 0.0,
    )


class BiasBounds(Spec):
    """The omitted-variable bias a confounder of stated strength would produce.

    ``bias`` is its absolute value; ``adjusted_estimate`` moves the
    estimate toward zero by ``bias`` (``reduce=True``) or away from it;
    ``adjusted_interval`` is the Wald interval on the adjusted estimate
    and ``adjusted_se`` with ``mass`` stated.
    """

    estimate: float
    se: float = Field(gt=0)
    df: int = Field(ge=2)
    r2_yz_dx: float = Field(ge=0, lt=1)
    r2_dz_x: float = Field(ge=0, lt=1)
    reduce: bool
    bias: float = Field(ge=0)
    adjusted_estimate: float
    adjusted_se: float = Field(gt=0)
    adjusted_t: float
    adjusted_interval: Interval


def bias_bounds(
    estimate: float,
    se: float,
    df: int,
    *,
    r2_yz_dx: float,
    r2_dz_x: float,
    reduce: bool = True,
    mass: float = 0.95,
) -> BiasBounds:
    """Price a confounder with partial R² ``(r2_yz_dx, r2_dz_x)`` (Cinelli & Hazlett 2020, §4.1).

    ``|bias| = se · sqrt(df) · sqrt(r2_yz_dx · r2_dz_x / (1 − r2_dz_x))``
    and the adjusted standard error rescales by
    ``sqrt((1 − r2_yz_dx) / (1 − r2_dz_x)) · sqrt(df / (df − 1))``.
    """
    if se <= 0:
        raise ValueError(f"se must be positive, got {se}")
    if df < 2:
        raise ValueError(f"df must be at least 2, got {df}")
    for name, r2 in (("r2_yz_dx", r2_yz_dx), ("r2_dz_x", r2_dz_x)):
        if not 0.0 <= r2 < 1.0:
            raise ValueError(f"{name} must lie in [0, 1), got {r2}")
    bias = float(se) * float(np.sqrt(df)) * float(np.sqrt(r2_yz_dx * r2_dz_x / (1.0 - r2_dz_x)))
    sign = 1.0 if estimate >= 0 else -1.0
    adjusted = float(estimate) - sign * bias if reduce else float(estimate) + sign * bias
    se_adj = (
        float(se)
        * float(np.sqrt((1.0 - r2_yz_dx) / (1.0 - r2_dz_x)))
        * float(np.sqrt(df / (df - 1.0)))
    )
    return BiasBounds(
        estimate=float(estimate),
        se=float(se),
        df=int(df),
        r2_yz_dx=float(r2_yz_dx),
        r2_dz_x=float(r2_dz_x),
        reduce=reduce,
        bias=bias,
        adjusted_estimate=adjusted,
        adjusted_se=se_adj,
        adjusted_t=adjusted / se_adj,
        adjusted_interval=wald(adjusted, se_adj, mass),
    )


class Benchmark(Spec):
    """Partial-R² bounds for a confounder ``k_d`` / ``k_y`` times an observed covariate.

    ``r2_dxj_x`` and ``r2_yxj_dx`` are the covariate's own partial R²s with
    the treatment (given the other covariates) and with the outcome (given
    the treatment and the other covariates). ``bounds`` prices the
    confounder those multiples imply.
    """

    covariate: str
    k_d: float = Field(gt=0)
    k_y: float = Field(gt=0)
    r2_dxj_x: float = Field(ge=0, lt=1)
    r2_yxj_dx: float = Field(ge=0, lt=1)
    r2_dz_x: float = Field(ge=0, lt=1)
    r2_yz_dx: float = Field(ge=0, lt=1)
    bounds: BiasBounds


def benchmark(
    estimate: float,
    se: float,
    df: int,
    *,
    covariate: str,
    r2_dxj_x: float,
    r2_yxj_dx: float,
    k_d: float = 1.0,
    k_y: float = 1.0,
    reduce: bool = True,
    mass: float = 0.95,
) -> Benchmark | Unsupported:
    """Bound a confounder by an observed covariate (Cinelli & Hazlett 2020, §4.4; sensemakr).

    ``r2_dz_x = k_d · r2_dxj_x / (1 − r2_dxj_x)`` and, with
    ``r2_zxj_dx = k_d · r2_dxj_x² / ((1 − k_d · r2_dxj_x)(1 − r2_dxj_x))``,
    ``r2_yz_dx = ((sqrt(k_y) + sqrt(r2_zxj_dx)) / sqrt(1 − r2_zxj_dx))² ·
    r2_yxj_dx / (1 − r2_yxj_dx)``. A multiple so large that a bound leaves
    ``[0, 1)`` is not a confounder the data can host: ``Unsupported``.
    """
    if k_d <= 0 or k_y <= 0:
        raise ValueError(f"k_d and k_y must be positive, got {k_d}, {k_y}")
    for name, r2 in (("r2_dxj_x", r2_dxj_x), ("r2_yxj_dx", r2_yxj_dx)):
        if not 0.0 <= r2 < 1.0:
            raise ValueError(f"{name} must lie in [0, 1), got {r2}")
    r2_dz_x = k_d * r2_dxj_x / (1.0 - r2_dxj_x)
    denom = (1.0 - k_d * r2_dxj_x) * (1.0 - r2_dxj_x)
    if r2_dz_x >= 1.0 or denom <= 0.0:
        return Unsupported(
            reason=f"k_d={k_d} times covariate {covariate!r} implies r2_dz_x >= 1",
            detail={"r2_dz_x": repr(r2_dz_x), "covariate": covariate},
        )
    r2_zxj_dx = k_d * r2_dxj_x**2 / denom
    if r2_zxj_dx >= 1.0:
        return Unsupported(
            reason=f"k_d={k_d} times covariate {covariate!r} implies r2_zxj_dx >= 1",
            detail={"r2_zxj_dx": repr(r2_zxj_dx), "covariate": covariate},
        )
    eta = (float(np.sqrt(k_y)) + float(np.sqrt(r2_zxj_dx))) / float(np.sqrt(1.0 - r2_zxj_dx))
    r2_yz_dx = eta**2 * r2_yxj_dx / (1.0 - r2_yxj_dx)
    if r2_yz_dx >= 1.0:
        return Unsupported(
            reason=f"k_y={k_y} times covariate {covariate!r} implies r2_yz_dx >= 1",
            detail={"r2_yz_dx": repr(r2_yz_dx), "covariate": covariate},
        )
    return Benchmark(
        covariate=covariate,
        k_d=float(k_d),
        k_y=float(k_y),
        r2_dxj_x=float(r2_dxj_x),
        r2_yxj_dx=float(r2_yxj_dx),
        r2_dz_x=r2_dz_x,
        r2_yz_dx=r2_yz_dx,
        bounds=bias_bounds(
            estimate, se, df, r2_yz_dx=r2_yz_dx, r2_dz_x=r2_dz_x, reduce=reduce, mass=mass
        ),
    )


# -- view (b): a bias parameter over existing draws ------------------------------------


def _flat(draws: npt.ArrayLike | Posterior, name: str | None) -> Array:
    if isinstance(draws, Posterior):
        if name is None:
            raise ValueError("a Posterior needs the name of the variable to shift")
        x = draws.flat(name)
    else:
        x = np.asarray(draws, dtype=float)
    x = x.reshape(-1)
    if x.size < 2:
        raise ValueError("need at least two draws")
    if not np.all(np.isfinite(x)):
        raise ValueError("draws must be finite")
    return x


@dataclass(frozen=True)
class ShiftedPosterior:
    """Estimate draws with a bias draw subtracted from each — the bias-adjusted posterior.

    ``posterior`` holds ``estimate``, ``bias`` and ``shifted`` draws (one
    chain) with the seed and pairing in its provenance; the three
    ``Summary`` objects carry the interval of each under the stated
    definition and mass.
    """

    posterior: Posterior
    estimate: Summary
    bias: Summary
    shifted: Summary

    @property
    def interval(self) -> Interval:
        return self.shifted.interval

    def draws(self, name: str) -> Array:
        return self.posterior.draws(name)

    def names(self) -> frozenset[str]:
        return self.posterior.names()


def shift_posterior(
    draws: npt.ArrayLike | Posterior,
    *,
    name: str | None = None,
    bias_draws: npt.ArrayLike | None = None,
    bias_mean: float = 0.0,
    bias_sd: float = 0.0,
    definition: IntervalDefinition = "eti",
    mass: float = 0.9,
    seed: int | None = None,
) -> ShiftedPosterior:
    """Subtract a bias from every estimate draw without refitting.

    The bias is either ``bias_draws`` (resampled with replacement to the
    estimate's length when the lengths differ) or a Gaussian
    ``N(bias_mean, bias_sd²)`` drawn independently per estimate draw — the
    closed-form convolution of the posterior with a Gaussian bias prior
    (``bias_sd=0`` is a constant shift). Pairing is random with ``seed``.
    """
    x = _flat(draws, name)
    rng = np.random.default_rng(seed)
    if bias_draws is not None:
        b = np.asarray(bias_draws, dtype=float).reshape(-1)
        if b.size == 0 or not np.all(np.isfinite(b)):
            raise ValueError("bias_draws must be non-empty and finite")
        if b.size != x.size:
            b = rng.choice(b, size=x.size, replace=True)
        else:
            b = rng.permutation(b)
        how = "bias_draws"
    else:
        if bias_sd < 0:
            raise ValueError(f"bias_sd must be non-negative, got {bias_sd}")
        b = bias_mean + bias_sd * rng.standard_normal(x.size)
        how = "gaussian"
    shifted = x - b
    post = Posterior(
        {"estimate": x[None, :], "bias": b[None, :], "shifted": shifted[None, :]},
        provenance={
            "method": "shift_posterior",
            "bias": how,
            "bias_mean": float(bias_mean),
            "bias_sd": float(bias_sd),
            "seed": seed,
            "definition": definition,
            "mass": float(mass),
        },
    )
    return ShiftedPosterior(
        posterior=post,
        estimate=summarize(x, definition=definition, mass=mass),
        bias=summarize(b, definition=definition, mass=mass),
        shifted=summarize(shifted, definition=definition, mass=mass),
    )


# -- view (c): decision-scale tipping points -------------------------------------------


class TippingPoint(Spec):
    """The smallest hypothesised bias at which a threshold decision flips.

    The decision at bias ``b`` is ``P(effect − b > threshold) >= certainty``.
    ``bias_grid`` and ``probabilities`` record the whole walk; ``bias`` is
    the first grid point whose decision differs from the one at zero bias
    (``None`` when none does — ``flipped=False``), with ``interval`` and
    ``probability`` at that bias.
    """

    threshold: float
    certainty: float = Field(gt=0, lt=1)
    n_draws: int = Field(ge=2)
    bias_grid: tuple[float, ...] = Field(min_length=1)
    probabilities: tuple[float, ...] = Field(min_length=1)
    decision_at_zero: bool
    probability_at_zero: float = Field(ge=0, le=1)
    flipped: bool
    bias: float | None
    probability: float | None
    interval: Interval | None

    @model_validator(mode="after")
    def _consistent(self) -> TippingPoint:
        if len(self.bias_grid) != len(self.probabilities):
            raise ValueError("bias_grid and probabilities must have the same length")
        if self.flipped != (self.bias is not None):
            raise ValueError("flipped must agree with whether a tipping bias was found")
        if self.flipped and (self.interval is None or self.probability is None):
            raise ValueError("a flipped decision needs its interval and probability")
        return self


def tipping_point(
    estimate_draws: npt.ArrayLike | Posterior,
    decision_threshold: float,
    bias_grid: npt.ArrayLike,
    *,
    name: str | None = None,
    certainty: float = 0.5,
    definition: IntervalDefinition = "eti",
    mass: float = 0.9,
) -> TippingPoint:
    """Walk ``bias_grid`` (sorted by absolute size) and find where the decision flips.

    ``certainty=0.5`` makes the decision the posterior median against the
    threshold; a stricter certainty demands that much posterior mass above
    it. Grid points are visited in order of ``|bias|`` so the reported
    tipping bias is the smallest in magnitude, whichever sign flips first.
    """
    x = _flat(estimate_draws, name)
    grid = np.asarray(bias_grid, dtype=float).reshape(-1)
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("bias_grid must be non-empty and finite")
    if not 0.0 < certainty < 1.0:
        raise ValueError(f"certainty must lie in (0, 1), got {certainty}")
    order = np.argsort(np.abs(grid), kind="stable")
    grid = grid[order]
    p0 = float(np.mean(x > decision_threshold))
    d0 = p0 >= certainty
    probs = [float(np.mean(x - b > decision_threshold)) for b in grid]
    hit = next((i for i, p in enumerate(probs) if (p >= certainty) != d0), None)
    return TippingPoint(
        threshold=float(decision_threshold),
        certainty=float(certainty),
        n_draws=int(x.size),
        bias_grid=tuple(float(b) for b in grid),
        probabilities=tuple(probs),
        decision_at_zero=bool(d0),
        probability_at_zero=p0,
        flipped=hit is not None,
        bias=float(grid[hit]) if hit is not None else None,
        probability=probs[hit] if hit is not None else None,
        interval=(
            summarize(x - grid[hit], definition=definition, mass=mass).interval
            if hit is not None
            else None
        ),
    )
