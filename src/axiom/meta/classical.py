"""Classical (sampler-free) meta-analysis: fixed effect, random effects with three
heterogeneity estimators, Q/I²/H², Knapp–Hartung intervals, and prediction intervals.

New code (roadmap Phase 7, ``meta/classical.py``); there is no parent module.
Imports only numpy, scipy, pydantic and ``axiom.core`` so that pooling works
on a core-only install (roadmap criterion 3).

Notation: ``k`` studies with estimates ``y_i`` and known sampling variances
``v_i = se_i²``. The random-effects model is ``y_i ~ N(θ_i, v_i)``,
``θ_i ~ N(μ, τ²)``.

* **Fixed effect.** ``w_i = 1/v_i``, ``μ̂ = Σ w_i y_i / Σ w_i``,
  ``se = (Σ w_i)^{-1/2}``.
* **Heterogeneity.** Cochran's ``Q = Σ w_i (y_i − μ̂_FE)²`` on ``k − 1`` df;
  ``I² = max(0, (Q − df)/Q)``; ``H² = Q/df`` (Higgins & Thompson 2002).
* **DerSimonian–Laird** (1986), closed form:
  ``τ² = max(0, (Q − df) / (Σ w_i − Σ w_i² / Σ w_i))``.
* **Paule–Mandel** (1982): the ``τ²`` at which the generalized
  ``Q(τ²) = Σ w_i(τ²) (y_i − μ̂(τ²))²`` with ``w_i(τ²) = 1/(v_i + τ²)``
  equals its expectation ``k − 1``; ``Q`` is decreasing in ``τ²``, so a
  unique root exists when ``Q(0) > k − 1`` and ``τ² = 0`` otherwise. Solved by
  ``scipy.optimize.brentq`` to ``xtol = 1e-12`` (absolute, on ``τ²``).
* **REML** (Harville 1977; Viechtbauer 2005): the restricted log-likelihood
  ``ℓ_R(τ²) = −½ [Σ log(v_i + τ²) + log Σ w_i(τ²) + Σ w_i(τ²)(y_i − μ̂(τ²))²]``
  maximized over ``τ² ≥ 0``. The score is ``∂ℓ_R/∂τ² = ½ [Σ w_i²(y_i − μ̂)² −
  Σ w_i + Σ w_i² / Σ w_i]``; when it is negative at zero the maximum is at the
  boundary ``τ² = 0``, else the root is bracketed and found by ``brentq`` to
  ``xtol = 1e-12``.
* **Random effects.** ``w_i* = 1/(v_i + τ²)``, ``μ̂ = Σ w_i* y_i / Σ w_i*``,
  ``se = (Σ w_i*)^{-1/2}``, interval ``μ̂ ± z·se`` (``wald``).
* **Knapp–Hartung** (2003): ``se_KH² = Σ w_i*(y_i − μ̂)² / ((k − 1) Σ w_i*)``
  and a ``t_{k−1}`` quantile in place of ``z``. This is the unmodified
  variant; it widens the interval whenever the scaled residual sum exceeds
  one, and can narrow it otherwise (``detail["knapp_hartung_scale"]`` says
  which happened).
* **Prediction interval** (Higgins, Thompson & Spiegelhalter 2009):
  ``μ̂ ± t_{k−2} · sqrt(τ² + se²)``; needs ``k ≥ 3``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator
from scipy import optimize, stats

from axiom.core import Interval, Spec, wald

__all__ = [
    "Heterogeneity",
    "PooledEstimate",
    "TauEstimate",
    "TauMethod",
    "fixed_effect",
    "heterogeneity",
    "prediction_interval",
    "random_effects",
    "reml_log_likelihood",
    "tau_dersimonian_laird",
    "tau_paule_mandel",
    "tau_reml",
]

TauMethod = Literal["dl", "pm", "reml"]
_XTOL = 1e-12
_Scalar = Callable[[float], float]
_MAXITER = 500


class Heterogeneity(Spec):
    """Cochran's ``Q`` on ``df = k − 1`` with its chi-square ``p``, ``I²`` and ``H²``."""

    q: float
    df: int
    p_value: float
    i2: float
    h2: float
    k: int

    @model_validator(mode="after")
    def _ranges(self) -> Heterogeneity:
        if self.q < 0 or not 0.0 <= self.i2 <= 1.0 or not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"Q={self.q}, I²={self.i2}, p={self.p_value} out of range")
        return self


class TauEstimate(Spec):
    """A between-study variance estimate with the record of how it was found.

    ``iterations`` is 0 for the closed form (DL) and the root-finder's
    function-evaluation count otherwise; ``converged`` is False only when an
    iterative method hit its limit, in which case the returned ``tau2`` is
    the last iterate and ``detail`` says so. ``truncated`` marks a negative
    moment estimate (DL) or a boundary solution (PM, REML) clipped to zero.
    """

    tau2: float
    method: TauMethod
    iterations: int
    converged: bool
    truncated: bool = False
    tolerance: float = _XTOL
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _nonneg(self) -> TauEstimate:
        if not (np.isfinite(self.tau2) and self.tau2 >= 0):
            raise ValueError(f"tau² must be finite and ≥ 0, got {self.tau2}")
        return self

    @property
    def tau(self) -> float:
        return float(np.sqrt(self.tau2))


class PooledEstimate(Spec):
    """The pooled mean, its standard error and interval, and everything that produced them.

    ``model`` is ``fixed`` or ``random``; ``tau2`` is zero for fixed effect
    and ``tau_method`` is ``"none"``. ``weights`` are the normalized weights
    (sum to one) in study order. ``interval`` is always ``wald``; under
    ``knapp_hartung`` the quantile is ``t_{k−1}`` rather than ``z`` and the
    se is the Knapp–Hartung se (``detail`` records both scale and quantile).
    """

    estimate: float
    se: float
    interval: Interval
    model: Literal["fixed", "random"]
    tau2: float
    tau_method: Literal["none", "dl", "pm", "reml"]
    k: int
    weights: tuple[float, ...]
    heterogeneity: Heterogeneity
    knapp_hartung: bool = False
    mass: float
    tau_estimate: TauEstimate | None = None
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> PooledEstimate:
        if len(self.weights) != self.k:
            raise ValueError(f"{len(self.weights)} weights for k={self.k}")
        if self.se <= 0 or not np.isfinite(self.se):
            raise ValueError(f"se must be finite and positive, got {self.se}")
        if self.tau2 < 0:
            raise ValueError(f"tau² must be ≥ 0, got {self.tau2}")
        if self.model == "fixed" and (self.tau2 != 0 or self.tau_method != "none"):
            raise ValueError("a fixed-effect estimate has tau²=0 and tau_method='none'")
        if self.interval.mass != self.mass:
            raise ValueError("interval mass must equal the estimate's mass")
        return self

    @property
    def tau(self) -> float:
        return float(np.sqrt(self.tau2))


# -- arrays -----------------------------------------------------------------------------


def _arrays(
    y: npt.ArrayLike, se: npt.ArrayLike, *, minimum: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ya = np.asarray(y, dtype=np.float64).ravel()
    sa = np.asarray(se, dtype=np.float64).ravel()
    if ya.shape != sa.shape:
        raise ValueError(f"y and se differ in length: {ya.size} vs {sa.size}")
    if ya.size < minimum:
        raise ValueError(f"need at least {minimum} studies, got {ya.size}")
    if not np.all(np.isfinite(ya)):
        raise ValueError("estimates must be finite")
    if not (np.all(np.isfinite(sa)) and np.all(sa > 0)):
        raise ValueError("standard errors must be finite and positive")
    return ya, sa


def _pooled(
    y: npt.NDArray[np.float64], v: npt.NDArray[np.float64], tau2: float
) -> tuple[npt.NDArray[np.float64], float]:
    w = 1.0 / (v + tau2)
    return w, float(np.sum(w * y) / np.sum(w))


def _generalized_q(y: npt.NDArray[np.float64], v: npt.NDArray[np.float64], tau2: float) -> float:
    w, mu = _pooled(y, v, tau2)
    return float(np.sum(w * (y - mu) ** 2))


# -- heterogeneity ----------------------------------------------------------------------


def heterogeneity(y: npt.ArrayLike, se: npt.ArrayLike) -> Heterogeneity:
    """Cochran's ``Q`` about the fixed-effect mean, with ``I²`` and ``H²``."""
    ya, sa = _arrays(y, se, minimum=2)
    v = sa**2
    k = ya.size
    df = k - 1
    q = _generalized_q(ya, v, 0.0)
    p = float(stats.chi2.sf(q, df))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return Heterogeneity(q=q, df=df, p_value=p, i2=float(i2), h2=float(q / df), k=k)


# -- tau² estimators --------------------------------------------------------------------


def tau_dersimonian_laird(y: npt.ArrayLike, se: npt.ArrayLike) -> TauEstimate:
    """DerSimonian–Laird closed form: ``(Q − df) / (Σw − Σw²/Σw)``, truncated at zero."""
    ya, sa = _arrays(y, se, minimum=2)
    w = 1.0 / sa**2
    df = ya.size - 1
    q = _generalized_q(ya, sa**2, 0.0)
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    raw = (q - df) / c
    return TauEstimate(
        tau2=max(0.0, float(raw)),
        method="dl",
        iterations=0,
        converged=True,
        truncated=bool(raw < 0),
        tolerance=0.0,
        detail={"q": repr(q), "c": repr(c), "raw": repr(float(raw))},
    )


def _bracket(f: _Scalar, v: npt.NDArray[np.float64]) -> tuple[float, float, int]:
    """Find ``hi`` with ``f(hi) < 0`` given ``f(0) > 0`` by doubling from the variance scale."""
    hi = float(np.max(v))
    evals = 0
    while f(hi) > 0:
        hi *= 2.0
        evals += 1
        if evals > 200:  # ~2^200 · max(v): the function has no root in any useful range
            raise RuntimeError("could not bracket tau²: the criterion never crossed zero")
    return 0.0, hi, evals


def _solve(
    f: _Scalar, v: npt.NDArray[np.float64], method: TauMethod, boundary_note: str
) -> TauEstimate:
    f0 = f(0.0)
    if f0 <= 0:
        return TauEstimate(
            tau2=0.0,
            method=method,
            iterations=1,
            converged=True,
            truncated=True,
            detail={"criterion_at_zero": repr(float(f0)), "note": boundary_note},
        )
    lo, hi, evals = _bracket(f, v)
    root, res = optimize.brentq(f, lo, hi, xtol=_XTOL, maxiter=_MAXITER, full_output=True)
    return TauEstimate(
        tau2=float(max(root, 0.0)),
        method=method,
        iterations=int(res.function_calls) + evals + 1,
        converged=bool(res.converged),
        truncated=False,
        detail={
            "bracket_upper": repr(hi),
            "criterion_at_root": repr(float(f(float(root)))),
            "criterion_at_zero": repr(float(f0)),
        },
    )


def tau_paule_mandel(y: npt.ArrayLike, se: npt.ArrayLike) -> TauEstimate:
    """Paule–Mandel: the ``τ²`` solving ``Q(τ²) = k − 1``; zero when ``Q(0) ≤ k − 1``."""
    ya, sa = _arrays(y, se, minimum=2)
    v = sa**2
    df = ya.size - 1

    def criterion(tau2: float) -> float:
        return _generalized_q(ya, v, tau2) - df

    return _solve(criterion, v, "pm", "Q(0) ≤ k − 1: no between-study variance needed")


def tau_reml(y: npt.ArrayLike, se: npt.ArrayLike) -> TauEstimate:
    """REML: the root of the restricted score ``Σw²(y−μ̂)² − Σw + Σw²/Σw`` on ``τ² ≥ 0``."""
    ya, sa = _arrays(y, se, minimum=2)
    v = sa**2

    def score(tau2: float) -> float:
        w, mu = _pooled(ya, v, tau2)
        return float(np.sum(w**2 * (ya - mu) ** 2) - np.sum(w) + np.sum(w**2) / np.sum(w))

    return _solve(score, v, "reml", "restricted score ≤ 0 at τ²=0: boundary maximum")


def reml_log_likelihood(y: npt.ArrayLike, se: npt.ArrayLike, tau2: float) -> float:
    """The restricted log-likelihood (up to a constant) at ``tau2``; for checking ``tau_reml``."""
    ya, sa = _arrays(y, se, minimum=2)
    v = sa**2
    w, mu = _pooled(ya, v, tau2)
    return float(-0.5 * (np.sum(np.log(v + tau2)) + np.log(np.sum(w)) + np.sum(w * (ya - mu) ** 2)))


def _tau(y: npt.ArrayLike, se: npt.ArrayLike, method: TauMethod) -> TauEstimate:
    if method == "dl":
        return tau_dersimonian_laird(y, se)
    if method == "pm":
        return tau_paule_mandel(y, se)
    if method == "reml":
        return tau_reml(y, se)
    raise ValueError(f"tau_method must be 'dl', 'pm' or 'reml', got {method!r}")


# -- pooling ----------------------------------------------------------------------------


def fixed_effect(y: npt.ArrayLike, se: npt.ArrayLike, *, mass: float = 0.95) -> PooledEstimate:
    """Inverse-variance fixed-effect pooling; the interval is ``μ̂ ± z·se`` (``wald``)."""
    ya, sa = _arrays(y, se, minimum=1)
    v = sa**2
    w, mu = _pooled(ya, v, 0.0)
    s = float(np.sqrt(1.0 / np.sum(w)))
    het = (
        heterogeneity(ya, sa)
        if ya.size >= 2
        else Heterogeneity(q=0.0, df=0, p_value=1.0, i2=0.0, h2=1.0, k=1)
    )
    return PooledEstimate(
        estimate=mu,
        se=s,
        interval=wald(mu, s, mass),
        model="fixed",
        tau2=0.0,
        tau_method="none",
        k=int(ya.size),
        weights=tuple(float(x) for x in w / np.sum(w)),
        heterogeneity=het,
        knapp_hartung=False,
        mass=mass,
        detail={"quantile": "z", "z": repr(float(stats.norm.ppf((1.0 + mass) / 2.0)))},
    )


def random_effects(
    y: npt.ArrayLike,
    se: npt.ArrayLike,
    *,
    tau_method: TauMethod = "dl",
    knapp_hartung: bool = False,
    mass: float = 0.95,
) -> PooledEstimate:
    """Random-effects pooling with ``τ²`` by DL, PM or REML; optional Knapp–Hartung interval.

    Weights are ``1/(v_i + τ²)``. Without Knapp–Hartung the interval is
    ``μ̂ ± z·se``; with it, ``se`` is replaced by the Knapp–Hartung se and the
    quantile by ``t_{k−1}``. Needs ``k ≥ 2``.
    """
    ya, sa = _arrays(y, se, minimum=2)
    v = sa**2
    tau = _tau(ya, sa, tau_method)
    w, mu = _pooled(ya, v, tau.tau2)
    sum_w = float(np.sum(w))
    s = float(np.sqrt(1.0 / sum_w))
    het = heterogeneity(ya, sa)
    k = int(ya.size)
    detail: dict[str, str] = {"tau_iterations": str(tau.iterations)}
    if knapp_hartung:
        scale = float(np.sum(w * (ya - mu) ** 2) / (k - 1))
        s_kh = float(np.sqrt(scale / sum_w))
        t = float(stats.t.ppf((1.0 + mass) / 2.0, k - 1))
        interval = Interval(lower=mu - t * s_kh, upper=mu + t * s_kh, definition="wald", mass=mass)
        s = s_kh
        detail.update(
            {
                "quantile": f"t_{k - 1}",
                "t": repr(t),
                "knapp_hartung_scale": repr(scale),
                "se_without_knapp_hartung": repr(float(np.sqrt(1.0 / sum_w))),
            }
        )
    else:
        interval = wald(mu, s, mass)
        detail.update({"quantile": "z", "z": repr(float(stats.norm.ppf((1.0 + mass) / 2.0)))})
    return PooledEstimate(
        estimate=mu,
        se=s,
        interval=interval,
        model="random",
        tau2=tau.tau2,
        tau_method=tau_method,
        k=k,
        weights=tuple(float(x) for x in w / sum_w),
        heterogeneity=het,
        knapp_hartung=knapp_hartung,
        mass=mass,
        tau_estimate=tau,
        detail=detail,
    )


def prediction_interval(pooled: PooledEstimate, mass: float | None = None) -> Interval:
    """``μ̂ ± t_{k−2} · sqrt(τ² + se²)``: where a new study's true effect is expected to lie.

    Higgins, Thompson & Spiegelhalter (2009). Uses the estimate's own
    ``se`` (the Knapp–Hartung one when that was requested). Needs ``k ≥ 3``.
    """
    if pooled.k < 3:
        raise ValueError(f"a prediction interval needs k ≥ 3 studies, got {pooled.k}")
    m = pooled.mass if mass is None else mass
    t = float(stats.t.ppf((1.0 + m) / 2.0, pooled.k - 2))
    half = t * float(np.sqrt(pooled.tau2 + pooled.se**2))
    return Interval(
        lower=pooled.estimate - half, upper=pooled.estimate + half, definition="wald", mass=m
    )
