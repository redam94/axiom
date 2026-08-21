"""Influence and small-study diagnostics for a set of pooled study estimates.

New in axiom (roadmap M7.5; no parent module). Everything here is computed
from ``(y_i, se_i)`` pairs and the pooled estimate produced by
``axiom.meta.classical``; nothing re-implements the pooling.

* **Leave-one-out** — re-pool without each study; the influence of study
  ``i`` is ``(μ̂ − μ̂₍₋ᵢ₎) / se(μ̂)``, the change in the pooled estimate in
  units of its own standard error (Viechtbauer & Cheung 2010, DFBETAS-style).
* **Egger's test** — the regression ``z_i = b₀ + b₁ · (1/se_i) + ε_i`` with
  ``z_i = y_i / se_i`` (Egger et al. 1997). This is the weighted (``1/se²``)
  regression of ``y`` on ``se``; the intercept ``b₀`` is the asymmetry
  statistic, tested with ``t = b₀ / se(b₀)`` on ``k − 2`` degrees of freedom.
* **Funnel contours** — the pseudo-confidence region ``μ̂ ± z_m · se`` for a
  grid of standard errors at masses 0.9 / 0.95 / 0.99 (Sterne & Egger 2001).
* **Forest data** — per-study Wald intervals, the pooled interval, and the
  Higgins–Thompson–Spiegelhalter prediction interval from ``classical``.
* **Baujat plot data** — per-study contribution to ``Q`` against its
  influence on the pooled estimate (Baujat et al. 2002).

Every interval is a ``core.Interval`` carrying its definition and mass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import field_validator, model_validator
from scipy import stats as _st

from axiom.core import Interval, NonEmptyStr, Spec, wald

__all__ = [
    "BaujatData",
    "EggerTest",
    "ForestData",
    "ForestRow",
    "FunnelContour",
    "FunnelData",
    "LeaveOneOut",
    "PoolMethod",
    "baujat",
    "egger",
    "forest_data",
    "funnel_data",
    "leave_one_out",
]

PoolMethod = Literal["fe", "dl", "pm", "reml"]
"""``"fe"`` is the fixed-effect model; the others are the random-effects ``tau`` methods."""


# -- helpers ---------------------------------------------------------------------------------


def _arrays(
    y: npt.ArrayLike, se: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ya = np.asarray(y, dtype=np.float64).ravel()
    sa = np.asarray(se, dtype=np.float64).ravel()
    if ya.shape != sa.shape:
        raise ValueError(f"y and se must have the same length, got {ya.size} and {sa.size}")
    if ya.size == 0:
        raise ValueError("need at least one study")
    if not np.all(np.isfinite(ya)):
        raise ValueError("y must be finite")
    if not (np.all(np.isfinite(sa)) and np.all(sa > 0)):
        raise ValueError("se must be finite and positive")
    return ya, sa


def _mass(v: float) -> float:
    if not 0.0 < v < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {v}")
    return float(v)


def _pool(
    y: npt.NDArray[np.float64], se: npt.NDArray[np.float64], method: PoolMethod, mass: float
) -> Any:
    """Pool through ``axiom.meta.classical`` (one implementation of the pooling; never copied)."""
    from axiom.meta import classical

    if method == "fe":
        return classical.fixed_effect(y, se, mass=mass)
    return classical.random_effects(y, se, tau_method=method, mass=mass)


def _split_pooled(pooled: Any) -> tuple[float, float | None, float | None]:
    """``(estimate, se, tau2)`` from a ``PooledEstimate`` or a bare float."""
    if isinstance(pooled, int | float | np.floating):
        return float(pooled), None, None
    est = getattr(pooled, "estimate", None)
    if est is None:
        raise TypeError("pooled must be a PooledEstimate or a float")
    se = getattr(pooled, "se", None)
    tau2 = getattr(pooled, "tau2", None)
    return float(est), (None if se is None else float(se)), (None if tau2 is None else float(tau2))


# -- leave-one-out ---------------------------------------------------------------------------


class LeaveOneOut(Spec):
    """Pooled estimates with each study removed, and each study's influence.

    ``estimates[i]``/``ses[i]``/``intervals[i]`` describe the pool without
    study ``i``; ``influence[i] = (full_estimate − estimates[i]) / full_se``.
    ``tau2s`` is the between-study variance of each reduced pool (all zero for
    ``method="fe"``).
    """

    method: PoolMethod
    k: int
    mass: float
    full_estimate: float
    full_se: float
    full_interval: Interval
    estimates: tuple[float, ...]
    ses: tuple[float, ...]
    intervals: tuple[Interval, ...]
    influence: tuple[float, ...]
    tau2s: tuple[float, ...]
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> LeaveOneOut:
        n = len(self.estimates)
        if self.k != n or n < 1:
            raise ValueError(f"k={self.k} must equal the number of reduced pools ({n}), ≥ 1")
        for name in ("ses", "intervals", "influence", "tau2s"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have length {n}")
        return self


def leave_one_out(
    y: npt.ArrayLike, se: npt.ArrayLike, *, method: PoolMethod = "dl", mass: float = 0.95
) -> LeaveOneOut:
    """Re-pool without each study; influence is the shift in pooled-``se`` units.

    Needs ``k ≥ 3`` so that every reduced pool still has two studies.
    """
    ya, sa = _arrays(y, se)
    mass = _mass(mass)
    k = ya.size
    if k < 3:
        raise ValueError(f"leave-one-out needs at least 3 studies, got {k}")
    full = _pool(ya, sa, method, mass)
    full_est, full_se = float(full.estimate), float(full.se)
    ests: list[float] = []
    ses: list[float] = []
    ivs: list[Interval] = []
    infl: list[float] = []
    tau2s: list[float] = []
    for i in range(k):
        keep = np.arange(k) != i
        p = _pool(ya[keep], sa[keep], method, mass)
        e, s = float(p.estimate), float(p.se)
        ests.append(e)
        ses.append(s)
        ivs.append(wald(e, s, mass))
        infl.append((full_est - e) / full_se)
        tau2s.append(float(getattr(p, "tau2", 0.0) or 0.0))
    return LeaveOneOut(
        method=method,
        k=k,
        mass=mass,
        full_estimate=full_est,
        full_se=full_se,
        full_interval=wald(full_est, full_se, mass),
        estimates=tuple(ests),
        ses=tuple(ses),
        intervals=tuple(ivs),
        influence=tuple(infl),
        tau2s=tuple(tau2s),
        detail={"influence": "(full_estimate - estimate_without_i) / full_se"},
    )


# -- Egger ------------------------------------------------------------------------------------


class EggerTest(Spec):
    """Egger's regression test for funnel asymmetry.

    ``intercept`` is ``b₀`` in ``y_i/se_i = b₀ + b₁/se_i + ε``; ``slope`` is
    ``b₁`` (the bias-adjusted pooled effect). ``t = intercept / se`` on
    ``df = k − 2``; ``p`` is two-sided. ``interval`` is ``intercept ± t_df · se``
    at ``mass`` (labelled ``wald``: estimate ± critical value · se).
    """

    k: int
    df: int
    intercept: float
    se: float
    t: float
    p: float
    slope: float
    slope_se: float
    mass: float
    interval: Interval
    detail: dict[str, str] = {}

    @field_validator("p")
    @classmethod
    def _p_in_unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _df(self) -> EggerTest:
        if self.df != self.k - 2 or self.df < 1:
            raise ValueError("df must equal k - 2 and be at least 1")
        return self


def egger(y: npt.ArrayLike, se: npt.ArrayLike, *, mass: float = 0.95) -> EggerTest:
    """Egger et al. (1997): OLS of ``z = y/se`` on precision ``1/se`` with an intercept."""
    ya, sa = _arrays(y, se)
    mass = _mass(mass)
    k = ya.size
    if k < 3:
        raise ValueError(f"Egger's test needs at least 3 studies, got {k}")
    z = ya / sa
    prec = 1.0 / sa
    x = np.column_stack([np.ones(k), prec])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ z
    resid = z - x @ beta
    df = k - 2
    s2 = float(resid @ resid) / df
    cov = s2 * xtx_inv
    b0, b1 = float(beta[0]), float(beta[1])
    se_b0, se_b1 = math.sqrt(float(cov[0, 0])), math.sqrt(float(cov[1, 1]))
    t = b0 / se_b0 if se_b0 > 0 else math.copysign(math.inf, b0) if b0 != 0 else 0.0
    p = float(2.0 * _st.t.sf(abs(t), df)) if math.isfinite(t) else 0.0
    crit = float(_st.t.ppf((1.0 + mass) / 2.0, df))
    return EggerTest(
        k=k,
        df=df,
        intercept=b0,
        se=se_b0,
        t=float(t),
        p=p,
        slope=b1,
        slope_se=se_b1,
        mass=mass,
        interval=Interval(
            lower=b0 - crit * se_b0, upper=b0 + crit * se_b0, definition="wald", mass=mass
        ),
        detail={"model": "y/se = b0 + b1 * (1/se) + e; t = b0/se(b0) on k-2 df"},
    )


# -- funnel -----------------------------------------------------------------------------------


class FunnelContour(Spec):
    """``pooled ± z_mass · se`` evaluated on a grid of standard errors."""

    mass: float
    se: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @model_validator(mode="after")
    def _lengths(self) -> FunnelContour:
        if not len(self.se) == len(self.lower) == len(self.upper):
            raise ValueError("se, lower and upper must have the same length")
        return self


class FunnelData(Spec):
    """Study points (``y`` against ``se``) with pseudo-confidence contours around ``pooled``."""

    k: int
    y: tuple[float, ...]
    se: tuple[float, ...]
    precision: tuple[float, ...]
    pooled: float
    contours: tuple[FunnelContour, ...]
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _lengths(self) -> FunnelData:
        if not len(self.y) == len(self.se) == len(self.precision) == self.k:
            raise ValueError("y, se and precision must have length k")
        return self


def funnel_data(
    y: npt.ArrayLike,
    se: npt.ArrayLike,
    pooled: Any,
    *,
    masses: Sequence[float] = (0.9, 0.95, 0.99),
    n_grid: int = 50,
) -> FunnelData:
    """Points and closed-form contours ``pooled ± z_m · se`` for ``se`` from 0 to ``max(se)``."""
    ya, sa = _arrays(y, se)
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")
    est, _, _ = _split_pooled(pooled)
    grid = np.linspace(0.0, float(sa.max()), n_grid)
    contours = []
    for m in masses:
        m = _mass(m)
        zc = float(_st.norm.ppf((1.0 + m) / 2.0))
        contours.append(
            FunnelContour(
                mass=m,
                se=tuple(grid.tolist()),
                lower=tuple((est - zc * grid).tolist()),
                upper=tuple((est + zc * grid).tolist()),
            )
        )
    return FunnelData(
        k=ya.size,
        y=tuple(ya.tolist()),
        se=tuple(sa.tolist()),
        precision=tuple((1.0 / sa).tolist()),
        pooled=est,
        contours=tuple(contours),
        detail={"contour": "pooled ± z_mass * se"},
    )


# -- forest -----------------------------------------------------------------------------------


class ForestRow(Spec):
    """One study on a forest plot: its estimate, Wald interval and pooling weight (share)."""

    label: NonEmptyStr
    estimate: float
    se: float
    interval: Interval
    weight: float | None = None


class ForestData(Spec):
    """Per-study rows, the pooled row, and the prediction interval when available."""

    rows: tuple[ForestRow, ...]
    mass: float
    pooled_estimate: float
    pooled_se: float | None
    pooled_interval: Interval | None
    prediction_interval: Interval | None
    tau2: float | None
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _non_empty(self) -> ForestData:
        if not self.rows:
            raise ValueError("a forest needs at least one row")
        return self


def _from_corpus(obj: Any) -> tuple[list[str], list[float], list[float]] | None:
    records = getattr(obj, "records", None)
    if records is None:
        return None
    labels = [str(r.study) for r in records]
    return labels, [float(r.estimate) for r in records], [float(r.se) for r in records]


def forest_data(
    y_or_corpus: Any,
    se: npt.ArrayLike | None,
    pooled: Any,
    *,
    labels: Sequence[str] | None = None,
    mass: float = 0.95,
) -> ForestData:
    """Forest-plot data from ``(y, se)`` arrays or a ``Corpus`` (``se=None``) plus the pooled fit.

    When ``pooled`` is a ``PooledEstimate`` its weights are reported per row
    and the prediction interval is taken from ``classical.prediction_interval``
    (``k ≥ 3`` and a ``tau2``); when it is a bare float only the pooled point is
    shown.
    """
    mass = _mass(mass)
    corpus = _from_corpus(y_or_corpus) if se is None else None
    if corpus is not None:
        labels_l, y_l, se_l = corpus
        ya, sa = _arrays(y_l, se_l)
        if labels is None:
            labels = labels_l
    else:
        if se is None:
            raise ValueError("se is required unless the first argument is a Corpus")
        ya, sa = _arrays(y_or_corpus, se)
    k = ya.size
    if labels is None:
        labels = [f"study_{i + 1}" for i in range(k)]
    if len(labels) != k:
        raise ValueError(f"labels must have length {k}")
    est, pse, tau2 = _split_pooled(pooled)
    weights = getattr(pooled, "weights", None)
    w_list: list[float | None] = [None] * k
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size == k and w.sum() > 0:
            w_list = [float(v) for v in (w / w.sum())]
    rows = tuple(
        ForestRow(
            label=str(lab),
            estimate=float(ya[i]),
            se=float(sa[i]),
            interval=wald(float(ya[i]), float(sa[i]), mass),
            weight=w_list[i],
        )
        for i, lab in enumerate(labels)
    )
    pred: Interval | None = None
    if pse is not None and tau2 is not None and k >= 3:
        from axiom.meta import classical

        pred = classical.prediction_interval(pooled, mass)
    return ForestData(
        rows=rows,
        mass=mass,
        pooled_estimate=est,
        pooled_se=pse,
        pooled_interval=None if pse is None else wald(est, pse, mass),
        prediction_interval=pred,
        tau2=tau2,
        detail={"weights": "normalized to sum to one"} if weights is not None else {},
    )


# -- Baujat -----------------------------------------------------------------------------------


class BaujatData(Spec):
    """Per-study ``Q`` contribution (x) against influence on the pooled estimate (y)."""

    k: int
    q_contribution: tuple[float, ...]
    influence: tuple[float, ...]
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _lengths(self) -> BaujatData:
        if not len(self.q_contribution) == len(self.influence) == self.k:
            raise ValueError("q_contribution and influence must have length k")
        return self


def baujat(y: npt.ArrayLike, se: npt.ArrayLike) -> BaujatData:
    """Baujat et al. (2002) under fixed-effect pooling.

    ``x_i = w_i (y_i − μ̂)²`` (contribution to ``Q``) and
    ``y_i = (μ̂ − μ̂₍₋ᵢ₎)² / var(μ̂₍₋ᵢ₎)`` (influence on the pooled estimate).
    """
    ya, sa = _arrays(y, se)
    k = ya.size
    if k < 3:
        raise ValueError(f"Baujat needs at least 3 studies, got {k}")
    full = _pool(ya, sa, "fe", 0.95)
    mu = float(full.estimate)
    w = 1.0 / sa**2
    qc = w * (ya - mu) ** 2
    infl = []
    for i in range(k):
        keep = np.arange(k) != i
        p = _pool(ya[keep], sa[keep], "fe", 0.95)
        infl.append((mu - float(p.estimate)) ** 2 / float(p.se) ** 2)
    return BaujatData(
        k=k,
        q_contribution=tuple(qc.tolist()),
        influence=tuple(infl),
        detail={"model": "fixed effect"},
    )
