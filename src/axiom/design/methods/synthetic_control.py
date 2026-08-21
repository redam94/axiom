"""Synthetic control: a simplex-weighted donor combination fit on the pre period.

For a treated unit with pre-period series ``y_pre`` and donor matrix ``Y_pre``
(one row per control unit) find weights on the simplex,

    w* = argmin ‖y_pre − Y_preᵀ w‖²   s.t.  w ≥ 0,  Σ w = 1,

(Abadie, Diamond & Hainmueller 2010) by SLSQP, then read the post-period gap
``g_t = y_t − Y_tᵀ w*``; the treated unit's effect is ``mean(g_post)`` and,
with several treated units, the effect is the mean of their per-unit effects
(each fit separately against the same donor pool).

The standard error comes from the placebo-in-space permutation (Abadie et
al. 2010) with the donor-noise correlation among treated gaps corrected
explicitly. Each donor ``j`` is in turn fit on the other donors and its post
gap ``g_j`` recorded. Writing ``ε`` for a unit's post-period noise net of its
pre-period fit, a gap is ``g = ε_unit − Σ_k w_k ε_k``, so if unit noise is
independent with variance ``σ²`` then ``Var(g_j) = σ² (1 + ‖w_j‖²)`` and the
placebos give

    σ̂² = Σ_j (g_j − ḡ)² / Σ_j (1 + ‖w_j‖²) · m / (m − 1),

over the ``m`` placebos. The treated mean gap uses the *average* weight vector
``w̄`` (each treated unit is fit separately on the same pool, and the mean of
simplex vectors is a simplex vector), so its variance is

    Var(effect) = σ̂² (1 / n_treated + ‖w̄‖²):

``1 / n_treated`` is the treated units' own noise and ``‖w̄‖²`` the shared
donor noise — with uniform weights it is ``1 / n_donors`` and the formula is
the difference-in-differences variance. Dividing the single-unit placebo sd
by ``sqrt(n_treated)`` would drop the shared term and, on the A/A panel in
``design.simulate`` (20 treated, 20 donors), report an SE 1.4× too small;
treating same-size donor subsets as pseudo-treated sets would leave them a
pool too small to match (20 treated of 40 leaves one donor per pseudo-set)
and report one 6× too large. With the correction, sd(effect) / rms(se) is
1.006 on that panel. The critical value is Student-t at ``m − 1`` degrees of
freedom (the placebo gaps are correlated, so this understates the loss of
information slightly; the measured A/A rate is in the gate test).
``detail["placebo_rank"]`` is the permutation p-value itself. A poorly
fitting placebo (pre-period RMSPE far above the treated units') inflates the
spread for no reason; ``estimate`` takes ``rmspe_ratio_max`` to drop placebos
whose pre-RMSPE exceeds that multiple of the treated units', and
``detail["rmspe_ratio_max"]`` / ``detail["n_placebos_dropped"]`` record the
filter. ``estimate_arrays`` does not filter.

Ported from the parent's ``planning/methods`` synthetic control with the
domain-specific aggregation removed; see ``docs/plan/02-porting-ledger.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import optimize

from axiom.core import Unsupported
from axiom.design.methods.registry import (
    MethodEstimate,
    PanelArrays,
    panel_arrays,
    t_critical,
    wald_t,
)

__all__ = ["estimate", "estimate_arrays", "estimate_synthetic_control", "simplex_weights"]

METHOD = "synthetic_control"
Array = npt.NDArray[np.float64]


def simplex_weights(target: Array, donors: Array) -> Array:
    """Non-negative weights summing to one that best reproduce ``target`` from ``donors`` rows.

    Solved with SLSQP from the uniform start; a single donor gets weight one.
    Raises ``RuntimeError`` if the solver reports failure (never silently
    returns a bad fit).
    """
    k = donors.shape[0]
    if k == 1:
        return np.ones(1)
    scale = float(np.abs(target).max())
    scale = scale if scale > 0 else 1.0
    t, d = target / scale, donors / scale

    def loss(w: npt.NDArray[np.float64]) -> float:
        r = t - d.T @ w
        return float(r @ r)

    def grad(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(-2.0 * d @ (t - d.T @ w), dtype=np.float64)

    res = optimize.minimize(
        loss,
        np.full(k, 1.0 / k, dtype=np.float64),
        jac=grad,
        method="SLSQP",
        bounds=optimize.Bounds(np.zeros(k), np.ones(k)),
        constraints=optimize.LinearConstraint(np.ones((1, k)), 1.0, 1.0),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"simplex weight solver failed: {res.message}")
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, None)
    return np.asarray(w / w.sum(), dtype=np.float64)


def _fit_gap(target: Array, donors: Array, pre: slice, post: slice) -> tuple[Array, float, Array]:
    """Post-period gap series, pre-period RMSPE, and weights of one unit against a donor pool."""
    w = simplex_weights(target[pre], donors[:, pre])
    synthetic = donors.T @ w
    rmspe = float(np.sqrt(np.mean((target[pre] - synthetic[pre]) ** 2)))
    return np.asarray(target[post] - synthetic[post], dtype=np.float64), rmspe, w


def _estimate(
    arrays: PanelArrays, mass: float, rmspe_ratio_max: float | None
) -> MethodEstimate | Unsupported:
    n_pre = arrays.pre.stop - arrays.pre.start
    n_post = arrays.post.stop - arrays.post.start
    if n_pre < 2:
        return Unsupported(
            reason=f"{METHOD} needs at least 2 pre periods to fit weights; got {n_pre}",
            missing=("pre_period",),
        )
    if n_post < 1:
        return Unsupported(
            reason=f"{METHOD} needs a non-empty post window", missing=("post_period",)
        )
    treated = list(arrays.treated)
    donors = list(arrays.control)
    n_t, n_d = len(treated), len(donors)
    if not treated:
        return Unsupported(reason=f"{METHOD} needs at least one treated unit", missing=("treated",))
    if n_d < 2:
        return Unsupported(
            reason=f"{METHOD} needs at least 2 donors for placebo gaps; got {n_d}",
            missing=("controls",),
        )
    y = arrays.outcome
    pool = y[donors]
    gaps: list[float] = []
    rmspes: list[float] = []
    weights: list[Array] = []
    for i in treated:
        g, r, w = _fit_gap(y[i], pool, arrays.pre, arrays.post)
        gaps.append(float(g.mean()))
        rmspes.append(r)
        weights.append(w)
    effect = float(np.mean(gaps))
    treated_rmspe = float(np.mean(rmspes))
    w_bar = np.mean(np.asarray(weights), axis=0)
    overlap = float(w_bar @ w_bar)

    # Placebo in space: each donor fit on the other donors; its gap's variance is
    # sigma^2 (1 + |w|^2), which lets the placebos estimate the per-unit noise sigma^2.
    placebo_gaps: list[float] = []
    placebo_rmspes: list[float] = []
    placebo_norms: list[float] = []
    for j in range(n_d):
        others = np.delete(pool, j, axis=0)
        g, r, w = _fit_gap(pool[j], others, arrays.pre, arrays.post)
        placebo_gaps.append(float(g.mean()))
        placebo_rmspes.append(r)
        placebo_norms.append(1.0 + float(w @ w))
    pg = np.asarray(placebo_gaps)
    pr = np.asarray(placebo_rmspes)
    pn = np.asarray(placebo_norms)
    dropped = 0
    if rmspe_ratio_max is not None and treated_rmspe > 0:
        keep = pr <= rmspe_ratio_max * treated_rmspe
        dropped = int((~keep).sum())
        if keep.sum() >= 2:
            pg, pn = pg[keep], pn[keep]
        else:
            dropped = 0  # The filter would leave no distribution; fall back to all placebos.
    m = int(pg.size)
    if m < 2:
        return Unsupported(
            reason=f"{METHOD}: fewer than 2 placebo gaps; no permutation distribution",
            detail={"n_placebos": str(m)},
        )
    sd = float(pg.std(ddof=1))
    sigma2 = float(((pg - pg.mean()) ** 2).sum() / pn.sum()) * m / (m - 1)
    se2 = sigma2 * (1.0 / n_t + overlap)
    if not math.isfinite(se2) or se2 <= 0.0:
        return Unsupported(
            reason=f"{METHOD}: placebo gaps have zero spread; no standard error exists",
            detail={"placebo_sd": repr(sd)},
        )
    se = math.sqrt(se2)
    df = m - 1
    return MethodEstimate(
        method=METHOD,
        effect=effect,
        se=se,
        interval=wald_t(effect, se, mass, df),
        n_treated=n_t,
        n_control=n_d,
        n_pre=n_pre,
        n_post=n_post,
        se_method="placebo_in_space_weight_corrected",
        critical="student_t",
        detail={
            "df": float(df),
            "critical_value": t_critical(mass, df),
            "n_placebos": float(m),
            "n_placebos_dropped": float(dropped),
            "rmspe_ratio_max": float(rmspe_ratio_max) if rmspe_ratio_max is not None else math.inf,
            "pre_rmspe": treated_rmspe,
            "placebo_pre_rmspe_median": float(np.median(pr)),
            "placebo_gap_sd": sd,
            "unit_noise_variance": sigma2,
            "weight_overlap": overlap,
            "placebo_weight_norm_mean": float(pn.mean()),
            "placebo_rank": float(np.mean(np.abs(pg) >= abs(effect))),
        },
    )


def estimate_arrays(arrays: PanelArrays, mass: float) -> MethodEstimate | Unsupported:
    """Synthetic control with the unfiltered placebo-in-space standard error."""
    return _estimate(arrays, mass, None)


def estimate(
    outcome: npt.ArrayLike,
    treated: Sequence[int],
    pre: slice,
    post: slice,
    *,
    mass: float = 0.95,
    rmspe_ratio_max: float | None = None,
) -> MethodEstimate | Unsupported:
    """Keyword convenience with the optional RMSPE-ratio placebo filter."""
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    if rmspe_ratio_max is not None and not rmspe_ratio_max > 0:
        raise ValueError("rmspe_ratio_max must be positive")
    return _estimate(panel_arrays(outcome, treated, pre, post), mass, rmspe_ratio_max)


estimate_synthetic_control = estimate
