"""Time-based regression: forecast the treated aggregate from the control aggregate.

Aggregate the treated units and the control units to two series, ``y_t``
(mean over treated units) and ``x_t`` (mean over control units). In the pre
period fit

    y_t = a + b · x_t + ε_t,   ε_t ~ N(0, σ²),

forecast ``ŷ_t = â + b̂ · x_t`` into the post period, and read the effect as the
mean post-period residual ``τ̂ = mean(y_t − ŷ_t)``. Because the effect is an
*average* over the ``n_post`` forecast points, its variance is the variance of
the mean forecast error, which carries both the noise and the parameter
uncertainty evaluated at the post-period mean regressor ``x̄ = (1, mean(x_post))``:

    se² = σ̂² · (1/n_post + x̄ᵀ (XᵀX)⁻¹ x̄),   σ̂² = RSS_pre / (n_pre − 2).

Summing the forecast errors one-by-one instead would ignore that they share
the same parameter error. The interval's critical value is Student-t at
``n_pre − 2`` degrees of freedom (``registry.wald_t``): with twelve pre
periods that is ``df = 10``, where the normal ``z`` turns a nominal 5 % A/A
rate into 10 %. The formula takes ``ε_t`` to be white; a common shock that is
serially correlated and not fully absorbed by ``b · x_t`` leaves correlated
residuals, and on the A/A panel in ``design.simulate`` (AR(1) shocks with
``rho = 0.5``) ``sd(τ̂) / rms(se)`` is 1.08 — the measured rate with the t
critical value is in the gate test. ``detail["cumulative_per_unit"]`` is
``n_post · τ̂`` (the usual TBR total) and ``detail["cumulative_total"]``
multiplies by the number of treated units.

Ported from the parent's ``planning/methods`` time-based regression
(Kerman, Wang & Vaver 2017) with the domain-specific aggregation removed; see
``docs/plan/02-porting-ledger.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from axiom.core import Unsupported
from axiom.design.methods.registry import (
    MethodEstimate,
    PanelArrays,
    panel_arrays,
    t_critical,
    wald_t,
)

__all__ = ["estimate", "estimate_arrays", "estimate_time_based_regression"]

METHOD = "time_based_regression"


def estimate_arrays(arrays: PanelArrays, mass: float) -> MethodEstimate | Unsupported:
    """Pre-period regression of the treated aggregate on the control aggregate, forecast post."""
    n_pre = arrays.pre.stop - arrays.pre.start
    n_post = arrays.post.stop - arrays.post.start
    if n_pre < 3:
        return Unsupported(
            reason=f"{METHOD} needs at least 3 pre periods for a residual variance; got {n_pre}",
            missing=("pre_period",),
        )
    if n_post < 1:
        return Unsupported(
            reason=f"{METHOD} needs a non-empty post window", missing=("post_period",)
        )
    treated = list(arrays.treated)
    control = list(arrays.control)
    if not treated or not control:
        return Unsupported(
            reason=f"{METHOD} needs at least one treated and one control unit",
            missing=("treated",) if not treated else ("controls",),
        )
    y_all = arrays.outcome[treated].mean(axis=0)
    x_all = arrays.outcome[control].mean(axis=0)
    x_pre, y_pre = x_all[arrays.pre], y_all[arrays.pre]
    x_post, y_post = x_all[arrays.post], y_all[arrays.post]
    if float(x_pre.var()) <= 0.0:
        return Unsupported(
            reason=f"{METHOD}: control aggregate is constant in the pre period; slope undefined",
            detail={"x_pre_var": repr(float(x_pre.var()))},
        )
    x_mat = np.column_stack([np.ones(n_pre), x_pre])
    xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
    beta = xtx_inv @ x_mat.T @ y_pre
    resid_pre = y_pre - x_mat @ beta
    df = n_pre - 2
    sigma2 = float(resid_pre @ resid_pre) / df
    # An exact linear relation leaves round-off, not a residual variance.
    if sigma2 <= 1e-20 * max(float(y_pre @ y_pre) / n_pre, np.finfo(float).tiny):
        sigma2 = 0.0
    forecast = beta[0] + beta[1] * x_post
    residual_post = y_post - forecast
    effect = float(residual_post.mean())
    xbar = np.array([1.0, float(x_post.mean())])
    se2 = sigma2 * (1.0 / n_post + float(xbar @ xtx_inv @ xbar))
    if not np.isfinite(se2) or se2 <= 0.0:
        return Unsupported(
            reason=f"{METHOD}: pre-period fit is exact (zero residual variance); no forecast error",
            detail={"sigma2": repr(sigma2)},
        )
    se = float(np.sqrt(se2))
    return MethodEstimate(
        method=METHOD,
        effect=effect,
        se=se,
        interval=wald_t(effect, se, mass, df),
        n_treated=len(treated),
        n_control=len(control),
        n_pre=n_pre,
        n_post=n_post,
        se_method="forecast_error_with_parameter_uncertainty",
        critical="student_t",
        detail={
            "df": float(df),
            "critical_value": t_critical(mass, df),
            "intercept": float(beta[0]),
            "slope": float(beta[1]),
            "sigma2": sigma2,
            "cumulative_per_unit": effect * n_post,
            "cumulative_total": effect * n_post * len(treated),
            "cumulative_se_per_unit": se * n_post,
        },
    )


def estimate(
    outcome: npt.ArrayLike,
    treated: Sequence[int],
    pre: slice,
    post: slice,
    *,
    mass: float = 0.95,
) -> MethodEstimate | Unsupported:
    """Keyword convenience: build the ``PanelArrays`` and run ``estimate_arrays``."""
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    return estimate_arrays(panel_arrays(outcome, treated, pre, post), mass)


estimate_time_based_regression = estimate
