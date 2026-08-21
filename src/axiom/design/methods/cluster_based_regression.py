"""Cluster-based regression: post-period mean on treatment with the pre-period mean as covariate.

Each unit (cluster) contributes one row,

    ȳ_i^post = α + τ · D_i + β · ȳ_i^pre + ε_i,

and ``τ`` is the effect. Adjusting for the pre-period mean removes the
between-unit level variance that a plain difference in post means would carry
(the ANCOVA gain), and under random assignment ``τ̂`` is unbiased whatever
``β`` is. The standard error is the classical OLS one,

    se² = σ̂² · [(XᵀX)⁻¹]_{ττ},   σ̂² = RSS / (n − 3),   df = n − 3,

which is the right one when units are exchangeable and the residual variance
is common across arms; the two-arm Welch alternative is the ``ghost`` method's
business. The effect is an average per treated unit per post period because
the response is a per-period mean.

Ported from the parent's ``planning/methods`` cluster-based regression with
the domain-specific aggregation removed; see ``docs/plan/02-porting-ledger.md``.
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

__all__ = ["estimate", "estimate_arrays", "estimate_cluster_based_regression"]

METHOD = "cluster_based_regression"


def estimate_arrays(arrays: PanelArrays, mass: float) -> MethodEstimate | Unsupported:
    """OLS of post-period unit means on the treatment indicator and pre-period means."""
    n_pre = arrays.pre.stop - arrays.pre.start
    n_post = arrays.post.stop - arrays.post.start
    if n_pre < 1 or n_post < 1:
        return Unsupported(
            reason=f"{METHOD} needs non-empty pre and post windows; got {n_pre} pre, {n_post} post",
            missing=("pre_period",) if n_pre < 1 else ("post_period",),
        )
    n = arrays.n_units
    n_t = len(arrays.treated)
    n_c = n - n_t
    if n_t < 1 or n_c < 1:
        return Unsupported(
            reason=f"{METHOD} needs at least one treated and one control unit",
            missing=("treated",) if n_t < 1 else ("controls",),
        )
    df = n - 3
    if df < 1:
        return Unsupported(
            reason=f"{METHOD} needs at least 4 units for a residual degree of freedom; got {n}",
            missing=("units",),
        )
    y = arrays.outcome
    post_mean = y[:, arrays.post].mean(axis=1)
    pre_mean = y[:, arrays.pre].mean(axis=1)
    d = np.zeros(n)
    d[list(arrays.treated)] = 1.0
    x = np.column_stack([np.ones(n), d, pre_mean])
    rank = int(np.linalg.matrix_rank(x))
    if rank < 3:
        return Unsupported(
            reason=f"{METHOD}: design matrix is rank {rank} < 3 (constant pre-period mean?)",
            detail={"rank": str(rank)},
        )
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ post_mean
    resid = post_mean - x @ beta
    sigma2 = float(resid @ resid) / df
    se2 = sigma2 * float(xtx_inv[1, 1])
    if not np.isfinite(se2) or se2 <= 0.0:
        return Unsupported(
            reason=f"{METHOD}: residual variance is zero; no standard error exists",
            detail={"sigma2": repr(sigma2)},
        )
    effect = float(beta[1])
    se = float(np.sqrt(se2))
    return MethodEstimate(
        method=METHOD,
        effect=effect,
        se=se,
        interval=wald_t(effect, se, mass, df),
        n_treated=n_t,
        n_control=n_c,
        n_pre=n_pre,
        n_post=n_post,
        se_method="ols_classical",
        critical="student_t",
        detail={
            "df": float(df),
            "critical_value": t_critical(mass, df),
            "intercept": float(beta[0]),
            "pre_coefficient": float(beta[2]),
            "sigma2": sigma2,
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


estimate_cluster_based_regression = estimate
