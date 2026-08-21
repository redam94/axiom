"""Difference in differences on a unit-by-period panel.

Each unit's *change* is its post-period mean minus its pre-period mean,

    Δ_i = mean(y_i, post) − mean(y_i, pre).

The effect is the difference of mean changes, treated minus control,

    τ̂ = mean(Δ_T) − mean(Δ_C),

which is the two-by-two difference in differences written so that its standard
error is a two-sample variance of unit-level changes,

    se² = s²(Δ_T)/n_T + s²(Δ_C)/n_C,   df = n_T + n_C − 2.

Collapsing each unit's series to one change before comparing is the
Bertrand–Duflo–Mullainathan remedy for serial correlation within unit: the
standard error needs no model of the within-unit noise. With a single treated
unit its change has no sample variance; the control changes' variance is then
used for both arms under a common-variance assumption and ``detail["pooled"]``
records that. The quantity is an average effect per treated unit per post
period, the common currency of ``design.methods``.

Ported from the parent's ``planning/methods`` difference-in-differences
estimator with domain-specific aggregation removed; see
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

__all__ = ["estimate", "estimate_arrays", "estimate_difference_in_differences"]

METHOD = "difference_in_differences"


def estimate_arrays(arrays: PanelArrays, mass: float) -> MethodEstimate | Unsupported:
    """Difference in differences of unit-level pre-to-post changes on ``arrays``."""
    n_pre = arrays.pre.stop - arrays.pre.start
    n_post = arrays.post.stop - arrays.post.start
    if n_pre < 1 or n_post < 1:
        return Unsupported(
            reason=f"{METHOD} needs non-empty pre and post windows; got {n_pre} pre, {n_post} post",
            missing=("pre_period",) if n_pre < 1 else ("post_period",),
        )
    treated = list(arrays.treated)
    control = list(arrays.control)
    if not treated or not control:
        return Unsupported(
            reason=f"{METHOD} needs at least one treated and one control unit",
            missing=("treated",) if not treated else ("controls",),
        )
    y = arrays.outcome
    change = y[:, arrays.post].mean(axis=1) - y[:, arrays.pre].mean(axis=1)
    d_t = change[treated]
    d_c = change[control]
    n_t, n_c = len(treated), len(control)
    if n_c < 2:
        return Unsupported(
            reason=f"{METHOD} needs at least two control units for a variance; got {n_c}",
            missing=("controls",),
        )
    var_c = float(d_c.var(ddof=1))
    pooled = n_t < 2
    var_t = var_c if pooled else float(d_t.var(ddof=1))
    se2 = var_t / n_t + var_c / n_c
    if not np.isfinite(se2) or se2 <= 0.0:
        return Unsupported(
            reason=f"{METHOD}: unit-level changes have zero variance; no standard error exists",
            detail={"var_treated": repr(var_t), "var_control": repr(var_c)},
        )
    effect = float(d_t.mean() - d_c.mean())
    se = float(np.sqrt(se2))
    df = n_t + n_c - 2
    return MethodEstimate(
        method=METHOD,
        effect=effect,
        se=se,
        interval=wald_t(effect, se, mass, df),
        n_treated=n_t,
        n_control=n_c,
        n_pre=n_pre,
        n_post=n_post,
        se_method="two_sample_unit_changes",
        critical="student_t",
        detail={
            "df": float(df),
            "critical_value": t_critical(mass, df),
            "pooled": 1.0 if pooled else 0.0,
            "mean_change_treated": float(d_t.mean()),
            "mean_change_control": float(d_c.mean()),
            "var_change_treated": var_t,
            "var_change_control": var_c,
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


estimate_difference_in_differences = estimate
