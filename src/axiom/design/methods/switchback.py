"""Switchback: within-unit regression on a time-randomized on/off schedule with a HAC variance.

Each unit alternates between treatment on (``a_it = 1``) and off (``0``)
according to ``arrays.assignment``. With unit fixed effects,

    y_it = μ_i + τ · a_it + ε_it,

the within estimator demeans both sides by unit,

    τ̂ = Σ ã_it ỹ_it / Σ ã_it²,

which is the on-minus-off contrast pooled across units. Periods within a unit
are serially correlated and the schedule can carry over, so the variance is
heteroskedasticity-and-autocorrelation consistent (Newey–West with the
Bartlett kernel) computed unit by unit on the scores ``s_it = ã_it · e_it``:

    Var(τ̂) = (N·T / df) · Σ_i LRV_i / (Σ ã²)²,        df = N·T − N − 1,
    LRV_i = Σ_t s_it² + 2 Σ_{ℓ=1}^{L} (1 − ℓ/(L+1)) Σ_t s_it s_{i,t−ℓ},

with bandwidth ``L = max(1, floor(4 · (T/100)^(2/9)))`` — the Newey–West rule
of thumb — recorded in ``detail["bandwidth"]``. The leading factor is the
usual small-sample degrees-of-freedom correction: the residuals have lost
``N`` unit fixed effects and one slope, and without it the variance runs
about ``(N + 1) / (N·T)`` low (measured on the A/A panel in
``design.simulate``: 40 units × 12 periods, sd(τ̂) / rms(se) = 1.053 without
the factor and 1.007 with it). The interval's critical value is Student-t at
``df`` (``registry.wald_t``). The estimate is computed on the
``post`` window, which for a switchback is the whole schedule (set
``pre=slice(0, 0)``); a ``pre`` window, if given, is simply not used. All
units take part: a unit whose schedule never switches is demeaned to zero and
contributes nothing to the contrast.

Ported from the parent's ``planning/methods`` switchback estimator; the
block-bootstrap alternative mentioned in the registry description is not
ported. See ``docs/plan/02-porting-ledger.md``.
"""

from __future__ import annotations

import math
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

__all__ = ["bartlett_bandwidth", "estimate", "estimate_arrays", "estimate_switchback"]

METHOD = "switchback"


def bartlett_bandwidth(n_periods: int) -> int:
    """Newey–West rule of thumb ``max(1, floor(4 · (T/100)^(2/9)))``."""
    if n_periods < 1:
        raise ValueError("n_periods must be positive")
    return max(1, int(math.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0))))


def _bartlett_lrv(scores: npt.NDArray[np.float64], bandwidth: int) -> float:
    """Long-run variance of one unit's score series with the Bartlett kernel."""
    total = float(scores @ scores)
    for lag in range(1, bandwidth + 1):
        if lag >= scores.size:
            break
        weight = 1.0 - lag / (bandwidth + 1.0)
        total += 2.0 * weight * float(scores[lag:] @ scores[:-lag])
    return total


def estimate_arrays(arrays: PanelArrays, mass: float) -> MethodEstimate | Unsupported:
    """Within-unit on/off contrast with a Bartlett HAC standard error."""
    if arrays.assignment is None:
        return Unsupported(
            reason=f"{METHOD} needs an `assignment` schedule of on/off periods",
            missing=("assignment",),
        )
    n_pre = arrays.pre.stop - arrays.pre.start
    n_post = arrays.post.stop - arrays.post.start
    if n_post < 2:
        return Unsupported(
            reason=f"{METHOD} needs two or more periods in the schedule window; got {n_post}",
            missing=("post_period",),
        )
    y = arrays.outcome[:, arrays.post]
    a = arrays.assignment[:, arrays.post]
    a_dm = a - a.mean(axis=1, keepdims=True)
    y_dm = y - y.mean(axis=1, keepdims=True)
    denom = float((a_dm**2).sum())
    if denom <= 0.0:
        return Unsupported(
            reason=f"{METHOD}: the assignment never switches within any unit; no contrast exists",
            detail={"on_share": repr(float(a.mean()))},
        )
    effect = float((a_dm * y_dm).sum() / denom)
    resid = y_dm - effect * a_dm
    scores = a_dm * resid
    bandwidth = bartlett_bandwidth(n_post)
    lrv = sum(_bartlett_lrv(scores[i], bandwidth) for i in range(scores.shape[0]))
    n_units = arrays.n_units
    df = n_units * n_post - n_units - 1
    if df < 1:
        return Unsupported(
            reason=f"{METHOD}: no residual degree of freedom ({n_units} units x {n_post} periods)",
            missing=("post_period",),
        )
    # Residuals after the unit fixed effects and the slope are short by (n_units + 1) degrees
    # of freedom; without this factor the HAC variance runs about n_units / df low (HC1).
    dof_factor = (n_units * n_post) / df
    se2 = dof_factor * lrv / denom**2
    if not np.isfinite(se2) or se2 <= 0.0:
        return Unsupported(
            reason=f"{METHOD}: HAC variance is not positive (zero residuals or too few periods)",
            detail={"lrv": repr(lrv), "df": str(df)},
        )
    se = float(np.sqrt(se2))
    switching = int(((a_dm**2).sum(axis=1) > 0).sum())
    return MethodEstimate(
        method=METHOD,
        effect=effect,
        se=se,
        interval=wald_t(effect, se, mass, df),
        n_treated=switching,
        n_control=n_units - switching,
        n_pre=n_pre,
        n_post=n_post,
        se_method="hac_bartlett_within_unit",
        critical="student_t",
        detail={
            "df": float(df),
            "critical_value": t_critical(mass, df),
            "dof_factor": dof_factor,
            "bandwidth": float(bandwidth),
            "on_share": float(a.mean()),
            "n_switching_units": float(switching),
            "sigma2": float((resid**2).sum() / df),
        },
    )


def estimate(
    outcome: npt.ArrayLike,
    assignment: npt.ArrayLike,
    *,
    treated: Sequence[int] = (),
    pre: slice = slice(0, 0),
    post: slice = slice(None),
    mass: float = 0.95,
) -> MethodEstimate | Unsupported:
    """Keyword convenience: build ``PanelArrays`` with a schedule and run ``estimate_arrays``."""
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    return estimate_arrays(panel_arrays(outcome, treated, pre, post, assignment=assignment), mass)


estimate_switchback = estimate
