"""Ghost-exposure comparison: treated-and-exposed versus control-and-would-have-been-exposed.

When treatment reaches only the units that would have been exposed, comparing
all treated units to all controls dilutes the effect by the exposure rate. If
the exposure that a control unit *would* have received is recorded (its
"ghost" exposure), the comparison among exposed units on both arms recovers
the effect on the exposed without dilution:

    τ̂ = mean(ȳ_i^post : i treated, exposed) − mean(ȳ_i^post : i control, exposed),

with the two-sample Welch standard error

    se² = s²_T / n_T + s²_C / n_C

and the Welch–Satterthwaite degrees of freedom in ``detail["df"]``. The method
rests on ``exposure_known``: a ghost-exposure rate in control that differs
from the treated exposure rate is the tell that it is not.

Ported from the parent's ``planning/methods`` ghost estimator with the
domain-specific vocabulary removed; see ``docs/plan/02-porting-ledger.md``.
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

__all__ = ["estimate", "estimate_arrays", "estimate_ghost"]

METHOD = "ghost"


def estimate_arrays(arrays: PanelArrays, mass: float) -> MethodEstimate | Unsupported:
    """Post-period comparison of exposed treated units with ghost-exposed control units."""
    if arrays.exposed is None:
        return Unsupported(
            reason=f"{METHOD} needs an `exposed` mask (exposure, or ghost exposure in control)",
            missing=("exposed",),
        )
    n_pre = arrays.pre.stop - arrays.pre.start
    n_post = arrays.post.stop - arrays.post.start
    if n_post < 1:
        return Unsupported(
            reason=f"{METHOD} needs a non-empty post window", missing=("post_period",)
        )
    exposed = arrays.exposed > 0.5
    treated_mask = np.zeros(arrays.n_units, dtype=bool)
    treated_mask[list(arrays.treated)] = True
    post_mean = arrays.outcome[:, arrays.post].mean(axis=1)
    cell_t = post_mean[treated_mask & exposed]
    cell_c = post_mean[~treated_mask & exposed]
    n_t, n_c = int(cell_t.size), int(cell_c.size)
    if n_t < 2 or n_c < 2:
        return Unsupported(
            reason=(
                f"{METHOD} needs at least two exposed units on each arm; "
                f"got {n_t} treated-exposed and {n_c} control-exposed"
            ),
            missing=("exposed_treated",) if n_t < 2 else ("exposed_controls",),
            detail={"n_treated_exposed": str(n_t), "n_control_exposed": str(n_c)},
        )
    var_t = float(cell_t.var(ddof=1))
    var_c = float(cell_c.var(ddof=1))
    a, b = var_t / n_t, var_c / n_c
    se2 = a + b
    if not np.isfinite(se2) or se2 <= 0.0:
        return Unsupported(
            reason=f"{METHOD}: post-period means have zero variance within both cells",
            detail={"var_treated": repr(var_t), "var_control": repr(var_c)},
        )
    # Welch–Satterthwaite; a cell with zero variance contributes nothing to the denominator.
    denom = (a**2 / (n_t - 1) if a > 0 else 0.0) + (b**2 / (n_c - 1) if b > 0 else 0.0)
    df = se2**2 / denom
    effect = float(cell_t.mean() - cell_c.mean())
    se = float(np.sqrt(se2))
    n_treated_all = int(treated_mask.sum())
    n_control_all = arrays.n_units - n_treated_all
    return MethodEstimate(
        method=METHOD,
        effect=effect,
        se=se,
        interval=wald_t(effect, se, mass, df),
        n_treated=n_treated_all,
        n_control=n_control_all,
        n_pre=n_pre,
        n_post=n_post,
        se_method="two_sample_welch",
        critical="student_t",
        detail={
            "df": float(df),
            "critical_value": t_critical(mass, df),
            "n_treated_exposed": float(n_t),
            "n_control_exposed": float(n_c),
            "exposure_rate_treated": n_t / n_treated_all if n_treated_all else 0.0,
            "exposure_rate_control": n_c / n_control_all if n_control_all else 0.0,
            "var_treated": var_t,
            "var_control": var_c,
        },
    )


def estimate(
    outcome: npt.ArrayLike,
    treated: Sequence[int],
    post: slice,
    exposed: npt.ArrayLike,
    *,
    pre: slice = slice(0, 0),
    mass: float = 0.95,
) -> MethodEstimate | Unsupported:
    """Keyword convenience: build ``PanelArrays`` with ``exposed`` and run ``estimate_arrays``."""
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    return estimate_arrays(panel_arrays(outcome, treated, pre, post, exposed=exposed), mass)


estimate_ghost = estimate
