"""Durbin–Wu–Hausman tests for endogeneity of a treatment given instruments.

Two forms of the same question — "does OLS on ``x`` estimate the same thing
2SLS does?":

* ``durbin_wu_hausman`` — the control-function (augmented-regression) form
  (Hausman 1978; Wooldridge 2010, §6.3.1). Regress ``x`` on the instruments
  and covariates, keep the residual ``v``, then regress ``y`` on ``x``, the
  covariates, and ``v``. Under the null that ``x`` is exogenous the
  coefficient on ``v`` is zero; its t-statistic is the test. A rejection
  says OLS is inconsistent *if the instruments are valid* — the test cannot
  tell a bad instrument from an endogenous treatment.
* ``hausman_iv_vs_ols`` — the classic contrast (Hausman 1978):
  ``(b_iv - b_ols)^2 / (se_iv^2 - se_ols^2) ~ chi^2(1)`` under the null,
  using the fact that OLS is efficient under the null so the covariance of
  the difference is the difference of the variances. In finite samples that
  difference can be non-positive; the contrast is then undefined.

Both return an ``EndogeneityTest`` that records the ``alpha`` the conclusion
was drawn at, or an ``axiom.core.Unverified`` when the statistic cannot be
formed — never a fabricated statistic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import field_validator
from scipy import special as _sp

from axiom.core.result import Unverified
from axiom.core.spec import Spec
from axiom.identify.estimators import (
    LinearEstimate,
    _check_roles,
    _names,
    design_matrix,
    least_squares,
)

__all__ = ["EndogeneityTest", "durbin_wu_hausman", "hausman_iv_vs_ols"]

Conclusion = Literal["endogenous", "exogenous"]


class EndogeneityTest(Spec):
    """Outcome of an endogeneity test, with the level the conclusion used.

    ``statistic`` is a t-statistic for the control-function form (``df`` is
    its residual degrees of freedom) and a chi-squared statistic for the
    classic contrast (``df = 1``). ``conclusion`` is ``endogenous`` when
    ``p_value < alpha`` and ``exogenous`` otherwise. A test whose statistic
    cannot be formed is an ``Unverified``, not an ``EndogeneityTest``.
    """

    statistic: float
    p_value: float
    df: int
    method: str
    conclusion: Conclusion
    alpha: float = 0.05
    treatment: str = ""
    outcome: str = ""
    detail: dict[str, float] = {}

    @field_validator("alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {v}")
        return v

    @field_validator("p_value")
    @classmethod
    def _p_in_closed_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {v}")
        return v

    @field_validator("statistic")
    @classmethod
    def _statistic_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"statistic must be finite, got {v}")
        return v


def _conclude(p_value: float, alpha: float) -> Conclusion:
    return "endogenous" if p_value < alpha else "exogenous"


def _as_text(table: dict[str, float]) -> dict[str, str]:
    return {k: repr(float(v)) for k, v in table.items()}


def durbin_wu_hausman(
    frame: pd.DataFrame,
    y: str,
    x: str,
    instruments: str | Sequence[str],
    covariates: str | Sequence[str] = (),
    alpha: float = 0.05,
) -> EndogeneityTest | Unverified:
    """Control-function Durbin–Wu–Hausman test of the exogeneity of ``x``.

    Stage one: ``x`` on ``[1, instruments, covariates]``, residual ``v``.
    Stage two: ``y`` on ``[1, x, covariates, v]``; the t-statistic on ``v``
    against ``t(n - k)`` is the test (Wooldridge 2010, §6.3.1). ``detail``
    carries the coefficient and SE on ``v`` and the first-stage F, because a
    weak first stage makes the test uninformative. When the SE on ``v`` is
    not a positive finite number the statistic does not exist and the
    result is an ``Unverified`` carrying the same diagnostics.
    """
    instruments, covariates = _names(instruments), _names(covariates)
    if not instruments:
        raise ValueError("durbin_wu_hausman needs at least one instrument")
    _check_roles(y=[y], x=[x], instruments=instruments, covariates=covariates)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    first_design = design_matrix(frame, [*instruments, *covariates])
    treatment = design_matrix(frame, [x])[:, 1]
    target = design_matrix(frame, [y])[:, 1]
    first = least_squares(first_design, treatment)
    restricted = least_squares(design_matrix(frame, list(covariates)), treatment)
    q = len(instruments)
    f_stat = ((restricted.rss - first.rss) / q) / first.sigma2

    augmented = np.column_stack([design_matrix(frame, [x, *covariates]), first.residuals])
    fit = least_squares(augmented, target)
    coef_v = float(fit.beta[-1])
    se_v = float(fit.se[-1])
    detail = {"coefficient_v": coef_v, "se_v": se_v, "first_stage_f": float(f_stat)}
    if not (np.isfinite(se_v) and se_v > 0.0):
        return Unverified(
            reason=(
                "durbin_wu_hausman: the standard error on the first-stage residual is not a "
                f"positive finite number (se_v = {se_v}); the t-statistic does not exist"
            ),
            detail=_as_text(detail) | {"treatment": x, "outcome": y},
        )
    t_stat = coef_v / se_v
    p_value = float(2.0 * _sp.stdtr(fit.df_resid, -abs(t_stat)))
    return EndogeneityTest(
        statistic=float(t_stat),
        p_value=p_value,
        df=fit.df_resid,
        method="durbin_wu_hausman_control_function",
        conclusion=_conclude(p_value, alpha),
        alpha=alpha,
        treatment=x,
        outcome=y,
        detail=detail,
    )


def hausman_iv_vs_ols(
    est_ols: LinearEstimate, est_2sls: LinearEstimate, alpha: float = 0.05
) -> EndogeneityTest | Unverified:
    """Classic Hausman contrast of an OLS and a 2SLS estimate of the same effect.

    ``statistic = (b_iv - b_ols)^2 / (se_iv^2 - se_ols^2)`` against
    ``chi^2(1)`` (Hausman 1978). The two estimates must be of the same
    treatment on the same outcome with the same covariates and the same
    ``n`` — the contrast compares two estimators of one parameter on one
    sample, and anything else is a ``ValueError``. The denominator relies on
    OLS being efficient under the null; when it is not positive the contrast
    is undefined and the result is an ``Unverified`` saying why.
    """
    if est_ols.method != "ols" or est_2sls.method != "2sls":
        raise ValueError(
            f"hausman_iv_vs_ols needs an 'ols' and a '2sls' estimate, got "
            f"{est_ols.method!r} and {est_2sls.method!r}"
        )
    if (est_ols.treatment, est_ols.outcome) != (est_2sls.treatment, est_2sls.outcome):
        raise ValueError(
            "the two estimates must target the same treatment and outcome: "
            f"({est_ols.treatment!r}, {est_ols.outcome!r}) vs "
            f"({est_2sls.treatment!r}, {est_2sls.outcome!r})"
        )
    if est_ols.covariates != est_2sls.covariates:
        raise ValueError(
            "the two estimates must adjust for the same covariates: "
            f"{list(est_ols.covariates)} vs {list(est_2sls.covariates)}"
        )
    if est_ols.n != est_2sls.n:
        raise ValueError(
            f"the two estimates must come from the same sample: n = {est_ols.n} vs {est_2sls.n}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    difference = est_2sls.estimate - est_ols.estimate
    variance = est_2sls.se**2 - est_ols.se**2
    detail = {
        "difference": difference,
        "variance_difference": variance,
        "estimate_ols": est_ols.estimate,
        "estimate_2sls": est_2sls.estimate,
    }
    if not (np.isfinite(variance) and variance > 0.0):
        return Unverified(
            reason=(
                "hausman_iv_vs_ols: se_2sls^2 - se_ols^2 = "
                f"{variance:.6g} is not positive, so the contrast is undefined; OLS is not "
                "more precise than 2SLS in this sample, which the test assumes under the null"
            ),
            detail=_as_text(detail) | {"treatment": est_ols.treatment, "outcome": est_ols.outcome},
        )
    statistic = difference**2 / variance
    p_value = float(_sp.chdtrc(1.0, statistic))
    return EndogeneityTest(
        statistic=float(statistic),
        p_value=p_value,
        df=1,
        method="hausman_iv_vs_ols",
        conclusion=_conclude(p_value, alpha),
        alpha=alpha,
        treatment=est_ols.treatment,
        outcome=est_ols.outcome,
        detail=detail,
    )
