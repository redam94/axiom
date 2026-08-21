"""Linear estimators for the identification routes: OLS, 2SLS, linear front-door.

Pure numpy. Each function takes a ``pandas.DataFrame`` and column names and
returns a ``LinearEstimate`` whose ``ci`` is a Wald ``Interval`` that says
so. Standard errors are the classical homoskedastic ones,
``sigma^2 (X'X)^{-1}`` with ``sigma^2 = RSS / (n - k)``; there is no
heteroskedasticity-robust option here, and the field names do not pretend
otherwise.

Routes and what each assumes:

* ``ols`` — the back-door route: with ``covariates`` a back-door admissible
  set (Pearl 2009, Def. 3.3.1), the coefficient on ``x`` is the average
  causal effect per unit of ``x`` in a linear model.
* ``two_stage_least_squares`` — the instrumental-variable route: with
  ``instruments`` relevant, excluded, and unconfounded with the outcome
  (Pearl 2009, §7.4.2; Angrist & Pischke 2009, Ch. 4), the 2SLS slope is
  consistent where OLS is not. Residuals for the SE use the *original* ``x``,
  not the first-stage fit (Wooldridge 2010, §5.2.1).
* ``frontdoor_linear`` — the front-door route (Pearl 2009, Def. 3.3.3 and
  Thm. 3.3.4): in a linear model the effect is the product of the
  mediator-on-treatment slope and the outcome-on-mediator slope adjusted for
  the treatment; the SE is by the delta method.

Every estimator raises ``KeyError`` naming missing columns and ``ValueError``
when a column plays two roles (``y`` as a covariate, an instrument as the
treatment, ...), when the frame's column labels are not unique, when ``n``
is not larger than the parameter count plus one, when the design is rank
deficient, or when the fit is perfect (a residual sum of squares of zero is
a column that was built from the others, not a finding). Rows are never
dropped silently: a non-finite value is an error, not a missing observation.

Ported by specification from the parent's ``estimators/causal.py``; no code
was copied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import field_validator

from axiom.core.intervals import Interval, wald
from axiom.core.spec import Spec
from axiom.core.verdict import Assumption

__all__ = [
    "WEAK_INSTRUMENT_F",
    "LeastSquaresFit",
    "LinearEstimate",
    "design_matrix",
    "frontdoor_linear",
    "least_squares",
    "ols",
    "two_stage_least_squares",
    "weak_instrument_check",
]

Array = npt.NDArray[np.float64]
Method = Literal["ols", "2sls", "frontdoor"]

WEAK_INSTRUMENT_F = 10.0
"""Staiger & Stock (1997) rule of thumb: a first-stage F below 10 is a weak instrument."""

_PERFECT_FIT_RELATIVE_RSS = 1e-18
"""``rss <= this * tss`` (or ``tss == 0``) is a perfect fit; ``least_squares`` rejects it.

Set near double precision so that a genuinely low-noise model (noise/signal
SD ratio of 1e-6 gives ``rss/tss ~ 1e-12``) is still fit, while an outcome
that is literally a linear function of the design (``rss/tss ~ 1e-32``) is
refused.
"""


class LinearEstimate(Spec):
    """A point estimate with its classical standard error and provenance.

    ``estimate`` is the effect of ``treatment`` on ``outcome`` per unit of
    treatment; ``covariates`` are the adjustment columns; ``detail`` holds
    method-specific diagnostics (``first_stage_f`` for 2SLS, the per-stage
    slopes for front-door). ``ci`` is a Wald interval labelled as such.
    """

    estimate: float
    se: float
    n: int
    method: Method
    treatment: str
    outcome: str
    covariates: tuple[str, ...] = ()
    detail: dict[str, float] = {}

    @field_validator("se")
    @classmethod
    def _se_finite_non_negative(cls, v: float) -> float:
        if not (np.isfinite(v) and v >= 0.0):
            raise ValueError(f"se must be finite and non-negative, got {v}")
        return v

    @field_validator("n")
    @classmethod
    def _n_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n must be at least 1, got {v}")
        return v

    def ci(self, mass: float) -> Interval:
        """``estimate ± z_{(1+mass)/2} · se`` as a ``wald`` ``Interval``."""
        return wald(self.estimate, self.se, mass)

    def z(self, null: float = 0.0) -> float:
        """``(estimate - null) / se``.

        ``0.0`` when ``estimate == null`` whatever ``se`` is; otherwise
        ``inf`` with the sign of the difference when ``se == 0``.
        """
        difference = self.estimate - null
        if difference == 0.0:
            return 0.0
        if self.se == 0.0:
            return float(np.sign(difference) * np.inf)
        return difference / self.se


@dataclass(frozen=True, slots=True)
class LeastSquaresFit:
    """The pieces of one OLS fit that every estimator here reuses.

    ``beta`` are the coefficients in design-column order, ``cov`` is
    ``sigma^2 (X'X)^{-1}``, ``residuals`` is ``y - X beta``, ``sigma2`` is
    ``RSS / df_resid``, and ``rss`` is the residual sum of squares.
    """

    beta: Array
    cov: Array
    residuals: Array
    sigma2: float
    rss: float
    df_resid: int

    @property
    def se(self) -> Array:
        return np.sqrt(np.diag(self.cov))


def _check_frame(frame: pd.DataFrame) -> None:
    if not frame.columns.is_unique:
        dupes = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
        raise ValueError(f"frame column labels are not unique; repeated: {dupes}")


def _names(v: str | Sequence[str]) -> tuple[str, ...]:
    """A bare string is one column name, not a sequence of characters."""
    return (v,) if isinstance(v, str) else tuple(v)


def _check_roles(**roles: str | Sequence[str]) -> None:
    """Raise ``ValueError`` if any column is named in two roles (or twice in one)."""
    where: dict[str, list[str]] = {}
    for role, columns in roles.items():
        for column in _names(columns):
            where.setdefault(column, []).append(role)
    clashes = {c: r for c, r in where.items() if len(r) > 1}
    if clashes:
        described = "; ".join(f"{c!r} as {' and '.join(r)}" for c, r in sorted(clashes.items()))
        raise ValueError(
            f"columns must be distinct across the roles {tuple(roles)}; repeated: {described}"
        )


def design_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> Array:
    """``[1, frame[columns]]`` as a float array.

    Raises ``KeyError`` on missing columns and ``ValueError`` on repeated
    columns, non-unique frame labels, or non-finite values.
    """
    _check_frame(frame)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")
    if len(set(columns)) != len(columns):
        dupes = sorted({c for c in columns if list(columns).count(c) > 1})
        raise ValueError(f"columns repeated in the design: {dupes}")
    body = _finite_columns(frame, columns)
    return np.column_stack([np.ones(len(frame), dtype=np.float64), body])


def _finite_columns(frame: pd.DataFrame, columns: Sequence[str]) -> Array:
    if not columns:
        return np.empty((len(frame), 0), dtype=np.float64)
    values = np.asarray(frame.loc[:, list(columns)].to_numpy(dtype=np.float64))
    bad = [c for c, ok in zip(columns, np.isfinite(values).all(axis=0), strict=True) if not ok]
    if bad:
        raise ValueError(f"non-finite values in columns: {bad}")
    return values


def _vector(frame: pd.DataFrame, column: str) -> Array:
    _check_frame(frame)
    if column not in frame.columns:
        raise KeyError(f"columns not in frame: {[column]}")
    return _finite_columns(frame, [column])[:, 0]


def least_squares(design: Array, y: Array) -> LeastSquaresFit:
    """OLS of ``y`` on ``design`` with classical covariance.

    Raises ``ValueError`` if ``n <= k + 1``, the design is rank deficient, or
    the fit is perfect (``rss`` is zero relative to the variance of ``y``):
    a perfect fit means ``y`` is a linear function of the design columns,
    which is a specification error, and its standard errors would be zero.
    """
    n, k = design.shape
    if n <= k + 1:
        raise ValueError(f"need n > k + 1 observations for k = {k} parameters, got n = {n}")
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < k:
        raise ValueError(f"design matrix is rank deficient (rank {rank} < {k} columns)")
    residuals = y - design @ beta
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    if tss == 0.0 or rss <= _PERFECT_FIT_RELATIVE_RSS * tss:
        raise ValueError(
            f"perfect fit: rss = {rss:.3g} against tss = {tss:.3g}; y is a linear function "
            "of the design columns (or constant), so the model is misspecified and the "
            "standard errors would be zero"
        )
    df_resid = n - k
    sigma2 = rss / df_resid
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return LeastSquaresFit(
        beta=np.asarray(beta, dtype=np.float64),
        cov=np.asarray(cov, dtype=np.float64),
        residuals=np.asarray(residuals, dtype=np.float64),
        sigma2=sigma2,
        rss=rss,
        df_resid=df_resid,
    )


def _r_squared(fit: LeastSquaresFit, y: Array) -> float:
    tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - fit.rss / tss if tss > 0 else 0.0


# -- OLS ----------------------------------------------------------------------------


def ols(
    frame: pd.DataFrame, y: str, x: str, covariates: str | Sequence[str] = ()
) -> LinearEstimate:
    """Coefficient on ``x`` in OLS of ``y`` on ``[1, x, covariates]``.

    The back-door estimator: consistent for the effect when ``covariates``
    is a back-door admissible set and the model is linear. Naive (empty
    ``covariates``) it is the association, which is what the recovery tests
    show being wrong.
    """
    covariates = _names(covariates)
    _check_roles(y=[y], x=[x], covariates=covariates)
    cols = [x, *covariates]
    design = design_matrix(frame, cols)
    target = _vector(frame, y)
    fit = least_squares(design, target)
    return LinearEstimate(
        estimate=float(fit.beta[1]),
        se=float(fit.se[1]),
        n=int(design.shape[0]),
        method="ols",
        treatment=x,
        outcome=y,
        covariates=tuple(covariates),
        detail={
            "sigma": float(np.sqrt(fit.sigma2)),
            "r_squared": _r_squared(fit, target),
            "df_resid": float(fit.df_resid),
        },
    )


# -- 2SLS -----------------------------------------------------------------------------


def two_stage_least_squares(
    frame: pd.DataFrame,
    y: str,
    x: str,
    instruments: str | Sequence[str],
    covariates: str | Sequence[str] = (),
) -> LinearEstimate:
    """2SLS of ``y`` on ``x`` with ``instruments``, adjusting for ``covariates``.

    First stage: ``x`` on ``[1, instruments, covariates]``. Second stage:
    ``y`` on ``[1, x_hat, covariates]``. The standard error uses residuals
    ``y - [1, x, covariates] beta`` with the original ``x`` — the second-stage
    OLS residuals are the wrong ones (Wooldridge 2010, §5.2.1). ``detail``
    carries ``first_stage_f`` (the F-test of the instruments in the first
    stage, against the model with covariates only), its degrees of freedom,
    and the 2SLS ``sigma``. Weak instruments are not an error here: see
    ``weak_instrument_check``.
    """
    if not instruments:
        raise ValueError("two_stage_least_squares needs at least one instrument")
    instruments, covariates = _names(instruments), _names(covariates)
    _check_roles(y=[y], x=[x], instruments=instruments, covariates=covariates)
    first_design = design_matrix(frame, [*instruments, *covariates])
    treatment = _vector(frame, x)
    target = _vector(frame, y)
    first = least_squares(first_design, treatment)
    restricted = least_squares(design_matrix(frame, list(covariates)), treatment)
    q = len(instruments)
    f_stat = ((restricted.rss - first.rss) / q) / first.sigma2
    x_hat = first_design @ first.beta

    second_design = design_matrix(frame, list(covariates))
    second_design = np.column_stack([second_design[:, :1], x_hat, second_design[:, 1:]])
    n, k = second_design.shape
    if n <= k + 1:
        raise ValueError(f"need n > k + 1 observations for k = {k} parameters, got n = {n}")
    beta, _, rank, _ = np.linalg.lstsq(second_design, target, rcond=None)
    if rank < k:
        raise ValueError(f"second-stage design is rank deficient (rank {rank} < {k} columns)")
    original_design = np.column_stack([second_design[:, :1], treatment, second_design[:, 2:]])
    residuals = target - original_design @ beta
    rss = float(residuals @ residuals)
    tss = float(((target - target.mean()) ** 2).sum())
    if tss == 0.0 or rss <= _PERFECT_FIT_RELATIVE_RSS * tss:
        raise ValueError(
            f"perfect fit: structural rss = {rss:.3g} against tss = {tss:.3g}; y is a "
            "linear function of x and the covariates, so the standard errors would be zero"
        )
    sigma2 = rss / (n - k)
    cov = sigma2 * np.linalg.inv(second_design.T @ second_design)
    return LinearEstimate(
        estimate=float(beta[1]),
        se=float(np.sqrt(cov[1, 1])),
        n=int(n),
        method="2sls",
        treatment=x,
        outcome=y,
        covariates=tuple(covariates),
        detail={
            "first_stage_f": float(f_stat),
            "first_stage_df_num": float(q),
            "first_stage_df_den": float(first.df_resid),
            "n_instruments": float(q),
            "sigma": float(np.sqrt(sigma2)),
        },
    )


def weak_instrument_check(est: LinearEstimate, threshold: float = WEAK_INSTRUMENT_F) -> Assumption:
    """The instrument-strength assumption, checked against the first-stage F.

    ``satisfied`` when ``first_stage_f >= threshold`` (Staiger & Stock
    1997: 10), ``violated`` otherwise. Only a 2SLS estimate carries a first
    stage; anything else is a ``ValueError``, as is a non-positive threshold.
    """
    if est.method != "2sls" or "first_stage_f" not in est.detail:
        raise ValueError("weak_instrument_check needs a 2SLS estimate with detail['first_stage_f']")
    if not (np.isfinite(threshold) and threshold > 0.0):
        raise ValueError(f"threshold must be a positive finite F value, got {threshold}")
    f = est.detail["first_stage_f"]
    assumption = Assumption(
        name="instrument_strength",
        facet="identification",
        statement=(
            f"the instruments for {est.treatment!r} are strong: first-stage F = {f:.3g} "
            f"against the rule-of-thumb threshold {threshold:g}"
        ),
        challenged_by=(
            "a weak first stage: 2SLS is then biased toward OLS and the Wald interval "
            "understates uncertainty (Staiger & Stock 1997; Stock & Yogo 2005)"
        ),
        detail={"first_stage_f": f"{f:.6g}", "threshold": f"{threshold:g}"},
    )
    return assumption.satisfied() if f >= threshold else assumption.violated()


# -- front-door ---------------------------------------------------------------------


def frontdoor_linear(
    frame: pd.DataFrame,
    y: str,
    x: str,
    mediators: str | Sequence[str],
    covariates: str | Sequence[str] = (),
) -> LinearEstimate:
    """Linear front-door estimate of the effect of ``x`` on ``y`` through ``mediators``.

    Stage one is a multivariate OLS of the mediators on ``[1, x, covariates]``;
    ``a_m`` is the slope on ``x`` for mediator ``m``. Stage two is one OLS of
    ``y`` on ``[1, mediators, x, covariates]``; ``b_m`` is the slope on ``m``
    (conditioning on ``x`` blocks the back-door path from ``m`` to ``y``
    through the latent confounder). The estimate is ``sum_m a_m b_m``
    (Pearl 2009, Thm. 3.3.4, linear case).

    Delta-method SE: ``Var = b' Cov(a) b + a' Cov(b) a``. ``Cov(a)`` is the
    full cross-mediator covariance of the stage-one slopes,
    ``(R'R / (n - k)) (X'X)^{-1}_{xx}`` with ``R`` the matrix of stage-one
    residuals (Seemingly-unrelated regressions with identical regressors
    reduce to equation-by-equation OLS; Zellner 1962), so mediators that
    share noise — ``M1 -> M2`` — are handled. ``Cov(b)`` is the stage-two
    covariance across mediators. The two stages are treated as independent,
    which holds exactly when the stage-one residuals are independent of the
    stage-two residual; the approximation is stated here rather than hidden.
    ``detail`` carries ``a[m]`` and ``b[m]`` per mediator.
    """
    if not mediators:
        raise ValueError("frontdoor_linear needs at least one mediator")
    mediators, covariates = _names(mediators), _names(covariates)
    _check_roles(y=[y], x=[x], mediators=mediators, covariates=covariates)
    target = _vector(frame, y)
    stage_one_design = design_matrix(frame, [x, *covariates])
    p = len(mediators)
    a = np.empty(p, dtype=np.float64)
    stage_one_residuals = np.empty((stage_one_design.shape[0], p), dtype=np.float64)
    df_resid = 0
    for i, m in enumerate(mediators):
        fit = least_squares(stage_one_design, _vector(frame, m))
        a[i] = fit.beta[1]
        stage_one_residuals[:, i] = fit.residuals
        df_resid = fit.df_resid
    residual_cov = stage_one_residuals.T @ stage_one_residuals / df_resid
    xx_inv_x = float(np.linalg.inv(stage_one_design.T @ stage_one_design)[1, 1])
    cov_a = residual_cov * xx_inv_x
    stage_two = least_squares(design_matrix(frame, [*mediators, x, *covariates]), target)
    idx = slice(1, 1 + p)
    b = stage_two.beta[idx]
    cov_b = stage_two.cov[idx, idx]
    estimate = float(a @ b)
    variance = float(b @ cov_a @ b + a @ cov_b @ a)
    detail = {"sigma": float(np.sqrt(stage_two.sigma2))}
    for i, m in enumerate(mediators):
        detail[f"a[{m}]"] = float(a[i])
        detail[f"b[{m}]"] = float(b[i])
    return LinearEstimate(
        estimate=estimate,
        se=float(np.sqrt(variance)),
        n=int(stage_one_design.shape[0]),
        method="frontdoor",
        treatment=x,
        outcome=y,
        covariates=tuple(covariates),
        detail=detail,
    )
