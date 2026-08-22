"""Residual diagnostics of a fitted surface: what the mean leaves behind, and whether it looks
like the likelihood's noise.

The residual is ``y − E_post[mu]``: the observed outcome minus the
posterior mean of the model's mean, one ``forward`` per posterior draw
(``surface.forward.predict`` with ``noise=False``). On the
``(n_units, n_periods)`` grid the report gives, each with its test
statistic, p-value, and the ``n`` it used:

* per unit: mean and sd of the residual (a unit whose mean is far from
  zero has an intercept the model does not capture);
* Durbin–Watson ``Σ (e_t − e_{t−1})² / Σ e_t²`` over the within-unit
  series pooled (no p-value: its null distribution depends on the design;
  values near 2 mean no lag-1 autocorrelation);
* Ljung–Box at each stated lag, ``Q = n(n+2) Σ_k ρ_k² / (n − k)`` on the
  pooled within-unit autocorrelations, ``χ²`` with ``lag`` degrees of
  freedom;
* normality: Shapiro–Wilk (``scipy.stats.shapiro``; skipped above 5000
  residuals, where scipy's p-value is unreliable) and Jarque–Bera;
* heteroscedasticity: Breusch–Pagan, the ``n · R²`` of ``e²`` regressed on
  the fitted values, ``χ²`` with one degree of freedom.

``flagged`` lists the tests whose p-value fell below ``alpha``.
Ported by specification from the parent's
``validation/residual_diagnostics.py`` (ledger row
``diagnose/residuals.py``, PORT).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import stats as _st

from axiom.core import Posterior, Spec, Unsupported, Unverified
from axiom.surface import FitResult
from axiom.surface.forward import predict

__all__ = [
    "ResidualReport",
    "ResidualTest",
    "UnitResiduals",
    "residuals",
]

Array = npt.NDArray[np.float64]
SHAPIRO_MAX_N = 5000


class ResidualTest(Spec):
    """One test: statistic, p-value (``None`` when the test has no reference distribution, as
    Durbin–Watson), the ``n`` it was computed on, and the lag where one applies."""

    name: str
    statistic: float
    p_value: float | None = None
    n: int = Field(ge=1)
    lag: int | None = None
    note: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> ResidualTest:
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"{self.name}: p-value {self.p_value} outside [0, 1]")
        return self


class UnitResiduals(Spec):
    """Residual mean and sd for one unit over its periods."""

    unit: str
    n: int = Field(ge=1)
    mean: float
    sd: float


class ResidualReport(Spec):
    """Per-unit summaries and the pooled tests. ``skipped`` names tests that could not run on
    this panel (a Ljung–Box lag beyond the series, Shapiro above its size limit), with why."""

    n_units: int = Field(ge=1)
    n_periods: int = Field(ge=1)
    n: int = Field(ge=1)
    alpha: float = Field(gt=0, lt=1)
    residual_sd: float
    units: tuple[UnitResiduals, ...]
    tests: tuple[ResidualTest, ...]
    skipped: tuple[str, ...] = ()
    flagged: tuple[str, ...]
    provenance: dict[str, Any] = {}

    @model_validator(mode="after")
    def _consistent(self) -> ResidualReport:
        if self.n != self.n_units * self.n_periods:
            raise ValueError("n must equal n_units * n_periods")
        expected = tuple(
            t.name for t in self.tests if t.p_value is not None and t.p_value < self.alpha
        )
        if self.flagged != expected:
            raise ValueError("flagged must list exactly the tests with p below alpha")
        return self


def _durbin_watson(e: Array) -> float:
    """Pooled over units: within-unit first differences only, so unit boundaries do not count."""
    d = np.diff(e, axis=1)
    den = float(np.sum(e**2))
    return float(np.sum(d**2) / den) if den > 0 else float("nan")


def _pooled_autocorr(e: Array, lag: int) -> float:
    """Lag-``lag`` autocorrelation of the within-unit series, pooled across units."""
    c = e - e.mean(axis=1, keepdims=True)
    num = float(np.sum(c[:, lag:] * c[:, :-lag]))
    den = float(np.sum(c**2))
    return num / den if den > 0 else float("nan")


def _ljung_box(e: Array, lag: int) -> tuple[float, float]:
    n = e.shape[1]
    q = 0.0
    for k in range(1, lag + 1):
        r = _pooled_autocorr(e, k)
        q += r * r / (n - k)
    q *= n * (n + 2) * e.shape[0]  # each unit's series contributes its own n terms
    return float(q), float(_st.chi2.sf(q, df=lag))


def _breusch_pagan(e: Array, fitted: Array) -> tuple[float, float]:
    """``n · R²`` of ``e²`` on the fitted values (Koenker's studentized form is not used: the
    classic LM statistic is what the parent reported)."""
    y = (e**2).reshape(-1)
    x = fitted.reshape(-1)
    n = y.size
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / tss if tss > 0 else 0.0
    lm = n * r2
    return float(lm), float(_st.chi2.sf(lm, df=1))


def residuals(
    result: FitResult,
    *,
    lags: tuple[int, ...] = (1, 5, 10),
    alpha: float = 0.05,
) -> ResidualReport | Unsupported:
    """The residual report of ``result`` on its fitted panel (module docstring).

    ``lags`` are the Ljung–Box lags; a lag at or beyond the number of
    periods is skipped and named. ``Unsupported`` when the fit has no
    posterior.
    """
    if not lags or any(lag < 1 for lag in lags) or len(set(lags)) != len(lags):
        raise ValueError(f"lags must be distinct positive integers, got {lags}")
    if isinstance(result.posterior, Unverified | Unsupported):
        return Unsupported(
            reason=f"residuals need a posterior; the fit has "
            f"{type(result.posterior).__name__}: {result.posterior.reason}",
            missing=("posterior",),
        )
    posterior: Posterior = result.posterior
    outcome = result.surface.spec.outcome.name
    y = np.asarray(result.data[outcome], dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"the fitted outcome must be an (n_units, n_periods) grid, got {y.shape}")
    pred = predict(result.surface, posterior, result.data, noise=False)
    fitted = np.asarray(pred.values, dtype=np.float64).reshape(-1, *y.shape).mean(axis=0)
    e = y - fitted
    n_units, n_periods = y.shape
    n = n_units * n_periods
    labels = result.unit_labels
    units = tuple(
        UnitResiduals(
            unit=str(labels[i]),
            n=n_periods,
            mean=float(e[i].mean()),
            sd=float(e[i].std(ddof=1)) if n_periods > 1 else 0.0,
        )
        for i in range(n_units)
    )
    tests: list[ResidualTest] = []
    skipped: list[str] = []
    if n_periods > 1:
        tests.append(
            ResidualTest(
                name="durbin_watson",
                statistic=_durbin_watson(e),
                n=n,
                note="no p-value: the null distribution depends on the design; ~2 means none",
            )
        )
    else:
        skipped.append("durbin_watson: needs at least two periods")
    for lag in lags:
        if lag >= n_periods:
            skipped.append(f"ljung_box[{lag}]: lag must be below n_periods={n_periods}")
            continue
        q, p = _ljung_box(e, lag)
        tests.append(ResidualTest(name=f"ljung_box[{lag}]", statistic=q, p_value=p, n=n, lag=lag))
    flat = e.reshape(-1)
    if 3 <= n <= SHAPIRO_MAX_N:
        w, p = _st.shapiro(flat)
        tests.append(ResidualTest(name="shapiro_wilk", statistic=float(w), p_value=float(p), n=n))
    else:
        skipped.append(f"shapiro_wilk: needs 3 <= n <= {SHAPIRO_MAX_N}, n={n}")
    if n >= 2:
        jb = _st.jarque_bera(flat)
        tests.append(
            ResidualTest(
                name="jarque_bera", statistic=float(jb.statistic), p_value=float(jb.pvalue), n=n
            )
        )
    else:
        skipped.append("jarque_bera: needs at least two residuals")
    if n >= 3 and np.ptp(fitted) > 0:
        lm, p = _breusch_pagan(e, fitted)
        tests.append(ResidualTest(name="breusch_pagan", statistic=lm, p_value=p, n=n))
    else:
        skipped.append("breusch_pagan: needs at least three residuals and varying fitted values")
    return ResidualReport(
        n_units=n_units,
        n_periods=n_periods,
        n=n,
        alpha=alpha,
        residual_sd=float(flat.std(ddof=1)) if n > 1 else 0.0,
        units=units,
        tests=tuple(tests),
        skipped=tuple(skipped),
        flagged=tuple(t.name for t in tests if t.p_value is not None and t.p_value < alpha),
        provenance={
            "spec_hash": result.surface.spec.content_hash(),
            "model_hash": result.provenance.get("model_hash", result.surface.model.content_hash()),
            "panel_hash": result.provenance.get("panel_hash"),
            "n_draws": posterior.n_draws(),
        },
    )
