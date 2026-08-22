"""Unit tests for ``axiom.diagnose.ppc`` and ``axiom.diagnose.residuals``."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from axiom.core import Prior, Unsupported, Unverified, eti
from axiom.diagnose.ppc import DEFAULT_STATISTICS, PPCResult, StatisticCheck, posterior_predictive
from axiom.diagnose.residuals import ResidualReport, ResidualTest, residuals
from axiom.sim import DosePlan, surface_world
from axiom.surface import FitResult, HillKernel, fit

AMP = Prior(family="lognormal", hyper={"mu": 0.0, "sigma": 0.5})


@pytest.fixture(scope="module")
def fitted() -> FitResult:
    w = surface_world(
        n_units=6,
        n_periods=14,
        treatments=("a",),
        kernels=HillKernel(reference_dose=1.0, amplitude_prior=AMP),
        intercept="shared",
        noise_sd=0.3,
        seed=7,
        doses=DosePlan(zero_fraction=0.2),
    )
    return fit(w.spec, w.panel, backend="laplace", draws=150, chains=1, seed=0)


def _declined(res: FitResult) -> FitResult:
    return FitResult(res.surface, Unverified(reason="declined"), res.data, None, res.provenance)


# -- ppc -------------------------------------------------------------------------------------


def test_posterior_predictive_default_statistics(fitted: FitResult) -> None:
    out = posterior_predictive(fitted, n_draws=80, seed=1)
    assert isinstance(out, PPCResult)
    assert out.n_draws == 80 and out.seed == 1 and out.skipped == ()
    assert [s.name for s in out.statistics] == list(DEFAULT_STATISTICS)
    for s in out.statistics:
        assert s.n == 80 and s.interval.definition == "eti" and s.interval.mass == 0.9
        assert 0.0 <= s.p_two_sided <= 1.0 and s.p_two_sided <= 2 * s.p_value + 1e-12
        assert s.extreme == (s.p_two_sided < 0.05)
    assert out.extreme_statistics == tuple(s.name for s in out.statistics if s.extreme)
    # a well-specified fit: the observed mean is not extreme
    mean = next(s for s in out.statistics if s.name == "mean")
    assert not mean.extreme and mean.interval.contains(mean.observed)
    assert PPCResult.from_json(out.to_json()) == out


def test_posterior_predictive_is_seeded_and_subsamples(fitted: FitResult) -> None:
    a = posterior_predictive(fitted, n_draws=30, seed=4)
    b = posterior_predictive(fitted, n_draws=30, seed=4)
    c = posterior_predictive(fitted, n_draws=30, seed=5)
    assert a == b and a != c
    full = posterior_predictive(fitted, n_draws=10_000, seed=0)
    assert isinstance(full, PPCResult) and full.n_draws == fitted.n_draws()


def test_posterior_predictive_custom_statistic_flags_an_extreme(fitted: FitResult) -> None:
    out = posterior_predictive(
        fitted, statistics={"shifted_mean": lambda y: float(np.mean(y)) + 10.0}, n_draws=50
    )
    assert isinstance(out, PPCResult)
    (s,) = out.statistics
    # every replicate's shifted mean is also shifted, so the observed value sits in the middle
    assert not s.extreme
    out2 = posterior_predictive(fitted, statistics={"always_huge": lambda y: 1e6}, n_draws=20)
    assert isinstance(out2, PPCResult) and not out2.statistics[0].extreme
    nan = posterior_predictive(fitted, statistics={"nan": lambda y: float("nan")}, n_draws=5)
    assert isinstance(nan, PPCResult) and nan.skipped == ("nan",) and nan.statistics == ()


def test_posterior_predictive_failures(fitted: FitResult) -> None:
    assert isinstance(posterior_predictive(_declined(fitted)), Unsupported)
    with pytest.raises(ValueError, match="n_draws"):
        posterior_predictive(fitted, n_draws=0)
    with pytest.raises(ValueError, match="at least one statistic"):
        posterior_predictive(fitted, statistics={})
    with pytest.raises(ValidationError, match="extreme must equal"):
        StatisticCheck(
            name="x",
            observed=0.0,
            interval=eti([0.0, 1.0], 0.9),
            p_value=0.5,
            p_two_sided=1.0,
            n=2,
            alpha=0.05,
            extreme=True,
        )


# -- residuals -------------------------------------------------------------------------------


def test_residuals_report(fitted: FitResult) -> None:
    out = residuals(fitted, lags=(1, 5, 20))
    assert isinstance(out, ResidualReport)
    assert out.n_units == 6 and out.n_periods == 14 and out.n == 84
    assert len(out.units) == 6 and all(u.n == 14 for u in out.units)
    names = [t.name for t in out.tests]
    assert names == [
        "durbin_watson",
        "ljung_box[1]",
        "ljung_box[5]",
        "shapiro_wilk",
        "jarque_bera",
        "breusch_pagan",
    ]
    assert out.skipped == ("ljung_box[20]: lag must be below n_periods=14",)
    dw = out.tests[0]
    assert dw.p_value is None and 1.0 < dw.statistic < 3.0
    for t in out.tests[1:]:
        assert t.p_value is not None and 0.0 <= t.p_value <= 1.0 and t.n == 84
    assert out.flagged == tuple(
        t.name for t in out.tests if t.p_value is not None and t.p_value < 0.05
    )
    assert out.residual_sd == pytest.approx(0.3, abs=0.1)
    assert ResidualReport.from_json(out.to_json()) == out


def test_residuals_failures_and_validation(fitted: FitResult) -> None:
    assert isinstance(residuals(_declined(fitted)), Unsupported)
    with pytest.raises(ValueError, match="lags"):
        residuals(fitted, lags=(0,))
    with pytest.raises(ValueError, match="lags"):
        residuals(fitted, lags=(1, 1))
    with pytest.raises(ValidationError):
        ResidualTest(name="t", statistic=1.0, p_value=1.5, n=3)


def test_residuals_one_period_world_skips_serial_tests() -> None:
    w = surface_world(n_units=12, n_periods=1, treatments=("a",), intercept="shared", seed=0)
    res = fit(w.spec, w.panel, backend="laplace", draws=40, chains=1, seed=0)
    out = residuals(res, lags=(1,))
    assert isinstance(out, ResidualReport)
    assert any(s.startswith("durbin_watson") for s in out.skipped)
    assert any(s.startswith("ljung_box[1]") for s in out.skipped)
    assert all(u.sd == 0.0 for u in out.units)
