from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from axiom.core import clopper_pearson, effective_sample_size, mc_standard_error, z_score


def test_clopper_pearson_region_contains_expected_count() -> None:
    r = clopper_pearson(500, 0.05, alpha=0.001)
    assert r.lower <= 25 <= r.upper
    assert r.accepts(25) and not r.accepts(0) and not r.accepts(80)
    lo, hi = r.rate_bounds
    assert lo < 0.05 < hi
    # region has (at least) the nominal coverage
    cov = stats.binom.cdf(r.upper, 500, 0.05) - stats.binom.cdf(r.lower - 1, 500, 0.05)
    assert cov >= 0.999 - 1e-9
    with pytest.raises(ValueError):
        clopper_pearson(10, 1.5)


def test_ess_iid_is_about_n_and_ar1_is_less() -> None:
    rng = np.random.default_rng(0)
    iid = rng.normal(size=(4, 2000))
    assert effective_sample_size(iid) == pytest.approx(8000, rel=0.15)
    rho = 0.9
    ar = np.zeros((4, 2000))
    for c in range(4):
        for t in range(1, 2000):
            ar[c, t] = rho * ar[c, t - 1] + rng.normal() * np.sqrt(1 - rho**2)
    ess = effective_sample_size(ar)
    assert ess < 2000
    # theory: n (1-rho)/(1+rho) ~= 421 for 8000 draws
    assert ess == pytest.approx(8000 * (1 - rho) / (1 + rho), rel=0.35)
    assert effective_sample_size(np.ones((2, 10))) == 20
    assert effective_sample_size([1.0, 2.0]) == 2


def test_mcse_and_z() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=10_000)
    assert mc_standard_error(x) == pytest.approx(0.01, rel=0.2)
    assert z_score(1.0, 0.0, 0.5) == 2.0
    with pytest.raises(ValueError):
        z_score(1, 0, 0)
