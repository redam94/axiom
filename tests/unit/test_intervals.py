from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats

from axiom.core import Interval, eti, hdi, interval, summarize


def test_interval_requires_provenance_and_sane_bounds() -> None:
    with pytest.raises(ValidationError):
        Interval(lower=0, upper=1)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Interval(lower=0, upper=1, definition="hdi", mass=1.0)
    with pytest.raises(ValueError):
        Interval(lower=1, upper=0, definition="eti", mass=0.9)
    with pytest.raises(ValueError):
        Interval(lower=float("nan"), upper=0, definition="eti", mass=0.9)
    i = Interval(lower=-1, upper=2, definition="hdi", mass=0.5)
    assert i.width == 3 and i.contains(0) and not i.contains(3)
    assert str(i) == "[-1, 2] (50% HDI)"


def test_eti_matches_quantiles() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=20_000)
    i = eti(x, 0.9)
    assert i.lower == pytest.approx(np.quantile(x, 0.05))
    assert i.upper == pytest.approx(np.quantile(x, 0.95))
    assert i.definition == "eti" and i.mass == 0.9


def test_hdi_is_narrower_than_eti_on_skewed_draws() -> None:
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 1.0, size=50_000)
    h, e = hdi(x, 0.9), eti(x, 0.9)
    assert h.width < e.width
    assert h.lower < e.lower
    # mass inside is right
    assert np.mean((x >= h.lower) & (x <= h.upper)) == pytest.approx(0.9, abs=0.005)


def test_hdi_approaches_analytic_mode_interval() -> None:
    # For a Gamma(5,1), the 90% HDI is known from the density; compare against
    # a fine numerical search on the true density.
    rng = np.random.default_rng(2)
    x = rng.gamma(5.0, 1.0, size=400_000)
    h = hdi(x, 0.9)
    dist = stats.gamma(5.0)
    best = None
    for lo in np.linspace(0.5, 3.0, 2501):
        hi = dist.ppf(dist.cdf(lo) + 0.9)
        if best is None or hi - lo < best[1] - best[0]:
            best = (lo, hi)
    assert best is not None
    assert h.lower == pytest.approx(best[0], abs=0.05)
    assert h.upper == pytest.approx(best[1], abs=0.05)


def test_hdi_symmetric_equals_eti_for_normal() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=200_000)
    assert hdi(x, 0.8).lower == pytest.approx(eti(x, 0.8).lower, abs=0.06)


def test_non_finite_draws_are_dropped_and_too_few_raise() -> None:
    x = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
    assert eti(x, 0.5).lower == pytest.approx(1.5)
    with pytest.raises(ValueError):
        hdi([1.0], 0.5)


def test_summarize_and_dispatch() -> None:
    x = np.arange(100.0)
    s = summarize(x, definition="eti", mass=0.5)
    assert s.n == 100 and s.mean == 49.5 and s.median == 49.5
    assert s.interval == interval(x, definition="eti", mass=0.5)
    assert s.interval.definition == "eti"
