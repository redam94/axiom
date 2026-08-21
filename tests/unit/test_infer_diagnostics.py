"""Convergence diagnostics: R̂, ESS, MCSE on known chains, the report's round trip,
non-finite draws, and a cross-check against arviz-stats when it is installed."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from axiom.core import Posterior
from axiom.infer.diagnostics import (
    ConvergenceReport,
    ConvergenceThresholds,
    ParameterDiagnostics,
    diagnose,
    ess_bulk,
    ess_tail,
    mcse_mean,
    split_rhat,
)


@pytest.fixture
def iid() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((4, 1000))


def ar1(phi: float, shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c, n = shape
    x = np.empty(shape)
    x[:, 0] = rng.standard_normal(c)
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + scale * rng.standard_normal(c)
    return x


def test_iid_chains_mix(iid: np.ndarray) -> None:
    assert abs(split_rhat(iid) - 1.0) < 0.01
    n = iid.size
    assert 0.8 * n < ess_bulk(iid) < 1.2 * n
    assert 0.6 * n < ess_tail(iid) < 1.4 * n
    assert abs(mcse_mean(iid) - 1.0 / np.sqrt(n)) < 0.2 / np.sqrt(n)


def test_shifted_chains_do_not_mix(iid: np.ndarray) -> None:
    x = iid + 5.0 * np.arange(4)[:, None]
    assert split_rhat(x) > 1.1
    assert ess_bulk(x) < 0.1 * x.size


def test_autocorrelated_chain_has_small_ess() -> None:
    x = ar1(0.9, (4, 2000))
    n = x.size
    # theory: ess = n (1 - phi) / (1 + phi) = n / 19
    assert 0.02 * n < ess_bulk(x) < 0.12 * n
    assert abs(split_rhat(x) - 1.0) < 0.05


def test_antithetic_chain_has_ess_above_n() -> None:
    """phi = -0.5: tau = 1/3 in theory; the floor is 1/log10(N), not 1/log10(n_split + 1)."""
    x = ar1(-0.5, (4, 1000))
    n = x.size
    assert ess_bulk(x) > 2.0 * n
    assert ess_bulk(x) <= n * np.log10(n) * 1.0001


def test_one_chain_drifting_is_caught() -> None:
    x = np.random.default_rng(2).standard_normal((4, 1000))
    x[0] += np.linspace(0.0, 4.0, 1000)
    assert split_rhat(x) > 1.05


def test_constant_draws_are_undefined() -> None:
    x = np.ones((2, 100))
    assert np.isnan(split_rhat(x))
    assert np.isnan(ess_bulk(x))
    assert np.isnan(mcse_mean(x))


def test_single_chain_is_split() -> None:
    x = np.random.default_rng(3).standard_normal(2000)
    assert abs(split_rhat(x) - 1.0) < 0.02
    assert 1500 < ess_bulk(x) < 2500


def test_bad_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        split_rhat(np.zeros((2, 3, 4)))


# -- arviz-stats cross-check -------------------------------------------------------------


def _arviz_array_stats() -> Any | None:
    try:
        return importlib.import_module("arviz_stats.base").array_stats
    except ImportError:
        return None


@pytest.mark.skipif(_arviz_array_stats() is None, reason="arviz-stats is not installed")
@pytest.mark.parametrize("phi", [0.0, 0.9, -0.5])
def test_matches_arviz_stats(phi: float) -> None:
    az = _arviz_array_stats()
    assert az is not None
    x = ar1(phi, (4, 1000), seed=11)
    kw = {"chain_axis": 0, "draw_axis": 1}
    assert ess_bulk(x) == pytest.approx(float(az.ess(x, method="bulk", **kw)), rel=1e-9)
    assert ess_tail(x) == pytest.approx(
        float(az.ess(x, method="tail", prob=(0.05, 0.95), **kw)), rel=1e-9
    )
    assert split_rhat(x) == pytest.approx(float(az.rhat(x, method="rank", **kw)), rel=1e-9)
    assert mcse_mean(x) == pytest.approx(float(az.mcse(x, method="mean", **kw)), rel=1e-9)


# -- report --------------------------------------------------------------------------


@pytest.fixture
def posterior(iid: np.ndarray) -> Posterior:
    rng = np.random.default_rng(5)
    return Posterior(
        {"mu": iid, "alpha": rng.standard_normal((4, 1000, 2))},
        coords={"alpha_dim0": [0, 1]},
        provenance={"seed": 5, "backend": "test"},
    )


def test_report_rows_and_verdict(posterior: Posterior) -> None:
    report = diagnose(posterior)
    assert isinstance(report, ConvergenceReport)
    assert [r.name for r in report.rows] == ["alpha[0]", "alpha[1]", "mu"]
    assert report.converged
    assert report.failing == ()
    assert report.divergences == 0
    assert report.nonfinite_draw_frac == 0.0
    assert (report.n_chains, report.n_draws) == (4, 1000)
    assert report.thresholds == ConvergenceThresholds()
    row = report.row("mu")
    assert isinstance(row, ParameterDiagnostics)
    assert row.rhat is not None and row.rhat < 1.01
    assert row.ess_bulk is not None and row.ess_bulk > 400
    assert row.n_nonfinite == 0
    assert row.mean is not None and row.sd is not None
    frame = report.to_frame()
    assert list(frame.index) == ["alpha[0]", "alpha[1]", "mu"]
    assert {"rhat", "ess_bulk", "ess_tail", "mcse_mean", "n_nonfinite"} <= set(frame.columns)


def test_report_round_trips_as_json(posterior: Posterior) -> None:
    report = diagnose(posterior, divergences=2, rhat_max=1.05)
    again = ConvergenceReport.from_json(report.to_json())
    assert again == report
    assert again.content_hash() == report.content_hash()
    assert again.thresholds.rhat_max == 1.05
    assert again.divergences == 2
    assert again.converged is False  # two divergences against a max of zero


def test_converged_flips_with_thresholds(posterior: Posterior) -> None:
    assert diagnose(posterior).converged
    assert not diagnose(posterior, ess_min=10_000).converged
    assert diagnose(posterior, divergences=3, divergences_max=5).converged
    assert not diagnose(posterior, divergences=3).converged
    shifted = Posterior({"mu": posterior.draws("mu") + 5.0 * np.arange(4)[:, None]})
    bad = diagnose(shifted)
    assert not bad.converged
    assert bad.failing == ("mu",)
    assert diagnose(shifted, rhat_max=5.0, ess_min=1.0).converged


def test_divergences_default_from_provenance(iid: np.ndarray) -> None:
    post = Posterior({"mu": iid}, provenance={"divergences": 4})
    report = diagnose(post)
    assert report.divergences == 4
    assert not report.converged
    assert diagnose(post, divergences=0).converged


def test_undefined_rows_never_converge() -> None:
    post = Posterior({"c": np.ones((2, 500))})
    report = diagnose(post)
    row = report.row("c")
    assert row.rhat is None and row.ess_bulk is None and row.mcse_mean is None
    assert not report.converged
    assert ConvergenceReport.from_json(report.to_json()) == report


def test_one_nonfinite_draw_fails_the_verdict() -> None:
    """One ``inf`` in 4 x 500 iid draws used to pass (rank R̂ ignores it) and break the JSON."""
    x = np.random.default_rng(0).standard_normal((4, 500))
    x[0, 10] = np.inf
    report = diagnose(x_post := Posterior({"mu": x}))
    row = report.row("mu")
    assert row.n_nonfinite == 1
    assert row.mean is None and row.sd is None
    assert not row.passes(report.thresholds)
    assert report.failing == ("mu",)
    assert not report.converged
    assert report.nonfinite_draw_frac == pytest.approx(1 / 2000)
    again = ConvergenceReport.from_json(report.to_json())
    assert again == report
    x[0, 10] = np.nan
    report = diagnose(Posterior({"mu": x}))
    assert report.row("mu").rhat is None and report.row("mu").mcse_mean is None
    assert not report.converged
    assert diagnose(x_post, rhat_max=10.0, ess_min=1.0).converged is False


def test_declared_nonfinite_fraction_fails_the_verdict(iid: np.ndarray) -> None:
    post = Posterior({"mu": iid}, provenance={"nonfinite_draw_frac": 0.01})
    report = diagnose(post)
    assert report.failing == ()
    assert report.nonfinite_draw_frac == 0.01
    assert not report.converged


def test_single_draw_has_no_sd() -> None:
    report = diagnose(Posterior({"mu": np.ones((1, 1))}))
    assert report.row("mu").sd is None
    assert report.row("mu").mean == 1.0
    assert not report.converged


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        ConvergenceThresholds(rhat_max=0.9)
    with pytest.raises(ValueError):
        ConvergenceThresholds(ess_min=0)
    with pytest.raises(ValueError):
        ParameterDiagnostics(
            name="x",
            rhat=None,
            ess_bulk=None,
            ess_tail=None,
            mcse_mean=None,
            mean=None,
            sd=None,
            n_nonfinite=-1,
        )
