"""Gate (Phase 8, exit criterion 1): SBC rank uniformity passes for the surface model and the
meta model — and a deliberately wrong refit fails the same check.

Two tiers:

* **fast** (default): ``N = 30`` simulations, family-wise ``alpha = 0.05``
  (each parameter at ``0.05 / k``), Laplace fits. With ``N = 30`` the
  chi-square histogram has two bins and the DKW band is wide, so this tier
  has little power against a *subtle* miscalibration — it is a smoke gate —
  but the negative controls (a refit whose prior on the amplitude /
  pooled mean is shifted and tight enough to ignore the data) fail it by a
  mile, which is what proves the check bites.
* **slow** (``-m slow``): ``N = 200`` on a larger panel, same ``alpha``, the
  roadmap's N. Under Laplace the surface is the linear-kernel surface on a
  ``20 × 30`` panel: Laplace is exact for the linear-Gaussian part and
  within SBC's resolution for ``log sigma`` at 600 observations. The Hill
  surface is checked under NumPyro (skipped when the extra is missing):
  Laplace on a Hill surface is *measurably* miscalibrated on the ``k``–``s``
  ridge at these panel sizes (``k`` ranks pile at the top when the true
  ``k`` is large) — a finding, not a bug, recorded here rather than hidden
  by a wide alpha.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import Prior, jax_available
from axiom.diagnose.sbc import SBCResult, SBCSpec, sbc_pool, sbc_surface
from axiom.meta import Corpus, PoolPriors, PoolSpec, StudyRecord
from axiom.sim import DosePlan, surface_world
from axiom.surface import HillKernel, LinearKernel, SurfaceSpec

AMPLITUDE = Prior(family="lognormal", hyper={"mu": 0.0, "sigma": 0.5})
WRONG_AMPLITUDE = Prior(family="lognormal", hyper={"mu": 3.0, "sigma": 0.05})
ALPHA = 0.05
FAST_N = 30
SLOW_N = 200


def _numpyro_available() -> bool:
    try:
        import numpyro  # noqa: F401
    except ImportError:
        return False
    return jax_available()


def _surface(n_units: int, n_periods: int, *, hill: bool = False) -> tuple[SurfaceSpec, object]:
    kernel = (
        HillKernel(reference_dose=1.0, amplitude_prior=AMPLITUDE)
        if hill
        else LinearKernel(reference_dose=1.0, amplitude_prior=AMPLITUDE)
    )
    world = surface_world(
        n_units=n_units,
        n_periods=n_periods,
        treatments=("a",),
        kernels=kernel,
        intercept="shared",
        noise_sd=0.3,
        seed=1,
        doses=DosePlan(zero_fraction=0.2),
    )
    return world.spec, world.panel


def _wrong(spec: SurfaceSpec) -> SurfaceSpec:
    kernel = spec.kernel_of("a").model_copy(update={"amplitude_prior": WRONG_AMPLITUDE})
    return spec.model_copy(update={"kernels": {"a": kernel}})


def _corpus(seed: int = 0, k: int = 12) -> Corpus:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(k):
        se = float(rng.uniform(0.1, 0.4))
        records.append(
            StudyRecord(
                study=f"s{i}",
                contributor=f"s{i}",
                quantity="elasticity",
                estimate=float(rng.normal(0.5, 0.3) + se * rng.normal()),
                se=se,
                read="experiment",
                family="f",
            )
        )
    return Corpus(records=tuple(records), name="sbc-gate")


POOL = PoolSpec(family="f")
WRONG_POOL = PoolSpec(family="f", priors=PoolPriors(mu_scale=0.02))


def _report(label: str, out: SBCResult) -> str:
    lines = [
        f"{label}: N={out.n_simulations} fitted={out.n_fitted} failed_fits={out.n_failed_fits} "
        f"alpha/param={out.alpha_per_parameter:.4f}"
    ]
    for p in out.parameters:
        lines.append(
            f"  {p.name}: chi2 p={p.chi2_p_value:.3f} (bins={p.bins}) "
            f"ecdf D={p.ecdf_statistic:.3f} band={p.ecdf_band:.3f} passed={p.passed}"
        )
    return "\n".join(lines)


# -- fast tier -------------------------------------------------------------------------------


def test_surface_sbc_passes_and_wrong_refit_fails_fast() -> None:
    spec, panel = _surface(8, 16)
    sbc_spec = SBCSpec(n_simulations=FAST_N, draws=200, seed=0, alpha=ALPHA)
    good = sbc_surface(spec, panel, sbc_spec=sbc_spec)
    assert isinstance(good, SBCResult)
    assert good.n_failed_fits == 0, good.failure_reasons
    assert good.passed, _report("surface", good)
    bad = sbc_surface(spec, panel, sbc_spec=sbc_spec, refit=_wrong(spec))
    assert isinstance(bad, SBCResult)
    assert not bad.passed and "beta_rate_a" in bad.failed_parameters, _report("wrong", bad)


def test_pool_sbc_passes_and_wrong_refit_fails_fast() -> None:
    corpus = _corpus()
    sbc_spec = SBCSpec(n_simulations=FAST_N, draws=200, seed=0, alpha=ALPHA)
    good = sbc_pool(POOL, corpus, sbc_spec=sbc_spec)
    assert isinstance(good, SBCResult)
    assert good.n_failed_fits == 0, good.failure_reasons
    assert {p.name for p in good.parameters} == {"mu_f", "tau_f"}
    assert good.passed, _report("pool", good)
    bad = sbc_pool(POOL, corpus, sbc_spec=sbc_spec, refit=WRONG_POOL)
    assert isinstance(bad, SBCResult)
    assert not bad.passed and "mu_f" in bad.failed_parameters, _report("wrong pool", bad)


# -- slow tier -------------------------------------------------------------------------------


@pytest.mark.slow
def test_surface_sbc_passes_and_wrong_refit_fails_slow() -> None:
    spec, panel = _surface(20, 30)
    sbc_spec = SBCSpec(n_simulations=SLOW_N, draws=200, seed=0, alpha=ALPHA)
    good = sbc_surface(spec, panel, sbc_spec=sbc_spec)
    assert isinstance(good, SBCResult)
    assert good.n_failed_fits == 0, good.failure_reasons
    assert good.passed, _report("surface", good)
    bad = sbc_surface(spec, panel, sbc_spec=sbc_spec, refit=_wrong(spec))
    assert isinstance(bad, SBCResult)
    assert not bad.passed and "beta_rate_a" in bad.failed_parameters, _report("wrong", bad)


@pytest.mark.slow
def test_pool_sbc_passes_and_wrong_refit_fails_slow() -> None:
    corpus = _corpus()
    sbc_spec = SBCSpec(n_simulations=SLOW_N, draws=200, seed=0, alpha=ALPHA)
    good = sbc_pool(POOL, corpus, sbc_spec=sbc_spec)
    assert isinstance(good, SBCResult)
    assert good.n_failed_fits == 0, good.failure_reasons
    assert good.passed, _report("pool", good)
    bad = sbc_pool(POOL, corpus, sbc_spec=sbc_spec, refit=WRONG_POOL)
    assert isinstance(bad, SBCResult)
    assert not bad.passed and "mu_f" in bad.failed_parameters, _report("wrong pool", bad)


@pytest.mark.slow
@pytest.mark.skipif(not _numpyro_available(), reason="numpyro extra not installed")
def test_hill_surface_sbc_passes_under_numpyro_slow() -> None:
    """The saturating surface under an exact sampler: NUTS, one chain of 200 draws per fit."""
    spec, panel = _surface(8, 16, hill=True)
    sbc_spec = SBCSpec(n_simulations=SLOW_N, draws=200, seed=0, alpha=ALPHA, backend="numpyro")
    good = sbc_surface(spec, panel, sbc_spec=sbc_spec)
    assert isinstance(good, SBCResult)
    assert good.n_failed_fits == 0, good.failure_reasons
    assert good.passed, _report("hill/numpyro", good)
