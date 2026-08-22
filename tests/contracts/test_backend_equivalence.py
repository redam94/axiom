"""Phase 8 exit criterion 4: the backends agree on the same Hill surface and data.

Three routes reach a posterior — Laplace, NumPyro NUTS, and PyMC NUTS — and
PyMC can itself dispatch NUTS to four samplers. All of them compile the *same*
``ModelSpec``: ``core.value`` for Laplace, ``core.interpret.jax`` for NumPyro,
``core.interpret.pytensor`` for PyMC. If any two disagree beyond Monte Carlo
noise, one of those three interpreters has drifted from the tree, which is the
failure this gate exists to catch.

This is a **normal-approximation check**. The Laplace posterior is Gaussian
in the unconstrained parameters by construction, so it can only agree with
NUTS where the true posterior is nearly Gaussian — a well-identified world
with enough data. The primary criterion is therefore moment-based:

* ``|mean_laplace − mean_nuts| < 0.25 · sd_nuts`` per parameter, and
* ``sd_laplace / sd_nuts ∈ [0.75, 1.25]``.

The two-sample Kolmogorov–Smirnov test on the marginals is secondary. With
thousands of draws KS has the power to detect the small skew a log-normal
scale or a bounded shape parameter carries, so its p-value is held to the
lenient ``0.01`` and a KS failure on a parameter whose moments agree is
reported as a normal-approximation caveat rather than as disagreement.

The **MCMC routes** are held to a much tighter standard than Laplace, because
they are sampling the same density rather than approximating it: their means
must agree within ``0.15`` posterior sd. A PyMC sampler that is installed but
does not work with the installed PyMC is skipped *by name* — ``samplers(probe=
True)`` decides, so the gate never silently covers less than it appears to.

Slow (NUTS with ``draws=1000, tune=1000, chains=2``); each part skips without
its extra.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats

from axiom.core import Posterior
from axiom.sim import DosePlan, surface_world
from axiom.surface import fit

pytestmark = pytest.mark.slow

MEAN_TOL_SD = 0.25
SD_RATIO = (0.75, 1.25)
KS_ALPHA = 0.01
#: Two exact samplers of the same density may differ only by Monte Carlo noise.
MCMC_TOL_SD = 0.15
MCMC_SD_RATIO = (0.85, 1.18)


@pytest.fixture(scope="module")
def world():  # type: ignore[no-untyped-def]
    """A single-Hill, shared-intercept panel: many units, a long horizon, zero-dose cells,
    low noise."""
    return surface_world(
        n_units=8,
        n_periods=40,
        treatments=("a",),
        intercept="shared",
        doses=DosePlan(scale=50.0, spread=0.8, zero_fraction=0.2),
        noise_sd=0.15,
        seed=808,
    )


def test_laplace_and_numpyro_agree_on_a_well_identified_hill_world(world) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("numpyro")
    t0 = time.perf_counter()
    lap = fit(world.spec, world.panel, backend="laplace", draws=1000, chains=2, seed=1)
    t_lap = time.perf_counter() - t0
    t0 = time.perf_counter()
    nuts = fit(world.spec, world.panel, backend="numpyro", draws=1000, tune=1000, chains=2, seed=2)
    t_nuts = time.perf_counter() - t0
    assert isinstance(lap.posterior, Posterior), lap.posterior
    assert isinstance(nuts.posterior, Posterior), nuts.posterior
    assert lap.converged, lap.posterior.provenance
    assert nuts.report is not None and nuts.report.converged, nuts.report

    names = sorted(lap.posterior.names() & nuts.posterior.names())
    assert set(names) == {p.name for p in world.model.parameters}
    rows: list[str] = []
    failures: list[str] = []
    caveats: list[str] = []
    for name in names:
        a = lap.posterior.flat(name).ravel()
        b = nuts.posterior.flat(name).ravel()
        mean_gap = abs(a.mean() - b.mean()) / b.std(ddof=1)
        ratio = a.std(ddof=1) / b.std(ddof=1)
        ks = stats.ks_2samp(a, b)
        truth = float(np.asarray(world.theta[name]).ravel()[0])
        rows.append(
            f"{name:>10s}: truth={truth:+.4f} laplace={a.mean():+.4f}±{a.std(ddof=1):.4f} "
            f"nuts={b.mean():+.4f}±{b.std(ddof=1):.4f} gap/sd={mean_gap:.3f} "
            f"sd_ratio={ratio:.3f} KS p={ks.pvalue:.3g}"
        )
        if mean_gap >= MEAN_TOL_SD:
            failures.append(f"{name}: mean gap {mean_gap:.3f} sd >= {MEAN_TOL_SD}")
        if not SD_RATIO[0] <= ratio <= SD_RATIO[1]:
            failures.append(f"{name}: sd ratio {ratio:.3f} outside {SD_RATIO}")
        if ks.pvalue <= KS_ALPHA:
            caveats.append(f"{name}: KS p={ks.pvalue:.3g} (normal-approximation caveat)")
    report = "\n".join(rows) + f"\nlaplace {t_lap:.1f}s, numpyro {t_nuts:.1f}s"
    if caveats:
        report += "\nKS caveats: " + "; ".join(caveats)
    print("\n" + report)
    assert not failures, "\n".join(failures) + "\n" + report
    # Secondary: KS must pass on the well-identified structural parameters, which are the
    # ones a backend disagreement would show up in. Nuisance scale parameters (lognormal
    # and half-normal posteriors) are allowed the documented caveat.
    structural = [n for n in names if n.startswith(("beta_", "k_"))]
    ks_failed = [c for c in caveats if c.split(":")[0] in structural]
    assert not ks_failed, "\n".join(ks_failed) + "\n" + report


# -- the exact samplers, against each other -------------------------------------------------


def _moments(posterior, names):  # type: ignore[no-untyped-def]
    return {
        n: (posterior.flat(n).ravel().mean(), posterior.flat(n).ravel().std(ddof=1)) for n in names
    }


def _compare(left, right, names, label, tol, ratio):  # type: ignore[no-untyped-def]
    """Two posteriors of the same model, moment by moment. Returns (report, failures)."""
    a, b = _moments(left, names), _moments(right, names)
    rows, failures = [], []
    for name in names:
        gap = abs(a[name][0] - b[name][0]) / b[name][1]
        sd_ratio = a[name][1] / b[name][1]
        rows.append(
            f"{name:>10s}: {a[name][0]:+.4f}+-{a[name][1]:.4f} vs "
            f"{b[name][0]:+.4f}+-{b[name][1]:.4f}  gap/sd={gap:.3f} sd_ratio={sd_ratio:.3f}"
        )
        if gap >= tol:
            failures.append(f"{label} {name}: mean gap {gap:.3f} sd >= {tol}")
        if not ratio[0] <= sd_ratio <= ratio[1]:
            failures.append(f"{label} {name}: sd ratio {sd_ratio:.3f} outside {ratio}")
    return label + "\n" + "\n".join(rows), failures


def _usable_pymc_samplers() -> list[str]:
    """The PyMC samplers that actually run here, probed rather than assumed."""
    pytest.importorskip("pymc")
    from axiom.infer import samplers

    return [name for name, status in samplers(probe=True).items() if status.usable]


def test_pymc_and_numpyro_sample_the_same_posterior(world) -> None:  # type: ignore[no-untyped-def]
    """The strong form: two exact samplers over two interpreters of one tree."""
    pytest.importorskip("pymc")
    pytest.importorskip("numpyro")
    pm_fit = fit(world.spec, world.panel, backend="pymc", draws=1000, tune=1000, chains=2, seed=3)
    np_fit = fit(
        world.spec, world.panel, backend="numpyro", draws=1000, tune=1000, chains=2, seed=2
    )
    assert isinstance(pm_fit.posterior, Posterior), pm_fit.posterior
    assert isinstance(np_fit.posterior, Posterior), np_fit.posterior
    assert pm_fit.posterior.provenance["backend"] == "pymc"
    assert pm_fit.posterior.provenance["nuts_sampler"] == "pymc"

    names = sorted(pm_fit.posterior.names() & np_fit.posterior.names())
    assert set(names) == {p.name for p in world.model.parameters}
    report, failures = _compare(
        pm_fit.posterior, np_fit.posterior, names, "pymc vs numpyro", MCMC_TOL_SD, MCMC_SD_RATIO
    )
    print("\n" + report)
    assert not failures, "\n".join(failures) + "\n" + report


def test_every_usable_pymc_sampler_gives_the_same_posterior(world) -> None:  # type: ignore[no-untyped-def]
    """The claim the PyMC backend is *for*: the sampler is a speed choice, not a modelling one."""
    usable = _usable_pymc_samplers()
    assert "pymc" in usable, "PyMC's own NUTS must always work when pymc is installed"
    if len(usable) == 1:
        pytest.skip("only PyMC's own sampler is usable here; nothing to compare it against")

    from axiom.infer import PymcBackend

    # ``fit`` takes a Backend instance as well as a name, which is the seam for any
    # backend-specific option; nothing PyMC-shaped leaks into ``fit``'s signature.
    fits = {
        name: fit(
            world.spec,
            world.panel,
            backend=PymcBackend(nuts_sampler=name),
            draws=1000,
            tune=1000,
            chains=2,
            seed=3,
        )
        for name in usable
    }
    for name, result in fits.items():
        assert isinstance(result.posterior, Posterior), (name, result.posterior)
        assert result.posterior.provenance["nuts_sampler"] == name

    reference = fits["pymc"].posterior
    names = sorted({p.name for p in world.model.parameters})
    reports, failures = [], []
    for name in usable:
        if name == "pymc":
            continue
        report, failed = _compare(
            fits[name].posterior, reference, names, f"{name} vs pymc", MCMC_TOL_SD, MCMC_SD_RATIO
        )
        reports.append(report)
        failures += failed
    print("\n" + "\n".join(reports))
    assert not failures, "\n".join(failures) + "\n" + "\n".join(reports)
