"""Phase 8 exit criterion 4: Laplace and NumPyro NUTS agree on the same Hill surface and data.

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

Slow (NUTS with ``draws=1000, tune=1000, chains=2``); skipped without the
``numpyro`` extra.
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
