"""Gate: ``‖X @ theta_lin + offset − forward(dose, theta)‖∞ < 1e-12`` over 200 random surfaces.

Random kernels, carryovers, intercept structures, interactions, nuisance
terms, and layouts (a 1-D dose grid, or a units × time panel). The design
matrix comes from evaluating the same tree ``forward`` evaluates with unit
vectors on the linear parameters, so a failure here means the surface mean
is no longer linear in a parameter it claims to be linear in — or that
``linearize`` and ``forward`` have drifted apart (rule 3).

A surface with carryover reads dose arrays as ``(n_units, n_periods)`` and
refuses a 1-D grid; on the grid layout such a surface is evaluated through
``steady_state()``, which this file also checks against the closed form.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from axiom.core import D, Outcome, Param, Treatment
from axiom.surface.carryover import (
    DelayedCarryover,
    GeometricCarryover,
    NoCarryover,
    WeibullCarryover,
)
from axiom.surface.kernels import KERNELS, HillKernel
from axiom.surface.linearize import check_linearization, column_names
from axiom.surface.model import Surface, SurfaceSpec
from axiom.surface.nuisance import FourierSeasonality, LinearTrend, NuisanceSet

TOL = 1e-12
N_CONFIGS = 200
UNITS = ("u0", "u1", "u2")
N_PERIODS = 5
N_RUNS = 6
OUTCOME = Outcome(name="y", dimension=D.outcome)


def _random_spec(rng: random.Random, i: int) -> SurfaceSpec:
    names = ["x1", "x2", "x3"][: rng.randint(1, 3)]
    treatments = tuple(Treatment(name=n, dimension=D.currency, unit="USD") for n in names)
    kernels = {n: KERNELS[rng.choice(sorted(KERNELS))](reference_dose=2.0) for n in names}
    carryover = {}
    for n in names:
        lag = rng.randint(2, 4)
        carryover[n] = rng.choice(
            [
                NoCarryover(),
                GeometricCarryover(max_lag=lag),
                DelayedCarryover(max_lag=lag),
                WeibullCarryover(max_lag=lag),
            ]
        )
    pairs = list(itertools.combinations(names, 2))
    rng.shuffle(pairs)
    interactions = tuple(pairs[: rng.randint(0, len(pairs))])
    intercept = rng.choice(["none", "shared", "per_unit", "hierarchical"])
    terms = []
    if rng.random() < 0.5:
        terms.append(LinearTrend(scale=5.0))
    if rng.random() < 0.5:
        terms.append(FourierSeasonality(period=4.0, order=1))
    return SurfaceSpec(
        name=f"random_{i}",
        treatments=treatments,
        outcome=OUTCOME,
        kernels=kernels,  # type: ignore[arg-type]
        carryover=carryover,  # type: ignore[arg-type]
        interactions=interactions,
        intercept=intercept,  # type: ignore[arg-type]
        unit_labels=UNITS if intercept in ("per_unit", "hierarchical") else (),
        nuisance=NuisanceSet(terms=tuple(terms)),
    )


def _random_value(p: Param, nrng: np.random.Generator) -> np.ndarray | float:
    assert p.prior is not None
    match p.prior.family:
        case "lognormal":
            v = nrng.uniform(1.0, 3.0, size=p.shape)
        case "gamma":
            v = nrng.uniform(0.6, 2.2, size=p.shape)
        case "halfnormal":
            v = nrng.uniform(0.2, 2.0, size=p.shape)
        case "beta":
            v = nrng.uniform(0.2, 0.9, size=p.shape)
        case "uniform":
            lo, hi = float(p.prior.hyper["low"]), float(p.prior.hyper["high"])  # type: ignore[arg-type]
            v = nrng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), size=p.shape)
        case _:
            v = nrng.uniform(-1.5, 1.5, size=p.shape)
    return v if p.shape else float(v)


def _random_dose(
    spec: SurfaceSpec, nrng: np.random.Generator, layout: str
) -> dict[str, np.ndarray]:
    shape = (len(UNITS), N_PERIODS) if layout == "panel" else (N_RUNS,)
    dose: dict[str, np.ndarray] = {
        n: nrng.uniform(0.1, 5.0, size=shape) for n in spec.treatment_names
    }
    for col in spec.nuisance.column_names():
        dose[col] = nrng.uniform(-1.0, 1.0, size=shape)
    if layout == "panel":
        dose[spec.unit_column] = np.arange(len(UNITS))[:, None]
    else:
        dose[spec.unit_column] = nrng.integers(0, len(UNITS), size=N_RUNS)
    return dose


def test_linearization_invariant_over_200_random_surfaces() -> None:
    rng = random.Random(2026)
    nrng = np.random.default_rng(2026)
    worst = 0.0
    checked = 0
    steady_checked = 0
    for i in range(N_CONFIGS):
        spec = _random_spec(rng, i)
        surface = Surface(spec)
        theta = {p.name: _random_value(p, nrng) for p in surface.model.parameters}
        layout = "panel" if (spec.intercept in ("per_unit", "hierarchical") or i % 2) else "arms"
        dose = _random_dose(spec, nrng, layout)
        if layout == "arms" and spec.carried:
            # independent rows: the carryover surface refuses them, the steady state takes them
            with pytest.raises(ValueError, match="steady_state"):
                surface.linearize(dose, theta)
            surface = surface.steady_state()
            steady_checked += 1
        dm = surface.linearize(dose, theta)
        assert dm.columns == column_names(surface.model, surface.linear)
        assert dm.X.shape == (dm.offset.size, len(dm.columns))
        assert np.all(np.isfinite(dm.X)) and np.all(np.isfinite(dm.offset))
        assert set(dm.at) == set(surface.nonlinear)
        resid = check_linearization(surface, dose, theta)
        assert resid < TOL, f"config {i} ({spec.name}): residual {resid:.3e} >= {TOL}"
        worst = max(worst, resid)
        checked += 1
    assert checked == N_CONFIGS
    assert steady_checked > 0
    assert worst < TOL


def test_steady_state_on_a_1d_grid_matches_the_closed_form() -> None:
    """A carryover spec on a 1-D grid: refused as is; exact through ``steady_state()``."""
    spec = SurfaceSpec(
        name="carried",
        treatments=(Treatment(name="x1", dimension=D.currency, unit="USD"),),
        outcome=OUTCOME,
        kernels={"x1": HillKernel(reference_dose=2.0)},
        carryover={"x1": GeometricCarryover(max_lag=4)},
    )
    surface = Surface(spec)
    steady = surface.steady_state()
    x = np.array([4.0, 0.5, 4.0, 0.5, 1.0, 3.0])
    theta = {"alpha": 0.25, "k_a": 0.0, "k_x1": 2.0, "s_x1": 1.5, "beta_x1": 2.0, "lam_x1": 0.6}
    with pytest.raises(ValueError, match="'x1'.*steady_state"):
        surface.forward({"x1": x}, theta)
    with pytest.raises(ValueError, match="'x1'.*steady_state"):
        surface.linearize({"x1": x}, theta)
    u = (x / 2.0) ** 1.5
    closed = 0.25 + 2.0 * u / (1.0 + u)
    np.testing.assert_allclose(steady.forward({"x1": x}, theta), closed, rtol=1e-12)
    assert check_linearization(steady, {"x1": x}, theta) < TOL
    dm = steady.linearize({"x1": x}, theta)
    assert dm.columns == ("alpha", "beta_x1") and set(dm.at) == {"k_x1", "s_x1"}
    np.testing.assert_allclose(dm.X[:, 1], u / (1.0 + u), rtol=1e-12)
    # the same rows held constant for max_lag periods reach that steady state in the last period
    held = {"x1": np.tile(x[:, None], (1, 4))}
    np.testing.assert_allclose(surface.forward(held, theta)[:, -1], closed, rtol=1e-12)
    assert not np.allclose(surface.forward(held, theta)[:, 0], closed)
    assert steady.provenance["steady_state"] is True
    assert steady.provenance["source_spec_hash"] == spec.content_hash()


def test_every_linear_parameter_is_a_column_and_no_nonlinear_one_is() -> None:
    rng = random.Random(7)
    for i in range(20):
        surface = Surface(_random_spec(rng, i))
        roles = surface.roles
        for name in surface.linear:
            assert roles[name] == "linear"
        for name in surface.nonlinear:
            assert roles[name] == "nonlinear"
        assert not set(surface.linear) & set(surface.nonlinear)
        assert set(surface.linear) | set(surface.nonlinear) | set(surface.auxiliary) == set(roles)
