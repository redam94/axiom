"""Recovery: a compiled dynamic system, simulated and refit, gets its truth back.

The two compilations are checked against each other and against the truth.
The marginal form generates the data — it is the definition of what the
system *means* over a horizon — and the conditional form is what gets fit,
which is the arrangement any real dynamic panel is in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core import D, Likelihood, Param, Posterior, Prior, dimensionless, value
from axiom.dynamics import (
    Unrolled,
    Variable,
    conditional_form,
    parse_system,
    prepare_panel,
    to_model_spec,
    unroll,
)
from axiom.infer import LaplaceBackend

pytestmark = pytest.mark.recovery

NONE = dimensionless()
TRUTH = {"decay": 0.6, "beta": 2.5, "sigma": 0.25}


def carryover_system():
    """``stock[t] = decay * stock[t-1] + beta * inflow[t]`` — one compartment, one input."""
    return parse_system(
        "stock = decay * stock[t-1] + beta * inflow",
        variables=(
            Variable(name="stock", dimension=D.outcome),
            Variable(name="inflow", dimension=D.currency, role="exogenous"),
        ),
        parameters=(
            Param(name="decay", dimension=NONE),
            Param(name="beta", dimension=D.outcome / D.currency),
        ),
        name="one-compartment",
    )


def simulate(periods: int, units: int, seed: int) -> pd.DataFrame:
    """Draw inflows, run the marginal form, add noise. The truth is TRUTH."""
    system = carryover_system()
    compiled = unroll(system, periods=periods)
    assert isinstance(compiled, Unrolled)
    rng = np.random.default_rng(seed)
    inflow = rng.gamma(shape=2.0, scale=1.0, size=(units, periods))
    data = {f"inflow.t{t}": inflow[:, t] for t in range(periods)}
    rows = []
    for t in range(periods):
        mean = np.asarray(value(compiled.expression("stock", t), data=data, params=TRUTH), float)
        observed = mean + rng.normal(0.0, TRUTH["sigma"], size=units)
        for unit in range(units):
            rows.append(
                {"unit": f"u{unit}", "t": t, "inflow": inflow[unit, t], "stock": observed[unit]}
            )
    return pd.DataFrame(rows)


def fitted_model():
    compiled = conditional_form(carryover_system())
    assert isinstance(compiled, Unrolled)
    return to_model_spec(
        compiled,
        "stock",
        likelihood=Likelihood(family="normal", scale="sigma"),
        priors={
            "decay": Prior(family="beta", hyper={"alpha": 1.5, "beta": 1.5}),
            "beta": Prior(family="lognormal", hyper={"mu": 0.0, "sigma": 1.5}),
        },
        extra_parameters=(
            Param(
                name="sigma",
                dimension=D.outcome,
                prior=Prior(family="halfnormal", hyper={"sigma": 2.0}),
            ),
        ),
        name="one-compartment-conditional",
    )


def test_the_two_compilations_agree_on_noiseless_data() -> None:
    """The conditional form fed the true lagged values reproduces the marginal one."""
    system = carryover_system()
    marginal = unroll(system, periods=6)
    conditional = conditional_form(system)
    assert isinstance(marginal, Unrolled)
    assert isinstance(conditional, Unrolled)
    rng = np.random.default_rng(0)
    inflow = rng.gamma(2.0, 1.0, size=6)
    wide = {f"inflow.t{t}": np.array(inflow[t]) for t in range(6)}
    trajectory = [
        float(np.ravel(value(marginal.expression("stock", t), data=wide, params=TRUTH))[0])
        for t in range(6)
    ]
    lagged = np.array([0.0, *trajectory[:-1]])
    one_step = np.asarray(
        value(
            conditional.expression("stock"),
            data={"inflow": inflow, "stock.l1": lagged},
            params=TRUTH,
        ),
        dtype=float,
    )
    assert np.allclose(one_step, trajectory)


def test_the_conditional_fit_recovers_the_truth() -> None:
    frame = prepare_panel(carryover_system(), simulate(30, 12, seed=7), unit="unit", time="t")
    model = fitted_model()
    data = {name: frame[name].to_numpy() for name in ("stock", "stock.l1", "inflow")}
    posterior = LaplaceBackend().sample(model, data, draws=800, tune=0, chains=1, seed=3)
    assert isinstance(posterior, Posterior)
    for name in ("decay", "beta"):
        draws = posterior.flat(name)
        mean, sd = float(np.mean(draws)), float(np.std(draws, ddof=1))
        assert abs(mean - TRUTH[name]) < 4.0 * sd, f"{name}: {mean:.3f} vs {TRUTH[name]}"
        assert abs(mean - TRUTH[name]) / TRUTH[name] < 0.15


def test_a_design_with_no_carryover_to_see_cannot_recover_the_decay() -> None:
    """A constant inflow makes the decay and the amplitude one number, not two.

    ``stock`` converges to ``beta * inflow / (1 - decay)``; only that ratio is
    in the data. This is the identifiability failure the design analysis names,
    and it is a property of the *schedule*, so it belongs in a recovery test.
    """
    from axiom.design import Observation, estimable_combinations

    system = carryover_system()
    compiled = conditional_form(system)
    assert isinstance(compiled, Unrolled)
    steady = {"inflow": np.full(40, 2.0), "stock.l1": np.full(40, 2.5 * 2.0 / (1 - 0.6))}
    report = estimable_combinations(
        [Observation("constant inflow", compiled.expression("stock"), steady, 0.25)],
        {"decay": 0.6, "beta": 2.5},
        tolerance=1e-8,
    )
    assert report.rank == 1
    assert report.deficiency == 1
