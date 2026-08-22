"""A dynamic system's estimability, and the observable that fixes it.

The two halves of this work meet here. `axiom.dynamics` compiles a system with
a latent intermediate state into ordinary expressions; `design.identifiability`
then says which of its parameters the design can separate, and which extra
*observable* would separate the rest. Nothing in either module knows about the
other — the compiled expression is the whole interface.
"""

from __future__ import annotations

import numpy as np

from axiom.core import D, Param, dimensionless
from axiom.design import Observation, estimable_combinations, prescribe_measurements
from axiom.dynamics import Unrolled, Variable, parse_system, time_ref, unroll

NONE = dimensionless()
PERIODS = 8
TRUTH = {"decay": 0.6, "uptake": 2.0, "gain": 3.0}


def two_compartment():
    """A latent compartment fills from the dose and the outcome reads it, scaled.

    ``outcome[t] = gain * sum_k decay^k * uptake * dose[t-k]``, so ``gain`` and
    ``uptake`` only ever appear multiplied together. No amount of outcome data
    separates them; measuring the compartment does.
    """
    return parse_system(
        """
        state   = decay * state[t-1] + uptake * dose
        outcome = gain * state
        """,
        variables=(
            Variable(name="state", dimension=D.currency, observed=False),
            Variable(name="outcome", dimension=D.outcome),
            Variable(name="dose", dimension=D.currency, role="exogenous"),
        ),
        parameters=(
            Param(name="decay", dimension=NONE),
            Param(name="uptake", dimension=NONE),
            Param(name="gain", dimension=D.outcome / D.currency),
        ),
        name="two-compartment",
    )


def doses() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4)
    return {time_ref("dose", t): rng.gamma(2.0, 1.0, size=6) for t in range(PERIODS)}


def observations(compiled: Unrolled, variable: str, label: str) -> list[Observation]:
    """One observation per period of a compiled variable's trajectory."""
    data = doses()
    return [
        Observation(f"{label}@{t}", compiled.expression(variable, t), data, noise_sd=0.1)
        for t in range(1, PERIODS)
    ]


def test_only_the_product_of_gain_and_uptake_is_estimable_from_the_outcome() -> None:
    compiled = unroll(two_compartment(), periods=PERIODS)
    assert isinstance(compiled, Unrolled)
    report = estimable_combinations(
        observations(compiled, "outcome", "outcome"),
        TRUTH,
        at=({"decay": 0.3, "uptake": 1.0, "gain": 1.0}, {"decay": 0.8, "uptake": 4.0, "gain": 0.5}),
    )
    assert report.rank == 2
    assert report.deficiency == 1
    # flat at every parameter point: this is the model, not the dose schedule
    assert report.persistent_deficiency == 1
    assert [c.exponents for c in report.symmetries] == [{"uptake": 1, "gain": -1}]
    rendered = {c.render() for c in report.estimable}
    assert "decay" in rendered
    assert "gain * uptake" in rendered


def test_measuring_the_latent_compartment_separates_them() -> None:
    compiled = unroll(two_compartment(), periods=PERIODS)
    assert isinstance(compiled, Unrolled)
    plan = prescribe_measurements(
        observations(compiled, "outcome", "outcome"),
        observations(compiled, "state", "state")[:1],
        TRUTH,
    )
    assert plan.rank_before == 2
    assert plan.rank_after == 3
    assert plan.complete
    assert plan.added == ("state@1",)
    assert plan.broken == (("uptake / gain",),)


def test_more_outcome_periods_do_not_help() -> None:
    """The distinction that matters: more of the same measurement cannot break a symmetry."""
    compiled = unroll(two_compartment(), periods=PERIODS)
    assert isinstance(compiled, Unrolled)
    plan = prescribe_measurements(
        observations(compiled, "outcome", "outcome")[:3],
        observations(compiled, "outcome", "later outcome")[3:],
        TRUTH,
    )
    assert not plan.complete
    assert plan.added == ()
    assert plan.still_flat == ("uptake / gain",)
