"""Phase 2 recovery: the route the graph licenses recovers the truth; the naive one does not.

Every case has a negative control (a world where the estimator must fail).
Tolerances are in standard errors of the estimator at the simulated n, so
the tests are sharp but not flaky at fixed seeds.
"""

from __future__ import annotations

import pytest

from axiom.identify import (
    durbin_wu_hausman,
    frontdoor_linear,
    identify,
    ols,
    two_stage_least_squares,
    weak_instrument_check,
)
from axiom.sim import (
    LinearSCM,
    confounded_world,
    feedback_world,
    frontdoor_world,
    hidden_confounder_world,
    iv_world,
    mediator_world,
)

pytestmark = pytest.mark.recovery
N = 20_000


def _z(est: float, truth: float, se: float) -> float:
    return abs(est - truth) / se


def test_backdoor_route_recovers_and_naive_is_biased() -> None:
    world = confounded_world()
    truth = world.total_effect("X", "Y")
    frame = world.observed(world.simulate(N, seed=0))
    v = identify(world.graph, "X", "Y")
    assert v.status == "identified" and v.route == "backdoor" and v.adjustment_set == ("Z",)
    naive = ols(frame, "Y", "X")
    adjusted = ols(frame, "Y", "X", covariates=v.adjustment_set)
    assert _z(naive.estimate, truth, naive.se) > 5
    assert _z(adjusted.estimate, truth, adjusted.se) < 3
    assert adjusted.ci(0.95).contains(truth) and adjusted.ci(0.95).definition == "wald"


def test_hidden_confounder_is_downgraded_not_estimated() -> None:
    world = hidden_confounder_world()
    v = identify(world.graph, "X", "Y")
    assert v.status == "downgraded" and v.route == "backdoor_unmeasured"
    assert v.unmeasured_required == ("Z",)
    assert any(
        a.name == "unmeasured_adjustment_set" and a.state == "unverified"
        for a in v.verdict.assumptions
    )
    assert not v.alternatives


def test_instrument_route_recovers_and_ols_is_biased() -> None:
    world = iv_world()
    truth = world.total_effect("X", "Y")
    frame = world.observed(world.simulate(N, seed=1))
    v = identify(world.graph, "X", "Y")
    assert v.route == "instrument" and v.instrument == "Z" and v.status == "downgraded"
    assert {a.name for a in v.verdict.assumptions} >= {
        "exclusion",
        "relevance",
        "effect_homogeneity_or_monotonicity",
    }
    naive = ols(frame, "Y", "X")
    iv = two_stage_least_squares(frame, "Y", "X", instruments=[v.instrument])
    assert _z(naive.estimate, truth, naive.se) > 5
    assert _z(iv.estimate, truth, iv.se) < 3
    assert weak_instrument_check(iv).state == "satisfied"
    dwh = durbin_wu_hausman(frame, "Y", "X", instruments=["Z"])
    assert getattr(dwh, "conclusion", None) == "endogenous"


def test_frontdoor_route_recovers_and_negative_control_fails() -> None:
    world = frontdoor_world()
    truth = world.total_effect("X", "Y")
    frame = world.observed(world.simulate(N, seed=2))
    v = identify(world.graph, "X", "Y")
    assert v.route == "frontdoor" and v.mediators == ("M",) and v.status == "identified"
    fd = frontdoor_linear(frame, "Y", "X", mediators=v.mediators)
    naive = ols(frame, "Y", "X")
    assert _z(fd.estimate, truth, fd.se) < 3
    assert _z(naive.estimate, truth, naive.se) > 5
    # negative control: a latent between X and M violates the front-door criterion
    broken = LinearSCM.from_text("X -> M: 1.2, M -> Y: 1.5, X <-> Y: 1.0, X <-> M: 1.0")
    assert identify(broken.graph, "X", "Y").status != "identified"
    bframe = broken.observed(broken.simulate(N, seed=3))
    bad = frontdoor_linear(bframe, "Y", "X", mediators=["M"])
    assert _z(bad.estimate, broken.total_effect("X", "Y"), bad.se) > 5


def test_mediator_total_vs_direct() -> None:
    world = mediator_world()
    frame = world.simulate(N, seed=4)
    v = identify(world.graph, "X", "Y")
    assert v.route == "backdoor" and v.adjustment_set == ()
    total = ols(frame, "Y", "X")
    direct = ols(frame, "Y", "X", covariates=["M"])
    assert _z(total.estimate, world.total_effect("X", "Y"), total.se) < 3
    assert _z(direct.estimate, world.direct_effect("X", "Y"), direct.se) < 3
    assert _z(direct.estimate, world.total_effect("X", "Y"), direct.se) > 5


def test_feedback_flag_downgrades_with_carryover() -> None:
    world = feedback_world()
    assert identify(world.graph, "X", "Y").status == "identified"
    v = identify(world.graph, "X", "Y", has_carryover=True)
    assert v.status == "downgraded"
    assert any(a.name == "no_time_varying_confounding" for a in v.verdict.assumptions)
