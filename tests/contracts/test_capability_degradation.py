"""Gate 7: a producer lacking a capability yields ``status="unsupported"`` with a reason.

For every ``Capability`` an estimand can require, construct a fitted
producer with that capability removed, request an estimand that needs it,
and assert a typed ``Unsupported`` with a non-empty reason — never a number,
never an exception.
"""

from __future__ import annotations

import pytest

from axiom.core import Capability, Intervention, Population, TimeWindow, Unsupported, is_failure
from axiom.estimands import Estimand, Level, Quantity, realize
from axiom.sim import arms_world, surface_world
from axiom.surface import fit

REQUIRES: dict[Capability, dict[str, object]] = {
    Capability.COUNTERFACTUAL: {"quantity": Quantity(kind="contrast")},
    Capability.MARGINAL: {"quantity": Quantity(kind="marginal"), "reference": None},
    Capability.TIME_WINDOW: {
        "quantity": Quantity(kind="contrast"),
        "window": TimeWindow(start=2, stop=6),
    },
    Capability.PER_UNIT: {"quantity": Quantity(kind="contrast"), "level": Level(unit="individual")},
}


def _estimand(world, **over):  # type: ignore[no-untyped-def]

    spec = world.spec
    t = spec.treatments[0]
    kind = over.get("quantity", Quantity(kind="contrast")).kind
    dim = {"contrast": spec.outcome.dim, "marginal": spec.outcome.dim / t.dim}[kind]
    base = dict(
        name="probe",
        quantity=Quantity(kind="contrast"),
        treatment=t,
        intervention=Intervention(doses={t.name: 60.0}),
        reference=Intervention(doses={t.name: 0.0}),
        outcome=spec.outcome,
        population=Population(name="all"),
        window=TimeWindow(start=0, stop=world.n_periods),
        level=Level(unit="aggregate"),
        dimension=dim,
    )
    base.update(over)
    return Estimand(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("capability", sorted(REQUIRES), ids=lambda c: c.value)
def test_missing_capability_is_unsupported_not_a_number(capability: Capability) -> None:
    world = surface_world(n_units=3, n_periods=12, treatments=("a",), seed=0)
    res = fit(world.spec, world.panel, backend="laplace", draws=200, seed=0)
    assert not is_failure(res.posterior), res.posterior
    full = realize(_estimand(world, **REQUIRES[capability]), res)
    assert not isinstance(full, Unsupported), f"baseline realization failed: {full}"
    crippled = res.restrict([capability])
    assert capability not in crippled.capabilities()
    out = realize(_estimand(world, **REQUIRES[capability]), crippled)
    assert isinstance(out, Unsupported), f"expected Unsupported, got {type(out).__name__}"
    assert out.reason.strip() and capability.value in out.missing


def test_every_capability_is_either_required_somewhere_or_documented() -> None:
    covered = set(REQUIRES)
    documented_only = {Capability.LOG_LIKELIHOOD, Capability.PREDICTIVE}
    assert covered | documented_only == set(Capability), set(Capability) - covered - documented_only


def test_arms_world_producer_reports_capabilities() -> None:
    world = arms_world(n_units=10, treatments=("dose",), seed=1)
    res = fit(world.spec, world.panel, backend="laplace", draws=100, seed=0)
    caps = res.capabilities()
    assert {Capability.COUNTERFACTUAL, Capability.MARGINAL} <= caps
