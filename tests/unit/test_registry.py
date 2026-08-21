from __future__ import annotations

import itertools
import math

import pytest
from _factories import _estimand

from axiom.core import D, Intervention, Outcome, Population, Spec, TimeWindow, Treatment
from axiom.estimands import Estimand, Level, Quantity, derived_dimension
from axiom.estimands.registry import (
    STANDARD_ESTIMAND_NAMES,
    EstimandRegistry,
    standard_estimands,
)

_BANNED = {"channel", "spend", "roas", "roi", "kpi", "geo", "dma", "impression", "media", "brand"}


def _marginal() -> Estimand:
    return _estimand(
        name="slope_at_100",
        quantity=Quantity(kind="marginal"),
        reference=None,
        dimension=D.outcome / D.currency,
    )


# -- registry -----------------------------------------------------------------------------


def test_register_returns_hash_and_keeps_order() -> None:
    reg = EstimandRegistry()
    e, m = _estimand(), _marginal()
    assert reg.register(e) == e.content_hash()
    assert reg.register(m) == m.content_hash()
    assert reg.names() == ("lift_at_100", "slope_at_100")
    assert len(reg) == 2 and "lift_at_100" in reg and "nope" not in reg
    assert list(reg) == [e, m]
    assert reg.get("slope_at_100") == m
    assert reg.by_hash(e.content_hash()) == e
    assert reg.hashes() == {"lift_at_100": e.content_hash(), "slope_at_100": m.content_hash()}


def test_duplicates_and_overwrite() -> None:
    reg = EstimandRegistry([_estimand()])
    # the identical spec again is a no-op
    assert reg.register(_estimand()) == _estimand().content_hash()
    assert len(reg) == 1
    changed = _estimand(intervention=Intervention(doses={"fertilizer": 150.0}, version="granular"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(changed)
    assert reg.get("lift_at_100") == _estimand()  # untouched after the refusal
    assert reg.register(changed, overwrite=True) == changed.content_hash()
    assert reg.get("lift_at_100") == changed and len(reg) == 1


def test_lookup_failures_name_what_is_registered() -> None:
    reg = EstimandRegistry([_estimand()])
    with pytest.raises(KeyError, match="lift_at_100"):
        reg.get("missing")
    with pytest.raises(KeyError, match="content hash"):
        reg.by_hash("0" * 64)
    with pytest.raises(KeyError, match="missing"):
        reg.remove("missing")
    assert reg.remove("lift_at_100") == _estimand() and len(reg) == 0


def test_round_trip_through_specs_and_json() -> None:
    reg = EstimandRegistry.from_specs([_estimand(), _marginal()])
    specs = reg.to_specs()
    assert isinstance(specs, tuple) and len(specs) == 2
    back = EstimandRegistry.from_specs([Spec.from_json(s.to_json()) for s in specs])  # type: ignore[misc]
    assert back == reg
    assert back.hashes() == reg.hashes()
    assert back != EstimandRegistry.from_specs([_marginal()])
    with pytest.raises(ValueError, match="already registered"):
        EstimandRegistry.from_specs([_estimand(), _estimand(window=TimeWindow(start=0, stop=12))])


def test_equality_is_order_sensitive() -> None:
    """A registry round-trips as a tuple, so its identity includes registration order;
    ``hashes()`` is the order-free comparison."""
    e, m = _estimand(), _marginal()
    forward, backward = EstimandRegistry([e, m]), EstimandRegistry([m, e])
    assert forward == EstimandRegistry([e, m])
    assert forward != backward
    assert forward.to_specs() == backward.to_specs()[::-1]
    assert forward.hashes() == backward.hashes()
    assert forward.names() == ("lift_at_100", "slope_at_100")
    assert backward.names() == ("slope_at_100", "lift_at_100")
    # re-registering an existing name keeps its original position
    backward.register(m)
    assert backward.names() == ("slope_at_100", "lift_at_100")
    # overwriting under an existing name keeps the position as well
    changed = _estimand(intervention=Intervention(doses={"fertilizer": 150.0}, version="granular"))
    backward.register(changed, overwrite=True)
    assert backward.names() == ("slope_at_100", "lift_at_100")
    assert backward != EstimandRegistry([m, e])
    # not comparable to other things, and not hashable
    assert forward != ("lift_at_100", "slope_at_100")
    assert forward.__eq__(object()) is NotImplemented
    with pytest.raises(TypeError):
        hash(forward)


# -- the standard library -----------------------------------------------------------------


def _standard(**over: object) -> EstimandRegistry:
    kw: dict[str, object] = dict(
        treatment=Treatment(name="fertilizer", dimension=D.currency, unit="USD"),
        outcome=Outcome(name="yield_total", dimension=D.outcome, unit="kg"),
        population=Population(name="north"),
        window=TimeWindow(start=0, stop=8),
        level=Level(unit="cluster"),
        dose=100.0,
    )
    kw.update(over)
    return standard_estimands(**kw)  # type: ignore[arg-type]


def test_standard_estimands_cover_the_five_kinds_with_derived_dimensions() -> None:
    reg = _standard()
    assert reg.names() == STANDARD_ESTIMAND_NAMES
    kinds = {e.name: e.quantity.kind for e in reg}
    assert kinds == {
        "contrast_at_dose": "contrast",
        "marginal_at_dose": "marginal",
        "average_response_ratio": "ratio",
        "elasticity_at_dose": "elasticity",
        "area_under_response": "area",
    }
    for e in reg:
        assert e.dimension == derived_dimension(e.quantity.kind, D.outcome, D.currency)
        assert e.intervention == Intervention(doses={"fertilizer": 100.0})
        if e.quantity.kind in ("contrast", "ratio", "area"):
            assert e.reference == Intervention(doses={"fertilizer": 0.0})
        else:
            assert e.reference is None
        assert e.conditioning == () and e.description
        assert not (_BANNED & set(e.description.lower().replace(",", " ").split()))
        assert not (_BANNED & set(e.name.split("_")))
        assert Spec.from_json(e.to_json()) == e


def test_standard_estimands_options_and_refusals() -> None:
    reg = _standard(reference_dose=20.0, version="liquid")
    c = reg.get("contrast_at_dose")
    assert c.reference is not None and c.reference.doses == {"fertilizer": 20.0}
    assert c.intervention.version == "liquid" and c.reference.version == "liquid"
    assert "20" in c.description and "100" in c.description
    with pytest.raises(ValueError, match="coincide"):
        _standard(reference_dose=100.0)
    with pytest.warns(UserWarning, match="without a dimension"):
        bare = Treatment(name="fertilizer")
    with pytest.raises(ValueError, match="must carry dimensions"):
        _standard(treatment=bare)
    # a negative reference and a dose below it are allowed: the library does not order them
    down = _standard(dose=50.0, reference_dose=100.0)
    assert down.get("contrast_at_dose").intervention.doses == {"fertilizer": 50.0}


def test_standard_estimands_refuse_non_finite_doses() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="dose must be finite"):
            _standard(dose=bad)
        with pytest.raises(ValueError, match="reference_dose must be finite"):
            _standard(reference_dose=bad)
    # the check precedes the coincidence check: nan never "coincides" silently
    with pytest.raises(ValueError, match="finite"):
        _standard(dose=math.nan, reference_dose=math.nan)


def test_standard_estimands_refuse_a_zero_dose_because_of_the_elasticity() -> None:
    with pytest.raises(ValueError, match="elasticity_at_dose.*degenerate") as info:
        _standard(dose=0.0, reference_dose=100.0)
    assert "choose a non-zero dose" in str(info.value)
    with pytest.raises(ValueError, match="degenerate"):
        _standard(dose=-0.0, reference_dose=50.0)
    # a zero REFERENCE dose is the default and fine: the elasticity is evaluated at ``dose``
    assert _standard(reference_dose=0.0).get("elasticity_at_dose").intervention.doses == {
        "fertilizer": 100.0
    }


def test_standard_estimands_satisfy_the_transfer_invariants() -> None:
    """Gate 11's property: no pair transfers as a silent ``identified`` unless facets are equal."""
    reg = _standard()
    for s, t in itertools.product(reg, repeat=2):
        plan = s.transfer_to(t)
        if s == t:
            assert plan.status == "identified" and not plan.differing and not plan.ledger_lines
            continue
        assert plan.status != "identified"
        assert "quantity" in plan.differing
        assert plan.status == "blocked" and "different functional" in plan.reason
        assert len(plan.ledger_lines) == len(plan.differing)
        assert {line.kind for line in plan.ledger_lines} == {f"facet:{f}" for f in plan.differing}
    # the same estimand at another dose is downgraded, never identified
    other = _standard(dose=200.0)
    plan = reg.get("contrast_at_dose").transfer_to(other.get("contrast_at_dose"))
    assert plan.status == "downgraded" and plan.differing == ("intervention",)
    assert {a.name for a in plan.assumptions} == {"surface_correct_between_doses"}
