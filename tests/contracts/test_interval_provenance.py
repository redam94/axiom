"""Gate 6: every interval carries its definition and mass.

Constructive half: ``core.Interval`` cannot be built without ``definition``
and ``mass`` (no defaults). Runtime half: every realized ``EstimandResult``
in the standard registry carries a ``Summary`` whose ``Interval`` has both,
and no public function annotated as returning an interval returns anything
else.
"""

from __future__ import annotations

import inspect
import typing

import pytest
from pydantic import ValidationError

import axiom.core
import axiom.estimands
import axiom.identify
import axiom.infer
import axiom.surface
from axiom.core import Interval, Summary


def test_interval_requires_definition_and_mass() -> None:
    assert Interval.model_fields["definition"].is_required()
    assert Interval.model_fields["mass"].is_required()
    with pytest.raises(ValidationError):
        Interval(lower=0.0, upper=1.0)  # type: ignore[call-arg]
    assert Summary.model_fields["interval"].is_required()


def _public_functions() -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for mod in (axiom.core, axiom.estimands, axiom.identify, axiom.infer, axiom.surface):
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            if inspect.isfunction(obj):
                out.append((f"{mod.__name__}.{name}", obj))
    return out


def test_functions_that_return_intervals_return_the_interval_type() -> None:
    """Any annotated return mentioning 'interval' or 'ci' must be ``Interval`` (or carry one)."""
    offenders = []
    for qualname, fn in _public_functions():
        try:
            hints = typing.get_type_hints(fn)
        except Exception as e:  # typing resolution can fail on forward refs; name it
            raise AssertionError(f"{qualname}: cannot resolve type hints: {e}") from e
        ret = hints.get("return")
        if ret is None:
            continue
        text = str(ret)
        if ("tuple[float, float]" in text) and (
            "interval" in qualname.lower() or qualname.endswith(("eti", "hdi", "wald", "ci"))
        ):
            offenders.append(f"{qualname} -> {text}")
    assert not offenders, f"interval-returning functions must return core.Interval: {offenders}"


def test_every_realized_estimand_carries_an_interval() -> None:
    from axiom.estimands import EstimandResult, evaluate, standard_estimands
    from axiom.sim import arms_world
    from axiom.surface import fit

    world = arms_world(n_units=24, treatments=("dose",), seed=0)
    res = fit(world.spec, world.panel, backend="laplace", draws=400, seed=0)
    spec = world.spec
    registry = standard_estimands(
        treatment=spec.treatments[0],
        outcome=spec.outcome,
        population=axiom.core.Population(name="all"),
        window=axiom.core.TimeWindow(start=0, stop=1),
        level=axiom.estimands.Level(unit="individual"),
        dose=float(world.panel.frame["dose"].median()),
    )
    results = evaluate(list(registry), res, definition="hdi", mass=0.9)
    assert results, "registry realized nothing"
    realized = [r for r in results.values() if isinstance(r, EstimandResult)]
    assert realized, f"no estimand realized: {results}"
    for r in realized:
        assert isinstance(r.summary.interval, Interval)
        assert r.summary.interval.definition == "hdi" and r.summary.interval.mass == 0.9
