from __future__ import annotations

import warnings

import pytest

from axiom.core import (
    D,
    Dose,
    Entity,
    Intervention,
    Population,
    TimeWindow,
    Treatment,
    UndimensionedWarning,
    dimension_of,
    dimensionless,
)


def test_entity_is_a_protocol_not_a_base_class() -> None:
    t = Treatment(name="x", dimension=D.currency)
    d = Dose(name="d", dimension=D.currency, numeraire="USD")
    assert isinstance(t, Entity) and isinstance(d, Entity)
    assert Treatment.__mro__[1].__name__ == "Spec"  # no intermediate base
    assert dimension_of(t) == D.currency == t.dim


def test_undimensioned_entity_warns_and_is_dimensionless() -> None:
    with pytest.warns(UndimensionedWarning):
        t = Treatment(name="x")
    assert t.dim == dimensionless()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Treatment(name="x", dimension=D.currency).dim == D.currency


def test_names_must_be_identifier_like() -> None:
    with pytest.raises(ValueError):
        Treatment(name="bad name", dimension=D.currency)
    Treatment(name="tv-ads.v2", dimension=D.currency)


def test_population_weights() -> None:
    Population(name="p", strata={"soil": {"a": 0.5, "b": 0.5}})
    with pytest.raises(ValueError):
        Population(name="p", strata={"soil": {"a": 0.6, "b": 0.5}})
    with pytest.raises(ValueError):
        Population(name="p", strata={"soil": {}})


def test_window_and_intervention() -> None:
    w = TimeWindow(start=2, stop=5)
    assert w.length == 3 and w.basis == "cumulative"
    with pytest.raises(ValueError):
        TimeWindow(start=3, stop=3)
    iv = Intervention(doses={"b": 1.0, "a": 2.0}, window=w)
    assert iv.treatments == ("a", "b") and iv.version == "unspecified"
    with pytest.raises(ValueError):
        Intervention(doses={})
    assert Intervention(doses={"a": 1.0}, version="v1") != Intervention(
        doses={"a": 1.0}, version="v2"
    )
