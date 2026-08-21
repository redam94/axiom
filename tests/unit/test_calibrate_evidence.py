"""``Measurement`` and ``combine_inverse_variance``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from axiom.calibrate.evidence import Measurement, combine_inverse_variance
from axiom.core import (
    Assumption,
    D,
    Intervention,
    Outcome,
    Population,
    TimeWindow,
    Treatment,
    load_spec,
    wald,
)
from axiom.estimands import Estimand, Level, Quantity

A = Treatment(name="a", dimension=D.currency, unit="USD")
Y = Outcome(name="y", dimension=D.outcome, unit="count")


def estimand(**over: Any) -> Estimand:
    base: dict[str, Any] = dict(
        name="lift",
        quantity=Quantity(kind="contrast"),
        treatment=A,
        intervention=Intervention(doses={"a": 5.0}),
        reference=Intervention(doses={"a": 0.0}),
        outcome=Y,
        population=Population(name="all"),
        window=TimeWindow(start=0, stop=8),
        level=Level(unit="individual"),
        dimension=D.outcome,
    )
    base.update(over)
    return Estimand(**base)


def measurement(**over: Any) -> Measurement:
    base: dict[str, Any] = dict(estimand=estimand(), estimate=2.5, se=0.4, source="study-1")
    base.update(over)
    return Measurement(**base)


def test_measurement_fields_and_interval() -> None:
    m = measurement(
        method="difference_in_differences",
        n_units=40,
        n_periods=8,
        assumptions=(Assumption(name="pt", facet="method", statement="parallel trends"),),
    )
    assert m.definition == "wald" and m.mass == 0.95
    assert m.interval == wald(2.5, 0.4, 0.95)
    assert m.precision == pytest.approx(1 / 0.16)
    assert m.target == estimand().content_hash()
    assert m.design_factor is None
    posterior = measurement(definition="hdi", mass=0.9)
    assert posterior.interval.definition == "hdi"
    assert posterior.interval.mass == 0.9
    assert posterior.interval.lower == pytest.approx(wald(2.5, 0.4, 0.9).lower)


def test_measurement_round_trips_and_hashes() -> None:
    m = measurement(n_units=12, design_factor=3.0)
    again = load_spec(m.to_json())
    assert again == m
    assert again.content_hash() == m.content_hash()
    assert measurement(estimate=2.6).content_hash() != m.content_hash()


@pytest.mark.parametrize(
    "bad",
    [
        dict(se=0.0),
        dict(se=-1.0),
        dict(se=math.inf),
        dict(estimate=math.nan),
        dict(mass=1.0),
        dict(mass=0.0),
        dict(n_units=0),
        dict(n_periods=-3),
        dict(design_factor=0.0),
        dict(source=""),
        dict(source="   "),
        dict(definition="bayes"),
    ],
)
def test_measurement_validation(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        measurement(**bad)


def test_combine_inverse_variance_closed_form() -> None:
    mean, se = combine_inverse_variance([2.5, 3.1, 1.9], [0.4, 0.9, 0.3])
    w = 1 / np.array([0.4, 0.9, 0.3]) ** 2
    assert mean == pytest.approx(float(np.sum(w * [2.5, 3.1, 1.9]) / w.sum()), rel=1e-14)
    assert se == pytest.approx(float(np.sqrt(1 / w.sum())), rel=1e-14)
    assert se < 0.3  # pooling never loses precision against the best study


def test_combine_single_is_identity_and_equal_ses_average() -> None:
    assert combine_inverse_variance([1.7], [0.2]) == (1.7, 0.2)
    mean, se = combine_inverse_variance([1.0, 3.0], [0.5, 0.5])
    assert mean == pytest.approx(2.0)
    assert se == pytest.approx(0.5 / math.sqrt(2))


@pytest.mark.parametrize(
    ("targets", "ses"),
    [([], []), ([1.0, 2.0], [0.1]), ([1.0], [0.0]), ([1.0], [-0.1]), ([math.nan], [0.1])],
)
def test_combine_rejects_malformed(targets: list[float], ses: list[float]) -> None:
    with pytest.raises(ValueError):
        combine_inverse_variance(targets, ses)
