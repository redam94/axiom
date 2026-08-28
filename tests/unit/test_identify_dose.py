from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core import Unverified
from axiom.identify import (
    EXCLUSION,
    UNIFORM_FIRST_STAGE,
    DerivativeReport,
    FirstStage,
    first_stage,
    local_derivative,
    response_to_dose,
)

_SLOPE = 0.8


def _world(*, lift: float = 2.0, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Assignment adds a heterogeneous amount of dose; the outcome responds per unit."""
    rng = np.random.default_rng(seed)
    assigned = (rng.random(n) < 0.5).astype(float)
    base = rng.gamma(2.0, 1.5, n)  # what a unit would take anyway
    dose = base + lift * assigned * rng.random(n)
    outcome = 10.0 + _SLOPE * dose + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"y": outcome, "assigned": assigned, "dose": dose, "base": base})


# -- the first stage ------------------------------------------------------------------------


def test_the_first_stage_is_the_shift_in_mean_exposure() -> None:
    stage = first_stage(_world(), "dose", "assigned")
    assert isinstance(stage, FirstStage)
    assert stage.shift == pytest.approx(stage.mean_assigned - stage.mean_control)
    assert stage.shift == pytest.approx(1.0, abs=0.1)
    assert stage.n == 4000 and stage.n_assigned + stage.n_control == stage.n


def test_the_standardized_shift_is_the_number_to_read() -> None:
    """A raw shift means nothing without the exposure's own spread."""
    stage = first_stage(_world(), "dose", "assigned")
    assert stage.standardized_shift == pytest.approx(stage.shift / stage.sd_exposure)
    assert 0.3 < stage.standardized_shift < 0.6
    thin = first_stage(_world(lift=0.02), "dose", "assigned")
    assert abs(thin.standardized_shift) < abs(stage.standardized_shift) / 10


def test_a_constant_exposure_has_no_standardized_shift() -> None:
    frame = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "assigned": [1.0, 1.0, 0.0, 0.0], "dose": [2.0] * 4}
    )
    stage = first_stage(frame, "dose", "assigned")
    assert stage.sd_exposure == 0.0
    assert np.isnan(stage.standardized_shift)
    assert stage.shift == 0.0


def test_the_first_stage_refuses_what_it_cannot_read() -> None:
    frame = _world()
    with pytest.raises(ValueError, match="same column"):
        first_stage(frame, "assigned", "assigned")
    with pytest.raises(KeyError, match="not in frame"):
        first_stage(frame, "nope", "assigned")
    with pytest.raises(ValueError, match="units in both arms"):
        first_stage(frame.assign(assigned=1.0), "dose", "assigned")
    with pytest.raises(ValueError, match="must be 0/1"):
        first_stage(frame, "assigned", "dose")


# -- the derivative -------------------------------------------------------------------------


def test_the_derivative_recovers_the_per_unit_slope() -> None:
    result = local_derivative(_world(), "y", "dose", "assigned")
    assert not isinstance(result, Unverified)
    assert result.method == "2sls" and result.treatment == "dose"
    assert result.estimate == pytest.approx(_SLOPE, abs=0.08)


def test_the_itt_and_the_derivative_are_on_different_scales() -> None:
    """``itt = derivative x shift`` — the same arithmetic as the binary case."""
    report = response_to_dose(_world(), "y", "assigned", "dose")
    assert isinstance(report, DerivativeReport)
    assert report.derivative is not None
    assert report.itt.estimate == pytest.approx(report.derivative.estimate * report.shift, rel=1e-6)
    assert report.itt.treatment == "assigned"  # per assignment
    assert report.derivative.treatment == "dose"  # per unit of dose


def test_covariates_reach_both_estimands() -> None:
    frame = _world()
    report = response_to_dose(frame, "y", "assigned", "dose", covariates=("base",))
    assert report.itt.covariates == ("base",)
    assert report.derivative is not None and report.derivative.covariates == ("base",)


# -- what licenses it -----------------------------------------------------------------------


def test_the_derivative_is_downgraded_and_never_identified() -> None:
    report = response_to_dose(_world(), "y", "assigned", "dose")
    verdict = report.verdict()
    assert verdict.status == "downgraded" and verdict.route == "2sls"
    assert [a.name for a in verdict.assumptions] == [
        "exclusion_restriction",
        "uniform_first_stage",
        "instrument_strength",
    ]
    assert "per unit of" in verdict.reason
    assert EXCLUSION.state == "unverified" and UNIFORM_FIRST_STAGE.state == "unverified"


def test_the_uniform_first_stage_is_monotonicitys_continuous_cousin() -> None:
    assert "same direction" in UNIFORM_FIRST_STAGE.statement
    assert "continuous analogue of a defier" in UNIFORM_FIRST_STAGE.challenged_by
    stage = first_stage(_world(), "dose", "assigned")
    assumption = stage.monotonicity()
    assert assumption.name == "uniform_first_stage"
    assert float(assumption.detail["shift"]) == pytest.approx(stage.shift, rel=1e-5)


def test_a_thin_first_stage_blocks_on_instrument_strength() -> None:
    report = response_to_dose(_world(lift=0.01, n=600, seed=3), "y", "assigned", "dose")
    assert report.verdict().status == "blocked"
    assert "instrument_strength" in report.verdict().reason


def test_an_exposure_the_assignment_never_moved_has_no_derivative() -> None:
    frame = _world()
    still = frame.assign(dose=frame["base"], y=frame["y"])
    # force an exactly-zero shift: the guard before 2SLS
    still = still.assign(dose=1.0 + (still.index % 2 == 0).astype(float) * 0.0)
    report = response_to_dose(still, "y", "assigned", "dose")
    assert report.derivative is None
    assert report.unverified is not None
    assert "did not move mean" in report.unverified.reason
    assert report.verdict().status == "blocked"


# -- the readout ----------------------------------------------------------------------------


def test_the_summary_says_what_each_number_is_per() -> None:
    summary = response_to_dose(_world(), "y", "assigned", "dose").summary()
    assert "per               being assigned" in summary
    assert "per               one unit of 'dose'" in summary
    assert "first stage" in summary


def test_the_ledger_names_the_first_stage_and_both_estimands() -> None:
    report = response_to_dose(_world(), "y", "assigned", "dose")
    lines = report.ledger()
    assert [line.kind for line in lines] == ["first_stage", "derivative_estimands"]
    assert lines[0].assumption is not None
    assert lines[0].assumption.name == "uniform_first_stage"
    assert lines[1].assumption == EXCLUSION
    assert "two estimands on two scales" in lines[1].statement


def test_a_report_carries_either_a_derivative_or_the_reason_there_is_none() -> None:
    report = response_to_dose(_world(), "y", "assigned", "dose")
    broken = report.model_copy(update={"unverified": Unverified(reason="both")})
    with pytest.raises(ValueError, match="either a derivative or the reason"):
        DerivativeReport.model_validate(broken.to_dict())


def test_the_report_round_trips_and_validates_its_level() -> None:
    report = response_to_dose(_world(), "y", "assigned", "dose")
    assert DerivativeReport.from_json(report.to_json()) == report
    with pytest.raises(ValueError, match="mass must be in"):
        response_to_dose(_world(), "y", "assigned", "dose", mass=0.0)
