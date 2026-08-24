"""A printed number stops where its uncertainty stops."""

from __future__ import annotations

import math

import numpy as np
import pytest

from axiom.core import (
    Interval,
    Summary,
    decimals_for,
    format_interval,
    format_measured,
    round_to,
    summarize,
)


class TestTheUncertaintyFixesThePlace:
    def test_the_value_rounds_to_two_digits_of_the_uncertainty(self) -> None:
        assert format_measured(12.3456789, 0.52) == "12.35"
        assert format_measured(12.3456789, 5.2) == "12.3"
        assert format_measured(12.3456789, 52.0) == "12"

    def test_a_wide_uncertainty_takes_digits_off_the_left_of_the_point(self) -> None:
        """1234 give or take 500 is known to the nearest ten, and says so."""
        assert decimals_for(500.0) == -1
        assert format_measured(1234.5678, 500.0) == "1230"

    def test_a_value_far_below_its_uncertainty_prints_as_the_zero_it_is(self) -> None:
        assert format_measured(0.003, 5.0) == "0.0"

    def test_a_negative_value_rounded_away_does_not_print_as_minus_zero(self) -> None:
        assert format_measured(-1e-9, 2.5) == "0.0"

    def test_the_sign_survives_when_the_value_does(self) -> None:
        assert format_measured(-12.3456789, 0.52) == "-12.35"

    def test_rounding_the_uncertainty_into_the_next_decade_does_not_add_a_digit(self) -> None:
        """0.0999 to two digits is 0.10, whose last digit sits a place higher."""
        assert decimals_for(0.0999) == 2
        assert format_measured(7.0, 0.0999) == "7.00"

    def test_the_digit_count_is_a_knob(self) -> None:
        assert format_measured(12.3456789, 0.52, digits=1) == "12.3"
        assert format_measured(12.3456789, 0.52, digits=3) == "12.346"

    def test_trailing_zeros_are_kept_because_they_are_the_claim(self) -> None:
        assert format_measured(12.0, 0.05) == "12.000"  # 0.05 to two digits is 0.050


class TestWhenThereIsNothingToRoundAgainst:
    def test_no_uncertainty_falls_back_to_a_plain_g(self) -> None:
        assert format_measured(12.3456789) == "12.3457"

    def test_a_zero_or_non_finite_uncertainty_states_no_resolution(self) -> None:
        assert format_measured(12.3456789, 0.0) == "12.3457"
        assert format_measured(12.3456789, float("nan")) == "12.3457"
        assert format_measured(12.3456789, float("inf")) == "12.3457"

    def test_a_non_finite_value_prints_as_itself(self) -> None:
        assert format_measured(float("nan"), 0.5) == "nan"
        assert format_measured(float("inf"), 0.5) == "inf"

    def test_asking_for_the_place_of_nothing_raises_rather_than_inventing_one(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            decimals_for(0.0)
        with pytest.raises(ValueError, match="finite"):
            decimals_for(float("nan"))
        with pytest.raises(ValueError, match="at least 1"):
            decimals_for(0.5, digits=0)


class TestTheExponentForm:
    def test_a_very_small_number_keeps_the_digits_its_uncertainty_supports(self) -> None:
        assert format_measured(1.2345e-7, 1.1e-9) == "1.235e-07"

    def test_a_very_large_number_does_not_print_ten_placeholder_digits(self) -> None:
        assert format_measured(12345678.9, 1000.0) == "1.23457e+07"

    def test_a_readable_magnitude_stays_in_fixed_notation(self) -> None:
        assert format_measured(1234000.0, 5000.0) == "1234000"
        assert format_measured(1234000.0, 5000.0, group=True) == "1,234,000"


class TestRoundTo:
    def test_it_is_the_numeric_form_of_the_same_rule(self) -> None:
        assert round_to(12.3456789, 0.52) == pytest.approx(12.35)
        assert round_to(1234.5678, 500.0) == pytest.approx(1230.0)

    def test_it_agrees_with_the_printed_form(self) -> None:
        for value, uncertainty in [(12.3456, 0.52), (0.00312345, 9.8e-5), (-4.44, 1.1)]:
            assert float(format_measured(value, uncertainty)) == pytest.approx(
                round_to(value, uncertainty)
            )


class TestAnIntervalIsItsOwnResolution:
    def test_both_bounds_round_to_the_same_place(self) -> None:
        assert format_interval(10.1234, 14.5678) == "[10.1, 14.6]"

    def test_the_half_width_is_what_they_round_against(self) -> None:
        interval = Interval(lower=10.1234, upper=14.5678, definition="hdi", mass=0.9)
        assert interval.half_width == pytest.approx(2.2222)
        assert decimals_for(interval.half_width) == 1

    def test_a_stated_uncertainty_overrides_the_half_width(self) -> None:
        """A caller showing the band beside its sigma prints both at one place."""
        assert format_interval(1.0123, 3.9876, uncertainty=0.7614) == "[1.01, 3.99]"

    def test_a_degenerate_interval_claims_no_resolution(self) -> None:
        assert format_interval(2.5, 2.5) == "[2.5, 2.5]"

    def test_an_interval_prints_its_bounds_its_definition_and_its_mass(self) -> None:
        interval = Interval(lower=10.1234, upper=14.5678, definition="hdi", mass=0.9)
        assert str(interval) == "[10.1, 14.6] (90% HDI)"

    def test_a_summary_stops_where_its_spread_stops(self) -> None:
        interval = Interval(lower=10.1234, upper=14.5678, definition="hdi", mass=0.9)
        summary = Summary(mean=12.3456789, median=12.2987, sd=1.23456, interval=interval, n=4000)
        assert str(summary) == "12.3 ± 1.2 [10.1, 14.6] (90% HDI) from 4000 draws"

    def test_a_summary_of_real_draws_carries_no_arithmetic_digits(self) -> None:
        draws = np.random.default_rng(0).normal(3.0, 1.0, 4000)
        text = str(summarize(draws, definition="hdi", mass=0.9))
        # two digits of the sd, and the mean at that same place — nothing longer
        assert not any(len(token.split(".")[-1]) > 3 for token in text.split() if "." in token)
        assert math.isfinite(float(text.split()[0]))
