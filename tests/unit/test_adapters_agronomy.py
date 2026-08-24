"""``axiom.adapters.agronomy``: trial frames → ``Panel``, the preset, the domain estimands,
and the economic optimum — including the case where the trial did not bracket it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.adapters import agronomy
from axiom.core import D, Unsupported, dimensionless
from axiom.estimands import EstimandResult
from axiom.surface import FitResult, fit

LEVELS = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0)
ASYMPTOTE, SCALE, FLOOR = 5.2, 70.0, 3.0


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for plot in range(28):
        for season in range(3):
            n = float(LEVELS[(plot + 2 * season) % len(LEVELS)])
            rows.append(
                {
                    "plot": f"p{plot:02d}",
                    "season": season,
                    "grain": FLOOR + ASYMPTOTE * (1.0 - np.exp(-n / SCALE)) + rng.normal(0.0, 0.30),
                    "nitrogen": n,
                    "soil_carbon": float(rng.normal(1.4, 0.2)),
                    "unused": "text",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def roles() -> agronomy.TrialRoles:
    return agronomy.TrialRoles(
        harvest="grain", nutrients=("nitrogen",), soil_tests=("soil_carbon",)
    )


@pytest.fixture
def fitted(roles: agronomy.TrialRoles) -> FitResult:
    panel = agronomy.panel_from_trial(_frame(), roles)
    spec = agronomy.trial_spec(panel, amplitude_scale=5.0).model_copy(
        update={"intercept_scale": 4.0, "noise_scale": 1.0}
    )
    return fit(spec, panel, backend="laplace", draws=800, seed=5)


# -- vocabulary ---------------------------------------------------------------------------------


def test_a_rate_and_a_yield_share_a_dimension_so_efficiency_is_dimensionless() -> None:
    rate = agronomy.nutrient_rate("nitrogen")
    assert rate.dimension == agronomy.MASS_PER_AREA == agronomy.MASS / agronomy.AREA
    assert rate.dimension is not None
    assert rate.dimension / agronomy.MASS_PER_AREA == dimensionless()


def test_declaring_a_base_twice_is_idempotent() -> None:
    assert agronomy.MASS == D.mass
    assert agronomy.AREA == D.area


def test_a_costed_dose_carries_a_numeraire_and_an_uncosted_one_does_not() -> None:
    assert agronomy.nutrient_rate("n", price=2.2).numeraire == "USD"
    assert agronomy.nutrient_rate("n").numeraire is None


def test_a_negative_price_is_refused_rather_than_absorbed() -> None:
    with pytest.raises(ValueError, match="price must be positive"):
        agronomy.nutrient_rate("nitrogen", price=-1.0)


def test_a_soil_test_is_dimensionless_because_it_cannot_honestly_be_anything_else() -> None:
    assert agronomy.soil_test("ph").dimension == dimensionless()


# -- roles and loading --------------------------------------------------------------------------


def test_roles_translate_to_a_role_map_with_the_right_entities(
    roles: agronomy.TrialRoles,
) -> None:
    mapped = agronomy.role_map(roles)
    assert mapped.unit == "plot" and mapped.time == "season"
    assert mapped.outcome[1].dimension == agronomy.MASS_PER_AREA
    assert mapped.outcome[1].unit == "t/ha"
    assert mapped.treatments["nitrogen"].unit == "kg/ha"
    assert mapped.covariates["soil_carbon"].dimension == dimensionless()


def test_a_column_cannot_play_two_roles() -> None:
    with pytest.raises(ValueError, match="more than one role"):
        agronomy.TrialRoles(harvest="grain", nutrients=("grain",))


def test_loading_drops_unnamed_columns_and_names_missing_ones(
    roles: agronomy.TrialRoles,
) -> None:
    panel = agronomy.panel_from_trial(_frame(), roles)
    assert "unused" not in panel.frame.columns
    assert panel.completeness().n_units == 28
    with pytest.raises(ValueError, match=r"lacks columns.*soil_carbon"):
        agronomy.panel_from_trial(_frame().drop(columns=["soil_carbon"]), roles)


# -- the preset ---------------------------------------------------------------------------------


def test_the_preset_defaults_to_mitscherlich_and_scales_on_the_mean_applied_rate(
    roles: agronomy.TrialRoles,
) -> None:
    panel = agronomy.panel_from_trial(_frame(), roles)
    spec = agronomy.trial_spec(panel)
    assert spec.kernels["nitrogen"].name == "exponential"
    applied = [level for level in LEVELS if level > 0]
    assert spec.kernels["nitrogen"].reference_dose == pytest.approx(np.mean(applied), rel=0.05)
    assert spec.intercept == "shared"
    assert not spec.carryover


def test_carryover_is_opt_in_and_an_unknown_nutrient_is_named(
    roles: agronomy.TrialRoles,
) -> None:
    panel = agronomy.panel_from_trial(_frame(), roles)
    assert agronomy.trial_spec(panel, residual_carryover=True).carryover["nitrogen"].name == (
        "geometric"
    )
    with pytest.raises(ValueError, match=r"potassium.*not treatment columns"):
        agronomy.trial_spec(panel, nutrients=("potassium",))


# -- domain quantities --------------------------------------------------------------------------


def test_the_response_is_per_hectare_per_season_not_summed_over_plots(
    fitted: FitResult,
) -> None:
    out = agronomy.response_to(fitted, "nitrogen", seed=1)
    assert isinstance(out, EstimandResult)
    # The truth is the mean of the response over the cells, not the response at the mean
    # rate: the curve is concave, so those differ by a full tonne a hectare here. Every
    # level appears equally often in the rotation, so the cell mean is the level mean.
    per_level = [ASYMPTOTE * (1.0 - np.exp(-n / SCALE)) for n in LEVELS]
    truth = float(np.mean(per_level))
    at_mean_rate = ASYMPTOTE * (1.0 - np.exp(-float(np.mean(LEVELS)) / SCALE))
    assert truth < at_mean_rate  # Jensen, and the estimand follows the truth
    assert out.summary.interval.lower < truth < out.summary.interval.upper
    assert out.unit == "t/ha"
    assert out.detail["level"] == "individual" and out.detail["basis"] == "per_period"


def test_agronomic_efficiency_is_dimensionless(fitted: FitResult) -> None:
    out = agronomy.agronomic_efficiency(fitted, "nitrogen", seed=1)
    assert isinstance(out, EstimandResult)
    assert out.dimension == dimensionless()
    assert out.summary.mean > 0.0


def test_the_marginal_product_is_below_the_average_on_a_concave_curve(
    fitted: FitResult,
) -> None:
    marginal = agronomy.marginal_product(fitted, "nitrogen", seed=1)
    average = agronomy.agronomic_efficiency(fitted, "nitrogen", seed=1)
    assert isinstance(marginal, EstimandResult) and isinstance(average, EstimandResult)
    assert marginal.summary.mean < average.summary.mean


def test_the_elasticity_is_dimensionless_and_between_zero_and_one(fitted: FitResult) -> None:
    out = agronomy.nutrient_elasticity(fitted, "nitrogen", seed=1)
    assert isinstance(out, EstimandResult)
    assert out.dimension == dimensionless()
    assert 0.0 < out.summary.mean < 1.0


def test_every_quantity_declines_when_the_fit_has_no_posterior(
    roles: agronomy.TrialRoles,
) -> None:
    panel = agronomy.panel_from_trial(_frame(), roles)
    thin = agronomy.trial_spec(panel, intercept="hierarchical", amplitude_scale=5.0).model_copy(
        update={"intercept_scale": 4.0, "noise_scale": 1.0}
    )
    attempt = fit(thin, panel, backend="laplace", draws=200, seed=5)
    for quantity in (agronomy.response_to, agronomy.agronomic_efficiency):
        assert isinstance(quantity(attempt, "nitrogen"), Unsupported)


# -- the economic optimum -----------------------------------------------------------------------


def test_prices_give_the_break_even_marginal_product() -> None:
    assert agronomy.Prices(harvest=220.0, nutrient=2.2).ratio == pytest.approx(0.01)


def test_the_optimum_recovers_the_rate_the_truth_implies(fitted: FitResult) -> None:
    prices = agronomy.Prices(harvest=220.0, nutrient=2.20)
    got = agronomy.economic_optimum(fitted, "nitrogen", prices, seed=2)
    assert isinstance(got, agronomy.EconomicOptimum)
    # dY/dN = (A / k) exp(-N / k) = ratio  ->  N = k log(A / (k ratio))
    truth = SCALE * np.log(ASYMPTOTE / (SCALE * prices.ratio))
    assert got.lower < truth < got.upper
    assert got.bracketed
    assert got.lower <= got.rate <= got.upper
    assert got.expected_gain > 0.0
    assert got.profit_over_zero == pytest.approx(
        prices.harvest * got.expected_gain - prices.nutrient * got.rate
    )


def test_a_cheaper_nutrient_moves_the_optimum_up(fitted: FitResult) -> None:
    dear = agronomy.economic_optimum(
        fitted, "nitrogen", agronomy.Prices(harvest=220.0, nutrient=2.20), seed=2
    )
    cheap = agronomy.economic_optimum(
        fitted, "nitrogen", agronomy.Prices(harvest=220.0, nutrient=1.10), seed=2
    )
    assert isinstance(dear, agronomy.EconomicOptimum) and isinstance(
        cheap, agronomy.EconomicOptimum
    )
    assert cheap.rate > dear.rate


def test_an_optimum_past_the_trials_rates_is_unsupported_not_extrapolated(
    fitted: FitResult,
) -> None:
    got = agronomy.economic_optimum(
        fitted, "nitrogen", agronomy.Prices(harvest=220.0, nutrient=0.20), seed=2
    )
    assert isinstance(got, Unsupported)
    assert "did not apply enough" in got.reason
    assert got.missing == ("dose_range",)
    assert got.detail["searched_high"] == "270"


def test_the_optimum_round_trips_and_orders_its_bounds() -> None:
    one = agronomy.EconomicOptimum(
        nutrient="nitrogen",
        rate=138.0,
        lower=131.0,
        upper=145.0,
        expected_gain=4.45,
        prices=agronomy.Prices(harvest=220.0, nutrient=2.2),
        price_ratio=0.01,
        definition="eti",
        mass=0.9,
        searched=(0.0, 270.0),
        bracketed=True,
    )
    assert agronomy.EconomicOptimum.from_json(one.to_json()) == one
    with pytest.raises(ValueError, match="exceeds upper"):
        one.model_copy(update={"lower": 200.0}).model_validate({**one.to_dict(), "lower": 200.0})
