"""Estimable combinations, profile likelihood, and what to measure next."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import (
    Add,
    Apply,
    Data,
    Div,
    Likelihood,
    ModelSpec,
    Mul,
    Param,
    Prior,
    Unsupported,
    dimensionless,
)
from axiom.design import (
    Combination,
    EstimabilityReport,
    Observation,
    Prescription,
    ProfileReport,
    SensitivityMatrix,
    estimable_combinations,
    observation_from_model,
    prescribe_measurements,
    profile_combination,
    profile_likelihood,
    sensitivity_matrix,
    simulated_identifiability,
)

NONE = dimensionless()
DOSE = Data(name="dose", dimension=NONE)


def hill() -> Mul:
    """``alpha * dose / (k + dose)``."""
    return Mul(
        factors=(
            Param(name="alpha", dimension=NONE),
            Div(numerator=DOSE, denominator=Add(terms=(Param(name="k", dimension=NONE), DOSE))),
        )
    )


def product() -> Mul:
    """``a * b * dose`` — only the product is ever estimable."""
    return Mul(factors=(Param(name="a", dimension=NONE), Param(name="b", dimension=NONE), DOSE))


def low_dose() -> Observation:
    return Observation("low dose", hill(), {"dose": np.linspace(0.01, 0.3, 12)}, 0.05)


def wide_dose() -> Observation:
    return Observation("wide dose", hill(), {"dose": np.linspace(0.5, 60.0, 12)}, 0.05)


def hill_model() -> ModelSpec:
    return ModelSpec(
        name="hill",
        mean=hill(),
        outcome=Data(name="y", dimension=NONE),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(
            Param(
                name="alpha",
                dimension=NONE,
                prior=Prior(family="lognormal", hyper={"mu": 1.0, "sigma": 1.0}),
            ),
            Param(
                name="k",
                dimension=NONE,
                prior=Prior(family="lognormal", hyper={"mu": 2.0, "sigma": 1.0}),
            ),
            Param(
                name="sigma", dimension=NONE, prior=Prior(family="halfnormal", hyper={"sigma": 1.0})
            ),
        ),
    )


# -- sensitivity --------------------------------------------------------------------


def test_the_log_sensitivity_is_the_analytic_derivative() -> None:
    doses = np.array([0.5, 2.0, 8.0])
    got = sensitivity_matrix([Observation("d", hill(), {"dose": doses})], {"alpha": 4.0, "k": 10.0})
    assert isinstance(got, SensitivityMatrix)
    assert got.parameters == ("alpha", "k")
    assert got.rows == 3
    # d/dlog alpha of alpha x/(k+x) is the mean itself; d/dlog k is -k x alpha/(k+x)^2
    mean = 4.0 * doses / (10.0 + doses)
    assert np.allclose(got.matrix[:, 0], mean, rtol=1e-6)
    assert np.allclose(got.matrix[:, 1], -10.0 * doses * 4.0 / (10.0 + doses) ** 2, rtol=1e-5)


def test_a_parameter_the_mean_does_not_use_is_reported_as_an_exact_zero() -> None:
    expr = Mul(factors=(Param(name="alpha", dimension=NONE), DOSE))
    got = sensitivity_matrix(
        [Observation("d", expr, {"dose": np.array([1.0, 2.0])})],
        {"alpha": 2.0, "unused": 3.0},
    )
    assert isinstance(got, SensitivityMatrix)
    assert got.zero_columns == ("unused",)
    assert np.all(got.matrix[:, 1] == 0.0)


def test_log_scaling_needs_positive_parameters_and_says_which_are_not() -> None:
    got = sensitivity_matrix(
        [Observation("d", product(), {"dose": np.array([1.0])})], {"a": 0.0, "b": 1.0}
    )
    assert isinstance(got, Unsupported)
    assert got.missing == ("a",)
    assert "positive" in got.reason


def test_absolute_scaling_needs_a_scale_for_a_parameter_at_zero() -> None:
    got = sensitivity_matrix(
        [Observation("d", product(), {"dose": np.array([1.0])})],
        {"a": 0.0, "b": 1.0},
        scaling="absolute",
    )
    assert isinstance(got, Unsupported)
    assert got.missing == ("a",)
    got = sensitivity_matrix(
        [Observation("d", product(), {"dose": np.array([1.0])})],
        {"a": 0.0, "b": 1.0},
        scaling="absolute",
        parameter_scales={"a": 1.0},
    )
    assert isinstance(got, SensitivityMatrix)


def test_an_observation_needs_a_label_and_a_positive_noise() -> None:
    with pytest.raises(ValueError, match="label"):
        Observation(" ", hill(), {})
    with pytest.raises(ValueError, match="noise_sd"):
        Observation("d", hill(), {}, 0.0)


def test_an_observation_can_be_taken_from_a_model_spec() -> None:
    observation = observation_from_model(
        hill_model(), {"dose": np.array([1.0])}, theta={"alpha": 1.0, "k": 1.0, "sigma": 0.25}
    )
    assert observation.noise_sd == 0.25
    assert observation.label == "hill"


# -- estimable combinations ----------------------------------------------------------


def test_only_the_product_is_estimable_when_only_the_product_appears() -> None:
    observations = [Observation("any", product(), {"dose": np.linspace(0.1, 10.0, 20)})]
    report = estimable_combinations(
        observations, {"a": 2.0, "b": 3.0}, at=({"a": 0.5, "b": 8.0}, {"a": 5.0, "b": 0.2})
    )
    assert isinstance(report, EstimabilityReport)
    assert report.rank == 1
    assert report.deficiency == 1
    # flat at every parameter point tried, and no design can break it: a * b is
    # all the model ever exposes
    assert report.persistent_deficiency == 1
    assert [c.render() for c in report.estimable] == ["a * b"]
    assert [c.exponents for c in report.symmetries] == [{"a": 1, "b": -1}]
    assert "a -> a*c" in report.symmetries[0].describe()
    assert not report.identified


def test_the_low_dose_design_estimates_only_the_ratio_at_a_practical_tolerance() -> None:
    report = estimable_combinations(
        [low_dose()],
        {"alpha": 4.0, "k": 10.0},
        at=({"alpha": 1.0, "k": 3.0}, {"alpha": 9.0, "k": 25.0}),
        tolerance=0.03,
    )
    assert isinstance(report, EstimabilityReport)
    assert report.rank == 1
    assert [c.render() for c in report.estimable] == ["alpha / k"]
    assert [c.exponents for c in report.symmetries] == [{"alpha": 1, "k": 1}]
    # Flat at every one of the three parameter points -- so not an accident of the
    # values. It is still the *design*, not the model: see the prescription test,
    # where one wide-dose measurement separates them.
    assert report.persistent_deficiency == 1
    assert report.n_points == 3
    assert "alpha / k" in report.summary()


def test_the_same_design_is_full_rank_at_machine_tolerance() -> None:
    report = estimable_combinations([low_dose()], {"alpha": 4.0, "k": 10.0})
    assert isinstance(report, EstimabilityReport)
    assert report.identified
    assert report.condition_number > 100
    assert sorted(c.render() for c in report.estimable) == ["alpha", "k"]


def test_a_wide_design_separates_the_parameters() -> None:
    report = estimable_combinations([wide_dose()], {"alpha": 4.0, "k": 10.0}, tolerance=0.03)
    assert isinstance(report, EstimabilityReport)
    assert report.identified
    assert report.condition_number < 20


def test_the_report_records_the_spectrum_and_the_tolerance_it_used() -> None:
    report = estimable_combinations([low_dose()], {"alpha": 4.0, "k": 10.0}, tolerance=0.03)
    assert isinstance(report, EstimabilityReport)
    assert len(report.singular_values) == 2
    assert report.tolerance == 0.03
    assert report.n_points == 1
    assert len(report.null_basis) == report.deficiency


def test_a_search_that_names_nothing_says_so_rather_than_returning_empty() -> None:
    report = estimable_combinations(
        [low_dose()], {"alpha": 4.0, "k": 10.0}, tolerance=0.03, max_coefficient=1, max_support=1
    )
    assert isinstance(report, EstimabilityReport)
    assert report.deficiency == 1
    assert report.symmetries == ()
    assert any("flat directions" in limit for limit in report.search_limits_hit)


def test_absolute_scaling_gives_linear_combinations() -> None:
    expr = Add(terms=(Param(name="a", dimension=NONE), Param(name="b", dimension=NONE)))
    observations = [Observation("sum", expr, {})]
    report = estimable_combinations(observations, {"a": 1.0, "b": 2.0}, scaling="absolute")
    assert isinstance(report, EstimabilityReport)
    assert report.rank == 1
    assert report.symmetries[0].exponents in ({"a": 1, "b": -1}, {"a": -1, "b": 1})
    assert "->" in report.symmetries[0].describe()


def test_a_combination_needs_a_non_zero_exponent() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        Combination(exponents={"a": 0}, kind="estimable", score=0.0)


def test_rendering_covers_products_quotients_and_pure_denominators() -> None:
    assert Combination(exponents={"a": 1, "b": -1}, kind="estimable", score=0.0).render() == "a / b"
    assert (
        Combination(exponents={"a": 2, "b": 1}, kind="estimable", score=0.0).render() == "a^2 * b"
    )
    assert Combination(exponents={"a": -1}, kind="estimable", score=0.0).render() == "1 / a"


# -- what to measure next ------------------------------------------------------------


def test_the_prescription_names_the_measurement_that_breaks_the_symmetry() -> None:
    plan = prescribe_measurements(
        [low_dose()], [wide_dose()], {"alpha": 4.0, "k": 10.0}, tolerance=0.03
    )
    assert isinstance(plan, Prescription)
    assert plan.added == ("wide dose",)
    assert plan.rank_before == 1
    assert plan.rank_after == 2
    assert plan.complete
    assert plan.broken == (("alpha * k",),)
    assert plan.still_flat == ()


def test_a_candidate_that_cannot_help_leaves_the_prescription_incomplete() -> None:
    observations = [Observation("any", product(), {"dose": np.linspace(0.1, 10.0, 20)})]
    candidate = [Observation("more of the same", product(), {"dose": np.linspace(20.0, 30.0, 5)})]
    plan = prescribe_measurements(observations, candidate, {"a": 2.0, "b": 3.0})
    assert isinstance(plan, Prescription)
    assert not plan.complete
    assert plan.added == ()
    assert plan.still_flat == ("a / b",)
    assert plan.considered == ("more of the same",)


# -- practical identifiability -------------------------------------------------------


def test_a_wide_design_recovers_both_parameters_with_finite_intervals() -> None:
    report = simulated_identifiability(
        hill_model(),
        {"dose": np.linspace(0.5, 60.0, 24)},
        {"alpha": 4.0, "k": 10.0, "sigma": 0.02},
        targets=["alpha", "k"],
        seed=3,
        n_grid=13,
    )
    assert isinstance(report, ProfileReport)
    assert report.identified
    for name in ("alpha", "k"):
        low, high = report.interval[name]
        assert np.isfinite(low) and np.isfinite(high)
        assert low <= report.recovered[name] <= high
        assert (high - low) / report.recovered[name] < 0.05
        assert abs(report.recovered[name] - report.truth[name]) / report.truth[name] < 0.05


def test_a_low_dose_design_leaves_each_parameter_unbounded_but_pins_the_ratio() -> None:
    ratio = Combination(exponents={"alpha": 1, "k": -1}, kind="estimable", score=0.0)
    report = simulated_identifiability(
        hill_model(),
        {"dose": np.linspace(0.01, 0.3, 24)},
        {"alpha": 4.0, "k": 10.0, "sigma": 0.002},
        targets=["alpha"],
        combinations=[ratio],
        seed=3,
        range_factor=50.0,
        n_grid=13,
    )
    assert isinstance(report, ProfileReport)
    assert "alpha" in report.flat
    assert not np.isfinite(report.interval["alpha"][1])
    low, high = report.interval["alpha / k"]
    assert np.isfinite(low) and np.isfinite(high)
    assert high - low < 0.05
    assert report.truth["alpha / k"] == pytest.approx(0.4)


def test_a_profile_rises_away_from_the_optimum() -> None:
    model = hill_model()
    data = {"dose": np.linspace(0.5, 60.0, 24)}
    truth = {"alpha": 4.0, "k": 10.0, "sigma": 0.05}
    rng = np.random.default_rng(0)
    from axiom.core import value

    data["y"] = np.asarray(value(model.mean, data=data, params=truth)) + rng.normal(0, 0.05, 24)
    grid = [2.0, 4.0, 8.0]
    values, drops = profile_likelihood(model, data, truth, "alpha", grid=grid)
    assert values == tuple(grid)
    assert drops[1] < drops[0] and drops[1] < drops[2]
    assert min(drops) >= 0.0


def test_profiling_an_unknown_target_names_it() -> None:
    with pytest.raises(KeyError, match="wobble"):
        profile_likelihood(hill_model(), {}, {}, "wobble", grid=[1.0])


def test_only_a_log_combination_can_be_held_fixed() -> None:
    absolute = Combination(exponents={"alpha": 1}, kind="estimable", scaling="absolute", score=0.0)
    with pytest.raises(ValueError, match="monomial"):
        profile_combination(hill_model(), {}, {}, absolute, grid=[1.0])


def test_a_non_normal_likelihood_is_not_simulated_silently() -> None:
    model = hill_model().model_copy(update={"likelihood": Likelihood(family="poisson")})
    out = simulated_identifiability(model, {"dose": np.array([1.0])}, {"alpha": 1.0, "k": 1.0})
    assert isinstance(out, Unsupported)
    assert "poisson" in out.reason


def test_simulation_without_a_noise_scale_says_what_is_missing() -> None:
    out = simulated_identifiability(
        hill_model(), {"dose": np.array([1.0])}, {"alpha": 1.0, "k": 1.0}
    )
    assert isinstance(out, Unsupported)
    assert out.missing == ("sigma",)


def test_a_target_that_is_not_a_free_parameter_is_an_error() -> None:
    with pytest.raises(KeyError):
        simulated_identifiability(
            hill_model(),
            {"dose": np.array([1.0, 2.0])},
            {"alpha": 1.0, "k": 1.0, "sigma": 0.1},
            targets=["nope"],
        )


def test_a_point_where_the_observation_is_not_finite_is_reported_not_returned() -> None:
    from axiom.core import Const

    # 1 / (a - 1) at a = 1: the observation itself is infinite, which is a fact about
    # the model at this point and not something to take a derivative of.
    expr = Div(
        numerator=Const(value=1.0, dimension=NONE),
        denominator=Add(terms=(Param(name="a", dimension=NONE), Const(value=-1.0, dimension=NONE))),
    )
    with np.errstate(divide="ignore"):
        out = sensitivity_matrix([Observation("d", expr, {})], {"a": 1.0})
    assert isinstance(out, Unsupported)
    assert "not finite" in out.reason


def test_a_step_lost_to_round_off_is_reported_as_an_exact_zero() -> None:
    expr = Apply(fn="log", arg=Param(name="a", dimension=NONE))
    out = sensitivity_matrix([Observation("d", expr, {})], {"a": 1e-320})
    assert isinstance(out, SensitivityMatrix)
    assert out.zero_columns == ("a",)
