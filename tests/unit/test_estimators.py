from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from axiom.core import Assumption, Interval, Unverified, clopper_pearson
from axiom.identify import endogeneity
from axiom.identify.endogeneity import EndogeneityTest, durbin_wu_hausman, hausman_iv_vs_ols
from axiom.identify.estimators import (
    LeastSquaresFit,
    LinearEstimate,
    design_matrix,
    frontdoor_linear,
    least_squares,
    ols,
    two_stage_least_squares,
    weak_instrument_check,
)
from axiom.sim.scm import LinearSCM
from axiom.sim.worlds import confounded_world, frontdoor_world, iv_world, mediator_world

N = 20_000


@pytest.fixture(scope="module")
def confounded() -> pd.DataFrame:
    return confounded_world().simulate(N, seed=0)


@pytest.fixture(scope="module")
def iv() -> pd.DataFrame:
    return iv_world().simulate(N, seed=0)


@pytest.fixture(scope="module")
def frontdoor() -> pd.DataFrame:
    return frontdoor_world().simulate(N, seed=0)


def _within(est: LinearEstimate, truth: float, k: float = 3.0) -> bool:
    return abs(est.estimate - truth) < k * est.se


def _as_test(result: EndogeneityTest | Unverified) -> EndogeneityTest:
    assert isinstance(result, EndogeneityTest), result
    return result


# -- OLS / back-door --------------------------------------------------------------------


def test_ols_naive_is_biased_and_adjusted_recovers(confounded: pd.DataFrame) -> None:
    naive = ols(confounded, "Y", "X")
    adjusted = ols(confounded, "Y", "X", covariates=["Z"])
    truth = confounded_world().total_effect("X", "Y")
    assert abs(naive.z(truth)) > 5.0
    assert naive.estimate == pytest.approx(2.0 + 1.5 * 0.8 / 1.64, abs=0.05)
    assert _within(adjusted, truth)
    assert adjusted.method == "ols" and adjusted.covariates == ("Z",)
    assert adjusted.n == N and adjusted.treatment == "X" and adjusted.outcome == "Y"
    assert set(adjusted.detail) == {"sigma", "r_squared", "df_resid"}
    assert adjusted.detail["sigma"] == pytest.approx(1.0, abs=0.03)


def test_ols_matches_closed_form() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    y = 1.0 + 3.0 * x + rng.normal(size=200)
    frame = pd.DataFrame({"x": x, "y": y})
    est = ols(frame, "y", "x")
    design = np.column_stack([np.ones(200), x])
    beta = np.linalg.solve(design.T @ design, design.T @ y)
    resid = y - design @ beta
    cov = (resid @ resid / 198) * np.linalg.inv(design.T @ design)
    assert est.estimate == pytest.approx(beta[1])
    assert est.se == pytest.approx(np.sqrt(cov[1, 1]))


def test_mediator_world_total_vs_direct() -> None:
    frame = mediator_world().simulate(N, seed=2)
    total = ols(frame, "Y", "X")
    direct = ols(frame, "Y", "X", covariates=["M"])
    assert _within(total, 1.5) and _within(direct, 1.0)
    assert abs(total.z(1.0)) > 5.0


def test_ci_is_a_wald_interval_containing_truth(confounded: pd.DataFrame) -> None:
    est = ols(confounded, "Y", "X", covariates=["Z"])
    ci = est.ci(0.95)
    assert isinstance(ci, Interval)
    assert ci.definition == "wald" and ci.mass == 0.95
    assert ci.contains(2.0)
    assert ci.width == pytest.approx(2 * 1.959964 * est.se, rel=1e-5)


def test_linear_estimate_round_trips(confounded: pd.DataFrame) -> None:
    est = ols(confounded, "Y", "X", covariates=["Z"])
    assert LinearEstimate.from_json(est.to_json()) == est


def test_linear_estimate_validates_se_and_n_and_z() -> None:
    base = dict(estimate=1.0, se=0.5, n=10, method="ols", treatment="X", outcome="Y")
    with pytest.raises(ValidationError, match="se"):
        LinearEstimate(**{**base, "se": -0.1})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="se"):
        LinearEstimate(**{**base, "se": float("inf")})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="se"):
        LinearEstimate(**{**base, "se": float("nan")})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="n"):
        LinearEstimate(**{**base, "n": 0})  # type: ignore[arg-type]
    est = LinearEstimate(**base)  # type: ignore[arg-type]
    assert est.z(0.0) == pytest.approx(2.0)
    assert est.z(1.0) == 0.0
    exact = LinearEstimate(**{**base, "se": 0.0})  # type: ignore[arg-type]
    assert exact.z(1.0) == 0.0
    assert exact.z(0.0) == np.inf and exact.z(2.0) == -np.inf


# -- errors --------------------------------------------------------------------------------


def test_missing_columns_raise_key_error(confounded: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="W"):
        ols(confounded, "Y", "X", covariates=["W"])
    with pytest.raises(KeyError, match="Q"):
        ols(confounded, "Q", "X")
    with pytest.raises(KeyError, match="Q"):
        two_stage_least_squares(confounded, "Y", "X", instruments=["Q"])
    with pytest.raises(KeyError, match="Q"):
        frontdoor_linear(confounded, "Y", "X", mediators=["Q"])
    with pytest.raises(KeyError, match="Q"):
        durbin_wu_hausman(confounded, "Y", "X", instruments=["Q"])


def test_too_few_rows_and_bad_inputs_raise_value_error(confounded: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n > k"):
        ols(confounded.head(3), "Y", "X")
    with pytest.raises(ValueError, match="n > k"):
        ols(confounded.head(4), "Y", "X", covariates=["Z"])
    with pytest.raises(ValueError, match="rank deficient"):
        ols(confounded.assign(W=confounded["X"] * 2), "Y", "X", covariates=["W"])
    with pytest.raises(ValueError, match="non-finite"):
        ols(confounded.assign(X=confounded["X"].where(confounded.index > 0)), "Y", "X")
    with pytest.raises(ValueError, match="instrument"):
        two_stage_least_squares(confounded, "Y", "X", instruments=[])
    with pytest.raises(ValueError, match="distinct"):
        two_stage_least_squares(confounded, "Y", "X", instruments=["X"])
    with pytest.raises(ValueError, match="mediator"):
        frontdoor_linear(confounded, "Y", "X", mediators=[])
    with pytest.raises(ValueError, match="distinct"):
        ols(confounded, "Y", "X", covariates=["X"])
    with pytest.raises(ValueError, match="repeated"):
        design_matrix(confounded, ["X", "X"])
    with pytest.raises(ValueError, match="n > k"):
        least_squares(np.ones((3, 2)), np.ones(3))


def test_outcome_must_be_distinct_from_every_other_role(confounded: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="'Y' as y and covariates"):
        ols(confounded, "Y", "X", covariates=["Y"])
    with pytest.raises(ValueError, match="'Y' as y and x"):
        ols(confounded, "Y", "Y")
    with pytest.raises(ValueError, match="'Y' as y and instruments"):
        two_stage_least_squares(confounded, "Y", "X", instruments=["Y"])
    with pytest.raises(ValueError, match="'Y' as y and covariates"):
        two_stage_least_squares(confounded, "Y", "X", instruments=["Z"], covariates=["Y"])
    with pytest.raises(ValueError, match="'Y' as y and mediators"):
        frontdoor_linear(confounded, "Y", "X", mediators=["Y"])
    with pytest.raises(ValueError, match="'Y' as y and covariates"):
        frontdoor_linear(confounded, "Y", "X", mediators=["Z"], covariates=["Y"])
    with pytest.raises(ValueError, match="'Y' as y and covariates"):
        durbin_wu_hausman(confounded, "Y", "X", instruments=["Z"], covariates=["Y"])
    with pytest.raises(ValueError, match="'Y' as y and instruments"):
        durbin_wu_hausman(confounded, "Y", "X", instruments=["Y"])
    with pytest.raises(ValueError, match="'Z' as covariates and covariates"):
        ols(confounded, "Y", "X", covariates=["Z", "Z"])


def test_perfect_fit_is_an_error_not_a_zero_standard_error(confounded: pd.DataFrame) -> None:
    x = confounded["X"].to_numpy()
    design = np.column_stack([np.ones(N), x])
    with pytest.raises(ValueError, match="perfect fit"):
        least_squares(design, 1.0 + 2.0 * x)
    with pytest.raises(ValueError, match="perfect fit"):
        ols(confounded.assign(W=3.0 * confounded["X"] - 1.0), "W", "X")
    with pytest.raises(ValueError, match="perfect fit"):
        two_stage_least_squares(
            confounded.assign(W=3.0 * confounded["X"] - 1.0), "W", "X", instruments=["Z"]
        )
    # A constant outcome is a perfect fit of the intercept alone.
    with pytest.raises(ValueError, match="perfect fit"):
        ols(confounded.assign(W=1.0), "W", "X")


def test_duplicate_frame_labels_are_named(confounded: pd.DataFrame) -> None:
    doubled = pd.concat([confounded, confounded[["Z"]]], axis=1)
    assert not doubled.columns.is_unique
    with pytest.raises(ValueError, match=r"not unique.*\['Z'\]"):
        design_matrix(doubled, ["X"])
    with pytest.raises(ValueError, match=r"not unique.*\['Z'\]"):
        ols(doubled, "Y", "X")
    with pytest.raises(ValueError, match=r"not unique.*\['Z'\]"):
        ols(doubled, "Y", "X", covariates=["Z"])


# -- 2SLS / instrument ---------------------------------------------------------------------


def test_two_stage_least_squares_recovers_where_ols_fails(iv: pd.DataFrame) -> None:
    est = two_stage_least_squares(iv, "Y", "X", instruments=["Z"])
    naive = ols(iv, "Y", "X")
    assert _within(est, 2.0)
    assert abs(naive.z(2.0)) > 5.0
    assert naive.estimate == pytest.approx(2.0 + 1.0 / 3.0, abs=0.05)
    assert est.method == "2sls" and est.n == N
    assert est.detail["first_stage_f"] > 1000.0
    assert est.detail["n_instruments"] == 1.0
    assert est.ci(0.95).contains(2.0)
    assert est.se > naive.se


def test_two_stage_least_squares_matches_closed_form(iv: pd.DataFrame) -> None:
    z = np.column_stack([np.ones(N), iv["Z"].to_numpy()])
    x = np.column_stack([np.ones(N), iv["X"].to_numpy()])
    y = iv["Y"].to_numpy()
    xhat = z @ np.linalg.solve(z.T @ z, z.T @ x)
    beta = np.linalg.solve(xhat.T @ x, xhat.T @ y)
    resid = y - x @ beta
    cov = (resid @ resid / (N - 2)) * np.linalg.inv(xhat.T @ xhat)
    est = two_stage_least_squares(iv, "Y", "X", instruments=["Z"])
    assert est.estimate == pytest.approx(beta[1])
    assert est.se == pytest.approx(np.sqrt(cov[1, 1]))


def test_two_stage_least_squares_with_covariates_and_two_instruments() -> None:
    scm = LinearSCM.from_text(
        "Z1 -> X: 1.0, Z2 -> X: 0.5, W -> X: 0.7, W -> Y: 1.0, X -> Y: 2.0, X <-> Y: 1.0"
    )
    frame = scm.simulate(N, seed=4)
    est = two_stage_least_squares(frame, "Y", "X", instruments=["Z1", "Z2"], covariates=["W"])
    assert _within(est, 2.0)
    assert est.covariates == ("W",)
    assert est.detail["first_stage_df_num"] == 2.0
    # Closed form: beta = (Xhat'X)^{-1} Xhat'y with Xhat = P_Z X, Z = [1, Z1, Z2, W], X = [1, x, W];
    # cov = sigma^2 (Xhat'Xhat)^{-1} with sigma^2 from the structural residual y - X beta.
    z = np.column_stack([np.ones(N), frame[["Z1", "Z2", "W"]].to_numpy()])
    x = np.column_stack([np.ones(N), frame[["X", "W"]].to_numpy()])
    y = frame["Y"].to_numpy()
    xhat = z @ np.linalg.solve(z.T @ z, z.T @ x)
    beta = np.linalg.solve(xhat.T @ x, xhat.T @ y)
    resid = y - x @ beta
    cov = (resid @ resid / (N - 3)) * np.linalg.inv(xhat.T @ xhat)
    assert est.estimate == pytest.approx(beta[1])
    assert est.se == pytest.approx(np.sqrt(cov[1, 1]))
    # First-stage F against the covariates-only model, by hand.
    treatment = frame["X"].to_numpy()
    full = np.linalg.lstsq(z, treatment, rcond=None)[0]
    small_design = np.column_stack([np.ones(N), frame["W"].to_numpy()])
    small = np.linalg.lstsq(small_design, treatment, rcond=None)[0]
    rss_full = float(((treatment - z @ full) ** 2).sum())
    rss_small = float(((treatment - small_design @ small) ** 2).sum())
    f_by_hand = ((rss_small - rss_full) / 2) / (rss_full / (N - 4))
    assert est.detail["first_stage_f"] == pytest.approx(f_by_hand)
    assert est.detail["first_stage_df_den"] == float(N - 4)


def test_weak_instrument_check_states(iv: pd.DataFrame) -> None:
    strong = two_stage_least_squares(iv, "Y", "X", instruments=["Z"])
    a = weak_instrument_check(strong)
    assert isinstance(a, Assumption)
    assert a.name == "instrument_strength" and a.facet == "identification"
    assert a.state == "satisfied"
    weak_scm = LinearSCM.from_text("Z -> X: 0.01, X -> Y: 2.0, X <-> Y: 1.0")
    weak = two_stage_least_squares(weak_scm.simulate(2_000, seed=0), "Y", "X", instruments=["Z"])
    b = weak_instrument_check(weak)
    assert b.state == "violated"
    assert float(b.detail["first_stage_f"]) < 10.0
    with pytest.raises(ValueError):
        weak_instrument_check(ols(iv, "Y", "X"))


def test_weak_instrument_check_threshold_is_honoured(iv: pd.DataFrame) -> None:
    strong = two_stage_least_squares(iv, "Y", "X", instruments=["Z"])
    f = strong.detail["first_stage_f"]
    assert weak_instrument_check(strong, threshold=f).state == "satisfied"
    above = weak_instrument_check(strong, threshold=f * 1.01)
    assert above.state == "violated"
    assert above.detail["threshold"] == f"{f * 1.01:g}"
    assert f"{f * 1.01:g}" in above.statement
    weak_scm = LinearSCM.from_text("Z -> X: 0.01, X -> Y: 2.0, X <-> Y: 1.0")
    weak = two_stage_least_squares(weak_scm.simulate(2_000, seed=0), "Y", "X", instruments=["Z"])
    assert weak_instrument_check(weak).state == "violated"
    assert weak_instrument_check(weak, threshold=1e-6).state == "satisfied"
    with pytest.raises(ValueError, match="threshold"):
        weak_instrument_check(strong, threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        weak_instrument_check(strong, threshold=float("nan"))


# -- front-door ---------------------------------------------------------------------------------


def test_frontdoor_recovers_where_ols_fails(frontdoor: pd.DataFrame) -> None:
    est = frontdoor_linear(frontdoor, "Y", "X", mediators=["M"])
    naive = ols(frontdoor, "Y", "X")
    assert _within(est, 1.8)
    assert abs(naive.z(1.8)) > 5.0
    assert naive.estimate == pytest.approx(2.3, abs=0.05)
    assert est.method == "frontdoor"
    assert est.detail["a[M]"] == pytest.approx(1.2, abs=0.05)
    assert est.detail["b[M]"] == pytest.approx(1.5, abs=0.05)
    assert est.ci(0.95).contains(1.8)


def test_frontdoor_delta_method_se_matches_hand_computation(frontdoor: pd.DataFrame) -> None:
    a = ols(frontdoor, "M", "X")
    b = ols(frontdoor, "Y", "M", covariates=["X"])
    est = frontdoor_linear(frontdoor, "Y", "X", mediators=["M"])
    assert est.estimate == pytest.approx(a.estimate * b.estimate)
    assert est.se == pytest.approx(np.sqrt(b.estimate**2 * a.se**2 + a.estimate**2 * b.se**2))


def test_frontdoor_with_two_mediators() -> None:
    scm = LinearSCM.from_text(
        "X -> M1: 1.0, X -> M2: 0.5, M1 -> Y: 1.0, M2 -> Y: 2.0, X <-> Y: 1.0"
    )
    frame = scm.simulate(N, seed=9)
    est = frontdoor_linear(frame, "Y", "X", mediators=["M1", "M2"])
    assert _within(est, scm.total_effect("X", "Y"))
    assert scm.total_effect("X", "Y") == pytest.approx(2.0)


CHAINED_MEDIATORS = "X -> M1: 1.0, M1 -> M2: 1.0, M2 -> Y: 1.0, M1 -> Y: 1.0, X <-> Y: 1.0"


def test_frontdoor_multi_mediator_se_uses_the_joint_first_stage_covariance() -> None:
    scm = LinearSCM.from_text(CHAINED_MEDIATORS)
    frame = scm.simulate(N, seed=3)
    est = frontdoor_linear(frame, "Y", "X", mediators=["M1", "M2"])
    assert _within(est, 2.0)
    design = np.column_stack([np.ones(N), frame["X"].to_numpy()])
    mediators = frame[["M1", "M2"]].to_numpy()
    coef = np.linalg.solve(design.T @ design, design.T @ mediators)
    resid = mediators - design @ coef
    cov_a = (resid.T @ resid / (N - 2)) * np.linalg.inv(design.T @ design)[1, 1]
    a = coef[1]
    second = np.column_stack([np.ones(N), mediators, frame["X"].to_numpy()])
    y = frame["Y"].to_numpy()
    beta = np.linalg.solve(second.T @ second, second.T @ y)
    r2 = y - second @ beta
    cov_b = (r2 @ r2 / (N - 4)) * np.linalg.inv(second.T @ second)[1:3, 1:3]
    b = beta[1:3]
    assert est.estimate == pytest.approx(a @ b)
    assert est.se == pytest.approx(np.sqrt(b @ cov_a @ b + a @ cov_b @ a))
    # M2 inherits M1's noise, so Cov(a_M1, a_M2) > 0 and dropping it understates the SE.
    naive_se = np.sqrt((b**2 * np.diag(cov_a)).sum() + a @ cov_b @ a)
    assert cov_a[0, 1] > 0.0
    assert naive_se < est.se / 1.05


def test_frontdoor_multi_mediator_se_is_calibrated_by_monte_carlo() -> None:
    scm = LinearSCM.from_text(CHAINED_MEDIATORS)
    rng = np.random.default_rng(1)
    reps, n = 200, 2_000
    estimates = np.empty(reps)
    reported = np.empty(reps)
    for r in range(reps):
        frame = scm.simulate(n, seed=int(rng.integers(2**31)))
        est = frontdoor_linear(frame, "Y", "X", mediators=["M1", "M2"])
        estimates[r] = est.estimate
        reported[r] = est.se
    ratio = estimates.std(ddof=1) / reported.mean()
    assert 0.9 < ratio < 1.1, ratio
    assert abs(estimates.mean() - 2.0) < 4 * reported.mean() / np.sqrt(reps)


def test_frontdoor_negative_control_latent_treatment_mediator_confounding() -> None:
    # X <-> M violates the front-door criterion (a back-door path from X to M);
    # the estimator must be visibly wrong, not quietly right.
    bad = LinearSCM.from_text("X -> M: 1.2, M -> Y: 1.5, X <-> M: 1.0, X <-> Y: 1.0")
    assert bad.total_effect("X", "Y") == pytest.approx(1.8)
    frame = bad.simulate(N, seed=0)
    est = frontdoor_linear(frame, "Y", "X", mediators=["M"])
    assert abs(est.z(1.8)) > 5.0
    assert not est.ci(0.95).contains(1.8)


# -- endogeneity ------------------------------------------------------------------------------


def test_durbin_wu_hausman_detects_endogeneity(iv: pd.DataFrame) -> None:
    test = _as_test(durbin_wu_hausman(iv, "Y", "X", instruments=["Z"]))
    assert test.conclusion == "endogenous"
    assert test.p_value < 1e-6
    assert test.df == N - 3
    assert test.alpha == 0.05
    assert test.detail["first_stage_f"] > 1000.0
    assert EndogeneityTest.from_json(test.to_json()) == test


def test_durbin_wu_hausman_finds_exogeneity_without_confounding() -> None:
    scm = LinearSCM.from_text("Z -> X: 1.0, X -> Y: 2.0")
    frame = scm.simulate(N, seed=0)
    test = _as_test(durbin_wu_hausman(frame, "Y", "X", instruments=["Z"]))
    assert test.conclusion == "exogenous"
    assert test.p_value > 0.05
    assert abs(test.statistic) < 2.0


def test_durbin_wu_hausman_has_nominal_size_under_the_null() -> None:
    scm = LinearSCM.from_text("Z -> X: 1.0, X -> Y: 2.0")
    rng = np.random.default_rng(0)
    reps, alpha = 200, 0.05
    rejections = 0
    for _ in range(reps):
        frame = scm.simulate(500, seed=int(rng.integers(2**31)))
        test = _as_test(durbin_wu_hausman(frame, "Y", "X", instruments=["Z"], alpha=alpha))
        rejections += test.conclusion == "endogenous"
    region = clopper_pearson(reps, alpha, 0.001)
    assert region.accepts(rejections), (rejections, region)


def test_durbin_wu_hausman_rejects_bad_alpha(iv: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="alpha"):
        durbin_wu_hausman(iv, "Y", "X", instruments=["Z"], alpha=1.5)
    with pytest.raises(ValueError, match="instrument"):
        durbin_wu_hausman(iv, "Y", "X", instruments=[])


def test_durbin_wu_hausman_degenerate_se_is_unverified(
    iv: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = least_squares

    def zero_cov_on_augmented(design: np.ndarray, y: np.ndarray) -> LeastSquaresFit:  # type: ignore[type-arg]
        fit = real(design, y)
        if design.shape[1] == 3:  # [1, x, v]: the augmented regression
            return dataclasses.replace(fit, cov=np.zeros_like(fit.cov))
        return fit

    monkeypatch.setattr(endogeneity, "least_squares", zero_cov_on_augmented)
    result = durbin_wu_hausman(iv, "Y", "X", instruments=["Z"])
    assert isinstance(result, Unverified)
    assert not result
    assert "se_v" in result.reason and "t-statistic" in result.reason
    assert result.detail["treatment"] == "X" and result.detail["outcome"] == "Y"
    assert float(result.detail["se_v"]) == 0.0
    assert float(result.detail["first_stage_f"]) > 1000.0


def test_hausman_iv_vs_ols(iv: pd.DataFrame) -> None:
    est_ols = ols(iv, "Y", "X")
    est_iv = two_stage_least_squares(iv, "Y", "X", instruments=["Z"])
    test = _as_test(hausman_iv_vs_ols(est_ols, est_iv))
    assert test.conclusion == "endogenous" and test.df == 1
    assert test.method == "hausman_iv_vs_ols"
    assert test.statistic > 100.0 and test.p_value < 1e-6


def test_hausman_unverified_when_variance_difference_not_positive() -> None:
    est_ols = LinearEstimate(estimate=1.0, se=0.5, n=100, method="ols", treatment="X", outcome="Y")
    est_iv = LinearEstimate(estimate=1.5, se=0.4, n=100, method="2sls", treatment="X", outcome="Y")
    result = hausman_iv_vs_ols(est_ols, est_iv)
    assert isinstance(result, Unverified)
    assert not result
    assert "not positive" in result.reason
    assert float(result.detail["variance_difference"]) == pytest.approx(0.16 - 0.25)
    assert result.detail["treatment"] == "X"
    with pytest.raises(ValueError, match="'ols' and a '2sls'"):
        hausman_iv_vs_ols(est_iv, est_ols)
    other = est_iv.model_copy(update={"outcome": "Q"})
    with pytest.raises(ValueError, match="same treatment and outcome"):
        hausman_iv_vs_ols(est_ols, other)


def test_hausman_requires_matching_covariates_and_sample(iv: pd.DataFrame) -> None:
    est_iv = two_stage_least_squares(iv, "Y", "X", instruments=["Z"])
    with pytest.raises(ValueError, match="same covariates"):
        hausman_iv_vs_ols(ols(iv, "Y", "X", covariates=["Z"]), est_iv)
    with pytest.raises(ValueError, match="same sample"):
        hausman_iv_vs_ols(ols(iv.head(N // 2), "Y", "X"), est_iv)
    with pytest.raises(ValueError, match="alpha"):
        hausman_iv_vs_ols(ols(iv, "Y", "X"), est_iv, alpha=0.0)
