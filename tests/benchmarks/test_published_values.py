"""Does axiom reproduce numbers other people published, from their raw data?

Every other test in this repository checks axiom against a world axiom made up.
That is the right way to check whether an estimator recovers a truth, and it is
circular in one specific respect: the data-generating process and the likelihood
are the same code. These tests close that loop from outside, using datasets
collected by other people and results computed by other software.

Two tiers:

* **Vendored** data is committed, so these run in CI on a fresh clone.
* **Fetched** data cannot be redistributed under this repository's licence, so
  those tests skip until someone runs ``python benchmarks/fetch.py``.

Tolerances are set to the precision of the PUBLISHED value, not to whatever makes
the test pass. Where the published number is quoted to four decimal places, that
is what is asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from registry import DATASETS, load, log_risk_ratios, model_based_i2

from axiom.core import BASES, D, Outcome, Treatment
from axiom.data import Panel, RoleMap
from axiom.diagnose import robustness_value
from axiom.identify import ols
from axiom.meta import heterogeneity, random_effects
from axiom.surface import ExponentialKernel, SurfaceSpec, fit

# ----------------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------------


def test_every_dataset_declares_its_terms() -> None:
    """A dataset without a licence and a citation has no business being here."""
    for name, spec in DATASETS.items():
        assert spec.citation, f"{name} has no citation"
        assert spec.licence, f"{name} has no licence"
        assert spec.source.startswith("http"), f"{name} has no source URL"
        assert spec.published, f"{name} has no published values to check against"
        if spec.availability == "fetch":
            assert spec.url.startswith("http"), f"{name} is fetchable but has no URL"


def test_vendored_datasets_are_present_and_stable() -> None:
    for spec in DATASETS.values():
        if spec.availability != "vendored":
            continue
        assert spec.available, f"{spec.name} should be committed at {spec.path}"
        assert len(spec.sha256()) == 64


# ----------------------------------------------------------------------------------
# BCG: meta-analysis against R's metafor
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bcg() -> tuple[np.ndarray, np.ndarray]:
    return log_risk_ratios(load("bcg"))


def test_bcg_shape() -> None:
    frame = load("bcg")
    assert len(frame) == 13
    assert int((frame.tpos + frame.tneg + frame.cpos + frame.cneg).sum()) == 357_347


def test_bcg_pooled_estimate_matches_metafor(bcg: tuple[np.ndarray, np.ndarray]) -> None:
    y, se = bcg
    published = DATASETS["bcg"].published
    pooled = random_effects(y, se, tau_method="reml")

    assert pooled.estimate == pytest.approx(published["estimate"], abs=1e-4)
    assert pooled.se == pytest.approx(published["se"], abs=1e-4)
    assert pooled.interval.lower == pytest.approx(published["ci_lower"], abs=1e-3)
    assert pooled.interval.upper == pytest.approx(published["ci_upper"], abs=1e-3)
    assert pooled.tau2 == pytest.approx(published["tau2_reml"], abs=1e-4)


def test_bcg_heterogeneity_matches_metafor(bcg: tuple[np.ndarray, np.ndarray]) -> None:
    y, se = bcg
    published = DATASETS["bcg"].published
    het = heterogeneity(y, se)

    assert het.q == pytest.approx(published["q"], abs=1e-3)
    assert het.df == published["q_df"]


def test_bcg_i2_definitions_differ_and_both_are_right(
    bcg: tuple[np.ndarray, np.ndarray],
) -> None:
    """metafor prints the model-based I²; axiom returns the Q-based one.

    They differ by about a tenth of a percentage point on this dataset. This test
    pins both, so that if either ever moves it is a deliberate change rather than
    a silent one.
    """
    y, se = bcg
    published = DATASETS["bcg"].published
    het = heterogeneity(y, se)
    pooled = random_effects(y, se, tau_method="reml")
    i2_model, h2_model = model_based_i2(pooled.tau2, se)

    # axiom's own definition: (Q - df) / Q
    assert het.i2 == pytest.approx((het.q - het.df) / het.q, rel=1e-12)
    # metafor's, reconstructed from axiom's outputs
    assert i2_model == pytest.approx(published["i2_model_based"], abs=1e-3)
    assert h2_model == pytest.approx(published["h2_model_based"], abs=1e-2)
    # and they are genuinely different numbers
    assert abs(het.i2 - i2_model) > 5e-4


def test_bcg_tau_estimators_agree_with_each_other(bcg: tuple[np.ndarray, np.ndarray]) -> None:
    y, se = bcg
    estimates = [random_effects(y, se, tau_method=m).estimate for m in ("dl", "pm", "reml")]
    assert max(estimates) - min(estimates) < 0.1


# ----------------------------------------------------------------------------------
# NIST Misra1a: a certified nonlinear fit
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def misra_fit() -> object:
    BASES.declare("volume", symbol="V")
    BASES.declare("pressure", symbol="P")
    volume = Outcome(name="volume", dimension=D.volume, unit="cc")
    pressure = Treatment(name="pressure", dimension=D.pressure, unit="mmHg")

    frame = load("misra1a")
    tidy = pd.DataFrame(
        {
            "unit": ["specimen"] * len(frame),
            "t": np.arange(len(frame)),
            "pressure": frame["pressure"].to_numpy(float),
            "volume": frame["volume"].to_numpy(float),
        }
    )
    panel = Panel(
        tidy,
        RoleMap(
            unit="unit",
            time="t",
            outcome=("volume", volume),
            treatments={"pressure": pressure},
        ),
    )
    spec = SurfaceSpec(
        name="misra1a",
        treatments=(pressure,),
        outcome=volume,
        kernels={"pressure": ExponentialKernel(reference_dose=1800.0, amplitude_scale=250.0)},
        intercept="none",
        unit_labels=("specimen",),
    )
    return fit(spec, panel, backend="laplace", draws=4000, seed=0)


def test_misra1a_recovers_nist_certified_values(misra_fit: object) -> None:
    """axiom fits a posterior; NIST certifies a least-squares optimum.

    They are different objects, so this asks whether the posterior concentrates on
    the certified values -- one part in a thousand is a generous tolerance for a
    two-parameter problem with fourteen clean points.
    """
    certified = DATASETS["misra1a"].published
    posterior = misra_fit.posterior  # type: ignore[attr-defined]

    beta = posterior.summary("beta_pressure").mean
    k = posterior.summary("k_pressure").mean

    assert beta == pytest.approx(certified["axiom_beta"], rel=1e-3)
    assert k == pytest.approx(certified["axiom_k"], rel=1e-3)
    # NIST parameterises by b2 = 1 / k
    assert 1.0 / k == pytest.approx(certified["b2"], rel=1e-3)


def test_misra1a_certified_values_lie_inside_the_posterior(misra_fit: object) -> None:
    certified = DATASETS["misra1a"].published
    posterior = misra_fit.posterior  # type: ignore[attr-defined]
    for name, target in (
        ("beta_pressure", certified["axiom_beta"]),
        ("k_pressure", certified["axiom_k"]),
    ):
        interval = posterior.summary(name, definition="hdi", mass=0.95).interval
        assert interval.lower <= target <= interval.upper, name


def test_misra1a_does_not_beat_the_certified_optimum(misra_fit: object) -> None:
    """A residual sum of squares below NIST's certified minimum would be a bug.

    The certified value IS the minimum over the parameter space, computed to 500
    digits. Anything lower means the model being fitted is not the model NIST
    certified, or the residuals are being computed wrongly.
    """
    certified = DATASETS["misra1a"].published
    frame = load("misra1a")
    posterior = misra_fit.posterior  # type: ignore[attr-defined]

    beta = posterior.summary("beta_pressure").mean
    k = posterior.summary("k_pressure").mean
    x = frame["pressure"].to_numpy(float)
    y = frame["volume"].to_numpy(float)
    rss = float(np.sum((y - beta * (1.0 - np.exp(-x / k))) ** 2))

    certified_rss = certified["residual_sum_of_squares"]
    assert rss >= certified_rss, "fitted RSS below the certified minimum"
    assert rss == pytest.approx(certified_rss, rel=1e-3)


# ----------------------------------------------------------------------------------
# fetched: skip cleanly when the data has not been downloaded
# ----------------------------------------------------------------------------------

needs_darfur = pytest.mark.skipif(
    not DATASETS["darfur"].available,
    reason="darfur not fetched; run `python benchmarks/fetch.py darfur`",
)
needs_lalonde = pytest.mark.skipif(
    not (DATASETS["lalonde_nsw"].available and DATASETS["psid_controls"].available),
    reason="lalonde not fetched; run `python benchmarks/fetch.py lalonde_nsw psid_controls`",
)


@needs_darfur
def test_darfur_regression_reproduces_the_paper() -> None:
    """The step axiom's unit tests skip: getting 0.0973 out of 1,276 rows.

    ``tests/unit/test_diagnose_sensitivity.py`` checks the sensitivity formulas
    against Cinelli & Hazlett's published values using hard-coded summary
    statistics. This checks the summary statistics themselves.
    """
    published = DATASETS["darfur"].published
    frame = load("darfur")
    assert len(frame) == published["n"]

    dummies = pd.get_dummies(frame["village"], prefix="village", drop_first=True, dtype=float)
    design = pd.concat([frame.drop(columns=["village"]), dummies], axis=1)
    covariates = [
        "age",
        "farmer_dar",
        "herder_dar",
        "pastvoted",
        "hhsize_darfur",
        "female",
        *dummies.columns,
    ]
    estimate = ols(design, "peacefactor", "directlyharmed", covariates=covariates)

    assert estimate.estimate == pytest.approx(published["coefficient"], abs=1e-4)
    assert estimate.se == pytest.approx(published["se"], abs=1e-4)
    assert int(estimate.detail["df_resid"]) == published["df"]


@needs_darfur
def test_darfur_sensitivity_from_the_raw_data() -> None:
    published = DATASETS["darfur"].published
    frame = load("darfur")
    dummies = pd.get_dummies(frame["village"], prefix="village", drop_first=True, dtype=float)
    design = pd.concat([frame.drop(columns=["village"]), dummies], axis=1)
    covariates = [
        "age",
        "farmer_dar",
        "herder_dar",
        "pastvoted",
        "hhsize_darfur",
        "female",
        *dummies.columns,
    ]
    estimate = ols(design, "peacefactor", "directlyharmed", covariates=covariates)
    rv = robustness_value(
        estimate=estimate.estimate,
        se=estimate.se,
        df=int(estimate.detail["df_resid"]),
        q=1.0,
        alpha=0.05,
    )
    assert rv.rv == pytest.approx(published["robustness_value"], abs=1e-3)
    assert rv.rv_alpha == pytest.approx(published["robustness_value_alpha"], abs=1e-3)
    assert rv.r2_yd_x == pytest.approx(published["partial_r2"], abs=1e-3)


@needs_lalonde
def test_lalonde_experimental_benchmark() -> None:
    """Randomization makes the difference in means the causal effect."""
    published = DATASETS["lalonde_nsw"].published
    frame = load("lalonde_nsw")

    assert len(frame) == published["n"]
    assert int(frame.treat.sum()) == published["n_treated"]
    assert int((1 - frame.treat).sum()) == published["n_control"]

    estimate = ols(frame, "re78", "treat")
    assert estimate.estimate == pytest.approx(published["experimental_ate"], abs=1.0)


@needs_lalonde
def test_lalonde_observational_version_misses_the_benchmark() -> None:
    """The finding the dataset is famous for, asserted rather than described.

    Swapping the randomized controls for PSID respondents moves the estimate by
    thousands of dollars. The point is not that the observational number is
    imprecise -- its standard error is small. It is confidently wrong.
    """
    frame = load("lalonde_nsw")
    psid = load("psid_controls")
    covariates = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

    truth = ols(frame, "re78", "treat").estimate
    observational = pd.concat([frame[frame.treat == 1], psid], ignore_index=True)

    naive = ols(observational, "re78", "treat")
    adjusted = ols(observational, "re78", "treat", covariates=covariates)

    # wrong sign, and by a mile
    assert naive.estimate < -10_000
    assert abs(naive.estimate - truth) > 15_000
    # adjustment helps a lot and still misses
    assert abs(adjusted.estimate - truth) > 500
    # and it is not that the observational estimate is vague
    assert naive.se < 2_000
