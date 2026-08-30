"""Likelihood families beyond the Gaussian, and the design math that follows them.

The claim under test is one identity. For any of these families the Fisher
information is ``J' W J`` with ``J = d mean / d theta`` and a single diagonal
weight ``w_i = 1 / (phi V(mu_i))``, so the design math needs a variance
function and nothing else — no link, no deviance, no IRLS. The first test
here is the only one that really matters: it checks that weight against the
*definition* of Fisher information, ``Var[score]``, by simulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from axiom.core import (
    Apply,
    Data,
    Likelihood,
    ModelSpec,
    Param,
    Prior,
    Unsupported,
    dimensionless,
    information_weight,
    log_density,
    log_likelihood,
    variance_weight,
)
from axiom.design import Weighting
from axiom.design.power import difference_se, power_from_se, proportion_difference_se
from axiom.design.structural import fisher_information
from axiom.sim import DosePlan, surface_world
from axiom.surface import HillKernel

NONE = dimensionless()
N = 12
_rng = np.random.default_rng(0)
X = _rng.uniform(0.5, 1.5, N)
TRIALS = _rng.integers(4, 20, N).astype(float)


def _model(family: str, **lik: Any) -> ModelSpec:
    y = Data(name="y", dimension=NONE)
    x = Data(name="x", dimension=NONE)
    a = Param(
        name="a", dimension=NONE, prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0})
    )
    params = [a]
    mean = a * x
    if family == "binomial":  # a probability, which is what a sigmoid produces
        mean = Apply(fn="sigmoid", arg=a * x)
    if lik.get("scale"):
        params.append(
            Param(name="s", dimension=NONE, prior=Prior(family="halfnormal", hyper={"sigma": 1.0}))
        )
    return ModelSpec(
        name=family,
        mean=mean,
        outcome=y,
        likelihood=Likelihood(family=family, **lik),
        parameters=tuple(params),
    )


# -- the identity ---------------------------------------------------------------------

# family, likelihood kwargs, theta, how to draw an outcome at a mean
CASES: list[tuple[str, dict[str, Any], dict[str, float], Any]] = [
    ("normal", {"scale": "s"}, {"a": 2.0, "s": 0.7}, lambda mu, r: r.normal(mu, 0.7)),
    (
        "student_t",
        {"scale": "s", "df": 6.0},
        {"a": 2.0, "s": 0.7},
        lambda mu, r: mu + 0.7 * r.standard_t(6.0, mu.shape),
    ),
    ("lognormal", {"scale": "s"}, {"a": 2.0, "s": 0.4}, lambda mu, r: r.lognormal(np.log(mu), 0.4)),
    ("gamma", {"scale": "s"}, {"a": 2.0, "s": 0.5}, lambda mu, r: r.gamma(4.0, mu / 4.0)),
    ("poisson", {}, {"a": 4.0}, lambda mu, r: r.poisson(mu)),
    (
        "binomial",
        {"trials": "n"},
        {"a": 0.4},
        lambda mu, r: r.binomial(TRIALS.astype(int), mu),
    ),
]


@pytest.mark.parametrize("family,lik,theta,draw", CASES, ids=[c[0] for c in CASES])
def test_the_weight_is_the_variance_of_the_score(
    family: str, lik: dict[str, Any], theta: dict[str, float], draw: Any
) -> None:
    """``sum_i w_i (dmu_i/da)^2`` must equal ``Var[d logL / da]``. That is the definition."""
    model = _model(family, **lik)
    data = {"x": X, "n": TRIALS}
    mu = _mean_of(model, {**data, "y": np.zeros(N)}, theta)
    w = np.broadcast_to(
        np.asarray(information_weight(model, {**data, "y": np.zeros(N)}, theta), dtype=float),
        (N,),
    )
    # J = d mu / d a, by central difference on the mean alone
    h = 1e-6
    up = {**theta, "a": theta["a"] + h}
    dn = {**theta, "a": theta["a"] - h}
    j = (
        _mean_of(model, {**data, "y": np.zeros(N)}, up)
        - _mean_of(model, {**data, "y": np.zeros(N)}, dn)
    ) / (2 * h)
    analytic = float(np.sum(w * j**2))

    rng = np.random.default_rng(7)
    scores = np.array(
        [
            _score(model, {**data, "y": np.asarray(draw(mu, rng), dtype=float)}, theta)
            for _ in range(20000)
        ]
    )
    mc = float(np.var(scores, ddof=1))
    se = mc * np.sqrt(2 / (scores.size - 1))
    assert abs(mc - analytic) < 5 * se, f"{family}: J'WJ={analytic:.4f} vs Var(score)={mc:.4f}"


def _mean_of(model: ModelSpec, data: dict[str, Any], theta: dict[str, float]) -> np.ndarray:
    from axiom.core import value

    return np.broadcast_to(
        np.asarray(value(model.mean, data=data, params=theta), dtype=float), (N,)
    ).copy()


def _score(
    model: ModelSpec, data: dict[str, Any], theta: dict[str, float], h: float = 1e-6
) -> float:
    up = {**theta, "a": theta["a"] + h}
    dn = {**theta, "a": theta["a"] - h}
    return (log_likelihood(model, data, up) - log_likelihood(model, data, dn)) / (2 * h)


def test_student_t_information_tends_to_the_gaussian_as_df_grows() -> None:
    """``(df+1)/((df+3) sigma²)`` -> ``1/sigma²``: the check to make when touching this."""
    normal = float(variance_weight("normal", 1.0, scale=2.0))
    for df, closeness in ((4.0, 0.72), (50.0, 0.96), (10_000.0, 0.999)):
        ratio = float(variance_weight("student_t", 1.0, scale=2.0, df=df)) / normal
        assert ratio == pytest.approx(closeness, abs=0.01)
        assert ratio < 1.0  # a t is always less informative about location than a normal


def test_lognormal_and_gamma_share_a_weight_but_not_a_density() -> None:
    """Both put a constant coefficient of variation on a positive mean."""
    mu = np.array([1.0, 4.0])
    assert np.allclose(
        variance_weight("lognormal", mu, scale=0.3), variance_weight("gamma", mu, scale=0.3)
    )
    a = log_likelihood(
        _model("lognormal", scale="s"), {"x": X, "y": np.full(N, 2.0)}, {"a": 2.0, "s": 0.3}
    )
    b = log_likelihood(
        _model("gamma", scale="s"), {"x": X, "y": np.full(N, 2.0)}, {"a": 2.0, "s": 0.3}
    )
    assert a != b


# -- the support each family needs ----------------------------------------------------


def test_a_mean_outside_its_family_support_raises_rather_than_returning_a_number() -> None:
    with pytest.raises(ValueError, match="positive"):
        variance_weight("poisson", np.array([1.0, -0.5]))
    with pytest.raises(ValueError, match="probability"):
        variance_weight("binomial", np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        variance_weight("gamma", np.array([1.0]), scale=-1.0)
    with pytest.raises(ValueError, match="no variance function"):
        variance_weight("weibull", np.array([1.0]))
    with pytest.raises(ValueError, match="df"):
        variance_weight("student_t", np.array([1.0]), scale=1.0, df=0.0)


def test_the_spec_refuses_a_scale_a_family_does_not_have() -> None:
    with pytest.raises(ValueError, match="no scale parameter"):
        Likelihood(family="binomial", scale="s")
    with pytest.raises(ValueError, match="needs a scale"):
        Likelihood(family="gamma")
    with pytest.raises(ValueError, match="trials"):
        Likelihood(family="poisson", trials="n")


# -- the design math ------------------------------------------------------------------


@pytest.fixture(scope="module")
def world() -> Any:
    return surface_world(
        n_units=2,
        n_periods=12,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        doses=DosePlan(scale=50.0),
        intercept="shared",
        truth={"beta_a": 10.0, "alpha": 5.0, "k_a": 50.0, "s_a": 2.0},
        noise_sd=0.5,
        seed=0,
    )


def test_a_normal_weighting_is_the_noise_sd_path_exactly(world: Any) -> None:
    """The Gaussian case is not a special case of the new code; it is the same number."""
    by_sd = fisher_information(world.surface, dict(world.data), world.theta, 0.5)
    by_weight = fisher_information(
        world.surface,
        dict(world.data),
        world.theta,
        weighting=Weighting(family="normal", scale=0.5),
    )
    assert not isinstance(by_sd, Unsupported) and not isinstance(by_weight, Unsupported)
    assert np.allclose(by_sd.as_array(), by_weight.as_array(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family,kw", [("poisson", {}), ("gamma", {"scale": 0.4})])
def test_the_weighted_information_is_j_w_j(world: Any, family: str, kw: dict[str, Any]) -> None:
    """Against a Jacobian built independently, by differencing ``forward`` directly."""
    data, theta, surface = dict(world.data), world.theta, world.surface
    weighting = Weighting(family=family, **kw)
    got = fisher_information(surface, data, theta, weighting=weighting)
    assert not isinstance(got, Unsupported), got

    mu = np.asarray(surface.forward(data, theta), dtype=float).ravel()
    jac = np.zeros((mu.size, len(got.parameters)))
    for j, name in enumerate(got.parameters):
        centre = float(np.mean(np.asarray(theta[name], dtype=float)))
        h = 1e-6 * max(abs(centre), 1.0)
        up = {**theta, name: centre + h}
        dn = {**theta, name: centre - h}
        jac[:, j] = (
            np.asarray(surface.forward(data, up)).ravel()
            - np.asarray(surface.forward(data, dn)).ravel()
        ) / (2 * h)
    w = np.broadcast_to(np.asarray(weighting.at(mu, data), dtype=float), mu.shape)
    hand = jac.T @ (w[:, None] * jac)
    assert np.allclose(got.as_array(), hand, rtol=1e-6, atol=1e-8 * float(np.max(np.abs(hand))))
    assert got.weighting == weighting


def test_information_from_two_different_likelihoods_does_not_add(world: Any) -> None:
    """Rows weighted by different variance functions are not in the same units."""
    a = fisher_information(world.surface, dict(world.data), world.theta, weighting=Weighting())
    b = fisher_information(
        world.surface, dict(world.data), world.theta, weighting=Weighting(family="poisson")
    )
    assert not isinstance(a, Unsupported) and not isinstance(b, Unsupported)
    with pytest.raises(ValueError, match="same weighting"):
        _ = a + b
    doubled = b + b
    assert np.allclose(doubled.as_array(), 2.0 * b.as_array())


def test_a_dispersion_cannot_be_applied_twice(world: Any) -> None:
    with pytest.raises(ValueError, match="not both"):
        fisher_information(
            world.surface,
            dict(world.data),
            world.theta,
            0.5,
            weighting=Weighting(family="poisson"),
        )


def test_a_weighting_undefined_at_the_design_is_unsupported_not_an_exception(world: Any) -> None:
    """A binomial weighting on a surface whose mean is not a probability."""
    got = fisher_information(
        world.surface, dict(world.data), world.theta, weighting=Weighting(family="binomial")
    )
    assert isinstance(got, Unsupported)
    assert "binomial" in got.reason


def test_a_weighting_reads_its_scale_from_a_fitted_model() -> None:
    model = _model("gamma", scale="s")
    weighting = Weighting.from_model(model, {"a": 1.0, "s": 0.25})
    assert weighting.family == "gamma" and weighting.scale == 0.25
    assert "scale=0.25" in weighting.label
    with pytest.raises(ValueError, match="pass theta"):
        Weighting.from_model(model)


# -- the two-arm gap a weight cannot close --------------------------------------------


def test_a_difference_in_proportions_is_not_a_difference_in_means() -> None:
    """The one place a variance function does not rescue the Gaussian formula."""
    p_c, p_t, n = 0.2, 0.6, 400
    exact = proportion_difference_se(p_c, p_t, n)
    # what the homoscedastic route gives, handed the pooled sd it would use
    pooled = float(np.sqrt(0.4 * 0.6))
    assert difference_se(n, pooled) != pytest.approx(exact, rel=1e-3)
    # and the further apart the arms, the worse it is -- the error grows with the
    # effect being powered for, which is the wrong direction for it to run
    near = abs(difference_se(n, float(np.sqrt(0.5 * 0.5))) - proportion_difference_se(0.5, 0.5, n))
    far = abs(difference_se(n, pooled) - exact)
    assert far > near
    assert 0.0 < power_from_se(p_t - p_c, exact).power <= 1.0


def test_the_mde_for_a_proportion_delivers_the_power_it_promises() -> None:
    """The defect: `mde` inverts a fixed SE, but a proportion's SE moves with the effect."""
    from axiom.design import proportion_mde, proportion_power
    from axiom.design.power import mde

    for p_c, n in [(0.05, 2000), (0.10, 400), (0.20, 400), (0.50, 400)]:
        honest = proportion_mde(p_c, n)
        assert not isinstance(honest, Unsupported)
        # the effect it reports really does carry the power it was asked for
        assert proportion_power(p_c, p_c + honest.effect, n).power == pytest.approx(0.8, abs=1e-6)
        assert honest.design == "difference_in_proportions"

        # while freezing the SE at the null misses, and misses *low* wherever the
        # base rate is under a half -- the direction that oversells a study
        frozen = mde(n, float(np.sqrt(p_c * (1 - p_c)))).effect
        delivered = proportion_power(p_c, p_c + frozen, n).power
        if p_c < 0.5:
            assert frozen < honest.effect
            assert delivered < 0.78
        else:
            assert delivered > 0.8


def test_a_fall_is_easier_to_detect_than_a_rise_of_the_same_size() -> None:
    """Not symmetry: the two land on different variances."""
    from axiom.design import proportion_mde

    up = proportion_mde(0.20, 400, direction="increase")
    down = proportion_mde(0.20, 400, direction="decrease")
    assert not isinstance(up, Unsupported) and not isinstance(down, Unsupported)
    assert down.effect < up.effect
    assert down.se < up.se


def test_a_proportion_sample_size_is_the_smallest_n_that_reaches_the_target() -> None:
    from axiom.design import proportion_power, proportion_sample_size

    for p_c, p_t in [(0.10, 0.15), (0.50, 0.55), (0.02, 0.03)]:
        got = proportion_sample_size(p_c, p_t)
        assert not isinstance(got, Unsupported)
        assert got.power >= 0.8
        assert got.n_treated is not None and got.n_control is not None
        assert got.n_treated + got.n_control == got.n
        # minimal: one unit fewer does not reach the target
        assert proportion_power(p_c, p_t, got.n - 1).power < 0.8


def test_a_proportion_design_that_cannot_reach_the_target_says_so() -> None:
    from axiom.design import proportion_mde, proportion_sample_size

    tiny = proportion_mde(0.5, 10)
    assert isinstance(tiny, Unsupported) and "power" in tiny.reason
    assert isinstance(proportion_sample_size(0.3, 0.3), Unsupported)
    with pytest.raises(ValueError, match="probability"):
        proportion_mde(1.5, 100)


def test_a_proportion_outside_zero_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="probability"):
        proportion_difference_se(0.5, 1.2, 100)
    with pytest.raises(ValueError, match="sampling variance"):
        proportion_difference_se(0.0, 1.0, 100)


# -- simulation -----------------------------------------------------------------------


def test_surface_world_draws_a_gamma_outcome_and_refuses_a_binomial_one() -> None:
    w = surface_world(
        n_units=2,
        n_periods=40,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        doses=DosePlan(scale=50.0),
        intercept="shared",
        truth={"beta_a": 10.0, "alpha": 20.0, "k_a": 50.0, "s_a": 2.0, "sigma": 0.3},
        likelihood=Likelihood(family="gamma", scale="sigma"),
        seed=0,
    )
    y = np.asarray(w.panel.frame[w.panel.roles.outcome[0]], dtype=float)
    assert np.all(y > 0.0)  # a gamma outcome is positive by construction

    with pytest.raises(ValueError, match="binomial"):
        surface_world(
            n_units=2,
            n_periods=8,
            treatments=("a",),
            kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
            doses=DosePlan(scale=50.0),
            intercept="shared",
            truth={"beta_a": 10.0, "alpha": 20.0, "k_a": 50.0, "s_a": 2.0},
            likelihood=Likelihood(family="binomial"),
            seed=0,
        )


def test_log_density_agrees_across_the_interpreters() -> None:
    """numpy, jax and pytensor must give the same number for the new families."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import pytensor.tensor as pt

    from axiom.core.interpret.jax import compile_log_density as jax_ld
    from axiom.core.interpret.pytensor import compile_log_density as pt_ld

    rng = np.random.default_rng(3)
    cases = [
        ("poisson", {}, {"a": 0.3}, rng.poisson(2.0, N).astype(float)),
        ("gamma", {"scale": "s"}, {"a": 0.3, "s": -0.4}, rng.gamma(4.0, 0.5, N)),
        (
            "binomial",
            {"trials": "n"},
            {"a": 0.5},
            rng.binomial(TRIALS.astype(int), 0.5).astype(float),
        ),
    ]
    for family, lik, z, y in cases:
        model = _model(family, **lik)
        data = {"x": X, "y": y, "n": TRIALS}
        core = log_density(model, data, z)
        as_jax = float(jax_ld(model)(data, {k: np.float64(v) for k, v in z.items()}))
        as_pt = float(
            pt_ld(model, data)(
                {k: pt.as_tensor_variable(np.float64(v)) for k, v in z.items()}
            ).eval()
        )
        assert as_jax == pytest.approx(core, rel=1e-10), family
        assert as_pt == pytest.approx(core, rel=1e-10), family
