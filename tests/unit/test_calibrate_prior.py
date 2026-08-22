"""The prior route: design factor, moment matching, ``derive_prior``, and ``build`` wiring."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import stats

from axiom.calibrate.evidence import Measurement
from axiom.calibrate.prior import (
    CalibratedSpec,
    amplitude_prior,
    combine_measurements,
    derive_prior,
    design_factor,
    lognormal_from_moments,
    mean_sd_to_gamma,
)
from axiom.core import (
    D,
    Intervention,
    LedgerLine,
    Outcome,
    Population,
    Prior,
    TimeWindow,
    Treatment,
    Unsupported,
    load_spec,
    log_prior,
    walk,
)
from axiom.estimands import Estimand, Level, Quantity
from axiom.surface import SurfaceSpec, build
from axiom.surface.kernels import (
    KERNELS,
    ExponentialKernel,
    HillKernel,
    LinearKernel,
    LogisticKernel,
    PowerKernel,
)

A = Treatment(name="a", dimension=D.currency, unit="USD")
B = Treatment(name="b", dimension=D.currency, unit="USD")
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


def spec(**over: Any) -> SurfaceSpec:
    base: dict[str, Any] = dict(
        name="arms",
        treatments=(A, B),
        outcome=Y,
        kernels={"a": HillKernel(reference_dose=2.0), "b": LinearKernel()},
    )
    base.update(over)
    return SurfaceSpec(**base)


def draws(factor: float = 1.8, n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = rng.lognormal(0.0, 0.3, n)
    contribution = beta * factor * np.exp(rng.normal(0.0, 0.05, n))
    return beta, contribution


def walk_param_names(expr: Any) -> list[str]:
    return [n.name for n in walk(expr) if type(n).__name__ == "Param"]


# -- closed forms ----------------------------------------------------------------------


def test_design_factor_is_mean_of_ratios_and_validates() -> None:
    b = np.array([1.0, 2.0, 4.0])
    c = np.array([10.0, 10.0, 10.0])
    assert design_factor(b, c) == pytest.approx(np.mean(c / b))
    assert design_factor(b, 3.0 * b) == pytest.approx(3.0)
    for bad in ((b, c[:2]), ([], []), ([0.0, 1.0], [1.0, 1.0]), ([1.0, np.nan], [1.0, 1.0])):
        with pytest.raises(ValueError):
            design_factor(*bad)


def test_mean_sd_to_gamma_matches_moments() -> None:
    shape, rate = mean_sd_to_gamma(2.5, 0.4)
    assert shape == pytest.approx(39.0625)
    assert rate == pytest.approx(15.625)
    g = stats.gamma(a=shape, scale=1 / rate)
    assert g.mean() == pytest.approx(2.5) and g.std() == pytest.approx(0.4)
    for m, s in ((0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (math.inf, 1.0)):
        with pytest.raises(ValueError):
            mean_sd_to_gamma(m, s)


def test_lognormal_from_moments_matches_moments() -> None:
    mu, sigma = lognormal_from_moments(2.5, 0.4)
    assert sigma == pytest.approx(math.sqrt(math.log(1 + (0.4 / 2.5) ** 2)), rel=1e-14)
    ln = stats.lognorm(s=sigma, scale=math.exp(mu))
    assert ln.mean() == pytest.approx(2.5, rel=1e-12)
    assert ln.std() == pytest.approx(0.4, rel=1e-12)
    with pytest.raises(ValueError):
        lognormal_from_moments(1.0, -1.0)


def test_amplitude_prior_families() -> None:
    assert amplitude_prior(2.0, 0.5).family == "lognormal"
    g = amplitude_prior(2.0, 0.5, "gamma")
    assert g.family == "gamma" and g.hyper == {"alpha": 16.0, "beta": 8.0}
    with pytest.raises(ValueError):
        amplitude_prior(2.0, 0.5, "halfnormal")  # type: ignore[arg-type]


# -- kernels carry an explicit amplitude prior ------------------------------------------


@pytest.mark.parametrize("kernel_cls", list(KERNELS.values()))
def test_every_kernel_accepts_amplitude_prior(kernel_cls: type[Any]) -> None:
    prior = Prior(family="lognormal", hyper={"mu": 0.2, "sigma": 0.3})
    kernel = kernel_cls(amplitude_prior=prior)
    params = kernel.parameters("a", D.currency, D.outcome)
    amplitudes = [p for p in params if kernel.roles[p.name.removesuffix("_a")] == "amplitude"]
    # the basis families declare one coefficient per basis function; each takes the prior
    assert amplitudes and all(p.prior == prior for p in amplitudes)
    names = {p.name for p in amplitudes}
    default = kernel_cls().parameters("a", D.currency, D.outcome)
    assert [p.prior for p in default if p.name not in names] == [
        p.prior for p in params if p.name not in names
    ]
    assert load_spec(kernel.to_json()) == kernel
    assert kernel.content_hash() != kernel_cls().content_hash()


@pytest.mark.parametrize("kernel_cls", list(KERNELS.values()))
def test_amplitude_prior_support_is_checked(kernel_cls: type[Any]) -> None:
    """Saturating families take positive priors only; a basis family may take a signed one."""
    signed = Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0})
    if set(kernel_cls().roles.values()) == {"amplitude"} and len(kernel_cls().roles) > 1:
        assert kernel_cls(amplitude_prior=signed).amplitude_prior == signed
    else:
        with pytest.raises(ValueError, match="positive support"):
            kernel_cls(amplitude_prior=signed)
    with pytest.raises(ValueError, match="other parameters"):
        kernel_cls(amplitude_prior=Prior(family="lognormal", hyper={"mu": "m", "sigma": 1.0}))
    with pytest.raises(ValueError, match="amplitude_prior must be one of"):
        kernel_cls(amplitude_prior=Prior(family="uniform", hyper={"low": 0.1, "high": 1.0}))


def test_default_kernels_unchanged() -> None:
    for cls in (HillKernel, LogisticKernel, ExponentialKernel, PowerKernel, LinearKernel):
        k = cls()
        assert k.amplitude_prior is None
        amp = [p for p in k.parameters("a", D.currency, D.outcome) if "beta" in p.name]
        assert amp[0].prior == Prior(family="halfnormal", hyper={"sigma": 1.0})


# -- combine ---------------------------------------------------------------------------


def test_combine_measurements_same_estimand() -> None:
    m1, m2 = measurement(), measurement(estimate=3.1, se=0.9, source="study-2")
    out = combine_measurements([m1, m2])
    assert not isinstance(out, Unsupported)
    mean, se, lines, statuses = out
    w = np.array([1 / 0.16, 1 / 0.81])
    assert mean == pytest.approx(float((w * [2.5, 3.1]).sum() / w.sum()))
    assert se == pytest.approx(float(np.sqrt(1 / w.sum())))
    assert statuses == {"study-1": "identified", "study-2": "identified"}
    (line,) = lines
    assert line.kind == "evidence:combine"
    assert line.assumption is not None
    assert float(line.detail["counterfactual"]) == 2.5  # the most precise study
    assert float(line.detail["value"]) == mean


def test_combine_refuses_different_estimand_without_correction() -> None:
    other = measurement(
        estimand=estimand(name="lift_10", intervention=Intervention(doses={"a": 10.0})),
        source="study-2",
    )
    out = combine_measurements([measurement(), other])
    assert isinstance(out, Unsupported)
    assert out.missing == ("correction:study-2",)
    assert "study-2" in out.reason


def test_combine_applies_correction_with_ledger_line() -> None:
    other = measurement(
        estimand=estimand(name="lift_10", intervention=Intervention(doses={"a": 10.0})),
        estimate=4.0,
        se=0.8,
        source="study-2",
    )
    out = combine_measurements([measurement(), other], corrections={"study-2": 0.5})
    assert not isinstance(out, Unsupported)
    mean, se, lines, statuses = out
    assert statuses["study-2"] == "corrected"
    assert [line.kind for line in lines] == ["transfer:correction", "evidence:combine"]
    corr = lines[0]
    assert float(corr.detail["counterfactual"]) == 4.0
    assert float(corr.detail["value"]) == 2.0
    assert corr.detail["facets"] == "intervention"
    assert corr.source == other.target and corr.target == measurement().target
    # corrected study has estimate 2.0, se 0.4 -> equal-weight average with 2.5
    assert mean == pytest.approx(2.25)
    assert se == pytest.approx(0.4 / math.sqrt(2))
    with pytest.raises(ValueError):
        combine_measurements([measurement(), other], corrections={"study-2": -1.0})


def test_combine_rejects_duplicate_sources_and_empty() -> None:
    with pytest.raises(ValueError):
        combine_measurements([measurement(), measurement()])
    assert isinstance(combine_measurements([]), Unsupported)


# -- derive_prior ----------------------------------------------------------------------


def test_derive_prior_lognormal_end_to_end() -> None:
    m1, m2 = measurement(), measurement(estimate=3.1, se=0.9, source="study-2")
    beta, contribution = draws()
    base = spec()
    out = derive_prior([m1, m2], base, "a", beta_draws=beta, contribution_draws=contribution)
    assert isinstance(out, CalibratedSpec)
    factor = float(np.mean(contribution / beta))
    assert out.design_factor == pytest.approx(factor)
    assert out.amplitude_mean == pytest.approx(out.target_mean / factor)
    assert out.amplitude_sd == pytest.approx(out.target_se / factor)
    mu, sigma = lognormal_from_moments(out.amplitude_mean, out.amplitude_sd)
    assert out.prior == Prior(family="lognormal", hyper={"mu": mu, "sigma": sigma})
    assert out.parameter == "beta_a" and out.treatment == "a"
    assert out.sources == ("study-1", "study-2")
    assert out.target == estimand().content_hash()
    # only the amplitude prior of kernel "a" changed
    assert out.spec.kernel_of("a") == HillKernel(reference_dose=2.0, amplitude_prior=out.prior)
    assert out.spec.kernel_of("b") == base.kernel_of("b")
    assert out.spec.model_dump(exclude={"kernels"}) == base.model_dump(exclude={"kernels"})
    assert out.spec.content_hash() != base.content_hash()
    # ledger: three steps, every line names an assumption and carries both numbers
    assert [line.kind for line in out.ledger_lines] == [
        "evidence:combine",
        "prior:design_factor",
        "prior:moment_match",
    ]
    for line in out.ledger_lines:
        assert isinstance(line, LedgerLine) and line.assumption is not None
        assert "counterfactual" in line.detail and "value" in line.detail
    df_line = out.ledger_lines[1]
    assert float(df_line.detail["counterfactual"]) == pytest.approx(out.target_mean)
    assert float(df_line.detail["value"]) == pytest.approx(out.amplitude_mean)
    # the result round-trips as a Spec
    assert load_spec(out.to_json()) == out


def test_derive_prior_gamma_family() -> None:
    beta, contribution = draws()
    out = derive_prior(
        [measurement()],
        spec(),
        "a",
        beta_draws=beta,
        contribution_draws=contribution,
        family="gamma",
    )
    assert isinstance(out, CalibratedSpec)
    shape, rate = mean_sd_to_gamma(out.amplitude_mean, out.amplitude_sd)
    assert out.prior == Prior(family="gamma", hyper={"alpha": shape, "beta": rate})


def test_derive_prior_linear_kernel_targets_rate() -> None:
    beta, contribution = draws()
    out = derive_prior(
        [
            measurement(
                estimand=estimand(
                    treatment=B,
                    intervention=Intervention(doses={"b": 5.0}),
                    reference=Intervention(doses={"b": 0.0}),
                )
            )
        ],
        spec(),
        "b",
        beta_draws=beta,
        contribution_draws=contribution,
    )
    assert isinstance(out, CalibratedSpec)
    assert out.parameter == "beta_rate_b"
    model = build(out.spec)
    (p,) = [p for p in model.parameters if p.name == "beta_rate_b"]
    assert p.prior == out.prior


def test_build_uses_the_calibrated_prior_and_log_prior_moves() -> None:
    beta, contribution = draws()
    base = spec()
    out = derive_prior([measurement()], base, "a", beta_draws=beta, contribution_draws=contribution)
    assert isinstance(out, CalibratedSpec)
    before, after = build(base), build(out.spec)
    (p_before,) = [p for p in before.parameters if p.name == "beta_a"]
    (p_after,) = [p for p in after.parameters if p.name == "beta_a"]
    assert p_before.prior == Prior(family="halfnormal", hyper={"sigma": 1.0})
    assert p_after.prior == out.prior
    assert [p.name for p in before.parameters] == [p.name for p in after.parameters]
    others = [
        (p.prior, q.prior)
        for p, q in zip(before.parameters, after.parameters, strict=True)
        if p.name != "beta_a"
    ]
    assert all(p == q for p, q in others)
    # the expression tree is unchanged apart from the prior the amplitude Param carries
    strip = {"parameters", "mean"}
    assert before.model_dump(exclude=strip) == after.model_dump(exclude=strip)
    assert [type(n).__name__ for n in walk(before.mean)] == [
        type(n).__name__ for n in walk(after.mean)
    ]
    assert walk_param_names(before.mean) == walk_param_names(after.mean)
    theta = {"alpha": 0.1, "k_a": 2.0, "s_a": 1.5, "beta_a": 1.3, "beta_rate_b": 0.3, "sigma": 0.5}
    delta = log_prior(after, theta) - log_prior(before, theta)
    mu, sigma = out.prior.hyper["mu"], out.prior.hyper["sigma"]
    expected = stats.lognorm.logpdf(1.3, s=sigma, scale=math.exp(mu)) - stats.halfnorm.logpdf(
        1.3, scale=1.0
    )
    assert delta == pytest.approx(float(expected), rel=1e-12)
    assert delta != 0.0


def test_derive_prior_passes_unsupported_through_and_validates() -> None:
    beta, contribution = draws()
    other = measurement(
        estimand=estimand(name="lift_10", intervention=Intervention(doses={"a": 10.0})),
        source="study-2",
    )
    out = derive_prior(
        [measurement(), other], spec(), "a", beta_draws=beta, contribution_draws=contribution
    )
    assert isinstance(out, Unsupported)
    assert out.missing == ("correction:study-2",)
    fixed = derive_prior(
        [measurement(), other],
        spec(),
        "a",
        beta_draws=beta,
        contribution_draws=contribution,
        corrections={"study-2": 0.5},
    )
    assert isinstance(fixed, CalibratedSpec)
    assert fixed.plan_statuses == {"study-1": "identified", "study-2": "corrected"}
    assert fixed.ledger_lines[0].kind == "transfer:correction"
    with pytest.raises(KeyError):
        derive_prior(
            [measurement()], spec(), "zzz", beta_draws=beta, contribution_draws=contribution
        )
    with pytest.raises(ValueError, match="not positive"):
        derive_prior(
            [measurement(estimate=-2.5)],
            spec(),
            "a",
            beta_draws=beta,
            contribution_draws=contribution,
        )
    with pytest.raises(ValueError):
        derive_prior(
            [measurement()], spec(), "a", beta_draws=beta, contribution_draws=contribution[:10]
        )


def test_derive_prior_is_deterministic_and_hashes_differ_by_evidence() -> None:
    beta, contribution = draws()
    a = derive_prior([measurement()], spec(), "a", beta_draws=beta, contribution_draws=contribution)
    b = derive_prior([measurement()], spec(), "a", beta_draws=beta, contribution_draws=contribution)
    c = derive_prior(
        [measurement(se=0.8)], spec(), "a", beta_draws=beta, contribution_draws=contribution
    )
    assert isinstance(a, CalibratedSpec) and isinstance(b, CalibratedSpec)
    assert isinstance(c, CalibratedSpec)
    assert a.content_hash() == b.content_hash()
    assert a.spec.content_hash() != c.spec.content_hash()
    assert c.amplitude_sd > a.amplitude_sd
