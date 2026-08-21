"""Unit tests for ``meta.pool``, ``meta.moderators``, ``meta.bias``, ``meta.priors`` and the
``Likelihood.scale_expr`` extension of ``core`` they rely on."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from axiom.core import (
    Add,
    Data,
    Gather,
    LedgerLine,
    Likelihood,
    ModelSpec,
    Param,
    Pow,
    Prior,
    Unsupported,
    Unverified,
    compile_log_density,
    dimensionless,
    jax_available,
    log_density,
)
from axiom.core.model import likelihood_scale
from axiom.meta.bias import NO_DUAL_READ, delta_identification
from axiom.meta.moderators import ModeratorDesign, moderator_matrix
from axiom.meta.pool import (
    EffectShrinkage,
    ParameterSummary,
    Pooled,
    PoolPriors,
    PoolResult,
    PoolSpec,
    pool,
    pool_model,
)
from axiom.meta.priors import prior_from_pool
from axiom.meta.schema import Corpus, StudyRecord

DL = dimensionless()


def _rec(
    study: str,
    estimate: float,
    se: float,
    *,
    contributor: str | None = None,
    read: str = "experiment",
    family: str = "f",
    quantity: str = "elasticity",
    moderators: dict[str, float] | None = None,
) -> StudyRecord:
    return StudyRecord(
        study=study,
        contributor=contributor or study,
        quantity=quantity,
        estimate=estimate,
        se=se,
        read=read,  # type: ignore[arg-type]
        family=family,
        moderators=moderators or {},
    )


def _corpus(seed: int = 0, k: int = 12, tau: float = 0.3, mu: float = 0.5) -> Corpus:
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(k):
        se = float(rng.uniform(0.1, 0.4))
        recs.append(_rec(f"s{i}", float(rng.normal(mu, tau) + se * rng.normal()), se))
    return Corpus(records=tuple(recs), name="unit")


# -- core: Likelihood.scale_expr -------------------------------------------------------------


def test_likelihood_scale_expr_validation() -> None:
    with pytest.raises(ValueError, match="either"):
        Likelihood(family="normal", scale="s", scale_expr=Data(name="se", dimension=DL))
    with pytest.raises(ValueError, match="scale parameter name or scale_expr"):
        Likelihood(family="normal")
    with pytest.raises(ValueError, match="poisson"):
        Likelihood(family="poisson", scale_expr=Data(name="se", dimension=DL))
    mu = Param(name="mu", dimension=DL, prior=Prior(family="normal", hyper={"mu": 0, "sigma": 1}))
    tau = Param(name="tau", dimension=DL, prior=Prior(family="halfnormal", hyper={"sigma": 1}))
    # a scale_expr may only use declared parameters
    with pytest.raises(ValueError, match="does not declare"):
        ModelSpec(
            name="m",
            mean=mu,
            outcome=Data(name="y", dimension=DL),
            likelihood=Likelihood(family="normal", scale_expr=tau),
            parameters=(mu,),
        )
    # and must carry the outcome's dimension
    from axiom.core import D, DimensionError

    with pytest.raises(DimensionError):
        ModelSpec(
            name="m",
            mean=mu,
            outcome=Data(name="y", dimension=DL),
            likelihood=Likelihood(family="normal", scale_expr=Data(name="se", dimension=D.time)),
            parameters=(mu,),
        )
    m = ModelSpec(
        name="m",
        mean=mu,
        outcome=Data(name="y", dimension=DL),
        likelihood=Likelihood(family="normal", scale_expr=Data(name="se", dimension=DL)),
        parameters=(mu,),
    )
    assert m.data_columns == ("y", "se")


def test_scale_expr_density_matches_scipy_and_jax() -> None:
    mu = Param(name="mu", dimension=DL, prior=Prior(family="normal", hyper={"mu": 0, "sigma": 2}))
    tau = Param(name="tau", dimension=DL, prior=Prior(family="halfnormal", hyper={"sigma": 1}))
    scale_expr = Pow(
        base=Add(
            terms=(
                Pow(base=Data(name="se", dimension=DL), exponent=2),
                Pow(base=tau, exponent=2),
            )
        ),
        exponent="1/2",
    )
    m = ModelSpec(
        name="m",
        mean=mu,
        outcome=Data(name="y", dimension=DL),
        likelihood=Likelihood(family="normal", scale_expr=scale_expr),
        parameters=(mu, tau),
    )
    data = {"y": np.array([0.2, 0.9, -0.3]), "se": np.array([0.1, 0.3, 0.2])}
    z = {"mu": np.asarray(0.4), "tau": np.asarray(np.log(0.5))}
    theta = {"mu": 0.4, "tau": 0.5}
    s = likelihood_scale(m, data, theta)
    np.testing.assert_allclose(s, np.sqrt(data["se"] ** 2 + 0.25))
    expected = (
        stats.norm.logpdf(0.4, 0, 2)
        + stats.halfnorm.logpdf(0.5, scale=1)
        + np.log(0.5)  # Jacobian of tau = exp(v)
        + stats.norm.logpdf(data["y"], 0.4, s).sum()
    )
    assert log_density(m, data, z) == pytest.approx(expected, rel=1e-12)
    if jax_available():
        import jax

        jax.config.update("jax_enable_x64", True)
        f = compile_log_density(m)
        assert float(f(data, z)) == pytest.approx(expected, rel=1e-10)


# -- moderators ------------------------------------------------------------------------------


def test_moderator_matrix_centers_and_refuses() -> None:
    recs = (
        _rec("a", 0.1, 0.1, moderators={"dose": 1.0, "weeks": 4}),
        _rec("b", 0.2, 0.1, moderators={"dose": 3.0, "weeks": 4}),
        _rec("c", 0.3, 0.1, moderators={"dose": 5.0, "weeks": 8}),
    )
    c = Corpus(records=recs)
    md = moderator_matrix(c, ["dose"])
    assert isinstance(md, ModeratorDesign)
    np.testing.assert_allclose(md.column("dose"), [-2.0, 0.0, 2.0])
    np.testing.assert_allclose(md.means, [3.0])
    np.testing.assert_allclose(md.center({"dose": 4.0}), [1.0])
    assert md.n_records == 3 and md.n_moderators == 1
    with pytest.raises(KeyError):
        md.column("weeks")
    missing = moderator_matrix(Corpus(records=(recs[0], _rec("d", 0.1, 0.1))), ["dose"])
    assert isinstance(missing, Unsupported) and "d" in missing.reason
    with pytest.raises(ValueError):
        moderator_matrix(c, ["dose", "dose"])
    const = moderator_matrix(Corpus(records=recs[:2]), ["weeks"])
    assert isinstance(const, Unsupported) and "constant" in const.reason


# -- bias ------------------------------------------------------------------------------------


def test_delta_identification_verdicts() -> None:
    single = Corpus(
        records=(
            _rec("a", 0.1, 0.1, contributor="p", read="model"),
            _rec("b", 0.2, 0.1, contributor="q", read="experiment"),
        )
    )
    v = delta_identification(single, "f")
    assert v.status == "blocked" and NO_DUAL_READ in v.reason and not v.licensed
    dual = Corpus(
        records=single.records + (_rec("c", 0.3, 0.1, contributor="p", read="experiment"),)
    )
    v = delta_identification(dual, "f")
    assert v.status == "identified" and "p" in v.route
    # other families do not lend identification
    other = Corpus(records=dual.records + (_rec("d", 0.1, 0.1, family="g", read="model"),))
    assert delta_identification(other, "g").status == "blocked"
    assert delta_identification(other).status == "blocked"  # all families must qualify
    assert delta_identification(Corpus(records=())).status == "blocked"


# -- spec ------------------------------------------------------------------------------------


def test_pool_spec_validation() -> None:
    PoolSpec(family="f")  # constructor example
    PoolPriors(mu_scale=2.0, tau_scale=0.5, tau_fixed=0.3)
    with pytest.raises(ValueError):
        PoolSpec(family="f", moderators=("a", "a"))
    with pytest.raises(ValueError):
        PoolSpec(family="f", bias_term=True, effect_key="study")
    with pytest.raises(ValueError):
        PoolSpec(family="f", mass=1.0)
    with pytest.raises(ValueError):
        PoolPriors(tau_fixed=-1.0)
    with pytest.raises(ValueError):
        PoolPriors(mu_scale=0.0)
    assert PoolSpec(family="f").marginal
    assert not PoolSpec(family="f", parametrization="centered").marginal
    assert PoolPriors(tau_fixed=0.2).tau_prior().family == "fixed"
    assert PoolPriors().tau_prior().family == "halfnormal"


# -- model construction ------------------------------------------------------------------------


def test_pool_model_centered_is_the_hierarchy_in_the_tree() -> None:
    c = _corpus()
    built = pool_model(PoolSpec(family="f", parametrization="centered"), c)
    assert not isinstance(built, Unsupported)
    model, data = built
    theta = model.parameter("theta_f")
    assert theta.shape == (12,)
    assert theta.prior is not None and theta.prior.parents == ("mu_f", "tau_f")
    assert model.likelihood.scale_expr == Data(name="se", dimension=DL)
    assert isinstance(model.mean, Add) and isinstance(model.mean.terms[0], Gather)
    assert set(data) == {"y", "se", "effect_index", "is_model_read"}
    assert model.free == (model.parameter("mu_f"), model.parameter("tau_f"), theta)
    assert model.content_hash()  # serializable


def test_pool_model_marginal_rotation_is_orthonormal_and_exact() -> None:
    # two records sharing a contributor, one alone
    recs = (
        _rec("a", 0.5, 0.2, contributor="p", read="model"),
        _rec("b", 0.1, 0.1, contributor="p", read="experiment"),
        _rec("c", 0.7, 0.3, contributor="q"),
    )
    c = Corpus(records=recs)
    built = pool_model(PoolSpec(family="f", priors=PoolPriors(tau_fixed=0.3)), c)
    assert not isinstance(built, Unsupported)
    model, data = built
    assert "theta_f" not in {p.name for p in model.parameters}
    assert model.likelihood.scale_expr is not None
    # loadings: one row per effect carries mu/se_g and tau²/se_g²
    se_p = 1.0 / np.sqrt(1 / 0.2**2 + 1 / 0.1**2)
    np.testing.assert_allclose(sorted(data["mu_load"]), sorted([1 / se_p, 0.0, 1 / 0.3]))
    np.testing.assert_allclose(data["tau2_load"], data["mu_load"] ** 2)
    # the rotated outcome keeps the whitened norm (orthonormal rotation)
    w = np.array([0.5 / 0.2, 0.1 / 0.1, 0.7 / 0.3])
    assert float(np.sum(data["outcome"] ** 2)) == pytest.approx(float(np.sum(w**2)))
    # the marginal likelihood equals the analytic marginal of the hierarchy at a few points
    theta_bar = lambda mu: np.array([mu, mu, mu])  # noqa: E731
    for mu in (0.0, 0.4):
        z = {"mu_f": np.asarray(mu)}
        ld = log_density(model, data, z)
        y = np.array([0.5, 0.1, 0.7])
        cov = np.diag([0.2**2, 0.1**2, 0.3**2]) + 0.09 * np.array(
            [[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float
        )
        analytic = stats.multivariate_normal.logpdf(y, theta_bar(mu), cov) + stats.norm.logpdf(
            mu, 0, 1
        )
        # the whitening drops the constant sum(log se); compare differences across mu instead
        if mu == 0.0:
            base = (ld, analytic)
        else:
            assert ld - base[0] == pytest.approx(analytic - base[1], rel=1e-10)


def test_pool_model_refusals() -> None:
    c = _corpus()
    assert isinstance(pool_model(PoolSpec(family="nope"), c), Unsupported)
    mixed = Corpus(records=c.records + (_rec("x", 0.2, 0.1, quantity="correlation"),))
    out = pool_model(PoolSpec(family="f"), mixed)
    assert isinstance(out, Unsupported) and "mixes" in out.reason
    out = pool_model(PoolSpec(family="f", moderators=("dose",)), c)
    assert isinstance(out, Unsupported) and "moderator" in out.reason
    scaled = Corpus(
        records=(
            StudyRecord(
                study="u",
                contributor="u",
                quantity="elasticity",
                estimate=0.1,
                se=0.1,
                read="model",
                family="f",
                unit_scale="per_kg",
            ),
        )
    )
    out = pool_model(PoolSpec(family="f"), scaled)
    assert isinstance(out, Unsupported) and "TransferPlan" in out.reason
    with pytest.raises(ValueError):
        pool(PoolSpec(family="f"), c, draws=1)
    assert isinstance(pool(PoolSpec(family="f"), c, backend="nope"), Unsupported)


# -- fitting -----------------------------------------------------------------------------------


def test_pool_fixed_tau_both_parametrizations_agree() -> None:
    c = _corpus()
    spec = PoolSpec(family="f", priors=PoolPriors(tau_fixed=0.3), effect_key="study")
    a = pool(spec, c, draws=4000, seed=1)
    b = pool(spec.model_copy(update={"parametrization": "centered"}), c, draws=4000, seed=1)
    assert isinstance(a, Pooled) and isinstance(b, Pooled)
    ra, rb = a.result, b.result
    assert isinstance(ra, PoolResult)
    assert ra.mu.mean == pytest.approx(rb.mu.mean, abs=4 * ra.mu.sd / np.sqrt(4000))
    assert ra.mu.sd == pytest.approx(rb.mu.sd, rel=0.05)
    assert ra.tau.mean == 0.3 and ra.tau.sd == 0.0 and ra.tau.note == "fixed by the spec"
    assert ra.k == 12 and ra.n_effects == 12 and len(ra.shrinkage) == 12
    assert ra.delta is None and ra.delta_verdict is None
    assert "theta_f" in a.posterior.names() and "theta_f" in b.posterior.names()
    assert a.posterior.flat("theta_f").shape == (4000, 12)
    for s in ra.shrinkage:
        assert isinstance(s, EffectShrinkage)
        assert s.analytic == pytest.approx(s.se**2 / (s.se**2 + 0.09))
    assert ra.mu.interval.definition == "eti" and ra.mu.interval.mass == 0.95
    assert ra.model_hash == a.model.content_hash()
    assert "marginal" in ra.detail["parametrization"]
    assert "centered" in rb.detail["parametrization"]


def test_pool_moderator_and_log_scale() -> None:
    rng = np.random.default_rng(3)
    recs = []
    for i in range(20):
        x = float(rng.uniform(-1, 1))
        se = 0.1
        y = 0.5 + 0.8 * x + 0.1 * rng.normal() + se * rng.normal()
        recs.append(_rec(f"s{i}", y, se, moderators={"x": x}))
    c = Corpus(records=tuple(recs))
    out = pool(PoolSpec(family="f", moderators=("x",)), c, draws=4000, seed=0)
    assert isinstance(out, Pooled)
    (g,) = out.result.gamma
    assert isinstance(g, ParameterSummary) and g.name == "gamma_x_f"
    assert abs(g.mean - 0.8) < 4 * g.sd
    # with tau fixed and flat priors the pool is exactly weighted least squares on centered x
    flat = PoolPriors(tau_fixed=0.1, mu_scale=100.0, gamma_scale=100.0)
    out = pool(PoolSpec(family="f", moderators=("x",), priors=flat), c, draws=50_000, seed=0)
    assert isinstance(out, Pooled)
    x = np.array([r.moderators["x"] for r in recs])
    y = np.array([r.estimate for r in recs])
    X = np.column_stack([np.ones(20), x - x.mean()])
    XtWX = X.T @ X / (0.1**2 + 0.1**2)
    beta = np.linalg.solve(XtWX, X.T @ y / (0.1**2 + 0.1**2))
    sd = np.sqrt(np.diag(np.linalg.inv(XtWX)))
    assert out.result.gamma[0].mean == pytest.approx(beta[1], abs=4 * sd[1] / np.sqrt(50_000))
    assert out.result.gamma[0].sd == pytest.approx(sd[1], rel=0.03)
    assert out.result.mu.mean == pytest.approx(beta[0], abs=4 * sd[0] / np.sqrt(50_000))
    # log-scale quantity: pooled as log(estimate) with delta-method se
    ratio = Corpus(
        records=tuple(
            _rec(f"r{i}", float(np.exp(0.3 + 0.05 * rng.normal())), 0.1, quantity="response_ratio")
            for i in range(10)
        )
    )
    out = pool(PoolSpec(family="f"), ratio, draws=2000, seed=0)
    assert isinstance(out, Pooled)
    assert out.result.scale == "log" and "log_scale" in out.result.detail
    assert abs(out.result.mu.mean - 0.3) < 4 * out.result.mu.sd


def test_pool_bias_term_identified_and_not() -> None:
    rng = np.random.default_rng(5)
    recs = []
    for i in range(10):
        th = float(rng.normal(0.5, 0.2))
        recs.append(_rec(f"e{i}", th + 0.1 * rng.normal(), 0.1, contributor=f"c{i}"))
        recs.append(
            _rec(f"m{i}", th + 0.4 + 0.1 * rng.normal(), 0.1, contributor=f"c{i}", read="model")
        )
    dual = Corpus(records=tuple(recs))
    out = pool(PoolSpec(family="f", bias_term=True), dual, draws=4000, seed=0)
    assert isinstance(out, Pooled)
    r = out.result
    assert r.delta is not None and r.delta.identified and r.delta_verdict is not None
    assert r.delta_verdict.status == "identified"
    assert abs(r.delta.mean - 0.4) < 3 * r.delta.sd
    assert r.n_effects == 10 and r.k == 20
    assert all(len(s.studies) == 2 for s in r.shrinkage)
    # no dual read: blocked, prior reported, flagged
    model_only = Corpus(records=tuple(x for x in recs if x.read == "model"))
    out = pool(PoolSpec(family="f", bias_term=True), model_only, draws=4000, seed=0)
    assert isinstance(out, Pooled)
    r = out.result
    assert r.delta is not None and not r.delta.identified
    assert r.delta_verdict is not None and r.delta_verdict.status == "blocked"
    assert NO_DUAL_READ in r.delta.note and "NOT identified" in r.detail["delta"]
    assert r.delta.sd == pytest.approx(1.0, rel=0.1)
    assert "model_read" not in out.data


def test_centered_free_tau_laplace_is_refused_not_wrong() -> None:
    # the joint centered density has no mode; laplace must say so rather than return numbers
    out = pool(PoolSpec(family="f", parametrization="centered"), _corpus(k=20), draws=500, seed=0)
    assert isinstance(out, Unverified)


# -- priors handoff -----------------------------------------------------------------------------


def test_prior_from_pool_targets_and_families() -> None:
    out = pool(PoolSpec(family="f"), _corpus(k=20), draws=4000, seed=0)
    assert isinstance(out, Pooled)
    r = out.result
    p, line = prior_from_pool(r, target="mu")
    assert p.family == "normal" and isinstance(line, LedgerLine)
    assert p.hyper["mu"] == r.mu.mean and p.hyper["sigma"] == r.mu.sd
    q, _ = prior_from_pool(r, target="predictive")
    expected = np.sqrt(r.tau.mean**2 + r.tau.sd**2 + r.mu.sd**2)
    assert q.hyper["sigma"] == pytest.approx(expected)
    assert q.hyper["sigma"] > p.hyper["sigma"]
    ln, line = prior_from_pool(r, target="mu", family="lognormal")
    assert ln.family == "lognormal"
    # moment matched: lognormal mean and sd equal the pool's
    s2 = float(ln.hyper["sigma"]) ** 2
    mean = np.exp(float(ln.hyper["mu"]) + s2 / 2)
    sd = mean * np.sqrt(np.expm1(s2))
    assert mean == pytest.approx(r.mu.mean) and sd == pytest.approx(r.mu.sd)
    assert line.kind == "prior_from_pool" and line.source == r.model_hash
    assert line.detail["target"] == "mu" and line.detail["prior_family"] == "lognormal"
    with pytest.raises(ValueError):
        prior_from_pool(r, target="nope")  # type: ignore[arg-type]
    neg = r.model_copy(update={"mu": r.mu.model_copy(update={"mean": -0.2})})
    with pytest.raises(ValueError, match="positive"):
        prior_from_pool(neg, family="lognormal")
    # a log-scale pool hands over the lognormal directly
    logged = r.model_copy(update={"scale": "log"})
    ln2, _ = prior_from_pool(logged, target="mu", family="lognormal")
    assert ln2.hyper == {"mu": r.mu.mean, "sigma": r.mu.sd}


def test_pool_accepts_scaled_record_admitted_by_normalize() -> None:
    """A record that ``normalize`` admitted under a licensed plan is poolable (Phase 9 gap)."""
    from axiom.meta import Corpus, PoolSpec, StudyRecord, pool_model

    base = dict(quantity="elasticity", read="experiment", family="f")
    admitted = StudyRecord(
        study="scaled",
        contributor="c3",
        estimate=0.4,
        se=0.1,
        unit_scale="per_area",
        detail={"transfer_status": "downgraded", "transfer_plan_hash": "ab" * 32},
        **base,
    )
    refused = admitted.model_copy(update={"study": "scaled2", "detail": {}})
    ok = [
        StudyRecord(study="s1", contributor="c1", estimate=0.3, se=0.1, **base),
        StudyRecord(study="s2", contributor="c2", estimate=0.5, se=0.1, **base),
    ]
    spec = PoolSpec(family="f")
    out = pool_model(spec, Corpus(records=(*ok, admitted)))
    assert not isinstance(out, Unsupported), out
    bad = pool_model(spec, Corpus(records=(*ok, refused)))
    assert isinstance(bad, Unsupported) and "scaled2" in bad.reason
