from __future__ import annotations

import numpy as np
import pytest

from axiom.core import Unsupported, Unverified
from axiom.meta import Corpus, Pooled, PoolPriors, PoolSpec, StudyRecord, pool, pool_model

MU, TAU_PARTY, TAU_STUDY = 0.5, 0.4, 0.15
_OFFSETS = (-1.0, 0.2, 0.9, -0.3)


def _corpus(
    *,
    n_parties: int = 3,
    n_studies: int = 4,
    seed: int = 0,
    reads: tuple[str, ...] = ("experiment",),
) -> tuple[Corpus, dict[str, float]]:
    """A corpus whose parties genuinely differ, with several studies inside each."""
    rng = np.random.default_rng(seed)
    parties = [f"party{i}" for i in range(n_parties)]
    alpha = {p: MU + TAU_PARTY * o for p, o in zip(parties, _OFFSETS, strict=False)}
    records = []
    for p in parties:
        for j in range(n_studies):
            theta = alpha[p] + TAU_STUDY * rng.standard_normal()
            for read in reads:
                se = float(rng.uniform(0.1, 0.2))
                records.append(
                    StudyRecord(
                        study=f"{p}-{j}" if len(reads) == 1 else f"{p}-{j}-{read}",
                        contributor=p,
                        quantity="elasticity",
                        family="fert",
                        estimate=float(theta + se * rng.standard_normal()),
                        se=se,
                        read=read,  # type: ignore[arg-type]
                    )
                )
    return Corpus(records=tuple(records), name="corpus"), alpha


def _nested(**priors: float) -> PoolSpec:
    return PoolSpec(family="fert", effect_key="nested", priors=PoolPriors(**priors))


def _fit(spec: PoolSpec, corpus: Corpus, **kw: object) -> Pooled:
    result = pool(spec, corpus, **kw)  # type: ignore[arg-type]
    assert isinstance(result, Pooled), getattr(result, "reason", result)
    return result


# -- the spec -------------------------------------------------------------------------------


def test_the_nested_key_resolves_to_the_nested_tree() -> None:
    spec = _nested()
    assert spec.tree == "nested" and spec.nested and spec.marginal
    assert PoolSpec(family="f").tree == "marginal"
    assert PoolSpec(family="f", parametrization="centered").tree == "centered"


def test_the_nested_tree_and_the_nested_key_come_together() -> None:
    with pytest.raises(ValueError, match="nested tree is the nested effect key"):
        PoolSpec(family="f", effect_key="nested", parametrization="centered")
    with pytest.raises(ValueError, match="nested tree is the nested effect key"):
        PoolSpec(family="f", effect_key="study", parametrization="nested")


def test_the_bias_term_is_allowed_with_nested_and_refused_with_study() -> None:
    """The nested tree keeps delta identified: a party's reads share its alpha."""
    assert PoolSpec(family="f", effect_key="nested", bias_term=True).bias_term
    with pytest.raises(ValueError, match="bias_term needs effect_key"):
        PoolSpec(family="f", effect_key="study", bias_term=True)


def test_the_party_prior_validates_like_the_others() -> None:
    with pytest.raises(ValueError):
        PoolPriors(tau_party_scale=0.0)
    with pytest.raises(ValueError, match="tau_party_fixed"):
        PoolPriors(tau_party_fixed=-1.0)
    assert PoolPriors(tau_party_fixed=0.4).tau_party_prior().family == "fixed"
    assert PoolPriors().tau_party_prior().family == "halfnormal"


# -- the tree -------------------------------------------------------------------------------


def test_the_model_carries_both_variance_components_and_the_party_offsets() -> None:
    corpus, _ = _corpus()
    built = pool_model(_nested(), corpus)
    assert not isinstance(built, Unsupported)
    model, data = built
    names = [p.name for p in model.parameters]
    assert names == ["mu_fert", "tau_fert", "tau_party_fert", "z_party_fert"]
    assert model.parameters[-1].shape == (3,)
    assert model.parameters[-1].prior.family == "normal"  # non-centered: z ~ N(0, 1)
    assert model.parameters[-1].prior.hyper == {"mu": 0.0, "sigma": 1.0}
    assert "party_index" in data and data["party_index"].size == data["outcome"].size


def test_each_record_loads_its_own_partys_offset() -> None:
    corpus, _ = _corpus(n_parties=3, n_studies=2)
    built = pool_model(_nested(), corpus)
    assert not isinstance(built, Unsupported)
    _, data = built
    # Twelve records over six studies in three parties: two studies per party.
    assert sorted(set(data["party_index"].tolist())) == [0.0, 1.0, 2.0]
    assert np.count_nonzero(data["mu_load"]) == 6  # one loading row per study


# -- what the level buys ---------------------------------------------------------------------


def test_the_family_mean_is_no_more_certain_than_the_number_of_parties() -> None:
    """The headline: 'study' treats k studies as k independent draws and mu is too tight."""
    corpus, _ = _corpus(n_parties=3, n_studies=4)
    flat = _fit(PoolSpec(family="fert", effect_key="study"), corpus, backend="laplace", draws=4000)
    nested = _fit(_nested(tau_party_fixed=TAU_PARTY), corpus, backend="laplace", draws=4000)
    assert nested.result.mu.interval.width > 2.0 * flat.result.mu.interval.width
    assert nested.result.n_parties == 3
    assert nested.result.n_effects == 12
    assert flat.result.n_parties == 0


def test_the_two_variance_components_separate() -> None:
    """'study' conflates them into one tau; nested splits it, and the within part is smaller."""
    corpus, _ = _corpus(n_parties=3, n_studies=4)
    flat = _fit(PoolSpec(family="fert", effect_key="study"), corpus, backend="laplace", draws=4000)
    nested = _fit(_nested(tau_party_fixed=TAU_PARTY), corpus, backend="laplace", draws=4000)
    assert nested.result.tau.mean < flat.result.tau.mean
    assert nested.result.tau.mean == pytest.approx(TAU_STUDY, abs=0.1)
    assert nested.result.tau_party is not None
    assert nested.result.tau_party.mean == pytest.approx(TAU_PARTY)
    assert flat.result.tau_party is None and flat.result.alphas == ()


def test_the_party_means_are_recovered_and_reported() -> None:
    corpus, alpha = _corpus(n_parties=3, n_studies=4)
    result = _fit(_nested(tau_party_fixed=TAU_PARTY), corpus, backend="laplace", draws=4000).result
    assert len(result.alphas) == 3
    for summary, truth in zip(result.alphas, alpha.values(), strict=True):
        assert summary.mean == pytest.approx(truth, abs=0.25)
        assert summary.interval.mass == 0.95
    assert [s.name for s in result.alphas] == [f"alpha_fert[party{i}]" for i in range(3)]


def test_an_effect_shrinks_toward_its_own_party_not_the_family_mean() -> None:
    corpus, _ = _corpus(n_parties=3, n_studies=4)
    result = _fit(_nested(tau_party_fixed=TAU_PARTY), corpus, backend="laplace", draws=4000).result
    by_party = {s.name.split("[")[1][:-1]: s.mean for s in result.alphas}
    for row in result.shrinkage:
        assert row.party in by_party
        assert row.toward == pytest.approx(by_party[row.party], abs=1e-9)
        assert row.toward != pytest.approx(result.mu.mean, abs=1e-9)
        # the shrunk effect lies between its raw estimate and its party's mean
        low, high = sorted((row.estimate, row.toward))
        assert low - 1e-9 <= row.theta <= high + 1e-9


def test_the_two_level_trees_still_shrink_toward_mu() -> None:
    corpus, _ = _corpus(n_parties=3, n_studies=4)
    result = _fit(PoolSpec(family="fert", effect_key="study"), corpus, backend="laplace").result
    for row in result.shrinkage:
        assert row.party == "" and row.toward is None


def test_the_detail_says_which_tree_was_fitted() -> None:
    corpus, _ = _corpus()
    result = _fit(_nested(tau_party_fixed=TAU_PARTY), corpus, backend="laplace").result
    assert "nested" in result.detail["parametrization"]
    assert "3 parties over 12 studies" in result.detail["levels"]
    assert result.detail["party_level"] == "non-centered: alpha = mu + tau_party * z_party"


def test_the_result_round_trips() -> None:
    corpus, _ = _corpus()
    from axiom.meta import PoolResult

    result = _fit(_nested(tau_party_fixed=TAU_PARTY), corpus, backend="laplace").result
    assert PoolResult.from_json(result.to_json()) == result


# -- what it refuses ------------------------------------------------------------------------


def test_one_party_is_not_two_levels() -> None:
    corpus, _ = _corpus(n_parties=1, n_studies=4)
    got = pool_model(_nested(), corpus)
    assert isinstance(got, Unsupported)
    assert "at least two parties" in got.reason


def test_one_study_per_party_is_not_two_levels_either() -> None:
    corpus, _ = _corpus(n_parties=3, n_studies=1)
    got = pool_model(_nested(), corpus)
    assert isinstance(got, Unsupported)
    assert "at least two studies" in got.reason


def test_a_study_belongs_to_one_party_by_construction() -> None:
    """The nesting is well defined because ``Corpus`` refuses a repeated study id."""
    shared = tuple(
        StudyRecord(
            study="s1",
            contributor=party,
            quantity="elasticity",
            family="f",
            estimate=1.0,
            se=0.1,
            read="experiment",
        )
        for party in ("a", "b")
    )
    with pytest.raises(ValueError, match="duplicate study ids"):
        Corpus(records=shared, name="c")


def test_a_free_between_party_sd_needs_a_sampler() -> None:
    """Measured, not assumed: see the slow test below for the numbers behind the refusal."""
    corpus, _ = _corpus()
    got = pool(_nested(), corpus, backend="laplace")
    assert isinstance(got, Unsupported)
    assert "needs a sampler" in got.reason
    assert got.missing == ("a sampler backend, or PoolPriors.tau_party_fixed",)
    # fixing it removes the ridge, and the same call goes through
    assert isinstance(pool(_nested(tau_party_fixed=0.4), corpus, backend="laplace"), Pooled)


# -- against a sampler ----------------------------------------------------------------------


@pytest.mark.slow
def test_a_sampler_recovers_both_variance_components() -> None:
    corpus, alpha = _corpus(n_parties=4, n_studies=4)
    got = pool(_nested(), corpus, backend="numpyro", draws=1000, seed=1)
    if isinstance(got, Unsupported):
        pytest.skip(f"numpyro unavailable: {got.reason}")
    assert not isinstance(got, Unverified)
    result = got.result
    assert result.mu.mean == pytest.approx(MU, abs=0.15)
    assert result.tau.mean == pytest.approx(TAU_STUDY, abs=0.1)
    assert result.tau_party is not None
    assert result.tau_party.mean == pytest.approx(TAU_PARTY, abs=0.2)
    for summary, truth in zip(result.alphas, alpha.values(), strict=True):
        assert summary.interval.contains(truth)


@pytest.mark.slow
def test_the_laplace_refusal_is_not_precious() -> None:
    """The refused fit really is wrong, and adding parties does not rescue it."""
    from axiom.meta.pool import _build, _layout, _run, _summarize, _with_effects
    from axiom.meta.pool import _names as pool_names

    spec = _nested()
    for n_parties in (4, 16):
        corpus, _ = _corpus(n_parties=min(n_parties, 4), n_studies=4, seed=n_parties)
        layout = _layout(spec, corpus)
        assert not isinstance(layout, Unsupported)
        model, data = _build(spec, layout)
        post = _run("laplace", model, data, draws=4000, seed=1)
        assert not isinstance(post, Unsupported | Unverified)
        post = _with_effects(spec, layout, post, seed=1)
        summarized = _summarize(
            spec,
            corpus,
            layout,
            model,
            post,
            pool_names(spec),
            backend="laplace",
            draws=4000,
            seed=1,
        )
        assert summarized.tau_party is not None
        # Three times the truth, which is what the guard exists to keep out of a result.
        assert summarized.tau_party.mean > 3.0 * TAU_PARTY
