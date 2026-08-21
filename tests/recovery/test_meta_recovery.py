"""Phase 7 recovery gate for ``meta``.

Simulated hierarchies with known ``tau`` and known study means; the classical
estimators recover ``tau²`` within their published bias, the Bayesian pool's
shrinkage matches the analytic normal–normal factor, the provenance offset
``delta`` is recovered when and only when a dual-read contributor exists,
and a corpus mixing scales is refused by ``normalize`` until licensed.

The setting (items 1–2)
-----------------------
``k = 30`` studies, ``theta_i ~ N(mu = 0.5, tau = 0.3)``, ``se_i ~ U(0.1, 0.4)``,
``y_i = theta_i + se_i · eps_i``. ``tau² = 0.09`` against ``se² ∈ [0.01, 0.16]``
is moderate heterogeneity (``I² ≈ 60 %``), the regime the published bias
numbers below are quoted for.

Bounds asserted and their references
------------------------------------
* **DerSimonian–Laird** ``tau²`` is negatively biased when heterogeneity is
  not small and its bias shrinks with ``k``; at ``k = 30`` and ``I² ≈ 0.6``
  the mean relative bias is a few percent (Viechtbauer 2005, *J. Educ.
  Behav. Stat.* 30(3), Table 1; Veroniki et al. 2016, *Res. Synth. Methods*
  7(1), §3.1). Asserted: ``|mean relative bias| ≤ 15 %`` over 200
  replications.
* **Paule–Mandel** and **REML** are approximately unbiased in this regime
  (Viechtbauer 2005, Tables 1–2; Veroniki et al. 2016, §3.3, §3.6).
  Asserted: ``|mean relative bias| ≤ 10 %`` over 200 replications.
  The Monte-Carlo sd of a mean relative bias over 200 replications is
  ``≈ 3 %`` here, so these bounds are one-sided tests of the published
  magnitudes with a ``≥ 2 sd`` margin, not noise.
* **Knapp–Hartung** 95 % intervals for ``mu`` hold their nominal level
  (Knapp & Hartung 2003, *Stat. Med.* 22(17); IntHout et al. 2014, *BMC Med.
  Res. Methodol.* 14:25). Asserted: the coverage count over 200
  replications lies in ``clopper_pearson(200, 0.95, 1e-3)``.

Fast variants run 20 replications with wider, stated regions (bias bounds
``25 % / 20 %``, MC sd ``≈ 9 %``; coverage region
``clopper_pearson(20, 0.95, 1e-3)``) so the gate is exercised on every run;
the 200-replication versions are ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import Unsupported, clopper_pearson
from axiom.estimands import TransferPlan
from axiom.meta.classical import random_effects
from axiom.meta.ingest import normalize
from axiom.meta.pool import Pooled, PoolPriors, PoolSpec, pool
from axiom.meta.schema import Corpus, StudyRecord

MU, TAU, K = 0.5, 0.3, 30


def _hierarchy(rng: np.random.Generator, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    se = rng.uniform(0.1, 0.4, size=k)
    theta = rng.normal(MU, TAU, size=k)
    y = theta + se * rng.normal(size=k)
    return y, se


def _records(
    y: np.ndarray, se: np.ndarray, *, prefix: str = "s", read: str = "experiment"
) -> tuple[StudyRecord, ...]:
    return tuple(
        StudyRecord(
            study=f"{prefix}{i}",
            contributor=f"{prefix}{i}",
            quantity="elasticity",
            estimate=float(y[i]),
            se=float(se[i]),
            read=read,  # type: ignore[arg-type]
            family="f",
        )
        for i in range(y.size)
    )


# -- (1) classical tau² recovery and Knapp–Hartung coverage ----------------------------------


def _classical_replications(
    n_rep: int, seed: int
) -> tuple[dict[str, float], int, dict[str, float]]:
    rng = np.random.default_rng(seed)
    tau2 = {"dl": [], "pm": [], "reml": []}  # type: dict[str, list[float]]
    covered = 0
    for _ in range(n_rep):
        y, se = _hierarchy(rng)
        for m in tau2:
            tau2[m].append(random_effects(y, se, tau_method=m).tau2)  # type: ignore[arg-type]
        kh = random_effects(y, se, tau_method="reml", knapp_hartung=True, mass=0.95)
        assert kh.interval.definition == "wald" and kh.interval.mass == 0.95
        covered += int(kh.interval.contains(MU))
    bias = {m: float(np.mean(v) / TAU**2 - 1.0) for m, v in tau2.items()}
    mc_sd = {m: float(np.std(v, ddof=1) / TAU**2 / np.sqrt(n_rep)) for m, v in tau2.items()}
    return bias, covered, mc_sd


def _assert_classical(n_rep: int, seed: int, *, dl_bound: float, pm_reml_bound: float) -> None:
    bias, covered, mc_sd = _classical_replications(n_rep, seed)
    assert abs(bias["dl"]) <= dl_bound, (bias, mc_sd)
    assert abs(bias["pm"]) <= pm_reml_bound, (bias, mc_sd)
    assert abs(bias["reml"]) <= pm_reml_bound, (bias, mc_sd)
    region = clopper_pearson(n_rep, 0.95, 1e-3)
    assert region.accepts(covered), (covered, region)


def test_classical_tau2_bias_and_kh_coverage_fast() -> None:
    """20 replications: DL ≤ 25 %, PM/REML ≤ 20 %, coverage in clopper_pearson(20, .95, 1e-3)."""
    _assert_classical(20, seed=2024, dl_bound=0.25, pm_reml_bound=0.20)


@pytest.mark.slow
def test_classical_tau2_bias_and_kh_coverage() -> None:
    """200 replications: DL ≤ 15 %, PM/REML ≤ 10 %, coverage in clopper_pearson(200, .95, 1e-3)."""
    _assert_classical(200, seed=7, dl_bound=0.15, pm_reml_bound=0.10)


# -- (2) Bayesian pool: shrinkage against the analytic normal–normal factor -------------------


@pytest.mark.parametrize("parametrization", ["marginal", "centered"])
def test_pool_shrinkage_matches_analytic_with_tau_fixed(parametrization: str) -> None:
    rng = np.random.default_rng(11)
    y, se = _hierarchy(rng)
    corpus = Corpus(records=_records(y, se), name="shrinkage")
    spec = PoolSpec(
        family="f",
        priors=PoolPriors(tau_fixed=TAU, mu_scale=10.0),
        parametrization=parametrization,  # type: ignore[arg-type]
    )
    n_draws = 400_000  # Laplace draws are Gaussian samples; MC error on a factor ≲ 0.3 %
    out = pool(spec, corpus, draws=n_draws, seed=3)
    assert isinstance(out, Pooled)
    r = out.result
    assert r.tau.mean == TAU
    B = se**2 / (se**2 + TAU**2)
    # posterior mean of every effect is (1 − B) y + B E[mu], within 4 MC sd
    theta_mean = np.array([t.mean for t in r.thetas])
    predicted = (1 - B) * y + B * r.mu.mean
    mc_sd = np.array([t.sd for t in r.thetas]) / np.sqrt(n_draws)
    assert np.all(np.abs(theta_mean - predicted) < 4 * mc_sd + 1e-9)
    # the shrinkage factors themselves, where the ratio is well conditioned (|y − mu| > se)
    checked = 0
    worst = 0.0
    for s, b in zip(r.shrinkage, B, strict=True):
        assert s.analytic == pytest.approx(b)
        if s.empirical is None or abs(s.estimate - r.mu.mean) <= s.se:
            continue
        checked += 1
        worst = max(worst, abs(s.empirical - b) / b)
    assert checked >= 10
    assert worst <= 0.02, worst


def test_pool_free_tau_recovers_within_posterior_uncertainty() -> None:
    zs = []
    for seed in range(6):
        y, se = _hierarchy(np.random.default_rng(100 + seed))
        out = pool(PoolSpec(family="f"), Corpus(records=_records(y, se)), draws=4000, seed=seed)
        assert isinstance(out, Pooled)
        r = out.result
        assert r.tau.sd > 0 and r.tau.interval.lower > 0
        zs.append((r.tau.mean - TAU) / r.tau.sd)
        assert abs(r.mu.mean - MU) < 3 * r.mu.sd
    # every replication within 2.5 posterior sd, and no systematic drift across them
    assert np.max(np.abs(zs)) < 2.5, zs
    assert abs(np.mean(zs)) < 1.0, zs


# -- (3) the provenance offset delta -------------------------------------------------------------


def _dual_read_corpus(
    rng: np.random.Generator, *, n_contributors: int, dual_fraction: float, delta: float
) -> Corpus:
    n_dual = round(dual_fraction * n_contributors)
    recs: list[StudyRecord] = []
    for c in range(n_contributors):
        theta = float(rng.normal(MU, TAU))
        dual = c < n_dual
        model_only = (not dual) and c % 2 == 0
        reads = ("model", "experiment") if dual else (("model",) if model_only else ("experiment",))
        for read in reads:
            se = float(rng.uniform(0.1, 0.3))
            shift = delta if read == "model" else 0.0
            recs.append(
                StudyRecord(
                    study=f"c{c}:{read}",
                    contributor=f"c{c}",
                    quantity="elasticity",
                    estimate=theta + shift + se * float(rng.normal()),
                    se=se,
                    read=read,  # type: ignore[arg-type]
                    family="f",
                )
            )
    return Corpus(records=tuple(recs), name="dual")


def test_delta_recovered_with_dual_read_contributors() -> None:
    delta_true = 0.4
    corpus = _dual_read_corpus(
        np.random.default_rng(21), n_contributors=30, dual_fraction=0.3, delta=delta_true
    )
    assert len(corpus.dual_read_contributors("f")) == 9
    out = pool(PoolSpec(family="f", bias_term=True), corpus, draws=4000, seed=0)
    assert isinstance(out, Pooled)
    r = out.result
    assert r.delta is not None and r.delta.identified
    assert r.delta_verdict is not None and r.delta_verdict.status == "identified"
    assert abs(r.delta.mean - delta_true) < 2 * r.delta.sd, (r.delta.mean, r.delta.sd)
    assert r.delta.sd < 0.5 * PoolPriors().delta_scale  # the data, not the prior, speaks
    assert abs(r.mu.mean - MU) < 3 * r.mu.sd
    assert abs(r.tau.mean - TAU) < 2.5 * r.tau.sd


def test_delta_not_identified_without_dual_reads() -> None:
    corpus = _dual_read_corpus(
        np.random.default_rng(22), n_contributors=30, dual_fraction=0.0, delta=0.4
    )
    assert corpus.dual_read_contributors("f") == ()
    prior_sd = 0.7
    spec = PoolSpec(family="f", bias_term=True, priors=PoolPriors(delta_scale=prior_sd))
    out = pool(spec, corpus, draws=8000, seed=0)
    assert isinstance(out, Pooled)
    r = out.result
    assert r.delta is not None and not r.delta.identified
    assert r.delta_verdict is not None and r.delta_verdict.status == "blocked"
    assert "no contributor reports both reads" in r.delta_verdict.reason
    assert abs(r.delta.sd / prior_sd - 1.0) <= 0.10, (r.delta.sd, prior_sd)
    assert abs(r.delta.mean) < 3 * prior_sd / np.sqrt(8000)
    assert "NOT identified" in r.detail["delta"]
    # and mu is still estimated — from the experimental reads' scale plus the model reads
    # without any offset, which is what "not identified" honestly means here
    assert r.mu.sd > 0


# -- (4) scales: refused until licensed ----------------------------------------------------------


def test_mixed_scale_corpus_refused_then_licensed() -> None:
    rng = np.random.default_rng(5)
    y, se = _hierarchy(rng, k=2)
    clean = _records(y, se)
    kg = clean[0].model_copy(update={"unit_scale": "per_kg"})
    lb = clean[1].model_copy(update={"unit_scale": "per_lb"})
    refused = normalize([kg, lb])
    assert isinstance(refused, Unsupported)
    assert kg.study in refused.reason and "TransferPlan" in refused.reason
    plan = TransferPlan(
        status="identified",
        source="per_kg",
        target="dimensionless",
        differing=(),
        entries=(),
        assumptions=(),
        ledger_lines=(),
    )
    still_refused = normalize([kg, lb], plans={kg.study: plan})
    assert isinstance(still_refused, Unsupported) and lb.study in still_refused.reason
    admitted = normalize([kg, lb], plans={kg.study: plan, lb.study: plan})
    assert isinstance(admitted, Corpus) and len(admitted) == 2
    assert all("transfer" in r.detail for r in admitted.records)
    assert isinstance(normalize(clean), Corpus)
