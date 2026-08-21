"""Prior-to-posterior learning: contraction, overlap, shift, and prior-dominated flags.

Ported from the parent's ``diagnostics/learning.py``. For each parameter
the prior is summarized from ``draw_prior`` samples (``axiom.diagnose.sbc``,
so a hierarchical prior is drawn the way the model declares it) and the
posterior from its draws:

* **contraction** ``1 − sd_post² / sd_prior²`` (Schad et al. 2021): one
  when the data fixed the parameter, zero when they left it where the
  prior put it, negative when the posterior is wider than the prior;
* **overlap** — the Bhattacharyya coefficient of the two normal
  approximations, ``sqrt(2 s1 s2 / (s1² + s2²)) · exp(−(m1 − m2)² /
  (4 (s1² + s2²)))``: one for identical distributions, near zero when they
  barely touch;
* **shift** ``(mean_post − mean_prior) / sd_prior`` in prior-sd units.

A parameter with contraction below ``threshold`` is **prior-dominated**;
vector parameters are reported element-wise as ``name[i]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import ModelSpec, Posterior, Spec, Unsupported, free_parameters
from axiom.surface import FitResult

__all__ = ["LearningReport", "ParameterLearning", "bhattacharyya", "learning"]

Array = npt.NDArray[np.float64]


def bhattacharyya(mean_a: float, sd_a: float, mean_b: float, sd_b: float) -> float:
    """Bhattacharyya coefficient of two normals (1 = identical, 0 = disjoint)."""
    if sd_a <= 0 or sd_b <= 0:
        raise ValueError("standard deviations must be positive")
    v = sd_a * sd_a + sd_b * sd_b
    return math.sqrt(2.0 * sd_a * sd_b / v) * math.exp(-((mean_a - mean_b) ** 2) / (4.0 * v))


class ParameterLearning(Spec):
    """Prior and posterior moments of one scalar parameter and what moved."""

    name: str
    prior_mean: float
    prior_sd: float = Field(gt=0)
    posterior_mean: float
    posterior_sd: float = Field(gt=0)
    contraction: float = Field(le=1)
    overlap: float = Field(ge=0, le=1)
    shift: float
    prior_dominated: bool


class LearningReport(Spec):
    """Per-parameter learning plus the prior-dominated set.

    ``n_prior`` prior draws (with ``seed``) and ``n_posterior`` posterior
    draws were used; ``threshold`` is the contraction below which a
    parameter is flagged; ``passed`` means nothing in ``parameters`` is
    prior-dominated.
    """

    parameters: tuple[ParameterLearning, ...] = Field(min_length=1)
    n_prior: int = Field(ge=2)
    n_posterior: int = Field(ge=2)
    seed: int | None
    threshold: float
    prior_dominated: tuple[str, ...]
    passed: bool

    @model_validator(mode="after")
    def _consistent(self) -> LearningReport:
        flagged = tuple(p.name for p in self.parameters if p.prior_dominated)
        if flagged != self.prior_dominated:
            raise ValueError("prior_dominated must list exactly the flagged parameters")
        if self.passed != (not flagged):
            raise ValueError("passed must be the absence of prior-dominated parameters")
        return self

    def get(self, name: str) -> ParameterLearning:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"no parameter {name!r} in the learning report")


def _prior_draws(model: ModelSpec, rng: np.random.Generator, n: int) -> Mapping[str, Array]:
    # Lazy: ``sbc`` is a sibling that also imports ``infer``; importing it here keeps
    # ``learning`` importable on its own and lets tests substitute the sampler.
    from axiom.diagnose.sbc import draw_prior

    return draw_prior(model, rng, n=n)


def learning(
    model: ModelSpec,
    posterior: Posterior | FitResult,
    *,
    parameters: Sequence[str] | None = None,
    n_prior: int = 4000,
    threshold: float = 0.1,
    seed: int | None = 0,
) -> LearningReport | Unsupported:
    """Contraction, overlap and shift of every (or the named) free parameter.

    ``posterior`` may be a ``FitResult`` (its posterior is used; without
    one the result is ``Unsupported``). ``parameters`` defaults to every
    free parameter present in the posterior.
    """
    if isinstance(posterior, FitResult):
        if not isinstance(posterior.posterior, Posterior):
            return Unsupported(
                reason=f"fit has no posterior: {posterior.posterior.reason}",
                missing=("posterior",),
            )
        post = posterior.posterior
    else:
        post = posterior
    if n_prior < 2:
        raise ValueError(f"n_prior must be at least 2, got {n_prior}")
    names = (
        tuple(parameters)
        if parameters is not None
        else tuple(p.name for p in free_parameters(model) if p.name in post.names())
    )
    if not names:
        raise ValueError("no requested parameter is present in the posterior")
    for n in names:
        if n not in post.names():
            raise KeyError(f"parameter {n!r} is not in the posterior")
    prior = _prior_draws(model, np.random.default_rng(seed), n_prior)

    rows: list[ParameterLearning] = []
    for n in names:
        pr = np.asarray(prior[n], dtype=float).reshape(n_prior, -1)
        po = post.flat(n).reshape(post.n_draws(), -1)
        if pr.shape[1] != po.shape[1]:
            raise ValueError(
                f"parameter {n!r}: prior draws have {pr.shape[1]} elements, posterior "
                f"{po.shape[1]}"
            )
        for i in range(po.shape[1]):
            label = n if po.shape[1] == 1 and post.flat(n).ndim == 1 else f"{n}[{i}]"
            m1, s1 = float(pr[:, i].mean()), float(pr[:, i].std(ddof=1))
            m2, s2 = float(po[:, i].mean()), float(po[:, i].std(ddof=1))
            if s1 <= 0 or s2 <= 0:
                return Unsupported(
                    reason=f"parameter {label!r} has zero prior or posterior spread",
                    detail={"prior_sd": repr(s1), "posterior_sd": repr(s2)},
                )
            contraction = 1.0 - (s2 * s2) / (s1 * s1)
            rows.append(
                ParameterLearning(
                    name=label,
                    prior_mean=m1,
                    prior_sd=s1,
                    posterior_mean=m2,
                    posterior_sd=s2,
                    contraction=contraction,
                    overlap=min(1.0, bhattacharyya(m1, s1, m2, s2)),
                    shift=(m2 - m1) / s1,
                    prior_dominated=contraction < threshold,
                )
            )
    flagged = tuple(r.name for r in rows if r.prior_dominated)
    return LearningReport(
        parameters=tuple(rows),
        n_prior=int(n_prior),
        n_posterior=post.n_draws(),
        seed=seed,
        threshold=float(threshold),
        prior_dominated=flagged,
        passed=not flagged,
    )
