"""Weak identification of structural parameters from the posterior's geometry.

Extracted from the parent's ``diagnostics/identification.py`` (porting
ledger: concept and bounds math port; the backend-specific guardrail is
rewritten per backend). Three signals, all computed from draws already in
hand:

* the **posterior correlation matrix** of the structural parameters and the
  pairs above a stated ``|rho|`` — the ``k``–``s``–``beta`` equifinality
  ridge of a saturating kernel shows up as a near-unit correlation;
* the **condition number** of the standardized posterior covariance (the
  correlation matrix): the ratio of its largest to smallest eigenvalue. A
  large value says some direction of parameter space is almost unmoved by
  the data;
* the **prior-to-posterior sd ratio** per parameter: a ratio near one means
  the data did not narrow that parameter at all.

Plus the bounds guardrail: a parameter on a bounded support (``beta``,
``uniform``) whose draws pile up at a bound is a saturated transform — the
parent's ``ZeroDivisionError`` inside a compiled gradient traced to exactly
that — and is reported rather than silently returned.

Hyperparameters that name another parameter are resolved at that
parameter's posterior mean when computing the prior sd; the report says so
in ``resolved_hypers``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import ModelSpec, Posterior, Prior, Spec, Unsupported, free_parameters
from axiom.surface import FitResult

__all__ = ["WeakIdReport", "prior_moments", "weak_identification"]

Array = npt.NDArray[np.float64]


def prior_moments(prior: Prior, resolved: Mapping[str, float] | None = None) -> tuple[float, float]:
    """``(mean, sd)`` of a prior family; a hyperparameter naming a parameter is looked up
    in ``resolved``. ``KeyError`` when it is not there; ``(value, 0)`` for ``fixed``."""
    look = dict(resolved or {})

    def h(k: str) -> float:
        v = prior.hyper[k]
        return float(look[v]) if isinstance(v, str) else float(v)

    match prior.family:
        case "normal":
            return h("mu"), h("sigma")
        case "halfnormal":
            s = h("sigma")
            return s * math.sqrt(2.0 / math.pi), s * math.sqrt(1.0 - 2.0 / math.pi)
        case "lognormal":
            mu, s = h("mu"), h("sigma")
            var = (math.exp(s * s) - 1.0) * math.exp(2.0 * mu + s * s)
            return math.exp(mu + 0.5 * s * s), math.sqrt(var)
        case "gamma":
            a, b = h("alpha"), h("beta")
            return a / b, math.sqrt(a) / b
        case "beta":
            a, b = h("alpha"), h("beta")
            return a / (a + b), math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
        case "uniform":
            lo, hi = h("low"), h("high")
            return 0.5 * (lo + hi), (hi - lo) / math.sqrt(12.0)
        case "fixed":
            return h("value"), 0.0
    raise ValueError(f"unknown prior family {prior.family!r}")  # pragma: no cover


class WeakIdReport(Spec):
    """Posterior geometry of the structural parameters and what it flags.

    ``parameters`` are the scalar names (``name`` or ``name[i]`` for a
    vector); ``correlation`` is row-major over them. ``high_pairs`` lists
    ``(a, b, rho)`` with ``|rho| >= rho_threshold``; ``sd_ratio`` is
    posterior sd over prior sd per *declared* parameter (vector: mean over
    elements), ``unlearned`` those with ratio ``>= ratio_threshold``;
    ``saturated`` names bounded parameters with more than
    ``saturation_share`` of their draws within ``saturation_eps`` (as a
    fraction of the support's width) of a bound. ``passed`` is the
    conjunction: no high pair, condition number below its threshold, no
    unlearned structural parameter, nothing saturated.
    """

    parameters: tuple[str, ...] = Field(min_length=1)
    n_draws: int = Field(ge=2)
    correlation: tuple[tuple[float, ...], ...]
    condition_number: float = Field(ge=1)
    rho_threshold: float = Field(gt=0, le=1)
    condition_threshold: float = Field(gt=1)
    ratio_threshold: float = Field(gt=0)
    high_pairs: tuple[tuple[str, str, float], ...]
    prior_sd: dict[str, float]
    posterior_sd: dict[str, float]
    sd_ratio: dict[str, float]
    unlearned: tuple[str, ...]
    saturated: tuple[str, ...]
    saturation_share: float = Field(gt=0, lt=1)
    saturation_eps: float = Field(gt=0, lt=0.5)
    resolved_hypers: dict[str, float] = {}
    passed: bool

    @model_validator(mode="after")
    def _consistent(self) -> WeakIdReport:
        k = len(self.parameters)
        if len(self.correlation) != k or any(len(r) != k for r in self.correlation):
            raise ValueError("correlation must be square over parameters")
        if set(self.sd_ratio) != set(self.prior_sd) or set(self.sd_ratio) != set(self.posterior_sd):
            raise ValueError("prior_sd, posterior_sd and sd_ratio must share keys")
        return self


def _scalar_columns(post: Posterior, names: Sequence[str]) -> tuple[list[str], Array]:
    labels: list[str] = []
    cols: list[Array] = []
    for n in names:
        a = post.flat(n)
        flat = a.reshape(a.shape[0], -1)
        for i in range(flat.shape[1]):
            labels.append(n if flat.shape[1] == 1 and a.ndim == 1 else f"{n}[{i}]")
            cols.append(flat[:, i])
    return labels, np.column_stack(cols)


def _structural(result: FitResult | Posterior, model: ModelSpec) -> tuple[str, ...]:
    if isinstance(result, FitResult):
        return tuple(result.surface.linear) + tuple(result.surface.nonlinear)
    return tuple(p.name for p in free_parameters(model))


def weak_identification(
    result: FitResult | Posterior,
    model: ModelSpec | None = None,
    *,
    parameters: Sequence[str] | None = None,
    rho_threshold: float = 0.9,
    condition_threshold: float = 1e3,
    ratio_threshold: float = 0.9,
    saturation_share: float = 0.1,
    saturation_eps: float = 1e-3,
) -> WeakIdReport | Unsupported:
    """Correlation ridge, condition number, prior-vs-posterior sd ratio, bound saturation.

    ``parameters`` defaults to the surface's linear and nonlinear
    parameters for a ``FitResult`` (hyperparameters and the likelihood
    scale are left out) and to every free parameter for a bare
    ``Posterior``, for which ``model`` is required. A fit without a
    posterior comes back as ``Unsupported``.
    """
    if isinstance(result, FitResult):
        if not isinstance(result.posterior, Posterior):
            return Unsupported(
                reason=f"fit has no posterior: {result.posterior.reason}",
                missing=("posterior",),
            )
        post = result.posterior
        model = result.surface.model if model is None else model
    else:
        post = result
        if model is None:
            raise ValueError("a bare Posterior needs the ModelSpec it was drawn under")
    names = tuple(parameters) if parameters is not None else _structural(result, model)
    names = tuple(n for n in names if n in post.names())
    if not names:
        raise ValueError("no requested parameter is present in the posterior")
    if post.n_draws() < 2:
        raise ValueError("need at least two draws")
    if not 0 < rho_threshold <= 1:
        raise ValueError(f"rho_threshold must lie in (0, 1], got {rho_threshold}")

    labels, mat = _scalar_columns(post, names)
    sds = mat.std(axis=0, ddof=1)
    if np.any(sds == 0.0):
        degenerate = [labels[i] for i in np.flatnonzero(sds == 0.0)]
        return Unsupported(
            reason=f"posterior draws are constant for {degenerate}",
            detail={"constant": ", ".join(degenerate)},
        )
    corr = np.corrcoef(mat, rowvar=False) if mat.shape[1] > 1 else np.ones((1, 1))
    corr = np.atleast_2d(corr)
    eig = np.linalg.eigvalsh(corr)
    cond = float(eig.max() / max(eig.min(), np.finfo(float).tiny))
    high: list[tuple[str, str, float]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rho = float(corr[i, j])
            if abs(rho) >= rho_threshold:
                high.append((labels[i], labels[j], rho))

    # prior sd with parent hyperparameters resolved at their posterior mean (documented)
    resolved: dict[str, float] = {}
    for n in post.names():
        resolved[n] = float(np.mean(post.flat(n)))
    used: dict[str, float] = {}
    prior_sd: dict[str, float] = {}
    post_sd: dict[str, float] = {}
    ratio: dict[str, float] = {}
    saturated: list[str] = []
    for n in names:
        p = model.parameter(n)
        assert p.prior is not None
        for parent in p.prior.parents:
            if parent not in resolved:
                return Unsupported(
                    reason=f"prior of {n!r} depends on {parent!r}, absent from the posterior",
                    missing=(parent,),
                )
            used[parent] = resolved[parent]
        _, sd_prior = prior_moments(p.prior, resolved)
        a = post.flat(n).reshape(post.n_draws(), -1)
        sd_post = float(np.mean(a.std(axis=0, ddof=1)))
        prior_sd[n] = sd_prior
        post_sd[n] = sd_post
        ratio[n] = sd_post / sd_prior if sd_prior > 0 else float("inf")
        if p.prior.family in ("beta", "uniform"):
            if p.prior.family == "beta":
                lo, hi = 0.0, 1.0
            else:
                lo_v, hi_v = p.prior.hyper["low"], p.prior.hyper["high"]
                lo = resolved[lo_v] if isinstance(lo_v, str) else float(lo_v)
                hi = resolved[hi_v] if isinstance(hi_v, str) else float(hi_v)
            width = hi - lo
            near = (a - lo <= saturation_eps * width) | (hi - a <= saturation_eps * width)
            if float(np.mean(near)) > saturation_share:
                saturated.append(n)
    unlearned = tuple(n for n in names if ratio[n] >= ratio_threshold)
    return WeakIdReport(
        parameters=tuple(labels),
        n_draws=post.n_draws(),
        correlation=tuple(tuple(float(v) for v in row) for row in corr),
        condition_number=max(cond, 1.0),
        rho_threshold=float(rho_threshold),
        condition_threshold=float(condition_threshold),
        ratio_threshold=float(ratio_threshold),
        high_pairs=tuple(high),
        prior_sd=prior_sd,
        posterior_sd=post_sd,
        sd_ratio=ratio,
        unlearned=unlearned,
        saturated=tuple(saturated),
        saturation_share=float(saturation_share),
        saturation_eps=float(saturation_eps),
        resolved_hypers=used,
        passed=not high and cond < condition_threshold and not unlearned and not saturated,
    )
