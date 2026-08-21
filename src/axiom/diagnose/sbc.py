"""Simulation-based calibration (Talts, Betancourt, Simpson, Vehtari & Gelman 2018).

For a model ``p(theta) p(y | theta)`` and any posterior approximation
``q(theta | y)``, the procedure is: draw ``theta* ~ p(theta)``, draw ``y ~
p(y | theta*)``, fit ``q(theta | y)``, and record the rank of ``theta*``
among ``L`` posterior draws. If ``q`` is the exact posterior, the rank is
uniform on ``{0, ..., L}`` for every parameter, whatever the data look
like — so a non-uniform rank histogram is evidence that the fitting
machinery (the backend, the approximation, the model's implementation) is
wrong, and nothing else. This is the calibration check the roadmap's
Phase 8 exit criterion 1 asks for.

Ported by specification from the parent's ``diagnostics/sbc.py`` (ledger
row ``diagnose/sbc.py``, PORT): the refit loop is rewired to
``axiom.infer.Backend``, the prior draw goes through ``axiom.core``'s prior
families, and the outcome is simulated from the model's own likelihood
family at ``mean(theta*)`` — the surface variant through
``Surface.forward`` (rule 3: one ``forward``).

Uniformity is assessed two ways, both reported:

* a chi-square goodness-of-fit test on the rank histogram. The ``L + 1``
  rank values are grouped into ``bins`` equiprobable bins (``bins − 1``
  degrees of freedom; Talts et al. §5 recommend a histogram coarse enough
  that every bin expects a handful of ranks — the chi-square approximation
  is unreliable below about ten per bin). ``bins`` must divide ``L + 1`` so
  the bins really are equiprobable; by default it is the largest divisor
  with at least ten expected ranks per bin (two when ``n`` is too small for
  even that);
* the ECDF-difference statistic ``sup_z |F_n(z) − F(z)|`` between the
  empirical CDF of the normalized ranks ``r / L`` and the CDF of the
  discrete uniform on ``{0, ..., L} / L``, with a **Dvoretzky–Kiefer–
  Wolfowitz simultaneous band** ``sqrt(ln(2 / alpha) / (2 n))`` (Massart's
  tight constant). This is the simpler Kolmogorov band the integrator
  accepted in place of Säilynoja, Vehtari & Gabry's (2022) simulation-
  calibrated band; it is conservative — exact for every ``n``, never
  anti-conservative — so it trades some power for a guarantee.

A parameter ``passed`` when the chi-square p-value is at least its
``alpha`` **and** the ECDF statistic lies inside the band. The spec's
``alpha`` is the *family-wise* level of the run: with ``k`` ranked scalar
parameters each is tested at ``alpha / k`` (Bonferroni), so a perfectly
calibrated fit fails the whole run with probability at most about
``2 · alpha`` (two checks per parameter, the DKW one conservative). A
fit the backend declines (``Unverified``) is counted in ``n_failed_fits``
with its reason — never dropped silently — and the ranks are computed over
the fits that succeeded.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import stats as _st

from axiom.core import (
    ModelSpec,
    Param,
    Posterior,
    Prior,
    Spec,
    Unsupported,
    Unverified,
    likelihood_scale,
    value,
)
from axiom.data import Panel
from axiom.infer import get_backend
from axiom.meta import Corpus, PoolSpec, pool_model
from axiom.surface import Surface, SurfaceSpec, prepare

__all__ = [
    "ParameterRanks",
    "SBCResult",
    "SBCSpec",
    "Simulator",
    "default_bins",
    "draw_prior",
    "rank_uniformity",
    "sbc",
    "sbc_pool",
    "sbc_surface",
    "simulate_outcome",
]

Array = npt.NDArray[np.float64]
DataDict = Mapping[str, npt.ArrayLike]
Simulator = Callable[[Mapping[str, Array], np.random.Generator], Array]
"""``(theta, rng) -> outcome array``: the data-generating step of one SBC replication."""


# -- prior draws ----------------------------------------------------------------------------


def _resolution_order(model: ModelSpec) -> tuple[Param, ...]:
    """Parameters with every hyper-reference resolved before them (``ModelSpec`` forbids
    cycles); the same order ``axiom.sim.surface_world`` resolves a truth in."""
    pending = list(model.parameters)
    done: set[str] = set()
    ordered: list[Param] = []
    while pending:
        ready = [p for p in pending if p.prior is None or all(q in done for q in p.prior.parents)]
        if not ready:  # pragma: no cover - ModelSpec validation rejects cycles
            raise ValueError("prior hierarchy could not be ordered")
        for p in ready:
            ordered.append(p)
            done.add(p.name)
            pending.remove(p)
    return tuple(ordered)


def _hyper(prior: Prior, key: str, resolved: Mapping[str, Array]) -> Array:
    v = prior.hyper[key]
    return np.asarray(resolved[v] if isinstance(v, str) else v, dtype=np.float64)


def _draw_one(
    prior: Prior, resolved: Mapping[str, Array], shape: tuple[int, ...], rng: np.random.Generator
) -> Array:
    """One constrained draw from ``prior`` with its hyperparameters taken from ``resolved``."""

    def h(k: str) -> Array:
        return _hyper(prior, k, resolved)

    match prior.family:
        case "normal":
            out = rng.normal(h("mu"), h("sigma"), size=shape)
        case "halfnormal":
            out = np.abs(rng.normal(0.0, h("sigma"), size=shape))
        case "lognormal":
            out = rng.lognormal(h("mu"), h("sigma"), size=shape)
        case "gamma":
            out = rng.gamma(h("alpha"), 1.0 / h("beta"), size=shape)
        case "beta":
            out = rng.beta(h("alpha"), h("beta"), size=shape)
        case "uniform":
            out = rng.uniform(h("low"), h("high"), size=shape)
        case "fixed":
            out = np.broadcast_to(h("value"), shape)
    return np.asarray(out, dtype=np.float64)


def draw_prior(model: ModelSpec, rng: np.random.Generator, *, n: int = 1) -> dict[str, Array]:
    """``n`` joint draws from the model's prior, constrained, shaped ``(n, *param_shape)``.

    Hierarchical hyperparameters are resolved in dependency order (a
    parent is drawn before the child whose prior names it), so every draw
    is a coherent point of the joint prior. ``fixed`` parameters are
    included at their value. A scalar parameter comes back as ``(n,)``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    order = _resolution_order(model)
    out: dict[str, list[Array]] = {p.name: [] for p in order}
    for _ in range(n):
        resolved: dict[str, Array] = {}
        for p in order:
            assert p.prior is not None  # ModelSpec guarantees a prior on every parameter
            resolved[p.name] = _draw_one(p.prior, resolved, p.shape, rng)
            out[p.name].append(resolved[p.name])
    return {name: np.stack(draws, axis=0) for name, draws in out.items()}


# -- simulating the outcome -------------------------------------------------------------------


def _draw_noise(
    model: ModelSpec, data: DataDict, mean: Array, theta: Mapping[str, Array], rng: Any
) -> Array:
    lik = model.likelihood
    match lik.family:
        case "normal":
            scale = likelihood_scale(model, data, theta)
            return np.asarray(mean + rng.normal(0.0, 1.0, size=mean.shape) * scale)
        case "student_t":
            scale = likelihood_scale(model, data, theta)
            df = float(lik.df or 0.0)
            return np.asarray(mean + rng.standard_t(df, size=mean.shape) * scale)
        case "lognormal":
            scale = likelihood_scale(model, data, theta)
            if np.any(mean <= 0.0):
                raise ValueError("a lognormal likelihood needs a positive mean in every cell")
            return np.asarray(mean * np.exp(rng.normal(0.0, 1.0, size=mean.shape) * scale))
        case "poisson":
            if np.any(mean <= 0.0):
                raise ValueError("a poisson likelihood needs a positive mean in every cell")
            return np.asarray(rng.poisson(mean), dtype=np.float64)
    raise ValueError(f"unknown likelihood family {lik.family!r}")  # pragma: no cover


def simulate_outcome(
    model: ModelSpec, data: DataDict, theta: Mapping[str, Array], rng: np.random.Generator
) -> Array:
    """One outcome array from ``model``'s likelihood at ``mean(theta)`` on ``data``.

    The mean is ``value(model.mean, data, theta)``; the noise follows the
    likelihood family with ``likelihood_scale`` (a named scale parameter or
    the ``scale_expr``). A lognormal or poisson likelihood needs a positive
    mean everywhere (``ValueError`` otherwise, never a ``nan``).
    """
    mean = np.asarray(value(model.mean, data=data, params=theta), dtype=np.float64)
    return _draw_noise(model, data, mean, theta, rng)


# -- specs and results ------------------------------------------------------------------------


class SBCSpec(Spec):
    """How many replications, how big each fit, and how ranks are resolved.

    ``n_simulations`` prior draws are each simulated, refitted with
    ``draws`` posterior draws from ``backend``, and ranked against
    ``rank_draws`` (``L``) of those draws, thinned at equal spacing so the
    ranks live on ``{0, ..., L}``; ``bins`` groups them for the chi-square
    histogram (a divisor of ``L + 1``; ``None`` picks the largest divisor
    expecting at least ten ranks per bin). ``parameters`` restricts the
    ranking to the named parameters (every free parameter when empty).
    ``alpha`` is the family-wise level: each parameter is checked at
    ``alpha / k`` over the ``k`` scalar parameters ranked.
    """

    n_simulations: int = Field(ge=2)
    draws: int = Field(default=200, ge=2)
    rank_draws: int = Field(default=19, ge=1)
    bins: int | None = Field(default=None, ge=2)
    seed: int = 0
    backend: str = "laplace"
    parameters: tuple[str, ...] = ()
    alpha: float = Field(default=0.05, gt=0, lt=1)

    @model_validator(mode="after")
    def _consistent(self) -> SBCSpec:
        if self.rank_draws > self.draws:
            raise ValueError(
                f"rank_draws={self.rank_draws} exceeds draws={self.draws}; ranks need at most "
                "as many thinned draws as the fit produces"
            )
        if self.bins is not None and (self.rank_draws + 1) % self.bins != 0:
            raise ValueError(
                f"bins={self.bins} must divide rank_draws + 1 = {self.rank_draws + 1} so the "
                "histogram bins are equiprobable"
            )
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError(f"parameter names must be distinct: {list(self.parameters)}")
        if not self.backend.strip():
            raise ValueError("backend must be a non-empty name")
        return self


class ParameterRanks(Spec):
    """The rank histogram of one (scalar slice of a) parameter and its two uniformity checks.

    ``ranks`` has one entry per successful fit, on ``{0, ..., n_ranks}``;
    ``histogram`` counts them in ``bins`` equiprobable bins (``bins``
    divides ``n_ranks + 1``). ``chi2_p_value`` is the chi-square
    goodness-of-fit p-value on the histogram (``bins − 1`` degrees of
    freedom); ``ecdf_statistic`` is
    ``sup |F_n − F|`` against the discrete uniform and ``ecdf_band`` the
    DKW simultaneous band at ``alpha``. ``passed`` is both checks at once.
    """

    name: str
    n: int = Field(ge=1)
    n_ranks: int = Field(ge=1)
    bins: int = Field(ge=2)
    ranks: tuple[int, ...]
    histogram: tuple[int, ...]
    chi2_statistic: float
    chi2_p_value: float
    ecdf_statistic: float
    ecdf_band: float
    alpha: float = Field(gt=0, lt=1)
    passed: bool

    @model_validator(mode="after")
    def _consistent(self) -> ParameterRanks:
        if len(self.ranks) != self.n:
            raise ValueError(f"{self.name}: {len(self.ranks)} ranks for n={self.n}")
        if (self.n_ranks + 1) % self.bins != 0:
            raise ValueError(f"{self.name}: bins must divide n_ranks + 1")
        if len(self.histogram) != self.bins or sum(self.histogram) != self.n:
            raise ValueError(f"{self.name}: histogram must have `bins` entries summing to n")
        if any(r < 0 or r > self.n_ranks for r in self.ranks):
            raise ValueError(f"{self.name}: ranks must lie on [0, n_ranks]")
        return self


class SBCResult(Spec):
    """Every parameter's rank check plus the bookkeeping of the run.

    ``n_fitted + n_failed_fits == n_simulations``; ``failure_reasons`` keeps
    one line per declined fit. ``alpha_per_parameter`` is the Bonferroni
    share of the spec's family-wise ``alpha`` each parameter was read at.
    ``passed`` is true when every ranked parameter passed;
    ``failed_parameters`` names the ones that did not.
    """

    spec: SBCSpec
    model_hash: str
    refit_model_hash: str
    alpha_per_parameter: float = Field(gt=0, lt=1)
    n_simulations: int = Field(ge=0)
    n_fitted: int = Field(ge=0)
    n_failed_fits: int = Field(ge=0)
    failure_reasons: tuple[str, ...] = ()
    parameters: tuple[ParameterRanks, ...]
    failed_parameters: tuple[str, ...]
    passed: bool

    @model_validator(mode="after")
    def _consistent(self) -> SBCResult:
        if self.n_fitted + self.n_failed_fits != self.n_simulations:
            raise ValueError("n_fitted + n_failed_fits must equal n_simulations")
        if len(self.failure_reasons) != self.n_failed_fits:
            raise ValueError("one failure reason per failed fit")
        expected = tuple(p.name for p in self.parameters if not p.passed)
        if self.failed_parameters != expected:
            raise ValueError("failed_parameters must list exactly the parameters that failed")
        if self.passed != (not expected and self.n_fitted > 0):
            raise ValueError("passed must mean every parameter passed on at least one fit")
        return self


# -- rank statistics --------------------------------------------------------------------------


def default_bins(n: int, n_ranks: int, *, min_expected: int = 10) -> int:
    """The largest divisor of ``n_ranks + 1`` that expects at least ``min_expected`` ranks per
    bin from ``n`` ranks; ``2`` when none does (a two-bin histogram is always defined)."""
    values = n_ranks + 1
    ok = [d for d in range(2, values + 1) if values % d == 0 and n // d >= min_expected]
    return max(ok) if ok else 2


def rank_uniformity(
    name: str, ranks: npt.ArrayLike, *, n_ranks: int, alpha: float, bins: int | None = None
) -> ParameterRanks:
    """Chi-square and ECDF-difference checks of ``ranks`` on ``{0, ..., n_ranks}`` (module
    docstring). ``ValueError`` for an empty rank vector, a rank off the support, or ``bins``
    that do not divide ``n_ranks + 1``."""
    r = np.asarray(ranks, dtype=np.int64).reshape(-1)
    if r.size == 0:
        raise ValueError(f"{name}: no ranks to test")
    if n_ranks < 1:
        raise ValueError(f"n_ranks must be at least 1, got {n_ranks}")
    if np.any(r < 0) or np.any(r > n_ranks):
        raise ValueError(f"{name}: ranks must lie on [0, {n_ranks}]")
    n = int(r.size)
    values = n_ranks + 1
    b = default_bins(n, n_ranks) if bins is None else bins
    if b < 2 or values % b != 0:
        raise ValueError(f"bins={b} must be at least 2 and divide n_ranks + 1 = {values}")
    hist = np.bincount(r * b // values, minlength=b)
    chi2 = float(np.sum((hist - n / b) ** 2 / (n / b)))
    p = float(_st.chi2.sf(chi2, df=b - 1))
    # ECDF of the normalized ranks against the discrete uniform's CDF, evaluated at every
    # support point: sup over the support equals sup over the line for step functions.
    support = np.arange(values)
    f_n = np.searchsorted(np.sort(r), support, side="right") / n
    f = (support + 1) / values
    d = float(np.max(np.abs(f_n - f)))
    band = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    return ParameterRanks(
        name=name,
        n=n,
        n_ranks=n_ranks,
        bins=b,
        ranks=tuple(int(x) for x in r),
        histogram=tuple(int(x) for x in hist),
        chi2_statistic=chi2,
        chi2_p_value=p,
        ecdf_statistic=d,
        ecdf_band=band,
        alpha=alpha,
        passed=bool(p >= alpha and d <= band),
    )


# -- the loop ---------------------------------------------------------------------------------


def _run_backend(
    backend_name: str, model: ModelSpec, data: DataDict, *, draws: int, seed: int
) -> Posterior | Unverified | Unsupported:
    b = get_backend(backend_name)
    if isinstance(b, Unsupported):
        return b
    if backend_name.strip().lower() == "laplace":
        return b.laplace(model, data, draws=draws, seed=seed)
    return b.sample(model, data, draws=draws, tune=draws, chains=1, seed=seed)


def _rank_names(model: ModelSpec, parameters: tuple[str, ...]) -> tuple[Param, ...]:
    free = {p.name: p for p in model.free}
    if not parameters:
        return tuple(free.values())
    unknown = [n for n in parameters if n not in free]
    if unknown:
        raise ValueError(
            f"parameters {unknown} are not free parameters of {model.name!r}; "
            f"free are {sorted(free)}"
        )
    return tuple(free[n] for n in parameters)


def _slices(p: Param) -> tuple[str, ...]:
    """Labels of a parameter's scalar slices in C order: ``name`` for a scalar, ``name[i,j]``
    per element of a vector or matrix."""
    if not p.shape:
        return (p.name,)
    return tuple(f"{p.name}[{','.join(str(i) for i in idx)}]" for idx in np.ndindex(*p.shape))


def _thin(flat: Array, n_ranks: int) -> Array:
    """``n_ranks`` draws at equal spacing through the chain (the first is always kept)."""
    idx = np.linspace(0, flat.shape[0] - 1, n_ranks).round().astype(int)
    return np.asarray(flat[idx], dtype=np.float64)


def sbc(
    model: ModelSpec,
    data_template: DataDict,
    *,
    spec: SBCSpec,
    simulate: Simulator | None = None,
    refit: ModelSpec | None = None,
) -> SBCResult | Unsupported:
    """Run SBC on ``model`` over ``data_template`` (the covariate columns; the outcome column
    is overwritten every replication).

    ``simulate`` replaces the default likelihood simulator (``simulate_outcome``)
    — it receives the prior draw and a generator and returns the outcome
    array. ``refit`` is the model the backend fits when it differs from the
    generating model — the negative control: ranking a correct prior draw
    against the posterior of a wrong model must *fail* uniformity. It must
    declare the same free parameters. ``Unsupported`` when the backend is
    not installed.
    """
    fit_model = model if refit is None else refit
    if {p.name for p in fit_model.free} != {p.name for p in model.free}:
        raise ValueError("refit must declare the same free parameters as the generating model")
    params = _rank_names(fit_model, spec.parameters)
    backend = get_backend(spec.backend)
    if isinstance(backend, Unsupported):
        return backend
    rng = np.random.default_rng(spec.seed)
    ranks: dict[str, list[int]] = {name: [] for p in params for name in _slices(p)}
    reasons: list[str] = []
    for i in range(spec.n_simulations):
        child = rng.spawn(1)[0]
        theta = {k: v[0] for k, v in draw_prior(model, child).items()}
        y = (
            simulate_outcome(model, data_template, theta, child)
            if simulate is None
            else np.asarray(simulate(theta, child), dtype=np.float64)
        )
        data = {**dict(data_template), model.outcome.name: y}
        fit_seed = int(child.integers(0, 2**31 - 1))
        post = _run_backend(spec.backend, fit_model, data, draws=spec.draws, seed=fit_seed)
        if isinstance(post, Unsupported):
            return post
        if isinstance(post, Unverified):
            reasons.append(f"simulation {i}: {post.reason}")
            continue
        for p in params:
            thinned = _thin(np.asarray(post.flat(p.name), dtype=np.float64), spec.rank_draws)
            columns = thinned.reshape(thinned.shape[0], -1)
            truths = np.asarray(theta[p.name], dtype=np.float64).reshape(-1)
            for j, name in enumerate(_slices(p)):
                ranks[name].append(int(np.sum(columns[:, j] < truths[j])))
    n_fitted = spec.n_simulations - len(reasons)
    ranked = {name: r for name, r in ranks.items() if r}
    per_parameter = spec.alpha / max(len(ranked), 1)
    checks = tuple(
        rank_uniformity(name, r, n_ranks=spec.rank_draws, alpha=per_parameter, bins=spec.bins)
        for name, r in ranked.items()
    )
    failed = tuple(c.name for c in checks if not c.passed)
    return SBCResult(
        spec=spec,
        model_hash=model.content_hash(),
        refit_model_hash=fit_model.content_hash(),
        alpha_per_parameter=per_parameter,
        n_simulations=spec.n_simulations,
        n_fitted=n_fitted,
        n_failed_fits=len(reasons),
        failure_reasons=tuple(reasons),
        parameters=checks,
        failed_parameters=failed,
        passed=bool(not failed and n_fitted > 0),
    )


def sbc_surface(
    spec: SurfaceSpec,
    panel_template: Panel,
    *,
    sbc_spec: SBCSpec,
    refit: SurfaceSpec | None = None,
) -> SBCResult | Unsupported:
    """SBC for a response surface: the outcome is simulated through ``Surface.forward`` at the
    prior draw plus the spec's likelihood noise, then refitted with ``refit`` (default: the
    same spec). ``panel_template`` supplies the doses and layout; its outcome is ignored."""
    surface = Surface(spec)
    data = prepare(spec, panel_template)
    model = surface.model

    def simulate(theta: Mapping[str, Array], rng: np.random.Generator) -> Array:
        mean = np.asarray(surface.forward(data, theta), dtype=np.float64)
        return _draw_noise(model, data, mean, theta, rng)

    refit_model = None if refit is None else Surface(refit).model
    return sbc(model, data, spec=sbc_spec, simulate=simulate, refit=refit_model)


def sbc_pool(
    pool_spec: PoolSpec,
    corpus: Corpus,
    *,
    sbc_spec: SBCSpec,
    refit: PoolSpec | None = None,
) -> SBCResult | Unsupported:
    """SBC for the meta-analytic pool: ``meta.pool_model`` gives the model and its data (the
    records' standard errors, loadings, moderators); the outcome column is re-simulated from
    the pool's likelihood at each prior draw. ``refit`` pools with a different spec (the
    negative control). ``Unsupported`` when the corpus cannot be laid out."""
    built = pool_model(pool_spec, corpus)
    if isinstance(built, Unsupported):
        return built
    model, data = built
    refit_model: ModelSpec | None = None
    if refit is not None:
        other = pool_model(refit, corpus)
        if isinstance(other, Unsupported):
            return other
        refit_model = other[0]
    return sbc(model, data, spec=sbc_spec, refit=refit_model)
