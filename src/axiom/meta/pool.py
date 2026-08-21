"""The Bayesian random-effects pool: a hierarchical normal–normal model through ``infer``.

The model, for one family with records ``i = 1..k`` grouped into effects
``g(i)`` (one per contributor by default, so a contributor's model read and
experimental read share an effect)::

    theta_g ~ N(mu, tau)                                    g = 1..G
    y_i     ~ N(theta_g(i) + x_i·gamma + delta·m_i,  se_i)  (known se_i)

``mu`` is the family mean on the experimental scale, ``tau`` the
between-effect sd, ``gamma`` the moderator coefficients on **centered**
moderators (``meta.moderators``), ``delta`` the provenance offset of a
model read (``m_i = 1``), identified only by dual-read contributors
(``meta.bias``). Priors are ``mu ~ N(0, s_mu)``, ``tau ~ HalfNormal(s_tau)``
(or fixed), ``gamma_j ~ N(0, s_gamma)``, ``delta ~ N(0, s_delta)``, every
scale explicit on the ``PoolSpec``.

Two parametrizations of the same posterior
------------------------------------------
``"centered"`` is the tree above, literally: a ``theta`` parameter of shape
``(G,)`` whose prior names ``mu`` and ``tau`` as parents, gathered by
``effect_index``, with the known per-record scale as the likelihood's
``scale_expr = Data("se")``. It is what a sampler (``backend="numpyro"``)
should see. It is **not** what a Laplace fit should see when ``tau`` is
free: the joint density in ``(theta, mu, log tau)`` has no mode — it grows
without bound as ``tau → 0`` with every ``theta_g → mu`` (the funnel) — so
``laplace`` reports ``Unverified`` or a meaningless Gaussian there.

``"marginal"`` (the ``"auto"`` default) integrates ``theta`` out exactly.
Whitening each record by its ``se`` and rotating each effect's records onto
the direction of the shared effect gives independent rows: one per effect
carrying ``mu / se_g`` with scale ``sqrt(1 + tau² / se_g²)`` (``se_g`` the
effect's precision-weighted se, ``1/se_g² = Σ_i 1/se_i²``), and ``n_g − 1``
within-effect contrasts carrying only the moderator and provenance terms
with unit scale. The rotation is orthonormal, so the likelihood is the
exact marginal of the hierarchy; the parameters left are ``(mu, tau, gamma,
delta)`` and the model is linear-Gaussian for fixed ``tau`` — the Laplace
posterior is then exact, and with free ``tau`` it is a Gaussian in ``log
tau`` with everything else exact conditionally. The effects are recovered
**draw by draw** from their conditional ``theta_g | rest ~ N((1 − B_g) ȳ_g
+ B_g mu, B_g se_g²)`` with ``B_g = se_g² / (se_g² + tau²)`` and ``ȳ_g`` the
effect's precision-weighted mean after removing the moderator and
provenance terms — so ``Pooled.posterior`` carries ``theta`` under either
parametrization and the shrinkage table is the textbook one.

Ported from the parent's ``benchmarks/meta_model.py`` (ledger row
``meta/pool.py``, PORT): per-family ``tau``, moderators, and ``delta_m``
identified by dual-read contributors; the PyMC block is rewritten against
``infer.get_backend``. The marginal form is new here (the parent sampled
the joint).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import (
    Add,
    Const,
    Data,
    Expr,
    Gather,
    Interval,
    Likelihood,
    ModelSpec,
    Mul,
    NonEmptyStr,
    Param,
    Posterior,
    Pow,
    Prior,
    Spec,
    Unsupported,
    Unverified,
    Verdict,
    dimensionless,
    interval,
)
from axiom.infer import get_backend
from axiom.meta.bias import delta_identification
from axiom.meta.moderators import ModeratorDesign, moderator_matrix
from axiom.meta.schema import Corpus, StudyRecord, poolable_quantity

__all__ = [
    "EffectShrinkage",
    "ParameterSummary",
    "PoolPriors",
    "PoolResult",
    "PoolSpec",
    "Pooled",
    "pool",
    "pool_model",
]

Array = npt.NDArray[np.float64]
EffectKey = Literal["study", "contributor"]
PoolDefinition = Literal["eti", "hdi"]
Parametrization = Literal["auto", "centered", "marginal"]


class PoolPriors(Spec):
    """Prior scales of the pool, all explicit.

    ``mu ~ N(0, mu_scale)``, ``tau ~ HalfNormal(tau_scale)``, ``gamma_j ~ N(0,
    gamma_scale)``, ``delta ~ N(0, delta_scale)``. ``tau_fixed`` replaces the
    half-normal with a point mass (``Prior(family="fixed")``) — the setting
    under which the pool is exactly the analytic normal–normal model.
    """

    mu_scale: float = Field(default=1.0, gt=0)
    tau_scale: float = Field(default=1.0, gt=0)
    gamma_scale: float = Field(default=1.0, gt=0)
    delta_scale: float = Field(default=1.0, gt=0)
    tau_fixed: float | None = None

    @model_validator(mode="after")
    def _finite(self) -> PoolPriors:
        for name in ("mu_scale", "tau_scale", "gamma_scale", "delta_scale"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.tau_fixed is not None and not (np.isfinite(self.tau_fixed) and self.tau_fixed >= 0):
            raise ValueError(f"tau_fixed must be finite and >= 0, got {self.tau_fixed}")
        return self

    def tau_prior(self) -> Prior:
        if self.tau_fixed is not None:
            return Prior(family="fixed", hyper={"value": float(self.tau_fixed)})
        return Prior(family="halfnormal", hyper={"sigma": float(self.tau_scale)})


class PoolSpec(Spec):
    """What to pool and how.

    ``family`` selects the records; ``moderators`` names the study-level
    covariates entering the mean (centered); ``bias_term`` adds the
    model-read offset ``delta``; ``effect_key`` says what a study effect
    ``theta`` is attached to — ``"contributor"`` (default: a contributor's
    reads share one effect, which is what identifies ``delta``) or
    ``"study"`` (every record its own effect, the classical layout; then
    ``bias_term`` is refused because no record can carry both reads).
    ``parametrization`` picks the tree (module docstring); ``"auto"`` is
    ``"marginal"``. ``mass``/``definition`` fix every interval the result
    reports.
    """

    family: NonEmptyStr
    moderators: tuple[str, ...] = ()
    bias_term: bool = False
    effect_key: EffectKey = "contributor"
    priors: PoolPriors = PoolPriors()
    parametrization: Parametrization = "auto"
    mass: float = 0.95
    definition: PoolDefinition = "eti"

    @property
    def marginal(self) -> bool:
        """True when ``theta`` is integrated out of the tree (``"marginal"`` or ``"auto"``)."""
        return self.parametrization != "centered"

    @model_validator(mode="after")
    def _consistent(self) -> PoolSpec:
        if not 0.0 < self.mass < 1.0:
            raise ValueError(f"mass must be in (0, 1), got {self.mass}")
        if len(set(self.moderators)) != len(self.moderators):
            raise ValueError(f"moderator names must be distinct: {list(self.moderators)}")
        if any(not m.strip() for m in self.moderators):
            raise ValueError("moderator names must be non-empty")
        if self.bias_term and self.effect_key != "contributor":
            raise ValueError(
                "bias_term needs effect_key='contributor': the provenance offset is identified "
                "by a contributor's two reads of one effect, and with effect_key='study' every "
                "record is its own effect"
            )
        return self


class ParameterSummary(Spec):
    """Posterior mean, sd, and one provenance-carrying interval for a scalar parameter.

    ``identified=False`` marks a parameter the data could not inform (the
    provenance offset without a dual-read contributor): what is reported is
    the prior, and ``note`` says so.
    """

    name: NonEmptyStr
    mean: float
    sd: float
    interval: Interval
    identified: bool = True
    note: str = ""


class EffectShrinkage(Spec):
    """One study effect's raw and shrunk estimate.

    ``estimate``/``se`` are the effect's precision-weighted combination of
    its records *after* removing the moderator and provenance terms at their
    posterior means; ``theta`` is the posterior mean of the effect;
    ``analytic`` is ``se² / (se² + tau²)`` at the posterior mean of ``tau``;
    ``empirical`` is ``1 − (theta − mu) / (estimate − mu)`` at posterior
    means, or ``None`` when the effect sits on ``mu`` and the ratio is
    undefined.
    """

    effect: NonEmptyStr
    studies: tuple[str, ...]
    estimate: float
    se: float
    theta: float
    interval: Interval
    analytic: float
    empirical: float | None


class PoolResult(Spec):
    """Posterior summaries of one family's pool, every interval at ``mass``/``definition``.

    ``k`` records over ``n_effects`` study effects; ``scale`` is the scale
    the numbers are on (``"log"`` when the catalog pools the quantity on the
    log scale — then ``mu`` is a log ratio). ``delta`` is ``None`` when the
    spec asked for no bias term; ``delta_verdict`` records whether it was
    identified. ``detail`` carries the provenance of every modelling choice.
    """

    family: NonEmptyStr
    quantity: NonEmptyStr
    scale: Literal["natural", "log"]
    k: int
    n_effects: int
    mu: ParameterSummary
    tau: ParameterSummary
    gamma: tuple[ParameterSummary, ...]
    delta: ParameterSummary | None
    thetas: tuple[ParameterSummary, ...]
    shrinkage: tuple[EffectShrinkage, ...]
    delta_verdict: Verdict | None
    mass: float
    definition: PoolDefinition
    backend: str
    draws: int
    seed: int | None
    model_hash: str
    detail: dict[str, str] = {}


@dataclass(frozen=True)
class Pooled:
    """``pool``'s return: the ``PoolResult`` plus the ``Posterior`` it was summarized from
    (``theta_<family>`` draws included under either parametrization), the ``ModelSpec`` and
    the data the backend saw — a plain carrier, so the draws stay out of the Spec."""

    result: PoolResult
    posterior: Posterior
    model: ModelSpec
    data: dict[str, Array]
    spec: PoolSpec


# -- names -------------------------------------------------------------------------------


def _tag(text: str) -> str:
    tag = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return tag or "family"


def _names(spec: PoolSpec) -> dict[str, str]:
    t = _tag(spec.family)
    out = {"mu": f"mu_{t}", "tau": f"tau_{t}", "theta": f"theta_{t}", "delta": f"delta_{t}"}
    for m in spec.moderators:
        out[f"gamma:{m}"] = f"gamma_{_tag(m)}_{t}"
    return out


# -- the corpus as arrays ------------------------------------------------------------------


@dataclass(frozen=True)
class _Layout:
    """The family's records as arrays on the pooling scale, grouped into effects."""

    records: tuple[StudyRecord, ...]
    y: Array
    se: Array
    scale: Literal["natural", "log"]
    labels: tuple[str, ...]
    index: npt.NDArray[np.int64]
    is_model: Array
    design: ModeratorDesign | None
    delta_in_mean: bool

    @property
    def n_effects(self) -> int:
        return len(self.labels)

    def members(self, g: int) -> npt.NDArray[np.int64]:
        return np.flatnonzero(self.index == g).astype(np.int64)

    def moderator(self, name: str) -> Array:
        assert self.design is not None
        return self.design.column(name)


def _family_records(spec: PoolSpec, corpus: Corpus) -> tuple[StudyRecord, ...] | Unsupported:
    records = corpus.by_family(spec.family).records
    if not records:
        return Unsupported(
            reason=f"corpus {corpus.name!r} has no records in family {spec.family!r}",
            detail={"families": ", ".join(corpus.families())},
        )
    not_poolable = [r.study for r in records if not r.is_dimensionless]
    if not_poolable:
        return Unsupported(
            reason="pooling is over dimensionless quantities; run meta.ingest.normalize with a "
            f"licensed TransferPlan first for {not_poolable}",
            missing=("TransferPlan",),
        )
    kinds = sorted({r.quantity for r in records})
    if len(kinds) != 1:
        return Unsupported(
            reason=f"family {spec.family!r} mixes quantities {kinds}; a pool is over one quantity",
            detail={"quantities": ", ".join(kinds)},
        )
    return records


def _on_pooling_scale(
    records: tuple[StudyRecord, ...],
) -> tuple[Array, Array, Literal["natural", "log"]]:
    """``(y, se, scale)``: log-scale quantities are pooled as ``log(estimate)`` with the
    delta-method se ``se / estimate``."""
    y = np.asarray([r.estimate for r in records], dtype=float)
    se = np.asarray([r.se for r in records], dtype=float)
    scale = poolable_quantity(records[0].quantity).scale
    if scale == "log":
        if np.any(y <= 0):
            bad = [r.study for r in records if r.estimate <= 0]
            raise ValueError(
                f"{records[0].quantity!r} is pooled on the log scale; non-positive estimates "
                f"on {bad} (ingest should have refused them)"
            )
        se = se / y
        y = np.log(y)
    return y, se, scale


def _layout(spec: PoolSpec, corpus: Corpus) -> _Layout | Unsupported:
    got = _family_records(spec, corpus)
    if isinstance(got, Unsupported):
        return got
    records = got
    y, se, scale = _on_pooling_scale(records)
    keys = [r.study if spec.effect_key == "study" else r.contributor for r in records]
    labels = tuple(dict.fromkeys(keys))  # first-appearance order
    index = np.asarray([labels.index(k) for k in keys], dtype=np.int64)
    design: ModeratorDesign | None = None
    if spec.moderators:
        md = moderator_matrix(Corpus(records=records, name=corpus.name), spec.moderators)
        if isinstance(md, Unsupported):
            return md
        design = md
    # Only a dual-read contributor licenses the offset; otherwise delta stays a prior-only
    # parameter (meta.bias explains why the exchangeability route is refused).
    delta_in_mean = spec.bias_term and delta_identification(corpus, spec.family).licensed
    return _Layout(
        records=records,
        y=y,
        se=se,
        scale=scale,
        labels=labels,
        index=index,
        is_model=np.asarray([1.0 if r.read == "model" else 0.0 for r in records]),
        design=design,
        delta_in_mean=delta_in_mean,
    )


# -- the tree ---------------------------------------------------------------------------------


def _hyper_params(spec: PoolSpec, names: Mapping[str, str]) -> list[Param]:
    dl = dimensionless()
    pri = spec.priors
    out = [
        Param(
            name=names["mu"],
            dimension=dl,
            prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": pri.mu_scale}),
        ),
        Param(name=names["tau"], dimension=dl, prior=pri.tau_prior()),
    ]
    for m in spec.moderators:
        out.append(
            Param(
                name=names[f"gamma:{m}"],
                dimension=dl,
                prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": pri.gamma_scale}),
            )
        )
    if spec.bias_term:
        out.append(
            Param(
                name=names["delta"],
                dimension=dl,
                prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": pri.delta_scale}),
            )
        )
    return out


def _centered(spec: PoolSpec, lay: _Layout) -> tuple[ModelSpec, dict[str, Array]]:
    names = _names(spec)
    dl = dimensionless()
    params = _hyper_params(spec, names)
    by_name = {p.name: p for p in params}
    theta = Param(
        name=names["theta"],
        dimension=dl,
        prior=Prior(family="normal", hyper={"mu": names["mu"], "sigma": names["tau"]}),
        shape=(lay.n_effects,),
    )
    terms: list[Expr] = [Gather(source=theta, index=Data(name="effect_index", dimension=dl))]
    data: dict[str, Array] = {
        "y": lay.y,
        "se": lay.se,
        "effect_index": lay.index.astype(np.float64),
        "is_model_read": lay.is_model,
    }
    for m in spec.moderators:
        col = f"mod_{_tag(m)}"
        terms.append(Mul(factors=(by_name[names[f"gamma:{m}"]], Data(name=col, dimension=dl))))
        data[col] = lay.moderator(m)
    if lay.delta_in_mean:
        terms.append(
            Mul(factors=(by_name[names["delta"]], Data(name="is_model_read", dimension=dl)))
        )
    model = ModelSpec(
        name=f"pool:{spec.family}",
        mean=Add(terms=tuple(terms)),
        outcome=Data(name="y", dimension=dl),
        likelihood=Likelihood(family="normal", scale_expr=Data(name="se", dimension=dl)),
        parameters=(*params, theta),
    )
    return model, data


def _rotation(lay: _Layout) -> tuple[Array, Array, Array]:
    """The orthonormal whitening rotation ``Q`` (``k × k``, block-diagonal over effects),
    the per-row ``mu`` loading ``a`` and the per-row ``tau²`` loading ``c``.

    Row ``i`` of the rotated system is ``Q[:, i]·(y/se)``. Within an effect
    the first row is the unit vector along ``u = 1/se``, so it carries
    ``mu·|u| = mu/se_g`` and picks up ``tau²/se_g²`` of extra variance; the
    remaining rows are orthogonal to ``u`` and carry neither.
    """
    k = lay.y.size
    Q = np.zeros((k, k))
    a = np.zeros(k)
    c = np.zeros(k)
    for g in range(lay.n_effects):
        idx = lay.members(g)
        u = 1.0 / lay.se[idx]
        norm = float(np.sqrt(np.sum(u**2)))
        q, _ = np.linalg.qr(u[:, None], mode="complete")
        if float(q[:, 0] @ u) < 0.0:
            q[:, 0] = -q[:, 0]
        Q[np.ix_(idx, idx)] = q
        a[idx[0]] = norm
        c[idx[0]] = norm**2
    return Q, a, c


def _marginal(spec: PoolSpec, lay: _Layout) -> tuple[ModelSpec, dict[str, Array]]:
    names = _names(spec)
    dl = dimensionless()
    params = _hyper_params(spec, names)
    by_name = {p.name: p for p in params}
    Q, a, c = _rotation(lay)
    w = lay.y / lay.se
    data: dict[str, Array] = {
        "outcome": Q.T @ w,
        "mu_load": a,
        "tau2_load": c,
        "se": lay.se,
    }
    terms: list[Expr] = [Mul(factors=(by_name[names["mu"]], Data(name="mu_load", dimension=dl)))]
    for m in spec.moderators:
        col = f"mod_{_tag(m)}"
        data[col] = Q.T @ (lay.moderator(m) / lay.se)
        terms.append(Mul(factors=(by_name[names[f"gamma:{m}"]], Data(name=col, dimension=dl))))
    if lay.delta_in_mean:
        data["model_read"] = Q.T @ (lay.is_model / lay.se)
        terms.append(Mul(factors=(by_name[names["delta"]], Data(name="model_read", dimension=dl))))
    one = Const(value=1.0, dimension=dl)
    tau2 = Pow(base=by_name[names["tau"]], exponent=Fraction(2))
    scale_expr = Pow(
        base=Add(terms=(one, Mul(factors=(tau2, Data(name="tau2_load", dimension=dl))))),
        exponent=Fraction(1, 2),
    )
    model = ModelSpec(
        name=f"pool:{spec.family}",
        mean=Add(terms=tuple(terms)),
        outcome=Data(name="outcome", dimension=dl),
        likelihood=Likelihood(family="normal", scale_expr=scale_expr),
        parameters=tuple(params),
    )
    return model, data


def pool_model(spec: PoolSpec, corpus: Corpus) -> tuple[ModelSpec, dict[str, Array]] | Unsupported:
    """The model and its data for ``spec`` on ``corpus``.

    Centered (``spec.parametrization == "centered"``): data ``y``, ``se``,
    ``effect_index``, ``is_model_read``, centered ``mod_<name>`` columns;
    parameters ``mu_<f>``, ``tau_<f>``, ``theta_<f>`` (shape ``(G,)``, prior
    naming ``mu``/``tau``), ``gamma_<m>_<f>``, ``delta_<f>``; likelihood
    ``normal`` with ``scale_expr = Data("se")``.

    Marginal (default): the rotated system of the module docstring — data
    ``outcome``, ``mu_load``, ``tau2_load``, rotated ``mod_<name>`` and
    ``model_read`` columns; parameters ``mu_<f>``, ``tau_<f>``,
    ``gamma_<m>_<f>``, ``delta_<f>``; likelihood ``normal`` with ``scale_expr
    = sqrt(1 + tau² · tau2_load)``. ``delta_<f>`` is declared whenever
    ``bias_term`` and enters the mean only when identified.
    """
    lay = _layout(spec, corpus)
    if isinstance(lay, Unsupported):
        return lay
    return _marginal(spec, lay) if spec.marginal else _centered(spec, lay)


# -- fit ---------------------------------------------------------------------------------


def _summary(
    name: str, draws: Array, *, definition: PoolDefinition, mass: float, **flags: object
) -> ParameterSummary:
    x = np.asarray(draws, dtype=float).ravel()
    return ParameterSummary(
        name=name,
        mean=float(x.mean()),
        sd=float(x.std(ddof=1)) if x.size > 1 else 0.0,
        interval=interval(x, definition=definition, mass=mass),
        **flags,  # type: ignore[arg-type]
    )


def _fixed_summary(
    name: str, value: float, *, definition: PoolDefinition, mass: float
) -> ParameterSummary:
    return ParameterSummary(
        name=name,
        mean=value,
        sd=0.0,
        interval=Interval(lower=value, upper=value, definition=definition, mass=mass),
        note="fixed by the spec",
    )


def _run(
    backend: str, model: ModelSpec, data: Mapping[str, Array], *, draws: int, seed: int | None
) -> Posterior | Unverified | Unsupported:
    b = get_backend(backend)
    if isinstance(b, Unsupported):
        return b
    if backend.strip().lower() == "laplace":
        return b.laplace(model, data, draws=draws, seed=seed)
    return b.sample(model, data, draws=draws, tune=draws, chains=4, seed=seed)


def pool(
    spec: PoolSpec,
    corpus: Corpus,
    *,
    backend: str = "laplace",
    draws: int = 2000,
    seed: int | None = 0,
) -> Pooled | Unverified | Unsupported:
    """Fit the pool and summarize it.

    ``backend`` is an ``infer`` backend name (``"laplace"`` needs nothing
    beyond core; ``"numpyro"`` needs the extra). A backend that cannot stand
    behind its draws returns ``Unverified`` unchanged; an unusable corpus
    returns ``Unsupported`` with the reason. ``delta`` (when asked for) is
    flagged ``identified=False`` — and is the prior — on a corpus with no
    dual-read contributor in the family.
    """
    if draws < 2:
        raise ValueError("draws must be at least 2")
    lay = _layout(spec, corpus)
    if isinstance(lay, Unsupported):
        return lay
    model, data = _marginal(spec, lay) if spec.marginal else _centered(spec, lay)
    post = _run(backend, model, data, draws=draws, seed=seed)
    if isinstance(post, Unverified | Unsupported):
        return post
    names = _names(spec)
    if spec.marginal:
        post = _with_effects(spec, lay, post, seed=seed)
    result = _summarize(
        spec, corpus, lay, model, post, names, backend=backend, draws=draws, seed=seed
    )
    return Pooled(result=result, posterior=post, model=model, data=data, spec=spec)


def _tau_draws(spec: PoolSpec, post: Posterior, names: Mapping[str, str]) -> Array:
    if spec.priors.tau_fixed is not None:
        return np.full(post.n_draws(), float(spec.priors.tau_fixed))
    return np.asarray(post.flat(names["tau"]), dtype=float)


def _adjusted(spec: PoolSpec, lay: _Layout, post: Posterior, names: Mapping[str, str]) -> Array:
    """``y`` minus the moderator and provenance terms, per draw: ``(n_draws, k)``."""
    adj = np.broadcast_to(lay.y, (post.n_draws(), lay.y.size)).copy()
    for m in spec.moderators:
        adj -= post.flat(names[f"gamma:{m}"])[:, None] * lay.moderator(m)[None, :]
    if lay.delta_in_mean:
        adj -= post.flat(names["delta"])[:, None] * lay.is_model[None, :]
    return np.asarray(adj, dtype=float)


def _with_effects(spec: PoolSpec, lay: _Layout, post: Posterior, *, seed: int | None) -> Posterior:
    """Add ``theta`` draws from their exact conditional ``N((1−B) ȳ_g + B mu, B se_g²)``."""
    names = _names(spec)
    n = post.n_draws()
    mu = post.flat(names["mu"])
    tau = _tau_draws(spec, post, names)
    adj = _adjusted(spec, lay, post, names)
    rng = np.random.default_rng(None if seed is None else int(seed) + 1)
    theta = np.empty((n, lay.n_effects))
    for g in range(lay.n_effects):
        idx = lay.members(g)
        w = 1.0 / lay.se[idx] ** 2
        se2_g = 1.0 / float(np.sum(w))
        ybar = adj[:, idx] @ w * se2_g
        B = se2_g / (se2_g + tau**2)
        theta[:, g] = (1.0 - B) * ybar + B * mu + np.sqrt(B * se2_g) * rng.standard_normal(n)
    draws = {name: post.draws(name) for name in sorted(post.names())}
    draws[names["theta"]] = theta.reshape(post.n_chains, -1, lay.n_effects)
    coords = dict(post.coords())
    coords[names["theta"]] = list(lay.labels)
    return Posterior(
        draws,
        coords=coords,
        provenance={**post.provenance, "theta": "conditional draws given (mu, tau, gamma, delta)"},
    )


def _summarize(
    spec: PoolSpec,
    corpus: Corpus,
    lay: _Layout,
    model: ModelSpec,
    post: Posterior,
    names: Mapping[str, str],
    *,
    backend: str,
    draws: int,
    seed: int | None,
) -> PoolResult:
    mass, definition = spec.mass, spec.definition
    detail: dict[str, str] = {
        "parametrization": (
            "marginal: theta integrated out, effects drawn from their conditional"
            if spec.marginal
            else "centered: theta ~ N(mu, tau) in the tree"
        ),
        "likelihood": "normal with the known per-record se as scale_expr",
        "effect_key": spec.effect_key,
        "moderators": "centered at the corpus mean" if spec.moderators else "none",
        "scale": lay.scale,
    }
    if lay.scale == "log":
        detail["log_scale"] = "estimate -> log(estimate); se -> se / estimate (delta method)"

    mu_s = _summary(names["mu"], post.flat(names["mu"]), definition=definition, mass=mass)
    if spec.priors.tau_fixed is not None:
        tau_hat = float(spec.priors.tau_fixed)
        tau_s = _fixed_summary(names["tau"], tau_hat, definition=definition, mass=mass)
    else:
        tau_s = _summary(names["tau"], post.flat(names["tau"]), definition=definition, mass=mass)
        tau_hat = tau_s.mean
    gamma_s = tuple(
        _summary(
            names[f"gamma:{m}"], post.flat(names[f"gamma:{m}"]), definition=definition, mass=mass
        )
        for m in spec.moderators
    )

    delta_s: ParameterSummary | None = None
    verdict: Verdict | None = None
    if spec.bias_term:
        verdict = delta_identification(corpus, spec.family)
        d = post.flat(names["delta"])
        if verdict.licensed:
            delta_s = _summary(names["delta"], d, definition=definition, mass=mass)
            detail["delta"] = (
                "identified by dual-read contributors; assumes a contributor's two reads "
                "estimate the same effect (" + verdict.route + ")"
            )
        else:
            delta_s = _summary(
                names["delta"],
                d,
                definition=definition,
                mass=mass,
                identified=False,
                note=f"not identified: {verdict.reason}; the summary is the prior "
                f"N(0, {spec.priors.delta_scale})",
            )
            detail["delta"] = (
                f"NOT identified ({verdict.reason}); delta was kept out of the mean rather "
                "than identified through exchangeability of model-only and experiment-only "
                "contributors, so its posterior is its prior"
            )

    adj = _adjusted(spec, lay, post, names).mean(axis=0)  # at posterior means (linear terms)
    theta_draws = post.flat(names["theta"])  # (n, G)
    thetas: list[ParameterSummary] = []
    shrink: list[EffectShrinkage] = []
    for g, label in enumerate(lay.labels):
        idx = lay.members(g)
        w = 1.0 / lay.se[idx] ** 2
        est_g = float(np.sum(w * adj[idx]) / np.sum(w))
        se_g = float(np.sqrt(1.0 / np.sum(w)))
        ts = _summary(
            f"{names['theta']}[{label}]", theta_draws[:, g], definition=definition, mass=mass
        )
        thetas.append(ts)
        analytic = se_g**2 / (se_g**2 + tau_hat**2)
        gap = est_g - mu_s.mean
        empirical = (
            None
            if abs(gap) <= 1e-12 * max(1.0, abs(est_g), abs(mu_s.mean))
            else 1.0 - (ts.mean - mu_s.mean) / gap
        )
        shrink.append(
            EffectShrinkage(
                effect=label,
                studies=tuple(lay.records[i].study for i in idx),
                estimate=est_g,
                se=se_g,
                theta=ts.mean,
                interval=ts.interval,
                analytic=float(analytic),
                empirical=None if empirical is None else float(empirical),
            )
        )

    prov = post.provenance
    for key in ("hessian_pd", "converged", "derivatives", "min_eigenvalue"):
        if key in prov:
            detail[f"posterior.{key}"] = repr(prov[key])
    return PoolResult(
        family=spec.family,
        quantity=lay.records[0].quantity,
        scale=lay.scale,
        k=len(lay.records),
        n_effects=lay.n_effects,
        mu=mu_s,
        tau=tau_s,
        gamma=gamma_s,
        delta=delta_s,
        thetas=tuple(thetas),
        shrinkage=tuple(shrink),
        delta_verdict=verdict,
        mass=mass,
        definition=definition,
        backend=backend,
        draws=int(post.n_draws()),
        seed=seed,
        model_hash=model.content_hash(),
        detail=detail,
    )
