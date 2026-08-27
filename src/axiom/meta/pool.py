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

Three levels: study within party
--------------------------------
``effect_key="contributor"`` says a party's studies all estimate one effect;
``effect_key="study"`` says a party's studies are no more alike than any two
studies. Neither is right for a house running repeated experiments in the same
markets, and the cost of getting it wrong lands on ``mu``: pool thirty studies
from three parties as thirty independent draws and the interval on the family
mean is the interval you would have earned from thirty independent markets.

``effect_key="nested"`` is the third option, with two variance components::

    alpha_p ~ N(mu, tau_party)                              p = 1..P
    theta_g ~ N(alpha_p(g), tau)                            g = 1..G
    y_i     ~ N(theta_g(i) + x_i·gamma + delta·m_i,  se_i)

``tau`` is now the between-study sd *within* a party and ``tau_party`` the sd
between parties, so ``mu``'s uncertainty reflects the number of parties. The
tree is the ``"nested"`` parametrization, and it is the marginal one with a
level on top: ``theta`` is integrated out exactly by the same within-study
rotation, and ``alpha`` stays in the tree because a diagonal likelihood has no
way to express the party-level correlation among study rows — rotating it away
would need a rotation that depends on ``tau``, which is a parameter.

The party level is **non-centered** (``alpha_p = mu + tau_party · z_p`` with
``z_p ~ N(0, 1)``), which is what a sampler wants and what makes the divergence
count small. It is also why a sampler is *required* unless ``tau_party`` is
fixed: the likelihood sees only the product ``tau_party · z_p``, and a Laplace
expansion at one point of that multiplicative ridge overstates ``tau_party``
threefold. ``pool`` returns ``Unsupported`` for that combination rather than
the number (see ``_capability_guard``).

``delta`` survives the move. It is identified by a contributor's two reads of
one shared effect, and under the nested tree a party's reads still share its
``alpha`` — so the nested pool is the only one of the three that identifies the
provenance offset *and* separates within-party from between-party
heterogeneity. With ``effect_key="contributor"`` you get ``delta`` and no
within-party heterogeneity; with ``"study"``, the reverse.

Ported from the parent's ``benchmarks/meta_model.py`` (ledger row
``meta/pool.py``, PORT): per-family ``tau``, moderators, and ``delta_m``
identified by dual-read contributors; the PyMC block is rewritten against
``infer.get_backend``. The marginal form is new here (the parent sampled
the joint), and so is the nested one (see
``docs/notes/0029-the-party-a-study-came-from.md``).
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
from axiom.estimands import Estimand
from axiom.infer import get_backend
from axiom.meta.bias import delta_identification
from axiom.meta.commensurate import Commensurability, commensurable
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
EffectKey = Literal["study", "contributor", "nested"]
PoolDefinition = Literal["eti", "hdi"]
Parametrization = Literal["auto", "centered", "marginal", "nested"]


class PoolPriors(Spec):
    """Prior scales of the pool, all explicit.

    ``mu ~ N(0, mu_scale)``, ``tau ~ HalfNormal(tau_scale)``, ``gamma_j ~ N(0,
    gamma_scale)``, ``delta ~ N(0, delta_scale)``. ``tau_fixed`` replaces the
    half-normal with a point mass (``Prior(family="fixed")``) — the setting
    under which the pool is exactly the analytic normal–normal model.

    ``tau_party_scale``/``tau_party_fixed`` are the same pair for the
    between-party sd of the nested tree (``effect_key="nested"``) and are
    ignored otherwise. Fixing ``tau_party`` is both a real modelling choice —
    with three parties the data say little about the spread between them — and
    the only way a sampler-free backend can fit the nested tree at all; see
    ``_capability_guard``.
    """

    mu_scale: float = Field(default=1.0, gt=0)
    tau_scale: float = Field(default=1.0, gt=0)
    gamma_scale: float = Field(default=1.0, gt=0)
    delta_scale: float = Field(default=1.0, gt=0)
    tau_fixed: float | None = None
    tau_party_scale: float = Field(default=1.0, gt=0)
    tau_party_fixed: float | None = None

    @model_validator(mode="after")
    def _finite(self) -> PoolPriors:
        for name in ("mu_scale", "tau_scale", "gamma_scale", "delta_scale", "tau_party_scale"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("tau_fixed", "tau_party_fixed"):
            v = getattr(self, name)
            if v is not None and not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        return self

    def tau_prior(self) -> Prior:
        if self.tau_fixed is not None:
            return Prior(family="fixed", hyper={"value": float(self.tau_fixed)})
        return Prior(family="halfnormal", hyper={"sigma": float(self.tau_scale)})

    def tau_party_prior(self) -> Prior:
        if self.tau_party_fixed is not None:
            return Prior(family="fixed", hyper={"value": float(self.tau_party_fixed)})
        return Prior(family="halfnormal", hyper={"sigma": float(self.tau_party_scale)})


class PoolSpec(Spec):
    """What to pool and how.

    ``family`` selects the records; ``moderators`` names the study-level
    covariates entering the mean (centered); ``bias_term`` adds the
    model-read offset ``delta``; ``effect_key`` says what a study effect
    ``theta`` is attached to:

    * ``"contributor"`` (default) — a contributor's reads share one effect,
      which is what identifies ``delta``, and which assumes a party's studies
      all estimate the *same* effect;
    * ``"study"`` — every record its own effect, the classical layout, which
      assumes studies from one party are no more alike than studies from
      different parties; then ``bias_term`` is refused because no record can
      carry both reads;
    * ``"nested"`` — study within party (module docstring): both assumptions
      dropped, ``tau`` becomes the between-study sd *within* a party and
      ``tau_party`` the between-party sd. ``delta`` stays identified, because
      a party's dual reads still share its ``alpha``.

    ``parametrization`` picks the tree (module docstring); ``"auto"`` is
    ``"marginal"``, or ``"nested"`` when ``effect_key`` is ``"nested"``.
    ``mass``/``definition`` fix every interval the result reports.
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
    def tree(self) -> Literal["centered", "marginal", "nested"]:
        """The parametrization actually used, with ``"auto"`` resolved."""
        if self.parametrization != "auto":
            return self.parametrization
        return "nested" if self.effect_key == "nested" else "marginal"

    @property
    def marginal(self) -> bool:
        """True when ``theta`` is integrated out of the tree (``"marginal"`` or ``"nested"``)."""
        return self.tree != "centered"

    @property
    def nested(self) -> bool:
        """True when the tree carries a party level above the study effects."""
        return self.tree == "nested"

    @model_validator(mode="after")
    def _consistent(self) -> PoolSpec:
        if not 0.0 < self.mass < 1.0:
            raise ValueError(f"mass must be in (0, 1), got {self.mass}")
        if len(set(self.moderators)) != len(self.moderators):
            raise ValueError(f"moderator names must be distinct: {list(self.moderators)}")
        if any(not m.strip() for m in self.moderators):
            raise ValueError("moderator names must be non-empty")
        if self.bias_term and self.effect_key == "study":
            raise ValueError(
                "bias_term needs effect_key='contributor' or 'nested': the provenance offset is "
                "identified by a contributor's two reads of one shared effect, and with "
                "effect_key='study' every record is its own effect"
            )
        if (self.effect_key == "nested") != (self.tree == "nested"):
            raise ValueError(
                "the nested tree is the nested effect key and vice versa: effect_key='nested' "
                "takes parametrization 'auto' or 'nested', and no other effect key takes "
                f"'nested'. Got parametrization={self.parametrization!r} with "
                f"effect_key={self.effect_key!r}"
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
    ``empirical`` is ``1 − (theta − target) / (estimate − target)`` at
    posterior means, or ``None`` when the effect sits on the target and the
    ratio is undefined. ``target`` is ``mu`` under the two-level trees; under
    the nested tree it is the effect's own party mean ``alpha_p``, reported as
    ``toward`` beside the ``party`` the effect belongs to.
    """

    effect: NonEmptyStr
    studies: tuple[str, ...]
    estimate: float
    se: float
    theta: float
    interval: Interval
    analytic: float
    empirical: float | None
    party: str = ""
    toward: float | None = None


class PoolResult(Spec):
    """Posterior summaries of one family's pool, every interval at ``mass``/``definition``.

    ``k`` records over ``n_effects`` study effects; ``scale`` is the scale
    the numbers are on (``"log"`` when the catalog pools the quantity on the
    log scale — then ``mu`` is a log ratio). ``delta`` is ``None`` when the
    spec asked for no bias term; ``delta_verdict`` records whether it was
    identified. ``detail`` carries the provenance of every modelling choice.

    Under the nested tree ``tau`` is the between-study sd *within* a party and
    ``tau_party`` the between-party sd, ``alphas`` are the ``n_parties`` party
    means, and ``mu`` is the mean across parties — whose interval reflects the
    number of parties rather than the number of studies. Both are ``None``/
    empty under the two-level trees, where there is one variance component and
    ``tau`` is it.
    """

    family: NonEmptyStr
    quantity: NonEmptyStr
    scale: Literal["natural", "log"]
    k: int
    n_effects: int
    n_parties: int = 0
    mu: ParameterSummary
    tau: ParameterSummary
    tau_party: ParameterSummary | None = None
    alphas: tuple[ParameterSummary, ...] = ()
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
    out = {
        "mu": f"mu_{t}",
        "tau": f"tau_{t}",
        "theta": f"theta_{t}",
        "delta": f"delta_{t}",
        "alpha": f"alpha_{t}",
        "party_offset": f"z_party_{t}",
        "tau_party": f"tau_party_{t}",
    }
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
    parties: tuple[str, ...] = ()
    party_of_effect: npt.NDArray[np.int64] | None = None

    @property
    def n_effects(self) -> int:
        return len(self.labels)

    @property
    def n_parties(self) -> int:
        return len(self.parties)

    def party_index(self) -> npt.NDArray[np.int64]:
        """The party of each effect; empty when the layout carries no party level."""
        if self.party_of_effect is None:
            raise ValueError("this layout has no party level")
        return self.party_of_effect

    def party_members(self, p: int) -> npt.NDArray[np.int64]:
        """The effects belonging to party ``p``."""
        return np.flatnonzero(self.party_index() == p).astype(np.int64)

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
    # A scaled record is poolable once ``meta.ingest.normalize`` admitted it under a
    # licensed TransferPlan (the plan's hash and status are stamped into ``detail``).
    not_poolable = [
        r.study
        for r in records
        if not r.is_dimensionless
        and r.detail.get("transfer_status") not in ("identified", "downgraded")
    ]
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
    keys = [r.contributor if spec.effect_key == "contributor" else r.study for r in records]
    labels = tuple(dict.fromkeys(keys))  # first-appearance order
    index = np.asarray([labels.index(k) for k in keys], dtype=np.int64)
    parties: tuple[str, ...] = ()
    party_of_effect: npt.NDArray[np.int64] | None = None
    if spec.effect_key == "nested":
        # ``Corpus`` already refuses duplicate study ids, so a study has one contributor.
        by_effect = {r.study: r.contributor for r in records}
        parties = tuple(dict.fromkeys(by_effect[label] for label in labels))
        party_of_effect = np.asarray(
            [parties.index(by_effect[label]) for label in labels], dtype=np.int64
        )
        guard = _nested_is_identified(labels, parties, party_of_effect)
        if isinstance(guard, Unsupported):
            return guard
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
        parties=parties,
        party_of_effect=party_of_effect,
    )


def _nested_is_identified(
    labels: tuple[str, ...], parties: tuple[str, ...], party_of_effect: npt.NDArray[np.int64]
) -> None | Unsupported:
    """Both variance components need something to vary over."""
    if len(parties) < 2:
        return Unsupported(
            reason=(
                f"a nested pool needs at least two parties; the family has one ({parties[0]!r}). "
                "Use effect_key='study' — with one party the two levels are one"
            ),
            detail={"parties": ", ".join(parties)},
            missing=("a second party",),
        )
    counts = np.bincount(party_of_effect, minlength=len(parties))
    if int(counts.max()) < 2:
        return Unsupported(
            reason=(
                f"a nested pool needs a party with at least two studies; all {len(parties)} "
                "parties contribute one, so the between-study and between-party sds are the "
                "same quantity. Use effect_key='study'"
            ),
            detail={"studies_per_party": ", ".join(str(int(c)) for c in counts)},
            missing=("a repeated study within one party",),
        )
    return None


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
    if spec.nested:
        out.append(Param(name=names["tau_party"], dimension=dl, prior=pri.tau_party_prior()))
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


def _nested(spec: PoolSpec, lay: _Layout) -> tuple[ModelSpec, dict[str, Array]]:
    """The marginal rotation with a party level above it.

    ``theta`` is integrated out exactly, as in ``_marginal``; what the first
    row of each study now loads is not ``mu`` but that study's party effect
    ``alpha_p = mu + tau_party · z_p``, which stays in the tree because a
    diagonal likelihood cannot express the party-level correlation any other
    way (module docstring). The party level is **non-centered** — the sampled
    parameter is the standard-normal ``z_p`` and ``alpha`` is reconstructed
    from it — so the tree has no funnel and ``laplace`` can fit it.
    """
    names = _names(spec)
    dl = dimensionless()
    params = _hyper_params(spec, names)
    by_name = {p.name: p for p in params}
    Q, a, c = _rotation(lay)
    offset = Param(
        name=names["party_offset"],
        dimension=dl,
        prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0}),
        shape=(lay.n_parties,),
    )
    party_of_row = lay.party_index()[lay.index].astype(np.float64)
    data: dict[str, Array] = {
        "outcome": Q.T @ (lay.y / lay.se),
        "mu_load": a,
        "tau2_load": c,
        "se": lay.se,
        "party_index": party_of_row,
    }
    alpha = Add(
        terms=(
            by_name[names["mu"]],
            Mul(
                factors=(
                    by_name[names["tau_party"]],
                    Gather(source=offset, index=Data(name="party_index", dimension=dl)),
                )
            ),
        )
    )
    terms: list[Expr] = [Mul(factors=(alpha, Data(name="mu_load", dimension=dl)))]
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
        parameters=(*params, offset),
    )
    return model, data


def _build(spec: PoolSpec, lay: _Layout) -> tuple[ModelSpec, dict[str, Array]]:
    if spec.tree == "centered":
        return _centered(spec, lay)
    return _nested(spec, lay) if spec.nested else _marginal(spec, lay)


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
    return _build(spec, lay)


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


def _capability_guard(spec: PoolSpec, backend: str) -> None | Unsupported:
    """A sampler-free backend cannot fit a free between-party sd.

    The party level is non-centered — the likelihood sees ``tau_party · z_p`` —
    so ``(tau_party, z)`` lies on a multiplicative ridge that only the priors
    break. HMC traverses the ridge; a Gaussian expanded at one point of a
    curved ridge does not. Measured: on simulated corpora with a true
    ``tau_party`` of 0.4, ``laplace`` returns 1.3-1.8 and does not improve with
    more parties, while ``numpyro`` returns 0.42
    (``tests/unit/test_meta_pool_nested.py``). Fixing ``tau_party`` removes
    the ridge — it is then a constant — and the fit is exact conditional on
    ``tau``.
    """
    if backend.strip().lower() != "laplace":
        return None
    if spec.nested and spec.priors.tau_party_fixed is None:
        return Unsupported(
            reason=(
                "a nested pool with a free tau_party needs a sampler: the party level is "
                "non-centered, so tau_party and the party offsets lie on a multiplicative "
                "ridge and a Laplace expansion at one point of it overstates tau_party by a "
                "factor of three at every party count. Fix it with "
                "PoolPriors(tau_party_fixed=...) — exact under laplace — or fit with a "
                "sampler backend"
            ),
            detail={"backend": backend, "parametrization": spec.tree},
            missing=("a sampler backend, or PoolPriors.tau_party_fixed",),
        )
    return None


def pool(
    spec: PoolSpec,
    corpus: Corpus,
    *,
    backend: str = "laplace",
    draws: int = 2000,
    seed: int | None = 0,
    estimands: Mapping[str, Estimand] | None = None,
) -> Pooled | Unverified | Unsupported:
    """Fit the pool and summarize it.

    ``backend`` is an ``infer`` backend name (``"laplace"`` needs nothing
    beyond core; ``"numpyro"`` needs the extra). A backend that cannot stand
    behind its draws returns ``Unverified`` unchanged; an unusable corpus
    returns ``Unsupported`` with the reason.

    ``estimands`` maps each record's ``study`` to the ``Estimand`` behind it.
    Supplied, it is checked with ``meta.commensurable`` and a corpus that is
    averaging different quantities comes back ``Unsupported`` naming the
    records and the facet. Omitted, the pool runs as it always has and says
    ``unchecked`` in ``PoolResult.detail["commensurability"]`` — the check is
    optional, the silence about it is not. ``delta`` (when asked for) is
    flagged ``identified=False`` — and is the prior — on a corpus with no
    dual-read contributor in the family.
    """
    if draws < 2:
        raise ValueError("draws must be at least 2")
    lay = _layout(spec, corpus)
    if isinstance(lay, Unsupported):
        return lay
    unusable = _capability_guard(spec, backend)
    if isinstance(unusable, Unsupported):
        return unusable
    agreement: Commensurability | None = None
    if estimands is not None:
        agreement = commensurable(corpus, estimands, family=spec.family)
        verdict = agreement.verdict()
        if verdict.status == "blocked":
            return Unsupported(
                reason=f"the corpus is not pooling one quantity: {verdict.reason}",
                detail={"family": spec.family, "reference": agreement.reference},
                missing=("commensurable estimands",),
            )
    model, data = _build(spec, lay)
    post = _run(backend, model, data, draws=draws, seed=seed)
    if isinstance(post, Unverified | Unsupported):
        return post
    names = _names(spec)
    if spec.marginal:
        post = _with_effects(spec, lay, post, seed=seed)
    result = _summarize(
        spec,
        corpus,
        lay,
        model,
        post,
        names,
        backend=backend,
        draws=draws,
        seed=seed,
        agreement=agreement,
    )
    return Pooled(result=result, posterior=post, model=model, data=data, spec=spec)


def _tau_draws(spec: PoolSpec, post: Posterior, names: Mapping[str, str]) -> Array:
    if spec.priors.tau_fixed is not None:
        return np.full(post.n_draws(), float(spec.priors.tau_fixed))
    return np.asarray(post.flat(names["tau"]), dtype=float)


def _tau_party_draws(spec: PoolSpec, post: Posterior, names: Mapping[str, str]) -> Array:
    if spec.priors.tau_party_fixed is not None:
        return np.full(post.n_draws(), float(spec.priors.tau_party_fixed))
    return np.asarray(post.flat(names["tau_party"]), dtype=float)


def _with_alpha(
    spec: PoolSpec, lay: _Layout, post: Posterior, names: Mapping[str, str]
) -> Posterior:
    """Reconstruct the party effects ``alpha_p = mu + tau_party · z_p`` from the sampled offsets."""
    n = post.n_draws()
    mu = np.asarray(post.flat(names["mu"]), dtype=float)
    tau_party = _tau_party_draws(spec, post, names)
    z = np.asarray(post.flat(names["party_offset"]), dtype=float).reshape(n, lay.n_parties)
    alpha = mu[:, None] + tau_party[:, None] * z
    draws = {name: post.draws(name) for name in sorted(post.names())}
    draws[names["alpha"]] = alpha.reshape(post.n_chains, -1, lay.n_parties)
    coords = dict(post.coords())
    coords[names["alpha"]] = list(lay.parties)
    return Posterior(
        draws,
        coords=coords,
        provenance={**post.provenance, "alpha": "mu + tau_party * z_party (non-centered)"},
    )


def _adjusted(spec: PoolSpec, lay: _Layout, post: Posterior, names: Mapping[str, str]) -> Array:
    """``y`` minus the moderator and provenance terms, per draw: ``(n_draws, k)``."""
    adj = np.broadcast_to(lay.y, (post.n_draws(), lay.y.size)).copy()
    for m in spec.moderators:
        adj -= post.flat(names[f"gamma:{m}"])[:, None] * lay.moderator(m)[None, :]
    if lay.delta_in_mean:
        adj -= post.flat(names["delta"])[:, None] * lay.is_model[None, :]
    return np.asarray(adj, dtype=float)


def _shrink_target(
    spec: PoolSpec, lay: _Layout, post: Posterior, names: Mapping[str, str]
) -> Array:
    """What each effect shrinks toward, per draw: ``(n_draws, n_effects)``.

    ``mu`` under the two-level trees; under the nested tree an effect shrinks
    toward *its own party's* ``alpha``, which is the whole point of the level.
    """
    n = post.n_draws()
    if not spec.nested:
        return np.repeat(post.flat(names["mu"])[:, None], lay.n_effects, axis=1)
    alpha = np.asarray(post.flat(names["alpha"]), dtype=float).reshape(n, lay.n_parties)
    return alpha[:, lay.party_index()]


def _with_effects(spec: PoolSpec, lay: _Layout, post: Posterior, *, seed: int | None) -> Posterior:
    """Add ``theta`` draws from their exact conditional ``N((1−B) ȳ_g + B mu, B se_g²)``."""
    names = _names(spec)
    if spec.nested:
        post = _with_alpha(spec, lay, post, names)
    n = post.n_draws()
    target = _shrink_target(spec, lay, post, names)
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
        theta[:, g] = (
            (1.0 - B) * ybar + B * target[:, g] + np.sqrt(B * se2_g) * rng.standard_normal(n)
        )
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
    agreement: Commensurability | None = None,
) -> PoolResult:
    mass, definition = spec.mass, spec.definition
    trees = {
        "centered": "centered: theta ~ N(mu, tau) in the tree",
        "marginal": "marginal: theta integrated out, effects drawn from their conditional",
        "nested": (
            "nested: theta integrated out within each study, alpha ~ N(mu, tau_party) per "
            "party left in the tree"
        ),
    }
    detail: dict[str, str] = {
        "parametrization": trees[spec.tree],
        "likelihood": "normal with the known per-record se as scale_expr",
        "effect_key": spec.effect_key,
        "moderators": "centered at the corpus mean" if spec.moderators else "none",
        "scale": lay.scale,
        "commensurability": (
            "unchecked: no estimands supplied, so nothing compared what the records measure"
            if agreement is None
            else agreement.ledger_line().statement
        ),
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

    tau_party_s: ParameterSummary | None = None
    alphas: tuple[ParameterSummary, ...] = ()
    if spec.nested:
        if spec.priors.tau_party_fixed is not None:
            tau_party_s = _fixed_summary(
                names["tau_party"],
                float(spec.priors.tau_party_fixed),
                definition=definition,
                mass=mass,
            )
        else:
            tau_party_s = _summary(
                names["tau_party"],
                post.flat(names["tau_party"]),
                definition=definition,
                mass=mass,
            )
        alpha_draws = np.asarray(post.flat(names["alpha"]), dtype=float).reshape(
            post.n_draws(), lay.n_parties
        )
        detail["party_level"] = "non-centered: alpha = mu + tau_party * z_party"
        alphas = tuple(
            _summary(
                f"{names['alpha']}[{party}]",
                alpha_draws[:, p_i],
                definition=definition,
                mass=mass,
            )
            for p_i, party in enumerate(lay.parties)
        )
        detail["levels"] = (
            f"{lay.n_parties} parties over {lay.n_effects} studies; tau is between studies "
            "within a party, tau_party between parties"
        )
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
    targets = _shrink_target(spec, lay, post, names).mean(axis=0)  # mu, or the party's alpha
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
        target = float(targets[g])
        gap = est_g - target
        empirical = (
            None
            if abs(gap) <= 1e-12 * max(1.0, abs(est_g), abs(target))
            else 1.0 - (ts.mean - target) / gap
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
                party=lay.parties[int(lay.party_index()[g])] if spec.nested else "",
                toward=target if spec.nested else None,
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
        n_parties=lay.n_parties,
        mu=mu_s,
        tau=tau_s,
        tau_party=tau_party_s,
        alphas=alphas,
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
