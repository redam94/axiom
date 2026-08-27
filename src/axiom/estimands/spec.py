"""``Estimand``: a complete transferability key, and ``transfer_to``.

Two quantities are the same quantity iff all eight facets match:

| Facet | Field(s) | What it fixes |
|---|---|---|
| ``quantity`` | ``quantity`` | the functional: contrast, marginal, ratio, elasticity, area |
| ``intervention`` | ``treatment``, ``intervention``, ``reference`` | what is set, to what, how |
| ``outcome`` | ``outcome`` | which outcome, at what aggregation, in what dimension |
| ``population`` | ``population`` | the target population and the strata defining it |
| ``window`` | ``window`` | the time window and its basis (per-period vs cumulative) |
| ``level`` | ``level`` | unit of analysis and the interference model |
| ``conditioning`` | ``conditioning`` | the strata it is conditional on; empty for marginal |
| ``dimension`` | ``dimension`` | derived from the others, asserted against the declaration |

``transfer_to`` compares facet by facet. For every differing facet it either
names the licensing ``Assumption`` (state ``unverified`` until something
checks it — the selection-diagram verdict in Phase 2, the surface in Phase
6) or returns ``blocked`` with a reason. No facet passes silently. The plan
is structural here; the correction operators arrive in ``calibrate``.

Licensing table (charter, with review B1/B2/B3 applied):

| differs | assumption | challenged by | derived from |
|---|---|---|---|
| population | S-admissibility given Z | overlap; moderator interaction | graph (Phase 2) |
| window | dynamics stationary; carryover contained | half-life vs window | surface |
| intervention (dose) | surface correct between the doses | curvature; chord-vs-marginal | surface |
| intervention (version) | version-irrelevance | — | asserted |
| level (unit) | linear aggregation or an aggregation model | Jensen gap | surface |
| level (interference) | **blocked** without a declared interference model | — | — |
| outcome (same dim) | commensurability / surrogate validity | not falsifiable | asserted |
| outcome (diff dim) | **blocked** | — | — |
| conditioning, collapsible | target strata weights known | strata provenance | population |
| conditioning, non-collapsible | **blocked** unless re-expressed as a contrast | — | — |
| quantity | **blocked** — a different functional is a different quantity | — | — |
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import model_validator

from axiom.core.dimensions import Dimension, DimensionError, dimensionless
from axiom.core.entities import (
    EntityName,
    Intervention,
    LatentSelection,
    Outcome,
    Population,
    TimeWindow,
    Treatment,
)
from axiom.core.result import Blocked, NonEmptyStr
from axiom.core.spec import Spec
from axiom.core.verdict import Assumption, LedgerLine, Status

__all__ = [
    "FACETS",
    "Estimand",
    "Facet",
    "FacetDiff",
    "Level",
    "Quantity",
    "QuantityKind",
    "TransferPlan",
    "derived_dimension",
]

Facet = Literal[
    "quantity",
    "intervention",
    "outcome",
    "population",
    "window",
    "level",
    "conditioning",
    "dimension",
]
FACETS: tuple[Facet, ...] = (
    "quantity",
    "intervention",
    "outcome",
    "population",
    "window",
    "level",
    "conditioning",
    "dimension",
)

QuantityKind = Literal["contrast", "marginal", "ratio", "elasticity", "area"]
COLLAPSIBLE: frozenset[str] = frozenset({"contrast", "marginal", "area"})


class Quantity(Spec):
    """The functional applied to the response.

    Every functional is taken over the *aggregated* outcome ``agg(Y)`` and
    dose ``agg(X)`` that the ``window`` (basis) and ``level`` facets define,
    so the same words mean the same thing for an arm, a unit, and a panel
    (decision 0002.20):

    * ``contrast`` — ``agg Y(iv) − agg Y(ref)``; dimension of the outcome.
      Scales with the window basis (a cumulative contrast is ``T`` times a
      per-period one).
    * ``marginal`` — ``d agg Y / d dose`` at ``iv``; outcome per dose.
    * ``ratio`` — ``(agg Y(iv) − agg Y(ref)) / (agg X(iv) − agg X(ref))``;
      outcome per unit of dose, *invariant* to the basis because numerator
      and denominator aggregate identically.
    * ``elasticity`` — ``marginal · agg X / agg Y`` at ``iv``; dimensionless,
      basis-invariant.
    * ``area`` — ``∫ agg Y d(dose)`` along the dose path from ``ref`` to
      ``iv``; outcome × dose; scales with the basis.
    """

    kind: QuantityKind
    scale: Literal["natural", "log"] = "natural"


class Level(Spec):
    """Unit of analysis and interference model (review B2: SUTVA-2)."""

    unit: Literal["individual", "cluster", "aggregate"]
    interference: Literal["none", "within_cluster", "declared"] = "none"
    interference_model: str = ""


def derived_dimension(kind: QuantityKind, outcome: Dimension, dose: Dimension) -> Dimension:
    match kind:
        case "contrast":
            return outcome
        case "marginal" | "ratio":
            return outcome / dose
        case "elasticity":
            return dimensionless()
        case "area":
            return outcome * dose
    raise ValueError(f"unknown quantity kind {kind!r}")  # pragma: no cover


class Estimand(Spec):
    """A named, versioned, content-hashed counterfactual quantity. All facets required."""

    name: EntityName
    quantity: Quantity
    treatment: Treatment
    intervention: Intervention
    reference: Intervention | None = None
    outcome: Outcome
    population: Population
    window: TimeWindow
    level: Level
    conditioning: tuple[str, ...] = ()
    dimension: Dimension
    description: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> Estimand:
        if self.treatment.name not in self.intervention.doses:
            raise ValueError(
                f"intervention does not set treatment {self.treatment.name!r}; "
                f"it sets {list(self.intervention.doses)}"
            )
        needs_ref = self.quantity.kind in ("contrast", "ratio", "area")
        if needs_ref and self.reference is None:
            raise ValueError(f"a {self.quantity.kind!r} needs a reference intervention")
        if self.reference is not None and self.treatment.name not in self.reference.doses:
            raise ValueError(f"reference does not set treatment {self.treatment.name!r}")
        if self.treatment.dimension is None or self.outcome.dimension is None:
            raise DimensionError(
                f"estimand {self.name!r}: treatment and outcome must carry dimensions"
            )
        got = derived_dimension(
            self.quantity.kind, self.outcome.dimension, self.treatment.dimension
        )
        if got != self.dimension:
            raise DimensionError(
                f"estimand {self.name!r}: declared dimension {self.dimension} but a "
                f"{self.quantity.kind} of {self.outcome.dimension} over {self.treatment.dimension} "
                f"derives to {got}"
            )
        if tuple(sorted(set(self.conditioning))) != self.conditioning:
            raise ValueError("conditioning strata must be sorted and unique")
        return self

    # -- facets ----------------------------------------------------------------

    def facet(self, name: Facet) -> tuple[Spec | None, ...] | tuple[str, ...] | Dimension:
        """The value of one facet, as the thing compared by ``transfer_to``."""
        match name:
            case "quantity":
                return (self.quantity,)
            case "intervention":
                return (self.treatment, self.intervention, self.reference)
            case "outcome":
                return (self.outcome,)
            case "population":
                return (self.population,)
            case "window":
                return (self.window,)
            case "level":
                return (self.level,)
            case "conditioning":
                return self.conditioning
            case "dimension":
                return self.dimension
        raise KeyError(name)  # pragma: no cover

    def differing_facets(self, other: Estimand) -> tuple[Facet, ...]:
        return tuple(f for f in FACETS if self.facet(f) != other.facet(f))

    def transfer_to(self, target: Estimand) -> TransferPlan:
        """A typed plan for reading this estimand as ``target``. See the module docstring."""
        diffs = self.differing_facets(target)
        entries: list[FacetDiff] = []
        for facet in diffs:
            entries.append(_RULES[facet](self, target))
        blocked = tuple(e for e in entries if e.blocked is not None)
        status: Status
        if not diffs:
            status = "identified"
        elif blocked:
            status = "blocked"
        else:
            status = "downgraded"
        assumptions = tuple(a for e in entries for a in e.assumptions)
        reason = "; ".join(
            f"{e.facet}: {e.blocked.reason}" for e in entries if e.blocked is not None
        )
        lines = tuple(
            LedgerLine(
                kind=f"facet:{e.facet}",
                statement=e.statement,
                assumption=e.assumptions[0] if e.assumptions else None,
                detail={
                    "status": "blocked" if e.blocked is not None else "assumed",
                    "facet": e.facet,
                },
                source=self.content_hash(),
                target=target.content_hash(),
            )
            for e in entries
        )
        return TransferPlan(
            status=status,
            source=self.content_hash(),
            target=target.content_hash(),
            differing=diffs,
            entries=tuple(entries),
            assumptions=assumptions,
            corrections=tuple(c for e in entries for c in e.corrections),
            ledger_lines=lines,
            reason=reason,
        )


class FacetDiff(Spec):
    """What one differing facet requires: assumptions, or a block, plus any correction."""

    facet: Facet
    statement: NonEmptyStr
    assumptions: tuple[Assumption, ...] = ()
    blocked: Blocked | None = None
    corrections: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _one_or_the_other(self) -> FacetDiff:
        if self.blocked is None and not self.assumptions:
            raise ValueError(
                f"facet {self.facet!r} differs but names neither an assumption nor a block"
            )
        return self


class TransferPlan(Spec):
    """The typed diff between two estimands and what licenses reading one as the other.

    ``status`` uses ``Verdict``'s vocabulary. Every differing facet appears in
    exactly one ledger line.
    """

    status: Status
    source: str
    target: str
    differing: tuple[Facet, ...]
    entries: tuple[FacetDiff, ...]
    assumptions: tuple[Assumption, ...]
    corrections: tuple[str, ...] = ()
    ledger_lines: tuple[LedgerLine, ...]
    reason: str = ""

    @property
    def licensed(self) -> bool:
        return self.status in ("identified", "downgraded")

    def entry(self, facet: Facet) -> FacetDiff:
        for e in self.entries:
            if e.facet == facet:
                return e
        raise KeyError(f"facet {facet!r} does not differ in this plan")


# -- the licensing rules ---------------------------------------------------------------

Rule = Callable[[Estimand, Estimand], FacetDiff]


def _quantity(s: Estimand, t: Estimand) -> FacetDiff:
    return FacetDiff(
        facet="quantity",
        statement=f"functional differs: {s.quantity.kind} vs {t.quantity.kind}",
        blocked=Blocked(reason="a different functional is a different quantity; re-express first"),
    )


def _intervention(s: Estimand, t: Estimand) -> FacetDiff:
    if s.treatment != t.treatment:
        return FacetDiff(
            facet="intervention",
            statement=f"treatment differs: {s.treatment.name} vs {t.treatment.name}",
            blocked=Blocked(reason="a different treatment is a different quantity"),
        )
    if s.intervention.mode != t.intervention.mode:
        return FacetDiff(
            facet="intervention",
            statement=f"intervention mode differs: {s.intervention.mode} vs {t.intervention.mode}",
            blocked=Blocked(
                reason=(
                    "set/scale/shift interventions are not interconvertible "
                    "without the observed dose path"
                )
            ),
        )
    assumptions: list[Assumption] = []
    corrections: list[str] = []
    if s.intervention.doses != t.intervention.doses or s.reference != t.reference:
        assumptions.append(
            Assumption(
                name="surface_correct_between_doses",
                facet="intervention",
                statement=(
                    "the response surface is correct between the source and target dose levels"
                ),
                challenged_by="curvature; the chord-versus-marginal correction",
                detail={
                    "source_doses": str(s.intervention.doses),
                    "target_doses": str(t.intervention.doses),
                },
            )
        )
        corrections.append("chord_to_marginal")
    if s.intervention.version != t.intervention.version:
        assumptions.append(
            Assumption(
                name="version_irrelevance",
                facet="intervention",
                statement=(
                    f"treatment version {s.intervention.version!r} and "
                    f"{t.intervention.version!r} have the same effect"
                ),
                challenged_by="not falsifiable from the data; must be asserted",
                detail={
                    "source_version": s.intervention.version,
                    "target_version": t.intervention.version,
                },
            )
        )
    if s.intervention.window != t.intervention.window:
        assumptions.append(
            Assumption(
                name="intervention_timing_irrelevant",
                facet="intervention",
                statement=(
                    "the effect does not depend on when within the window the dose is applied"
                ),
                challenged_by="carryover shape",
            )
        )
    return FacetDiff(
        facet="intervention",
        statement="intervention differs in dose, version, or timing",
        assumptions=tuple(assumptions),
        corrections=tuple(corrections),
    )


def _outcome(s: Estimand, t: Estimand) -> FacetDiff:
    if s.outcome.dimension != t.outcome.dimension:
        return FacetDiff(
            facet="outcome",
            statement=f"outcome dimension differs: {s.outcome.dimension} vs {t.outcome.dimension}",
            blocked=Blocked(reason="outcomes of different dimension are not commensurable"),
        )
    assumptions = [
        Assumption(
            name="outcome_commensurability",
            facet="outcome",
            statement=(
                f"{s.outcome.name} is commensurable with (or a valid surrogate for) "
                f"{t.outcome.name}"
            ),
            challenged_by="usually not falsifiable; must be asserted",
        )
    ]
    if s.outcome.aggregation != t.outcome.aggregation:
        assumptions.append(
            Assumption(
                name="aggregation_commensurable",
                facet="outcome",
                statement=(
                    f"a {s.outcome.aggregation} outcome can be read as a {t.outcome.aggregation}"
                ),
                challenged_by="the number of units and periods aggregated over",
            )
        )
    if s.outcome.unit != t.outcome.unit:
        assumptions.append(
            Assumption(
                name="outcome_unit_conversion_registered",
                facet="outcome",
                statement=(
                    f"a conversion from {s.outcome.unit} to {t.outcome.unit} is registered "
                    "in the unit system"
                ),
                challenged_by="UnitSystem.factor raising",
            )
        )
    return FacetDiff(facet="outcome", statement="outcome differs", assumptions=tuple(assumptions))


def _s_admissibility(s: Estimand, t: Estimand) -> Assumption:
    return Assumption(
        name="s_admissibility",
        facet="population",
        statement=(
            f"the effect transports from {s.population.name} to {t.population.name} "
            "given an S-admissible set"
        ),
        challenged_by="overlap; moderator interaction in meta",
        detail={"verdict": "pending selection-diagram verdict (identify.transport)"},
    )


def _latent_homogeneity(latent: LatentSelection, *, into: bool) -> Assumption:
    """The condition that bridges a latent subpopulation and the population around it.

    Not falsifiable, and it is worth being exact about why: a never-taker is by
    construction never exposed under this instrument, so no design using it ever
    observes the effect being assumed equal. Only a *different* instrument can,
    and that recruits a different subpopulation.
    """
    direction = "outward to the whole population" if into else "inward to the subpopulation"
    return Assumption(
        name="latent_type_homogeneity",
        facet="population",
        statement=(
            f"the effect on the {latent.kind}s of {latent.instrument} is the effect on every "
            f"unit, so the quantity carries {direction}"
        ),
        challenged_by=(
            "an instrument of different strength giving a different effect, which recruits a "
            "different subpopulation and so is evidence about heterogeneity across types; "
            "never the units themselves, who are not observed under both exposures"
        ),
        detail={
            "kind": latent.kind,
            "instrument": latent.instrument,
            "exposure": latent.exposure,
            "share": "" if latent.share is None else f"{latent.share:.6g}",
        },
    )


def _population(s: Estimand, t: Estimand) -> FacetDiff:
    source, target = s.population.latent, t.population.latent
    if source is not None and target is not None and not source.same_stratum(target):
        return FacetDiff(
            facet="population",
            statement=(f"two latent subpopulations differ: {source} vs {target}"),
            blocked=Blocked(
                reason=(
                    "a latent subpopulation is defined by which units respond to which "
                    f"instrument, so the {source.kind}s of {source.instrument} and the "
                    f"{target.kind}s of {target.instrument} are different sets of units; no "
                    "assumption over observables bridges them, because no observable "
                    "distinguishes a member of either"
                )
            ),
        )
    assumptions: list[Assumption] = []
    if (source is None) != (target is None):
        latent = source if source is not None else target
        if latent is None:  # unreachable: exactly one of the two is set here
            raise RuntimeError("a one-sided latent difference must have a latent side")
        assumptions.append(_latent_homogeneity(latent, into=source is not None))
    if s.population.name != t.population.name or s.population.strata != t.population.strata:
        assumptions.append(_s_admissibility(s, t))
    if not assumptions:  # only the description or the stratum's share differs
        assumptions.append(_s_admissibility(s, t))
    return FacetDiff(
        facet="population",
        statement=f"population differs: {s.population.name} vs {t.population.name}",
        assumptions=tuple(assumptions),
    )


def _window(s: Estimand, t: Estimand) -> FacetDiff:
    assumptions = [
        Assumption(
            name="stationary_dynamics",
            facet="window",
            statement="the response dynamics are stationary across the two windows",
            challenged_by="half-life against window length",
            detail={
                "source": f"[{s.window.start},{s.window.stop})",
                "target": f"[{t.window.start},{t.window.stop})",
            },
        ),
        Assumption(
            name="carryover_contained",
            facet="window",
            statement="carryover from doses in the window is contained within it",
            challenged_by="half-life against window length",
        ),
    ]
    corrections: list[str] = []
    if s.window.basis != t.window.basis and s.quantity.kind in ("contrast", "area"):
        # ratio, marginal and elasticity aggregate numerator and denominator alike and are
        # basis-invariant (Quantity docstring, decision 0002.20); only level-type quantities rescale
        corrections.append(
            "per_period_to_cumulative"
            if t.window.basis == "cumulative"
            else "cumulative_to_per_period"
        )
    return FacetDiff(
        facet="window",
        statement="time window or basis differs",
        assumptions=tuple(assumptions),
        corrections=tuple(corrections),
    )


def _level(s: Estimand, t: Estimand) -> FacetDiff:
    if s.level.interference != t.level.interference and t.level.interference != "declared":
        return FacetDiff(
            facet="level",
            statement=f"interference differs: {s.level.interference} vs {t.level.interference}",
            blocked=Blocked(
                reason="a differing interference structure needs a declared interference model"
            ),
        )
    assumptions = []
    if s.level.unit != t.level.unit:
        assumptions.append(
            Assumption(
                name="linear_aggregation",
                facet="level",
                statement=(
                    f"the effect aggregates linearly from {s.level.unit} to {t.level.unit} "
                    "level, or an explicit aggregation model is supplied"
                ),
                challenged_by="the Jensen gap under a nonlinear response",
            )
        )
    if s.level.interference != t.level.interference:
        assumptions.append(
            Assumption(
                name="declared_interference_model",
                facet="level",
                statement=(
                    "interference is modelled as declared: "
                    f"{t.level.interference_model or 'unspecified'}"
                ),
                challenged_by="spillover diagnostics",
                state="asserted" if t.level.interference_model else "unverified",
            )
        )
    if not assumptions:  # only interference_model text differs
        assumptions.append(
            Assumption(
                name="declared_interference_model",
                facet="level",
                statement="the two declared interference models are equivalent",
                challenged_by="spillover diagnostics",
            )
        )
    return FacetDiff(facet="level", statement="level differs", assumptions=tuple(assumptions))


def _conditioning(s: Estimand, t: Estimand) -> FacetDiff:
    if s.quantity.kind not in COLLAPSIBLE or t.quantity.kind not in COLLAPSIBLE:
        return FacetDiff(
            facet="conditioning",
            statement="conditioning differs for a non-collapsible quantity",
            blocked=Blocked(
                reason=(
                    f"a {s.quantity.kind} does not collapse across strata; "
                    "re-express as a contrast first"
                )
            ),
        )
    collapsed = sorted(set(s.conditioning) ^ set(t.conditioning))
    known = all(c in t.population.strata for c in collapsed)
    return FacetDiff(
        facet="conditioning",
        statement=f"conditioning strata differ: {collapsed}",
        assumptions=(
            Assumption(
                name="target_strata_weights_known",
                facet="conditioning",
                statement=f"the target population's weights over {collapsed} are known",
                challenged_by="the provenance of the strata distribution",
                state="satisfied" if known and collapsed else "unverified",
                detail={"strata": ",".join(collapsed)},
            ),
        ),
        corrections=("standardize_over_strata",),
    )


def _dimension(s: Estimand, t: Estimand) -> FacetDiff:
    return FacetDiff(
        facet="dimension",
        statement=f"dimension differs: {s.dimension} vs {t.dimension}",
        blocked=Blocked(reason="quantities of different dimension are different quantities"),
    )


_RULES: dict[Facet, Rule] = {
    "quantity": _quantity,
    "intervention": _intervention,
    "outcome": _outcome,
    "population": _population,
    "window": _window,
    "level": _level,
    "conditioning": _conditioning,
    "dimension": _dimension,
}
assert set(_RULES) == set(FACETS)
