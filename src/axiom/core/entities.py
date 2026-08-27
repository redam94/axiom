"""The domain-general vocabulary. Every entity carries a ``Dimension``.

| parent (marketing)   | axiom        |
|----------------------|--------------|
| channel              | ``Treatment`` |
| spend / impressions  | ``Dose``      |
| geo / DMA            | ``Unit``      |
| KPI / sales          | ``Outcome``   |
| control variable     | ``Covariate`` |

The five entity specs are independent classes that share field types and a
validator by composition; ``Entity`` is the *protocol* they all satisfy, for
functions that accept any of them. A ``Dose`` also carries a ``numeraire`` so
cost-per-outcome arithmetic is checked rather than conventional.
``Intervention`` and ``TimeWindow`` live here too because the
``SupportsIntervention`` protocol needs them and ``core`` imports nothing
above itself.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import AfterValidator, field_validator, model_validator

from axiom.core.dimensions import Dimension, dimensionless
from axiom.core.spec import Spec

__all__ = [
    "Covariate",
    "Dose",
    "Entity",
    "EntityName",
    "Intervention",
    "Outcome",
    "Population",
    "TimeWindow",
    "Treatment",
    "Unit",
    "UndimensionedWarning",
    "dimension_of",
]


class UndimensionedWarning(UserWarning):
    """An entity was declared without a dimension and is treated as dimensionless (D6)."""


def _identifier_like(v: str) -> str:
    if not v or not v.replace("-", "_").replace(".", "_").isidentifier():
        raise ValueError(f"entity name {v!r} must be identifier-like")
    return v


EntityName = Annotated[str, AfterValidator(_identifier_like)]
"""A non-empty, identifier-like name (``-`` and ``.`` allowed)."""


@runtime_checkable
class Entity(Protocol):
    """What every entity spec provides: a name, an optional dimension and unit."""

    @property
    def name(self) -> str: ...
    @property
    def dimension(self) -> Dimension | None: ...
    @property
    def unit(self) -> str | None: ...
    @property
    def description(self) -> str: ...


def dimension_of(entity: Entity) -> Dimension:
    """The entity's dimension; dimensionless when none was declared (D6)."""
    return entity.dimension if entity.dimension is not None else dimensionless()


def _warn_undimensioned[E: Entity](entity: E) -> E:
    """Shared ``model_validator``: user code may omit a dimension, but it is said out loud.

    Gate 10 still rejects anything *shipped* in ``src/axiom`` that is undimensioned.
    """
    if entity.dimension is None:
        warnings.warn(
            f"{type(entity).__name__} {entity.name!r} declared without a dimension; "
            "treating as dimensionless",
            UndimensionedWarning,
            stacklevel=3,
        )
    return entity


class Treatment(Spec):
    """The thing you can intervene on. Its dimension is the dose's dimension."""

    name: EntityName
    dimension: Dimension | None = None
    unit: str | None = None
    description: str = ""

    _check = model_validator(mode="after")(_warn_undimensioned)
    dim = property(dimension_of)


class Dose(Spec):
    """The magnitude of an intervention on a treatment, with units and a numeraire.

    ``numeraire`` is the unit in which this dose is *costed* (often the same
    as ``unit`` for a currency dose, a price per unit otherwise). It is what
    lets value-of-information arithmetic be dimension-checked.
    """

    name: EntityName
    dimension: Dimension | None = None
    unit: str | None = None
    description: str = ""
    numeraire: str | None = None

    _check = model_validator(mode="after")(_warn_undimensioned)
    dim = property(dimension_of)


class Unit(Spec):
    """The randomization / observation unit: a plot, a patient, a region.

    Dimension ``entity`` by convention; a count of units is ``entity``.
    """

    name: EntityName
    dimension: Dimension | None = None
    unit: str | None = None
    description: str = ""
    kind: Literal["individual", "cluster", "aggregate"] = "individual"

    _check = model_validator(mode="after")(_warn_undimensioned)
    dim = property(dimension_of)


class Outcome(Spec):
    """What you are trying to move."""

    name: EntityName
    dimension: Dimension | None = None
    unit: str | None = None
    description: str = ""
    aggregation: Literal["sum", "mean", "rate"] = "sum"

    _check = model_validator(mode="after")(_warn_undimensioned)
    dim = property(dimension_of)


class Covariate(Spec):
    """Measured, not intervened on."""

    name: EntityName
    dimension: Dimension | None = None
    unit: str | None = None
    description: str = ""

    _check = model_validator(mode="after")(_warn_undimensioned)
    dim = property(dimension_of)


class LatentSelection(Spec):
    """A subpopulation defined by a response nobody observes: sized, never listed.

    The compliers of an encouragement design are the standard case. They are a
    real set of units with a real average effect, their share is identified
    from two exposure rates, and no covariate distinguishes a member from a
    never-taker — you can say how many there are and never which ones. That
    breaks the usual description of a population as strata weights, which is
    why it gets its own field on ``Population`` rather than an entry in
    ``strata``.

    Identity is ``(kind, instrument, exposure)`` and **not** ``share``. A
    complier is a unit whose ``exposure`` responds to ``instrument``, so
    changing the instrument changes the set: the compliers of a letter and the
    compliers of a phone call are different people, and their effects are
    different quantities however similar the two designs look. ``share`` is a
    property of the population the stratum was taken from and varies across
    them; it is carried for reading, not for identity.

    ``kind`` is free text with a convention: ``"complier"``, ``"always_taker"``,
    ``"never_taker"`` for an instrumental design, and any principal stratum
    (Frangakis and Rubin) otherwise — ``"survivor"`` for truncation by death.
    """

    kind: EntityName
    instrument: EntityName
    exposure: EntityName
    share: float | None = None
    description: str = ""

    @model_validator(mode="after")
    def _share_is_a_share(self) -> LatentSelection:
        if self.share is not None and not 0.0 <= self.share <= 1.0:
            raise ValueError(f"share must be in [0, 1], got {self.share}")
        return self

    def same_stratum(self, other: LatentSelection) -> bool:
        """Whether two selections pick out the same set of units, share aside."""
        return (self.kind, self.instrument, self.exposure) == (
            other.kind,
            other.instrument,
            other.exposure,
        )

    def __str__(self) -> str:
        size = "" if self.share is None else f", {self.share:.1%}"
        return f"{self.kind}s of {self.instrument} on {self.exposure}{size}"


class Population(Spec):
    """A target population: a name and, where known, the strata weights that define it.

    ``strata`` maps a covariate name to ``{level: weight}`` with weights
    summing to one. Transfer across populations is licensed by S-admissibility
    (graph) and, for conditional-to-marginal moves, by these weights being
    known (review B1).

    ``latent`` narrows the population to a subpopulation no covariate can
    describe — the compliers of an instrument, the survivors of a truncation.
    A latent population is a different population from the one it sits inside,
    and ``Estimand.transfer_to`` treats it as one: see note 0032.
    """

    name: EntityName
    description: str = ""
    strata: dict[str, dict[str, float]] = {}
    latent: LatentSelection | None = None

    @property
    def is_latent(self) -> bool:
        """True when the population is a subpopulation that can be sized but not listed."""
        return self.latent is not None

    @field_validator("strata")
    @classmethod
    def _weights_sum_to_one(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        for cov, weights in v.items():
            if not weights:
                raise ValueError(f"strata for {cov!r} are empty")
            total = sum(weights.values())
            if any(w < 0 for w in weights.values()) or abs(total - 1.0) > 1e-9:
                raise ValueError(f"strata weights for {cov!r} must be non-negative and sum to 1")
        return v


class TimeWindow(Spec):
    """A half-open index window ``[start, stop)`` on the panel's time axis, with a basis.

    ``basis`` says whether a quantity over the window is a per-period rate or
    a cumulative total — the per-period-versus-cumulative bug class.
    """

    start: int
    stop: int
    basis: Literal["per_period", "cumulative"] = "cumulative"

    @model_validator(mode="after")
    def _ordered(self) -> TimeWindow:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(
                f"window must satisfy 0 <= start < stop, got [{self.start}, {self.stop})"
            )
        return self

    @property
    def length(self) -> int:
        return self.stop - self.start


class Intervention(Spec):
    """What is set, to what value, how, over what support.

    ``doses`` maps treatment name to the dose level. ``mode`` distinguishes
    setting a level (``"set"``), scaling the observed dose (``"scale"``), and
    shifting it additively (``"shift"``). ``version`` records *how* the
    treatment is delivered (review B2: SUTVA-1); two interventions with the
    same dose and a different version are different interventions.
    """

    doses: dict[str, float]
    mode: Literal["set", "scale", "shift"] = "set"
    version: str = "unspecified"
    window: TimeWindow | None = None

    @field_validator("doses")
    @classmethod
    def _nonempty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("an intervention must set at least one treatment")
        return dict(sorted(v.items()))

    @property
    def treatments(self) -> tuple[str, ...]:
        return tuple(self.doses)
