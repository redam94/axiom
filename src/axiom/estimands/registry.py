"""``EstimandRegistry``: a named collection of estimands, and the standard library.

A registry is a plain class over a dict (composition, 0002.13). It keeps one
``Estimand`` per name, reports each one's content hash — the identity a
fitted producer records in ``declared_estimands`` — and round-trips as a
tuple of specs, so an analysis manifest can store it without a new ``Spec``.

``standard_estimands`` builds the domain-general library for one treatment:
a contrast, a marginal, an average-response ratio, an elasticity, and an
area under the response. The names and descriptions are free of marketing
vocabulary; return-on-dose presets belong to ``adapters`` (Phase 8), where
they are declared as these estimands under domain names.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence

from axiom.core.entities import Intervention, Outcome, Population, TimeWindow, Treatment
from axiom.estimands.spec import Estimand, Level, Quantity, QuantityKind, derived_dimension

__all__ = ["STANDARD_ESTIMAND_NAMES", "EstimandRegistry", "standard_estimands"]


class EstimandRegistry:
    """Estimands by name, with their content hashes.

    ``register`` refuses a second, *different* estimand under an existing
    name unless ``overwrite=True``; registering the identical spec again is a
    no-op that returns the same hash. Iteration order is registration order.
    """

    def __init__(self, estimands: Iterable[Estimand] = ()) -> None:
        self._by_name: dict[str, Estimand] = {}
        for e in estimands:
            self.register(e)

    # -- mutation ---------------------------------------------------------------

    def register(self, estimand: Estimand, *, overwrite: bool = False) -> str:
        """Add ``estimand`` and return its content hash."""
        existing = self._by_name.get(estimand.name)
        if existing is not None and existing != estimand and not overwrite:
            raise ValueError(
                f"estimand {estimand.name!r} is already registered with hash "
                f"{existing.content_hash()[:12]}…; pass overwrite=True to replace it"
            )
        self._by_name[estimand.name] = estimand
        return estimand.content_hash()

    def remove(self, name: str) -> Estimand:
        """Drop and return the estimand under ``name``; ``KeyError`` if absent."""
        if name not in self._by_name:
            raise KeyError(f"no estimand named {name!r}; registered: {list(self._by_name)}")
        return self._by_name.pop(name)

    # -- lookup -----------------------------------------------------------------

    def get(self, name: str) -> Estimand:
        if name not in self._by_name:
            raise KeyError(f"no estimand named {name!r}; registered: {list(self._by_name)}")
        return self._by_name[name]

    def by_hash(self, content_hash: str) -> Estimand:
        """The estimand with this content hash; ``KeyError`` if none is registered."""
        for e in self._by_name.values():
            if e.content_hash() == content_hash:
                return e
        raise KeyError(f"no registered estimand has content hash {content_hash!r}")

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def hashes(self) -> dict[str, str]:
        """``{name: content_hash}`` for every registered estimand."""
        return {name: e.content_hash() for name, e in self._by_name.items()}

    def __iter__(self) -> Iterator[Estimand]:
        return iter(tuple(self._by_name.values()))

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __eq__(self, other: object) -> bool:
        """Equal iff the two registries hold equal specs *in the same registration order*.

        A registry round-trips as a tuple, so its identity includes order;
        compare ``hashes()`` for an order-free check.
        """
        if not isinstance(other, EstimandRegistry):
            return NotImplemented
        return self.to_specs() == other.to_specs()

    __hash__ = None  # type: ignore[assignment]  # mutable; not hashable

    def __repr__(self) -> str:
        return f"EstimandRegistry({list(self._by_name)})"

    # -- round trip -------------------------------------------------------------

    def to_specs(self) -> tuple[Estimand, ...]:
        """The registry as a tuple of ``Estimand`` specs, in registration order."""
        return tuple(self._by_name.values())

    @classmethod
    def from_specs(cls, specs: Sequence[Estimand]) -> EstimandRegistry:
        """Build a registry from specs; duplicate names raise unless the specs are equal."""
        return cls(specs)


# -- the standard library -----------------------------------------------------------------

STANDARD_ESTIMAND_NAMES: tuple[str, ...] = (
    "contrast_at_dose",
    "marginal_at_dose",
    "average_response_ratio",
    "elasticity_at_dose",
    "area_under_response",
)
"""Names produced by ``standard_estimands``, in registration order."""

_STANDARD_KINDS: dict[str, QuantityKind] = {
    "contrast_at_dose": "contrast",
    "marginal_at_dose": "marginal",
    "average_response_ratio": "ratio",
    "elasticity_at_dose": "elasticity",
    "area_under_response": "area",
}


def _describe(name: str, treatment: Treatment, outcome: Outcome, dose: float, ref: float) -> str:
    t, y = treatment.name, outcome.name
    match name:
        case "contrast_at_dose":
            return (
                f"expected {y} with {t} set to {dose:g} minus expected {y} with {t} set to "
                f"{ref:g}, aggregated over the window"
            )
        case "marginal_at_dose":
            return f"derivative of expected {y} with respect to the {t} dose, at dose {dose:g}"
        case "average_response_ratio":
            return (
                f"contrast in expected {y} between {t} doses {ref:g} and {dose:g}, divided by "
                f"the dose difference: the average response per unit dose over that interval"
            )
        case "elasticity_at_dose":
            return (
                f"proportional change in expected {y} per proportional change in the {t} dose, "
                f"at dose {dose:g}"
            )
        case "area_under_response":
            return f"integral of expected {y} over {t} doses from {ref:g} to {dose:g}"
    raise KeyError(name)  # pragma: no cover


def standard_estimands(
    treatment: Treatment,
    outcome: Outcome,
    population: Population,
    window: TimeWindow,
    level: Level,
    *,
    dose: float,
    reference_dose: float = 0.0,
    version: str = "unspecified",
) -> EstimandRegistry:
    """The five domain-general estimands for one treatment at one dose.

    Every estimand uses ``mode="set"`` interventions at ``dose`` (and, where
    a reference is needed, at ``reference_dose``), the same ``version``, and
    no conditioning. ``treatment`` and ``outcome`` must carry dimensions;
    each estimand's dimension is derived from them. Both doses must be
    finite and distinct. ``dose`` must be non-zero: the elasticity at a zero
    dose is degenerate (``0 · marginal / E[Y(0)]`` — ``nan`` wherever the
    response vanishes at zero), and the library is refused with
    ``ValueError`` rather than registering a quantity that cannot be
    realized; build the other four by hand when the dose of interest is
    zero. ``reference_dose`` may be zero (the default).
    """
    for label, given in (("dose", dose), ("reference_dose", reference_dose)):
        if not math.isfinite(given):
            raise ValueError(f"{label} must be finite, got {given!r}")
    if dose == reference_dose:
        raise ValueError(
            f"dose and reference_dose coincide at {dose:g}; the ratio and area estimands "
            "need a dose interval"
        )
    if dose == 0.0:
        raise ValueError(
            "dose=0 is refused: 'elasticity_at_dose' is degenerate at a zero dose "
            "(0 · marginal / E[Y(0)]); choose a non-zero dose, or declare the other "
            "four estimands by hand"
        )
    if treatment.dimension is None or outcome.dimension is None:
        raise ValueError(
            f"treatment {treatment.name!r} and outcome {outcome.name!r} must carry dimensions "
            "to derive an estimand's dimension"
        )
    at = Intervention(doses={treatment.name: float(dose)}, mode="set", version=version)
    ref = Intervention(doses={treatment.name: float(reference_dose)}, mode="set", version=version)
    registry = EstimandRegistry()
    for name in STANDARD_ESTIMAND_NAMES:
        kind = _STANDARD_KINDS[name]
        needs_ref = kind in ("contrast", "ratio", "area")
        registry.register(
            Estimand(
                name=name,
                quantity=Quantity(kind=kind),
                treatment=treatment,
                intervention=at,
                reference=ref if needs_ref else None,
                outcome=outcome,
                population=population,
                window=window,
                level=level,
                conditioning=(),
                dimension=derived_dimension(kind, outcome.dimension, treatment.dimension),
                description=_describe(name, treatment, outcome, dose, reference_dose),
            )
        )
    return registry
