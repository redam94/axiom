"""``VariableBuilder``: name, kind, dimension, unit → one of the five entity specs.

Ported from the parent's ``builders/variable.py`` (ledger: PORT). The
parent built ``Channel`` / ``Control`` / ``KPI`` configurations; here the
kinds are the domain-general entities — ``Treatment``, ``Outcome``,
``Covariate``, ``Dose``, ``Unit`` — and the marketing names live in
``adapters``. An entity built without a dimension is dimensionless and says
so (``UndimensionedWarning``), exactly as constructing the entity directly
would.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from axiom.build.base import BuildError, Fields
from axiom.core import Covariate, Dimension, Dose, Outcome, Treatment, Unit

__all__ = ["EntitySpec", "VariableBuilder", "VariableKind"]

VariableKind = Literal["treatment", "outcome", "covariate", "dose", "unit"]
EntitySpec = Treatment | Outcome | Covariate | Dose | Unit

_CLASSES: dict[str, type[EntitySpec]] = {
    "treatment": Treatment,
    "outcome": Outcome,
    "covariate": Covariate,
    "dose": Dose,
    "unit": Unit,
}


@dataclass(frozen=True)
class VariableBuilder:
    """``VariableBuilder().treatment("x").dimension(D.currency).unit("USD").build()``.

    One of ``treatment`` / ``outcome`` / ``covariate`` / ``dose`` / ``unit``
    fixes the kind and the name; ``dimension``, ``unit``, ``describe`` apply
    to every kind; ``numeraire`` is for a dose, ``aggregation`` for an
    outcome, ``cluster`` / ``individual`` / ``aggregate`` for a unit.
    """

    fields: Fields = Fields()

    def with_(self, **updates: Any) -> VariableBuilder:
        return replace(self, fields=self.fields.with_(**updates))

    # -- kind + name --------------------------------------------------------------------

    def kind(self, kind: VariableKind) -> VariableBuilder:
        if kind not in _CLASSES:
            raise BuildError(f"unknown variable kind {kind!r}; known: {sorted(_CLASSES)}")
        return self.with_(kind=kind)

    def name(self, name: str) -> VariableBuilder:
        return self.with_(name=name)

    def treatment(self, name: str) -> VariableBuilder:
        return self.kind("treatment").name(name)

    def outcome(self, name: str) -> VariableBuilder:
        return self.kind("outcome").name(name)

    def covariate(self, name: str) -> VariableBuilder:
        return self.kind("covariate").name(name)

    def dose(self, name: str) -> VariableBuilder:
        return self.kind("dose").name(name)

    def unit(self, name: str) -> VariableBuilder:
        """The observation / randomization unit entity (not the measurement unit string)."""
        return self.kind("unit").name(name)

    # -- shared attributes --------------------------------------------------------------

    def dimension(self, dimension: Dimension) -> VariableBuilder:
        return self.with_(dimension=dimension)

    def measured_in(self, unit: str) -> VariableBuilder:
        """The measurement unit string (``"USD"``, ``"mg"``)."""
        return self.with_(unit=unit)

    def describe(self, description: str) -> VariableBuilder:
        return self.with_(description=description)

    # -- kind-specific ------------------------------------------------------------------

    def numeraire(self, numeraire: str) -> VariableBuilder:
        return self.with_(numeraire=numeraire)

    def aggregation(self, aggregation: Literal["sum", "mean", "rate"]) -> VariableBuilder:
        return self.with_(aggregation=aggregation)

    def cluster(self) -> VariableBuilder:
        return self.with_(unit_kind="cluster")

    def individual(self) -> VariableBuilder:
        return self.with_(unit_kind="individual")

    def aggregate(self) -> VariableBuilder:
        return self.with_(unit_kind="aggregate")

    # -- build --------------------------------------------------------------------------

    def build(self) -> EntitySpec:
        self.fields.require("kind", "name", builder="VariableBuilder")
        kind: str = self.fields.get("kind")
        payload: dict[str, Any] = {
            "name": self.fields.get("name"),
            "dimension": self.fields.get("dimension"),
            "unit": self.fields.get("unit"),
            "description": self.fields.get("description", ""),
        }
        specific = {"numeraire": "dose", "aggregation": "outcome", "unit_kind": "unit"}
        for key, owner in specific.items():
            if key in self.fields and kind != owner:
                raise BuildError(f"VariableBuilder: {key!r} applies to a {owner}, not a {kind}")
        if kind == "dose" and "numeraire" in self.fields:
            payload["numeraire"] = self.fields.get("numeraire")
        if kind == "outcome" and "aggregation" in self.fields:
            payload["aggregation"] = self.fields.get("aggregation")
        if kind == "unit" and "unit_kind" in self.fields:
            payload["kind"] = self.fields.get("unit_kind")
        return _CLASSES[kind](**payload)
