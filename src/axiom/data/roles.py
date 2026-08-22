"""Column roles. A column adopts the dimension of its role (review D3).

A ``RoleMap`` says which column is the observation unit, which is time, and
which entity — ``Treatment``, ``Outcome``, ``Covariate`` — each remaining
column is. Units are declared once here and nowhere else in an analysis.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from axiom.core.dimensions import Dimension
from axiom.core.entities import Covariate, Entity, Outcome, Treatment, dimension_of
from axiom.core.spec import Spec

__all__ = ["RoleKind", "RoleMap"]

RoleKind = Literal["unit", "time", "treatment", "outcome", "covariate"]


class RoleMap(Spec):
    """Column → role for a panel.

    ``treatments`` / ``covariates`` map column name to entity; ``outcome`` is
    the single outcome column. Entity ``name`` need not equal the column
    name; the column is the storage, the entity is the meaning.
    """

    unit: str
    time: str
    outcome: tuple[str, Outcome]
    treatments: dict[str, Treatment] = {}
    covariates: dict[str, Covariate] = {}

    @model_validator(mode="after")
    def _columns_distinct(self) -> RoleMap:
        cols = [self.unit, self.time, self.outcome[0], *self.treatments, *self.covariates]
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        if dupes:
            raise ValueError(f"columns assigned more than one role: {dupes}")
        entities: list[Entity] = [*self.treatments.values(), *self.covariates.values()]
        names = [e.name for e in entities]
        names.append(self.outcome[1].name)
        dupe_names = sorted({n for n in names if names.count(n) > 1})
        if dupe_names:
            raise ValueError(f"entity names used more than once: {dupe_names}")
        return self

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.unit, self.time, self.outcome[0], *self.treatments, *self.covariates)

    def kind_of(self, column: str) -> RoleKind:
        if column == self.unit:
            return "unit"
        if column == self.time:
            return "time"
        if column == self.outcome[0]:
            return "outcome"
        if column in self.treatments:
            return "treatment"
        if column in self.covariates:
            return "covariate"
        raise KeyError(f"column {column!r} has no role")

    def dimension_of(self, column: str) -> Dimension:
        """The dimension a column carries, by its role. Unit and time columns are indices."""
        kind = self.kind_of(column)
        if kind == "outcome":
            return dimension_of(self.outcome[1])
        if kind == "treatment":
            return dimension_of(self.treatments[column])
        if kind == "covariate":
            return dimension_of(self.covariates[column])
        raise KeyError(f"column {column!r} is an index ({kind}), not a measured quantity")

    def unit_of(self, column: str) -> str | None:
        kind = self.kind_of(column)
        if kind == "outcome":
            return self.outcome[1].unit
        if kind == "treatment":
            return self.treatments[column].unit
        if kind == "covariate":
            return self.covariates[column].unit
        return None

    @property
    def measured(self) -> tuple[str, ...]:
        """Every column that is a quantity (not an index)."""
        return (self.outcome[0], *self.treatments, *self.covariates)
