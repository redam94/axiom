"""The record a study contributes to a pool, the corpus that holds them, and the catalog
of poolable quantities.

Ported from the parent's ``benchmarks/schema.py`` (``ContributionRecord``,
``LiftStudyRecord``, ``StudyForm``, ``Provenance``) and ``benchmarks/estimands.py``
(the catalog), ledger rows ``meta/schema.py`` (PORT). The parent pooled
*contribution-shaped* marketing records; here a ``StudyRecord`` is one
estimate of one dimensionless quantity with its standard error and its
provenance, and the catalog names the quantities that may be pooled at all.

Two invariants the schema enforces, because every downstream number depends
on them (roadmap criterion 5):

* **Pooling is over dimensionless quantities.** A record whose ``dimension``
  is not dimensionless must declare the ``unit_scale`` it is reported on;
  ``meta.ingest.normalize`` then refuses it unless a licensed
  ``TransferPlan`` is supplied for that study. A dimensioned record with no
  declared scale cannot be constructed.
* **Provenance is part of the record.** ``read`` says whether the number is a
  model read or an experimental read; the provenance bias term in
  ``meta.pool`` is identified only by contributors that report both in the
  same family (``Corpus.dual_read_contributors``).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import field_validator, model_validator

from axiom.core import Dimension, NonEmptyStr, Spec, dimensionless

__all__ = [
    "POOLABLE_QUANTITIES",
    "Corpus",
    "PoolableQuantity",
    "ReadKind",
    "StudyRecord",
    "poolable_quantity",
]

ReadKind = Literal["model", "experiment"]
"""Where the number came from: a fitted model's read or a randomized experiment."""

Scale = Literal["natural", "log"]


@dataclass(frozen=True)
class PoolableQuantity:
    """One entry of the catalog: what the quantity is, whether it is dimensionless by
    construction, and the scale it is pooled on.

    ``scale == "log"`` means the estimate is a strictly positive ratio that the
    pool works with on the log scale; ``normalize`` refuses non-positive
    estimates for such quantities. ``dimensionless == False`` marks a
    quantity that carries units and so always needs a licensed transfer to
    enter a pool.
    """

    name: str
    description: str
    dimensionless: bool
    scale: Scale


POOLABLE_QUANTITIES: dict[str, PoolableQuantity] = {
    q.name: q
    for q in (
        PoolableQuantity(
            "elasticity",
            "d log(outcome) / d log(dose): the local proportional response.",
            True,
            "natural",
        ),
        PoolableQuantity(
            "ratio_log",
            "log of a ratio of like-dimensioned quantities (treated over control outcome).",
            True,
            "natural",
        ),
        PoolableQuantity(
            "response_ratio",
            "treated over control outcome; pooled on the log scale.",
            True,
            "log",
        ),
        PoolableQuantity(
            "standardized_contrast",
            "contrast divided by the outcome's pooled standard deviation (Hedges' g).",
            True,
            "natural",
        ),
        PoolableQuantity(
            "relative_change",
            "(treated - control) / control outcome.",
            True,
            "natural",
        ),
        PoolableQuantity(
            "log_odds_ratio",
            "log odds ratio of a binary outcome.",
            True,
            "natural",
        ),
        PoolableQuantity(
            "odds_ratio",
            "odds ratio of a binary outcome; pooled on the log scale.",
            True,
            "log",
        ),
        PoolableQuantity(
            "correlation",
            "a product-moment correlation; pooled on the natural scale (no z transform).",
            True,
            "natural",
        ),
        PoolableQuantity(
            "value_ratio",
            "outcome value per unit dose cost: carries units, needs a licensed transfer.",
            False,
            "natural",
        ),
        PoolableQuantity(
            "cost_per_outcome",
            "dose cost per unit outcome: carries units, needs a licensed transfer.",
            False,
            "log",
        ),
    )
}
"""Name → catalog entry. A ``StudyRecord.quantity`` must be one of these names."""


def poolable_quantity(name: str) -> PoolableQuantity:
    """The catalog entry for ``name``; ``ValueError`` naming the catalog otherwise."""
    try:
        return POOLABLE_QUANTITIES[name]
    except KeyError:
        raise ValueError(
            f"{name!r} is not a poolable quantity; known: {sorted(POOLABLE_QUANTITIES)}"
        ) from None


class StudyRecord(Spec):
    """One study's estimate of one poolable quantity, with its uncertainty and provenance.

    ``study`` is the record's unique id within a corpus; ``contributor`` the
    party that produced it (one contributor may submit many studies);
    ``family`` the pooling family (a treatment class) the record belongs to;
    ``quantity`` a ``POOLABLE_QUANTITIES`` name; ``estimand_hash`` ties the
    record to an ``estimands.Estimand`` when one exists. ``estimate``/``se``
    are on the quantity's reporting scale; ``n`` the study's size when
    known. ``dimension`` must be dimensionless unless ``unit_scale`` names
    the scale the number is on — such a record is refused at ingest without
    a licensed ``TransferPlan``. ``moderators`` are study-level covariates
    for meta-regression; ``period`` and ``source`` are free provenance text.
    """

    study: NonEmptyStr
    contributor: NonEmptyStr
    quantity: NonEmptyStr
    estimate: float
    se: float
    read: ReadKind
    family: NonEmptyStr
    estimand_hash: str = ""
    n: int | None = None
    moderators: dict[str, float] = {}
    dimension: Dimension = dimensionless()
    unit_scale: str = ""
    period: str = ""
    source: str = ""
    detail: dict[str, str] = {}

    @field_validator("quantity")
    @classmethod
    def _known_quantity(cls, v: str) -> str:
        poolable_quantity(v)
        return v

    @model_validator(mode="after")
    def _sound(self) -> StudyRecord:
        if not np.isfinite(self.estimate):
            raise ValueError(f"study {self.study!r}: estimate must be finite, got {self.estimate}")
        if not (np.isfinite(self.se) and self.se > 0):
            raise ValueError(f"study {self.study!r}: se must be finite and > 0, got {self.se}")
        if self.n is not None and self.n <= 0:
            raise ValueError(f"study {self.study!r}: n must be positive, got {self.n}")
        if not self.dimension.is_dimensionless and not self.unit_scale:
            raise ValueError(
                f"study {self.study!r}: pooling is over dimensionless quantities; a record "
                f"with dimension {self.dimension} must declare its unit_scale so ingest can "
                "demand a licensed TransferPlan"
            )
        for name, value in self.moderators.items():
            if not np.isfinite(value):
                raise ValueError(f"study {self.study!r}: moderator {name!r} is not finite")
        return self

    @property
    def is_dimensionless(self) -> bool:
        """True when the record can enter a pool without a transfer."""
        return self.dimension.is_dimensionless and not self.unit_scale


class Corpus(Spec):
    """An ordered collection of ``StudyRecord`` with unique study ids."""

    records: tuple[StudyRecord, ...]
    name: str = "corpus"

    @model_validator(mode="after")
    def _unique_ids(self) -> Corpus:
        seen: set[str] = set()
        dupes: list[str] = []
        for r in self.records:
            if r.study in seen:
                dupes.append(r.study)
            seen.add(r.study)
        if dupes:
            raise ValueError(f"duplicate study ids in corpus: {sorted(set(dupes))}")
        return self

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[StudyRecord]:  # type: ignore[override]
        return iter(self.records)

    def families(self) -> tuple[str, ...]:
        """The distinct families present, sorted."""
        return tuple(sorted({r.family for r in self.records}))

    def by_family(self, family: str) -> Corpus:
        """The sub-corpus of one family (order preserved)."""
        return Corpus(
            records=tuple(r for r in self.records if r.family == family),
            name=f"{self.name}/{family}",
        )

    def contributors(self) -> tuple[str, ...]:
        return tuple(sorted({r.contributor for r in self.records}))

    def dual_read_contributors(self, family: str | None = None) -> tuple[str, ...]:
        """Contributors that report both a model read and an experimental read within the
        same family (the ones that identify the provenance bias term). Restricted to
        ``family`` when given."""
        reads: dict[tuple[str, str], set[str]] = {}
        for r in self.records:
            if family is not None and r.family != family:
                continue
            reads.setdefault((r.contributor, r.family), set()).add(r.read)
        return tuple(sorted({c for (c, _), kinds in reads.items() if len(kinds) == 2}))

    def arrays(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(estimates, standard errors)`` in record order, for ``meta.classical``."""
        y = np.asarray([r.estimate for r in self.records], dtype=np.float64)
        se = np.asarray([r.se for r in self.records], dtype=np.float64)
        return y, se

    def to_frame(self) -> pd.DataFrame:
        """One row per record; moderators expanded to ``mod_<name>`` columns."""
        rows = []
        for r in self.records:
            row: dict[str, object] = {
                "study": r.study,
                "contributor": r.contributor,
                "family": r.family,
                "quantity": r.quantity,
                "read": r.read,
                "estimate": r.estimate,
                "se": r.se,
                "n": r.n,
                "dimension": str(r.dimension),
                "unit_scale": r.unit_scale,
                "period": r.period,
                "source": r.source,
                "estimand_hash": r.estimand_hash,
            }
            for k, v in r.moderators.items():
                row[f"mod_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)
