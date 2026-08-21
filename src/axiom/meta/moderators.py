"""Study-level moderators for meta-regression: the centered design the pool reads.

A moderator is a study-level covariate (a dose level, a follow-up length, a
population share) whose coefficient ``gamma`` explains part of the
between-study heterogeneity. The pool's mean is ``theta + X gamma + ...``
with ``X`` **centered at the corpus mean**, so the pooled ``mu`` keeps its
meaning as the family mean at the average study rather than at ``X = 0``
(which for a follow-up length is nowhere). The means are kept on the
design so a prediction at a new study's moderators can be centered the same
way.

Ported by specification from the moderator handling in the parent's
``benchmarks/meta_model.py`` (ledger row ``meta/pool.py``); the parent
centered inside the PyMC block, here the design is a plain carrier the
model reads as ``Data`` columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from axiom.core import Unsupported
from axiom.meta.schema import Corpus

__all__ = ["ModeratorDesign", "moderator_matrix"]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ModeratorDesign:
    """Centered moderator columns ``(n_records, n_moderators)``, their names, and the means
    subtracted — a plain carrier (Specs hold no arrays)."""

    names: tuple[str, ...]
    columns: Array
    means: Array

    @property
    def n_records(self) -> int:
        return int(self.columns.shape[0])

    @property
    def n_moderators(self) -> int:
        return len(self.names)

    def column(self, name: str) -> Array:
        """The centered column for ``name``."""
        try:
            j = self.names.index(name)
        except ValueError:
            raise KeyError(f"no moderator {name!r}; have {list(self.names)}") from None
        return np.asarray(self.columns[:, j], dtype=np.float64)

    def center(self, values: dict[str, float]) -> Array:
        """Center a new study's moderator values the way the design was centered."""
        missing = [n for n in self.names if n not in values]
        if missing:
            raise KeyError(f"moderator values missing for {missing}")
        return np.asarray([values[n] for n in self.names], dtype=np.float64) - self.means


def moderator_matrix(corpus: Corpus, names: Sequence[str]) -> ModeratorDesign | Unsupported:
    """The centered moderator design of ``corpus`` for ``names``, in record order.

    Every record must carry every named moderator; a record missing one is a
    typed ``Unsupported`` naming the studies and moderators involved, never a
    silently imputed zero (which, after centering, would be the corpus mean
    and would quietly shrink the coefficient). A moderator that is constant
    across the corpus is likewise refused — its coefficient would be
    unidentified and the prior alone would be reported as a posterior.
    """
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError(f"moderator names must be distinct: {list(names)}")
    if not corpus.records:
        raise ValueError("moderator_matrix needs a non-empty corpus")
    missing: list[str] = []
    for r in corpus.records:
        absent = [n for n in names if n not in r.moderators]
        if absent:
            missing.append(f"{r.study}: {absent}")
    if missing:
        return Unsupported(
            reason="every record must carry every moderator the pool uses; missing on "
            + "; ".join(missing),
            missing=tuple(
                sorted({n for r in corpus.records for n in names if n not in r.moderators})
            ),
            detail={"studies": ", ".join(m.split(":")[0] for m in missing)},
        )
    raw = np.asarray(
        [[r.moderators[n] for n in names] for r in corpus.records], dtype=np.float64
    ).reshape(len(corpus.records), len(names))
    means = raw.mean(axis=0) if names else np.zeros(0)
    centered = raw - means
    constant = [
        n for j, n in enumerate(names) if len(corpus.records) > 1 and np.ptp(raw[:, j]) == 0.0
    ]
    if constant:
        return Unsupported(
            reason=f"moderators {constant} are constant across the corpus; their coefficients "
            "are not identified",
            missing=tuple(constant),
        )
    return ModeratorDesign(names=names, columns=centered, means=np.asarray(means, dtype=float))
