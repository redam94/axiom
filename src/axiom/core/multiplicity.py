"""Testing many things at once: the two corrections, and the one implementation of them.

``diagnose.structure`` needed this to refute a graph against forty implied
independences; ``diagnose.delivery`` needs it for a balance table; ``design``
needs it for a book of experiments. Three subpackages, one arithmetic, so it
lives at the bottom where all three can reach it rather than being written
again in each.

``holm`` controls the family-wise error rate — the chance of **any** false
positive — and is right when the output is a single verdict that one lucky test
should not be able to flip. ``benjamini_hochberg`` controls the false discovery
rate — the expected share of the positives that are false — and is right when
the output is a ranked list somebody will work through. ``none`` leaves the
values alone and, because it is a named choice rather than an omission, says so
on whatever result carries it.

Neither adjustment knows anything about dependence between the tests. Holm is
valid under any; Benjamini-Hochberg needs independence or positive regression
dependence, which a book of experiments sharing markets and seasons does not
obviously have. ``design.program`` carries an e-value route for exactly that
case.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

__all__ = ["Multiplicity", "adjust"]

Multiplicity = Literal["none", "holm", "benjamini_hochberg"]
"""Which error rate a family of tests controls, or ``none`` for uncorrected."""


def adjust(p_values: Sequence[float], correction: Multiplicity) -> tuple[float, ...]:
    """Adjusted p-values, in the order given.

    ``holm`` is the step-down family-wise procedure: the ``k``-th smallest of
    ``m`` p-values is multiplied by ``m − k + 1`` and the running maximum is
    taken so the sequence is monotone. ``benjamini_hochberg`` is the step-up
    false-discovery procedure: the ``k``-th smallest is multiplied by ``m / k``
    and the running minimum is taken from the largest down. Both cap at 1.

    An adjusted value is compared to the same ``alpha`` an unadjusted one would
    be, which is what makes the two comparable side by side.
    """
    values = list(p_values)
    total = len(values)
    if any(not (math.isfinite(p) and 0.0 <= p <= 1.0) for p in values):
        raise ValueError(f"p-values must be finite and in [0, 1], got {values[:5]}")
    if total == 0 or correction == "none":
        return tuple(values)
    order = sorted(range(total), key=lambda i: values[i])
    adjusted = [0.0] * total
    if correction == "holm":
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (total - rank) * values[index])
            adjusted[index] = min(1.0, running)
        return tuple(adjusted)
    running = 1.0
    for rank in range(total - 1, -1, -1):
        index = order[rank]
        running = min(running, total * values[index] / (rank + 1))
        adjusted[index] = min(1.0, running)
    return tuple(adjusted)
