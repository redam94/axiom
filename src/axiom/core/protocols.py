"""The seam. Upper layers are written against these protocols, not a model class.

``SupportsPosterior`` is satisfied by a NumPyro fit, a Laplace approximation,
a stored npz, or a hand-built dict of draws. ``SupportsIntervention`` is
anything you can ask a counterfactual of. ``Capability`` is the typed
replacement for the parent's runtime ``inspect`` checks: a producer declares
what it can do, and a request needing more returns ``Unsupported``.

``SupportsForward`` is the design layer's dependency: a deterministic
response surface whose ``forward`` is ``interpret.value`` over ``expr``.
``linearize`` joins it in Phase 3 with the design matrix; ``SupportsEstimands``
joins in Phase 4 with realization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from axiom.core.entities import Intervention, TimeWindow, Treatment
from axiom.core.expr import Model

__all__ = [
    "Capability",
    "PredictiveDraws",
    "SupportsForward",
    "SupportsIntervention",
    "SupportsPosterior",
    "missing_capabilities",
]


class Capability(StrEnum):
    """What a producer can do. A request needing an absent one degrades, typed."""

    COUNTERFACTUAL = "counterfactual"  # predict_under an arbitrary Intervention
    MARGINAL = "marginal"  # closed-form d(outcome)/d(dose)
    TIME_WINDOW = "time_window"  # restrict predictions to a TimeWindow
    PER_UNIT = "per_unit"  # per-observation-unit predictions
    LOG_LIKELIHOOD = "log_likelihood"  # pointwise log-likelihood draws (LOO)
    PREDICTIVE = "predictive"  # outcome-scale posterior predictive draws


@runtime_checkable
class SupportsPosterior(Protocol):
    """Anything with draws you can ask questions of.

    ``draws(name)`` returns ``(chain, draw, *shape)``. ``coords`` maps a
    dimension name to its labels for the trailing axes.
    """

    def draws(self, name: str) -> npt.NDArray[np.float64]: ...
    def names(self) -> frozenset[str]: ...
    def coords(self) -> Mapping[str, Sequence[Any]]: ...
    def n_draws(self) -> int: ...


@dataclass(frozen=True)
class PredictiveDraws:
    """Outcome draws under an intervention: ``values`` is ``(chain, draw, *shape)``."""

    values: npt.NDArray[np.float64]
    intervention: Intervention
    window: TimeWindow | None = None
    coords: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    seed: int | None = None


@runtime_checkable
class SupportsIntervention(Protocol):
    """Anything you can ask a counterfactual of."""

    @property
    def treatments(self) -> Sequence[Treatment]: ...
    def predict_under(
        self, iv: Intervention, window: TimeWindow | None = None, seed: int | None = None
    ) -> PredictiveDraws: ...
    def capabilities(self) -> frozenset[Capability]: ...


@runtime_checkable
class SupportsForward(Protocol):
    """A deterministic response surface: ``forward`` == ``interpret.value(expr)``.

    ``dose`` maps data column name to array; ``theta`` maps parameter name to
    array. Implementations must not reimplement the transform chain — they
    call the interpreter on ``expr`` (rule 3, "one forward()").
    """

    @property
    def expr(self) -> Model: ...
    def forward(
        self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
    ) -> npt.NDArray[np.float64]: ...


def missing_capabilities(
    producer: SupportsIntervention, required: frozenset[Capability] | set[Capability]
) -> tuple[str, ...]:
    """Names of required capabilities the producer lacks, sorted; empty means go."""
    return tuple(sorted(c.value for c in set(required) - producer.capabilities()))
