"""The seam. Upper layers are written against these protocols, not a model class.

``SupportsPosterior`` is satisfied by a NumPyro fit, a Laplace approximation,
a stored npz, or a hand-built dict of draws. ``SupportsIntervention`` is
anything you can ask a counterfactual of. ``Capability`` is the typed
replacement for the parent's runtime ``inspect`` checks: a producer declares
what it can do, and a request needing more returns ``Unsupported``.

``SupportsForward`` is the design layer's dependency: a deterministic
response surface whose ``forward`` is ``interpret.value`` over ``expr``.
``SupportsEstimands`` is what a *fitted* thing offers: it is a posterior, it
answers counterfactuals, and it names the estimands it was declared with.
``estimands.realize`` is written against it; a fitted surface, a stored
analysis, or a hand-built test double all satisfy it.
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
from axiom.core.result import Unsupported

__all__ = [
    "Capability",
    "DesignMatrix",
    "PredictiveDraws",
    "SupportsEstimands",
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
    ) -> PredictiveDraws | Unsupported: ...
    def capabilities(self) -> frozenset[Capability]: ...


@dataclass(frozen=True)
class DesignMatrix:
    """``mu = offset + X @ theta[columns]`` at a fixed point of the nonlinear parameters.

    ``columns`` names the linear parameters (one per column of ``X``);
    ``offset`` collects every term that does not depend on them; ``at`` is
    the nonlinear parameter point the linearization was taken at. The
    invariant ``‖X @ theta − forward(dose, theta)‖∞ < 1e-12`` is gate 9.
    """

    X: npt.NDArray[np.float64]
    columns: tuple[str, ...]
    offset: npt.NDArray[np.float64]
    at: Mapping[str, npt.NDArray[np.float64]]

    def predict(self, theta: Mapping[str, npt.ArrayLike]) -> npt.NDArray[np.float64]:
        """``offset + X @ theta[columns]``; a column named ``name[i]`` reads element ``i``."""
        beta = np.empty(len(self.columns))
        for j, col in enumerate(self.columns):
            name, _, idx = col.partition("[")
            v = np.asarray(theta[name], dtype=float)
            if idx:
                beta[j] = v.ravel()[int(idx.rstrip("]"))]
            elif v.size == 1:
                beta[j] = float(v.ravel()[0])
            else:
                raise ValueError(
                    f"parameter {name!r} has {v.size} values but column {col!r} needs one; "
                    "pass a point value, not draws"
                )
        return np.asarray(self.offset + self.X @ beta, dtype=np.float64)


@runtime_checkable
class SupportsForward(Protocol):
    """A deterministic response surface: ``forward`` == ``interpret.value(expr)``.

    ``dose`` maps data column name to array; ``theta`` maps parameter name to
    array. Implementations must not reimplement the transform chain — they
    call the interpreter on ``expr`` (rule 3, "one forward()").
    ``linearize`` returns the design matrix in the *linear* parameters at a
    fixed point of the nonlinear ones.
    """

    @property
    def expr(self) -> Model: ...
    def forward(
        self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
    ) -> npt.NDArray[np.float64]: ...
    def linearize(
        self, dose: Mapping[str, npt.ArrayLike], theta_at: Mapping[str, npt.ArrayLike]
    ) -> DesignMatrix: ...


@runtime_checkable
class SupportsEstimands(SupportsPosterior, SupportsIntervention, Protocol):
    """A fitted producer: posterior draws + counterfactual predictions + declared estimands.

    ``declared_estimands`` holds the content hashes of the estimands the
    producer was declared with (the ``Estimand`` type lives above ``core``).
    ``outcome_unit`` / ``dose_unit(treatment)`` report the units the
    posterior was fit in, so realization can convert — with a ledger line —
    when an estimand is declared in another unit of the same dimension.
    """

    @property
    def declared_estimands(self) -> Sequence[str]: ...
    @property
    def outcome_unit(self) -> str | None: ...
    def dose_unit(self, treatment: str) -> str | None: ...
    def marginal_under(
        self,
        iv: Intervention,
        treatment: str,
        window: TimeWindow | None = None,
        seed: int | None = None,
    ) -> PredictiveDraws | Unsupported: ...


def missing_capabilities(
    producer: SupportsIntervention, required: frozenset[Capability] | set[Capability]
) -> tuple[str, ...]:
    """Names of required capabilities the producer lacks, sorted; empty means go."""
    return tuple(sorted(c.value for c in set(required) - producer.capabilities()))
