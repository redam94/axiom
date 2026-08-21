"""The efficient frontier: optimal expected outcome as a function of the budget.

``frontier`` runs ``allocate`` at each budget, ascending, warm-starting every
solve from the previous budget's allocation (shrunk toward the lower bounds
when it would overspend). The result is a ``Frontier`` spec — budgets,
outcomes, and the full ``Allocation`` at each — whose ``shadow_prices`` are
the finite-difference slope ``d outcome / d budget``: the marginal value of
one more unit of budget, the number a planner compares against its cost.

A single budget that fails to converge fails the whole frontier, typed
(``Unsupported`` naming the budget): a curve with a silently interpolated
point would be a wrong number. So does a budget whose optimum falls below
the previous, smaller budget's: the feasible set only grows with the
budget, so a drop is a missed basin, not a curve. A surface with carryover is refused up
front, as in ``allocate`` — budgets are independent candidates, not periods
of one series; pass ``surface.steady_state()``. Non-treatment columns the
surface reads go through ``context`` exactly as in ``allocate``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import model_validator

from axiom.core import (
    Blocked,
    Failure,
    Spec,
    SupportsForward,
    SupportsPosterior,
    Unsupported,
    Unverified,
)
from axiom.surface.design import Bounds
from axiom.surface.optimize import (
    Allocation,
    AllocationMethod,
    Context,
    Objective,
    allocate,
    carryover_failure,
)

__all__ = ["Frontier", "frontier"]

Array = npt.NDArray[np.float64]
Theta = Mapping[str, npt.ArrayLike]


class Frontier(Spec):
    """Optimal outcome per budget: ``budgets`` ascend; ``allocations[i]`` reaches ``outcomes[i]``.

    ``shadow_prices`` is the slope ``d outcome / d budget`` along the curve.
    """

    budgets: tuple[float, ...]
    outcomes: tuple[float, ...]
    allocations: tuple[Allocation, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Frontier:
        n = len(self.budgets)
        if n == 0:
            raise ValueError("a Frontier needs at least one budget")
        if len(self.outcomes) != n or len(self.allocations) != n:
            raise ValueError("budgets, outcomes, and allocations must be equally long")
        if any(not math.isfinite(b) for b in self.budgets):
            raise ValueError("budgets must be finite")
        if any(b1 <= b0 for b0, b1 in zip(self.budgets, self.budgets[1:], strict=False)):
            raise ValueError("budgets must be strictly increasing")
        for b, y, a in zip(self.budgets, self.outcomes, self.allocations, strict=True):
            if a.budget != b or a.expected_outcome != y:
                raise ValueError(f"allocation at budget {b} disagrees with the frontier arrays")
        names = {a.treatments for a in self.allocations}
        if len(names) != 1:
            raise ValueError("every allocation must name the same treatments")
        return self

    @property
    def treatments(self) -> tuple[str, ...]:
        return self.allocations[0].treatments

    @property
    def n(self) -> int:
        return len(self.budgets)

    def shadow_prices(self) -> Array:
        """``d outcome / d budget`` by finite differences along the frontier (one per budget).

        With a single budget the slope is undefined and ``nan`` is returned.
        """
        if self.n < 2:
            return np.full(1, np.nan)
        return np.asarray(
            np.gradient(np.asarray(self.outcomes), np.asarray(self.budgets)), dtype=np.float64
        )

    def doses(self) -> Array:
        """``(n_budgets, k)`` doses, columns in ``treatments`` order."""
        return np.vstack([a.as_array() for a in self.allocations])

    def as_frame(self) -> pd.DataFrame:
        """One row per budget: budget, expected outcome, shadow price, and one column per dose."""
        prices = self.shadow_prices()
        rows = [
            {
                "budget": a.budget,
                "expected_outcome": a.expected_outcome,
                "shadow_price": float(p),
                **a.doses,
            }
            for a, p in zip(self.allocations, prices, strict=True)
        ]
        return pd.DataFrame(rows)


def frontier(
    surface: SupportsForward,
    posterior_or_theta: SupportsPosterior | Theta | Failure,
    budgets: Sequence[float],
    bounds: Bounds,
    *,
    objective: Objective = "mean",
    q: float = 0.5,
    seed: int | None = None,
    method: AllocationMethod = "slsqp",
    n_starts: int = 1,
    n_draws: int | None = None,
    maxiter: int = 200,
    tol: float = 1e-12,
    n_grid: int = 101,
    context: Context | None = None,
) -> Frontier | Failure:
    """``allocate`` at every budget (sorted ascending), each warm-started from the last.

    Arguments after ``bounds`` are passed through to ``allocate`` (the warm
    start joins its fixed start set of corners, two-way splits, centre, and
    pre-search winners). Duplicate budgets are rejected. A surface with
    carryover returns ``Unsupported`` asking for ``surface.steady_state()``;
    any budget that does not converge, or whose optimum falls below the
    previous budget's, returns ``Unsupported`` naming it, with the solver's
    message; a typed failure passed as the posterior is returned unchanged.
    """
    if isinstance(posterior_or_theta, Unsupported | Blocked | Unverified):
        return posterior_or_theta
    carried = carryover_failure(surface)
    if carried is not None:
        return carried
    if len(budgets) == 0:
        raise ValueError("frontier needs at least one budget")
    ordered = sorted(float(b) for b in budgets)
    if any(b1 <= b0 for b0, b1 in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("budgets must be distinct")
    allocations: list[Allocation] = []
    start: Mapping[str, float] | None = None
    for b in ordered:
        result = allocate(
            surface,
            posterior_or_theta,
            budget=b,
            bounds=bounds,
            objective=objective,
            q=q,
            seed=seed,
            method=method,
            start=start,
            n_starts=n_starts,
            n_draws=n_draws,
            maxiter=maxiter,
            tol=tol,
            n_grid=n_grid,
            context=context,
        )
        if not isinstance(result, Allocation):
            if not isinstance(result, Unsupported):  # pragma: no cover - propagated above
                return result
            return Unsupported(
                reason=f"frontier failed at budget {b:.6g}: {result.reason}",
                detail={**result.detail, "budget": f"{b:.6g}"},
                missing=result.missing,
            )
        if allocations:
            prev = allocations[-1]
            spread = max(abs(a.expected_outcome - prev.expected_outcome) for a in allocations)
            tol = 1e-9 * max(1.0, abs(prev.expected_outcome), spread)
            if result.expected_outcome < prev.expected_outcome - tol:
                return Unsupported(
                    reason=f"frontier failed at budget {b:.6g}: its optimum "
                    f"({result.expected_outcome:.6g}) is below the optimum at budget "
                    f"{prev.budget:.6g} ({prev.expected_outcome:.6g}); the feasible set only "
                    "grows with the budget, so the allocator missed a basin — raise n_starts",
                    detail={
                        "budget": f"{b:.6g}",
                        "outcome": f"{result.expected_outcome:.6g}",
                        "previous_budget": f"{prev.budget:.6g}",
                        "previous_outcome": f"{prev.expected_outcome:.6g}",
                    },
                )
        allocations.append(result)
        start = result.doses
    return Frontier(
        budgets=tuple(a.budget for a in allocations),
        outcomes=tuple(a.expected_outcome for a in allocations),
        allocations=tuple(allocations),
    )
