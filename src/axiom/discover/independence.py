"""Conditional independence, as a test with a p-value *and* an effect size.

Everything that learns or checks a structure rests on one question asked many
times: is ``X`` independent of ``Y`` once ``Z`` is accounted for? For jointly
Gaussian data the answer is the partial correlation, and Fisher's z-transform
turns it into a test.

The reason both numbers come back is not tidiness. A conditional-independence
test has a sample-size problem in *both* directions, and reporting only ``p``
hides it:

* with few rows there is no power, so nothing is ever rejected and a graph
  passes by being untested;
* with very many rows every approximation is detectably false, so a partial
  correlation of 0.01 rejects and a graph fails for being an idealization —
  which every graph is.

So ``IndependenceResult`` carries ``correlation`` beside ``p_value``, and the
callers that use it (``diagnose.refute_structure``, the FCI adjacency search)
take an effect threshold as well as an ``alpha``. This is the same lesson as
the rank tolerance in ``design.identifiability`` and the sparsity penalty in
``discover.search``: the threshold *is* the question, so it is a parameter and
never a constant.

Only rows in which none of the variables under test were intervened on are
used. Randomizing ``X`` destroys exactly the association a test of ``X`` is
asking about, so including those rows would answer a different question.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import stats as _st

from axiom.core import NonEmptyStr, Spec, Unsupported
from axiom.discover.score import Dataset

__all__ = [
    "IndependenceResult",
    "PartialCorrelation",
    "partial_correlation",
]

Array = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]


class IndependenceResult(Spec):
    """One conditional-independence test: what was asked, and both answers.

    ``correlation`` is the partial correlation — the *size* of the dependence
    — and ``p_value`` the evidence against it being zero. A decision needs
    both: significance without magnitude is a sample-size artifact, magnitude
    without significance is noise.
    """

    x: NonEmptyStr
    y: NonEmptyStr
    given: tuple[str, ...] = ()
    correlation: float = Field(ge=-1.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    n: int = Field(ge=0)
    statistic: float = 0.0

    @model_validator(mode="after")
    def _distinct(self) -> IndependenceResult:
        if self.x == self.y:
            raise ValueError(f"a variable is not tested against itself: {self.x!r}")
        clash = sorted({self.x, self.y} & set(self.given))
        if clash:
            raise ValueError(f"cannot condition on the variables under test: {clash}")
        return self

    @property
    def effect(self) -> float:
        """``|correlation|`` — the magnitude a practical threshold compares against."""
        return abs(self.correlation)

    def describe(self) -> str:
        given = f" | {', '.join(self.given)}" if self.given else ""
        return (
            f"{self.x} vs {self.y}{given}: r = {self.correlation:+.3f}, "
            f"p = {self.p_value:.3g} (n = {self.n})"
        )


def partial_correlation(values: Array, n_conditioning: int) -> float:
    """The partial correlation of the first two columns given the rest.

    Computed from the inverse of the correlation matrix, which is the
    numerically better-behaved route: a near-singular conditioning set shows up
    as a failure to invert rather than as a plausible-looking number.
    """
    matrix = np.corrcoef(values, rowvar=False)
    if matrix.ndim == 0:  # pragma: no cover - a single column cannot be a pair
        raise ValueError("a partial correlation needs at least two variables")
    try:
        precision = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(matrix)
    denominator = math.sqrt(abs(precision[0, 0] * precision[1, 1]))
    if denominator == 0.0:
        return 0.0
    del n_conditioning
    return float(np.clip(-precision[0, 1] / denominator, -1.0, 1.0))


@dataclass(frozen=True)
class PartialCorrelation:
    """A Gaussian conditional-independence test over a dataset.

    Assumes the variables are jointly Gaussian, or at least that dependence
    shows up as linear correlation. A nonlinear dependence that happens to be
    uncorrelated passes this test, which is a real limitation and the reason a
    graph "not refuted" is never "confirmed".
    """

    data: Dataset

    def usable_rows(self, variables: Iterable[str]) -> Mask:
        """Rows in which none of ``variables`` was randomized."""
        wanted = set(variables)
        return np.array([not (wanted & regime) for regime in self.data.regimes], dtype=bool)

    def test(self, x: str, y: str, given: Sequence[str] = ()) -> IndependenceResult | Unsupported:
        """``x`` against ``y`` given ``given``; ``Unsupported`` when the rows cannot answer.

        Fisher's z-transform gives the p-value: with ``n`` usable rows and
        ``k`` conditioning variables the statistic is
        ``sqrt(n - k - 3) * atanh(r)``, standard normal under independence.
        """
        names = self.data.names
        for name in (x, y, *given):
            if name not in names:
                raise KeyError(f"no variable {name!r}; have {list(names)}")
        conditioning = [g for g in dict.fromkeys(given) if g not in (x, y)]
        usable = self.usable_rows([x, y, *conditioning])
        rows = int(np.count_nonzero(usable))
        needed = len(conditioning) + 4
        if rows < needed:
            return Unsupported(
                reason=(
                    f"testing {x} vs {y} given {conditioning} needs at least {needed} rows "
                    f"in which none of them was randomized; there are {rows}"
                ),
                detail={"rows": str(rows), "needed": str(needed)},
            )
        columns = np.array([names.index(n) for n in (x, y, *conditioning)], dtype=int)
        block = self.data.values[np.ix_(np.flatnonzero(usable), columns)]
        if float(np.min(np.std(block, axis=0))) == 0.0:
            return Unsupported(
                reason=(
                    f"one of {[x, y, *conditioning]} does not vary in the rows that could "
                    "answer this test, so no correlation is defined"
                ),
            )
        correlation = partial_correlation(block, len(conditioning))
        degrees = rows - len(conditioning) - 3
        bounded = float(np.clip(correlation, -0.999999999, 0.999999999))
        statistic = math.sqrt(degrees) * math.atanh(bounded)
        p_value = float(2.0 * _st.norm.sf(abs(statistic)))
        return IndependenceResult(
            x=x,
            y=y,
            given=tuple(conditioning),
            correlation=correlation,
            p_value=p_value,
            n=rows,
            statistic=statistic,
        )

    def independent(
        self, x: str, y: str, given: Sequence[str] = (), *, alpha: float = 0.05
    ) -> bool:
        """Whether the test *fails to reject* independence — which is not the same as proving it."""
        result = self.test(x, y, given)
        if isinstance(result, Unsupported):
            return True  # no evidence of dependence is not evidence of dependence
        return result.p_value >= alpha
