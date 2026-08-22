"""The estimand formula: what an identification algorithm *returns*, not just whether it succeeded.

``identify()`` answers "is this effect identified, and by which route", and
hands back an adjustment set. That is enough when the route is back-door — the
estimand is the adjustment formula and everyone knows it. It stops being enough
the moment the answer is neither back-door nor front-door, which is most of the
time once latent confounding is drawn honestly.

So identification needs a return value with the same standing as the verdict:
the **formula**, a symbolic expression in the observational distribution. Four
node types are enough for everything the ID algorithm produces.

* ``Density`` — ``P(Y | X, Z)``, the atom.
* ``Product`` — a product of formulas.
* ``Marginal`` — a sum over named variables.
* ``Ratio`` — a quotient, which is how a conditional of an already-derived
  expression is written.

The formula renders (``to_text``, ``to_latex``) so it can go in a report, and
it *evaluates* (``evaluate``) against a discrete joint distribution, which is
what makes an identification claim checkable rather than merely plausible: run
the algorithm, evaluate its formula on a simulated world's observational
distribution, and compare against that world's true interventional
distribution. ``tests/recovery/test_id_recovery.py`` does exactly that.

Constructors (``product``, ``marginal``, ``ratio``) simplify as they build —
sums with nothing to sum over disappear, nested sums merge, and a factor that
does not mention a summed variable is lifted out of the sum. None of that
changes the value; all of it changes whether a human can read the result.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core.spec import Spec

__all__ = [
    "Density",
    "Formula",
    "JointTable",
    "Marginal",
    "Product",
    "Ratio",
    "evaluate",
    "free_variables",
    "joint_from_counts",
    "marginal",
    "product",
    "ratio",
    "to_latex",
    "to_text",
    "variables",
    "walk",
]

Array = npt.NDArray[np.float64]


class Density(Spec):
    """``P(outcomes | given)`` — a conditional of the observational distribution."""

    node: Literal["density"] = "density"
    outcomes: tuple[str, ...] = Field(min_length=1)
    given: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _disjoint(self) -> Density:
        shared = sorted(set(self.outcomes) & set(self.given))
        if shared:
            raise ValueError(f"a density cannot condition on its own outcome: {shared}")
        return self


class Product(Spec):
    """A product of formulas."""

    node: Literal["product"] = "product"
    factors: tuple[Formula, ...] = Field(min_length=1)


class Marginal(Spec):
    """``sum over `over` of `term``` — marginalization."""

    node: Literal["marginal"] = "marginal"
    over: tuple[str, ...] = Field(min_length=1)
    term: Formula


class Ratio(Spec):
    """``numerator / denominator``; how a conditional of a derived expression is written."""

    node: Literal["ratio"] = "ratio"
    numerator: Formula
    denominator: Formula


Formula = Annotated[Density | Product | Marginal | Ratio, Field(discriminator="node")]

for _cls in (Product, Marginal, Ratio):
    _cls.model_rebuild()


# -- traversal -----------------------------------------------------------------------


def walk(formula: Formula) -> Iterator[Formula]:
    """Every node of the formula, pre-order."""
    yield formula
    match formula:
        case Product():
            for factor in formula.factors:
                yield from walk(factor)
        case Marginal():
            yield from walk(formula.term)
        case Ratio():
            yield from walk(formula.numerator)
            yield from walk(formula.denominator)
        case _:
            return


def variables(formula: Formula) -> frozenset[str]:
    """Every variable the formula mentions, summed-over ones included."""
    out: set[str] = set()
    for node in walk(formula):
        if isinstance(node, Density):
            out |= set(node.outcomes) | set(node.given)
        elif isinstance(node, Marginal):
            out |= set(node.over)
    return frozenset(out)


def free_variables(formula: Formula) -> frozenset[str]:
    """The variables the formula is a function of: mentioned, minus summed out."""
    match formula:
        case Density():
            return frozenset(formula.outcomes) | frozenset(formula.given)
        case Product():
            out: set[str] = set()
            for factor in formula.factors:
                out |= free_variables(factor)
            return frozenset(out)
        case Marginal():
            return free_variables(formula.term) - set(formula.over)
        case Ratio():
            return free_variables(formula.numerator) | free_variables(formula.denominator)
    raise TypeError(f"not a formula node: {type(formula).__name__}")  # pragma: no cover


# -- constructors that simplify -------------------------------------------------------


def product(*factors: Formula) -> Formula:
    """A product, flattening nested products; a single factor is returned as itself."""
    flat: list[Formula] = []
    for factor in factors:
        if isinstance(factor, Product):
            flat.extend(factor.factors)
        else:
            flat.append(factor)
    if not flat:
        raise ValueError("a product needs at least one factor")
    if len(flat) == 1:
        return flat[0]
    return Product(factors=tuple(flat))


def marginal(over: Iterable[str], term: Formula) -> Formula:
    """A sum, dropping variables the term does not depend on and merging nested sums.

    Three exact identities are applied as it builds. Summing a joint density
    over some of its own variables is the smaller joint
    (``sum_x P(x, z) = P(z)``). Nested sums merge. And a factor that does not
    mention a summed variable is lifted out of the sum:
    ``sum_z P(y|z) P(z) P(w)`` becomes ``P(w) sum_z P(y|z) P(z)``. None of them
    changes the value; all of them change whether the result can be read.
    """
    names = frozenset(over) & free_variables(term)
    if not names:
        return term
    if isinstance(term, Density) and not term.given and names < set(term.outcomes):
        return Density(outcomes=tuple(n for n in term.outcomes if n not in names))
    if isinstance(term, Marginal):
        return marginal(names | set(term.over), term.term)
    if isinstance(term, Product):
        inside = [f for f in term.factors if free_variables(f) & names]
        outside = [f for f in term.factors if not (free_variables(f) & names)]
        if outside:
            return product(*outside, marginal(names, product(*inside)))
    return Marginal(over=tuple(sorted(names)), term=term)


def ratio(numerator: Formula, denominator: Formula) -> Formula:
    """A quotient, recognizing the definition of a conditional as it builds.

    ``P(a, b) / P(b)`` is ``P(a | b)`` — the identity that turns the ID
    algorithm's ratios of marginals back into the densities a reader expects.
    An equal numerator and denominator would be the constant one, which is not
    a formula, and is refused as the bug it would be.
    """
    if numerator == denominator:
        raise ValueError("a ratio of identical formulas is the constant one, not a formula")
    if (
        isinstance(numerator, Density)
        and isinstance(denominator, Density)
        and not numerator.given
        and not denominator.given
        and set(denominator.outcomes) < set(numerator.outcomes)
    ):
        given = tuple(n for n in numerator.outcomes if n in set(denominator.outcomes))
        outcomes = tuple(n for n in numerator.outcomes if n not in set(denominator.outcomes))
        return Density(outcomes=outcomes, given=given)
    return Ratio(numerator=numerator, denominator=denominator)


# -- rendering -------------------------------------------------------------------------


def to_text(formula: Formula) -> str:
    """``sum_{Z} P(Y | X, Z) P(Z)`` — the formula as a line of text."""
    match formula:
        case Density():
            given = f" | {', '.join(formula.given)}" if formula.given else ""
            return f"P({', '.join(formula.outcomes)}{given})"
        case Product():
            return " ".join(_bracketed(f) for f in formula.factors)
        case Marginal():
            return f"sum_{{{', '.join(formula.over)}}} {_bracketed(formula.term)}"
        case Ratio():
            return f"{_bracketed(formula.numerator)} / {_bracketed(formula.denominator)}"
    raise TypeError(f"not a formula node: {type(formula).__name__}")  # pragma: no cover


def _bracketed(formula: Formula) -> str:
    text = to_text(formula)
    return text if isinstance(formula, Density) else f"[{text}]"


def to_latex(formula: Formula) -> str:
    """The same, as LaTeX."""
    match formula:
        case Density():
            given = f" \\mid {', '.join(formula.given)}" if formula.given else ""
            return f"P({', '.join(formula.outcomes)}{given})"
        case Product():
            return " \\, ".join(_latex_bracketed(f) for f in formula.factors)
        case Marginal():
            return f"\\sum_{{{', '.join(formula.over)}}} {_latex_bracketed(formula.term)}"
        case Ratio():
            return f"\\frac{{{to_latex(formula.numerator)}}}{{{to_latex(formula.denominator)}}}"
    raise TypeError(f"not a formula node: {type(formula).__name__}")  # pragma: no cover


def _latex_bracketed(formula: Formula) -> str:
    text = to_latex(formula)
    return text if isinstance(formula, Density | Ratio) else f"\\left[{text}\\right]"


# -- evaluation ------------------------------------------------------------------------


@dataclass(frozen=True)
class JointTable:
    """A joint distribution over discrete variables, as an array with one axis per variable.

    Not a ``Spec``: it is the numbers, not the declaration. It exists so an
    identification formula can be *checked* — evaluate it on a world's
    observational joint and compare against that world's true interventional
    distribution.
    """

    names: tuple[str, ...]
    table: Array

    def __post_init__(self) -> None:
        if self.table.ndim != len(self.names):
            raise ValueError(
                f"the table has {self.table.ndim} axes but {len(self.names)} variables are named"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"variable names must be distinct: {list(self.names)}")
        total = float(np.sum(self.table))
        if not np.isclose(total, 1.0):
            raise ValueError(f"a joint distribution sums to one, not {total:.6g}")

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.table.shape)

    def axis(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no variable {name!r}; have {list(self.names)}") from None

    def marginal(self, keep: Iterable[str]) -> Array:
        """The joint over ``keep``, with singleton axes for everything else."""
        kept = {self.axis(n) for n in keep}
        summed = tuple(i for i in range(len(self.names)) if i not in kept)
        return np.asarray(np.sum(self.table, axis=summed, keepdims=True), dtype=np.float64)

    def conditional(self, outcomes: Sequence[str], given: Sequence[str]) -> Array:
        """``P(outcomes | given)``, with singleton axes for the variables it does not use."""
        joint = self.marginal([*outcomes, *given])
        base = self.marginal(given) if given else np.ones((1,) * len(self.names))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(base > 0.0, joint / base, 0.0)
        return np.asarray(out, dtype=np.float64)


def evaluate(formula: Formula, joint: JointTable) -> Array:
    """Evaluate the formula against a discrete joint distribution.

    The result carries one axis per variable of the joint, singleton where the
    formula does not depend on it, so a caller can compare it against a
    reference distribution by broadcasting.
    """
    match formula:
        case Density():
            return joint.conditional(formula.outcomes, formula.given)
        case Product():
            out = evaluate(formula.factors[0], joint)
            for factor in formula.factors[1:]:
                out = out * evaluate(factor, joint)
            return out
        case Marginal():
            axes = tuple(joint.axis(n) for n in formula.over)
            return np.asarray(
                np.sum(evaluate(formula.term, joint), axis=axes, keepdims=True), dtype=np.float64
            )
        case Ratio():
            top = evaluate(formula.numerator, joint)
            bottom = evaluate(formula.denominator, joint)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(bottom > 0.0, top / bottom, 0.0)
            return np.asarray(out, dtype=np.float64)
    raise TypeError(f"not a formula node: {type(formula).__name__}")  # pragma: no cover


def joint_from_counts(names: Sequence[str], counts: Mapping[tuple[int, ...], float]) -> JointTable:
    """A ``JointTable`` from a mapping of value-tuples to weights; the weights are normalized."""
    levels = [max(key[i] for key in counts) + 1 for i in range(len(names))]
    table = np.zeros(levels, dtype=np.float64)
    for key, weight in counts.items():
        table[key] += float(weight)
    total = table.sum()
    if total <= 0:
        raise ValueError("the counts are all zero")
    return JointTable(names=tuple(names), table=table / total)
