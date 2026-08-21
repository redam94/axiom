"""Gate 10: everything shipped dimension-checks, and the checker agrees with the evaluator.

1. Every example ``Estimand`` derives to its declared dimension (the class
   enforces it; the gate proves the enforcement bites).
2. Every nonlinear node (``Apply``, ``Link``, variable-exponent ``Pow``) in
   every shipped model receives a dimensionless argument — by construction
   of ``interpret.dimension``; tested on the example corpus and on
   deliberately broken trees.
3. Every shipped ``Equation`` / ``System`` / ``ODESystem`` balances.
4. Over random trees, ``interpret.dimension`` accepts exactly the trees
   ``interpret.value`` handles on dimensionally consistent inputs, and every
   rejection names a node.

The "shipped" corpus is ``tests/contracts/_factories.py`` plus the kernel
registry once ``surface`` exists (parametrize there when it lands).
"""

from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest
from _factories import EXAMPLES

from axiom.core import (
    Add,
    Apply,
    Const,
    D,
    Data,
    Dimension,
    DimensionError,
    Div,
    Equation,
    Link,
    Model,
    Mul,
    ODESystem,
    Param,
    Pow,
    Spec,
    System,
    dimension,
    dimensionless,
    value,
)
from axiom.estimands import Estimand, derived_dimension

_MODELS = [
    cls
    for cls in EXAMPLES
    if issubclass(cls, Add | Mul | Div | Pow | Apply | Link | Equation | System | ODESystem)
]


@pytest.mark.parametrize("cls", _MODELS, ids=lambda c: c.__name__)
def test_every_shipped_model_dimension_checks(cls: type[Spec]) -> None:
    model = EXAMPLES[cls]()
    assert isinstance(dimension(model), Dimension)  # type: ignore[arg-type]


def test_every_shipped_estimand_derives_to_its_declaration() -> None:
    e = EXAMPLES[Estimand]()
    assert isinstance(e, Estimand)
    assert derived_dimension(e.quantity.kind, e.outcome.dim, e.treatment.dim) == e.dimension
    with pytest.raises(DimensionError, match="derives to"):
        e.model_copy(update={"dimension": D.time}).model_validate(
            e.model_copy(update={"dimension": D.time}).to_dict()
        )


def test_nonlinear_nodes_reject_dimensioned_arguments() -> None:
    dose = Data(name="dose", dimension=D.currency)
    for bad in (
        Apply(fn="log", arg=dose),
        Link(fn="logit", arg=dose),
        Pow(base=dose, exponent=Param(name="s", dimension=dimensionless())),
        Pow(base=Div(numerator=dose, denominator=dose), exponent=Param(name="s", dimension=D.time)),
    ):
        with pytest.raises(DimensionError) as e:
            dimension(bad)
        assert bad.node in str(e.value)


def test_unbalanced_equations_name_the_node() -> None:
    y = Data(name="y", dimension=D.outcome)
    t = Data(name="t", dimension=D.time)
    with pytest.raises(DimensionError, match=r"system\[1\]\.equation: equation 'second'"):
        dimension(System(equations=(Equation(lhs=y, rhs=y), Equation(lhs=y, rhs=t, name="second"))))
    with pytest.raises(DimensionError, match=r"d\(S\)/dt"):
        dimension(ODESystem(states=(y.model_copy(update={"name": "S"}),), rhs=(y,), time=t))


# -- random trees ----------------------------------------------------------------------

_BASES = [D.currency, D.time, D.outcome, dimensionless()]


def _random_tree(rng: random.Random, depth: int, consistent: bool) -> Model:
    """Build a tree; when ``consistent`` every node is well-typed, else one node may not be."""
    if depth == 0 or rng.random() < 0.25:
        dim = rng.choice(_BASES)
        kind = rng.random()
        if kind < 0.4:
            return Data(name=f"x{rng.randrange(3)}", dimension=dim)
        if kind < 0.8:
            return Param(name=f"p{rng.randrange(3)}", dimension=dim)
        return Const(value=rng.uniform(0.5, 2.0), dimension=dim)
    choice = rng.randrange(6)
    child = lambda: _random_tree(rng, depth - 1, consistent)  # noqa: E731
    if choice == 0:
        first = child()
        d = dimension(first) if consistent else None
        terms = [first]
        for _ in range(rng.randrange(1, 3)):
            t = child()
            if consistent and d is not None:
                # force the same dimension by dividing out and multiplying back in
                t = Mul(
                    factors=(
                        Div(numerator=t, denominator=Const(value=1.0, dimension=dimension(t))),
                        Const(value=1.0, dimension=d),
                    )
                )
            terms.append(t)
        return Add(terms=tuple(terms))
    if choice == 1:
        return Mul(factors=tuple(child() for _ in range(rng.randrange(1, 4))))
    if choice == 2:
        return Div(numerator=child(), denominator=child())
    if choice == 3:
        return Pow(base=child(), exponent=Fraction(rng.choice([1, 2, -1]), rng.choice([1, 2])))
    if choice == 4:
        arg = child()
        if consistent:
            arg = Div(numerator=arg, denominator=Const(value=1.0, dimension=dimension(arg)))
        return Apply(fn=rng.choice(["exp", "tanh", "softplus"]), arg=arg)
    arg = child()
    if consistent:
        arg = Div(numerator=arg, denominator=Const(value=1.0, dimension=dimension(arg)))
    return Link(fn="identity", arg=arg)


def _evaluates(model: Model) -> bool:
    data = {f"x{i}": np.array([0.5, 1.5, 2.5]) for i in range(3)}
    params = {f"p{i}": 1.25 + 0.5 * i for i in range(3)}
    out = value(model, data=data, params=params)
    return bool(np.all(np.isfinite(out)))


def test_checker_and_evaluator_agree_on_200_random_trees() -> None:
    rng = random.Random(2026)
    accepted = rejected = 0
    for i in range(200):
        consistent = i % 2 == 0
        tree = _random_tree(rng, depth=4, consistent=consistent)
        try:
            dimension(tree)
            ok = True
        except DimensionError as e:
            ok = False
            # every rejection names a node path
            assert any(tok in str(e) for tok in ("add", "apply", "link", "pow", "equation"))
        if consistent:
            assert ok, f"consistent tree {i} rejected"
        if ok:
            accepted += 1
            assert _evaluates(tree), f"accepted tree {i} did not evaluate"
        else:
            rejected += 1
            # the evaluator still runs on raw numbers: the checker is what catches it
            assert _evaluates(tree)
    assert accepted >= 100 and rejected >= 20, (accepted, rejected)
