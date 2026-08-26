"""The arithmetic operators on ``Expr`` nodes build exactly what the constructors build.

The operators are sugar and nothing more: every test here pins an operator
expression against the constructor form it is supposed to be identical to, so a
model written as mathematics and the same model written out longhand are one
tree, one hash, and one set of dimension errors.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, get_args

import numpy as np
import pytest

from axiom.core import (
    Add,
    Const,
    D,
    Data,
    DimensionError,
    Div,
    Expr,
    Gather,
    Mul,
    Param,
    Pow,
    Spec,
    dimension,
    dimensionless,
    node_path,
    params,
    value,
)

dose = Data(name="dose", dimension=D.currency)
k = Param(name="k", dimension=D.currency)
beta = Param(name="beta", dimension=D.outcome)
s = Param(name="s", dimension=dimensionless())
unit = Data(name="unit_index", dimension=dimensionless())
alpha = Param(name="alpha", dimension=D.outcome, shape=(3,))

ONE = Const(value=1.0, dimension=dimensionless())


def _expr_members() -> tuple[type[Spec], ...]:
    """The classes in the ``Expr`` discriminated union."""
    members: tuple[type[Spec], ...] = get_args(get_args(Expr)[0])
    assert len(members) >= 13
    return members


def test_every_expr_node_carries_the_operators() -> None:
    """A new node class that forgets the alias block fails here, not in a user's model."""
    wanted = (
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__rtruediv__",
        "__pow__",
        "__rpow__",
        "__neg__",
        "__getitem__",
    )
    for cls in _expr_members():
        missing = [name for name in wanted if name not in vars(cls)]
        assert not missing, f"{cls.__name__} is missing {missing}"


def test_operators_build_the_constructor_form() -> None:
    assert dose / k == Div(numerator=dose, denominator=k)
    assert beta * dose == Mul(factors=(beta, dose))
    assert beta + dose == Add(terms=(beta, dose))
    assert (dose / k) ** s == Pow(base=Div(numerator=dose, denominator=k), exponent=s)
    assert alpha[unit] == Gather(source=alpha, index=unit)

    hill = beta * (dose / k) ** s / (1.0 + (dose / k) ** s)
    x = Div(numerator=dose, denominator=k)
    longhand = Div(
        numerator=Mul(factors=(beta, Pow(base=x, exponent=s))),
        denominator=Add(terms=(ONE, Pow(base=x, exponent=s))),
    )
    assert hill == longhand
    assert hill.content_hash() == longhand.content_hash()


def test_add_and_mul_flatten_and_canonicalize_associativity() -> None:
    assert (beta + beta + beta).terms == (beta, beta, beta)
    assert (beta + beta) + beta == beta + (beta + beta)
    assert (s * s * s).factors == (s, s, s)
    assert (s * s) * s == s * (s * s)
    # Div and Pow are binary and do not flatten.
    assert (dose / k / s) == Div(numerator=Div(numerator=dose, denominator=k), denominator=s)


def test_a_bare_number_is_dimensionless() -> None:
    assert s + 1 == Add(terms=(s, ONE))
    assert 2 * s == Mul(factors=(Const(value=2.0, dimension=dimensionless()), s))
    assert 1.0 / s == Div(numerator=ONE, denominator=s)
    assert dimension(2.0**s) == dimensionless()
    assert dimension(1.0 - s) == dimensionless()
    # A literal that carries units is still written out; it is not inferred.
    with pytest.raises(DimensionError, match=r"terms of a sum"):
        dimension(dose - 100.0)
    assert dimension(dose - Const(value=100.0, dimension=D.currency)) == D.currency


def test_negation_keeps_the_dimension() -> None:
    """``-a`` is ``(-1) · a``; ``Apply(fn="neg")`` would demand a dimensionless argument."""
    minus_one = Const(value=-1.0, dimension=dimensionless())
    assert -dose == Mul(factors=(minus_one, dose))
    assert dimension(-dose) == D.currency
    assert dose - k == Add(terms=(dose, Mul(factors=(minus_one, k))))
    assert dimension(dose - k) == D.currency


def test_reflected_operators() -> None:
    assert 1 + s == Add(terms=(ONE, s))
    assert 1 - s == Add(terms=(ONE, Mul(factors=(Const(value=-1.0, dimension=dimensionless()), s))))
    assert 2 * s == Mul(factors=(Const(value=2.0, dimension=dimensionless()), s))
    assert 1 / k == Div(numerator=ONE, denominator=k)
    assert 2**s == Pow(base=Const(value=2.0, dimension=dimensionless()), exponent=s)


def test_exponent_coercion_matches_the_constructor() -> None:
    assert (dose**2).exponent == Fraction(2)
    assert (dose**0.5).exponent == Pow(base=dose, exponent=0.5).exponent
    assert dimension(dose ** Fraction(1, 2)) == D.currency ** Fraction(1, 2)
    with pytest.raises(TypeError, match="bool is not an exponent"):
        dose**True
    # numpy scalars are numbers too, on both sides of every operator
    assert dose ** np.int64(2) == dose**2
    assert np.float64(2.0) * beta == 2.0 * beta


def test_operands_that_are_not_expressions_are_refused() -> None:
    with pytest.raises(TypeError, match="bool is not an expression operand"):
        beta * True
    with pytest.raises(TypeError, match="indexed by a Data column"):
        alpha[0]
    with pytest.raises(TypeError, match="str is not an expression operand"):
        beta + "dose"  # type: ignore[operator]


def test_structural_equality_is_untouched() -> None:
    """No ``__eq__`` overload: equality, hashing, and traversal still work on the tree."""
    built = beta * dose + s
    assert built == Add(terms=(Mul(factors=(beta, dose)), s))
    assert [p.name for p in params(built)] == ["beta", "s"]
    assert node_path(built, s).endswith("param")
    assert Spec.from_json(built.to_json()) == built
    assert hash(built) == hash(Add(terms=(Mul(factors=(beta, dose)), s)))


def test_sum_builds_the_tree_a_reader_expects() -> None:
    """``sum`` starts from ``0``; a bare zero is the additive identity in any dimension."""
    terms = (dose, dose, dose)
    assert sum(terms) == Add(terms=terms)
    assert dimension(sum(terms)) == D.currency
    assert dose + 0 == dose
    assert dose - 0 == dose
    assert 0 - dose == -dose
    # An explicit zero node is a term the author wrote, and stays.
    zero = Const(value=0.0, dimension=D.currency)
    assert dose + zero == Add(terms=(dose, zero))


def test_value_agrees_with_numpy() -> None:
    dose_v = np.array([1.0, 2.0, 4.0])
    env: dict[str, Any] = {"dose": dose_v}
    theta: dict[str, Any] = {"beta": 3.0, "k": 2.0, "s": 1.5}
    expr = beta * (dose / k) ** s / (1.0 + (dose / k) ** s)
    u = dose_v / 2.0
    expected = 3.0 * u**1.5 / (1.0 + u**1.5)
    np.testing.assert_allclose(value(expr, params=theta, data=env), expected)


def test_gather_reaches_the_panel() -> None:
    env: dict[str, Any] = {"unit_index": np.array([0, 2, 1])}
    theta: dict[str, Any] = {"alpha": np.array([10.0, 20.0, 30.0])}
    got = value(alpha[unit], params=theta, data=env)
    np.testing.assert_allclose(got, np.array([10.0, 30.0, 20.0]))
    assert dimension(alpha[unit]) == D.outcome
