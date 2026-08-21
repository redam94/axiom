from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from axiom.core import (
    Add,
    Apply,
    Const,
    Convolve,
    D,
    Data,
    DimensionError,
    Div,
    Equation,
    Link,
    Mul,
    ODESystem,
    Opaque,
    Param,
    Pow,
    Prior,
    Spec,
    SupportsForward,
    System,
    Unsupported,
    causal_convolve,
    check,
    children,
    data_names,
    dimension,
    dimensionless,
    latex,
    latex_or_unsupported,
    node_path,
    params,
    value,
    walk,
)

dose = Data(name="dose", dimension=D.currency)
k = Param(
    name="k", dimension=D.currency, prior=Prior(family="lognormal", hyper={"mu": 3, "sigma": 1})
)
s = Param(name="s", dimension=dimensionless())
beta = Param(name="beta", dimension=D.outcome)
x = Div(numerator=dose, denominator=k)
hill = Mul(
    factors=(
        beta,
        Div(
            numerator=Pow(base=x, exponent=s),
            denominator=Add(
                terms=(Const(value=1, dimension=dimensionless()), Pow(base=x, exponent=s))
            ),
        ),
    )
)


def test_roundtrip_and_traversal() -> None:
    assert Spec.from_json(hill.to_json()) == hill
    assert [p.name for p in params(hill)] == ["beta", "k", "s"]
    assert data_names(hill) == ("dose",)
    assert k.is_shape is False and s.is_shape is True
    paths = [p for p, _ in walk(hill)]
    assert paths[0] == "mul" and paths[1] == "mul[0].param"
    assert node_path(hill, s).endswith("param")
    assert children(Pow(base=dose, exponent=Fraction(1, 2))) == (dose,)
    with pytest.raises(KeyError):
        node_path(hill, Data(name="zzz", dimension=D.time))


def test_pow_exponent_coercion() -> None:
    assert Pow(base=dose, exponent=0.5).exponent == Fraction(1, 2)
    assert Pow(base=dose, exponent="1/3").exponent == Fraction(1, 3)
    assert Pow(base=x, exponent=s).exponent == s
    with pytest.raises(ValueError):
        Pow(base=dose, exponent=True)


def test_dimension_rules() -> None:
    assert dimension(hill) == D.outcome
    assert dimension(Pow(base=dose, exponent=Fraction(1, 2))) == D.currency ** Fraction(1, 2)
    assert dimension(Pow(base=Mul(factors=(dose, dose)), exponent="1/2")) == D.currency
    assert check(hill, D.outcome) == D.outcome
    with pytest.raises(DimensionError, match="declared"):
        check(hill, D.time)
    with pytest.raises(DimensionError, match=r"add: terms of a sum"):
        dimension(Add(terms=(dose, beta)))
    with pytest.raises(DimensionError, match="variable exponent"):
        dimension(Pow(base=x, exponent=k))
    with pytest.raises(DimensionError, match="weights must be dimensionless"):
        dimension(Convolve(signal=dose, kernel=k))
    assert dimension(Opaque(name="f", inputs=(dose,), dimension=D.time)) == D.time


def test_value_matches_numpy() -> None:
    d = np.array([0.0, 50.0, 100.0])
    out = value(hill, data={"dose": d}, params={"k": 50.0, "s": 2.0, "beta": 10.0})
    np.testing.assert_allclose(out, 10 * (d / 50) ** 2 / (1 + (d / 50) ** 2))
    # params as draws broadcast
    out = value(
        hill,
        data={"dose": d[None, :]},
        params={"k": np.array([[25.0], [50.0]]), "s": 1.0, "beta": 1.0},
    )
    assert out.shape == (2, 3)
    assert value(Apply(fn="sigmoid", arg=Const(value=0.0, dimension=dimensionless()))) == 0.5
    assert value(Link(fn="log", arg=Const(value=1.0, dimension=dimensionless()))) == 0.0
    with pytest.raises(KeyError, match="'dose'"):
        value(hill, params={"k": 1, "s": 1, "beta": 1})
    with pytest.raises(KeyError, match="'beta'"):
        value(hill, data={"dose": d}, params={"k": 1, "s": 1})


def test_convolve_and_opaque() -> None:
    w = Opaque(
        name="geom",
        inputs=(Param(name="lam", dimension=dimensionless()),),
        dimension=dimensionless(),
    )
    carry = Convolve(signal=dose, kernel=w)
    assert dimension(carry) == D.currency
    y = value(
        carry,
        data={"dose": np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])},
        params={"lam": 0.5},
        opaque={"geom": lambda lam: lam ** np.arange(4)},
    )
    np.testing.assert_allclose(y, [[1, 0.5, 0.25, 0.125], [0, 1, 0.5, 0.25]])
    np.testing.assert_allclose(causal_convolve(np.ones(3), np.array([1.0, 1.0])), [1, 2, 2])
    with pytest.raises(KeyError, match="not registered"):
        value(carry, data={"dose": np.ones(2)}, params={"lam": 0.5})
    assert isinstance(latex_or_unsupported(carry), Unsupported)
    assert "geom" in latex(carry)


def test_equation_system_ode() -> None:
    y = Data(name="y", dimension=D.outcome)
    t = Data(name="t", dimension=D.time)
    eq = Equation(lhs=y, rhs=hill, name="response")
    assert dimension(eq) == D.outcome
    with pytest.raises(DimensionError, match="'response' does not balance"):
        dimension(Equation(lhs=t, rhs=hill, name="response"))
    sysm = System(equations=(eq, Equation(lhs=Data(name="z", dimension=D.currency), rhs=k)))
    out = value(
        sysm, data={"dose": np.array([50.0, 100.0])}, params={"k": 50.0, "s": 1.0, "beta": 2.0}
    )
    assert out.shape == (2, 2) and np.allclose(out[1], 50.0)
    ode = ODESystem(
        states=(Data(name="S", dimension=D.outcome),),
        rhs=(
            Mul(
                factors=(Param(name="r", dimension=D.time**-1), Data(name="S", dimension=D.outcome))
            ),
        ),
        time=t,
    )
    assert dimension(ode) == D.outcome
    assert "\\frac{d S}{d t}" in latex(ode)
    with pytest.raises(NotImplementedError):
        value(ode, data={"S": 1.0, "t": 0.0}, params={"r": 0.1})
    with pytest.raises(ValueError, match="right-hand sides"):
        ODESystem(states=(y,), rhs=(y, y), time=t)


def test_latex_renders() -> None:
    tex = latex(hill)
    assert "\\beta" in tex and "\\frac" in tex and "^{s}" in tex
    assert latex(Pow(base=dose, exponent=Fraction(1, 2))) == "{\\mathrm{dose}}^{1/2}"
    assert latex(Apply(fn="exp", arg=s)) == "\\exp\\left(s\\right)"
    assert latex(Link(fn="identity", arg=s)) == "s"
    assert "cases" in latex(
        System(equations=(Equation(lhs=Data(name="y", dimension=D.outcome), rhs=hill),))
    )


def test_supports_forward_protocol() -> None:
    class Surface:
        expr = hill

        def forward(self, dose, theta):  # type: ignore[no-untyped-def]
            return value(self.expr, data=dose, params=theta)

    assert isinstance(Surface(), SupportsForward)
