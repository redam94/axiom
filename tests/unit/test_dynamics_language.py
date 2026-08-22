"""The modelling language: parsing, validation, blocks, and what a bad system says."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import D, DimensionError, Param, dimensionless, value
from axiom.dynamics import (
    DynamicEquation,
    DynamicsError,
    DynamicSystem,
    LeadNotSupportedError,
    ParseError,
    Variable,
    block_order,
    lag_ref,
    parse_equations,
    parse_ref,
    parse_system,
    refs_in,
    strongly_connected_components,
    time_ref,
)
from axiom.dynamics.algebra import add, div, is_one, is_zero, mul, neg, one, size, substitute, zero

NONE = dimensionless()


def stock_variables() -> tuple[Variable, ...]:
    return (
        Variable(name="stock", dimension=D.outcome),
        Variable(name="inflow", dimension=D.outcome, role="exogenous"),
    )


def stock_system() -> DynamicSystem:
    return parse_system(
        "stock = decay * stock[t-1] + inflow",
        variables=stock_variables(),
        parameters=(Param(name="decay", dimension=NONE),),
        name="one-compartment",
    )


# -- references ---------------------------------------------------------------------


def test_lag_and_time_references_round_trip() -> None:
    assert lag_ref("y", 0) == "y"
    assert lag_ref("y", 2) == "y.l2"
    assert parse_ref("y.l2") == ("y", 2)
    assert parse_ref("y") == ("y", 0)
    assert time_ref("y", 3) == "y.t3"


def test_a_lead_is_refused_with_the_reason() -> None:
    with pytest.raises(LeadNotSupportedError, match="rational-expectations"):
        parse_ref("y.f1")


def test_an_absolute_reference_is_not_an_equation_reference() -> None:
    with pytest.raises(DynamicsError, match="absolute period"):
        parse_ref("y.t3")


def test_a_variable_name_may_not_contain_a_dot() -> None:
    with pytest.raises(ValueError, match="plain identifier"):
        Variable(name="y.l1", dimension=NONE)


# -- parsing ------------------------------------------------------------------------


def test_the_two_lag_spellings_agree() -> None:
    first = parse_equations(
        "stock = decay * stock[t-1]",
        variables=stock_variables(),
        parameters=(Param(name="decay", dimension=NONE),),
    )
    second = parse_equations(
        "stock = decay * stock.l1",
        variables=stock_variables(),
        parameters=(Param(name="decay", dimension=NONE),),
    )
    assert first[0].rhs == second[0].rhs


def test_precedence_and_functions_parse() -> None:
    variables = (
        Variable(name="y", dimension=NONE),
        Variable(name="x", dimension=NONE, role="exogenous"),
    )
    parameters = (Param(name="a", dimension=NONE), Param(name="b", dimension=NONE))
    equations = parse_equations(
        "y = a + b * exp(-x) / 2 - x^2", variables=variables, parameters=parameters
    )
    got = value(equations[0].rhs, data={"x": np.array([1.5])}, params={"a": 2.0, "b": 4.0})
    assert np.allclose(got, 2.0 + 4.0 * np.exp(-1.5) / 2 - 1.5**2)


def test_an_unknown_name_names_both_lists() -> None:
    with pytest.raises(ParseError, match="neither a declared variable"):
        parse_equations("y = wobble", variables=(Variable(name="y", dimension=NONE),))


def test_a_parameter_may_not_be_lagged() -> None:
    with pytest.raises(ParseError, match="does not vary with time"):
        parse_equations(
            "y = a[t-1]",
            variables=(Variable(name="y", dimension=NONE),),
            parameters=(Param(name="a", dimension=NONE),),
        )


def test_a_lead_in_the_text_is_refused() -> None:
    with pytest.raises(ParseError, match="forward-looking"):
        parse_equations(
            "y = x[t+1]",
            variables=(
                Variable(name="y", dimension=NONE),
                Variable(name="x", dimension=NONE, role="exogenous"),
            ),
        )


def test_an_exponent_must_be_a_rational_constant() -> None:
    variables = (
        Variable(name="y", dimension=NONE),
        Variable(name="x", dimension=NONE, role="exogenous"),
    )
    parsed = parse_equations("y = x^(1/2)", variables=variables)
    assert value(parsed[0].rhs, data={"x": np.array([9.0])}) == pytest.approx(3.0)
    with pytest.raises(ParseError, match="rational constant"):
        parse_equations(
            "y = x^x",
            variables=variables,
        )


def test_a_line_without_an_equals_is_an_error_naming_the_line() -> None:
    with pytest.raises(ParseError, match="line 1"):
        parse_equations("y + 1", variables=(Variable(name="y", dimension=NONE),))


def test_comments_and_blank_lines_are_skipped() -> None:
    equations = parse_equations(
        "# a comment\n\ny = x  # trailing\n",
        variables=(
            Variable(name="y", dimension=NONE),
            Variable(name="x", dimension=NONE, role="exogenous"),
        ),
    )
    assert len(equations) == 1


# -- system validation ---------------------------------------------------------------


def test_the_system_checks_dimensions_across_the_equals() -> None:
    with pytest.raises(DimensionError, match="right-hand side"):
        parse_system(
            "y = x",
            variables=(
                Variable(name="y", dimension=D.outcome),
                Variable(name="x", dimension=D.currency, role="exogenous"),
            ),
        )


def test_an_exogenous_variable_may_not_have_an_equation() -> None:
    with pytest.raises(ValueError, match="exogenous"):
        parse_system(
            "x = y",
            variables=(
                Variable(name="y", dimension=NONE),
                Variable(name="x", dimension=NONE, role="exogenous"),
            ),
        )


def test_an_endogenous_variable_needs_an_equation() -> None:
    with pytest.raises(ValueError, match="no equation"):
        DynamicSystem(
            variables=(Variable(name="y", dimension=NONE), Variable(name="z", dimension=NONE)),
            equations=(),
        )


def test_the_system_reports_its_shape() -> None:
    system = stock_system()
    assert system.endogenous == ("stock",)
    assert system.exogenous == ("inflow",)
    assert system.max_lag == 1
    assert [p.name for p in system.parameters] == ["decay"]
    assert system.equation("stock").refs == (("stock", 1), ("inflow", 0))
    assert refs_in(system.equation("stock").rhs) == (("stock", 1), ("inflow", 0))
    assert system.variable("inflow").role == "exogenous"
    with pytest.raises(KeyError):
        system.equation("inflow")


def test_a_duplicated_equation_is_refused() -> None:
    equation = stock_system().equation("stock")
    with pytest.raises(ValueError, match="more than one equation"):
        DynamicSystem(variables=stock_variables(), equations=(equation, equation))


def test_an_equation_may_not_be_written_with_a_lagged_target() -> None:
    with pytest.raises(ParseError, match="takes no lag"):
        parse_equations("y[t-1] = y", variables=(Variable(name="y", dimension=NONE),))


def test_a_reference_dimension_must_match_the_declaration() -> None:
    from axiom.core import Data

    with pytest.raises(DimensionError, match="is declared"):
        DynamicSystem(
            variables=(Variable(name="y", dimension=D.outcome),),
            equations=(DynamicEquation(target="y", rhs=Data(name="y.l1", dimension=D.currency)),),
        )


# -- blocks --------------------------------------------------------------------------


def test_a_recursive_system_has_singleton_blocks() -> None:
    order = block_order(stock_system())
    assert order.recursive
    assert order.order == ("stock",)
    assert order.largest_block == 1
    assert order.simultaneous_blocks == ()


def test_a_cycle_becomes_one_simultaneous_block() -> None:
    system = parse_system(
        """
        q = a - b * p
        p = c + e * q
        """,
        variables=(Variable(name="q", dimension=NONE), Variable(name="p", dimension=NONE)),
        parameters=tuple(Param(name=n, dimension=NONE) for n in "abce"),
    )
    order = block_order(system)
    assert not order.recursive
    assert order.blocks[0].variables == ("p", "q")
    assert order.blocks[0].simultaneous
    assert system.cycles_through() == ("p", "q")


def test_blocks_are_ordered_so_a_dependency_is_solved_first() -> None:
    system = parse_system(
        """
        first = x
        second = first + x
        """,
        variables=(
            Variable(name="first", dimension=NONE),
            Variable(name="second", dimension=NONE),
            Variable(name="x", dimension=NONE, role="exogenous"),
        ),
    )
    assert block_order(system).order == ("first", "second")


def test_a_self_reference_at_lag_zero_is_simultaneous() -> None:
    system = parse_system(
        "y = 0.5 * y + x",
        variables=(
            Variable(name="y", dimension=NONE),
            Variable(name="x", dimension=NONE, role="exogenous"),
        ),
    )
    order = block_order(system)
    assert order.blocks[0].simultaneous


def test_tarjan_finds_components_and_orders_them() -> None:
    components = strongly_connected_components(
        ("a", "b", "c", "d"), (("a", "b"), ("b", "a"), ("b", "c"), ("c", "d"))
    )
    assert ("a", "b") in components
    assert components.index(("d",)) < components.index(("a", "b"))


# -- algebra -------------------------------------------------------------------------


def test_the_constructors_fold_identities_without_changing_the_value() -> None:
    from axiom.core import Data

    x = Data(name="x", dimension=NONE)
    assert add(x, zero(NONE)) == x
    assert mul(x, one()) == x
    assert is_zero(mul(x, zero(NONE)))
    assert is_one(one())
    assert neg(neg(x)) == x
    assert div(x, one()) == x
    assert size(x) == 1


def test_substitution_replaces_by_name() -> None:
    from axiom.core import Data

    x = Data(name="x", dimension=NONE)
    y = Data(name="y", dimension=NONE)
    assert substitute(add(x, y), {"x": y}) == add(y, y)
