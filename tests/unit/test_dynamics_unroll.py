"""Unrolling and simultaneity: does the compiled tree give the number the system means?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core import (
    D,
    Likelihood,
    Param,
    Prior,
    Unsupported,
    dimensionless,
    log_likelihood,
    value,
)
from axiom.dynamics import (
    Unrolled,
    Variable,
    affine_split,
    conditional_form,
    lagged_columns,
    parse_system,
    prepare_panel,
    reduced_form,
    solve_block,
    time_ref,
    to_model_spec,
    unroll,
    unrolled_edges,
)

NONE = dimensionless()


def stock_system(initial: float = 0.0):
    return parse_system(
        "stock = decay * stock[t-1] + inflow",
        variables=(
            Variable(name="stock", dimension=D.outcome, initial=initial),
            Variable(name="inflow", dimension=D.outcome, role="exogenous"),
        ),
        parameters=(Param(name="decay", dimension=NONE),),
        name="one-compartment",
    )


def market_system(momentum: bool = False):
    variables = (
        Variable(name="quantity", dimension=D.outcome),
        Variable(name="price", dimension=D.currency),
        Variable(name="income", dimension=D.currency, role="exogenous"),
        Variable(name="cost", dimension=D.currency, role="exogenous"),
    )
    parameters = (
        Param(name="a", dimension=D.outcome),
        Param(name="b", dimension=D.outcome / D.currency),
        Param(name="c", dimension=D.outcome / D.currency),
        Param(name="d", dimension=D.currency),
        Param(name="e", dimension=D.currency / D.outcome),
        Param(name="f", dimension=NONE),
    )
    text = """
    quantity = a - b * price + c * income
    price    = d + e * quantity + cost
    """
    if momentum:
        text = text.rstrip() + " + f * price[t-1]\n"
    return parse_system(text, variables=variables, parameters=parameters, name="market")


def scalar(expr, **kwargs) -> float:
    return float(np.ravel(value(expr, **kwargs))[0])


# -- the marginal form -------------------------------------------------------------


def test_the_unrolled_recursion_matches_the_loop_it_compiles() -> None:
    compiled = unroll(stock_system(), periods=5)
    assert isinstance(compiled, Unrolled)
    inflow = [1.0, 2.0, 0.5, 3.0, -1.0]
    data = {time_ref("inflow", t): np.array(v) for t, v in enumerate(inflow)}
    got = [
        scalar(compiled.expression("stock", t), data=data, params={"decay": 0.6}) for t in range(5)
    ]
    want, running = [], 0.0
    for v in inflow:
        running = 0.6 * running + v
        want.append(running)
    assert np.allclose(got, want)


def test_the_initial_condition_is_the_declared_one() -> None:
    compiled = unroll(stock_system(initial=10.0), periods=2)
    assert isinstance(compiled, Unrolled)
    data = {"inflow.t0": np.array(0.0), "inflow.t1": np.array(0.0)}
    assert scalar(
        compiled.expression("stock", 0), data=data, params={"decay": 0.5}
    ) == pytest.approx(5.0)
    assert scalar(
        compiled.expression("stock", 1), data=data, params={"decay": 0.5}
    ) == pytest.approx(2.5)


def test_the_marginal_form_reads_only_exogenous_columns() -> None:
    compiled = unroll(stock_system(), periods=3)
    assert isinstance(compiled, Unrolled)
    assert set(compiled.columns) == {"inflow.t0", "inflow.t1", "inflow.t2"}
    assert compiled.exact
    assert compiled.approximate_blocks == ()


def test_unrolling_zero_periods_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        unroll(stock_system(), periods=0)


def test_a_horizon_past_the_node_budget_is_reported_not_hung() -> None:
    system = parse_system(
        "y = exp(a * y[t-1]) + x",
        variables=(
            Variable(name="y", dimension=NONE),
            Variable(name="x", dimension=NONE, role="exogenous"),
        ),
        parameters=(Param(name="a", dimension=NONE),),
        name="explosive",
    )
    out = unroll(system, periods=40, max_nodes=200)
    assert isinstance(out, Unsupported)
    assert "node budget" in out.reason


# -- the conditional form ----------------------------------------------------------


def test_the_conditional_form_uses_lagged_columns() -> None:
    compiled = conditional_form(stock_system())
    assert isinstance(compiled, Unrolled)
    assert lagged_columns(stock_system()) == ("stock.l1",)
    assert set(compiled.columns) == {"stock.l1", "inflow"}
    got = scalar(
        compiled.expression("stock"),
        data={"stock.l1": np.array([2.0]), "inflow": np.array([1.0])},
        params={"decay": 0.5},
    )
    assert got == pytest.approx(2.0)


def test_a_latent_lag_has_no_column_and_says_so() -> None:
    system = parse_system(
        "state = decay * state[t-1] + shock",
        variables=(
            Variable(name="state", dimension=NONE, observed=False),
            Variable(name="shock", dimension=NONE, role="exogenous"),
        ),
        parameters=(Param(name="decay", dimension=NONE),),
    )
    out = conditional_form(system)
    assert isinstance(out, Unsupported)
    assert "latent" in out.reason
    assert out.missing == ("state",)


def test_prepare_panel_lags_within_each_unit() -> None:
    frame = pd.DataFrame(
        {
            "unit": ["a", "a", "b", "b"],
            "t": [0, 1, 0, 1],
            "stock": [1.0, 2.0, 5.0, 6.0],
            "inflow": [1.0, 1.0, 5.0, 1.0],
        }
    )
    out = prepare_panel(stock_system(initial=-1.0), frame, unit="unit", time="t")
    assert list(out["stock.l1"]) == [-1.0, 1.0, -1.0, 5.0]


def test_prepare_panel_names_the_missing_column() -> None:
    with pytest.raises(KeyError, match="missing columns"):
        prepare_panel(stock_system(), pd.DataFrame({"t": [0]}), time="t")


# -- simultaneity ------------------------------------------------------------------


def test_the_linear_reduced_form_is_exact() -> None:
    compiled = conditional_form(market_system())
    assert isinstance(compiled, Unrolled)
    assert [s.method for s in compiled.solutions] == ["linear"]
    assert compiled.exact
    theta = {"a": 10.0, "b": 2.0, "c": 0.5, "d": 1.0, "e": 0.25, "f": 0.0}
    data = {"income": np.array([4.0]), "cost": np.array([2.0])}
    q = scalar(compiled.expression("quantity"), data=data, params=theta)
    p = scalar(compiled.expression("price"), data=data, params=theta)
    assert q == pytest.approx(theta["a"] - theta["b"] * p + theta["c"] * 4.0)
    assert p == pytest.approx(theta["d"] + theta["e"] * q + 2.0)
    closed = (10.0 - 2.0 * (1.0 + 2.0) + 0.5 * 4.0) / (1 + 2.0 * 0.25)
    assert q == pytest.approx(closed)


def test_a_simultaneous_system_with_a_lag_matches_a_period_by_period_solve() -> None:
    compiled = unroll(market_system(momentum=True), periods=3)
    assert isinstance(compiled, Unrolled)
    theta = {"a": 10.0, "b": 2.0, "c": 0.5, "d": 1.0, "e": 0.25, "f": 0.3}
    income, cost = [4.0, 4.0, 5.0], [2.0, 2.5, 2.0]
    data = {}
    for t in range(3):
        data[f"income.t{t}"] = np.array(income[t])
        data[f"cost.t{t}"] = np.array(cost[t])
    previous, want_q, want_p = 0.0, [], []
    for t in range(3):
        q = (
            theta["a"]
            - theta["b"] * (theta["d"] + cost[t] + theta["f"] * previous)
            + theta["c"] * income[t]
        ) / (1 + theta["b"] * theta["e"])
        p = theta["d"] + theta["e"] * q + cost[t] + theta["f"] * previous
        want_q.append(q)
        want_p.append(p)
        previous = p
    got_q = [scalar(compiled.expression("quantity", t), data=data, params=theta) for t in range(3)]
    got_p = [scalar(compiled.expression("price", t), data=data, params=theta) for t in range(3)]
    assert np.allclose(got_q, want_q)
    assert np.allclose(got_p, want_p)


def test_a_nonlinear_block_needs_a_declared_number_of_sweeps() -> None:
    system = nonlinear_system()
    out = conditional_form(system)
    assert isinstance(out, Unsupported)
    assert "not affine" in out.reason
    assert out.missing == ("start", "sweeps")


def test_the_swept_block_converges_and_carries_its_residual() -> None:
    system = nonlinear_system()
    compiled = conditional_form(system, sweeps=8)
    assert isinstance(compiled, Unrolled)
    assert not compiled.exact
    assert [s.method for s in compiled.solutions] == ["fixed_point"]
    data = {"drive": np.array([1.0])}
    theta = {"k": 2.0, "g": 0.5}
    got = scalar(compiled.expression("response"), data=data, params=theta)
    truth = 0.0
    for _ in range(500):
        truth = 2.0 * (1.0 + 0.5 * truth) / (1.0 + 1.0 + 0.5 * truth)
    assert got == pytest.approx(truth, abs=1e-5)
    residuals = compiled.approximate_blocks[0].residuals
    assert set(residuals) == {"load", "response"}
    worst = max(abs(scalar(r, data=data, params=theta)) for r in residuals.values())
    assert worst < 1e-4


def test_too_many_sweeps_is_reported_rather_than_run() -> None:
    out = conditional_form(nonlinear_system(), sweeps=40)
    assert isinstance(out, Unsupported)
    assert "geometrically" in out.reason


def nonlinear_system():
    return parse_system(
        """
        response = k * load / (1 + load)
        load     = drive + g * response
        """,
        variables=(
            Variable(name="load", dimension=NONE),
            Variable(name="response", dimension=NONE),
            Variable(name="drive", dimension=NONE, role="exogenous"),
        ),
        parameters=(Param(name="k", dimension=NONE), Param(name="g", dimension=NONE)),
        name="saturating-feedback",
    )


def test_affine_split_sees_through_products_by_constants_only() -> None:
    from axiom.core import Apply, Data, Mul

    x = Data(name="x", dimension=NONE)
    y = Data(name="y", dimension=NONE)
    assert affine_split(Mul(factors=(Param(name="a", dimension=NONE), x)), ["x"]) is not None
    assert affine_split(Mul(factors=(x, y)), ["x", "y"]) is None
    assert affine_split(Apply(fn="exp", arg=x), ["x"]) is None
    assert affine_split(Apply(fn="exp", arg=y), ["x"]) is not None


def test_a_block_no_equation_determines_is_reported() -> None:
    from axiom.core import Data

    x = Data(name="x", dimension=NONE)
    out = reduced_form({"x": x}, {"x": NONE})
    assert isinstance(out, Unsupported)
    assert "singular" in out.reason


def test_solve_block_on_a_single_variable_is_a_substitution() -> None:
    from axiom.core import Data

    solution = solve_block({"y": Data(name="x", dimension=NONE)}, {"y": NONE}, simultaneous=False)
    assert not isinstance(solution, Unsupported)
    assert solution.method == "substitution"
    assert solution.exact


# -- the unrolled graph and the model spec -----------------------------------------


def test_the_reduced_form_graph_has_no_arrow_inside_a_simultaneous_block() -> None:
    edges = unrolled_edges(market_system(), periods=1)
    assert ("quantity.t0", "price.t0") not in edges
    assert ("price.t0", "quantity.t0") not in edges
    assert ("cost.t0", "quantity.t0") in edges
    structural = unrolled_edges(market_system(), periods=1, reduced=False)
    assert ("quantity.t0", "price.t0") in structural


def test_a_compiled_node_becomes_a_model_spec_that_evaluates() -> None:
    compiled = conditional_form(stock_system())
    assert isinstance(compiled, Unrolled)
    model = to_model_spec(
        compiled,
        "stock",
        likelihood=Likelihood(family="normal", scale="sigma"),
        priors={"decay": Prior(family="beta", hyper={"alpha": 2.0, "beta": 2.0})},
        extra_parameters=(
            Param(
                name="sigma",
                dimension=D.outcome,
                prior=Prior(family="halfnormal", hyper={"sigma": 1.0}),
            ),
        ),
    )
    assert model.outcome.name == "stock"
    assert {p.name for p in model.parameters} == {"decay", "sigma"}
    data = {
        "stock": np.array([2.0, 3.0]),
        "stock.l1": np.array([1.0, 2.0]),
        "inflow": np.array([1.0, 1.0]),
    }
    assert np.isfinite(log_likelihood(model, data, {"decay": 0.5, "sigma": 1.0}))


def test_a_missing_prior_names_the_parameter() -> None:
    compiled = conditional_form(stock_system())
    assert isinstance(compiled, Unrolled)
    with pytest.raises(KeyError, match="decay"):
        to_model_spec(
            compiled,
            "stock",
            likelihood=Likelihood(family="normal", scale="sigma"),
            priors={},
        )
