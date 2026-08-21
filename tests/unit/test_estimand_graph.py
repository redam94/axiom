from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

import numpy as np
import pytest
from _factories import _estimand

from axiom.core import (
    Add,
    Apply,
    Const,
    Convolve,
    D,
    Data,
    DimensionError,
    Div,
    Expr,
    Gather,
    Intervention,
    Link,
    Mul,
    Opaque,
    Outcome,
    Param,
    Population,
    Pow,
    Reduce,
    Spec,
    TimeWindow,
    Treatment,
    Unsupported,
    dimension,
    dimensionless,
    value,
    walk,
)
from axiom.estimands import Estimand, Level, Quantity
from axiom.estimands.graph import check_estimand_dimension, estimand_expr, substitute
from axiom.estimands.registry import standard_estimands
from axiom.sim import arms_world
from axiom.surface.forward import marginal_expr

DOSE = Data(name="dose", dimension=D.currency)
K = Param(name="k", dimension=D.currency)
S = Param(name="s", dimension=dimensionless())
BETA = Param(name="beta", dimension=D.outcome)
ONE = Const(value=1.0, dimension=dimensionless())
PARAMS = {"beta": 2.0, "k": 50.0, "s": 1.5}
DRAWS = {"beta": np.array([1.0, 2.0, 3.0]), "k": np.array([40.0, 50.0, 60.0]), "s": 1.5}


def _u(dose: Expr) -> Div:
    return Div(numerator=dose, denominator=K)


def _hill(dose: Expr = DOSE) -> Expr:
    """``beta · u^s / (1 + u^s)``."""
    us = Pow(base=_u(dose), exponent=S)
    return Mul(factors=(BETA, Div(numerator=us, denominator=Add(terms=(ONE, us)))))


def _hill_derivative(dose: Expr = DOSE) -> Expr:
    """``beta · s · u^(s-1) / (k · (1 + u^s)^2)``."""
    us = Pow(base=_u(dose), exponent=S)
    u_sm1 = Pow(
        base=_u(dose), exponent=Add(terms=(S, Const(value=-1.0, dimension=dimensionless())))
    )
    return Div(
        numerator=Mul(factors=(BETA, S, u_sm1)),
        denominator=Mul(factors=(K, Pow(base=Add(terms=(ONE, us)), exponent=Fraction(2)))),
    )


def _mean_at(dose: float, params: Mapping[str, object] = PARAMS) -> np.ndarray:
    return value(_hill(), data={"dose": dose}, params=params)


def _kind(kind: str, **over: object) -> Estimand:
    base: dict[str, object] = {
        "quantity": Quantity(kind=kind),  # type: ignore[arg-type]
        "dimension": {
            "contrast": D.outcome,
            "ratio": D.outcome / D.currency,
            "marginal": D.outcome / D.currency,
            "elasticity": dimensionless(),
            "area": D.outcome * D.currency,
        }[kind],
    }
    if kind in ("marginal", "elasticity"):
        base["reference"] = None
    base.update(over)
    return _estimand(**base)


def _iv(level: float, mode: str = "set", **over: object) -> Intervention:
    return Intervention(doses={"fertilizer": level}, mode=mode, **over)  # type: ignore[arg-type]


def _carried_hill() -> tuple[Expr, dict[str, object]]:
    """A Hill response of a 3-lag geometric carryover of the dose, with its parameters."""
    lags = Const(value=(0.0, 1.0, 2.0), dimension=dimensionless())
    lam = Param(name="lam", dimension=dimensionless())
    raw = Pow(base=lam, exponent=lags)
    w = Div(numerator=raw, denominator=Reduce(op="sum", arg=raw, keepdims=True))
    return _hill(Convolve(signal=DOSE, kernel=w)), {**PARAMS, "lam": 0.5}


# -- substitute ---------------------------------------------------------------------------


def test_substitute_rebuilds_and_round_trips() -> None:
    mean = _hill()
    at = substitute(mean, {"dose": Const(value=100.0, dimension=D.currency)})
    assert not any(isinstance(n, Data) for _, n in walk(at))
    assert Spec.from_json(at.to_json()) == at
    assert dimension(at) == dimension(mean)
    np.testing.assert_allclose(value(at, params=PARAMS), _mean_at(100.0), rtol=1e-12)
    # untouched subtrees are the very same objects; an empty map returns the tree itself
    assert substitute(mean, {}) is mean
    assert substitute(mean, {"other": ONE}) is mean
    other = Data(name="rain", dimension=dimensionless())
    both = Add(terms=(mean, Mul(factors=(BETA, other))))
    sub = substitute(both, {"rain": Const(value=0.5, dimension=dimensionless())})
    assert isinstance(sub, Add) and sub.terms[0] is mean


def test_substitute_reaches_every_node_type() -> None:
    idx = Data(name="unit", dimension=dimensionless())
    lags = Const(value=(0.0, 1.0, 2.0), dimension=dimensionless())
    lam = Data(name="lam", dimension=dimensionless())
    raw = Pow(base=lam, exponent=lags)
    w = Div(numerator=raw, denominator=Reduce(op="sum", arg=raw, keepdims=True))
    tree = Add(
        terms=(
            Gather(source=Param(name="alpha", dimension=D.outcome, shape=(2,)), index=idx),
            Mul(
                factors=(
                    Param(name="rate", dimension=D.outcome / D.currency),
                    Convolve(signal=DOSE, kernel=w),
                )
            ),
        )
    )
    new_idx = Data(name="unit2", dimension=dimensionless())
    sub = substitute(tree, {"unit": new_idx, "lam": Const(value=0.5, dimension=dimensionless())})
    names = {n.name for _, n in walk(sub) if isinstance(n, Data)}
    assert names == {"unit2", "dose"}
    assert dimension(sub) == D.outcome
    data = {"unit2": np.array([0, 1, 1]), "dose": np.array([[1.0, 2.0, 3.0]] * 3)}
    params = {"alpha": np.array([10.0, 20.0]), "rate": 2.0}
    got = value(sub, data=data, params=params)
    ref = value(tree, data={**data, "unit": data["unit2"], "lam": 0.5}, params=params)
    np.testing.assert_allclose(got, ref, rtol=1e-12)
    with pytest.raises(TypeError, match="Gather index"):
        substitute(tree, {"unit": ONE})


def test_substitute_reaches_apply_link_and_opaque() -> None:
    u = _u(DOSE)
    tree = Add(
        terms=(
            Mul(factors=(BETA, Apply(fn="tanh", arg=u))),
            Mul(factors=(BETA, Link(fn="log", arg=Add(terms=(ONE, u))))),
            Opaque(name="user", inputs=(DOSE, K), dimension=D.outcome),
        )
    )
    at = substitute(tree, {"dose": Const(value=25.0, dimension=D.currency)})
    assert not any(isinstance(n, Data) for _, n in walk(at))
    assert isinstance(at, Add)
    assert isinstance(at.terms[0].factors[1], Apply) and at.terms[0].factors[1].fn == "tanh"  # type: ignore[union-attr]
    assert isinstance(at.terms[1].factors[1], Link) and at.terms[1].factors[1].fn == "log"  # type: ignore[union-attr]
    assert isinstance(at.terms[2], Opaque) and at.terms[2].name == "user"
    assert at.terms[2].dimension == D.outcome
    assert Spec.from_json(at.to_json()) == at
    opaque = {"user": lambda x, k: x / k}
    np.testing.assert_allclose(
        value(at, params=PARAMS, opaque=opaque),
        value(tree, data={"dose": 25.0}, params=PARAMS, opaque=opaque),
        rtol=1e-12,
    )
    # an Apply/Link/Opaque whose argument is untouched is returned as the same object
    assert substitute(tree, {"rain": ONE}) is tree


# -- estimand_expr ------------------------------------------------------------------------


def test_contrast_is_the_difference_of_means() -> None:
    e = _kind("contrast")
    expr = estimand_expr(e, mean=_hill(), treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    assert check_estimand_dimension(e, expr) == D.outcome
    np.testing.assert_allclose(
        value(expr, params=PARAMS), _mean_at(100.0) - _mean_at(0.0), rtol=1e-12
    )
    # draws broadcast: one value per draw
    np.testing.assert_allclose(
        value(expr, params=DRAWS), _mean_at(100.0, DRAWS) - _mean_at(0.0, DRAWS), rtol=1e-12
    )
    assert Spec.from_json(expr.to_json()) == expr
    assert not any(isinstance(n, Data) for _, n in walk(expr))


def test_ratio_divides_by_the_dose_difference() -> None:
    e = _kind(
        "ratio",
        intervention=Intervention(doses={"fertilizer": 120.0}, version="granular"),
        reference=Intervention(doses={"fertilizer": 20.0}, version="granular"),
    )
    expr = estimand_expr(e, mean=_hill(), treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    assert check_estimand_dimension(e, expr) == D.outcome / D.currency
    np.testing.assert_allclose(
        value(expr, params=PARAMS), (_mean_at(120.0) - _mean_at(20.0)) / 100.0, rtol=1e-12
    )
    with pytest.raises(ValueError, match="dose difference is zero"):
        estimand_expr(
            _kind("ratio", reference=Intervention(doses={"fertilizer": 100.0}, version="granular")),
            mean=_hill(),
            treatment_data=DOSE,
        )


def test_marginal_and_elasticity_use_the_supplied_derivative() -> None:
    m = _kind("marginal")
    assert isinstance(estimand_expr(m, mean=_hill(), treatment_data=DOSE), Unsupported)
    un = estimand_expr(m, mean=_hill(), treatment_data=DOSE)
    assert isinstance(un, Unsupported) and "derivative" in un.missing
    assert "ResponseKernel.derivative" in un.reason

    expr = estimand_expr(m, mean=_hill(), treatment_data=DOSE, derivative=_hill_derivative())
    assert not isinstance(expr, Unsupported)
    assert check_estimand_dimension(m, expr) == D.outcome / D.currency
    h = 1e-4
    fd = (_mean_at(100.0 + h) - _mean_at(100.0 - h)) / (2 * h)
    np.testing.assert_allclose(value(expr, params=PARAMS), fd, rtol=1e-7)

    el = _kind("elasticity")
    assert isinstance(estimand_expr(el, mean=_hill(), treatment_data=DOSE), Unsupported)
    expr = estimand_expr(el, mean=_hill(), treatment_data=DOSE, derivative=_hill_derivative())
    assert not isinstance(expr, Unsupported)
    assert check_estimand_dimension(el, expr).is_dimensionless
    np.testing.assert_allclose(value(expr, params=PARAMS), fd * 100.0 / _mean_at(100.0), rtol=1e-7)
    # Hill elasticity is s / (1 + u^s) in closed form
    u = 100.0 / PARAMS["k"]
    np.testing.assert_allclose(
        value(expr, params=PARAMS), PARAMS["s"] / (1 + u ** PARAMS["s"]), rtol=1e-10
    )


def test_unsupported_paths_are_typed() -> None:
    area = estimand_expr(_kind("area"), mean=_hill(), treatment_data=DOSE)
    assert isinstance(area, Unsupported) and "quadrature" in area.reason
    log = estimand_expr(
        _kind("contrast", quantity=Quantity(kind="contrast", scale="log")),
        mean=_hill(),
        treatment_data=DOSE,
    )
    assert isinstance(log, Unsupported) and "log" in log.reason
    cond = estimand_expr(
        _kind("contrast", conditioning=("soil",)), mean=_hill(), treatment_data=DOSE
    )
    assert isinstance(cond, Unsupported) and "conditions on" in cond.reason
    multi = estimand_expr(
        _kind(
            "contrast",
            intervention=Intervention(
                doses={"fertilizer": 100.0, "water": 2.0}, version="granular"
            ),
        ),
        mean=_hill(),
        treatment_data=DOSE,
    )
    assert isinstance(multi, Unsupported) and "water" in multi.reason
    for u in (area, log, cond, multi):
        assert u.detail["estimand"] == "lift_at_100" and not u
        assert Spec.from_json(u.to_json()) == u


def test_intervention_support_window_is_unsupported() -> None:
    """A dose applied over part of the horizon is a time path; the per-row tree cannot say it."""
    window = TimeWindow(start=2, stop=5)
    on_iv = estimand_expr(
        _kind("contrast", intervention=_iv(100.0, window=window)), mean=_hill(), treatment_data=DOSE
    )
    assert isinstance(on_iv, Unsupported) and on_iv.missing == ("time_window",)
    assert "intervention" in on_iv.reason and "[2, 5)" in on_iv.reason
    assert "evaluate.realize" in on_iv.reason
    on_ref = estimand_expr(
        _kind("contrast", reference=_iv(0.0, window=window)), mean=_hill(), treatment_data=DOSE
    )
    assert isinstance(on_ref, Unsupported) and on_ref.missing == ("time_window",)
    assert "reference" in on_ref.reason
    # a marginal with a windowed intervention is refused the same way
    m = estimand_expr(
        _kind("marginal", intervention=_iv(100.0, window=window)),
        mean=_hill(),
        treatment_data=DOSE,
        derivative=_hill_derivative(),
    )
    assert isinstance(m, Unsupported) and m.missing == ("time_window",)
    # and the estimand's own reporting window is not the intervention's support: it is fine
    ok = estimand_expr(
        _kind("contrast", window=TimeWindow(start=3, stop=4)), mean=_hill(), treatment_data=DOSE
    )
    assert not isinstance(ok, Unsupported)


def test_malformed_requests_raise() -> None:
    with pytest.raises(ValueError, match="does not appear in the mean tree"):
        estimand_expr(
            _kind("contrast"), mean=_hill(), treatment_data=Data(name="x", dimension=D.currency)
        )
    with pytest.raises(DimensionError, match="carries"):
        estimand_expr(
            _kind("contrast"),
            mean=_hill(Data(name="dose", dimension=D.time)),
            treatment_data=Data(name="dose", dimension=D.time),
        )
    with pytest.raises(ValueError, match="does not appear"):
        estimand_expr(_kind("contrast"), mean=_hill(), treatment_data=DOSE, reference={"rain": 1.0})
    with pytest.raises(ValueError, match="pins the treatment column"):
        estimand_expr(_kind("contrast"), mean=_hill(), treatment_data=DOSE, reference={"dose": 1.0})


def test_tree_column_dimension_must_match_treatment_data() -> None:
    """Matching the column by name alone is not enough: the tree's node must carry the same
    dimension as ``treatment_data`` (and so the treatment)."""
    bad = _hill(Data(name="dose", dimension=D.time))
    with pytest.raises(DimensionError, match=r"mean tree declares column 'dose' with dimension T"):
        estimand_expr(_kind("contrast"), mean=bad, treatment_data=DOSE)
    with pytest.raises(DimensionError, match="derivative tree declares column 'dose'"):
        estimand_expr(
            _kind("marginal"),
            mean=_hill(),
            treatment_data=DOSE,
            derivative=_hill_derivative(Data(name="dose", dimension=D.time)),
        )


def test_mean_and_derivative_must_carry_the_estimands_dimensions() -> None:
    per_dose = Div(numerator=_hill(), denominator=K)  # Y / $, not the outcome Y
    with pytest.raises(DimensionError, match=r"mean tree derives to \$\^-1·Y") as info:
        estimand_expr(_kind("contrast"), mean=per_dose, treatment_data=DOSE)
    assert "'yield_total'" in str(info.value) and "dimension Y" in str(info.value)
    with pytest.raises(DimensionError, match="derivative tree derives to Y") as info:
        estimand_expr(_kind("marginal"), mean=_hill(), treatment_data=DOSE, derivative=_hill())
    assert "d yield_total / d fertilizer" in str(info.value)
    assert "$^-1·Y" in str(info.value)
    # an outcome declared in another dimension is caught the same way
    e = _kind(
        "contrast",
        outcome=Outcome(name="revenue", dimension=D.currency),
        dimension=D.currency,
    )
    with pytest.raises(DimensionError, match="derives to Y but outcome 'revenue'"):
        estimand_expr(e, mean=_hill(), treatment_data=DOSE)


def test_pinning_a_gather_index_is_a_value_error_naming_the_estimand() -> None:
    idx = Data(name="unit", dimension=dimensionless())
    mean = Add(
        terms=(
            Gather(source=Param(name="alpha", dimension=D.outcome, shape=(2,)), index=idx),
            _hill(),
        )
    )
    with pytest.raises(ValueError, match=r"estimand 'lift_at_100': \['unit'\] index a Gather"):
        estimand_expr(_kind("contrast"), mean=mean, treatment_data=DOSE, reference={"unit": 0.0})
    # the index column can never be the treatment column either
    with pytest.raises(ValueError, match="index a Gather"):
        estimand_expr(
            _kind(
                "contrast",
                treatment=Treatment(name="fertilizer", dimension=dimensionless()),
                dimension=D.outcome,
            ),
            mean=Add(
                terms=(
                    Gather(
                        source=Param(name="alpha", dimension=D.outcome, shape=(2,)),
                        index=Data(name="dose", dimension=dimensionless()),
                    ),
                    Mul(factors=(BETA, Data(name="dose", dimension=dimensionless()))),
                )
            ),
            treatment_data=Data(name="dose", dimension=dimensionless()),
        )
    # with the index left alone, the per-unit intercept cancels in the contrast
    expr = estimand_expr(_kind("contrast"), mean=mean, treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    got = value(expr, data={"unit": np.array([0, 1, 1])}, params={**PARAMS, "alpha": [5.0, 9.0]})
    np.testing.assert_allclose(got, np.full(3, _mean_at(100.0) - _mean_at(0.0)), rtol=1e-12)


def test_check_estimand_dimension_names_both_sides() -> None:
    e = _kind("ratio")
    with pytest.raises(DimensionError, match=r"declares dimension \$\^-1·Y"):
        check_estimand_dimension(e, _hill())
    with pytest.raises(DimensionError, match="derives to Y"):
        check_estimand_dimension(e, _hill())


def test_reference_pins_other_columns() -> None:
    rain = Data(name="rain", dimension=dimensionless())
    gamma = Param(name="gamma", dimension=D.outcome)
    mean = Add(terms=(_hill(), Mul(factors=(gamma, rain))))
    e = _kind("contrast")
    expr = estimand_expr(e, mean=mean, treatment_data=DOSE, reference={"rain": 0.7})
    assert not isinstance(expr, Unsupported)
    assert not any(isinstance(n, Data) for _, n in walk(expr))
    params = {**PARAMS, "gamma": 3.0}
    # the covariate cancels in a contrast; it does not in the mean itself
    np.testing.assert_allclose(
        value(expr, params=params), _mean_at(100.0) - _mean_at(0.0), rtol=1e-12
    )
    el = _kind("elasticity")
    expr = estimand_expr(
        el, mean=mean, treatment_data=DOSE, reference={"rain": 0.7}, derivative=_hill_derivative()
    )
    assert not isinstance(expr, Unsupported)
    y = value(mean, data={"dose": 100.0, "rain": 0.7}, params=params)
    slope = value(_hill_derivative(), data={"dose": 100.0}, params=params)
    np.testing.assert_allclose(value(expr, params=params), slope * 100.0 / y, rtol=1e-12)


def test_scale_and_shift_modes_keep_the_observed_dose_path() -> None:
    doses = np.array([0.0, 25.0, 50.0, 200.0])
    sc = _kind("contrast", intervention=_iv(2.0, "scale"), reference=_iv(1.0, "scale"))
    expr = estimand_expr(sc, mean=_hill(), treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    assert check_estimand_dimension(sc, expr) == D.outcome
    got = value(expr, data={"dose": doses}, params=PARAMS)
    want = value(_hill(), data={"dose": 2 * doses}, params=PARAMS) - value(
        _hill(), data={"dose": doses}, params=PARAMS
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)

    sh = _kind("contrast", intervention=_iv(30.0, "shift"), reference=_iv(0.0, "shift"))
    expr = estimand_expr(sh, mean=_hill(), treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    got = value(expr, data={"dose": doses}, params=PARAMS)
    want = value(_hill(), data={"dose": doses + 30.0}, params=PARAMS) - value(
        _hill(), data={"dose": doses}, params=PARAMS
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)

    # a shift-versus-shift ratio has a row-constant dose difference: it is the per-row
    # contrast over that constant, and its row average is Δagg(Y)/Δagg(X)
    ratio = _kind("ratio", intervention=_iv(30.0, "shift"), reference=_iv(0.0, "shift"))
    expr = estimand_expr(ratio, mean=_hill(), treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    assert check_estimand_dimension(ratio, expr) == D.outcome / D.currency
    got = value(expr, data={"dose": doses}, params=PARAMS)
    np.testing.assert_allclose(got, want / 30.0, rtol=1e-12)
    np.testing.assert_allclose(got.mean(), want.sum() / (30.0 * doses.size), rtol=1e-12)


def test_ratio_with_a_row_varying_dose_difference_is_unsupported() -> None:
    """0002.20: a ratio is Δagg(Y)/Δagg(X); per-row Δmean/Δdose is a different number when the
    dose difference depends on the observed dose."""
    for iv, ref in (
        (_iv(2.0, "scale"), _iv(1.0, "scale")),
        (_iv(100.0, "set"), _iv(1.0, "scale")),
        (_iv(100.0, "set"), _iv(0.0, "shift")),
        (_iv(30.0, "shift"), _iv(2.0, "scale")),
    ):
        e = _kind("ratio", intervention=iv, reference=ref)
        un = estimand_expr(e, mean=_hill(), treatment_data=DOSE)
        assert isinstance(un, Unsupported), (iv.mode, ref.mode)
        assert "Δagg(Y)/Δagg(X)" in un.reason and "evaluate.realize" in un.reason
        assert un.missing == ("dose_path",)
        assert un.detail["intervention_mode"] == iv.mode
        assert un.detail["reference_mode"] == ref.mode
        assert Spec.from_json(un.to_json()) == un
    # the zero-difference refusal applies to like-mode arms only
    with pytest.raises(ValueError, match="both scale 'fertilizer' to 2"):
        estimand_expr(
            _kind("ratio", intervention=_iv(2.0, "scale"), reference=_iv(2.0, "scale")),
            mean=_hill(),
            treatment_data=DOSE,
        )
    mixed_same_level = estimand_expr(
        _kind("ratio", intervention=_iv(1.0, "set"), reference=_iv(1.0, "scale")),
        mean=_hill(),
        treatment_data=DOSE,
    )
    assert isinstance(mixed_same_level, Unsupported)


def test_mixed_mode_arms_in_a_contrast_substitute_each_arm_on_its_own() -> None:
    """``set 100`` against ``scale 1.0`` (the observed dose) is ``mean(100) − mean(x)``."""
    doses = np.array([0.0, 25.0, 50.0, 200.0])
    e = _kind("contrast", intervention=_iv(100.0, "set"), reference=_iv(1.0, "scale"))
    expr = estimand_expr(e, mean=_hill(), treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    got = value(expr, data={"dose": doses}, params=PARAMS)
    want = _mean_at(100.0) - value(_hill(), data={"dose": doses}, params=PARAMS)
    np.testing.assert_allclose(got, want, rtol=1e-12)
    # the sign follows the arms, not the intervention's mode
    assert got[0] > 0 and got[-1] < 0
    flipped = _kind("contrast", intervention=_iv(1.0, "scale"), reference=_iv(100.0, "set"))
    expr = estimand_expr(flipped, mean=_hill(), treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    np.testing.assert_allclose(value(expr, data={"dose": doses}, params=PARAMS), -want, rtol=1e-12)
    # shift against set: mean(x + 30) − mean(0)
    e = _kind("contrast", intervention=_iv(30.0, "shift"), reference=_iv(0.0, "set"))
    expr = estimand_expr(e, mean=_hill(), treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    np.testing.assert_allclose(
        value(expr, data={"dose": doses}, params=PARAMS),
        value(_hill(), data={"dose": doses + 30.0}, params=PARAMS) - _mean_at(0.0),
        rtol=1e-12,
    )


def test_carryover_refuses_a_set_dose_in_either_arm_but_accepts_a_scaled_one() -> None:
    mean, params = _carried_hill()
    un = estimand_expr(_kind("contrast"), mean=mean, treatment_data=DOSE)
    assert isinstance(un, Unsupported) and "Convolve" in un.reason and "time_window" in un.missing
    # a set REFERENCE arm is a time path too (the intervention arm is a scale)
    un = estimand_expr(
        _kind("contrast", intervention=_iv(2.0, "scale"), reference=_iv(0.0, "set")),
        mean=mean,
        treatment_data=DOSE,
    )
    assert isinstance(un, Unsupported) and "Convolve" in un.reason and "time_window" in un.missing
    sc = _kind("contrast", intervention=_iv(2.0, "scale"), reference=_iv(1.0, "scale"))
    expr = estimand_expr(sc, mean=mean, treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    path = np.array([[10.0, 0.0, 40.0, 40.0, 0.0]])
    got = value(expr, data={"dose": path}, params=params)
    want = value(mean, data={"dose": 2 * path}, params=params) - value(
        mean, data={"dose": path}, params=params
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)
    # a shift keeps the path as well
    sh = _kind("contrast", intervention=_iv(5.0, "shift"), reference=_iv(0.0, "shift"))
    expr = estimand_expr(sh, mean=mean, treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    got = value(expr, data={"dose": path}, params=params)
    want = value(mean, data={"dose": path + 5.0}, params=params) - value(
        mean, data={"dose": path}, params=params
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_reduce_and_opaque_refuse_a_scalar_column() -> None:
    """A set dose or a pinned covariate feeding a Reduce or an Opaque collapses the time axis:
    ``Reduce(sum)`` of a scalar is the scalar, not ``T`` times it (a factor-of-T error)."""
    gamma = Param(name="gamma", dimension=D.outcome)
    total = Reduce(op="sum", arg=DOSE, keepdims=True)  # the horizon total of the dose
    mean = Add(terms=(_hill(), Mul(factors=(gamma, Div(numerator=total, denominator=K)))))
    path = np.array([[10.0, 0.0, 40.0, 40.0, 0.0, 20.0, 30.0, 20.0]])
    params = {**PARAMS, "gamma": 3.0}

    un = estimand_expr(_kind("contrast"), mean=mean, treatment_data=DOSE)
    assert isinstance(un, Unsupported) and "'dose' feeds a Reduce" in un.reason
    assert un.missing == ("time_window",) and "evaluate.realize" in un.reason
    # a set reference arm alone triggers it as well
    un = estimand_expr(
        _kind("contrast", intervention=_iv(2.0, "scale"), reference=_iv(0.0, "set")),
        mean=mean,
        treatment_data=DOSE,
    )
    assert isinstance(un, Unsupported) and "Reduce" in un.reason
    # a scaled dose keeps the path and the expression is exact
    sc = _kind("contrast", intervention=_iv(2.0, "scale"), reference=_iv(1.0, "scale"))
    expr = estimand_expr(sc, mean=mean, treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    got = value(expr, data={"dose": path}, params=params)
    want = value(mean, data={"dose": 2 * path}, params=params) - value(
        mean, data={"dose": path}, params=params
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)
    # a pinned covariate that feeds a Reduce is refused too, naming the covariate
    rain = Data(name="rain", dimension=dimensionless())
    with_rain = Add(
        terms=(_hill(), Mul(factors=(gamma, Reduce(op="mean", arg=rain, keepdims=True))))
    )
    un = estimand_expr(
        _kind("contrast"), mean=with_rain, treatment_data=DOSE, reference={"rain": 0.5}
    )
    assert isinstance(un, Unsupported) and "'rain' feeds a Reduce" in un.reason
    # without pinning it, the covariate column stays a column
    ok = estimand_expr(_kind("contrast"), mean=with_rain, treatment_data=DOSE)
    assert not isinstance(ok, Unsupported)

    # Opaque: the user function sees the whole column; a constant in its place is not the column
    user = Opaque(name="user", inputs=(DOSE, K), dimension=D.outcome)
    un = estimand_expr(_kind("contrast"), mean=user, treatment_data=DOSE)
    assert isinstance(un, Unsupported) and "'dose' feeds a Opaque" in un.reason
    assert un.missing == ("time_window",)
    expr = estimand_expr(sc, mean=user, treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    fn = {"user": lambda x, k: np.cumsum(x, axis=-1) / k}
    got = value(expr, data={"dose": path}, params=PARAMS, opaque=fn)
    want = value(user, data={"dose": 2 * path}, params=PARAMS, opaque=fn) - value(
        user, data={"dose": path}, params=PARAMS, opaque=fn
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_standard_estimands_all_derive_to_their_declared_dimension() -> None:
    reg = standard_estimands(
        Treatment(name="fertilizer", dimension=D.currency, unit="USD"),
        Outcome(name="yield_total", dimension=D.outcome, unit="kg"),
        Population(name="north"),
        TimeWindow(start=0, stop=8),
        Level(unit="cluster"),
        dose=100.0,
    )
    for e in reg:
        expr = estimand_expr(e, mean=_hill(), treatment_data=DOSE, derivative=_hill_derivative())
        if e.quantity.kind == "area":
            assert isinstance(expr, Unsupported)
            continue
        assert not isinstance(expr, Unsupported), e.name
        assert check_estimand_dimension(e, expr) == e.dimension
        assert np.isfinite(value(expr, params=PARAMS)).all()


# -- recovery against a simulated world ---------------------------------------------------


def test_estimand_exprs_recover_the_arms_world_truth() -> None:
    """On a one-period arms world the per-row expression *is* the functional: its value at
    the truth equals ``world.forward`` / ``world.marginal`` arithmetic to 1e-10."""
    world = arms_world(n_units=6, doses={"dose": np.linspace(0.0, 150.0, 6)}, seed=0)
    treatment = world.spec.treatment("dose")
    column = Data(name="dose", dimension=D.currency)
    assert treatment.dimension == column.dimension
    reg = standard_estimands(
        treatment,
        world.spec.outcome,
        Population(name="arms"),
        TimeWindow(start=0, stop=1),
        Level(unit="individual"),
        dose=100.0,
    )
    derivative = marginal_expr(world.surface, "dose")
    y100 = world.forward({"dose": 100.0})
    y0 = world.forward({"dose": 0.0})
    slope = world.marginal("dose", dose={"dose": 100.0})
    assert not isinstance(slope, Unsupported)
    # every unit sits at the same dose, so each truth is one number replicated over units
    for arr in (y100, y0, slope):
        assert arr.shape == (6, 1) and np.ptp(arr) == 0.0
    truth = {
        "contrast_at_dose": y100[0, 0] - y0[0, 0],
        "average_response_ratio": (y100[0, 0] - y0[0, 0]) / 100.0,
        "marginal_at_dose": slope[0, 0],
        "elasticity_at_dose": slope[0, 0] * 100.0 / y100[0, 0],
    }
    assert truth["contrast_at_dose"] > 0 and truth["marginal_at_dose"] > 0
    for name, want in truth.items():
        e = reg.get(name)
        expr = estimand_expr(e, mean=world.model.mean, treatment_data=column, derivative=derivative)
        assert not isinstance(expr, Unsupported), name
        assert check_estimand_dimension(e, expr) == e.dimension
        assert not any(isinstance(n, Data) for _, n in walk(expr))
        got = value(expr, data=world.data, params=world.theta)
        np.testing.assert_allclose(got, want, rtol=1e-10, err_msg=name)
    area = estimand_expr(
        reg.get("area_under_response"), mean=world.model.mean, treatment_data=column
    )
    assert isinstance(area, Unsupported) and "quadrature" in area.missing


# -- residual defects from verification ---------------------------------------------------


def test_pinned_column_must_carry_one_dimension_across_both_trees() -> None:
    """``rain`` is dimensionless in the mean and a time in the derivative: one pinned Const
    cannot serve both, and the refusal names the column and both dimensions up front rather
    than failing ``dimension()`` on an internal node of whichever tree lost."""
    gamma = Param(name="gamma", dimension=D.outcome)
    eta = Param(name="eta", dimension=D.outcome / D.currency / D.time)
    mean = Add(terms=(_hill(), Mul(factors=(gamma, Data(name="rain", dimension=dimensionless())))))
    derivative = Add(
        terms=(_hill_derivative(), Mul(factors=(eta, Data(name="rain", dimension=D.time))))
    )
    assert dimension(mean) == D.outcome and dimension(derivative) == D.outcome / D.currency
    with pytest.raises(DimensionError, match=r"pins 'rain', whose dimension differs") as info:
        estimand_expr(
            _kind("elasticity"),
            mean=mean,
            treatment_data=DOSE,
            reference={"rain": 0.7},
            derivative=derivative,
        )
    msg = str(info.value)
    assert "'lift_at_100'" in msg  # names the estimand
    assert "mean tree declares it with 1" in msg and "derivative tree declares it with T" in msg
    # the same tree declaring the column twice with different dimensions is caught as well
    twice = Add(
        terms=(
            mean,
            Mul(factors=(eta, Data(name="rain", dimension=D.time), K)),
        )
    )
    with pytest.raises(DimensionError, match="mean tree declares it with 1 and T"):
        estimand_expr(_kind("contrast"), mean=twice, treatment_data=DOSE, reference={"rain": 0.7})
    # agreeing trees pin as before
    same = Add(
        terms=(
            _hill_derivative(),
            Mul(
                factors=(
                    Param(name="eta2", dimension=D.outcome / D.currency),
                    Data(name="rain", dimension=dimensionless()),
                )
            ),
        )
    )
    ok = estimand_expr(
        _kind("elasticity"),
        mean=mean,
        treatment_data=DOSE,
        reference={"rain": 0.7},
        derivative=same,
    )
    assert not isinstance(ok, Unsupported)
    assert check_estimand_dimension(_kind("elasticity"), ok).is_dimensionless


def test_set_dose_feeding_a_convolve_kernel_is_unsupported() -> None:
    """A kernel is a weight vector along the time axis; a ``set`` dose there is a length-one
    kernel, which convolves to a different number rather than a factor-of-T one."""
    rain = Data(name="rain", dimension=D.outcome)
    mean = Convolve(signal=rain, kernel=_u(DOSE))
    assert dimension(mean) == D.outcome
    un = estimand_expr(_kind("contrast"), mean=mean, treatment_data=DOSE)
    assert isinstance(un, Unsupported) and "'dose' feeds a Convolve" in un.reason
    assert un.missing == ("time_window",) and "evaluate.realize" in un.reason
    # a set reference arm alone is enough, exactly as for a signal
    un = estimand_expr(
        _kind("contrast", intervention=_iv(2.0, "scale"), reference=_iv(0.0, "set")),
        mean=mean,
        treatment_data=DOSE,
    )
    assert isinstance(un, Unsupported) and "'dose' feeds a Convolve" in un.reason
    # a pinned covariate in the kernel is refused the same way, naming the covariate
    w = Data(name="w", dimension=dimensionless())
    kernel_pinned = Add(terms=(_hill(), Convolve(signal=Mul(factors=(BETA, _u(DOSE))), kernel=w)))
    un = estimand_expr(
        _kind("contrast"), mean=kernel_pinned, treatment_data=DOSE, reference={"w": 0.5}
    )
    assert isinstance(un, Unsupported) and "'w' feeds a Convolve" in un.reason
    # scale arms keep the dose a column, so the kernel stays a vector and the value is exact
    sc = _kind("contrast", intervention=_iv(2.0, "scale"), reference=_iv(1.0, "scale"))
    expr = estimand_expr(sc, mean=mean, treatment_data=DOSE)
    assert not isinstance(expr, Unsupported)
    path = np.array([10.0, 0.0, 40.0, 40.0, 0.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = value(expr, data={"dose": path, "rain": y}, params=PARAMS)
    want = value(mean, data={"dose": 2 * path, "rain": y}, params=PARAMS) - value(
        mean, data={"dose": path, "rain": y}, params=PARAMS
    )
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_negative_set_or_scale_levels_raise_as_on_the_producer_path() -> None:
    """Hill is not defined below zero: a negative ``set`` level or ``scale`` factor would give
    nan silently. ``predict_under`` refuses the realized grid; the expression refuses the
    level. A ``shift`` is negative only relative to the observed dose and cannot be checked."""
    with pytest.raises(ValueError, match=r"estimand 'lift_at_100': the set level -5 for") as info:
        estimand_expr(_kind("contrast", intervention=_iv(-5.0)), mean=_hill(), treatment_data=DOSE)
    assert "'fertilizer'" in str(info.value) and "non-negative" in str(info.value)
    # the reference arm is checked too
    with pytest.raises(ValueError, match="the set level -1 for"):
        estimand_expr(_kind("contrast", reference=_iv(-1.0)), mean=_hill(), treatment_data=DOSE)
    with pytest.raises(ValueError, match="the scale level -0.5 for"):
        estimand_expr(
            _kind("contrast", intervention=_iv(-0.5, "scale"), reference=_iv(1.0, "scale")),
            mean=_hill(),
            treatment_data=DOSE,
        )
    # marginal and elasticity go through the same gate
    with pytest.raises(ValueError, match="the set level -5 for"):
        estimand_expr(
            _kind("marginal", intervention=_iv(-5.0)),
            mean=_hill(),
            treatment_data=DOSE,
            derivative=_hill_derivative(),
        )
    # a zero level is fine; a negative shift is accepted (the observed dose may absorb it)
    ok = estimand_expr(
        _kind("contrast", intervention=_iv(0.0), reference=_iv(0.0, "scale")),
        mean=_hill(),
        treatment_data=DOSE,
    )
    assert not isinstance(ok, Unsupported)
    sh = estimand_expr(
        _kind("contrast", intervention=_iv(-5.0, "shift"), reference=_iv(0.0, "shift")),
        mean=_hill(),
        treatment_data=DOSE,
    )
    assert not isinstance(sh, Unsupported)
    got = value(sh, data={"dose": np.array([50.0, 100.0])}, params=PARAMS)
    assert np.isfinite(got).all()
