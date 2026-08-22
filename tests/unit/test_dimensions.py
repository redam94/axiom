from __future__ import annotations

from fractions import Fraction

import pytest

from axiom.core import (
    BASES,
    D,
    Dimension,
    DimensionError,
    LedgerLine,
    UndeclaredBaseError,
    UnitConversionError,
    UnitSystem,
    dimensionless,
)


def test_group_laws() -> None:
    a, b, c = D.currency, D.time, D.outcome
    assert (a * b) * c == a * (b * c)
    assert a * b == b * a
    assert a * dimensionless() == a
    assert a / a == dimensionless()
    assert (a * b) / b == a
    assert (a**2).root(2) == a
    assert a ** Fraction(1, 2) * a ** Fraction(1, 2) == a
    assert (a / b) ** -1 == b / a


def test_zero_exponents_vanish_and_str_is_stable() -> None:
    d = Dimension(exponents={"currency": 1, "time": 0})
    assert d == D.currency and d.exponents == {"currency": Fraction(1)}
    assert str(D.outcome / D.currency) == "$^-1·Y"
    assert str(dimensionless()) == "1"
    assert (D.outcome / D.currency).is_dimensionless is False


def test_undeclared_base_raises_naming_it() -> None:
    with pytest.raises(UndeclaredBaseError, match="'temperature'"):
        Dimension(exponents={"temperature": 1})
    with pytest.raises(UndeclaredBaseError):
        _ = D.temperature


def test_declare_is_idempotent_and_checks_symbol() -> None:
    m = BASES.declare("mass_test", symbol="M")
    assert m == BASES.declare("mass_test", symbol="M") == D.mass_test
    with pytest.raises(DimensionError):
        BASES.declare("mass_test", symbol="kg")
    assert "mass_test" in BASES.declarations()
    with pytest.raises(DimensionError):
        BASES.declare("not an identifier")


def test_require_helpers_name_both_sides() -> None:
    with pytest.raises(DimensionError, match=r"\$ vs T"):
        D.currency.require_equal(D.time)
    with pytest.raises(DimensionError, match="dimensionless"):
        D.currency.require_dimensionless(context="log argument")
    D.currency.require_equal(D.currency)


def test_unit_system_converts_transitively_with_ledger_line() -> None:
    u = UnitSystem()
    u.declare("USD", "currency")
    u.declare("EUR", "currency")
    u.declare("GBP", "currency")
    u.declare("day", "time")
    u.register("USD", "EUR", Fraction(9, 10))
    u.register("EUR", "GBP", Fraction(4, 5))
    value, line = u.convert(100.0, "USD", "GBP")
    assert value == pytest.approx(72.0)
    assert isinstance(line, LedgerLine) and line.kind == "unit_conversion"
    assert line.detail == {"from_unit": "USD", "to_unit": "GBP", "factor": "18/25"}
    assert u.factor("GBP", "USD") == Fraction(25, 18)
    assert u.factor("USD", "USD") == 1
    assert u.dimension_of("day") == D.time


def test_unit_system_refuses_cross_dimension_and_unregistered() -> None:
    u = UnitSystem()
    u.declare("USD", "currency")
    u.declare("day", "time")
    u.declare("JPY", "currency")
    with pytest.raises(DimensionError, match="different dimensions"):
        u.register("USD", "day", 1)
    with pytest.raises(DimensionError):
        u.factor("USD", "day")
    with pytest.raises(UnitConversionError, match="no conversion registered"):
        u.convert(1.0, "USD", "JPY")
    with pytest.raises(UnitConversionError, match="not declared"):
        u.factor("USD", "CHF")
    with pytest.raises(DimensionError):
        u.register("USD", "JPY", 0)
