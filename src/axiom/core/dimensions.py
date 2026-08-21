"""Dimensions, a declarable base registry, and a unit system with conversions.

Three things are kept distinct (``docs/plan/05-open-decisions.md`` D6):

* **Dimension** — ``currency`` vs ``time`` vs ``outcome``. A mismatch raises,
  always. A ``Dimension`` is a frozen mapping from base name to ``Fraction``
  exponent; ``Fraction`` because a standard deviation is the square root of a
  variance.
* **Unit** — USD vs EUR, day vs week. Convert when a conversion is registered
  in the ``UnitSystem`` (returning the factor *and* a ``LedgerLine``), raise
  when not.
* **Scope** — this population, this window. Not a units problem; settled by
  ``TransferPlan`` in ``calibrate``.

The base set is declarable. ``BASES`` ships ``time``, ``currency``,
``outcome``, ``entity`` and a domain declares what else it needs::

    BASES.declare("mass", symbol="M")
    D.mass / D.time
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import Any, ClassVar

from pydantic import field_validator

from axiom.core.spec import Spec, SpecError
from axiom.core.verdict import LedgerLine

__all__ = [
    "BASES",
    "BaseRegistry",
    "D",
    "DEFAULT_BASES",
    "Dimension",
    "DimensionError",
    "UNITS",
    "UndeclaredBaseError",
    "UnitConversionError",
    "UnitSystem",
    "dimensionless",
]

DEFAULT_BASES: tuple[tuple[str, str], ...] = (
    ("time", "T"),
    ("currency", "$"),
    ("outcome", "Y"),
    ("entity", "N"),
)


class DimensionError(SpecError):
    """Two quantities of different dimension were combined."""


class UndeclaredBaseError(DimensionError):
    """A dimension named a base that the registry has not declared."""


class UnitConversionError(DimensionError):
    """No conversion is registered between two units of the same dimension."""


class BaseRegistry:
    """The declarable set of base dimensions.

    Process-global and mutable on purpose: a domain declares its bases once at
    import time. ``io`` records non-default declarations in the analysis
    manifest and re-declares them on load (note 0002.6).
    """

    def __init__(self, defaults: tuple[tuple[str, str], ...] = DEFAULT_BASES) -> None:
        self._symbols: dict[str, str] = {}
        self._defaults = frozenset(name for name, _ in defaults)
        for name, symbol in defaults:
            self.declare(name, symbol=symbol)

    def declare(self, name: str, *, symbol: str | None = None) -> Dimension:
        """Declare a base (idempotent if the symbol agrees) and return its dimension."""
        if not name.isidentifier():
            raise DimensionError(f"base name {name!r} must be a Python identifier")
        sym = symbol or name
        if name in self._symbols and self._symbols[name] != sym:
            raise DimensionError(
                f"base {name!r} already declared with symbol {self._symbols[name]!r}, not {sym!r}"
            )
        self._symbols[name] = sym
        return Dimension.model_construct(exponents={name: Fraction(1)})

    def symbol(self, name: str) -> str:
        self.require(name)
        return self._symbols[name]

    def require(self, name: str) -> None:
        if name not in self._symbols:
            raise UndeclaredBaseError(
                f"base dimension {name!r} is not declared; declared: {sorted(self._symbols)}"
            )

    def __contains__(self, name: object) -> bool:
        return name in self._symbols

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._symbols))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._symbols)

    def declarations(self) -> dict[str, str]:
        """Non-default ``{name: symbol}`` declarations, for the manifest."""
        return {n: s for n, s in sorted(self._symbols.items()) if n not in self._defaults}

    def __getattr__(self, name: str) -> Dimension:
        if name.startswith("_"):
            raise AttributeError(name)
        self.require(name)
        return Dimension(exponents={name: Fraction(1)})


def _as_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise DimensionError("exponents must be rational numbers, not bool")
    if isinstance(value, int | str):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(10_000)
    raise DimensionError(f"cannot interpret {value!r} as a rational exponent")


class Dimension(Spec):
    """A product of declared base dimensions raised to rational exponents.

    Zero exponents are dropped on construction, so equality and hashing are
    structural: ``D.currency / D.currency == dimensionless()``.
    """

    exponents: dict[str, Fraction]

    @field_validator("exponents", mode="before")
    @classmethod
    def _normalize(cls, v: Any) -> dict[str, Fraction]:
        if not isinstance(v, Mapping):
            raise DimensionError("exponents must be a mapping of base name to exponent")
        out: dict[str, Fraction] = {}
        for name, q in v.items():
            BASES.require(str(name))
            f = _as_fraction(q)
            if f != 0:
                out[str(name)] = f
        return dict(sorted(out.items()))

    # -- algebra ------------------------------------------------------------

    def __mul__(self, other: Dimension) -> Dimension:
        exps = dict(self.exponents)
        for k, q in other.exponents.items():
            exps[k] = exps.get(k, Fraction(0)) + q
        return Dimension(exponents=exps)

    def __truediv__(self, other: Dimension) -> Dimension:
        return self * other**-1

    def __pow__(self, q: Fraction | int) -> Dimension:
        f = _as_fraction(q)
        return Dimension(exponents={k: v * f for k, v in self.exponents.items()})

    def root(self, n: int) -> Dimension:
        """``self ** (1/n)``; the square root of a variance is ``root(2)``."""
        return self ** Fraction(1, n)

    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        parts = []
        for name, q in self.exponents.items():
            sym = BASES.symbol(name)
            parts.append(sym if q == 1 else f"{sym}^{q}")
        return "·".join(parts)

    def __repr__(self) -> str:
        return f"Dimension({self})"

    def __hash__(self) -> int:
        return hash(tuple(self.exponents.items()))

    def require_equal(self, other: Dimension, *, context: str = "") -> None:
        """Raise ``DimensionError`` naming both sides if the dimensions differ."""
        if self != other:
            where = f" in {context}" if context else ""
            raise DimensionError(f"dimension mismatch{where}: {self} vs {other}")

    def require_dimensionless(self, *, context: str = "") -> None:
        if not self.is_dimensionless:
            where = f" in {context}" if context else ""
            raise DimensionError(f"expected a dimensionless quantity{where}, got {self}")


def dimensionless() -> Dimension:
    return Dimension(exponents={})


BASES = BaseRegistry()
D = BASES  # ``D.currency`` reads better than ``BASES.currency`` in model code.


class UnitSystem:
    """Units of measure, each attached to a base dimension, with conversions.

    Conversions are stored as a weighted graph per dimension and composed by
    breadth-first search, so registering USD→EUR and EUR→GBP makes USD→GBP
    available. Every conversion performed returns a ``LedgerLine`` so the
    caller can append it to the analysis ledger (D6).
    """

    CONVERSION_KIND: ClassVar[str] = "unit_conversion"

    def __init__(self) -> None:
        self._dimension_of: dict[str, str] = {}
        self._edges: dict[str, dict[str, Fraction]] = {}

    def declare(self, unit: str, base: str) -> None:
        BASES.require(base)
        if unit in self._dimension_of and self._dimension_of[unit] != base:
            raise DimensionError(
                f"unit {unit!r} already declared for {self._dimension_of[unit]!r}, not {base!r}"
            )
        self._dimension_of[unit] = base
        self._edges.setdefault(unit, {})

    def dimension_of(self, unit: str) -> Dimension:
        if unit not in self._dimension_of:
            raise UnitConversionError(f"unit {unit!r} is not declared in the unit system")
        return Dimension(exponents={self._dimension_of[unit]: Fraction(1)})

    def register(self, src: str, dst: str, factor: Fraction | float | int) -> None:
        """Declare ``1 src == factor dst`` (and the inverse)."""
        for u in (src, dst):
            if u not in self._dimension_of:
                raise UnitConversionError(f"declare unit {u!r} before registering a conversion")
        if self._dimension_of[src] != self._dimension_of[dst]:
            raise DimensionError(
                f"cannot convert {src!r} [{self._dimension_of[src]}] to "
                f"{dst!r} [{self._dimension_of[dst]}]: different dimensions"
            )
        f = _as_fraction(factor)
        if f <= 0:
            raise DimensionError("conversion factors must be positive")
        self._edges[src][dst] = f
        self._edges[dst][src] = 1 / f

    def factor(self, src: str, dst: str) -> Fraction:
        """The multiplicative factor taking ``src`` to ``dst``; raises if none."""
        for u in (src, dst):
            if u not in self._dimension_of:
                raise UnitConversionError(f"unit {u!r} is not declared in the unit system")
        if src == dst:
            return Fraction(1)
        if self._dimension_of[src] != self._dimension_of[dst]:
            raise DimensionError(
                f"cannot convert {src!r} [{self._dimension_of[src]}] to "
                f"{dst!r} [{self._dimension_of[dst]}]: different dimensions"
            )
        seen: dict[str, Fraction] = {src: Fraction(1)}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nxt, f in self._edges[node].items():
                if nxt not in seen:
                    seen[nxt] = seen[node] * f
                    if nxt == dst:
                        return seen[nxt]
                    queue.append(nxt)
        raise UnitConversionError(
            f"no conversion registered from {src!r} to {dst!r}; "
            "register one with UnitSystem.register()"
        )

    def convert(self, value: float, src: str, dst: str) -> tuple[float, LedgerLine]:
        """Convert a scalar and return it with the ledger line that records it."""
        f = self.factor(src, dst)
        line = LedgerLine(
            kind=self.CONVERSION_KIND,
            statement=f"converted {src} to {dst} by factor {f}",
            detail={"from_unit": src, "to_unit": dst, "factor": str(f)},
        )
        return value * float(f), line

    def units(self) -> dict[str, str]:
        return dict(sorted(self._dimension_of.items()))

    def conversions(self) -> list[tuple[str, str, str]]:
        """``(src, dst, factor)`` for every registered edge, for the manifest."""
        return sorted(
            (s, d, str(f)) for s, targets in self._edges.items() for d, f in targets.items()
        )


UNITS = UnitSystem()
