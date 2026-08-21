"""The fluent-builder pattern, by composition.

Ported from the parent's ``builders/base.py`` (ledger: PORT). The parent
expressed the pattern as an abstract base class with ``_fields`` and
``_set``; here there is no hierarchy. A builder is a frozen dataclass that
*holds* a ``Fields`` — an immutable mapping of the values collected so far —
and every fluent method returns a **new** builder with one more value set.
Nothing mutates, so a builder can be forked at any point and the two forks
cannot see each other (the parent's ``copy()`` dance goes away).

``Fields.with_`` is the one operation every builder uses; ``require`` names
what is missing in a ``BuildError`` rather than letting pydantic's message
speak for a builder. ``Builder`` is the protocol an object satisfies when it
can ``build()`` a ``Spec``; every builder's ``build`` produces a ``Spec``
that round-trips through ``load_spec(spec.to_json())``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from axiom.core import Spec

__all__ = ["BuildError", "Builder", "Fields"]


class BuildError(ValueError):
    """A builder was asked to build before it had what it needs, or was given a bad value."""


@dataclass(frozen=True)
class Fields:
    """An immutable ``name → value`` mapping with a copy-on-write ``with_``.

    Values are stored behind a ``MappingProxyType`` so a builder cannot be
    mutated through its fields; ``with_`` returns a new ``Fields`` and
    leaves this one untouched.
    """

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def with_(self, **updates: Any) -> Fields:
        """A copy with ``updates`` applied; setting a name to ``None`` clears it."""
        merged = dict(self.values)
        for k, v in updates.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
        return Fields(merged)

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def __contains__(self, name: object) -> bool:
        return name in self.values

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def require(self, *names: str, builder: str = "builder") -> None:
        """Raise ``BuildError`` naming every name in ``names`` that has not been set."""
        missing = [n for n in names if n not in self.values]
        if missing:
            raise BuildError(f"{builder}: set {missing} before build(); have {sorted(self.values)}")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@runtime_checkable
class Builder(Protocol):
    """Anything that can assemble a ``Spec`` from what it has collected."""

    def build(self) -> Spec: ...
