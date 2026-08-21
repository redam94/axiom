"""Typed failures. Returned, never raised, by code that must not guess.

``design commitment #5``: a function that cannot answer returns one of these
(or raises). It never returns a number it is not entitled to. Each carries a
non-empty ``reason``; each is a ``Spec`` so it serializes into a result and
shows up in the ledger. The three are independent classes sharing a field
type, not a hierarchy; ``Failure`` is their union.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeGuard

from pydantic import AfterValidator

from axiom.core.spec import Spec

__all__ = ["Blocked", "Failure", "NonEmptyStr", "Unsupported", "Unverified", "is_failure"]


def _non_empty(v: str) -> str:
    if not v.strip():
        raise ValueError("must be non-empty")
    return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
"""A string that is not blank; used for every ``reason`` and ``statement`` field."""


class Unsupported(Spec):
    """The producer lacks a capability the request needs."""

    reason: NonEmptyStr
    detail: dict[str, str] = {}
    missing: tuple[str, ...] = ()
    status: Literal["unsupported"] = "unsupported"

    def __bool__(self) -> bool:
        return False


class Blocked(Spec):
    """The request is not licensed: no route, no transfer, no identification."""

    reason: NonEmptyStr
    detail: dict[str, str] = {}
    status: Literal["blocked"] = "blocked"

    def __bool__(self) -> bool:
        return False


class Unverified(Spec):
    """An assumption the machinery cannot check; the caller decides."""

    reason: NonEmptyStr
    detail: dict[str, str] = {}
    status: Literal["unverified"] = "unverified"

    def __bool__(self) -> bool:
        return False


Failure = Unsupported | Blocked | Unverified


def is_failure(x: object) -> TypeGuard[Failure]:
    return isinstance(x, Unsupported | Blocked | Unverified)
