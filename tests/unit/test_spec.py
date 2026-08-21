from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from axiom.core import (
    D,
    Dimension,
    Interval,
    SchemaVersionError,
    Spec,
    SpecError,
    TimeWindow,
    UnknownSpecError,
    load_spec,
)


class Toy(Spec):
    SCHEMA_VERSION = "2"
    a: int
    b: str = "x"


class ToyV1Loader(Spec):
    SCHEMA_VERSION = "2"
    value: int


ToyV1Loader.register_migration("1", "2", lambda d: {"value": d["old_value"]})


def test_frozen_and_forbids_extra() -> None:
    t = Toy(a=1)
    with pytest.raises(ValidationError):
        t.a = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Toy(a=1, c=3)  # type: ignore[call-arg]


def test_hash_is_order_independent_and_type_sensitive() -> None:
    d1 = Dimension(exponents={"currency": 1, "time": -1})
    d2 = Dimension(exponents={"time": -1, "currency": 1})
    assert d1 == d2 and d1.content_hash() == d2.content_hash() and hash(d1) == hash(d2)
    w1 = TimeWindow(start=0, stop=1)
    assert w1.content_hash() != Toy(a=0).content_hash()


def test_diff_names_paths() -> None:
    a, b = Toy(a=1, b="p"), Toy(a=2, b="p")
    d = a.diff(b)
    assert d.changed == {"a": (1, 2)} and not d.is_empty
    assert a.diff(a).is_empty
    with pytest.raises(SpecError):
        a.diff(TimeWindow(start=0, stop=1))


def test_from_json_rejects_wrong_class_and_bad_envelopes() -> None:
    s = Toy(a=1).to_json()
    with pytest.raises(UnknownSpecError):
        TimeWindow.from_json(s)
    with pytest.raises(SpecError):
        Spec.from_json('{"nope": 1}')
    with pytest.raises(UnknownSpecError):
        load_spec('{"spec": "axiom.core:NoSuchSpec", "schema_version": "1", "data": {}}')
    with pytest.raises(UnknownSpecError):
        load_spec('{"spec": "os:path", "schema_version": "1", "data": {}}')


def test_schema_migration_applies_or_raises() -> None:
    env = f'{{"spec": "{__name__}:ToyV1Loader", "schema_version": "1", "data": {{"old_value": 7}}}}'
    assert load_spec(env) == ToyV1Loader(value=7)
    env_bad = f'{{"spec": "{__name__}:Toy", "schema_version": "1", "data": {{"a": 1}}}}'
    with pytest.raises(SchemaVersionError) as e:
        load_spec(env_bad)
    assert e.value.found == "1" and e.value.expected == "2"


def test_fraction_fields_round_trip() -> None:
    d = D.outcome ** Fraction(1, 3)
    assert Dimension.from_json(d.to_json()) == d
    assert '"1/3"' in d.to_json()


def test_subclasses_are_discoverable_and_sorted() -> None:
    subs = Spec.subclasses()
    assert Interval in subs and Toy in subs
    assert subs == sorted(subs, key=lambda c: f"{c.__module__}:{c.__qualname__}")
