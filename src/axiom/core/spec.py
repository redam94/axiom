"""The frozen, versioned, content-hashed base for every configuration object.

Every declarative thing in axiom — a dimension, an entity, an estimand, a
graph, a design, a ledger line — is a ``Spec``. A ``Spec``:

* is immutable (pydantic ``frozen=True``) and rejects unknown fields;
* serializes to JSON through one envelope, ``{"spec", "schema_version",
  "data"}``, and deserializes back to an equal object;
* has a stable ``content_hash`` (blake2b-256 over canonical JSON) that does
  not depend on ``PYTHONHASHSEED``, dict insertion order, or the process;
* can ``diff`` itself against another spec of the same type;
* carries a class-level ``SCHEMA_VERSION``; loading an older version applies a
  registered migration or raises ``SchemaVersionError`` naming both versions.

Specs hold no large arrays. Data is referenced by the ``Panel``'s hash and a
posterior by its own; see ``docs/notes/0002-foundation-decisions.md`` §5.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Iterator, Mapping
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Spec",
    "SpecDiff",
    "SpecError",
    "SchemaVersionError",
    "UnknownSpecError",
    "load_spec",
    "spec_type_name",
]


class SpecError(Exception):
    """Base class for spec construction and (de)serialization failures."""


class SchemaVersionError(SpecError):
    """A serialized spec's version has no migration to the current class version."""

    def __init__(self, spec: str, found: str, expected: str) -> None:
        self.spec, self.found, self.expected = spec, found, expected
        super().__init__(
            f"{spec}: serialized schema_version {found!r} cannot be loaded by "
            f"schema_version {expected!r}; no migration is registered"
        )


class UnknownSpecError(SpecError):
    """The envelope names a class that cannot be imported or is not a Spec."""


Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_type_name(cls: type[Spec]) -> str:
    """``"<module>:<qualname>"`` — the name written into the envelope."""
    return f"{cls.__module__}:{cls.__qualname__}"


def _flatten(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from _flatten(f"{prefix}.{k}" if prefix else str(k), v)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _flatten(f"{prefix}[{i}]", v)
    else:
        yield prefix, value


class SpecDiff(BaseModel, frozen=True):
    """Field-level differences between two specs of the same type.

    Paths are dotted (``"exponents.currency"``, ``"terms[2].name"``). Each
    entry maps a path to ``(left, right)`` in JSON form; a path present on one
    side only has ``None`` on the other.
    """

    spec: str
    changed: dict[str, tuple[Any, Any]]

    @property
    def is_empty(self) -> bool:
        return not self.changed


class Spec(BaseModel):
    """Frozen, versioned, hashable configuration object. Subclass freely."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    SCHEMA_VERSION: ClassVar[str] = "1"
    _MIGRATIONS: ClassVar[dict[tuple[str, str], Migration]] = {}

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The JSON-mode payload (the ``data`` half of the envelope)."""
        return self.model_dump(mode="json")

    def envelope(self) -> dict[str, Any]:
        return {
            "spec": spec_type_name(type(self)),
            "schema_version": type(self).SCHEMA_VERSION,
            "data": self.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        env = self.envelope()
        if indent is None:
            return _canonical(env)
        return json.dumps(env, sort_keys=True, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, schema_version: str | None = None) -> Self:
        """Build from a payload, migrating from ``schema_version`` if given."""
        payload = dict(data)
        if schema_version is not None and schema_version != cls.SCHEMA_VERSION:
            payload = cls._migrate(payload, schema_version)
        return cls.model_validate(payload)

    @classmethod
    def from_json(cls, s: str) -> Self:
        env = json.loads(s)
        if not isinstance(env, dict) or set(env) != {"spec", "schema_version", "data"}:
            raise SpecError("not a spec envelope: expected keys {'spec', 'schema_version', 'data'}")
        found = _resolve(env["spec"])
        if not issubclass(found, cls):
            raise UnknownSpecError(
                f"envelope names {env['spec']}, which is not a {spec_type_name(cls)}"
            )
        return found.from_dict(env["data"], schema_version=env["schema_version"])

    @classmethod
    def _migrate(cls, payload: dict[str, Any], found: str) -> dict[str, Any]:
        version = found
        seen = {version}
        while version != cls.SCHEMA_VERSION:
            step = next(((src, dst) for (src, dst) in cls._MIGRATIONS if src == version), None)
            if step is None or step[1] in seen:
                raise SchemaVersionError(spec_type_name(cls), found, cls.SCHEMA_VERSION)
            payload = cls._MIGRATIONS[step](payload)
            version = step[1]
            seen.add(version)
        return payload

    @classmethod
    def register_migration(cls, src: str, dst: str, fn: Migration) -> None:
        """Register ``fn`` to carry a payload from schema ``src`` to ``dst``."""
        if "_MIGRATIONS" not in cls.__dict__:
            cls._MIGRATIONS = {}
        cls._MIGRATIONS[(src, dst)] = fn

    # -- identity -----------------------------------------------------------

    def content_hash(self) -> str:
        """blake2b-256 hex digest of the canonical envelope."""
        return hashlib.blake2b(self.to_json().encode("utf-8"), digest_size=32).hexdigest()

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def diff(self, other: Spec) -> SpecDiff:
        if type(other) is not type(self):
            raise SpecError(
                f"cannot diff {spec_type_name(type(self))} against {spec_type_name(type(other))}"
            )
        left = dict(_flatten("", self.to_dict()))
        right = dict(_flatten("", other.to_dict()))
        changed = {
            k: (left.get(k), right.get(k))
            for k in sorted(set(left) | set(right))
            if left.get(k) != right.get(k) or (k in left) != (k in right)
        }
        return SpecDiff(spec=spec_type_name(type(self)), changed=changed)

    # -- discovery ----------------------------------------------------------

    @classmethod
    def subclasses(cls) -> list[type[Spec]]:
        """Every transitively known subclass, in a deterministic order."""
        out: list[type[Spec]] = []
        stack: list[type[Spec]] = [cls]
        while stack:
            node = stack.pop()
            for sub in node.__subclasses__():
                if sub not in out:
                    out.append(sub)
                    stack.append(sub)
        return sorted(out, key=spec_type_name)


def _resolve(name: str) -> type[Spec]:
    module_name, _, qual = name.partition(":")
    if not module_name or not qual:
        raise UnknownSpecError(f"malformed spec name {name!r}; expected '<module>:<Class>'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise UnknownSpecError(f"cannot import module for spec {name!r}") from e
    obj: Any = module
    for part in qual.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            raise UnknownSpecError(f"{name!r} does not exist")
    if not (isinstance(obj, type) and issubclass(obj, Spec)):
        raise UnknownSpecError(f"{name!r} is not a Spec subclass")
    return obj


def load_spec(s: str) -> Spec:
    """Deserialize any spec from its envelope, resolving the class by name."""
    return Spec.from_json(s)
