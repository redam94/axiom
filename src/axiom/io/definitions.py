"""What a party means by "conversion", version by version.

`meta.commensurable` compares the estimands behind two records and refuses a
pool that is averaging two quantities. It can only do that if somebody supplied
the estimands, and an estimand names a `core.Outcome` — which is a name, a
dimension, a unit and an aggregation. The gap this module closes is one level
further back: **who decided what that outcome is, and when did they change it?**

`StudyRecord.estimand_hash` is the hook and it defaults to the empty string.
Note 0027 §D27.6.8 puts the failure plainly: *optional provenance is absent
provenance once there are thirty parties and one of them has quietly redefined
its outcome.* Two things go wrong and they are different:

* **Disagreement across parties.** Party A's "conversion" counts a trial
  signup; party B's counts a paid one. Both file records under the same
  quantity name and nothing compares them, because the comparison lives in a
  `Spec` neither record carries.
* **Drift within a party.** A definition changes in week nine. Records before
  and after it are the same name, the same party, and different quantities,
  and the readouts on either side of the change are pooled without comment.

A `DefinitionRegistry` names, versions and content-hashes whatever `Spec` a
party has agreed a term means — an `Outcome`, a `Treatment`, a whole `RoleMap` —
so both questions have answers rather than opinions. It is composed over
`io.Catalog`, so the definitions live in their party's scope like everything
else and cross a boundary only by a ledgered transfer.

**Versions are content, not intent.** Registering an identical spec again is a
no-op that returns the existing version; registering a different one under the
same name is version `n + 1` with `supersedes` pointing at the digest it
replaced. Nobody has to remember to bump anything, and nobody can bump without
changing something.

**A change is a diff, not a flag.** `changes` returns the field-level
`Spec.diff` between consecutive versions, so "somebody redefined conversion in
week nine" comes back as `aggregation: sum -> mean`, which is the sentence an
analyst needs.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec, Verdict
from axiom.core.spec import spec_type_name
from axiom.io.catalog import Catalog, Program

__all__ = [
    "Change",
    "Consensus",
    "Definition",
    "DefinitionRegistry",
]

_LABEL = "definition"


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


class Definition(Spec):
    """One version of what a party means by one term.

    ``digest`` is the content hash of the spec itself, which lives in the
    party's ``ProgramStore``; this record is the *index entry* that gives it a
    name, a version and a place in a history. ``supersedes`` is the digest of
    the version this one replaced, empty for the first.
    """

    scope: NonEmptyStr
    name: NonEmptyStr
    version: int = Field(ge=1)
    digest: NonEmptyStr
    type_name: NonEmptyStr
    registered: str = ""
    supersedes: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _first_supersedes_nothing(self) -> Definition:
        if self.version == 1 and self.supersedes:
            raise ValueError("version 1 supersedes nothing")
        if self.version > 1 and not self.supersedes:
            raise ValueError(f"version {self.version} must name the digest it replaced")
        return self

    @property
    def key(self) -> str:
        return f"{self.scope}:{self.name}@{self.version}"

    def __str__(self) -> str:
        return f"{self.key} ({self.type_name.split(':')[-1]}, {self.digest[:12]}…)"


class Change(Spec):
    """What changed between two consecutive versions of one definition.

    ``changed`` maps a dotted field path to ``"before -> after"``. It is the
    ``Spec.diff`` of the two specs rendered for reading; ``digests`` keeps both
    hashes so the raw specs can be fetched and re-diffed.
    """

    scope: NonEmptyStr
    name: NonEmptyStr
    from_version: int = Field(ge=1)
    to_version: int = Field(ge=2)
    at: str = ""
    changed: dict[str, str] = {}
    digests: tuple[str, str] = ("", "")
    note: str = ""

    @model_validator(mode="after")
    def _consecutive(self) -> Change:
        if self.to_version != self.from_version + 1:
            raise ValueError(
                f"a change is between consecutive versions, got {self.from_version} "
                f"and {self.to_version}"
            )
        return self

    def __str__(self) -> str:
        fields = ", ".join(f"{k}: {v}" for k, v in sorted(self.changed.items()))
        return f"{self.scope}:{self.name} v{self.from_version}->v{self.to_version} — {fields}"


class Consensus(Spec):
    """Whether several parties mean the same thing by one term.

    ``agreed`` is the digest they share when they do. When they do not,
    ``by_digest`` groups the scopes by what they mean and ``differences``
    renders the field-level diff of each dissenting definition against the
    first, which is the form somebody can act on.
    """

    name: NonEmptyStr
    scopes: tuple[str, ...]
    missing: tuple[str, ...] = ()
    by_digest: dict[str, tuple[str, ...]] = {}
    differences: dict[str, dict[str, str]] = {}

    @property
    def agreed(self) -> str:
        """The one digest every scope shares, or the empty string."""
        return next(iter(self.by_digest)) if len(self.by_digest) == 1 else ""

    def verdict(self) -> Verdict:
        """``identified`` when every scope agrees, ``blocked`` when they do not.

        ``unverified`` when some scope has never registered the term: a party
        that has not said what it means by "conversion" has not disagreed, and
        recording that as agreement would be the failure this module exists for.
        """
        if self.missing:
            return Verdict(
                status="unverified",
                reason=(
                    f"{list(self.missing)} have not registered a definition of {self.name!r}; "
                    "silence is not agreement"
                ),
                route="definition",
            )
        if self.agreed:
            return Verdict(status="identified", route="definition")
        rendered = "; ".join(
            f"{scope}: {', '.join(f'{k} {v}' for k, v in sorted(fields.items()))}"
            for scope, fields in sorted(self.differences.items())
        )
        return Verdict(
            status="blocked",
            reason=(
                f"{len(self.by_digest)} different definitions of {self.name!r} across "
                f"{len(self.scopes)} scope(s): {rendered}"
            ),
            route="definition",
        )

    def summary(self) -> str:
        verdict = self.verdict()
        lines = [f"{self.name!r} across {len(self.scopes)} scope(s): {verdict.status}"]
        for digest, scopes in sorted(self.by_digest.items()):
            lines.append(f"  {digest[:12]}…  {', '.join(scopes)}")
        if self.missing:
            lines.append(f"  (not registered) {', '.join(self.missing)}")
        return "\n".join(lines)


class DefinitionRegistry:
    """Named, versioned definitions per party, over an ``io.Catalog``.

    Every definition spec is stored in its party's scope; the ``Definition``
    index entries are stored beside them, so a registry is readable from the
    catalog alone and needs no second store.
    """

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    # -- writing ------------------------------------------------------------

    def register(
        self,
        program: Program,
        name: str,
        spec: Spec,
        *,
        note: str = "",
        at: str | None = None,
    ) -> Definition:
        """Record what ``program`` means by ``name``, as a new version if it changed.

        Idempotent by content: registering an identical spec returns the
        existing ``Definition`` unchanged, so a pipeline that registers on every
        run does not manufacture versions.
        """
        if not name.strip():
            raise ValueError("a definition needs a name")
        store = self.catalog.store(program)
        digest = store.put(spec, label=f"{_LABEL}:{name}", tags={"definition": name})
        existing = self.history(program, name)
        if existing and existing[-1].digest == digest:
            return existing[-1]
        version = len(existing) + 1
        entry = Definition(
            scope=program.scope,
            name=name,
            version=version,
            digest=digest,
            type_name=spec_type_name(type(spec)),
            registered=at if at is not None else _now(),
            supersedes=existing[-1].digest if existing else "",
            note=note,
        )
        store.put(
            entry,
            label=f"{_LABEL}-entry:{name}",
            tags={"definition": name, "version": str(version)},
        )
        return entry

    # -- reading ------------------------------------------------------------

    def history(self, program: Program, name: str) -> tuple[Definition, ...]:
        """Every version of one term in one scope, oldest first."""
        store = self.catalog.store(program)
        rows = store.find(type_name="Definition", tags={"definition": name})
        entries = []
        for row in rows:
            entry = store.get(row.digest)
            if isinstance(entry, Definition) and entry.name == name:
                entries.append(entry)
        return tuple(sorted(entries, key=lambda d: d.version))

    def current(self, program: Program, name: str) -> Definition:
        """The latest version, or ``KeyError`` when the term was never registered."""
        entries = self.history(program, name)
        if not entries:
            raise KeyError(f"{program.scope} has no definition of {name!r}")
        return entries[-1]

    def get(self, program: Program, name: str, version: int | None = None) -> Spec:
        """The spec itself, at ``version`` or at the latest."""
        entries = self.history(program, name)
        if not entries:
            raise KeyError(f"{program.scope} has no definition of {name!r}")
        if version is None:
            entry = entries[-1]
        else:
            found = [e for e in entries if e.version == version]
            if not found:
                raise KeyError(
                    f"{program.scope} has no version {version} of {name!r}; "
                    f"it has 1..{len(entries)}"
                )
            entry = found[0]
        return self.catalog.store(program).get(entry.digest)

    def names(self, program: Program) -> tuple[str, ...]:
        """Every term this scope has defined, sorted."""
        rows = self.catalog.store(program).find(type_name="Definition")
        return tuple(sorted({row.tags["definition"] for row in rows if "definition" in row.tags}))

    # -- the two questions --------------------------------------------------

    def changes(self, program: Program, name: str) -> tuple[Change, ...]:
        """Field-level diffs between consecutive versions, oldest first.

        This is the answer to "somebody redefined conversion in week nine": not
        a flag, but ``aggregation: sum -> mean`` and the date it happened.
        """
        entries = self.history(program, name)
        store = self.catalog.store(program)
        out = []
        for older, newer in zip(entries, entries[1:], strict=False):
            before, after = store.get(older.digest), store.get(newer.digest)
            diff = before.diff(after)
            out.append(
                Change(
                    scope=program.scope,
                    name=name,
                    from_version=older.version,
                    to_version=newer.version,
                    at=newer.registered,
                    changed={k: f"{a!r} -> {b!r}" for k, (a, b) in diff.changed.items()},
                    digests=(older.digest, newer.digest),
                    note=newer.note,
                )
            )
        return tuple(out)

    def consensus(self, name: str, programs: Sequence[Program]) -> Consensus:
        """Whether these parties currently mean the same thing by ``name``.

        A party that has never registered the term is ``missing`` and makes the
        verdict ``unverified`` rather than agreed: silence is not agreement.
        """
        if not programs:
            raise ValueError("consensus needs at least one program")
        by_digest: dict[str, list[str]] = {}
        missing: list[str] = []
        specs: dict[str, Spec] = {}
        for program in programs:
            try:
                entry = self.current(program, name)
            except KeyError:
                missing.append(program.scope)
                continue
            by_digest.setdefault(entry.digest, []).append(program.scope)
            specs[program.scope] = self.catalog.store(program).get(entry.digest)

        differences: dict[str, dict[str, str]] = {}
        if len(by_digest) > 1:
            reference_scope = sorted(specs)[0]
            reference = specs[reference_scope]
            for scope, spec in sorted(specs.items()):
                if scope == reference_scope or spec.content_hash() == reference.content_hash():
                    continue
                if type(spec) is not type(reference):
                    differences[scope] = {
                        "type": f"{spec_type_name(type(reference))} -> {spec_type_name(type(spec))}"
                    }
                    continue
                diff = reference.diff(spec)
                differences[scope] = {k: f"{a!r} -> {b!r}" for k, (a, b) in diff.changed.items()}
        return Consensus(
            name=name,
            scopes=tuple(p.scope for p in programs),
            missing=tuple(missing),
            by_digest={d: tuple(sorted(s)) for d, s in sorted(by_digest.items())},
            differences=differences,
        )

    def __repr__(self) -> str:
        return f"DefinitionRegistry({self.catalog.root})"


def registered(registry: DefinitionRegistry, program: Program) -> Mapping[str, Definition]:
    """Every term this scope has defined, at its current version."""
    return {name: registry.current(program, name) for name in registry.names(program)}
