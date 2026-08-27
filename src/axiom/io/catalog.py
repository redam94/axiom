"""Scoped artifact storage: a ``Program`` owns a scope, and a ``Catalog`` holds many.

``ArtifactRegistry`` answers "is this artifact here". It cannot answer "whose is
it", which is the question that matters as soon as one house runs experiments
for more than one party: two parties' corpora in one root are indistinguishable,
and nothing but directory discipline keeps one party's panel out of another
party's pool. See ``docs/notes/0027-scope-and-the-experiment-lifecycle.md``.

A ``Program`` is a scope — a ``party`` and a line of work within it. The
vocabulary is general (rule 2): a commercial house reads ``party`` as the
client, a trial as the site, a consortium as the member. Artifacts live under
``root/scopes/<party>/<program>/``, each scope backed by its own
``ArtifactRegistry``, and one index across all of them carries columns rather
than the flat registry's ``hash\\ttype``.

Two rules the module enforces, because everything downstream of a cross-party
comparison rests on them:

* **An artifact belongs to exactly one scope.** ``ProgramStore.get`` for a
  digest held under a different scope raises ``CrossScopeError`` naming both.
  It does not return the artifact and it does not return ``None``: "no such
  artifact" and "that is somebody else's artifact" are different facts.
* **Crossing a boundary is a transfer, and takes a ledger line.**
  ``Catalog.transfer`` requires a ``LedgerLine``, stores it in the target
  scope, and records its digest on the target entry, so ``Catalog.crossings``
  is the list of every artifact that ever moved between parties.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator

from axiom.core.spec import Spec, spec_type_name
from axiom.core.verdict import LedgerLine
from axiom.io.registry import ArtifactRegistry

__all__ = [
    "Catalog",
    "CatalogEntry",
    "CrossScopeError",
    "Program",
    "ProgramStore",
]

_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_INDEX = "index.tsv"
_COLUMNS = ("digest", "type", "scope", "label", "created", "derived_from", "tags")


def _token(v: str) -> str:
    if not _TOKEN.match(v):
        raise ValueError(
            f"{v!r} is not a scope token: lowercase alphanumerics, dot, dash and "
            "underscore, starting with a letter or digit"
        )
    return v


ScopeToken = Annotated[str, AfterValidator(_token)]
"""A path-safe scope component. Cannot contain a separator, so cannot climb out of a root."""


def _clean(field: str, value: str) -> str:
    if "\t" in value or "\n" in value:
        raise ValueError(f"{field} must not contain a tab or a newline: {value!r}")
    return value


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


class CrossScopeError(LookupError):
    """The artifact exists, and it belongs to another scope."""

    def __init__(self, digest: str, *, holder: str, asked_by: str) -> None:
        self.digest, self.holder, self.asked_by = digest, holder, asked_by
        super().__init__(
            f"artifact {digest} belongs to scope {holder!r}, not {asked_by!r}; "
            "use Catalog.transfer with a LedgerLine to move it"
        )


class Program(Spec):
    """One party's line of work: the scope every artifact in it belongs to.

    ``party`` is who the work is for — the client, the site, the member — and
    ``program`` the line of work within it. Both are ``ScopeToken``s, so
    ``scope`` is a two-component path that cannot escape the catalog root.
    ``started`` is free text (an ISO date by convention) and, like
    ``description``, is provenance rather than a key: two programs are the same
    program when their party and name match.
    """

    party: ScopeToken
    program: ScopeToken
    description: str = ""
    started: str = ""

    @property
    def scope(self) -> str:
        """``"<party>/<program>"`` — the key this program's artifacts are filed under."""
        return f"{self.party}/{self.program}"

    def __str__(self) -> str:
        return self.scope


class CatalogEntry(Spec):
    """One index row: where an artifact lives and what it is, without opening it.

    ``derived_from`` is the digest of the ``LedgerLine`` that licensed a
    crossing into this scope, and is empty for an artifact that was written
    here. ``tags`` are free key/value strings the caller filters on; neither
    they nor ``label`` may contain a tab or a newline, because the index is a
    TSV.
    """

    digest: str
    type_name: str
    scope: str
    label: str = ""
    created: str = ""
    derived_from: str = ""
    tags: dict[str, str] = {}

    def model_post_init(self, __context: object) -> None:
        _clean("label", self.label)
        for k, v in self.tags.items():
            _clean(f"tag {k!r}", v)

    def row(self) -> str:
        return "\t".join(
            (
                self.digest,
                self.type_name,
                self.scope,
                self.label,
                self.created,
                self.derived_from,
                json.dumps(self.tags, sort_keys=True, separators=(",", ":")),
            )
        )

    @classmethod
    def from_row(cls, row: str) -> CatalogEntry:
        fields = row.split("\t")
        if len(fields) != len(_COLUMNS):
            raise ValueError(
                f"index row has {len(fields)} fields, expected {len(_COLUMNS)}: {row!r}"
            )
        digest, type_name, scope, label, created, derived_from, tags = fields
        return cls(
            digest=digest,
            type_name=type_name,
            scope=scope,
            label=label,
            created=created,
            derived_from=derived_from,
            tags=json.loads(tags),
        )

    def matches(
        self,
        *,
        scope: str | None,
        type_name: str | None,
        label: str | None,
        tags: Mapping[str, str] | None,
    ) -> bool:
        if scope is not None and self.scope != scope:
            return False
        if type_name is not None and not self.type_name.endswith(type_name):
            return False
        if label is not None and self.label != label:
            return False
        if tags:
            return all(self.tags.get(k) == v for k, v in tags.items())
        return True


class Catalog:
    """Many scopes over one root, with an index that can be queried.

    ``root/index.tsv`` is the index; ``root/scopes/<party>/<program>/`` holds
    one ``ArtifactRegistry`` per program. Use ``store`` for the scoped handle a
    caller should hold; the catalog itself is the cross-scope view — ``find``,
    ``crossings``, and the ``transfer`` that is the only way between scopes.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, CatalogEntry] = {}
        self._index_size = -1

    # -- the index ----------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / _INDEX

    def _load(self) -> dict[str, CatalogEntry]:
        p = self.index_path
        size = p.stat().st_size if p.exists() else 0
        if size == self._index_size:
            return self._index
        entries: dict[str, CatalogEntry] = {}
        if size:
            for line in p.read_text().splitlines()[1:]:
                if line:
                    entry = CatalogEntry.from_row(line)
                    entries[f"{entry.scope}/{entry.digest}"] = entry
        self._index, self._index_size = entries, size
        return entries

    def _append(self, entry: CatalogEntry) -> None:
        p = self.index_path
        header = "" if p.exists() else "\t".join(_COLUMNS) + "\n"
        with p.open("a") as f:
            f.write(header + entry.row() + "\n")
        self._index[f"{entry.scope}/{entry.digest}"] = entry
        self._index_size = p.stat().st_size

    def entries(self) -> tuple[CatalogEntry, ...]:
        """Every row, in the order written."""
        return tuple(self._load().values())

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self.entries())

    def __len__(self) -> int:
        return len(self._load())

    # -- scopes -------------------------------------------------------------

    def _scope_root(self, scope: str) -> Path:
        return self.root / "scopes" / scope

    def store(self, program: Program) -> ProgramStore:
        """The handle a caller holds: this program's artifacts and nobody else's."""
        return ProgramStore(self, program)

    def register(self, program: Program) -> str:
        """Store the ``Program`` spec in its own scope, so the catalog lists it."""
        return self.store(program).put(program, label="program")

    def programs(self) -> tuple[Program, ...]:
        """Every registered program, sorted by scope."""
        name = spec_type_name(Program)
        out: list[Program] = []
        for entry in self.entries():
            if entry.type_name != name:
                continue
            spec = ArtifactRegistry(self._scope_root(entry.scope)).get(entry.digest)
            if not isinstance(spec, Program):
                raise TypeError(
                    f"index row {entry.digest} claims {name} but the artifact is a "
                    f"{type(spec).__name__}"
                )
            out.append(spec)
        return tuple(sorted(out, key=lambda p: p.scope))

    def scopes(self) -> tuple[str, ...]:
        """Every scope that holds at least one artifact, sorted."""
        return tuple(sorted({e.scope for e in self.entries()}))

    # -- queries ------------------------------------------------------------

    def find(
        self,
        *,
        scope: str | None = None,
        type_name: str | None = None,
        label: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> tuple[CatalogEntry, ...]:
        """Index rows matching every constraint given. ``type_name`` matches a suffix,
        so ``"ExperimentRun"`` finds ``axiom.io.experiment:ExperimentRun``."""
        return tuple(
            e
            for e in self.entries()
            if e.matches(scope=scope, type_name=type_name, label=label, tags=tags)
        )

    def entry(self, digest: str, *, scope: str | None = None) -> CatalogEntry:
        """The row for one artifact. Raises ``KeyError`` when the catalog has no such row."""
        for e in self.entries():
            if e.digest == digest and (scope is None or e.scope == scope):
                return e
        where = f" in scope {scope!r}" if scope else ""
        raise KeyError(f"no artifact {digest}{where} in {self.root}")

    def scope_of(self, digest: str) -> tuple[str, ...]:
        """Every scope holding this artifact — empty when unknown, more than one after
        a licensed ``transfer``."""
        return tuple(sorted({e.scope for e in self.entries() if e.digest == digest}))

    def crossings(self) -> tuple[CatalogEntry, ...]:
        """Every artifact that reached its scope by ``transfer``, with the line that licensed it."""
        return tuple(e for e in self.entries() if e.derived_from)

    # -- the boundary -------------------------------------------------------

    def transfer(
        self,
        digest: str,
        *,
        source: Program,
        target: Program,
        line: LedgerLine,
        label: str = "",
        tags: Mapping[str, str] | None = None,
    ) -> str:
        """Copy one artifact from ``source`` to ``target`` under a ledger line.

        The line is stored in the target scope as an artifact of its own and its
        digest is recorded on the new entry, so the move is findable from either
        end (``Catalog.crossings``). Returns the digest of the stored line, not
        of the artifact, which is unchanged by definition.
        """
        if source.scope == target.scope:
            raise ValueError(f"source and target are the same scope: {source.scope!r}")
        spec = self.store(source).get(digest)
        into = self.store(target)
        line_digest = into.put(
            line, label="crossing", tags={"artifact": digest, "from": source.scope}
        )
        into._write(
            spec,
            label=label or f"from {source.scope}",
            tags=dict(tags or {}),
            derived_from=line_digest,
        )
        return line_digest


class ProgramStore:
    """One scope's artifacts: ``ArtifactRegistry`` restricted to a ``Program``.

    Reads are scoped as strictly as writes. A digest held elsewhere in the
    catalog raises ``CrossScopeError``; an unknown digest raises ``KeyError``.
    """

    def __init__(self, catalog: Catalog, program: Program) -> None:
        self.catalog = catalog
        self.program = program
        self.registry = ArtifactRegistry(catalog._scope_root(program.scope))

    @property
    def scope(self) -> str:
        return self.program.scope

    @property
    def root(self) -> Path:
        return self.registry.root

    def _write(
        self,
        spec: Spec,
        *,
        label: str,
        tags: dict[str, str],
        derived_from: str,
        created: str | None = None,
    ) -> str:
        digest = self.registry.put(spec)
        if f"{self.scope}/{digest}" not in self.catalog._load():
            self.catalog._append(
                CatalogEntry(
                    digest=digest,
                    type_name=spec_type_name(type(spec)),
                    scope=self.scope,
                    label=label,
                    created=created if created is not None else _now(),
                    derived_from=derived_from,
                    tags=tags,
                )
            )
        return digest

    def put(
        self,
        spec: Spec,
        *,
        label: str = "",
        tags: Mapping[str, str] | None = None,
        created: str | None = None,
    ) -> str:
        """Store a spec in this scope and index it. Idempotent by content hash."""
        return self._write(
            spec, label=label, tags=dict(tags or {}), derived_from="", created=created
        )

    def get(self, digest: str) -> Spec:
        """The artifact, if it is this scope's."""
        if digest not in self.registry:
            holders = [s for s in self.catalog.scope_of(digest) if s != self.scope]
            if holders:
                raise CrossScopeError(digest, holder=holders[0], asked_by=self.scope)
            raise KeyError(f"no artifact {digest} in scope {self.scope!r}")
        return self.registry.get(digest)

    def find(
        self,
        *,
        type_name: str | None = None,
        label: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> tuple[CatalogEntry, ...]:
        """This scope's index rows, filtered."""
        return self.catalog.find(scope=self.scope, type_name=type_name, label=label, tags=tags)

    def __contains__(self, digest: object) -> bool:
        return isinstance(digest, str) and digest in self.registry

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self.find())

    def __len__(self) -> int:
        return len(self.find())

    def __repr__(self) -> str:
        return f"ProgramStore({self.scope!r}, {len(self)} artifacts)"
