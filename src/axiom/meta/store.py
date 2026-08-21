"""A content-addressed store of records and corpora, composed over ``io.ArtifactRegistry``.

Rewritten from the parent's ``benchmarks/store.py`` (ledger row ``meta/store.py``,
REWRITE): the parent kept its own sqlite database; here a corpus is a spec
like any other and lives as ``root/<hash>.json`` in an ``ArtifactRegistry``.
``put`` is idempotent by content hash, so re-contributing an identical record
is a no-op and two corpora that differ in one record are two artifacts.
"""

from __future__ import annotations

from pathlib import Path

from axiom.core import Spec
from axiom.io import ArtifactRegistry
from axiom.meta.schema import Corpus, StudyRecord

__all__ = ["CorpusStore"]


class CorpusStore:
    """Records and corpora by content hash; corpora also findable by name."""

    def __init__(self, root: str | Path) -> None:
        self.registry = ArtifactRegistry(root)

    @property
    def root(self) -> Path:
        return self.registry.root

    def put_record(self, record: StudyRecord) -> str:
        return self.registry.put(record)

    def put_corpus(self, corpus: Corpus, *, with_records: bool = True) -> str:
        """Store the corpus (and, by default, each record on its own) and return its hash."""
        if with_records:
            for r in corpus.records:
                self.registry.put(r)
        return self.registry.put(corpus)

    def put(self, spec: StudyRecord | Corpus) -> str:
        if isinstance(spec, Corpus):
            return self.put_corpus(spec)
        return self.put_record(spec)

    def get(self, digest: str) -> StudyRecord | Corpus:
        spec: Spec = self.registry.get(digest)
        if not isinstance(spec, StudyRecord | Corpus):
            raise TypeError(f"artifact {digest} is a {type(spec).__name__}, not a meta record")
        return spec

    def get_record(self, digest: str) -> StudyRecord:
        spec = self.get(digest)
        if not isinstance(spec, StudyRecord):
            raise TypeError(f"artifact {digest} is a Corpus, not a StudyRecord")
        return spec

    def get_corpus(self, digest: str) -> Corpus:
        spec = self.get(digest)
        if not isinstance(spec, Corpus):
            raise TypeError(f"artifact {digest} is a StudyRecord, not a Corpus")
        return spec

    def list(self, kind: str | None = None) -> list[tuple[str, str]]:
        """``(hash, type name)`` rows in insertion order; ``kind`` filters by type name
        (``"StudyRecord"`` / ``"Corpus"``)."""
        rows = list(self.registry)
        if kind is None:
            return rows
        return [(h, t) for h, t in rows if t.rsplit(":", 1)[-1] == kind]

    def corpora(self) -> dict[str, str]:
        """Corpus name → hash of the most recently stored corpus with that name."""
        out: dict[str, str] = {}
        for digest, _ in self.list("Corpus"):
            out[self.get_corpus(digest).name] = digest
        return out

    def load_corpus(self, name: str) -> Corpus:
        """The most recently stored corpus called ``name``; ``KeyError`` if none."""
        try:
            digest = self.corpora()[name]
        except KeyError:
            raise KeyError(f"no corpus named {name!r} in {self.root}") from None
        return self.get_corpus(digest)

    def __contains__(self, digest: object) -> bool:
        return digest in self.registry

    def __len__(self) -> int:
        return len(self.registry)
