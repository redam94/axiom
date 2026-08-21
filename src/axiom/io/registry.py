"""A content-addressed store of specs on disk. ``put`` is idempotent by hash."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from axiom.core.spec import Spec, load_spec, spec_type_name

__all__ = ["ArtifactRegistry"]


class ArtifactRegistry:
    """``root/<hash>.json`` per spec, plus ``root/index.tsv`` of ``hash\\ttype``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        return self.root / f"{digest}.json"

    def put(self, spec: Spec) -> str:
        digest = spec.content_hash()
        p = self._path(digest)
        if not p.exists():
            p.write_text(spec.to_json(indent=2))
            with (self.root / "index.tsv").open("a") as f:
                f.write(f"{digest}\t{spec_type_name(type(spec))}\n")
        return digest

    def get(self, digest: str) -> Spec:
        p = self._path(digest)
        if not p.exists():
            raise KeyError(f"no artifact {digest} in {self.root}")
        spec = load_spec(p.read_text())
        if spec.content_hash() != digest:
            raise ValueError(
                f"artifact {digest} is corrupt: content hashes to {spec.content_hash()}"
            )
        return spec

    def __contains__(self, digest: object) -> bool:
        return isinstance(digest, str) and self._path(digest).exists()

    def __iter__(self) -> Iterator[tuple[str, str]]:
        idx = self.root / "index.tsv"
        if not idx.exists():
            return iter(())
        rows: list[tuple[str, str]] = []
        for line in idx.read_text().splitlines():
            if line:
                digest, type_name = line.split("\t", 1)
                rows.append((digest, type_name))
        return iter(rows)

    def __len__(self) -> int:
        return sum(1 for _ in self)
