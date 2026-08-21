"""A plain ``Posterior``: draws in a dict, no sampler, an npz round-trip.

Implements ``SupportsPosterior`` so everything above ``infer`` can be
exercised before a backend exists — and so a posterior produced anywhere
(NUTS, Laplace, a file, a hand-built dict) is the same kind of thing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from axiom.core.intervals import IntervalDefinition, Summary, summarize

__all__ = ["Posterior"]

# numpy's stub types ``**kwds`` against ``allow_pickle``; alias to keep mypy strict quiet.
_savez: Callable[..., None] = np.savez


class Posterior:
    """Draws keyed by name, each ``(chain, draw, *shape)``, plus coords and provenance.

    ``provenance`` is free-form JSON-able metadata: the seed, the backend,
    the model spec's content hash. It is written alongside the draws.
    """

    def __init__(
        self,
        draws: Mapping[str, npt.ArrayLike],
        *,
        coords: Mapping[str, Sequence[Any]] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if not draws:
            raise ValueError("a Posterior needs at least one variable")
        arrays: dict[str, npt.NDArray[np.float64]] = {}
        shape: tuple[int, int] | None = None
        for name, value in draws.items():
            a = np.asarray(value, dtype=float)
            if a.ndim < 2:
                raise ValueError(
                    f"draws for {name!r} must be (chain, draw, *shape); got shape {a.shape}"
                )
            cd = (int(a.shape[0]), int(a.shape[1]))
            if shape is None:
                shape = cd
            elif cd != shape:
                raise ValueError(
                    f"{name!r} has (chain, draw)={cd}, expected {shape} like the other variables"
                )
            arrays[name] = a
        assert shape is not None
        self._draws = arrays
        self._chains, self._n = shape
        self._coords: dict[str, list[Any]] = {k: list(v) for k, v in (coords or {}).items()}
        self._provenance: dict[str, Any] = dict(provenance or {})

    # -- SupportsPosterior ----------------------------------------------------

    def draws(self, name: str) -> npt.NDArray[np.float64]:
        if name not in self._draws:
            raise KeyError(f"no variable {name!r}; have {sorted(self._draws)}")
        return self._draws[name]

    def names(self) -> frozenset[str]:
        return frozenset(self._draws)

    def coords(self) -> Mapping[str, Sequence[Any]]:
        return dict(self._coords)

    def n_draws(self) -> int:
        return self._chains * self._n

    # -- extras ---------------------------------------------------------------

    @property
    def n_chains(self) -> int:
        return self._chains

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def flat(self, name: str) -> npt.NDArray[np.float64]:
        """``(chain * draw, *shape)``."""
        a = self.draws(name)
        return a.reshape(self.n_draws(), *a.shape[2:])

    def summary(
        self, name: str, *, definition: IntervalDefinition = "hdi", mass: float = 0.9
    ) -> Summary:
        a = self.flat(name)
        if a.ndim != 1:
            raise ValueError(
                f"{name!r} has trailing shape {a.shape[1:]}; summarize a scalar slice instead"
            )
        return summarize(a, definition=definition, mass=mass)

    def with_provenance(self, **extra: Any) -> Posterior:
        return Posterior(self._draws, coords=self._coords, provenance={**self._provenance, **extra})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Posterior):
            return NotImplemented
        return (
            self.names() == other.names()
            and all(np.array_equal(self._draws[k], other._draws[k]) for k in self._draws)
            and self._coords == other._coords
            and self._provenance == other._provenance
        )

    def __hash__(self) -> int:  # pragma: no cover - identity semantics
        return id(self)

    def __repr__(self) -> str:
        return f"Posterior(names={sorted(self._draws)}, chains={self._chains}, draws={self._n})"

    # -- persistence ----------------------------------------------------------

    def to_npz(self, path: str | Path) -> Path:
        """Write draws plus a JSON ``__meta__`` entry. No pickle is involved."""
        p = Path(path)
        meta = json.dumps({"coords": self._coords, "provenance": self._provenance})
        payload: dict[str, npt.NDArray[Any]] = {"__meta__": np.array(meta), **self._draws}
        _savez(p, allow_pickle=False, **payload)
        return p if p.suffix == ".npz" else p.with_suffix(p.suffix + ".npz")

    @classmethod
    def from_npz(cls, path: str | Path) -> Posterior:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            draws = {k: z[k] for k in z.files if k != "__meta__"}
        return cls(draws, coords=meta.get("coords"), provenance=meta.get("provenance"))
