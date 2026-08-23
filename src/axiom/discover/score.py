"""The score a structure search maximizes, and the data format that makes interventions count.

Structure learning needs a number to hill-climb, and it has to be
*decomposable* — a sum of one term per variable given its parents — so that
changing one edge changes only a few terms. For linear-Gaussian data that is
the BIC: the maximized log-likelihood of each variable regressed on its
parents, minus a penalty for the parameters spent.

The interventional part is one line and it is the whole point. A row in which
a variable was *randomized* carries no information about what causes that
variable — its value came from the experimenter, not from its parents — so
that row is excluded from that variable's local score and from no other's
(Hauser & Bühlmann 2012). Every other variable in the same row is still
informative, which is why interventional data is worth so much more than its
row count suggests.

Two properties matter downstream and both are tested. The score is
*decomposable*, so a search can cache local terms; and it is
**score-equivalent** on observational data — Markov-equivalent DAGs get
identical scores, which is exactly why a search over equivalence classes is
the right search, and why interventional data is what breaks the tie.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["Dataset", "GaussianBIC"]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Dataset:
    """Rows of measurements, each labelled with what was intervened on when it was taken.

    ``regimes[i]`` is the set of variables randomized for row ``i`` — empty for
    an observational row. Not a ``Spec``: it holds the data.
    """

    values: Array
    names: tuple[str, ...]
    regimes: tuple[frozenset[str], ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        object.__setattr__(self, "values", values)
        if values.ndim != 2:
            raise ValueError(f"data is a matrix of rows by variables; got shape {values.shape}")
        if values.shape[1] != len(self.names):
            raise ValueError(f"{values.shape[1]} columns but {len(self.names)} names")
        if len(self.regimes) != values.shape[0]:
            raise ValueError(
                f"{len(self.regimes)} regimes but {values.shape[0]} rows; every row is "
                "labelled with what was intervened on when it was taken"
            )
        unknown = sorted({v for r in self.regimes for v in r} - set(self.names))
        if unknown:
            raise ValueError(f"regimes name variables the data does not have: {unknown}")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"variable names must be distinct: {list(self.names)}")

    @classmethod
    def observational(cls, values: Array, names: Sequence[str]) -> Dataset:
        """Every row taken under no intervention."""
        rows = np.asarray(values, dtype=np.float64).shape[0]
        return cls(np.asarray(values, dtype=np.float64), tuple(names), (frozenset(),) * rows)

    @classmethod
    def stack(cls, *parts: Dataset) -> Dataset:
        """Pool observational and interventional batches into one dataset."""
        if not parts:
            raise ValueError("nothing to stack")
        names = parts[0].names
        for part in parts[1:]:
            if part.names != names:
                raise ValueError(f"datasets disagree on variables: {names} vs {part.names}")
        return cls(
            np.vstack([p.values for p in parts]),
            names,
            tuple(r for p in parts for r in p.regimes),
        )

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def targets(self) -> tuple[tuple[str, ...], ...]:
        """The distinct non-empty intervention targets present, sorted."""
        return tuple(sorted({tuple(sorted(r)) for r in self.regimes if r}))

    def rows_free_of(self, variable: str) -> Array:
        """The rows in which ``variable`` was *not* intervened on — the ones that inform it."""
        return np.array([variable not in regime for regime in self.regimes], dtype=bool)


@dataclass(frozen=True)
class GaussianBIC:
    """Decomposable BIC for linear-Gaussian structure, aware of which rows were randomized.

    ``penalty`` multiplies the usual ``(k/2) log n`` term; ``1.0`` is BIC. It
    is exposed because the right sparsity is a judgement about the problem,
    not a constant, and a search that hides it makes that judgement silently.

    It is a dial, not a monotone one. A greedy search follows a *path*, and
    changing the score changes the path as well as the destination: raising
    the penalty can land the search in a local optimum with *more* edges than
    a lower one found. ``tests/recovery/test_discover_recovery.py`` pins a
    case where it does. Treat a result that changes sharply with the penalty
    as a warning about the search, not a finding about the world.
    """

    data: Dataset
    penalty: float = 1.0
    _cache: dict[tuple[str, frozenset[str]], float] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.penalty <= 0:
            raise ValueError(f"penalty must be positive, got {self.penalty}")

    def local(self, node: str, parents: Iterable[str]) -> float:
        """The score of one variable given a parent set — the term a search caches."""
        key = (node, frozenset(parents))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._compute(node, key[1])
        self._cache[key] = value
        return value

    def _compute(self, node: str, parents: frozenset[str]) -> float:
        names = self.data.names
        if node not in names:
            raise KeyError(f"no variable {node!r}; have {list(names)}")
        unknown = sorted(parents - set(names))
        if unknown:
            raise KeyError(f"unknown parents {unknown}")
        usable = self.data.rows_free_of(node)
        rows = int(np.count_nonzero(usable))
        if rows < len(parents) + 2:
            # Not enough rows to estimate this local term: refuse it rather than
            # returning a score that would win by being unconstrained.
            return -np.inf
        y = self.data.values[usable, names.index(node)]
        columns = [self.data.values[usable, names.index(p)] for p in sorted(parents)]
        design = np.column_stack([np.ones(rows), *columns]) if columns else np.ones((rows, 1))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        variance = float(residual @ residual) / rows
        if variance <= 0:
            variance = np.finfo(float).tiny
        log_likelihood = -0.5 * rows * (np.log(2.0 * np.pi * variance) + 1.0)
        parameters = len(parents) + 2  # coefficients, intercept, variance
        return float(log_likelihood - self.penalty * 0.5 * parameters * np.log(rows))

    def total(self, parents_of: dict[str, frozenset[str]]) -> float:
        """The score of a whole DAG: the sum of its local terms."""
        return sum(self.local(node, parents) for node, parents in parents_of.items())
