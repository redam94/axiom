"""How much of a discovered graph survives resampling the data.

A structure search returns one graph and says nothing about how much of it was
determined by the data and how much by the last few rows. That is the gap
between "here is a graph" and something a person can act on, and it closes with
a loop that costs nothing conceptual: resample the rows, run the search again,
count how often each edge comes back.

What comes out is per-edge rather than per-graph, which is the useful
granularity. A discovered graph is rarely all right or all wrong; it is usually
a stable core with a fringe of edges that appear in half the resamples. The
core is what a decision should rest on and the fringe is what an experiment
should target — and ``discover.orientation_gain`` prices that experiment.

Three numbers per pair, because an edge can be unstable in two different ways:

* ``adjacent`` — how often the two variables are connected at all. Low here
  means the *adjacency* is in doubt.
* ``forward`` / ``backward`` / ``undirected`` — given a connection, how often
  it pointed each way. A pair with ``adjacent = 1.0`` and ``forward = 0.5``
  is not an unstable edge; it is a **stable edge whose direction the data
  cannot settle**, which is a completely different finding and calls for an
  intervention rather than more rows.

Bootstrap resampling measures sampling variability, and only that. It cannot
see a bias shared by every resample: a confounder missing from the data is
missing from all of them, and shows up as a beautifully stable wrong edge.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec
from axiom.discover.score import Dataset, GaussianBIC
from axiom.discover.search import ges, gies

__all__ = ["EdgeSupport", "StabilityReport", "edge_stability"]


class EdgeSupport(Spec):
    """How often one pair of variables was joined, and which way it pointed.

    ``forward``, ``backward`` and ``undirected`` are fractions of *all*
    resamples, so they sum to ``adjacent``.
    """

    a: NonEmptyStr
    b: NonEmptyStr
    adjacent: float = Field(ge=0.0, le=1.0)
    forward: float = Field(ge=0.0, le=1.0)
    backward: float = Field(ge=0.0, le=1.0)
    undirected: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> EdgeSupport:
        """Store the ends sorted, swapping the two directions along with them."""
        if self.a == self.b:
            raise ValueError(f"an edge joins two variables, not {self.a!r} to itself")
        if self.a > self.b:
            a, b, forward, backward = self.b, self.a, self.backward, self.forward
            object.__setattr__(self, "a", a)
            object.__setattr__(self, "b", b)
            object.__setattr__(self, "forward", forward)
            object.__setattr__(self, "backward", backward)
        total = self.forward + self.backward + self.undirected
        if abs(total - self.adjacent) > 1e-9:
            raise ValueError(
                f"the orientations sum to {total:.6f} but the pair is adjacent "
                f"{self.adjacent:.6f} of the time"
            )
        return self

    @property
    def oriented(self) -> float:
        """How often the search committed to a direction, either way."""
        return self.forward + self.backward

    @property
    def decisive(self) -> float:
        """The dominant direction's share of the resamples that oriented it at all."""
        return max(self.forward, self.backward) / self.oriented if self.oriented else 0.0

    def describe(self) -> str:
        if self.adjacent < 0.5:
            return f"{self.a} - {self.b}: present in only {self.adjacent:.0%} of resamples"
        if self.oriented < 0.5 * self.adjacent:
            return f"{self.a} - {self.b}: {self.adjacent:.0%} adjacent, direction not settled"
        arrow = (
            f"{self.a} -> {self.b}" if self.forward >= self.backward else f"{self.b} -> {self.a}"
        )
        return f"{arrow}: {self.adjacent:.0%} adjacent, {self.decisive:.0%} of those agree"


class StabilityReport(Spec):
    """Per-edge support across bootstrap resamples of the same data.

    ``edges`` is sorted by how often the pair was adjacent, so the stable core
    is at the top and the fringe at the bottom.
    """

    variables: tuple[NonEmptyStr, ...] = Field(min_length=2)
    edges: tuple[EdgeSupport, ...] = ()
    n_bootstrap: int = Field(ge=1)
    n_rows: int = Field(ge=1)
    targets: tuple[tuple[str, ...], ...] = ()
    penalty: float = Field(gt=0)
    seed: int | None = None
    detail: dict[str, str] = {}

    def stable(self, threshold: float = 0.8) -> tuple[EdgeSupport, ...]:
        """Edges present in at least ``threshold`` of the resamples."""
        return tuple(e for e in self.edges if e.adjacent >= threshold)

    def contested(self, low: float = 0.2, high: float = 0.8) -> tuple[EdgeSupport, ...]:
        """Edges the resamples disagree about — the fringe an experiment should target."""
        return tuple(e for e in self.edges if low <= e.adjacent < high)

    def undecided_direction(self, threshold: float = 0.8) -> tuple[EdgeSupport, ...]:
        """Edges that are reliably *there* and reliably not oriented.

        The finding that more rows will not fix: observational data has said
        everything it can about these, and only an intervention will settle
        them.
        """
        return tuple(
            e for e in self.edges if e.adjacent >= threshold and e.undirected > 0.5 * e.adjacent
        )

    def summary(self) -> str:
        return (
            f"{len(self.stable())} stable edges, {len(self.contested())} contested, "
            f"{len(self.undecided_direction())} present but unoriented, "
            f"over {self.n_bootstrap} resamples"
        )


def edge_stability(
    data: Dataset,
    *,
    n_bootstrap: int = 100,
    penalty: float = 1.0,
    targets: Sequence[Iterable[str]] | None = None,
    seed: int | None = None,
) -> StabilityReport:
    """Resample the rows, re-run the search, and count how often each edge returns.

    ``targets`` chooses the search: ``None`` runs ``gies`` when the data has
    intervention targets and ``ges`` when it does not; an explicit family
    forces ``gies``. Resampling is over rows, so each resample keeps the
    regime label attached to the row it came from — an interventional row
    stays interventional.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least one, got {n_bootstrap}")
    family = list(targets) if targets is not None else [list(t) for t in data.targets]
    rng = np.random.default_rng(seed)
    counts: dict[tuple[str, str], list[float]] = {}
    names = data.names
    for a_index, a in enumerate(names):
        for b in names[a_index + 1 :]:
            counts[(a, b) if a <= b else (b, a)] = [0.0, 0.0, 0.0]

    for _ in range(n_bootstrap):
        rows = rng.integers(0, data.n_rows, size=data.n_rows)
        resample = Dataset(data.values[rows], names, tuple(data.regimes[int(i)] for i in rows))
        score = GaussianBIC(resample, penalty=penalty)
        found = gies(score, family) if family else ges(score)
        for a, b in found.essential.directed:
            key = (a, b) if a <= b else (b, a)
            counts[key][0 if key == (a, b) else 1] += 1
        for a, b in found.essential.undirected:
            counts[(a, b) if a <= b else (b, a)][2] += 1

    edges = []
    for (a, b), (forward, backward, undirected) in counts.items():
        adjacent = (forward + backward + undirected) / n_bootstrap
        if adjacent == 0.0:
            continue
        edges.append(
            EdgeSupport(
                a=a,
                b=b,
                adjacent=adjacent,
                forward=forward / n_bootstrap,
                backward=backward / n_bootstrap,
                undirected=undirected / n_bootstrap,
            )
        )
    edges.sort(key=lambda e: (-e.adjacent, e.a, e.b))
    return StabilityReport(
        variables=names,
        edges=tuple(edges),
        n_bootstrap=n_bootstrap,
        n_rows=data.n_rows,
        targets=tuple(tuple(sorted(t)) for t in family),
        penalty=penalty,
        seed=seed,
        detail={"search": "gies" if family else "ges"},
    )
