"""Block-triangular decomposition: which equations are genuinely simultaneous.

A system is *recursive* when its contemporaneous dependencies can be ordered
— then it is already a DAG within a period and nothing needs solving. It is
*simultaneous* where they cannot, and the strongly connected components of
the contemporaneous graph are exactly the sets that have to be solved
together (Tarjan 1972; the econometric "block-recursive" form of Fisher
1966, ch. 4).

The decomposition matters twice over:

* it says which blocks ``dynamics.solve`` has to invert, and how large the
  inversion is — a system of forty equations with two-variable cycles is
  forty scalar substitutions and two 2x2 solves, not one 40x40 solve;
* it is the honest place to report simultaneity to the reader. A cycle is
  not an error, but it *is* a claim: these variables are determined
  together, and no ordering of them is causal within the period.

Lagged edges never enter a block: ``y.l1`` is a different node from ``y``,
which is precisely why unrolling a dynamic system yields a DAG.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec
from axiom.dynamics.spec import DynamicSystem

__all__ = ["Block", "BlockOrder", "block_order", "strongly_connected_components"]


def strongly_connected_components(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, ...], ...]:
    """Tarjan's SCCs, each sorted, the tuple in reverse topological order (sources last).

    Iterative, so a long chain cannot overflow the stack. ``edges`` are
    ``(from, to)``; nodes not in ``nodes`` are ignored.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[tuple[str, ...]] = []
    counter = 0
    adjacency: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].append(b)

    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_i = work[-1]
            if child_i == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            recursed = False
            neighbours = adjacency[node]
            while child_i < len(neighbours):
                nxt = neighbours[child_i]
                child_i += 1
                if nxt not in index:
                    work[-1] = (node, child_i)
                    work.append((nxt, 0))
                    recursed = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            else:
                work[-1] = (node, child_i)
            if recursed:
                continue
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == node:
                        break
                out.append(tuple(sorted(component)))
            work.pop()
            if work:
                parent, _ = work[-1]
                low[parent] = min(low[parent], low[node])
    return tuple(out)


class Block(Spec):
    """One set of variables solved together, and whether it is a cycle.

    ``simultaneous`` is true when the block holds more than one variable, or
    one variable whose own equation reads it at lag 0 (``y = f(y, ...)``, an
    implicit definition). A block that is not simultaneous is a single
    substitution.
    """

    variables: tuple[NonEmptyStr, ...] = Field(min_length=1)
    simultaneous: bool

    @property
    def size(self) -> int:
        return len(self.variables)


class BlockOrder(Spec):
    """The system's blocks in solution order, sources first.

    Solving in this order means every reference a block makes to another
    endogenous variable at lag 0 has already been resolved, so the only
    unknowns left inside a block are its own members.
    """

    blocks: tuple[Block, ...] = ()
    system: str = ""

    @model_validator(mode="after")
    def _distinct(self) -> BlockOrder:
        seen: set[str] = set()
        for b in self.blocks:
            for v in b.variables:
                if v in seen:
                    raise ValueError(f"variable {v!r} appears in two blocks")
                seen.add(v)
        return self

    @property
    def recursive(self) -> bool:
        """True when no block is simultaneous — the period is already a DAG."""
        return not any(b.simultaneous for b in self.blocks)

    @property
    def order(self) -> tuple[str, ...]:
        """Every endogenous variable, in solution order."""
        return tuple(v for b in self.blocks for v in b.variables)

    @property
    def simultaneous_blocks(self) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if b.simultaneous)

    @property
    def largest_block(self) -> int:
        return max((b.size for b in self.blocks), default=0)


def block_order(system: DynamicSystem) -> BlockOrder:
    """Block-triangular decomposition of the contemporaneous dependency graph."""
    endogenous = system.endogenous
    edges = tuple(
        (u, v) for u, v in system.contemporaneous_edges() if u in endogenous and v in endogenous
    )
    self_loops = {
        eq.target for eq in system.equations for u, lag in eq.refs if lag == 0 and u == eq.target
    }
    components = strongly_connected_components(endogenous, edges)
    blocks = [
        Block(
            variables=component,
            simultaneous=len(component) > 1 or component[0] in self_loops,
        )
        for component in reversed(components)
    ]
    return BlockOrder(blocks=tuple(blocks), system=system.name)
