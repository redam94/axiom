"""``GraphBuilder``: edges, unmeasured nodes → ``identify.CausalGraph``; plus an ``Estimand``.

Rewritten from the parent's ``builders/model.py`` (ledger: REWRITE). The
graph side is thin — ``CausalGraph`` normalizes and validates itself — but
the builder gives a place to say ``.edge("Z", "X").edge("X", "Y")
.unmeasured("U")`` and to declare the quantity of interest on the same
object: ``estimand(treatment, outcome, ...)`` checks both entities name
nodes of the graph and hands the rest to ``estimands.Estimand`` with the
dimension derived from the entities (``derived_dimension``), so an estimand
built here cannot disagree with its own facets.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from axiom.build.base import BuildError, Fields
from axiom.core import Intervention, Outcome, Population, TimeWindow, Treatment
from axiom.estimands import Estimand, Level, Quantity, QuantityKind, derived_dimension
from axiom.identify import CausalGraph

__all__ = ["GraphBuilder"]


@dataclass(frozen=True)
class GraphBuilder:
    """``GraphBuilder().edge("Z", "X").edge("X", "Y").unmeasured("U").build()``."""

    fields: Fields = Fields()

    def with_(self, **updates: Any) -> GraphBuilder:
        return replace(self, fields=self.fields.with_(**updates))

    def name(self, name: str) -> GraphBuilder:
        return self.with_(name=name)

    def edge(self, source: str, target: str) -> GraphBuilder:
        edges: tuple[tuple[str, str], ...] = self.fields.get("edges", ())
        return self.with_(edges=(*edges, (str(source), str(target))))

    def edges(self, text: str) -> GraphBuilder:
        """Add edges from ``"Z -> X, X -> Y, X <-> W"`` text (``identify.parse_edges``)."""
        from axiom.identify import parse_edges

        directed, bi = parse_edges(text)
        out = self
        for a, b in directed:
            out = out.edge(a, b)
        for a, b in bi:
            out = out.bidirected(a, b)
        return out

    def bidirected(self, a: str, b: str) -> GraphBuilder:
        pairs: tuple[tuple[str, str], ...] = self.fields.get("bidirected", ())
        return self.with_(bidirected=(*pairs, (str(a), str(b))))

    def node(self, *names: str) -> GraphBuilder:
        nodes: tuple[str, ...] = self.fields.get("nodes", ())
        return self.with_(nodes=(*nodes, *(str(n) for n in names)))

    def unmeasured(self, *names: str) -> GraphBuilder:
        current: tuple[str, ...] = self.fields.get("unmeasured", ())
        return self.with_(unmeasured=(*current, *(str(n) for n in names))).node(*names)

    def selection(self, *names: str) -> GraphBuilder:
        current: tuple[str, ...] = self.fields.get("selection", ())
        return self.with_(selection=(*current, *(str(n) for n in names))).node(*names)

    def feedback(self, on: bool = True) -> GraphBuilder:
        return self.with_(feedback=on)

    def build(self) -> CausalGraph:
        if not self.fields.get("edges") and not self.fields.get("nodes"):
            raise BuildError("GraphBuilder: add at least one edge or node before build()")
        keys = {"nodes", "edges", "bidirected", "unmeasured", "selection", "feedback", "name"}
        payload = {k: v for k, v in self.fields.to_dict().items() if k in keys}
        return CausalGraph(**payload)

    def estimand(
        self,
        treatment: Treatment,
        outcome: Outcome,
        *,
        kind: QuantityKind = "contrast",
        dose: float,
        reference_dose: float | None = 0.0,
        window: TimeWindow | None = None,
        population: Population | None = None,
        level: Level | None = None,
        conditioning: Iterable[str] = (),
        version: str = "unspecified",
        name: str | None = None,
    ) -> Estimand:
        """An ``Estimand`` on this graph's treatment and outcome nodes.

        ``treatment.name`` and ``outcome.name`` must be nodes of the graph.
        ``window`` defaults to one period, ``population`` to ``"all"``, and
        ``level`` to the individual unit without interference.
        ``reference_dose`` is required for a contrast, ratio, or area and
        ignored for a marginal or elasticity. The dimension is derived.
        """
        graph = self.build()
        window = window if window is not None else TimeWindow(start=0, stop=1)
        population = population if population is not None else Population(name="all")
        level = level if level is not None else Level(unit="individual")
        for entity in (treatment, outcome):
            if entity.name not in graph.nodes:
                raise BuildError(
                    f"{entity.name!r} is not a node of the graph; nodes are {list(graph.nodes)}"
                )
        if treatment.dimension is None or outcome.dimension is None:
            raise BuildError("treatment and outcome must carry dimensions to derive the estimand's")
        needs_ref = kind in ("contrast", "ratio", "area")
        if needs_ref and reference_dose is None:
            raise BuildError(f"a {kind!r} estimand needs reference_dose")
        at = Intervention(doses={treatment.name: float(dose)}, version=version)
        ref = (
            Intervention(doses={treatment.name: float(reference_dose)}, version=version)
            if needs_ref and reference_dose is not None
            else None
        )
        return Estimand(
            name=name or f"{kind}_{treatment.name}_{outcome.name}",
            quantity=Quantity(kind=kind),
            treatment=treatment,
            intervention=at,
            reference=ref,
            outcome=outcome,
            population=population,
            window=window,
            level=level,
            conditioning=tuple(sorted(set(conditioning))),
            dimension=derived_dimension(kind, outcome.dimension, treatment.dimension),
        )
