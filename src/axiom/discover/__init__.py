"""discover: what the data can orient, and what an experiment would buy.

Observation identifies a Markov *equivalence class*, not a DAG. Every member
entails the same independencies, so no amount of observational data separates
them; the honest output of structure learning is the class — an **essential
graph**, directed where every member agrees and undirected where they do not.

Interventions are what break the remaining ties, and they do so in a way that
can be priced before anything is run. Randomizing a variable severs its
incoming edges, so an edge with exactly one endpoint in the target has its
direction revealed:

    from axiom.discover import orientation_gain
    orientation_gain(graph, [["a"]])   # the edges intervening on `a` would settle

That is the design question — *which experiment is worth running* — answered
with graph theory rather than with a pilot study. ``ges`` and ``gies``
(Chickering 2002; Hauser & Bühlmann 2012) are the estimation half: greedy
search over equivalence classes, scored with a decomposable BIC that knows
which rows were randomized.

Two cautions this subpackage states rather than assumes. Every discovery
method here assumes **faithfulness** — that the data's independencies are
exactly the graph's, with no coincidental cancellation — which is not
testable from the data it is assumed about. And greedy is greedy: the search
records every move in ``DiscoveryResult.steps`` so a suspicious answer can be
read back. A discovered graph is a hypothesis, and the rest of ``axiom``
treats it as one.

Layer 5 — reads ``axiom.identify`` for the graph types and ``axiom.core``.
"""

from axiom.discover.essential import (
    EssentialGraph,
    consistent_extension,
    consistent_extensions,
    cpdag,
    interventional_essential_graph,
    markov_equivalent,
    meek_closure,
    orientation_gain,
    v_structures,
)
from axiom.discover.score import Dataset, GaussianBIC
from axiom.discover.search import DiscoveryResult, ges, gies

__all__ = [
    "Dataset",
    "DiscoveryResult",
    "EssentialGraph",
    "GaussianBIC",
    "consistent_extension",
    "consistent_extensions",
    "cpdag",
    "ges",
    "gies",
    "interventional_essential_graph",
    "markov_equivalent",
    "meek_closure",
    "orientation_gain",
    "v_structures",
]
