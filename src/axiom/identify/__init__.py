"""identify: DAGs, adjustment sets, front-door and IV routes, transport, honest verdicts.

Imports ``axiom.core`` and ``axiom.dynamics`` (for the unrolled graph of a
system with feedback or simultaneity). ``networkx`` is not a dependency.

``CausalGraph`` is the acyclic case and ``MixedGraph`` the general one: a
cycle is not an error, it is a different separation criterion
(``sigma_separated``), and ``acyclify`` converts between them.

``identify`` searches a menu of named routes; ``identify_effect`` runs the ID
algorithm, which is sound *and complete* under latent confounding and returns
the estimand as a ``Formula`` or a ``Hedge`` proving there is none. ``swig``
puts the potential outcome on the graph, so the ignorability an estimator
assumes is visible rather than implied.
"""

from axiom.identify.backdoor import (
    Role,
    RoleAssignment,
    adjustment_sets,
    admissible_set_exists,
    assign_roles,
    backdoor_admissible,
    canonical_adjustment_set,
    minimal_adjustment_sets,
    requires_unmeasured,
    roles,
)
from axiom.identify.cluster import ClusterDAG, compatible, identify_cluster_effect
from axiom.identify.cyclic import MixedGraph, acyclify, sigma_separated
from axiom.identify.dynamic import (
    SequentialPlan,
    sequential_backdoor_admissible,
    sequential_plan,
    unrolled_graph,
    unrolled_mixed_graph,
)
from axiom.identify.endogeneity import EndogeneityTest, durbin_wu_hausman, hausman_iv_vs_ols
from axiom.identify.estimators import (
    LinearEstimate,
    frontdoor_linear,
    ols,
    two_stage_least_squares,
    weak_instrument_check,
)
from axiom.identify.formula import Density, Formula, JointTable, Marginal, Product, Ratio
from axiom.identify.frontdoor import (
    FrontDoorRoute,
    InstrumentRoute,
    conditional_instruments,
    frontdoor_admissible,
    frontdoor_sets,
    instrument_admissible,
    instruments,
)
from axiom.identify.graph import CausalGraph, GraphError, parse_edges
from axiom.identify.id_algorithm import (
    Hedge,
    IdentifiedEffect,
    districts,
    identify_conditional_effect,
    identify_effect,
    latent_projection,
)
from axiom.identify.swig import (
    sequential_ignorability,
    single_world_ignorability,
    swig,
)
from axiom.identify.transport import (
    TransportVerdict,
    directly_transportable,
    minimal_s_admissible_sets,
    s_admissible,
    s_admissible_sets,
    selection_diagram,
    transport_verdict,
    trivially_transportable,
)
from axiom.identify.verdict import IdentificationVerdict, Route, identify

__all__ = [
    "CausalGraph",
    "ClusterDAG",
    "Density",
    "EndogeneityTest",
    "Formula",
    "FrontDoorRoute",
    "GraphError",
    "Hedge",
    "IdentificationVerdict",
    "IdentifiedEffect",
    "InstrumentRoute",
    "JointTable",
    "LinearEstimate",
    "Marginal",
    "MixedGraph",
    "Product",
    "Ratio",
    "Role",
    "RoleAssignment",
    "Route",
    "SequentialPlan",
    "TransportVerdict",
    "acyclify",
    "adjustment_sets",
    "admissible_set_exists",
    "assign_roles",
    "backdoor_admissible",
    "canonical_adjustment_set",
    "compatible",
    "conditional_instruments",
    "directly_transportable",
    "districts",
    "durbin_wu_hausman",
    "frontdoor_admissible",
    "frontdoor_linear",
    "frontdoor_sets",
    "hausman_iv_vs_ols",
    "identify",
    "identify_cluster_effect",
    "identify_conditional_effect",
    "identify_effect",
    "instrument_admissible",
    "instruments",
    "latent_projection",
    "minimal_adjustment_sets",
    "minimal_s_admissible_sets",
    "ols",
    "parse_edges",
    "requires_unmeasured",
    "roles",
    "s_admissible",
    "s_admissible_sets",
    "selection_diagram",
    "sequential_backdoor_admissible",
    "sequential_ignorability",
    "sequential_plan",
    "sigma_separated",
    "single_world_ignorability",
    "swig",
    "transport_verdict",
    "trivially_transportable",
    "two_stage_least_squares",
    "unrolled_graph",
    "unrolled_mixed_graph",
    "weak_instrument_check",
]
