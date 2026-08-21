"""estimands: declarative, content-hashed counterfactual quantities.

Phase 1b ships the declaration half — the facets, the derived dimension, and
``transfer_to``. Realization against a posterior lands in Phase 4.
"""

from axiom.estimands.spec import (
    FACETS,
    Estimand,
    Facet,
    FacetDiff,
    Level,
    Quantity,
    QuantityKind,
    TransferPlan,
    derived_dimension,
)

__all__ = [
    "FACETS",
    "Estimand",
    "Facet",
    "FacetDiff",
    "Level",
    "Quantity",
    "QuantityKind",
    "TransferPlan",
    "derived_dimension",
]
