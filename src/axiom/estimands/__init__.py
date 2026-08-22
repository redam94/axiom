"""estimands: declarative, content-hashed counterfactual quantities.

The declaration half (facets, derived dimension, ``transfer_to``) and the
realization half (``realize`` against any ``SupportsEstimands`` producer,
the standard registry, and the estimand as an expression tree). Imports
``core`` only.
"""

from axiom.estimands.evaluate import (
    EstimandResult,
    RealizedDraws,
    ResultStatus,
    evaluate,
    realize,
)
from axiom.estimands.graph import check_estimand_dimension, estimand_expr, substitute
from axiom.estimands.registry import EstimandRegistry, standard_estimands
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
    "EstimandRegistry",
    "EstimandResult",
    "Facet",
    "FacetDiff",
    "Level",
    "Quantity",
    "QuantityKind",
    "RealizedDraws",
    "ResultStatus",
    "TransferPlan",
    "check_estimand_dimension",
    "derived_dimension",
    "estimand_expr",
    "evaluate",
    "realize",
    "standard_estimands",
    "substitute",
]
