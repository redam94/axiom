"""Meta-analysis: study records, classical and Bayesian pooling, provenance bias, the
prior handoff, influence diagnostics, and privacy-gated publication.

See ``docs/plan/03-roadmap.md`` (Phase 7) and ``nbs/meta/``.
"""

from __future__ import annotations

from axiom.meta.bias import (
    NO_DUAL_READ,
    delta_identification,
)
from axiom.meta.classical import (
    Heterogeneity,
    PooledEstimate,
    TauEstimate,
    TauMethod,
    fixed_effect,
    heterogeneity,
    prediction_interval,
    random_effects,
    reml_log_likelihood,
    tau_dersimonian_laird,
    tau_paule_mandel,
    tau_reml,
)
from axiom.meta.contribute import (
    record_from_result,
    record_from_summary,
    se_from_summary,
)
from axiom.meta.influence import (
    BaujatData,
    EggerTest,
    ForestData,
    ForestRow,
    FunnelContour,
    FunnelData,
    LeaveOneOut,
    PoolMethod,
    baujat,
    egger,
    forest_data,
    funnel_data,
    leave_one_out,
)
from axiom.meta.ingest import (
    DEFAULT_COLUMNS,
    from_frame,
    normalize,
)
from axiom.meta.moderators import (
    ModeratorDesign,
    moderator_matrix,
)
from axiom.meta.pool import (
    EffectShrinkage,
    ParameterSummary,
    Pooled,
    PoolPriors,
    PoolResult,
    PoolSpec,
    pool,
    pool_model,
)
from axiom.meta.priors import (
    PriorTarget,
    prior_from_pool,
)
from axiom.meta.privacy import (
    Cell,
    EpsilonCharge,
    EpsilonLedger,
    Mechanism,
    PrivacyPolicy,
    cell_from_records,
    charge,
    check_cell,
    contributor_totals,
)
from axiom.meta.publish import (
    EpsilonSplit,
    Release,
    carry_forward,
    gaussian_sigma,
    gaussian_sigma_classical,
    jaccard_distance,
    laplace_scale,
    orthogonal_split,
    release,
)
from axiom.meta.schema import (
    POOLABLE_QUANTITIES,
    Corpus,
    PoolableQuantity,
    ReadKind,
    StudyRecord,
    poolable_quantity,
)
from axiom.meta.store import (
    CorpusStore,
)

__all__ = [
    "BaujatData",
    "Cell",
    "Corpus",
    "CorpusStore",
    "DEFAULT_COLUMNS",
    "EffectShrinkage",
    "EggerTest",
    "EpsilonCharge",
    "EpsilonLedger",
    "EpsilonSplit",
    "ForestData",
    "ForestRow",
    "FunnelContour",
    "FunnelData",
    "Heterogeneity",
    "LeaveOneOut",
    "Mechanism",
    "ModeratorDesign",
    "NO_DUAL_READ",
    "POOLABLE_QUANTITIES",
    "ParameterSummary",
    "PoolMethod",
    "PoolPriors",
    "PoolResult",
    "PoolSpec",
    "PoolableQuantity",
    "Pooled",
    "PooledEstimate",
    "PriorTarget",
    "PrivacyPolicy",
    "ReadKind",
    "Release",
    "StudyRecord",
    "TauEstimate",
    "TauMethod",
    "baujat",
    "carry_forward",
    "cell_from_records",
    "charge",
    "check_cell",
    "contributor_totals",
    "delta_identification",
    "egger",
    "fixed_effect",
    "forest_data",
    "from_frame",
    "funnel_data",
    "gaussian_sigma",
    "gaussian_sigma_classical",
    "heterogeneity",
    "jaccard_distance",
    "laplace_scale",
    "leave_one_out",
    "moderator_matrix",
    "normalize",
    "orthogonal_split",
    "pool",
    "pool_model",
    "poolable_quantity",
    "prediction_interval",
    "prior_from_pool",
    "random_effects",
    "record_from_result",
    "record_from_summary",
    "release",
    "reml_log_likelihood",
    "se_from_summary",
    "tau_dersimonian_laird",
    "tau_paule_mandel",
    "tau_reml",
]
