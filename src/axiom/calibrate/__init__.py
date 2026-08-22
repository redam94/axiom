"""Folding experimental evidence into surface models: evidence records, the prior and
likelihood routes, scope transfer with typed corrections, and the assumption ledger.

See ``docs/plan/03-roadmap.md`` (Phase 6) and ``nbs/calibrate/``.
"""

from __future__ import annotations

from axiom.calibrate.check import (
    Agreement,
    AgreementVerdict,
    agreement,
)
from axiom.calibrate.evidence import (
    Measurement,
    combine_inverse_variance,
)
from axiom.calibrate.ledger import (
    FACET_PREFIX,
    UNCORRECTED,
    Ledger,
    facet_of,
)
from axiom.calibrate.likelihood import (
    attach,
    constraint_for,
    fit_calibrated,
    lognormal_mu_from_moments,
    lognormal_sigma_from_moments,
)
from axiom.calibrate.prior import (
    CalibratedSpec,
    PriorFamily,
    amplitude_prior,
    combine_measurements,
    derive_prior,
    design_factor,
    lognormal_from_moments,
    mean_sd_to_gamma,
)
from axiom.calibrate.transfer import (
    Correction,
    CorrectionKind,
    ResolvedTransfer,
    aggregation_level,
    carryover_window_factor,
    chord_to_marginal,
    dose_path_accumulation,
    resolve,
    variance_reweight,
)

__all__ = [
    "Agreement",
    "AgreementVerdict",
    "CalibratedSpec",
    "Correction",
    "CorrectionKind",
    "FACET_PREFIX",
    "Ledger",
    "Measurement",
    "PriorFamily",
    "ResolvedTransfer",
    "UNCORRECTED",
    "aggregation_level",
    "agreement",
    "amplitude_prior",
    "attach",
    "carryover_window_factor",
    "chord_to_marginal",
    "combine_inverse_variance",
    "combine_measurements",
    "constraint_for",
    "derive_prior",
    "design_factor",
    "dose_path_accumulation",
    "facet_of",
    "fit_calibrated",
    "lognormal_from_moments",
    "lognormal_mu_from_moments",
    "lognormal_sigma_from_moments",
    "mean_sd_to_gamma",
    "resolve",
    "variance_reweight",
]
