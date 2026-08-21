"""sim: data-generating processes with known causal ground truth.

Every recovery test in the suite is written against these worlds. Phase 2
ships the linear-Gaussian SCM; surface and panel worlds arrive in Phase 3.
"""

from axiom.sim.panel import (
    DoseDistribution,
    DosePlan,
    draw_doses,
    long_frame,
    panel_from_arrays,
    unit_labels,
)
from axiom.sim.scm import LinearSCM, SCMError, coefficient_key, latent_key
from axiom.sim.surface_world import (
    Horizon,
    SurfaceWorld,
    TruthMode,
    arms_world,
    surface_world,
    true_parameters,
)
from axiom.sim.worlds import (
    confounded_world,
    feedback_world,
    frontdoor_world,
    hidden_confounder_world,
    iv_world,
    mediator_world,
    transport_pair,
)

__all__ = [
    "DoseDistribution",
    "DosePlan",
    "Horizon",
    "LinearSCM",
    "SCMError",
    "SurfaceWorld",
    "TruthMode",
    "arms_world",
    "coefficient_key",
    "confounded_world",
    "draw_doses",
    "feedback_world",
    "frontdoor_world",
    "hidden_confounder_world",
    "iv_world",
    "latent_key",
    "long_frame",
    "mediator_world",
    "panel_from_arrays",
    "surface_world",
    "transport_pair",
    "true_parameters",
    "unit_labels",
]
