"""sim: data-generating processes with known causal ground truth.

Every recovery test in the suite is written against these worlds. Phase 2
ships the linear-Gaussian SCM; surface and panel worlds arrive in Phase 3.
"""

from axiom.sim.scm import LinearSCM, SCMError, coefficient_key, latent_key
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
    "LinearSCM",
    "SCMError",
    "coefficient_key",
    "confounded_world",
    "feedback_world",
    "frontdoor_world",
    "hidden_confounder_world",
    "iv_world",
    "latent_key",
    "mediator_world",
    "transport_pair",
]
