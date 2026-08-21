"""infer: the sampler seam. Backends are optional extras; ``Posterior`` is sampler-free.

``Posterior`` is defined in ``axiom.core.posterior`` (so ``io`` can persist
it without importing upward) and re-exported here because ``infer`` is where
a user expects to find it. Nothing imported here may pull jax, numpyro, or
pymc into ``sys.modules``.
"""

from axiom.core.posterior import Posterior

__all__ = ["Posterior"]
