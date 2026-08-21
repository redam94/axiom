"""Interpreters over ``axiom.core.expr``: one tree, several meanings.

* ``dimension`` — abstract interpretation over ``Dimension``; the type checker.
* ``value`` — numpy evaluation; this *is* ``forward()``.
* ``latex`` — rendering.

The jax interpreter lands in Phase 3 alongside ``infer``.
"""

from axiom.core.interpret.dimension import check, dimension
from axiom.core.interpret.jax import compile as compile_jax
from axiom.core.interpret.jax import (
    compile_log_density,
    jax_available,
    require_jax,
)
from axiom.core.interpret.latex import latex, latex_or_unsupported
from axiom.core.interpret.value import OpaqueFn, OpaqueRegistry, causal_convolve, value

__all__ = [
    "OpaqueFn",
    "compile_jax",
    "compile_log_density",
    "jax_available",
    "require_jax",
    "OpaqueRegistry",
    "causal_convolve",
    "check",
    "dimension",
    "latex",
    "latex_or_unsupported",
    "value",
]
