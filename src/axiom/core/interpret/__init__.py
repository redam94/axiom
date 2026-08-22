"""Interpreters over ``axiom.core.expr``: one tree, several meanings.

* ``dimension`` — abstract interpretation over ``Dimension``; the type checker.
* ``value`` — numpy evaluation; this *is* ``forward()``.
* ``latex`` — rendering.

* ``jax`` — a traced graph for NumPyro.
* ``pytensor`` — a symbolic graph for PyMC and its four NUTS samplers.
"""

from axiom.core.interpret.dimension import check, dimension
from axiom.core.interpret.jax import compile as compile_jax
from axiom.core.interpret.jax import (
    compile_log_density,
    jax_available,
    require_jax,
)
from axiom.core.interpret.latex import latex, latex_or_unsupported
from axiom.core.interpret.pytensor import compile as compile_pytensor
from axiom.core.interpret.pytensor import (
    compile_log_density as compile_log_density_pytensor,
)
from axiom.core.interpret.pytensor import pytensor_available, require_pytensor
from axiom.core.interpret.value import OpaqueFn, OpaqueRegistry, causal_convolve, value

__all__ = [
    "OpaqueFn",
    "OpaqueRegistry",
    "causal_convolve",
    "check",
    "compile_jax",
    "compile_log_density",
    "compile_log_density_pytensor",
    "compile_pytensor",
    "dimension",
    "jax_available",
    "latex",
    "latex_or_unsupported",
    "pytensor_available",
    "require_jax",
    "require_pytensor",
    "value",
]
