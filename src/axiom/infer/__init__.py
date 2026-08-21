"""infer: the sampler seam — Backend protocol, Laplace, NumPyro NUTS, and diagnostics.

Importing this package never pulls jax, numpyro, or arviz into
``sys.modules``; backends import them lazily and ``get_backend`` returns a
typed ``Unsupported`` when an extra is missing.
"""

from axiom.core.posterior import Posterior
from axiom.infer._arviz import from_inference_data, to_inference_data
from axiom.infer.backend import BACKEND_NAMES, Backend, PointEstimate, SampleSettings, get_backend
from axiom.infer.diagnostics import (
    ConvergenceReport,
    ConvergenceThresholds,
    ParameterDiagnostics,
    diagnose,
    ess_bulk,
    ess_tail,
    mcse_mean,
    split_rhat,
)
from axiom.infer.laplace import LaplaceBackend, find_mode, hessian_at, laplace
from axiom.infer.numpyro_backend import NumpyroBackend

__all__ = [
    "BACKEND_NAMES",
    "Backend",
    "ConvergenceReport",
    "ConvergenceThresholds",
    "LaplaceBackend",
    "NumpyroBackend",
    "ParameterDiagnostics",
    "PointEstimate",
    "Posterior",
    "SampleSettings",
    "diagnose",
    "ess_bulk",
    "ess_tail",
    "find_mode",
    "from_inference_data",
    "get_backend",
    "hessian_at",
    "laplace",
    "mcse_mean",
    "split_rhat",
    "to_inference_data",
]
