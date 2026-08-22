"""infer: the sampler seam — Backend protocol, Laplace, NumPyro NUTS, PyMC NUTS, diagnostics.

Importing this package never pulls jax, numpyro, pymc, pytensor or arviz into
``sys.modules``; backends import them lazily and ``get_backend`` returns a
typed ``Unsupported`` when an extra is missing.

The PyMC backend is the one with a choice inside it: the same PyTensor graph
runs under PyMC's own NUTS, nutpie, numpyro or blackjax, and
``infer.samplers()`` says which of the four are installed.
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
from axiom.infer.pymc_backend import (
    NUTS_SAMPLERS,
    NutsSampler,
    PymcBackend,
    SamplerStatus,
    samplers,
)

__all__ = [
    "BACKEND_NAMES",
    "Backend",
    "ConvergenceReport",
    "ConvergenceThresholds",
    "LaplaceBackend",
    "NUTS_SAMPLERS",
    "NumpyroBackend",
    "NutsSampler",
    "PymcBackend",
    "SamplerStatus",
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
    "samplers",
    "split_rhat",
    "to_inference_data",
]
