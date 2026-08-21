"""The backend seam: a protocol, two small specs, and a resolver.

A ``Backend`` takes a ``ModelSpec`` — not a Python callable (review A2) —
and compiles the tree itself. That is what lets a fitted model be saved
without pickle and what keeps every sampler behind this one door.

Nothing here imports a sampler. ``get_backend`` resolves by name and returns
a typed ``Unsupported`` when the extra is missing, so ``import axiom.infer``
costs nothing (gate 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy.typing as npt
from pydantic import field_validator

from axiom.core.model import ModelSpec
from axiom.core.posterior import Posterior
from axiom.core.result import Unsupported, Unverified
from axiom.core.spec import Spec

__all__ = [
    "BACKEND_NAMES",
    "Backend",
    "PointEstimate",
    "SampleSettings",
    "get_backend",
]

BACKEND_NAMES: tuple[str, ...] = ("laplace", "numpyro")
"""Names ``get_backend`` understands. ``laplace`` is always available."""


class PointEstimate(Spec):
    """A mode (MAP) in *constrained* coordinates with the optimizer's verdict.

    ``theta`` maps parameter name to a float (scalar) or tuple (vector
    parameter, flattened in C order). ``log_density`` is the unnormalized
    log posterior that was maximized — in unconstrained coordinates, Jacobian
    included — so two estimates of the same model are comparable.

    ``converged`` means the Newton decrement at the point is below ``1e-8``
    and the Hessian there is positive semi-definite; ``hessian_pd`` (the
    stricter relative test), ``min_eigenvalue`` and ``newton_decrement`` say
    why when it is not. A saddle is never a converged mode.
    """

    theta: dict[str, float | tuple[float, ...]]
    log_density: float
    converged: bool
    method: str
    n_iter: int
    hessian_pd: bool | None = None
    min_eigenvalue: float | None = None
    newton_decrement: float | None = None

    @field_validator("n_iter")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("n_iter must be non-negative")
        return v


class SampleSettings(Spec):
    """How much to sample. The defaults are the usual NUTS defaults."""

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9

    @field_validator("draws", "chains")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be at least 1")
        return v

    @field_validator("tune")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tune must be non-negative")
        return v

    @field_validator("target_accept")
    @classmethod
    def _in_unit(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("target_accept must be in (0, 1)")
        return v


@runtime_checkable
class Backend(Protocol):
    """Something that turns a ``ModelSpec`` plus data into a ``Posterior``.

    Every method records the seed, the backend name, and the sizes it was
    asked for in ``Posterior.provenance``. ``sample`` and ``laplace`` return
    ``Unverified`` instead of draws they cannot stand behind (the Laplace
    path: a mode search that did not converge or a Hessian that is not
    numerically positive definite).
    """

    name: str

    def sample(
        self,
        model: ModelSpec,
        data: Mapping[str, npt.ArrayLike],
        *,
        draws: int,
        tune: int,
        chains: int,
        seed: int | None,
    ) -> Posterior | Unverified: ...

    def optimize(
        self, model: ModelSpec, data: Mapping[str, npt.ArrayLike], *, seed: int | None
    ) -> PointEstimate: ...

    def laplace(
        self,
        model: ModelSpec,
        data: Mapping[str, npt.ArrayLike],
        *,
        draws: int,
        seed: int | None,
    ) -> Posterior | Unverified: ...


def get_backend(name: str) -> Backend | Unsupported:
    """Resolve a backend by name; ``Unsupported`` names the missing extra.

    ``"laplace"`` needs only numpy and scipy. ``"numpyro"`` needs the
    ``axiom[numpyro]`` extra (jax + numpyro). The imports happen here, not at
    module top, so asking is free.
    """
    key = name.strip().lower()
    if key == "laplace":
        from axiom.infer.laplace import LaplaceBackend

        return LaplaceBackend()
    if key == "numpyro":
        from axiom.infer import numpyro_backend

        if not numpyro_backend.available():
            return Unsupported(
                reason="numpyro backend needs jax and numpyro; install axiom[numpyro]",
                missing=numpyro_backend.missing(),
            )
        return numpyro_backend.NumpyroBackend()
    return Unsupported(
        reason=f"unknown backend {name!r}; known backends are {list(BACKEND_NAMES)}",
        detail={"requested": name},
    )
