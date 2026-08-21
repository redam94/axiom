"""The NumPyro backend: NUTS over the jax-compiled log density of a ``ModelSpec``.

No numpyro model function is written anywhere: ``potential_fn`` is
``-compile_log_density(model)(data, z)`` over the unconstrained parameter
dict, so the likelihood NUTS sees is the same tree ``axiom.core.value``
evaluates (review A2). Draws come back in unconstrained coordinates and are
mapped through ``axiom.core.constrain`` — the same transform the Laplace
path uses — into a ``Posterior``.

Every jax / numpyro import lives inside a function; ``available()`` says
whether the extra is installed.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from axiom.core.interpret.jax import compile_log_density
from axiom.core.model import ModelSpec, free_parameters, unconstrain
from axiom.core.posterior import Posterior
from axiom.core.result import Unverified
from axiom.infer.backend import PointEstimate
from axiom.infer.laplace import (
    _coords,
    _draw_seed,
    _fixed_values,
    _mode_estimate,
    _model_data,
    constrain_draws,
    find_mode,
    flat_layout,
)

__all__ = ["NumpyroBackend", "available", "missing", "sample"]

log = logging.getLogger(__name__)


def missing() -> tuple[str, ...]:
    """Which of ``jax`` and ``numpyro`` cannot be imported."""
    out: list[str] = []
    for mod in ("jax", "numpyro"):
        try:
            importlib.import_module(mod)
        except ImportError:
            out.append(mod)
    return tuple(out)


def available() -> bool:
    return not missing()


def _validate_nuts(target_accept: float, max_tree_depth: int) -> None:
    if not 0.0 < target_accept < 1.0:
        raise ValueError(f"target_accept must be in (0, 1); got {target_accept}")
    if max_tree_depth < 1:
        raise ValueError(f"max_tree_depth must be at least 1; got {max_tree_depth}")


def _init_z(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    init: Mapping[str, npt.ArrayLike] | None,
    *,
    from_mode: bool,
    seed: int,
) -> tuple[dict[str, npt.NDArray[np.float64]], str, str | None]:
    """Unconstrained starting point: a user point, the Laplace mode, or zeros.

    A mode start is taken only from a *verified* mode — converged **and**
    numerically positive-definite Hessian. A converged point on a flat or
    ill-posed direction (a collinear pair, a funnel's neck) is not a place
    to start chains. Returns ``(z, source, note)``; ``note`` says why a
    requested mode start fell back to zeros so provenance can record it.
    """
    layout = flat_layout(model)
    fixed = _fixed_values(model)
    if init is not None:
        # ``unconstrain`` needs the fixed parameters too: an interval bound may be one.
        z = unconstrain(model, {**fixed, **init})
        return {k: np.asarray(v, dtype=float) for k, v in z.items()}, "init", None
    if from_mode:
        mode = _mode_estimate(model, data, init=None, derivatives="jax", seed=seed)
        verified = isinstance(mode, PointEstimate) and mode.converged and mode.hessian_pd is True
        if isinstance(mode, PointEstimate) and verified:
            theta = {
                name: np.asarray(mode.theta[name], dtype=float).reshape(shape)
                for name, shape in layout
            }
            z = unconstrain(model, {**fixed, **theta})
            return {k: np.asarray(v, dtype=float) for k, v in z.items()}, "laplace_mode", None
        if isinstance(mode, Unverified):
            note = f"mode search unverified: {mode.reason}"
        elif not mode.converged:
            note = (
                f"mode search did not converge (optimizer {mode.method}, "
                f"hessian_pd={mode.hessian_pd}, newton_decrement={mode.newton_decrement})"
            )
        else:
            note = (
                f"mode is not verified: Hessian not positive definite (optimizer {mode.method}, "
                f"min_eigenvalue={mode.min_eigenvalue})"
            )
        log.warning("numpyro: %s; initializing chains at zero plus jitter", note)
        return {name: np.zeros(shape) for name, shape in layout}, "zeros", note
    return {name: np.zeros(shape) for name, shape in layout}, "zeros", None


def sample(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int | None,
    target_accept: float = 0.9,
    init: Mapping[str, npt.ArrayLike] | None = None,
    init_from_mode: bool = True,
    jitter: float = 0.1,
    max_tree_depth: int = 10,
) -> Posterior:
    """Run NUTS and return constrained draws with NUTS diagnostics in provenance.

    Chains start at the Laplace mode (or ``init``, or zero), each jittered by
    ``N(0, jitter²)`` in unconstrained space so they are not identical.
    Provenance records ``divergences`` (total over chains), ``accept_prob``
    (mean), ``max_tree_depth_hits`` (transitions whose tree reached
    ``max_tree_depth``), ``mean_num_steps``, ``seed`` (drawn and recorded
    when ``None`` was passed), sizes, and the model hash.
    """
    if draws < 1 or chains < 1 or tune < 0:
        raise ValueError("need draws >= 1, chains >= 1, tune >= 0")
    _validate_nuts(target_accept, max_tree_depth)
    import jax
    import jax.numpy as jnp

    # numpyro ships no type information; go through importlib so mypy sees ``Any``.
    numpyro: Any = importlib.import_module("numpyro")
    numpyro_infer: Any = importlib.import_module("numpyro.infer")
    numpyro.enable_x64()
    NUTS, MCMC = numpyro_infer.NUTS, numpyro_infer.MCMC

    layout = flat_layout(model)
    if not layout:
        raise ValueError(f"model {model.name!r} has no free parameters")
    used_seed = _draw_seed() if seed is None else int(seed)
    cols = _model_data(model, data)
    jdata = {k: jnp.asarray(v) for k, v in cols.items()}
    ld = compile_log_density(model)

    def potential(z: Mapping[str, Any]) -> Any:
        return -ld(jdata, z)

    z0, init_source, init_note = _init_z(
        model, data, init, from_mode=init_from_mode, seed=used_seed
    )
    rng = np.random.default_rng(used_seed)
    init_params = {
        name: jnp.asarray(
            z0[name][None, ...] + jitter * rng.standard_normal((chains, *np.shape(z0[name])))
        )
        for name, _ in layout
    }
    if chains == 1:
        init_params = {k: v[0] for k, v in init_params.items()}

    kernel = NUTS(
        potential_fn=potential,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=False,
    )
    key = jax.random.PRNGKey(used_seed)
    mcmc.run(key, init_params=init_params, extra_fields=("diverging", "accept_prob", "num_steps"))
    z_samples = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    divergences = int(np.sum(np.asarray(extra["diverging"])))
    accept = float(np.mean(np.asarray(extra["accept_prob"])))
    num_steps = np.asarray(extra["num_steps"])
    # a tree of depth d holds 2^d - 1 leapfrog steps; reaching the cap means the
    # trajectory was truncated rather than U-turned
    depth_hits = int(np.sum(num_steps >= 2**max_tree_depth - 1))

    z_flat = {
        name: np.asarray(z_samples[name], dtype=float).reshape((chains * draws, *shape))
        for name, shape in layout
    }
    theta = constrain_draws(model, z_flat)
    n = chains * draws
    bad = np.zeros(n, dtype=bool)
    for v in theta.values():
        bad |= ~np.all(np.isfinite(v.reshape(n, -1)), axis=1)
    nonfinite_frac = float(bad.mean())
    if divergences:
        log.warning(
            "numpyro: %d divergent transitions in %d draws of %r", divergences, n, model.name
        )
    if depth_hits:
        log.warning(
            "numpyro: %d of %d transitions hit max_tree_depth=%d on %r",
            depth_hits,
            n,
            max_tree_depth,
            model.name,
        )
    if nonfinite_frac > 0:
        log.warning(
            "numpyro: %.1f%% of constrained draws are non-finite; they are kept, not filtered",
            100 * nonfinite_frac,
        )
    free = free_parameters(model)
    provenance: dict[str, Any] = {
        "method": "nuts",
        "backend": "numpyro",
        "seed": used_seed,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept,
        "max_tree_depth": max_tree_depth,
        "init": init_source,
        "divergences": divergences,
        "accept_prob": accept,
        "max_tree_depth_hits": depth_hits,
        "mean_num_steps": float(np.mean(num_steps)),
        "nonfinite_draw_frac": nonfinite_frac,
        "model_hash": model.content_hash(),
        "model_name": model.name,
        "fixed": _fixed_values(model),
    }
    if init_note is not None:
        provenance["init_note"] = init_note
    return Posterior(
        {name: v.reshape((chains, draws, *v.shape[1:])) for name, v in theta.items()},
        coords=_coords(free),
        provenance=provenance,
    )


class NumpyroBackend:
    """``Backend`` over NumPyro's NUTS. ``optimize`` and ``laplace`` reuse ``infer.laplace``."""

    name: str = "numpyro"

    def __init__(self, *, target_accept: float = 0.9, max_tree_depth: int = 10) -> None:
        _validate_nuts(target_accept, max_tree_depth)
        self.target_accept = target_accept
        self.max_tree_depth = max_tree_depth

    def sample(
        self,
        model: ModelSpec,
        data: Mapping[str, npt.ArrayLike],
        *,
        draws: int,
        tune: int,
        chains: int,
        seed: int | None,
    ) -> Posterior:
        return sample(
            model,
            data,
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed,
            target_accept=self.target_accept,
            max_tree_depth=self.max_tree_depth,
        )

    def optimize(
        self, model: ModelSpec, data: Mapping[str, npt.ArrayLike], *, seed: int | None
    ) -> PointEstimate:
        """``find_mode``; ``Backend.optimize`` has no ``Unverified`` return, so it is raised."""
        est = find_mode(model, data, derivatives="jax", seed=seed)
        if isinstance(est, Unverified):
            raise ValueError(est.reason)
        return est

    def laplace(
        self,
        model: ModelSpec,
        data: Mapping[str, npt.ArrayLike],
        *,
        draws: int,
        seed: int | None,
        allow_unverified: bool = False,
    ) -> Posterior | Unverified:
        from axiom.infer.laplace import laplace as _laplace

        return _laplace(
            model,
            data,
            draws=draws,
            seed=seed,
            derivatives="jax",
            allow_unverified=allow_unverified,
        )
