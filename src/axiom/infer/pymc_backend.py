"""The PyMC backend: NUTS over the PyTensor-compiled log density of a ``ModelSpec``.

No PyMC model is written, in the same sense the NumPyro backend writes no
NumPyro model. The PyMC model here is one ``pm.Flat`` per free parameter in
*unconstrained* coordinates plus a single ``pm.Potential`` holding
``core.interpret.pytensor.compile_log_density`` — so the density PyMC's samplers
see is the same tree ``core.value`` evaluates. Writing a ``pm.Normal`` per
parameter and a ``pm.Normal`` likelihood would have been shorter and would have
created a second definition of the model, free to drift from the first; that is
the failure mode rule 3 exists for.

**The reason to have this backend at all is that PyMC has several samplers.**
One PyTensor graph can be sampled by

* ``"pymc"`` — PyMC's own NUTS, no extra install, C/numba-compiled;
* ``"nutpie"`` — a Rust NUTS, usually the fastest, needs ``nutpie``;
* ``"numpyro"`` — the graph lowered to jax, needs ``numpyro``;
* ``"blackjax"`` — likewise, needs ``blackjax``.

``sample(..., nuts_sampler=...)`` picks one and :func:`samplers` says which are
installed. They are genuinely the same posterior — the graph does not change —
so the choice is about speed and nothing else, and
``tests/contracts/test_backend_equivalence.py`` holds them to that.

Every pymc / pytensor import lives inside a function; ``available()`` says
whether the extra is installed.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from axiom.core.interpret.pytensor import compile_log_density
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

__all__ = [
    "NUTS_SAMPLERS",
    "NutsSampler",
    "PymcBackend",
    "SamplerStatus",
    "available",
    "missing",
    "sample",
    "samplers",
]

log = logging.getLogger(__name__)

NutsSampler = Literal["pymc", "nutpie", "numpyro", "blackjax"]
"""Which NUTS implementation runs over the graph. Same posterior, different speed."""

NUTS_SAMPLERS: tuple[NutsSampler, ...] = ("pymc", "nutpie", "numpyro", "blackjax")
"""Every sampler PyMC can dispatch to, in the order of how little they need installed."""

#: What each sampler needs on top of the ``pymc`` extra itself.
_SAMPLER_REQUIRES: dict[str, tuple[str, ...]] = {
    "pymc": (),
    "nutpie": ("nutpie",),
    "numpyro": ("numpyro", "jax"),
    "blackjax": ("blackjax", "jax"),
}


def missing() -> tuple[str, ...]:
    """Which of ``pymc`` and ``pytensor`` cannot be imported."""
    return tuple(m for m in ("pymc", "pytensor") if importlib.util.find_spec(m) is None)


def available() -> bool:
    """Is the ``pymc`` extra installed? Asking never imports it."""
    return not missing()


@dataclass(frozen=True)
class SamplerStatus:
    """Whether one NUTS sampler can actually be used, and why not when it cannot.

    ``missing`` is what could not be imported. ``error`` is set only by a
    :func:`samplers` call with ``probe=True``, and holds the failure from
    actually running two draws through it — because PyMC's external-sampler
    integrations track fast-moving upstream packages, and *installed* is not the
    same as *works with the PyMC you have*.
    """

    name: str
    missing: tuple[str, ...] = ()
    error: str = ""

    @property
    def usable(self) -> bool:
        return not self.missing and not self.error

    def __str__(self) -> str:
        if self.missing:
            return f"needs {', '.join(self.missing)}"
        return self.error or "usable"


def samplers(*, probe: bool = False) -> dict[str, SamplerStatus]:
    """Every NUTS sampler PyMC can dispatch to, and whether it is usable here.

    Without ``probe`` this is a fast import check and nothing more. With it,
    each importable sampler is asked to draw twice from a one-parameter model,
    which is the only way to find out whether the installed versions actually
    work together — a mismatched blackjax or nutpie imports perfectly and then
    fails inside ``pm.sample``. The probe takes a few seconds and is what
    ``nbs/infer/`` and the backend-equivalence gate use.
    """
    if not available():
        return {
            name: SamplerStatus(name=name, missing=("pymc", *reqs))
            for name, reqs in _SAMPLER_REQUIRES.items()
        }
    out: dict[str, SamplerStatus] = {}
    for name, reqs in _SAMPLER_REQUIRES.items():
        absent = tuple(r for r in reqs if importlib.util.find_spec(r) is None)
        if absent or not probe:
            out[name] = SamplerStatus(name=name, missing=absent)
            continue
        out[name] = SamplerStatus(name=name, error=_probe(name))
    return out


#: What a sampler that is installed but out of step with the installed PyMC fails with.
#: A wrong keyword (``TypeError``), a shape PyMC's integration layer did not expect
#: (``ValueError``), an API that moved (``AttributeError``, ``ImportError``), or PyMC
#: refusing outright (``NotImplementedError``, ``RuntimeError``). Anything *outside* this
#: list is a real bug and propagates out of the probe rather than being reported as a
#: version mismatch -- ``except Exception`` is banned here for exactly that reason
#: (``tests/contracts/test_no_silent_degradation.py``).
_MISMATCH = (
    TypeError,
    ValueError,
    AttributeError,
    ImportError,
    NotImplementedError,
    RuntimeError,
)


def _probe(name: str) -> str:
    """Two draws from ``x ~ N(0, 1)``. Empty string when it works, else the failure.

    The exception is turned into a string *here and only here*: this function's
    whole job is to answer "does this combination work", so a failure is the
    answer rather than an error. Everywhere else a sampler failure propagates.
    """
    pm: Any = importlib.import_module("pymc")
    try:
        with pm.Model():
            pm.Normal("x", 0.0, 1.0)
            with _quiet(False):
                pm.sample(
                    draws=2,
                    tune=2,
                    chains=1,
                    cores=1,
                    nuts_sampler=name,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
    except _MISMATCH as e:
        return f"installed but not usable with pymc {getattr(pm, '__version__', '?')}: {e}"
    return ""


def _check_sampler(name: str) -> NutsSampler:
    if name not in NUTS_SAMPLERS:
        raise ValueError(f"unknown nuts_sampler {name!r}; PyMC dispatches to {list(NUTS_SAMPLERS)}")
    status = samplers()[name]
    if status.missing:
        ready = [s for s, st in samplers().items() if not st.missing]
        raise ImportError(
            f"the {name!r} NUTS sampler needs {list(status.missing)}; "
            f"install them, or use one of {ready}"
        )
    return name


@contextlib.contextmanager
def _quiet(progressbar: bool) -> Iterator[None]:
    """Keep PyMC's chatter out of a library call.

    PyMC narrates to a logger and to stdout. Every other backend here is silent,
    ``diagnose`` is where convergence is reported, and a fit inside a loop should
    not print. ``progressbar=True`` opts back in.
    """
    if progressbar:
        yield
        return
    logger = logging.getLogger("pymc")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            # pm.Flat has no moment, so PyMC warns about an initial point it cannot
            # guess. We always supply one, so the warning is noise.
            warnings.filterwarnings("ignore", message=".*initial.*", category=UserWarning)
            yield
    finally:
        logger.setLevel(previous)


def _init_theta(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    init: Mapping[str, npt.ArrayLike] | None,
    *,
    from_mode: bool,
    seed: int,
) -> tuple[dict[str, npt.NDArray[np.float64]], str, str | None]:
    """Unconstrained start: a user point, a *verified* Laplace mode, or zeros.

    The same rule the NumPyro backend applies, for the same reason: a converged
    point on a flat direction is not a place to start chains, so a mode is used
    only when its Hessian is numerically positive definite. ``note`` records a
    fallback so provenance can carry it.
    """
    layout = flat_layout(model)
    fixed = _fixed_values(model)
    if init is not None:
        z = unconstrain(model, {**fixed, **init})
        return {k: np.asarray(v, dtype=float) for k, v in z.items()}, "init", None
    if from_mode:
        mode = _mode_estimate(model, data, init=None, derivatives="auto", seed=seed)
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
            note = f"mode search did not converge (optimizer {mode.method})"
        else:
            note = (
                f"mode is not verified: Hessian not positive definite "
                f"(min_eigenvalue={mode.min_eigenvalue})"
            )
        log.warning("pymc: %s; initializing chains at zero plus jitter", note)
        return {name: np.zeros(shape) for name, shape in layout}, "zeros", note
    return {name: np.zeros(shape) for name, shape in layout}, "zeros", None


def _sample_stat(idata: Any, *names: str) -> Any | None:
    """The first of ``names`` present in ``sample_stats``.

    The four samplers do not agree on what to call things — PyMC's own emits
    ``diverging`` and ``acceptance_rate``, the jax ones ``diverging`` and
    ``accept_prob``, nutpie its own set — so provenance records what was
    actually reported instead of inventing a value for what was not.
    """
    stats = getattr(idata, "sample_stats", None)
    if stats is None:
        return None
    for name in names:
        if name in stats:
            return np.asarray(stats[name])
    return None


def sample(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int | None,
    target_accept: float = 0.9,
    nuts_sampler: NutsSampler = "pymc",
    init: Mapping[str, npt.ArrayLike] | None = None,
    init_from_mode: bool = True,
    jitter: float = 0.1,
    progressbar: bool = False,
) -> Posterior:
    """Run NUTS through PyMC and return constrained draws with diagnostics in provenance.

    ``nuts_sampler`` chooses which NUTS implementation runs over the graph; the
    graph, and therefore the posterior, is the same for all four.
    """
    if draws < 1 or chains < 1 or tune < 0:
        raise ValueError("need draws >= 1, chains >= 1, tune >= 0")
    if not 0.0 < target_accept < 1.0:
        raise ValueError(f"target_accept must be in (0, 1); got {target_accept}")
    chosen = _check_sampler(nuts_sampler)

    # pymc and pytensor ship no type information; go through importlib so mypy sees Any.
    pm: Any = importlib.import_module("pymc")
    pt: Any = importlib.import_module("pytensor.tensor")

    layout = flat_layout(model)
    if not layout:
        raise ValueError(f"model {model.name!r} has no free parameters")
    used_seed = _draw_seed() if seed is None else int(seed)
    columns = _model_data(model, data)
    logp = compile_log_density(model, columns)

    z0, init_source, init_note = _init_theta(
        model, data, init, from_mode=init_from_mode, seed=used_seed
    )
    rng = np.random.default_rng(used_seed)

    with pm.Model():
        variables = {
            name: pm.Flat(name, shape=shape, initval=np.asarray(z0[name], dtype=float))
            for name, shape in layout
        }
        pm.Potential("axiom_log_density", pt.sum(logp(variables)))
        # Only PyMC's own sampler accepts a start per chain; nutpie refuses outright and
        # the jax samplers take one point. So chains are jittered here where that is
        # possible and left to the sampler's own jitter where it is not -- and which
        # happened goes into provenance rather than being invisible.
        per_chain = chosen == "pymc"
        initvals: Any = (
            [
                {
                    name: z0[name] + jitter * rng.standard_normal(np.shape(z0[name]))
                    for name, _ in layout
                }
                for _ in range(chains)
            ]
            if per_chain
            else {name: z0[name] for name, _ in layout}
        )
        with _quiet(progressbar):
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                # One process. PyMC's default forks, and a fork with jax already
                # loaded is a documented deadlock; sequential chains also make the
                # run reproducible from ``seed`` alone, which is what the NumPyro
                # backend does for the same reason.
                cores=1,
                random_seed=used_seed,
                nuts_sampler=chosen,
                target_accept=target_accept,
                initvals=initvals,
                progressbar=progressbar,
                compute_convergence_checks=False,
            )

    posterior = idata.posterior
    z_flat = {
        name: np.asarray(posterior[name], dtype=float).reshape((chains * draws, *shape))
        for name, shape in layout
    }
    theta = constrain_draws(model, z_flat)
    # The same accounting the NumPyro backend does: a non-finite constrained draw is
    # kept and reported, never filtered out behind the caller's back.
    n_total = chains * draws
    bad = np.zeros(n_total, dtype=bool)
    for v in theta.values():
        bad |= ~np.all(np.isfinite(v.reshape(n_total, -1)), axis=1)
    nonfinite_frac = float(bad.mean())
    free = free_parameters(model)

    diverging = _sample_stat(idata, "diverging")
    accept = _sample_stat(idata, "acceptance_rate", "accept_prob", "mean_tree_accept")
    depth = _sample_stat(idata, "tree_depth", "treedepth")
    steps = _sample_stat(idata, "n_steps", "num_steps")

    provenance: dict[str, Any] = {
        "backend": "pymc",
        "method": "nuts",
        "nuts_sampler": chosen,
        "pymc_version": getattr(pm, "__version__", "unknown"),
        "seed": used_seed,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept,
        "init": init_source,
        "init_jitter": jitter if per_chain else 0.0,
        "init_per_chain": per_chain,
        "nonfinite_draw_frac": nonfinite_frac,
        "model_hash": model.content_hash(),
        "model_name": model.name,
        "fixed": _fixed_values(model),
    }
    if diverging is not None:
        provenance["divergences"] = int(np.sum(diverging))
    if accept is not None:
        provenance["accept_prob"] = float(np.mean(accept))
    if depth is not None:
        provenance["mean_tree_depth"] = float(np.mean(depth))
    if steps is not None:
        provenance["mean_num_steps"] = float(np.mean(steps))
    if init_note is not None:
        provenance["init_note"] = init_note
    return Posterior(
        {name: v.reshape((chains, draws, *v.shape[1:])) for name, v in theta.items()},
        coords=_coords(free),
        provenance=provenance,
    )


class PymcBackend:
    """``Backend`` over PyMC's NUTS. ``optimize`` and ``laplace`` reuse ``infer.laplace``.

    ``nuts_sampler`` is the reason to reach for this backend rather than the
    NumPyro one: the same graph runs under PyMC's own sampler, nutpie, numpyro
    or blackjax, and :func:`samplers` says which are installed.
    """

    name: str = "pymc"

    def __init__(self, *, target_accept: float = 0.9, nuts_sampler: NutsSampler = "pymc") -> None:
        if not 0.0 < target_accept < 1.0:
            raise ValueError(f"target_accept must be in (0, 1); got {target_accept}")
        if nuts_sampler not in NUTS_SAMPLERS:
            raise ValueError(
                f"unknown nuts_sampler {nuts_sampler!r}; PyMC dispatches to "
                f"{list(NUTS_SAMPLERS)}"
            )
        self.target_accept = target_accept
        self.nuts_sampler = nuts_sampler

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
            nuts_sampler=self.nuts_sampler,
        )

    def optimize(
        self, model: ModelSpec, data: Mapping[str, npt.ArrayLike], *, seed: int | None
    ) -> PointEstimate:
        """``find_mode``; ``Backend.optimize`` has no ``Unverified`` return, so it is raised."""
        est = find_mode(model, data, seed=seed)
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

        return _laplace(model, data, draws=draws, seed=seed, allow_unverified=allow_unverified)
