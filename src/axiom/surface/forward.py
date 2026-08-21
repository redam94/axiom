"""The one ``forward()``, the posterior predictive of the mean, and closed-form marginals.

``forward`` is ``axiom.core.value`` over the surface's expression tree —
nothing else in axiom evaluates a transform chain (rule 3). ``predict``
pushes every posterior draw through it (optionally adding likelihood noise,
``Capability.PREDICTIVE``). ``marginal`` and ``marginal_total`` use the
kernels' closed-form derivatives (review C2) *through the chain*:

* the kernel sees the carried dose ``c_t = Σ_l w_l · x_{t−l}`` and its
  derivative ``g_t = ∂mu_t / ∂c_t`` is a tree (``marginal_expr``), built
  from ``kernel.derivative`` plus, for each interaction the treatment is
  in, ``gamma · ∂f_i/∂c · f_j``;
* carryover is linear, so ``∂c_{t+l} / ∂x_t = w_l``. The effect of one
  extra unit of dose in period ``t`` on the **period-``t``** outcome is
  ``w_0 · g_t`` (``marginal``); its **total** effect over the horizon is
  ``Σ_l w_l · g_{t+l}`` (``marginal_total``), which equals ``g_t`` at a
  steady state because the weights sum to one, and is truncated at the end
  of the observed horizon otherwise.

Both take a *point* ``theta`` (scalars per parameter); loop over draws for
a posterior of marginals.

Every entry point that evaluates a ``Surface`` with carryover applies the
layout contract from ``axiom.surface.model``: dose arrays are ``(n_units,
n_periods)`` with time last, and a 1-D grid is refused with ``ValueError``
(use ``Surface.steady_state()`` for independent rows).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from axiom.core import (
    Add,
    Const,
    Data,
    Div,
    Expr,
    Intervention,
    ModelSpec,
    Mul,
    PredictiveDraws,
    SupportsForward,
    SupportsPosterior,
    causal_convolve,
    dimension,
    dimension_of,
    dimensionless,
    value,
)
from axiom.surface.kernels import LinearKernel, ResponseKernel
from axiom.surface.model import Surface, _require_panel_layout, interaction_name

__all__ = ["GRID_VERSION", "forward", "marginal", "marginal_expr", "marginal_total", "predict"]

Array = npt.NDArray[np.float64]

GRID_VERSION = "grid"
"""``PredictiveDraws.intervention.version`` when the dose rows differ: ``doses`` then holds the
first cell (C order) of each treatment's array and the exact per-row doses are in ``coords``."""


def forward(
    surface: SupportsForward,
    dose: Mapping[str, npt.ArrayLike],
    theta: Mapping[str, npt.ArrayLike],
) -> Array:
    """``value(surface.expr, data=dose, params=theta)`` — the one forward()."""
    if isinstance(surface, Surface):
        _require_panel_layout(surface.spec, dose, "forward")
    return value(surface.expr, data=dose, params=theta)


# -- posterior predictive ----------------------------------------------------------------


def _fixed_values(model: ModelSpec) -> dict[str, float]:
    return {
        p.name: float(p.prior.hyper["value"])
        for p in model.parameters
        if p.prior is not None and p.prior.family == "fixed"
    }


def _require_positive_mean(family: str, mu: Array, draw: int | None) -> None:
    """A lognormal or Poisson mean must be strictly positive; name the offending cells."""
    bad = ~(mu > 0.0)
    if not np.any(bad):
        return
    cells = [tuple(int(i) for i in idx) for idx in np.argwhere(bad)[:8]]
    where = f" in draw {draw}" if draw is not None else ""
    raise ValueError(
        f"{family} likelihood needs a strictly positive mean{where}; {int(bad.sum())} of "
        f"{mu.size} cells are non-positive or non-finite, first at {cells}"
        f"{'...' if bad.sum() > len(cells) else ''}. Constrain the mean (intercept, "
        "amplitudes) or use a normal likelihood."
    )


def _noise(
    model: ModelSpec,
    mu: Array,
    theta: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    draw: int | None = None,
) -> Array:
    lik = model.likelihood
    match lik.family:
        case "normal":
            sigma = np.asarray(theta[lik.scale or ""], dtype=float)
            return np.asarray(mu + rng.normal(0.0, 1.0, size=mu.shape) * sigma, dtype=np.float64)
        case "student_t":
            sigma = np.asarray(theta[lik.scale or ""], dtype=float)
            t = rng.standard_t(float(lik.df or 0.0), size=mu.shape)
            return np.asarray(mu + t * sigma, dtype=np.float64)
        case "lognormal":
            _require_positive_mean("lognormal", mu, draw)
            sigma = np.asarray(theta[lik.scale or ""], dtype=float)
            return np.asarray(rng.lognormal(np.log(mu), sigma, size=mu.shape), dtype=np.float64)
        case "poisson":
            _require_positive_mean("poisson", mu, draw)
            return np.asarray(rng.poisson(mu, size=mu.shape), dtype=np.float64)
    raise ValueError(f"unknown likelihood family {lik.family!r}")  # pragma: no cover


def _intervention(surface: Surface, dose: Mapping[str, npt.ArrayLike]) -> Intervention:
    """A truthful ``Intervention`` for a dose grid (see ``GRID_VERSION``)."""
    grid = {
        t: np.asarray(dose[t], dtype=float).reshape(-1)
        for t in surface.spec.treatment_names
        if t in dose
    }
    if not grid:
        raise ValueError("dose must include at least one treatment column")
    empty = sorted(t for t, v in grid.items() if v.size == 0)
    if empty:
        raise ValueError(f"dose arrays are empty for treatments {empty}")
    shared = all(np.all(v == v[0]) for v in grid.values())
    first = {t: float(v[0]) for t, v in grid.items()}
    if shared:
        return Intervention(doses=first, mode="set")
    return Intervention(doses=first, mode="set", version=GRID_VERSION)


def predict(
    surface: Surface,
    posterior: SupportsPosterior,
    dose: Mapping[str, npt.ArrayLike],
    *,
    seed: int | None = None,
    noise: bool = False,
) -> PredictiveDraws:
    """Every draw pushed through ``forward``: ``values`` is ``(chain, draw, *forward shape)``.

    ``noise=True`` adds likelihood noise per draw (the outcome-scale
    posterior predictive); ``seed`` is recorded on the result. Under a
    lognormal or Poisson likelihood a draw whose mean is not strictly
    positive somewhere is a ``ValueError`` naming the cells, never a
    ``nan``. Draws are evaluated one at a time so vector parameters and
    carryover weights never mix across draws.

    The ``Intervention`` on the result is honest about the grid: when every
    row of every treatment shares one dose it is ``mode="set"`` at that
    level; otherwise ``version == GRID_VERSION`` (``"grid"``), ``doses``
    holds the first cell of each treatment's array (C order) and the exact
    per-row doses are in ``coords`` under each treatment's name.
    """
    model = surface.model
    _require_panel_layout(surface.spec, dose, "predict")
    names = sorted(posterior.names())
    if not names:
        raise ValueError("posterior has no variables")
    first = np.asarray(posterior.draws(names[0]), dtype=float)
    chains, per_chain = int(first.shape[0]), int(first.shape[1])
    n = chains * per_chain
    flat = {
        name: np.asarray(posterior.draws(name), dtype=float).reshape(
            n, *np.asarray(posterior.draws(name)).shape[2:]
        )
        for name in names
    }
    fixed = _fixed_values(model)
    rng = np.random.default_rng(seed)
    out: list[Array] = []
    for d in range(n):
        theta: dict[str, Any] = {**fixed, **{name: flat[name][d] for name in names}}
        mu = np.asarray(value(model.mean, data=dose, params=theta), dtype=np.float64)
        out.append(_noise(model, mu, theta, rng, draw=d) if noise else mu)
    stacked = np.stack(np.broadcast_arrays(*out))
    values = stacked.reshape(chains, per_chain, *stacked.shape[1:])
    intervention = _intervention(surface, dose)
    coords: dict[str, list[Any]] = {
        t: [float(x) for x in np.asarray(dose[t], dtype=float).reshape(-1)]
        for t in surface.spec.treatment_names
        if t in dose
    }
    return PredictiveDraws(
        values=np.asarray(values, dtype=np.float64),
        intervention=intervention,
        coords=coords,
        seed=seed,
    )


# -- marginals ---------------------------------------------------------------------------


def _saturation_derivative(
    surface: Surface, kernel: ResponseKernel, carried: Expr, treatment: str
) -> Expr:
    """``∂f/∂c`` for the dimensionless saturation ``f``: ``derivative / amplitude``.

    Every shipped kernel's response is ``amplitude · f`` with one amplitude
    parameter, so its saturation derivative is the response derivative with
    the amplitude divided out. Both are built in the spec's outcome
    dimension and the divisor is the amplitude ``Param`` the model
    declares, so the amplitude appears with one dimension throughout the
    marginal tree. ``LinearKernel`` is the exception — its ``saturation``
    is ``dose / reference_dose`` — and is handled directly.
    """
    dose_dim = dimension(carried)
    if isinstance(kernel, LinearKernel):
        return Div(
            numerator=Const(value=1.0, dimension=dimensionless()),
            denominator=Const(value=kernel.reference_dose, dimension=dose_dim),
        )
    out_dim = surface.spec.outcome_dimension
    amplitudes = [
        p
        for p in kernel.parameters(treatment, dose_dim, out_dim)
        if kernel.roles[p.name.removesuffix(f"_{treatment}")] == "amplitude"
    ]
    if len(amplitudes) != 1:  # pragma: no cover - kernel contract
        raise ValueError(f"kernel {kernel.name!r} must declare exactly one amplitude")
    return Div(
        numerator=kernel.derivative(carried, treatment, out_dim),
        denominator=surface.model.parameter(amplitudes[0].name),
    )


def _carried(surface: Surface) -> dict[str, Expr]:
    spec = surface.spec
    return {
        t.name: spec.carryover_of(t.name).apply(
            Data(name=t.name, dimension=dimension_of(t)), t.name
        )
        for t in spec.treatments
    }


def marginal_expr(surface: Surface, treatment: str) -> Expr:
    """``∂mu / ∂c`` with respect to a treatment's *carried* dose, as a tree.

    ``kernel.derivative`` of the main effect plus ``gamma · ∂f_i/∂c · f_j``
    for every interaction the treatment takes part in. Dimension
    ``outcome / dose``; every parameter appears with the dimension the
    model declares for it.
    """
    spec = surface.spec
    spec.treatment(treatment)
    carried = _carried(surface)
    out_dim = spec.outcome_dimension
    terms: list[Expr] = [
        spec.kernel_of(treatment).derivative(carried[treatment], treatment, out_dim)
    ]
    for first, second in spec.interactions:
        if treatment not in (first, second):
            continue
        other = second if first == treatment else first
        terms.append(
            Mul(
                factors=(
                    surface.model.parameter(interaction_name(first, second)),
                    _saturation_derivative(
                        surface, spec.kernel_of(treatment), carried[treatment], treatment
                    ),
                    spec.kernel_of(other).saturation(carried[other], other),
                )
            )
        )
    return terms[0] if len(terms) == 1 else Add(terms=tuple(terms))


def _weights(surface: Surface, theta: Mapping[str, npt.ArrayLike], treatment: str) -> Array:
    w = value(surface.spec.carryover_of(treatment).weights(treatment), params=theta)
    return np.atleast_1d(np.asarray(w, dtype=np.float64))


def marginal(
    surface: Surface,
    theta: Mapping[str, npt.ArrayLike],
    dose: Mapping[str, npt.ArrayLike],
    treatment: str,
) -> Array:
    """``w_0 · ∂mu_t/∂c_t``: the effect of one more unit of dose in period ``t`` on outcome ``t``.

    Same shape as ``forward``. Without carryover ``w_0 = 1`` and this is the
    kernel's derivative at the dose.
    """
    _require_panel_layout(surface.spec, dose, "marginal")
    g = np.asarray(value(marginal_expr(surface, treatment), data=dose, params=theta), dtype=float)
    w = _weights(surface, theta, treatment)
    return np.asarray(w[..., 0] * g, dtype=np.float64)


def marginal_total(
    surface: Surface,
    theta: Mapping[str, npt.ArrayLike],
    dose: Mapping[str, npt.ArrayLike],
    treatment: str,
) -> Array:
    """``Σ_l w_l · ∂mu_{t+l}/∂c_{t+l}``: the total effect of one more unit in period ``t``.

    Summed over the lags that fall inside the observed horizon (time is the
    last axis); lags beyond the horizon — all of them past ``n_periods``
    when ``max_lag`` exceeds it — are truncated, so the total then
    under-counts relative to an infinite horizon. Equals the same-period
    derivative ``∂mu_t/∂c_t`` when the carried dose is constant over the
    next ``max_lag`` periods, since the weights sum to one; with
    ``NoCarryover`` it equals ``marginal``.
    """
    _require_panel_layout(surface.spec, dose, "marginal_total")
    g = np.asarray(value(marginal_expr(surface, treatment), data=dose, params=theta), dtype=float)
    w = _weights(surface, theta, treatment)
    if w.shape[-1] == 1:
        return np.asarray(w[..., 0] * g, dtype=np.float64)
    # Σ_l w_l g_{t+l} is the causal convolution run backwards in time.
    reversed_total = causal_convolve(np.flip(g, axis=-1), w)
    return np.asarray(np.flip(reversed_total, axis=-1), dtype=np.float64)
