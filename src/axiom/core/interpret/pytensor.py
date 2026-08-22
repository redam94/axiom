"""The PyTensor interpreter: the same tree, as a symbolic graph PyMC can sample.

The third interpreter of ``axiom.core.expr``, and it exists for one reason: it
is what lets PyMC's samplers see the *same* log density ``core.value`` and
``core.interpret.jax`` evaluate. No PyMC model is written anywhere. There is no
second expression of the priors, the transforms, or the likelihood — which is
the drift rule 3 is about, and which writing a ``pm.Normal`` per parameter
would have reintroduced immediately.

**Data is concrete, parameters are symbolic.** Unlike the jax interpreter,
which traces both, this one folds the data in as constants and leaves only the
unconstrained parameter vector symbolic. That is what a sampler needs, and it
makes every array shape static at graph-build time — which the causal
convolution depends on.

**Why this buys several samplers.** A PyMC model built from a ``pm.Flat`` per
free parameter plus one ``pm.Potential`` can be sampled by any of PyMC's NUTS
backends — its own C/numba sampler, ``nutpie``, ``numpyro`` or ``blackjax`` —
because they all consume the same PyTensor graph. Every op used here has a
numba and a jax lowering, so none of those four is closed off.

Nothing at module scope imports pytensor; gate 1 requires ``import axiom`` to
leave it out of ``sys.modules``.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from axiom.core.expr import (
    Add,
    Apply,
    Const,
    Convolve,
    Data,
    Div,
    Equation,
    Gather,
    Link,
    Model,
    Mul,
    ODESystem,
    Opaque,
    Param,
    Pow,
    Prior,
    Reduce,
    System,
)
from axiom.core.model import Constraint, ModelSpec, free_parameters, support_of

__all__ = [
    "PytensorOpaqueRegistry",
    "compile",
    "compile_log_density",
    "missing",
    "pytensor_available",
    "require_pytensor",
]

PytensorOpaqueRegistry = Mapping[str, Callable[..., Any]]
"""Opaque-function name to a pytensor implementation, for models that use ``Opaque``."""


def missing() -> tuple[str, ...]:
    """Which of ``pytensor`` and ``pymc`` cannot be imported."""
    out: list[str] = []
    for module in ("pytensor", "pymc"):
        try:
            importlib.import_module(module)
        except ImportError:
            out.append(module)
    return tuple(out)


def pytensor_available() -> bool:
    """Is the ``pymc`` extra installed? Asking never imports it."""
    import importlib.util

    return importlib.util.find_spec("pytensor") is not None


def require_pytensor() -> None:
    """Raise a message naming the extra, rather than an opaque ``ImportError``."""
    if not pytensor_available():
        raise ImportError("this needs pytensor; install the pymc extra (pip install 'axiom[pymc]')")


def _f64(value: float) -> Any:
    """A float64 scalar constant.

    ``pt.as_tensor_variable(0.0)`` on a *Python* float gives **float32** even when
    ``pytensor.config.floatX`` is float64, and a float32 hyperparameter quietly
    costs the log density about eight digits. Every scalar constant in this module
    goes through here.
    """
    pt: Any = importlib.import_module("pytensor.tensor")

    return pt.as_tensor_variable(np.asarray(value, dtype=np.float64))


def _static_length(x: Any, what: str) -> int:
    """The last axis of ``x`` as a Python int, or a message saying why it is not known.

    The convolution needs it. Every array that reaches one comes either from the
    data (a constant, so its shape is known) or from a carryover's weight vector
    (built on a ``Const`` lag index, likewise known), so an unknown length here
    means a model shape this interpreter has not seen rather than a limitation
    to work around silently.
    """
    shape = getattr(getattr(x, "type", None), "shape", ())
    if shape and shape[-1] is not None:
        return int(shape[-1])
    raise ValueError(
        f"the pytensor interpreter needs a statically known length for {what}; "
        f"got shape {shape!r}. Data arrays are folded in as constants, so this means "
        "a parameter-shaped array reached a convolution."
    )


def compile(  # noqa: A001 - matches the jax interpreter's name
    model: Model, *, opaque: PytensorOpaqueRegistry | None = None
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Any]:
    """Build ``f(data, params) -> pytensor variable`` for one expression tree.

    ``data`` values are concrete arrays folded in as constants; ``params``
    values are pytensor variables. Term for term the same as
    ``core.interpret.value`` and ``core.interpret.jax``.
    """
    require_pytensor()
    # pytensor ships no usable type information; go through importlib so mypy sees Any.
    pt: Any = importlib.import_module("pytensor.tensor")

    apply_fns: dict[str, Callable[[Any], Any]] = {
        "exp": pt.exp,
        "log": pt.log,
        "log1p": pt.log1p,
        "expm1": pt.expm1,
        "tanh": pt.tanh,
        "sigmoid": pt.sigmoid,
        "logit": lambda x: pt.log(x) - pt.log1p(-x),
        "softplus": lambda x: pt.softplus(x),
        "neg": lambda x: -x,
        "relu": lambda x: pt.maximum(x, _f64(0.0)),
        "step": lambda x: pt.switch(pt.gt(x, _f64(0.0)), _f64(1.0), _f64(0.0)),
        "sin": pt.sin,
        "cos": pt.cos,
    }
    link_fns: dict[str, Callable[[Any], Any]] = {
        "identity": lambda x: x,
        "log": pt.log,
        "logit": lambda x: pt.log(x) - pt.log1p(-x),
    }
    reduce_fns: dict[str, Callable[[Any, bool], Any]] = {
        "sum": lambda x, k: pt.sum(pt.atleast_1d(x), axis=-1, keepdims=k),
        "mean": lambda x, k: pt.mean(pt.atleast_1d(x), axis=-1, keepdims=k),
        "max": lambda x, k: pt.max(pt.atleast_1d(x), axis=-1, keepdims=k),
    }
    registry = dict(opaque or {})

    def causal_convolve(x: Any, w: Any) -> Any:
        """``y[..., t] = sum_l w[..., l] x[..., t-l]`` — the same sum as the other two."""
        w = pt.atleast_1d(w)
        n = _static_length(x, "the signal of a Convolve")
        lags = _static_length(w, "the kernel of a Convolve")
        out = None
        for lag in range(min(lags, n)):
            shifted = (
                x
                if lag == 0
                else pt.concatenate([pt.zeros_like(x[..., :lag]), x[..., :-lag]], axis=-1)
            )
            term = w[..., lag : lag + 1] * shifted
            out = term if out is None else out + term
        return pt.zeros_like(x) if out is None else out

    def ev(node: Model, data: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
        match node:
            case Const():
                return pt.as_tensor_variable(np.asarray(node.value, dtype=np.float64))
            case Data():
                return pt.as_tensor_variable(np.asarray(data[node.name], dtype=np.float64))
            case Param():
                return params[node.name]
            case Add():
                out = ev(node.terms[0], data, params)
                for t in node.terms[1:]:
                    out = out + ev(t, data, params)
                return out
            case Mul():
                out = ev(node.factors[0], data, params)
                for f in node.factors[1:]:
                    out = out * ev(f, data, params)
                return out
            case Div():
                return ev(node.numerator, data, params) / ev(node.denominator, data, params)
            case Pow():
                base = ev(node.base, data, params)
                if isinstance(node.exponent, Fraction):
                    return pt.power(base, float(node.exponent))
                # The same double-switch guard the jax interpreter uses: at base == 0 with
                # a variable exponent the value is 0 for e > 0, but the gradient is inf*0
                # through any upstream product. Clamping the dead branch keeps both finite.
                exponent = ev(node.exponent, data, params)
                positive = pt.gt(base, _f64(0.0))
                safe = pt.switch(positive, base, _f64(1.0))
                return pt.switch(positive, pt.power(safe, exponent), _f64(0.0))
            case Apply():
                return apply_fns[node.fn](ev(node.arg, data, params))
            case Link():
                return link_fns[node.fn](ev(node.arg, data, params))
            case Reduce():
                return reduce_fns[node.op](ev(node.arg, data, params), node.keepdims)
            case Gather():
                # The index keeps its shape: a hierarchical intercept's index is
                # ``(n_units, 1)`` so the gathered vector broadcasts down the time axis
                # (see the layout contract in ``surface.model``). Flattening it here
                # silently transposes the result.
                index = np.asarray(data[node.index.name]).astype(np.int64)
                return pt.take(ev(node.source, data, params), index, axis=-1)
            case Convolve():
                return causal_convolve(ev(node.signal, data, params), ev(node.kernel, data, params))
            case Opaque():
                if node.name not in registry:
                    raise KeyError(
                        f"opaque function {node.name!r} has no pytensor implementation registered"
                    )
                return registry[node.name](*(ev(i, data, params) for i in node.inputs))
            case Equation():
                return ev(node.rhs, data, params)
            case System():
                return pt.stack([ev(eq, data, params) for eq in node.equations])
            case ODESystem():
                raise NotImplementedError("ODESystem is dimension-check-only in 1.0")
        raise TypeError(f"not an expression node: {type(node).__name__}")

    return lambda data, params: ev(model, data, params)


def _log_prior_pt(prior: Prior, x: Any, theta: Mapping[str, Any]) -> Any:
    """One prior's log density. Written out rather than taken from ``pymc.logp``.

    Using PyMC's distributions here would make the prior PyMC sees a *second*
    definition of the prior ``core.model.log_density`` uses, and the two would
    be free to drift. These are the same closed forms as the jax interpreter.
    """
    pt: Any = importlib.import_module("pytensor.tensor")

    def h(k: str) -> Any:
        v = prior.hyper[k]
        return theta[v] if isinstance(v, str) else _f64(float(v))

    half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    match prior.family:
        case "normal":
            s = h("sigma")
            return -pt.log(s) - half_log_2pi - 0.5 * ((x - h("mu")) / s) ** 2
        case "halfnormal":
            s = h("sigma")
            return -pt.log(s) - half_log_2pi - 0.5 * (x / s) ** 2 + math.log(2.0)
        case "lognormal":
            s = h("sigma")
            lx = pt.log(x)
            return -lx - pt.log(s) - half_log_2pi - 0.5 * ((lx - h("mu")) / s) ** 2
        case "gamma":
            a, b = h("alpha"), h("beta")
            return a * pt.log(b) - pt.gammaln(a) + (a - 1.0) * pt.log(x) - b * x
        case "beta":
            a, b = h("alpha"), h("beta")
            log_beta = pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)
            return (a - 1.0) * pt.log(x) + (b - 1.0) * pt.log1p(-x) - log_beta
        case "uniform":
            lo, hi = h("low"), h("high")
            inside = pt.and_(pt.ge(x, lo), pt.le(x, hi))
            return pt.switch(inside, -pt.log(hi - lo), _f64(-np.inf))
    return pt.zeros_like(x)  # fixed


def _log_constraint_pt(constraint: Constraint, x: Any) -> Any:
    """The same term as ``core.model.log_constraint``: ``-inf`` for a lognormal at ``x <= 0``."""
    pt: Any = importlib.import_module("pytensor.tensor")

    x = pt.reshape(x, ())
    obs, s = constraint.observed, constraint.scale
    half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    if constraint.family == "normal":
        return -math.log(s) - half_log_2pi - 0.5 * ((obs - x) / s) ** 2
    positive = pt.gt(x, _f64(0.0))
    safe = pt.switch(positive, x, _f64(1.0))
    lp = (
        -math.log(obs)
        - math.log(s)
        - half_log_2pi
        - 0.5 * ((math.log(obs) - pt.log(safe)) / s) ** 2
    )
    return pt.switch(positive, lp, _f64(-np.inf))


def _likelihood_pt(family: str, y: Any, mu: Any, scale: Any, df: float) -> Any:
    pt: Any = importlib.import_module("pytensor.tensor")

    half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    match family:
        case "normal":
            return -pt.log(scale) - half_log_2pi - 0.5 * ((y - mu) / scale) ** 2
        case "lognormal":
            ly = pt.log(y)
            return -ly - pt.log(scale) - half_log_2pi - 0.5 * ((ly - pt.log(mu)) / scale) ** 2
        case "student_t":
            z = (y - mu) / scale
            return (
                pt.gammaln(0.5 * (df + 1.0))
                - pt.gammaln(0.5 * df)
                - 0.5 * math.log(math.pi)
                - 0.5 * pt.log(df)
                - pt.log(scale)
                - 0.5 * (df + 1.0) * pt.log1p(z**2 / df)
            )
        case "poisson":
            k = pt.round(y)
            return k * pt.log(mu) - mu - pt.gammaln(k + 1.0)
    raise ValueError(f"unknown likelihood family {family!r}")


def compile_log_density(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    *,
    opaque: PytensorOpaqueRegistry | None = None,
) -> Callable[[Mapping[str, Any]], Any]:
    """``f(z) -> log posterior`` as a pytensor graph, in unconstrained coordinates.

    Matches ``core.model.log_density`` term for term — priors, the Jacobian, the
    outcome likelihood, every soft ``Constraint`` — which is what
    ``tests/contracts/test_backend_equivalence.py`` checks numerically against
    the numpy and jax paths.
    """
    require_pytensor()
    # pytensor ships no usable type information; go through importlib so mypy sees Any.
    pt: Any = importlib.import_module("pytensor.tensor")

    mean = compile(model.mean, opaque=opaque)
    constraints = tuple((c, compile(c.expr, opaque=opaque)) for c in model.constraints)
    free = free_parameters(model)
    lik = model.likelihood
    scale_expr = None if lik.scale_expr is None else compile(lik.scale_expr, opaque=opaque)
    columns = {str(k): np.asarray(v, dtype=np.float64) for k, v in data.items()}

    def scale(theta: Mapping[str, Any]) -> Any:
        if scale_expr is not None:
            return scale_expr(columns, theta)
        return theta[lik.scale or ""]

    def constrain(z: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        theta: dict[str, Any] = {}
        log_jac: Any = _f64(0.0)
        for p in model.parameters:
            assert p.prior is not None
            if p.prior.family == "fixed":
                theta[p.name] = _f64(float(p.prior.hyper["value"]))
                continue
            v = z[p.name]
            match support_of(p.prior):
                case "real":
                    theta[p.name] = v
                case "positive":
                    theta[p.name] = pt.exp(v)
                    log_jac = log_jac + pt.sum(v)
                case "unit":
                    u = pt.sigmoid(v)
                    theta[p.name] = u
                    log_jac = log_jac + pt.sum(pt.log(u) + pt.log1p(-u))
                case _:
                    lo_h, hi_h = p.prior.hyper["low"], p.prior.hyper["high"]
                    lo = theta[lo_h] if isinstance(lo_h, str) else _f64(float(lo_h))
                    hi = theta[hi_h] if isinstance(hi_h, str) else _f64(float(hi_h))
                    u = pt.sigmoid(v)
                    theta[p.name] = lo + (hi - lo) * u
                    log_jac = log_jac + pt.sum(pt.log(u) + pt.log1p(-u) + pt.log(hi - lo))
        return theta, log_jac

    def f(z: Mapping[str, Any]) -> Any:
        theta, log_jac = constrain(z)
        lp = log_jac
        for p in free:
            assert p.prior is not None
            lp = lp + pt.sum(_log_prior_pt(p.prior, theta[p.name], theta))
        y = pt.as_tensor_variable(columns[model.outcome.name])
        mu = mean(columns, theta)
        ll = _likelihood_pt(lik.family, y, mu, scale(theta), float(lik.df or 0.0))
        out = lp + pt.sum(ll)
        for c, fn in constraints:
            out = out + _log_constraint_pt(c, fn(columns, theta))
        return out

    return f
