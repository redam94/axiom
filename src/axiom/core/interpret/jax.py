"""The jax interpreter: the same tree, traced for ``jax.grad`` and for the NumPyro backend.

``jax`` is an optional extra and is imported lazily inside the functions, so
importing this module costs nothing and ``import axiom`` never pulls jax
into ``sys.modules`` (gate 1). Call ``require_jax()`` to get a typed
``Unsupported`` instead of an ``ImportError`` when it is absent.

``compile(expr)`` returns ``f(data, params) -> jnp.ndarray`` evaluating the
tree; ``compile_log_density(model)`` returns ``f(data, z) -> scalar`` over
unconstrained parameters matching ``core.model.log_density`` to 1e-10 —
that agreement is a Phase 3 gate and replaces the AST heuristic (review A2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

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
from axiom.core.result import Unsupported

__all__ = [
    "JaxFn",
    "JaxOpaqueRegistry",
    "compile",
    "compile_log_density",
    "jax_available",
    "require_jax",
]

JaxFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
JaxOpaqueRegistry = Mapping[str, Callable[..., Any]]


def jax_available() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def require_jax() -> Unsupported | None:
    if jax_available():
        return None
    return Unsupported(
        reason="jax is not installed; install axiom[numpyro] for the jax interpreter",
        missing=("jax",),
    )


def compile(model: Model, *, opaque: JaxOpaqueRegistry | None = None) -> JaxFn:  # noqa: A001
    """Build a traceable ``f(data, params)`` for ``model``."""
    import jax.numpy as jnp
    from jax.nn import sigmoid, softplus
    from jax.scipy.special import logit

    apply_fns: dict[str, Callable[[Any], Any]] = {
        "exp": jnp.exp,
        "log": jnp.log,
        "log1p": jnp.log1p,
        "expm1": jnp.expm1,
        "tanh": jnp.tanh,
        "sigmoid": sigmoid,
        "logit": logit,
        "softplus": softplus,
        "neg": jnp.negative,
        "relu": lambda x: jnp.maximum(x, 0.0),
        "step": lambda x: jnp.where(x > 0.0, 1.0, 0.0),
    }
    link_fns: dict[str, Callable[[Any], Any]] = {
        "identity": lambda x: x,
        "log": jnp.log,
        "logit": logit,
    }
    reduce_fns: dict[str, Callable[[Any, bool], Any]] = {
        "sum": lambda x, k: jnp.sum(jnp.atleast_1d(x), axis=-1, keepdims=k),
        "mean": lambda x, k: jnp.mean(jnp.atleast_1d(x), axis=-1, keepdims=k),
        "max": lambda x, k: jnp.max(jnp.atleast_1d(x), axis=-1, keepdims=k),
    }
    registry = dict(opaque or {})

    def causal_convolve(x: Any, w: Any) -> Any:
        w = jnp.atleast_1d(w)
        n = x.shape[-1]
        lead = jnp.broadcast_shapes(tuple(w.shape[:-1]), tuple(x.shape[:-1]))
        out = jnp.zeros(tuple(lead) + (n,), dtype=x.dtype)
        for lag in range(min(int(w.shape[-1]), n)):
            if lag == 0:
                shifted = x
            else:
                pad = jnp.zeros(tuple(x.shape[:-1]) + (lag,), dtype=x.dtype)
                shifted = jnp.concatenate([pad, x[..., :-lag]], axis=-1)
            out = out + w[..., lag : lag + 1] * shifted
        return out

    def ev(node: Model, data: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
        match node:
            case Const():
                return jnp.asarray(node.value, dtype=jnp.float64 if _x64() else jnp.float32)
            case Data():
                return jnp.asarray(data[node.name])
            case Param():
                return jnp.asarray(params[node.name])
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
                    return jnp.power(base, float(node.exponent))
                # Guarded power: at base == 0 with a variable exponent the value is 0 for
                # e > 0 but jax's cotangent is inf·0 = nan through any upstream product
                # (a carried dose of exactly zero under s < 1). The double-where keeps
                # the value exact and the gradient finite (zero) at that point.
                exponent = ev(node.exponent, data, params)
                positive = base > 0
                safe = jnp.where(positive, base, 1.0)
                return jnp.where(positive, jnp.power(safe, exponent), 0.0)
            case Apply():
                return apply_fns[node.fn](ev(node.arg, data, params))
            case Link():
                return link_fns[node.fn](ev(node.arg, data, params))
            case Reduce():
                return reduce_fns[node.op](ev(node.arg, data, params), node.keepdims)
            case Gather():
                idx = jnp.asarray(data[node.index.name], dtype=jnp.int32)
                return jnp.take(ev(node.source, data, params), idx, axis=-1)
            case Convolve():
                return causal_convolve(ev(node.signal, data, params), ev(node.kernel, data, params))
            case Opaque():
                if node.name not in registry:
                    raise KeyError(
                        f"opaque function {node.name!r} has no jax implementation registered"
                    )
                return registry[node.name](*(ev(i, data, params) for i in node.inputs))
            case Equation():
                return ev(node.rhs, data, params)
            case System():
                return jnp.stack(
                    jnp.broadcast_arrays(*(ev(eq, data, params) for eq in node.equations))
                )
            case ODESystem():
                raise NotImplementedError("ODESystem is dimension-check-only in 1.0")
        raise TypeError(f"not an expression node: {type(node).__name__}")

    return lambda data, params: ev(model, data, params)


def _x64() -> bool:
    import jax

    return bool(getattr(jax.config, "jax_enable_x64", False))


def _log_prior_jax(prior: Prior, x: Any, theta: Mapping[str, Any]) -> Any:
    import jax.numpy as jnp
    from jax.scipy import stats as jst

    def h(k: str) -> Any:
        v = prior.hyper[k]
        return theta[v] if isinstance(v, str) else jnp.asarray(float(v))

    match prior.family:
        case "normal":
            return jst.norm.logpdf(x, loc=h("mu"), scale=h("sigma"))
        case "halfnormal":
            return jst.norm.logpdf(x, loc=0.0, scale=h("sigma")) + jnp.log(2.0)
        case "lognormal":
            s = h("sigma")
            lx = jnp.log(x)
            return -lx - jnp.log(s) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((lx - h("mu")) / s) ** 2
        case "gamma":
            return jst.gamma.logpdf(x, a=h("alpha"), scale=1.0 / h("beta"))
        case "beta":
            return jst.beta.logpdf(x, a=h("alpha"), b=h("beta"))
        case "uniform":
            lo, hi = h("low"), h("high")
            return jnp.where((x >= lo) & (x <= hi), -jnp.log(hi - lo), -jnp.inf)
    return jnp.zeros_like(x)  # fixed


def _log_constraint_jax(constraint: Constraint, x: Any) -> Any:
    """The same term as ``core.model.log_constraint``: ``-inf`` for a lognormal at ``x <= 0``."""
    import jax.numpy as jnp
    from jax.scipy import stats as jst

    x = jnp.reshape(x, ())
    obs, s = constraint.observed, constraint.scale
    if constraint.family == "normal":
        return jst.norm.logpdf(obs, loc=x, scale=s)
    # Guarded log: where x <= 0 the density is zero; the where keeps both the value and the
    # gradient finite on the positive branch (log of a clamped 1.0 on the dead branch).
    positive = x > 0
    safe = jnp.where(positive, x, 1.0)
    lp = (
        -jnp.log(obs)
        - jnp.log(s)
        - 0.5 * jnp.log(2 * jnp.pi)
        - 0.5 * ((jnp.log(obs) - jnp.log(safe)) / s) ** 2
    )
    return jnp.where(positive, lp, -jnp.inf)


def compile_log_density(
    model: ModelSpec, *, opaque: JaxOpaqueRegistry | None = None
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Any]:
    """``f(data, z) -> log posterior`` in unconstrained coordinates, traceable by jax.

    Matches ``core.model.log_density`` term for term: priors, the Jacobian,
    the outcome likelihood, and every soft ``Constraint``.
    """
    import jax.numpy as jnp
    from jax.scipy import stats as jst

    mean = compile(model.mean, opaque=opaque)
    constraints = tuple((c, compile(c.expr, opaque=opaque)) for c in model.constraints)
    free = free_parameters(model)
    lik = model.likelihood
    scale_expr = None if lik.scale_expr is None else compile(lik.scale_expr, opaque=opaque)

    def scale(data: Mapping[str, Any], theta: Mapping[str, Any]) -> Any:
        # The same resolution as ``core.model.likelihood_scale``: the named parameter, or
        # ``scale_expr`` evaluated on the data and the constrained parameters.
        if scale_expr is not None:
            return scale_expr(data, theta)
        return theta[lik.scale or ""]

    def constrain(z: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        theta: dict[str, Any] = {}
        log_jac = jnp.asarray(0.0)
        for p in model.parameters:
            assert p.prior is not None
            if p.prior.family == "fixed":
                theta[p.name] = jnp.asarray(float(p.prior.hyper["value"]))
                continue
            v = jnp.asarray(z[p.name])
            s = support_of(p.prior)
            if s == "real":
                theta[p.name] = v
            elif s == "positive":
                theta[p.name] = jnp.exp(v)
                log_jac = log_jac + jnp.sum(v)
            elif s == "unit":
                u = 1.0 / (1.0 + jnp.exp(-v))
                theta[p.name] = u
                log_jac = log_jac + jnp.sum(jnp.log(u) + jnp.log1p(-u))
            else:
                lo_h, hi_h = p.prior.hyper["low"], p.prior.hyper["high"]
                lo = theta[lo_h] if isinstance(lo_h, str) else jnp.asarray(float(lo_h))
                hi = theta[hi_h] if isinstance(hi_h, str) else jnp.asarray(float(hi_h))
                u = 1.0 / (1.0 + jnp.exp(-v))
                theta[p.name] = lo + (hi - lo) * u
                log_jac = log_jac + jnp.sum(jnp.log(u) + jnp.log1p(-u) + jnp.log(hi - lo))
        return theta, log_jac

    def f(data: Mapping[str, Any], z: Mapping[str, Any]) -> Any:
        theta, log_jac = constrain(z)
        lp = log_jac
        for p in free:
            assert p.prior is not None
            lp = lp + jnp.sum(_log_prior_jax(p.prior, theta[p.name], theta))
        y = jnp.asarray(data[model.outcome.name])
        mu = mean(data, theta)
        match lik.family:
            case "normal":
                ll = jst.norm.logpdf(y, loc=mu, scale=scale(data, theta))
            case "lognormal":
                s = scale(data, theta)
                ly = jnp.log(y)
                ll = (
                    -ly
                    - jnp.log(s)
                    - 0.5 * jnp.log(2 * jnp.pi)
                    - 0.5 * ((ly - jnp.log(mu)) / s) ** 2
                )
            case "student_t":
                ll = jst.t.logpdf(y, df=float(lik.df or 0.0), loc=mu, scale=scale(data, theta))
            case "poisson":
                ll = jst.poisson.logpmf(jnp.round(y), mu)
        out = lp + jnp.sum(ll)
        for c, fn in constraints:
            out = out + _log_constraint_jax(c, fn(data, theta))
        return out

    return f
