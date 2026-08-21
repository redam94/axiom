"""``ModelSpec``: an expression for the mean, a likelihood, and priors — the thing a backend fits.

``Backend.sample`` takes a model *spec*, not a Python callable (review A2):
the backend compiles the tree. That is also what lets a fitted model be
saved without pickle — the spec replays.

This module also provides the sampler-free numpy log density over
*unconstrained* parameters (``log_density``), with the change-of-variables
Jacobian, used by ``infer.laplace`` and by the ``value == jax`` gate.

A ``ModelSpec`` may also carry **soft constraints** (``Constraint``, decision
D6.3): a scalar expression over the model's parameters and data — typically
an estimand realized at an experiment's doses and reduced over its units and
periods — with an observed value and a scale. Each adds ``log p(observed |
expr(theta), scale)`` to the likelihood, which is how ``calibrate`` folds a
randomized measurement into the graph without a second model. The jax
interpreter adds the identical term (gate 9).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import stats as _st

from axiom.core.dimensions import Dimension, DimensionError
from axiom.core.expr import Data, Expr, Param, Prior, params
from axiom.core.interpret.dimension import dimension
from axiom.core.interpret.value import value
from axiom.core.result import NonEmptyStr
from axiom.core.spec import Spec

__all__ = [
    "Constraint",
    "ConstraintFamily",
    "Likelihood",
    "LikelihoodFamily",
    "ModelSpec",
    "constrain",
    "free_parameters",
    "log_constraint",
    "log_density",
    "log_likelihood",
    "log_prior",
    "unconstrain",
]

LikelihoodFamily = Literal["normal", "lognormal", "student_t", "poisson"]
ConstraintFamily = Literal["normal", "lognormal"]
Array = npt.NDArray[np.float64]


class Likelihood(Spec):
    """How the outcome column is distributed around the mean expression.

    ``scale`` names the scale parameter (``sigma``) for the continuous
    families; ``df`` is the Student-t degrees of freedom (fixed, not inferred).
    """

    family: LikelihoodFamily
    scale: str | None = None
    df: float | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Likelihood:
        if self.family in ("normal", "lognormal", "student_t") and not self.scale:
            raise ValueError(f"{self.family} likelihood needs a scale parameter name")
        if self.family == "poisson" and self.scale:
            raise ValueError("poisson likelihood has no scale parameter")
        if self.family == "student_t" and (self.df is None or self.df <= 0):
            raise ValueError("student_t likelihood needs df > 0")
        return self


class Constraint(Spec):
    """A soft constraint on a scalar function of the parameters: ``observed ~ p(expr(θ), scale)``.

    ``expr`` is an expression over the model's ``Param`` and ``Data`` nodes
    whose value is a scalar (an estimand at an experiment's doses, reduced
    over its units and periods); it must dimension-check and may only use
    parameters the model declares. ``family="normal"`` contributes
    ``log N(observed | expr, scale)``; ``family="lognormal"`` contributes
    ``log LogNormal(observed | log expr, scale)`` — a multiplicative error
    with ``scale`` on the log scale — and requires ``observed > 0``; where
    ``expr`` is not positive the term is ``-inf`` (never an exception
    inside the density). ``detail`` carries provenance: the estimand hash,
    the measurement source, what the doses were.
    """

    name: NonEmptyStr
    expr: Expr
    family: ConstraintFamily
    observed: float
    scale: float = Field(gt=0)
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Constraint:
        if not np.isfinite(self.observed) or not np.isfinite(self.scale):
            raise ValueError(f"constraint {self.name!r}: observed and scale must be finite")
        if self.family == "lognormal" and self.observed <= 0.0:
            raise ValueError(
                f"constraint {self.name!r}: a lognormal constraint needs observed > 0, "
                f"got {self.observed}"
            )
        try:
            dimension(self.expr)
        except DimensionError as e:
            raise DimensionError(f"constraint {self.name!r}: {e}") from e
        return self


class ModelSpec(Spec):
    """Mean expression + outcome column + likelihood + every parameter's prior.

    ``parameters`` lists every ``Param`` with a prior — those in ``mean`` and
    any that appear only in the likelihood or as hyperparameters of other
    priors. Construction checks that the set is closed (every referenced
    parameter is declared exactly once, every hyper-reference resolves, no
    cycles) and that ``mean`` has the outcome's dimension. ``constraints``
    are soft constraints whose expressions may only use declared
    parameters (with the declared dimension and shape); their data columns
    should be ones the mean already reads, since backends lay out only
    those.
    """

    name: str
    mean: Expr
    outcome: Data
    likelihood: Likelihood
    parameters: tuple[Param, ...]
    constraints: tuple[Constraint, ...] = ()

    @model_validator(mode="after")
    def _closed(self) -> ModelSpec:
        declared = {p.name: p for p in self.parameters}
        if len(declared) != len(self.parameters):
            dupes = sorted(
                {
                    p.name
                    for p in self.parameters
                    if sum(q.name == p.name for q in self.parameters) > 1
                }
            )
            raise ValueError(f"parameters declared more than once: {dupes}")
        for p in params(self.mean):
            if p.name not in declared:
                raise ValueError(f"parameter {p.name!r} appears in mean but is not declared")
            if declared[p.name].dimension != p.dimension or declared[p.name].shape != p.shape:
                raise ValueError(f"parameter {p.name!r} is declared inconsistently")
        if self.likelihood.scale and self.likelihood.scale not in declared:
            raise ValueError(f"scale parameter {self.likelihood.scale!r} is not declared")
        names = [c.name for c in self.constraints]
        if len(set(names)) != len(names):
            raise ValueError(f"constraint names must be distinct: {names}")
        for c in self.constraints:
            for p in params(c.expr):
                if p.name not in declared:
                    raise ValueError(
                        f"constraint {c.name!r} uses parameter {p.name!r}, which the model "
                        "does not declare"
                    )
                if declared[p.name].dimension != p.dimension or declared[p.name].shape != p.shape:
                    raise ValueError(
                        f"constraint {c.name!r} uses parameter {p.name!r} with a different "
                        "dimension or shape than the model declares"
                    )
        for p in self.parameters:
            if p.prior is None:
                raise ValueError(f"parameter {p.name!r} has no prior")
            for parent in p.prior.parents:
                if parent not in declared:
                    raise ValueError(
                        f"prior of {p.name!r} refers to undeclared parameter {parent!r}"
                    )
        # no cycles among hyper-references
        seen: set[str] = set()
        for p in self.parameters:
            stack = [p.name]
            path: set[str] = set()
            while stack:
                n = stack.pop()
                if n in path:
                    raise ValueError(f"prior hierarchy has a cycle through {n!r}")
                path.add(n)
                pr = declared[n].prior
                stack.extend(pr.parents if pr is not None else ())
            seen |= path
        got = dimension(self.mean)
        if got != self.outcome.dimension:
            raise DimensionError(
                f"model {self.name!r}: mean has dimension {got} but outcome "
                f"{self.outcome.name!r} is {self.outcome.dimension}"
            )
        if self.likelihood.scale:
            scale_dim = declared[self.likelihood.scale].dimension
            expected = (
                self.outcome.dimension
                if self.likelihood.family in ("normal", "student_t")
                else Dimension(exponents={})
            )
            if scale_dim != expected:
                raise DimensionError(
                    f"scale parameter {self.likelihood.scale!r} has dimension {scale_dim}; "
                    f"a {self.likelihood.family} likelihood needs {expected}"
                )
        return self

    def parameter(self, name: str) -> Param:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"no parameter {name!r} in model {self.name!r}")

    @property
    def free(self) -> tuple[Param, ...]:
        return free_parameters(self)

    @property
    def shapes(self) -> tuple[Param, ...]:
        return tuple(
            p
            for p in self.parameters
            if p.is_shape and p.prior is not None and p.prior.family != "fixed"
        )

    @property
    def scales(self) -> tuple[Param, ...]:
        return tuple(
            p
            for p in self.parameters
            if not p.is_shape and p.prior is not None and p.prior.family != "fixed"
        )


def free_parameters(model: ModelSpec) -> tuple[Param, ...]:
    """Parameters that are inferred (everything not ``fixed``), in declaration order."""
    return tuple(p for p in model.parameters if p.prior is not None and p.prior.family != "fixed")


# -- transforms ------------------------------------------------------------------------

Support = Literal["real", "positive", "unit", "interval"]


def support_of(prior: Prior) -> Support:
    return {
        "normal": "real",
        "halfnormal": "positive",
        "lognormal": "positive",
        "gamma": "positive",
        "beta": "unit",
        "uniform": "interval",
        "fixed": "real",
    }[
        prior.family
    ]  # type: ignore[return-value]


def _bounds(prior: Prior, theta: Mapping[str, Any]) -> tuple[float, float]:
    lo, hi = prior.hyper["low"], prior.hyper["high"]
    lo_v = float(theta[lo]) if isinstance(lo, str) else float(lo)
    hi_v = float(theta[hi]) if isinstance(hi, str) else float(hi)
    return lo_v, hi_v


def unconstrain(model: ModelSpec, theta: Mapping[str, npt.ArrayLike]) -> dict[str, Array]:
    """Map constrained parameter values to R^k (log / logit per support)."""
    out: dict[str, Array] = {}
    for p in free_parameters(model):
        assert p.prior is not None
        x = np.asarray(theta[p.name], dtype=float)
        s = support_of(p.prior)
        if s == "real":
            out[p.name] = x
        elif s == "positive":
            out[p.name] = np.log(x)
        elif s == "unit":
            out[p.name] = np.log(x) - np.log1p(-x)
        else:
            lo, hi = _bounds(p.prior, theta)
            u = (x - lo) / (hi - lo)
            out[p.name] = np.log(u) - np.log1p(-u)
    return out


def constrain(model: ModelSpec, z: Mapping[str, npt.ArrayLike]) -> tuple[dict[str, Array], float]:
    """Inverse of ``unconstrain``; also returns the summed log-Jacobian ``log|dθ/dz|``.

    Fixed parameters are filled in from their prior. Interval bounds that
    are themselves parameters are resolved in declaration order, so a bound
    parameter must be declared before the parameter it bounds.
    """
    theta: dict[str, Array] = {}
    log_jac = 0.0
    for p in model.parameters:
        assert p.prior is not None
        if p.prior.family == "fixed":
            theta[p.name] = np.asarray(float(p.prior.hyper["value"]), dtype=float)
            continue
        v = np.asarray(z[p.name], dtype=float)
        s = support_of(p.prior)
        if s == "real":
            theta[p.name] = v
        elif s == "positive":
            theta[p.name] = np.exp(v)
            log_jac += float(np.sum(v))
        elif s == "unit":
            u = 1.0 / (1.0 + np.exp(-v))
            theta[p.name] = u
            log_jac += float(np.sum(np.log(u) + np.log1p(-u)))
        else:
            lo, hi = _bounds(p.prior, theta)
            u = 1.0 / (1.0 + np.exp(-v))
            theta[p.name] = lo + (hi - lo) * u
            log_jac += float(np.sum(np.log(u) + np.log1p(-u) + np.log(hi - lo)))
    return theta, log_jac


# -- densities -------------------------------------------------------------------------


def _hyper(prior: Prior, key: str, theta: Mapping[str, Any]) -> Array:
    v = prior.hyper[key]
    return np.asarray(theta[v] if isinstance(v, str) else v, dtype=float)


def log_prior(model: ModelSpec, theta: Mapping[str, npt.ArrayLike]) -> float:
    """Sum of log prior densities of the free parameters at constrained values."""
    total = 0.0
    for p in free_parameters(model):
        pr = p.prior
        assert pr is not None
        x = np.asarray(theta[p.name], dtype=float)

        def h(k: str, _pr: Prior = pr) -> Array:
            return _hyper(_pr, k, theta)

        match pr.family:
            case "normal":
                lp = _st.norm.logpdf(x, loc=h("mu"), scale=h("sigma"))
            case "halfnormal":
                lp = _st.halfnorm.logpdf(x, scale=h("sigma"))
            case "lognormal":
                lp = _st.lognorm.logpdf(x, s=h("sigma"), scale=np.exp(h("mu")))
            case "gamma":
                lp = _st.gamma.logpdf(x, a=h("alpha"), scale=1.0 / h("beta"))
            case "beta":
                lp = _st.beta.logpdf(x, a=h("alpha"), b=h("beta"))
            case "uniform":
                lo, hi = h("low"), h("high")
                lp = _st.uniform.logpdf(x, loc=lo, scale=hi - lo)
            case _:  # pragma: no cover - fixed is excluded by free_parameters
                lp = np.zeros_like(x)
        total += float(np.sum(lp))
    return total


def log_constraint(
    constraint: Constraint, data: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
) -> float:
    """``log p(observed | expr(theta), scale)`` for one soft constraint.

    The expression must evaluate to one number (a trailing singleton axis is
    fine); anything else is a ``ValueError`` — a malformed constraint, not a
    point of the density. A lognormal constraint at a non-positive value is
    ``-inf``.
    """
    v = np.asarray(value(constraint.expr, data=data, params=theta), dtype=float)
    if v.size != 1:
        raise ValueError(
            f"constraint {constraint.name!r} must evaluate to a scalar; got shape {v.shape}"
        )
    x = float(v.reshape(-1)[0])
    if constraint.family == "normal":
        return float(_st.norm.logpdf(constraint.observed, loc=x, scale=constraint.scale))
    if not (x > 0.0):
        return float("-inf")
    return float(_st.lognorm.logpdf(constraint.observed, s=constraint.scale, scale=x))


def log_likelihood(
    model: ModelSpec, data: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
) -> float:
    """Outcome log-likelihood at constrained values, plus every soft constraint's term."""
    y = np.asarray(data[model.outcome.name], dtype=float)
    mu = value(model.mean, data=data, params=theta)
    lik = model.likelihood
    match lik.family:
        case "normal":
            ll = _st.norm.logpdf(y, loc=mu, scale=np.asarray(theta[lik.scale or ""], dtype=float))
        case "lognormal":
            sigma = np.asarray(theta[lik.scale or ""], dtype=float)
            ll = _st.lognorm.logpdf(y, s=sigma, scale=mu)
        case "student_t":
            df = float(lik.df or 0.0)
            scale = np.asarray(theta[lik.scale or ""], dtype=float)
            ll = _st.t.logpdf(y, df=df, loc=mu, scale=scale)
        case "poisson":
            ll = _st.poisson.logpmf(np.round(y), mu)
    total = float(np.sum(ll))
    for c in model.constraints:
        total += log_constraint(c, data, theta)
    return total


def log_density(
    model: ModelSpec, data: Mapping[str, npt.ArrayLike], z: Mapping[str, npt.ArrayLike]
) -> float:
    """Unnormalized log posterior in unconstrained coordinates: prior + Jacobian + likelihood."""
    theta, log_jac = constrain(model, z)
    return log_prior(model, theta) + log_jac + log_likelihood(model, data, theta)
