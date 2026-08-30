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
from axiom.core.expr import Data, Expr, Param, Prior, data_names, params
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
    "binomial_trials",
    "constrain",
    "free_parameters",
    "information_weight",
    "variance_weight",
    "likelihood_scale",
    "log_constraint",
    "log_density",
    "log_likelihood",
    "log_prior",
    "unconstrain",
]

LikelihoodFamily = Literal["normal", "lognormal", "student_t", "poisson", "binomial", "gamma"]
ConstraintFamily = Literal["normal", "lognormal"]
Array = npt.NDArray[np.float64]
_TINY = float(np.finfo(np.float64).tiny)


class Likelihood(Spec):
    """How the outcome column is distributed around the mean expression.

    ``scale`` names the scale parameter (``sigma``) for the continuous
    families; ``df`` is the Student-t degrees of freedom (fixed, not inferred).

    **What ``mean`` is, per family.** For ``normal``, ``student_t`` and
    ``poisson`` it is ``E[y]``. For ``lognormal`` it is the *median*
    (scipy's ``lognorm`` scale), so ``log y ~ N(log mean, scale²)``. For
    ``gamma`` it is ``E[y]`` and ``scale`` is the **coefficient of
    variation**, so ``Var[y] = (scale · mean)²`` — dimensionless, like
    ``lognormal``'s, and the two families then share an information
    geometry. For ``binomial`` it is the **success probability**, not the
    expected count: the outcome column holds successes out of ``trials``,
    and ``E[y] = trials · mean``. That is the GLM convention (the mean
    expression is what a ``sigmoid`` produces) and it is what makes
    ``d mean / d theta`` the Jacobian the design math wants.

    ``scale_expr`` is the alternative to ``scale`` for a **known or
    structured per-observation scale**: an expression over ``Data`` and
    ``Param`` nodes evaluated alongside the mean — a meta-analysis's
    ``sqrt(se_i² + tau²)`` with ``se`` a data column and ``tau`` a parameter,
    or a fixed heteroscedastic weight column. Exactly one of ``scale`` and
    ``scale_expr`` is set for a continuous family; it must dimension-check
    like a scale parameter would (the outcome's dimension for ``normal`` and
    ``student_t``, dimensionless for ``lognormal``).
    """

    family: LikelihoodFamily
    scale: str | None = None
    scale_expr: Expr | None = None
    df: float | None = None
    trials: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Likelihood:
        if self.scale and self.scale_expr is not None:
            raise ValueError("a likelihood takes either a scale parameter name or a scale_expr")
        if self.family in ("normal", "lognormal", "student_t", "gamma") and not (
            self.scale or self.scale_expr is not None
        ):
            raise ValueError(f"{self.family} likelihood needs a scale parameter name or scale_expr")
        if self.family in ("poisson", "binomial") and (self.scale or self.scale_expr is not None):
            raise ValueError(f"{self.family} likelihood has no scale parameter")
        if self.family == "student_t" and (self.df is None or self.df <= 0):
            raise ValueError("student_t likelihood needs df > 0")
        if self.trials is not None and self.family != "binomial":
            raise ValueError(f"trials is a binomial column; a {self.family} likelihood has none")
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
        if self.likelihood.scale_expr is not None:
            for p in params(self.likelihood.scale_expr):
                if p.name not in declared:
                    raise ValueError(
                        f"likelihood scale_expr uses parameter {p.name!r}, which the model "
                        "does not declare"
                    )
                if declared[p.name].dimension != p.dimension or declared[p.name].shape != p.shape:
                    raise ValueError(
                        f"likelihood scale_expr uses parameter {p.name!r} with a different "
                        "dimension or shape than the model declares"
                    )
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
        if self.likelihood.scale or self.likelihood.scale_expr is not None:
            expected = (
                self.outcome.dimension
                if self.likelihood.family in ("normal", "student_t")
                else Dimension(exponents={})
            )
            if self.likelihood.scale:
                scale_dim = declared[self.likelihood.scale].dimension
                what = f"scale parameter {self.likelihood.scale!r}"
            else:
                assert self.likelihood.scale_expr is not None
                scale_dim = dimension(self.likelihood.scale_expr)
                what = "likelihood scale_expr"
            if scale_dim != expected:
                raise DimensionError(
                    f"{what} has dimension {scale_dim}; "
                    f"a {self.likelihood.family} likelihood needs {expected}"
                )
        return self

    @property
    def data_columns(self) -> tuple[str, ...]:
        """Every data column the model reads: mean, outcome, scale_expr, constraints."""
        out: list[str] = list(data_names(self.mean))
        for name in (self.outcome.name, *self._scale_columns(), *self._constraint_columns()):
            if name not in out:
                out.append(name)
        return tuple(out)

    def _scale_columns(self) -> tuple[str, ...]:
        trials = (self.likelihood.trials,) if self.likelihood.trials else ()
        if self.likelihood.scale_expr is None:
            return trials
        return (*data_names(self.likelihood.scale_expr), *trials)

    def _constraint_columns(self) -> tuple[str, ...]:
        return tuple(n for c in self.constraints for n in data_names(c.expr))

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


def likelihood_scale(
    model: ModelSpec, data: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
) -> Array:
    """The likelihood's scale at constrained values: the named parameter, or ``scale_expr``
    evaluated on ``data`` and ``theta``. ``ValueError`` for a family without one."""
    lik = model.likelihood
    if lik.scale:
        return np.asarray(theta[lik.scale], dtype=float)
    if lik.scale_expr is not None:
        return np.asarray(value(lik.scale_expr, data=data, params=theta), dtype=float)
    raise ValueError(f"a {lik.family} likelihood has no scale")


def binomial_trials(model: ModelSpec, data: Mapping[str, npt.ArrayLike]) -> Array:
    """The binomial trial counts: the ``trials`` column, or ones (Bernoulli).

    ``ValueError`` for a family that has none, so a caller cannot silently
    treat a Poisson model as one-trial binomial.
    """
    lik = model.likelihood
    if lik.family != "binomial":
        raise ValueError(f"a {lik.family} likelihood has no trials")
    if lik.trials is None:
        return np.ones(1, dtype=float)
    n = np.asarray(data[lik.trials], dtype=float)
    if np.any(n <= 0) or np.any(n != np.round(n)):
        raise ValueError(f"trials column {lik.trials!r} must be positive whole numbers")
    return n


def variance_weight(
    family: str,
    mu: npt.ArrayLike,
    *,
    scale: npt.ArrayLike | None = None,
    df: float | None = None,
    trials: npt.ArrayLike | None = None,
) -> Array:
    """``w = 1 / (phi · V(mu))`` for one family — the whole of what design needs.

    Split out from :func:`information_weight` because the design layer asks
    this question at candidate designs where there is no fitted model to
    read a mean off: it has ``mu`` from ``forward()`` already. Both callers
    go through here, so the variance functions live in exactly one place.
    """
    m = np.asarray(mu, dtype=float)
    match family:
        case "normal":
            return 1.0 / _scale_of(scale, family) ** 2
        case "student_t":
            d = float(df or 0.0)
            if d <= 0.0:
                raise ValueError("student_t needs df > 0 to weight an observation")
            return ((d + 1.0) / (d + 3.0)) / _scale_of(scale, family) ** 2
        case "lognormal" | "gamma":
            _positive(m, family, "mean")
            return np.asarray(1.0 / (_scale_of(scale, family) ** 2 * m**2), dtype=np.float64)
        case "poisson":
            _positive(m, family, "mean")
            return 1.0 / m
        case "binomial":
            if np.any(m <= 0.0) or np.any(m >= 1.0):
                raise ValueError(
                    "a binomial mean is a success probability and must lie strictly inside "
                    "(0, 1); it is outside at this point, so the information is undefined"
                )
            n = 1.0 if trials is None else np.asarray(trials, dtype=float)
            return n / (m * (1.0 - m))
        case _:
            raise ValueError(f"no variance function for likelihood family {family!r}")


def _scale_of(scale: npt.ArrayLike | None, family: str) -> Array:
    if scale is None:
        raise ValueError(f"a {family} weight needs the likelihood's scale")
    s = np.asarray(scale, dtype=float)
    if np.any(s <= 0.0):
        raise ValueError(f"a {family} scale must be positive")
    return s


def _positive(mu: Array, family: str, what: str) -> None:
    if np.any(mu <= 0.0):
        raise ValueError(
            f"a {family} {what} must be positive; it is not at this point, so the "
            "information is undefined there"
        )


def information_weight(
    model: ModelSpec, data: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
) -> Array:
    """``w_i = 1 / (phi · V(mu_i))`` — the diagonal of ``W`` in ``FI = J' W J``.

    ``J = d mean / d theta``. Because axiom differentiates the **mean**
    rather than a linear predictor, the link function never appears here:
    it is already inside ``J``. All that distinguishes one family from
    another at this level is its variance function.

    ==============  ===================  ==========================
    family          ``phi · V(mu)``      ``w``
    ==============  ===================  ==========================
    ``normal``      ``sigma²``           ``1 / sigma²``
    ``student_t``   —                    ``(df+1) / ((df+3) sigma²)``
    ``lognormal``   ``(sigma · mu)²``    ``1 / (sigma · mu)²``
    ``gamma``       ``(cv · mu)²``       ``1 / (cv · mu)²``
    ``poisson``     ``mu``               ``1 / mu``
    ``binomial``    ``mu(1-mu) / n``     ``n / (mu(1-mu))``
    ==============  ===================  ==========================

    ``student_t`` is not an exponential family, but its location
    information is still a constant multiple of the Gaussian one — the
    score is bounded, which is what buys the robustness — so it fits the
    same shape. It tends to ``1 / sigma²`` as ``df`` grows, which is the
    check to make when changing this. ``lognormal`` and ``gamma`` share a
    weight because both put a constant coefficient of variation on a
    positive mean; they differ in their density, not in this geometry.

    Returns an array broadcastable against the outcome. ``Unsupported`` is
    not used here: a mean outside its family's support is a modelling
    error, and raises.
    """
    lik = model.likelihood
    mu = value(model.mean, data=data, params=theta)
    has_scale = bool(lik.scale) or lik.scale_expr is not None
    return variance_weight(
        lik.family,
        mu,
        scale=likelihood_scale(model, data, theta) if has_scale else None,
        df=lik.df,
        trials=binomial_trials(model, data) if lik.family == "binomial" else None,
    )


def log_likelihood(
    model: ModelSpec, data: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
) -> float:
    """Outcome log-likelihood at constrained values, plus every soft constraint's term."""
    y = np.asarray(data[model.outcome.name], dtype=float)
    mu = value(model.mean, data=data, params=theta)
    lik = model.likelihood
    match lik.family:
        case "normal":
            ll = _st.norm.logpdf(y, loc=mu, scale=likelihood_scale(model, data, theta))
        case "lognormal":
            ll = _st.lognorm.logpdf(y, s=likelihood_scale(model, data, theta), scale=mu)
        case "student_t":
            df = float(lik.df or 0.0)
            ll = _st.t.logpdf(y, df=df, loc=mu, scale=likelihood_scale(model, data, theta))
        case "poisson":
            ll = _st.poisson.logpmf(np.round(y), mu)
        case "binomial":
            # ``mu`` is the success probability; ``y`` the successes out of ``trials``.
            n = binomial_trials(model, data)
            p = np.asarray(mu, dtype=float)
            ll = np.where(
                (p > 0.0) & (p < 1.0),
                _st.binom.logpmf(np.round(y), n, np.clip(p, _TINY, 1.0 - _TINY)),
                -np.inf,
            )
        case "gamma":
            # ``scale`` is the coefficient of variation: shape = 1 / cv², and the
            # scipy scale is mean / shape, so Var = (cv · mean)².
            cv = likelihood_scale(model, data, theta)
            shape = 1.0 / np.asarray(cv, dtype=float) ** 2
            m = np.asarray(mu, dtype=float)
            ll = np.where(
                m > 0.0,
                _st.gamma.logpdf(y, a=shape, scale=np.where(m > 0.0, m, 1.0) / shape),
                -np.inf,
            )
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
