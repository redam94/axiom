"""Saturation (dose-response) kernels as expression builders.

A kernel does not compute anything. It *builds* the ``axiom.core`` expression
for one treatment's response — ``beta · f(dose / k)`` — so that the
likelihood, the simulator, the design math, and the optimizer all evaluate
the same tree through ``axiom.core.value`` or ``axiom.core.compile_jax``
(rule 3, "one ``forward()``").

Every kernel is a flat ``Spec`` (no base class; 0002.13) satisfying the
``ResponseKernel`` protocol (D3), and declares its parameter split through
``roles``:

* **scale** — the half-saturation ``k``; carries the *dose* dimension. Its
  default prior is centred on the kernel's ``reference_dose`` (review B9: the
  reference is a per-treatment convention that ``meta`` pools ``k`` against).
* **shape** — ``s`` and friends; dimensionless. Shapes are the Pi groups
  that are invariant to the unit the dose is measured in (Phase 3 exit 6).
* **amplitude** — ``beta``; carries the *outcome* dimension (``LinearKernel``
  is the exception: its rate carries ``outcome / dose``).

There is no ``Deriv`` node (review C2), so each kernel ships its derivative
``d response / d dose`` in closed form, built from the same node set.

**Zero-dose gradients.** The families with a shape exponent (``hill``,
``power``) are written as ``x̃^s / (k̃^s + x̃^s)`` and ``x̃^s / k̃^s`` with
``x̃ = dose / c``, ``k̃ = k / c`` and ``c = Const(1.0, dose dimension)``,
rather than as ``(dose / k)^s``. The values are identical, but the naive
form differentiates ``(dose / k)^s`` through ``k`` at ``dose = 0``, which for
``s < 1`` is ``∞ · 0 = nan`` under automatic differentiation; the scaled form
only ever raises ``k̃ > 0`` to the power ``s``, so the jax gradient with
respect to ``k`` is finite on zero-dose rows. ``c`` is the unit of the dose
dimension, so the Pi groups (and unit invariance) are unchanged.

What this does *not* fix: for ``s < 1`` the slope of ``x̃^s`` at zero is
genuinely infinite, so when the dose expression itself depends on
parameters (a carryover in front of the kernel) and a row's *carried* dose
is exactly zero, the chain rule meets ``∞ · 0`` and the gradient in the
carryover parameters is ``nan`` on that row. That is a property of the
family, not of the tree; keeping the shape prior's mass at ``s ≥ 1`` (the
``hill`` default) avoids it, and ``tests/unit/test_kernels.py`` pins the
limitation with a strict ``xfail``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import (
    Add,
    Apply,
    Const,
    D,
    Dimension,
    Div,
    Expr,
    Mul,
    Param,
    Pow,
    Prior,
    Spec,
    dimension,
    dimensionless,
)

__all__ = [
    "AMPLITUDE_PRIOR_FAMILIES",
    "BASIS_PRIOR_FAMILIES",
    "KERNELS",
    "AnyKernel",
    "CovarianceFamily",
    "ExponentialKernel",
    "GaussianProcessKernel",
    "HillKernel",
    "KernelRole",
    "LinearKernel",
    "LogisticKernel",
    "PiecewiseLinearKernel",
    "PolynomialKernel",
    "PowerKernel",
    "ResponseKernel",
    "SplineKernel",
    "kernel_from_name",
]

Array = npt.NDArray[np.float64]

KernelRole = Literal["scale", "shape", "amplitude"]
"""What a kernel parameter is: a dose-dimensioned scale, a dimensionless shape, or an amplitude."""

AMPLITUDE_PRIOR_FAMILIES: frozenset[str] = frozenset({"halfnormal", "lognormal", "gamma"})
"""Prior families with support on R+, the only ones an explicit ``amplitude_prior`` may use."""


BASIS_PRIOR_FAMILIES: frozenset[str] = AMPLITUDE_PRIOR_FAMILIES | frozenset({"normal"})
"""What a *basis* family's coefficients may use: the positive families, plus ``normal``.

A saturating kernel's amplitude is an asymptote, signed by convention, so its prior
has positive support. A basis coefficient is not: a polynomial or a spline represents
a dose-response that *turns over* precisely by letting some coefficients go negative,
so ``normal`` is their default and is admissible here. Declaring a positive family
instead is how a caller asks a basis family for a monotone fit.
"""


def _check_amplitude_prior(
    prior: Prior | None, families: frozenset[str] = AMPLITUDE_PRIOR_FAMILIES
) -> None:
    """An explicit amplitude prior must come from ``families`` and be fully numeric (D6.2)."""
    if prior is None:
        return
    if prior.family not in families:
        why = " (positive support)" if families == AMPLITUDE_PRIOR_FAMILIES else " (signed allowed)"
        raise ValueError(
            f"amplitude_prior must be one of {sorted(families)}{why}, got {prior.family!r}"
        )
    if prior.parents:
        raise ValueError("amplitude_prior cannot reference other parameters")


@runtime_checkable
class ResponseKernel(Protocol):
    """A dose-response family that builds its own expression tree.

    ``parameters`` names and dimensions the ``Param`` nodes the family
    introduces for one treatment (``"k_tv"``, ``"s_tv"``, ``"beta_tv"``), with
    default priors. ``saturation`` is the dimensionless ``f(dose / k)``;
    ``response`` is ``beta · f``; ``derivative`` is the closed-form
    ``d response / d dose`` with dimension ``outcome / dose``. The dose
    dimension is read off the ``dose`` expression; the outcome dimension
    defaults to ``D.outcome``.

    ``saturation_derivative`` is ``d saturation / d dose`` (dimension
    ``1 / dose``). It is what an interaction term differentiates, and it is
    why a family does not have to declare exactly one amplitude in order to
    take part in one.

    For every family with a ``scale`` parameter, ``response == amplitude ·
    saturation`` holds node for node. ``LinearKernel`` is the documented
    exception: it has no scale, its ``saturation`` is the identity shape
    ``dose / reference_dose`` (a dimensionless convention, not a factor of
    the response), and its ``response`` is ``beta_rate · dose`` with
    ``beta_rate`` carrying ``outcome / dose``. Callers that decompose a
    response into amplitude and saturation must branch on ``name ==
    "linear"`` (or on ``"scale" not in roles.values()``).

    The **basis** families — ``polynomial``, ``spline``, ``piecewise_linear``
    — extend that exception. They declare *several* amplitudes, one per basis
    function and signed; their ``saturation`` is likewise the identity shape
    ``dose / reference_dose``; and no single parameter can be factored out of
    the response. ``roles`` tells them apart: more than one ``"amplitude"``.
    """

    @property
    def name(self) -> str: ...
    @property
    def roles(self) -> Mapping[str, KernelRole]: ...
    @property
    def saturating(self) -> bool: ...
    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]: ...
    def saturation(self, dose: Expr, treatment: str) -> Expr: ...
    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr: ...
    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr: ...
    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr: ...


# -- shared builders (composition, not inheritance) -----------------------------------


def _one() -> Const:
    return Const(value=1.0, dimension=dimensionless())


def _num(v: float) -> Const:
    return Const(value=float(v), dimension=dimensionless())


def _outcome(d: Dimension | None) -> Dimension:
    return d if d is not None else D.outcome


def _scale_param(role: str, treatment: str, dose_dimension: Dimension, reference: float) -> Param:
    """A half-saturation scale: lognormal centred on the reference dose, one e-fold wide."""
    return Param(
        name=f"{role}_{treatment}",
        dimension=dose_dimension,
        prior=Prior(family="lognormal", hyper={"mu": math.log(reference), "sigma": 1.0}),
    )


def _shape_param(role: str, treatment: str) -> Param:
    """A dimensionless shape on R+: gamma(4, 2), mean 2, most mass on (0.5, 4)."""
    return Param(
        name=f"{role}_{treatment}",
        dimension=dimensionless(),
        prior=Prior(family="gamma", hyper={"alpha": 4.0, "beta": 2.0}),
    )


def _unit_shape_param(role: str, treatment: str) -> Param:
    """A dimensionless shape on (0, 1): beta(2, 2), mean 1/2, symmetric, vanishing at the ends."""
    return Param(
        name=f"{role}_{treatment}",
        dimension=dimensionless(),
        prior=Prior(family="beta", hyper={"alpha": 2.0, "beta": 2.0}),
    )


def _amplitude_param(
    role: str, treatment: str, dim: Dimension, scale: float, prior: Prior | None = None
) -> Param:
    """A non-negative amplitude: the explicit ``prior`` when set, else halfnormal(``scale``).

    The explicit prior is how ``calibrate`` hands randomized evidence to the
    surface (the prior route, D6.2): the kernel stays the same family and
    shape, only the amplitude's prior changes.
    """
    if prior is None:
        prior = Prior(family="halfnormal", hyper={"sigma": float(scale)})
    return Param(name=f"{role}_{treatment}", dimension=dim, prior=prior)


def _ratio(dose: Expr, k: Param) -> Div:
    """``u = dose / k`` — the dimensionless argument of the families without a shape exponent."""
    return Div(numerator=dose, denominator=k)


def _unit_dose(dose: Expr) -> Const:
    """``c = 1`` in the dose dimension: the unit that makes ``x̃ = dose / c`` dimensionless."""
    return Const(value=1.0, dimension=dimension(dose))


def _scaled(beta: Param, f: Expr) -> Mul:
    return Mul(factors=(beta, f))


def _tilde(dose: Expr, k: Param) -> tuple[Const, Div, Div]:
    """``(c, x̃, k̃)`` with ``x̃ = dose / c`` and ``k̃ = k / c`` (see the module docstring)."""
    c = _unit_dose(dose)
    return c, Div(numerator=dose, denominator=c), Div(numerator=k, denominator=c)


def _over_k(beta: Param, k: Param, df_du: Expr) -> Div:
    """``beta · f'(u) / k`` — the chain rule through ``u = dose / k``."""
    return Div(numerator=Mul(factors=(beta, df_du)), denominator=k)


# -- basis families: shared pieces -------------------------------------------------------


def _reduced(dose: Expr, reference: float) -> Div:
    """``u = dose / reference_dose`` — the dimensionless coordinate a basis is built on."""
    return Div(numerator=dose, denominator=Const(value=float(reference), dimension=dimension(dose)))


def _inverse_reference(dose: Expr, reference: float) -> Div:
    """``du / d dose = 1 / reference_dose``, with dimension ``1 / dose``."""
    return Div(
        numerator=_one(), denominator=Const(value=float(reference), dimension=dimension(dose))
    )


def _coefficient_param(
    stem: str, treatment: str, dim: Dimension, scale: float, prior: Prior | None = None
) -> Param:
    """A *signed* basis coefficient: the explicit ``prior`` when set, else ``normal(0, scale)``."""
    if prior is None:
        prior = Prior(family="normal", hyper={"mu": 0.0, "sigma": float(scale)})
    return Param(name=f"{stem}_{treatment}", dimension=dim, prior=prior)


def _combine(coefficients: Sequence[Param], basis: Sequence[Expr]) -> Expr:
    """``Σ_j beta_j · B_j`` — the one shape every basis family's response and derivative take."""
    terms = tuple(Mul(factors=(c, b)) for c, b in zip(coefficients, basis, strict=True))
    return terms[0] if len(terms) == 1 else Add(terms=terms)


def _hinge(u: Expr, knot: float) -> Apply:
    """``(u − t)_+``, the truncated-power hinge."""
    return Apply(fn="relu", arg=Add(terms=(u, _num(-float(knot)))))


def _hinge_step(u: Expr, knot: float) -> Apply:
    """``d (u − t)_+ / du``: ``1`` past the knot, ``0`` before it."""
    return Apply(fn="step", arg=Add(terms=(u, _num(-float(knot)))))


def _check_knots(knots: tuple[float, ...], minimum: int, reference: float) -> None:
    """Knots are in dose units: strictly increasing, strictly positive, finite."""
    if len(knots) < minimum:
        raise ValueError(f"need at least {minimum} knots, got {len(knots)}")
    previous = 0.0
    for t in knots:
        if not math.isfinite(t):
            raise ValueError(f"knots must be finite, got {knots}")
        if t <= previous:
            raise ValueError(
                f"knots must be strictly increasing and strictly positive, got {knots}"
            )
        previous = t
    if reference <= 0.0:  # pragma: no cover - Field(gt=0) already guarantees it
        raise ValueError("reference_dose must be positive")


# -- families --------------------------------------------------------------------------


class HillKernel(Spec):
    """``f(u) = u^s / (1 + u^s)``, ``u = dose / k``. Saturating; ``f(0) = 0``, ``f(1) = 1/2``.

    ``k`` is the half-saturation dose (scale), ``s`` the steepness (shape),
    ``beta`` the asymptote (amplitude). ``s < 1`` gives infinite slope at
    zero dose; the default ``gamma(4, 2)`` prior keeps most mass above 1.

    Built as ``x̃^s / (k̃^s + x̃^s)`` with ``x̃ = dose / c``, ``k̃ = k / c``,
    ``c`` the dose unit, so the gradient in ``k`` is finite at zero dose
    (module docstring). Overflow: for ``s · log(x̃) > 709`` (or the same for
    ``k̃``) the power overflows to ``inf`` in float64 and the saturation
    evaluates to ``nan`` while the derivative correctly underflows to zero.
    That needs ``dose / k`` beyond ``e^(709 / s)`` — astronomically far
    outside any observed range for ``s ≥ 1`` and only reachable for very
    small ``s`` with extreme ``dose / k`` ratios; it is not guarded.
    """

    name: Literal["hill"] = "hill"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None

    @model_validator(mode="after")
    def _positive_amplitude_prior(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior)
        return self

    @property
    def roles(self) -> dict[str, KernelRole]:
        return {"k": "scale", "s": "shape", "beta": "amplitude"}

    @property
    def saturating(self) -> bool:
        return True

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        return (
            _scale_param("k", treatment, dose_dimension, self.reference_dose),
            _shape_param("s", treatment),
            _amplitude_param(
                "beta", treatment, outcome_dimension, self.amplitude_scale, self.amplitude_prior
            ),
        )

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        k, s, _ = self.parameters(treatment, dimension(dose), D.outcome)
        _, x_tilde, k_tilde = _tilde(dose, k)
        xs = Pow(base=x_tilde, exponent=s)
        ks = Pow(base=k_tilde, exponent=s)
        return Div(numerator=xs, denominator=Add(terms=(ks, xs)))

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        _, _, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation(dose, treatment))

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        """``s · x̃^(s-1) · k̃^s / (c · (k̃^s + x̃^s)^2)`` = ``s · u^(s-1) / (k · (1 + u^s)^2)``."""
        k, s, _ = self.parameters(treatment, dimension(dose), D.outcome)
        c, x_tilde, k_tilde = _tilde(dose, k)
        xs = Pow(base=x_tilde, exponent=s)
        ks = Pow(base=k_tilde, exponent=s)
        x_sm1 = Pow(base=x_tilde, exponent=Add(terms=(s, _num(-1.0))))
        denominator = Mul(factors=(c, Pow(base=Add(terms=(ks, xs)), exponent=Fraction(2))))
        return Div(numerator=Mul(factors=(s, x_sm1, ks)), denominator=denominator)

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta · s · x̃^(s-1) · k̃^s / (c · (k̃^s + x̃^s)^2)``."""
        _, _, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation_derivative(dose, treatment))


class LogisticKernel(Spec):
    """``f(u) = 2 / (1 + exp(-u)) − 1 = tanh(u / 2)``, ``u = dose / k``. Saturating; ``f(0) = 0``.

    A one-shape-parameter-free sigmoid: ``k`` sets the dose at which the
    response reaches ``tanh(1/2) ≈ 0.46`` of its asymptote ``beta``.
    """

    name: Literal["logistic"] = "logistic"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None

    @model_validator(mode="after")
    def _positive_amplitude_prior(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior)
        return self

    @property
    def roles(self) -> dict[str, KernelRole]:
        return {"k": "scale", "beta": "amplitude"}

    @property
    def saturating(self) -> bool:
        return True

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        return (
            _scale_param("k", treatment, dose_dimension, self.reference_dose),
            _amplitude_param(
                "beta", treatment, outcome_dimension, self.amplitude_scale, self.amplitude_prior
            ),
        )

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        k, _ = self.parameters(treatment, dimension(dose), D.outcome)
        sig = Apply(fn="sigmoid", arg=_ratio(dose, k))
        return Add(terms=(Mul(factors=(_num(2.0), sig)), _num(-1.0)))

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        _, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation(dose, treatment))

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        """``2 σ(u) (1 − σ(u)) / k``."""
        k, _ = self.parameters(treatment, dimension(dose), D.outcome)
        sig = Apply(fn="sigmoid", arg=_ratio(dose, k))
        df_du = Mul(factors=(_num(2.0), sig, Add(terms=(_one(), Apply(fn="neg", arg=sig)))))
        return Div(numerator=df_du, denominator=k)

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta · 2 σ(u) (1 − σ(u)) / k``."""
        _, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation_derivative(dose, treatment))


class ExponentialKernel(Spec):
    """``f(u) = 1 − exp(−u)``, ``u = dose / k``. Saturating; ``f(0) = 0``; concave everywhere.

    ``k`` is the dose at which ``1 − 1/e ≈ 63%`` of the asymptote ``beta`` is reached.
    """

    name: Literal["exponential"] = "exponential"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None

    @model_validator(mode="after")
    def _positive_amplitude_prior(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior)
        return self

    @property
    def roles(self) -> dict[str, KernelRole]:
        return {"k": "scale", "beta": "amplitude"}

    @property
    def saturating(self) -> bool:
        return True

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        return (
            _scale_param("k", treatment, dose_dimension, self.reference_dose),
            _amplitude_param(
                "beta", treatment, outcome_dimension, self.amplitude_scale, self.amplitude_prior
            ),
        )

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        k, _ = self.parameters(treatment, dimension(dose), D.outcome)
        # 1 - exp(-u) == -expm1(-u); expm1 keeps precision at small doses.
        return Apply(fn="neg", arg=Apply(fn="expm1", arg=Apply(fn="neg", arg=_ratio(dose, k))))

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        _, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation(dose, treatment))

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        """``exp(−u) / k``."""
        k, _ = self.parameters(treatment, dimension(dose), D.outcome)
        df_du = Apply(fn="exp", arg=Apply(fn="neg", arg=_ratio(dose, k)))
        return Div(numerator=df_du, denominator=k)

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta · exp(−u) / k``."""
        _, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation_derivative(dose, treatment))


class PowerKernel(Spec):
    """``f(u) = u^s``, ``u = dose / k``. **Not saturating**: unbounded as the dose grows.

    ``f(0) = 0`` for ``s > 0``; concave for ``s < 1``, convex for ``s > 1``.
    ``k`` is the dose at which the response equals ``beta`` (``f(1) = 1``),
    so ``beta`` is the response at the reference scale rather than an
    asymptote. Use it for diminishing returns without a ceiling, never for
    extrapolation far beyond the observed dose range.

    The default shape prior is ``s ~ beta(2, 2)``: the family is shipped for
    *diminishing* returns, and that prior puts all of its mass on the
    concave range ``0 < s < 1`` (mean ``1/2``, density vanishing at both
    ends, so ``s`` is pulled away from the linear ``s = 1`` and the step
    ``s = 0``). A convex ``s > 1`` is outside the default support; declare
    a different prior for ``s_<treatment>`` in the ``ModelSpec`` to allow it.

    Built as ``x̃^s / k̃^s`` with ``x̃ = dose / c``, ``k̃ = k / c`` (module
    docstring) so the gradient in ``k`` is finite at zero dose.
    """

    name: Literal["power"] = "power"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None

    @model_validator(mode="after")
    def _positive_amplitude_prior(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior)
        return self

    @property
    def roles(self) -> dict[str, KernelRole]:
        return {"k": "scale", "s": "shape", "beta": "amplitude"}

    @property
    def saturating(self) -> bool:
        return False

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        return (
            _scale_param("k", treatment, dose_dimension, self.reference_dose),
            _unit_shape_param("s", treatment),
            _amplitude_param(
                "beta", treatment, outcome_dimension, self.amplitude_scale, self.amplitude_prior
            ),
        )

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        k, s, _ = self.parameters(treatment, dimension(dose), D.outcome)
        _, x_tilde, k_tilde = _tilde(dose, k)
        return Div(
            numerator=Pow(base=x_tilde, exponent=s), denominator=Pow(base=k_tilde, exponent=s)
        )

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        _, _, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation(dose, treatment))

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        """``s · x̃^(s−1) / (k̃^s · c)`` = ``s · u^(s−1) / k``."""
        k, s, _ = self.parameters(treatment, dimension(dose), D.outcome)
        c, x_tilde, k_tilde = _tilde(dose, k)
        x_sm1 = Pow(base=x_tilde, exponent=Add(terms=(s, _num(-1.0))))
        denominator = Mul(factors=(Pow(base=k_tilde, exponent=s), c))
        return Div(numerator=Mul(factors=(s, x_sm1)), denominator=denominator)

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta · s · u^(s−1) / k``."""
        _, _, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation_derivative(dose, treatment))


class LinearKernel(Spec):
    """``response = beta_rate · dose``. No scale, no shape; never saturates.

    ``beta_rate`` carries ``outcome / dose`` and is the only parameter; its
    role is recorded as ``amplitude``. The default prior is
    ``halfnormal(amplitude_scale / reference_dose)`` so that the response at
    the reference dose has scale ``amplitude_scale``, like the other families.
    ``saturation`` returns ``dose / reference_dose`` — the identity shape in
    units of the reference — and ``derivative`` is the constant ``beta_rate``
    (a scalar that broadcasts against any dose grid).
    """

    name: Literal["linear"] = "linear"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None

    @model_validator(mode="after")
    def _positive_amplitude_prior(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior)
        return self

    @property
    def roles(self) -> dict[str, KernelRole]:
        return {"beta_rate": "amplitude"}

    @property
    def saturating(self) -> bool:
        return False

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        return (
            _amplitude_param(
                "beta_rate",
                treatment,
                outcome_dimension / dose_dimension,
                self.amplitude_scale / self.reference_dose,
                self.amplitude_prior,
            ),
        )

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        reference = Const(value=self.reference_dose, dimension=dimension(dose))
        return Div(numerator=dose, denominator=reference)

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        (beta_rate,) = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return Mul(factors=(beta_rate, dose))

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        """``1 / reference_dose`` — the identity shape's constant slope."""
        return _inverse_reference(dose, self.reference_dose)

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta_rate`` — constant in the dose."""
        (beta_rate,) = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return beta_rate


# -- basis families ----------------------------------------------------------------------


class PolynomialKernel(Spec):
    """``response = Σ_{j=1..degree} beta_j · u^j``, ``u = dose / reference_dose``.

    The first family that can bend back down. Every coefficient is signed and
    the response is **linear in all of them**, so ``linearize`` is exact
    rather than a local approximation, ``design_matrix`` is the real design
    matrix, and the alphabet-optimality criteria in ``surface.designs`` mean
    what they say. There is no constant term: the surface supplies the
    intercept, and ``response(0) = 0`` like every other family.

    ``reference_dose`` is not a parameter here, it is the coordinate: set it
    to the **top of the dose range you intend to fit**, so that ``u`` lands in
    ``[0, 1]`` and the default ``normal(0, amplitude_scale)`` prior means the
    same thing for every power. Leaving it at 1.0 with doses in the tens
    makes ``u^3`` four orders of magnitude larger than ``u``, and no single
    prior scale is then sensible for both.

    Not saturating, and not to be extrapolated: outside the fitted range a
    polynomial does whatever its leading term says, which for ``degree >= 3``
    is a lot. ``degree`` is capped at 8 because the powers of ``u`` are badly
    conditioned long before that; past a cubic, prefer ``SplineKernel``,
    whose basis is local and whose tails are linear.
    """

    name: Literal["polynomial"] = "polynomial"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None
    degree: int = Field(default=3, ge=1, le=8)

    @model_validator(mode="after")
    def _signed_amplitude_prior(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior, BASIS_PRIOR_FAMILIES)
        return self

    @property
    def stems(self) -> tuple[str, ...]:
        """The coefficient stems, in basis order: ``beta1`` is the linear term."""
        return tuple(f"beta{j}" for j in range(1, self.degree + 1))

    @property
    def roles(self) -> dict[str, KernelRole]:
        return dict.fromkeys(self.stems, "amplitude")

    @property
    def saturating(self) -> bool:
        return False

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        return tuple(
            _coefficient_param(
                stem, treatment, outcome_dimension, self.amplitude_scale, self.amplitude_prior
            )
            for stem in self.stems
        )

    def basis(self, dose: Expr) -> tuple[Expr, ...]:
        """``(u, u^2, ..., u^degree)`` — the dimensionless basis the coefficients multiply."""
        u = _reduced(dose, self.reference_dose)
        return tuple(
            u if j == 1 else Pow(base=u, exponent=Fraction(j)) for j in range(1, self.degree + 1)
        )

    def basis_derivative(self, dose: Expr) -> tuple[Expr, ...]:
        """``d basis / d dose`` — ``(j · u^(j-1) / reference_dose)_j``."""
        u = _reduced(dose, self.reference_dose)
        slope = _inverse_reference(dose, self.reference_dose)
        out: list[Expr] = []
        for j in range(1, self.degree + 1):
            if j == 1:
                out.append(slope)
            elif j == 2:
                out.append(Mul(factors=(_num(2.0), u, slope)))
            else:
                out.append(
                    Mul(factors=(_num(float(j)), Pow(base=u, exponent=Fraction(j - 1)), slope))
                )
        return tuple(out)

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        """``u`` — the identity shape, as for ``LinearKernel``; see the module docstring."""
        return _reduced(dose, self.reference_dose)

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        return _inverse_reference(dose, self.reference_dose)

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        betas = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _combine(betas, self.basis(dose))

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``Σ_j j · beta_j · u^(j-1) / reference_dose``."""
        betas = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _combine(betas, self.basis_derivative(dose))


class SplineKernel(Spec):
    """A **natural cubic spline** in ``u = dose / reference_dose``, on fixed knots.

    Cubic between the boundary knots and **linear outside them**, which is
    what makes it the family to reach for when a dose-response might turn
    over: it can bend as often as the interior knots allow and still
    extrapolate like a straight line instead of like a cubic. Coefficients
    are signed and the response is linear in all of them, so everything said
    about ``PolynomialKernel`` and ``linearize`` holds here too.

    The basis is the standard reduced one (Hastie, Tibshirani & Friedman,
    *ESL* 5.2.1) with the constant dropped, since the surface supplies the
    intercept::

        B_1(u)     = u
        B_{k+1}(u) = d_k(u) - d_{K-1}(u),          k = 1 .. K-2
        d_k(u)     = [ (u-t_k)_+^3 - (u-t_K)_+^3 ] / (t_K - t_k)

    for ``K`` knots ``t_1 < ... < t_K``, giving ``K - 1`` coefficients. Every
    basis function vanishes at ``u = 0`` because the knots are strictly
    positive, so ``response(0) = 0`` like every other family.

    ``knots`` are in **dose units** — write them where you would place them on
    a dose axis — and are divided by ``reference_dose`` internally. Three
    knots are the minimum (two coefficients: a slope and one bend); four to
    six is the usual range for a dose-response study, and more knots than
    distinct dose levels is a way to fit noise.

    A truncated-power basis is not the best-conditioned way to write a spline
    — a B-spline basis is — but it is exact, it is expressible in the node
    set, and at the handful of knots a dose-finding study supports the
    conditioning is not the binding constraint. ``surface.check_linearization``
    and ``design.identifiability_ridge`` will say if it becomes one.
    """

    name: Literal["spline"] = "spline"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None
    knots: tuple[float, ...] = (0.25, 0.5, 0.75)

    @model_validator(mode="after")
    def _valid(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior, BASIS_PRIOR_FAMILIES)
        _check_knots(self.knots, 3, self.reference_dose)
        return self

    @property
    def reduced_knots(self) -> tuple[float, ...]:
        """The knots on the ``u`` scale: ``knot / reference_dose``."""
        return tuple(t / self.reference_dose for t in self.knots)

    @property
    def stems(self) -> tuple[str, ...]:
        return tuple(f"beta{j}" for j in range(1, len(self.knots)))

    @property
    def roles(self) -> dict[str, KernelRole]:
        return dict.fromkeys(self.stems, "amplitude")

    @property
    def saturating(self) -> bool:
        return False

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        return tuple(
            _coefficient_param(
                stem, treatment, outcome_dimension, self.amplitude_scale, self.amplitude_prior
            )
            for stem in self.stems
        )

    def _d(self, u: Expr, index: int, power: int) -> Expr:
        """``d_k`` at ``power = 3``, or ``3 · d d_k / du`` at ``power = 2``."""
        t = self.reduced_knots
        last = t[-1]
        scale = last - t[index]
        left = Pow(base=_hinge(u, t[index]), exponent=Fraction(power))
        right = Pow(base=_hinge(u, last), exponent=Fraction(power))
        return Div(
            numerator=Add(terms=(left, Mul(factors=(_num(-1.0), right)))),
            denominator=_num(scale),
        )

    def basis(self, dose: Expr) -> tuple[Expr, ...]:
        """``(u, B_2, ..., B_{K-1})`` — the natural cubic basis, constant dropped."""
        u = _reduced(dose, self.reference_dose)
        out: list[Expr] = [u]
        penultimate = len(self.knots) - 2
        for k in range(len(self.knots) - 2):
            out.append(
                Add(
                    terms=(
                        self._d(u, k, 3),
                        Mul(factors=(_num(-1.0), self._d(u, penultimate, 3))),
                    )
                )
            )
        return tuple(out)

    def basis_derivative(self, dose: Expr) -> tuple[Expr, ...]:
        """``d basis / d dose``; the cubic terms differentiate to ``3 (u-t)_+^2``."""
        u = _reduced(dose, self.reference_dose)
        slope = _inverse_reference(dose, self.reference_dose)
        out: list[Expr] = [slope]
        penultimate = len(self.knots) - 2
        for k in range(len(self.knots) - 2):
            inner = Add(
                terms=(self._d(u, k, 2), Mul(factors=(_num(-1.0), self._d(u, penultimate, 2))))
            )
            out.append(Mul(factors=(_num(3.0), inner, slope)))
        return tuple(out)

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        """``u`` — the identity shape; see the module docstring on the basis families."""
        return _reduced(dose, self.reference_dose)

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        return _inverse_reference(dose, self.reference_dose)

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        betas = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _combine(betas, self.basis(dose))

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        betas = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _combine(betas, self.basis_derivative(dose))


class PiecewiseLinearKernel(Spec):
    """A **linear spline** in ``u = dose / reference_dose``: a broken stick on fixed knots.

    ``response = beta_1 · u + Σ_{j>=1} beta_{j+1} · (u - t_j)_+``. The first
    coefficient is the initial slope and each later one is the **change** in
    slope at its knot, so the fitted parameters read directly as "the response
    per unit dose was this, and at 20 mg it changed by that" — the least
    interpretive of the non-monotone families, and the one whose derivative is
    a step function a titration decision can act on.

    Linear in its coefficients, signed, ``response(0) = 0``, not saturating,
    and — unlike the polynomial — its extrapolation is the last segment's
    slope rather than a runaway power. What it gives up is smoothness: the
    derivative jumps at each knot, and a marginal-effect plot will show it.
    Prefer ``SplineKernel`` when the underlying response is believed smooth
    and this one when the interpretation matters more than the curvature.
    """

    name: Literal["piecewise_linear"] = "piecewise_linear"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None
    knots: tuple[float, ...] = (0.5,)

    @model_validator(mode="after")
    def _valid(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior, BASIS_PRIOR_FAMILIES)
        _check_knots(self.knots, 1, self.reference_dose)
        return self

    @property
    def reduced_knots(self) -> tuple[float, ...]:
        """The knots on the ``u`` scale: ``knot / reference_dose``."""
        return tuple(t / self.reference_dose for t in self.knots)

    @property
    def stems(self) -> tuple[str, ...]:
        """``beta1`` is the initial slope; ``beta{j+1}`` is the slope change at knot ``j``."""
        return tuple(f"beta{j}" for j in range(1, len(self.knots) + 2))

    @property
    def roles(self) -> dict[str, KernelRole]:
        return dict.fromkeys(self.stems, "amplitude")

    @property
    def saturating(self) -> bool:
        return False

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        return tuple(
            _coefficient_param(
                stem, treatment, outcome_dimension, self.amplitude_scale, self.amplitude_prior
            )
            for stem in self.stems
        )

    def basis(self, dose: Expr) -> tuple[Expr, ...]:
        """``(u, (u - t_1)_+, ..., (u - t_K)_+)``."""
        u = _reduced(dose, self.reference_dose)
        return (u, *(_hinge(u, t) for t in self.reduced_knots))

    def basis_derivative(self, dose: Expr) -> tuple[Expr, ...]:
        """``(1, step(u - t_1), ..., step(u - t_K)) / reference_dose``."""
        u = _reduced(dose, self.reference_dose)
        slope = _inverse_reference(dose, self.reference_dose)
        return (
            slope,
            *(Mul(factors=(_hinge_step(u, t), slope)) for t in self.reduced_knots),
        )

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        """``u`` — the identity shape; see the module docstring on the basis families."""
        return _reduced(dose, self.reference_dose)

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        return _inverse_reference(dose, self.reference_dose)

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        betas = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _combine(betas, self.basis(dose))

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        betas = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _combine(betas, self.basis_derivative(dose))


# -- Gaussian process --------------------------------------------------------------------

CovarianceFamily = Literal["squared_exponential", "matern32", "matern52"]
"""The stationary covariance a ``GaussianProcessKernel`` approximates."""


def _spectral_density(family: CovarianceFamily, omega: float, lengthscale: Expr) -> Expr:
    """``sqrt(S(omega))`` for the unit-amplitude 1-D spectral density, as a tree in ``ell``.

    The square root is taken symbolically so the coefficient of each basis
    function is ``beta · sqrt(S(w_j)) · z_j`` with ``z_j ~ N(0, 1)`` — the
    non-centred parameterization, which is the one that samples.
    """
    w2 = _num(float(omega) ** 2)
    if family == "squared_exponential":
        # S(w) = sqrt(2 pi) l exp(-l^2 w^2 / 2), so
        # sqrt(S) = (2 pi)^(1/4) l^(1/2) exp(-l^2 w^2 / 4)
        decay = Apply(
            fn="exp",
            arg=Mul(
                factors=(
                    _num(-0.25),
                    w2,
                    Pow(base=lengthscale, exponent=Fraction(2)),
                )
            ),
        )
        return Mul(
            factors=(
                _num(float((2.0 * math.pi) ** 0.25)),
                Pow(base=lengthscale, exponent=Fraction(1, 2)),
                decay,
            )
        )
    # Matern: S(w) = a(nu) l^-(2 nu) (b(nu)/l^2 + w^2)^-(nu + 1/2), and sqrt halves the exponent.
    nu2, power = (3.0, Fraction(-1)) if family == "matern32" else (5.0, Fraction(-3, 2))
    constant = (4.0 * 3.0**1.5 if family == "matern32" else (16.0 / 3.0) * 5.0**2.5) ** 0.5
    inverse_sq = Pow(base=lengthscale, exponent=Fraction(-2))
    inner = Add(terms=(Mul(factors=(_num(nu2), inverse_sq)), w2))
    return Mul(
        factors=(
            _num(float(constant)),
            Pow(
                base=lengthscale,
                exponent=Fraction(-3, 2) if family == "matern32" else Fraction(-5, 2),
            ),
            Pow(base=inner, exponent=power),
        )
    )


class GaussianProcessKernel(Spec):
    """A Gaussian process over the dose axis, as its Hilbert-space basis approximation.

    The family for "I do not want to commit to a shape at all". Where
    ``HillKernel`` assumes saturation and ``SplineKernel`` assumes a
    particular knot set, this puts a **stationary GP prior** on the
    dose-response and lets the smoothness be estimated: a lengthscale ``ell``
    says how fast the curve may wiggle and an amplitude ``beta`` says how far
    it may travel, and the data pick both.

    **It is an approximation, and the docstring is where that is said.** The
    exact GP would need a multivariate normal over the latent function, which
    is not what ``core.Likelihood`` evaluates. Instead this is the
    Hilbert-space reduced-rank construction (Solin & Särkkä 2020; Riutort-Mayol
    et al. 2023): on a bounded interval the Laplacian eigenfunctions are a
    fixed sine basis, and a stationary GP is recovered by giving basis ``j``
    the prior standard deviation ``sqrt(S(w_j))``, where ``S`` is the
    covariance's spectral density and ``w_j`` the ``j``-th eigenvalue's root::

        f(u) = beta · Σ_j sqrt(S(w_j; ell)) · z_j · [ phi_j(u) − phi_j(0) ],
        phi_j(u) = sin(w_j (u + L)) / sqrt(L),   w_j = pi j / (2 L),   z_j ~ N(0, 1)

    written non-centred so it samples. Subtracting ``phi_j(0)`` is what keeps
    ``response(0) = 0`` like every other family, so the surface's intercept
    still means the response at zero dose rather than competing with the GP
    for it.

    **The two numbers that decide whether it is a GP at all** are ``n_basis``
    (how many eigenfunctions) and ``boundary_factor`` (how far past the data
    the domain is padded, ``L = boundary_factor / 2`` on the ``u`` scale). Too
    few basis functions and short lengthscales cannot be represented; too
    small a boundary and the sine basis's periodicity leaks into the fit.
    :meth:`covariance_error` measures the approximation directly — it compares
    the implied covariance against the exact one and returns the worst
    absolute discrepancy — and :meth:`sufficient_for` turns that into a yes or
    no for a lengthscale you have in mind.

    The two are not interchangeable: ``boundary_factor`` sets the **long**
    lengthscale limit and ``n_basis`` the **short** one, and raising the
    boundary costs basis functions to hold the short end. The defaults
    (``n_basis=24``, ``boundary_factor=3.0``) keep the covariance error under
    0.02 for lengthscales from **0.10 to 0.70** on the ``u`` scale — a tenth
    to seven tenths of ``reference_dose`` — which is the 90 % interval of the
    default lengthscale prior. Outside that band the approximation, not the
    data, is what limits the fit, and ``covariance_error`` says so rather than
    leaving it to be inferred from a bad posterior.

    Like the saturating families and unlike the basis families, this declares
    exactly **one amplitude**: the sign of the response comes from the ``z``
    coefficients, so ``beta`` is a positive GP marginal scale and
    ``response == beta · saturation`` holds node for node.
    """

    name: Literal["gaussian_process"] = "gaussian_process"
    reference_dose: float = Field(default=1.0, gt=0)
    amplitude_scale: float = Field(default=1.0, gt=0)
    amplitude_prior: Prior | None = None
    n_basis: int = Field(default=24, ge=2, le=128)
    boundary_factor: float = Field(default=3.0, gt=1.0)
    covariance: CovarianceFamily = "squared_exponential"
    lengthscale_median: float = Field(default=0.3, gt=0)
    lengthscale_spread: float = Field(default=0.5, gt=0)

    @model_validator(mode="after")
    def _positive_amplitude_prior(self) -> Self:
        _check_amplitude_prior(self.amplitude_prior)
        return self

    @property
    def half_width(self) -> float:
        """``S``: half the width of the ``u`` domain the basis is laid out on."""
        return 0.5

    @property
    def boundary(self) -> float:
        """``L = boundary_factor · S`` — the padded half-domain of the eigenbasis."""
        return self.boundary_factor * self.half_width

    @property
    def frequencies(self) -> tuple[float, ...]:
        """``w_j = pi j / (2 L)`` for ``j = 1 … n_basis``."""
        return tuple(math.pi * j / (2.0 * self.boundary) for j in range(1, self.n_basis + 1))

    @property
    def stems(self) -> tuple[str, ...]:
        return ("ell", *(f"z{j}" for j in range(1, self.n_basis + 1)), "beta")

    @property
    def roles(self) -> dict[str, KernelRole]:
        out: dict[str, KernelRole] = {"ell": "shape"}
        for j in range(1, self.n_basis + 1):
            out[f"z{j}"] = "shape"
        out["beta"] = "amplitude"
        return out

    @property
    def saturating(self) -> bool:
        return False

    def parameters(
        self, treatment: str, dose_dimension: Dimension, outcome_dimension: Dimension
    ) -> tuple[Param, ...]:
        lengthscale = Param(
            name=f"ell_{treatment}",
            dimension=dimensionless(),
            prior=Prior(
                family="lognormal",
                hyper={
                    "mu": math.log(self.lengthscale_median),
                    "sigma": float(self.lengthscale_spread),
                },
            ),
        )
        weights = tuple(
            Param(
                name=f"z{j}_{treatment}",
                dimension=dimensionless(),
                prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0}),
            )
            for j in range(1, self.n_basis + 1)
        )
        amplitude = _amplitude_param(
            "beta", treatment, outcome_dimension, self.amplitude_scale, self.amplitude_prior
        )
        return (lengthscale, *weights, amplitude)

    def _shifted(self, dose: Expr) -> Expr:
        """``u − S``: the reduced dose recentred on the basis domain, so zero dose is ``−S``."""
        return Add(terms=(_reduced(dose, self.reference_dose), _num(-self.half_width)))

    def basis(self, dose: Expr) -> tuple[Expr, ...]:
        """``phi_j(u) − phi_j(0)``, the centred eigenfunctions."""
        centred = self._shifted(dose)
        scale = _num(1.0 / math.sqrt(self.boundary))
        out: list[Expr] = []
        for omega in self.frequencies:
            shifted = Mul(factors=(_num(omega), Add(terms=(centred, _num(self.boundary)))))
            at_zero = math.sin(omega * (self.boundary - self.half_width)) / math.sqrt(self.boundary)
            out.append(
                Add(
                    terms=(
                        Mul(factors=(scale, Apply(fn="sin", arg=shifted))),
                        _num(-at_zero),
                    )
                )
            )
        return tuple(out)

    def basis_derivative(self, dose: Expr) -> tuple[Expr, ...]:
        """``d phi_j / d dose = w_j cos(w_j (u − S + L)) / (sqrt(L) · reference_dose)``."""
        centred = self._shifted(dose)
        slope = _inverse_reference(dose, self.reference_dose)
        out: list[Expr] = []
        for omega in self.frequencies:
            shifted = Mul(factors=(_num(omega), Add(terms=(centred, _num(self.boundary)))))
            out.append(
                Mul(
                    factors=(
                        _num(omega / math.sqrt(self.boundary)),
                        Apply(fn="cos", arg=shifted),
                        slope,
                    )
                )
            )
        return tuple(out)

    def _weighted(self, treatment: str, basis: Sequence[Expr]) -> Expr:
        """``Σ_j sqrt(S(w_j; ell)) · z_j · basis_j`` — dimensionless."""
        lengthscale, *rest = self.parameters(treatment, dimensionless(), D.outcome)
        weights = rest[: self.n_basis]
        terms = tuple(
            Mul(
                factors=(
                    _spectral_density(self.covariance, omega, lengthscale),
                    z,
                    b,
                )
            )
            for omega, z, b in zip(self.frequencies, weights, basis, strict=True)
        )
        return Add(terms=terms) if len(terms) > 1 else terms[0]

    def saturation(self, dose: Expr, treatment: str) -> Expr:
        return self._weighted(treatment, self.basis(dose))

    def saturation_derivative(self, dose: Expr, treatment: str) -> Expr:
        return self._weighted(treatment, self.basis_derivative(dose))

    def response(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        *_, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation(dose, treatment))

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        *_, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return _scaled(beta, self.saturation_derivative(dose, treatment))

    # -- how good is the approximation? ---------------------------------------------

    def implied_covariance(self, lengthscale: float, u: Array) -> Array:
        """``Σ_j S(w_j) phi_j(u) phi_j(u')`` — the covariance the basis actually implies."""
        from axiom.core import value as _value

        grid = np.asarray(u, dtype=np.float64) - self.half_width
        omegas = np.asarray(self.frequencies, dtype=np.float64)
        phi = np.sin(omegas[:, None] * (grid[None, :] + self.boundary)) / math.sqrt(self.boundary)
        ell = Param(name="ell", dimension=dimensionless())
        density = np.asarray(
            [
                float(
                    _value(
                        _spectral_density(self.covariance, float(w), ell),
                        params={"ell": float(lengthscale)},
                    )
                )
                ** 2
                for w in omegas
            ]
        )
        return np.asarray(phi.T @ (density[:, None] * phi), dtype=np.float64)

    def exact_covariance(self, lengthscale: float, u: Array) -> Array:
        """The stationary covariance the basis is approximating, on the same grid."""
        grid = np.asarray(u, dtype=np.float64)
        r = np.abs(grid[:, None] - grid[None, :])
        if self.covariance == "squared_exponential":
            return np.asarray(np.exp(-0.5 * (r / lengthscale) ** 2), dtype=np.float64)
        if self.covariance == "matern32":
            a = math.sqrt(3.0) * r / lengthscale
            return np.asarray((1.0 + a) * np.exp(-a), dtype=np.float64)
        a = math.sqrt(5.0) * r / lengthscale
        return np.asarray((1.0 + a + a**2 / 3.0) * np.exp(-a), dtype=np.float64)

    def covariance_error(self, lengthscale: float, *, n_grid: int = 41) -> float:
        """Worst absolute gap between the implied and exact covariance, on ``u`` in ``[0, 1]``.

        The honest measure of whether ``n_basis`` and ``boundary_factor`` are
        large enough for a lengthscale: the covariance is on a unit scale, so
        this reads as a fraction. Below about 0.02 the approximation is not
        the thing limiting the fit.
        """
        if lengthscale <= 0.0:
            raise ValueError(f"lengthscale must be positive, got {lengthscale}")
        u = np.linspace(0.0, 1.0, int(n_grid))
        return float(
            np.max(
                np.abs(
                    self.implied_covariance(lengthscale, u) - self.exact_covariance(lengthscale, u)
                )
            )
        )

    def sufficient_for(self, lengthscale: float, *, tolerance: float = 0.02) -> bool:
        """Is this basis big enough to represent a GP of that lengthscale?"""
        return self.covariance_error(lengthscale) <= tolerance


AnyKernel = Annotated[
    HillKernel
    | LogisticKernel
    | ExponentialKernel
    | PowerKernel
    | LinearKernel
    | PolynomialKernel
    | SplineKernel
    | PiecewiseLinearKernel
    | GaussianProcessKernel,
    Field(discriminator="name"),
]
"""The shipped families as a discriminated union on ``name``, for embedding in other specs."""

KERNELS: dict[str, type[Spec]] = {
    "hill": HillKernel,
    "logistic": LogisticKernel,
    "exponential": ExponentialKernel,
    "power": PowerKernel,
    "linear": LinearKernel,
    "polynomial": PolynomialKernel,
    "spline": SplineKernel,
    "piecewise_linear": PiecewiseLinearKernel,
    "gaussian_process": GaussianProcessKernel,
}
"""Family name to kernel class. Every entry satisfies ``ResponseKernel``."""


def kernel_from_name(name: str, **fields: Any) -> ResponseKernel:
    """Build a kernel by family name, e.g. ``kernel_from_name("hill", reference_dose=50.0)``."""
    if name not in KERNELS:
        raise ValueError(f"unknown kernel family {name!r}; known: {sorted(KERNELS)}")
    kernel = KERNELS[name](**fields)
    if not isinstance(kernel, ResponseKernel):  # pragma: no cover - registry invariant
        raise TypeError(f"{type(kernel).__name__} does not satisfy ResponseKernel")
    return kernel
