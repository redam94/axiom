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
from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

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
    "KERNELS",
    "AnyKernel",
    "ExponentialKernel",
    "HillKernel",
    "KernelRole",
    "LinearKernel",
    "LogisticKernel",
    "PowerKernel",
    "ResponseKernel",
    "kernel_from_name",
]

KernelRole = Literal["scale", "shape", "amplitude"]
"""What a kernel parameter is: a dose-dimensioned scale, a dimensionless shape, or an amplitude."""

AMPLITUDE_PRIOR_FAMILIES: frozenset[str] = frozenset({"halfnormal", "lognormal", "gamma"})
"""Prior families with support on R+, the only ones an explicit ``amplitude_prior`` may use."""


def _check_amplitude_prior(prior: Prior | None) -> None:
    """An explicit amplitude prior must be positive-support and fully numeric (D6.2)."""
    if prior is None:
        return
    if prior.family not in AMPLITUDE_PRIOR_FAMILIES:
        raise ValueError(
            f"amplitude_prior must be one of {sorted(AMPLITUDE_PRIOR_FAMILIES)} "
            f"(positive support), got {prior.family!r}"
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

    For every family with a ``scale`` parameter, ``response == amplitude ·
    saturation`` holds node for node. ``LinearKernel`` is the documented
    exception: it has no scale, its ``saturation`` is the identity shape
    ``dose / reference_dose`` (a dimensionless convention, not a factor of
    the response), and its ``response`` is ``beta_rate · dose`` with
    ``beta_rate`` carrying ``outcome / dose``. Callers that decompose a
    response into amplitude and saturation must branch on ``name ==
    "linear"`` (or on ``"scale" not in roles.values()``).
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

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta · s · x̃^(s-1) · k̃^s / (c · (k̃^s + x̃^s)^2)``.

        Equal to ``beta · s · u^(s-1) / (k · (1 + u^s)^2)`` with ``u = dose / k``.
        """
        k, s, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        c, x_tilde, k_tilde = _tilde(dose, k)
        xs = Pow(base=x_tilde, exponent=s)
        ks = Pow(base=k_tilde, exponent=s)
        x_sm1 = Pow(base=x_tilde, exponent=Add(terms=(s, _num(-1.0))))
        numerator = Mul(factors=(beta, s, x_sm1, ks))
        denominator = Mul(factors=(c, Pow(base=Add(terms=(ks, xs)), exponent=Fraction(2))))
        return Div(numerator=numerator, denominator=denominator)


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

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta · 2 σ(u) (1 − σ(u)) / k``."""
        k, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        sig = Apply(fn="sigmoid", arg=_ratio(dose, k))
        df_du = Mul(factors=(_num(2.0), sig, Add(terms=(_one(), Apply(fn="neg", arg=sig)))))
        return _over_k(beta, k, df_du)


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

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta · exp(−u) / k``."""
        k, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        df_du = Apply(fn="exp", arg=Apply(fn="neg", arg=_ratio(dose, k)))
        return _over_k(beta, k, df_du)


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

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta · s · x̃^(s−1) / (k̃^s · c)`` = ``beta · s · u^(s−1) / k``."""
        k, s, beta = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        c, x_tilde, k_tilde = _tilde(dose, k)
        x_sm1 = Pow(base=x_tilde, exponent=Add(terms=(s, _num(-1.0))))
        numerator = Mul(factors=(beta, s, x_sm1))
        denominator = Mul(factors=(Pow(base=k_tilde, exponent=s), c))
        return Div(numerator=numerator, denominator=denominator)


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

    def derivative(
        self, dose: Expr, treatment: str, outcome_dimension: Dimension | None = None
    ) -> Expr:
        """``beta_rate`` — constant in the dose."""
        (beta_rate,) = self.parameters(treatment, dimension(dose), _outcome(outcome_dimension))
        return beta_rate


AnyKernel = Annotated[
    HillKernel | LogisticKernel | ExponentialKernel | PowerKernel | LinearKernel,
    Field(discriminator="name"),
]
"""The shipped families as a discriminated union on ``name``, for embedding in other specs."""

KERNELS: dict[str, type[Spec]] = {
    "hill": HillKernel,
    "logistic": LogisticKernel,
    "exponential": ExponentialKernel,
    "power": PowerKernel,
    "linear": LinearKernel,
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
