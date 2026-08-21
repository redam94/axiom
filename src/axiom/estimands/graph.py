"""The estimand as an expression over the model's mean tree.

``estimand_expr`` builds, from a mean expression and the ``Data`` node that
carries the treatment's dose, the ``axiom.core`` expression whose value is
the estimand *per row of the panel*: a contrast is ``mean(at iv) − mean(at
ref)``; a ratio divides that by the dose difference; a marginal is the
caller-supplied closed-form derivative at ``iv``; an elasticity is
``derivative · dose / mean``. The arms are made by ``substitute``, which
rebuilds the tree with the treatment column replaced by the intervened dose
— a ``Const`` for ``mode="set"``, ``dose · factor`` for ``"scale"``,
``dose + shift`` for ``"shift"``.

Because it *is* an expression, the result goes through the same three
interpreters as the model: ``dimension`` proves gate 10 ("every declared
estimand's expression derives to its declared dimension",
``check_estimand_dimension``), ``value`` evaluates it over posterior draws,
``latex`` renders it.

Every functional is defined over the aggregated outcome and dose
(decision 0002.20; the ``Quantity`` docstring is the reference). A per-row
expression can stand in for the aggregated functional only when aggregating
its rows gives that functional: a contrast sums, a marginal sums, a ratio
averages *provided the dose difference is the same in every row*. What
cannot be expressed that way is returned as a typed ``Unsupported``, never
approximated:

* an ``area`` (quadrature is not an expression);
* a marginal without a supplied derivative (there is no ``Deriv`` node,
  review C2; kernels ship ``ResponseKernel.derivative``);
* a log-scale quantity and a conditional estimand;
* a ratio whose arms differ in mode, or both scale the observed dose: the
  per-row ``Δmean/Δdose`` there is not ``Δagg(Y)/Δagg(X)``;
* an intervention with a support ``window``: a dose applied over part of
  the horizon is a time path, not a per-row substitution;
* a ``set`` dose or a pinned covariate that feeds a ``Convolve`` (signal or
  kernel), a ``Reduce``, or an ``Opaque`` — a scalar there collapses the
  time axis (a factor-of-``T`` error, or a length-one kernel), so the arm
  is a time path too.

A negative ``set`` level or ``scale`` factor is a ``ValueError``, as it is
on the producer path; a ``shift`` is only negative relative to the observed
dose, which the tree does not hold, so it is checked by ``predict_under``.

All of those are realized by ``predict_under`` in ``estimands.evaluate``,
which holds the observed dose arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction

from axiom.core.dimensions import Dimension, DimensionError, dimensionless
from axiom.core.entities import Intervention
from axiom.core.expr import (
    Add,
    Apply,
    Const,
    Convolve,
    Data,
    Div,
    Expr,
    Gather,
    Link,
    Mul,
    Opaque,
    Param,
    Pow,
    Reduce,
    data_names,
    walk,
)
from axiom.core.interpret.dimension import dimension
from axiom.core.result import Unsupported
from axiom.estimands.spec import Estimand

__all__ = ["check_estimand_dimension", "estimand_expr", "substitute"]

_REALIZE = "realize with predict_under (estimands.evaluate.realize)"


# -- substitution -------------------------------------------------------------------------


def substitute(expr: Expr, replacements: Mapping[str, Expr]) -> Expr:
    """Rebuild ``expr`` with every ``Data`` node named in ``replacements`` swapped out.

    Unnamed subtrees are returned as the same objects. A ``Gather`` index
    column can only be replaced by another ``Data`` node (it must stay an
    integer column); anything else raises ``TypeError``.
    """
    if not replacements:
        return expr
    return _Substitution(replacements).go(expr)


class _Substitution:
    def __init__(self, replacements: Mapping[str, Expr]) -> None:
        self.rep = replacements

    def go(self, node: Expr) -> Expr:
        match node:
            case Data():
                return self.rep[node.name] if node.name in self.rep else node
            case Const() | Param():
                return node
            case Add():
                terms = tuple(self.go(t) for t in node.terms)
                return (
                    node
                    if all(a is b for a, b in zip(terms, node.terms, strict=True))
                    else Add(terms=terms)
                )
            case Mul():
                factors = tuple(self.go(f) for f in node.factors)
                return (
                    node
                    if all(a is b for a, b in zip(factors, node.factors, strict=True))
                    else Mul(factors=factors)
                )
            case Div():
                num, den = self.go(node.numerator), self.go(node.denominator)
                if num is node.numerator and den is node.denominator:
                    return node
                return Div(numerator=num, denominator=den)
            case Pow():
                base = self.go(node.base)
                exponent: Expr | Fraction = (
                    node.exponent if isinstance(node.exponent, Fraction) else self.go(node.exponent)
                )
                if base is node.base and exponent is node.exponent:
                    return node
                return Pow(base=base, exponent=exponent)
            case Apply():
                arg = self.go(node.arg)
                return node if arg is node.arg else Apply(fn=node.fn, arg=arg)
            case Link():
                arg = self.go(node.arg)
                return node if arg is node.arg else Link(fn=node.fn, arg=arg)
            case Reduce():
                arg = self.go(node.arg)
                if arg is node.arg:
                    return node
                return Reduce(op=node.op, arg=arg, keepdims=node.keepdims)
            case Convolve():
                signal, kernel = self.go(node.signal), self.go(node.kernel)
                if signal is node.signal and kernel is node.kernel:
                    return node
                return Convolve(signal=signal, kernel=kernel)
            case Gather():
                source = self.go(node.source)
                index: Data = node.index
                if node.index.name in self.rep:
                    new_index = self.rep[node.index.name]
                    if not isinstance(new_index, Data):
                        raise TypeError(
                            f"Gather index {node.index.name!r} can only be replaced by a Data "
                            f"node, not {type(new_index).__name__}"
                        )
                    index = new_index
                if source is node.source and index is node.index:
                    return node
                return Gather(source=source, index=index)
            case Opaque():
                inputs = tuple(self.go(i) for i in node.inputs)
                if all(a is b for a, b in zip(inputs, node.inputs, strict=True)):
                    return node
                return Opaque(name=node.name, inputs=inputs, dimension=node.dimension)
        raise TypeError(f"not an expression node: {type(node).__name__}")  # pragma: no cover


# -- helpers ------------------------------------------------------------------------------


def _minus(a: Expr, b: Expr) -> Expr:
    """``a − b``; ``Apply("neg")`` needs a dimensionless argument, so multiply by −1."""
    return Add(terms=(a, Mul(factors=(Const(value=-1.0, dimension=dimensionless()), b))))


def _dose_expr(column: Data, iv: Intervention, treatment: str, estimand: str) -> Expr:
    """The dose the treatment column takes under ``iv``: a level, a multiple, or a shift.

    A negative ``set`` level or ``scale`` factor realizes a negative dose in
    every row and is a ``ValueError`` here, as it is on the producer path
    (``surface.forward.predict_under``): the kernels are not defined below
    zero and would evaluate to ``nan`` silently. A ``shift`` is only negative
    relative to the observed dose, which this expression does not hold, so it
    cannot be checked statically; ``predict_under`` checks the realized grid.
    """
    level = float(iv.doses[treatment])
    if iv.mode in ("set", "scale") and level < 0.0:
        raise ValueError(
            f"estimand {estimand!r}: the {iv.mode} level {level:g} for {treatment!r} realizes a "
            "negative dose in every row; doses must be non-negative"
        )
    match iv.mode:
        case "set":
            return Const(value=level, dimension=column.dimension)
        case "scale":
            return Mul(factors=(column, Const(value=level, dimension=dimensionless())))
        case "shift":
            return Add(terms=(column, Const(value=level, dimension=column.dimension)))
    raise ValueError(f"unknown intervention mode {iv.mode!r}")  # pragma: no cover


def _constant_dose_difference(
    column: Data, iv: Intervention, ref: Intervention, treatment: str
) -> Const | None:
    """``dose(iv) − dose(ref)`` when it is the same in every row; ``None`` when it varies.

    The difference of the two arms' dose expressions is row-independent
    exactly when both arms ``set`` a level (``a − b``) or both ``shift`` the
    observed dose (``(x + a) − (x + b) = a − b``). Two ``scale`` arms give
    ``(a − b) · x`` and mixed modes give ``a − x·b``-shaped differences,
    both of which vary with the observed dose. When the arms share a mode
    and a level the difference is identically zero and that is a
    ``ValueError`` — a ratio over a zero dose interval is not a number.
    """
    a, b = float(iv.doses[treatment]), float(ref.doses[treatment])
    if iv.mode == ref.mode and a == b:
        raise ValueError(
            f"intervention and reference both {iv.mode} {treatment!r} to {a:g}; "
            "the dose difference is zero"
        )
    if iv.mode == ref.mode and iv.mode in ("set", "shift"):
        return Const(value=a - b, dimension=column.dimension)
    return None


def _column_dimensions(trees: Iterable[Expr], name: str) -> tuple[Dimension, ...]:
    """Every distinct dimension the ``Data`` column ``name`` is declared with across ``trees``."""
    out: list[Dimension] = []
    for tree in trees:
        for _, node in walk(tree):
            if isinstance(node, Data) and node.name == name and node.dimension not in out:
                out.append(node.dimension)
    return tuple(out)


def _gather_indices(trees: Iterable[Expr]) -> set[str]:
    return {n.index.name for t in trees for _, n in walk(t) if isinstance(n, Gather)}


def _time_axis_hits(trees: Iterable[Expr], names: set[str]) -> tuple[str, ...]:
    """``"'col' feeds a Reduce"`` for every name in ``names`` reaching a node that acts along
    the last (time) axis: a ``Convolve`` signal or kernel, a ``Reduce`` argument, an ``Opaque``
    input. A kernel is a weight vector along that axis: a scalar in its place is a length-one
    kernel, which is a different convolution."""
    hits: set[tuple[str, str]] = set()
    for tree in trees:
        for _, node in walk(tree):
            subtrees: tuple[Expr, ...]
            match node:
                case Convolve():
                    subtrees = (node.signal, node.kernel)
                case Reduce():
                    subtrees = (node.arg,)
                case Opaque():
                    subtrees = node.inputs
                case _:
                    continue
            for sub in subtrees:
                for col in names & set(data_names(sub)):
                    hits.add((col, type(node).__name__))
    return tuple(f"{col!r} feeds a {kind}" for col, kind in sorted(hits))


# -- the estimand as an expression ---------------------------------------------------------


def estimand_expr(
    estimand: Estimand,
    *,
    mean: Expr,
    treatment_data: Data,
    reference: Mapping[str, float] | None = None,
    derivative: Expr | None = None,
) -> Expr | Unsupported:
    """The estimand as a per-row expression over ``mean``, or a typed ``Unsupported``.

    ``treatment_data`` is the ``Data`` node in ``mean`` that carries the
    estimand's treatment dose; its dimension must equal the treatment's and
    the one the tree declares for that column. ``mean`` must derive to the
    estimand's outcome dimension and ``derivative`` — ``d mean / d dose`` as
    an expression over the same nodes (``ResponseKernel.derivative`` builds
    one), required for ``marginal`` and ``elasticity`` — to outcome per
    dose; either mismatch is a ``DimensionError`` naming both sides.
    ``reference`` pins *other* data columns (covariates, other treatments)
    to constant values in every arm; names not in the tree, the treatment
    column, and a ``Gather`` index column raise ``ValueError``.

    The result is one value per row of the data the tree is evaluated on;
    the window/level/population aggregation is ``estimands.evaluate``'s.
    Under 0002.20 that aggregation recovers the declared functional for a
    contrast and a marginal (sum the rows) and for a ratio whose dose
    difference is row-constant (average the rows); a ratio with two
    ``scale`` arms or mixed modes is returned as ``Unsupported``. A per-row
    elasticity ``g · dose / mean`` equals the windowed elasticity only
    where the mean is the same in every row of the window (one arm, a
    steady state); over a varying panel use ``evaluate.realize``. The
    intervention's ``version`` is carried and not interpreted; its
    ``window`` (a support inside the horizon) is not expressible here and
    is ``Unsupported`` (see the module docstring for the full list).
    """
    kind = estimand.quantity.kind
    name = estimand.name
    detail = {"estimand": name, "quantity": kind}
    if kind == "area":
        return Unsupported(
            reason=(
                f"estimand {name!r} is an area under the response: quadrature is not an "
                "expression; realize it with estimands.evaluate"
            ),
            detail=detail,
            missing=("quadrature",),
        )
    if estimand.quantity.scale != "natural":
        return Unsupported(
            reason=(
                f"estimand {name!r} is declared on the {estimand.quantity.scale!r} scale; "
                "only natural-scale quantities are expressible over the mean tree"
            ),
            detail=detail,
        )
    if estimand.conditioning:
        return Unsupported(
            reason=(
                f"estimand {name!r} conditions on {list(estimand.conditioning)}; a conditional "
                "estimand restricts units and is not an expression over the mean tree"
            ),
            detail=detail,
        )

    # -- a malformed request raises -------------------------------------------------------
    treatment = estimand.treatment.name
    column = treatment_data.name
    trees: dict[str, Expr] = {"mean": mean}
    if derivative is not None:
        trees["derivative"] = derivative
    if column not in data_names(mean):
        raise ValueError(
            f"treatment column {column!r} does not appear in the mean tree; "
            f"its columns are {list(data_names(mean))}"
        )
    if estimand.treatment.dimension != treatment_data.dimension:
        raise DimensionError(
            f"estimand {name!r}: treatment {treatment!r} has dimension "
            f"{estimand.treatment.dimension} but column {column!r} carries "
            f"{treatment_data.dimension}"
        )
    for label, tree in trees.items():
        for declared in _column_dimensions((tree,), column):
            if declared != treatment_data.dimension:
                raise DimensionError(
                    f"estimand {name!r}: the {label} tree declares column {column!r} with "
                    f"dimension {declared} but treatment_data carries {treatment_data.dimension}"
                )
    outcome_dim = estimand.outcome.dimension
    assert outcome_dim is not None  # the Estimand validator guarantees it
    mean_dim = dimension(mean)
    if mean_dim != outcome_dim:
        raise DimensionError(
            f"estimand {name!r}: the mean tree derives to {mean_dim} but outcome "
            f"{estimand.outcome.name!r} has dimension {outcome_dim}"
        )
    if derivative is not None:
        want = outcome_dim / treatment_data.dimension
        got = dimension(derivative)
        if got != want:
            raise DimensionError(
                f"estimand {name!r}: the derivative tree derives to {got} but "
                f"d {estimand.outcome.name} / d {treatment} has dimension {want}"
            )
    pinned: dict[str, Expr] = {}
    for col, v in (reference or {}).items():
        per_tree = {label: _column_dimensions((tree,), col) for label, tree in trees.items()}
        dims = _column_dimensions(trees.values(), col)
        if not dims:
            raise ValueError(
                f"reference= pins {col!r}, which does not appear in the tree; columns are "
                f"{sorted({c for t in trees.values() for c in data_names(t)})}"
            )
        if len(dims) > 1:
            # One Const stands in for the column in every tree, so the trees must agree on
            # its dimension; pinning with either one would fail dimension() deep inside the
            # other tree, naming an internal node instead of the column.
            where = "; ".join(
                f"the {label} tree declares it with {' and '.join(str(d) for d in ds)}"
                for label, ds in per_tree.items()
                if ds
            )
            raise DimensionError(
                f"estimand {name!r}: reference= pins {col!r}, whose dimension differs between "
                f"the trees — {where}; a pinned column must carry one dimension everywhere"
            )
        pinned[col] = Const(value=float(v), dimension=dims[0])
    if column in pinned:
        raise ValueError(
            f"reference= pins the treatment column {column!r}; the treatment's "
            "dose comes from the estimand's intervention"
        )
    indexed = sorted((set(pinned) | {column}) & _gather_indices(trees.values()))
    if indexed:
        raise ValueError(
            f"estimand {name!r}: {indexed} index a Gather (a unit-level lookup) and cannot "
            "be set to a constant dose or pinned through reference="
        )

    # -- what is not an expression is Unsupported -----------------------------------------
    arms: dict[str, Intervention] = {"intervention": estimand.intervention}
    if estimand.reference is not None:
        arms["reference"] = estimand.reference
    extra = sorted({t for iv in arms.values() for t in iv.doses} - {treatment})
    if extra:
        return Unsupported(
            reason=(
                f"estimand {name!r} also sets {extra}; estimand_expr substitutes one treatment "
                "column — pin the others through reference= or realize with estimands.evaluate"
            ),
            detail=detail,
        )
    if derivative is None and kind in ("marginal", "elasticity"):
        return Unsupported(
            reason=(
                f"estimand {name!r} is a {kind}: no closed-form derivative supplied; "
                "kernels provide one via ResponseKernel.derivative"
            ),
            detail=detail,
            missing=("derivative",),
        )
    windowed = [label for label, iv in arms.items() if iv.window is not None]
    if windowed:
        spans = ", ".join(
            f"{label} [{iv.window.start}, {iv.window.stop})"
            for label, iv in arms.items()
            if iv.window is not None
        )
        return Unsupported(
            reason=(
                f"estimand {name!r}: the {' and '.join(windowed)} has a support window "
                f"({spans}); a dose applied over part of the horizon is a time path, not a "
                f"per-row substitution — {_REALIZE}"
            ),
            detail=detail,
            missing=("time_window",),
        )
    scalar_columns = set(pinned)
    if any(iv.mode == "set" for iv in arms.values()):
        scalar_columns.add(column)
    hits = _time_axis_hits(trees.values(), scalar_columns)
    if hits:
        return Unsupported(
            reason=(
                f"estimand {name!r}: {'; '.join(hits)}; a constant there collapses the time "
                f"axis — a set dose or a pinned covariate is a time path, not a scalar — "
                f"{_REALIZE}"
            ),
            detail=detail,
            missing=("time_window",),
        )

    # -- build ------------------------------------------------------------------------------
    iv_dose = _dose_expr(treatment_data, estimand.intervention, treatment, name)
    at_iv = {**pinned, column: iv_dose}
    mean_iv = substitute(mean, at_iv)

    match kind:
        case "contrast" | "ratio":
            ref = estimand.reference
            assert ref is not None  # the Estimand validator guarantees it
            ref_dose = _dose_expr(treatment_data, ref, treatment, name)
            mean_ref = substitute(mean, {**pinned, column: ref_dose})
            contrast = _minus(mean_iv, mean_ref)
            if kind == "contrast":
                return contrast
            diff = _constant_dose_difference(treatment_data, estimand.intervention, ref, treatment)
            if diff is None:
                return Unsupported(
                    reason=(
                        f"estimand {name!r} is a ratio Δagg(Y)/Δagg(X) (0002.20); with a "
                        f"{estimand.intervention.mode!r} intervention against a {ref.mode!r} "
                        "reference the dose difference varies across rows, so a per-row "
                        f"Δmean/Δdose is not that functional — {_REALIZE}"
                    ),
                    detail={
                        **detail,
                        "intervention_mode": estimand.intervention.mode,
                        "reference_mode": ref.mode,
                    },
                    missing=("dose_path",),
                )
            return Div(numerator=contrast, denominator=diff)
        case "marginal":
            assert derivative is not None
            return substitute(derivative, at_iv)
        case "elasticity":
            assert derivative is not None
            slope = substitute(derivative, at_iv)
            return Div(numerator=Mul(factors=(slope, iv_dose)), denominator=mean_iv)
    raise ValueError(f"unknown quantity kind {kind!r}")  # pragma: no cover


def check_estimand_dimension(estimand: Estimand, expr: Expr) -> Dimension:
    """``dimension(expr)``, asserted equal to the estimand's declared dimension (gate 10)."""
    got = dimension(expr)
    if got != estimand.dimension:
        raise DimensionError(
            f"estimand {estimand.name!r} declares dimension {estimand.dimension} but its "
            f"expression derives to {got}"
        )
    return got
