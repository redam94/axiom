"""Render an expression tree as LaTeX, for docs and ``viz``.

An ``Opaque`` node cannot be introspected; it renders as ``\\mathrm{name}(...)``
and ``latex_or_unsupported`` reports it as ``Unsupported`` when the caller
needs a faithful rendering.
"""

from __future__ import annotations

from fractions import Fraction

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
    Reduce,
    System,
    walk,
)
from axiom.core.result import Unsupported

__all__ = ["latex", "latex_or_unsupported"]

_FN = {
    "exp": "\\exp",
    "log": "\\log",
    "log1p": "\\operatorname{log1p}",
    "expm1": "\\operatorname{expm1}",
    "tanh": "\\tanh",
    "sigmoid": "\\sigma",
    "logit": "\\operatorname{logit}",
    "softplus": "\\operatorname{softplus}",
    "neg": "-",
    "relu": "\\operatorname{relu}",
    "step": "\\operatorname{step}",
}
_GREEK = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "rho",
    "sigma",
    "tau",
    "phi",
    "psi",
    "omega",
}


def _sym(name: str) -> str:
    head, _, sub = name.partition("_")
    h = f"\\{head}" if head in _GREEK else (head if len(head) == 1 else f"\\mathrm{{{head}}}")
    return f"{h}_{{{sub}}}" if sub else h


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _frac(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _paren(node: Model, s: str) -> str:
    return f"\\left({s}\\right)" if isinstance(node, Add) else s


def latex(model: Model) -> str:
    match model:
        case Const():
            if isinstance(model.value, tuple):
                shown = ", ".join(_num(v) for v in model.value[:4])
                return f"({shown}{', \\ldots' if len(model.value) > 4 else ''})"
            return _num(model.value)
        case Data() | Param():
            return _sym(model.name)
        case Add():
            return " + ".join(latex(t) for t in model.terms)
        case Mul():
            return " \\cdot ".join(_paren(f, latex(f)) for f in model.factors)
        case Div():
            return f"\\frac{{{latex(model.numerator)}}}{{{latex(model.denominator)}}}"
        case Pow():
            e = (
                _frac(model.exponent)
                if isinstance(model.exponent, Fraction)
                else latex(model.exponent)
            )
            b = latex(model.base)
            if isinstance(model.base, Div):
                b = f"\\left({b}\\right)"
            else:
                b = _paren(model.base, b)
            return f"{{{b}}}^{{{e}}}"
        case Apply():
            return f"{_FN[model.fn]}\\left({latex(model.arg)}\\right)"
        case Link():
            return (
                f"{model.fn}^{{-1}}\\left({latex(model.arg)}\\right)"
                if model.fn != "identity"
                else latex(model.arg)
            )
        case Gather():
            return f"{latex(model.source)}_{{[{_sym(model.index.name)}]}}"
        case Reduce():
            op = {"sum": "\\sum", "mean": "\\operatorname{mean}", "max": "\\max"}[model.op]
            return f"{op}\\left({latex(model.arg)}\\right)"
        case Convolve():
            return f"\\left({latex(model.kernel)}\\right) * \\left({latex(model.signal)}\\right)"
        case Opaque():
            args = ", ".join(latex(i) for i in model.inputs)
            return f"\\mathrm{{{model.name}}}\\left({args}\\right)"
        case Equation():
            return f"{latex(model.lhs)} = {latex(model.rhs)}"
        case System():
            rows = " \\\\ ".join(latex(eq) for eq in model.equations)
            return f"\\begin{{cases}} {rows} \\end{{cases}}"
        case ODESystem():
            t = _sym(model.time.name)
            rows = " \\\\ ".join(
                f"\\frac{{d {_sym(s.name)}}}{{d {t}}} = {latex(r)}"
                for s, r in zip(model.states, model.rhs, strict=True)
            )
            return f"\\begin{{cases}} {rows} \\end{{cases}}"
    raise TypeError(f"not an expression node: {type(model).__name__}")


def latex_or_unsupported(model: Model) -> str | Unsupported:
    """``latex``, unless the tree contains an ``Opaque`` node — then a typed refusal."""
    opaque = [n.name for _, n in walk(model) if isinstance(n, Opaque)]
    if opaque:
        return Unsupported(
            reason=f"tree contains opaque node(s) {opaque}; their body cannot be rendered",
            missing=("introspection",),
        )
    return latex(model)
