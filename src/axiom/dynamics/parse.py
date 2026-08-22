"""A small surface syntax for the equations, over declarations you write in Python.

The declarations — a variable's dimension, a parameter's prior — stay in
Python, where they are checked. Only the *equations* get a syntax, because
that is the part that is unreadable as a tree::

    quantity = alpha - beta * price + gamma * income
    price    = delta + epsilon * quantity + cost
    stock    = decay * stock[t-1] + inflow

Grammar (one equation per line; ``#`` starts a comment)::

    equation := reference '=' expr
    expr     := term (('+' | '-') term)*
    term     := unary (('*' | '/') unary)*
    unary    := ('-' | '+')? power
    power    := atom (('^' | '**') rational)?
    atom     := number | reference | name '(' expr ')' | '(' expr ')'
    reference:= name | name '[t]' | name '[t-' int ']' | name '.l' int

A name resolves to a parameter if one is declared with it and to a variable
otherwise; an unknown name is an error naming both lists. A number is
dimensionless — which is what makes ``2 * price`` legal and ``2 + price``
a dimension error, caught when the ``DynamicSystem`` is built rather than
here.

This is deliberately small. It has no distributions, no priors, no ``do``
operator, no plate notation: everything else in axiom is a ``Spec``, and a
second way to declare priors would be a second thing to keep in sync.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from fractions import Fraction

from axiom.core import Apply, ApplyFn, Const, Data, Div, Expr, Param, Pow, dimensionless
from axiom.dynamics.algebra import add, div, mul, neg, sub
from axiom.dynamics.spec import DynamicEquation, DynamicsError, DynamicSystem, Variable, lag_ref

__all__ = ["ParseError", "parse_equations", "parse_system"]

_FUNCTIONS: frozenset[str] = frozenset(
    (
        "exp",
        "log",
        "log1p",
        "expm1",
        "tanh",
        "sigmoid",
        "logit",
        "softplus",
        "neg",
        "relu",
        "step",
        "sin",
        "cos",
    )
)

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>\d+\.\d*([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?|\d+([eE][-+]?\d+)?)
  | (?P<name>[A-Za-z_]\w*(\.l\d+)?)
  | (?P<power>\*\*|\^)
  | (?P<op>[-+*/()\[\]=,])
    """,
    re.VERBOSE,
)


class ParseError(DynamicsError):
    """The equation text does not parse, or names something undeclared."""


class _Token:
    __slots__ = ("kind", "text", "line", "column")

    def __init__(self, kind: str, text: str, line: int, column: int) -> None:
        self.kind, self.text, self.line, self.column = kind, text, line, column

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.kind}:{self.text}"


def _tokenize(line: str, number: int) -> list[_Token]:
    out: list[_Token] = []
    position = 0
    while position < len(line):
        match = _TOKEN.match(line, position)
        if match is None:
            raise ParseError(
                f"line {number}: cannot read {line[position]!r} at column {position + 1}"
            )
        position = match.end()
        kind = match.lastgroup or ""
        if kind == "space":
            continue
        out.append(_Token(kind, match.group(), number, match.start() + 1))
    return out


class _Parser:
    """Recursive descent over one line's tokens, resolving names as it goes."""

    def __init__(
        self,
        tokens: Sequence[_Token],
        variables: Mapping[str, Variable],
        parameters: Mapping[str, Param],
        line: int,
    ) -> None:
        self.tokens = list(tokens)
        self.variables = variables
        self.parameters = parameters
        self.line = line
        self.position = 0

    # -- token handling ---------------------------------------------------------

    def peek(self) -> _Token | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> _Token:
        token = self.peek()
        if token is None:
            raise ParseError(f"line {self.line}: the equation ends too early")
        self.position += 1
        return token

    def accept(self, text: str) -> bool:
        token = self.peek()
        if token is not None and token.text == text:
            self.position += 1
            return True
        return False

    def expect(self, text: str) -> _Token:
        token = self.peek()
        if token is None or token.text != text:
            got = "the end of the line" if token is None else repr(token.text)
            raise ParseError(f"line {self.line}: expected {text!r}, found {got}")
        self.position += 1
        return token

    # -- grammar ----------------------------------------------------------------

    def expression(self) -> Expr:
        node = self.term()
        while True:
            if self.accept("+"):
                node = add(node, self.term())
            elif self.accept("-"):
                node = sub(node, self.term())
            else:
                return node

    def term(self) -> Expr:
        node = self.unary()
        while True:
            if self.accept("*"):
                node = mul(node, self.unary())
            elif self.accept("/"):
                node = div(node, self.unary())
            else:
                return node

    def unary(self) -> Expr:
        if self.accept("-"):
            return neg(self.unary())
        self.accept("+")
        return self.power()

    def power(self) -> Expr:
        base = self.atom()
        token = self.peek()
        if token is not None and token.kind == "power":
            self.take()
            return _power(base, self.unary(), self.line)
        return base

    def atom(self) -> Expr:
        token = self.take()
        if token.text == "(":
            inner = self.expression()
            self.expect(")")
            return inner
        if token.kind == "number":
            return Const(value=float(token.text), dimension=dimensionless())
        if token.kind != "name":
            raise ParseError(f"line {self.line}: expected a name or number, found {token.text!r}")
        following = self.peek()
        if following is not None and following.text == "(" and token.text in _FUNCTIONS:
            self.take()
            argument = self.expression()
            self.expect(")")
            fn: ApplyFn = token.text  # type: ignore[assignment]
            return Apply(fn=fn, arg=argument)
        return self.reference(token)

    def reference(self, token: _Token) -> Expr:
        name, _, suffix = token.text.partition(".")
        lag = int(suffix[1:]) if suffix else 0
        if self.peek() is not None and self.peek().text == "[":  # type: ignore[union-attr]
            if suffix:
                raise ParseError(
                    f"line {self.line}: {token.text!r} already names a lag; write "
                    f"'{name}[t-{lag}]' or '{token.text}', not both"
                )
            self.take()
            self.expect("t")
            if self.accept("-"):
                offset = self.take()
                if offset.kind != "number" or not offset.text.isdigit():
                    raise ParseError(
                        f"line {self.line}: a lag must be a whole number, found {offset.text!r}"
                    )
                lag = int(offset.text)
            elif self.peek() is not None and self.peek().text == "+":  # type: ignore[union-attr]
                raise ParseError(
                    f"line {self.line}: {name!r} is read at t+; a forward-looking equation "
                    "needs a rational-expectations solution, which axiom.dynamics does not do"
                )
            self.expect("]")
        if name in self.parameters:
            if lag:
                raise ParseError(
                    f"line {self.line}: {name!r} is a parameter and does not vary with time; "
                    "a time-varying coefficient is a variable"
                )
            return self.parameters[name]
        if name in self.variables:
            variable = self.variables[name]
            return Data(name=lag_ref(name, lag), dimension=variable.dimension)
        raise ParseError(
            f"line {self.line}: {name!r} is neither a declared variable "
            f"{sorted(self.variables)} nor a declared parameter {sorted(self.parameters)}"
        )


def _power(base: Expr, exponent: Expr, line: int) -> Expr:
    """``base ^ q`` for a rational constant ``q``; anything else is an error naming why."""
    rational = _as_rational(exponent)
    if rational is None:
        raise ParseError(
            f"line {line}: an exponent must be a rational constant (2, 0.5, (1/3)); "
            "raise to an expression only where both sides are dimensionless, which "
            "this syntax does not offer"
        )
    return Pow(base=base, exponent=rational)


def _as_rational(expr: Expr) -> Fraction | None:
    if isinstance(expr, Const) and not expr.is_vector:
        assert isinstance(expr.value, float)
        return Fraction(expr.value).limit_denominator(10_000)
    if isinstance(expr, Div):
        top, bottom = _as_rational(expr.numerator), _as_rational(expr.denominator)
        if top is not None and bottom is not None and bottom != 0:
            return top / bottom
    return None


def parse_equations(
    text: str,
    *,
    variables: Sequence[Variable],
    parameters: Sequence[Param] = (),
) -> tuple[DynamicEquation, ...]:
    """Parse ``target = expression`` lines against declared variables and parameters."""
    by_name = {v.name: v for v in variables}
    by_parameter = {p.name: p for p in parameters}
    clash = sorted(set(by_name) & set(by_parameter))
    if clash:
        raise ParseError(f"{clash} are declared as both a variable and a parameter")
    equations: list[DynamicEquation] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = _tokenize(line, number)
        if not any(t.text == "=" for t in tokens):
            raise ParseError(f"line {number}: an equation needs an '=': {line!r}")
        split = next(i for i, t in enumerate(tokens) if t.text == "=")
        head, tail = tokens[:split], tokens[split + 1 :]
        if not head or head[0].kind != "name":
            raise ParseError(f"line {number}: the left side must be a variable: {line!r}")
        target = head[0].text.partition(".")[0]
        if len(head) > 1 and head[1].text != "[":
            raise ParseError(f"line {number}: the left side must be a single variable: {line!r}")
        if "." in head[0].text or (len(head) > 1 and any(t.text == "-" for t in head)):
            raise ParseError(
                f"line {number}: the left side is the variable at t and takes no lag: {line!r}"
            )
        parser = _Parser(tail, by_name, by_parameter, number)
        rhs = parser.expression()
        if parser.peek() is not None:
            token = parser.peek()
            assert token is not None
            raise ParseError(f"line {number}: unexpected {token.text!r} after the expression")
        equations.append(DynamicEquation(target=target, rhs=rhs, name=f"line {number}"))
    if not equations:
        raise ParseError("no equations were given")
    return tuple(equations)


def parse_system(
    text: str,
    *,
    variables: Sequence[Variable],
    parameters: Sequence[Param] = (),
    name: str = "",
    description: str = "",
) -> DynamicSystem:
    """Parse equations and assemble the ``DynamicSystem``, dimension checks included."""
    return DynamicSystem(
        name=name,
        description=description,
        variables=tuple(variables),
        equations=parse_equations(text, variables=variables, parameters=parameters),
    )
