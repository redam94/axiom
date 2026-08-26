# 0024 — A model reads like the mathematics, and builds the same tree

*Kind: decision. Opened 2026-08-26. Status: implemented on `feature/expr-operators`.*

A Hill response written through the constructors is

```python
Mul(factors=(beta, Div(
    numerator=Pow(base=Div(numerator=dose, denominator=k), exponent=s),
    denominator=Add(terms=(one, Pow(base=Div(numerator=dose, denominator=k), exponent=s))),
)))
```

and written as mathematics it is `beta * (dose / k) ** s / (1 + (dose / k) ** s)`.
The second is the model; the first is a serialization of it that a reader has to
decode before they can check the algebra. `surface/kernels.py` had already
noticed: `_ratio`, `_scaled`, `_over_k`, `_combine`, and `_hinge` exist only to
give a name to a `Div` or a `Mul` so the call site stops reading like a
constructor.

## D24.1 — the operators are sugar, and nothing else

Every node class in the `Expr` union carries `+ - * / **`, unary `-`, the
reflected forms, and `[]`. Each builds *exactly* the node its constructor
builds: same spec, same `content_hash`, same dimension error in the same place.
Nothing in the codebase changed, no call site was rewritten, and the
constructors remain the canonical form the plan and the ledger describe.

Deliberately absent: `__eq__` and the comparisons. Structural equality is
load-bearing — `node_path`, the `params` dedup, `content_hash`, every golden
fixture — and there is no comparison node to build.

## D24.2 — where the methods live, given there is no node base class

`expr.py` promises there is no node base class and that shared behaviour lives
in functions; the style rule in CLAUDE.md rules out a mixin. Both hold: the
operators are module-level functions typed `self: Any`, aliased in each class
body.

```python
class Param(Spec):
    ...
    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
```

An alias in a class body is a method to Python *and* to mypy, which is what
distinguishes it from the obvious alternative — assigning the dunders after the
class statement, which runs identically and is invisible to `mypy --strict`, so
`beta * dose` would fail the type gate at every call site. A contract test
derives the union's members from `Expr` itself and asserts each carries all
twelve, so a fourteenth node that forgets the block fails in CI rather than in
a user's model.

## D24.3 — a bare number is dimensionless, and that is the point

`_lift` turns any real (numpy scalars included; `bool` is refused) into
`Const(value=..., dimension=dimensionless())`. There is no local information
that could give it another dimension, and a literal that silently borrowed its
neighbour's would be precisely the wrong-number class rule 4 exists to prevent.
So `beta * 2.0` works and `revenue - 100.0` raises a `DimensionError` naming
both terms; a literal that carries units is still written out as a `Const`.

Two consequences worth stating:

- **`Add` and `Mul` flatten, so associativity is canonicalized.** They are
  declared n-ary; `(a + b) + c`, `a + (b + c)`, and `Add(terms=(a, b, c))` are
  one spec with one hash. This is a real commitment, not an implementation
  detail: two authors who bracket a sum differently get the same content hash.
- **A bare literal `0` is the additive identity and is dropped.** `sum` starts
  from `0`, and a dimensionless zero left in the sum would fail the dimension
  check for every model whose terms carry units — so `sum(terms)` builds the
  tree a reader expects. An explicit `Const(value=0.0, ...)` is a term the
  author wrote and is never dropped.

## D24.4 — the two nodes with no natural operator

`-a` is `(-1) · a`, not `Apply(fn="neg")`: `Apply` requires a dimensionless
argument and a negated quantity keeps whatever dimension it had. `a[i]` is
`Gather` and requires `i` to be a `Data` column; anything else is a `TypeError`
naming what was expected, rather than a pydantic validation error about a
discriminator.

`Convolve`, `Reduce`, `Apply`, and `Link` get no operator. There is no
punctuation that means "causal convolution", and inventing one (`@`, `>>`)
would make the tree harder to read than the constructor does.

## What is not done

`latex` renders `dose - k` as `\mathrm{dose} + -1 \cdot k`, which is the tree
faithfully and the mathematics badly. Teaching the renderer to recognize a
`Mul` whose leading factor is `Const(-1)` is a change to `interpret/latex.py`
and to whatever golden strings depend on it; it is a rendering concern, not an
algebra one, and is left open.
