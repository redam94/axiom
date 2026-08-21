"""Gate 9: one forward(). The numpy and jax interpreters agree on the same tree.

Review A2/C8 replaced the AST reachability heuristic with this numerical
statement: over random trees and random inputs, ``interpret.value`` and the
jax interpreter agree to 1e-10, and so do the numpy and jax log densities of
every shipped ``ModelSpec``. Drift between the likelihood (jax) and the design
math (numpy) is therefore impossible without failing this gate.

Skipped when jax is not installed (the core CI job); the full job runs it.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from _factories import EXAMPLES

from axiom.core import (
    Add,
    Apply,
    Const,
    D,
    Data,
    DimensionError,
    Div,
    Model,
    ModelSpec,
    Mul,
    Param,
    Pow,
    Reduce,
    dimension,
    jax_available,
    log_density,
    unconstrain,
    value,
)

pytestmark = pytest.mark.skipif(not jax_available(), reason="jax not installed")


def _random_tree(rng: random.Random, depth: int) -> Model:
    if depth == 0 or rng.random() < 0.3:
        kind = rng.random()
        if kind < 0.4:
            return Data(name=f"x{rng.randrange(3)}", dimension=D.currency)
        if kind < 0.8:
            return Param(name=f"p{rng.randrange(3)}", dimension=D.currency)
        return Const(value=rng.uniform(0.5, 2.0), dimension=D.currency)
    c = rng.randrange(6)
    sub = lambda: _random_tree(rng, depth - 1)  # noqa: E731
    if c == 0:
        return Add(terms=tuple(sub() for _ in range(rng.randrange(2, 4))))
    if c == 1:
        return Mul(
            factors=(
                sub(),
                Div(numerator=sub(), denominator=Const(value=1.0, dimension=D.currency)),
            )
        )
    if c == 2:
        return Mul(
            factors=(
                Div(numerator=sub(), denominator=sub()),
                Const(value=1.0, dimension=D.currency),
            )
        )
    if c == 3:
        return Mul(
            factors=(
                Pow(
                    base=Div(numerator=sub(), denominator=Const(value=1.0, dimension=D.currency)),
                    exponent="1/2",
                ),
                Const(value=1.0, dimension=D.currency),
            )
        )
    if c == 4:
        return Mul(
            factors=(
                Apply(
                    fn=rng.choice(["exp", "tanh", "softplus", "sigmoid", "log1p"]),
                    arg=Div(numerator=sub(), denominator=Const(value=1.0, dimension=D.currency)),
                ),
                Const(value=1.0, dimension=D.currency),
            )
        )
    return Mul(
        factors=(
            Reduce(
                op=rng.choice(["sum", "mean", "max"]),
                arg=Div(numerator=sub(), denominator=Const(value=1.0, dimension=D.currency)),
            ),
            Const(value=1.0, dimension=D.currency),
        )
    )


def test_value_and_jax_agree_on_200_random_trees() -> None:
    import jax

    from axiom.core import compile_jax

    jax.config.update("jax_enable_x64", True)
    rng = random.Random(7)
    nrng = np.random.default_rng(7)
    checked = 0
    for _ in range(200):
        tree = _random_tree(rng, depth=4)
        try:
            assert dimension(tree) == D.currency
        except DimensionError:  # pragma: no cover - generator keeps trees consistent
            continue
        data = {f"x{i}": nrng.uniform(0.5, 2.0, size=5) for i in range(3)}
        params = {f"p{i}": nrng.uniform(0.5, 2.0) for i in range(3)}
        expected = value(tree, data=data, params=params)
        got = np.asarray(compile_jax(tree)(data, params))
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
        checked += 1
    assert checked >= 190


def test_every_shipped_model_log_density_agrees() -> None:
    import jax

    from axiom.core import compile_log_density

    jax.config.update("jax_enable_x64", True)
    models = [f() for cls, f in EXAMPLES.items() if issubclass(cls, ModelSpec)]
    assert models, "no ModelSpec examples to check"
    nrng = np.random.default_rng(11)
    for m in models:
        assert isinstance(m, ModelSpec)
        n = 12
        data: dict[str, np.ndarray] = {name: nrng.uniform(1.0, 100.0, n) for name in ("dose",)}
        data["unit"] = nrng.integers(0, 3, n)
        theta = {p.name: (np.full(p.shape, 1.0) if p.shape else 1.0) for p in m.parameters}
        for p in m.parameters:
            if p.prior is not None and p.prior.family == "beta":
                theta[p.name] = 0.4
            if p.prior is not None and p.prior.family == "uniform":
                theta[p.name] = 0.5 * (float(p.prior.hyper["low"]) + float(p.prior.hyper["high"]))  # type: ignore[arg-type]
        data[m.outcome.name] = value(m.mean, data=data, params=theta) + nrng.normal(0, 1, n)
        z = unconstrain(m, theta)
        assert float(compile_log_density(m)(data, z)) == pytest.approx(
            log_density(m, data, z), abs=1e-8
        )
