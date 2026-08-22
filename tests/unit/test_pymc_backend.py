"""The PyTensor interpreter and the PyMC backend.

The interpreter is checked against ``core.value`` term for term, because that is
the whole claim: PyMC samples the same tree, not a second model that happens to
look like it. The backend is checked for the thing it is *for* — several NUTS
samplers over one graph — and for the provenance it must record.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from axiom.core import Posterior, Unsupported, log_density, unconstrain
from axiom.infer import BACKEND_NAMES, Backend, get_backend
from axiom.sim import DosePlan, arms_world, surface_world
from axiom.surface import (
    GaussianProcessKernel,
    GeometricCarryover,
    HillKernel,
    LogisticKernel,
    PiecewiseLinearKernel,
    PolynomialKernel,
    PowerKernel,
    SplineKernel,
    WeibullCarryover,
)

HAS_PYMC = importlib.util.find_spec("pymc") is not None
needs_pymc = pytest.mark.skipif(not HAS_PYMC, reason="the pymc extra is not installed")


def _panel_world(**over):  # type: ignore[no-untyped-def]
    kwargs = {
        "n_units": 3,
        "n_periods": 12,
        "treatments": ("a",),
        "kernels": HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        "doses": DosePlan(scale=50.0, zero_fraction=0.2),
        "intercept": "shared",
        "noise_sd": 0.5,
        "seed": 2,
    }
    return surface_world(**{**kwargs, **over})


def _columns(world):  # type: ignore[no-untyped-def]
    data = dict(world.data)
    shape = (world.panel.frame["unit"].nunique(), -1)
    data[world.model.outcome.name] = np.asarray(
        world.panel.frame[world.model.outcome.name]
    ).reshape(shape)
    return data


def _evaluate(model, data, z):  # type: ignore[no-untyped-def]
    """The pytensor log density at ``z``, as a float."""
    import pytensor
    import pytensor.tensor as pt

    from axiom.core.interpret.pytensor import compile_log_density

    symbols = {
        name: pt.tensor(name=name, dtype="float64", shape=tuple(np.shape(v)))
        for name, v in z.items()
    }
    graph = compile_log_density(model, data)(symbols)
    fn = pytensor.function(list(symbols.values()), graph, on_unused_input="ignore")
    return float(fn(*[z[name] for name in symbols])), graph


# -- the interpreter ------------------------------------------------------------------------


@needs_pymc
@pytest.mark.parametrize(
    "kernel",
    [
        HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        LogisticKernel(reference_dose=50.0),
        PowerKernel(reference_dose=50.0),
        PolynomialKernel(reference_dose=60.0, degree=3),
        SplineKernel(reference_dose=60.0, knots=(15.0, 30.0, 45.0)),
        PiecewiseLinearKernel(reference_dose=60.0, knots=(20.0, 40.0)),
        GaussianProcessKernel(reference_dose=60.0, n_basis=6),
    ],
    ids=lambda k: k.name,
)
def test_the_log_density_matches_core_value_for_every_kernel(kernel) -> None:  # type: ignore[no-untyped-def]
    """Including the families that use relu, step, sin and cos."""
    world = arms_world(
        n_units=40,
        treatments=("a",),
        kernels=kernel,
        doses={"a": np.linspace(0.0, 60.0, 40)},
        noise_sd=0.5,
        seed=1,
    )
    data = dict(world.data)
    data[world.model.outcome.name] = np.asarray(world.panel.frame[world.model.outcome.name])
    z = {
        k: np.asarray(v, dtype=float)
        for k, v in unconstrain(
            world.model, {k: np.asarray(v) for k, v in world.theta.items()}
        ).items()
    }
    got, graph = _evaluate(world.model, data, z)
    assert graph.dtype == "float64", "a float32 log density silently costs eight digits"
    assert got == pytest.approx(float(log_density(world.model, data, z)), rel=1e-10, abs=1e-9)


@needs_pymc
@pytest.mark.parametrize(
    "carryover",
    [GeometricCarryover(max_lag=4), WeibullCarryover(max_lag=5), None],
    ids=["geometric", "weibull", "none"],
)
def test_the_log_density_matches_through_carryover_and_hierarchy(carryover) -> None:  # type: ignore[no-untyped-def]
    world = _panel_world(carryover=carryover, intercept="hierarchical")
    data = _columns(world)
    z = {
        k: np.asarray(v, dtype=float)
        for k, v in unconstrain(
            world.model, {k: np.asarray(v) for k, v in world.theta.items()}
        ).items()
    }
    got, _ = _evaluate(world.model, data, z)
    assert got == pytest.approx(float(log_density(world.model, data, z)), rel=1e-10, abs=1e-9)


@needs_pymc
def test_the_gradient_matches_the_jax_interpreter() -> None:
    """Two independently written interpreters of one tree, differentiated."""
    pytest.importorskip("jax")
    import jax
    import pytensor

    from axiom.core.interpret.jax import compile_log_density as jax_ld

    jax.config.update("jax_enable_x64", True)
    world = _panel_world(carryover=GeometricCarryover(max_lag=4), intercept="hierarchical")
    data = _columns(world)
    z = {
        k: np.asarray(v, dtype=float)
        for k, v in unconstrain(
            world.model, {k: np.asarray(v) for k, v in world.theta.items()}
        ).items()
    }
    import pytensor.tensor as pt

    from axiom.core.interpret.pytensor import compile_log_density

    symbols = {
        n: pt.tensor(name=n, dtype="float64", shape=tuple(np.shape(v))) for n, v in z.items()
    }
    graph = compile_log_density(world.model, data)(symbols)
    grad_fn = pytensor.function(
        list(symbols.values()),
        pytensor.grad(graph, list(symbols.values())),
        on_unused_input="ignore",
    )
    got = dict(zip(symbols, grad_fn(*[z[n] for n in symbols]), strict=True))
    want = jax.grad(
        lambda zz: jax_ld(world.model)({k: np.asarray(v) for k, v in data.items()}, zz)
    )(z)
    for name in symbols:
        np.testing.assert_allclose(
            np.asarray(got[name]), np.asarray(want[name]), rtol=1e-6, atol=1e-7, err_msg=name
        )


@needs_pymc
def test_a_convolution_of_unknown_length_is_refused_by_name() -> None:
    """The one shape this interpreter cannot do, said out loud rather than guessed."""
    import pytensor.tensor as pt

    from axiom.core.interpret.pytensor import _static_length

    assert _static_length(pt.as_tensor_variable(np.zeros((2, 5))), "x") == 5
    with pytest.raises(ValueError, match="statically known length"):
        _static_length(pt.vector("v"), "the kernel of a Convolve")


def test_asking_whether_pytensor_is_available_never_imports_it() -> None:
    import subprocess
    import sys

    code = (
        "import sys; from axiom.core.interpret.pytensor import pytensor_available; "
        "pytensor_available(); "
        "print('pytensor' in sys.modules, 'pymc' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "False False", out


# -- the backend ----------------------------------------------------------------------------


def test_pymc_is_a_known_backend() -> None:
    assert "pymc" in BACKEND_NAMES
    resolved = get_backend("pymc")
    if HAS_PYMC:
        assert isinstance(resolved, Backend)
        assert resolved.name == "pymc"
    else:
        assert isinstance(resolved, Unsupported)
        assert "pymc" in resolved.missing


@needs_pymc
def test_samplers_reports_what_is_installed_without_running_anything() -> None:
    from axiom.infer import NUTS_SAMPLERS, samplers

    status = samplers()
    assert set(status) == set(NUTS_SAMPLERS)
    assert status["pymc"].usable, "PyMC's own sampler needs nothing beyond pymc"
    for name, entry in status.items():
        assert entry.name == name
        assert entry.error == "", "the fast path never probes, so it never has an error"
        if entry.missing:
            assert not entry.usable
            assert "needs" in str(entry)


@needs_pymc
def test_probing_finds_a_sampler_that_imports_but_does_not_work() -> None:
    """Installed is not the same as usable, and the report says which it means."""
    from axiom.infer import samplers

    probed = samplers(probe=True)
    assert probed["pymc"].usable, str(probed["pymc"])
    for entry in probed.values():
        if entry.error:
            assert not entry.usable
            assert "not usable with pymc" in entry.error


@needs_pymc
def test_an_absent_sampler_is_refused_by_name() -> None:
    from axiom.infer.pymc_backend import _check_sampler

    with pytest.raises(ValueError, match="unknown nuts_sampler"):
        _check_sampler("stan")
    with pytest.raises(ValueError, match="unknown nuts_sampler"):
        from axiom.infer import PymcBackend

        PymcBackend(nuts_sampler="stan")  # type: ignore[arg-type]


@needs_pymc
def test_sampling_records_its_provenance() -> None:
    world = _panel_world()
    backend = get_backend("pymc")
    assert isinstance(backend, Backend)
    posterior = backend.sample(world.model, _columns(world), draws=150, tune=150, chains=2, seed=7)
    assert isinstance(posterior, Posterior)
    provenance = posterior.provenance
    assert provenance["backend"] == "pymc"
    assert provenance["method"] == "nuts"
    assert provenance["nuts_sampler"] == "pymc"
    assert provenance["seed"] == 7
    assert provenance["draws"] == 150 and provenance["chains"] == 2
    assert provenance["model_hash"] == world.model.content_hash()
    assert provenance["init"] in ("laplace_mode", "zeros", "init")
    assert provenance["init_per_chain"] is True
    assert 0.0 <= provenance["nonfinite_draw_frac"] <= 1.0
    assert "pymc_version" in provenance
    for name in ("alpha", "beta_a", "k_a", "s_a", "sigma"):
        assert posterior.draws(name).shape[:2] == (2, 150)


@needs_pymc
def test_an_external_sampler_records_that_it_jittered_its_own_chains() -> None:
    """nutpie refuses a start per chain, so it does its own -- and provenance says so."""
    from axiom.infer import samplers
    from axiom.infer.pymc_backend import sample

    if not samplers(probe=True)["nutpie"].usable:
        pytest.skip("nutpie is not usable here")
    world = _panel_world()
    posterior = sample(
        world.model, _columns(world), draws=100, tune=100, chains=2, seed=7, nuts_sampler="nutpie"
    )
    assert posterior.provenance["nuts_sampler"] == "nutpie"
    assert posterior.provenance["init_per_chain"] is False
    assert posterior.provenance["init_jitter"] == 0.0


@needs_pymc
def test_the_same_seed_gives_the_same_draws() -> None:
    world = _panel_world()
    data = _columns(world)
    backend = get_backend("pymc")
    assert isinstance(backend, Backend)
    first = backend.sample(world.model, data, draws=100, tune=100, chains=1, seed=11)
    second = backend.sample(world.model, data, draws=100, tune=100, chains=1, seed=11)
    assert isinstance(first, Posterior) and isinstance(second, Posterior)
    np.testing.assert_allclose(first.draws("beta_a"), second.draws("beta_a"))


@needs_pymc
def test_optimize_and_laplace_come_from_the_shared_implementation() -> None:
    """The backend is a NUTS backend; the other two methods are not reimplemented."""
    world = _panel_world()
    data = _columns(world)
    backend = get_backend("pymc")
    assert isinstance(backend, Backend)
    mode = backend.optimize(world.model, data, seed=1)
    assert mode.converged
    approx = backend.laplace(world.model, data, draws=200, seed=1)
    assert isinstance(approx, Posterior)
    assert approx.provenance["method"] == "laplace"


@needs_pymc
def test_a_model_with_no_free_parameters_is_refused() -> None:
    from axiom.core import D, Data, Likelihood, ModelSpec, Param, Prior
    from axiom.infer.pymc_backend import sample

    fixed = Param(name="mu", dimension=D.outcome, prior=Prior(family="fixed", hyper={"value": 1.0}))
    model = ModelSpec(
        name="degenerate",
        mean=fixed,
        outcome=Data(name="y", dimension=D.outcome),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(
            fixed,
            Param(
                name="sigma",
                dimension=D.outcome,
                prior=Prior(family="fixed", hyper={"value": 1.0}),
            ),
        ),
    )
    with pytest.raises(ValueError, match="no free parameters"):
        sample(model, {"y": np.zeros(4)}, draws=5, tune=5, chains=1, seed=0)


@needs_pymc
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"draws": 0}, "draws >= 1"),
        ({"chains": 0}, "draws >= 1"),
        ({"tune": -1}, "draws >= 1"),
        ({"target_accept": 1.0}, "target_accept must be in"),
    ],
)
def test_sizes_are_validated_before_anything_is_compiled(kwargs, match) -> None:  # type: ignore[no-untyped-def]
    from axiom.infer.pymc_backend import sample

    world = _panel_world()
    call = {"draws": 10, "tune": 10, "chains": 1, "seed": 0, **kwargs}
    with pytest.raises(ValueError, match=match):
        sample(world.model, _columns(world), **call)
