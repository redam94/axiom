"""Gate: a surface never travels without its uncertainty.

A response curve is a posterior quantity. Drawing or reporting one as a line —
``forward`` at the posterior mean of the parameters — states that the curve is
known when it is not, and for a nonlinear surface it is not even the curve the
model believes. Two halves are asserted here:

1. Every public function that evaluates a fitted surface over a dose grid
   returns a ``ResponseBand``, which carries an interval *with its definition
   and mass* at every grid point (rule 4).
2. Every ``viz`` figure of a surface draws that band, and there is no argument
   anywhere that turns it off.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from axiom.core import D, Interval, Unsupported
from axiom.sim import DosePlan, surface_world
from axiom.surface import HillKernel, ResponseBand, fit, marginal_band, response_band
from axiom.viz import available, marginal_curve, response_curve

#: Every ``viz`` entry point that draws a fitted surface, and the band builder behind it.
SURFACE_FIGURES = {"response_curve": response_band, "marginal_curve": marginal_band}


@pytest.fixture(scope="module")
def fitted():  # type: ignore[no-untyped-def]
    world = surface_world(
        n_units=4,
        n_periods=10,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        doses=DosePlan(scale=50.0, zero_fraction=0.2),
        intercept="shared",
        noise_sd=0.5,
        seed=1,
    )
    return fit(world.spec, world.panel, backend="laplace", draws=120, chains=1, seed=1)


@pytest.mark.parametrize("builder", [response_band, marginal_band])
def test_a_grid_evaluation_returns_a_band_with_its_interval(builder, fitted) -> None:  # type: ignore[no-untyped-def]
    band = builder(fitted, "a", n_grid=6, mass=0.9, definition="eti")
    assert isinstance(band, ResponseBand), band
    assert band.n_grid == 6
    assert band.n_draws == 120
    # the interval at every grid point carries the definition and the mass
    for index in range(band.n_grid):
        interval = band.interval_at(index)
        assert isinstance(interval, Interval)
        assert interval.definition == "eti"
        assert interval.mass == 0.9
        assert interval.lower <= interval.upper
    assert all(w >= 0.0 for w in band.width)
    # and it is a Spec, so it round-trips and hashes like every other number here
    assert ResponseBand.from_dict(band.to_dict()) == band
    assert band.content_hash()


def test_the_band_is_over_draws_not_the_posterior_mean(fitted) -> None:  # type: ignore[no-untyped-def]
    """A non-degenerate posterior gives a band with width. A mean curve would not."""
    band = response_band(fitted, "a", n_grid=6)
    assert max(band.width) > 0.0
    # the mean of the draws is not the same as the median for a nonlinear surface
    assert band.mean != band.median


@pytest.mark.skipif(not available(), reason="plotly not installed")
@pytest.mark.parametrize("name", sorted(SURFACE_FIGURES))
def test_every_surface_figure_draws_its_band(name: str, fitted) -> None:  # type: ignore[no-untyped-def]
    figure = {"response_curve": response_curve, "marginal_curve": marginal_curve}[name](
        fitted, "a", n_grid=6
    )
    assert not isinstance(figure, Unsupported), figure
    # exactly two traces: the filled band, then the curve on top of it
    assert len(figure.data) == 2, [t.name for t in figure.data]
    band_trace, line = figure.data
    assert band_trace.fill == "toself", "the first trace of a surface figure is its band"
    assert "%" in str(band_trace.name), band_trace.name
    assert line.mode == "lines"
    assert np.asarray(band_trace.y).size == 2 * np.asarray(line.y).size


@pytest.mark.skipif(not available(), reason="plotly not installed")
@pytest.mark.parametrize("name", sorted(SURFACE_FIGURES))
def test_no_argument_turns_the_band_off(name: str) -> None:
    """The band is not a display option. Nothing in the signature can suppress it."""
    function = {"response_curve": response_curve, "marginal_curve": marginal_curve}[name]
    parameters = inspect.signature(function).parameters
    assert "mass" in parameters and "definition" in parameters
    forbidden = {"band", "show_band", "with_band", "uncertainty", "interval", "ribbon"}
    assert not forbidden & set(parameters), sorted(forbidden & set(parameters))


@pytest.mark.skipif(not available(), reason="plotly not installed")
def test_a_figure_can_be_drawn_from_a_band_without_refitting(fitted) -> None:  # type: ignore[no-untyped-def]
    """A report computes the band once and renders it; the two routes agree."""
    band = response_band(fitted, "a", n_grid=6)
    assert isinstance(band, ResponseBand)
    from_band = response_curve(band)
    from_fit = response_curve(fitted, "a", n_grid=6)
    assert not isinstance(from_band, Unsupported)
    np.testing.assert_allclose(np.asarray(from_band.data[1].y), np.asarray(from_fit.data[1].y))


def test_a_band_refuses_to_be_built_inconsistently() -> None:
    base: dict[str, object] = {
        "treatment": "a",
        "outcome": "y",
        "kind": "response",
        "doses": (0.0, 1.0),
        "mean": (0.0, 1.0),
        "median": (0.0, 1.0),
        "lower": (-1.0, 0.5),
        "upper": (1.0, 1.5),
        "definition": "eti",
        "mass": 0.9,
        "n_draws": 10,
        "dimension": D.outcome,
    }
    assert ResponseBand(**base)
    with pytest.raises(ValueError, match="at least two grid points"):
        ResponseBand(
            **{
                **base,
                "doses": (0.0,),
                "mean": (0.0,),
                "median": (0.0,),
                "lower": (0.0,),
                "upper": (0.0,),
            }
        )
    with pytest.raises(ValueError, match="values for 2 doses"):
        ResponseBand(**{**base, "mean": (0.0,)})
    with pytest.raises(ValueError, match="exceeds upper"):
        ResponseBand(**{**base, "lower": (2.0, 0.5)})
    with pytest.raises(ValueError, match="mass must be in"):
        ResponseBand(**{**base, "mass": 1.0})
