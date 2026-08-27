"""GEIGER-1911: the synthetic world the Rutherford case-study notebooks share.

One experiment, planned before it is run: **is the positive charge of an atom a
hard dense centre, or is it spread through the whole atom?** The two answers
predict different angular distributions of scattered alpha particles, and this
module holds the one data-generating process that contains both of them, so
that five notebooks can carry a single question from "what would tell them
apart" to "point the detector here, cut the aperture like this, count for this
long, and stop when you see this."

The question, as a number
------------------------

A beam of alpha particles of energy ``E_ALPHA`` is fired at a gold foil. An
alpha of charge ``z e`` aimed head-on at a charge ``Z e`` stops and turns
around at

    D = 2 z Z e² / (4 pi eps0 E)                      (``D_CLOSEST``)

which for radium-C' alphas on gold is about 59 femtometres. ``D`` is the whole
story, because it is the *smallest distance the beam can reach*. If the
positive charge of the atom sits inside a ball of radius ``R``, then a
trajectory scattered through angle ``theta`` has its closest approach at

    r_min(theta) = (D / 2) · (1 + 1 / sin(theta / 2))

and the Coulomb law it was following is only the truth while ``r_min > R``.
The angle at which the beam first reaches the edge of the charge therefore has

    sin(theta_cut / 2) = 1 / (2 R / D − 1)             (``cutoff_sine``)

and beyond it the scattering dies, because a particle inside the ball feels
only the charge enclosed beneath it and cannot be turned far. This is the one
parameter the whole case study is about. Two atoms, two values:

* **the diffuse atom** — positive charge spread through the whole atom,
  ``R = R_ATOM = 1.35e-10 m``. Then ``sin(theta_cut/2) = 2.2e-4``: nothing
  survives past about a fiftieth of a degree, and what is seen at wider angles
  is the pile-up of many tiny deflections, which has a Gaussian tail and dies
  faster than anything.
* **the hard centre** — ``R`` below ``D / 2 = 30 fm``. Then the trajectory
  never reaches the charge at all, at any angle, and the Coulomb law holds
  out to 180 degrees.

Between them the model is continuous in ``R``, which is what makes this a
design problem rather than a debate: the experiment estimates ``R``, and the
two hypotheses are two values of it, four decades apart.

What the detector sees
----------------------

The mean the surface evaluates is the **expected count** in a detector of
solid angle ``omega`` steradians left open for ``exposure`` seconds at
scattering angle ``theta`` radians, as the sum of two terms:

* the **single-scattering tail** — one close encounter with one positive
  charge, ``u^-4`` in ``u = sin(theta/2)`` (Rutherford's law), suppressed by
  ``exp(-(u/s)^2)`` once the trajectory would have to enter the charge ball;
* the **multiple-scattering core** — the compounding of the thousand-odd
  glancing encounters every alpha has crossing the foil, Gaussian in
  ``theta`` with width ``CORE_WIDTH``. Both atoms produce this, and produce
  very nearly the same one, which is exactly why the small-angle measurements
  of 1909 settled nothing.

Counting is Poisson, and the design machinery in ``axiom.design`` is written
for a Gaussian scale. The bridge is exact rather than approximate: ``forward``
returns ``2 sqrt(mu)``, the variance-stabilizing transform of a Poisson count,
whose variance is one whatever the rate. For any parameter ``psi``,

    (d[2 sqrt(mu)] / d psi)² / 1²  ==  (d mu / d psi)² / mu

and the right-hand side is the Poisson Fisher information. So
``fisher_information(surface, ..., noise_sd=1.0)`` *is* the Poisson
information, with no approximation to apologize for; notebook 2 checks it
against a direct calculation. ``counts()`` inverts the transform, so the
expected count is read back through the same ``forward`` rather than from a
second copy of the formula (rule 3).

Nothing here is imported by ``axiom``; it is notebook scaffolding, and every
statistical claim it supports is computed with ``axiom`` in the notebooks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from axiom.core import (
    BASES,
    Add,
    Apply,
    Const,
    Data,
    DesignMatrix,
    Dimension,
    Likelihood,
    ModelSpec,
    Mul,
    Param,
    Pow,
    Prior,
    dimensionless,
)
from axiom.surface import design_matrix

Array = npt.NDArray[np.float64]

# -- the apparatus ---------------------------------------------------------------------

#: ``e² / (4 pi eps0)`` in MeV·fm — the only physical constant that is not a choice.
COULOMB_MEV_FM = 1.439964

#: Radium-C' alphas, the source Geiger and Marsden used. MeV.
E_ALPHA = 7.68
#: Charge number of the alpha particle, and of the gold foil's atoms.
Z_ALPHA, Z_FOIL = 2, 79

#: Head-on distance of closest approach, in metres. Everything scales off this.
D_CLOSEST = 2.0 * Z_ALPHA * Z_FOIL * COULOMB_MEV_FM / E_ALPHA * 1e-15

#: Gold: atoms per cubic metre, foil thickness in metres, atomic radius in metres.
N_VOLUME_GOLD = 5.90e28
FOIL_THICKNESS = 4.0e-7
R_ATOM = 1.35e-10
#: Atoms per square metre of foil, which is what a beam crossing it meets.
AREAL_DENSITY = N_VOLUME_GOLD * FOIL_THICKNESS

#: Alphas per second entering the foil through the collimator.
BEAM_RATE = 2.0e5

#: Flashes per second a human at a scintillation screen can count without missing
#: any. Ninety a minute was the working figure; it is a hard constraint on the
#: aperture at small angles and it is what makes this an allocation problem.
MAX_COUNT_RATE = 1.5

#: Total observing time the experiment has, in seconds. Two hundred hours.
TIME_BUDGET = 200 * 3600.0

#: Width of the multiple-scattering core, in radians. The compounding of the
#: ``n_encounters`` glancing deflections every alpha has crossing the foil; both
#: atoms predict very nearly this, which is why it carries no information about
#: which one is right and why it must be modelled rather than ignored.
CORE_WIDTH = 0.0114

#: Counts per second per steradian the counter records with the foil taken out:
#: alphas off the collimator lip, scintillations in the glass, and a dark-adapted
#: eye that has been at the microscope for an hour. It is the only thing standing
#: between a wide-angle flash and a discovery, so the design has to measure it as
#: carefully as it measures the scattering.
BACKGROUND_DENSITY = 1.5e-3

#: The widest aperture the apparatus can be given, in steradians — the annulus of
#: scintillating screen the microscope can be swung over.
OMEGA_MAX = 0.2

#: The gold nucleus, for the record. The experiment cannot see it: it is far
#: inside ``D_CLOSEST / 2``, and no number of counts will bring it into view.
R_NUCLEUS = 7.3e-15

OUTCOME_UNIT = "counts"

count = BASES.declare("count", symbol="N")


# -- the one parameter -----------------------------------------------------------------


def cutoff_sine(radius: float) -> float:
    """``sin(theta_cut / 2)``: where a positive charge of radius ``R`` kills the tail.

    ``inf`` when ``R <= D_CLOSEST / 2`` — the beam never reaches the charge at
    any angle, so there is no cut-off and Rutherford's law holds to 180
    degrees. That threshold is the experiment's resolving power and it is set
    by the beam energy alone.
    """
    ratio = 2.0 * radius / D_CLOSEST
    return math.inf if ratio <= 1.0 else 1.0 / (ratio - 1.0)


def radius_from_cutoff(sine: float) -> float:
    """The radius of positive charge implied by a cut-off at ``sin(theta/2) = sine``."""
    if not sine > 0.0:
        raise ValueError(f"cut-off sine must be positive, got {sine}")
    return D_CLOSEST * (1.0 + sine) / (2.0 * sine)


def lam_of(radius: float) -> float:
    """``log sin(theta_cut / 2)``, the model's parameter, from a radius in metres."""
    return math.log(cutoff_sine(radius)) if math.isfinite(cutoff_sine(radius)) else LAM_NONE


def radius_of(lam: float) -> float:
    """The radius in metres implied by ``lam``; ``D_CLOSEST / 2`` once it is out of reach."""
    return D_CLOSEST / 2.0 if lam >= LAM_NONE else radius_from_cutoff(math.exp(lam))


#: ``lam`` large enough that ``exp(-(u/s)²) == 1`` for every ``u <= 1``: a charge the
#: beam never reaches. Any larger value is the same experiment, which is the point.
LAM_NONE = math.log(1e3)

#: The two atoms, as values of one parameter.
LAM_DIFFUSE = math.log(cutoff_sine(R_ATOM))
LAM_HARD = LAM_NONE

#: Amplitude of the single-scattering tail, in counts per second per steradian at
#: ``u = 1`` (back-scattering): beam rate x areal density x (D/4)². It is O(1) by
#: arithmetic, not by tuning.
TAIL_AMPLITUDE = BEAM_RATE * AREAL_DENSITY * (D_CLOSEST / 4.0) ** 2
#: Peak of the multiple-scattering core, counts per second per steradian: the
#: whole beam spread over the solid angle the core occupies.
CORE_AMPLITUDE = BEAM_RATE / (2.0 * math.pi * CORE_WIDTH**2)

LOG_A_TRUE = math.log(TAIL_AMPLITUDE)
LOG_C_TRUE = math.log(CORE_AMPLITUDE)
LOG_W_TRUE = math.log(CORE_WIDTH)
LOG_B_TRUE = math.log(BACKGROUND_DENSITY)


def truth(lam: float) -> dict[str, float]:
    """The parameter point for an atom whose positive charge has cut-off ``lam``."""
    return {
        "log_a": LOG_A_TRUE,
        "lam": float(lam),
        "log_c": LOG_C_TRUE,
        "log_w": LOG_W_TRUE,
        "log_b": LOG_B_TRUE,
    }


#: The two hypotheses, ready to hand to ``forward``.
HARD_CENTRE = truth(LAM_HARD)
DIFFUSE = truth(LAM_DIFFUSE)


# -- the surface -----------------------------------------------------------------------

#: One count per second, the reference the log-amplitudes are measured against. A
#: logarithm needs a dimensionless argument, so the rate scale has to be named
#: somewhere; naming it here keeps it out of the parameters.
RATE_UNIT = Const(value=1.0, dimension=count / BASES.time)


def _exp(node: Any) -> Apply:
    return Apply(fn="exp", arg=node)


def model() -> ModelSpec:
    """The mean: ``2 sqrt(expected count)`` in a detector at ``theta``, open ``omega`` for ``t``.

    Four parameters, all dimensionless logarithms so that the information
    matrix is scale-free and the two hypotheses are two points on one axis:

    ``log_a``  amplitude of the single-scattering tail (beam x foil x charge)
    ``lam``    ``log sin(theta_cut / 2)`` — the size of the positive charge
    ``log_c``  amplitude of the multiple-scattering core
    ``log_w``  ``log`` of the core's width in radians
    ``log_b``  amplitude of the counting background, per steradian

    The data column ``foil`` is ``1`` with the foil in the beam and ``0`` with
    it out, which switches both scattering terms off and leaves the background
    alone. That is how the experiment measures its own background, and making
    it a column of the design rather than a second model is what lets one
    search allocate time between the two kinds of run.
    """
    theta = Data(name="theta", dimension=dimensionless())
    u = Apply(fn="sin", arg=Mul(factors=(Const(value=0.5, dimension=dimensionless()), theta)))

    log_a = Param(
        name="log_a",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 2.0}),
    )
    lam = Param(
        name="lam",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": -4.0, "sigma": 6.0}),
    )
    log_c = Param(
        name="log_c",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": 19.0, "sigma": 3.0}),
    )
    log_w = Param(
        name="log_w",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": -4.5, "sigma": 1.0}),
    )

    tail = Mul(
        factors=(
            _exp(log_a),
            Pow(base=u, exponent=Fraction(-4)),
            _exp(
                Mul(
                    factors=(
                        Const(value=-1.0, dimension=dimensionless()),
                        Pow(base=_ratio(u, lam), exponent=Fraction(2)),
                    )
                )
            ),
        )
    )
    core = Mul(
        factors=(
            _exp(log_c),
            _exp(
                Mul(
                    factors=(
                        Const(value=-0.5, dimension=dimensionless()),
                        Pow(base=_ratio(theta, log_w), exponent=Fraction(2)),
                    )
                )
            ),
        )
    )
    log_b = Param(
        name="log_b",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": -6.0, "sigma": 3.0}),
    )
    scattered = Mul(factors=(Data(name="foil", dimension=dimensionless()), Add(terms=(tail, core))))
    rate = Mul(factors=(RATE_UNIT, Add(terms=(scattered, _exp(log_b)))))
    expected = Mul(
        factors=(
            rate,
            Data(name="omega", dimension=dimensionless()),
            Data(name="exposure", dimension=BASES.time),
        )
    )
    mean = Mul(
        factors=(
            Const(value=2.0, dimension=dimensionless()),
            Pow(base=expected, exponent=Fraction(1, 2)),
        )
    )
    return ModelSpec(
        name="alpha_scattering",
        mean=mean,
        outcome=Data(name="root_count", dimension=_root_count()),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(
            log_a,
            lam,
            log_c,
            log_w,
            log_b,
            # Not inferred: on the variance-stabilized scale the residual sd *is*
            # one, by construction, and estimating it would be estimating a
            # constant. Declaring it ``fixed`` is what keeps it out of every fit.
            Param(
                name="sigma",
                dimension=_root_count(),
                prior=Prior(family="fixed", hyper={"value": 1.0}),
            ),
        ),
    )


def _root_count() -> Dimension:
    return (count ** Fraction(1)).root(2)


def _ratio(numerator: Any, log_scale: Param) -> Any:
    """``numerator / exp(log_scale)`` — the dimensionless ratio a transform needs."""
    from axiom.core import Div

    return Div(numerator=numerator, denominator=_exp(log_scale))


@dataclass(frozen=True)
class Scattering:
    """The scattering surface: a ``SupportsForward`` over one hand-built expression tree.

    ``forward`` is the interpreter on ``expr`` and nothing else; ``linearize``
    is ``surface.design_matrix`` with no linear parameters, which is the
    truthful answer — the mean is nonlinear in all four, so the whole Jacobian
    comes from derivatives of ``forward`` (rule 3, and the reason
    ``fisher_information`` needs no help here).
    """

    spec: ModelSpec

    @property
    def expr(self) -> Any:
        return self.spec.mean

    def forward(
        self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
    ) -> Array:
        from axiom.core import value

        return np.asarray(value(self.spec.mean, data=dose, params=theta), dtype=np.float64)

    def linearize(
        self, dose: Mapping[str, npt.ArrayLike], theta_at: Mapping[str, npt.ArrayLike]
    ) -> DesignMatrix:
        return design_matrix(self.spec, (), dose, theta_at)

    def counts(
        self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
    ) -> Array:
        """Expected counts, read back through the same ``forward``: ``(2 sqrt(mu) / 2)²``."""
        return (self.forward(dose, theta) / 2.0) ** 2


def surface() -> Scattering:
    """The one surface every notebook uses."""
    return Scattering(model())


def point_charge_model() -> ModelSpec:
    """The same model with ``lam`` held at "no cut-off anywhere" rather than inferred.

    The question "how large could the positive charge be" has no answer in a
    world where the tail was never seen: ``lam`` and ``log_a`` are both dead
    columns there, and a profile of either is exactly flat. The question that
    *does* have an answer is the other one — **if there were a hard centre, how
    strong could its scattering be?** — and asking it means fixing the cut-off
    and profiling the amplitude. A ``fixed`` prior is how a ``ModelSpec`` says
    a parameter is not being inferred.
    """
    base = model()
    held = tuple(
        (
            p.model_copy(update={"prior": Prior(family="fixed", hyper={"value": LAM_HARD})})
            if p.name == "lam"
            else p
        )
        for p in base.parameters
    )
    return base.model_copy(update={"name": "alpha_scattering_point_charge", "parameters": held})


# -- the apparatus, as data ------------------------------------------------------------


def station(
    theta_deg: npt.ArrayLike,
    omega: npt.ArrayLike,
    exposure: npt.ArrayLike,
    foil: npt.ArrayLike = 1.0,
) -> dict[str, Array]:
    """The data mapping ``forward`` reads, from angles in **degrees**.

    Angles are quoted in degrees everywhere a human reads them and carried in
    radians everywhere the model does; this is the only place the two meet.
    """
    theta = np.radians(np.asarray(theta_deg, dtype=np.float64))
    n = theta.size

    def spread(v: npt.ArrayLike) -> Array:
        return np.broadcast_to(np.asarray(v, dtype=np.float64), (n,)).astype(np.float64).copy()

    return {
        "theta": theta,
        "omega": spread(omega),
        "exposure": spread(exposure),
        "foil": spread(foil),
    }


def rate_per_steradian(theta_deg: npt.ArrayLike, theta_p: Mapping[str, float]) -> Array:
    """Counts per second per steradian at each angle, background included, through ``forward``."""
    return surface().counts(station(theta_deg, 1.0, 1.0), theta_p)


def scattered_per_steradian(theta_deg: npt.ArrayLike, theta_p: Mapping[str, float]) -> Array:
    """The same with the background taken out — what the foil is responsible for.

    Computed as the difference of two ``forward`` calls, foil in and foil out,
    which is also exactly the measurement the experiment makes.
    """
    surf = surface()
    return surf.counts(station(theta_deg, 1.0, 1.0, 1.0), theta_p) - surf.counts(
        station(theta_deg, 1.0, 1.0, 0.0), theta_p
    )


def widest_aperture(theta_deg: npt.ArrayLike, theta_p: Mapping[str, float]) -> Array:
    """The widest aperture the counter can be given before flashes start being missed.

    ``MAX_COUNT_RATE / rate``, capped at ``OMEGA_MAX``. Below a few degrees this
    is a pinhole; past ninety degrees the apparatus, not the counter, is what
    binds. It is the first thing that decides what the slit looks like, and it
    decides it before any statistics are done.
    """
    return np.minimum(OMEGA_MAX, MAX_COUNT_RATE / rate_per_steradian(theta_deg, theta_p))


# -- the aperture ----------------------------------------------------------------------


#: The narrowest radial half-width the workshop will cut, in degrees. At the
#: hundred-millimetre arm of the apparatus this is a slot 0.17 mm across, and it
#: is a statement about brass, not about statistics.
SLIT_MIN_HALF_WIDTH = 0.05


def slit_for(theta_deg: float, omega: float) -> tuple[float, float]:
    """``(half_width_deg, arc_deg)``: the least-smearing slit of a given solid angle.

    A slit of half-width ``delta`` and azimuthal arc ``phi`` subtends exactly
    ``phi * 2 sin(theta) sin(delta)``, so the solid angle can be bought in
    either direction — and only the radial one smears, because the pattern
    does not depend on azimuth. The prescription follows: cut the slot as
    narrow as the workshop can and take the rest of the aperture out in
    azimuth; open the slot only once the arc has gone all the way round.
    """
    if not omega > 0.0:
        raise ValueError(f"solid angle must be positive, got {omega}")
    ring = 2.0 * math.sin(math.radians(theta_deg))
    arc = math.degrees(omega / (ring * math.sin(math.radians(SLIT_MIN_HALF_WIDTH))))
    if arc <= 360.0:
        return SLIT_MIN_HALF_WIDTH, arc
    sine = omega / (2.0 * math.pi * ring)
    if sine >= 1.0:
        raise ValueError(f"no annulus at {theta_deg} deg subtends {omega} sr")
    return math.degrees(math.asin(sine)), 360.0


def slit_solid_angle(theta_deg: float, half_width_deg: float, arc_deg: float) -> float:
    """Solid angle of an annular slit: half-width ``delta`` in ``theta``, ``arc`` in azimuth.

    ``int sin(theta) dtheta dphi`` over the opening, exactly.
    """
    lo = math.radians(max(theta_deg - half_width_deg, 0.0))
    hi = math.radians(min(theta_deg + half_width_deg, 180.0))
    return math.radians(arc_deg) * (math.cos(lo) - math.cos(hi))


def slit_counts(
    theta_deg: float,
    half_width_deg: float,
    arc_deg: float,
    exposure: float,
    theta_p: Mapping[str, float],
    *,
    n_nodes: int = 201,
) -> float:
    """Expected counts through a real slit: the rate integrated over what it admits.

    ``int rho(theta') sin(theta') dtheta' dphi x exposure``, by Simpson's rule
    on ``rho`` evaluated through ``forward``. This is the number the counter
    actually reports; ``rho(theta) x slit_solid_angle x exposure`` is what a
    model that ignores the width of its own aperture would predict, and
    notebook 3 is about the difference.
    """
    lo = max(theta_deg - half_width_deg, 1e-6)
    hi = min(theta_deg + half_width_deg, 180.0)
    nodes = np.linspace(lo, hi, n_nodes if n_nodes % 2 else n_nodes + 1)
    density = rate_per_steradian(nodes, theta_p) * np.sin(np.radians(nodes))
    weights = np.ones(nodes.size)
    weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
    step = math.radians(nodes[1] - nodes[0])
    integral = float(np.sum(weights * density) * step / 3.0)
    return integral * math.radians(arc_deg) * exposure


def aperture_bias(
    theta_deg: float, half_width_deg: float, arc_deg: float, theta_p: Mapping[str, float]
) -> float:
    """Relative error a point-value model makes at this slit: ``(through / point) - 1``.

    Positive because the falling ``u^-4`` law is convex: a slit collects more
    from its inner edge than it loses at its outer one, and the counter reports
    a rate that belongs to no single angle.
    """
    omega = slit_solid_angle(theta_deg, half_width_deg, arc_deg)
    through = slit_counts(theta_deg, half_width_deg, arc_deg, 1.0, theta_p)
    point = float(rate_per_steradian([theta_deg], theta_p)[0]) * omega
    return through / point - 1.0


# -- weighing the two atoms against each other -----------------------------------------


def weight_of_evidence(mu_hard: Array | float, mu_diffuse: Array | float) -> Array:
    """Expected log Bayes factor per detector, in nats, when the hard centre is true.

    The Kullback-Leibler divergence between two Poisson counts,
    ``mu_H log(mu_H / mu_D) - (mu_H - mu_D)``. It is the exact expected
    information gain for a choice between two hypotheses, in the same nats
    ``design.eig_gaussian`` reports, and it is what the angle allocation in
    notebook 2 maximizes.

    Both means include the background, and that is what keeps the logarithm
    finite: the diffuse atom does not predict *nothing* at ninety degrees, it
    predicts the background. The whole experiment is the difference between
    those two statements.
    """
    h = np.asarray(mu_hard, dtype=np.float64)
    d = np.asarray(mu_diffuse, dtype=np.float64)
    if np.any(d <= 0.0):
        raise ValueError("the alternative must predict a positive rate; include the background")
    return h * np.log(h / d) - (h - d)


def n_encounters() -> float:
    """Atoms whose electron cloud one alpha crosses on its way through the foil."""
    return N_VOLUME_GOLD * math.pi * R_ATOM**2 * FOIL_THICKNESS


def adversary(width_factor: float) -> dict[str, float]:
    """The diffuse atom with its multiple-scattering core stretched by ``width_factor``.

    The defence of a diffuse atom against a wide-angle count is always the
    same: *your core is wider than you think*. The beam is conserved, so a
    core ``f`` times wider is ``f²`` times lower at the peak, which is why
    stretching it is not free and why there is a worst case rather than a
    slippery slope. Notebook 2 minimizes the evidence over this family; the
    stations whose evidence does not move are the ones the argument cannot
    reach.
    """
    if not width_factor > 0.0:
        raise ValueError(f"width_factor must be positive, got {width_factor}")
    width = CORE_WIDTH * width_factor
    out = dict(DIFFUSE)
    out["log_w"] = math.log(width)
    out["log_c"] = math.log(BEAM_RATE / (2.0 * math.pi * width**2))
    return out


def surviving_evidence(
    theta_deg: npt.ArrayLike,
    omega: npt.ArrayLike,
    exposure: npt.ArrayLike,
    *,
    factors: npt.ArrayLike | None = None,
) -> tuple[Array, Array]:
    """``(evidence, worst_factor)`` per station, minimized over the adversary's core.

    The evidence a station would still carry if the diffuse atom were allowed
    to choose the most convenient core it can, and the width factor that
    chooses it. Nothing constrains the adversary here except conservation of
    the beam, so this is a *lower* bound; the anchor stations exist to take
    that freedom away, and notebook 2 says by how much.
    """
    grid = np.geomspace(0.3, 30.0, 41) if factors is None else np.asarray(factors, dtype=np.float64)
    surf = surface()
    data = station(theta_deg, omega, exposure)
    mu_hard = surf.counts(data, HARD_CENTRE)
    per_factor = np.stack(
        [weight_of_evidence(mu_hard, surf.counts(data, adversary(float(f)))) for f in grid]
    )
    best = np.argmin(per_factor, axis=0)
    return per_factor[best, np.arange(per_factor.shape[1])], grid[best]


@dataclass(frozen=True)
class Station:
    """One place the counter is put, for one length of time, with one aperture."""

    theta_deg: float
    hours: float
    role: str
    foil: float = 1.0

    @property
    def seconds(self) -> float:
        return self.hours * 3600.0


#: The plan notebooks 2, 3 and 4 arrive at, and notebook 5 runs. Four roles:
#: an **anchor** that pins the multiple-scattering core and the normalization,
#: a **bank** that carries the information the model believes in, a **witness**
#: at wide angle whose evidence no story about the core can touch, and a
#: **background** run with the foil out. Two hundred hours in all.
#:
#: The shape of it is notebook 4's argument in one table: buy every constraint
#: the argument will need, as cheaply as it can be bought, and spend everything
#: left at a hundred and fifty degrees — because that is the only station whose
#: evidence cannot be talked out of.
PLAN: tuple[Station, ...] = (
    Station(1.0, 5.0, "anchor"),
    Station(2.5, 5.0, "anchor"),
    Station(5.0, 10.0, "bank"),
    Station(10.0, 10.0, "bank"),
    Station(20.0, 10.0, "bank"),
    Station(45.0, 15.0, "bank"),
    Station(90.0, 25.0, "witness"),
    Station(150.0, 110.0, "witness"),
    Station(90.0, 10.0, "background", foil=0.0),
)


def plan_data(plan: Sequence[Station] = PLAN) -> dict[str, Array]:
    """The data mapping ``forward`` reads for a whole observing plan.

    Each station gets the widest aperture the counter can be given at its
    angle (``widest_aperture``); the foil-out run gets the same aperture as
    the wide-angle station it is checking, because a background measured
    through a different hole is not the background.
    """
    angles = np.array([s.theta_deg for s in plan], dtype=np.float64)
    omega = widest_aperture(angles, HARD_CENTRE)
    return station(
        angles,
        omega,
        np.array([s.seconds for s in plan], dtype=np.float64),
        np.array([s.foil for s in plan], dtype=np.float64),
    )


#: Prior sds for the five parameters, in the log units they are declared in. Wide
#: on the one the experiment is about, informative on the ones the apparatus is
#: already understood to within a factor of a few.
PRIOR_SDS = {"log_a": 2.0, "lam": 6.0, "log_c": 3.0, "log_w": 1.0, "log_b": 3.0}


# -- the house style -------------------------------------------------------------------

# The shared notebook style: one registered plotly template and one validated palette
# for all eighty-four notebooks. Imported for its side effect (the template) and for the
# colours below, so this case study cannot drift into a look of its own.
import sys as _sys
from pathlib import Path as _Path

_NBS = str(_Path(__file__).resolve().parents[2])
if _NBS not in _sys.path:
    _sys.path.insert(0, _NBS)

from _style import AQUA, AXIS, BLUE, CRITICAL, GRID as _GRID, INK, MUTED, ORANGE, SUBTLE, SURFACE, VIOLET

# The two competing hypotheses are the two categorical slots with the widest separation
# under every colour-vision simulation; the truth is a reference mark, not a series.
HARD_COLOR = ORANGE
DIFFUSE_COLOR = BLUE
TRUTH_COLOR = MUTED
ACCENT = VIOLET
SLIT_COLOR = AQUA
GRID = _GRID


def figure(title: str, x: str, y: str, *, height: int = 400, **kwargs: object) -> Any:
    """A plotly figure with the case study's house layout already applied."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        height=height,
        template="axiom",
        margin={"l": 70, "r": 30, "t": 60, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def degrees_axis(fig: Any, *, log: bool = False) -> Any:
    """Label the x axis in degrees, with the angles a human thinks in."""
    if log:
        fig.update_xaxes(
            type="log",
            tickvals=[0.3, 1, 3, 10, 30, 90, 180],
            ticktext=["0.3°", "1°", "3°", "10°", "30°", "90°", "180°"],
        )
    else:
        fig.update_xaxes(
            tickvals=list(range(0, 181, 30)), ticktext=[f"{d}°" for d in range(0, 181, 30)]
        )
    return fig


def band(
    fig: Any, x: Sequence[float], lo: Sequence[float], hi: Sequence[float], color: str, name: str
) -> Any:
    """A shaded interval, drawn the same way in every notebook."""
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(hi) + list(lo)[::-1],
            fill="toself",
            fillcolor=color,
            opacity=0.18,
            line={"width": 0},
            name=name,
            hoverinfo="skip",
        )
    )
    return fig
