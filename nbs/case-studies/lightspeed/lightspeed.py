"""LUMEN-299: the synthetic world the speed-of-light case-study notebooks share.

One constant, measured by five kinds of instrument over a hundred and twenty
years, and the question the whole programme turns on: **how far should the
next experiment believe the ones before it?** This module holds the one
data-generating process every notebook draws from, so that five notebooks can
carry a single measurement programme from "what is this a measurement *of*"
through the design of one famous apparatus, the running of it, the pooling of
everything ever filed, and the decisions the pool would have taken along the
way -- with the truth beside every number, because the world is synthetic and
the truth is known.

The constant, and the two numbers it is reported as
--------------------------------------------------

Since 1983 the metre is defined so that light in vacuo travels exactly

    c = 299 792.458 km/s                                    (``C_TRUE``)

Air slows it by a factor of the refractive index, ``N_AIR = 1.000293`` for dry
air at 0 C and one atmosphere, so a measurement made across a mountain valley
reads ``c / N_AIR``, about 88 km/s low -- which is *larger than every standard
error reported after 1926*. Whether a published number is "in air" or "in
vacuo" is therefore not a footnote; it is a definition of the outcome, and the
programme has to register which one each laboratory meant (notebook 1).

Fizeau reported 70 948 leagues per second. Which league decides the third
significant figure: the *lieue de 25 au degre* is 4.4448 km and gives
315 300 km/s; the 4.416 km league that later accounts assumed gives
313 300 km/s. Both numbers circulate in textbooks for the same experiment.
That is the unit story in one line, and it is why ``UNITS`` declares the
league and registers the conversion rather than leaving it to a footnote.

What a rotating mirror measures
-------------------------------

The Foucault method sends light to a distant mirror at ``D`` and back onto a
mirror spinning at ``omega`` revolutions per second. During the round trip the
mirror turns ``2 pi omega . 2D / c`` radians, and a beam reflected off a mirror
turned by ``theta`` is deflected by ``2 theta``, so the returned image is
displaced through

    delta = 8 pi omega D / c                                (``Mirror.forward``)

Michelson's 1926 apparatus used an ``n``-faced mirror and tuned ``omega`` so
that the *next* face was in position when the light returned; the observable
is then the residual displacement ``delta - 4 pi / n``, which the data column
``null_angle`` carries (``0`` for a plane mirror). Everything in the case study
that touches a rotating mirror evaluates that one expression tree (rule 3):
the Fisher information in notebook 2, the readings in notebook 3, and the
reduction of a reading back to a speed, which is ``forward`` inverted.

Two things limit the reading. The image is blurred by the air it crossed
(``SEEING``, an angle that does not depend on ``D``) and dimmed by the distance
(``DIMMING``, an angular error growing like the square root of ``D`` because
the returned intensity falls with the path). So

    sd(delta) = sqrt(SEEING^2 + DIMMING^2 . D)              (``angular_noise``)

and the information about ``c`` per reading grows like ``D`` rather than
``D^2`` -- a longer baseline keeps paying, but less than the geometry alone
promises. A speed is a distance over a time, so ``c`` and ``D`` enter only as
the ratio ``D / c``: a rotating mirror alone cannot separate them, and the
survey of the baseline (``survey_observation``) is the second observation that
does. That is the identifiability story of notebook 2.

The century, as a world
-----------------------

``HISTORY`` is the published record: sixteen experiments in four instrument
families, each with the value and the uncertainty its authors reported. The
synthetic world does not replay those numbers; it draws each experiment's
readout as

    truth in the reported medium
      + a systematic the whole family carried in that era     (``family_offset``)
      + a systematic of the experiment's own                  (``own_offset``)
      + noise at the reported standard error

seeded, so the numbers resemble history without being it. The structure is the
point. A *generation* -- one design of instrument, which a family builds a few
of -- carries one systematic, drawn once; the early generations carry large
ones and quote optimistic uncertainties. The two generations active between
1928 and 1945, rotating mirrors in vacuum and Kerr cells, carry offsets drawn
around the same ``Truth.bandwagon`` of about -17 km/s, which is the pattern
Birge's 1941 consensus of 299 776 +/- 4 km/s recorded and the electromagnetic
methods of the 1950s overturned. After 1945 the offsets are small.

Nothing here is imported by ``axiom``; it is notebook scaffolding, and every
statistical claim it supports is computed with ``axiom`` in the notebooks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from axiom.core import (
    BASES,
    UNITS,
    Add,
    Apply,
    Const,
    Data,
    DesignMatrix,
    Div,
    Intervention,
    Likelihood,
    ModelSpec,
    Mul,
    Outcome,
    Param,
    Population,
    Prior,
    Spec,
    TimeWindow,
    Treatment,
    dimensionless,
)
from axiom.estimands import Estimand, Level, Quantity
from axiom.io import Program
from axiom.surface import design_matrix

Array = npt.NDArray[np.float64]

# -- the constant --------------------------------------------------------------------

#: The speed of light in vacuo, km/s, exact since the 1983 definition of the metre.
C_TRUE = 299_792.458
#: Refractive index of dry air at 0 C and one atmosphere. Every measurement made
#: across open air reads ``C_TRUE / N_AIR`` unless its authors corrected it.
N_AIR = 1.000_293
C_AIR_TRUE = C_TRUE / N_AIR

#: The league Fizeau counted in: 25 to the degree of a great circle, 4.4448 km.
LEAGUE_KM = Fraction(44_448, 10_000)
#: What Fizeau published: leagues per second, no uncertainty stated.
FIZEAU_LEAGUES = 70_948.0

#: A base dimension for length, which the core's four defaults do not include.
length = BASES.declare("length", symbol="L")
SPEED = length / BASES.time
PER_SECOND = dimensionless() / BASES.time

UNITS.declare("km", "length")
UNITS.declare("league", "length")
UNITS.register("league", "km", LEAGUE_KM)


def to_vacuo(c_in_air: float) -> float:
    """A speed measured across air, corrected to vacuo: ``c_air . N_AIR``."""
    return c_in_air * N_AIR


def ppm(c: float) -> float:
    """Parts per million from the defined value -- the dimensionless quantity the corpus pools."""
    return (c - C_TRUE) / C_TRUE * 1e6


def from_ppm(x: float) -> float:
    """The speed, km/s, that ``x`` parts per million from the defined value names."""
    return C_TRUE * (1.0 + x * 1e-6)


# -- the estimand ----------------------------------------------------------------------

#: One second of transit -- the dose every instrument sets, whatever its mechanism.
TRANSIT = Treatment(
    name="transit",
    dimension=BASES.time,
    unit="s",
    description="the time light is allowed to travel before the apparatus reads it",
)
DISTANCE_VACUO = Outcome(
    name="distance",
    dimension=length,
    unit="km",
    aggregation="mean",
    description="path length covered, corrected to vacuo",
)
DISTANCE_AIR = Outcome(
    name="distance",
    dimension=length,
    unit="km",
    aggregation="mean",
    description="path length covered, in the air the light actually crossed",
)


def estimand(*, medium: str = "vacuo", radiation: str = "visible", name: str = "c") -> Estimand:
    """The ratio *distance covered per second of transit*, which is the speed of light.

    A constant of nature is not the response to an intervention, and no
    ``QuantityKind`` describes a level without pretence. What every instrument
    in the programme actually does is set a transit time and read a distance,
    and the **ratio** of the two is ``c`` in km/s -- with the medium the light
    crossed carried as the intervention's ``version`` and the radiation used as
    the population. Those are the two facets the century's records differ in,
    and both are things a pool has to be told about.
    """
    outcome = DISTANCE_VACUO if medium == "vacuo" else DISTANCE_AIR
    return Estimand(
        name=name,
        quantity=Quantity(kind="ratio"),
        treatment=TRANSIT,
        intervention=Intervention(doses={"transit": 1.0}, version=f"in {medium}"),
        reference=Intervention(doses={"transit": 0.0}, version=f"in {medium}"),
        outcome=outcome,
        population=Population(name=radiation, description=f"{radiation} radiation"),
        window=TimeWindow(start=0, stop=1, basis="per_period"),
        level=Level(unit="aggregate"),
        dimension=SPEED,
        description=f"distance covered per second of transit, {radiation} light, in {medium}",
    )


# -- the record ------------------------------------------------------------------------

#: The four instrument families, which are the parties of the programme.
FAMILIES: tuple[str, ...] = ("toothed-wheel", "rotating-mirror", "kerr-cell", "electromagnetic")

FAMILY_LABEL: dict[str, str] = {
    "toothed-wheel": "toothed wheel",
    "rotating-mirror": "rotating mirror",
    "kerr-cell": "Kerr cell",
    "electromagnetic": "cavity / microwave / laser",
}


@dataclass(frozen=True)
class Experiment:
    """One published measurement: who, when, with what, and what they said."""

    key: str
    year: int
    lab: str
    who: str
    family: str
    reported: float
    reported_se: float
    medium: str
    generation: str = ""
    radiation: str = "visible"
    baseline_km: float | None = None
    omega: float | None = None
    faces: int | None = None
    se_assumed: bool = False

    @property
    def era(self) -> str:
        if self.year < 1928:
            return "classical"
        if self.year < 1945:
            return "bandwagon"
        return "modern"

    @property
    def reported_vacuo(self) -> float:
        """The published number read as a speed in vacuo."""
        return to_vacuo(self.reported) if self.medium == "air" else self.reported

    @property
    def program(self) -> Program:
        return Program(party=self.family, program=self.lab)


#: The century, as published. ``reported`` is in the medium the authors quoted.
HISTORY: tuple[Experiment, ...] = (
    Experiment(
        "fizeau_1849",
        1849,
        "fizeau",
        "Fizeau",
        "toothed-wheel",
        FIZEAU_LEAGUES * float(LEAGUE_KM),
        3000.0,
        "air",
        "fizeau",
        baseline_km=8.633,
        se_assumed=True,
    ),
    Experiment(
        "foucault_1862",
        1862,
        "foucault",
        "Foucault",
        "rotating-mirror",
        298_000.0,
        500.0,
        "air",
        "foucault",
        baseline_km=0.020,
        omega=400.0,
    ),
    Experiment(
        "cornu_1874",
        1874,
        "cornu",
        "Cornu",
        "toothed-wheel",
        299_990.0,
        200.0,
        "air",
        "cornu",
        baseline_km=22.910,
    ),
    Experiment(
        "michelson_1879",
        1879,
        "michelson",
        "Michelson (Annapolis)",
        "rotating-mirror",
        299_944.0,
        50.0,
        "vacuo",
        "michelson-i",
        baseline_km=0.605,
        omega=257.0,
    ),
    Experiment(
        "newcomb_1882",
        1882,
        "newcomb",
        "Newcomb (Washington)",
        "rotating-mirror",
        299_860.0,
        30.0,
        "vacuo",
        "michelson-i",
        baseline_km=3.721,
        omega=230.0,
    ),
    Experiment(
        "perrotin_1902",
        1902,
        "perrotin",
        "Perrotin (Nice)",
        "toothed-wheel",
        299_880.0,
        84.0,
        "air",
        "cornu",
        baseline_km=45.950,
    ),
    Experiment(
        "michelson_1926",
        1926,
        "michelson",
        "Michelson (Mt Wilson)",
        "rotating-mirror",
        299_796.0,
        4.0,
        "vacuo",
        "michelson-ii",
        baseline_km=35.385,
        omega=528.0,
        faces=8,
    ),
    Experiment(
        "karolus_1928",
        1928,
        "karolus",
        "Karolus & Mittelstaedt",
        "kerr-cell",
        299_778.0,
        20.0,
        "vacuo",
        "kerr",
    ),
    Experiment(
        "pease_pearson_1935",
        1935,
        "pease-pearson",
        "Michelson, Pease & Pearson",
        "rotating-mirror",
        299_774.0,
        11.0,
        "vacuo",
        "vacuum",
        baseline_km=8.0,
        omega=585.0,
        faces=32,
    ),
    Experiment(
        "anderson_1937", 1937, "anderson", "Anderson", "kerr-cell", 299_771.0, 12.0, "vacuo", "kerr"
    ),
    Experiment(
        "huettel_1940", 1940, "huettel", "Huettel", "kerr-cell", 299_768.0, 10.0, "vacuo", "kerr"
    ),
    Experiment(
        "anderson_1941", 1941, "anderson", "Anderson", "kerr-cell", 299_776.0, 14.0, "vacuo", "kerr"
    ),
    Experiment(
        "essen_1950",
        1950,
        "essen",
        "Essen (cavity resonator)",
        "electromagnetic",
        299_792.5,
        3.0,
        "vacuo",
        "electromagnetic",
        radiation="radio",
    ),
    Experiment(
        "bergstrand_1951",
        1951,
        "bergstrand",
        "Bergstrand (geodimeter)",
        "electromagnetic",
        299_793.1,
        0.3,
        "vacuo",
        "electromagnetic",
    ),
    Experiment(
        "froome_1958",
        1958,
        "froome",
        "Froome (microwave interferometer)",
        "electromagnetic",
        299_792.5,
        0.1,
        "vacuo",
        "electromagnetic",
        radiation="radio",
    ),
    Experiment(
        "evenson_1972",
        1972,
        "evenson",
        "Evenson (laser)",
        "electromagnetic",
        299_792.4574,
        0.0011,
        "vacuo",
        "laser",
    ),
)

BY_KEY: dict[str, Experiment] = {e.key: e for e in HISTORY}


def programs() -> dict[str, Program]:
    """One ``io.Program`` per laboratory, keyed by the lab token, party = instrument family."""
    return {e.lab: e.program for e in HISTORY}


def ledger_frame() -> pd.DataFrame:
    """The published record as a table, with the truth in each experiment's own medium."""
    rows = []
    for e in HISTORY:
        truth = C_AIR_TRUE if e.medium == "air" else C_TRUE
        rows.append(
            {
                "year": e.year,
                "who": e.who,
                "family": FAMILY_LABEL[e.family],
                "medium": e.medium,
                "reported": e.reported,
                "reported_se": e.reported_se,
                "truth_in_medium": truth,
                "sigma_off": (e.reported - truth) / e.reported_se,
            }
        )
    return pd.DataFrame(rows)


# -- the world -------------------------------------------------------------------------


@dataclass(frozen=True)
class Truth:
    """Every parameter of the data-generating process, in km/s.

    A *generation* is one design of instrument in one family -- Foucault's
    twenty-metre mirror, Michelson's Annapolis apparatus that Newcomb copied,
    the Mount Wilson octagon, the vacuum tube, the Kerr cells of 1928-41. Each
    carries one systematic, drawn once, with the spread its builders could not
    see; the two generations of the bandwagon era are drawn around the same
    centre, which is the one fact about the world no hierarchy over families
    can see and only a new kind of instrument reveals.
    """

    c: float = C_TRUE
    n_air: float = N_AIR
    #: sd of the systematic each generation of instrument carried, km/s.
    generation_sd: Mapping[str, float] = field(
        default_factory=lambda: {
            "fizeau": 3000.0,
            "foucault": 400.0,
            "cornu": 120.0,
            "michelson-i": 60.0,
            "michelson-ii": 3.0,
            "vacuum": 2.5,
            "kerr": 2.5,
            "electromagnetic": 0.15,
            "laser": 0.001,
        }
    )
    #: the generations that shared an expectation, and what they shared.
    bandwagon_generations: tuple[str, ...] = ("vacuum", "kerr")
    bandwagon: float = -17.0
    #: each experiment's own systematic, as a multiple of its reported se, by era.
    own_factor: Mapping[str, float] = field(
        default_factory=lambda: {"classical": 1.5, "bandwagon": 0.6, "modern": 0.5}
    )

    def in_medium(self, medium: str) -> float:
        return self.c / self.n_air if medium == "air" else self.c


TRUTH = Truth()


@dataclass(frozen=True)
class Readout:
    """One experiment as the synthetic world ran it."""

    experiment: Experiment
    read: float
    se: float
    family_offset: float
    own_offset: float

    @property
    def read_vacuo(self) -> float:
        return to_vacuo(self.read) if self.experiment.medium == "air" else self.read

    @property
    def truth_in_medium(self) -> float:
        return TRUTH.in_medium(self.experiment.medium)


@dataclass(frozen=True)
class World:
    """The century, once: every readout and the offsets that produced it."""

    readouts: tuple[Readout, ...]
    generation_offsets: dict[str, float]
    seed: int

    def __getitem__(self, key: str) -> Readout:
        for r in self.readouts:
            if r.experiment.key == key:
                return r
        raise KeyError(key)

    def frame(self) -> pd.DataFrame:
        rows = []
        for r in self.readouts:
            e = r.experiment
            rows.append(
                {
                    "key": e.key,
                    "year": e.year,
                    "who": e.who,
                    "family": e.family,
                    "era": e.era,
                    "medium": e.medium,
                    "radiation": e.radiation,
                    "read": r.read,
                    "read_vacuo": r.read_vacuo,
                    "se": r.se,
                    "ppm": ppm(r.read_vacuo),
                    "ppm_se": r.se / C_TRUE * 1e6,
                    "generation": e.generation,
                    "family_offset": r.family_offset,
                    "own_offset": r.own_offset,
                    "published": e.reported,
                }
            )
        return pd.DataFrame(rows)


def generation_offsets(seed: int = 299, truth: Truth = TRUTH) -> dict[str, float]:
    """The systematic each generation of instrument carries, drawn once per world."""
    rng = np.random.default_rng(seed)
    offsets: dict[str, float] = {}
    for generation, sd in truth.generation_sd.items():
        centre = truth.bandwagon if generation in truth.bandwagon_generations else 0.0
        offsets[generation] = float(rng.normal(centre, sd))
    return offsets


def simulate(seed: int = 299, truth: Truth = TRUTH, *, season_seed: int = 1926) -> World:
    """Run the century once.

    A systematic is drawn once per generation of instrument; each experiment
    adds its own systematic and noise at its reported standard error. The
    bandwagon generations are drawn around ``truth.bandwagon`` rather than
    around zero.

    One experiment is not drawn but *run*: Michelson's 1926 season is the
    reading-by-reading simulation ``mount_wilson`` produces, reduced at the
    re-surveyed baseline to a mean and an error budget the way notebook 3
    reduces it, so that the number the corpus holds for 1926 is the number the
    programme filed.
    """
    offsets = generation_offsets(seed, truth)
    rng = np.random.default_rng(seed + 1)
    readouts = []
    for e in HISTORY:
        fam = offsets[e.generation]
        if e.key == "michelson_1926":
            season = mount_wilson(seed=season_seed, offsets=offsets, truth=truth)
            speeds = season_speeds(season, MOUNT_WILSON_RESURVEY)
            stat_se = float(speeds.std(ddof=1) / math.sqrt(speeds.size))
            read = float(speeds.mean())
            se = math.sqrt(
                stat_se**2
                + (OMEGA_CALIBRATION * truth.c) ** 2
                + (MOUNT_WILSON_RESURVEY.relative_error * truth.c) ** 2
            )
            readouts.append(Readout(e, read, se, fam, read - truth.c - fam))
            continue
        own = float(rng.normal(0.0, truth.own_factor[e.era] * e.reported_se))
        noise = float(rng.normal(0.0, e.reported_se))
        read = truth.in_medium(e.medium) + fam + own + noise
        readouts.append(Readout(e, read, e.reported_se, fam, own))
    return World(tuple(readouts), offsets, seed)


# -- the rotating mirror, as an expression tree ------------------------------------------

#: The reference scales the two logarithms are measured against: one km/s and one km.
C_UNIT = Const(value=1.0, dimension=SPEED)
D_UNIT = Const(value=1.0, dimension=length)
EIGHT_PI = Const(value=8.0 * math.pi, dimension=dimensionless())

#: The blur of the returned image, radians, whatever the baseline.
SEEING = 5.0e-6
#: The angular error the dimming of the return adds, radians per root-km.
DIMMING = 3.36e-5
#: The fastest a mirror could be spun against a tuning fork, revolutions per second.
OMEGA_MAX = 600.0


def angular_noise(baseline_km: npt.ArrayLike) -> Any:
    """``sd(delta)`` for one reading at this baseline: seeing plus dimming."""
    d = np.asarray(baseline_km, dtype=np.float64)
    return np.sqrt(SEEING**2 + DIMMING**2 * d)


def _exp(node: Any) -> Apply:
    return Apply(fn="exp", arg=node)


def _params() -> tuple[Param, Param]:
    log_c = Param(
        name="log_c",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": math.log(299_860.0), "sigma": 0.005}),
    )
    log_d = Param(
        name="log_D",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": math.log(35.385), "sigma": 0.02}),
    )
    return log_c, log_d


def speed_expr(log_c: Param) -> Any:
    return Mul(factors=(C_UNIT, _exp(log_c)))


def baseline_expr(log_d: Param) -> Any:
    return Mul(factors=(D_UNIT, _exp(log_d)))


def model(*, prior_log_d: float | None = None, prior_sd_log_d: float = 0.02) -> ModelSpec:
    """The mean: the residual image displacement, ``8 pi omega D / c - null_angle``.

    Two parameters, both dimensionless logarithms so that the information
    matrix is scale-free:

    ``log_c``   the speed of light, in km/s
    ``log_D``   the baseline, in km

    and two data columns: ``omega`` (revolutions per second) and
    ``null_angle`` (``4 pi / n`` for an ``n``-faced mirror tuned to the next
    face, ``0`` for a plane mirror). The residual scale ``sigma`` is fixed at
    one; the apparatus's own noise is passed to the design functions as
    ``noise_sd`` (see ``angular_noise``).
    """
    log_c, log_d = _params()
    if prior_log_d is not None:
        log_d = log_d.model_copy(
            update={
                "prior": Prior(family="normal", hyper={"mu": prior_log_d, "sigma": prior_sd_log_d})
            }
        )
    omega = Data(name="omega", dimension=PER_SECOND)
    null_angle = Data(name="null_angle", dimension=dimensionless())
    deflection = Mul(
        factors=(
            EIGHT_PI,
            omega,
            Div(numerator=baseline_expr(log_d), denominator=speed_expr(log_c)),
        )
    )
    mean = Add(
        terms=(deflection, Mul(factors=(Const(value=-1.0, dimension=dimensionless()), null_angle)))
    )
    return ModelSpec(
        name="rotating_mirror",
        mean=mean,
        outcome=Data(name="delta", dimension=dimensionless()),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(
            log_c,
            log_d,
            Param(
                name="sigma",
                dimension=dimensionless(),
                prior=Prior(family="fixed", hyper={"value": 1.0}),
            ),
        ),
    )


@dataclass(frozen=True)
class Mirror:
    """The rotating-mirror surface: a ``SupportsForward`` over one hand-built tree.

    ``forward`` is the interpreter on ``expr`` and nothing else; ``linearize``
    is ``surface.design_matrix`` with no linear parameters, which is the
    truthful answer -- the mean is nonlinear in both -- so every Jacobian the
    design functions use comes from derivatives of ``forward``.
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

    def speed(
        self, dose: Mapping[str, npt.ArrayLike], delta: npt.ArrayLike, baseline_km: float
    ) -> Array:
        """Invert ``forward`` for ``c``: the speed each residual displacement implies.

        ``forward`` is linear in ``1 / c`` at fixed ``omega`` and ``D``, so the
        inverse is the same tree read backwards: ``8 pi omega D / (delta + null_angle)``.
        The notebooks check it against ``forward`` rather than trusting it.
        """
        omega = np.asarray(dose["omega"], dtype=np.float64)
        null = np.asarray(dose["null_angle"], dtype=np.float64)
        return 8.0 * math.pi * omega * baseline_km / (np.asarray(delta, dtype=np.float64) + null)


def surface(**kwargs: Any) -> Mirror:
    """The one surface every rotating-mirror notebook uses."""
    return Mirror(model(**kwargs))


def apparatus(
    omega: npt.ArrayLike, faces: int | None = None, *, n: int | None = None
) -> dict[str, Array]:
    """The data mapping ``forward`` reads: rotation rate and the null angle of the mirror."""
    w = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if n is not None:
        w = np.broadcast_to(w, (n,)).astype(np.float64).copy()
    null = 4.0 * math.pi / faces if faces else 0.0
    return {"omega": w, "null_angle": np.full(w.shape, null, dtype=np.float64)}


def point(baseline_km: float, c: float = C_TRUE) -> dict[str, float]:
    """The parameter point for an apparatus at this baseline, in this world."""
    return {"log_c": math.log(c), "log_D": math.log(baseline_km)}


def null_omega(baseline_km: float, faces: int, c: float = C_TRUE) -> float:
    """The rotation rate at which the next face is in position when the light returns."""
    return c / (2.0 * faces * baseline_km)


def survey_observation(baseline_km: float, relative_error: float, *, label: str = "survey") -> Any:
    """The baseline survey as a second observation: ``D`` itself, to the surveyor's precision."""
    from axiom.design import Observation

    _, log_d = _params()
    return Observation(
        label=label, expr=baseline_expr(log_d), data={}, noise_sd=baseline_km * relative_error
    )


class BaselineSurvey(Spec):
    """A surveyed baseline: what the distant mirror stands from the spinning one, and how well."""

    length_km: float
    relative_error: float
    surveyed_by: str
    year: int

    @property
    def error_km(self) -> float:
        return self.length_km * self.relative_error


#: The two baselines the case study designs and runs. The 1879 tape survey and
#: the 1924 triangulation by the U.S. Coast and Geodetic Survey.
ANNAPOLIS = BaselineSurvey(
    length_km=0.60540, relative_error=1.0e-5, surveyed_by="steel tape, Michelson", year=1879
)
MOUNT_WILSON = BaselineSurvey(
    length_km=35.38553,
    relative_error=1.4e-7,
    surveyed_by="U.S. Coast and Geodetic Survey",
    year=1924,
)
MOUNT_WILSON_RESURVEY = BaselineSurvey(
    length_km=35.38548,
    relative_error=1.4e-7,
    surveyed_by="U.S. Coast and Geodetic Survey",
    year=1926,
)

#: The tuning-fork calibration of ``omega``: a relative error the reading cannot see.
OMEGA_CALIBRATION = 1.0e-5

#: What Michelson published from the first Mount Wilson season, 1924: the value the
#: 1926 season was checking, and the null the running mean is read against.
PRELIMINARY_1924 = 299_802.0


# -- Mount Wilson, night by night ------------------------------------------------------

#: How the 1926 season was run: nights, readings a night, and the mirrors tried.
NIGHTS = 40
READINGS_PER_NIGHT = 40
CONFIGURATIONS: tuple[str, ...] = ("glass-8", "steel-8", "steel-12", "glass-16")
FACES_OF: dict[str, int] = {"glass-8": 8, "steel-8": 8, "steel-12": 12, "glass-16": 16}
#: Night-to-night systematic (refraction, temperature of the fork), km/s.
NIGHT_SD = 1.5
#: What a mirror configuration adds of its own, km/s: a mis-figured face is a bias.
CONFIGURATION_SD = 0.8
#: The seed of the block assignment of nights to mirrors; the rule re-derives from it.
PLAN_SEED = 1926


def nights() -> tuple[str, ...]:
    return tuple(f"night-{i:02d}" for i in range(NIGHTS))


def assign_nights() -> Any:
    """Forty nights into four mirrors in permuted blocks: ten each, and a rule that travels.

    The assignment is design work and ``axiom.design.assign`` does it; the
    module only fixes the seed so that ``simulate`` and notebook 3 run the same
    season.
    """
    from axiom.design import ArmAllocation, assign

    return assign(nights(), ArmAllocation.equal(*CONFIGURATIONS), method="block", seed=PLAN_SEED)


def mount_wilson(
    *,
    seed: int = 1926,
    survey: BaselineSurvey = MOUNT_WILSON,
    configuration_of_night: Sequence[str] | None = None,
    offsets: Mapping[str, float] | None = None,
    truth: Truth = TRUTH,
) -> pd.DataFrame:
    """Every reading of the 1926 season, one row each, through ``forward``.

    ``configuration_of_night`` is the arm each night was run in; by default it
    is ``assign_nights()``, which is what notebook 3 files as the plan. Each night draws its own
    systematic, each configuration a bias of its own, and each reading the
    angular noise of a 35 km baseline. Two things the season cannot see are in
    it as well: the Mount Wilson generation's own systematic, and one
    calibration error of the tuning fork that every recorded ``omega`` shares
    (``OMEGA_CALIBRATION``) -- the light was timed against the true rate, the
    notebook reads the recorded one.
    """
    rng = np.random.default_rng(seed)
    if configuration_of_night is None:
        assigned = assign_nights()
        configuration_of_night = tuple(assigned.arm(n) for n in nights())
    configs = tuple(configuration_of_night)
    if len(configs) != NIGHTS:
        raise ValueError(f"{len(configs)} configurations for {NIGHTS} nights")
    generation = (offsets or generation_offsets(truth=truth))["michelson-ii"]
    c_world = truth.c + generation
    fork = 1.0 + float(rng.normal(0.0, OMEGA_CALIBRATION))
    config_bias = {k: float(rng.normal(0.0, CONFIGURATION_SD)) for k in CONFIGURATIONS}
    surf = surface()
    rows = []
    for night, config in enumerate(configs):
        faces = FACES_OF[config]
        c_night = c_world + float(rng.normal(0.0, NIGHT_SD)) + config_bias[config]
        omega = null_omega(survey.length_km, faces, c_night) * (1.0 + rng.normal(0.0, 2e-5))
        data = apparatus(omega, faces, n=READINGS_PER_NIGHT)
        expected = surf.forward(data, point(survey.length_km, c_night))
        noise = rng.normal(0.0, float(angular_noise(survey.length_km)), size=READINGS_PER_NIGHT)
        delta = expected + noise
        for j in range(READINGS_PER_NIGHT):
            rows.append(
                {
                    "night": night,
                    "reading": j,
                    "configuration": config,
                    "faces": faces,
                    "omega": float(omega * fork),
                    "delta": float(delta[j]),
                }
            )
    return pd.DataFrame(rows)


def season_speeds(frame: pd.DataFrame, survey: BaselineSurvey) -> Array:
    """The speed each reading implies, through ``Mirror.speed`` at the surveyed baseline."""
    data = {
        "omega": frame["omega"].to_numpy(),
        "null_angle": 4.0 * math.pi / frame["faces"].to_numpy(),
    }
    return surface().speed(data, frame["delta"].to_numpy(), survey.length_km)


# -- the house style -------------------------------------------------------------------

# The shared notebook style: one registered plotly template and one validated palette
# for every notebook under nbs/. Imported for its side effect (the template) and for the
# colours below, so this case study cannot drift into a look of its own.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_NBS = str(_Path(__file__).resolve().parents[2])
if _NBS not in _sys.path:
    _sys.path.insert(0, _NBS)

from _style import (  # noqa: E402
    AQUA,
    AXIS,
    BLUE,
    CRITICAL,
    GRID,
    INK,
    MUTED,
    ORANGE,
    SUBTLE,
    SURFACE,
    VIOLET,
    YELLOW,
)

#: One hue per instrument family, in the palette's validated order.
FAMILY_COLOR: dict[str, str] = {
    "toothed-wheel": ORANGE,
    "rotating-mirror": BLUE,
    "kerr-cell": VIOLET,
    "electromagnetic": AQUA,
}
TRUTH_COLOR = MUTED
ACCENT = YELLOW

__all__ = [
    "ACCENT",
    "ANNAPOLIS",
    "AQUA",
    "AXIS",
    "BLUE",
    "BY_KEY",
    "CONFIGURATIONS",
    "CRITICAL",
    "C_AIR_TRUE",
    "C_TRUE",
    "DIMMING",
    "FACES_OF",
    "FAMILIES",
    "FAMILY_COLOR",
    "FAMILY_LABEL",
    "FIZEAU_LEAGUES",
    "GRID",
    "HISTORY",
    "INK",
    "LEAGUE_KM",
    "MOUNT_WILSON",
    "MOUNT_WILSON_RESURVEY",
    "MUTED",
    "NIGHTS",
    "NIGHT_SD",
    "PLAN_SEED",
    "PRELIMINARY_1924",
    "N_AIR",
    "OMEGA_CALIBRATION",
    "OMEGA_MAX",
    "ORANGE",
    "READINGS_PER_NIGHT",
    "SEEING",
    "SPEED",
    "SUBTLE",
    "SURFACE",
    "TRANSIT",
    "TRUTH",
    "TRUTH_COLOR",
    "VIOLET",
    "BaselineSurvey",
    "Experiment",
    "Mirror",
    "Readout",
    "Truth",
    "World",
    "angular_noise",
    "apparatus",
    "assign_nights",
    "estimand",
    "from_ppm",
    "generation_offsets",
    "ledger_frame",
    "length",
    "model",
    "mount_wilson",
    "nights",
    "null_omega",
    "point",
    "ppm",
    "programs",
    "season_speeds",
    "simulate",
    "surface",
    "survey_observation",
    "to_vacuo",
]
