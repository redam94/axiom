"""``StudyBuilder``: one experiment's plan → ``design`` specs and a ``calibrate.Measurement``.

Rewritten from the parent's ``builders/model.py`` (ledger: REWRITE). A
study is described once — its size, its method, its schedule, its economic
inputs — and built into whichever typed object the next step needs:

* ``build()`` / ``build_simulation()`` → ``design.SimulationSpec`` (the
  panel the power simulation draws);
* ``build_candidate()`` → ``design.DesignCandidate`` (what the optimizer
  scores; needs ``experiment_se`` and ``cost``);
* ``build_schedule()`` → ``design.Schedule`` through the named generator
  (``constant`` / ``pulse`` / ``alternating`` / ``ramp`` /
  ``random_switchback``);
* ``build_measurement()`` → ``calibrate.Measurement`` once the study has
  been run and reports ``(estimate, se)`` for an ``Estimand``.

Each result is a ``Spec`` and round-trips; the builder itself is immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from axiom.build.base import BuildError, Fields
from axiom.calibrate import Measurement
from axiom.core import Assumption
from axiom.design import (
    METHODS,
    DesignCandidate,
    Schedule,
    SimulationSpec,
    alternating,
    constant,
    pulse,
    ramp,
    random_switchback,
)
from axiom.estimands import Estimand

__all__ = ["SchedulePattern", "StudyBuilder"]

SchedulePattern = Literal["constant", "pulse", "alternating", "ramp", "random_switchback"]

_SIMULATION_FIELDS = frozenset(SimulationSpec.model_fields)
_CANDIDATE_FIELDS = frozenset(DesignCandidate.model_fields)


@dataclass(frozen=True)
class StudyBuilder:
    """Collect a study's plan; build the design spec, candidate, schedule, or measurement."""

    fields: Fields = Fields()

    def with_(self, **updates: Any) -> StudyBuilder:
        return replace(self, fields=self.fields.with_(**updates))

    # -- identity -----------------------------------------------------------------------

    def name(self, name: str) -> StudyBuilder:
        return self.with_(name=name)

    def method(self, method: str) -> StudyBuilder:
        """A ``design.methods`` registry name (``"difference_in_differences"``, ...)."""
        if method not in METHODS:
            raise BuildError(f"unknown method {method!r}; registered: {sorted(METHODS)}")
        return self.with_(method=method)

    # -- size and variance --------------------------------------------------------------

    def size(
        self,
        *,
        n_units: int | None = None,
        n_periods: int | None = None,
        n_pre: int | None = None,
        n_treated: int | None = None,
        n_clusters: int | None = None,
    ) -> StudyBuilder:
        return self.with_(
            n_units=n_units,
            n_periods=n_periods,
            n_pre=n_pre,
            n_treated=n_treated,
            n_clusters=n_clusters,
        )

    def variance(
        self,
        *,
        unit_sd: float | None = None,
        noise_sd: float | None = None,
        period_sd: float | None = None,
        rho: float | None = None,
    ) -> StudyBuilder:
        return self.with_(unit_sd=unit_sd, noise_sd=noise_sd, period_sd=period_sd, rho=rho)

    def effect(self, effect: float) -> StudyBuilder:
        return self.with_(effect=float(effect))

    def simulations(self, n_simulations: int, *, seed: int | None = None) -> StudyBuilder:
        return self.with_(n_simulations=int(n_simulations), seed=seed)

    def mass(self, mass: float) -> StudyBuilder:
        return self.with_(mass=float(mass))

    def switchback(self, on_fraction: float) -> StudyBuilder:
        return self.with_(on_fraction=float(on_fraction))

    def exposure(self, exposure_rate: float) -> StudyBuilder:
        return self.with_(exposure_rate=float(exposure_rate))

    # -- economics (candidate) ----------------------------------------------------------

    def holdout(self, fraction: float) -> StudyBuilder:
        return self.with_(holdout_fraction=float(fraction))

    def precision(self, experiment_se: float) -> StudyBuilder:
        """The standard error the design achieves on the decision parameter."""
        return self.with_(experiment_se=float(experiment_se))

    def cost(self, cost: float, *, cooldown_periods: int | None = None) -> StudyBuilder:
        return self.with_(cost=float(cost), cooldown_periods=cooldown_periods)

    # -- schedule -----------------------------------------------------------------------

    def schedule(
        self, pattern: SchedulePattern, *, treatment: str = "dose", **params: Any
    ) -> StudyBuilder:
        """Name the dose schedule generator and its parameters (``high``, ``low``, ``on``,
        ``off``, ``start``, ``stop``, ``dose``, ``seed``, ``p_high``). ``n_periods`` comes
        from ``size`` unless given here."""
        if pattern not in ("constant", "pulse", "alternating", "ramp", "random_switchback"):
            raise BuildError(f"unknown schedule pattern {pattern!r}")
        return self.with_(
            schedule_pattern=pattern, schedule_treatment=treatment, schedule_params=dict(params)
        )

    # -- measurement --------------------------------------------------------------------

    def measured(
        self,
        estimand: Estimand,
        estimate: float,
        se: float,
        *,
        source: str,
        definition: Literal["wald", "eti", "hdi"] = "wald",
        design_factor: float | None = None,
        assumptions: tuple[Assumption, ...] = (),
    ) -> StudyBuilder:
        """What the study found: ``(estimate, se)`` of ``estimand`` from ``source``."""
        return self.with_(
            estimand=estimand,
            estimate=float(estimate),
            se=float(se),
            source=source,
            definition=definition,
            design_factor=design_factor,
            assumptions=assumptions,
        )

    # -- builds -------------------------------------------------------------------------

    def build(self) -> SimulationSpec:
        return self.build_simulation()

    def build_simulation(self) -> SimulationSpec:
        payload = {k: v for k, v in self.fields.to_dict().items() if k in _SIMULATION_FIELDS}
        return SimulationSpec(**payload)

    def build_candidate(self) -> DesignCandidate:
        self.fields.require(
            "name",
            "method",
            "n_units",
            "n_periods",
            "holdout_fraction",
            "experiment_se",
            "cost",
            builder="StudyBuilder.build_candidate",
        )
        payload = {k: v for k, v in self.fields.to_dict().items() if k in _CANDIDATE_FIELDS}
        return DesignCandidate(**payload)

    def build_schedule(self) -> Schedule:
        self.fields.require("schedule_pattern", builder="StudyBuilder.build_schedule")
        pattern: str = self.fields.get("schedule_pattern")
        params = dict(self.fields.get("schedule_params", {}))
        n_periods = params.pop("n_periods", self.fields.get("n_periods"))
        if n_periods is None:
            raise BuildError("StudyBuilder.build_schedule: n_periods is not set")
        treatment = self.fields.get("schedule_treatment", "dose")
        try:
            if pattern == "constant":
                return constant(n_periods, params["dose"], treatment=treatment)
            if pattern == "pulse":
                return pulse(
                    n_periods,
                    params["high"],
                    params["low"],
                    on=params["on"],
                    off=params["off"],
                    start_on=params.get("start_on", True),
                    treatment=treatment,
                )
            if pattern == "alternating":
                return alternating(n_periods, params["high"], params["low"], treatment=treatment)
            if pattern == "ramp":
                return ramp(n_periods, params["start"], params["stop"], treatment=treatment)
            return random_switchback(
                n_periods,
                params["high"],
                params["low"],
                seed=params.get("seed", self.fields.get("seed")),
                p_high=params.get("p_high", 0.5),
                treatment=treatment,
            )
        except KeyError as e:
            raise BuildError(f"schedule {pattern!r} needs parameter {e.args[0]!r}") from e

    def build_measurement(self) -> Measurement:
        self.fields.require(
            "estimand", "estimate", "se", "source", builder="StudyBuilder.build_measurement"
        )
        f = self.fields
        return Measurement(
            estimand=f.get("estimand"),
            estimate=f.get("estimate"),
            se=f.get("se"),
            definition=f.get("definition", "wald"),
            mass=f.get("mass", 0.95),
            method=f.get("method", ""),
            n_units=f.get("n_units"),
            n_periods=f.get("n_periods"),
            design_factor=f.get("design_factor"),
            source=f.get("source"),
            assumptions=tuple(f.get("assumptions", ())),
        )
