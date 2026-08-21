"""``SurfaceBuilder``: treatments, kernels, carryover, nuisance, intercept → ``SurfaceSpec``.

Rewritten from the parent's ``builders/model.py`` (ledger: REWRITE — the
*pattern* ports, the ``BayesianMMM``-specific code does not). The builder
collects what ``surface.SurfaceSpec`` needs and hands it over in one
constructor call, so everything the spec validates (distinct names,
interaction pairs, unit labels for a per-unit intercept, the dimension
check of the built model) is validated exactly once, by the spec. The
builder adds only the conveniences: kernels and carryover by family name,
nuisance terms by keyword, one ``treatment`` call per treatment.

Kernels and carryovers are looked up through ``surface.kernel_from_name`` /
``surface.carryover_from_name``; a ``SurfaceSpec`` built here is the same
object a hand-written one would be, and ``build()`` round-trips through
``load_spec``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from axiom.build.base import BuildError, Fields
from axiom.core import D, Dimension, Likelihood, LikelihoodFamily, Outcome, Prior, Treatment
from axiom.surface import (
    AnyNuisanceTerm,
    CarryoverKernel,
    EventIndicators,
    FourierSeasonality,
    InterceptKind,
    LinearTrend,
    NuisanceSet,
    ResponseKernel,
    SurfaceSpec,
    carryover_from_name,
    kernel_from_name,
)

__all__ = ["SurfaceBuilder"]


@dataclass(frozen=True)
class _TreatmentEntry:
    treatment: Treatment
    kernel: ResponseKernel | None
    carryover: CarryoverKernel | None


@dataclass(frozen=True)
class SurfaceBuilder:
    """Fluent construction of a ``SurfaceSpec``; every method returns a new builder.

    ``SurfaceBuilder().name("s").treatment("a", kernel="hill",
    carryover="geometric", max_lag=4, reference_dose=50.0)
    .outcome("y").intercept("hierarchical", units=("u0", "u1")).build()``.
    """

    fields: Fields = Fields()

    def with_(self, **updates: Any) -> SurfaceBuilder:
        """Set raw ``SurfaceSpec`` fields (``time_column``, ``noise_scale``, ...)."""
        return replace(self, fields=self.fields.with_(**updates))

    def name(self, name: str) -> SurfaceBuilder:
        return self.with_(name=name)

    # -- treatments ---------------------------------------------------------------------

    def treatment(
        self,
        treatment: Treatment | str,
        *,
        dimension: Dimension | None = None,
        unit: str | None = None,
        kernel: ResponseKernel | str | None = None,
        carryover: CarryoverKernel | str | None = None,
        max_lag: int | None = None,
        reference_dose: float | None = None,
        amplitude_scale: float | None = None,
        amplitude_prior: Prior | None = None,
        **kernel_fields: Any,
    ) -> SurfaceBuilder:
        """Add one treatment with its kernel and carryover.

        A string ``treatment`` becomes ``Treatment(name, dimension, unit)``
        with ``dimension`` defaulting to currency (the common dose
        dimension; pass ``D.outcome``, a declared base, or any
        ``Dimension`` otherwise). ``kernel`` is a family name (``"hill"``,
        ``"logistic"``, ``"exponential"``, ``"power"``, ``"linear"``) or a
        kernel spec; ``reference_dose`` / ``amplitude_scale`` /
        ``amplitude_prior`` and any extra ``kernel_fields`` go to the
        named family. ``carryover`` is a family name (``"geometric"``,
        ``"delayed"``, ``"weibull"``, ``"none"``) or a carryover spec;
        ``max_lag`` goes to the named family. Omitting a kernel or a
        carryover leaves the spec's default (``HillKernel()`` and
        ``NoCarryover()``).
        """
        if isinstance(treatment, str):
            treatment = Treatment(
                name=treatment,
                dimension=dimension if dimension is not None else D.currency,
                unit=unit,
            )
        elif dimension is not None or unit is not None:
            raise BuildError(
                "SurfaceBuilder.treatment: pass dimension/unit with a name, not with an entity"
            )
        kernel_spec = _kernel(
            kernel, reference_dose, amplitude_scale, amplitude_prior, kernel_fields
        )
        carry_spec = _carryover(carryover, max_lag)
        entries: tuple[_TreatmentEntry, ...] = self.fields.get("treatments", ())
        if any(e.treatment.name == treatment.name for e in entries):
            raise BuildError(f"SurfaceBuilder: treatment {treatment.name!r} was already added")
        return self.with_(
            treatments=(*entries, _TreatmentEntry(treatment, kernel_spec, carry_spec))
        )

    def outcome(
        self,
        outcome: Outcome | str,
        *,
        dimension: Dimension | None = None,
        unit: str | None = None,
        aggregation: str = "sum",
    ) -> SurfaceBuilder:
        """The outcome entity; a string gets ``dimension`` (default ``D.outcome``)."""
        if isinstance(outcome, str):
            outcome = Outcome(
                name=outcome,
                dimension=dimension if dimension is not None else D.outcome,
                unit=unit,
                aggregation=aggregation,  # type: ignore[arg-type]
            )
        return self.with_(outcome=outcome)

    # -- structure ----------------------------------------------------------------------

    def intercept(
        self, kind: InterceptKind, *, units: Iterable[str] = (), scale: float | None = None
    ) -> SurfaceBuilder:
        """``"none"`` / ``"shared"`` / ``"per_unit"`` / ``"hierarchical"``; the last two
        need ``units`` (the panel's unit labels, in sorted order)."""
        out = self.with_(intercept=kind)
        labels = tuple(str(u) for u in units)
        if labels:
            out = out.with_(unit_labels=labels)
        if scale is not None:
            out = out.with_(intercept_scale=float(scale))
        return out

    def units(self, labels: Iterable[str]) -> SurfaceBuilder:
        return self.with_(unit_labels=tuple(str(u) for u in labels))

    def interaction(self, first: str, second: str, *, scale: float | None = None) -> SurfaceBuilder:
        pairs: tuple[tuple[str, str], ...] = self.fields.get("interactions", ())
        out = self.with_(interactions=(*pairs, (first, second)))
        if scale is not None:
            out = out.with_(interaction_scale=float(scale))
        return out

    def nuisance(
        self,
        *,
        fourier: tuple[float, int] | None = None,
        trend: bool | LinearTrend = False,
        events: Sequence[str] = (),
        terms: Sequence[AnyNuisanceTerm] = (),
        coefficient_scale: float = 1.0,
    ) -> SurfaceBuilder:
        """Baseline terms: ``fourier=(period, order)``, ``trend=True``, ``events=(cols,)``,
        plus any explicit ``terms``. Replaces previously set nuisance terms."""
        collected: list[AnyNuisanceTerm] = []
        if fourier is not None:
            period, order = fourier
            collected.append(
                FourierSeasonality(
                    period=float(period), order=int(order), coefficient_scale=coefficient_scale
                )
            )
        if isinstance(trend, LinearTrend):
            collected.append(trend)
        elif trend:
            collected.append(LinearTrend(coefficient_scale=coefficient_scale))
        if events:
            collected.append(
                EventIndicators(events=tuple(events), coefficient_scale=coefficient_scale)
            )
        collected.extend(terms)
        return self.with_(nuisance=NuisanceSet(terms=tuple(collected)))

    def likelihood(
        self,
        family: LikelihoodFamily,
        *,
        scale: str | None = "sigma",
        noise_scale: float = 1.0,
        df: float | None = None,
    ) -> SurfaceBuilder:
        """The outcome likelihood; ``df`` is the Student-t degrees of freedom."""
        fields: dict[str, Any] = {"family": family, "scale": scale}
        if df is not None:
            fields["df"] = float(df)
        out = self.with_(likelihood=Likelihood(**fields))
        return out.with_(noise_scale=float(noise_scale))

    # -- build --------------------------------------------------------------------------

    def build(self) -> SurfaceSpec:
        self.fields.require("treatments", "outcome", builder="SurfaceBuilder")
        entries: tuple[_TreatmentEntry, ...] = self.fields.get("treatments")
        kernels = {e.treatment.name: e.kernel for e in entries if e.kernel is not None}
        carryover = {e.treatment.name: e.carryover for e in entries if e.carryover is not None}
        payload: dict[str, Any] = {
            k: v for k, v in self.fields.to_dict().items() if k not in ("treatments",)
        }
        payload.setdefault("name", "surface")
        payload["treatments"] = tuple(e.treatment for e in entries)
        payload["kernels"] = kernels
        payload["carryover"] = carryover
        return SurfaceSpec(**payload)


def _kernel(
    kernel: ResponseKernel | str | None,
    reference_dose: float | None,
    amplitude_scale: float | None,
    amplitude_prior: Prior | None,
    extra: dict[str, Any],
) -> ResponseKernel | None:
    given = {k: v for k, v in extra.items()}
    if reference_dose is not None:
        given["reference_dose"] = float(reference_dose)
    if amplitude_scale is not None:
        given["amplitude_scale"] = float(amplitude_scale)
    if amplitude_prior is not None:
        given["amplitude_prior"] = amplitude_prior
    if kernel is None:
        if given:
            kernel = "hill"  # kernel fields without a family: the default family
        else:
            return None
    if isinstance(kernel, str):
        return kernel_from_name(kernel, **given)
    if given:
        raise BuildError("SurfaceBuilder.treatment: kernel fields go with a family name")
    return kernel


def _carryover(
    carryover: CarryoverKernel | str | None, max_lag: int | None
) -> CarryoverKernel | None:
    if carryover is None:
        if max_lag is not None:
            carryover = "geometric"  # a lag without a family: the default family
        else:
            return None
    if isinstance(carryover, str):
        fields: dict[str, Any] = {}
        if max_lag is not None:
            fields["max_lag"] = int(max_lag)
        return carryover_from_name(carryover, **fields)
    if max_lag is not None:
        raise BuildError("SurfaceBuilder.treatment: max_lag goes with a carryover family name")
    return carryover
