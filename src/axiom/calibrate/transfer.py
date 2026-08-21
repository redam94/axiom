"""Transfer correction operators and ``resolve``: reading a measurement as a different estimand.

``Estimand.transfer_to`` says *which* facets differ and names the licensing
assumption for each; it does not produce a number. The operators here do.
Each is a pure function returning a ``Correction``: a factor (or offset, or
SE multiplier) together with the ``LedgerLine`` that records it — the
assumption it rests on, the reading **before** (``detail["counterfactual"]``)
and **after** (``detail["value"]``) the correction, as strings — so that no
corrected number can exist without its provenance (rule 4).

Operators (provenance: the chord/marginal and adstock-window corrections
are the parent's calibration bridging logic, ported by specification; the
variance formulas are textbook Kish design effects):

* ``chord_to_marginal`` — an experiment contrasting doses ``d0`` and ``d1``
  measures the **chord** ``(f(d1) − f(d0)) / (d1 − d0)``; a ``marginal``
  estimand wants ``f'(at)``. Both are evaluated through the one
  ``forward()`` (``surface.forward`` and ``surface.marginal``) on the
  steady-state surface (0002.19), never a re-implemented Hill. Facet:
  ``intervention``.
* ``carryover_window_factor`` — a measurement taken over ``w`` periods
  after a dose sees only the share ``Σ_{l<w} w_l`` of the carryover mass;
  ``1 / share`` scales it up to the steady-state total. Facet: ``window``.
* ``dose_path_accumulation`` — the effective (carried) dose a dose path
  delivered inside its window, ``Σ_t (w ∗ x)_t``, against the raw
  ``Σ_t x_t``; the convolution is ``core.causal_convolve`` with the
  kernel's own weights. Facet: ``intervention``.
* ``variance_reweight`` — an SE for a different aggregation size:
  ``se · sqrt(n_source / n_target) · sqrt(deff)`` with the Kish design
  effect ``deff = 1 + (m − 1)·icc`` for the target's clusters of size
  ``m`` (``deff = 1`` when no ``icc`` is given). Facet: ``level``.
* ``aggregation_level`` — individual ↔ cluster ↔ aggregate. The mean
  effect *per unit* is invariant under aggregation of an additive effect
  (point factor 1, under the ``linear_aggregation`` assumption); the SE is
  not, and that part is delegated to ``variance_reweight``. A difference
  in interference structure is ``Unsupported``. Facet: ``level``.

``resolve`` folds a ``TransferPlan``, an optional transport ``Verdict`` and
a sequence of corrections into a ``ResolvedTransfer``: the composite
``factor`` / ``offset`` / ``se_scale``, the ledger, and the completeness
verdict. The ledger starts from ``Ledger.from_plan`` (every differing facet
with the ``UNCORRECTED`` counterfactual); each correction's line then
**replaces** the plan line for its facet — several corrections for one
facet are folded into one line whose detail lists them — so the
"exactly one line per differing facet" rule of ``Ledger.check_complete``
holds by construction, and a facet no operator touched is still visible,
as an explicitly uncorrected one. A transport verdict adds a ``transport``
line and, when it is ``identified``, marks the population facet's
assumption satisfied.

``ResolvedTransfer.apply(estimate, se)`` is ``(estimate·factor + offset,
se·factor·se_scale)``: a multiplicative correction rescales the SE with the
estimate; an additive one does not; ``se_scale`` corrections touch only
the SE.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.calibrate.ledger import FACET_PREFIX, Ledger, facet_of
from axiom.core import Assumption, LedgerLine, Spec, Unsupported, Verdict, value
from axiom.core import causal_convolve as _causal_convolve
from axiom.core.verdict import Status
from axiom.estimands import Level, TransferPlan
from axiom.surface import CarryoverKernel, Surface, forward, marginal

__all__ = [
    "Correction",
    "CorrectionKind",
    "ResolvedTransfer",
    "aggregation_level",
    "carryover_window_factor",
    "chord_to_marginal",
    "dose_path_accumulation",
    "resolve",
    "variance_reweight",
]

CorrectionKind = Literal["multiplicative", "additive", "se_scale"]
Theta = Mapping[str, npt.ArrayLike]


class Correction(Spec):
    """One correction operator's output: the number applied and the line that records it.

    ``value`` is what ``resolve`` composes — a factor on the estimate
    (``multiplicative``), an offset (``additive``) or a factor on the SE
    (``se_scale``). ``counterfactual`` and ``corrected`` are the quantity
    the operator reasoned about, before and after (chord and marginal
    slope; the SE before and after; the share of carryover mass seen and
    the full unit mass). ``ledger_line`` carries the same two numbers as
    strings in ``detail["counterfactual"]`` / ``detail["value"]``, a typed
    ``assumption`` and ``kind == "facet:<facet>"``; the validator refuses
    anything less.
    """

    name: str = Field(min_length=1)
    kind: CorrectionKind
    value: float
    counterfactual: float
    corrected: float
    detail: dict[str, str] = {}
    ledger_line: LedgerLine

    @model_validator(mode="after")
    def _well_formed(self) -> Correction:
        for field in ("value", "counterfactual", "corrected"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"correction {self.name!r}: {field} must be finite")
        if self.kind in ("multiplicative", "se_scale") and self.value <= 0.0:
            raise ValueError(
                f"correction {self.name!r}: a {self.kind} factor must be strictly positive"
            )
        line = self.ledger_line
        if line.assumption is None:
            raise ValueError(f"correction {self.name!r}: its ledger line names no assumption")
        if facet_of(line) is None:
            raise ValueError(f"correction {self.name!r}: its ledger line must be a facet line")
        for key in ("counterfactual", "value"):
            if key not in line.detail:
                raise ValueError(f"correction {self.name!r}: ledger detail lacks {key!r}")
        return self

    @property
    def facet(self) -> str:
        f = facet_of(self.ledger_line)
        assert f is not None  # validator
        return f


def _line(
    *,
    name: str,
    facet: str,
    statement: str,
    assumption: Assumption,
    before: float,
    after: float,
    factor: float,
    detail: Mapping[str, str],
) -> LedgerLine:
    return LedgerLine(
        kind=f"{FACET_PREFIX}{facet}",
        statement=statement,
        assumption=assumption,
        detail={
            "status": "corrected",
            "facet": facet,
            "correction": name,
            "counterfactual": repr(float(before)),
            "value": repr(float(after)),
            "factor": repr(float(factor)),
            **dict(detail),
        },
    )


def _weights(carryover: CarryoverKernel, theta: Theta, treatment: str) -> npt.NDArray[np.float64]:
    """The kernel's normalized weights at ``theta``, through the interpreter (rule 3)."""
    w = np.atleast_1d(np.asarray(value(carryover.weights(treatment), params=theta), dtype=float))
    if w.ndim != 1:
        raise ValueError(
            f"carryover weights for {treatment!r} must be a vector at a point theta; "
            f"got shape {w.shape} (pass scalars, not draws)"
        )
    return np.asarray(w, dtype=np.float64)


# -- operators ---------------------------------------------------------------------------


def chord_to_marginal(
    surface: Surface,
    theta: Theta,
    treatment: str,
    d0: float,
    d1: float,
    at: float,
    *,
    held: Mapping[str, npt.ArrayLike] | None = None,
) -> Correction | Unsupported:
    """``factor = f'(at) / ((f(d1) − f(d0)) / (d1 − d0))``: read a chord as a marginal.

    Evaluated on ``surface.steady_state()`` whenever the spec declares
    carryover (0002.19) so the three doses are independent rows. ``held``
    supplies every other data column the mean reads (other treatments'
    doses, a unit index), each broadcast to the three rows. ``Unsupported``
    when ``d1 == d0`` or the chord is numerically zero — there is nothing
    to divide by, and a number would be a lie.
    """
    if not (math.isfinite(d0) and math.isfinite(d1) and math.isfinite(at)):
        raise ValueError("doses must be finite")
    if d0 == d1:
        return Unsupported(
            reason="chord_to_marginal: d0 == d1 — a chord needs two distinct doses",
            missing=("distinct doses",),
            detail={"d0": repr(d0), "d1": repr(d1)},
        )
    steady = surface.steady_state() if surface.spec.carried else surface
    dose: dict[str, npt.ArrayLike] = {treatment: np.array([d0, d1, at], dtype=float)}
    for name, v in (held or {}).items():
        arr = np.asarray(v)
        dose[name] = np.broadcast_to(arr, (3,)) if arr.ndim == 0 else arr
    f = np.asarray(forward(steady, dose, theta), dtype=float).reshape(-1)
    g = np.asarray(marginal(steady, theta, dose, treatment), dtype=float).reshape(-1)
    chord = float((f[1] - f[0]) / (d1 - d0))
    slope = float(g[2])
    if not (math.isfinite(chord) and math.isfinite(slope)):
        return Unsupported(
            reason="chord_to_marginal: the surface is not finite at the requested doses",
            detail={"chord": repr(chord), "marginal": repr(slope)},
        )
    if abs(chord) <= 1e-12 * max(1.0, abs(slope)):
        return Unsupported(
            reason="chord_to_marginal: the chord is numerically zero; the factor is undefined",
            detail={"chord": repr(chord), "marginal": repr(slope)},
        )
    factor = slope / chord
    if factor <= 0.0:
        return Unsupported(
            reason=(
                "chord_to_marginal: chord and marginal have opposite signs; a multiplicative "
                "correction cannot bridge them"
            ),
            detail={"chord": repr(chord), "marginal": repr(slope)},
        )
    assumption = Assumption(
        name="surface_correct_between_doses",
        facet="intervention",
        statement=(
            f"the response surface is correct between doses {d0} and {d1} and at {at} "
            f"for treatment {treatment!r}"
        ),
        challenged_by="curvature; the chord-versus-marginal correction",
        detail={"d0": repr(d0), "d1": repr(d1), "at": repr(at)},
    )
    detail = {
        "treatment": treatment,
        "d0": repr(d0),
        "d1": repr(d1),
        "at": repr(at),
        "chord": repr(chord),
        "marginal": repr(slope),
        "surface": steady.spec.name,
    }
    return Correction(
        name="chord_to_marginal",
        kind="multiplicative",
        value=factor,
        counterfactual=chord,
        corrected=slope,
        detail=detail,
        ledger_line=_line(
            name="chord_to_marginal",
            facet="intervention",
            statement=(
                f"chord over [{d0}, {d1}] read as the marginal at {at}: " f"factor {factor:.6g}"
            ),
            assumption=assumption,
            before=chord,
            after=slope,
            factor=factor,
            detail=detail,
        ),
    )


def carryover_window_factor(
    carryover: CarryoverKernel,
    theta: Theta,
    window_periods: int,
    *,
    treatment: str,
) -> Correction | Unsupported:
    """``factor = 1 / Σ_{l < window_periods} w_l``: a short-window total scaled to steady state.

    ``NoCarryover`` gives share 1 and factor 1 — still a ``Correction`` with
    its line, because "no carryover" is itself the assumption being made.
    ``Unsupported`` when no weight falls inside the window.
    """
    if window_periods < 1:
        raise ValueError(f"window_periods must be >= 1, got {window_periods}")
    w = _weights(carryover, theta, treatment)
    share = float(w[:window_periods].sum())
    total = float(w.sum())
    if not math.isfinite(share) or share <= 0.0:
        return Unsupported(
            reason=(
                f"carryover_window_factor: no carryover mass within the first {window_periods} "
                "period(s); the window does not see the effect"
            ),
            detail={"share": repr(share), "max_lag": str(carryover.max_lag)},
        )
    factor = total / share
    assumption = Assumption(
        name="carryover_contained",
        facet="window",
        statement=(
            f"the {carryover.name!r} carryover (max_lag {carryover.max_lag}) describes how the "
            f"effect of a dose spills past a {window_periods}-period window"
        ),
        challenged_by="half-life against window length",
        detail={"share_within_window": repr(share)},
    )
    detail = {
        "carryover": carryover.name,
        "max_lag": str(carryover.max_lag),
        "window_periods": str(window_periods),
        "share_within_window": repr(share),
        "treatment": treatment,
    }
    return Correction(
        name="carryover_window_factor",
        kind="multiplicative",
        value=factor,
        counterfactual=share,
        corrected=total,
        detail=detail,
        ledger_line=_line(
            name="carryover_window_factor",
            facet="window",
            statement=(
                f"{share:.6g} of the carryover mass lands within {window_periods} period(s); "
                f"scaled to the steady-state total by {factor:.6g}"
            ),
            assumption=assumption,
            before=share,
            after=total,
            factor=factor,
            detail=detail,
        ),
    )


def dose_path_accumulation(
    carryover: CarryoverKernel,
    theta: Theta,
    dose_path: npt.ArrayLike,
    *,
    treatment: str,
) -> Correction | Unsupported:
    """``factor = Σ_t (w ∗ x)_t / Σ_t x_t``: the effective dose delivered inside the path's window.

    The convolution is ``core.causal_convolve`` with the kernel's own
    normalized weights; carried dose that spills past the end of the path
    is what makes the factor fall below one. ``Unsupported`` for an
    all-zero path.
    """
    x = np.asarray(dose_path, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("dose_path must be a non-empty finite vector")
    if np.any(x < 0.0):
        raise ValueError("dose_path must be non-negative")
    raw = float(x.sum())
    if raw <= 0.0:
        return Unsupported(
            reason="dose_path_accumulation: the dose path is all zero; no dose was delivered",
            detail={"n_periods": str(x.size)},
        )
    w = _weights(carryover, theta, treatment)
    effective = float(_causal_convolve(x, w).sum())
    factor = effective / raw
    assumption = Assumption(
        name="intervention_timing_irrelevant",
        facet="intervention",
        statement=(
            f"the effect of the delivered dose path depends on its timing only through the "
            f"{carryover.name!r} carryover"
        ),
        challenged_by="carryover shape",
        detail={"n_periods": str(x.size)},
    )
    detail = {
        "carryover": carryover.name,
        "max_lag": str(carryover.max_lag),
        "n_periods": str(x.size),
        "raw_dose": repr(raw),
        "effective_dose": repr(effective),
        "treatment": treatment,
    }
    return Correction(
        name="dose_path_accumulation",
        kind="multiplicative",
        value=factor,
        counterfactual=raw,
        corrected=effective,
        detail=detail,
        ledger_line=_line(
            name="dose_path_accumulation",
            facet="intervention",
            statement=(
                f"raw dose {raw:.6g} over {x.size} period(s) delivered an effective "
                f"{effective:.6g} inside the window: factor {factor:.6g}"
            ),
            assumption=assumption,
            before=raw,
            after=effective,
            factor=factor,
            detail=detail,
        ),
    )


def _design_effect(icc: float | None, cluster_size: int | None) -> float:
    if (icc is None) != (cluster_size is None):
        raise ValueError("icc and cluster_size must be given together")
    if icc is None or cluster_size is None:
        return 1.0
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must lie in [0, 1], got {icc}")
    if cluster_size < 1:
        raise ValueError(f"cluster_size must be >= 1, got {cluster_size}")
    return 1.0 + (cluster_size - 1) * icc


def variance_reweight(
    se: float,
    n_source: int,
    n_target: int,
    *,
    icc: float | None = None,
    cluster_size: int | None = None,
) -> Correction:
    """``se · sqrt(n_source / n_target) · sqrt(1 + (m − 1)·icc)``: the SE at another size.

    ``n_source`` / ``n_target`` are the numbers of independent units behind
    the source and the target estimate; ``icc`` and ``cluster_size`` (given
    together) describe the target's clustering through the Kish design
    effect. Returns a ``se_scale`` correction whose ``value`` is the factor.
    """
    if not math.isfinite(se) or se <= 0.0:
        raise ValueError(f"se must be a positive finite number, got {se}")
    if n_source < 1 or n_target < 1:
        raise ValueError("n_source and n_target must be >= 1")
    deff = _design_effect(icc, cluster_size)
    factor = math.sqrt(n_source / n_target) * math.sqrt(deff)
    after = se * factor
    assumption = Assumption(
        name="independent_units_within_level",
        facet="level",
        statement=(
            f"the source SE reflects {n_source} independent unit(s) and the target has "
            f"{n_target}, with a design effect of {deff:.6g}"
        ),
        challenged_by="the intra-cluster correlation; unequal cluster sizes",
        detail={"design_effect": repr(deff)},
    )
    detail = {
        "n_source": str(n_source),
        "n_target": str(n_target),
        "icc": "" if icc is None else repr(icc),
        "cluster_size": "" if cluster_size is None else str(cluster_size),
        "design_effect": repr(deff),
    }
    return Correction(
        name="variance_reweight",
        kind="se_scale",
        value=factor,
        counterfactual=se,
        corrected=after,
        detail=detail,
        ledger_line=_line(
            name="variance_reweight",
            facet="level",
            statement=(
                f"SE {se:.6g} for {n_source} unit(s) reweighted to {n_target} "
                f"(design effect {deff:.6g}): {after:.6g}"
            ),
            assumption=assumption,
            before=se,
            after=after,
            factor=factor,
            detail=detail,
        ),
    )


_LEVEL_RANK = {"individual": 0, "cluster": 1, "aggregate": 2}


def aggregation_level(
    level_source: Level,
    level_target: Level,
    *,
    cluster_size: int,
    icc: float,
) -> Correction | Unsupported:
    """Individual ↔ cluster ↔ aggregate: point factor 1, SE through ``variance_reweight``.

    ``cluster_size`` is the number of finer-level units per coarser-level
    unit and ``icc`` their intra-cluster correlation. Aggregating (finer →
    coarser) multiplies the SE by ``sqrt(deff / m)``; disaggregating by its
    inverse. The returned ``se_scale`` correction's detail records
    ``point_factor = 1`` under the ``linear_aggregation`` assumption.
    ``Unsupported`` when the interference structure differs — that needs a
    declared interference model, not a factor.
    """
    if level_source.interference != level_target.interference:
        return Unsupported(
            reason=(
                "aggregation_level: interference differs "
                f"({level_source.interference} vs {level_target.interference}); a declared "
                "interference model is needed, not a factor"
            ),
            missing=("interference_model",),
        )
    rank_s, rank_t = _LEVEL_RANK[level_source.unit], _LEVEL_RANK[level_target.unit]
    if rank_s == rank_t:
        factor = 1.0
        deff = 1.0
    else:
        # the SE of a mean over m units with intra-cluster correlation icc relative to one unit
        inner = variance_reweight(1.0, 1, cluster_size, icc=icc, cluster_size=cluster_size)
        deff = float(inner.detail["design_effect"])
        factor = inner.value if rank_t > rank_s else 1.0 / inner.value
    assumption = Assumption(
        name="linear_aggregation",
        facet="level",
        statement=(
            f"the effect aggregates linearly from {level_source.unit} to {level_target.unit} "
            "level: the mean effect per unit is invariant and only its SE changes"
        ),
        challenged_by="the Jensen gap under a nonlinear response",
        detail={"cluster_size": str(cluster_size), "icc": repr(icc)},
    )
    detail = {
        "level_source": level_source.unit,
        "level_target": level_target.unit,
        "interference": level_source.interference,
        "cluster_size": str(cluster_size),
        "icc": repr(icc),
        "design_effect": repr(deff),
        "point_factor": "1.0",
    }
    return Correction(
        name="aggregation_level",
        kind="se_scale",
        value=factor,
        counterfactual=1.0,
        corrected=factor,
        detail=detail,
        ledger_line=_line(
            name="aggregation_level",
            facet="level",
            statement=(
                f"{level_source.unit} read at {level_target.unit} level: point factor 1 "
                f"(additive effect), SE factor {factor:.6g}"
            ),
            assumption=assumption,
            before=1.0,
            after=factor,
            factor=factor,
            detail=detail,
        ),
    )


# -- resolve -----------------------------------------------------------------------------


class ResolvedTransfer(Spec):
    """A plan with its corrections applied: the composite operator and the ledger that licenses it.

    ``status`` is ``blocked`` when the plan or the transport verdict is,
    else the plan's status. ``completeness`` is ``ledger.check_complete(plan)``.
    """

    status: Status
    plan: TransferPlan
    factor: float = 1.0
    offset: float = 0.0
    se_scale: float = 1.0
    corrections: tuple[Correction, ...] = ()
    transport: Verdict | None = None
    ledger: Ledger
    completeness: Verdict
    reason: str = ""

    @model_validator(mode="after")
    def _finite(self) -> ResolvedTransfer:
        if self.factor <= 0.0 or not math.isfinite(self.factor):
            raise ValueError("factor must be positive and finite")
        if self.se_scale <= 0.0 or not math.isfinite(self.se_scale):
            raise ValueError("se_scale must be positive and finite")
        if not math.isfinite(self.offset):
            raise ValueError("offset must be finite")
        return self

    @property
    def licensed(self) -> bool:
        return self.status in ("identified", "downgraded")

    def apply(self, estimate: float, se: float) -> tuple[float, float]:
        """``(estimate·factor + offset, se·factor·se_scale)``."""
        if se < 0.0:
            raise ValueError("se must be non-negative")
        return estimate * self.factor + self.offset, se * self.factor * self.se_scale


def _merge_facet(plan_line: LedgerLine | None, group: Sequence[Correction]) -> LedgerLine:
    """One facet line for all the corrections serving one facet (see the module docstring)."""
    if len(group) == 1 and plan_line is None:
        return group[0].ledger_line
    first, last = group[0], group[-1]
    mult = math.prod(c.value for c in group if c.kind == "multiplicative")
    add = sum(c.value for c in group if c.kind == "additive")
    ses = math.prod(c.value for c in group if c.kind == "se_scale")
    names = ",".join(c.name for c in group)
    base = first.ledger_line
    detail = {
        **base.detail,
        "correction": names,
        "counterfactual": first.ledger_line.detail["counterfactual"],
        "value": last.ledger_line.detail["value"],
        "factor": repr(float(mult)),
        "offset": repr(float(add)),
        "se_scale": repr(float(ses)),
        "status": "corrected",
    }
    if plan_line is not None:
        detail["plan_assumption"] = (
            plan_line.assumption.name if plan_line.assumption is not None else ""
        )
    statement = (
        base.statement if len(group) == 1 else "; ".join(c.ledger_line.statement for c in group)
    )
    return base.model_copy(
        update={
            "statement": statement,
            "detail": detail,
            "source": plan_line.source if plan_line is not None else base.source,
            "target": plan_line.target if plan_line is not None else base.target,
        }
    )


def resolve(
    plan: TransferPlan,
    *,
    transport: Verdict | None = None,
    corrections: Sequence[Correction] = (),
) -> ResolvedTransfer:
    """Compose the corrections over the plan and write the ledger (module docstring).

    A correction whose facet the plan does not list as differing is refused
    with ``ValueError``: correcting a facet that does not differ is a sign
    the wrong plan or the wrong operator was used.
    """
    extraneous = sorted({c.facet for c in corrections} - set(plan.differing))
    if extraneous:
        raise ValueError(
            f"corrections serve facets the plan does not list as differing: {extraneous}; "
            f"the plan differs in {list(plan.differing)}"
        )
    factor = math.prod(c.value for c in corrections if c.kind == "multiplicative")
    offset = float(sum(c.value for c in corrections if c.kind == "additive"))
    se_scale = math.prod(c.value for c in corrections if c.kind == "se_scale")

    by_facet: dict[str, list[Correction]] = {}
    for c in corrections:
        by_facet.setdefault(c.facet, []).append(c)

    lines: list[LedgerLine] = []
    for line in Ledger.from_plan(plan).lines:
        facet = facet_of(line)
        if facet is not None and facet in by_facet:
            lines.append(_merge_facet(line, by_facet.pop(facet)))
        elif (
            facet == "population"
            and transport is not None
            and transport.status == "identified"
            and line.assumption is not None
        ):
            lines.append(
                line.model_copy(
                    update={
                        "assumption": line.assumption.satisfied(),
                        "detail": {**line.detail, "transport": transport.route or "identified"},
                    }
                )
            )
        else:
            lines.append(line)
    assert not by_facet  # every correction facet is in plan.differing (checked above)

    if transport is not None:
        lines.append(
            LedgerLine(
                kind="transport",
                statement=(
                    f"transport verdict {transport.status}"
                    + (f" via {transport.route}" if transport.route else "")
                    + (f": {transport.reason}" if transport.reason else "")
                ),
                assumption=transport.assumptions[0] if transport.assumptions else None,
                detail={
                    "status": transport.status,
                    "route": transport.route,
                    "assumptions": ",".join(a.name for a in transport.assumptions),
                    "counterfactual": "no transport verdict",
                    "value": transport.status,
                },
                source=plan.source,
                target=plan.target,
            )
        )
    ledger = Ledger(lines=tuple(lines))

    status: Status
    reason = plan.reason
    if plan.status == "blocked":
        status = "blocked"
    elif transport is not None and transport.status == "blocked":
        status = "blocked"
        reason = f"transport blocked: {transport.reason}"
    else:
        status = plan.status
    return ResolvedTransfer(
        status=status,
        plan=plan,
        factor=float(factor),
        offset=offset,
        se_scale=float(se_scale),
        corrections=tuple(corrections),
        transport=transport,
        ledger=ledger,
        completeness=ledger.check_complete(plan),
        reason=reason,
    )
