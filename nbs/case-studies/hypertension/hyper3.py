"""HYPER-3: the synthetic world the hypertension case-study notebooks share.

One data-generating process, one enrollment plan, one set of analysis helpers, and
a small plotting vocabulary — so that six notebooks tell one story about the same
trial rather than six stories about six trials.

**The trial.** Adults with elevated or stage-1 hypertension are randomized to the
standard of care or to one of three daily doses of an investigational agent
(10, 20, 40 mg). Randomization is stratified by age band — 25-35, 36-50, 51+ —
in 2:1:1:1 blocks, so the shared control arm is twice the size of each dose arm.
Seated systolic blood pressure is measured weekly for 24 weeks. Enrollment is
staggered over 16 calendar weeks, which is what makes an interim look a real
event: at calendar week 20 some units have 20 weeks of follow-up and some have
four.

**The truth the notebooks are trying to recover.** For a unit in stratum ``s`` on
dose ``d``, ``w`` weeks after randomization, the mean systolic pressure is

    baseline + settling(w) + regression(baseline, w) − benefit(d, s, w) + harm(d, s, w)

where ``settling`` is the shared improvement every arm gets from being in a trial
at all, ``regression`` is regression to the mean proportional to how far the
unit's baseline sat above its stratum mean, ``benefit`` is an Emax curve in dose
with an exponential onset, and ``harm`` is a *superlinear* pressor effect that is
negligible below 20 mg and concentrated in the oldest stratum, where clearance is
slowest.

That last term is the point of the case study. Averaged over the trial population
the 40 mg arm looks almost exactly like the control arm — the benefit in the two
younger strata cancels the harm in the oldest one — so an arm-level safety
boundary never fires while a stratum-level one fires early. Everything in the
notebooks follows from wanting to catch that before it reaches many units.

Nothing here is imported by ``axiom``; it is notebook scaffolding, and every
statistical claim it supports is computed with ``axiom`` in the notebooks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = npt.NDArray[np.float64]

# -- the protocol ----------------------------------------------------------------------

ARMS: tuple[str, ...] = ("standard_of_care", "dose_10", "dose_20", "dose_40")
DOSE: Mapping[str, float] = {
    "standard_of_care": 0.0,
    "dose_10": 10.0,
    "dose_20": 20.0,
    "dose_40": 40.0,
}
DOSE_UNIT = "mg"
OUTCOME_UNIT = "mmHg"

STRATA: tuple[str, ...] = ("age_25_35", "age_36_50", "age_51_plus")
STRATUM_SHARE: Mapping[str, float] = {"age_25_35": 0.25, "age_36_50": 0.40, "age_51_plus": 0.35}
STRATUM_LABEL: Mapping[str, str] = {
    "age_25_35": "25-35",
    "age_36_50": "36-50",
    "age_51_plus": "51+",
}

ALLOCATION: Mapping[str, int] = {"standard_of_care": 2, "dose_10": 1, "dose_20": 1, "dose_40": 1}
BLOCK = sum(ALLOCATION.values())

N_UNITS = 400
ENROLLMENT_WEEKS = 16
FOLLOW_UP_WEEKS = 24
PRIMARY_WEEK = 12
TRIAL_WEEKS = ENROLLMENT_WEEKS + FOLLOW_UP_WEEKS


@dataclass(frozen=True)
class Truth:
    """Every parameter of the data-generating process, in mmHg and mg."""

    baseline_mean: Mapping[str, float] = field(
        default_factory=lambda: {"age_25_35": 138.0, "age_36_50": 143.0, "age_51_plus": 149.0}
    )
    baseline_sd: float = 11.0
    settling: float = 4.0
    settling_weeks: float = 4.0
    regression_to_mean: float = 0.22
    emax: Mapping[str, float] = field(
        default_factory=lambda: {"age_25_35": 9.5, "age_36_50": 11.0, "age_51_plus": 12.5}
    )
    ed50: float = 18.0
    onset_weeks: float = 3.0
    harm: float = 32.0
    harm_share: Mapping[str, float] = field(
        default_factory=lambda: {"age_25_35": 0.0, "age_36_50": 0.28, "age_51_plus": 1.0}
    )
    harm_weeks: float = 6.0
    harm_exponent: float = 3.0
    noise_sd: float = 8.0
    autocorrelation: float = 0.45
    adherence_mean: float = 0.88
    adherence_dose_penalty: float = 0.10
    adherence_concentration: float = 22.0
    dropout_intercept: float = -4.6
    dropout_per_mmhg: float = 0.035
    dropout_older: float = 0.35


TRUTH = Truth()


# -- the mean response -----------------------------------------------------------------


def benefit(dose: Array | float, stratum: str, week: Array | float, truth: Truth = TRUTH) -> Any:
    """Emax reduction in systolic pressure, in mmHg, with an exponential onset."""
    d = np.asarray(dose, dtype=np.float64)
    w = np.asarray(week, dtype=np.float64)
    onset = 1.0 - np.exp(-np.maximum(w, 0.0) / truth.onset_weeks)
    return truth.emax[stratum] * d / (truth.ed50 + d) * onset


def harm(dose: Array | float, stratum: str, week: Array | float, truth: Truth = TRUTH) -> Any:
    """The pressor effect: superlinear in dose, and only where clearance is slow."""
    d = np.asarray(dose, dtype=np.float64)
    w = np.asarray(week, dtype=np.float64)
    onset = 1.0 - np.exp(-np.maximum(w, 0.0) / truth.harm_weeks)
    top = max(DOSE.values())
    return truth.harm * truth.harm_share[stratum] * (d / top) ** truth.harm_exponent * onset


def mean_change(
    dose: Array | float, stratum: str, week: Array | float, truth: Truth = TRUTH
) -> Any:
    """Mean change from baseline, in mmHg, at the *stratum* mean baseline.

    Negative is better. The shared settling term is in here too, so the *contrast*
    against the control arm — which is what every estimand in the notebooks is —
    is ``mean_change(d, s, w) - mean_change(0, s, w)``.
    """
    w = np.asarray(week, dtype=np.float64)
    settling = -truth.settling * (1.0 - np.exp(-np.maximum(w, 0.0) / truth.settling_weeks))
    return settling - benefit(dose, stratum, w, truth) + harm(dose, stratum, w, truth)


def per_protocol_contrast(
    dose: float, stratum: str | None = None, week: float = PRIMARY_WEEK, truth: Truth = TRUTH
) -> float:
    """Dose minus control at the *prescribed* dose — the effect of taking every tablet.

    This is not what a randomized trial estimates. Adherence is a consequence of
    the assignment, so the quantity randomization licenses is
    :func:`intent_to_treat_contrast`, which averages over how much was taken.
    """
    if stratum is not None:
        return float(
            mean_change(dose, stratum, week, truth) - mean_change(0.0, stratum, week, truth)
        )
    return float(
        sum(STRATUM_SHARE[s] * per_protocol_contrast(dose, s, week, truth) for s in STRATA)
    )


def adherence_grid(dose: float, truth: Truth = TRUTH, n: int = 2001) -> tuple[Array, Array]:
    """The adherence distribution at an assigned dose, as a quadrature grid and weights."""
    from scipy import stats as _stats

    mean = truth.adherence_mean - truth.adherence_dose_penalty * dose / max(DOSE.values())
    grid = np.linspace(1e-4, 1.0 - 1e-4, n)
    weights = np.asarray(
        _stats.beta.pdf(
            grid,
            mean * truth.adherence_concentration,
            (1.0 - mean) * truth.adherence_concentration,
        ),
        dtype=np.float64,
    )
    return grid, weights / weights.sum()


def intent_to_treat_contrast(
    dose: float, stratum: str | None = None, week: float = PRIMARY_WEEK, truth: Truth = TRUTH
) -> float:
    """The estimand randomization licenses: assigned dose minus control, adherence and all.

    Averages the mean response over the adherence distribution at that dose. It is
    the number every analysis in the notebooks is trying to recover, and it is
    smaller in magnitude than :func:`per_protocol_contrast` — for the harm term much
    smaller, because the pressor effect is cubic in the dose actually taken.
    """
    if stratum is None:
        return float(
            sum(STRATUM_SHARE[s] * intent_to_treat_contrast(dose, s, week, truth) for s in STRATA)
        )
    if dose == 0.0:
        return 0.0
    grid, weights = adherence_grid(dose, truth)
    value = mean_change(dose * grid, stratum, week, truth) - mean_change(0.0, stratum, week, truth)
    return float((np.asarray(value, dtype=np.float64) * weights).sum())


# -- enrollment and follow-up ----------------------------------------------------------


def enroll(n_units: int = N_UNITS, *, seed: int, truth: Truth = TRUTH) -> pd.DataFrame:
    """Stratified block randomization with staggered entry.

    Strata are filled to their planned shares; within a stratum, units are
    randomized in 2:1:1:1 blocks so the allocation is balanced at every point in
    the enrollment period rather than only at the end.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    counts = _stratum_counts(n_units)
    for stratum in STRATA:
        arms = _blocked_arms(counts[stratum], rng)
        baseline = rng.normal(truth.baseline_mean[stratum], truth.baseline_sd, size=counts[stratum])
        for i, (arm, base) in enumerate(zip(arms, baseline, strict=True)):
            dose = DOSE[arm]
            mean = truth.adherence_mean - truth.adherence_dose_penalty * dose / max(DOSE.values())
            rows.append(
                {
                    "unit": f"{stratum}_{i:03d}",
                    "stratum": stratum,
                    "arm": arm,
                    "dose": dose,
                    "baseline_sbp": float(base),
                    "adherence": float(
                        rng.beta(
                            mean * truth.adherence_concentration,
                            (1.0 - mean) * truth.adherence_concentration,
                        )
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    # Entry is uniform over the enrollment window, independent of stratum and arm.
    order = rng.permutation(len(frame))
    frame = frame.iloc[order].reset_index(drop=True)
    frame["enrolled_week"] = np.floor(
        np.linspace(0.0, ENROLLMENT_WEEKS, len(frame), endpoint=False)
    ).astype(int)
    return frame.sort_values("unit").reset_index(drop=True)


def _stratum_counts(n_units: int) -> dict[str, int]:
    counts = {s: int(round(n_units * STRATUM_SHARE[s] / BLOCK)) * BLOCK for s in STRATA}
    counts[STRATA[-1]] += n_units - sum(counts.values())
    return counts


def _blocked_arms(n: int, rng: np.random.Generator) -> list[str]:
    block = [arm for arm, k in ALLOCATION.items() for _ in range(k)]
    out: list[str] = []
    while len(out) < n:
        out += list(rng.permutation(block))
    return out[:n]


def follow_up(units: pd.DataFrame, *, seed: int, truth: Truth = TRUTH) -> pd.DataFrame:
    """Weekly systolic measurements for every enrolled unit, with AR(1) noise and dropout.

    Returns one row per unit-visit with ``week`` (weeks since randomization),
    ``calendar_week`` (weeks since the trial opened), the measured ``sbp``, and the
    ``change`` from that unit's own baseline. Visits after a unit drops out are not
    in the frame at all — a stopped observation is absent, not imputed.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(1, FOLLOW_UP_WEEKS + 1, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for record in units.to_dict("records"):
        stratum = str(record["stratum"])
        base = float(record["baseline_sbp"])
        # Adherence attenuates the dose actually taken; it is a consequence of the
        # assignment, never a thing the protocol sets.
        taken = float(record["dose"]) * float(record["adherence"])
        drift = np.asarray(mean_change(taken, stratum, weeks, truth), dtype=np.float64)
        regression = (
            -truth.regression_to_mean
            * (base - truth.baseline_mean[stratum])
            * (1.0 - np.exp(-weeks / 2.0))
        )
        noise = _ar1(len(weeks), truth.noise_sd, truth.autocorrelation, rng)
        sbp = base + drift + regression + noise
        older = 1.0 if stratum == "age_51_plus" else 0.0
        for step, week in enumerate(weeks):
            hazard = _logistic(
                truth.dropout_intercept
                + truth.dropout_per_mmhg * (sbp[step] - base)
                + truth.dropout_older * older
            )
            if rng.random() < hazard:
                break
            rows.append(
                {
                    "unit": record["unit"],
                    "stratum": stratum,
                    "arm": record["arm"],
                    "dose": record["dose"],
                    "baseline_sbp": base,
                    "adherence": record["adherence"],
                    "week": int(week),
                    "calendar_week": int(record["enrolled_week"] + week),
                    "sbp": float(sbp[step]),
                    "change": float(sbp[step] - base),
                }
            )
    return pd.DataFrame(rows).sort_values(["unit", "week"]).reset_index(drop=True)


def _ar1(n: int, sd: float, rho: float, rng: np.random.Generator) -> Array:
    innovation = rng.normal(0.0, sd * np.sqrt(1.0 - rho**2), size=n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + innovation[i]
    return out


def _logistic(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def trial(*, seed: int = 20260821, n_units: int = N_UNITS, truth: Truth = TRUTH) -> Trial:
    """Enroll and follow one whole trial. This is the object every notebook starts from."""
    units = enroll(n_units, seed=seed, truth=truth)
    return Trial(units=units, visits=follow_up(units, seed=seed + 1, truth=truth), truth=truth)


@dataclass(frozen=True)
class Trial:
    """One realized trial: its units, its visits, and the truth behind both."""

    units: pd.DataFrame
    visits: pd.DataFrame
    truth: Truth

    @property
    def n_units(self) -> int:
        return len(self.units)

    def arm_counts(self) -> pd.DataFrame:
        return (
            self.units.pivot_table(index="stratum", columns="arm", values="unit", aggfunc="count")
            .reindex(index=list(STRATA), columns=list(ARMS))
            .fillna(0)
            .astype(int)
        )

    def completion(self) -> pd.Series:
        """Share of units still on study at each week since randomization."""
        counts = self.visits.groupby("week")["unit"].nunique()
        return (counts / self.n_units).reindex(range(1, FOLLOW_UP_WEEKS + 1), fill_value=0.0)


# -- what an interim look sees ---------------------------------------------------------

Endpoint = Literal["safety", "primary"]

#: Both endpoints average the weekly readings inside a window rather than reading one
#: visit. This is why weekly monitoring is worth its cost: averaging four AR(1)
#: readings roughly halves the variance of a unit's endpoint, which is the difference
#: between a safety boundary that can fire inside a stratum and one that cannot.
WINDOW: Mapping[str, tuple[int, int]] = {"safety": (5, 8), "primary": (9, 12)}
MIN_VISITS = 3


def look_frame(
    trial_: Trial,
    calendar_week: int,
    *,
    endpoint: Endpoint = "safety",
    min_visits: int = MIN_VISITS,
) -> pd.DataFrame:
    """The one row per unit an analysis at ``calendar_week`` is entitled to see.

    A unit contributes once it has at least ``min_visits`` readings inside the
    endpoint's window, and its ``change`` is the mean of those readings less its own
    baseline. Nothing recorded after ``calendar_week`` is visible, which is the whole
    discipline of an interim analysis and the reason the information fraction is
    below one.

    Because a unit's window either is complete or is not, the set of contributing
    units only grows with ``calendar_week`` — the information scale a sequential
    boundary needs is monotone by construction.
    """
    low, high = WINDOW[endpoint]
    seen = trial_.visits
    seen = seen[
        (seen["calendar_week"] <= calendar_week) & (seen["week"] >= low) & (seen["week"] <= high)
    ]
    if seen.empty:
        return trial_.units.head(0).assign(
            change=pd.Series(dtype=float), visits=pd.Series(dtype=int)
        )
    grouped = seen.groupby("unit").agg(change=("change", "mean"), visits=("change", "size"))
    grouped = grouped[grouped["visits"] >= min_visits].reset_index()
    return (
        trial_.units.merge(grouped, on="unit", how="inner")
        .sort_values("unit")
        .reset_index(drop=True)
    )


def accrual(trial_: Trial, weeks: Sequence[int], *, endpoint: Endpoint = "safety") -> pd.DataFrame:
    """How many units each candidate look would see, and at what share of the total."""
    total = len(look_frame(trial_, TRIAL_WEEKS, endpoint=endpoint))
    rows = [
        {
            "calendar_week": int(w),
            "units": len(look_frame(trial_, int(w), endpoint=endpoint)),
            "share": len(look_frame(trial_, int(w), endpoint=endpoint)) / max(total, 1),
        }
        for w in weeks
    ]
    return pd.DataFrame(rows)


def contrast_frame(
    frame: pd.DataFrame, arm: str, *, stratum: str | None = None, control: str = ARMS[0]
) -> pd.DataFrame:
    """Two arms, with the columns an ANCOVA needs: ``treated``, ``change``, ``baseline_sbp``.

    ``treated`` is 1 for the dose arm and 0 for the control arm, and the covariates
    are all pre-randomization: baseline pressure always, and stratum indicators when
    the contrast is pooled over strata.
    """
    rows = frame[frame["arm"].isin((arm, control))]
    if stratum is not None:
        rows = rows[rows["stratum"] == stratum]
    out = rows.copy()
    out["treated"] = (out["arm"] == arm).astype(float)
    out["baseline_centred"] = out["baseline_sbp"] - out["baseline_sbp"].mean()
    for level in STRATA[1:]:
        out[f"is_{level}"] = (out["stratum"] == level).astype(float)
    return out.reset_index(drop=True)


def ancova_covariates(stratum: str | None) -> list[str]:
    """The adjustment set: baseline always, stratum indicators only when pooling."""
    if stratum is not None:
        return ["baseline_centred"]
    return ["baseline_centred"] + [f"is_{level}" for level in STRATA[1:]]


# -- plotting --------------------------------------------------------------------------

ARM_COLOR: Mapping[str, str] = {
    "standard_of_care": "#5b6472",
    "dose_10": "#2f7fd1",
    "dose_20": "#8a63c4",
    "dose_40": "#d1483f",
}
STRATUM_COLOR: Mapping[str, str] = {
    "age_25_35": "#3aa17e",
    "age_36_50": "#c9a227",
    "age_51_plus": "#b5453b",
}
ARM_LABEL: Mapping[str, str] = {
    "standard_of_care": "standard of care",
    "dose_10": "10 mg",
    "dose_20": "20 mg",
    "dose_40": "40 mg",
}
GRID = "rgba(128,128,128,0.20)"


def figure(title: str, x: str, y: str, *, height: int = 380, **kwargs: object) -> Any:
    """A plotly figure with the case study's house layout already applied."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        height=height,
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def band(
    fig: Any,
    x: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    color: str,
    *,
    name: str = "",
    opacity: float = 0.18,
) -> Any:
    """A filled interval band, drawn under whatever line it belongs to."""
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor=_rgba(color, opacity),
            line={"width": 0},
            hoverinfo="skip",
            showlegend=bool(name),
            name=name,
        )
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def mean_with_error(frame: pd.DataFrame, by: Sequence[str], value: str) -> pd.DataFrame:
    """Group means with their standard errors and counts — the plotting workhorse."""
    grouped = frame.groupby(list(by))[value]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out
