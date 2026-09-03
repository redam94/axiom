"""HOUSE-3: the synthetic world the media-measurement case-study notebooks share.

One measurement house, three advertisers, forty markets, fifty-two weeks. The
house runs incrementality experiments for each advertiser — a *party*, in the
vocabulary of ``axiom.io`` — and has to say at the end of the year which
channels moved which KPI, for whom, by how much, and which of those claims
survive being made all at once.

This is the one case study in ``nbs/`` where marketing words appear, and they
appear only through ``axiom.adapters.marketing``: a **channel** is a
``Treatment`` whose dose is spend, a **geo** (a DMA, here "market") is the
panel ``Unit``, a **KPI** is the ``Outcome``. Nothing under ``axiom`` outside
``adapters/`` knows those words (rule 2), and every number in the notebooks is
produced by the general machinery — ``design``, ``identify``, ``diagnose``,
``calibrate``, ``meta``, ``io`` — with the adapter supplying names at the edge.

The three parties
-----------------

=========  ==============  ===========================  ====================
party      business        what a conversion is         channels
=========  ==============  ===========================  ====================
northwind  retail          a paid order                 search, video
acme       subscription    a trial start, later a paid  social, search
                           subscription
globex     consumer goods  a unit sold at shelf         tv, display, search
=========  ==============  ===============================  ================

Every party's KPI column is ``conversions`` — the same ``Outcome`` entity,
which is what lets a search lift measured for one party be compared with a
search lift measured for another at all (``meta.commensurable`` refuses two
outcomes). What each party *means* by a conversion is a different question,
and it is the ``io.DefinitionRegistry``'s: notebook 1 registers the three
answers, and acme's changes in week twelve.

Northwind also reports ``revenue``, its orders times an average order value,
so its frame can be read with a currency KPI and the adapter's ``roas`` has
something to compute. For a count KPI the adapter returns a typed
``Unsupported`` rather than a mislabelled number, and notebook 1 shows both.

Why a market is the unit
------------------------

Media is bought and switched off by market, not by person. A holdout that
turns a channel off in ten markets for eight weeks is a cluster-randomized
experiment with forty clusters, and every design number in notebook 2 —
the ICC, the design effect, the minimum detectable effect at each holdout
share — is about *markets*. The forty markets come in three size tiers, and
size is what the assignments are stratified on.

The truth the notebooks are trying to recover
---------------------------------------------

Each party's KPI in market ``i`` in week ``t`` is

    alpha_i + season(t) + sum_c beta_c · hill(adstock_c(spend_c))
        + gamma · size_i · demand_it + noise

evaluated by ``axiom.core.value`` on the ``ModelSpec`` that ``axiom.surface``
builds from the party's ``SurfaceSpec`` — the same Hill saturation, the same
geometric carryover, the same Fourier seasonality the analyst will fit. There
is no second copy of the transform chain anywhere in this file (rule 3); the
world's ``mean`` *is* the model's mean, plus one term the model does not
have.

That term is the point of notebook 1. ``demand_it`` is an unobserved,
market-level, week-to-week demand shock — a competitor's stock-out, local
weather, a regional promotion — that raises the KPI directly **and** raises
the week's spend, because every one of these advertisers buys more media
when demand is up. It is not seasonal, so the Fourier terms cannot absorb it,
and it is not in any column the analyst has. An observational model fitted
to this panel attributes some of the demand to the channels. A geo experiment
does not, because the experiment chooses the spend.

Experiments, as the world applies them
--------------------------------------

An ``Experiment`` on the calendar names a party, a channel, a kind, a window
and how many markets it treats. The world applies one to a party's dose
arrays given the markets the design assigned:

* a **holdout** sets the channel's spend to zero in the treated markets for
  the window — except in markets that *kept spending* (``GX-03``: retailers'
  co-op search ads the house does not control), which is the compliance
  story;
* a **dose test** scales the channel's spend up by ``uplift`` in the treated
  markets — and the platform delivers only part of the increase in some of
  them (``NW-02``), which is the dose-is-not-a-switch story;
* a **rollout** is not an experiment at all: acme ships a new onboarding
  flow to its fifteen largest markets over ten weeks, which moves the KPI
  and was not randomized, and which the calendar has to know about.

Two more things happen in the field and are recorded here because the world
has to produce them: acme's KPI feed drops out in some control markets during
``AC-04`` — the ones whose demand fell, which is selection on the outcome —
and every party carries two **guardrail** metrics that no channel affects: a
null a programme's error control has to be judged against.

Nothing here is imported by ``axiom``; it is notebook scaffolding, and every
statistical claim it supports is computed with ``axiom`` in the notebooks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from axiom.adapters import KPI, Channel, MarketingRoles, panel_from_marketing_frame
from axiom.build import SurfaceBuilder
from axiom.core import D, TimeWindow, value
from axiom.data import Panel
from axiom.design import Occupancy
from axiom.sim import true_parameters
from axiom.surface import SurfaceSpec, build, prepare

Array = npt.NDArray[np.float64]

# -- the markets -----------------------------------------------------------------------

N_MARKETS = 40
N_WEEKS = 52
#: Zero-padded so the labels sort the way ``Panel`` sorts its rows.
MARKETS: tuple[str, ...] = tuple(f"m{i:02d}" for i in range(N_MARKETS))

#: Three size tiers: eight large markets, sixteen medium, sixteen small. A market's
#: size multiplies its baseline KPI and its spend, and it is what assignments are
#: stratified on, because a holdout that lands seven large markets in one arm is a
#: holdout nobody can read.
TIERS: tuple[str, ...] = tuple(["large"] * 8 + ["medium"] * 16 + ["small"] * 16)
_SIZE_BY_TIER = {"large": 2.2, "medium": 1.0, "small": 0.55}
SIZES: Array = np.array([_SIZE_BY_TIER[t] for t in TIERS], dtype=np.float64)

#: Weeks a design reads before an experiment starts: the pre-period.
PRE_WEEKS = 8

SPEND_UNIT = "kUSD"


# -- the parties -----------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelTruth:
    """One channel's true response: a Hill curve on adstocked spend.

    ``share`` is the fraction of the party's baseline KPI the channel drives at
    its typical spend — the number every holdout is trying to measure — and the
    amplitude ``beta`` is derived from it rather than tuned by hand.
    """

    spend: float  # typical weekly spend per unit of market size, kUSD
    share: float  # of baseline KPI, at typical spend
    shape: float = 1.8  # Hill s
    half: float = 1.0  # Hill k, as a multiple of typical spend
    decay: float = 0.45  # geometric carryover lambda


@dataclass(frozen=True)
class Party:
    """One advertiser: its KPI, its channels, and the truth behind both."""

    name: str
    business: str
    conversion: str  # what the party means by one
    baseline: float  # conversions per week in a size-1 market
    channels: Mapping[str, ChannelTruth]
    guardrails: tuple[str, ...]
    order_value: float | None = None  # kUSD per conversion, when the party reports revenue
    noise_share: float = 0.04  # residual sd as a share of baseline
    demand_share: float = 0.04  # sd of the unobserved demand term, as a share
    demand_on_spend: float = 0.6  # log-spend response to one sd of demand
    season_amplitude: float = 0.12  # of baseline, the first Fourier harmonic
    seed: int = 0

    @property
    def program(self) -> str:
        return "incrementality"


PARTIES: dict[str, Party] = {
    "northwind": Party(
        name="northwind",
        business="retail",
        conversion="paid_order",
        baseline=120.0,
        order_value=0.34,
        channels={
            "search": ChannelTruth(spend=2.5, share=0.12, shape=2.0),
            "video": ChannelTruth(spend=1.5, share=0.06, shape=1.6, half=1.5, decay=0.6),
        },
        guardrails=("return_rate", "basket_size"),
        seed=11,
    ),
    "acme": Party(
        name="acme",
        business="subscription",
        conversion="trial_start",
        baseline=60.0,
        channels={
            "social": ChannelTruth(spend=0.8, share=0.10, shape=1.7),
            "search": ChannelTruth(spend=0.6, share=0.07, shape=2.0),
        },
        guardrails=("churn", "tickets"),
        seed=23,
    ),
    "globex": Party(
        name="globex",
        business="consumer goods",
        conversion="unit_sold",
        baseline=300.0,
        channels={
            "tv": ChannelTruth(spend=8.0, share=0.14, shape=1.5, decay=0.55),
            "display": ChannelTruth(spend=2.5, share=0.09, shape=2.0),
            "search": ChannelTruth(spend=2.0, share=0.05, shape=1.8),
        },
        guardrails=("stockouts", "price_index"),
        seed=37,
    ),
}


KPI_COLUMN = "conversions"


def roles_for(party: Party, *, kpi: str = KPI_COLUMN) -> MarketingRoles:
    """The adapter's role declaration for one party's weekly frame.

    ``kpi="revenue"`` reads northwind's frame with its currency column as the
    KPI instead — the same markets, the same spend, a KPI the adapter can put
    a ROAS on.
    """
    if kpi == "revenue" and party.order_value is None:
        raise ValueError(f"{party.name} does not report revenue")
    return MarketingRoles(
        kpi=kpi,
        channels=tuple(party.channels),
        geo="market",
        date="week",
        currency=SPEND_UNIT,
        kpi_dimension="currency" if kpi == "revenue" else "outcome",
        kpi_unit=SPEND_UNIT if kpi == "revenue" else KPI_COLUMN,
    )


def kpi_entity(party: Party, *, kpi: str = KPI_COLUMN) -> KPI:
    return roles_for(party, kpi=kpi).kpi_entity


def channel_entity(name: str) -> Channel:
    return Channel(name=name, dimension=D.currency, unit=SPEND_UNIT)


# -- the surface -----------------------------------------------------------------------


def spec_for(party: Party, *, name: str | None = None) -> SurfaceSpec:
    """The party's ``SurfaceSpec``: Hill + geometric carryover per channel, hierarchical
    market intercepts, second-order annual seasonality.

    The same shape ``adapters.marketing_spec`` builds from a panel — this one is
    built from the truth's reference spends rather than a panel's means, so the
    world can exist before any frame does. The prior scales are set to the
    KPI's magnitude: a prior of sd 1 on an intercept of 300 is the trap note
    0009 §3 records, and it is easy to fall into with a count KPI.
    """
    b = SurfaceBuilder().name(name or f"{party.name}_mmm").outcome(kpi_entity(party))
    for channel, truth in party.channels.items():
        b = b.treatment(
            channel_entity(channel),
            kernel="hill",
            carryover="geometric",
            max_lag=4,
            reference_dose=truth.spend,
            amplitude_scale=party.baseline * 0.5,
        )
    b = b.intercept("hierarchical", units=MARKETS, scale=party.baseline)
    b = b.nuisance(fourier=(float(N_WEEKS), 2), coefficient_scale=party.baseline * 0.2)
    spec = b.build()
    return spec.model_copy(update={"noise_scale": party.baseline * 0.2})


def analyst_spec(panel: Panel, party: Party, *, intercept: str = "per_unit") -> SurfaceSpec:
    """The spec the house's analyst fits to a party's panel: ``adapters.marketing_spec``
    with its prior scales set to the KPI's magnitude.

    ``marketing_spec`` defaults every scale to one, which is right for a KPI of
    order one and is the prior-scale trap of note 0009 §3 for a KPI of order a
    hundred: a half-normal(1) prior on an amplitude of ten does not fail
    loudly, it fits badly. The scales below are the only thing this function
    adds: an amplitude prior that says no channel adds more than a quarter of
    the baseline at saturation, and intercept, seasonality and noise scales
    on the KPI's magnitude.

    Two choices the analyst made and the world did not. The intercept is per
    market rather than hierarchical, because the Laplace backend does not
    certify the hierarchical fit on a forty-market panel and with fifty-two
    weeks per market the pooling buys nothing. The carryover is two weeks
    (``max_lag=2``) rather than the world's four: with four lags and a
    per-market intercept the posterior has a near-flat direction the Laplace
    Hessian will not certify, and two lags carry most of a decay of 0.5.
    """
    from axiom.adapters import marketing_spec
    from axiom.surface import FourierSeasonality, NuisanceSet

    spec = marketing_spec(
        panel,
        name=f"{party.name}_mmm",
        seasonality=(float(N_WEEKS), 2),
        intercept=intercept,  # type: ignore[arg-type]
        amplitude_scale=party.baseline * 0.25,
        max_lag=2,
    )
    return spec.model_copy(
        update={
            "intercept_scale": party.baseline,
            "noise_scale": party.baseline * 0.2,
            "nuisance": NuisanceSet(
                terms=(
                    FourierSeasonality(
                        period=float(N_WEEKS), order=2, coefficient_scale=party.baseline * 0.2
                    ),
                )
            ),
        }
    )


def hill(spend: Array | float, truth: ChannelTruth) -> Any:
    """``u^s / (1 + u^s)`` at ``u = spend / k`` — used only to *size* ``beta`` from
    ``share``; the world's mean is never computed with it (rule 3)."""
    u = np.asarray(spend, dtype=np.float64) / (truth.half * truth.spend)
    return u**truth.shape / (1.0 + u**truth.shape)


def truth_for(party: Party, spec: SurfaceSpec) -> dict[str, Array]:
    """Every parameter of the built model at the truth.

    Intercepts scale with market size; the two harmonics put the seasonal peak
    in the last weeks of the year; each channel's amplitude is the one that
    makes its steady-state contribution ``share`` of baseline at typical spend.
    """
    model = build(spec)
    overrides: dict[str, npt.ArrayLike] = {
        "alpha_mean": party.baseline,
        "alpha_sd": party.baseline * 0.5,
        "alpha_unit": party.baseline * SIZES,
        "n0_coef_sin_1": -0.35 * party.season_amplitude * party.baseline,
        "n0_coef_cos_1": party.season_amplitude * party.baseline,
        "n0_coef_sin_2": 0.02 * party.baseline,
        "n0_coef_cos_2": 0.03 * party.baseline,
        "sigma": party.noise_share * party.baseline,
    }
    for channel, truth in party.channels.items():
        # At typical spend, size 1, the adstocked spend is the spend itself (the
        # carryover weights are normalized), so hill(typical) is the steady state.
        overrides[f"beta_{channel}"] = (
            truth.share * party.baseline / float(hill(truth.spend, truth))
        )
        overrides[f"k_{channel}"] = truth.half * truth.spend
        overrides[f"s_{channel}"] = truth.shape
        overrides[f"lam_{channel}"] = truth.decay
    return true_parameters(model, truth=overrides, seed=party.seed)


# -- the world -------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """One party's year: the spec, its truth, the spend every market ran, and the
    unobserved demand and noise that every counterfactual shares.

    ``doses`` is what the advertiser spent with no experiment running;
    ``kpi(doses)`` is what the KPI would have been under any other spend, with
    the *same* demand and the *same* noise, so the difference between two
    calls is exactly the channel's effect and nothing else.
    """

    party: Party
    spec: SurfaceSpec
    theta: dict[str, Array]
    doses: dict[str, Array]
    demand: Array
    noise: Array
    guardrail_values: dict[str, Array]
    order_value: Array
    year: int

    @property
    def markets(self) -> tuple[str, ...]:
        return MARKETS

    @property
    def roles(self) -> MarketingRoles:
        return roles_for(self.party)

    def _panel_for(self, doses: Mapping[str, Array], outcome: Array) -> Panel:
        return panel_from_marketing_frame(self.frame_from(doses, outcome), self.roles)

    def mean(self, doses: Mapping[str, Array] | None = None) -> Array:
        """The noise-free KPI: the model's mean through ``axiom.core.value``, plus demand."""
        d = dict(self.doses if doses is None else doses)
        laid_out = prepare(self.spec, self._panel_for(d, np.zeros_like(self.demand)))
        model_mean = np.asarray(value(build(self.spec).mean, data=laid_out, params=self.theta))
        model_mean = np.broadcast_to(model_mean, self.demand.shape)
        demand_term = self.party.demand_share * self.party.baseline * SIZES[:, None] * self.demand
        return np.asarray(model_mean + demand_term, dtype=np.float64)

    def kpi(self, doses: Mapping[str, Array] | None = None) -> Array:
        """The KPI the party sees: mean plus the year's fixed noise draw."""
        return self.mean(doses) + self.noise

    def contribution(self, channel: str, doses: Mapping[str, Array] | None = None) -> Array:
        """What the channel adds, cell by cell: ``mean(doses) − mean(doses with it off)``."""
        d = dict(self.doses if doses is None else doses)
        off = {**d, channel: np.zeros_like(d[channel])}
        return self.mean(d) - self.mean(off)

    def frame_from(
        self, doses: Mapping[str, Array], outcome: Array, *, revenue: bool = False
    ) -> pd.DataFrame:
        """A wide market × week frame in the columns the adapter's roles name.

        ``revenue=True`` adds northwind's revenue column: conversions times the
        average order value, with the week's mix noise on top.
        """
        rows = []
        for i, market in enumerate(MARKETS):
            for t in range(N_WEEKS):
                row: dict[str, Any] = {"market": market, "week": t, KPI_COLUMN: outcome[i, t]}
                if revenue:
                    row["revenue"] = outcome[i, t] * self.order_value[i, t]
                for channel in self.party.channels:
                    row[channel] = float(doses[channel][i, t])
                rows.append(row)
        return pd.DataFrame(rows)

    def frame(
        self, doses: Mapping[str, Array] | None = None, *, revenue: bool = False
    ) -> pd.DataFrame:
        d = dict(self.doses if doses is None else doses)
        return self.frame_from(d, self.kpi(d), revenue=revenue)

    def panel(self, doses: Mapping[str, Array] | None = None, *, kpi: str = KPI_COLUMN) -> Panel:
        """The party's ``Panel`` through the adapter; ``kpi="revenue"`` for northwind's
        currency reading of the same year."""
        d = dict(self.doses if doses is None else doses)
        frame = self.frame_from(d, self.kpi(d), revenue=kpi == "revenue")
        return panel_from_marketing_frame(frame, roles_for(self.party, kpi=kpi))

    def spend_share(self, channel: str) -> float:
        """The channel's share of the party's total spend over the year."""
        total = sum(float(v.sum()) for v in self.doses.values())
        return float(self.doses[channel].sum()) / total


def world(party: str | Party, *, year: int = 1) -> World:
    """The party's world for one year. ``year=0`` is last year — the observational
    record the MMM is fitted to in notebook 1; ``year=1`` is the year the
    experiments run in. Same truth, different demand and noise."""
    p = PARTIES[party] if isinstance(party, str) else party
    rng = np.random.default_rng([p.seed, year])
    spec = spec_for(p)
    theta = truth_for(p, spec)
    # Unobserved demand: AR(1) per market, standardized.
    demand = np.zeros((N_MARKETS, N_WEEKS))
    innovations = rng.normal(size=(N_MARKETS, N_WEEKS))
    rho = 0.6
    demand[:, 0] = innovations[:, 0]
    for t in range(1, N_WEEKS):
        demand[:, t] = rho * demand[:, t - 1] + math.sqrt(1.0 - rho**2) * innovations[:, t]
    # Spend follows size, the season, the demand the advertiser can see, and a
    # planner's week-to-week noise.
    weeks = np.arange(N_WEEKS, dtype=np.float64)
    season = 1.0 + 0.2 * np.cos(2.0 * np.pi * weeks / N_WEEKS)
    doses: dict[str, Array] = {}
    for channel, truth in p.channels.items():
        planner = rng.normal(0.0, 0.25, size=(N_MARKETS, N_WEEKS))
        doses[channel] = (
            truth.spend
            * SIZES[:, None]
            * season[None, :]
            * np.exp(p.demand_on_spend * demand + planner)
        )
    noise = rng.normal(0.0, p.noise_share * p.baseline, size=(N_MARKETS, N_WEEKS)) * np.sqrt(
        SIZES[:, None]
    )
    guard_rng = np.random.default_rng([p.seed, year, 7])
    guardrails = {
        name: (0.5 + 0.1 * j) * SIZES[:, None]
        + guard_rng.normal(0.0, 0.05, size=(N_MARKETS, N_WEEKS))
        for j, name in enumerate(p.guardrails)
    }
    order_value = (p.order_value or 0.0) * (1.0 + rng.normal(0.0, 0.03, size=(N_MARKETS, N_WEEKS)))
    return World(
        party=p,
        spec=spec,
        theta=theta,
        doses=doses,
        demand=demand,
        noise=noise,
        guardrail_values=guardrails,
        order_value=order_value,
        year=year,
    )


# -- the calendar ----------------------------------------------------------------------

Kind = Literal["holdout", "dose", "rollout"]


@dataclass(frozen=True)
class Experiment:
    """One entry on the house's calendar, as proposed."""

    id: str
    party: str
    channel: str
    kind: Kind
    start: int
    stop: int
    n_treated: int = 10
    uplift: float = 0.0
    randomized: bool = True
    note: str = ""

    @property
    def window(self) -> TimeWindow:
        return TimeWindow(start=self.start, stop=self.stop)

    @property
    def pre(self) -> slice:
        return slice(self.start - PRE_WEEKS, self.start)

    @property
    def post(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def weeks(self) -> int:
        return self.stop - self.start


#: The year as the three parties proposed it, before the house looked at it together.
#: Nine entries; notebook 2 decides which of them run, and when.
CALENDAR: tuple[Experiment, ...] = (
    Experiment("NW-01", "northwind", "search", "holdout", 8, 16, note="watched weekly"),
    Experiment("NW-02", "northwind", "video", "dose", 20, 28, uplift=1.0, note="spend doubled"),
    Experiment("NW-03", "northwind", "search", "holdout", 12, 20, note="a second search test"),
    Experiment("AC-01", "acme", "social", "holdout", 10, 18, n_treated=20),
    Experiment("AC-02", "acme", "search", "holdout", 14, 22, n_treated=20),
    Experiment(
        "AC-03",
        "acme",
        "onboarding",
        "rollout",
        30,
        40,
        n_treated=15,
        randomized=False,
        note="new onboarding flow, largest markets first",
    ),
    Experiment("AC-04", "acme", "search", "holdout", 34, 42, n_treated=20),
    Experiment("GX-01", "globex", "tv", "holdout", 8, 16),
    Experiment("GX-02", "globex", "display", "holdout", 30, 38, note="four scheduled looks"),
    Experiment("GX-03", "globex", "search", "holdout", 40, 48),
)


def experiment(experiment_id: str) -> Experiment:
    for e in CALENDAR:
        if e.id == experiment_id:
            return e
    raise KeyError(experiment_id)


def occupancy(exp: Experiment, markets: Sequence[str]) -> Occupancy:
    """What the experiment occupies on the calendar: the markets it *touches* (both
    arms — a holdout's control markets are still spoken for), its weeks, its lever."""
    return Occupancy(
        experiment=exp.id,
        units=tuple(markets),
        window=exp.window,
        treatments=(exp.channel,),
        randomized=exp.randomized,
    )


# -- what happens in the field ---------------------------------------------------------


@dataclass(frozen=True)
class Fielded:
    """One experiment as it actually ran: the doses the world saw, and the gaps."""

    experiment: Experiment
    treated: tuple[str, ...]
    control: tuple[str, ...]
    doses: dict[str, Array]
    #: markets assigned to treatment that did not receive it as assigned
    delivered_fraction: dict[str, float] = field(default_factory=dict)
    #: markets whose KPI was not reported, by week
    missing: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def markets(self) -> tuple[str, ...]:
        return tuple(sorted((*self.treated, *self.control)))

    def treated_indices(self) -> list[int]:
        return [MARKETS.index(m) for m in self.treated]

    def control_indices(self) -> list[int]:
        return [MARKETS.index(m) for m in self.control]

    def exposed(self, *, threshold: float = 0.8) -> dict[str, int]:
        """Markets per arm that received what they were assigned."""
        got = sum(1 for m in self.treated if self.delivered_fraction.get(m, 1.0) >= threshold)
        return {"control": len(self.control), "treated": got}


def field_experiment(
    w: World,
    exp: Experiment,
    treated: Sequence[str],
    control: Sequence[str],
    *,
    doses: Mapping[str, Array] | None = None,
) -> Fielded:
    """Apply one experiment to a party's doses, with whatever the field did to it.

    ``doses`` lets experiments stack: pass the doses a previous experiment
    left behind and this one is laid on top. The world's quirks are keyed by
    experiment id, because they are facts about the year, not parameters.
    """
    d = {k: v.copy() for k, v in (w.doses if doses is None else doses).items()}
    rng = np.random.default_rng([w.party.seed, hash(exp.id) % 10_000])
    rows = [MARKETS.index(m) for m in treated]
    delivered: dict[str, float] = {}
    missing: dict[str, tuple[int, ...]] = {}
    if exp.kind == "holdout":
        keep_spending: set[str] = set()
        if exp.id == "GX-03":
            # Three of the treated markets have retailer co-op search ads the house
            # does not control; the holdout never reached them.
            keep_spending = set(sorted(treated, key=lambda m: -SIZES[MARKETS.index(m)])[:3])
        for m, i in zip(treated, rows, strict=True):
            if m in keep_spending:
                delivered[m] = 0.0
                continue
            d[exp.channel][i, exp.post] = 0.0
            delivered[m] = 1.0
    elif exp.kind == "dose":
        for m, i in zip(treated, rows, strict=True):
            fraction = 1.0
            if exp.id == "NW-02":
                # The platform could not clear the doubled budget everywhere.
                fraction = float(np.clip(rng.beta(5.0, 2.0), 0.3, 1.0))
            d[exp.channel][i, exp.post] *= 1.0 + exp.uplift * fraction
            delivered[m] = fraction
    elif exp.kind == "rollout":
        pass  # the rollout moves the KPI, not the spend; see ``rollout_bump``
    if exp.id == "AC-04":
        # acme's KPI feed vendor drops a market after a week with too few events to
        # bill for. The treated markets were being watched and were re-enabled by
        # hand; the control markets were not — and the ones that went dark are the
        # ones whose demand fell in the window, which is selection on the outcome.
        low_demand = sorted(
            control, key=lambda m: float(w.demand[MARKETS.index(m), exp.post].mean())
        )
        for m in low_demand[:10]:
            missing[m] = tuple(range(exp.start + 2, exp.stop))
    return Fielded(exp, tuple(treated), tuple(control), d, delivered, missing)


def rollout_bump(w: World, exp: Experiment, markets: Sequence[str]) -> Array:
    """What acme's onboarding rollout adds to the KPI: a step in the rolled-out markets."""
    bump = np.zeros((N_MARKETS, N_WEEKS))
    for m in markets:
        bump[MARKETS.index(m), exp.post] = 0.06 * w.party.baseline * SIZES[MARKETS.index(m)]
    return bump


def with_missing(kpi: Array, missing: Mapping[str, Sequence[int]]) -> Array:
    """The KPI array with the unreported cells as NaN — what the analyst was handed."""
    out = kpi.copy()
    for m, weeks in missing.items():
        out[MARKETS.index(m), list(weeks)] = np.nan
    return out


def observed_markets(kpi: Array, post: slice) -> list[int]:
    """Row indices whose post-period KPI is complete."""
    return [i for i in range(kpi.shape[0]) if np.all(np.isfinite(kpi[i, post]))]


def market_kind(markets: Sequence[str]) -> list[str]:
    return [TIERS[MARKETS.index(m)] for m in markets]


# -- the house style -------------------------------------------------------------------

# The shared notebook style: one registered plotly template and one validated palette
# for every notebook under nbs/. Imported for its side effect (the template) and for
# the colours below, so this case study cannot drift into a look of its own.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_NBS = str(_Path(__file__).resolve().parents[2])
if _NBS not in _sys.path:
    _sys.path.insert(0, _NBS)

from _style import AQUA, BLUE, CRITICAL, GOOD, MUTED, ORANGE, VIOLET, WARNING  # noqa: E402

PARTY_COLOR = {"northwind": BLUE, "acme": ORANGE, "globex": AQUA}
TRUTH_COLOR = MUTED
ACCENT = VIOLET
__all__ = [
    "ACCENT",
    "CALENDAR",
    "CRITICAL",
    "GOOD",
    "MARKETS",
    "N_MARKETS",
    "N_WEEKS",
    "PARTIES",
    "PARTY_COLOR",
    "PRE_WEEKS",
    "SIZES",
    "SPEND_UNIT",
    "TIERS",
    "TRUTH_COLOR",
    "WARNING",
    "Experiment",
    "Fielded",
    "Party",
    "World",
    "analyst_spec",
    "channel_entity",
    "experiment",
    "field_experiment",
    "kpi_entity",
    "market_kind",
    "observed_markets",
    "occupancy",
    "roles_for",
    "rollout_bump",
    "spec_for",
    "truth_for",
    "KPI_COLUMN",
    "with_missing",
    "world",
]
