"""Identification for systems that are not DAGs until you unroll them.

``CausalGraph`` carries a ``feedback`` flag whose only honest use is to say
"the static summary graph hides treatment-outcome feedback over time, so
static adjustment is not enough". That flag is a refusal. This module is the
answer to it: take the ``axiom.dynamics.DynamicSystem`` the feedback
actually lives in, unroll it over the periods you plan to analyse, and ask
the ordinary graphical questions of the ordinary DAG that comes out.

Two things become sayable that the summary graph cannot say.

**Simultaneity becomes a reduced form.** A contemporaneous cycle is not a
DAG and never will be. Its *reduced form* is: solve the block and every
member depends on the block's parents and on nothing inside the block.
``unrolled_graph`` emits that, so a simultaneous system has a causal graph —
just not one with arrows between the simultaneous variables, which is
exactly the claim that no ordering of them is causal within the period.

**Time-varying confounding becomes visible.** With ``dose@1 -> outcome@2 ->
dose@3 -> outcome@4``, ``outcome@2`` is a confounder of the later dose *and*
a mediator of the earlier one. Adjusting for it is required and forbidden by
the same static criterion, which is why static adjustment fails and why the
g-formula exists (Robins 1986). ``sequential_plan`` runs the sequential
back-door criterion (Pearl 2009, §4.4.3) stage by stage on the unrolled
graph and returns the per-stage adjustment sets, or says which stage has no
admissible set and why.

What it checks, at stage ``k`` of treatments ``A_1 ... A_n``:

1. every element of ``Z_1 ... Z_k`` is measured and is not a descendant of
   ``A_k, ..., A_n``;
2. ``A_k`` and the outcome are d-separated by ``{A_1..A_{k-1}} u {Z_1..Z_k}``
   in the graph with every edge *out of* ``A_k ... A_n`` deleted.

Together these give sequential ignorability at every stage, which is what
licenses the g-formula. Positivity — that every treatment history the plan
asks about actually occurs — is not a graphical property and cannot be
checked here; it rides along as a named, unverified assumption, so a plan
that passes comes back ``downgraded`` rather than ``identified``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import Field, model_validator

from axiom.core import Assumption, NonEmptyStr, Spec, Verdict
from axiom.dynamics import DynamicSystem, time_ref, unrolled_edges
from axiom.identify.graph import CausalGraph, GraphError

__all__ = [
    "SequentialPlan",
    "sequential_backdoor_admissible",
    "sequential_plan",
    "unrolled_graph",
]


def unrolled_graph(
    system: DynamicSystem,
    periods: int,
    *,
    reduced: bool = True,
    name: str = "",
) -> CausalGraph:
    """The time-indexed causal graph of a dynamic system, over ``periods`` periods.

    Nodes are ``"variable.t0" ... "variable.t{periods-1}"`` for every declared
    variable; a variable declared ``observed=False`` is marked unmeasured at
    every period, so an adjustment set that needs it is a downgrade rather
    than a silent lie.

    ``reduced=True`` emits the reduced form of each simultaneous block, which
    is acyclic. ``reduced=False`` emits the structural edges as written; if
    the system has a contemporaneous cycle that is not a DAG and
    construction raises ``GraphError`` naming the cycle — the honest outcome,
    since no DAG algorithm can answer a question about it.
    """
    if periods < 1:
        raise ValueError(f"periods must be at least 1, got {periods}")
    nodes = tuple(time_ref(v.name, t) for v in system.variables for t in range(periods))
    unmeasured = tuple(
        time_ref(v.name, t) for v in system.variables if not v.observed for t in range(periods)
    )
    edges = unrolled_edges(system, periods, reduced=reduced)
    try:
        return CausalGraph(
            nodes=nodes,
            edges=edges,
            unmeasured=unmeasured,
            name=name or (f"{system.name}@{periods}" if system.name else f"unrolled@{periods}"),
        )
    except GraphError as e:
        raise GraphError(
            f"the structural graph of {system.name or 'the system'} is cyclic ({e}); "
            "call unrolled_graph(..., reduced=True) for the reduced form, which is a DAG"
        ) from e


def _mutilated(graph: CausalGraph, treatments: Sequence[str]) -> CausalGraph:
    """``G`` with every edge out of every named treatment deleted."""
    out = graph
    for a in treatments:
        out = out.remove_edges_out_of(a)
    return out


def sequential_backdoor_admissible(
    graph: CausalGraph,
    treatments: Sequence[str],
    outcome: str,
    adjustments: Sequence[Iterable[str]],
    *,
    measured_only: bool = True,
) -> tuple[bool, str]:
    """The sequential back-door criterion; ``(True, "")`` or ``(False, why)``.

    ``adjustments[k]`` is the covariate set introduced at stage ``k``; the
    conditioning set at that stage is every earlier treatment together with
    the union of ``adjustments[:k+1]``.
    """
    if len(adjustments) != len(treatments):
        raise ValueError(
            f"{len(treatments)} treatments need {len(treatments)} adjustment sets, "
            f"got {len(adjustments)}"
        )
    for node in (*treatments, outcome):
        graph._require(node)
    sets = [frozenset(z) for z in adjustments]
    for stage, (a, z) in enumerate(zip(treatments, sets, strict=True)):
        for n in z:
            graph._require(n)
        history = frozenset(treatments[:stage]) | frozenset().union(*sets[: stage + 1], frozenset())
        later = tuple(treatments[stage:])
        forbidden = graph.descendants(later[0], include_self=True)
        for a_later in later[1:]:
            forbidden |= graph.descendants(a_later, include_self=True)
        offending = sorted((history - frozenset(treatments[:stage])) & forbidden)
        if offending:
            return False, (
                f"stage {stage + 1} ({a}): {offending} are descendants of the treatment "
                "at this stage or a later one, so conditioning on them opens a path "
                "the g-formula cannot close"
            )
        if measured_only:
            hidden = sorted(n for n in history if n in graph.unmeasured)
            if hidden:
                return False, f"stage {stage + 1} ({a}): {hidden} are not measured"
        mutilated = _mutilated(graph, later)
        if not mutilated.d_separated(a, outcome, history - {a}):
            return False, (
                f"stage {stage + 1} ({a}): a back-door path to {outcome} stays open given "
                f"{sorted(history - {a})}"
            )
    return True, ""


class SequentialPlan(Spec):
    """Per-stage adjustment sets for a sequence of treatments, and the verdict on them.

    ``stages`` are the treatment nodes in time order; ``adjustments[k]`` is
    what to condition on when stage ``k`` is modelled, *in addition to* the
    treatments before it. ``licensed`` follows the verdict: a plan that
    satisfies the graphical criterion is ``downgraded``, not ``identified``,
    because positivity is not graphical.
    """

    stages: tuple[NonEmptyStr, ...] = Field(min_length=1)
    outcome: NonEmptyStr
    graph_hash: str
    adjustments: tuple[tuple[str, ...], ...]
    verdict: Verdict
    static_adjustment_fails: bool = False
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _aligned(self) -> SequentialPlan:
        if len(self.adjustments) != len(self.stages):
            raise ValueError("one adjustment set per stage")
        return self

    @property
    def licensed(self) -> bool:
        return self.verdict.licensed

    @property
    def conditioning_at(self) -> tuple[tuple[str, ...], ...]:
        """The full conditioning set at each stage: earlier treatments plus the adjustments."""
        out: list[tuple[str, ...]] = []
        seen: set[str] = set()
        for stage, extra in enumerate(self.adjustments):
            seen |= set(extra)
            out.append(tuple(sorted(seen | set(self.stages[:stage]))))
        return tuple(out)


def _positivity(stages: Sequence[str]) -> Assumption:
    return Assumption(
        name="positivity",
        facet="identification",
        statement=(
            "every treatment history the plan asks about occurs with positive probability "
            f"at each of the {len(stages)} stages, within every level of the adjustment sets"
        ),
        challenged_by="a stage where the observed regime is deterministic given the history",
    )


def _sequential_exchangeability(stages: Sequence[str]) -> Assumption:
    return Assumption(
        name="sequential_exchangeability",
        facet="identification",
        statement=(
            "at every stage the treatment is independent of the potential outcome given the "
            "measured history, which the unrolled graph's sequential back-door criterion checks"
        ),
        challenged_by="an unmeasured cause shared by a treatment stage and the outcome",
        state="satisfied",
    )


def sequential_plan(
    graph: CausalGraph,
    treatments: Sequence[str],
    outcome: str,
    *,
    candidates: Iterable[str] | None = None,
    prune: bool = True,
) -> SequentialPlan:
    """Search for per-stage adjustment sets satisfying the sequential back-door criterion.

    The search starts from the largest defensible set at each stage — every
    measured node that is not a descendant of this or a later treatment, and
    is an ancestor of the outcome or of some treatment — and, when ``prune``,
    greedily drops elements that are not needed. If the starting set is not
    admissible, no subset is either, and the plan comes back blocked naming
    the stage.

    ``static_adjustment_fails`` records the thing worth knowing even when a
    plan is found: whether one time-invariant adjustment set would have done
    the job. When it is true, a static analysis of this system is wrong, not
    merely less efficient.
    """
    stages = tuple(treatments)
    if not stages:
        raise ValueError("a sequential plan needs at least one treatment stage")
    for node in (*stages, outcome):
        graph._require(node)
    pool = frozenset(candidates) if candidates is not None else graph.measured
    pool = pool - set(stages) - {outcome}
    relevant = graph.ancestors(outcome, include_self=False) | graph.ancestors(
        stages, include_self=False
    )

    chosen: list[tuple[str, ...]] = []
    for stage, a in enumerate(stages):
        later = stages[stage:]
        forbidden: frozenset[str] = frozenset()
        for a_later in later:
            forbidden |= graph.descendants(a_later, include_self=True)
        already = {n for earlier in chosen for n in earlier}
        start = sorted((pool & relevant) - forbidden - already)
        candidate = [*chosen, tuple(start)]
        ok, why = sequential_backdoor_admissible(
            graph, stages[: stage + 1], outcome, candidate, measured_only=True
        )
        if not ok:
            # The stages after the failure were never searched; they are reported as
            # empty rather than omitted, so the tuple still lines up with `stages`
            # and `detail["failed_stage"]` says where the search stopped.
            unreached = [()] * (len(stages) - stage - 1)
            return SequentialPlan(
                stages=stages,
                outcome=outcome,
                graph_hash=graph.content_hash(),
                adjustments=tuple([*chosen, tuple(start), *unreached]),
                verdict=Verdict(
                    status="blocked",
                    reason=(
                        f"no admissible adjustment set exists at stage {stage + 1}: {why}. "
                        "Measure the variable the open path runs through, or intervene at "
                        "that stage rather than conditioning on it"
                    ),
                    route="sequential_backdoor",
                ),
                detail={"failed_stage": str(stage + 1), "treatment": a},
            )
        kept = list(start)
        if prune:
            for name in start:
                trial = [n for n in kept if n != name]
                fine, _ = sequential_backdoor_admissible(
                    graph, stages[: stage + 1], outcome, [*chosen, tuple(trial)], measured_only=True
                )
                if fine:
                    kept = trial
        chosen.append(tuple(kept))

    static = _static_would_fail(graph, stages, outcome)
    return SequentialPlan(
        stages=stages,
        outcome=outcome,
        graph_hash=graph.content_hash(),
        adjustments=tuple(chosen),
        verdict=Verdict(
            status="downgraded",
            reason="",
            assumptions=(_sequential_exchangeability(stages), _positivity(stages)),
            route="sequential_backdoor",
        ),
        static_adjustment_fails=static,
        detail={"stages": str(len(stages))},
    )


def _static_would_fail(graph: CausalGraph, stages: Sequence[str], outcome: str) -> bool:
    """True when some stage's adjustment set contains a descendant of an earlier stage.

    That is the signature of time-varying confounding: a variable that must
    be conditioned on for a later treatment and must not be for an earlier
    one. One time-invariant set cannot do both.
    """
    for stage, a in enumerate(stages[1:], start=1):
        earlier = graph.descendants(stages[:stage], include_self=False)
        later = tuple(stages[stage:])
        forbidden: frozenset[str] = frozenset()
        for a_later in later:
            forbidden |= graph.descendants(a_later, include_self=True)
        needed = (graph.measured & earlier) - forbidden - {outcome} - set(stages)
        for z in sorted(needed):
            mutilated = _mutilated(graph, later)
            without = frozenset(stages[:stage])
            if not mutilated.d_separated(a, outcome, without) and mutilated.d_separated(
                a, outcome, without | {z}
            ):
                return True
    return False
