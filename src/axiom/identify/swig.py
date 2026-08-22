"""SWIGs: the graph and the potential outcome, in one picture.

A DAG does not show potential outcomes. That sounds like a presentational
complaint and is not: the usual way of reading a DAG as a causal model (the
NPSEM-IE) quietly asserts *cross-world* independences — statements relating
``Y(0)`` and ``Y(1)``, which no experiment can ever check — and those
assumptions are strictly stronger than identification needs.

A **Single-World Intervention Graph** (Richardson & Robins 2013) fixes this by
splitting nodes. To intervene on ``X``, cut it in half: a *random* half that
keeps ``X``'s incoming edges, and a *fixed* half ``x`` that emits its outgoing
edges. Everything downstream becomes a potential outcome ``Y(x)``. Then
ordinary d-separation on the split graph reads off exactly the counterfactual
independences that hold in one hypothetical world — the FFRCISTG model — with
no cross-world assumption anywhere.

Two things follow, and both are checkable rather than decorative.

* **Ignorability becomes visible.** ``Y(x) ⫫ X | Z`` on the SWIG is the
  assumption an adjustment estimator actually needs. It coincides with the
  back-door criterion for ``Z`` — a theorem, and
  ``tests/unit/test_identify_swig.py`` checks it holds over random graphs
  rather than taking it on faith.
* **Sequential regimes get the same treatment.** Split every stage and the
  independences the g-formula needs are read off one picture, which is what
  ``sequential_plan`` checks stage by stage. The two agree, and that is also
  tested.

Node naming is a label, not an identifier: a fixed half is written ``x`` and a
potential outcome ``Y(x)``, following the papers. The parentheses mean these
graphs are built rather than parsed from an edge string.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from axiom.identify.graph import CausalGraph, GraphError

__all__ = [
    "labels_of",
    "potential_outcome",
    "renamed_in",
    "sequential_ignorability",
    "single_world_ignorability",
    "swig",
]


def _labels(
    interventions: Mapping[str, str] | Iterable[str], taken: frozenset[str]
) -> dict[str, str]:
    """Fixed-half names: the literature's lower case where that is free, else a starred name."""
    if isinstance(interventions, Mapping):
        return {str(k): str(v) for k, v in interventions.items()}
    out: dict[str, str] = {}
    for node in interventions:
        name = str(node)
        lowered = name.lower()
        out[name] = lowered if lowered != name and lowered not in taken else f"{name}*"
    return out


def potential_outcome(node: str, labels: Iterable[str]) -> str:
    """``Y(x)`` — the name a node takes once it is downstream of a fixed value."""
    names = sorted(set(labels))
    return f"{node}({', '.join(names)})" if names else node


def swig(
    graph: CausalGraph,
    interventions: Mapping[str, str] | Iterable[str],
) -> CausalGraph:
    """Split every intervened node; relabel its descendants as potential outcomes.

    ``interventions`` names the nodes to intervene on, optionally mapping each
    to the label its fixed half carries (``{"X": "x"}``). The default follows
    the papers — ``X`` becomes ``x`` — falling back to ``X*`` when the lower
    case form is the node's own name or already taken, as it is for a
    time-indexed node like ``dose.t0``.

    The random half keeps the node's incoming edges — it is still caused by
    whatever caused it — and the fixed half takes the outgoing ones. A node is
    relabelled ``Y(x)`` when a fixed half is among its ancestors, using only
    the labels that actually reach it, so a node untouched by the intervention
    keeps its name and its meaning.
    """
    fixed = _labels(interventions, frozenset(graph.nodes))
    if not fixed:
        raise ValueError("a SWIG needs at least one intervention")
    for node in fixed:
        graph._require(node)
    clash = sorted(set(fixed.values()) & set(graph.nodes))
    if clash:
        raise GraphError(
            f"the fixed halves {clash} collide with existing nodes; pass explicit labels"
        )

    # The split, in one line: an edge leaves the *fixed* half of an intervened node
    # and arrives at the *random* half of one, which keeps the node's own name.
    edges = [(fixed.get(a, a), b) for a, b in graph.edges]
    bidirected = list(graph.bidirected)

    # which fixed labels reach each node
    children: dict[str, set[str]] = {}
    for a, b in edges:
        children.setdefault(a, set()).add(b)
    reached: dict[str, set[str]] = {n: set() for n in graph.nodes}
    for label in fixed.values():
        stack = list(children.get(label, ()))
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            reached.setdefault(current, set()).add(label)
            stack.extend(children.get(current, ()))

    def rename(node: str) -> str:
        if node in fixed.values():
            return node
        return potential_outcome(node, reached.get(node, set()))

    return CausalGraph(
        nodes=tuple(sorted({*(rename(n) for n in graph.nodes), *fixed.values()})),
        edges=tuple(sorted({(rename(a), rename(b)) for a, b in edges})),
        bidirected=tuple(sorted({(rename(a), rename(b)) for a, b in bidirected})),
        unmeasured=tuple(sorted({rename(n) for n in graph.unmeasured})),
        name=f"swig({graph.name}; {', '.join(sorted(fixed))})" if graph.name else "",
    )


def labels_of(name: str) -> tuple[str, ...]:
    """The fixed labels a SWIG node carries: ``Y(a, b)`` gives ``("a", "b")``.

    A node that carries any label is a *potential outcome*, and that matters
    for more than display: it is a variable in a hypothetical world, so
    conditioning on it is not something data can do unless the regime that
    labels it is the one the data followed.
    """
    if not name.endswith(")") or "(" not in name:
        return ()
    inside = name[name.index("(") + 1 : -1]
    return tuple(part.strip() for part in inside.split(",") if part.strip())


def renamed_in(split: CausalGraph, node: str) -> str:
    """Find ``node`` in a SWIG, whether or not it was relabelled a potential outcome.

    ``Y`` may have become ``Y(x)``; a caller who asks about ``Y`` means the
    same variable either way, and should not have to reconstruct the label.
    """
    if node in split.nodes:
        return node
    matches = [n for n in split.nodes if n.startswith(f"{node}(")]
    if len(matches) == 1:
        return matches[0]
    raise GraphError(f"no node {node!r} in the SWIG; nodes are {list(split.nodes)}")


def single_world_ignorability(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    adjustment: Iterable[str] = (),
    *,
    label: str = "",
) -> bool:
    """Whether ``Y(x)`` is independent of ``X`` given ``Z`` on the SWIG.

    This is the assumption an adjustment estimator needs, stated about the
    potential outcome rather than about paths. For an acyclic graph it holds
    exactly when ``Z`` satisfies the back-door criterion — the theorem that
    lets a graphical check stand in for a counterfactual one.

    An adjustment variable that the split *relabelled* is a variable in the
    hypothetical world, not the observed one; conditioning on it is not
    something data can do, and no d-separation on the SWIG rescues that. Such
    a set is refused rather than certified — which is the same refusal the
    back-door criterion makes when it bars descendants of the treatment.
    """
    graph._require(treatment)
    graph._require(outcome)
    split = swig(graph, {treatment: label} if label else [treatment])
    conditioning = []
    for node in adjustment:
        graph._require(node)
        renamed = renamed_in(split, node)
        if labels_of(renamed):
            return False
        conditioning.append(renamed)
    return split.d_separated(renamed_in(split, outcome), renamed_in(split, treatment), conditioning)


def sequential_ignorability(
    graph: CausalGraph,
    stages: Sequence[str],
    outcome: str,
    adjustments: Sequence[Iterable[str]],
) -> tuple[bool, str]:
    """Sequential ignorability, read off one SWIG with every stage split.

    At stage ``k`` the requirement is that the potential outcome is
    independent of the stage's treatment given the earlier treatments and the
    covariates chosen so far. Returns ``(True, "")`` or ``(False, why)``.

    This is the potential-outcome statement of what ``sequential_plan``
    checks with the sequential back-door criterion; the two agreeing is a
    property the tests hold them to.
    """
    if len(stages) != len(adjustments):
        raise ValueError(
            f"{len(stages)} stages need {len(stages)} adjustment sets, got {len(adjustments)}"
        )
    split = swig(graph, list(stages))
    stage_label = {stage: renamed_in(split, stage) for stage in stages}
    fixed_label = {
        stage: next(
            n
            for n in split.nodes
            if n not in stage_label.values() and (n == stage.lower() or n == f"{stage}*")
        )
        for stage in stages
    }
    target = renamed_in(split, outcome)
    sets = [frozenset(z) for z in adjustments]
    for index, stage in enumerate(stages):
        here = stage_label[stage]
        later = {fixed_label[s] for s in stages[index:]}
        history = {renamed_in(split, earlier) for earlier in stages[:index]}
        for chosen in sets[: index + 1]:
            for node in chosen:
                renamed = renamed_in(split, node)
                carried = set(labels_of(renamed))
                if carried & later:
                    return False, (
                        f"stage {index + 1} ({stage}): {renamed} is a potential outcome under "
                        f"this stage or a later one, so conditioning on it is a statement about "
                        "a world the data did not follow"
                    )
                history.add(renamed)
        if not split.d_separated(target, here, history):
            return False, (
                f"stage {index + 1} ({stage}): {target} is not independent of {here} given "
                f"{sorted(history)}"
            )
    return True, ""
