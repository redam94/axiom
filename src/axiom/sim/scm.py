"""``LinearSCM``: a linear-Gaussian structural causal model with known ground truth.

Every recovery test in the suite runs an estimator against one of these and
asks whether it found the number the model was built with. The model is a
``Spec``: the graph, the coefficients, the noise scales, and the latent
scales are all data, so a world round-trips as JSON and carries a content
hash like every other declarative object.

Structural equations (Pearl 2009, §1.4; Wright 1921 for the path rule)::

    V_j = c_j + sum_{i in pa(j)} beta_{i->j} V_i + sum_{k : j in U_k} U_k + eps_j

with ``eps_j ~ N(0, noise_sd[j]^2)`` independent across nodes and one latent
``U_k ~ N(0, latent_sd[k]^2)`` per bidirected edge, added to both of its
endpoints. ``do(X = x)`` replaces ``X``'s equation by the constant ``x`` and
leaves every other equation alone (Pearl 2009, Def. 3.2.1). In a linear model
the average causal effect of ``X`` on ``Y`` per unit of ``X`` is the sum over
directed paths of the product of edge coefficients — Wright's rule — and
``total_effect`` returns exactly that, so a test can compare an estimator to
the truth analytically as well as by Monte Carlo through ``simulate``.

The graph is an ``axiom.identify.CausalGraph`` held in the ``graph`` field:
the same object the identification machinery reads, so a world and its
verdict can never disagree about the edges. Node order in a simulated frame
is ``graph.topological_order()``.
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import field_validator, model_validator

from axiom.core.spec import Spec, SpecError
from axiom.identify.graph import CausalGraph

__all__ = ["LinearSCM", "SCMError", "coefficient_key", "latent_key"]

Edge = tuple[str, str]

_NAME = r"[A-Za-z_][\w.\-]*"
_KEY_RE = re.compile(rf"^\s*({_NAME})\s*->\s*({_NAME})\s*$")
_LATENT_KEY_RE = re.compile(rf"^\s*({_NAME})\s*<->\s*({_NAME})\s*$")
_ITEM_RE = re.compile(rf"^\s*({_NAME})\s*(->|<->|<-)\s*({_NAME})\s*:\s*([^\s:]+)\s*$")


class SCMError(SpecError):
    """The structural model is malformed (missing coefficient, unknown node, bad key)."""


def coefficient_key(a: str, b: str) -> str:
    """The ``coefficients`` key for the directed edge ``a -> b``: ``"a->b"``."""
    return f"{a}->{b}"


def latent_key(a: str, b: str) -> str:
    """The ``latent_sd`` key for the bidirected edge ``a <-> b``: ``"a<->b"`` with ``a < b``."""
    x, y = sorted((a, b))
    return f"{x}<->{y}"


def _parse_key(key: str, pattern: re.Pattern[str], kind: str) -> Edge:
    m = pattern.match(key)
    if not m:
        raise SCMError(f"cannot parse {kind} key {key!r}")
    return m.group(1), m.group(2)


class LinearSCM(Spec):
    """A linear-Gaussian SCM over a ``CausalGraph`` with optional latent confounding.

    ``coefficients`` is keyed ``"a->b"`` and must have exactly one entry per
    directed edge of ``graph``. ``noise_sd`` and ``intercepts`` are per node
    and default to ``1.0`` and ``0.0``; ``latent_sd`` is keyed ``"a<->b"``
    (endpoints sorted) and defaults to ``1.0`` per bidirected edge. The
    graph's ``unmeasured``, ``selection``, ``feedback`` and ``name`` are read
    through the properties of the same name.
    """

    graph: CausalGraph
    coefficients: dict[str, float] = {}
    noise_sd: dict[str, float] = {}
    latent_sd: dict[str, float] = {}
    intercepts: dict[str, float] = {}

    # -- construction -----------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        nodes: Iterable[str] = (),
        unmeasured: Iterable[str] = (),
        selection: Iterable[str] = (),
        feedback: bool = False,
        name: str = "",
        noise_sd: Mapping[str, float] | None = None,
        intercepts: Mapping[str, float] | None = None,
    ) -> LinearSCM:
        """Parse ``"Z -> X: 0.8, X -> Y: 2.0, X <-> Y: 0.7"`` into a model.

        Each item is ``edge: number``. A directed item sets the edge's
        coefficient; ``A <- B: c`` means ``B -> A: c``; a bidirected item sets
        the latent scale for that pair. Separators are commas, semicolons,
        or newlines. An edge given twice is an ``SCMError``.
        """
        edges: list[Edge] = []
        bidirected: list[Edge] = []
        coefficients: dict[str, float] = {}
        latent_sd: dict[str, float] = {}
        for chunk in re.split(r"[,;\n]", text):
            if not chunk.strip():
                continue
            m = _ITEM_RE.match(chunk)
            if not m:
                raise SCMError(
                    f"cannot parse item {chunk.strip()!r}; expected 'A -> B: 1.0' or 'A <-> B: 1.0'"
                )
            a, op, b, raw = m.groups()
            try:
                value = float(raw)
            except ValueError as e:
                raise SCMError(f"cannot parse number {raw!r} in item {chunk.strip()!r}") from e
            if op == "<->":
                key = latent_key(a, b)
                if key in latent_sd:
                    raise SCMError(f"bidirected edge {key!r} given more than once")
                bidirected.append((a, b))
                latent_sd[key] = value
                continue
            if op == "<-":
                a, b = b, a
            key = coefficient_key(a, b)
            if key in coefficients:
                raise SCMError(f"edge {key!r} given more than once")
            edges.append((a, b))
            coefficients[key] = value
        graph = CausalGraph(
            nodes=tuple(nodes),
            edges=tuple(edges),
            bidirected=tuple(bidirected),
            unmeasured=tuple(unmeasured),
            selection=tuple(selection),
            feedback=feedback,
            name=name,
        )
        return cls(
            graph=graph,
            coefficients=coefficients,
            noise_sd=dict(noise_sd or {}),
            latent_sd=latent_sd,
            intercepts=dict(intercepts or {}),
        )

    @field_validator("coefficients", mode="before")
    @classmethod
    def _norm_coefficients(cls, v: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in v.items():
            a, b = _parse_key(str(key), _KEY_RE, "coefficient")
            norm = coefficient_key(a, b)
            if norm in out:
                raise SCMError(f"coefficient key {key!r} duplicates {norm!r}")
            out[norm] = float(value)
        return dict(sorted(out.items()))

    @field_validator("latent_sd", mode="before")
    @classmethod
    def _norm_latent_sd(cls, v: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in v.items():
            a, b = _parse_key(str(key), _LATENT_KEY_RE, "latent_sd")
            norm = latent_key(a, b)
            if norm in out:
                raise SCMError(f"latent_sd key {key!r} duplicates {norm!r}")
            out[norm] = float(value)
        return dict(sorted(out.items()))

    @field_validator("noise_sd", "intercepts", mode="before")
    @classmethod
    def _norm_per_node(cls, v: Mapping[str, float]) -> dict[str, float]:
        return dict(sorted((str(k), float(x)) for k, x in v.items()))

    @model_validator(mode="after")
    def _well_formed(self) -> LinearSCM:
        known = set(self.graph.nodes)
        for label, names in (
            ("noise_sd", tuple(self.noise_sd)),
            ("intercepts", tuple(self.intercepts)),
        ):
            unknown = sorted(set(names) - known)
            if unknown:
                raise SCMError(f"{label} names unknown nodes: {unknown}")
        expected = {coefficient_key(a, b) for a, b in self.graph.edges}
        missing = sorted(expected - set(self.coefficients))
        extra = sorted(set(self.coefficients) - expected)
        if missing:
            raise SCMError(f"edges without a coefficient: {missing}")
        if extra:
            raise SCMError(f"coefficients for edges not in the graph: {extra}")
        expected_latent = {latent_key(a, b) for a, b in self.graph.bidirected}
        extra_latent = sorted(set(self.latent_sd) - expected_latent)
        if extra_latent:
            raise SCMError(f"latent_sd for bidirected edges not in the graph: {extra_latent}")
        for label, table in (("noise_sd", self.noise_sd), ("latent_sd", self.latent_sd)):
            bad = sorted(k for k, s in table.items() if not (np.isfinite(s) and s >= 0.0))
            if bad:
                raise SCMError(f"{label} must be finite and non-negative; offending: {bad}")
        for label, table in (("coefficients", self.coefficients), ("intercepts", self.intercepts)):
            bad = sorted(k for k, c in table.items() if not np.isfinite(c))
            if bad:
                raise SCMError(f"{label} must be finite; offending: {bad}")
        return self

    # -- graph views (delegated to ``graph``) ------------------------------------------

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.graph.nodes

    @property
    def edges(self) -> tuple[Edge, ...]:
        return self.graph.edges

    @property
    def bidirected(self) -> tuple[Edge, ...]:
        return self.graph.bidirected

    @property
    def unmeasured(self) -> tuple[str, ...]:
        return self.graph.unmeasured

    @property
    def selection(self) -> tuple[str, ...]:
        return self.graph.selection

    @property
    def feedback(self) -> bool:
        return self.graph.feedback

    @property
    def name(self) -> str:
        return self.graph.name

    @property
    def measured(self) -> tuple[str, ...]:
        """The measured nodes in ``graph.nodes`` order (the columns ``observed`` keeps)."""
        return tuple(n for n in self.graph.nodes if n not in self.graph.unmeasured)

    def parents(self, node: str) -> frozenset[str]:
        self._require(node)
        return self.graph.parents(node)

    def children(self, node: str) -> frozenset[str]:
        self._require(node)
        return self.graph.children(node)

    def topological_order(self) -> tuple[str, ...]:
        """``graph.topological_order()``: the column order of a simulated frame."""
        return self.graph.topological_order()

    def directed_paths(self, source: str, target: str) -> tuple[tuple[str, ...], ...]:
        self._require(source)
        self._require(target)
        return self.graph.directed_paths(source, target)

    # -- ground truth ---------------------------------------------------------------

    def coefficient(self, a: str, b: str) -> float:
        """The structural coefficient on ``a -> b``; ``0.0`` when there is no such edge."""
        self._require(a)
        self._require(b)
        return self.coefficients.get(coefficient_key(a, b), 0.0)

    def direct_effect(self, x: str, y: str) -> float:
        """The controlled direct effect of ``x`` on ``y``: the coefficient on ``x -> y`` or 0."""
        return self.coefficient(x, y)

    def total_effect(self, x: str, y: str) -> float:
        """The average causal effect of ``x`` on ``y`` per unit of ``x``.

        Wright's path rule: the sum over directed paths ``x -> ... -> y`` of
        the product of the edge coefficients along each path. Equal to
        ``d E[Y | do(X = x)] / dx`` in a linear SCM.
        """
        total = 0.0
        for path in self.directed_paths(x, y):
            product = 1.0
            for a, b in zip(path[:-1], path[1:], strict=True):
                product *= self.coefficients[coefficient_key(a, b)]
            total += product
        return total

    def interventional_mean(self, node: str, intervene: Mapping[str, float] | None = None) -> float:
        """``E[node | do(intervene)]`` computed exactly by propagating means.

        With no intervention this is the observational mean. Latents and
        noise have mean zero, so only intercepts and coefficients enter.
        """
        self._require(node)
        fixed = self._check_intervention(intervene)
        means: dict[str, float] = {}
        for n in self.graph.topological_order():
            if n in fixed:
                means[n] = fixed[n]
                continue
            m = self.intercepts.get(n, 0.0)
            for p in self.graph.parents(n):
                m += self.coefficients[coefficient_key(p, n)] * means[p]
            means[n] = m
        return means[node]

    # -- simulation -------------------------------------------------------------------

    def simulate(
        self,
        n: int,
        seed: int | None = None,
        *,
        intervene: Mapping[str, float] | None = None,
    ) -> pd.DataFrame:
        """Draw ``n`` rows from the model, optionally under ``do(intervene)``.

        Returns every node — unmeasured ones included — as a float column, in
        topological order. ``intervene={"X": 1.0}`` fixes ``X`` to the
        constant and ignores the edges and latents into it (Pearl 2009,
        Def. 3.2.1). Draw order is fixed: one latent per bidirected edge in
        sorted order, then each node's noise in topological order, so the
        same seed gives the same frame under every intervention and the
        observational and interventional frames share noise draws.
        """
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")
        fixed = self._check_intervention(intervene)
        rng = np.random.default_rng(seed)
        return self._simulate(n, rng, fixed)

    def _simulate(
        self, n: int, rng: np.random.Generator, fixed: Mapping[str, float]
    ) -> pd.DataFrame:
        bidirected = self.graph.bidirected
        latents: dict[str, npt.NDArray[np.float64]] = {
            latent_key(a, b): rng.normal(0.0, self.latent_sd.get(latent_key(a, b), 1.0), n)
            for a, b in bidirected
        }
        columns: dict[str, npt.NDArray[np.float64]] = {}
        for node in self.graph.topological_order():
            noise = rng.normal(0.0, self.noise_sd.get(node, 1.0), n)
            if node in fixed:
                columns[node] = np.full(n, fixed[node], dtype=np.float64)
                continue
            value = np.full(n, self.intercepts.get(node, 0.0), dtype=np.float64)
            for p in sorted(self.graph.parents(node)):
                value += self.coefficients[coefficient_key(p, node)] * columns[p]
            for a, b in bidirected:
                if node in (a, b):
                    value += latents[latent_key(a, b)]
            columns[node] = value + noise
        return pd.DataFrame(columns)

    def observed(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The frame an analyst sees: ``frame`` without the ``unmeasured`` columns."""
        drop = [c for c in self.graph.unmeasured if c in frame.columns]
        return frame.drop(columns=drop)

    # -- helpers ------------------------------------------------------------------------

    def to_text(self) -> str:
        """The ``from_text`` form, with ``repr`` floats so ``from_text(to_text())`` round-trips."""
        parts = [
            f"{a} -> {b}: {self.coefficients[coefficient_key(a, b)]!r}" for a, b in self.graph.edges
        ]
        parts += [
            f"{a} <-> {b}: {self.latent_sd.get(latent_key(a, b), 1.0)!r}"
            for a, b in self.graph.bidirected
        ]
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.to_text() or f"LinearSCM(nodes={list(self.graph.nodes)})"

    def _check_intervention(self, intervene: Mapping[str, float] | None) -> dict[str, float]:
        fixed: dict[str, float] = {}
        for key, raw in (intervene or {}).items():
            node = str(key)
            self._require(node)
            if isinstance(raw, bool | np.complexfloating) or not isinstance(
                raw, numbers.Real | np.integer | np.floating
            ):
                raise SCMError(
                    f"intervention on {node!r} must be a real number, "
                    f"got {type(raw).__name__} {raw!r}"
                )
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"intervention on {node!r} must be finite, got {value}")
            fixed[node] = value
        return fixed

    def _require(self, node: str) -> None:
        if node not in self.graph.nodes:
            raise SCMError(f"unknown node {node!r}; nodes are {list(self.graph.nodes)}")
