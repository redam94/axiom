"""The ``analysis.axiom`` directory format. Spec replay, no pickle.

::

    analysis.axiom/
    ├── manifest.json      format version, axiom version, hashes, declared bases, units
    ├── specs/<role>.json  one envelope per Spec
    ├── panel.csv          the Panel's frame (17 significant digits) + roles in specs/
    ├── posterior.npz      draws + JSON meta
    └── evidence/          evidence.jsonl, ledger.jsonl

Loading re-declares the manifest's base dimensions and units, replays each
spec through ``load_spec`` (class resolved by qualified name, never by
executing stored code), rehydrates the posterior from npz, and rebuilds the
``Panel`` from the frame and its stored ``RoleMap``.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from axiom import __version__
from axiom.core.dimensions import BASES, UNITS
from axiom.core.posterior import Posterior
from axiom.core.spec import Spec, SpecError, load_spec
from axiom.core.verdict import LedgerLine
from axiom.data.frame import Panel
from axiom.data.roles import RoleMap
from axiom.io.analysis import Analysis
from axiom.io.provenance import Provenance

__all__ = ["FORMAT_VERSION", "FormatError", "load_analysis", "save_analysis"]

FORMAT_VERSION = "1"
_ROLE_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


class FormatError(SpecError):
    """The directory is not a readable ``analysis.axiom`` of a known format version."""


def _role_file(role: str) -> str:
    if not _ROLE_RE.match(role):
        raise FormatError(f"spec role {role!r} must match {_ROLE_RE.pattern}")
    return role.replace(":", "__") + ".json"


def _role_from_file(name: str) -> str:
    return name.removesuffix(".json").replace("__", ":")


def save_analysis(
    analysis: Analysis, path: str | Path, *, seed: int | None = None, overwrite: bool = False
) -> Analysis:
    """Write an analysis and return it with its provenance stamped.

    Refuses to overwrite unless asked; never writes executable state. The
    returned ``Analysis`` compares equal to ``load_analysis(path)``.
    """
    root = Path(path)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists; pass overwrite=True to replace it")
        if not (root / "manifest.json").exists():
            raise FormatError(
                f"{root} exists and is not an analysis directory; refusing to overwrite"
            )
    (root / "specs").mkdir(parents=True, exist_ok=True)
    (root / "evidence").mkdir(exist_ok=True)

    for role, spec in analysis.specs.items():
        (root / "specs" / _role_file(role)).write_text(spec.to_json(indent=2))

    panel_meta: dict[str, Any] | None = None
    if analysis.panel is not None:
        frame = analysis.panel.frame
        (root / "panel.csv").write_text(analysis.panel.to_csv())
        (root / "panel_roles.json").write_text(analysis.panel.roles.to_json(indent=2))
        panel_meta = {
            "format": "csv",
            "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
            "hash": analysis.panel.content_hash(),
        }

    if analysis.posterior is not None:
        analysis.posterior.to_npz(root / "posterior.npz")

    with (root / "evidence" / "evidence.jsonl").open("w") as f:
        for rec in analysis.evidence:
            f.write(rec.to_json() + "\n")
    with (root / "evidence" / "ledger.jsonl").open("w") as f:
        for line in analysis.ledger:
            f.write(line.to_json() + "\n")

    provenance = analysis.provenance or Provenance.now(
        __version__, hashes=analysis.hashes(), seed=seed
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "axiom_version": __version__,
        "provenance": provenance.envelope(),
        "specs": sorted(analysis.specs),
        "panel": panel_meta,
        "posterior": "posterior.npz" if analysis.posterior is not None else None,
        "bases": BASES.declarations(),
        "units": UNITS.units(),
        "conversions": UNITS.conversions(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return replace(analysis, provenance=provenance)


def load_analysis(path: str | Path) -> Analysis:
    root = Path(path)
    mpath = root / "manifest.json"
    if not mpath.exists():
        raise FormatError(f"{root} has no manifest.json")
    manifest = json.loads(mpath.read_text())
    fv = manifest.get("format_version")
    if fv != FORMAT_VERSION:
        raise FormatError(
            f"{root}: format_version {fv!r} is not supported (expected {FORMAT_VERSION!r})"
        )

    for name, symbol in manifest.get("bases", {}).items():
        BASES.declare(name, symbol=symbol)
    for unit, base in manifest.get("units", {}).items():
        UNITS.declare(unit, base)
    for src, dst, factor in manifest.get("conversions", []):
        UNITS.register(src, dst, factor)

    specs: dict[str, Spec] = {}
    for role in manifest["specs"]:
        specs[role] = load_spec((root / "specs" / _role_file(role)).read_text())

    panel: Panel | None = None
    if manifest.get("panel") is not None:
        meta = manifest["panel"]
        if meta.get("format") != "csv":
            raise FormatError(f"unknown panel format {meta.get('format')!r}")
        roles = RoleMap.from_json((root / "panel_roles.json").read_text())
        frame = pd.read_csv(
            root / "panel.csv",
            dtype={roles.unit: "str", **_pandas_dtypes(meta["dtypes"])},
            float_precision="round_trip",
        )
        panel = Panel(frame, roles)
        if panel.content_hash() != meta["hash"]:
            raise FormatError("panel.csv does not match the hash recorded in the manifest")

    posterior: Posterior | None = None
    if manifest.get("posterior"):
        posterior = Posterior.from_npz(root / manifest["posterior"])

    evidence = tuple(
        load_spec(line)
        for line in (root / "evidence" / "evidence.jsonl").read_text().splitlines()
        if line.strip()
    )
    ledger_raw = [
        load_spec(line)
        for line in (root / "evidence" / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ledger: list[LedgerLine] = []
    for item in ledger_raw:
        if not isinstance(item, LedgerLine):
            raise FormatError(f"ledger.jsonl contains a non-LedgerLine spec: {type(item).__name__}")
        ledger.append(item)

    prov_env = manifest["provenance"]
    provenance = Provenance.from_json(json.dumps(prov_env))
    return Analysis(
        specs=specs,
        panel=panel,
        posterior=posterior,
        evidence=evidence,
        ledger=tuple(ledger),
        provenance=provenance,
    )


def _pandas_dtypes(recorded: dict[str, str]) -> dict[str, str]:
    # Let pandas parse numerics; pin only object/string columns so unit labels
    # like "001" do not become integers.
    return {c: "str" for c, t in recorded.items() if t in ("object", "string", "str")}
