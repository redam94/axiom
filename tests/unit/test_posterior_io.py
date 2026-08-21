from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from axiom.core import (
    D,
    Interval,
    LedgerLine,
    Outcome,
    Posterior,
    SupportsPosterior,
    TimeWindow,
    Treatment,
)
from axiom.data import Panel, RoleMap, fit_scaling
from axiom.io import Analysis, ArtifactRegistry, FormatError, load_analysis, save_analysis


def _posterior() -> Posterior:
    rng = np.random.default_rng(0)
    return Posterior(
        {"beta": rng.normal(size=(2, 100)), "alpha": rng.normal(size=(2, 100, 3))},
        coords={"treatment": ["a", "b", "c"]},
        provenance={"seed": 0, "backend": "hand-built"},
    )


def _panel() -> Panel:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "u": np.repeat(["01", "02"], 5),
            "t": np.tile(range(5), 2),
            "y": rng.gamma(2, 1, 10),
            "x": rng.uniform(0, 10, 10),
        }
    )
    roles = RoleMap(
        unit="u",
        time="t",
        outcome=("y", Outcome(name="y_total", dimension=D.outcome)),
        treatments={"x": Treatment(name="x_dose", dimension=D.currency, unit="USD")},
    )
    return Panel(df, roles)


def test_posterior_protocol_and_shapes() -> None:
    p = _posterior()
    assert isinstance(p, SupportsPosterior)
    assert p.names() == {"beta", "alpha"} and p.n_draws() == 200 and p.n_chains == 2
    assert p.flat("alpha").shape == (200, 3)
    s = p.summary("beta", definition="eti", mass=0.5)
    assert isinstance(s.interval, Interval) and s.interval.mass == 0.5
    with pytest.raises(ValueError, match="trailing shape"):
        p.summary("alpha")
    with pytest.raises(KeyError):
        p.draws("nope")
    with pytest.raises(ValueError, match="chain, draw"):
        Posterior({"a": np.zeros(5)})
    with pytest.raises(ValueError, match="expected"):
        Posterior({"a": np.zeros((2, 5)), "b": np.zeros((2, 6))})
    with pytest.raises(ValueError):
        Posterior({})


def test_posterior_npz_roundtrip(tmp_path: Path) -> None:
    p = _posterior()
    f = p.to_npz(tmp_path / "post.npz")
    q = Posterior.from_npz(f)
    assert q == p and q.provenance == {"seed": 0, "backend": "hand-built"}
    assert p.with_provenance(note="x").provenance["note"] == "x"


def test_analysis_roundtrip_equal(tmp_path: Path) -> None:
    panel = _panel()
    a = Analysis(
        specs={
            "roles": panel.roles,
            "scaling": fit_scaling(panel),
            "window": TimeWindow(start=0, stop=5),
        }
    )
    a = a.with_panel(panel).with_posterior(_posterior())
    a = a.with_ledger_line(
        LedgerLine(kind="unit_conversion", statement="USD→USD", detail={"factor": "1"})
    )
    a = a.with_evidence(TimeWindow(start=1, stop=2))
    saved = save_analysis(a, tmp_path / "run.axiom", seed=7)
    assert saved.provenance is not None and saved.provenance.seed == 7
    loaded = load_analysis(tmp_path / "run.axiom")
    assert loaded == saved
    assert loaded.panel is not None and loaded.panel.content_hash() == panel.content_hash()
    assert loaded.spec("window") == TimeWindow(start=0, stop=5)
    assert set(saved.hashes()) >= {"spec:roles", "panel", "ledger:0", "evidence:0"}
    with pytest.raises(KeyError):
        loaded.spec("missing")
    assert loaded.summary()["ledger"] == 1


def test_save_refuses_to_clobber_and_load_checks_format(tmp_path: Path) -> None:
    a = Analysis(specs={"w": TimeWindow(start=0, stop=1)})
    save_analysis(a, tmp_path / "x.axiom")
    with pytest.raises(FileExistsError):
        save_analysis(a, tmp_path / "x.axiom")
    save_analysis(a, tmp_path / "x.axiom", overwrite=True)
    (tmp_path / "notaxiom").mkdir()
    with pytest.raises(FormatError):
        save_analysis(a, tmp_path / "notaxiom", overwrite=True)
    with pytest.raises(FormatError, match="manifest"):
        load_analysis(tmp_path / "notaxiom")
    m = tmp_path / "x.axiom" / "manifest.json"
    m.write_text(m.read_text().replace('"format_version": "1"', '"format_version": "99"'))
    with pytest.raises(FormatError, match="format_version"):
        load_analysis(tmp_path / "x.axiom")


def test_tampered_panel_is_detected(tmp_path: Path) -> None:
    a = Analysis().with_panel(_panel())
    save_analysis(a, tmp_path / "p.axiom")
    csv = tmp_path / "p.axiom" / "panel.csv"
    lines = csv.read_text().splitlines()
    lines[1] = lines[1].replace(lines[1].split(",")[2], "999")
    csv.write_text("\n".join(lines) + "\n")
    with pytest.raises(FormatError, match="hash"):
        load_analysis(tmp_path / "p.axiom")


def test_registry_is_content_addressed(tmp_path: Path) -> None:
    r = ArtifactRegistry(tmp_path / "reg")
    w = TimeWindow(start=0, stop=3)
    h = r.put(w)
    assert r.put(w) == h and len(r) == 1 and h in r
    assert r.get(h) == w
    assert list(r) == [(h, "axiom.core.entities:TimeWindow")]
    with pytest.raises(KeyError):
        r.get("0" * 64)
    (tmp_path / "reg" / f"{h}.json").write_text(TimeWindow(start=0, stop=4).to_json())
    with pytest.raises(ValueError, match="corrupt"):
        r.get(h)
