"""Unit tests for meta.schema, meta.ingest, meta.contribute, meta.store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from _factories import _estimand, _estimand_result

from axiom.core import D, Intervention, Population, Spec, TimeWindow, Unsupported, dimensionless
from axiom.estimands import TransferPlan
from axiom.meta.contribute import record_from_result, record_from_summary, se_from_summary
from axiom.meta.ingest import from_frame, normalize
from axiom.meta.schema import POOLABLE_QUANTITIES, Corpus, StudyRecord, poolable_quantity
from axiom.meta.store import CorpusStore


def _record(**over: object) -> StudyRecord:
    base: dict[str, object] = dict(
        study="s1",
        contributor="c1",
        quantity="elasticity",
        estimate=0.3,
        se=0.1,
        read="experiment",
        family="fertilizer",
        moderators={"season": 1.0},
    )
    base.update(over)
    return StudyRecord(**base)  # type: ignore[arg-type]


def _licensed_plan() -> TransferPlan:
    e = _estimand()
    t = _estimand(
        population=Population(name="all", strata={"soil": {"clay": 0.5, "loam": 0.5}}),
        window=TimeWindow(start=0, stop=12, basis="per_period"),
        intervention=Intervention(doses={"fertilizer": 200.0}, version="liquid"),
    )
    plan = e.transfer_to(t)
    assert plan.licensed
    return plan


def _blocked_plan() -> TransferPlan:
    from axiom.estimands import Level

    plan = _estimand().transfer_to(
        _estimand(level=Level(unit="cluster", interference="within_cluster"))
    )
    assert not plan.licensed
    return plan


# -- schema -----------------------------------------------------------------------------


def test_catalog_and_record_roundtrip() -> None:
    assert poolable_quantity("elasticity").dimensionless
    assert POOLABLE_QUANTITIES["response_ratio"].scale == "log"
    assert not POOLABLE_QUANTITIES["value_ratio"].dimensionless
    with pytest.raises(ValueError, match="not a poolable quantity"):
        poolable_quantity("lift")
    r = _record()
    assert Spec.from_json(r.to_json()) == r
    assert r.is_dimensionless and r.dimension == dimensionless()


def test_record_validation() -> None:
    with pytest.raises(ValueError, match="not a poolable quantity"):
        _record(quantity="lift")
    with pytest.raises(ValueError, match="se must be finite"):
        _record(se=0.0)
    with pytest.raises(ValueError, match="n must be positive"):
        _record(n=0)
    with pytest.raises(ValueError, match="dimensionless"):
        _record(dimension=D.outcome)
    # a dimensioned record is constructible once it declares its scale; ingest gates it
    scaled = _record(dimension=D.outcome, unit_scale="kg")
    assert not scaled.is_dimensionless


def test_corpus_helpers() -> None:
    recs = (
        _record(study="a", contributor="c1", read="model"),
        _record(study="b", contributor="c1", read="experiment"),
        _record(study="c", contributor="c2", read="model"),
        _record(study="d", contributor="c2", read="experiment", family="water"),
        _record(study="e", contributor="c3", read="experiment", family="water"),
    )
    corpus = Corpus(records=recs, name="demo")
    assert len(corpus) == 5 and corpus.families() == ("fertilizer", "water")
    assert corpus.by_family("water").contributors() == ("c2", "c3")
    assert corpus.dual_read_contributors() == ("c1",)
    assert corpus.dual_read_contributors("water") == ()
    y, se = corpus.arrays()
    assert y.shape == (5,) and (se == 0.1).all()
    frame = corpus.to_frame()
    assert list(frame["study"]) == ["a", "b", "c", "d", "e"] and "mod_season" in frame
    assert Spec.from_json(corpus.to_json()) == corpus
    with pytest.raises(ValueError, match="duplicate study ids"):
        Corpus(records=(_record(), _record()))


# -- ingest -----------------------------------------------------------------------------


def test_normalize_admits_dimensionless_records() -> None:
    out = normalize([_record(study="a"), _record(study="b")], name="ok")
    assert isinstance(out, Corpus) and out.name == "ok" and len(out) == 2


def test_normalize_refuses_unit_scale_without_plan_and_accepts_with_one() -> None:
    # two studies report the same shape in different units
    kg = _record(study="kg", quantity="value_ratio", unit_scale="kg/USD")
    lb = _record(study="lb", quantity="value_ratio", unit_scale="lb/USD")
    refused = normalize([kg, lb])
    assert isinstance(refused, Unsupported)
    assert "'kg'" in refused.reason and "licensed_transfer_plan" in refused.missing
    dimensioned = _record(study="dim", dimension=D.outcome, unit_scale="kg")
    refused2 = normalize([dimensioned])
    assert isinstance(refused2, Unsupported) and "'dim'" in refused2.reason

    blocked = normalize([kg, lb], plans={"kg": _blocked_plan(), "lb": _licensed_plan()})
    assert isinstance(blocked, Unsupported) and "blocked" in blocked.reason

    plan = _licensed_plan()
    admitted = normalize([kg, lb, dimensioned], plans={"kg": plan, "lb": plan, "dim": plan})
    assert isinstance(admitted, Corpus) and len(admitted) == 3
    for r in admitted.records:
        assert r.detail["transfer_plan_hash"] == plan.content_hash()
        assert r.detail["transfer_status"] == "downgraded"
        assert "admitted on scale" in r.detail["transfer"]
        assert "stationary_dynamics" in r.detail["transfer_assumptions"]


def test_normalize_refuses_duplicates_and_nonpositive_log_scale() -> None:
    dup = normalize([_record(), _record()])
    assert isinstance(dup, Unsupported) and "duplicate" in dup.reason
    neg = normalize([_record(quantity="response_ratio", estimate=-0.2)])
    assert isinstance(neg, Unsupported) and "log scale" in neg.reason
    ok = normalize([_record(quantity="response_ratio", estimate=1.2)])
    assert isinstance(ok, Corpus)
    empty = normalize([])
    assert isinstance(empty, Unsupported)


def test_from_frame_with_explicit_mapping() -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "who": ["c1", "c2"],
            "quantity": ["elasticity", "elasticity"],
            "est": [0.2, 0.4],
            "se": [0.1, 0.2],
            "read": ["model", "experiment"],
            "family": ["fertilizer", "fertilizer"],
            "size": [100, None],
            "rain": [1.0, 2.0],
        }
    )
    recs = from_frame(
        df,
        columns={"study": "id", "contributor": "who", "estimate": "est", "n": "size"},
        moderators=["rain"],
    )
    assert recs[0].n == 100 and recs[1].n is None
    assert recs[1].moderators == {"rain": 2.0} and recs[1].read == "experiment"
    corpus = normalize(
        df, columns={"study": "id", "contributor": "who", "estimate": "est"}, moderators=["rain"]
    )
    assert isinstance(corpus, Corpus) and len(corpus) == 2
    with pytest.raises(ValueError, match="lacks mapped columns"):
        from_frame(df)
    with pytest.raises(ValueError, match="read must be"):
        from_frame(
            df.assign(read=["x", "y"]),
            columns={"study": "id", "contributor": "who", "estimate": "est"},
        )


# -- contribute -------------------------------------------------------------------------


def test_record_from_result_and_summary() -> None:
    result = _estimand_result()  # dimension = outcome, unit kg, hdi interval
    rec = record_from_result(
        result,
        study="s1",
        contributor="proj",
        read="model",
        family="fertilizer",
        quantity="standardized_contrast",
        moderators={"season": 2.0},
    )
    assert rec.estimand_hash == result.estimand_hash and rec.dimension == D.outcome
    assert rec.unit_scale == "kg" and rec.se == result.summary.sd
    assert rec.detail["se_from"] == "posterior_sd" and rec.source == "lift_at_100"
    assert isinstance(normalize([rec]), Unsupported)  # carries units -> needs a plan
    assert isinstance(normalize([rec], plans={"s1": _licensed_plan()}), Corpus)

    with pytest.raises(ValueError, match="not a poolable quantity"):
        record_from_result(result, study="s", contributor="c", read="model", family="f")

    from axiom.core import Summary, wald

    s = Summary(mean=0.3, median=0.3, sd=0.7, interval=wald(0.3, 0.5, 0.9), n=100)
    se, rule = se_from_summary(s)
    assert se == pytest.approx(0.5, rel=1e-12) and rule.startswith("wald_width")

    hand = record_from_summary(
        study="h",
        contributor="c",
        quantity="elasticity",
        estimate=0.2,
        se=0.05,
        read="experiment",
        family="fertilizer",
        n=40,
        source="table 2",
    )
    assert hand.detail == {"se_from": "reported"} and hand.n == 40


# -- store ------------------------------------------------------------------------------


def test_store_roundtrip(tmp_path: Path) -> None:
    store = CorpusStore(tmp_path / "meta")
    corpus = Corpus(records=(_record(study="a"), _record(study="b")), name="demo")
    digest = store.put_corpus(corpus)
    assert digest == corpus.content_hash() and digest in store
    assert store.get_corpus(digest) == corpus
    assert store.put_corpus(corpus) == digest and len(store) == 3  # idempotent
    assert store.load_corpus("demo") == corpus
    assert store.get_record(store.put_record(_record(study="a"))) == _record(study="a")
    assert len(store.list("Corpus")) == 1 and len(store.list("StudyRecord")) == 2
    later = Corpus(records=(_record(study="c"),), name="demo")
    store.put(later)
    assert store.load_corpus("demo") == later
    with pytest.raises(KeyError, match="no corpus named"):
        store.load_corpus("missing")
    with pytest.raises(TypeError):
        store.get_corpus(corpus.records[0].content_hash())
