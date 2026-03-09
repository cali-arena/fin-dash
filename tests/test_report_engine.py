"""
Unit tests for deterministic report engine. Synthetic data only; no Streamlit runtime.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.reporting.report_pack import (
    ReportPack,
    FIRM_SNAPSHOT_COLUMNS,
    RANK_COLUMNS,
    TIME_SERIES_COLUMNS,
    ANOMALIES_COLUMNS,
)
from app.reporting.report_engine import (
    SectionOutput,
    render_overview,
    render_channel_commentary,
    render_product_commentary,
    render_geo_commentary,
    render_anomalies,
    render_recommendations,
    OGR_STRONG,
)

# Canonical anomalies columns (gateway schema)
ANOMALIES_CANONICAL_COLUMNS = [
    "level", "entity", "metric", "value_current", "baseline", "zscore",
    "rule_id", "reason", "severity", "month_end",
]


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _meta() -> dict:
    return {"row_counts": {}, "top_n": 5, "dataset_version": "test", "filter_hash": "test"}


def make_pack_zero_data() -> ReportPack:
    """ReportPack with empty dataframes (correct columns)."""
    return ReportPack(
        dataset_version="test",
        filter_hash="zero",
        firm_snapshot=_empty_df(FIRM_SNAPSHOT_COLUMNS),
        channel_rank=_empty_df(RANK_COLUMNS),
        ticker_rank=_empty_df(RANK_COLUMNS),
        etf_rank=_empty_df(RANK_COLUMNS),
        geo_rank=_empty_df(RANK_COLUMNS),
        time_series=_empty_df(TIME_SERIES_COLUMNS),
        anomalies=_empty_df(ANOMALIES_COLUMNS),
        meta=_meta(),
    )


def make_pack_single_month() -> ReportPack:
    """ReportPack with 1 month in time_series and 1-row firm_snapshot; ranks may be empty."""
    me = pd.Timestamp("2024-06-30")
    snap = pd.DataFrame([{
        "month_end": me,
        "begin_aum": 1000.0,
        "end_aum": 1050.0,
        "nnb": 20.0,
        "nnf": 1.0,
        "mom_pct": float("nan"),
        "ytd_pct": 0.05,
        "yoy_pct": float("nan"),
        "ogr": 0.02,
        "market_impact_abs": 30.0,
        "market_impact_rate": 0.03,
        "fee_yield": 0.001,
    }])
    ts = pd.DataFrame([{
        "month_end": me,
        "end_aum": 1050.0,
        "nnb": 20.0,
        "nnf": 1.0,
        "mom_pct": float("nan"),
        "ytd_pct": 0.05,
    }])
    return ReportPack(
        dataset_version="test",
        filter_hash="single",
        firm_snapshot=snap.reindex(columns=[c for c in FIRM_SNAPSHOT_COLUMNS if c in snap.columns], copy=False),
        channel_rank=_empty_df(RANK_COLUMNS),
        ticker_rank=_empty_df(RANK_COLUMNS),
        etf_rank=_empty_df(RANK_COLUMNS),
        geo_rank=_empty_df(RANK_COLUMNS),
        time_series=ts,
        anomalies=_empty_df(ANOMALIES_COLUMNS),
        meta=_meta(),
    )


def make_pack_negative_market_month() -> ReportPack:
    """ReportPack where latest month market_impact_rate < 0 and ogr > 0 (strong flows, headwind)."""
    me = pd.Timestamp("2024-07-31")
    snap = pd.DataFrame([{
        "month_end": me,
        "begin_aum": 1100.0,
        "end_aum": 1120.0,
        "nnb": 50.0,
        "nnf": 0.5,
        "mom_pct": 0.02,
        "ytd_pct": 0.12,
        "yoy_pct": 0.08,
        "ogr": OGR_STRONG + 0.01,
        "market_impact_abs": -15.0,
        "market_impact_rate": -0.014,
        "fee_yield": 0.001,
    }])
    ts = pd.DataFrame([{
        "month_end": me,
        "end_aum": 1120.0,
        "nnb": 50.0,
        "nnf": 0.5,
        "mom_pct": 0.02,
        "ytd_pct": 0.12,
    }])
    return ReportPack(
        dataset_version="test",
        filter_hash="neg_mkt",
        firm_snapshot=snap.reindex(columns=[c for c in FIRM_SNAPSHOT_COLUMNS if c in snap.columns], copy=False),
        channel_rank=_empty_df(RANK_COLUMNS),
        ticker_rank=_empty_df(RANK_COLUMNS),
        etf_rank=_empty_df(RANK_COLUMNS),
        geo_rank=_empty_df(RANK_COLUMNS),
        time_series=ts,
        anomalies=_empty_df(ANOMALIES_COLUMNS),
        meta=_meta(),
    )


def make_pack_extreme_outliers() -> ReportPack:
    """ReportPack with anomalies (high severity, zscores) and a big mix shift in channel rank."""
    me = pd.Timestamp("2024-08-31")
    snap = pd.DataFrame([{
        "month_end": me,
        "begin_aum": 1200.0,
        "end_aum": 1250.0,
        "nnb": 40.0,
        "nnf": 1.0,
        "mom_pct": 0.01,
        "ytd_pct": 0.10,
        "yoy_pct": 0.05,
        "ogr": 0.03,
        "market_impact_abs": 10.0,
        "market_impact_rate": 0.008,
        "fee_yield": 0.001,
    }])
    ts = pd.DataFrame([{
        "month_end": me,
        "end_aum": 1250.0,
        "nnb": 40.0,
        "nnf": 1.0,
        "mom_pct": 0.01,
        "ytd_pct": 0.10,
    }])
    anomalies = pd.DataFrame([
        {
            "level": "ticker",
            "entity": "TICK_A",
            "metric": "NNB",
            "value_current": 500.0,
            "baseline": 50.0,
            "zscore": 3.5,
            "rule_id": "dim_zscore_cross",
            "reason": "NNB |z|=3.50 (current=500.00, mean=50.00)",
            "severity": "high",
            "month_end": me,
        },
        {
            "level": "firm",
            "entity": "FIRM",
            "metric": "OGR",
            "value_current": 0.05,
            "baseline": 0.02,
            "zscore": 2.2,
            "rule_id": "firm_zscore_12m",
            "reason": "OGR |z|=2.20 (current=0.0500, baseline=0.0200)",
            "severity": "med",
            "month_end": me,
        },
    ])
    channel_rank = pd.DataFrame([
        {"dim_value": "Chan_A", "aum_end": 800.0, "nnb": 100.0, "aum_share_delta": 0.05, "nnb_share_delta": 0.08, "bucket": "top"},
        {"dim_value": "Chan_B", "aum_end": 200.0, "nnb": -20.0, "aum_share_delta": -0.03, "nnb_share_delta": -0.02, "bucket": "bottom"},
        {"dim_value": "Chan_C", "aum_end": 250.0, "nnb": 10.0, "aum_share_delta": 0.015, "nnb_share_delta": 0.01, "bucket": "top"},
    ])
    return ReportPack(
        dataset_version="test",
        filter_hash="outliers",
        firm_snapshot=snap.reindex(columns=[c for c in FIRM_SNAPSHOT_COLUMNS if c in snap.columns], copy=False),
        channel_rank=channel_rank,
        ticker_rank=_empty_df(RANK_COLUMNS),
        etf_rank=_empty_df(RANK_COLUMNS),
        geo_rank=_empty_df(RANK_COLUMNS),
        time_series=ts,
        anomalies=anomalies.reindex(columns=[c for c in ANOMALIES_CANONICAL_COLUMNS if c in anomalies.columns], copy=False),
        meta=_meta(),
    )


# --- Zero data -----------------------------------------------------------------


def test_zero_data_overview_returns_section_output() -> None:
    pack = make_pack_zero_data()
    out = render_overview(pack)
    assert isinstance(out, SectionOutput)
    assert hasattr(out, "bullets") and hasattr(out, "table_title") and hasattr(out, "table") and hasattr(out, "meta")


def test_zero_data_all_renderers_return_section_output() -> None:
    pack = make_pack_zero_data()
    for renderer in [render_overview, render_channel_commentary, render_product_commentary, render_geo_commentary, render_anomalies, render_recommendations]:
        out = renderer(pack)
        assert isinstance(out, SectionOutput)
        assert isinstance(out.bullets, list)
        assert isinstance(out.table_title, str)
        assert isinstance(out.table, pd.DataFrame)
        assert isinstance(out.meta, dict)


def test_zero_data_bullets_non_empty_no_crash() -> None:
    pack = make_pack_zero_data()
    for renderer in [render_overview, render_channel_commentary, render_product_commentary, render_geo_commentary, render_anomalies, render_recommendations]:
        out = renderer(pack)
        assert len(out.bullets) >= 1
        for b in out.bullets:
            assert isinstance(b, str) and len(b) > 0


def test_zero_data_table_may_be_empty_title_exists() -> None:
    pack = make_pack_zero_data()
    for renderer in [render_overview, render_channel_commentary, render_product_commentary, render_geo_commentary, render_anomalies, render_recommendations]:
        out = renderer(pack)
        assert out.table_title != ""
        assert out.table is not None


# --- Single month -----------------------------------------------------------------


def test_single_month_overview_bullets_render() -> None:
    pack = make_pack_single_month()
    out = render_overview(pack)
    assert isinstance(out, SectionOutput)
    assert len(out.bullets) >= 2
    assert any("AUM" in b or "1050" in b or "1,050" in b for b in out.bullets)
    assert any("MoM" in b or "—" in b or "NNB" in b for b in out.bullets)


def test_single_month_no_division_by_zero() -> None:
    pack = make_pack_single_month()
    for renderer in [render_overview, render_channel_commentary, render_product_commentary, render_geo_commentary, render_anomalies, render_recommendations]:
        out = renderer(pack)
        assert out.bullets is not None
        for b in out.bullets:
            assert "inf" not in b.lower() and "nan" not in b.lower()


# --- Negative market month -----------------------------------------------------------------


def test_negative_market_month_overview_includes_headwind_bullet() -> None:
    pack = make_pack_negative_market_month()
    out = render_overview(pack)
    assert any("markets headwind" in b or "Flows strong" in b for b in out.bullets), f"Expected 'markets headwind' or 'Flows strong' in bullets: {out.bullets}"


def test_negative_market_month_determinism_exact_bullets() -> None:
    """Exact bullet string match for negative market month to prevent template drift."""
    pack = make_pack_negative_market_month()
    out = render_overview(pack)
    headwind_bullets = [b for b in out.bullets if "headwind" in b or ("Flows strong" in b and "markets" in b)]
    assert len(headwind_bullets) >= 1
    exact_expected = "Flows strong, markets headwind; growth driven by net new business."
    assert exact_expected in out.bullets, f"Expected exact bullet '{exact_expected}' in {out.bullets}"


# --- Extreme outliers -----------------------------------------------------------------


def test_extreme_outliers_anomalies_bullets_mention_z_and_entity() -> None:
    pack = make_pack_extreme_outliers()
    out = render_anomalies(pack)
    assert len(out.bullets) >= 1
    z_related = [b for b in out.bullets if "z=" in b or "zscore" in b.lower()]
    assert len(z_related) >= 1, f"Expected at least one bullet with 'z=' or 'zscore': {out.bullets}"
    entity_related = [b for b in out.bullets if "TICK_A" in b or "FIRM" in b or "entity" in b]
    assert len(entity_related) >= 1, f"Expected at least one bullet with entity name: {out.bullets}"


def test_extreme_outliers_recommendations_include_triggered_bullet() -> None:
    pack = make_pack_extreme_outliers()
    out = render_recommendations(pack)
    triggered = [
        b for b in out.bullets
        if "Investigate" in b or "Allocate" in b or "sales focus" in b or "mix shift" in b or "anomal" in b.lower()
    ]
    assert len(triggered) >= 1, f"Expected at least one recommendation triggered by anomaly or mix shift: {out.bullets}"
