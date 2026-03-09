from __future__ import annotations

import pandas as pd

from app.analytics.commentary_engine import build_executive_insight_sections


def test_build_executive_insight_sections_from_metrics() -> None:
    snapshot_df = pd.DataFrame(
        [
            {
                "month_end": "2026-01-31",
                "end_aum": 17_870_000_000.0,
                "nnb": 120_000_000.0,
                "market_impact": 430_000_000.0,
                "mom_growth": 0.012,
                "ytd_growth": 0.044,
            }
        ]
    )
    ranked_channels = pd.DataFrame(
        [
            {"channel": "Direct", "AUM": 8_041_500_000.0, "NNB": 10_000_000.0},
            {"channel": "Wholesale", "AUM": 5_000_000_000.0, "NNB": -5_000_000.0},
            {"channel": "Institutional", "AUM": 4_828_500_000.0, "NNB": 15_000_000.0},
        ]
    )
    ranked_tickers = pd.DataFrame(
        [
            {"ticker": "AGG", "AUM": 6_000_000_000.0, "NNB": 20_000_000.0},
            {"ticker": "ETF1", "AUM": 3_000_000_000.0, "NNB": -1_000_000.0},
            {"ticker": "ETF2", "AUM": 1_000_000_000.0, "NNB": -2_000_000.0},
        ]
    )

    sections = build_executive_insight_sections(
        snapshot_df=snapshot_df,
        ranked_channels=ranked_channels,
        ranked_tickers=ranked_tickers,
    )

    assert [s.title for s in sections] == [
        "Portfolio Performance",
        "Distribution Dynamics",
        "Product Dynamics",
        "Movers",
    ]
    assert all(len(s.sentences) == 2 for s in sections)
    assert "market-driven" in sections[0].sentences[1]
    assert "Direct" in sections[1].sentences[0]
    assert "2 tickers show declining flows" in sections[2].sentences[1]
    assert "Top mover is AGG" in sections[3].sentences[0]


def test_build_executive_insight_sections_handles_missing_data() -> None:
    sections = build_executive_insight_sections(
        snapshot_df=pd.DataFrame(),
        ranked_channels=pd.DataFrame(),
        ranked_tickers=pd.DataFrame(),
    )
    assert all(len(s.sentences) == 2 for s in sections)
    assert "unavailable" in sections[0].sentences[0].lower()
