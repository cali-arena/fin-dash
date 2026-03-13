from __future__ import annotations

import pandas as pd

from app.pages.visualisations import _resample_to_quarter


def test_resample_to_quarter_returns_empty_schema_when_missing_required_columns() -> None:
    df = pd.DataFrame({"month_end": ["2024-01-31"], "end_aum": [100.0]})
    out = _resample_to_quarter(df)
    assert out.empty
    assert list(out.columns) == [
        "quarter",
        "month_end",
        "begin_aum",
        "end_aum",
        "nnb",
        "market_pnl",
        "ogr",
        "market_impact_rate",
    ]


def test_resample_to_quarter_one_row_derives_market_pnl_and_rates() -> None:
    df = pd.DataFrame(
        {
            "month_end": ["2024-01-31"],
            "begin_aum": [100.0],
            "end_aum": [110.0],
            "nnb": [5.0],
        }
    )
    out = _resample_to_quarter(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["begin_aum"] == 100.0
    assert row["end_aum"] == 110.0
    assert row["nnb"] == 5.0
    assert row["market_pnl"] == 5.0
    assert pd.notna(row["ogr"])
    assert pd.notna(row["market_impact_rate"])


def test_resample_to_quarter_uses_first_begin_last_end_and_sums_flows() -> None:
    df = pd.DataFrame(
        {
            "month_end": ["2024-01-31", "2024-02-29", "2024-03-31"],
            "begin_aum": [100.0, 105.0, 107.0],
            "end_aum": [105.0, 107.0, 110.0],
            "nnb": [3.0, 1.0, 1.0],
            "market_pnl": [2.0, 1.0, 2.0],
        }
    )
    out = _resample_to_quarter(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["begin_aum"] == 100.0
    assert row["end_aum"] == 110.0
    assert row["nnb"] == 5.0
    assert row["market_pnl"] == 5.0


def test_resample_to_quarter_coerces_non_numeric_inputs_safely() -> None:
    df = pd.DataFrame(
        {
            "month_end": ["2024-01-31", "2024-02-29"],
            "begin_aum": ["100", "bad"],
            "end_aum": ["110", "115"],
            "nnb": ["5", "x"],
            "market_pnl": ["3", None],
        }
    )
    out = _resample_to_quarter(df)
    assert len(out) == 1
    row = out.iloc[0]
    # first valid begin, last valid end, numeric-safe sums
    assert row["begin_aum"] == 100.0
    assert row["end_aum"] == 115.0
    assert row["nnb"] == 5.0
    assert row["market_pnl"] == 3.0
