"""
Tests for single-month dataset and missingness policy: rates not computable, coverage_incomplete.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_gateway import _build_firm_snapshot_canonical
from app.queries.firm_snapshot import build_firm_snapshot_payload


def _single_month_df() -> pd.DataFrame:
    """One row: first available month (no prior -> begin_aum undefined for rates)."""
    return pd.DataFrame({
        "month_end": [pd.Timestamp("2021-06-30")],
        "begin_aum": [np.nan],
        "end_aum": [1100.0],
        "nnb": [50.0],
        "nnf": [10.0],
        "market_pnl": [np.nan],
        "ogr": [np.nan],
        "market_impact_rate": [np.nan],
    })


def test_single_month_payload_rates_not_computable(tmp_path: Path) -> None:
    """Single-month dataset: prev_month_end is None -> rates_not_computable_reason and coverage_incomplete set."""
    with patch("app.queries.firm_snapshot.load_firm_monthly_last_n", return_value=_single_month_df()):
        payload = build_firm_snapshot_payload(months=24, root=tmp_path)

    assert payload.get("rates_not_computable_reason") == "begin_aum undefined for first available month"
    assert payload.get("coverage_incomplete") is True
    assert payload.get("validation_skip_reason") == "SKIP_INCOMPLETE_COVERAGE"
    raw = payload.get("raw") or {}
    # mom_growth should be NaN (no prior)
    mom = raw.get("mom_growth")
    assert mom is None or (isinstance(mom, float) and math.isnan(mom))


def test_single_month_payload_has_kpis_but_rates_na(tmp_path: Path) -> None:
    """Single month still produces kpis; end_aum present, mom/ogr/ytd can be NaN."""
    with patch("app.queries.firm_snapshot.load_firm_monthly_last_n", return_value=_single_month_df()):
        payload = build_firm_snapshot_payload(months=24, root=tmp_path)

    kpis = payload.get("kpis") or []
    assert len(kpis) >= 1
    # mom_growth, ogr, market_impact should show as not computable (display "—" or NaN)
    raw = payload.get("raw") or {}
    assert raw.get("end_aum") == 1100.0 or (isinstance(raw.get("end_aum"), float) and raw.get("end_aum") == 1100.0)


def test_gateway_single_month_snapshot_rates_nan_and_meta() -> None:
    """Gateway: single-month df -> prior None -> mom_pct, ogr, market_impact_rate NaN; meta has rates_not_computable_reason."""
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-06-30")],
        "begin_aum": [np.nan],
        "end_aum": [1100.0],
        "nnb": [50.0],
        "nnf": [10.0],
    })
    current = pd.Timestamp("2021-06-30")
    period_frames = {
        "current_month_end": current,
        "prior_month_end": None,
        "ytd_start_month_end": current,
        "ytd_frame": df.copy(),
        "yoy_frame": pd.DataFrame(),
        "aum_at_year_start": float("nan"),
    }
    meta: dict = {}
    snapshot, meta = _build_firm_snapshot_canonical(df, period_frames, meta)

    assert meta.get("prior_month_missing") is True
    assert meta.get("rates_not_computable_reason") == "begin_aum undefined for first available month"
    assert len(snapshot) == 1
    row = snapshot.iloc[0]
    assert math.isnan(row["mom_pct"])
    assert math.isnan(row["ogr"])
    assert math.isnan(row["market_impact_rate"])
    assert row["end_aum"] == 1100.0
