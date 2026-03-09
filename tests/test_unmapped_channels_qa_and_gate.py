"""
Pytest: unmapped channel extraction, persist, and gate. Deterministic; fast.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.persist.qa_store import persist_unmapped_channels
from legacy.legacy_src.qa.unmapped_channels import (
    KEY_COLS,
    STATUS_COL,
    extract_unmapped_channel_keys,
)
from legacy.legacy_src.quality.unmapped_gate import gate_unmapped


def test_extract_unmapped_channel_keys_aggregation_and_sort() -> None:
    """Extract: 3 MAPPED, 2 FALLBACK_BEST same key, 1 FALLBACK_STANDARD other key -> 2 rows, row_count correct, sample_month_end ISO, sorted by row_count desc."""
    # 3 MAPPED (key A,B,C), 2 FALLBACK_BEST (key R1,S1,B1), 1 FALLBACK_STANDARD (key R2,S2,"")
    df = pd.DataFrame({
        "month_end": pd.to_datetime([
            "2021-01-31", "2021-02-28", "2021-03-31",  # MAPPED
            "2021-04-30", "2021-05-31",  # FALLBACK_BEST same key
            "2021-06-30",  # FALLBACK_STANDARD
        ]),
        "product_ticker": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "channel_raw": ["A", "A", "A", "R1", "R1", "R2"],
        "channel_standard": ["B", "B", "B", "S1", "S1", "S2"],
        "channel_best": ["C", "C", "C", "B1", "B1", ""],
        STATUS_COL: [
            "MAPPED", "MAPPED", "MAPPED",
            "FALLBACK_BEST", "FALLBACK_BEST",
            "FALLBACK_STANDARD",
        ],
    })

    out = extract_unmapped_channel_keys(df)

    assert len(out) == 2, "2 distinct unmapped keys"
    assert list(out.columns) == [
        *KEY_COLS,
        "row_count",
        "distinct_months",
        "distinct_tickers",
        "sample_month_end",
        "sample_ticker",
        "sample_status",
    ]

    # First row: key with row_count 2 (FALLBACK_BEST)
    assert out["row_count"].iloc[0] == 2
    assert out["channel_raw"].iloc[0] == "R1"
    assert out["channel_standard"].iloc[0] == "S1"
    assert out["channel_best"].iloc[0] == "B1"
    assert out["sample_status"].iloc[0] == "FALLBACK_BEST"

    # Second row: key with row_count 1 (FALLBACK_STANDARD)
    assert out["row_count"].iloc[1] == 1
    assert out["channel_raw"].iloc[1] == "R2"
    assert out["sample_status"].iloc[1] == "FALLBACK_STANDARD"

    # sample_month_end formatted YYYY-MM-DD
    assert out["sample_month_end"].iloc[0] == "2021-04-30"
    assert out["sample_month_end"].iloc[1] == "2021-06-30"

    # Sorted by row_count desc
    assert list(out["row_count"].values) == [2, 1]


def test_persist_unmapped_channels_tmp_path(tmp_path: Path) -> None:
    """Persist writes CSV + meta under tmp_path/qa; meta includes dataset_version and file_sha256."""
    qa_dir = tmp_path / "qa"
    path_csv = qa_dir / "unmapped_channels.csv"
    path_meta = qa_dir / "unmapped_channels.meta.json"

    df = pd.DataFrame({
        "channel_raw": ["R"],
        "channel_standard": ["S"],
        "channel_best": ["B"],
        "row_count": [1],
        "distinct_months": [1],
        "distinct_tickers": [1],
        "sample_month_end": ["2021-01-31"],
        "sample_ticker": ["T1"],
        "sample_status": ["FALLBACK_BEST"],
    })

    meta = persist_unmapped_channels(
        df,
        dataset_version="v1",
        path_csv=path_csv,
        path_meta=path_meta,
    )

    assert path_csv.exists()
    assert path_meta.exists()
    assert meta["dataset_version"] == "v1"
    assert "file_sha256" in meta
    assert isinstance(meta["file_sha256"], str)
    assert len(meta["file_sha256"]) == 64
    with path_meta.open(encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["dataset_version"] == "v1"
    assert loaded["file_sha256"] == meta["file_sha256"]


def test_gate_unmapped_warn_always_ok() -> None:
    """mode='warn' with high ratio -> ok True, message contains WARNING."""
    ok, message, stats = gate_unmapped(
        total_rows=100,
        unmapped_rows=50,
        unmapped_keys=10,
        mode="warn",
        fail_above_ratio=0.01,
    )
    assert ok is True
    assert "WARNING" in message
    assert stats["mode"] == "warn"
    assert stats["ratio"] == 0.5


def test_gate_unmapped_fail_above_ratio() -> None:
    """mode='fail' with ratio above threshold -> ok False."""
    ok, message, stats = gate_unmapped(
        total_rows=100,
        unmapped_rows=5,
        unmapped_keys=2,
        mode="fail",
        fail_above_ratio=0.01,
    )
    assert ok is False
    assert "Unmapped channel keys" in message
    assert stats["ratio"] == 0.05
    assert stats["fail_above_ratio"] == 0.01
