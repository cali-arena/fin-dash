"""
Pytest: quality gates and raw persistence outputs. Deterministic; no Streamlit.
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

from legacy.legacy_src.quality.gates import run_quality_gates
from legacy.legacy_src.persist.raw_store import persist_raw_outputs

KEY_FIELDS = ["date", "product_ticker", "channel", "src_country"]
NUMERIC_FIELDS = ["asset_under_management", "net_new_business", "net_new_base_fees"]


def test_gates_pass() -> None:
    """Gates pass: 2 rows, key_fields non-null, numeric floats with no NaNs."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-31"), pd.Timestamp("2021-02-28")],
        "product_ticker": ["AGG", "BND"],
        "channel": ["BD", "RIA"],
        "src_country": ["US", "US"],
        "asset_under_management": [1000.0, 2000.0],
        "net_new_business": [50.0, 60.0],
        "net_new_base_fees": [1.0, 2.0],
    })
    ok, errors, stats = run_quality_gates(
        df,
        key_fields=KEY_FIELDS,
        numeric_fields=NUMERIC_FIELDS,
        min_rows=1,
    )
    assert ok is True
    assert len(errors) == 0
    assert stats["row_count"] == 2
    assert stats["nan_ratio_by_col"]["asset_under_management"] == 0.0


def test_gates_fail_missing_key() -> None:
    """Gates fail when key field has empty string or NaN; errors mention product_ticker."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-31"), pd.Timestamp("2021-02-28")],
        "product_ticker": ["AGG", ""],
        "channel": ["BD", "RIA"],
        "src_country": ["US", "US"],
        "asset_under_management": [1000.0, 2000.0],
        "net_new_business": [50.0, 60.0],
        "net_new_base_fees": [1.0, 2.0],
    })
    ok, errors, _ = run_quality_gates(
        df,
        key_fields=KEY_FIELDS,
        numeric_fields=NUMERIC_FIELDS,
    )
    assert ok is False
    assert any("product_ticker" in e for e in errors)

    df_nan = df.copy()
    df_nan.loc[1, "product_ticker"] = float("nan")
    ok2, errors2, _ = run_quality_gates(
        df_nan,
        key_fields=KEY_FIELDS,
        numeric_fields=NUMERIC_FIELDS,
    )
    assert ok2 is False
    assert any("product_ticker" in e for e in errors2)


def test_gates_fail_nan_explosion() -> None:
    """Gates fail when numeric column NaN ratio exceeds threshold; errors mention column and ratio."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-31")] * 10,
        "product_ticker": ["X"] * 10,
        "channel": ["Y"] * 10,
        "src_country": ["US"] * 10,
        "asset_under_management": [100.0] * 5 + [float("nan")] * 5,
        "net_new_business": [10.0] * 10,
        "net_new_base_fees": [1.0] * 10,
    })
    ok, errors, stats = run_quality_gates(
        df,
        key_fields=KEY_FIELDS,
        numeric_fields=NUMERIC_FIELDS,
        max_nan_ratio_by_col=0.01,
        max_total_nan_ratio=0.001,
    )
    assert ok is False
    assert any("asset_under_management" in e for e in errors)
    assert any("nan_ratio" in e for e in errors)
    assert stats["nan_ratio_by_col"]["asset_under_management"] == 0.5


def test_persistence_writes_artifacts(tmp_path: Path) -> None:
    """Persist writes data_raw (parquet or CSV), data_raw_rejects, ingest_report.json, data_raw.meta.json."""
    raw_dir = tmp_path / "raw"
    df_clean = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-31")],
        "product_ticker": ["AGG"],
        "channel": ["BD"],
        "src_country": ["US"],
        "asset_under_management": [1000.0],
        "net_new_business": [50.0],
        "net_new_base_fees": [1.0],
    })
    df_rejects = pd.DataFrame({
        "date": ["bad-date"],
        "amount": ["x"],
        "_reject_reason": ["bad_date"],
    })
    ingest_report = {
        "version": {"dataset_version": "test-dv-123", "pipeline_version": "p1", "source_sha256": "abc"},
        "load_report": {"sheet_name": "DATA RAW", "rows_read": 2},
        "type_enforcement_stats": {"rows_clean": 1, "rows_rejected": 1},
        "rejects_summary": {"total": 1, "by_reason": {"bad_date": 1}},
        "timestamp": "2021-01-01T00:00:00Z",
    }
    version_manifest = {
        "dataset_version": "test-dv-123",
        "pipeline_version": "p1",
        "source_sha256": "abc",
        "created_at": "2021-01-01T00:00:00Z",
    }

    persist_raw_outputs(
        df_clean,
        df_rejects,
        ingest_report,
        version_manifest,
        raw_dir=raw_dir,
    )

    # data_raw: parquet or CSV fallback
    assert (raw_dir / "data_raw.parquet").exists() or (raw_dir / "data_raw.csv").exists()
    # data_raw_rejects: parquet or CSV fallback
    assert (raw_dir / "data_raw_rejects.parquet").exists() or (raw_dir / "data_raw_rejects.csv").exists()
    assert (raw_dir / "ingest_report.json").exists()
    assert (raw_dir / "data_raw.meta.json").exists()

    ingest = json.loads((raw_dir / "ingest_report.json").read_text(encoding="utf-8"))
    assert "dataset_version" in ingest.get("version", {}) or "dataset_version" in ingest
    # Rejects count can be in version/load_report/type_enforcement_stats/rejects_summary
    assert ingest.get("rejects_summary", {}).get("total") == 1
    assert ingest.get("version", {}).get("dataset_version") == "test-dv-123"

    meta = json.loads((raw_dir / "data_raw.meta.json").read_text(encoding="utf-8"))
    assert meta.get("dataset_version") == "test-dv-123"
    assert "row_counts" in meta
