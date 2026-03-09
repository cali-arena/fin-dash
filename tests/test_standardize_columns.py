"""
Pytest: deterministic renaming + alias resolution (standardize_columns).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.transform.standardize_columns import standardize_columns

SCHEMA_PATH = PROJECT_ROOT / "schemas" / "canonical_columns.yml"


def test_trim_and_unicode_normalization_maps_to_canonical() -> None:
    """NBSP + zero-width in column name is normalized and maps to net_new_base_fees."""
    weird_col = " net\u00A0new\u200Bbase fees "
    df = pd.DataFrame({
        "Date": ["2021-01-15"],
        weird_col: ["1.0"],
        "product_ticker": ["AGG"],
    })
    df_out, audit = standardize_columns(df, canonical_schema_path=SCHEMA_PATH)

    assert "net_new_base_fees" in df_out.columns
    assert audit["normalized_headers"].get(weird_col) == "net new base fees"


def test_alias_resolution_single_winner_aum() -> None:
    """Single column 'aum' maps to asset_under_management."""
    df = pd.DataFrame({
        "Date": ["2021-01-15"],
        "aum": [1000.0],
        "product_ticker": ["AGG"],
    })
    df_out, audit = standardize_columns(df, canonical_schema_path=SCHEMA_PATH)
    assert "asset_under_management" in df_out.columns
    assert audit["resolved_to_canonical"].get("aum") == "asset_under_management"


def test_alias_resolution_single_winner_asset_under_management() -> None:
    """Single column 'Asset Under Management' maps to asset_under_management."""
    df = pd.DataFrame({
        "Date": ["2021-01-15"],
        "Asset Under Management": [1000.0],
        "product_ticker": ["AGG"],
    })
    df_out, audit = standardize_columns(df, canonical_schema_path=SCHEMA_PATH)
    assert "asset_under_management" in df_out.columns
    assert audit["resolved_to_canonical"].get("Asset Under Management") == "asset_under_management"


def test_collision_hard_fail() -> None:
    """Two columns mapping to same canonical raise ValueError with canonical + both originals."""
    df = pd.DataFrame({
        "Date": ["2021-01-15"],
        "aum": [1000.0],
        "Asset Under Management": [2000.0],
        "product_ticker": ["AGG"],
    })
    with pytest.raises(ValueError) as excinfo:
        standardize_columns(df, canonical_schema_path=SCHEMA_PATH)
    msg = str(excinfo.value)
    assert "asset_under_management" in msg
    assert "aum" in msg
    assert "Asset Under Management" in msg
    assert "Duplicate source columns" in msg or "canonical" in msg


def test_unmatched_columns_remain_and_in_audit() -> None:
    """Unmatched column 'Random Col' remains in output and is listed in audit.unmatched_columns."""
    df = pd.DataFrame({
        "Date": ["2021-01-15"],
        "product_ticker": ["AGG"],
        "Random Col": ["x"],
    })
    df_out, audit = standardize_columns(df, canonical_schema_path=SCHEMA_PATH)

    assert "Random Col" in df_out.columns
    assert "Random Col" in audit["unmatched_columns"]


def test_deterministic_audit_ordering() -> None:
    """Audit list/dict keys are sorted for deterministic output."""
    df = pd.DataFrame({
        "Date": ["2021-01-15"],
        "product_ticker": ["AGG"],
        "Random Col": ["x"],
    })
    _, audit = standardize_columns(df, canonical_schema_path=SCHEMA_PATH)

    assert audit["unmatched_columns"] == sorted(audit["unmatched_columns"])
    assert list(audit["normalized_headers"].keys()) == sorted(audit["normalized_headers"].keys())
    assert list(audit["final_rename_map"].keys()) == sorted(audit["final_rename_map"].keys())
    if audit["resolved_to_canonical"]:
        assert list(audit["resolved_to_canonical"].keys()) == sorted(audit["resolved_to_canonical"].keys())
    if audit["canonical_to_source"]:
        assert list(audit["canonical_to_source"].keys()) == sorted(audit["canonical_to_source"].keys())
