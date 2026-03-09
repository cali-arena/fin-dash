"""
Pytest: canonical column mapping and collisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.ingest.canonicalize import canonicalize_dataframe
from legacy.legacy_src.schemas.canonical_resolver import build_alias_index, load_canonical_columns


def _write_minimal_schema(path: Path, extra_columns: list[dict] | None = None) -> None:
    cols = [
        {
            "canonical_name": "month_end",
            "accepted_source_names": ["Date"],
            "dtype": "datetime",
            "required": True,
            "description": "Month-end date",
        },
        {
            "canonical_name": "asset_under_management",
            "accepted_source_names": [
                "aum",
                "Asset Under Management",
                " Asset Under Management ",
            ],
            "dtype": "float64",
            "required": True,
            "description": "AUM",
        },
        {
            "canonical_name": "product_ticker",
            "accepted_source_names": ["product_ticker"],
            "dtype": "string",
            "required": True,
            "description": "Ticker",
        },
    ]
    if extra_columns:
        cols.extend(extra_columns)
    content = {
        "schema_version": "2026-03-03",
        "description": "Test canonical schema",
        "columns": cols,
    }
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for tests") from e
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def test_alias_matching_asset_under_management(tmp_path: Path) -> None:
    """Alias matching: ' Asset Under Management ' in schema matches 'Asset Under Management' header."""
    schema_path = tmp_path / "canonical_columns.yml"
    _write_minimal_schema(schema_path)

    df = pd.DataFrame(
        {
            "Date": ["2021-01-31"],
            "Asset Under Management": ["1000"],
            "product_ticker": ["AGG"],
        }
    )
    df_canon, report = canonicalize_dataframe(df, canonical_schema_path=schema_path)

    assert "asset_under_management" in df_canon.columns
    # resolved is canonical_name -> original_header
    assert report["resolved"]["asset_under_management"] == "Asset Under Management"


def test_required_enforcement_missing_product_ticker(tmp_path: Path) -> None:
    """Missing required canonical (product_ticker) raises with canonical name in message."""
    schema_path = tmp_path / "canonical_columns.yml"
    _write_minimal_schema(schema_path)

    df = pd.DataFrame(
        {
            "Date": ["2021-01-31"],
            "Asset Under Management": ["1000"],
            # product_ticker missing
        }
    )
    with pytest.raises(ValueError) as excinfo:
        canonicalize_dataframe(df, canonical_schema_path=schema_path)
    msg = str(excinfo.value)
    assert "product_ticker" in msg
    assert "Missing required canonical columns" in msg


def test_collision_detection_in_schema(tmp_path: Path) -> None:
    """Two canonicals sharing alias 'aum' cause build_alias_index to raise ValueError."""
    schema_path = tmp_path / "canonical_columns_collision.yml"
    extra = [
        {
            "canonical_name": "aum_alt",
            "accepted_source_names": ["aum"],
            "dtype": "float64",
            "required": False,
            "description": "Alt AUM",
        }
    ]
    _write_minimal_schema(schema_path, extra_columns=extra)
    schema = load_canonical_columns(schema_path)

    with pytest.raises(ValueError) as excinfo:
        build_alias_index(schema)
    msg = str(excinfo.value)
    assert "Canonical alias collision" in msg
    assert "aum" in msg


def test_duplicate_mapping_in_input_headers(tmp_path: Path) -> None:
    """If df has both 'aum' and 'Asset Under Management', canonicalize_dataframe must fail."""
    schema_path = tmp_path / "canonical_columns.yml"
    _write_minimal_schema(schema_path)

    df = pd.DataFrame(
        {
            "aum": ["1000"],
            "Asset Under Management": ["1000"],
            "Date": ["2021-01-31"],
            "product_ticker": ["AGG"],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        canonicalize_dataframe(df, canonical_schema_path=schema_path)
    msg = str(excinfo.value)
    assert "Duplicate source mappings" in msg
    assert "asset_under_management" in msg


def test_canonicalization_report_contents(tmp_path: Path) -> None:
    """Report includes resolved and unmatched_headers lists."""
    schema_path = tmp_path / "canonical_columns.yml"
    _write_minimal_schema(schema_path)

    df = pd.DataFrame(
        {
            "Date": ["2021-01-31"],
            "Asset Under Management": ["1000"],
            "product_ticker": ["AGG"],
            "Extra Column": ["x"],
        }
    )
    df_canon, report = canonicalize_dataframe(df, canonical_schema_path=schema_path)

    assert "resolved" in report and "unmatched_headers" in report
    # All required canonicals resolved
    assert set(report["resolved"].keys()) >= {"month_end", "asset_under_management", "product_ticker"}
    # Extra column should be in unmatched_headers
    assert "Extra Column" in report["unmatched_headers"]

