"""Tests for app.agg_store: manifest, dataset_version, firm_monthly month range."""
import pytest

from app.agg_store import (
    get_dataset_version,
    get_firm_monthly_month_range,
    get_table_path,
)


def test_get_dataset_version_from_manifest() -> None:
    """dataset_version comes from manifest for cache keys."""
    assert get_dataset_version({"dataset_version": "abc123"}) == "abc123"
    assert get_dataset_version({"dataset_version": "  x  "}) == "x"
    assert get_dataset_version({}) == "unknown"


def test_get_firm_monthly_month_range_from_manifest() -> None:
    """Min/max month_end for firm_monthly from manifest table entry (no table read)."""
    manifest = {
        "tables": [
            {"name": "firm_monthly", "min_month_end": "2024-01-01", "max_month_end": "2024-12-01"},
        ]
    }
    assert get_firm_monthly_month_range(manifest) == ("2024-01-01", "2024-12-01")
    manifest["tables"][0]["min_month_end"] = None
    manifest["tables"][0]["min_month"] = "2024-01-01"
    assert get_firm_monthly_month_range(manifest) == ("2024-01-01", "2024-12-01")
    assert get_firm_monthly_month_range({"tables": []}) == (None, None)
