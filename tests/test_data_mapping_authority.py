"""
Pytest: DATA MAPPING authority — required columns, duplicate key hard_fail, dedupe_prefer_non_null, enrichment report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.mapping.channel_enrichment import KEY_COLS, enrich_channels
from legacy.legacy_src.mapping.data_mapping_loader import load_data_mapping


def test_missing_required_column_hard_fails(tmp_path: Path) -> None:
    """Mapping CSV lacking a column that resolves to channel_standard raises ValueError mentioning channel_standard."""
    csv = tmp_path / "mapping.csv"
    # Only Channel and Best of Source (no Standard Channel) -> missing channel_standard
    csv.write_text(
        "Channel,Best of Source\nRIA,RIA\n",
        encoding="utf-8",
    )
    schema_path = PROJECT_ROOT / "schemas" / "data_mapping.schema.yml"
    with pytest.raises(ValueError) as exc_info:
        load_data_mapping(csv, schema_path)
    assert "channel_standard" in str(exc_info.value).lower()


def test_duplicate_composite_key_hard_fails(tmp_path: Path) -> None:
    """Two rows with same (Channel, Standard Channel, Best of Source) and mode hard_fail -> raise with sample keys."""
    csv = tmp_path / "mapping.csv"
    csv.write_text(
        "Channel,Standard Channel,Best of Source\nRIA,RIA,RIA\nRIA,RIA,RIA\n",
        encoding="utf-8",
    )
    schema_path = PROJECT_ROOT / "schemas" / "data_mapping.schema.yml"
    with pytest.raises(ValueError) as exc_info:
        load_data_mapping(csv, schema_path)
    msg = str(exc_info.value)
    assert "duplicate" in msg.lower() or "hard_fail" in msg.lower()
    assert "sample" in msg.lower() or "key" in msg.lower() or "RIA" in msg


def test_dedupe_prefer_non_null_keeps_richer_row(tmp_path: Path) -> None:
    """Two duplicate-key rows, one with preferred_label set; dedupe_prefer_non_null keeps the richer row; report correct."""
    csv = tmp_path / "mapping.csv"
    # Two rows same key: first no preferred_label, second has Preferred Label
    csv.write_text(
        "Channel,Standard Channel,Best of Source,Preferred Label\n"
        "RIA,RIA,RIA,\n"
        "RIA,RIA,RIA,Preferred RIA\n",
        encoding="utf-8",
    )
    schema_path = PROJECT_ROOT / "schemas" / "data_mapping.schema.yml"
    schema = yaml.safe_load(schema_path.read_text())
    schema["uniqueness_policy"]["mode"] = "dedupe_prefer_non_null"
    tmp_schema = tmp_path / "data_mapping.schema.yml"
    tmp_schema.write_text(yaml.dump(schema))

    df, report = load_data_mapping(csv, tmp_schema)
    assert report["duplicates_count"] > 0
    assert report["rows_in"] == 2
    assert report["rows_out"] == 1
    assert len(df) == 1
    # Kept row should have preferred_label set (richer row)
    assert "preferred_label" in df.columns
    val = df["preferred_label"].iloc[0]
    assert pd.notna(val) and str(val).strip() == "Preferred RIA"


def test_enrichment_join_report_one_mapped_one_unmapped() -> None:
    """Fact has two keys; mapping covers one. enrich_channels returns rows_mapped==1, rows_unmapped==1."""
    df_fact = pd.DataFrame({
        "channel_raw": ["RIA", "BD"],
        "channel_standard": ["RIA", "BD"],
        "channel_best": ["RIA", "BD"],
        "value": [1, 2],
    })
    df_mapping = pd.DataFrame({
        "channel_raw": ["RIA"],
        "channel_standard": ["RIA"],
        "channel_best": ["RIA"],
        "channel_l1": ["L1"],
        "channel_l2": ["L2"],
        "preferred_label": ["RIA Label"],
    })
    enriched, report = enrich_channels(df_fact, df_mapping)
    assert report["rows_in_fact"] == 2
    assert report["rows_mapped"] == 1
    assert report["rows_unmapped"] == 1
    assert len(report["top_unmapped_keys_sample"]) == 1
    assert report["top_unmapped_keys_sample"][0] == ("BD", "BD", "BD")
    assert enriched["preferred_label"].iloc[0] == "RIA Label"
    assert pd.isna(enriched["preferred_label"].iloc[1])
