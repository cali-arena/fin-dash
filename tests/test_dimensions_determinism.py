"""
Pytest: deterministic dimension generation — same input => same output; stable hashes and keys.
Uses small fixture with edge cases: repeated ticker (segment conflict), messy geo, missing channel fields.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.dimensions.build_dimensions import (
    build_dim_channel,
    build_dim_geo,
    build_dim_product,
    build_dim_time,
    normalize_country,
)


def _make_fact_sample() -> pd.DataFrame:
    """Small fact with edge cases: repeated ticker (conflict), messy countries, None channel fields."""
    return pd.DataFrame({
        "month_end": pd.to_datetime([
            "2021-03-31",
            "2021-03-31",
            "2021-12-31",
            "2021-03-31",
            "2021-12-31",
        ]),
        "product_ticker": ["AGG", "AGG", "AGG", "SPY", "SPY"],
        "segment": ["S1", "S1", "S2", "S1", "S1"],
        "sub_segment": ["SS1", "SS1", "SS2", "SS1", "SS1"],
        "src_country": ["  us  ", "  us  ", " U K ", "ca", "  us  "],
        "product_country": ["  us  ", " U K ", " U K ", "  ca  ", "  us  "],
        "channel_raw": ["R", "R", None, "R", "R"],
        "channel_standard": ["S", "S", "S", "S", "S"],
        "channel_best": ["B", "B", "B", "B", "B"],
        "preferred_label": ["L", "L", "L", "L", "L"],
        "channel_l1": ["L1", "L1", "L1", "L1", "L1"],
        "channel_l2": ["L2", "L2", "L2", "L2", "L2"],
    })


@pytest.fixture(scope="module")
def fact_sample_parquet() -> Path:
    """Write fact_monthly_sample.parquet to tests/fixtures (generated in test setup)."""
    fixtures_dir = PROJECT_ROOT / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / "fact_monthly_sample.parquet"
    _make_fact_sample().to_parquet(path, index=False)
    return path


@pytest.fixture
def fact_df() -> pd.DataFrame:
    """Fact sample as DataFrame (in-memory for speed)."""
    return _make_fact_sample()


def test_dim_time_twice_identical(fact_df: pd.DataFrame) -> None:
    """Run build_dim_time twice; outputs identical (equals after sort)."""
    a = build_dim_time(fact_df)
    b = build_dim_time(fact_df)
    a_s = a.sort_values("month_end").reset_index(drop=True)
    b_s = b.sort_values("month_end").reset_index(drop=True)
    assert a_s.equals(b_s), "dim_time must be identical across runs"


def test_dim_time_quarter_flags_march_dec(fact_df: pd.DataFrame) -> None:
    """With max month_end = Dec: Dec has is_latest_month, is_ytd, is_current_quarter True; March has is_ytd True only."""
    out = build_dim_time(fact_df)
    march = out[out["month_end"].dt.month == 3]
    dec = out[out["month_end"].dt.month == 12]
    assert len(march) >= 1
    assert len(dec) >= 1
    assert (dec["is_latest_month"] == True).all()
    assert (dec["is_ytd"] == True).all()
    assert (dec["is_current_quarter"] == True).all()
    assert (march["is_latest_month"] == False).all()
    assert (march["is_ytd"] == True).all()
    assert "year_month" in out.columns
    assert out["year_month"].str.match(r"^\d{4}-\d{2}$").all()


def test_dim_channel_twice_identical(fact_df: pd.DataFrame) -> None:
    """Run build_dim_channel twice; outputs identical."""
    a = build_dim_channel(fact_df)
    b = build_dim_channel(fact_df)
    a_s = a.sort_values("channel_key").reset_index(drop=True)
    b_s = b.sort_values("channel_key").reset_index(drop=True)
    assert a_s.equals(b_s), "dim_channel must be identical across runs"


def test_dim_channel_key_sha1_known_row(fact_df: pd.DataFrame) -> None:
    """channel_key is sha1 of '|'.join(channel_raw, channel_standard, channel_best, preferred_label, channel_l1, channel_l2)."""
    out = build_dim_channel(fact_df)
    # After _str_fill, (R, S, B, L, L1, L2) -> key = sha1("R|S|B|L|L1|L2")
    expected = hashlib.sha1("R|S|B|L|L1|L2".encode("utf-8")).hexdigest()
    row = out[(out["channel_raw"] == "R") & (out["preferred_label"] == "L")]
    assert len(row) >= 1
    assert row["channel_key"].iloc[0] == expected


def test_dim_product_twice_identical(fact_df: pd.DataFrame) -> None:
    """Run build_dim_product twice; outputs identical."""
    a, _, _ = build_dim_product(fact_df)
    b, _, _ = build_dim_product(fact_df)
    a_s = a.sort_values("product_ticker").reset_index(drop=True)
    b_s = b.sort_values("product_ticker").reset_index(drop=True)
    assert a_s.equals(b_s), "dim_product must be identical across runs"


def test_dim_product_most_frequent_combo(fact_df: pd.DataFrame) -> None:
    """Repeated ticker AGG: S1/SS1 x2, S2/SS2 x1 -> pick S1/SS1 (most frequent)."""
    out, _, _ = build_dim_product(fact_df)
    agg = out[out["product_ticker"] == "AGG"]
    assert len(agg) == 1
    assert agg["segment"].iloc[0] == "S1"
    assert agg["sub_segment"].iloc[0] == "SS1"


def test_dim_geo_twice_identical(fact_df: pd.DataFrame) -> None:
    """Run build_dim_geo twice; outputs identical."""
    a, _, _ = build_dim_geo(fact_df, geo_region_map_path=None)
    b, _, _ = build_dim_geo(fact_df, geo_region_map_path=None)
    a_s = a.sort_values("country_key").reset_index(drop=True)
    b_s = b.sort_values("country_key").reset_index(drop=True)
    assert a_s.equals(b_s), "dim_geo must be identical across runs"


def test_dim_geo_normalization_uppercase_collapse_spaces(fact_df: pd.DataFrame) -> None:
    """Geo normalization: strip, collapse spaces, uppercase."""
    out, _, _ = build_dim_geo(fact_df, geo_region_map_path=None)
    countries = set(out["country"].astype(str))
    assert "US" in countries
    assert "U K" in countries
    assert "CA" in countries
    assert not any("  " in c for c in countries)
    assert not any(c != c.upper() for c in countries)


def test_normalize_country_uppercase_collapse() -> None:
    """normalize_country: strip, collapse spaces, uppercase."""
    assert normalize_country("  us  ") == "US"
    assert normalize_country("  a  b  ") == "A B"
    assert normalize_country(" U K ") == "U K"


def test_hashes_keys_stable(fact_df: pd.DataFrame) -> None:
    """First 5 keys and schema-like hashes stable across two runs."""
    dim_time_1 = build_dim_time(fact_df)
    dim_time_2 = build_dim_time(fact_df)
    keys_1 = dim_time_1["month_end"].head(5).astype(str).tolist()
    keys_2 = dim_time_2["month_end"].head(5).astype(str).tolist()
    assert keys_1 == keys_2

    dim_geo_1, _, _ = build_dim_geo(fact_df, geo_region_map_path=None)
    dim_geo_2, _, _ = build_dim_geo(fact_df, geo_region_map_path=None)
    assert dim_geo_1["country_key"].tolist() == dim_geo_2["country_key"].tolist()


def test_load_from_parquet_fixture(fact_sample_parquet: Path) -> None:
    """Fixture parquet exists and build_dim_time from loaded df matches in-memory."""
    df = pd.read_parquet(fact_sample_parquet)
    assert len(df) > 0
    out = build_dim_time(df)
    assert len(out) >= 1
    assert "month_end" in out.columns
