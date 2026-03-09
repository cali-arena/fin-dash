"""
Pytest: apply_channel_map behavior — MAPPED and FALLBACK_* statuses, rowcount, non-null outputs, stats.
Deterministic; no I/O.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.curate.apply_channel_map import (
    KEY_COLS,
    OUT_COLS,
    STATUS_COL,
    STATUS_FALLBACK_BEST,
    STATUS_FALLBACK_RAW,
    STATUS_FALLBACK_STANDARD,
    STATUS_MAPPED,
    UNMAPPED_PLACEHOLDER,
    apply_channel_map,
)


def _fact_df(
    channel_raw: str,
    channel_standard: str = "",
    channel_best: str = "",
) -> pd.DataFrame:
    """Minimal fact with KEY_COLS only."""
    return pd.DataFrame({
        "channel_raw": [channel_raw],
        "channel_standard": [channel_standard],
        "channel_best": [channel_best],
    })


def _channel_map_df(
    channel_raw: str,
    channel_standard: str,
    channel_best: str,
    channel_l1: str,
    channel_l2: str,
    preferred_label: str,
) -> pd.DataFrame:
    """Minimal channel_map with KEY_COLS + OUT_COLS."""
    return pd.DataFrame({
        "channel_raw": [channel_raw],
        "channel_standard": [channel_standard],
        "channel_best": [channel_best],
        "channel_l1": [channel_l1],
        "channel_l2": [channel_l2],
        "preferred_label": [preferred_label],
    })


def test_apply_channel_map_mapped() -> None:
    """Key (A,B,C) in map with preferred_label=X, channel_l1=L1, channel_l2=L2 -> status MAPPED, outputs from map."""
    df_fact = _fact_df(channel_raw="A", channel_standard="B", channel_best="C")
    df_map = _channel_map_df(
        channel_raw="A", channel_standard="B", channel_best="C",
        channel_l1="L1", channel_l2="L2", preferred_label="X",
    )
    result, stats = apply_channel_map(df_fact, df_map)

    assert len(result) == len(df_fact), "output rowcount equals input rowcount"
    assert result[STATUS_COL].iloc[0] == STATUS_MAPPED
    assert result["preferred_label"].iloc[0] == "X"
    assert result["channel_l1"].iloc[0] == "L1"
    assert result["channel_l2"].iloc[0] == "L2"

    for c in OUT_COLS + [STATUS_COL]:
        assert result[c].iloc[0] is not None and not pd.isna(result[c].iloc[0])
        assert isinstance(result[c].iloc[0], str)

    assert stats["rows_in"] == 1
    assert stats["mapped_count"] == 1
    assert stats["fallback_best_count"] == 0
    assert stats["fallback_standard_count"] == 0
    assert stats["fallback_raw_count"] == 0


def test_apply_channel_map_fallback_best() -> None:
    """Fact key not in map; channel_best=BestVal -> preferred_label=BestVal, l1/l2=UNMAPPED, status FALLBACK_BEST."""
    df_fact = _fact_df(channel_raw="R", channel_standard="S", channel_best="BestVal")
    # Map has different key so no match
    df_map = _channel_map_df(
        channel_raw="A", channel_standard="B", channel_best="C",
        channel_l1="L1", channel_l2="L2", preferred_label="X",
    )
    result, stats = apply_channel_map(df_fact, df_map)

    assert len(result) == len(df_fact)
    assert result[STATUS_COL].iloc[0] == STATUS_FALLBACK_BEST
    assert result["preferred_label"].iloc[0] == "BestVal"
    assert result["channel_l1"].iloc[0] == UNMAPPED_PLACEHOLDER
    assert result["channel_l2"].iloc[0] == UNMAPPED_PLACEHOLDER

    for c in OUT_COLS + [STATUS_COL]:
        assert result[c].iloc[0] is not None and not pd.isna(result[c].iloc[0])
        assert isinstance(result[c].iloc[0], str)

    assert stats["rows_in"] == 1
    assert stats["mapped_count"] == 0
    assert stats["fallback_best_count"] == 1
    assert stats["fallback_standard_count"] == 0
    assert stats["fallback_raw_count"] == 0


def test_apply_channel_map_fallback_standard() -> None:
    """channel_best empty, channel_standard=StdVal -> preferred_label=StdVal, status FALLBACK_STANDARD."""
    df_fact = _fact_df(channel_raw="R", channel_standard="StdVal", channel_best="")
    df_map = _channel_map_df(
        channel_raw="A", channel_standard="B", channel_best="C",
        channel_l1="L1", channel_l2="L2", preferred_label="X",
    )
    result, stats = apply_channel_map(df_fact, df_map)

    assert len(result) == len(df_fact)
    assert result[STATUS_COL].iloc[0] == STATUS_FALLBACK_STANDARD
    assert result["preferred_label"].iloc[0] == "StdVal"
    assert result["channel_l1"].iloc[0] == UNMAPPED_PLACEHOLDER
    assert result["channel_l2"].iloc[0] == UNMAPPED_PLACEHOLDER

    for c in OUT_COLS + [STATUS_COL]:
        assert result[c].iloc[0] is not None and not pd.isna(result[c].iloc[0])
        assert isinstance(result[c].iloc[0], str)

    assert stats["rows_in"] == 1
    assert stats["mapped_count"] == 0
    assert stats["fallback_best_count"] == 0
    assert stats["fallback_standard_count"] == 1
    assert stats["fallback_raw_count"] == 0


def test_apply_channel_map_fallback_raw() -> None:
    """channel_best and channel_standard empty, channel_raw=RawVal -> preferred_label=RawVal, status FALLBACK_RAW."""
    df_fact = _fact_df(channel_raw="RawVal", channel_standard="", channel_best="")
    df_map = _channel_map_df(
        channel_raw="A", channel_standard="B", channel_best="C",
        channel_l1="L1", channel_l2="L2", preferred_label="X",
    )
    result, stats = apply_channel_map(df_fact, df_map)

    assert len(result) == len(df_fact)
    assert result[STATUS_COL].iloc[0] == STATUS_FALLBACK_RAW
    assert result["preferred_label"].iloc[0] == "RawVal"
    assert result["channel_l1"].iloc[0] == UNMAPPED_PLACEHOLDER
    assert result["channel_l2"].iloc[0] == UNMAPPED_PLACEHOLDER

    for c in OUT_COLS + [STATUS_COL]:
        assert result[c].iloc[0] is not None and not pd.isna(result[c].iloc[0])
        assert isinstance(result[c].iloc[0], str)

    assert stats["rows_in"] == 1
    assert stats["mapped_count"] == 0
    assert stats["fallback_best_count"] == 0
    assert stats["fallback_standard_count"] == 0
    assert stats["fallback_raw_count"] == 1


def test_apply_channel_map_mixed_rowcount_and_stats() -> None:
    """Multiple rows: one MAPPED, one FALLBACK_BEST, one FALLBACK_RAW; rowcount preserved, stats sum."""
    df_fact = pd.DataFrame({
        "channel_raw": ["A", "R2", "R3"],
        "channel_standard": ["B", "S2", ""],
        "channel_best": ["C", "Best2", ""],
    })
    df_map = _channel_map_df(
        channel_raw="A", channel_standard="B", channel_best="C",
        channel_l1="L1", channel_l2="L2", preferred_label="X",
    )
    result, stats = apply_channel_map(df_fact, df_map)

    assert len(result) == 3 == len(df_fact)
    assert stats["rows_in"] == 3
    assert stats["mapped_count"] == 1
    assert stats["fallback_best_count"] == 1
    assert stats["fallback_standard_count"] == 0
    assert stats["fallback_raw_count"] == 1
    assert (
        stats["mapped_count"]
        + stats["fallback_best_count"]
        + stats["fallback_standard_count"]
        + stats["fallback_raw_count"]
        == 3
    )

    assert result[STATUS_COL].iloc[0] == STATUS_MAPPED
    assert result[STATUS_COL].iloc[1] == STATUS_FALLBACK_BEST
    assert result[STATUS_COL].iloc[2] == STATUS_FALLBACK_RAW
    for c in OUT_COLS + [STATUS_COL]:
        assert result[c].notna().all()
        assert result[c].astype(str).str.len().ge(0).all()
