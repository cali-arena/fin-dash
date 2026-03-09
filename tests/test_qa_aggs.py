"""
Tests for pipelines.agg.qa_aggs: duplicate keys failure, missing month_end failure.
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

from legacy.legacy_pipelines.agg.qa_aggs import (
    FAIL_PREFIX,
    QA_DIR,
    AggQAError,
    validate_agg_qa,
)
from legacy.legacy_pipelines.contracts.agg_policy_contract import load_and_validate_agg_policy

MINIMAL_POLICY_YAML = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum", "nnb"]
    rates: []
  dims:
    channel_l1: "channel_l1"
    product_ticker: "product_ticker"
    src_country_canonical: "src_country_canonical"
    segment: "segment"
  grains:
    firm_monthly: []
    channel_monthly: [["channel_l1"]]
    ticker_monthly: ["product_ticker"]
    geo_monthly: ["src_country_canonical"]
    segment_monthly: [["segment"]]
  null_handling:
    strategy: "UNKNOWN"
    unknown_label: "UNKNOWN"
  rollup:
    additive_method: "sum"
    rates_method: "store_numerators"
    weights: {}
"""


def test_qa_duplicate_keys_writes_fail_context_and_raises(tmp_path: Path) -> None:
    """When agg table has duplicate keys on month_end + grain_dims, QA writes qa/agg_qa_fail_<name>.json and raises AggQAError."""
    (tmp_path / "p.yml").write_text(MINIMAL_POLICY_YAML, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")

    # Duplicate (month_end, channel_l1) in two rows
    df = pd.DataFrame({
        "month_end": [
            pd.Timestamp("2024-01-31"),
            pd.Timestamp("2024-01-31"),
            pd.Timestamp("2024-02-29"),
        ],
        "channel_l1": ["A", "A", "B"],
        "begin_aum": [100.0, 200.0, 300.0],
        "nnb": [10.0, 20.0, 30.0],
    })

    with pytest.raises(AggQAError) as exc_info:
        validate_agg_qa(
            df,
            "month_end",
            ["channel_l1"],
            ["begin_aum", "nnb"],
            "channel_monthly",
            policy,
            tmp_path,
        )

    assert "duplicate" in str(exc_info.value).lower()
    assert "channel_monthly" in str(exc_info.value)

    fail_path = tmp_path / QA_DIR / f"{FAIL_PREFIX}channel_monthly.json"
    assert fail_path.exists(), "qa/agg_qa_fail_<name>.json should be written"
    ctx = json.loads(fail_path.read_text())
    assert ctx["table_name"] == "channel_monthly"
    assert "reason" in ctx
    assert "duplicate" in ctx["reason"].lower()
    assert "counts" in ctx
    assert ctx["counts"].get("n_duplicate_rows") == 2
    assert "sample_duplicate_keys" in ctx
    assert len(ctx["sample_duplicate_keys"]) >= 1
    assert "dtypes" in ctx
    assert "policy_excerpt" in ctx
    assert "time_key" in ctx["policy_excerpt"] or "measures" in str(ctx["policy_excerpt"])


def test_qa_missing_month_end_fails_and_writes_context(tmp_path: Path) -> None:
    """When month_end column is missing, QA fails and writes qa/agg_qa_fail_<name>.json."""
    (tmp_path / "p.yml").write_text(MINIMAL_POLICY_YAML, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")

    df = pd.DataFrame({
        "wrong_key": [pd.Timestamp("2024-01-31")],
        "begin_aum": [100.0],
        "nnb": [10.0],
    })

    with pytest.raises(AggQAError) as exc_info:
        validate_agg_qa(
            df,
            "month_end",
            [],
            ["begin_aum", "nnb"],
            "firm_monthly",
            policy,
            tmp_path,
        )

    msg = str(exc_info.value)
    assert "month_end" in msg or "time_key" in msg
    assert "missing" in msg.lower() or "firm_monthly" in msg

    fail_path = tmp_path / QA_DIR / f"{FAIL_PREFIX}firm_monthly.json"
    assert fail_path.exists()
    ctx = json.loads(fail_path.read_text())
    assert ctx["table_name"] == "firm_monthly"
    assert "reason" in ctx
    assert "policy_excerpt" in ctx
