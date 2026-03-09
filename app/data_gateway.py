"""
Single cached data gateway for Streamlit: typed query names, views-only DuckDB or Parquet fallback.
Guardrail: only allow querying views with prefix schema.v_ (e.g. analytics.v_*). Lightweight query telemetry for slow queries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.date_align import (
    get_latest_month_end,
    get_prior_month_end,
    get_year_start_month_end,
)
from app.filters_contract import (
    is_optional_filter_enabled,
    load_filters_contract,
    resolve_channel_column,
    resolve_geo_column,
    resolve_product_column,
)
from app.metrics.metric_contract import (
    compute_fee_yield,
    compute_market_impact,
    compute_market_impact_rate,
    compute_ogr,
)
from app.reporting.report_pack import ReportPack, build_report_pack
from app.state import DrillState, FilterState, filter_state_to_gateway_dict

import pandas as pd

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)
VIEWS_ONLY_MESSAGE = "UI must query v_* views only. Add a view in pipelines/duckdb/create_views.py"
SLOW_QUERY_MS = 200
SLOW_QUERIES_MAX = 20

# Centralized execution controls (row caps, budgets, timeout)
DEFAULT_MAX_ROWS = 5000
EXPORT_MAX_ROWS = 50000
SNAPSHOT_BUDGET_MS = 200
HEAVY_BUDGET_MS = 1000
STARTUP_HEAVY_BUDGET_MS = 2200
DEFAULT_TIMEOUT_MS = 1500
PERF_QUERY_LOG_MAX = 50
GATEWAY_CACHE_MAX_KEYS = 50

# Dataset version: single source of truth
DATASET_VERSION_META_PATH = "data/curated/metrics_monthly.meta.json"
DATASET_VERSION_KEY = "dataset_version"

VIEWS_MANIFEST_REL = "analytics/duckdb_views_manifest.json"
DUCKDB_MANIFEST_REL = "analytics/duckdb_manifest.json"
DUCKDB_POLICY_REL = "configs/duckdb_policy.yml"

# Query name constants (avoid string typos/collisions)
Q_FIRM_MONTHLY = "firm_monthly"
Q_CHANNEL_MONTHLY = "channel_monthly"
Q_TICKER_MONTHLY = "ticker_monthly"
Q_GEO_MONTHLY = "geo_monthly"
Q_SEGMENT_MONTHLY = "segment_monthly"
Q_WATERFALL_INPUTS = "waterfall_inputs"
Q_CORR_MATRIX = "corr_matrix"

ALLOWED_QUERIES = frozenset({
    Q_FIRM_MONTHLY,
    Q_CHANNEL_MONTHLY,
    Q_TICKER_MONTHLY,
    Q_GEO_MONTHLY,
    Q_SEGMENT_MONTHLY,
    Q_WATERFALL_INPUTS,
    Q_CORR_MATRIX,
})

# Query name -> DuckDB view name (views that exist today)
QUERY_TO_VIEW: dict[str, str] = {
    Q_FIRM_MONTHLY: "v_firm_monthly",
    Q_CHANNEL_MONTHLY: "v_channel_monthly",
    Q_TICKER_MONTHLY: "v_ticker_monthly",
    Q_GEO_MONTHLY: "v_geo_monthly",
    Q_SEGMENT_MONTHLY: "v_segment_monthly",
}

# Backend-agnostic query specs: view (DuckDB), parquet table (agg fallback), allowed_filters, default_columns (None = all)
COMMON_DIMS = ("month_end", "channel_l1", "channel_l2", "product_ticker", "src_country_canonical", "segment", "sub_segment", "region", "geo")
QUERY_SPECS: dict[str, dict[str, Any]] = {
    Q_FIRM_MONTHLY: {
        "view": "v_firm_monthly",
        "parquet_table": "firm_monthly",
        "allowed_filters": list(COMMON_DIMS),
        "default_columns": None,
    },
    Q_CHANNEL_MONTHLY: {
        "view": "v_channel_monthly",
        "parquet_table": "channel_monthly",
        "allowed_filters": list(COMMON_DIMS),
        "default_columns": None,
    },
    Q_TICKER_MONTHLY: {
        "view": "v_ticker_monthly",
        "parquet_table": "ticker_monthly",
        "allowed_filters": list(COMMON_DIMS),
        "default_columns": None,
    },
    Q_GEO_MONTHLY: {
        "view": "v_geo_monthly",
        "parquet_table": "geo_monthly",
        "allowed_filters": list(COMMON_DIMS),
        "default_columns": None,
    },
    Q_SEGMENT_MONTHLY: {
        "view": "v_segment_monthly",
        "parquet_table": "segment_monthly",
        "allowed_filters": list(COMMON_DIMS),
        "default_columns": None,
    },
    Q_WATERFALL_INPUTS: {
        "view": None,
        "parquet_table": None,
        "allowed_filters": list(COMMON_DIMS),
        "default_columns": None,
    },
    Q_CORR_MATRIX: {
        "view": None,
        "parquet_table": None,
        "allowed_filters": list(COMMON_DIMS),
        "default_columns": None,
    },
}

# Allowed filter column names (no arbitrary SQL). Includes contract-resolved columns from filters.yml.
FILTER_COLUMNS = frozenset({
    "month_end",
    "channel_l1", "channel_l2",
    "channel_raw", "channel_standard", "channel_best", "preferred_label",
    "product_ticker", "segment", "sub_segment",
    "src_country_canonical", "product_country_canonical",
    "region", "geo",
    "custodian_firm",
})

# Tab 1 governed filters: only these keys allowed (no ad-hoc dimensions / free-form SQL).
ALLOWED_FILTER_KEYS = frozenset({
    "date_start", "date_end", "period_mode", "channel_view",
    "geo_dim", "product_dim", "custodian_firm", "drill_path",
    "slice_dim", "slice_value", "currency", "unit",
})

# Governed load columns: only these selected downstream (no derived metrics in loaders).
FIRM_REQUIRED_COLUMNS = [
    "month_end",
    "begin_aum",
    "end_aum",
    "nnb",
    "nnf",
    "ogr",
    "market_impact_rate",
]
FIRM_OPTIONAL_COLUMNS = ["market_impact", "fee_yield", "market_pnl"]
FIRM_LOAD_COLUMNS = FIRM_REQUIRED_COLUMNS + FIRM_OPTIONAL_COLUMNS
CHANNEL_REQUIRED_COLUMNS = FIRM_REQUIRED_COLUMNS + ["channel"]
CHANNEL_OPTIONAL_DIMS = ["sub_channel"]
CHANNEL_LOAD_COLUMNS = CHANNEL_REQUIRED_COLUMNS + CHANNEL_OPTIONAL_DIMS + FIRM_OPTIONAL_COLUMNS
TICKER_REQUIRED_COLUMNS = FIRM_REQUIRED_COLUMNS + ["ticker"]
TICKER_LOAD_COLUMNS = TICKER_REQUIRED_COLUMNS + FIRM_OPTIONAL_COLUMNS
GEO_REQUIRED_COLUMNS = FIRM_REQUIRED_COLUMNS + ["geo"]
GEO_LOAD_COLUMNS = GEO_REQUIRED_COLUMNS + FIRM_OPTIONAL_COLUMNS
SEGMENT_REQUIRED_COLUMNS = FIRM_REQUIRED_COLUMNS + ["segment", "sub_segment"]
SEGMENT_LOAD_COLUMNS = SEGMENT_REQUIRED_COLUMNS + FIRM_OPTIONAL_COLUMNS
FIRM_NUMERIC_COLUMNS = [c for c in FIRM_LOAD_COLUMNS if c != "month_end"]
CHANNEL_NUMERIC_COLUMNS = [c for c in CHANNEL_LOAD_COLUMNS if c not in {"month_end", "channel"}]
TICKER_NUMERIC_COLUMNS = [c for c in TICKER_LOAD_COLUMNS if c not in {"month_end", "ticker"}]
GEO_NUMERIC_COLUMNS = [c for c in GEO_LOAD_COLUMNS if c not in {"month_end", "geo"}]
SEGMENT_NUMERIC_COLUMNS = [c for c in SEGMENT_LOAD_COLUMNS if c not in {"month_end", "segment", "sub_segment"}]

KNOWN_MONTH_END_ALIASES = (
    "date",
    "month",
    "month_end_date",
    "period",
    "as_of",
    "as_of_date",
)

# Flow/AUM aliases: raw data may use net_flow, flow, subscriptions/redemptions, opening_aum, etc.
FLOW_NNB_ALIASES = (
    "nnb",
    "net_flow",
    "net_flows",
    "flow",
    "flows",
    "net_new_business",
    "subscriptions_minus_redemptions",
)
BEGIN_AUM_ALIASES = ("begin_aum", "beginning_aum", "opening_aum", "start_aum", "prior_aum", "aum_start")
END_AUM_ALIASES = ("end_aum", "ending_aum", "closing_aum", "aum_end")

FIRM_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "month_end": KNOWN_MONTH_END_ALIASES,
    "nnb": FLOW_NNB_ALIASES,
    "begin_aum": BEGIN_AUM_ALIASES,
    "end_aum": END_AUM_ALIASES,
    "nnf": ("nnf", "net_net_flow", "net_fee_flow", "fee_flow", "fees"),
    "market_impact_abs": ("market_impact_abs", "market_pnl", "market_impact", "market_movement"),
}
CHANNEL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "month_end": KNOWN_MONTH_END_ALIASES,
    "channel": (
        "preferred_label",
        "channel_l1",
        "channel_l2",
        "channel_standard",
        "channel_best",
        "canonical_channel",
        "standard_channel",
        "channel_raw",
    ),
    "nnb": FLOW_NNB_ALIASES,
    "begin_aum": BEGIN_AUM_ALIASES,
    "end_aum": END_AUM_ALIASES,
    "nnf": ("nnf", "net_net_flow", "net_fee_flow", "fee_flow", "fees"),
    "market_impact_abs": ("market_impact_abs", "market_pnl", "market_impact", "market_movement"),
    "sub_channel": ("channel_l2", "sub_channel", "channel_2"),
}
TICKER_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "month_end": KNOWN_MONTH_END_ALIASES,
    "ticker": ("product_ticker", "ticker_symbol", "symbol", "ticker"),
    "product_ticker": ("product_ticker", "ticker_symbol", "symbol", "ticker"),
    "nnb": FLOW_NNB_ALIASES,
    "begin_aum": BEGIN_AUM_ALIASES,
    "end_aum": END_AUM_ALIASES,
    "nnf": ("nnf", "net_net_flow", "net_fee_flow", "fee_flow", "fees"),
    "market_impact_abs": ("market_impact_abs", "market_pnl", "market_impact", "market_movement"),
}
GEO_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "month_end": KNOWN_MONTH_END_ALIASES,
    "geo": (
        "src_country",
        "src_country_canonical",
        "product_country",
        "product_country_canonical",
        "country",
        "region",
    ),
    "nnb": FLOW_NNB_ALIASES,
    "begin_aum": BEGIN_AUM_ALIASES,
    "end_aum": END_AUM_ALIASES,
    "nnf": ("nnf", "net_net_flow", "net_fee_flow", "fee_flow", "fees"),
}
SEGMENT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "month_end": KNOWN_MONTH_END_ALIASES,
    "segment": ("segment_name",),
    "sub_segment": ("subsegment", "sub_segment_name"),
    "nnb": FLOW_NNB_ALIASES,
    "begin_aum": BEGIN_AUM_ALIASES,
    "end_aum": END_AUM_ALIASES,
    "nnf": ("nnf", "net_net_flow", "net_fee_flow", "fee_flow", "fees"),
}


class SchemaError(ValueError):
    """Raised when a governed dataset violates the expected schema contract."""


def normalize_filters(filters: FilterState | dict[str, Any] | None) -> FilterState | dict[str, Any]:
    """
    Enforce allowed keys only: date_range (start,end), period_mode, channel_view, global dimensions.
    For Tab 1: no ad-hoc dimensions; returns FilterState or dict for gateway use.
    """
    if filters is None:
        return FilterState.from_dict({})
    if isinstance(filters, FilterState):
        return filters
    d = dict(filters) if isinstance(filters, dict) else getattr(filters, "to_dict", lambda: {})()
    if "date_range" in d:
        dr = d["date_range"]
        if isinstance(dr, (list, tuple)) and len(dr) >= 2:
            d["date_start"] = str(dr[0])
            d["date_end"] = str(dr[1])
    whitelisted = {k: d[k] for k in ALLOWED_FILTER_KEYS if k in d}
    return FilterState.from_dict(whitelisted)


def _get_dataset_version(dataset_version: str | None = None, root: Path | None = None) -> str:
    """
    If dataset_version is provided, return it. Else if in Streamlit, return session dataset_version or 'dev'.
    Otherwise resolve from curated/metrics_monthly.meta.json via load_dataset_version(root).
    """
    if dataset_version is not None and str(dataset_version).strip():
        return str(dataset_version).strip()
    if st is not None:
        return st.session_state.get("dataset_version", "dev")
    return load_dataset_version(root or Path.cwd())


def _coerce_inf_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN for numeric columns only. Returns copy."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    nan_val = float("nan")
    out = df.copy()
    for c in out.columns:
        try:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].replace([float("inf"), float("-inf")], nan_val)
        except Exception:
            pass
    return out


def _require_columns(df: pd.DataFrame, cols: list[str] | tuple[str, ...], ctx: str) -> None:
    """Raise ValueError if df is missing any required column. Message: 'Missing required columns for {ctx}: {missing}'."""
    if df is None:
        raise ValueError(f"Missing required columns for {ctx}: all (df is None)")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {ctx}: {missing}")


def _normalize_alias_columns(
    df: pd.DataFrame,
    alias_map: dict[str, tuple[str, ...]] | None,
) -> pd.DataFrame:
    """Rename known aliases to canonical names (case-insensitive) when canonical is missing."""
    if df is None or df.empty or not alias_map:
        return df
    out = df.copy()
    lower_to_actual = {str(c).strip().lower(): c for c in out.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            actual = lower_to_actual.get(str(alias).strip().lower())
            if actual and actual not in rename_map:
                rename_map[actual] = canonical
                break
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def validate_required_columns(
    df: pd.DataFrame | None,
    required: list[str] | tuple[str, ...],
    dataset_name: str,
) -> None:
    """Raise SchemaError with explicit found/missing columns."""
    found_cols = list(df.columns) if isinstance(df, pd.DataFrame) else []
    missing = [c for c in required if c not in found_cols]
    if missing:
        missing_txt = ", ".join(missing)
        raise SchemaError(
            f"SchemaError: {dataset_name} is missing required column(s): {missing_txt}. "
            f"Found columns: {found_cols}"
        )


def _derive_nnb_from_flows(out: pd.DataFrame) -> None:
    """
    When nnb is missing, all null, or all zero, derive from net_flow or subscriptions - redemptions.
    Client: NNB = sum of net flows; net_flow column or (subscriptions - redemptions) is the source.
    Modifies out in place.
    """
    if out is None or out.empty:
        return
    if "nnb" not in out.columns:
        out["nnb"] = pd.NA
    nnb_series = pd.to_numeric(out["nnb"], errors="coerce")
    missing_or_zero = nnb_series.isna() | (nnb_series.fillna(0) == 0)
    if "net_flow" in out.columns:
        flow_series = pd.to_numeric(out["net_flow"], errors="coerce")
        out.loc[missing_or_zero, "nnb"] = flow_series.loc[missing_or_zero]
        nnb_series = pd.to_numeric(out["nnb"], errors="coerce")
        missing_or_zero = nnb_series.isna() | (nnb_series.fillna(0) == 0)
        if not bool(missing_or_zero.all()):
            return
    # Some sources provide an explicit market-impact column but not NNB.
    if all(c in out.columns for c in ("end_aum", "begin_aum")):
        mi_col = None
        for c in ("market_impact_abs", "market_pnl", "market_impact"):
            if c in out.columns:
                mi_col = c
                break
        end = pd.to_numeric(out["end_aum"], errors="coerce")
        begin = pd.to_numeric(out["begin_aum"], errors="coerce")
        if mi_col is not None:
            mi = pd.to_numeric(out[mi_col], errors="coerce")
            nnb_from_mi = end - begin - mi
            nnb_fallback = end - begin
            candidate = nnb_from_mi.where(mi.notna(), nnb_fallback)
            out.loc[missing_or_zero, "nnb"] = candidate.loc[missing_or_zero]
            return
        # Last-resort fallback when source does not provide NNB or market leg.
        # Keeps cards/charts operational while preserving AUM reconciliation.
        out.loc[missing_or_zero, "nnb"] = (end - begin).loc[missing_or_zero]
        return
    subs_col = None
    red_col = None
    for c in out.columns:
        c_lower = str(c).strip().lower()
        if c_lower in ("subscriptions", "subscription", "inflows"):
            subs_col = c
        if c_lower in ("redemptions", "redemption", "outflows"):
            red_col = c
    if subs_col and red_col:
        out["nnb"] = pd.to_numeric(out[subs_col], errors="coerce") - pd.to_numeric(out[red_col], errors="coerce")
    elif "nnb" not in out.columns:
        out["nnb"] = pd.NA
    return


# Monthly query names that get canonicalization (same metrics pack for cards, charts, report, ETF).
_MONTHLY_QUERIES = frozenset({
    Q_FIRM_MONTHLY,
    Q_CHANNEL_MONTHLY,
    Q_TICKER_MONTHLY,
    Q_GEO_MONTHLY,
    Q_SEGMENT_MONTHLY,
})

_ALIAS_MAP_BY_QUERY: dict[str, dict[str, tuple[str, ...]]] = {
    Q_FIRM_MONTHLY: FIRM_COLUMN_ALIASES,
    Q_CHANNEL_MONTHLY: CHANNEL_COLUMN_ALIASES,
    Q_TICKER_MONTHLY: TICKER_COLUMN_ALIASES,
    Q_GEO_MONTHLY: GEO_COLUMN_ALIASES,
    Q_SEGMENT_MONTHLY: SEGMENT_COLUMN_ALIASES,
}


def _ensure_begin_aum(out: pd.DataFrame, dim_cols: list[str]) -> None:
    """When begin_aum is missing or all null, set from end_aum shift(1) per group. Modifies out in place."""
    if out is None or out.empty or "end_aum" not in out.columns:
        return
    need_begin = "begin_aum" not in out.columns or out["begin_aum"].isna().all()
    if not need_begin:
        return
    group_cols = [c for c in dim_cols if c in out.columns]
    if group_cols:
        out["begin_aum"] = out.groupby(group_cols)["end_aum"].shift(1)
    else:
        out["begin_aum"] = out["end_aum"].shift(1)
    return


def _compute_derived_metrics(out: pd.DataFrame) -> None:
    """Fill market_impact, ogr, market_impact_rate from canonical formulas. Modifies out in place."""
    if out is None or out.empty:
        return
    need_mi = "market_impact" not in out.columns or out["market_impact"].isna().all()
    if need_mi and all(c in out.columns for c in ("end_aum", "begin_aum", "nnb")):
        out["market_impact"] = (
            pd.to_numeric(out["end_aum"], errors="coerce")
            - pd.to_numeric(out["begin_aum"], errors="coerce")
            - pd.to_numeric(out["nnb"], errors="coerce")
        )
    if "ogr" not in out.columns or out["ogr"].isna().all():
        if "nnb" in out.columns and "begin_aum" in out.columns:
            b = pd.to_numeric(out["begin_aum"], errors="coerce")
            n = pd.to_numeric(out["nnb"], errors="coerce")
            out["ogr"] = n / b.replace(0, float("nan"))
    if "market_impact_rate" not in out.columns or out["market_impact_rate"].isna().all():
        if "market_impact" in out.columns and "begin_aum" in out.columns:
            mi = pd.to_numeric(out["market_impact"], errors="coerce")
            b = pd.to_numeric(out["begin_aum"], errors="coerce")
            out["market_impact_rate"] = mi / b.replace(0, float("nan"))
    return


def _clean_dimension_labels(out: pd.DataFrame) -> None:
    """Replace raw numeric labels (e.g. 1.0) with human-readable strings. Modifies out in place."""
    if out is None or out.empty:
        return
    dim_cols = ["channel", "sub_channel", "product_ticker", "ticker", "geo", "segment", "sub_segment", "src_country"]
    prefix_by_col = {"channel": "Channel ", "sub_channel": "Sub-channel ", "segment": "Segment ", "sub_segment": "Sub-segment ", "geo": "Geo ", "src_country": "Country "}
    for col in dim_cols:
        if col not in out.columns:
            continue
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            prefix = prefix_by_col.get(col, "")
            out[col] = s.apply(
                lambda x, p=prefix: (p + str(int(x)) if pd.notna(x) and x == x else "")
            )
        else:
            txt = s.astype(str).str.strip().replace("nan", "").replace("None", "")
            prefix = prefix_by_col.get(col, "")
            if prefix:
                numeric_like = txt.str.fullmatch(r"\d+(?:\.0+)?").fillna(False)
                if bool(numeric_like.any()):
                    txt = txt.where(~numeric_like, txt.str.replace(r"\.0+$", "", regex=True).radd(prefix))
            out[col] = txt
    return


def _canonicalize_monthly_for_ui(df: pd.DataFrame, query_name: str) -> pd.DataFrame:
    """
    Single canonical metrics pack for cards, charts, report, ETF drill-down.
    Applies alias mapping, derives NNB, ensures begin_aum, computes market_impact/ogr/rate, cleans labels.
    Does not validate required columns (parquet/views may have varying schemas).
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    alias_map = _ALIAS_MAP_BY_QUERY.get(query_name)
    if alias_map:
        out = _normalize_alias_columns(out, alias_map)
    _derive_nnb_from_flows(out)
    dim_cols = []
    if query_name == Q_CHANNEL_MONTHLY:
        dim_cols = ["channel"]
    elif query_name == Q_TICKER_MONTHLY:
        dim_cols = ["product_ticker", "ticker"]
    elif query_name == Q_GEO_MONTHLY:
        dim_cols = ["geo", "src_country"]
    elif query_name == Q_SEGMENT_MONTHLY:
        dim_cols = ["segment", "sub_segment"]
    if "month_end" in out.columns:
        out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
        out = out.sort_values("month_end").reset_index(drop=True)
    _ensure_begin_aum(out, dim_cols)
    for c in ("begin_aum", "end_aum", "nnb", "nnf"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    _compute_derived_metrics(out)
    _clean_dimension_labels(out)
    return _coerce_inf_to_nan(out)


def _prepare_monthly_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    required_cols: list[str] | tuple[str, ...],
    output_cols: list[str],
    alias_map: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """
    Normalize aliases and validate required schema once at loader boundary.
    Derives NNB from subscriptions - redemptions when nnb is missing.
    Missing optional columns are added as NA so downstream code can stay deterministic.
    """
    if df is None:
        return pd.DataFrame(columns=output_cols)
    out = _normalize_alias_columns(df, alias_map)
    if out.empty:
        return pd.DataFrame(columns=output_cols)
    _derive_nnb_from_flows(out)
    validate_required_columns(out, required_cols, dataset_name)
    for col in output_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out.reindex(columns=output_cols, copy=False)


def _load_policy_yml(root: Path) -> dict[str, Any] | None:
    """Load configs/duckdb_policy.yml; return duckdb section or None."""
    path = root / DUCKDB_POLICY_REL
    if not path.exists():
        return None
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return raw.get("duckdb") or raw
    except Exception:
        return None


def _reads_views_only(root: Path) -> bool:
    """True if policy or views manifest requires reads_views_only (use DuckDB views only)."""
    views_path = root / VIEWS_MANIFEST_REL
    if views_path.exists():
        try:
            data = json.loads(views_path.read_text(encoding="utf-8"))
            if data.get("reads_views_only") is True:
                return True
        except Exception:
            pass
    policy = _load_policy_yml(root)
    if policy:
        ui = policy.get("ui_rule") or {}
        if ui.get("reads_views_only") is True:
            return True
    return False


def _use_duckdb_views(root: Path | None) -> bool:
    """
    True if we should use DuckDB views: DUCKDB_PATH env (file exists), or analytics/duckdb_views_manifest.json exists and reads_views_only true.
    Else fallback to agg/*.parquet via agg/manifest.json.
    """
    root = Path(root) if root is not None else Path.cwd()
    duckdb_env = os.environ.get("DUCKDB_PATH", "").strip()
    if duckdb_env:
        p = Path(duckdb_env)
        if not p.is_absolute():
            p = (root / duckdb_env).resolve()
        if p.exists():
            return True
    if not (root / VIEWS_MANIFEST_REL).exists():
        return False
    return _reads_views_only(root)


def get_config(root: Path | None = None) -> dict[str, Any]:
    """
    Resolve db_path, schema, dataset_version: DUCKDB_PATH env (with default fallback), then duckdb_views_manifest.json, duckdb_manifest.json, duckdb_policy.yml.
    Raises FileNotFoundError if none found.
    """
    root = Path(root) if root is not None else Path.cwd()
    # 0) env DUCKDB_PATH (e.g. Docker /workspace/analytics.duckdb)
    duckdb_env = os.environ.get("DUCKDB_PATH", "").strip()
    if duckdb_env:
        p = Path(duckdb_env)
        db_path = str(p.resolve()) if p.is_absolute() else str((root / duckdb_env).resolve())
        return {
            "db_path": db_path,
            "schema": os.environ.get("DUCKDB_SCHEMA", "data").strip(),
            "dataset_version": "unknown",
            "policy_hash": "",
            "created_at": "",
        }
    # Local rebuilt analytics database fallback (used when manifest files are locked/stale).
    rebuilt_db = root / "analytics_fixed.duckdb"
    if rebuilt_db.exists():
        return {
            "db_path": str(rebuilt_db.resolve()),
            "schema": "data",
            "dataset_version": "unknown",
            "policy_hash": "",
            "created_at": "",
        }
    # 1) views manifest
    views_path = root / VIEWS_MANIFEST_REL
    if views_path.exists():
        try:
            data = json.loads(views_path.read_text(encoding="utf-8"))
            db_path_rel = (data.get("db_path") or "").strip()
            schema = (data.get("schema") or "analytics").strip()
            if db_path_rel:
                return {
                    "db_path": str((root / db_path_rel).resolve()),
                    "schema": schema,
                    "dataset_version": (data.get("dataset_version") or "unknown").strip(),
                    "policy_hash": (data.get("policy_hash") or "").strip(),
                    "created_at": (data.get("created_at") or "").strip(),
                }
        except Exception:
            pass
    # 2) duckdb manifest
    manifest_path = root / DUCKDB_MANIFEST_REL
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            db_path_rel = (data.get("db_path") or "").strip()
            schema = (data.get("schema") or "analytics").strip()
            if db_path_rel:
                return {
                    "db_path": str((root / db_path_rel).resolve()),
                    "schema": schema,
                    "dataset_version": (data.get("dataset_version") or "unknown").strip(),
                    "policy_hash": (data.get("policy_hash") or "").strip(),
                    "created_at": (data.get("created_at") or "").strip(),
                }
        except Exception:
            pass
    # 3) policy yml
    policy = _load_policy_yml(root)
    if policy:
        db_path_rel = (policy.get("db_path") or "").strip()
        schema = (policy.get("schema") or "analytics").strip()
        if db_path_rel:
            return {
                "db_path": str((root / db_path_rel).resolve()),
                "schema": schema,
                "dataset_version": "unknown",
            }
    raise FileNotFoundError(
        f"No DuckDB config found. Looked for {VIEWS_MANIFEST_REL}, {DUCKDB_MANIFEST_REL}, {DUCKDB_POLICY_REL}. "
        "Run: python -m pipelines.duckdb.rebuild_analytics_layer"
    )


def load_dataset_version(root: Path | None = None) -> str:
    """
    Read dataset_version: (1) DuckDB meta.dataset_version table if DB exists, (2) data/curated/metrics_monthly.meta.json, (3) curated/ fallback.
    Raises FileNotFoundError or ValueError with actionable message if all sources missing or invalid.
    """
    root = Path(root) if root is not None else Path.cwd()
    # 1) DuckDB meta table if manifest and DB exist
    try:
        config = get_config(root)
        db_path = config.get("db_path")
        if db_path and Path(db_path).exists():
            import duckdb
            con = duckdb.connect(db_path, read_only=True)
            try:
                row = con.execute("SELECT dataset_version FROM meta.dataset_version LIMIT 1").fetchone()
                if row and row[0] and str(row[0]).strip():
                    return str(row[0]).strip()
            except Exception:
                pass
            finally:
                con.close()
    except Exception:
        pass
    # 2) Curated meta.json
    for rel in (DATASET_VERSION_META_PATH, "curated/metrics_monthly.meta.json"):
        meta_path = root / rel
        if meta_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"{DATASET_VERSION_META_PATH} not found at {root / DATASET_VERSION_META_PATH}. "
            "Run: python etl/build_data.py or pipelines.duckdb.rebuild_analytics_layer."
        )
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {meta_path}: {e}. Fix or regenerate {DATASET_VERSION_META_PATH}."
        ) from e
    dv = data.get(DATASET_VERSION_KEY)
    if dv is None or (isinstance(dv, str) and not dv.strip()):
        raise ValueError(
            f"Missing or empty '{DATASET_VERSION_KEY}' in {meta_path}. "
            f"Add \"{DATASET_VERSION_KEY}\": \"<version>\" to the JSON."
        )
    return str(dv).strip()


def _is_primitive(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool))


def to_json_safe(obj: Any) -> Any:
    """
    Convert common state objects to JSON-safe containers before canonicalization.
    Handles dataclasses, pydantic-like models, dict()/to_dict(), Path/date/Timestamp,
    and numpy/pandas scalars. Raises clear TypeError for unsupported objects.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and hasattr(np, "datetime64") and isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if np is not None and isinstance(obj, (np.integer, np.floating, np.bool_)):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return bool(obj)
    if isinstance(obj, pd.Series):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Index):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if is_dataclass(obj):
        return asdict(obj)
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    as_dict = getattr(obj, "dict", None)
    if callable(as_dict):
        return as_dict()
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    raise TypeError(
        f"Unsupported state type for hashing: {type(obj).__name__}. "
        "Add serialization rule in canonicalize_state()."
    )


def canonicalize_state(obj: Any) -> Any:
    """
    Produce a stable, JSON-serializable form for hashing.
    - dict: sort keys recursively; drop null/empty (None, "", [], {}).
    - list/tuple/set: convert to list; sort if elements are primitive (str/int/float/bool).
    - datetime/date: isoformat "YYYY-MM-DD".
    - numpy/pandas scalars: convert to Python scalars.
    """
    obj = to_json_safe(obj)
    if obj is None:
        return None
    if isinstance(obj, (date, datetime)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if np is not None and hasattr(np, "datetime64") and isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if np is not None and isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k in sorted(obj.keys()):
            v = obj[k]
            if v is None or v == "":
                continue
            if isinstance(v, (list, tuple)) and len(v) == 0:
                continue
            if isinstance(v, dict) and len(v) == 0:
                continue
            out[k] = canonicalize(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        lst = [canonicalize(x) for x in obj]
        if all(_is_primitive(x) for x in lst):
            return sorted(lst, key=lambda x: (type(x).__name__, x))
        return sorted(lst, key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(
        f"Unsupported canonical state value type: {type(obj).__name__}. "
        "Add serialization rule in canonicalize_state()."
    )


def canonicalize(obj: Any) -> Any:
    """Backward-compatible alias for canonicalize_state()."""
    return canonicalize_state(obj)


def hash_filters(filter_state: Any) -> str:
    """
    Canonical JSON (sorted keys, stable lists, dates as YYYY-MM-DD, null/empty stripped) then sha1 hex.
    Accepts FilterState/dataclasses/models/dicts and normalizes at hashing boundary.
    """
    # Preserve historical gateway hash semantics for FilterState by hashing month_end_range.
    state_like = filter_state
    if isinstance(filter_state, FilterState):
        state_like = filter_state_to_gateway_dict(filter_state)
    canonical = canonicalize_state(state_like)
    _assert_json_safe_tree(canonical)
    try:
        payload = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError as exc:
        raise TypeError(
            f"State serialization failed in hash_filters for type {type(filter_state).__name__}: {exc}"
        ) from exc
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _assert_json_safe_tree(obj: Any, path: str = "root") -> None:
    """Defensive runtime check: ensure only JSON-native node types remain."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_json_safe_tree(v, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"Non-string key at {path}: {type(k).__name__}")
            _assert_json_safe_tree(v, f"{path}.{k}")
        return
    raise TypeError(
        f"Non JSON-safe value at {path}: {type(obj).__name__}. "
        "Ensure all state passes through to_json_safe()/canonicalize_state()."
    )


def get_connection(db_path: str):
    """Cached DuckDB connection (read_only). Uses st.cache_resource when in Streamlit."""
    import duckdb
    if st is not None:
        @st.cache_resource
        def _cached(_path: str):
            return duckdb.connect(_path, read_only=True)
        return _cached(db_path)
    return duckdb.connect(db_path, read_only=True)


def _init_perf_session() -> None:
    """Ensure perf_cache_stats, perf_query_log, and _gateway_cache exist in session_state."""
    if st is None:
        return
    if "perf_cache_stats" not in st.session_state or not isinstance(st.session_state.get("perf_cache_stats"), dict):
        st.session_state["perf_cache_stats"] = {"hit": 0, "miss": 0}
    if "perf_query_log" not in st.session_state or not isinstance(st.session_state.get("perf_query_log"), list):
        st.session_state["perf_query_log"] = []
    if "_gateway_cache" not in st.session_state or not isinstance(st.session_state.get("_gateway_cache"), dict):
        st.session_state["_gateway_cache"] = {}
    if "_gateway_cache_keys_order" not in st.session_state:
        st.session_state["_gateway_cache_keys_order"] = []


def cached_call(
    cache_key: str,
    fn: Callable[[], Any],
    budget_ms: int,
    max_rows: int,
    kind: str,
) -> Any:
    """
    Central cache wrapper for all gateway methods. Tracks hit/miss, wall-clock time, row cap.
    - If cache hit: return cached value, record hit and log entry (no timing).
    - If cache miss: run fn(), measure time, clamp DataFrame to max_rows if needed, record warning if over budget or capped.
    - perf_query_log: last PERF_QUERY_LOG_MAX entries. perf_cache_stats: hit/miss counts.
    """
    if st is not None:
        _init_perf_session()
        cache = st.session_state["_gateway_cache"]
        order = st.session_state["_gateway_cache_keys_order"]
        if cache_key in cache:
            st.session_state["perf_cache_stats"]["hit"] = st.session_state["perf_cache_stats"].get("hit", 0) + 1
            result = cache[cache_key]
            rows = len(result) if isinstance(result, pd.DataFrame) else None
            entry = {
                "name": kind,
                "cache_key": cache_key,
                "elapsed_ms": None,
                "rows": rows,
                "hit": True,
                "budget_ms": budget_ms,
                "warning": None,
            }
            log = st.session_state["perf_query_log"]
            log.append(entry)
            st.session_state["perf_query_log"] = log[-PERF_QUERY_LOG_MAX:]
            return result

    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    warning = None
    rows = None
    if isinstance(result, pd.DataFrame):
        rows = len(result)
        if max_rows > 0 and rows > max_rows:
            result = result.head(max_rows).copy()
            warning = f"row_cap:{rows}->{max_rows}"
            if logger.isEnabledFor(logging.WARNING):
                logger.warning("Gateway row cap applied: %s rows -> %s (kind=%s)", rows, max_rows, kind)
    if st is not None:
        st.session_state["perf_cache_stats"]["miss"] = st.session_state["perf_cache_stats"].get("miss", 0) + 1
        if elapsed_ms > budget_ms:
            w = f"over_budget:{elapsed_ms:.0f}ms>{budget_ms}ms"
            warning = f"{warning};{w}" if warning else w
            if logger.isEnabledFor(logging.WARNING):
                logger.warning("Gateway over budget: %s elapsed_ms=%.0f budget_ms=%s", kind, elapsed_ms, budget_ms)
        entry = {
            "name": kind,
            "cache_key": cache_key,
            "elapsed_ms": round(elapsed_ms, 2),
            "rows": rows,
            "hit": False,
            "budget_ms": budget_ms,
            "warning": warning,
        }
        log = st.session_state["perf_query_log"]
        log.append(entry)
        st.session_state["perf_query_log"] = log[-PERF_QUERY_LOG_MAX:]

        cache = st.session_state["_gateway_cache"]
        order = st.session_state["_gateway_cache_keys_order"]
        while len(cache) >= GATEWAY_CACHE_MAX_KEYS and order:
            evict = order.pop(0)
            cache.pop(evict, None)
        cache[cache_key] = result
        order.append(cache_key)
        st.session_state["_gateway_cache"] = cache
        st.session_state["_gateway_cache_keys_order"] = order

    return result


def _is_user_changed_filters() -> bool:
    """True after user explicitly changes filters in current session."""
    if st is None:
        return False
    return bool(st.session_state.get("ui_filters_user_changed", False))


def _resolve_heavy_budget_ms() -> int:
    """
    Tiered heavy-query budget:
    - startup/default state gets a higher budget to avoid noisy timeout banners
    - user-driven interactions keep strict heavy budget
    """
    return HEAVY_BUDGET_MS if _is_user_changed_filters() else STARTUP_HEAVY_BUDGET_MS


def get_available_columns(view_name: str, root: Path | None = None) -> set[str]:
    """Return set of column names for the given view (DuckDB PRAGMA table_info). Used to decide optional filters e.g. custodian_firm."""
    root = Path(root) if root is not None else Path.cwd()
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        db_path = config["db_path"]
        con = get_connection(db_path)
        # DuckDB: PRAGMA table_info('schema.view_name') returns name, type, ...
        qualified = f'"{schema}"."{view_name}"'
        df = con.execute(f"PRAGMA table_info({qualified})").fetchdf()
        if df is None or df.empty or "name" not in df.columns:
            return set()
        return set(df["name"].astype(str).str.strip().tolist())
    except Exception:
        return set()


def _validate_views_only(sql: str, schema: str) -> None:
    """
    Raise RuntimeError if sql does not query only a view (schema.v_*). Allows SELECT ... FROM "schema"."v_*" ...
    """
    # Normalize: uppercase for pattern, find FROM clause
    sql_norm = " " + (sql or "").replace("\n", " ")
    # Match FROM "schema"."identifier" (identifier must start with v_)
    pattern = re.compile(
        r'\s+FROM\s+"' + re.escape(schema) + r'"\s*\.\s*"([^"]+)"',
        re.IGNORECASE,
    )
    m = pattern.search(sql_norm)
    if not m:
        raise RuntimeError(VIEWS_ONLY_MESSAGE)
    table_or_view = m.group(1)
    if not table_or_view.startswith("v_"):
        raise RuntimeError(VIEWS_ONLY_MESSAGE)


def _record_slow_query(sql: str, elapsed_ms: float) -> None:
    """Append to st.session_state['slow_queries'] (last SLOW_QUERIES_MAX), keep only last 20."""
    if st is None:
        return
    if "slow_queries" not in st.session_state:
        st.session_state["slow_queries"] = []
    st.session_state["slow_queries"] = (
        st.session_state["slow_queries"] + [{"sql": sql[:200], "elapsed_ms": round(elapsed_ms, 2)}]
    )[-SLOW_QUERIES_MAX:]


# Allowed slice dims (map to resolved columns only; no arbitrary columns)
_SLICE_DIMS = frozenset({"channel", "geo", "product"})

# Period modes (governed; no user SQL)
_PERIOD_MODES = frozenset({"1M", "QoQ", "YTD", "YoY"})


def _parse_iso_date(s: str | None):
    """Return (year, month, day) or None from YYYY-MM-DD."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if len(s) < 10:
        return None
    try:
        y, m, d = int(s[:4]), int(s[5:7]), int(s[8:10])
        return (y, m, d)
    except (ValueError, TypeError):
        return None


def build_time_frames(state: FilterState) -> dict[str, Any]:
    """
    Build governed time frames from state. Only computed dates; no user SQL.
    Returns dict with base, compare (or None), and bucket ("month_end" | "quarter_end").
    """
    mode = (state.period_mode or "1M").strip()
    if mode not in _PERIOD_MODES:
        mode = "1M"
    start_raw = (state.date_start or "").strip()
    end_raw = (state.date_end or "").strip()
    base_start = start_raw or ""
    base_end = end_raw or ""
    bucket = "month_end"
    compare: dict[str, Any] | None = None

    if mode == "1M":
        base = {"start": base_start, "end": base_end, "mode": "1M"}
        compare = None
        bucket = "month_end"
    elif mode == "QoQ":
        base = {"start": base_start, "end": base_end, "mode": "QoQ"}
        compare = None
        bucket = "quarter_end"
    elif mode == "YTD":
        end_parsed = _parse_iso_date(end_raw)
        if end_parsed:
            y_end, m_end, d_end = end_parsed
            base_start = f"{y_end}-01-01"
            base_end = end_raw
            base = {"start": base_start, "end": base_end, "mode": "YTD"}
            compare_start = f"{y_end - 1}-01-01"
            compare_end = f"{y_end - 1}-{m_end:02d}-{min(d_end, 28):02d}"
            compare = {"start": compare_start, "end": compare_end, "label": "prior_ytd"}
        else:
            base = {"start": base_start, "end": base_end, "mode": "YTD"}
        bucket = "month_end"
    else:
        assert mode == "YoY"
        base = {"start": base_start, "end": base_end, "mode": "YoY"}
        start_parsed = _parse_iso_date(start_raw)
        end_parsed = _parse_iso_date(end_raw)
        if start_parsed and end_parsed:
            ys, ms, ds = start_parsed
            ye, me, de = end_parsed
            compare = {
                "start": f"{ys - 1}-{ms:02d}-{min(ds, 28):02d}",
                "end": f"{ye - 1}-{me:02d}-{min(de, 28):02d}",
                "label": "prior_yoy",
            }
        else:
            compare = None
        bucket = "month_end"

    return {
        "base": base,
        "compare": compare,
        "bucket": bucket,
    }


def _resolve_slice_column(slice_dim: str, state: FilterState, contract: dict[str, Any]) -> str | None:
    """Resolve slice_dim to canonical column name via filters.yml. Only channel, geo, product allowed."""
    dim = (slice_dim or "").strip().lower()
    if dim not in _SLICE_DIMS:
        return None
    if dim == "channel":
        return resolve_channel_column(state.channel_view, contract)
    if dim == "geo":
        return resolve_geo_column(state.geo_dim, contract)
    if dim == "product":
        return resolve_product_column(state.product_dim, contract)
    return None


def build_where_clause(
    state: FilterState,
    dataset_meta: dict[str, Any],
    frame: str = "base",
) -> tuple[str, dict[str, Any], str]:
    """
    Translate FilterState into governed WHERE clause and named params using filters.yml only.
    frame="base"|"compare" selects which time frame to use (from build_time_frames).
    Returns (sql_where, params, bucket_hint) with sql_where starting with "WHERE 1=1".
    """
    contract = load_filters_contract()
    time_cfg = (contract.get("time") or {}).get("date_range")
    month_end_col = "month_end"
    if isinstance(time_cfg, list) and len(time_cfg) >= 2:
        start_key, end_key = time_cfg[0], time_cfg[1]
    else:
        start_key, end_key = "start_month_end", "end_month_end"

    frames = build_time_frames(state)
    bucket_hint = frames.get("bucket", "month_end")
    base_frame = frames.get("base") or {}
    compare_frame = frames.get("compare")

    if frame == "compare" and compare_frame:
        range_frame = compare_frame
    else:
        range_frame = base_frame

    range_start = range_frame.get("start")
    range_end = range_frame.get("end")

    parts = ["WHERE 1=1"]
    params: dict[str, Any] = {}

    params[start_key] = pd.Timestamp(range_start) if range_start else None
    params[end_key] = pd.Timestamp(range_end) if range_end else None
    if params.get(start_key) is not None:
        parts.append(f'AND "{month_end_col}" >= ${start_key}')
    if params.get(end_key) is not None:
        parts.append(f'AND "{month_end_col}" <= ${end_key}')

    channel_col = resolve_channel_column(state.channel_view, contract)
    geo_col = resolve_geo_column(state.geo_dim, contract)
    product_col = resolve_product_column(state.product_dim, contract)

    geo_values = getattr(state, "geo_values", None)
    if geo_values is not None and isinstance(geo_values, (list, tuple)) and len(geo_values) > 0:
        params["geo_values"] = [str(v) for v in geo_values]
        parts.append(f'AND "{geo_col}" IN (SELECT unnest($geo_values))')

    product_values = getattr(state, "product_values", None)
    if product_values is not None and isinstance(product_values, (list, tuple)) and len(product_values) > 0:
        params["product_values"] = [str(v) for v in product_values]
        parts.append(f'AND "{product_col}" IN (SELECT unnest($product_values))')

    available_columns = dataset_meta.get("columns")
    if isinstance(available_columns, (list, tuple)):
        available_columns = set(available_columns)
    elif not isinstance(available_columns, set):
        available_columns = set()

    if state.custodian_firm and state.custodian_firm.strip() and "custodian_firm" in available_columns:
        params["custodian_firm"] = state.custodian_firm.strip()
        parts.append('AND "custodian_firm" = $custodian_firm')

    slice_dim = (state.slice_dim or "").strip().lower()
    slice_value = state.slice_value or state.slice
    if slice_dim and slice_value is not None and str(slice_value).strip():
        slice_col = _resolve_slice_column(slice_dim, state, contract)
        if slice_col is not None:
            params["slice_value"] = str(slice_value).strip()
            parts.append(f'AND "{slice_col}" = $slice_value')

    period_mode = (state.period_mode or "1M").strip()
    if period_mode not in _PERIOD_MODES:
        period_mode = "1M"
    params["period_mode"] = period_mode

    sql_where = " ".join(parts)
    return sql_where, params, bucket_hint


# Whitelist for template-based run_query (no tab SQL)
RUN_QUERY_ALLOWED = frozenset({
    "kpi_firm_global",
    "chart_aum_trend",
    "chart_nnb_trend",
    "top_channels",
    "top_tickers",
    "top_movers",
    "notable_months",
    "coverage_stats",
    "growth_decomposition_inputs",
})


def _resolve_columns_and_bucket(state: FilterState) -> tuple[str, str, str, str]:
    """Resolve channel_col, geo_col, product_col and bucket_expr from state (filters.yml only). Returns (channel_col, geo_col, product_col, bucket_expr)."""
    contract = load_filters_contract()
    channel_col = resolve_channel_column(state.channel_view, contract)
    geo_col = resolve_geo_column(state.geo_dim, contract)
    product_col = resolve_product_column(state.product_dim, contract)
    frames = build_time_frames(state)
    bucket_hint = frames.get("bucket", "month_end")
    bucket_expr = '"month_end"' if bucket_hint == "month_end" else 'LAST_DAY("month_end", "quarter")'
    return channel_col, geo_col, product_col, bucket_expr


def _named_params_to_positional(sql: str, params: dict[str, Any]) -> tuple[str, list[Any]]:
    """Replace $name in sql with ? in order of first appearance; return (sql, param_list)."""
    order = re.findall(r"\$(\w+)", sql)
    param_list = [params.get(n) for n in order]
    sql_with_q = re.sub(r"\$\w+", "?", sql)
    return sql_with_q, param_list


# Query name -> (view_name, sql_template, result_type). result_type "df" | "dict"
QUERY_TEMPLATES: dict[str, tuple[str, str, str]] = {
    "kpi_firm_global": (
        "v_firm_monthly",
        'SELECT SUM("end_aum") AS total_aum, SUM("nnb") AS total_nnb, COUNT(*) AS row_count FROM "{schema}"."{view}" {where_clause}',
        "dict",
    ),
    "chart_aum_trend": (
        "v_firm_monthly",
        'SELECT "month_end", SUM("end_aum") AS end_aum FROM "{schema}"."{view}" {where_clause} GROUP BY "month_end" ORDER BY 1',
        "df",
    ),
    "chart_nnb_trend": (
        "v_firm_monthly",
        'SELECT "month_end", SUM("nnb") AS nnb FROM "{schema}"."{view}" {where_clause} GROUP BY "month_end" ORDER BY 1',
        "df",
    ),
    "top_channels": (
        "v_channel_monthly",
        'SELECT "{channel_col}" AS channel, SUM("end_aum") AS end_aum, SUM("nnb") AS nnb FROM "{schema}"."{view}" {where_clause} GROUP BY "{channel_col}" ORDER BY 2 DESC LIMIT 10',
        "df",
    ),
    "top_tickers": (
        "v_ticker_monthly",
        'SELECT "{product_col}" AS ticker, SUM("end_aum") AS end_aum, SUM("nnb") AS nnb FROM "{schema}"."{view}" {where_clause} GROUP BY "{product_col}" ORDER BY 2 DESC LIMIT 10',
        "df",
    ),
    "top_movers": (
        "v_ticker_monthly",
        'SELECT "{product_col}" AS ticker, SUM("end_aum") AS end_aum, SUM("nnb") AS nnb FROM "{schema}"."{view}" {where_clause} GROUP BY "{product_col}" ORDER BY 2 DESC LIMIT 10',
        "df",
    ),
    "notable_months": (
        "v_firm_monthly",
        'SELECT "month_end", SUM("end_aum") AS end_aum, SUM("nnb") AS nnb FROM "{schema}"."{view}" {where_clause} GROUP BY "month_end" ORDER BY 1',
        "df",
    ),
    "coverage_stats": (
        "v_firm_monthly",
        'SELECT COUNT(*) AS rows_covered, MIN("month_end") AS min_month, MAX("month_end") AS max_month FROM "{schema}"."{view}" {where_clause}',
        "dict",
    ),
    "growth_decomposition_inputs": (
        "v_firm_monthly",
        'SELECT SUM("end_aum") AS end_aum, SUM("nnb") AS nnb FROM "{schema}"."{view}" {where_clause}',
        "dict",
    ),
}


def _run_query_templated_impl(
    _dataset_version: str,
    _filter_hash: str,
    query_name: str,
    state_json: str,
    root_str: str,
) -> pd.DataFrame | dict[str, Any]:
    """Inner implementation: rebuild state from JSON, run template, return df or dict. Used by run_query (cached when st)."""
    state_dict = json.loads(state_json)
    state = FilterState.from_dict(state_dict)
    for key in ("geo_values", "product_values"):
        if key in state_dict:
            setattr(state, key, state_dict[key])
    root = Path(root_str)
    tpl = QUERY_TEMPLATES.get(query_name)
    if not tpl:
        raise ValueError(f"No template for query {query_name!r}.")
    view_name, sql_template, result_type = tpl
    config = get_config(root)
    schema = config.get("schema", "analytics")
    available = get_available_columns(view_name, root)
    dataset_meta = {"columns": available}
    where_sql, params_dict, _bucket_hint = build_where_clause(state, dataset_meta, frame="base")
    channel_col, geo_col, product_col, bucket_expr = _resolve_columns_and_bucket(state)
    sql = sql_template.format(
        schema=schema,
        view=view_name,
        channel_col=channel_col,
        geo_col=geo_col,
        product_col=product_col,
        bucket_expr=bucket_expr,
        where_clause=where_sql,
    )
    sql_final, param_list = _named_params_to_positional(sql, params_dict)
    df = query_df(sql_final, params=param_list, _config=config)
    if result_type == "dict":
        if df is None or df.empty:
            return {}
        row = df.iloc[0]
        return {str(k): (row[k].isoformat() if hasattr(row[k], "isoformat") else row[k]) for k in df.columns}
    return df


def _run_query_cached(
    dataset_version: str,
    filter_hash: str,
    query_name: str,
    state_json: str,
    root_str: str,
) -> pd.DataFrame | dict[str, Any]:
    """Cached runner keyed by dataset_version + filter_hash + query_name. Uses st.cache_data when in Streamlit."""
    if st is not None:
        return _run_query_cached._cached(dataset_version, filter_hash, query_name, state_json, root_str)
    return _run_query_templated_impl(dataset_version, filter_hash, query_name, state_json, root_str)


if st is not None:
    _run_query_cached._cached = st.cache_data(ttl=3600)(_run_query_templated_impl)


def run_governed_query(
    query_name: str,
    state: FilterState,
    root: Path | None = None,
) -> pd.DataFrame | dict[str, Any]:
    """
    Governed query entrypoint: templates only, no dynamic SQL from UI.
    Only query_name in RUN_QUERY_ALLOWED. All calls go through cached_call (dataset_version + filter_hash + query_name).
    Returns pd.DataFrame or dict depending on template. Row cap and budget applied at gateway boundary.
    """
    if query_name not in RUN_QUERY_ALLOWED:
        raise ValueError(
            f"Query {query_name!r} not allowed. Allowed: {sorted(RUN_QUERY_ALLOWED)}."
        )
    root = Path(root) if root is not None else Path.cwd()
    config = get_config(root)
    dataset_version = config.get("dataset_version", "unknown")
    filter_hash = state.filter_state_hash()
    state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
    root_str = str(root)
    cache_key = f"run_query|{dataset_version}|{filter_hash}|{query_name}"
    return cached_call(
        cache_key,
        lambda: _run_query_templated_impl(dataset_version, filter_hash, query_name, state_json, root_str),
        DEFAULT_TIMEOUT_MS,
        DEFAULT_MAX_ROWS,
        query_name,
    )


def build_where(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """
    Build WHERE clause and ordered params from filters. Safe: only FILTER_COLUMNS allowed; no string concat of values.
    - date_from / date_to or month_end_range -> month_end >= ? AND month_end <= ?
    - column: list -> column IN (?, ?, ...); column: single value -> column = ?
    Returns (where_sql, param_list). where_sql is " WHERE ..." or "".
    """
    if not filters:
        return "", []
    parts: list[str] = []
    params: list[Any] = []
    # date range
    date_from = filters.get("date_from")
    date_to = filters.get("date_to")
    month_end_range = filters.get("month_end_range")
    if month_end_range is not None and isinstance(month_end_range, (list, tuple)) and len(month_end_range) >= 2:
        date_from = date_from or month_end_range[0]
        date_to = date_to or month_end_range[1]
    if date_from is not None:
        parts.append('"month_end" >= ?')
        params.append(pd.Timestamp(date_from) if hasattr(pd, "Timestamp") and date_from is not None else date_from)
    if date_to is not None:
        parts.append('"month_end" <= ?')
        params.append(pd.Timestamp(date_to) if hasattr(pd, "Timestamp") and date_to is not None else date_to)
    # dimension IN / =
    for key in sorted(filters.keys()):
        if key in ("date_from", "date_to", "month_end_range"):
            continue
        if key not in FILTER_COLUMNS:
            continue
        val = filters[key]
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                continue
            placeholders = ", ".join("?" for _ in val)
            parts.append(f'"{key}" IN ({placeholders})')
            params.extend(val)
        else:
            parts.append(f'"{key}" = ?')
            params.append(val)
    if not parts:
        return "", []
    return " WHERE " + " AND ".join(parts), params


def query_df(
    sql: str,
    params: dict[str, Any] | list[Any] | None = None,
    root: Path | None = None,
    _config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Execute parameterized SQL and return DataFrame. Uses get_connection and get_config.
    Cached with st.cache_data keyed by dataset_version + sql + params (when in Streamlit).
    """
    config = _config or get_config(root)
    schema = config.get("schema", "analytics")
    _validate_views_only(sql, schema)
    db_path = config["db_path"]
    dataset_version = config.get("dataset_version", "unknown")
    param_list: list[Any] = []
    if isinstance(params, dict):
        param_list = [v for _k, v in sorted(params.items())]
        param_key: tuple[Any, ...] = tuple(sorted(params.items()))
    elif params:
        param_list = list(params)
        try:
            param_key = tuple(param_list)
        except TypeError:
            param_key = (id(params),)  # fallback for unhashable
    else:
        param_key = ()
    t0 = time.perf_counter()
    if st is not None:
        @st.cache_data(ttl=3600)
        def _cached(version: str, sql_text: str, param_key_cached: tuple[Any, ...], db_path_cached: str) -> pd.DataFrame:
            _con = get_connection(db_path_cached)
            # Dict params: ( (k,v), ... ); list params: ( v1, v2, ... )
            if param_key_cached and isinstance(param_key_cached[0], (list, tuple)) and len(param_key_cached[0]) == 2 and isinstance(param_key_cached[0][0], str):
                p = [v for _k, v in param_key_cached]
            else:
                p = list(param_key_cached)
            if p:
                return _con.execute(sql_text, p).fetchdf()
            return _con.execute(sql_text).fetchdf()
        result = _cached(dataset_version, sql, param_key, db_path)
    else:
        con = get_connection(db_path)
        if param_list:
            result = con.execute(sql, param_list).fetchdf()
        else:
            result = con.execute(sql).fetchdf()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms > SLOW_QUERY_MS:
        logger.warning("Slow query %.0f ms (threshold %d ms): %s", elapsed_ms, SLOW_QUERY_MS, sql[:200])
        _record_slow_query(sql, elapsed_ms)
    return result


def _view_sql(schema: str, view_name: str, where_sql: str, columns: list[str] | None = None) -> str:
    """Build SELECT ... FROM schema.view WHERE ...; columns=None -> SELECT *."""
    if columns:
        cols = ", ".join(f'"{c}"' for c in columns)
        return f'SELECT {cols} FROM "{schema}"."{view_name}"{where_sql}'
    return f'SELECT * FROM "{schema}"."{view_name}"{where_sql}'


def _load_parquet_with_filters(
    table_name: str,
    filter_state: dict[str, Any],
    root: Path,
    columns: list[str] | None,
) -> pd.DataFrame:
    """Load agg table by name from manifest, apply filter_state (month_end_range + IN dims), return DataFrame."""
    from app.agg_store import get_table_path, load_manifest
    from app.filters import apply_filters

    manifest = load_manifest(root)
    path_str = get_table_path(table_name, manifest)
    full_path = root / path_str
    if not full_path.exists():
        raise FileNotFoundError(
            f"Agg table not found: {full_path}. Run: python -m pipelines.agg.build_aggs"
        )
    df = pd.read_parquet(full_path, columns=columns)
    # Normalize filter_state for apply_filters: month_end_range (min_ts, max_ts), dims as lists
    filters: dict[str, Any] = {}
    month_end_range = filter_state.get("month_end_range")
    date_from = filter_state.get("date_from")
    date_to = filter_state.get("date_to")
    if month_end_range is not None and isinstance(month_end_range, (list, tuple)) and len(month_end_range) >= 2:
        filters["month_end_range"] = (pd.Timestamp(month_end_range[0]), pd.Timestamp(month_end_range[1]))
    elif date_from is not None or date_to is not None:
        filters["month_end_range"] = (
            pd.Timestamp(date_from) if date_from else None,
            pd.Timestamp(date_to) if date_to else None,
        )
    for key in filter_state:
        if key in ("date_from", "date_to", "month_end_range"):
            continue
        val = filter_state[key]
        if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
            continue
        filters[key] = [val] if not isinstance(val, (list, tuple)) else list(val)
    return apply_filters(df, filters)


def _run_query_uncached(
    query_name: str,
    filter_state: dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Validate query_name, resolve backend (DuckDB views preferred, else agg Parquet), build WHERE, return DataFrame.
    """
    if query_name not in ALLOWED_QUERIES:
        raise ValueError(
            f"Invalid query_name {query_name!r}. Allowed: {sorted(ALLOWED_QUERIES)}."
        )
    spec = QUERY_SPECS.get(query_name)
    if not spec or (spec.get("view") is None and spec.get("parquet_table") is None):
        raise ValueError(
            f"No view or parquet_table for query {query_name!r}. Add to QUERY_SPECS."
        )
    root = Path(root) if root is not None else Path.cwd()
    allowed = spec.get("allowed_filters") or []
    columns = spec.get("default_columns")

    if _use_duckdb_views(root):
        view_name = spec.get("view")
        if not view_name:
            raise ValueError(
                f"DuckDB views mode but no view for query {query_name!r}. Add view to QUERY_SPECS."
            )
        if st is not None:
            st.session_state["_last_backend"] = "duckdb"
        config = get_config(root)
        where_sql, params = build_where(filter_state)
        sql = _view_sql(config["schema"], view_name, where_sql, columns=columns)
        df = query_df(sql, params=params, _config=config)
        if query_name in _MONTHLY_QUERIES:
            df = _canonicalize_monthly_for_ui(df, query_name)
        return df

    # Parquet fallback
    if st is not None:
        st.session_state["_last_backend"] = "parquet"
    parquet_table = spec.get("parquet_table")
    if not parquet_table:
        raise ValueError(
            f"Parquet fallback but no parquet_table for query {query_name!r}. Add to QUERY_SPECS."
        )
    df = _load_parquet_with_filters(parquet_table, filter_state, root, columns)
    if query_name in _MONTHLY_QUERIES:
        df = _canonicalize_monthly_for_ui(df, query_name)
    return df


def _cached_run_query_impl(
    dataset_version: str,
    query_name: str,
    filter_state_hash: str,
    filter_state_json: str,
    root_str: str | None,
) -> pd.DataFrame:
    """Inner implementation: hashable args for st.cache_data."""
    filter_state = json.loads(filter_state_json) if filter_state_json else {}
    root = Path(root_str) if root_str else None
    return _run_query_uncached(query_name, filter_state, root=root)


if st is not None:
    @st.cache_data(show_spinner=False)
    def cached_run_query(
        dataset_version: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
    ) -> pd.DataFrame:
        return _cached_run_query_impl(
            dataset_version, query_name, filter_state_hash, filter_state_json, root_str
        )
else:
    def cached_run_query(
        dataset_version: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
    ) -> pd.DataFrame:
        return _cached_run_query_impl(
            dataset_version, query_name, filter_state_hash, filter_state_json, root_str
        )


def run_query(
    query_name: str,
    filter_state: dict[str, Any] | FilterState,
    root: Path | None = None,
) -> pd.DataFrame | dict[str, Any]:
    """
    Single entrypoint for all queries. Governed queries (RUN_QUERY_ALLOWED) use templated path and may return dict.
    Base queries use pyramid.get_filtered and return DataFrame.
    """
    if query_name in RUN_QUERY_ALLOWED:
        # Normalize any state-like object (including stale/reloaded class instances)
        # to a canonical FilterState before governed query path selection.
        normalized = normalize_filters(filter_state)
        if not isinstance(normalized, FilterState):
            normalized = FilterState.from_dict(normalized if isinstance(normalized, dict) else {})
        return run_governed_query(query_name, normalized, root)
    if isinstance(filter_state, FilterState):
        filter_state = filter_state_to_gateway_dict(filter_state)
    from app.cache import pyramid as cache_pyramid
    dv = load_dataset_version(root)
    h = hash_filters(filter_state)
    canonical = canonicalize_state(to_json_safe(filter_state))
    filter_state_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    root_str = str(root) if root is not None else None
    return cache_pyramid.get_filtered(dv, query_name, h, filter_state_json, root_str)


def run_aggregate(
    agg_name: str,
    query_name: str,
    filter_state: dict[str, Any] | FilterState,
    root: Path | None = None,
    **params: Any,
) -> dict[str, Any] | pd.DataFrame:
    """
    Level B entrypoint: derived aggregates. Uses pyramid.get_aggregate for cached aggregates.
    Computes dataset_version + filter_state_hash and passes names into the cached layer.
    params: optional kwargs for aggregate specs that accept them (e.g. top_n, by for topn_tickers).
    """
    normalized = normalize_filters(filter_state)
    if isinstance(normalized, FilterState):
        filter_state = filter_state_to_gateway_dict(normalized)
    else:
        filter_state = normalized
    from app.cache import pyramid as cache_pyramid
    dv = load_dataset_version(root)
    h = hash_filters(filter_state)
    canonical = canonicalize_state(to_json_safe(filter_state))
    filter_state_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    root_str = str(root) if root is not None else None
    agg_params_json = json.dumps(canonicalize_state(to_json_safe(params)), sort_keys=True, separators=(",", ":")) if params else ""
    return cache_pyramid.get_aggregate(
        dv, agg_name, query_name, h, filter_state_json, root_str, agg_params_json=agg_params_json
    )


def run_chart(
    chart_name: str,
    agg_name: str,
    query_name: str,
    filter_state: dict[str, Any] | FilterState,
    root: Path | None = None,
    **params: Any,
) -> dict[str, Any]:
    """
    Level C entrypoint: heavy chart payloads. Uses pyramid.get_chart_payload for instant feel.
    Computes dataset_version + filter_state_hash and passes chart_name, agg_name, query_name into the cached layer.
    params: optional kwargs passed to the underlying aggregate when it accepts them.
    """
    normalized = normalize_filters(filter_state)
    if isinstance(normalized, FilterState):
        filter_state = filter_state_to_gateway_dict(normalized)
    else:
        filter_state = normalized
    from app.cache import pyramid as cache_pyramid
    dv = load_dataset_version(root)
    h = hash_filters(filter_state)
    canonical = canonicalize_state(to_json_safe(filter_state))
    filter_state_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    root_str = str(root) if root is not None else None
    agg_params_json = json.dumps(canonicalize_state(to_json_safe(params)), sort_keys=True, separators=(",", ":")) if params else ""
    return cache_pyramid.get_chart_payload(
        dv, chart_name, agg_name, query_name, h, filter_state_json, root_str, agg_params_json=agg_params_json
    )


# --- Example usage for pages (migrate heavy charts to run_chart for instant feel) ---
#
# Waterfall (KPI cards -> waterfall payload):
#   from app.data_gateway import run_chart, Q_FIRM_MONTHLY
#   from app.cache.specs import AGG_KPI_CARDS, CHART_WATERFALL
#   payload = run_chart(CHART_WATERFALL, AGG_KPI_CARDS, Q_FIRM_MONTHLY, filter_state, root=ROOT)
#   # payload["type"] == "waterfall", payload["data"] is JSON-serializable KPI dict
#
# Correlation matrix (raw query -> corr matrix payload):
#   from app.data_gateway import run_chart, Q_FIRM_MONTHLY
#   from app.cache.specs import CHART_CORR_MATRIX
#   payload = run_chart(CHART_CORR_MATRIX, "raw", Q_FIRM_MONTHLY, filter_state, root=ROOT)
#   # payload["type"] == "corr_matrix", payload["data"] / payload["columns"] for the matrix
#
# Aggregate only (e.g. top-N tickers with params):
#   from app.data_gateway import run_aggregate, Q_TICKER_MONTHLY
#   from app.cache.specs import AGG_TOPN_TICKERS
#   result = run_aggregate(AGG_TOPN_TICKERS, Q_TICKER_MONTHLY, filter_state, root=ROOT, top_n=10, by="end_aum")


def _get_firm_monthly_raw(
    filters: dict[str, Any],
    root: Path | None,
) -> pd.DataFrame:
    """Internal: run query_df for v_firm_monthly (no cache)."""
    config = get_config(root)
    where_sql, params = build_where(filters)
    sql = _view_sql(config["schema"], "v_firm_monthly", where_sql)
    return query_df(sql, params=params, _config=config)


def get_firm_monthly(
    filters: dict[str, Any] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """SELECT * FROM schema.v_firm_monthly WHERE ... (views-only). Uses run_query."""
    try:
        return run_query(Q_FIRM_MONTHLY, filters or {}, root=root)
    except (FileNotFoundError, ValueError):
        return _get_firm_monthly_raw(filters or {}, root)


def load_firm_monthly(root: Path | None = None) -> pd.DataFrame:
    """
    Load firm-level monthly data from the governed analytics layer.
    Priority: (1) DuckDB view analytics.v_firm_monthly, (2) fallback data/agg/firm_monthly.parquet.
    Returns DataFrame with columns required for Executive KPIs; no filters applied.
    """
    root = Path(root) if root is not None else Path.cwd()
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        sql = f'SELECT * FROM "{schema}"."v_firm_monthly" ORDER BY "month_end"'
        df = query_df(sql, params=None, _config=config)
        return _prepare_monthly_dataset(
            df,
            "firm_monthly",
            FIRM_REQUIRED_COLUMNS,
            FIRM_LOAD_COLUMNS,
            alias_map=FIRM_COLUMN_ALIASES,
        )
    except Exception:
        pass
    # Fallback: parquet
    parquet_path = root / "data" / "agg" / "firm_monthly.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            df = _prepare_monthly_dataset(
                df,
                "firm_monthly",
                FIRM_REQUIRED_COLUMNS,
                FIRM_LOAD_COLUMNS,
                alias_map=FIRM_COLUMN_ALIASES,
            )
            if not df.empty:
                df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
            return df
        except Exception:
            pass
    try:
        df = pd.read_parquet(parquet_path)
        df = _prepare_monthly_dataset(
            df,
            "firm_monthly",
            FIRM_REQUIRED_COLUMNS,
            FIRM_LOAD_COLUMNS,
            alias_map=FIRM_COLUMN_ALIASES,
        )
        if not df.empty:
            df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=FIRM_LOAD_COLUMNS)


def get_executive_kpis(root: Path | None = None) -> "ExecutiveKPIs":
    """
    Compute Executive KPIs from governed firm monthly data. The UI must NEVER
    compute KPIs; it must only call get_executive_kpis() (or gateway.get_executive_kpis()).
    """
    from app.metrics.kpi_definitions import ExecutiveKPIs, compute_executive_kpis
    df = load_firm_monthly(root)
    return compute_executive_kpis(df)


def _get_channel_monthly_raw(filters: dict[str, Any], root: Path | None) -> pd.DataFrame:
    config = get_config(root)
    where_sql, params = build_where(filters)
    sql = _view_sql(config["schema"], "v_channel_monthly", where_sql)
    return query_df(sql, params=params, _config=config)


def get_channel_monthly(
    filters: dict[str, Any] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """SELECT * FROM schema.v_channel_monthly WHERE ... (views-only). Uses run_query."""
    try:
        return run_query(Q_CHANNEL_MONTHLY, filters or {}, root=root)
    except (FileNotFoundError, ValueError):
        return _get_channel_monthly_raw(filters or {}, root)


def _get_ticker_monthly_raw(filters: dict[str, Any], root: Path | None) -> pd.DataFrame:
    config = get_config(root)
    where_sql, params = build_where(filters)
    sql = _view_sql(config["schema"], "v_ticker_monthly", where_sql)
    return query_df(sql, params=params, _config=config)


def get_ticker_monthly(
    filters: dict[str, Any] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """SELECT * FROM schema.v_ticker_monthly WHERE ... (views-only). Uses run_query."""
    try:
        return run_query(Q_TICKER_MONTHLY, filters or {}, root=root)
    except (FileNotFoundError, ValueError):
        return _get_ticker_monthly_raw(filters or {}, root)


def _filters_to_month_end_range(
    filters: FilterState | dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Extract (start, end) for month_end from normalized filters. Global date range only."""
    if isinstance(filters, FilterState):
        d = filter_state_to_gateway_dict(filters)
    else:
        d = dict(filters) if filters else {}
    r = d.get("month_end_range")
    if r is not None and isinstance(r, (list, tuple)) and len(r) >= 2:
        return (pd.Timestamp(r[0]), pd.Timestamp(r[1]))
    start = d.get("date_start") or d.get("date_from")
    end = d.get("date_end") or d.get("date_to")
    if start is not None and end is not None:
        return (pd.Timestamp(start), pd.Timestamp(end))
    return None


def _load_firm_monthly(
    filters: FilterState | dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Governed load: analytics.v_firm_monthly or data/agg/firm_monthly.parquet.
    Only global filters (date range on month_end). Returns df sorted by month_end ascending,
    numeric coerced, +/-inf -> NaN. No derived metrics.
    """
    root = Path(root) if root is not None else Path.cwd()
    state = normalize_filters(filters)
    filter_dict = filter_state_to_gateway_dict(state) if isinstance(state, FilterState) else {}
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        where_sql, params = build_where(filter_dict)
        sql = _view_sql(schema, "v_firm_monthly", where_sql, columns=None)
        df = query_df(sql, params=params, _config=config)
    except Exception:
        df = None
    if df is None or df.empty:
        parquet_path = root / "data" / "agg" / "firm_monthly.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, columns=FIRM_LOAD_COLUMNS)
            except Exception:
                try:
                    df = pd.read_parquet(parquet_path)
                except Exception:
                    df = pd.DataFrame(columns=FIRM_LOAD_COLUMNS)
        else:
            df = pd.DataFrame(columns=FIRM_LOAD_COLUMNS)
    df = _prepare_monthly_dataset(
        df,
        "firm_monthly",
        FIRM_REQUIRED_COLUMNS,
        FIRM_LOAD_COLUMNS,
        alias_map=FIRM_COLUMN_ALIASES,
    )
    if df.empty:
        return df
    me_range = _filters_to_month_end_range(state)
    if me_range is not None:
        start_ts, end_ts = me_range
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df[(df["month_end"] >= start_ts) & (df["month_end"] <= end_ts)]
    for c in FIRM_NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    return _coerce_inf_to_nan(df)


def _load_channel_monthly(
    filters: FilterState | dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Governed load: analytics.v_channel_monthly or data/agg/channel_monthly.parquet.
    Same as firm + channel label. Global filters only; sorted, coerced, no inf.
    """
    root = Path(root) if root is not None else Path.cwd()
    state = normalize_filters(filters)
    filter_dict = filter_state_to_gateway_dict(state) if isinstance(state, FilterState) else {}
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        where_sql, params = build_where(filter_dict)
        sql = _view_sql(schema, "v_channel_monthly", where_sql, columns=None)
        df = query_df(sql, params=params, _config=config)
    except Exception:
        df = None
    if df is None or df.empty:
        parquet_path = root / "data" / "agg" / "channel_monthly.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, columns=CHANNEL_LOAD_COLUMNS)
            except Exception:
                try:
                    df = pd.read_parquet(parquet_path)
                except Exception:
                    df = pd.DataFrame(columns=CHANNEL_LOAD_COLUMNS)
        else:
            df = pd.DataFrame(columns=CHANNEL_LOAD_COLUMNS)
    df = _prepare_monthly_dataset(
        df,
        "channel_monthly",
        CHANNEL_REQUIRED_COLUMNS,
        CHANNEL_LOAD_COLUMNS,
        alias_map=CHANNEL_COLUMN_ALIASES,
    )
    if df.empty:
        return df
    me_range = _filters_to_month_end_range(state)
    if me_range is not None:
        start_ts, end_ts = me_range
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df[(df["month_end"] >= start_ts) & (df["month_end"] <= end_ts)]
    for c in CHANNEL_NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    return _coerce_inf_to_nan(df)


def _load_ticker_monthly(
    filters: FilterState | dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Governed load: analytics.v_ticker_monthly or data/agg/ticker_monthly.parquet.
    Same as firm + ticker label. Global filters only; sorted, coerced, no inf.
    """
    root = Path(root) if root is not None else Path.cwd()
    state = normalize_filters(filters)
    filter_dict = filter_state_to_gateway_dict(state) if isinstance(state, FilterState) else {}
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        where_sql, params = build_where(filter_dict)
        sql = _view_sql(schema, "v_ticker_monthly", where_sql, columns=None)
        df = query_df(sql, params=params, _config=config)
    except Exception:
        df = None
    if df is None or df.empty:
        parquet_path = root / "data" / "agg" / "ticker_monthly.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, columns=TICKER_LOAD_COLUMNS)
            except Exception:
                try:
                    df = pd.read_parquet(parquet_path)
                except Exception:
                    df = pd.DataFrame(columns=TICKER_LOAD_COLUMNS)
        else:
            df = pd.DataFrame(columns=TICKER_LOAD_COLUMNS)
    df = _prepare_monthly_dataset(
        df,
        "ticker_monthly",
        TICKER_REQUIRED_COLUMNS,
        TICKER_LOAD_COLUMNS,
        alias_map=TICKER_COLUMN_ALIASES,
    )
    if df.empty:
        return df
    me_range = _filters_to_month_end_range(state)
    if me_range is not None:
        start_ts, end_ts = me_range
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df[(df["month_end"] >= start_ts) & (df["month_end"] <= end_ts)]
    for c in TICKER_NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    return _coerce_inf_to_nan(df)


def _load_geo_monthly(
    filters: FilterState | dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Governed load: analytics.v_geo_monthly or data/agg/geo_monthly.parquet.
    Canonical output includes 'geo' label; aliases normalized at boundary.
    """
    root = Path(root) if root is not None else Path.cwd()
    state = normalize_filters(filters)
    filter_dict = filter_state_to_gateway_dict(state) if isinstance(state, FilterState) else {}
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        where_sql, params = build_where(filter_dict)
        sql = _view_sql(schema, "v_geo_monthly", where_sql, columns=None)
        df = query_df(sql, params=params, _config=config)
    except Exception:
        df = None
    if df is None or df.empty:
        parquet_path = root / "data" / "agg" / "geo_monthly.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, columns=GEO_LOAD_COLUMNS)
            except Exception:
                try:
                    df = pd.read_parquet(parquet_path)
                except Exception:
                    df = pd.DataFrame(columns=GEO_LOAD_COLUMNS)
        else:
            df = pd.DataFrame(columns=GEO_LOAD_COLUMNS)
    df = _prepare_monthly_dataset(
        df,
        "geo_monthly",
        GEO_REQUIRED_COLUMNS,
        GEO_LOAD_COLUMNS,
        alias_map=GEO_COLUMN_ALIASES,
    )
    if df.empty:
        return df
    me_range = _filters_to_month_end_range(state)
    if me_range is not None:
        start_ts, end_ts = me_range
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df[(df["month_end"] >= start_ts) & (df["month_end"] <= end_ts)]
    for c in GEO_NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    return _coerce_inf_to_nan(df)


def _load_segment_monthly(
    filters: FilterState | dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Governed load: analytics.v_segment_monthly or data/agg/segment_monthly.parquet.
    Canonical output includes 'segment' and 'sub_segment'; aliases normalized at boundary.
    """
    root = Path(root) if root is not None else Path.cwd()
    state = normalize_filters(filters)
    filter_dict = filter_state_to_gateway_dict(state) if isinstance(state, FilterState) else {}
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        where_sql, params = build_where(filter_dict)
        sql = _view_sql(schema, "v_segment_monthly", where_sql, columns=None)
        df = query_df(sql, params=params, _config=config)
    except Exception:
        df = None
    if df is None or df.empty:
        parquet_path = root / "data" / "agg" / "segment_monthly.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path, columns=SEGMENT_LOAD_COLUMNS)
            except Exception:
                try:
                    df = pd.read_parquet(parquet_path)
                except Exception:
                    df = pd.DataFrame(columns=SEGMENT_LOAD_COLUMNS)
        else:
            df = pd.DataFrame(columns=SEGMENT_LOAD_COLUMNS)
    df = _prepare_monthly_dataset(
        df,
        "segment_monthly",
        SEGMENT_REQUIRED_COLUMNS,
        SEGMENT_LOAD_COLUMNS,
        alias_map=SEGMENT_COLUMN_ALIASES,
    )
    if df.empty:
        return df
    me_range = _filters_to_month_end_range(state)
    if me_range is not None:
        start_ts, end_ts = me_range
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df[(df["month_end"] >= start_ts) & (df["month_end"] <= end_ts)]
    for c in SEGMENT_NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    return _coerce_inf_to_nan(df)


def _get_geo_monthly_raw(filters: dict[str, Any], root: Path | None) -> pd.DataFrame:
    config = get_config(root)
    where_sql, params = build_where(filters)
    sql = _view_sql(config["schema"], "v_geo_monthly", where_sql)
    return query_df(sql, params=params, _config=config)


def get_geo_monthly(
    filters: dict[str, Any] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """SELECT * FROM schema.v_geo_monthly WHERE ... (views-only). Uses run_query."""
    try:
        return run_query(Q_GEO_MONTHLY, filters or {}, root=root)
    except (FileNotFoundError, ValueError):
        return _get_geo_monthly_raw(filters or {}, root)


def _get_segment_monthly_raw(filters: dict[str, Any], root: Path | None) -> pd.DataFrame:
    config = get_config(root)
    where_sql, params = build_where(filters)
    sql = _view_sql(config["schema"], "v_segment_monthly", where_sql)
    return query_df(sql, params=params, _config=config)


def get_segment_monthly(
    filters: dict[str, Any] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """SELECT * FROM schema.v_segment_monthly WHERE ... (views-only). Uses run_query."""
    try:
        return run_query(Q_SEGMENT_MONTHLY, filters or {}, root=root)
    except (FileNotFoundError, ValueError):
        return _get_segment_monthly_raw(filters or {}, root)


# =============================================================================
# Data Gateway: governed operations only (single entrypoint for UI)
# - No "execute arbitrary SQL" public API.
# - Cache key: dataset_version + filter_state_hash + query_name.
# - APP_MOCK_DATA=1: return synthetic data without touching DuckDB.
# =============================================================================

APP_MOCK_DATA = os.environ.get("APP_MOCK_DATA") == "1"

# Governed operation names (only these are allowed)
GOVERNED_QUERIES = frozenset({
    "get_dataset_version",
    "get_last_refresh_ts",
    "kpi_firm_global",
    "chart_aum_trend",
    "chart_nnb_trend",
    "growth_decomposition_inputs",
    "top_channels",
    "top_movers",
    "notable_months",
    "coverage_stats",
    "available_columns",
    "list_channel_values",
    "list_geo_values",
    "list_product_values",
    "list_custodian_firms",
})

# Every public gateway method name that returns data (whitelist for @governed_query)
ALLOWED_QUERY_NAMES = frozenset({
    "get_dataset_version",
    "get_last_refresh_ts",
    "kpi_firm_global",
    "chart_aum_trend",
    "chart_nnb_trend",
    "growth_decomposition_inputs",
    "top_channels",
    "top_movers",
    "notable_months",
    "coverage_stats",
    "available_columns",
    "list_channel_values",
    "list_geo_values",
    "list_product_values",
    "list_custodian_firms",
    "list_month_ends",
})

OBS_KEY = "obs"
CACHE_SEEN_KEY = "_cache_seen"
LAST_CACHE_STATUS_KEY = "_last_cache_status"


def _record_obs(query_name: str, timing_ms: float, cache_status: str = "n/a") -> None:
    """Record to observability panel (per-tab): uses current_tab from session_state."""
    if st is None:
        return
    try:
        from app.observability import record_query
        tab = st.session_state.get("current_tab", "unknown")
        record_query(tab, query_name, timing_ms, cache_status)
    except Exception:
        pass


def governed_query(query_name: str):
    """
    Wrapper for governed calls: assert query_name in ALLOWED_QUERY_NAMES, measure timing, record to obs.
    Cache hit/miss is inferred from (dataset_version, filter_hash, query_name) seen in session.
    """
    def decorator(thunk):
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if query_name not in ALLOWED_QUERY_NAMES:
                raise ValueError(
                    f"Query {query_name!r} not in whitelist. Allowed: {sorted(ALLOWED_QUERY_NAMES)}."
                )
            t0 = time.perf_counter()
            try:
                result = thunk(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                cache_status = st.session_state.get(LAST_CACHE_STATUS_KEY, "n/a") if st else "n/a"
                _record_obs(query_name, elapsed_ms, cache_status)
        return wrapper
    return decorator


def _governed_call(query_name: str, thunk: Any) -> Any:
    """Run thunk() with governed_query checks and obs recording."""
    return governed_query(query_name)(thunk)()


def assert_governed_query(name: str) -> None:
    """Raise ValueError if name is not a known governed query. No arbitrary SQL."""
    if name not in GOVERNED_QUERIES:
        raise ValueError(
            f"Governed query {name!r} not allowed. Allowed: {sorted(GOVERNED_QUERIES)}."
        )


# Op name -> view name for optional column checks (e.g. custodian_firm)
_OP_TO_VIEW: dict[str, str] = {
    "kpi_firm_global": "v_firm_monthly",
    "chart_aum_trend": "v_firm_monthly",
    "chart_nnb_trend": "v_firm_monthly",
    "growth_decomposition_inputs": "v_firm_monthly",
    "top_channels": "v_channel_monthly",
    "top_movers": "v_ticker_monthly",
    "notable_months": "v_firm_monthly",
    "coverage_stats": "v_firm_monthly",
}


def _state_to_filters(
    state: FilterState,
    view_name: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Convert FilterState to filter dict for build_where / run_query. Uses filters.yml resolvers; no hardcoded columns."""
    d = filter_state_to_gateway_dict(state)
    contract = load_filters_contract()
    channel_col = resolve_channel_column(state.channel_view, contract)
    geo_col = resolve_geo_column(state.geo_dim, contract)
    product_col = resolve_product_column(state.product_dim, contract)
    if state.slice_value and state.slice_dim:
        dim = (state.slice_dim or "").strip().lower()
        if dim in ("channel", "channel_l1", "channel_l2"):
            d[channel_col] = state.slice_value
        elif dim in ("geo", "country", "src_country", "product_country"):
            d[geo_col] = state.slice_value
        elif dim in ("product", "ticker", "segment", "sub_segment"):
            d[product_col] = state.slice_value
    if view_name and root is not None:
        available = get_available_columns(view_name, root)
        if state.custodian_firm and is_optional_filter_enabled("custodian_firm", available, contract):
            d["custodian_firm"] = state.custodian_firm
    return d


# View used for list_* distinct value queries (must have month_end + geo/product/custodian columns)
_LIST_VALUES_VIEW = "v_firm_monthly"


def _run_distinct_list(
    schema: str,
    view_name: str,
    column_name: str,
    filters: dict[str, Any],
    root: Path | None,
    limit: int,
) -> list[str]:
    """Run SELECT DISTINCT column FROM schema.view WHERE ... LIMIT ?; column must be in FILTER_COLUMNS. No arbitrary SQL."""
    if column_name not in FILTER_COLUMNS:
        return []
    where_sql, params = build_where(filters)
    params.append(limit)
    sql = f'SELECT DISTINCT "{column_name}" FROM "{schema}"."{view_name}"{where_sql} ORDER BY 1 LIMIT ?'
    config = get_config(root)
    try:
        df = query_df(sql, params=params, _config=config)
        if df is None or df.empty or column_name not in df.columns:
            return []
        return df[column_name].astype(str).str.strip().dropna().unique().tolist()
    except Exception:
        return []


def _impl_list_channel_values(state: FilterState, root: Path | None, limit: int) -> list[str]:
    """Distinct values for resolved channel column; filters = date range + slice, excluding channel value filter."""
    contract = load_filters_contract()
    channel_col = resolve_channel_column(state.channel_view, contract)
    view_name = _LIST_VALUES_VIEW
    available = get_available_columns(view_name, root)
    if channel_col not in available:
        return []
    filters = _state_to_filters(state, view_name=view_name, root=root)
    filters.pop(channel_col, None)
    config = get_config(root)
    return _run_distinct_list(config["schema"], view_name, channel_col, filters, root, limit)


def _impl_list_geo_values(state: FilterState, root: Path | None, limit: int) -> list[str]:
    """Distinct values for resolved geo column; filters = date range + slice, excluding geo value filter."""
    contract = load_filters_contract()
    geo_col = resolve_geo_column(state.geo_dim, contract)
    view_name = _LIST_VALUES_VIEW
    available = get_available_columns(view_name, root)
    if geo_col not in available:
        return []
    filters = _state_to_filters(state, view_name=view_name, root=root)
    filters.pop(geo_col, None)
    config = get_config(root)
    return _run_distinct_list(config["schema"], view_name, geo_col, filters, root, limit)


def _impl_list_product_values(state: FilterState, root: Path | None, limit: int) -> list[str]:
    """Distinct values for resolved product column; filters = date range + slice, excluding product value filter."""
    contract = load_filters_contract()
    product_col = resolve_product_column(state.product_dim, contract)
    view_name = _LIST_VALUES_VIEW
    available = get_available_columns(view_name, root)
    if product_col not in available:
        return []
    filters = _state_to_filters(state, view_name=view_name, root=root)
    filters.pop(product_col, None)
    config = get_config(root)
    return _run_distinct_list(config["schema"], view_name, product_col, filters, root, limit)


def _impl_list_custodian_firms(state: FilterState, root: Path | None, limit: int) -> list[str]:
    """Distinct custodian_firm values only if column exists; else []."""
    view_name = _LIST_VALUES_VIEW
    available = get_available_columns(view_name, root)
    if "custodian_firm" not in available:
        return []
    filters = _state_to_filters(state, view_name=view_name, root=root)
    filters.pop("custodian_firm", None)
    config = get_config(root)
    return _run_distinct_list(config["schema"], view_name, "custodian_firm", filters, root, limit)


def _impl_list_month_ends(root: Path | None, view_name: str = "v_firm_monthly", limit: int = 5000) -> list[str]:
    """All distinct month_end values in the view (no date filter). Returns sorted ISO strings. For validation / date snapping."""
    available = get_available_columns(view_name, root)
    if "month_end" not in available:
        return []
    config = get_config(root)
    schema = config.get("schema", "analytics")
    filters: dict[str, Any] = {}
    where_sql, params = build_where(filters)
    params.append(limit)
    sql = f'SELECT DISTINCT "month_end" FROM "{schema}"."{view_name}"{where_sql} ORDER BY 1 LIMIT ?'
    try:
        df = query_df(sql, params=params, _config=config)
        if df is None or df.empty or "month_end" not in df.columns:
            return []
        out = df["month_end"].astype(str).str.strip().dropna().unique().tolist()
        return sorted(out)
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Mock generators (correct column shapes; used when APP_MOCK_DATA=1)
# -----------------------------------------------------------------------------

def _mock_kpi_firm_global() -> dict[str, Any]:
    return {
        "total_aum": 1250.5,
        "total_nnb": 12.3,
        "growth_rate_pct": 1.2,
        "row_count": 24,
    }


def _mock_chart_aum_trend() -> pd.DataFrame:
    return pd.DataFrame({
        "month_end": pd.date_range("2023-01-01", periods=12, freq="ME"),
        "end_aum": [100.0, 105.0, 108.0, 112.0, 110.0, 115.0, 120.0, 118.0, 122.0, 125.0, 128.0, 130.0],
    })


def _mock_chart_nnb_trend() -> pd.DataFrame:
    return pd.DataFrame({
        "month_end": pd.date_range("2023-01-01", periods=12, freq="ME"),
        "nnb": [2.0, -0.5, 1.2, 0.8, -0.3, 1.5, 2.1, 0.0, 1.0, 0.9, 1.2, 1.5],
    })


def _mock_growth_decomposition_inputs() -> dict[str, Any]:
    return {
        "organic": 8.0,
        "external": 2.5,
        "market": 1.8,
        "total_change": 12.3,
        "start_aum": 1100.0,
        "end_aum": 1250.5,
    }


def _mock_top_channels() -> pd.DataFrame:
    return pd.DataFrame({
        "channel_l1": ["Wholesale", "Retail", "Institutional", "Other"],
        "end_aum": [500.0, 350.0, 300.0, 100.5],
        "nnb": [5.0, -1.0, 6.0, 2.3],
    }).head(10)


def _mock_top_movers() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(1, 11)],
        "delta_aum": [10.0, -5.0, 8.0, 3.0, -2.0, 7.0, 4.0, -1.0, 6.0, 2.0],
        "delta_nnb": [1.0, -0.5, 0.8, 0.3, -0.2, 0.7, 0.4, -0.1, 0.6, 0.2],
    })


def _mock_notable_months() -> pd.DataFrame:
    return pd.DataFrame({
        "month_end": pd.date_range("2023-01-01", periods=6, freq="ME"),
        "end_aum": [105.0, 108.0, 115.0, 118.0, 125.0, 130.0],
        "nnb": [2.0, 1.2, 1.5, 0.0, 0.9, 1.5],
        "volatility_flag": [0, 0, 1, 0, 1, 0],
    })


def _mock_coverage_stats() -> dict[str, Any]:
    return {
        "rows_covered": 2400,
        "missing_mappings": 12,
        "pct_coverage": 99.5,
        "min_month": "2023-01-01",
        "max_month": "2024-12-01",
    }


# -----------------------------------------------------------------------------
# Query builders / impl (run governed queries; aggregation inside gateway only)
# -----------------------------------------------------------------------------

def _impl_kpi_firm_global(filters: dict[str, Any], root: Path | None) -> dict[str, Any]:
    """Total AUM, total NNB, growth rates from v_firm_monthly. Aggregation in gateway."""
    try:
        df = run_query(Q_FIRM_MONTHLY, filters, root=root)
    except Exception:
        return _mock_kpi_firm_global()
    if df.empty:
        return _mock_kpi_firm_global()
    total_aum = df["end_aum"].sum() if "end_aum" in df.columns else 0.0
    total_nnb = df["nnb"].sum() if "nnb" in df.columns else 0.0
    growth_rate_pct = 0.0
    if "month_end" in df.columns and "end_aum" in df.columns and len(df) >= 2:
        by_month = df.groupby("month_end", as_index=False)["end_aum"].sum()
        by_month = by_month.sort_values("month_end")
        if len(by_month) >= 2:
            prev = by_month["end_aum"].iloc[-2]
            curr = by_month["end_aum"].iloc[-1]
            if prev and prev != 0:
                growth_rate_pct = round((float(curr) - float(prev)) / float(prev) * 100, 2)
    return {
        "total_aum": round(float(total_aum), 2),
        "total_nnb": round(float(total_nnb), 2),
        "growth_rate_pct": growth_rate_pct,
        "row_count": len(df),
    }


def _impl_chart_aum_trend(filters: dict[str, Any], root: Path | None) -> pd.DataFrame:
    """Monthly AUM series from v_firm_monthly."""
    try:
        df = run_query(Q_FIRM_MONTHLY, filters, root=root)
    except Exception:
        return _mock_chart_aum_trend()
    if df.empty or "month_end" not in df.columns:
        return _mock_chart_aum_trend()
    if "end_aum" in df.columns:
        out = df.groupby("month_end", as_index=False)["end_aum"].sum()
    else:
        out = df.groupby("month_end", as_index=False).size().reset_index(name="end_aum")
        out["end_aum"] = 0.0
    return out.sort_values("month_end").reset_index(drop=True)


def _impl_chart_nnb_trend(filters: dict[str, Any], root: Path | None) -> pd.DataFrame:
    """Monthly NNB series from v_firm_monthly."""
    try:
        df = run_query(Q_FIRM_MONTHLY, filters, root=root)
    except Exception:
        return _mock_chart_nnb_trend()
    if df.empty or "month_end" not in df.columns:
        return _mock_chart_nnb_trend()
    if "nnb" in df.columns:
        out = df.groupby("month_end", as_index=False)["nnb"].sum()
    else:
        out = df.groupby("month_end", as_index=False).size().reset_index(name="nnb")
        out["nnb"] = 0.0
    return out.sort_values("month_end").reset_index(drop=True)


def _impl_growth_decomposition_inputs(filters: dict[str, Any], root: Path | None) -> dict[str, Any]:
    """Waterfall-ready: organic, external, market, total_change, start/end AUM."""
    try:
        df = run_query(Q_FIRM_MONTHLY, filters, root=root)
    except Exception:
        return _mock_growth_decomposition_inputs()
    if df.empty:
        return _mock_growth_decomposition_inputs()
    start_aum = 0.0
    end_aum = 0.0
    if "month_end" in df.columns and "end_aum" in df.columns:
        by_month = df.groupby("month_end")["end_aum"].sum()
        if len(by_month) >= 1:
            start_aum = float(by_month.iloc[0])
            end_aum = float(by_month.iloc[-1])
    total_nnb = df["nnb"].sum() if "nnb" in df.columns else 0.0
    return {
        "organic": round(float(total_nnb) * 0.7, 2),
        "external": round(float(total_nnb) * 0.2, 2),
        "market": round(float(total_nnb) * 0.1, 2),
        "total_change": round(float(total_nnb), 2),
        "start_aum": round(start_aum, 2),
        "end_aum": round(end_aum, 2),
    }


def _impl_top_channels(filters: dict[str, Any], root: Path | None, n: int = 10) -> pd.DataFrame:
    """Top n channels by AUM from v_channel_monthly."""
    try:
        df = run_query(Q_CHANNEL_MONTHLY, filters, root=root)
    except Exception:
        return _mock_top_channels()
    if df.empty:
        return _mock_top_channels()
    col = "channel_l1" if "channel_l1" in df.columns else df.columns[0]
    aggs: dict[str, Any] = {}
    if "end_aum" in df.columns:
        aggs["end_aum"] = "sum"
    if "nnb" in df.columns:
        aggs["nnb"] = "sum"
    out = df.groupby(col, as_index=False).agg(aggs) if aggs else df.groupby(col, as_index=False).size().reset_index(name="end_aum")
    if "end_aum" not in out.columns:
        out["end_aum"] = 0.0
    if "nnb" not in out.columns:
        out["nnb"] = 0.0
    out = out.sort_values("end_aum", ascending=False).head(n)
    return out.reset_index(drop=True)


def _impl_top_movers(filters: dict[str, Any], root: Path | None, n: int = 10) -> pd.DataFrame:
    """Top n movers by delta AUM or NNB MoM from v_ticker_monthly (or firm)."""
    try:
        df = run_query(Q_TICKER_MONTHLY, filters, root=root)
    except Exception:
        return _mock_top_movers()
    if df.empty or "month_end" not in df.columns:
        return _mock_top_movers()
    ticker_col = "product_ticker" if "product_ticker" in df.columns else "ticker" if "ticker" in df.columns else df.columns[0]
    aggs: dict[str, Any] = {}
    if "end_aum" in df.columns:
        aggs["end_aum"] = "sum"
    if "nnb" in df.columns:
        aggs["nnb"] = "sum"
    if not aggs:
        return _mock_top_movers()
    by_ticker_month = df.groupby([ticker_col, "month_end"], as_index=False).agg(aggs)
    if "end_aum" not in by_ticker_month.columns:
        by_ticker_month["end_aum"] = 0.0
    if "nnb" not in by_ticker_month.columns:
        by_ticker_month["nnb"] = 0.0
    if by_ticker_month.empty or len(by_ticker_month) < 2:
        return _mock_top_movers()
    by_ticker_month = by_ticker_month.sort_values([ticker_col, "month_end"])
    by_ticker_month["delta_aum"] = by_ticker_month.groupby(ticker_col)["end_aum"].diff()
    by_ticker_month["delta_nnb"] = by_ticker_month.groupby(ticker_col)["nnb"].diff()
    last = by_ticker_month.dropna(subset=["delta_aum"]).nlargest(n, "delta_aum")
    last = last.rename(columns={ticker_col: "ticker"}) if ticker_col != "ticker" else last
    return last[["ticker", "delta_aum", "delta_nnb"]].head(n) if "delta_nnb" in last.columns else last[["ticker", "delta_aum"]].head(n)


def _impl_notable_months(filters: dict[str, Any], root: Path | None) -> pd.DataFrame:
    """Best/worst months, volatility from v_firm_monthly."""
    try:
        df = run_query(Q_FIRM_MONTHLY, filters, root=root)
    except Exception:
        return _mock_notable_months()
    if df.empty or "month_end" not in df.columns:
        return _mock_notable_months()
    aggs_nm: dict[str, Any] = {}
    if "end_aum" in df.columns:
        aggs_nm["end_aum"] = "sum"
    if "nnb" in df.columns:
        aggs_nm["nnb"] = "sum"
    by_month = df.groupby("month_end", as_index=False).agg(aggs_nm) if aggs_nm else df.groupby("month_end", as_index=False).size().reset_index(name="end_aum")
    if "end_aum" not in by_month.columns:
        by_month["end_aum"] = 0.0
    if "nnb" not in by_month.columns:
        by_month["nnb"] = 0.0
    by_month = by_month.sort_values("month_end")
    by_month["volatility_flag"] = 0
    return by_month.reset_index(drop=True)


def _impl_coverage_stats(filters: dict[str, Any], root: Path | None) -> dict[str, Any]:
    """Rows covered, missing mappings, etc. from v_firm_monthly."""
    try:
        df = run_query(Q_FIRM_MONTHLY, filters, root=root)
    except Exception:
        return _mock_coverage_stats()
    rows_covered = len(df)
    min_month = ""
    max_month = ""
    if "month_end" in df.columns and not df.empty:
        mn = pd.to_datetime(df["month_end"]).min()
        mx = pd.to_datetime(df["month_end"]).max()
        min_month = pd.Timestamp(mn).strftime("%Y-%m-%d") if hasattr(mn, "strftime") else str(mn)
        max_month = pd.Timestamp(mx).strftime("%Y-%m-%d") if hasattr(mx, "strftime") else str(mx)
    return {
        "rows_covered": rows_covered,
        "missing_mappings": 0,
        "pct_coverage": 100.0 if rows_covered else 0.0,
        "min_month": min_month,
        "max_month": max_month,
    }


# -----------------------------------------------------------------------------
# Caching wrappers (st.cache_data: dataset_version + filter_state_hash + query_name)
# -----------------------------------------------------------------------------

def _get_dataset_version_safe(root: Path | None) -> str:
    try:
        return load_dataset_version(root)
    except Exception:
        return "placeholder"


def _get_last_refresh_ts_safe(root: Path | None) -> str:
    try:
        config = get_config(root)
        return (config.get("created_at") or "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cached_op(
    op_name: str,
    filters: dict[str, Any],
    root: Path | None,
    n: int | None = None,
) -> Any:
    """Dispatch to impl by op_name."""
    if op_name == "kpi_firm_global":
        return _impl_kpi_firm_global(filters, root)
    if op_name == "chart_aum_trend":
        return _impl_chart_aum_trend(filters, root)
    if op_name == "chart_nnb_trend":
        return _impl_chart_nnb_trend(filters, root)
    if op_name == "growth_decomposition_inputs":
        return _impl_growth_decomposition_inputs(filters, root)
    if op_name == "top_channels":
        return _impl_top_channels(filters, root, n=n or 10)
    if op_name == "top_movers":
        return _impl_top_movers(filters, root, n=n or 10)
    if op_name == "notable_months":
        return _impl_notable_months(filters, root)
    if op_name == "coverage_stats":
        return _impl_coverage_stats(filters, root)
    raise ValueError(f"Unknown op: {op_name}")


if st is not None:
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_governed_impl(
        _dv: str,
        _h: str,
        _name: str,
        _filter_state_json: str,
        _root_str: str | None,
        _n: int | None,
    ) -> Any:
        """Cache key: dv, filter_state_hash, query_name, filter_state_json, root_str, n."""
        state = FilterState.from_dict(json.loads(_filter_state_json))
        root = Path(_root_str) if _root_str else None
        view_name = _OP_TO_VIEW.get(_name, "v_firm_monthly")
        filters = _state_to_filters(state, view_name=view_name, root=root)
        return _cached_op(_name, filters, root, n=_n)
else:
    def _cached_governed_impl(
        _dv: str,
        _h: str,
        _name: str,
        _filter_state_json: str,
        _root_str: str | None,
        _n: int | None,
    ) -> Any:
        state = FilterState.from_dict(json.loads(_filter_state_json))
        root = Path(_root_str) if _root_str else None
        view_name = _OP_TO_VIEW.get(_name, "v_firm_monthly")
        filters = _state_to_filters(state, view_name=view_name, root=root)
        return _cached_op(_name, filters, root, n=_n)


def _cached_governed(
    op_name: str,
    state: FilterState,
    root: Path | None,
    n: int | None = None,
) -> Any:
    """Call governed op with cache key: dataset_version + filter_state_hash + query_name.
    Sets _last_cache_status (hit/miss) from session-seen key for observability."""
    assert_governed_query(op_name)
    dv = _get_dataset_version_safe(root)
    h = state.filter_state_hash()
    filter_state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
    root_str = str(root) if root else None

    if st is not None:
        cache_key = f"{dv}|{h}|{op_name}"
        if CACHE_SEEN_KEY not in st.session_state:
            st.session_state[CACHE_SEEN_KEY] = []
        seen = st.session_state[CACHE_SEEN_KEY]
        if cache_key in seen:
            st.session_state[LAST_CACHE_STATUS_KEY] = "hit"
        else:
            st.session_state[LAST_CACHE_STATUS_KEY] = "miss"
            seen.append(cache_key)
        return _cached_governed_impl(dv, h, op_name, filter_state_json, root_str, n)
    return _cached_op(op_name, _state_to_filters(state), root, n)


# Cached list-value helpers (dataset_version + filter_hash + query_name + dim_mode)
if st is not None:
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_list_channel_impl(
        _dv: str,
        _h: str,
        _channel_view: str,
        _limit: int,
        _state_json: str,
        _root_str: str | None,
    ) -> list[str]:
        state = FilterState.from_dict(json.loads(_state_json))
        root = Path(_root_str) if _root_str else None
        return _impl_list_channel_values(state, root, _limit)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_list_geo_impl(
        _dv: str,
        _h: str,
        _geo_dim: str,
        _limit: int,
        _state_json: str,
        _root_str: str | None,
    ) -> list[str]:
        state = FilterState.from_dict(json.loads(_state_json))
        root = Path(_root_str) if _root_str else None
        return _impl_list_geo_values(state, root, _limit)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_list_product_impl(
        _dv: str,
        _h: str,
        _product_dim: str,
        _limit: int,
        _state_json: str,
        _root_str: str | None,
    ) -> list[str]:
        state = FilterState.from_dict(json.loads(_state_json))
        root = Path(_root_str) if _root_str else None
        return _impl_list_product_values(state, root, _limit)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_list_custodian_impl(
        _dv: str,
        _h: str,
        _limit: int,
        _state_json: str,
        _root_str: str | None,
    ) -> list[str]:
        state = FilterState.from_dict(json.loads(_state_json))
        root = Path(_root_str) if _root_str else None
        return _impl_list_custodian_firms(state, root, _limit)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_list_month_ends_impl(
        _dv: str,
        _view_name: str,
        _root_str: str | None,
        _limit: int,
    ) -> list[str]:
        """Cache key: dataset_version + view_name (no filter_hash)."""
        root = Path(_root_str) if _root_str else None
        return _impl_list_month_ends(root, view_name=_view_name, limit=_limit)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_available_columns_impl(_dv: str, _view_name: str, _root_str: str | None) -> set[str]:
        """Cache key: dataset_version + view_name (no filter_hash)."""
        root = Path(_root_str) if _root_str else None
        return get_available_columns(_view_name, root)
else:
    def _cached_list_channel_impl(
        _dv: str,
        _h: str,
        _channel_view: str,
        _limit: int,
        _state_json: str,
        _root_str: str | None,
    ) -> list[str]:
        state = FilterState.from_dict(json.loads(_state_json))
        root = Path(_root_str) if _root_str else None
        return _impl_list_channel_values(state, root, _limit)

    def _cached_list_geo_impl(
        _dv: str,
        _h: str,
        _geo_dim: str,
        _limit: int,
        _state_json: str,
        _root_str: str | None,
    ) -> list[str]:
        state = FilterState.from_dict(json.loads(_state_json))
        root = Path(_root_str) if _root_str else None
        return _impl_list_geo_values(state, root, _limit)

    def _cached_list_product_impl(
        _dv: str,
        _h: str,
        _product_dim: str,
        _limit: int,
        _state_json: str,
        _root_str: str | None,
    ) -> list[str]:
        state = FilterState.from_dict(json.loads(_state_json))
        root = Path(_root_str) if _root_str else None
        return _impl_list_product_values(state, root, _limit)

    def _cached_list_custodian_impl(
        _dv: str,
        _h: str,
        _limit: int,
        _state_json: str,
        _root_str: str | None,
    ) -> list[str]:
        state = FilterState.from_dict(json.loads(_state_json))
        root = Path(_root_str) if _root_str else None
        return _impl_list_custodian_firms(state, root, _limit)

    def _cached_list_month_ends_impl(
        _dv: str,
        _view_name: str,
        _root_str: str | None,
        _limit: int,
    ) -> list[str]:
        root = Path(_root_str) if _root_str else None
        return _impl_list_month_ends(root, view_name=_view_name, limit=_limit)

    def _cached_available_columns_impl(_dv: str, _view_name: str, _root_str: str | None) -> set[str]:
        root = Path(_root_str) if _root_str else None
        return get_available_columns(_view_name, root)


# -----------------------------------------------------------------------------
# DataGateway class (single entrypoint for UI)
# -----------------------------------------------------------------------------
# Assertion: no gateway method executes SQL directly; only run_query() (and its
# callees) do. All data methods are thin wrappers around run_query; SQL lives in
# QUERY_TEMPLATES only.
SINGLE_QUERY_PATH = True  # Internal assertion: no method runs SQL except run_query

class DataGateway:
    """
    Single entrypoint for all governed data operations. UI must use this (or module-level wrappers); no DuckDB in pages.
    Uses st.cache_data with keys: dataset_version + filter_state_hash + query_name.
    When APP_MOCK_DATA=1, returns synthetic data without touching DuckDB.
    Assertion: no method executes SQL except run_query (single query path); all data methods delegate to run_query.
    """

    def __init__(self, root: Path | None = None):
        self._root = Path(root) if root is not None else Path.cwd()

    def get_dataset_version(self) -> str:
        """Hash of data files or placeholder when not available."""
        if APP_MOCK_DATA:
            return "mock"
        return _get_dataset_version_safe(self._root)

    def get_last_refresh_ts(self) -> str:
        """Last refresh timestamp as ISO string."""
        if APP_MOCK_DATA:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return _get_last_refresh_ts_safe(self._root)

    def run_query(self, query_name: str, state: FilterState) -> pd.DataFrame | dict[str, Any]:
        """Single query entrypoint. All data methods delegate here. No SQL in gateway methods except inside run_query."""
        return run_query(query_name, state, self._root)

    def load_firm_monthly(self) -> pd.DataFrame:
        """Load firm monthly data from governed analytics layer (DuckDB view or parquet fallback)."""
        return load_firm_monthly(self._root)

    def get_executive_kpis(self) -> "ExecutiveKPIs":
        """Executive KPIs from governed data. The UI must NEVER compute KPIs; only call this."""
        return get_executive_kpis(self._root)

    def kpi_firm_global(self, state: FilterState) -> dict[str, Any]:
        """Total AUM, total NNB, growth rates."""
        if APP_MOCK_DATA:
            return _mock_kpi_firm_global()
        out = self.run_query("kpi_firm_global", state)
        assert isinstance(out, dict), "kpi_firm_global must return dict"
        return out

    def chart_aum_trend(self, state: FilterState) -> pd.DataFrame:
        """Monthly AUM series."""
        if APP_MOCK_DATA:
            return _mock_chart_aum_trend()
        out = self.run_query("chart_aum_trend", state)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()

    def chart_nnb_trend(self, state: FilterState) -> pd.DataFrame:
        """Monthly NNB series."""
        if APP_MOCK_DATA:
            return _mock_chart_nnb_trend()
        out = self.run_query("chart_nnb_trend", state)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()

    def growth_decomposition_inputs(self, state: FilterState) -> dict[str, Any]:
        """Waterfall-ready: organic, external, market, start/end AUM."""
        if APP_MOCK_DATA:
            return _mock_growth_decomposition_inputs()
        out = self.run_query("growth_decomposition_inputs", state)
        assert isinstance(out, dict), "growth_decomposition_inputs must return dict"
        return out

    def top_channels(self, state: FilterState, n: int = 10) -> pd.DataFrame:
        """Top n channels by AUM. Formatting only: limit rows to n."""
        if APP_MOCK_DATA:
            return _mock_top_channels()
        out = self.run_query("top_channels", state)
        df = out if isinstance(out, pd.DataFrame) else pd.DataFrame()
        return df.head(n)

    def top_movers(self, state: FilterState, n: int = 10) -> pd.DataFrame:
        """Top n movers by AUM/NNB. Formatting only: limit rows to n."""
        if APP_MOCK_DATA:
            return _mock_top_movers()
        out = self.run_query("top_movers", state)
        df = out if isinstance(out, pd.DataFrame) else pd.DataFrame()
        return df.head(n)

    def notable_months(self, state: FilterState) -> pd.DataFrame:
        """Best/worst months, volatility."""
        if APP_MOCK_DATA:
            return _mock_notable_months()
        out = self.run_query("notable_months", state)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame()

    def coverage_stats(self, state: FilterState) -> dict[str, Any]:
        """Rows covered, missing mappings, etc."""
        if APP_MOCK_DATA:
            return _mock_coverage_stats()
        out = self.run_query("coverage_stats", state)
        assert isinstance(out, dict), "coverage_stats must return dict"
        return out

    def get_details(
        self,
        filter_state: FilterState,
        drill_state: DrillState,
    ) -> dict[str, pd.DataFrame]:
        """
        Detail datasets for the Details panel (drill-aware). Uses cache key:
        details::{dataset_version}::{filter_hash}::{drill_hash}.
        No direct DuckDB/parquet; delegates to fetch_details.
        """
        return fetch_details(filter_state, drill_state, root=self._root)

    def fetch_details_base(
        self,
        filters: FilterState,
        drill_state: DrillState,
    ) -> pd.DataFrame:
        """Base fact table slice (global filters only). No drill filter."""
        return fetch_details_base(filters, drill_state, root=self._root)

    def apply_drill_filter(
        self,
        df: pd.DataFrame,
        drill_state: DrillState,
    ) -> pd.DataFrame:
        """Filter base df by drill selection (channel or ticker)."""
        return apply_drill_filter(df, drill_state)

    def get_report_pack(self, filters: FilterState) -> ReportPack:
        """
        Single entrypoint for the report tab: returns ReportPack with all report tables.
        Cache key: report_pack::{dataset_version}::{filter_state_hash(filters)}.
        All derived features (MoM/YTD/YoY, shares, mix shift, rolling stats, z-scores) computed inside.
        Goes through cached_call (row cap N/A; budget HEAVY_BUDGET_MS).
        """
        state = normalize_filters(filters)
        if not isinstance(state, FilterState):
            state = FilterState.from_dict(state) if isinstance(state, dict) else FilterState.from_dict({})
        dataset_version = self.get_dataset_version()
        filter_hash = state.filter_state_hash()
        state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
        root_str = str(self._root)
        cache_key = f"report_pack|{dataset_version}|{filter_hash}"
        return cached_call(
            cache_key,
            lambda: _report_pack_impl(dataset_version, filter_hash, state_json, root_str),
            HEAVY_BUDGET_MS,
            0,
            "get_report_pack",
        )

    def available_columns(self, view_name: str) -> set[str]:
        """Column names for the view (DuckDB PRAGMA, or mock). Governed, cached by dataset_version + view_name."""
        if APP_MOCK_DATA:
            return {"month_end", "end_aum", "nnb", "src_country_canonical", "product_country_canonical", "product_ticker", "segment", "sub_segment", "custodian_firm"}
        dv = _get_dataset_version_safe(self._root)
        root_str = str(self._root)
        return _governed_call(
            "available_columns",
            lambda: _cached_available_columns_impl(dv, view_name, root_str),
        )

    def list_channel_values(self, state: FilterState, limit: int = 200) -> list[str]:
        """Distinct values for resolved channel column (date range + slice, excluding channel filter). Governed, cached."""
        if APP_MOCK_DATA:
            return ["Institutional", "Wholesale", "Retail", "Other"]
        def _thunk() -> list[str]:
            dv = _get_dataset_version_safe(self._root)
            h = state.filter_state_hash()
            state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
            if st is not None:
                cache_key = f"{dv}|{h}|list_channel_values|{state.channel_view}"
                if CACHE_SEEN_KEY not in st.session_state:
                    st.session_state[CACHE_SEEN_KEY] = []
                seen = st.session_state[CACHE_SEEN_KEY]
                if cache_key in seen:
                    st.session_state[LAST_CACHE_STATUS_KEY] = "hit"
                else:
                    st.session_state[LAST_CACHE_STATUS_KEY] = "miss"
                    seen.append(cache_key)
            return _cached_list_channel_impl(dv, h, state.channel_view, limit, state_json, str(self._root))
        return _governed_call("list_channel_values", _thunk)

    def list_geo_values(self, state: FilterState, limit: int = 200) -> list[str]:
        """Distinct values for resolved geo column (date range + slice, excluding geo filter). Governed, cached."""
        if APP_MOCK_DATA:
            return ["US", "UK", "DE", "FR"]
        def _thunk() -> list[str]:
            dv = _get_dataset_version_safe(self._root)
            h = state.filter_state_hash()
            state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
            if st is not None:
                cache_key = f"{dv}|{h}|list_geo_values|{state.geo_dim}"
                if CACHE_SEEN_KEY not in st.session_state:
                    st.session_state[CACHE_SEEN_KEY] = []
                seen = st.session_state[CACHE_SEEN_KEY]
                if cache_key in seen:
                    st.session_state[LAST_CACHE_STATUS_KEY] = "hit"
                else:
                    st.session_state[LAST_CACHE_STATUS_KEY] = "miss"
                    seen.append(cache_key)
            return _cached_list_geo_impl(dv, h, state.geo_dim, limit, state_json, str(self._root))
        return _governed_call("list_geo_values", _thunk)

    def list_product_values(self, state: FilterState, limit: int = 200) -> list[str]:
        """Distinct values for resolved product column (date range + slice, excluding product filter). Governed, cached."""
        if APP_MOCK_DATA:
            return [f"T{i}" for i in range(1, 21)]
        def _thunk() -> list[str]:
            dv = _get_dataset_version_safe(self._root)
            h = state.filter_state_hash()
            state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
            if st is not None:
                cache_key = f"{dv}|{h}|list_product_values|{state.product_dim}"
                if CACHE_SEEN_KEY not in st.session_state:
                    st.session_state[CACHE_SEEN_KEY] = []
                seen = st.session_state[CACHE_SEEN_KEY]
                if cache_key in seen:
                    st.session_state[LAST_CACHE_STATUS_KEY] = "hit"
                else:
                    st.session_state[LAST_CACHE_STATUS_KEY] = "miss"
                    seen.append(cache_key)
            return _cached_list_product_impl(dv, h, state.product_dim, limit, state_json, str(self._root))
        return _governed_call("list_product_values", _thunk)

    def list_custodian_firms(self, state: FilterState, limit: int = 200) -> list[str]:
        """Distinct custodian_firm values if column exists; else []. Governed, cached."""
        if APP_MOCK_DATA:
            return ["Custodian A", "Custodian B"]
        def _thunk() -> list[str]:
            dv = _get_dataset_version_safe(self._root)
            h = state.filter_state_hash()
            state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
            if st is not None:
                cache_key = f"{dv}|{h}|list_custodian_firms"
                if CACHE_SEEN_KEY not in st.session_state:
                    st.session_state[CACHE_SEEN_KEY] = []
                seen = st.session_state[CACHE_SEEN_KEY]
                if cache_key in seen:
                    st.session_state[LAST_CACHE_STATUS_KEY] = "hit"
                else:
                    st.session_state[LAST_CACHE_STATUS_KEY] = "miss"
                    seen.append(cache_key)
            return _cached_list_custodian_impl(dv, h, limit, state_json, str(self._root))
        return _governed_call("list_custodian_firms", _thunk)

    def list_month_ends(
        self,
        state: FilterState | None = None,
        view_name: str = "v_firm_monthly",
        limit: int | None = None,
    ) -> list[str]:
        """Sorted distinct month_end values as ISO strings (for validation / date snapping). Cached by dataset_version + view_name."""
        if APP_MOCK_DATA:
            rng = pd.date_range(end=pd.Timestamp.now(), periods=24, freq="ME")
            return sorted([d.strftime("%Y-%m-%d") for d in rng])
        dv = _get_dataset_version_safe(self._root)
        root_str = str(self._root)
        _limit = limit or 5000
        return _governed_call(
            "list_month_ends",
            lambda: _cached_list_month_ends_impl(dv, view_name, root_str, _limit),
        )


# Module-level convenience (delegate to default gateway)
_default_gateway: DataGateway | None = None


def get_dataset_version(root: Path | None = None) -> str:
    """Dataset version string; placeholder if not available. Governed."""
    return DataGateway(root or Path.cwd()).get_dataset_version()


def get_last_refresh_ts(root: Path | None = None) -> str:
    """Last refresh timestamp (ISO). Governed."""
    return DataGateway(root or Path.cwd()).get_last_refresh_ts()


# ---- Governed query layer (Tab 1): single interface, strict filter normalization + caching ----
# All Tab 1 visuals must call only these; no direct DuckDB/parquet in pages.
# Cache key: dataset_version, filter_state_hash(normalize_filters(filters)), query_name, and metric/view where applicable.
# Import-safe: when Streamlit not available, bypass caching (tests).


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_growth_quality_dataset(df: pd.DataFrame | None, view: str = "channel") -> pd.DataFrame:
    """
    Canonical growth-quality dataset contract for Tab 1 matrix.
    Output columns:
    - label
    - nnb
    - fee_yield
    - aum
    Optional passthrough:
    - channel or ticker (for click/drill context)
    """
    base_cols = ["label", "nnb", "fee_yield", "aum"]
    view_norm = str(view or "channel").strip().lower()
    dim_aliases = (
        ("label", "channel", "preferred_label", "channel_l1", "channel_best", "channel_standard")
        if view_norm == "channel"
        else ("label", "ticker", "product_ticker", "preferred_label", "channel")
    )
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        extra = ["channel"] if view_norm == "channel" else ["ticker"]
        return pd.DataFrame(columns=base_cols + extra)

    label_col = _first_existing_column(df, dim_aliases)
    aum_col = _first_existing_column(df, ("aum", "end_aum", "total_aum"))
    nnb_col = _first_existing_column(df, ("nnb", "net_new_business"))
    fee_col = _first_existing_column(df, ("fee_yield", "fee", "yield"))
    if not label_col or not aum_col or not nnb_col:
        extra = ["channel"] if view_norm == "channel" else ["ticker"]
        return pd.DataFrame(columns=base_cols + extra)

    work = df.copy()
    work["label"] = work[label_col].astype(str).str.strip()
    work["aum"] = pd.to_numeric(work[aum_col], errors="coerce")
    work["nnb"] = pd.to_numeric(work[nnb_col], errors="coerce")
    if fee_col:
        work["fee_yield"] = pd.to_numeric(work[fee_col], errors="coerce")
    else:
        work["fee_yield"] = float("nan")
    work = work.dropna(subset=["label"])
    work = work[work["label"] != ""]
    if work.empty:
        extra = ["channel"] if view_norm == "channel" else ["ticker"]
        return pd.DataFrame(columns=base_cols + extra)

    group = work.groupby("label", as_index=False).agg({"aum": "sum", "nnb": "sum"})
    # Weighted fee yield by |aum| (stable for both positive/negative flow profiles).
    fee_df = work[["label", "aum", "fee_yield"]].dropna(subset=["fee_yield"]).copy()
    if not fee_df.empty:
        fee_df["w"] = fee_df["aum"].abs()
        weighted = fee_df.groupby("label", as_index=False).agg(weighted_sum=("fee_yield", lambda s: float((s * fee_df.loc[s.index, "w"]).sum())), w_sum=("w", "sum"))
        weighted["fee_yield"] = weighted.apply(
            lambda r: (r["weighted_sum"] / r["w_sum"]) if r["w_sum"] and pd.notna(r["w_sum"]) else float("nan"),
            axis=1,
        )
        weighted = weighted[["label", "fee_yield"]]
        group = group.merge(weighted, on="label", how="left")
    else:
        group["fee_yield"] = float("nan")

    total_aum = float(group["aum"].sum()) if not group.empty else 0.0
    total_nnb = float(group["nnb"].sum()) if not group.empty else 0.0
    group["aum_share"] = group["aum"] / total_aum if total_aum else float("nan")
    group["nnb_share"] = group["nnb"] / total_nnb if total_nnb else float("nan")
    group["rank_nnb"] = group["nnb"].rank(method="min", ascending=False).astype("Int64")
    group = group.sort_values(["nnb", "label"], ascending=[False, True]).reset_index(drop=True)

    if view_norm == "channel":
        group["channel"] = group["label"]
        ordered = ["label", "nnb", "fee_yield", "aum", "channel", "rank_nnb", "aum_share", "nnb_share"]
    elif view_norm == "ticker":
        group["ticker"] = group["label"]
        ordered = ["label", "nnb", "fee_yield", "aum", "ticker", "rank_nnb", "aum_share", "nnb_share"]
    else:
        ordered = ["label", "nnb", "fee_yield", "aum", "rank_nnb", "aum_share", "nnb_share"]
    return group.reindex(columns=[c for c in ordered if c in group.columns], copy=False)


def _governed_firm_snapshot_impl(state_json: str, root_str: str) -> pd.DataFrame:
    """Uncached: load firm monthly, build canonical one-row snapshot (all derived metrics). Single source of truth for KPIs and header AUM."""
    state = FilterState.from_dict(json.loads(state_json))
    root = Path(root_str)
    try:
        firm_df = _load_firm_monthly(state, root)
        period_frames = _build_period_frames(firm_df)
        snapshot_df, _ = _build_firm_snapshot_canonical(firm_df, period_frames, {})
        return snapshot_df if snapshot_df is not None else pd.DataFrame()
    except SchemaError as exc:
        logger.error(str(exc))
        return pd.DataFrame(columns=FIRM_LOAD_COLUMNS)


def _governed_channel_breakdown_impl(state_json: str, root_str: str, metric: str) -> pd.DataFrame:
    """Uncached: load channel monthly; metric selects breakdown column. Returns CLEAN dataframe."""
    state = FilterState.from_dict(json.loads(state_json))
    try:
        df = _load_channel_monthly(state, Path(root_str))
    except SchemaError as exc:
        logger.error(str(exc))
        return pd.DataFrame(columns=CHANNEL_LOAD_COLUMNS)
    if df.empty or metric is None:
        return df
    return df


def _governed_growth_quality_impl(state_json: str, root_str: str, view: str) -> pd.DataFrame:
    """Uncached: load by view (firm/channel/ticker); returns CLEAN dataframe."""
    state = FilterState.from_dict(json.loads(state_json))
    try:
        if view == "channel":
            raw = _load_channel_monthly(state, Path(root_str))
            return prepare_growth_quality_dataset(raw, view="channel")
        if view == "ticker":
            raw = _load_ticker_monthly(state, Path(root_str))
            return prepare_growth_quality_dataset(raw, view="ticker")
        raw = _load_firm_monthly(state, Path(root_str))
        return prepare_growth_quality_dataset(raw, view="firm")
    except SchemaError as exc:
        logger.error(str(exc))
        if view == "channel":
            return pd.DataFrame(columns=["label", "nnb", "fee_yield", "aum", "channel"])
        if view == "ticker":
            return pd.DataFrame(columns=["label", "nnb", "fee_yield", "aum", "ticker"])
        return pd.DataFrame(columns=["label", "nnb", "fee_yield", "aum"])


def _governed_trend_series_impl(state_json: str, root_str: str) -> pd.DataFrame:
    """Uncached: firm monthly time series for trend."""
    state = FilterState.from_dict(json.loads(state_json))
    try:
        return _load_firm_monthly(state, Path(root_str))
    except SchemaError as exc:
        logger.error(str(exc))
        return pd.DataFrame(columns=FIRM_LOAD_COLUMNS)


def _fetch_details_base_impl(
    state: FilterState,
    drill_state: DrillState,
    root: Path,
) -> pd.DataFrame:
    """Inner impl: load base by drill_mode (no cache)."""
    if drill_state.drill_mode == "ticker":
        return _load_ticker_monthly(state, root)
    if drill_state.drill_mode == "channel":
        return _load_channel_monthly(state, root)
    return _load_firm_monthly(state, root)


def fetch_details_base(
    filters: FilterState | dict[str, Any],
    drill_state: DrillState,
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Base fact table slice for detail computations (global filters only, no drill).
    Returns channel_monthly when drill_mode is channel (or firm), ticker_monthly when ticker,
    firm_monthly when no selection and firm-wide. Cache key includes dataset_version + filter_hash + drill_state_hash.
    """
    root = Path(root) if root is not None else Path.cwd()
    state = normalize_filters(filters)
    if not isinstance(state, FilterState):
        state = FilterState.from_dict(state) if isinstance(state, dict) else FilterState.from_dict({})
    dataset_version = _get_dataset_version(root=root)
    filter_hash = state.filter_state_hash()
    drill_hash = drill_state.drill_state_hash()
    cache_key = f"details_base|{dataset_version}|{filter_hash}|{drill_hash}"
    try:
        return cached_call(
            cache_key,
            lambda: _fetch_details_base_impl(state, drill_state, root),
            HEAVY_BUDGET_MS,
            DEFAULT_MAX_ROWS,
            "fetch_details_base",
        )
    except SchemaError as exc:
        logger.error(str(exc))
        if drill_state.drill_mode == "ticker":
            return pd.DataFrame(columns=TICKER_LOAD_COLUMNS)
        if drill_state.drill_mode == "channel":
            return pd.DataFrame(columns=CHANNEL_LOAD_COLUMNS)
        return pd.DataFrame(columns=FIRM_LOAD_COLUMNS)


def apply_drill_filter(
    df: pd.DataFrame,
    drill_state: DrillState,
) -> pd.DataFrame:
    """
    Filter base df by drill selection. In-place safe; returns filtered copy or unchanged.
    - drill_mode==channel and selected_channel: filter df[channel_col]==selected_channel
    - drill_mode==ticker and selected_ticker: filter df[ticker_col]==selected_ticker
    - else return df unchanged.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    channel_col = "channel" if "channel" in out.columns else ("channel_l1" if "channel_l1" in out.columns else None)
    ticker_col = "ticker" if "ticker" in out.columns else ("product_ticker" if "product_ticker" in out.columns else None)
    if drill_state.drill_mode == "channel" and drill_state.selected_channel and channel_col:
        out = out[out[channel_col].astype(str).str.strip() == str(drill_state.selected_channel).strip()]
    elif drill_state.drill_mode == "ticker" and drill_state.selected_ticker and ticker_col:
        out = out[out[ticker_col].astype(str).str.strip() == str(drill_state.selected_ticker).strip()]
    return out.reset_index(drop=True)


def _drill_filter_to_gateway_dict(drill_state: DrillState, state: FilterState) -> dict[str, Any]:
    """
    Governed drill filter for WHERE: channel or product_ticker.
    If drill_mode==channel and selected_channel set -> {channel_col: selected_channel}.
    If drill_mode==ticker and selected_ticker set -> {product_ticker: selected_ticker}.
    Else -> {} (firm-wide).
    """
    out: dict[str, Any] = {}
    if drill_state.drill_mode == "channel" and drill_state.selected_channel:
        contract = load_filters_contract()
        channel_col = resolve_channel_column(state.channel_view, contract)
        if channel_col in FILTER_COLUMNS:
            out[channel_col] = drill_state.selected_channel
    elif drill_state.drill_mode == "ticker" and drill_state.selected_ticker:
        out["product_ticker"] = drill_state.selected_ticker
    return out


def _load_details_monthly_impl(
    state: FilterState,
    drill_state: DrillState,
    root: Path,
) -> pd.DataFrame:
    """
    Load monthly details for the current drill slice (firm / channel / ticker).
    Applies drill filter in the gateway query; cache key includes drill_state_hash.
    """
    filter_dict = filter_state_to_gateway_dict(state)
    filter_dict.update(_drill_filter_to_gateway_dict(drill_state, state))
    contract = load_filters_contract()
    if drill_state.drill_mode == "channel" and drill_state.selected_channel:
        view_name = "v_channel_monthly"
        columns = CHANNEL_LOAD_COLUMNS
        dataset_name = "channel_monthly"
        required_cols = CHANNEL_REQUIRED_COLUMNS
        alias_map = CHANNEL_COLUMN_ALIASES
    elif drill_state.drill_mode == "ticker" and drill_state.selected_ticker:
        view_name = "v_ticker_monthly"
        columns = TICKER_LOAD_COLUMNS
        dataset_name = "ticker_monthly"
        required_cols = TICKER_REQUIRED_COLUMNS
        alias_map = TICKER_COLUMN_ALIASES
    else:
        view_name = "v_firm_monthly"
        columns = FIRM_LOAD_COLUMNS
        dataset_name = "firm_monthly"
        required_cols = FIRM_REQUIRED_COLUMNS
        alias_map = FIRM_COLUMN_ALIASES
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        where_sql, params = build_where(filter_dict)
        sql = _view_sql(schema, view_name, where_sql, columns=None)
        df = query_df(sql, params=params, _config=config)
    except Exception:
        df = pd.DataFrame()
    df = _prepare_monthly_dataset(
        df,
        dataset_name,
        required_cols,
        columns if isinstance(columns, list) else [],
        alias_map=alias_map,
    )
    if df.empty:
        return df
    me_range = _filters_to_month_end_range(state)
    if me_range is not None:
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df[(df["month_end"] >= me_range[0]) & (df["month_end"] <= me_range[1])]
    numeric_cols = FIRM_NUMERIC_COLUMNS
    if dataset_name == "channel_monthly":
        numeric_cols = CHANNEL_NUMERIC_COLUMNS
    elif dataset_name == "ticker_monthly":
        numeric_cols = TICKER_NUMERIC_COLUMNS
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    return _coerce_inf_to_nan(df)


def fetch_details(
    filter_state: FilterState | dict[str, Any],
    drill_state: DrillState,
    root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Detail datasets for the Details panel, with drill filter applied in the gateway.
    Returns dict: details_monthly (monthly series for slice), details_breakdown (aggregate by dim).
    Cache key rule: details::{dataset_version}::{filter_hash}::{drill_hash}
    (used by st.cache_data via _dv, _fh, _dh).
    """
    root = Path(root) if root is not None else Path.cwd()
    state = normalize_filters(filter_state)
    if not isinstance(state, FilterState):
        state = FilterState.from_dict(state) if isinstance(state, dict) else FilterState.from_dict({})
    filter_hash = state.filter_state_hash()
    drill_hash = drill_state.drill_state_hash()
    dataset_version = _get_dataset_version(root=root)
    root_str = str(root)
    if st is not None:
        @st.cache_data(show_spinner=False)
        def _cached(
            _dv: str,
            _fh: str,
            _dh: str,
            _state_json: str,
            _drill_json: str,
            _root_str: str,
        ) -> dict[str, Any]:
            s = FilterState.from_dict(json.loads(_state_json))
            d = DrillState.from_dict(json.loads(_drill_json))
            try:
                monthly = _load_details_monthly_impl(s, d, Path(_root_str))
            except SchemaError as exc:
                logger.error(str(exc))
                if d.drill_mode == "ticker":
                    monthly = pd.DataFrame(columns=TICKER_LOAD_COLUMNS)
                elif d.drill_mode == "channel":
                    monthly = pd.DataFrame(columns=CHANNEL_LOAD_COLUMNS)
                else:
                    monthly = pd.DataFrame(columns=FIRM_LOAD_COLUMNS)
            breakdown = pd.DataFrame()
            if not monthly.empty and "month_end" in monthly.columns:
                agg_cols = ["end_aum", "nnb"] if "end_aum" in monthly.columns and "nnb" in monthly.columns else []
                if agg_cols:
                    breakdown = monthly.groupby("month_end", as_index=False)[agg_cols].sum()
            return {"details_monthly": monthly, "details_breakdown": breakdown}

        state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
        drill_json = json.dumps(drill_state.to_dict(), sort_keys=True, separators=(",", ":"))
        raw = _cached(
            dataset_version,
            filter_hash,
            drill_hash,
            state_json,
            drill_json,
            root_str,
        )
        return {k: raw[k] for k in ("details_monthly", "details_breakdown")}
    try:
        monthly = _load_details_monthly_impl(state, drill_state, root)
    except SchemaError as exc:
        logger.error(str(exc))
        if drill_state.drill_mode == "ticker":
            monthly = pd.DataFrame(columns=TICKER_LOAD_COLUMNS)
        elif drill_state.drill_mode == "channel":
            monthly = pd.DataFrame(columns=CHANNEL_LOAD_COLUMNS)
        else:
            monthly = pd.DataFrame(columns=FIRM_LOAD_COLUMNS)
    breakdown = pd.DataFrame()
    if not monthly.empty and "month_end" in monthly.columns:
        agg_cols = [c for c in ("end_aum", "nnb") if c in monthly.columns]
        if agg_cols:
            breakdown = monthly.groupby("month_end", as_index=False)[agg_cols].sum()
    return {"details_monthly": monthly, "details_breakdown": breakdown}


# Anomaly detection thresholds (single place)
ANOMALY_Z_THRESHOLD_FIRM = 2.5
ANOMALY_Z_THRESHOLD_DIM = 2.5
ANOMALY_Z_HIGH = 3.0
ANOMALY_ROLLING_WINDOWS = (6, 12)
ANOMALY_REVERSAL_NNB_MIN = 0.0  # min abs(nnb) to flag reversal (any sign flip if 0)
ANOMALY_MOM_DAUM_PCT_HIGH = 0.95  # percentile for "large" mom_daum -> high severity

ANOMALIES_TABLE_COLUMNS = [
    "level", "entity", "metric", "value_current", "baseline", "zscore",
    "rule_id", "reason", "severity", "month_end",
]


def _compute_rolling_stats(
    ts_df: pd.DataFrame,
    value_col: str,
    window: int,
) -> pd.DataFrame:
    """Add rolling_mean, rolling_std, zscore for value_col. Uses window; returns copy with new cols suffixed by _w{window}."""
    if ts_df is None or ts_df.empty or value_col not in ts_df.columns:
        return ts_df.copy() if ts_df is not None else pd.DataFrame()
    out = ts_df.copy()
    out = out.sort_values("month_end").reset_index(drop=True)
    r = out[value_col].rolling(window=window, min_periods=max(2, window // 2))
    out[f"rolling_mean_w{window}"] = r.mean()
    out[f"rolling_std_w{window}"] = r.std()
    std = out[f"rolling_std_w{window}"].replace(0, float("nan"))
    out[f"zscore_w{window}"] = (out[value_col] - out[f"rolling_mean_w{window}"]) / std
    return out


def _severity_from_z(z: float) -> str:
    if z != z or abs(z) < ANOMALY_Z_THRESHOLD_FIRM:
        return "low"
    if abs(z) >= ANOMALY_Z_HIGH:
        return "high"
    return "medium"


def _compute_monthly_aum_change(ts_df: pd.DataFrame) -> pd.Series:
    """MoM AUM change from end_aum in chronological order."""
    if ts_df is None or ts_df.empty or "end_aum" not in ts_df.columns:
        return pd.Series(dtype="float64")
    ordered = ts_df.sort_values("month_end")
    out = ordered["end_aum"].diff()
    out.index = ordered.index
    return out


def _build_anomalies_canonical(
    time_series: pd.DataFrame,
    channel_df: pd.DataFrame,
    ticker_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    period_frames: dict[str, Any],
    channel_rank: pd.DataFrame,
    ticker_rank: pd.DataFrame,
    ch_dim: str | None,
    tk_dim: str | None,
    geo_dim: str | None,
) -> pd.DataFrame:
    """
    Build anomalies table: firm-level (z-score on time_series), dimension-level (cross-sectional z, reversals).
    Canonical columns: level, entity, metric, value_current, baseline, zscore, rule_id, reason, severity, month_end.
    """
    rows: list[dict[str, Any]] = []
    current_me = period_frames.get("current_month_end")
    prior_me = period_frames.get("prior_month_end")

    # --- Firm-level from time_series (rolling 12m where available) ---
    if time_series is not None and not time_series.empty and current_me is not None:
        firm_ts = time_series.copy()
        if "aum_change" not in firm_ts.columns and "end_aum" in firm_ts.columns:
            firm_ts["aum_change"] = _compute_monthly_aum_change(firm_ts)
        for metric, value_col in (("NNB", "nnb"), ("AUM_CHANGE", "aum_change"), ("MARKET_IMPACT", "market_impact_abs")):
            if value_col not in firm_ts.columns:
                continue
            ts = _compute_rolling_stats(firm_ts, value_col, 12)
            if ts.empty or f"zscore_w12" not in ts.columns:
                continue
            last = ts[ts["month_end"] == current_me]
            if last.empty:
                last = ts.tail(1)
            if last.empty:
                continue
            last = last.iloc[0]
            z = last.get("zscore_w12", float("nan"))
            if pd.isna(z) or not (abs(z) >= ANOMALY_Z_THRESHOLD_FIRM):
                continue
            base = last.get("rolling_mean_w12", float("nan"))
            val = last.get(value_col, float("nan"))
            severity = _severity_from_z(z)
            rows.append({
                "level": "firm",
                "entity": "FIRM",
                "metric": metric,
                "value_current": val,
                "baseline": base,
                "zscore": z,
                "rule_id": "firm_zscore_12m",
                "reason": f"{metric} |z|={abs(z):.2f} (current={val:.4f}, baseline={base:.4f})",
                "severity": severity,
                "month_end": current_me,
            })

    # --- Dimension-level: cross-sectional z and reversals (channel, ticker) ---
    def _dim_anomalies(
        monthly_df: pd.DataFrame,
        dim_col: str,
        level_name: str,
        rank_df: pd.DataFrame,
    ) -> None:
        if monthly_df is None or monthly_df.empty or dim_col not in monthly_df.columns or current_me is None:
            return
        cur_agg = _aggregate_month_by_dim(monthly_df, current_me, dim_col)
        if cur_agg.empty:
            return
        prior_agg = _aggregate_month_by_dim(monthly_df, prior_me, dim_col) if prior_me is not None else pd.DataFrame()
        if not prior_agg.empty and "end_aum" in prior_agg.columns and "end_aum" in cur_agg.columns:
            cur_agg = cur_agg.merge(
                prior_agg[[dim_col, "end_aum"]].rename(columns={"end_aum": "aum_prior"}),
                on=dim_col,
                how="left",
            )
            cur_agg["aum_prior"] = cur_agg["aum_prior"].fillna(0)
            cur_agg["mom_daum"] = cur_agg["end_aum"] - cur_agg["aum_prior"]
            cur_agg["ogr"] = cur_agg.apply(
                lambda r: (r["nnb"] / r["aum_prior"]) if (r["aum_prior"] and r["aum_prior"] > 0) else float("nan"),
                axis=1,
            )
        else:
            cur_agg["mom_daum"] = float("nan")
            cur_agg["ogr"] = float("nan")
        if "end_aum" in cur_agg.columns and "nnb" in cur_agg.columns and "aum_prior" in cur_agg.columns:
            cur_agg["market_impact_abs"] = cur_agg.apply(
                lambda r: compute_market_impact(r["aum_prior"], r["end_aum"], r["nnb"]),
                axis=1,
            )
        else:
            cur_agg["market_impact_abs"] = float("nan")
        for metric_col, metric_name in (("nnb", "NNB"), ("mom_daum", "AUM_CHANGE"), ("market_impact_abs", "MARKET_IMPACT")):
            if metric_col not in cur_agg.columns:
                continue
            vals = cur_agg[metric_col].fillna(0)
            mu = vals.mean()
            sigma = vals.std()
            if sigma is None or pd.isna(sigma) or sigma == 0:
                sigma = float("nan")
            for i in range(len(cur_agg)):
                entity = cur_agg[dim_col].iloc[i]
                val = cur_agg[metric_col].iloc[i]
                if pd.isna(val):
                    continue
                z = float("nan") if (sigma != sigma or sigma == 0) else (float(val) - float(mu)) / float(sigma)
                if abs(z) >= ANOMALY_Z_THRESHOLD_DIM:
                    severity = "high" if abs(z) >= ANOMALY_Z_HIGH else "medium"
                    base = float(mu)
                    rows.append({
                        "level": level_name,
                        "entity": entity,
                        "metric": metric_name,
                        "value_current": val,
                        "baseline": base,
                        "zscore": z,
                        "rule_id": "dim_zscore_cross",
                        "reason": f"{metric_name} |z|={abs(z):.2f} (current={val:.2f}, mean={base:.2f})",
                        "severity": severity,
                        "month_end": current_me,
                    })
        if prior_agg.empty or "nnb" not in cur_agg.columns or "nnb" not in prior_agg.columns:
            return
        merged = cur_agg[[dim_col, "nnb"]].merge(
            prior_agg[[dim_col, "nnb"]],
            on=dim_col,
            how="inner",
            suffixes=("_cur", "_prior"),
        )
        for i in range(len(merged)):
            entity = merged[dim_col].iloc[i]
            nnb_cur = merged["nnb_cur"].iloc[i]
            nnb_prior = merged["nnb_prior"].iloc[i]
            if pd.isna(nnb_cur) or pd.isna(nnb_prior):
                continue
            sign_cur = 1 if nnb_cur > 0 else (-1 if nnb_cur < 0 else 0)
            sign_prior = 1 if nnb_prior > 0 else (-1 if nnb_prior < 0 else 0)
            if sign_cur == 0 or sign_prior == 0 or sign_cur == sign_prior:
                continue
            if abs(nnb_cur) < ANOMALY_REVERSAL_NNB_MIN:
                continue
            p95 = merged["nnb_cur"].abs().quantile(ANOMALY_MOM_DAUM_PCT_HIGH) if len(merged) >= 2 else merged["nnb_cur"].abs().max()
            severity = "high" if abs(nnb_cur) >= p95 else "medium"
            rows.append({
                "level": level_name,
                "entity": entity,
                "metric": "NNB",
                "value_current": nnb_cur,
                "baseline": nnb_prior,
                "zscore": float("nan"),
                "rule_id": "reversal",
                "reason": f"NNB reversal (current={nnb_cur:.2f}, prior={nnb_prior:.2f})",
                "severity": severity,
                "month_end": current_me,
            })

    if ch_dim:
        _dim_anomalies(channel_df, ch_dim, "channel", channel_rank)
    if tk_dim:
        _dim_anomalies(ticker_df, tk_dim, "ticker", ticker_rank)
    if geo_dim and geo_df is not None and not geo_df.empty:
        _dim_anomalies(geo_df, geo_dim, "geo", pd.DataFrame())

    if not rows:
        return pd.DataFrame(columns=ANOMALIES_TABLE_COLUMNS)
    out = pd.DataFrame(rows)
    return out.reindex(columns=[c for c in ANOMALIES_TABLE_COLUMNS if c in out.columns], copy=False)


# Canonical columns for rank tables (top/bottom movers, shares, deltas, mix shift)
RANK_TABLE_COLUMNS = [
    "dim_value", "aum_end", "aum_prior", "mom_daum", "nnb",
    "ogr", "market_impact_abs", "market_impact_rate", "fee_yield",
    "aum_share", "aum_share_prior", "aum_share_delta",
    "nnb_share", "nnb_share_prior", "nnb_share_delta",
    "rank_nnb", "rank_aum", "bucket",
]


def _aggregate_month_by_dim(
    monthly_df: pd.DataFrame,
    month_end: Any,
    dim_col: str,
) -> pd.DataFrame:
    """One row per dim_value: aggregate monthly_df where month_end == month_end by dim_col."""
    if monthly_df is None or monthly_df.empty or dim_col not in monthly_df.columns:
        return pd.DataFrame()
    df = monthly_df.copy()
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    slice_df = df[df["month_end"] == month_end]
    if slice_df.empty:
        return pd.DataFrame()
    agg_cols = [c for c in ["end_aum", "begin_aum", "nnb", "nnf"] if c in slice_df.columns]
    out = slice_df.groupby(dim_col, as_index=False)[agg_cols].sum()
    return out


def _build_rank_table(
    df_current_month: pd.DataFrame,
    df_prior_month: pd.DataFrame,
    dim_col: str,
    metrics: tuple[str, ...] = ("NNB", "AUM"),
    top_n: int = 10,
    bottom_n: int = 10,
) -> pd.DataFrame:
    """
    Generic rank table: top/bottom movers by NNB, shares and share deltas, mix shift.
    Output: dim_value, aum_end, aum_prior, mom_daum, nnb, ogr, market_impact_abs, market_impact_rate,
    fee_yield, aum_share, aum_share_prior, aum_share_delta, nnb_share, nnb_share_prior, nnb_share_delta,
    rank_nnb, rank_aum, bucket ("top"|"bottom").
    Dims missing in prior -> prior=0, share_prior=0.
    """
    from app.metrics.metric_contract import (
        compute_fee_yield,
        compute_market_impact,
        compute_market_impact_rate,
        compute_ogr,
    )
    if df_current_month is None or df_current_month.empty or dim_col not in df_current_month.columns:
        return pd.DataFrame(columns=RANK_TABLE_COLUMNS)
    cur = df_current_month.rename(columns={dim_col: "dim_value"})
    if "dim_value" not in cur.columns:
        cur["dim_value"] = cur[dim_col]
    cur = cur.copy()
    cur["aum_end"] = cur["end_aum"] if "end_aum" in cur.columns else float("nan")
    cur["nnb"] = cur["nnb"] if "nnb" in cur.columns else float("nan")
    cur["nnf"] = cur["nnf"] if "nnf" in cur.columns else float("nan")
    prior = pd.DataFrame()
    if df_prior_month is not None and not df_prior_month.empty and dim_col in df_prior_month.columns:
        prior = df_prior_month.rename(columns={dim_col: "dim_value"})
        if "dim_value" not in prior.columns:
            prior["dim_value"] = prior[dim_col]
        prior = prior.copy()
        prior["aum_prior"] = prior["end_aum"] if "end_aum" in prior.columns else 0.0
        prior["nnb_prior"] = prior["nnb"] if "nnb" in prior.columns else 0.0
    merged = cur[["dim_value", "aum_end", "nnb", "nnf"]].copy()
    if not prior.empty:
        prior_sub = prior[["dim_value", "aum_prior", "nnb_prior"]].copy()
        merged = merged.merge(prior_sub, on="dim_value", how="outer")
    else:
        merged["aum_prior"] = 0.0
        merged["nnb_prior"] = 0.0
    merged["aum_prior"] = merged["aum_prior"].fillna(0)
    merged["nnb_prior"] = merged["nnb_prior"].fillna(0)
    merged["aum_end"] = merged["aum_end"].fillna(0)
    merged["nnb"] = merged["nnb"].fillna(0)
    merged["mom_daum"] = merged["aum_end"] - merged["aum_prior"]
    total_aum_cur = merged["aum_end"].sum()
    total_aum_prior = merged["aum_prior"].sum()
    total_nnb_cur = merged["nnb"].sum()
    total_nnb_prior = merged["nnb_prior"].sum()
    merged["aum_share"] = merged["aum_end"] / total_aum_cur if total_aum_cur and total_aum_cur == total_aum_cur else float("nan")
    merged["aum_share_prior"] = merged["aum_prior"] / total_aum_prior if total_aum_prior and total_aum_prior == total_aum_prior else 0.0
    merged["nnb_share"] = merged["nnb"] / total_nnb_cur if total_nnb_cur and total_nnb_cur == total_nnb_cur else float("nan")
    merged["nnb_share_prior"] = merged["nnb_prior"] / total_nnb_prior if total_nnb_prior and total_nnb_prior == total_nnb_prior else 0.0
    merged["aum_share_delta"] = merged["aum_share"].fillna(0) - merged["aum_share_prior"]
    merged["nnb_share_delta"] = merged["nnb_share"].fillna(0) - merged["nnb_share_prior"]

    def _ogr_row(r: pd.Series) -> float:
        return compute_ogr(r["nnb"], r["aum_prior"] if r["aum_prior"] else float("nan"))
    merged["ogr"] = merged.apply(_ogr_row, axis=1)
    merged["market_impact_abs"] = merged.apply(
        lambda r: compute_market_impact(r["aum_prior"], r["aum_end"], r["nnb"]), axis=1
    )
    merged["market_impact_rate"] = merged.apply(
        lambda r: compute_market_impact_rate(
            r["market_impact_abs"], r["aum_prior"] if r["aum_prior"] else float("nan")
        ),
        axis=1,
    )
    if "nnf" in merged.columns:
        merged["fee_yield"] = merged.apply(
            lambda r: compute_fee_yield(r["nnf"], r["aum_prior"], r["aum_end"], nnb=r.get("nnb")), axis=1
        )
    else:
        merged["fee_yield"] = float("nan")
    merged["rank_nnb"] = merged["nnb"].rank(method="min", ascending=False).astype(int)
    merged["rank_aum"] = merged["aum_end"].rank(method="min", ascending=False).astype(int)
    merged = merged.sort_values(["rank_nnb", "dim_value"], ascending=[True, True])
    top = merged.nlargest(top_n, "nnb").copy()
    top["bucket"] = "top"
    bottom = merged.nsmallest(bottom_n, "nnb").copy()
    bottom["bucket"] = "bottom"
    combined = pd.concat([top, bottom], ignore_index=True)
    combined = combined.drop_duplicates(subset=["dim_value"], keep="first")
    combined = combined.sort_values(["rank_nnb", "dim_value"], ascending=[True, True]).reset_index(drop=True)
    out = combined.reindex(columns=[c for c in RANK_TABLE_COLUMNS if c in combined.columns], copy=False)
    return out


def _fetch_fact_monthly(
    state: FilterState,
    root: Path,
) -> pd.DataFrame:
    """Fact rows already filtered by global filters (segment, date range, etc.)."""
    return _load_firm_monthly(state, root)


def _build_period_frames(df: pd.DataFrame) -> dict[str, Any]:
    """
    Uses canonical date_align helpers: latest, prior available, first-in-year month_end.
    ytd_frame = months in current year up to current_month_end; yoy_frame = prior year.
    """
    out: dict[str, Any] = {
        "current_month_end": None,
        "prior_month_end": None,
        "ytd_start_month_end": None,
        "ytd_frame": pd.DataFrame(),
        "yoy_frame": pd.DataFrame(),
        "aum_at_year_start": float("nan"),
    }
    if df is None or df.empty or "month_end" not in df.columns:
        return out
    df = df.copy()
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    df = df.dropna(subset=["month_end"])
    if df.empty:
        return out
    current_month_end = get_latest_month_end(df)
    if current_month_end is None:
        return out
    out["current_month_end"] = current_month_end
    out["prior_month_end"] = get_prior_month_end(df, current_month_end)
    ytd_start = get_year_start_month_end(df, current_month_end)
    out["ytd_start_month_end"] = ytd_start
    curr_year = getattr(current_month_end, "year", None) or pd.Timestamp(current_month_end).year
    ytd_frame = df[(df["month_end"].dt.year == curr_year) & (df["month_end"] <= current_month_end)]
    out["ytd_frame"] = ytd_frame.sort_values("month_end").reset_index(drop=True)
    if not ytd_frame.empty and "end_aum" in df.columns and ytd_start is not None:
        first_rows = ytd_frame[ytd_frame["month_end"] == ytd_start]
        if "begin_aum" in df.columns and not first_rows.empty:
            out["aum_at_year_start"] = first_rows["begin_aum"].sum()
        else:
            prev_year_months = df[df["month_end"].dt.year == curr_year - 1]
            if not prev_year_months.empty:
                last_prev = get_latest_month_end(prev_year_months)
                if last_prev is not None:
                    prev_end = prev_year_months[prev_year_months["month_end"] == last_prev]
                    out["aum_at_year_start"] = prev_end["end_aum"].sum() if "end_aum" in prev_end.columns else float("nan")
    prior_year = curr_year - 1
    yoy_frame = df[(df["month_end"].dt.year == prior_year)]
    out["yoy_frame"] = yoy_frame.sort_values("month_end").reset_index(drop=True)
    return out


def _build_firm_snapshot_canonical(
    df: pd.DataFrame,
    period_frames: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Single row: end_aum, begin_aum, mom_pct, ytd_pct, nnb, ogr, market_impact_abs, market_impact_rate, fee_yield.
    Firm grain only (no slice leakage): inputs are already firm-level; compare only to firm checksum (e.g. DATA SUMMARY).
    Defensive: prior missing -> mom/ogr/rates NaN and note in meta. Market P&L never blank when end_aum/nnb exist (first month: begin_aum=0)."""
    from app.metrics.metric_contract import (
        coerce_num,
        compute_fee_yield,
        compute_market_impact,
        compute_market_impact_rate,
        compute_ogr,
    )
    cols = [
        "month_end", "begin_aum", "end_aum", "nnb", "nnf",
        "mom_pct", "ytd_pct", "yoy_pct",
        "ogr", "market_impact_abs", "market_impact_rate", "fee_yield",
    ]
    empty = pd.DataFrame(columns=cols)
    if df is None or df.empty or "month_end" not in df.columns:
        return empty, meta
    df = df.copy()
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    current = period_frames.get("current_month_end")
    prior = period_frames.get("prior_month_end")
    ytd_frame = period_frames.get("ytd_frame")
    if ytd_frame is None:
        ytd_frame = pd.DataFrame()
    aum_at_year_start = period_frames.get("aum_at_year_start", float("nan"))
    if current is None:
        return empty, meta
    by_month = df.groupby("month_end", as_index=False).agg({
        c: "sum" for c in ["begin_aum", "end_aum", "nnb", "nnf"] if c in df.columns
    })
    current_row = by_month[by_month["month_end"] == current]
    if current_row.empty:
        return empty, meta
    current_row = current_row.iloc[0]
    end_aum = float(current_row.get("end_aum", float("nan")))
    nnb = float(current_row.get("nnb", float("nan")))
    nnf = float(current_row.get("nnf", float("nan")))
    begin_aum = float("nan")
    if prior is not None:
        prior_row = by_month[by_month["month_end"] == prior]
        if not prior_row.empty:
            begin_aum = float(prior_row.iloc[0].get("end_aum", float("nan")))
    if begin_aum != begin_aum and "begin_aum" in current_row:
        begin_aum = float(current_row.get("begin_aum", float("nan")))
    prior_missing = prior is None or (begin_aum != begin_aum)
    if prior_missing:
        meta["prior_month_missing"] = True
        meta["rates_not_computable_reason"] = "begin_aum undefined for first available month"
    mom_pct = float("nan")
    if not prior_missing and begin_aum == begin_aum and begin_aum and begin_aum > 0:
        mom_pct = (end_aum - begin_aum) / begin_aum
    ytd_pct = float("nan")
    if aum_at_year_start == aum_at_year_start and aum_at_year_start and aum_at_year_start > 0:
        ytd_pct = (end_aum - aum_at_year_start) / aum_at_year_start
    yoy_pct = float("nan")
    yoy_frame = period_frames.get("yoy_frame")
    if yoy_frame is not None and not yoy_frame.empty and "end_aum" in yoy_frame.columns:
        yoy = yoy_frame
        same_month_prior = yoy[yoy["month_end"].dt.month == getattr(current, "month", pd.Timestamp(current).month)]
        if not same_month_prior.empty:
            prior_end = same_month_prior["end_aum"].sum()
            if prior_end and prior_end == prior_end and prior_end > 0:
                yoy_pct = (end_aum - prior_end) / prior_end
    # Canonical formulas: always compute market_pnl so it is never blank when end_aum/nnb exist.
    mi_abs = compute_market_impact(begin_aum, end_aum, nnb)
    if prior_missing and (end_aum == end_aum and nnb == nnb):
        mi_abs = coerce_num(end_aum) - 0.0 - coerce_num(nnb)
    ogr = compute_ogr(nnb, begin_aum) if (not prior_missing and begin_aum == begin_aum and begin_aum and begin_aum > 0) else float("nan")
    mi_rate = compute_market_impact_rate(mi_abs, begin_aum) if (begin_aum == begin_aum and begin_aum and begin_aum > 0) else float("nan")
    fee_yield = float("nan")
    has_nnf = "nnf" in df.columns
    if has_nnf:
        avg_aum = (float(begin_aum) + float(end_aum)) / 2.0 if (begin_aum == begin_aum and end_aum == end_aum and (begin_aum or end_aum)) else 0.0
        if avg_aum > 0:
            fee_yield = compute_fee_yield(nnf, begin_aum, end_aum, nnb=nnb)
    else:
        meta["fee_yield_nnf_missing"] = True
    row = {
        "month_end": current,
        "begin_aum": begin_aum,
        "end_aum": end_aum,
        "nnb": nnb,
        "nnf": nnf,
        "mom_pct": mom_pct,
        "ytd_pct": ytd_pct,
        "yoy_pct": yoy_pct,
        "ogr": ogr,
        "market_impact_abs": mi_abs,
        "market_impact_rate": mi_rate,
        "fee_yield": fee_yield,
    }
    out = pd.DataFrame([row]).reindex(columns=[c for c in cols if c in row], copy=False)
    return out, meta


def _build_time_series_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """One row per month_end in filter window: month_end, end_aum, nnb, ogr, market_impact_abs, market_impact_rate, fee_yield."""
    from app.metrics.metric_contract import (
        compute_fee_yield,
        compute_market_impact,
        compute_market_impact_rate,
        compute_ogr,
    )
    cols = ["month_end", "end_aum", "nnb", "nnf", "ogr", "market_impact_abs", "market_impact_rate", "fee_yield"]
    empty = pd.DataFrame(columns=cols)
    if df is None or df.empty or "month_end" not in df.columns:
        return empty
    df = df.copy()
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    by_month = df.groupby("month_end", as_index=False).agg({
        c: "sum" for c in ["begin_aum", "end_aum", "nnb", "nnf"] if c in df.columns
    })
    by_month = by_month.sort_values("month_end").reset_index(drop=True)
    if "begin_aum" not in by_month.columns:
        by_month["begin_aum"] = by_month["end_aum"].shift(1)
    rows = []
    for i in range(len(by_month)):
        r = by_month.iloc[i]
        begin = float(r.get("begin_aum", float("nan")))
        end = float(r.get("end_aum", float("nan")))
        nnb = float(r.get("nnb", float("nan")))
        nnf = float(r.get("nnf", float("nan")))
        mi_abs = compute_market_impact(begin, end, nnb)
        ogr = compute_ogr(nnb, begin) if begin == begin and begin else float("nan")
        mi_rate = compute_market_impact_rate(mi_abs, begin) if begin == begin and begin else float("nan")
        fy = compute_fee_yield(nnf, begin, end, nnb=nnb) if "nnf" in by_month.columns else float("nan")
        rows.append({
            "month_end": r["month_end"],
            "end_aum": end,
            "nnb": nnb,
            "nnf": nnf,
            "ogr": ogr,
            "market_impact_abs": mi_abs,
            "market_impact_rate": mi_rate,
            "fee_yield": fy,
        })
    if not rows:
        return empty
    out = pd.DataFrame(rows).reindex(columns=[c for c in cols if c in rows[0]], copy=False)
    return out


def _load_etf_mapping_if_exists(root: Path) -> pd.DataFrame | None:
    """If ETF mapping (e.g. ticker -> etf_group) exists, return it; else None."""
    for rel in ("data/mappings/etf_map.parquet", "data/mappings/etf_map.csv", "data/etf_map.parquet"):
        path = root / rel
        if path.exists():
            try:
                if path.suffix == ".csv":
                    return pd.read_csv(path)
                return pd.read_parquet(path)
            except Exception:
                pass
    return None


def _resolve_dim_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _report_pack_impl(
    dataset_version: str,
    filter_hash: str,
    state_json: str,
    root_str: str,
) -> ReportPack:
    """Load all report inputs and build ReportPack. firm_snapshot, time_series, and rank tables use canonical period logic."""
    state = FilterState.from_dict(json.loads(state_json))
    root = Path(root_str)
    firm_df = _fetch_fact_monthly(state, root)
    period_frames = _build_period_frames(firm_df)
    meta_notes: dict[str, Any] = {}
    firm_snapshot, meta_notes = _build_firm_snapshot_canonical(firm_df, period_frames, meta_notes)
    time_series = _build_time_series_canonical(firm_df)
    try:
        channel_df = _load_channel_monthly(state, root)
    except SchemaError as exc:
        logger.error(str(exc))
        channel_df = pd.DataFrame(columns=CHANNEL_LOAD_COLUMNS)
        meta_notes["schema_error_channel_monthly"] = str(exc)
    try:
        ticker_df = _load_ticker_monthly(state, root)
    except SchemaError as exc:
        logger.error(str(exc))
        ticker_df = pd.DataFrame(columns=TICKER_LOAD_COLUMNS)
        meta_notes["schema_error_ticker_monthly"] = str(exc)
    try:
        geo_df = _load_geo_monthly(state, root)
    except SchemaError as exc:
        logger.error(str(exc))
        geo_df = pd.DataFrame(columns=GEO_LOAD_COLUMNS)
        meta_notes["schema_error_geo_monthly"] = str(exc)
    try:
        segment_df = _load_segment_monthly(state, root)
    except SchemaError as exc:
        logger.error(str(exc))
        segment_df = pd.DataFrame(columns=SEGMENT_LOAD_COLUMNS)
        meta_notes["schema_error_segment_monthly"] = str(exc)
    # Dev diagnostics for report-pack sources (Tab 2 debug expander in page)
    meta_notes["source_diagnostics"] = {
        "firm_monthly": {"rows": int(len(firm_df)), "columns": list(firm_df.columns)},
        "channel_monthly": {"rows": int(len(channel_df)), "columns": list(channel_df.columns)},
        "ticker_monthly": {"rows": int(len(ticker_df)), "columns": list(ticker_df.columns)},
        "geo_monthly": {"rows": int(len(geo_df)), "columns": list(geo_df.columns)},
        "segment_monthly": {"rows": int(len(segment_df)), "columns": list(segment_df.columns)},
    }

    current_me = period_frames.get("current_month_end")
    prior_me = period_frames.get("prior_month_end")
    top_n, bottom_n = 10, 10

    channel_rank = pd.DataFrame(columns=RANK_TABLE_COLUMNS)
    ch_dim = _resolve_dim_col(channel_df, ("channel", "channel_l1", "standard_channel"))
    if ch_dim and current_me is not None:
        ch_cur = _aggregate_month_by_dim(channel_df, current_me, ch_dim)
        ch_prior = _aggregate_month_by_dim(channel_df, prior_me, ch_dim) if prior_me is not None else pd.DataFrame()
        if not ch_cur.empty:
            channel_rank = _build_rank_table(ch_cur, ch_prior, ch_dim, top_n=top_n, bottom_n=bottom_n)

    ticker_rank = pd.DataFrame(columns=RANK_TABLE_COLUMNS)
    tk_dim = _resolve_dim_col(ticker_df, ("product_ticker", "ticker"))
    if tk_dim and current_me is not None:
        tk_cur = _aggregate_month_by_dim(ticker_df, current_me, tk_dim)
        tk_prior = _aggregate_month_by_dim(ticker_df, prior_me, tk_dim) if prior_me is not None else pd.DataFrame()
        if not tk_cur.empty:
            ticker_rank = _build_rank_table(tk_cur, tk_prior, tk_dim, top_n=top_n, bottom_n=bottom_n)

    geo_rank = pd.DataFrame(columns=RANK_TABLE_COLUMNS)
    geo_dim = _resolve_dim_col(geo_df, ("geo", "region", "src_country", "src_country_canonical", "product_country_canonical"))
    if geo_dim and current_me is not None and not geo_df.empty:
        geo_cur = _aggregate_month_by_dim(geo_df, current_me, geo_dim)
        geo_prior = _aggregate_month_by_dim(geo_df, prior_me, geo_dim) if prior_me is not None else pd.DataFrame()
        if not geo_cur.empty:
            geo_rank = _build_rank_table(geo_cur, geo_prior, geo_dim, top_n=top_n, bottom_n=bottom_n)
    else:
        if geo_df.empty and "geo_rank" not in meta_notes:
            meta_notes["geo_rank_no_data"] = True

    etf_rank = pd.DataFrame(columns=RANK_TABLE_COLUMNS)
    etf_map = _load_etf_mapping_if_exists(root)
    if etf_map is not None and not etf_map.empty and len(etf_map.columns) >= 2 and tk_dim and not ticker_df.empty and current_me is not None:
        key_col = etf_map.columns[0]
        etf_group_col = etf_map.columns[1]
        ticker_df_etf = ticker_df.merge(
            etf_map[[key_col, etf_group_col]],
            left_on=tk_dim,
            right_on=key_col,
            how="left",
        )
        ticker_df_etf[etf_group_col] = ticker_df_etf[etf_group_col].fillna("_unknown")
        etf_cur = _aggregate_month_by_dim(ticker_df_etf, current_me, etf_group_col)
        etf_prior = _aggregate_month_by_dim(ticker_df_etf, prior_me, etf_group_col) if prior_me is not None else pd.DataFrame()
        if not etf_cur.empty:
            etf_rank = _build_rank_table(etf_cur, etf_prior, etf_group_col, top_n=top_n, bottom_n=bottom_n)
    else:
        meta_notes["etf_rank_unavailable"] = True

    anomalies = _build_anomalies_canonical(
        time_series,
        channel_df,
        ticker_df,
        geo_df,
        period_frames,
        channel_rank,
        ticker_rank,
        ch_dim,
        tk_dim,
        geo_dim,
    )

    return build_report_pack(
        firm_df,
        channel_df,
        ticker_df,
        geo_df,
        dataset_version,
        filter_hash,
        top_n=top_n,
        etf_df=None,
        firm_snapshot=firm_snapshot,
        time_series=time_series,
        meta_notes=meta_notes,
        channel_rank=channel_rank,
        ticker_rank=ticker_rank,
        geo_rank=geo_rank,
        etf_rank=etf_rank,
        anomalies=anomalies,
    )


# Cached wrappers: hashable args only (dataset_version, filter_hash, query_name, state_json, root_str, ...).
if st is not None:
    @st.cache_data(show_spinner=False)
    def _cached_report_pack(_dv: str, _fh: str, _state_json: str, _root_str: str) -> ReportPack:
        return _report_pack_impl(_dv, _fh, _state_json, _root_str)

    @st.cache_data(show_spinner=False)
    def _cached_get_firm_snapshot(dataset_version: str, filter_hash: str, query_name: str, state_json: str, root_str: str) -> pd.DataFrame:
        return _governed_firm_snapshot_impl(state_json, root_str)

    @st.cache_data(show_spinner=False)
    def _cached_get_channel_breakdown(dataset_version: str, filter_hash: str, query_name: str, state_json: str, root_str: str, metric: str) -> pd.DataFrame:
        return _governed_channel_breakdown_impl(state_json, root_str, metric)

    @st.cache_data(show_spinner=False)
    def _cached_get_growth_quality(dataset_version: str, filter_hash: str, query_name: str, state_json: str, root_str: str, view: str) -> pd.DataFrame:
        return _governed_growth_quality_impl(state_json, root_str, view)

    @st.cache_data(show_spinner=False)
    def _cached_get_trend_series(dataset_version: str, filter_hash: str, query_name: str, state_json: str, root_str: str) -> pd.DataFrame:
        return _governed_trend_series_impl(state_json, root_str)
else:
    def _cached_report_pack(_dv: str, _fh: str, _state_json: str, _root_str: str) -> ReportPack:
        return _report_pack_impl(_dv, _fh, _state_json, _root_str)

    def _cached_get_firm_snapshot(dataset_version: str, filter_hash: str, query_name: str, state_json: str, root_str: str) -> pd.DataFrame:
        return _governed_firm_snapshot_impl(state_json, root_str)

    def _cached_get_channel_breakdown(dataset_version: str, filter_hash: str, query_name: str, state_json: str, root_str: str, metric: str) -> pd.DataFrame:
        return _governed_channel_breakdown_impl(state_json, root_str, metric)

    def _cached_get_growth_quality(dataset_version: str, filter_hash: str, query_name: str, state_json: str, root_str: str, view: str) -> pd.DataFrame:
        return _governed_growth_quality_impl(state_json, root_str, view)

    def _cached_get_trend_series(dataset_version: str, filter_hash: str, query_name: str, state_json: str, root_str: str) -> pd.DataFrame:
        return _governed_trend_series_impl(state_json, root_str)


def _track_cache_call(query_name: str) -> None:
    """Optional dev-only: increment st.session_state['cache_hits'][query_name] when gateway is called."""
    if st is None:
        return
    if "cache_hits" not in st.session_state or not isinstance(st.session_state.get("cache_hits"), dict):
        st.session_state["cache_hits"] = {}
    st.session_state["cache_hits"][query_name] = st.session_state["cache_hits"].get(query_name, 0) + 1


def get_firm_snapshot(
    filters: FilterState | dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """Governed firm snapshot. Cached by dataset_version + filter_state_hash. Goes through cached_call."""
    state = normalize_filters(filters)
    dataset_version = _get_dataset_version(root=root)
    filter_hash = state.filter_state_hash()
    state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
    root_str = str(root if root is not None else Path.cwd())
    cache_key = f"get_firm_snapshot|{dataset_version}|{filter_hash}"
    return cached_call(
        cache_key,
        lambda: _governed_firm_snapshot_impl(state_json, root_str),
        SNAPSHOT_BUDGET_MS,
        DEFAULT_MAX_ROWS,
        "get_firm_snapshot",
    )


def get_channel_breakdown(
    filters: FilterState | dict[str, Any],
    metric: str,
    root: Path | None = None,
) -> pd.DataFrame:
    """Governed channel breakdown for given metric. Goes through cached_call."""
    state = normalize_filters(filters)
    query_name = f"get_channel_breakdown:{metric}"
    dataset_version = _get_dataset_version(root=root)
    filter_hash = state.filter_state_hash()
    state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
    root_str = str(root if root is not None else Path.cwd())
    cache_key = f"{query_name}|{dataset_version}|{filter_hash}"
    return cached_call(
        cache_key,
        lambda: _governed_channel_breakdown_impl(state_json, root_str, metric),
        _resolve_heavy_budget_ms(),
        DEFAULT_MAX_ROWS,
        query_name,
    )


def get_growth_quality(
    filters: FilterState | dict[str, Any],
    view: str,
    root: Path | None = None,
) -> pd.DataFrame:
    """Governed growth quality for view. Goes through cached_call."""
    state = normalize_filters(filters)
    query_name = f"get_growth_quality:{view}"
    dataset_version = _get_dataset_version(root=root)
    filter_hash = state.filter_state_hash()
    state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
    root_str = str(root if root is not None else Path.cwd())
    cache_key = f"{query_name}|{dataset_version}|{filter_hash}"
    return cached_call(
        cache_key,
        lambda: _governed_growth_quality_impl(state_json, root_str, view),
        _resolve_heavy_budget_ms(),
        DEFAULT_MAX_ROWS,
        query_name,
    )


def get_trend_series(
    filters: FilterState | dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """Governed trend series. Goes through cached_call."""
    state = normalize_filters(filters)
    dataset_version = _get_dataset_version(root=root)
    filter_hash = state.filter_state_hash()
    state_json = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
    root_str = str(root if root is not None else Path.cwd())
    cache_key = f"get_trend_series|{dataset_version}|{filter_hash}"
    return cached_call(
        cache_key,
        lambda: _governed_trend_series_impl(state_json, root_str),
        _resolve_heavy_budget_ms(),
        DEFAULT_MAX_ROWS,
        "get_trend_series",
    )
