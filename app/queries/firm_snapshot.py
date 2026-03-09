"""
Single firm snapshot query powering the KPI strip and narrative (global slice only).
Pulls from the governed firm dataset (DuckDB analytics.v_firm_monthly or fallback
data/agg/firm_monthly.parquet). No UI aggregation or extra filters; global slice only.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data_gateway import get_config, query_df
from app.date_align import (
    get_latest_month_end,
    get_prior_month_end,
    get_year_start_month_end,
    has_month_gaps,
    is_single_month,
)
from app.metrics.metric_contract import (
    coerce_num,
    compute_market_impact,
    compute_market_impact_rate,
    compute_ogr,
)
from app.metrics.reconciliation import (
    format_no_data_panel,
    reconcile_waterfall_from_contract,
)

# Columns selected from firm monthly (must match governed schema)
_FIRM_REQUIRED_COLUMNS = (
    "month_end",
    "begin_aum",
    "end_aum",
    "nnb",
    "nnf",
    "ogr",
    "market_impact_rate",
)
_FIRM_OPTIONAL_COLUMNS = ("market_pnl", "market_impact", "fee_yield")
_FIRM_COLUMNS = _FIRM_REQUIRED_COLUMNS + _FIRM_OPTIONAL_COLUMNS
_FIRM_NUMERIC_COLUMNS = tuple(c for c in _FIRM_COLUMNS if c != "month_end")
_MONTH_END_ALIASES = ("date", "month", "month_end_date", "period", "as_of")


def _normalize_firm_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "month_end" not in out.columns:
        lower_to_actual = {str(c).strip().lower(): c for c in out.columns}
        for alias in _MONTH_END_ALIASES:
            actual = lower_to_actual.get(alias.lower())
            if actual:
                out = out.rename(columns={actual: "month_end"})
                break
    missing = [c for c in _FIRM_REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        return pd.DataFrame(columns=list(_FIRM_COLUMNS))
    for c in _FIRM_OPTIONAL_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out.reindex(columns=list(_FIRM_COLUMNS), copy=False)


def safe_divide(a: float | None, b: float | None) -> float:
    """
    Deterministic division for KPI math. Returns np.nan if b is None or 0,
    or if the result is inf/-inf.
    """
    if b is None:
        return np.nan
    try:
        bf = float(b)
    except (TypeError, ValueError):
        return np.nan
    if bf == 0 or math.isnan(bf):
        return np.nan
    try:
        af = float(a) if a is not None else np.nan
    except (TypeError, ValueError):
        af = np.nan
    out = af / bf
    if math.isinf(out):
        return np.nan
    return out


def coerce_numeric_series(df: pd.DataFrame, cols: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """
    Force the given columns to numeric where possible; keep NaNs. Returns df (mutates or returns copy).
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        try:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        except Exception:
            pass
    return out


def _replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN in all numeric columns. No Streamlit."""
    out = df.copy()
    for c in out.columns:
        try:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = s.replace([float("inf"), float("-inf")], np.nan)
        except Exception:
            pass
    return out


def _apply_canonical_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add canonical derived columns using app.metrics.metric_contract. Do not overwrite
    ogr/market_impact_rate if already present; fill only when missing or all null.
    """
    if df is None or df.empty:
        return df
    out = coerce_numeric_series(df, list(_FIRM_NUMERIC_COLUMNS))
    out = _replace_inf_with_nan(out)
    # Required for derived: begin_aum, end_aum, nnb
    for col in ("begin_aum", "end_aum", "nnb"):
        if col not in out.columns:
            return out
    # 1) market_impact (currency) = end - begin - nnb
    out["market_impact"] = out.apply(
        lambda r: compute_market_impact(r.get("begin_aum"), r.get("end_aum"), r.get("nnb")),
        axis=1,
    )
    # 2) ogr: only if missing or all null
    if "ogr" not in out.columns or out["ogr"].isna().all():
        out["ogr"] = out.apply(
            lambda r: compute_ogr(r.get("nnb"), r.get("begin_aum")),
            axis=1,
        )
    # 3) market_impact_rate: only if missing or all null
    if "market_impact_rate" not in out.columns or out["market_impact_rate"].isna().all():
        out["market_impact_rate"] = out.apply(
            lambda r: compute_market_impact_rate(r.get("market_impact"), r.get("begin_aum")),
            axis=1,
        )
    return out


def load_firm_monthly_last_n(
    months: int = 24,
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Load the last N months of firm-level data from the governed layer (global slice only).
    Preferred: DuckDB analytics.v_firm_monthly with ORDER BY month_end DESC LIMIT N,
    then sorted ascending in pandas. Fallback: data/agg/firm_monthly.parquet, sorted
    then tail(N). Always returns a DataFrame sorted by month_end ascending.
    """
    root = Path(root) if root is not None else Path.cwd()
    # 1) DuckDB
    try:
        config = get_config(root)
        schema = config.get("schema", "analytics")
        sql = f'SELECT * FROM "{schema}"."v_firm_monthly" ORDER BY "month_end" DESC LIMIT ?'
        df = query_df(sql, params=[months], _config=config)
        if df is None or df.empty:
            return pd.DataFrame(columns=list(_FIRM_COLUMNS))
        df = _normalize_firm_schema(df)
        if df.empty:
            return df
        if "month_end" in df.columns:
            df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
        return df
    except Exception:
        pass
    # 2) Parquet fallback
    parquet_path = root / "data" / "agg" / "firm_monthly.parquet"
    if not parquet_path.exists():
        return pd.DataFrame(columns=list(_FIRM_COLUMNS))
    try:
        df = pd.read_parquet(parquet_path, columns=list(_FIRM_COLUMNS))
    except Exception:
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            return pd.DataFrame(columns=list(_FIRM_COLUMNS))
    df = _normalize_firm_schema(df)
    if df.empty:
        return df
    if "month_end" not in df.columns:
        return pd.DataFrame(columns=list(_FIRM_COLUMNS))
    df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    df = df.tail(months).reset_index(drop=True)
    return df


def compute_context_months(df: pd.DataFrame) -> dict[str, Any]:
    """
    Canonical context months using app.date_align helpers (gap-aware).
    Returns latest_month_end, prev_month_end, ytd_start_month_end; None where missing.
    """
    out: dict[str, Any] = {
        "latest_month_end": None,
        "prev_month_end": None,
        "ytd_start_month_end": None,
    }
    if df is None or df.empty or "month_end" not in df.columns:
        return out
    latest = get_latest_month_end(df)
    if latest is None:
        return out
    out["latest_month_end"] = latest
    out["prev_month_end"] = get_prior_month_end(df, latest)
    out["ytd_start_month_end"] = get_year_start_month_end(df, latest)
    return out


def _scalar_at_month(df: pd.DataFrame, month_end: Any, col: str) -> float:
    """Value of col in the row where month_end matches; np.nan if missing or invalid."""
    if df is None or df.empty or col not in df.columns or month_end is None:
        return np.nan
    try:
        mask = pd.to_datetime(df["month_end"], errors="coerce") == pd.Timestamp(month_end)
        if not mask.any():
            return np.nan
        val = df.loc[mask, col].iloc[0]
        return float(pd.to_numeric(val, errors="coerce")) if pd.notna(val) else np.nan
    except Exception:
        return np.nan


def compute_kpi_raw(df: pd.DataFrame, context: dict[str, Any]) -> dict[str, float]:
    """
    Compute raw KPI values (numbers only) from firm monthly df and context.
    df must be sorted ascending by month_end. Uses safe_divide for growth rates;
    never throws on missing values. Returns dict with raw numeric values only.
    """
    nan = np.nan
    out: dict[str, float] = {
        "end_aum": nan,
        "mom_growth": nan,
        "ytd_growth": nan,
        "nnb": nan,
        "nnf": nan,
        "ogr": nan,
        "market_impact": nan,
        "market_pnl": nan,
    }
    if df is None or df.empty:
        return out
    df = coerce_numeric_series(df, list(_FIRM_NUMERIC_COLUMNS))
    if "month_end" not in df.columns:
        return out
    df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    latest_month = context.get("latest_month_end")
    prev_month = context.get("prev_month_end")
    ytd_start_month = context.get("ytd_start_month_end")

    # Latest row values
    end_aum_latest = _scalar_at_month(df, latest_month, "end_aum")
    out["end_aum"] = end_aum_latest if not math.isnan(end_aum_latest) else nan
    out["nnb"] = _scalar_at_month(df, latest_month, "nnb")
    out["nnf"] = _scalar_at_month(df, latest_month, "nnf")
    out["ogr"] = _scalar_at_month(df, latest_month, "ogr")
    out["market_impact"] = _scalar_at_month(df, latest_month, "market_impact_rate")
    out["market_pnl"] = _scalar_at_month(df, latest_month, "market_pnl")

    # mom_growth: (end_aum_latest - end_aum_prev) / end_aum_prev; if prev missing -> NaN
    if prev_month is not None:
        end_aum_prev = _scalar_at_month(df, prev_month, "end_aum")
        out["mom_growth"] = safe_divide(end_aum_latest - end_aum_prev, end_aum_prev)
    else:
        out["mom_growth"] = nan

    # ytd_growth: (end_aum_latest - end_aum_ytd_start) / end_aum_ytd_start; if ytd_start missing -> NaN
    if ytd_start_month is not None:
        end_aum_ytd_start = _scalar_at_month(df, ytd_start_month, "end_aum")
        out["ytd_growth"] = safe_divide(end_aum_latest - end_aum_ytd_start, end_aum_ytd_start)
    else:
        out["ytd_growth"] = nan

    # Coerce any inf to nan
    for k in out:
        try:
            if math.isinf(out[k]):
                out[k] = nan
        except Exception:
            pass
    return out


def _is_na(x: float | None) -> bool:
    if x is None:
        return True
    try:
        return isinstance(x, float) and math.isnan(x)
    except (TypeError, ValueError):
        return True


def format_currency(x: float | None) -> str:
    """NaN/None -> "—"; else format by magnitude as $12.3B, $123.4M, $1.2K or $123.45."""
    if _is_na(x):
        return "—"
    try:
        v = float(x)
        if math.isinf(v):
            return "—"
        av = abs(v)
        if av >= 1e9:
            return f"${v / 1e9:,.2f}B"
        if av >= 1e6:
            return f"${v / 1e6:,.2f}M"
        if av >= 1e3:
            return f"${v / 1e3:,.2f}K"
        return f"${v:,.2f}"
    except (TypeError, ValueError):
        return "—"


def format_percent(x: float | None) -> str:
    """NaN/None -> "—"; else format as 2.31% (x is decimal, e.g. 0.0231)."""
    if _is_na(x):
        return "—"
    try:
        v = float(x)
        if math.isinf(v):
            return "—"
        return f"{v * 100:,.2f}%"
    except (TypeError, ValueError):
        return "—"


def _status_growth(v: float | None) -> str:
    """Growth-style metrics: >0 good, <0 bad, ==0 neutral, NaN na."""
    if _is_na(v):
        return "na"
    try:
        f = float(v)
        if math.isnan(f):
            return "na"
        if f > 0:
            return "good"
        if f < 0:
            return "bad"
        return "neutral"
    except (TypeError, ValueError):
        return "na"


def _status_absolute(v: float | None) -> str:
    """Absolute metrics (end_aum, nnb, nnf): NaN na, else neutral."""
    if _is_na(v):
        return "na"
    return "neutral"


def _status_market_pnl(v: float | None) -> str:
    """Market PnL: >0 good, <0 bad, ==0 neutral, NaN na."""
    return _status_growth(v)


def build_kpi_strip(
    raw: dict[str, float],
    deltas: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Build the KPI strip payload: list of dicts with name, key, value, display, format, status.
    Optional deltas (e.g. change vs prev month) for end_aum, nnb, nnf, market_pnl; if prev missing, omit or None.
    """
    deltas = deltas or {}
    nan = np.nan

    def _val(k: str) -> float:
        return raw.get(k, nan) if isinstance(raw.get(k), (int, float)) else nan

    def _delta(k: str) -> float | None:
        d = deltas.get(k)
        if d is None or (_is_na(d)):
            return None
        return float(d)

    kpis = [
        {
            "name": "End AUM",
            "key": "end_aum",
            "value": _val("end_aum"),
            "display": format_currency(_val("end_aum")),
            "format": "currency",
            "status": _status_absolute(_val("end_aum")),
            "delta": _delta("end_aum"),
        },
        {
            "name": "MoM growth",
            "key": "mom_growth",
            "value": _val("mom_growth"),
            "display": format_percent(_val("mom_growth")),
            "format": "percent",
            "status": _status_growth(_val("mom_growth")),
        },
        {
            "name": "YTD growth",
            "key": "ytd_growth",
            "value": _val("ytd_growth"),
            "display": format_percent(_val("ytd_growth")),
            "format": "percent",
            "status": _status_growth(_val("ytd_growth")),
        },
        {
            "name": "NNB",
            "key": "nnb",
            "value": _val("nnb"),
            "display": format_currency(_val("nnb")),
            "format": "currency",
            "status": _status_absolute(_val("nnb")),
            "delta": _delta("nnb"),
        },
        {
            "name": "NNF",
            "key": "nnf",
            "value": _val("nnf"),
            "display": format_currency(_val("nnf")),
            "format": "currency",
            "status": _status_absolute(_val("nnf")),
            "delta": _delta("nnf"),
        },
        {
            "name": "OGR",
            "key": "ogr",
            "value": _val("ogr"),
            "display": format_percent(_val("ogr")),
            "format": "percent",
            "status": _status_growth(_val("ogr")),
        },
        {
            "name": "Market impact",
            "key": "market_impact",
            "value": _val("market_impact"),
            "display": format_percent(_val("market_impact")),
            "format": "percent",
            "status": _status_growth(_val("market_impact")),
        },
        {
            "name": "Market PnL",
            "key": "market_pnl",
            "value": _val("market_pnl"),
            "display": format_currency(_val("market_pnl")),
            "format": "currency",
            "status": _status_market_pnl(_val("market_pnl")),
            "delta": _delta("market_pnl"),
        },
    ]
    return kpis


def _compute_deltas(
    df: pd.DataFrame,
    context: dict[str, Any],
    raw: dict[str, float],
) -> dict[str, float]:
    """Deltas vs prev month: end_aum, nnb, nnf, market_pnl. Empty if prev missing."""
    out: dict[str, float] = {}
    prev = context.get("prev_month_end")
    if prev is None or df is None or df.empty:
        return out
    latest_month = context.get("latest_month_end")
    end_aum_prev = _scalar_at_month(df, prev, "end_aum")
    end_aum_latest = raw.get("end_aum", np.nan)
    if not _is_na(end_aum_latest) and not _is_na(end_aum_prev):
        out["end_aum"] = float(end_aum_latest - end_aum_prev)
    nnb_prev = _scalar_at_month(df, prev, "nnb")
    nnb_latest = raw.get("nnb", np.nan)
    if not _is_na(nnb_latest) and not _is_na(nnb_prev):
        out["nnb"] = float(nnb_latest - nnb_prev)
    nnf_prev = _scalar_at_month(df, prev, "nnf")
    nnf_latest = raw.get("nnf", np.nan)
    if not _is_na(nnf_latest) and not _is_na(nnf_prev):
        out["nnf"] = float(nnf_latest - nnf_prev)
    pnl_prev = _scalar_at_month(df, prev, "market_pnl")
    pnl_latest = raw.get("market_pnl", np.nan)
    if not _is_na(pnl_latest) and not _is_na(pnl_prev):
        out["market_pnl"] = float(pnl_latest - pnl_prev)
    return out


def _to_iso(d: Any) -> str | None:
    """Convert timestamp/date to ISO string or None."""
    if d is None:
        return None
    try:
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    except Exception:
        return None


def _latest_row(df: pd.DataFrame, context: dict[str, Any]) -> pd.Series | None:
    """Return the latest row (series) by month_end, or None."""
    if df is None or df.empty or "month_end" not in df.columns:
        return None
    latest_month = context.get("latest_month_end")
    if latest_month is None:
        return None
    try:
        mask = pd.to_datetime(df["month_end"], errors="coerce") == pd.Timestamp(latest_month)
        if not mask.any():
            return None
        return df.loc[mask].iloc[0]
    except Exception:
        return None


def build_firm_snapshot_payload(
    months: int = 24,
    root: Path | None = None,
) -> dict[str, Any]:
    """
    Build the full firm snapshot payload for the KPI strip: kpis (with value + display),
    context (latest/prev/ytd_start as ISO strings), and raw numbers. Uses canonical
    metric contract for derived columns; adds _qa (waterfall_reconcile, ogr_diff, market_impact_rate_diff).
    Empty data returns format_no_data_panel plus minimal structure.
    """
    root = Path(root) if root is not None else Path.cwd()
    df = load_firm_monthly_last_n(months=months, root=root)

    # Empty data: return no_data payload so UI does not break
    if df is None or df.empty:
        no_data = format_no_data_panel("No firm data for selected range")
        return {
            **no_data,
            "kpis": [],
            "context": {},
            "raw": {},
            "series": {"month_end": [], "end_aum": []},
            "_qa": {},
            "rates_not_computable_reason": None,
            "coverage_incomplete": True,
            "validation_skip_reason": "SKIP_INCOMPLETE_COVERAGE",
        }

    df = _apply_canonical_derived(df)
    context = compute_context_months(df)
    # Missingness policy: first available month has no prior -> rates not computable
    rates_not_computable_reason = (
        "begin_aum undefined for first available month"
        if context.get("prev_month_end") is None
        else None
    )
    coverage_incomplete = is_single_month(df) or has_month_gaps(df)
    validation_skip_reason = "SKIP_INCOMPLETE_COVERAGE" if coverage_incomplete else None

    raw = compute_kpi_raw(df, context)
    deltas = _compute_deltas(df, context, raw)
    kpis = build_kpi_strip(raw, deltas=deltas)

    # _qa: waterfall reconciliation and consistency diffs (dataset vs canonical)
    _qa: dict[str, Any] = {}
    latest = _latest_row(df, context)
    if latest is not None:
        begin = coerce_num(latest.get("begin_aum"))
        end = coerce_num(latest.get("end_aum"))
        nnb = coerce_num(latest.get("nnb"))
        mi = coerce_num(latest.get("market_impact"))
        _qa["waterfall_reconcile"] = reconcile_waterfall_from_contract(begin, end, nnb, mi)
        # If dataset provided ogr / market_impact_rate, compare to canonical
        dataset_ogr = coerce_num(latest.get("ogr"))
        computed_ogr = compute_ogr(latest.get("nnb"), latest.get("begin_aum"))
        if not math.isnan(dataset_ogr) and not math.isnan(computed_ogr):
            _qa["ogr_diff"] = abs(float(dataset_ogr - computed_ogr))
        else:
            _qa["ogr_diff"] = None
        dataset_mir = coerce_num(latest.get("market_impact_rate"))
        computed_mir = compute_market_impact_rate(latest.get("market_impact"), latest.get("begin_aum"))
        if not math.isnan(dataset_mir) and not math.isnan(computed_mir):
            _qa["market_impact_rate_diff"] = abs(float(dataset_mir - computed_mir))
        else:
            _qa["market_impact_rate_diff"] = None

    # Last 12 months series for sparkline (raw end_aum only; no KPI computation)
    series: dict[str, Any] = {"month_end": [], "end_aum": []}
    if df is not None and not df.empty and "month_end" in df.columns and "end_aum" in df.columns:
        tail = df.sort_values("month_end", ascending=True).tail(12)
        series["month_end"] = [_to_iso(t) for t in tail["month_end"].tolist()]
        try:
            vals = [float(pd.to_numeric(x, errors="coerce")) for x in tail["end_aum"].tolist()]
            series["end_aum"] = [x if math.isfinite(x) else np.nan for x in vals]
        except Exception:
            series["end_aum"] = []

    return {
        "kpis": kpis,
        "context": {
            "latest_month_end": _to_iso(context.get("latest_month_end")),
            "prev_month_end": _to_iso(context.get("prev_month_end")),
            "ytd_start_month_end": _to_iso(context.get("ytd_start_month_end")),
        },
        "raw": raw,
        "series": series,
        "_qa": _qa,
        "rates_not_computable_reason": rates_not_computable_reason,
        "coverage_incomplete": coverage_incomplete,
        "validation_skip_reason": validation_skip_reason,
    }


def _cached_firm_snapshot_impl(
    dataset_version: str,
    filter_state_hash: str,
    query_name: str,
    months: int,
) -> dict[str, Any]:
    """Inner implementation: build payload. Wrapped with st.cache_data when Streamlit is available."""
    return build_firm_snapshot_payload(months=months)


try:
    import streamlit as _st
except ImportError:
    _st = None  # type: ignore[assignment]

if _st is not None:
    _cached_firm_snapshot = _st.cache_data(show_spinner=False)(_cached_firm_snapshot_impl)
else:
    _cached_firm_snapshot = _cached_firm_snapshot_impl


def get_firm_snapshot_cached(
    months: int,
    dataset_version: str,
    filter_state_hash: str,
) -> dict[str, Any]:
    """
    Return firm snapshot payload with caching. Cache key: dataset_version, filter_state_hash,
    query_name='firm_snapshot', months. Global slice only (filter_state_hash represents global filter state).
    When Streamlit is not available (e.g. unit tests), calls build_firm_snapshot_payload directly.
    """
    payload = dict(_cached_firm_snapshot(
        dataset_version=dataset_version,
        filter_state_hash=filter_state_hash,
        query_name="firm_snapshot",
        months=months,
    ))
    payload["_meta"] = {
        "query_name": "firm_snapshot",
        "dataset_version": dataset_version,
        "filter_state_hash": filter_state_hash,
        "months": months,
    }
    return payload
