"""
AggregateSpecs and ChartSpecs: typed names and deterministic outputs.
All computations: stable sort (break ties consistently), stable column order, JSON-serializable chart payloads.
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict

import pandas as pd

# --- Aggregate names (Level B) ---
AGG_TOPN_TICKERS = "topn_tickers"
AGG_CHANNEL_MIX = "channel_mix"
AGG_ROLLING_AVG = "rolling_avg"
AGG_KPI_CARDS = "kpi_cards"

ALL_AGG_NAMES = frozenset({
    AGG_TOPN_TICKERS,
    AGG_CHANNEL_MIX,
    AGG_ROLLING_AVG,
    AGG_KPI_CARDS,
})

# --- Chart payload names (Level C) ---
CHART_WATERFALL = "waterfall"
CHART_CORR_MATRIX = "corr_matrix"
CHART_PIVOT_HEATMAP = "pivot_heatmap"

ALL_CHART_NAMES = frozenset({
    CHART_WATERFALL,
    CHART_CORR_MATRIX,
    CHART_PIVOT_HEATMAP,
})


class AggSpec(TypedDict, total=False):
    requires_cols: list[str]
    params: list[str]
    fn: Callable[..., dict[str, Any] | pd.DataFrame]


class ChartSpec(TypedDict):
    agg: str
    fn: Callable[..., dict[str, Any]]


def _to_json_safe(obj: Any) -> Any:
    """Convert to JSON-serializable types (no Timestamp/numpy)."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return obj.item()
    if hasattr(obj, "isoformat"):  # date/datetime/Timestamp
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# --- Deterministic compute functions (Level B) ---

def compute_topn_tickers(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Top-N tickers by end_aum or nnb. Deterministic: sort by value desc, then by ticker asc for ties.
    params: {"top_n": int, "by": "end_aum"|"nnb"} (default top_n=10, by="end_aum").
    """
    if df.empty:
        return {"tickers": [], "values": [], "by": "end_aum", "top_n": 0}
    p = params or {}
    top_n = int(p.get("top_n", 10))
    by = str(p.get("by", "end_aum"))
    if by not in ("end_aum", "nnb"):
        by = "end_aum"
    ticker_col = "product_ticker" if "product_ticker" in df.columns else "ticker"
    if ticker_col not in df.columns or by not in df.columns:
        return {"tickers": [], "values": [], "by": by, "top_n": top_n}
    # Aggregate per ticker (sum), then sort: value desc, ticker asc for ties
    agg = df.groupby(ticker_col, as_index=False)[by].sum()
    agg = agg.sort_values(
        [by, ticker_col],
        ascending=[False, True],
        kind="mergesort",
    )
    agg = agg.head(top_n)
    # Stable column order
    tickers = agg[ticker_col].astype(str).tolist()
    values = agg[by].astype(float).tolist()
    return {"tickers": tickers, "values": values, "by": by, "top_n": len(tickers)}


def compute_channel_mix(df: pd.DataFrame) -> dict[str, Any]:
    """Share of end_aum per channel_l1 for latest month. Deterministic: channels sorted by name."""
    if df.empty or "channel_l1" not in df.columns or "end_aum" not in df.columns:
        return {"channels": [], "shares": [], "month_end": None}
    if "month_end" in df.columns:
        latest = df["month_end"].max()
        df = df[df["month_end"] == latest].copy()
    else:
        latest = None
    if df.empty:
        return {"channels": [], "shares": [], "month_end": None}
    grp = df.groupby("channel_l1", as_index=False)["end_aum"].sum()
    grp = grp.sort_values("channel_l1", kind="mergesort")
    tot = grp["end_aum"].sum()
    shares = (grp["end_aum"] / tot).astype(float).tolist() if tot else []
    channels = grp["channel_l1"].astype(str).tolist()
    month_str = str(pd.Timestamp(latest).isoformat()) if latest is not None else None
    return {"channels": channels, "shares": shares, "month_end": month_str}


def compute_rolling_avg(df: pd.DataFrame) -> dict[str, Any]:
    """
    3-month rolling average for firm end_aum. Deterministic: sort by month_end asc.
    """
    if df.empty or "month_end" not in df.columns or "end_aum" not in df.columns:
        return {"month_end": [], "end_aum": [], "rolling_3m": []}
    df = df.sort_values("month_end", kind="mergesort").copy()
    firm = df.groupby("month_end", as_index=False)["end_aum"].sum()
    firm["rolling_3m"] = firm["end_aum"].rolling(3, min_periods=1).mean()
    month_end = [str(pd.Timestamp(x).isoformat()) for x in firm["month_end"]]
    return {
        "month_end": month_end,
        "end_aum": firm["end_aum"].astype(float).tolist(),
        "rolling_3m": firm["rolling_3m"].astype(float).tolist(),
    }


def compute_kpi_cards(df: pd.DataFrame) -> dict[str, Any]:
    """KPI totals (row_count, begin_aum, end_aum, nnb, nnf, market_pnl). Stable key order."""
    out: dict[str, Any] = {"row_count": len(df)}
    for col in ("begin_aum", "end_aum", "nnb", "nnf", "market_pnl"):
        if col in df.columns:
            out[col] = float(df[col].sum())
    return out


# --- Chart payload builders (Level C); return JSON-serializable dicts ---

def build_waterfall_payload(agg_result: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """Waterfall chart: expects kpi_cards-style dict or DataFrame with KPI columns."""
    if isinstance(agg_result, pd.DataFrame):
        data = compute_kpi_cards(agg_result)
    else:
        data = dict(agg_result) if agg_result else {}
    return _to_json_safe({"type": CHART_WATERFALL, "data": data})


def build_corr_matrix_payload(agg_result: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """Correlation matrix for selected numeric measures. Stable column/row order."""
    if isinstance(agg_result, pd.DataFrame):
        numeric = agg_result.select_dtypes(include=["number"])
        cols = sorted(numeric.columns.tolist())
        if not cols:
            return _to_json_safe({"type": CHART_CORR_MATRIX, "data": {}, "columns": []})
        sub = numeric[cols]
        corr = sub.corr()
        corr = corr.reindex(index=cols, columns=cols)
        data = corr.astype(float).to_dict()
    else:
        data = agg_result if isinstance(agg_result, dict) else {}
        cols = sorted(data.keys()) if data else []
    return _to_json_safe({"type": CHART_CORR_MATRIX, "data": data, "columns": cols})


def build_pivot_heatmap_payload(agg_result: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """Pivot heatmap: from aggregate (e.g. by_month or raw), produce matrix-style payload."""
    if isinstance(agg_result, pd.DataFrame):
        if agg_result.empty:
            return _to_json_safe({"type": CHART_PIVOT_HEATMAP, "rows": [], "columns": [], "values": []})
        cols = sorted(agg_result.columns.tolist())
        data = agg_result[cols].astype(str).where(agg_result[cols].notna(), None)
        # Orient as list of rows (stable order)
        rows = data.to_dict(orient="records")
        rows = [_to_json_safe(r) for r in rows]
        return _to_json_safe({"type": CHART_PIVOT_HEATMAP, "rows": rows, "columns": cols, "values": []})
    return _to_json_safe({"type": CHART_PIVOT_HEATMAP, "rows": [], "columns": [], "values": []})


# --- Spec registries ---

AGG_SPECS: dict[str, AggSpec] = {
    AGG_TOPN_TICKERS: {
        "requires_cols": ["product_ticker", "end_aum", "nnb"],
        "params": ["top_n", "by"],
        "fn": compute_topn_tickers,
    },
    AGG_CHANNEL_MIX: {
        "requires_cols": ["channel_l1", "end_aum", "month_end"],
        "params": [],
        "fn": compute_channel_mix,
    },
    AGG_ROLLING_AVG: {
        "requires_cols": ["month_end", "end_aum"],
        "params": [],
        "fn": compute_rolling_avg,
    },
    AGG_KPI_CARDS: {
        "requires_cols": ["begin_aum", "end_aum", "nnb", "nnf", "market_pnl"],
        "params": [],
        "fn": compute_kpi_cards,
    },
}

CHART_SPECS: dict[str, ChartSpec] = {
    CHART_WATERFALL: {"agg": AGG_KPI_CARDS, "fn": build_waterfall_payload},
    CHART_CORR_MATRIX: {"agg": "raw", "fn": build_corr_matrix_payload},
    CHART_PIVOT_HEATMAP: {"agg": "raw", "fn": build_pivot_heatmap_payload},
}


def validate_agg_name(agg_name: str) -> None:
    """Raise ValueError if agg_name is not in AGG_SPECS, with message telling dev to add a spec."""
    if agg_name not in AGG_SPECS:
        raise ValueError(
            f"Invalid agg_name {agg_name!r}. Allowed: {sorted(AGG_SPECS)}. "
            "Add a spec to app/cache/specs.py AGG_SPECS if you need a new aggregate."
        )


def validate_chart_name(chart_name: str) -> None:
    """Raise ValueError if chart_name is not in CHART_SPECS, with message telling dev to add a spec."""
    if chart_name not in CHART_SPECS:
        raise ValueError(
            f"Invalid chart_name {chart_name!r}. Allowed: {sorted(CHART_SPECS)}. "
            "Add a spec to app/cache/specs.py CHART_SPECS if you need a new chart."
        )
