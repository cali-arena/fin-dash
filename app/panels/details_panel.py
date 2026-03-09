"""
Drill-aware Details Panel: one entrypoint, gateway-only, strict cache keys.
All drill logic for the details view lives here; no direct DuckDB/parquet.
Uses app.ui.guardrails and app.ui.formatters for empty state and display.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from app.ui.formatters import format_df, fmt_currency, fmt_percent, infer_common_formats
from app.ui.exports import FULL_EXPORT_MAX_ROWS, render_export_buttons
from app.ui.guardrails import (
    ensure_min_points,
    ensure_non_empty,
    fallback_note,
    render_chart_or_fallback,
    render_empty_state,
)
from app.ui.theme import PALETTE, apply_enterprise_plotly_style, safe_render_plotly

if TYPE_CHECKING:
    from app.data_gateway import DataGateway
    from app.state import DrillState, FilterState

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def drill_state_hash(drill_state: "DrillState") -> str:
    """
    Canonical hash for cache keys: SHA1 of {"mode", "channel", "ticker"}.
    """
    canonical = {
        "mode": getattr(drill_state, "drill_mode", "channel"),
        "channel": getattr(drill_state, "selected_channel", None),
        "ticker": getattr(drill_state, "selected_ticker", None),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def _details_cache_key(
    dataset_version: str,
    filter_hash: str,
    drill_hash: str,
) -> str:
    """Cache key rule: details::{dataset_version}::{filter_hash}::{drill_hash}."""
    return f"details::{dataset_version}::{filter_hash}::{drill_hash}"


def _safe_float(x: Any) -> float:
    """Coerce to float; NaN if invalid."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    try:
        v = float(x)
        return v if not math.isnan(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def compute_selected_kpis(df: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """
    Compute selection-aware KPIs from drilled_df using governed metric formulas.
    Returns (kpis_dict, list of unavailable metric names).
    kpis_dict: AUM, NNB, OGR, market_impact_rate, fee_yield (values or None if unavailable).
    """
    from app.metrics.metric_contract import (
        compute_fee_yield,
        compute_market_impact,
        compute_market_impact_rate,
        compute_ogr,
    )

    out: dict[str, Any] = {
        "AUM": None,
        "NNB": None,
        "OGR": None,
        "market_impact_rate": None,
        "fee_yield": None,
    }
    unavailable: list[str] = []

    if df is None or df.empty:
        return out, ["AUM", "NNB", "OGR", "market_impact_rate", "fee_yield"]

    # Period-end AUM: last month_end's end_aum (or sum of end_aum if single period)
    if "end_aum" not in df.columns:
        unavailable.append("AUM")
    else:
        if "month_end" in df.columns:
            last = df.sort_values("month_end").tail(1)
            out["AUM"] = last["end_aum"].sum() if not last.empty else _safe_float(df["end_aum"].sum())
        else:
            out["AUM"] = _safe_float(df["end_aum"].sum())

    if "nnb" not in df.columns:
        unavailable.append("NNB")
    else:
        out["NNB"] = _safe_float(df["nnb"].sum())

    need_begin = "begin_aum" in df.columns
    if not need_begin:
        unavailable.extend(["OGR", "market_impact_rate", "fee_yield"])
    else:
        begin = _safe_float(df["begin_aum"].sum())
        end = _safe_float(df["end_aum"].sum()) if "end_aum" in df.columns else float("nan")
        nnb = _safe_float(df["nnb"].sum()) if "nnb" in df.columns else float("nan")
        nnf = _safe_float(df["nnf"].sum()) if "nnf" in df.columns else float("nan")
        if "market_impact" in df.columns:
            mi = _safe_float(df["market_impact"].sum())
        else:
            mi = compute_market_impact(begin, end, nnb)
        out["OGR"] = compute_ogr(nnb, begin) if begin and begin == begin else None
        out["market_impact_rate"] = compute_market_impact_rate(mi, begin) if begin and begin == begin else None
        out["fee_yield"] = compute_fee_yield(nnf, begin, end, nnb=nnb) if (begin == begin or end == end) else None
        if out["OGR"] is None or (isinstance(out["OGR"], float) and math.isnan(out["OGR"])):
            out["OGR"] = None
            if "OGR" not in unavailable:
                unavailable.append("OGR")
        if out["market_impact_rate"] is None or (isinstance(out["market_impact_rate"], float) and math.isnan(out["market_impact_rate"])):
            out["market_impact_rate"] = None
            if "market_impact_rate" not in unavailable:
                unavailable.append("market_impact_rate")
        if out["fee_yield"] is None or (isinstance(out["fee_yield"], float) and math.isnan(out["fee_yield"])):
            out["fee_yield"] = None
            if "fee_yield" not in unavailable:
                unavailable.append("fee_yield")

    return out, list(dict.fromkeys(unavailable))


def _build_trend_monthly(
    df: pd.DataFrame,
    last_n_months: int = 24,
) -> pd.DataFrame:
    """
    Build monthly trend table (last N months in df): month_end, AUM_end, NNB, OGR, market_impact_rate, fee_yield.
    Bounded to months present in df; no hard-coded dates.
    """
    from app.metrics.metric_contract import (
        compute_fee_yield,
        compute_market_impact,
        compute_market_impact_rate,
        compute_ogr,
    )

    if df is None or df.empty or "month_end" not in df.columns:
        return pd.DataFrame()

    df = df.sort_values("month_end").copy()
    months = df["month_end"].dropna().unique()
    months = sorted(pd.Series(months).dropna().tolist(), key=lambda x: x)
    months = months[-last_n_months:] if len(months) > last_n_months else months
    if not months:
        return pd.DataFrame()

    out_rows = []
    for me in months:
        row_df = df[df["month_end"] == me]
        if row_df.empty:
            continue
        begin = row_df["begin_aum"].sum() if "begin_aum" in row_df.columns else float("nan")
        end = row_df["end_aum"].sum() if "end_aum" in row_df.columns else float("nan")
        nnb = row_df["nnb"].sum() if "nnb" in row_df.columns else float("nan")
        nnf = row_df["nnf"].sum() if "nnf" in row_df.columns else float("nan")
        mi = row_df["market_impact"].sum() if "market_impact" in row_df.columns else compute_market_impact(begin, end, nnb)
        ogr = compute_ogr(nnb, begin) if begin == begin else float("nan")
        mir = compute_market_impact_rate(mi, begin) if begin == begin else float("nan")
        fy = compute_fee_yield(nnf, begin, end, nnb=nnb) if (begin == begin or end == end) else float("nan")
        out_rows.append({
            "month_end": me,
            "AUM_end": end,
            "NNB": nnb,
            "OGR": ogr,
            "market_impact_rate": mir,
            "fee_yield": fy,
        })
    return pd.DataFrame(out_rows)


def _geo_column(df: pd.DataFrame) -> str | None:
    """First available geo-like column in df; else None."""
    for c in ("geo", "region", "src_country_canonical", "product_country_canonical"):
        if c in df.columns:
            return c
    return None


def _entity_column(df: pd.DataFrame, drill_state: "DrillState") -> str | None:
    """Preferred peer entity column based on drill mode."""
    if df is None or df.empty:
        return None
    if drill_state.drill_mode == "ticker":
        return "ticker" if "ticker" in df.columns else ("product_ticker" if "product_ticker" in df.columns else None)
    return "channel" if "channel" in df.columns else ("channel_l1" if "channel_l1" in df.columns else None)


def _peer_rank_diagnostics(base_df: pd.DataFrame, drill_state: "DrillState") -> dict[str, Any]:
    """Peer rank + contribution for selected channel/ticker versus full peer set."""
    if base_df is None or base_df.empty:
        return {}
    entity_col = _entity_column(base_df, drill_state)
    if not entity_col or entity_col not in base_df.columns:
        return {}
    selected = drill_state.selected_ticker if drill_state.drill_mode == "ticker" else drill_state.selected_channel
    if not selected:
        return {}
    agg_cols = [c for c in ("end_aum", "nnb", "begin_aum") if c in base_df.columns]
    if not agg_cols:
        return {}
    peers = base_df.groupby(entity_col, as_index=False)[agg_cols].sum().rename(columns={entity_col: "entity"})
    if peers.empty:
        return {}
    if "end_aum" in peers.columns:
        peers["rank_aum"] = peers["end_aum"].rank(method="min", ascending=False)
    if "nnb" in peers.columns:
        peers["rank_nnb"] = peers["nnb"].rank(method="min", ascending=False)
    row = peers[peers["entity"].astype(str).str.strip() == str(selected).strip()]
    if row.empty:
        return {}
    r = row.iloc[0]
    out: dict[str, Any] = {"entity": selected, "peer_count": int(len(peers))}
    if "end_aum" in peers.columns:
        total_aum = float(peers["end_aum"].sum()) if len(peers) else float("nan")
        out["aum"] = float(r.get("end_aum", float("nan")))
        out["rank_aum"] = int(r.get("rank_aum", float("nan"))) if pd.notna(r.get("rank_aum")) else None
        out["aum_contrib_pct"] = (float(r.get("end_aum")) / total_aum) if total_aum and pd.notna(total_aum) else float("nan")
    if "nnb" in peers.columns:
        total_nnb = float(peers["nnb"].sum()) if len(peers) else float("nan")
        out["nnb"] = float(r.get("nnb", float("nan")))
        out["rank_nnb"] = int(r.get("rank_nnb", float("nan"))) if pd.notna(r.get("rank_nnb")) else None
        out["nnb_contrib_pct"] = (float(r.get("nnb")) / total_nnb) if total_nnb and pd.notna(total_nnb) else float("nan")
    return out


def _movers_table(base_df: pd.DataFrame, drill_state: "DrillState", top_n: int = 5) -> pd.DataFrame:
    """Top gainers/decliners by MoM AUM delta and NNB using current vs prior month."""
    if base_df is None or base_df.empty or "month_end" not in base_df.columns or "end_aum" not in base_df.columns:
        return pd.DataFrame()
    entity_col = _entity_column(base_df, drill_state)
    if not entity_col:
        return pd.DataFrame()
    df = base_df.copy()
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    df = df.dropna(subset=["month_end"]).sort_values("month_end")
    if df.empty:
        return pd.DataFrame()
    current_me = df["month_end"].max()
    prior_me = df[df["month_end"] < current_me]["month_end"].max() if (df["month_end"] < current_me).any() else None
    if prior_me is None:
        return pd.DataFrame()
    cur = df[df["month_end"] == current_me].groupby(entity_col, as_index=False).agg({"end_aum": "sum", "nnb": "sum"} if "nnb" in df.columns else {"end_aum": "sum"})
    prior = df[df["month_end"] == prior_me].groupby(entity_col, as_index=False)["end_aum"].sum().rename(columns={"end_aum": "end_aum_prior"})
    out = cur.merge(prior, on=entity_col, how="left")
    out["end_aum_prior"] = out["end_aum_prior"].fillna(0.0)
    out["mom_daum"] = out["end_aum"] - out["end_aum_prior"]
    out["mom_daum_pct"] = out.apply(
        lambda r: (r["mom_daum"] / r["end_aum_prior"]) if r["end_aum_prior"] not in (0, None) else float("nan"),
        axis=1,
    )
    out = out.rename(columns={entity_col: "entity", "end_aum": "aum_current", "nnb": "nnb_current"})
    gain = out.sort_values(["mom_daum", "entity"], ascending=[False, True]).head(top_n).assign(direction="gainer")
    det = out.sort_values(["mom_daum", "entity"], ascending=[True, True]).head(top_n).assign(direction="decliner")
    combined = pd.concat([gain, det], ignore_index=True)
    display_cols = [c for c in ("direction", "entity", "aum_current", "end_aum_prior", "mom_daum", "mom_daum_pct", "nnb_current") if c in combined.columns]
    return combined[display_cols].reset_index(drop=True)


def _mix_shift_table(base_df: pd.DataFrame, drill_state: "DrillState", top_n: int = 10) -> pd.DataFrame:
    """Strongest share shifts (AUM and NNB) current vs prior month."""
    if base_df is None or base_df.empty or "month_end" not in base_df.columns:
        return pd.DataFrame()
    entity_col = _entity_column(base_df, drill_state)
    if not entity_col:
        return pd.DataFrame()
    df = base_df.copy()
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    df = df.dropna(subset=["month_end"]).sort_values("month_end")
    if df.empty:
        return pd.DataFrame()
    current_me = df["month_end"].max()
    prior_me = df[df["month_end"] < current_me]["month_end"].max() if (df["month_end"] < current_me).any() else None
    if prior_me is None:
        return pd.DataFrame()
    agg_map = {"end_aum": "sum"}
    if "nnb" in df.columns:
        agg_map["nnb"] = "sum"
    cur = df[df["month_end"] == current_me].groupby(entity_col, as_index=False).agg(agg_map)
    prior = df[df["month_end"] == prior_me].groupby(entity_col, as_index=False).agg(agg_map)
    cur = cur.rename(columns={entity_col: "entity", "end_aum": "aum_cur", "nnb": "nnb_cur"})
    prior = prior.rename(columns={entity_col: "entity", "end_aum": "aum_prior", "nnb": "nnb_prior"})
    out = cur.merge(prior, on="entity", how="outer").fillna(0.0)
    total_aum_cur = float(out["aum_cur"].sum()) if "aum_cur" in out.columns else 0.0
    total_aum_prior = float(out["aum_prior"].sum()) if "aum_prior" in out.columns else 0.0
    out["aum_share_cur"] = out["aum_cur"] / total_aum_cur if total_aum_cur else float("nan")
    out["aum_share_prior"] = out["aum_prior"] / total_aum_prior if total_aum_prior else float("nan")
    out["aum_share_shift"] = out["aum_share_cur"].fillna(0) - out["aum_share_prior"].fillna(0)
    if "nnb_cur" in out.columns and "nnb_prior" in out.columns:
        total_nnb_cur = float(out["nnb_cur"].sum()) if len(out) else 0.0
        total_nnb_prior = float(out["nnb_prior"].sum()) if len(out) else 0.0
        out["nnb_share_cur"] = out["nnb_cur"] / total_nnb_cur if total_nnb_cur else float("nan")
        out["nnb_share_prior"] = out["nnb_prior"] / total_nnb_prior if total_nnb_prior else float("nan")
        out["nnb_share_shift"] = out["nnb_share_cur"].fillna(0) - out["nnb_share_prior"].fillna(0)
    out["mix_shift_abs"] = out[["aum_share_shift", "nnb_share_shift"]].abs().max(axis=1) if "nnb_share_shift" in out.columns else out["aum_share_shift"].abs()
    out = out.sort_values(["mix_shift_abs", "entity"], ascending=[False, True]).head(top_n)
    display_cols = [c for c in ("entity", "aum_share_shift", "nnb_share_shift", "mix_shift_abs", "aum_cur", "nnb_cur") if c in out.columns]
    return out[display_cols].reset_index(drop=True)


def _geo_split_table(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Geo allocation/flow concentration table."""
    geo_col = _geo_column(df)
    if df is None or df.empty or not geo_col:
        return pd.DataFrame()
    agg_map = {"end_aum": "sum"} if "end_aum" in df.columns else {}
    if "nnb" in df.columns:
        agg_map["nnb"] = "sum"
    if not agg_map:
        return pd.DataFrame()
    out = df.groupby(geo_col, as_index=False).agg(agg_map).rename(columns={geo_col: "geo"})
    if "end_aum" in out.columns:
        total = float(out["end_aum"].sum())
        out["aum_contrib_pct"] = out["end_aum"] / total if total else float("nan")
    if "nnb" in out.columns:
        total_nnb = float(out["nnb"].sum())
        out["nnb_contrib_pct"] = out["nnb"] / total_nnb if total_nnb else float("nan")
    sort_col = "end_aum" if "end_aum" in out.columns else "nnb"
    out = out.sort_values([sort_col, "geo"], ascending=[False, True]).head(top_n)
    return out.reset_index(drop=True)


def _signal_table(base_df: pd.DataFrame, drill_state: "DrillState", top_n: int = 12) -> pd.DataFrame:
    """Deterministic anomaly/signal table with low/medium/high severity."""
    if base_df is None or base_df.empty or "month_end" not in base_df.columns:
        return pd.DataFrame()
    entity_col = _entity_column(base_df, drill_state)
    if not entity_col:
        return pd.DataFrame()
    df = base_df.copy()
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    df = df.dropna(subset=["month_end"]).sort_values("month_end")
    if df.empty:
        return pd.DataFrame()
    current_me = df["month_end"].max()
    prior_me = df[df["month_end"] < current_me]["month_end"].max() if (df["month_end"] < current_me).any() else None
    agg_map = {c: "sum" for c in ("end_aum", "nnb", "market_impact", "nnf", "begin_aum") if c in df.columns}
    cur = df[df["month_end"] == current_me].groupby(entity_col, as_index=False).agg(agg_map).rename(columns={entity_col: "entity"})
    if cur.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    def _sev(z: float) -> str:
        if z != z:
            return "low"
        if abs(z) >= 3:
            return "high"
        if abs(z) >= 2:
            return "medium"
        return "low"
    for metric in ("nnb", "market_impact", "nnf"):
        if metric not in cur.columns:
            continue
        series = pd.to_numeric(cur[metric], errors="coerce")
        mu = float(series.mean())
        sigma = float(series.std()) if pd.notna(series.std()) else float("nan")
        if sigma == 0 or sigma != sigma:
            continue
        for _, r in cur.iterrows():
            z = (float(r[metric]) - mu) / sigma
            s = _sev(z)
            if s == "low":
                continue
            rows.append(
                {
                    "entity": r["entity"],
                    "metric": metric.upper(),
                    "value_current": float(r[metric]),
                    "baseline": mu,
                    "zscore": z,
                    "rule_id": "cross_section_z",
                    "reason": f"{metric.upper()} z-score={z:.2f} vs peer mean {mu:.2f}",
                    "severity": s,
                    "month_end": current_me,
                }
            )
    if prior_me is not None and "nnb" in agg_map:
        prior = df[df["month_end"] == prior_me].groupby(entity_col, as_index=False).agg({"nnb": "sum"}).rename(columns={entity_col: "entity", "nnb": "nnb_prior"})
        merged = cur.merge(prior, on="entity", how="left")
        for _, r in merged.iterrows():
            cur_n = float(r.get("nnb", float("nan")))
            prv_n = float(r.get("nnb_prior", float("nan")))
            if pd.isna(cur_n) or pd.isna(prv_n) or cur_n == 0 or prv_n == 0:
                continue
            if (cur_n > 0 > prv_n) or (cur_n < 0 < prv_n):
                rows.append(
                    {
                        "entity": r["entity"],
                        "metric": "NNB",
                        "value_current": cur_n,
                        "baseline": prv_n,
                        "zscore": float("nan"),
                        "rule_id": "reversal",
                        "reason": f"NNB reversal from {prv_n:.2f} to {cur_n:.2f}",
                        "severity": "high",
                        "month_end": current_me,
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    out["_sev"] = out["severity"].map(sev_rank).fillna(3)
    out["_absz"] = out["zscore"].abs().fillna(0.0)
    out = out.sort_values(["_sev", "_absz", "entity"], ascending=[True, False, True]).drop(columns=["_sev", "_absz"]).head(top_n)
    return out.reset_index(drop=True)


def _compute_breakdown_full(
    df: pd.DataFrame,
    drill_state: "DrillState",
    split_by_geo: bool = False,
) -> pd.DataFrame:
    """
    Full breakdown table (no top_n/search). Groupby name; aggregate NNB, AUM, NNF; compute OGR, Fee Yield.
    Used for caching; caller applies search and top_n.
    """
    from app.metrics.metric_contract import compute_fee_yield, compute_market_impact, compute_market_impact_rate, compute_ogr

    if df is None or df.empty:
        return pd.DataFrame()

    channel_col = "channel" if "channel" in df.columns else ("channel_l1" if "channel_l1" in df.columns else None)
    ticker_col = "ticker" if "ticker" in df.columns else ("product_ticker" if "product_ticker" in df.columns else None)
    geo_col = _geo_column(df) if split_by_geo else None

    agg_cols = [c for c in ("begin_aum", "end_aum", "nnb", "nnf") if c in df.columns]
    if not agg_cols or "nnb" not in df.columns:
        return pd.DataFrame()

    # Choose groupby and title logic
    if drill_state.selected_channel and ticker_col:
        group_col = ticker_col
    elif drill_state.selected_ticker:
        group_col = (geo_col if (split_by_geo and geo_col) else channel_col) or channel_col
        if not group_col or group_col not in df.columns:
            group_col = channel_col
    else:
        group_col = (channel_col if getattr(drill_state, "drill_mode", "channel") == "channel" else ticker_col) or channel_col

    if not group_col or group_col not in df.columns:
        return pd.DataFrame()

    g = df.groupby(group_col, as_index=False)
    agg = g.agg({c: "sum" for c in agg_cols if c in df.columns})
    agg = agg.rename(columns={group_col: "Name", "nnb": "NNB", "end_aum": "AUM", "nnf": "NNF"})
    if "Name" not in agg.columns:
        agg["Name"] = agg[group_col] if group_col in agg.columns else agg.index.astype(str)
    if "NNB" not in agg.columns and "nnb" in agg.columns:
        agg["NNB"] = agg["nnb"]
    if "NNB" not in agg.columns:
        return pd.DataFrame()

    agg = agg.sort_values("NNB", ascending=False)
    agg = agg.sort_values("Name", ascending=True, kind="mergesort").sort_values("NNB", ascending=False)

    begin = agg["begin_aum"] if "begin_aum" in agg.columns else float("nan")
    end = agg["AUM"] if "AUM" in agg.columns else (agg["end_aum"] if "end_aum" in agg.columns else float("nan"))
    nnb = agg["NNB"]
    nnf = agg["NNF"] if "NNF" in agg.columns else (agg["nnf"] if "nnf" in agg.columns else float("nan"))
    mi = compute_market_impact(begin, end, nnb) if "begin_aum" in agg.columns else float("nan")
    agg["OGR"] = compute_ogr(nnb, begin) if "begin_aum" in agg.columns else float("nan")
    agg["market_impact_rate"] = compute_market_impact_rate(mi, begin) if "begin_aum" in agg.columns else float("nan")
    agg["fee_yield"] = compute_fee_yield(nnf, begin, end, nnb=nnb) if "begin_aum" in agg.columns and ("AUM" in agg.columns or "end_aum" in agg.columns) else float("nan")
    if "AUM" not in agg.columns and "end_aum" in agg.columns:
        agg["AUM"] = agg["end_aum"]
    if "NNF" not in agg.columns and "nnf" in agg.columns:
        agg["NNF"] = agg["nnf"]
    display = ["Name", "NNB", "AUM", "NNF", "OGR", "fee_yield"]
    display = [c for c in display if c in agg.columns]
    return agg[display].copy()


def build_breakdown(
    df: pd.DataFrame,
    drill_state: "DrillState",
    top_n: int = 25,
    search_term: str | None = None,
    cache_key: str | None = None,
    split_by_geo: bool = False,
) -> tuple[str, pd.DataFrame]:
    """
    (title, table_df). Rank by NNB desc, then name asc. Apply search (contains, case-insensitive) then top_n.
    Full groupby result cached by cache_key in session_state; search/top_n applied in memory.
    """
    channel_col = "channel" if (df is not None and "channel" in df.columns) else ("channel_l1" if (df is not None and "channel_l1" in df.columns) else None)
    ticker_col = "ticker" if (df is not None and "ticker" in df.columns) else ("product_ticker" if (df is not None and "product_ticker" in df.columns) else None)
    geo_col = _geo_column(df) if df is not None else None

    if drill_state.selected_channel and ticker_col:
        title = "Top Tickers inside Channel"
    elif drill_state.selected_ticker:
        title = "Geo split" if (split_by_geo and geo_col) else "Channel split"
    else:
        mode = getattr(drill_state, "drill_mode", "channel")
        title = "Top Channels" if mode == "channel" else "Top Tickers"

    if df is None or df.empty:
        return title, pd.DataFrame()

    store_key = f"details_breakdown_full_{cache_key or 'none'}_{split_by_geo}"
    full_df = None
    if st is not None and cache_key:
        if store_key in st.session_state:
            stored = st.session_state[store_key]
            if isinstance(stored, pd.DataFrame) and not stored.empty:
                full_df = stored
    if full_df is None:
        full_df = _compute_breakdown_full(df, drill_state, split_by_geo=split_by_geo)
        if st is not None and cache_key and not full_df.empty:
            st.session_state[store_key] = full_df

    if full_df.empty:
        return title, pd.DataFrame()

    name_col = "Name" if "Name" in full_df.columns else full_df.columns[0]
    out = full_df.copy()
    if search_term and str(search_term).strip():
        term = str(search_term).strip()
        out = out[out[name_col].astype(str).str.contains(term, case=False, na=False)]
    out = out.sort_values("NNB", ascending=False) if "NNB" in out.columns else out
    if "Name" in out.columns:
        out = out.sort_values("Name", ascending=True, kind="mergesort").sort_values("NNB", ascending=False)
    out = out.head(top_n)
    return title, out


def _valid_sets_from_base(base_df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Extract valid_channels and valid_tickers from base_df for validate_drill_selection."""
    valid_channels: set[str] = set()
    valid_tickers: set[str] = set()
    if base_df is None or base_df.empty:
        return valid_channels, valid_tickers
    channel_col = "channel" if "channel" in base_df.columns else ("channel_l1" if "channel_l1" in base_df.columns else None)
    ticker_col = "ticker" if "ticker" in base_df.columns else ("product_ticker" if "product_ticker" in base_df.columns else None)
    if channel_col:
        valid_channels = set(base_df[channel_col].dropna().astype(str).str.strip().unique())
    if ticker_col:
        valid_tickers = set(base_df[ticker_col].dropna().astype(str).str.strip().unique())
    return valid_channels, valid_tickers


def render_details_panel(
    filters: "FilterState",
    drill_state: "DrillState",
    gateway: "DataGateway",
) -> None:
    """
    Single entrypoint for the Details panel. Queries via gateway only; renders
    selection-aware KPIs, mini trend, breakdown tables; enforces guardrails
    (auto-clear invalid/empty selections). Cache key used by gateway:
    details::{dataset_version}::{filter_hash}::{drill_hash}.
    """
    from app.state import (
        get_drill_state,
        set_selected_channel,
        set_selected_ticker,
        update_drill_state,
        validate_drill_selection,
    )

    # Gateway-only: base then apply drill filter (no direct DuckDB/parquet)
    if st is not None:
        with st.spinner("Loading…"):
            base_df = gateway.fetch_details_base(filters, drill_state)
    else:
        base_df = gateway.fetch_details_base(filters, drill_state)
    if not ensure_non_empty(base_df, "No data under current filters.", "Widen date range or relax filters."):
        return

    valid_channels, valid_tickers = _valid_sets_from_base(base_df)
    ok, reset_msg = validate_drill_selection(valid_channels, valid_tickers)
    if not ok and reset_msg and st is not None:
        render_empty_state(
            reset_msg or "Selection reset (not available under current filters).",
            "Pick another channel/ticker or widen date range.",
            icon="ℹ️",
        )
    drill_state = get_drill_state()

    drilled_df = gateway.apply_drill_filter(base_df, drill_state)

    # Guardrail: selection set but 0 rows -> auto-clear and recompute so details always show something if base has rows
    has_selection = (
        drill_state.selected_channel is not None or drill_state.selected_ticker is not None
    )
    if has_selection and (drilled_df is None or drilled_df.empty) and st is not None:
        update_drill_state(details_level="firm")
        set_selected_channel(None)
        set_selected_ticker(None)
        render_empty_state(
            "Selection reset because it's not available under current filters.",
            "Pick another channel/ticker or widen date range.",
        )
        drill_state = get_drill_state()
        drilled_df = gateway.apply_drill_filter(base_df, drill_state)

    if drilled_df is None:
        drilled_df = pd.DataFrame()

    # Build breakdown from drilled_df
    details_breakdown = pd.DataFrame()
    if not drilled_df.empty and "month_end" in drilled_df.columns:
        agg_cols = [c for c in ("end_aum", "nnb") if c in drilled_df.columns]
        if agg_cols:
            details_breakdown = drilled_df.groupby("month_end", as_index=False)[agg_cols].sum()

    # Header
    if drill_state.selected_channel:
        st.subheader(f"Details — Channel: {drill_state.selected_channel}")
    elif drill_state.selected_ticker:
        st.subheader(f"Details — Ticker: {drill_state.selected_ticker}")
    else:
        st.subheader("Details — All")

    if drilled_df.empty:
        render_empty_state("No detail data for the current filters.", "Widen date range or relax filters.")
        return

    # Selection-aware KPIs
    kpis, unavailable = compute_selected_kpis(drilled_df)
    if unavailable and st is not None:
        render_empty_state(
            "Unavailable metrics: " + ", ".join(unavailable),
            "Data may be missing for this slice.",
        )
    if st is not None:
        def _fmt_metric(k: str, v: Any) -> str:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "—"
            if k in ("OGR", "market_impact_rate", "fee_yield"):
                return fmt_percent(v, decimals=2, signed=True)
            return fmt_currency(v, unit="auto", decimals=2)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("AUM", _fmt_metric("AUM", kpis.get("AUM")))
        with c2:
            st.metric("NNB", _fmt_metric("NNB", kpis.get("NNB")))
        with c3:
            st.metric("OGR", _fmt_metric("OGR", kpis.get("OGR")))
        with c4:
            st.metric("Market Impact Rate", _fmt_metric("market_impact_rate", kpis.get("market_impact_rate")))
        with c5:
            st.metric("Fee Yield", _fmt_metric("fee_yield", kpis.get("fee_yield")))

    # Mini trend: last 12–24 months, OGR and Market Impact Rate (guardrails: min points + fallback)
    trend_df = _build_trend_monthly(drilled_df, last_n_months=36)
    trend_mode = "12M"
    if st is not None:
        trend_mode = st.radio(
            "Detail trend window",
            options=["1M", "3M", "YTD", "12M", "Full"],
            index=3,
            horizontal=True,
            key="details_trend_mode",
        )
    if not trend_df.empty and "month_end" in trend_df.columns:
        trend_df = trend_df.copy()
        trend_df["month_end"] = pd.to_datetime(trend_df["month_end"], errors="coerce")
        trend_df = trend_df.dropna(subset=["month_end"]).sort_values("month_end")
        if trend_mode == "1M":
            trend_df = trend_df.tail(1)
        elif trend_mode == "3M":
            trend_df = trend_df.tail(3)
        elif trend_mode == "12M":
            trend_df = trend_df.tail(12)
        elif trend_mode == "YTD" and not trend_df.empty:
            cy = int(trend_df["month_end"].max().year)
            trend_df = trend_df[trend_df["month_end"].dt.year == cy]
    if go is not None and st is not None:
        fallback_cols = [c for c in ("month_end", "OGR", "market_impact_rate") if c in trend_df.columns] or list(trend_df.columns)[:3]
        def _draw_trend():
            has_ogr = "OGR" in trend_df.columns and trend_df["OGR"].notna().any()
            has_mir = "market_impact_rate" in trend_df.columns and trend_df["market_impact_rate"].notna().any()
            if not (has_ogr or has_mir):
                return
            fig = go.Figure()
            if "AUM_end" in trend_df.columns and len(trend_df) >= 4:
                roll = trend_df["AUM_end"].rolling(window=3, min_periods=2)
                trend_df["AUM_roll"] = roll.mean()
                trend_df["AUM_std"] = roll.std()
                trend_df["AUM_upper"] = trend_df["AUM_roll"] + trend_df["AUM_std"]
                trend_df["AUM_lower"] = trend_df["AUM_roll"] - trend_df["AUM_std"]
                fig.add_trace(go.Scatter(x=trend_df["month_end"], y=trend_df["AUM_end"], name="AUM", mode="lines"))
                fig.add_trace(go.Scatter(x=trend_df["month_end"], y=trend_df["AUM_roll"], name="AUM 3M avg", mode="lines"))
                fig.add_trace(go.Scatter(x=trend_df["month_end"], y=trend_df["AUM_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(
                    go.Scatter(
                        x=trend_df["month_end"],
                        y=trend_df["AUM_lower"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(148,163,184,0.18)",
                        name="AUM volatility band",
                        hoverinfo="skip",
                    )
                )
            if has_ogr:
                fig.add_trace(
                    go.Scatter(
                        x=trend_df["month_end"],
                        y=trend_df["OGR"],
                        name="OGR",
                        mode="lines+markers",
                        line=dict(color=PALETTE["primary"], width=2),
                        marker=dict(color=PALETTE["primary"], size=6),
                    )
                )
            if has_mir:
                fig.add_trace(
                    go.Scatter(
                        x=trend_df["month_end"],
                        y=trend_df["market_impact_rate"],
                        name="Market Impact Rate",
                        mode="lines+markers",
                        line=dict(color=PALETTE["secondary"], width=2),
                        marker=dict(color=PALETTE["secondary"], size=6),
                    )
                )
            fig.update_layout(title="OGR & Market Impact Rate (last months)", xaxis_title="Month", yaxis_title="Rate", height=280)
            apply_enterprise_plotly_style(fig, height=280)
            safe_render_plotly(fig, user_message="Selected item diagnostic unavailable for this selection.")
        render_chart_or_fallback(
            _draw_trend,
            trend_df,
            fallback_cols,
            fallback_note("insufficient_trend", {"min_points": 2}),
            min_points=2,
            empty_reason="No trend data for the current filters.",
            empty_hint="Widen date range or relax filters.",
        )

    # Breakdown table: cache key for expensive groupby (dataset_version + filter_hash + drill_hash)
    filter_hash = (
        filters.filter_state_hash()
        if hasattr(filters, "filter_state_hash") and callable(getattr(filters, "filter_state_hash"))
        else hashlib.sha1(json.dumps(filters if isinstance(filters, dict) else {}, sort_keys=True).encode()).hexdigest()
    )
    try:
        dataset_version = gateway.get_dataset_version()
    except Exception:
        dataset_version = "dev"
    cache_key = _details_cache_key(dataset_version, filter_hash, drill_state_hash(drill_state))

    top_n = int(st.session_state.get("tab1_top_n", 25)) if st is not None else 25
    search_term = None
    if st is not None:
        search_term = st.text_input(
            "Search",
            key="details_search",
            placeholder="Filter tickers/channels…",
        )
    split_by_geo = False
    if drill_state.selected_ticker and _geo_column(drilled_df):
        if st is not None:
            split_by_geo = st.radio(
                "Split by",
                options=["Channel", "Geo"],
                index=0,
                key="details_split_by",
                horizontal=True,
            ) == "Geo"
        else:
            split_by_geo = False

    breakdown_title, breakdown_df = build_breakdown(
        drilled_df,
        drill_state,
        top_n=top_n,
        search_term=search_term or None,
        cache_key=cache_key,
        split_by_geo=split_by_geo,
    )
    store_key = f"details_breakdown_full_{cache_key or 'none'}_{split_by_geo}"
    if st is not None and breakdown_title:
        st.subheader(breakdown_title)
    if st is not None and not breakdown_df.empty and len(breakdown_df) >= top_n and not (search_term and str(search_term).strip()):
        st.caption(f"Showing Top {top_n}. Use search to refine.")
    if not breakdown_df.empty:
        breakdown_show = format_df(breakdown_df, infer_common_formats(breakdown_df))
        st.dataframe(breakdown_show, height=360, use_container_width=True, hide_index=True)
        if st is not None:
            def _full_breakdown_provider() -> pd.DataFrame:
                stored = st.session_state.get(store_key) if st else None
                if stored is not None and isinstance(stored, pd.DataFrame) and not stored.empty:
                    return stored.head(FULL_EXPORT_MAX_ROWS)
                return pd.DataFrame()
            allow_full = st.session_state.get("export_mode_toggle", False)
            render_export_buttons(
                breakdown_df,
                _full_breakdown_provider,
                "details_breakdown",
                allow_full=allow_full,
            )

    # Selected item diagnostic (selection-aware; deterministic).
    peer_diag = _peer_rank_diagnostics(base_df, drill_state)
    movers_df = _movers_table(base_df, drill_state, top_n=min(10, max(5, top_n // 2)))
    mix_shift_df = _mix_shift_table(base_df, drill_state, top_n=min(12, top_n))
    geo_split_df = _geo_split_table(drilled_df, top_n=min(10, top_n))
    signal_df = _signal_table(base_df, drill_state, top_n=min(15, top_n))
    with st.expander("Selected Item Diagnostic", expanded=bool(drill_state.selected_channel or drill_state.selected_ticker)):
        if peer_diag:
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("Peer Rank (AUM)", peer_diag.get("rank_aum", "—"))
            with d2:
                st.metric("Peer Rank (NNB)", peer_diag.get("rank_nnb", "—"))
            with d3:
                aum_contrib = peer_diag.get("aum_contrib_pct")
                st.metric("AUM Contribution", fmt_percent(aum_contrib, decimals=2, signed=False) if aum_contrib == aum_contrib else "—")
            with d4:
                nnb_contrib = peer_diag.get("nnb_contrib_pct")
                st.metric("NNB Contribution", fmt_percent(nnb_contrib, decimals=2, signed=True) if nnb_contrib == nnb_contrib else "—")
        else:
            st.caption("Select a channel or ticker to view peer diagnostics.")

        if not movers_df.empty:
            st.markdown("**What Changed This Period? (Top/Bottom Movers)**")
            movers_show = format_df(movers_df, infer_common_formats(movers_df))
            st.dataframe(movers_show, use_container_width=True, hide_index=True, height=260)
            render_export_buttons(movers_df, None, "details_movers")
        if not mix_shift_df.empty:
            st.markdown("**Allocation / Flow Concentration (Mix Shift)**")
            mix_show = format_df(mix_shift_df, infer_common_formats(mix_shift_df))
            st.dataframe(mix_show, use_container_width=True, hide_index=True, height=260)
            render_export_buttons(mix_shift_df, None, "details_mix_shift")
        if not geo_split_df.empty:
            st.markdown("**Geo Allocation Signals**")
            geo_show = format_df(geo_split_df, infer_common_formats(geo_split_df))
            st.dataframe(geo_show, use_container_width=True, hide_index=True, height=250)
            render_export_buttons(geo_split_df, None, "details_geo_split")
        if not signal_df.empty:
            st.markdown("**Anomaly & Risk Signals**")
            signal_show = format_df(signal_df, infer_common_formats(signal_df))
            st.dataframe(signal_show, use_container_width=True, hide_index=True, height=280)
            render_export_buttons(signal_df, None, "details_signal_table")

    # Selection-aware tables (formatted for display; raw df used for computations)
    drilled_view = drilled_df.head(50)
    drilled_show = format_df(drilled_view, infer_common_formats(drilled_df))
    st.dataframe(
        drilled_show,
        use_container_width=True,
        hide_index=True,
        height=320,
    )
    if st is not None:
        render_export_buttons(drilled_view, None, "details_drilled")
    if not details_breakdown.empty:
        st.caption("Monthly totals (slice)")
        details_breakdown_show = format_df(details_breakdown, infer_common_formats(details_breakdown))
        st.dataframe(
            details_breakdown_show,
            use_container_width=True,
            hide_index=True,
            height=220,
        )
        if st is not None:
            render_export_buttons(details_breakdown, None, "details_monthly_totals")
    # AUM line chart: guardrails so single-month/empty never breaks
    trend_slice = pd.DataFrame()
    if "month_end" in drilled_df.columns and "end_aum" in drilled_df.columns and len(drilled_df) >= 1:
        try:
            trend_slice = drilled_df.groupby("month_end", as_index=False)["end_aum"].sum()
            trend_slice = trend_slice.sort_values("month_end")
        except Exception:
            pass
    if st is not None and not trend_slice.empty:
        def _draw_aum_line():
            st.line_chart(trend_slice.set_index("month_end")[["end_aum"]])
        fallback_cols = [c for c in ("month_end", "end_aum") if c in trend_slice.columns] or list(trend_slice.columns)
        render_chart_or_fallback(
            _draw_aum_line,
            trend_slice,
            fallback_cols,
            fallback_note("insufficient_trend", {"min_points": 2}),
            min_points=2,
            empty_reason="No AUM trend for the current filters.",
            empty_hint="Widen date range or relax filters.",
        )
