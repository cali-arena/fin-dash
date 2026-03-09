"""
Canonical metric output schema: narrative placeholders map to computed numbers from df + filters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd

from app.ui.formatters import fmt_number, fmt_percent

# Stable column names for tables produced by parameterized metrics
MOVERS_COLUMNS = ["name", "value", "rank", "segment"]
MIX_SHIFT_COLUMNS = ["name", "share", "share_delta", "share_prior"]


def _coerce(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _fmt_num(x: Any, decimals: int = 1) -> str:
    return fmt_number(x, decimals=decimals)


def _fmt_pct(x: Any, decimals: int = 1) -> str:
    return fmt_percent(x, decimals=decimals, signed=False)


def _fmt_rate(x: Any, decimals: int = 4) -> str:
    return fmt_percent(x, decimals=decimals, signed=False)


@dataclass
class MetricSpec:
    id: str
    description: str
    deps: list[str]
    level: Literal["firm", "channel", "ticker", "geo"]
    compute_fn: Callable[..., Any]
    output_fields: list[str] = field(default_factory=list)


# --- Firm-level metric implementations (df has month_end, begin_aum, end_aum, nnb, nnf) ---

def _firm_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    agg = df.agg({
        "begin_aum": "sum" if "begin_aum" in df.columns else "first",
        "end_aum": "sum" if "end_aum" in df.columns else "first",
        "nnb": "sum" if "nnb" in df.columns else "first",
        "nnf": "sum" if "nnf" in df.columns else "first",
    })
    return agg


def _compute_firm_end_aum(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    if df is None or df.empty or "end_aum" not in df.columns:
        return {"end_aum": float("nan")}
    if "month_end" in df.columns:
        last = df.sort_values("month_end").tail(1)
        v = last["end_aum"].sum()
    else:
        v = df["end_aum"].sum()
    return {"end_aum": v}


def _compute_firm_mom_pct(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    if df is None or df.empty or "month_end" not in df.columns or "end_aum" not in df.columns:
        return {"mom_pct": float("nan")}
    by_month = df.groupby("month_end", as_index=False)["end_aum"].sum()
    by_month = by_month.sort_values("month_end")
    if len(by_month) < 2:
        return {"mom_pct": float("nan")}
    cur = by_month["end_aum"].iloc[-1]
    prior = by_month["end_aum"].iloc[-2]
    if _coerce(prior) == 0:
        return {"mom_pct": float("nan")}
    return {"mom_pct": (cur - prior) / prior}


def _compute_firm_ytd_pct(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    if df is None or df.empty or "month_end" not in df.columns or "end_aum" not in df.columns:
        return {"ytd_pct": float("nan")}
    by_month = df.groupby("month_end", as_index=False)["end_aum"].sum()
    by_month = by_month.sort_values("month_end")
    if len(by_month) < 2:
        return {"ytd_pct": float("nan")}
    cur = by_month["end_aum"].iloc[-1]
    start = by_month["end_aum"].iloc[0]
    if _coerce(start) == 0:
        return {"ytd_pct": float("nan")}
    return {"ytd_pct": (cur - start) / start}


def _compute_firm_nnb(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    if df is None or df.empty or "nnb" not in df.columns:
        return {"nnb": float("nan")}
    return {"nnb": df["nnb"].sum()}


def _compute_firm_ogr(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from app.metrics.metric_contract import compute_ogr
    s = _firm_series(df)
    begin = s.get("begin_aum", float("nan"))
    nnb = s.get("nnb", float("nan"))
    return {"ogr": compute_ogr(nnb, begin)}


def _compute_firm_market_impact_abs(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from app.metrics.metric_contract import compute_market_impact
    s = _firm_series(df)
    begin = s.get("begin_aum", float("nan"))
    end = s.get("end_aum", float("nan"))
    nnb = s.get("nnb", float("nan"))
    return {"market_impact_abs": compute_market_impact(begin, end, nnb)}


def _compute_firm_market_impact_rate(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from app.metrics.metric_contract import compute_market_impact, compute_market_impact_rate
    s = _firm_series(df)
    begin = s.get("begin_aum", float("nan"))
    end = s.get("end_aum", float("nan"))
    nnb = s.get("nnb", float("nan"))
    mi = compute_market_impact(begin, end, nnb)
    return {"market_impact_rate": compute_market_impact_rate(mi, begin)}


def _compute_firm_fee_yield(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from app.metrics.metric_contract import compute_fee_yield
    s = _firm_series(df)
    begin = s.get("begin_aum", float("nan"))
    end = s.get("end_aum", float("nan"))
    nnf = s.get("nnf", float("nan"))
    nnb_val = s.get("nnb", float("nan"))
    return {"fee_yield": compute_fee_yield(nnf, begin, end, nnb=nnb_val)}


def _compute_movers_top_bottom(
    df: pd.DataFrame,
    filters: dict[str, Any],
    *,
    dimension: str = "channel",
    metric: str = "nnb",
    n: int = 5,
    **kwargs: Any,
) -> pd.DataFrame:
    name_col = dimension if dimension in (df.columns if df is not None else []) else None
    val_col = metric if df is not None and metric in df.columns else "nnb"
    if df is None or df.empty or name_col is None:
        return pd.DataFrame(columns=MOVERS_COLUMNS)
    agg = df.groupby(name_col, as_index=False)[val_col].sum()
    agg = agg.sort_values(val_col, ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    top = agg.head(n).copy()
    top["segment"] = "top"
    bottom = agg.tail(n).copy()
    bottom["segment"] = "bottom"
    out = pd.concat([top, bottom], ignore_index=True)
    out = out.rename(columns={name_col: "name", val_col: "value"})
    for c in MOVERS_COLUMNS:
        if c not in out.columns:
            out[c] = None
    return out[MOVERS_COLUMNS]


def _compute_mix_shift_share_delta(
    df: pd.DataFrame,
    filters: dict[str, Any],
    *,
    dimension: str = "channel",
    share_on: str = "end_aum",
    **kwargs: Any,
) -> pd.DataFrame:
    name_col = dimension if df is not None and dimension in df.columns else None
    if df is None or df.empty or name_col is None or share_on not in (df.columns if df is not None else []):
        return pd.DataFrame(columns=MIX_SHIFT_COLUMNS)
    by_dim = df.groupby(name_col, as_index=False)[share_on].sum()
    total = by_dim[share_on].sum()
    if _coerce(total) == 0:
        by_dim["share"] = float("nan")
        by_dim["share_delta"] = float("nan")
        by_dim["share_prior"] = float("nan")
    else:
        by_dim["share"] = by_dim[share_on] / total
        by_dim["share_delta"] = float("nan")
        by_dim["share_prior"] = float("nan")
    by_dim = by_dim.rename(columns={name_col: "name"})
    for c in MIX_SHIFT_COLUMNS:
        if c not in by_dim.columns:
            by_dim[c] = None
    return by_dim[MIX_SHIFT_COLUMNS]


def _compute_anomalies_zscore(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    if df is None or df.empty or "month_end" not in df.columns:
        return {"zscore_flag": ""}
    by_m = df.groupby("month_end", as_index=False).sum(numeric_only=True)
    if "end_aum" not in by_m.columns or len(by_m) < 3:
        return {"zscore_flag": ""}
    s = by_m["end_aum"].rolling(3, min_periods=2).mean()
    std = by_m["end_aum"].rolling(3, min_periods=2).std()
    z = (by_m["end_aum"] - s) / std.replace(0, float("nan"))
    flags = z.abs() > 2
    out = ", ".join(by_m.loc[flags, "month_end"].astype(str).tolist()) if flags.any() else "none"
    return {"zscore_flag": out}


def _compute_anomalies_rolling_std(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    if df is None or df.empty or "month_end" not in df.columns:
        return {"rolling_std_flag": ""}
    by_m = df.groupby("month_end", as_index=False).sum(numeric_only=True)
    if "end_aum" not in by_m.columns or len(by_m) < 3:
        return {"rolling_std_flag": ""}
    r = by_m["end_aum"].rolling(3, min_periods=2).std()
    high = r > (r.mean() + 2 * r.std()) if r.notna().any() else pd.Series(dtype=bool)
    out = ", ".join(by_m.loc[high, "month_end"].astype(str).tolist()) if high.any() else "none"
    return {"rolling_std_flag": out}


def _compute_reversals(df: pd.DataFrame, filters: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    if df is None or df.empty or "month_end" not in df.columns or "nnb" not in df.columns:
        return {"reversals_list": "", "anomaly_count": 0}
    by_m = df.groupby("month_end", as_index=False)["nnb"].sum()
    by_m = by_m.sort_values("month_end")
    if len(by_m) < 2:
        return {"reversals_list": "", "anomaly_count": 0}
    sign = by_m["nnb"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    rev = sign.diff().fillna(0).ne(0) & sign.ne(0)
    rev_dates = by_m.loc[rev, "month_end"].astype(str).tolist()
    return {"reversals_list": ", ".join(rev_dates), "anomaly_count": len(rev_dates) + 0}


# --- Registry ---

METRIC_REGISTRY: dict[str, MetricSpec] = {
    "firm_end_aum": MetricSpec(
        id="firm_end_aum",
        description="Period-end AUM at firm level",
        deps=["end_aum", "month_end"],
        level="firm",
        compute_fn=_compute_firm_end_aum,
        output_fields=["end_aum"],
    ),
    "firm_mom_pct": MetricSpec(
        id="firm_mom_pct",
        description="Firm MoM AUM growth percent",
        deps=["month_end", "end_aum"],
        level="firm",
        compute_fn=_compute_firm_mom_pct,
        output_fields=["mom_pct"],
    ),
    "firm_ytd_pct": MetricSpec(
        id="firm_ytd_pct",
        description="Firm YTD AUM growth percent",
        deps=["month_end", "end_aum"],
        level="firm",
        compute_fn=_compute_firm_ytd_pct,
        output_fields=["ytd_pct"],
    ),
    "firm_nnb": MetricSpec(
        id="firm_nnb",
        description="Firm net new business",
        deps=["nnb"],
        level="firm",
        compute_fn=_compute_firm_nnb,
        output_fields=["nnb"],
    ),
    "firm_ogr": MetricSpec(
        id="firm_ogr",
        description="Firm organic growth rate",
        deps=["begin_aum", "nnb"],
        level="firm",
        compute_fn=_compute_firm_ogr,
        output_fields=["ogr"],
    ),
    "firm_market_impact_abs": MetricSpec(
        id="firm_market_impact_abs",
        description="Firm market impact (absolute)",
        deps=["begin_aum", "end_aum", "nnb"],
        level="firm",
        compute_fn=_compute_firm_market_impact_abs,
        output_fields=["market_impact_abs"],
    ),
    "firm_market_impact_rate": MetricSpec(
        id="firm_market_impact_rate",
        description="Firm market impact rate",
        deps=["begin_aum", "end_aum", "nnb"],
        level="firm",
        compute_fn=_compute_firm_market_impact_rate,
        output_fields=["market_impact_rate"],
    ),
    "firm_fee_yield": MetricSpec(
        id="firm_fee_yield",
        description="Firm fee yield",
        deps=["begin_aum", "end_aum", "nnf"],
        level="firm",
        compute_fn=_compute_firm_fee_yield,
        output_fields=["fee_yield"],
    ),
    "movers_top_bottom": MetricSpec(
        id="movers_top_bottom",
        description="Top/bottom movers by dimension and metric",
        deps=[],
        level="channel",
        compute_fn=_compute_movers_top_bottom,
        output_fields=[],
    ),
    "mix_shift_share_delta": MetricSpec(
        id="mix_shift_share_delta",
        description="Share-of-total delta",
        deps=[],
        level="channel",
        compute_fn=_compute_mix_shift_share_delta,
        output_fields=[],
    ),
    "anomalies_zscore": MetricSpec(
        id="anomalies_zscore",
        description="Anomaly detection via z-score",
        deps=["month_end", "end_aum"],
        level="firm",
        compute_fn=_compute_anomalies_zscore,
        output_fields=["zscore_flag"],
    ),
    "anomalies_rolling_std": MetricSpec(
        id="anomalies_rolling_std",
        description="Anomaly via rolling std",
        deps=["month_end", "end_aum"],
        level="firm",
        compute_fn=_compute_anomalies_rolling_std,
        output_fields=["rolling_std_flag"],
    ),
    "reversals": MetricSpec(
        id="reversals",
        description="Sign reversals",
        deps=["month_end", "nnb"],
        level="firm",
        compute_fn=_compute_reversals,
        output_fields=["reversals_list", "anomaly_count"],
    ),
}


def _section_dimension(section_id: str) -> str:
    if section_id == "channel_commentary":
        return "channel"
    if section_id == "product_etf_commentary":
        return "ticker"
    if section_id == "geo_commentary":
        return "geo"
    return "channel"


def _format_context_values(raw: dict[str, Any], section_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k in ("mom_pct", "ytd_pct", "share", "share_delta"):
            out[k] = _fmt_pct(v)
        elif k in ("ogr", "market_impact_rate", "fee_yield"):
            out[k] = _fmt_rate(v)
        elif k in ("end_aum", "nnb", "market_impact_abs"):
            out[k] = _fmt_num(v, 0)
        elif k == "mkt_abs":
            out[k] = _fmt_num(raw.get("market_impact_abs", v), 0)
        elif k == "mkt_rate":
            out[k] = _fmt_rate(raw.get("market_impact_rate", v))
        elif isinstance(v, (pd.DataFrame, list)) or k.endswith("_list") or k.endswith("_flag"):
            out[k] = str(v) if v is not None else "—"
        else:
            out[k] = str(v) if v is not None else "—"
    if "market_impact_abs" in raw and "mkt_abs" not in out:
        out["mkt_abs"] = _fmt_num(raw["market_impact_abs"], 0)
    if "market_impact_rate" in raw and "mkt_rate" not in out:
        out["mkt_rate"] = _fmt_rate(raw["market_impact_rate"])
    return out


def compute_section_context(
    section_id: str,
    df: pd.DataFrame,
    filters: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """
    Returns a dict of computed values and prepared tables for the section.
    All placeholders are derived from df + filters; values formatted for template insertion.
    """
    if contract is None:
        contract = {}
    sections = contract.get("sections") or []
    sec = next((s for s in sections if s.get("id") == section_id), None)
    if not sec:
        return {}

    required_metrics = sec.get("required_metrics") or []
    dim = _section_dimension(section_id)
    filters = filters or {}
    raw: dict[str, Any] = {}
    tables: dict[str, pd.DataFrame] = {}

    for mid in required_metrics:
        spec = METRIC_REGISTRY.get(mid)
        if spec is None:
            continue
        try:
            if mid == "movers_top_bottom":
                tbl = spec.compute_fn(df, filters, dimension=dim, metric="nnb", n=5)
                tables["movers"] = tbl
                if not tbl.empty and "name" in tbl.columns:
                    raw["movers_list"] = ", ".join(tbl["name"].astype(str).tolist())
                else:
                    raw["movers_list"] = ""
            elif mid == "mix_shift_share_delta":
                tbl = spec.compute_fn(df, filters, dimension=dim, share_on="end_aum")
                tables["mix_shift"] = tbl
                if not tbl.empty and len(tbl) > 0:
                    row = tbl.iloc[0]
                    raw["share"] = row.get("share", float("nan"))
                    raw["share_delta"] = row.get("share_delta", float("nan"))
                    raw["share_prior"] = row.get("share_prior", float("nan"))
                    raw["name"] = row.get("name", "")
                if dim == "channel" and "name" in raw:
                    raw["channel"] = raw.get("name", "")
                elif dim == "ticker" and "name" in raw:
                    raw["ticker"] = raw.get("name", "")
                elif dim == "geo" and "name" in raw:
                    raw["geo_country"] = raw.get("name", "")
            else:
                res = spec.compute_fn(df, filters)
                if isinstance(res, dict):
                    raw.update(res)
                elif isinstance(res, pd.DataFrame) and not res.empty:
                    tables[mid] = res
        except Exception:
            pass

    if section_id == "overview_firm":
        raw["period_label"] = str(filters.get("date_start", "")) + " – " + str(filters.get("date_end", ""))
    if section_id in ("channel_commentary", "product_etf_commentary", "geo_commentary") and tables.get("movers") is not None and not tables["movers"].empty:
        if "movers_list" not in raw:
            raw["movers_list"] = ", ".join(tables["movers"]["name"].astype(str).tolist())
    name_col = dim if dim in (df.columns if df is not None else []) else None
    if section_id in ("channel_commentary", "product_etf_commentary", "geo_commentary") and name_col and df is not None and not df.empty:
        agg_cols = [c for c in ("end_aum", "nnb") if c in df.columns]
        if agg_cols:
            by_dim = df.groupby(name_col, as_index=False)[agg_cols].sum()
            if not by_dim.empty:
                row0 = by_dim.iloc[0]
                raw.setdefault("nnb", row0.get("nnb", float("nan")))
                raw.setdefault("end_aum", row0.get("end_aum", float("nan")))
                raw.setdefault("name", row0.get(name_col, ""))
                if dim == "channel":
                    raw.setdefault("channel", row0.get(name_col, ""))
                elif dim == "ticker":
                    raw.setdefault("ticker", row0.get(name_col, ""))
                elif dim == "geo":
                    raw.setdefault("geo_country", row0.get(name_col, ""))
    if section_id in ("channel_commentary", "product_etf_commentary", "geo_commentary"):
        if dim in ("channel", "ticker") and tables.get("mix_shift") is not None and not tables["mix_shift"].empty:
            ms = tables["mix_shift"]
            for _, row in ms.head(1).iterrows():
                raw["share"] = row.get("share", raw.get("share", float("nan")))
                raw["share_delta"] = row.get("share_delta", raw.get("share_delta", float("nan")))
                raw["name"] = row.get("name", "")
                if dim == "channel":
                    raw["channel"] = row.get("name", "")
                elif dim == "ticker":
                    raw["ticker"] = row.get("name", "")
                break
        if dim == "geo" and tables.get("mix_shift") is not None and not tables["mix_shift"].empty:
            row = tables["mix_shift"].iloc[0]
            raw["geo_country"] = row.get("name", "")
            raw["share"] = row.get("share", float("nan"))
            raw["share_delta"] = row.get("share_delta", float("nan"))
        rank_col = tables.get("movers")
        if rank_col is not None and not rank_col.empty and "rank" in rank_col.columns and "name" in rank_col.columns:
            first_name = raw.get("channel") or raw.get("ticker") or raw.get("geo_country") or raw.get("name")
            if first_name is not None:
                r = rank_col[rank_col["name"].astype(str) == str(first_name)]
                raw["rank"] = int(r["rank"].iloc[0]) if len(r) > 0 else 0
            else:
                raw["rank"] = 0
        else:
            raw["rank"] = 0
        if section_id == "product_etf_commentary":
            raw["etf_flag"] = filters.get("etf_flag", "—")
    if section_id == "recommendations":
        raw["focus_channel"] = filters.get("focus_channel", "—")
        raw["focus_ticker"] = filters.get("focus_ticker", "—")
        raw["focus_geo"] = filters.get("focus_geo", "—")
        raw["action_summary"] = filters.get("action_summary", "—")

    formatted = _format_context_values(raw, section_id)
    out = dict(formatted)
    out["_tables"] = tables
    out["_raw"] = raw
    return out


def assert_placeholders_fulfilled(
    section_id: str,
    contract: dict[str, Any],
    section_context: dict[str, Any],
) -> None:
    """
    Before rendering templates: assert every placeholder required by the section's templates
    exists in section_context. If any missing -> raise ValueError listing missing + section/template id.
    """
    from app.reporting.report_contract import _parse_placeholders_from_text

    narrative = contract.get("narrative") or {}
    templates = (narrative.get("templates") or {}).get(section_id) or []
    allowed = set(section_context.keys()) - {"_tables", "_raw"}
    missing_by_template: list[tuple[str, set[str]]] = []
    for t in templates:
        tid = t.get("id", "?")
        placeholders = set(t.get("placeholders") or [])
        in_text = _parse_placeholders_from_text(t.get("text") or "")
        required = placeholders | in_text
        missing = required - allowed
        if missing:
            missing_by_template.append((tid, missing))
    if missing_by_template:
        parts = [f"section {section_id!r}, template {tid!r}: missing placeholders {sorted(miss)}" for tid, miss in missing_by_template]
        raise ValueError("Placeholders not in section_context: " + "; ".join(parts))
