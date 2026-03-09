"""Deterministic executive insight generation from dashboard metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.ui.formatters import fmt_currency, fmt_percent


@dataclass(frozen=True)
class ExecutiveInsightSection:
    title: str
    sentences: list[str]


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return v


def _fmt_money(x: Any) -> str:
    v = _num(x)
    if v is None:
        return "-"
    core = fmt_currency(abs(v), unit="auto", decimals=2)
    return f"-${core}" if v < 0 else f"${core}"


def _fmt_pct(x: Any) -> str:
    v = _num(x)
    if v is None:
        return "-"
    return fmt_percent(v, decimals=2, signed=False)


def _latest_row(snapshot_df: pd.DataFrame) -> dict[str, Any]:
    if snapshot_df is None or snapshot_df.empty:
        return {}
    if "month_end" in snapshot_df.columns:
        work = snapshot_df.copy()
        work["month_end"] = pd.to_datetime(work["month_end"], errors="coerce")
        work = work.sort_values("month_end")
        if not work.empty:
            return work.iloc[-1].to_dict()
    return snapshot_df.iloc[-1].to_dict()


def _portfolio_section(latest: dict[str, Any]) -> ExecutiveInsightSection:
    end_aum = _num(latest.get("end_aum"))
    mom_growth = _num(latest.get("mom_growth"))
    ytd_growth = _num(latest.get("ytd_growth"))
    nnb = _num(latest.get("nnb"))
    market_impact = _num(latest.get("market_impact"))

    if end_aum is None:
        return ExecutiveInsightSection(
            title="Portfolio Performance",
            sentences=[
                "AUM performance is unavailable for the selected filters.",
                "Select a wider period to populate month-over-month and year-to-date context.",
            ],
        )

    direction = "increased" if (mom_growth or 0.0) > 0 else "decreased" if (mom_growth or 0.0) < 0 else "was flat"
    first = (
        f"Assets under management {direction} to {_fmt_money(end_aum)}, "
        f"with MoM growth of {_fmt_pct(mom_growth)} and YTD growth of {_fmt_pct(ytd_growth)}."
    )

    if nnb is None and market_impact is None:
        second = "Primary drivers are unavailable for this selection."
    elif nnb is None:
        second = f"Performance was driven by market impact of {_fmt_money(market_impact)} while net inflow data is unavailable."
    elif market_impact is None:
        second = f"Performance was flow-driven, with net new business of {_fmt_money(nnb)}."
    elif abs(market_impact) > abs(nnb):
        second = (
            f"Period change was primarily market-driven, with market impact of {_fmt_money(market_impact)} "
            f"versus net new business of {_fmt_money(nnb)}."
        )
    elif abs(nnb) > abs(market_impact):
        second = (
            f"Period change was primarily flow-driven, with net new business of {_fmt_money(nnb)} "
            f"versus market impact of {_fmt_money(market_impact)}."
        )
    else:
        second = (
            f"Flows and market effects were balanced, with net new business of {_fmt_money(nnb)} "
            f"and market impact of {_fmt_money(market_impact)}."
        )
    return ExecutiveInsightSection(title="Portfolio Performance", sentences=[first, second])


def _distribution_section(ranked_channels: pd.DataFrame) -> ExecutiveInsightSection:
    if ranked_channels is None or ranked_channels.empty:
        return ExecutiveInsightSection(
            title="Distribution Dynamics",
            sentences=[
                "Channel contribution is unavailable for the selected filters.",
                "Use a broader slice to evaluate concentration and organic growth by channel.",
            ],
        )

    work = ranked_channels.copy()
    work["AUM"] = pd.to_numeric(work.get("AUM"), errors="coerce").fillna(0.0)
    work["NNB"] = pd.to_numeric(work.get("NNB"), errors="coerce").fillna(0.0)
    total_aum = float(work["AUM"].sum())
    top_aum_row = work.sort_values(["AUM", "channel"], ascending=[False, True]).iloc[0]
    top_nnb_row = work.sort_values(["NNB", "channel"], ascending=[False, True]).iloc[0]
    positive_channels = int((work["NNB"] > 0).sum())
    share = (float(top_aum_row["AUM"]) / total_aum * 100.0) if total_aum else 0.0

    first = (
        f"Distribution remains concentrated in {top_aum_row['channel']}, "
        f"representing {share:.1f}% of tracked channel AUM."
    )
    if float(top_nnb_row["NNB"]) <= 0:
        second = (
            f"Organic growth was limited, with the strongest channel NNB at {_fmt_money(top_nnb_row['NNB'])} "
            f"and only {positive_channels} channels in positive flow."
        )
    else:
        second = (
            f"{top_nnb_row['channel']} led flow generation with {_fmt_money(top_nnb_row['NNB'])}, "
            f"while {positive_channels} channels delivered positive net new business."
        )
    return ExecutiveInsightSection(title="Distribution Dynamics", sentences=[first, second])


def _product_section(ranked_tickers: pd.DataFrame) -> ExecutiveInsightSection:
    if ranked_tickers is None or ranked_tickers.empty:
        return ExecutiveInsightSection(
            title="Product Dynamics",
            sentences=[
                "Product contribution is unavailable for the selected filters.",
                "Expand the window to evaluate concentration and flow dispersion across tickers.",
            ],
        )

    work = ranked_tickers.copy()
    work["AUM"] = pd.to_numeric(work.get("AUM"), errors="coerce").fillna(0.0)
    work["NNB"] = pd.to_numeric(work.get("NNB"), errors="coerce").fillna(0.0)
    total_aum = float(work["AUM"].sum())
    top_aum_row = work.sort_values(["AUM", "ticker"], ascending=[False, True]).iloc[0]
    top_nnb_row = work.sort_values(["NNB", "ticker"], ascending=[False, True]).iloc[0]
    declining_count = int((work["NNB"] < 0).sum())
    share = (float(top_aum_row["AUM"]) / total_aum * 100.0) if total_aum else 0.0

    first = (
        f"Product contribution is concentrated in {top_aum_row['ticker']}, "
        f"which represents {share:.1f}% of tracked product AUM."
    )
    second = (
        f"{top_nnb_row['ticker']} is the top flow contributor at {_fmt_money(top_nnb_row['NNB'])}, "
        f"while {declining_count} tickers show declining flows."
    )
    return ExecutiveInsightSection(title="Product Dynamics", sentences=[first, second])


def _movers_section(ranked_tickers: pd.DataFrame) -> ExecutiveInsightSection:
    if ranked_tickers is None or ranked_tickers.empty:
        return ExecutiveInsightSection(
            title="Movers",
            sentences=[
                "Top and bottom movers are unavailable for the selected filters.",
                "Expand the date window to compare leading and lagging product flows.",
            ],
        )

    work = ranked_tickers.copy()
    work["NNB"] = pd.to_numeric(work.get("NNB"), errors="coerce").fillna(0.0)
    top_row = work.sort_values(["NNB", "ticker"], ascending=[False, True]).iloc[0]
    bottom_row = work.sort_values(["NNB", "ticker"], ascending=[True, True]).iloc[0]
    non_positive = int((work["NNB"] <= 0).sum())

    first = (
        f"Top mover is {top_row['ticker']} with net flow of {_fmt_money(top_row['NNB'])}, "
        f"while the weakest mover is {bottom_row['ticker']} at {_fmt_money(bottom_row['NNB'])}."
    )
    second = (
        f"Flow breadth remains selective, with {non_positive} of {len(work)} tracked tickers "
        "delivering flat or negative net flow."
    )
    return ExecutiveInsightSection(title="Movers", sentences=[first, second])


def build_executive_insight_sections(
    *,
    snapshot_df: pd.DataFrame,
    ranked_channels: pd.DataFrame,
    ranked_tickers: pd.DataFrame,
) -> list[ExecutiveInsightSection]:
    """
    Return deterministic executive insight sections driven only by dashboard metrics.
    Each section contains exactly two short sentences.
    """
    latest = _latest_row(snapshot_df if isinstance(snapshot_df, pd.DataFrame) else pd.DataFrame())
    return [
        _portfolio_section(latest),
        _distribution_section(ranked_channels if isinstance(ranked_channels, pd.DataFrame) else pd.DataFrame()),
        _product_section(ranked_tickers if isinstance(ranked_tickers, pd.DataFrame) else pd.DataFrame()),
        _movers_section(ranked_tickers if isinstance(ranked_tickers, pd.DataFrame) else pd.DataFrame()),
    ]
