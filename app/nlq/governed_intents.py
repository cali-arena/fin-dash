"""
Governed NLQ intents and templates. Strict whitelist only; no user injection.
All planned-query text is from TEMPLATE_TEXT constants.
"""
from __future__ import annotations

# Allowed intents (map ONLY to these; no other values allowed).
ALLOWED_INTENTS = [
    "firm_trend",
    "channel_mix",
    "ticker_ranking",
    "geo_split",
    "segment_view",
    "nnb_trend",
    "growth_decomposition",
    "coverage_stats",
]

# Intent -> template name (1:1; template is the intent for display).
INTENT_TO_TEMPLATE: dict[str, str] = {
    "firm_trend": "firm_trend",
    "channel_mix": "channel_mix",
    "ticker_ranking": "ticker_ranking",
    "geo_split": "geo_split",
    "segment_view": "segment_view",
    "nnb_trend": "nnb_trend",
    "growth_decomposition": "growth_decomposition",
    "coverage_stats": "coverage_stats",
}

# Constant template text per intent (no user injection). Shown as "planned query".
TEMPLATE_TEXT: dict[str, str] = {
    "firm_trend": "SELECT month_end, SUM(end_aum) AS end_aum FROM v_firm_monthly WHERE month_end BETWEEN :date_start AND :date_end GROUP BY 1 ORDER BY 1",
    "channel_mix": "SELECT channel_l1, SUM(end_aum) AS end_aum, SUM(nnb) AS nnb FROM v_channel_monthly WHERE month_end BETWEEN :date_start AND :date_end GROUP BY channel_l1 ORDER BY end_aum DESC LIMIT 10",
    "ticker_ranking": "SELECT ticker, delta_aum, delta_nnb FROM (MoM diff over v_ticker_monthly) WHERE month_end BETWEEN :date_start AND :date_end ORDER BY delta_aum DESC LIMIT 10",
    "geo_split": "SELECT geo, SUM(end_aum) AS end_aum FROM v_geo_monthly WHERE month_end BETWEEN :date_start AND :date_end GROUP BY geo ORDER BY end_aum DESC",
    "segment_view": "SELECT segment, SUM(end_aum) AS end_aum, SUM(nnb) AS nnb FROM v_segment_monthly WHERE month_end BETWEEN :date_start AND :date_end GROUP BY segment ORDER BY end_aum DESC",
    "nnb_trend": "SELECT month_end, SUM(nnb) AS nnb FROM v_firm_monthly WHERE month_end BETWEEN :date_start AND :date_end GROUP BY 1 ORDER BY 1",
    "growth_decomposition": "Waterfall inputs: organic, external, market from v_firm_monthly over :date_start..:date_end",
    "coverage_stats": "SELECT COUNT(*) AS rows_covered, min(month_end), max(month_end) FROM v_firm_monthly WHERE month_end BETWEEN :date_start AND :date_end",
}

# Intent -> gateway query name(s) executed (for commentary).
INTENT_TO_GATEWAY_QUERIES: dict[str, list[str]] = {
    "firm_trend": ["chart_aum_trend"],
    "channel_mix": ["top_channels"],
    "ticker_ranking": ["top_movers"],
    "geo_split": ["top_channels"],
    "segment_view": ["top_channels"],
    "nnb_trend": ["chart_nnb_trend"],
    "growth_decomposition": ["growth_decomposition_inputs"],
    "coverage_stats": ["coverage_stats"],
}


def get_template_text(intent: str) -> str:
    """Return constant template string for intent. No injection."""
    return TEMPLATE_TEXT.get(intent, "(unsupported)")


def is_allowed_intent(intent: str) -> bool:
    """True if intent is in whitelist."""
    return intent in ALLOWED_INTENTS
