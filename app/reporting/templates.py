"""Deterministic report templates.

Each template slot is filled only with computed values from ReportPack.
"""
from __future__ import annotations

OVERVIEW_TEMPLATES = [
    "Month-end AUM {end_aum} with MoM {mom_pct} and YTD {ytd_pct}.",
    "NNB {nnb}, OGR {ogr}, market impact rate {market_impact_rate}.",
]

CHANNEL_TEMPLATES = [
    "Top channel: {top_channel} ({top_channel_value}).",
    "Bottom channel: {bottom_channel} ({bottom_channel_value}).",
]

PRODUCT_TEMPLATES = [
    "Top ticker: {top_ticker} ({top_ticker_value}).",
    "Bottom ticker: {bottom_ticker} ({bottom_ticker_value}).",
]

GEO_TEMPLATES = [
    "Top geography: {top_geo} ({top_geo_value}).",
    "Bottom geography: {bottom_geo} ({bottom_geo_value}).",
]

