"""
External search layer for Market Intelligence mode.
Placeholder: returns mock or empty context. Wire to Tavily/SerpAPI/Bing when credentials are available.
Claude receives only this retrieved context; no internal data.
"""
from __future__ import annotations


def search_market_context(query: str, max_snippets: int = 5) -> str:
    """
    Placeholder: fetch external snippets for the query. Returns a single string of context.
    When implementing: call your search API (e.g. Tavily, SerpAPI), then join snippets.
    """
    # Placeholder: no credentials, return a short message so Claude can still respond
    return (
        "[External search not configured. Set up a search API (e.g. Tavily, SerpAPI) and wire it here.] "
        f"User asked: {query!r}. No external context was retrieved."
    )
