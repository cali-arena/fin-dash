"""
HTML export pipeline: audit-safe, deterministic, self-contained (inline CSS).
Table numbers formatted via app.ui.formatters for display.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from app.reporting.report_engine import SectionOutput
from app.ui.formatters import format_df, infer_common_formats


def _safe_filename(s: str, max_len: int = 60) -> str:
    """Sanitize for use in file names."""
    if not s:
        return "report"
    out = re.sub(r"[^\w\-.]", "_", str(s))
    return out[:max_len] if len(out) > max_len else out


def _stable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with deterministic column order (sorted)."""
    if df is None or df.empty:
        return df
    cols = sorted(df.columns.tolist(), key=str)
    return df.reindex(columns=[c for c in cols if c in df.columns])


def _table_to_html(df: pd.DataFrame) -> str:
    """Compact HTML table: stable columns, zebra-friendly classes, right-align numeric."""
    if df is None or df.empty:
        return "<p class=\"table-empty\"><em>No data.</em></p>"
    stable = _stable_columns(df)
    cols = list(stable.columns)
    num_indices = {i for i, c in enumerate(cols) if pd.api.types.is_numeric_dtype(stable[c])}
    html = stable.to_html(
        index=False,
        classes=["report-table"],
        na_rep="—",
        escape=True,
    )
    for i, c in enumerate(cols):
        if i in num_indices:
            html = html.replace(f"<th>{c}</th>", f"<th class=\"num\">{c}</th>", 1)
    ncols = len(cols)
    td_count = 0
    start = 0
    insert_at = []
    while True:
        pos = html.find("<td>", start)
        if pos == -1:
            break
        if (td_count % ncols) in num_indices:
            insert_at.append(pos + 4)
        td_count += 1
        start = pos + 4
    for pos in reversed(insert_at):
        html = html[:pos] + " class=\"num\"" + html[pos:]
    return html


def _escape(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


INLINE_CSS = """
.report-container { max-width: 980px; margin: 0 auto; padding: 1rem; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 16px; line-height: 1.5; color: #222; }
.report-container h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }
.report-container h2 { font-size: 1.25rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }
.report-container ul { margin: 0.5rem 0 1rem 1.25rem; padding: 0; }
.report-container li { margin: 0.25rem 0; }
.report-container .report-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin: 0.75rem 0; }
.report-container .report-table th, .report-container .report-table td { border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }
.report-container .report-table th { background: #f0f0f0; font-weight: 600; }
.report-container .report-table tr:nth-child(even) { background: #f9f9f9; }
.report-container .report-table .num { text-align: right; }
.report-container .table-empty { margin: 0.5rem 0; color: #666; font-size: 0.875rem; }
.report-container .meta-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ccc; font-size: 0.8125rem; color: #555; }
.report-container .meta-footer dt { font-weight: 600; margin-top: 0.35rem; }
.report-container .meta-footer dd { margin-left: 0; }
.report-container .meta-footer pre { background: #f5f5f5; padding: 0.75rem; overflow-x: auto; font-size: 0.8125rem; }
"""


def build_report_html(
    title: str,
    sections: list[tuple[str, SectionOutput]],
    meta: dict[str, Any],
) -> str:
    """
    Build self-contained HTML report: inline CSS, H1, per-section H2 + bullets + table, footer (meta).
    Deterministic: stable table column order, no external assets, no JS.
    """
    parts = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\"/>",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>",
        "<title>" + _escape(title) + "</title>",
        "<style>" + INLINE_CSS + "</style>",
        "</head>",
        "<body>",
        "<div class=\"report-container\">",
        "<h1>" + _escape(title) + "</h1>",
        "",
    ]
    for section_title, out in sections:
        parts.append("<h2>" + _escape(section_title) + "</h2>")
        parts.append("<ul>")
        for b in out.bullets:
            parts.append("<li>" + _escape(b) + "</li>")
        parts.append("</ul>")
        if out.table_title:
            parts.append("<p><strong>" + _escape(out.table_title) + "</strong></p>")
        tbl = out.table
        if tbl is not None and not tbl.empty:
            tbl = format_df(tbl, infer_common_formats(tbl))
        parts.append(_table_to_html(tbl))
        parts.append("")

    parts.append("<h2>Report Metadata</h2>")
    filters_used = meta.get("filters_used")
    if filters_used is not None:
        parts.append("<p><strong>Filters (JSON)</strong></p>")
        try:
            filters_json = json.dumps(filters_used, sort_keys=True, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            filters_json = str(filters_used)
        parts.append("<pre class=\"meta-footer\">" + _escape(filters_json) + "</pre>")
    recon = meta.get("reconciliation")
    if isinstance(recon, list) and recon:
        parts.append("<p><strong>Reconciliation</strong></p>")
        recon_df = pd.DataFrame(recon)
        parts.append(_table_to_html(recon_df))
    parts.append("<footer class=\"meta-footer\">")
    parts.append("<dl>")
    for k in sorted(meta.keys(), key=str):
        if k in ("filters_used", "reconciliation"):
            continue
        v = meta[k]
        parts.append("<dt>" + _escape(k) + "</dt>")
        parts.append("<dd>" + _escape(v) + "</dd>")
    parts.append("</dl>")
    parts.append("</footer>")
    parts.append("</div>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)
