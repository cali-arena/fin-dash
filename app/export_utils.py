"""
Traceable exports: dataset_version in filenames and metadata.
CSV: sidecar .meta.json. PDF: footer or metadata when supported.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def export_filename(base_name: str, dataset_version: str, ext: str) -> str:
    """e.g. export_kpis__ds-abc12345__2026-03-03.csv"""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ds_short = (dataset_version[:8]) if len(dataset_version) >= 8 else dataset_version
    ext = ext.lstrip(".")
    return f"{base_name}__ds-{ds_short}__{date_str}.{ext}"


def csv_export_with_meta(
    df: pd.DataFrame,
    dataset_version: str,
    pipeline_version: str,
    created_at: str,
    source_sha256: str | None = None,
    base_name: str = "export_data",
) -> tuple[str, str, str, str]:
    """
    Returns (suggested_csv_filename, csv_content, suggested_meta_filename, meta_json_content).
    Sidecar .meta.json contains dataset_version, pipeline_version, created_at, source_sha256.
    Table content unchanged; metadata in separate file.
    """
    csv_name = export_filename(base_name, dataset_version, "csv")
    meta_name = export_filename(base_name, dataset_version, "meta.json")

    csv_content = df.to_csv(index=False)
    meta: dict[str, Any] = {
        "dataset_version": dataset_version,
        "pipeline_version": pipeline_version,
        "created_at": created_at,
    }
    if source_sha256 is not None:
        meta["source_sha256"] = source_sha256
    meta_content = json.dumps(meta, sort_keys=True, indent=2)

    return csv_name, csv_content, meta_name, meta_content


def pdf_footer_text(dataset_version: str, pipeline_version: str) -> str:
    """Footer for each PDF page: dataset_version and pipeline_version (first 8 chars)."""
    ds = (dataset_version[:8]) if len(dataset_version) >= 8 else dataset_version
    pv = (pipeline_version[:8]) if len(pipeline_version) >= 8 else pipeline_version
    return f"dataset_version: {ds} | pipeline_version: {pv}"


def make_pdf_with_footer(
    title: str,
    body_lines: list[str],
    dataset_version: str,
    pipeline_version: str,
) -> bytes | None:
    """
    Build a minimal PDF with footer on each page. Returns PDF bytes or None if reportlab missing.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except ImportError:
        return None

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    footer = pdf_footer_text(dataset_version, pipeline_version)
    width, height = letter
    y = height - 0.75 * inch
    c.setFont("Helvetica", 14)
    c.drawString(0.75 * inch, y, title)
    c.setFont("Helvetica", 10)
    y -= 0.35 * inch
    for line in body_lines[:30]:
        c.drawString(0.75 * inch, y, line[:100])
        y -= 0.2 * inch
        if y < 1.5 * inch:
            c.drawString(0.75 * inch, 0.75 * inch, footer)
            c.showPage()
            y = height - 0.75 * inch
    c.drawString(0.75 * inch, 0.75 * inch, footer)
    c.save()
    buf.seek(0)
    return buf.read()
