"""
Dashboard data loader: single entry point for enriched fact (join contract).
UI must load curated/fact_enriched.parquet OR call load_fact_enriched() — no direct merges in app.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

FACT_ENRICHED_REL = "curated/fact_enriched.parquet"
QA_JOIN_COVERAGE_REL = "qa/join_coverage.json"


def get_dataset(
    mode: str = "enriched",
    root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load the dataset for the UI. mode="enriched" only (enforced).
    - Tries to read curated/fact_enriched.parquet.
    - If missing, calls load_fact_enriched() and writes it.
    - Returns (df, join_coverage). join_coverage is from qa/join_coverage.json if exists, else {}.
    """
    root = Path(root) if root is not None else Path.cwd()
    if mode != "enriched":
        raise ValueError(f"Only mode='enriched' is allowed for UI; got mode={mode!r}")

    enriched_path = root / FACT_ENRICHED_REL
    if enriched_path.exists():
        df = pd.read_parquet(enriched_path)
        logger.info("Loaded fact_enriched from %s (%d rows)", enriched_path, len(df))
    else:
        from app.contracts.star_contract import load_fact_enriched

        df = load_fact_enriched(root=root, write_output=True)
        logger.info("Built fact_enriched via join contract (%d rows)", len(df))

    join_coverage: dict[str, Any] = {}
    coverage_path = root / QA_JOIN_COVERAGE_REL
    if coverage_path.exists():
        try:
            join_coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not load join_coverage from %s: %s", coverage_path, e)

    return df, join_coverage


def check_ui_no_merge(app_dir: Path | None = None, strict: bool | None = None) -> list[tuple[str, int, str]]:
    """
    Static anti-pattern check: find pd.merge( or .merge( in app Python files (actual code, not comments).
    Returns list of (file_path, line_no, line_text). If strict=True and list non-empty, raises.
    strict: from env STRICT_UI_JOINS (true/false) if None.
    """
    if app_dir is None:
        app_dir = Path(__file__).resolve().parent
    app_dir = Path(app_dir)
    if strict is None:
        strict = os.environ.get("STRICT_UI_JOINS", "").strip().lower() == "true"

    violations: list[tuple[str, int, str]] = []
    this_file = Path(__file__).resolve()
    for py in app_dir.rglob("*.py"):
        if py.resolve() == this_file:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "pd.merge(" in line or ".merge(" in line:
                violations.append((str(py), i, stripped))
    if violations:
        msg_lines = [
            "UI join anti-pattern: direct .merge( / pd.merge( found in app. Use get_dataset() / fact_enriched only.",
            "",
        ]
        for path, line_no, line in violations:
            msg_lines.append(f"  {path}:{line_no}: {line[:80]}")
        msg = "\n".join(msg_lines)
        logger.warning(msg)
        if strict:
            raise RuntimeError(msg)
    return violations


def check_ui_no_groupby(app_dir: Path | None = None, strict: bool | None = None) -> list[tuple[str, int, str]]:
    """
    Static anti-pattern check: find .groupby( in app Python files (actual code, not comments).
    Core metrics must come from curated/metrics_monthly.parquet filtered by (path_id, slice_id); no groupby in UI.
    Returns list of (file_path, line_no, line_text). If strict=True and list non-empty, raises.
    strict: from env STRICT_UI_SLICES (true/false) if None.
    """
    if app_dir is None:
        app_dir = Path(__file__).resolve().parent
    app_dir = Path(app_dir)
    if strict is None:
        strict = os.environ.get("STRICT_UI_SLICES", "").strip().lower() == "true"

    violations: list[tuple[str, int, str]] = []
    this_file = Path(__file__).resolve()
    for py in app_dir.rglob("*.py"):
        if py.resolve() == this_file:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "groupby(" in line:
                violations.append((str(py), i, stripped))
    if violations:
        msg_lines = [
            "UI slice contract: .groupby( found in app. Use drill path + slice and metrics_monthly only (filtering + plotting).",
            "",
        ]
        for path, line_no, line in violations:
            msg_lines.append(f"  {path}:{line_no}: {line[:80]}")
        msg = "\n".join(msg_lines)
        logger.warning(msg)
        if strict:
            raise RuntimeError(msg)
    return violations
