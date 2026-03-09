"""
Integration: ingest DATA RAW from Excel, validate columns, apply type enforcement, quality gates, persistence.
Missing required columns hard-fail before type enforcement. Returns clean df, rejects df, and LoadReport.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.ingest.excel_reader import load_data_raw_excel
from legacy.legacy_src.ingest.load_report import LoadReport
from legacy.legacy_src.persist.raw_store import persist_raw_outputs
from legacy.legacy_src.quality.gates import run_quality_gates, summarize_rejects
from legacy.legacy_src.schemas.canonical_resolver import load_canonical_columns, required_canonicals
from legacy.legacy_src.schemas.loader import load_yaml_schema, validate_schema_shape
from legacy.legacy_src.transform.standardize_columns import standardize_columns
from legacy.legacy_src.transform.profiling import apply_optional_categoricals, profile_cardinality
from legacy.legacy_src.transform.type_enforcement import DATE_FORMATS, enforce_types_data_raw

# Quality gate and persistence config (canonical names from canonical_columns.yml)
GATE_KEY_FIELDS = ["month_end", "product_ticker", "channel_raw", "src_country"]
GATE_NUMERIC_FIELDS = ["asset_under_management", "net_new_business", "net_new_base_fees"]
RAW_OUTPUT_DIR = Path("raw")
CANONICAL_SCHEMA_PATH = Path("schemas/canonical_columns.yml")


def _enable_categoricals() -> bool:
    """True if ENABLE_CATEGORICALS env var is set to a truthy value (true, 1, yes)."""
    v = os.environ.get("ENABLE_CATEGORICALS", "").strip().lower()
    return v in ("true", "1", "yes")


def _schema_type_enforcement_params(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract date_column, currency_columns, id_columns, date_formats for enforce_types_data_raw."""
    date_col = schema.get("date_column", "date")
    currency_cols = list(schema.get("currency_columns") or [])
    id_cols = list(schema.get("id_columns") or [])
    date_rules = (schema.get("parsing_rules") or {}).get("date") or {}
    date_formats = date_rules.get("date_formats")  # optional override; None -> use default
    return {
        "date_col": date_col,
        "currency_cols": currency_cols,
        "id_cols": id_cols,
        "date_formats": date_formats if date_formats is not None else DATE_FORMATS,
    }


def _try_load_manifest(manifest_path: str | Path = "data/.version.json") -> dict[str, Any] | None:
    """Load full version manifest or None if missing."""
    try:
        from legacy.legacy_src.versioning.version_manifest import load_manifest
    except ImportError:
        return None
    return load_manifest(manifest_path)


def ingest_data_raw_from_excel(
    xlsx_path: str | Path = "data/input/source.xlsx",
    sheet_name: str = "DATA RAW",
    schema_path: str | Path = "schemas/data_raw.schema.yml",
    *,
    hard_fail_missing: bool = True,
    allow_extra_columns: bool = True,
    version_manifest_path: str | Path | None = "data/.version.json",
) -> tuple[pd.DataFrame, pd.DataFrame, LoadReport]:
    """
    Load DATA RAW sheet from Excel (raw strings), validate columns (hard-fail on missing
    required columns), rename to canonical names, run type enforcement. Returns (df_clean,
    df_rejects, report). Missing required columns raise before type enforcement.
    Attaches to df_clean: load_report, type_enforcement_stats; dataset_version if manifest exists.
    """
    # 1) Read Excel as strings and validate columns (hard-fail on missing required columns)
    df_raw, report = load_data_raw_excel(
        xlsx_path,
        sheet_name=sheet_name,
        schema_path=schema_path,
        hard_fail_missing=hard_fail_missing,
        allow_extra_columns=allow_extra_columns,
    )

    schema = load_yaml_schema(schema_path)
    validate_schema_shape(schema)

    manifest = _try_load_manifest(version_manifest_path) if version_manifest_path else None

    if df_raw.empty:
        df_clean = pd.DataFrame()
        df_rejects = pd.DataFrame()
        stats = {
            "rows_in": 0,
            "rows_clean": 0,
            "rows_rejected": 0,
            "reject_counts": {},
            "currency_parse_nulls": {},
        }
        df_clean.attrs["profiling"] = {}
        rename_audit: dict[str, Any] = {}
    else:
        # 2) Standardize headers + resolve aliases to canonical names (keep unmatched; fail on collisions)
        canonical_schema = load_canonical_columns(CANONICAL_SCHEMA_PATH)
        try:
            df_canonical, rename_audit = standardize_columns(df_raw, canonical_schema_path=CANONICAL_SCHEMA_PATH)
            # 3) Validate required canonicals exist after standardization
            required = required_canonicals(canonical_schema)
            missing_required = [c for c in required if c not in df_canonical.columns]
            if missing_required:
                raise ValueError(
                    "Missing required canonical columns after standardization: "
                    f"{missing_required}. Available columns: {sorted(df_canonical.columns.tolist())}."
                )

            # 4) Type enforcement on canonical columns only
            from legacy.legacy_src.ingest.canonicalize import type_enforcement_params_from_canonical

            params = type_enforcement_params_from_canonical(canonical_schema)
            params["date_formats"] = DATE_FORMATS

            df_clean, df_rejects, stats = enforce_types_data_raw(
                df_canonical,
                date_col=params["date_col"],
                currency_cols=params["currency_cols"],
                id_cols=params["id_cols"],
                date_formats=params["date_formats"],
            )

            # 5) Optional profiling and categoricals for id_cols
            id_cols = params["id_cols"]
            id_cols_present = [c for c in id_cols if c in df_clean.columns]
            profile = profile_cardinality(df_clean, id_cols_present) if id_cols_present else {}
            if _enable_categoricals() and profile:
                df_clean = apply_optional_categoricals(df_clean, profile)
            df_clean.attrs["profiling"] = profile
        except ValueError as e:
            # Best-effort: persist ingest_report with rename_audit and error, but no clean parquet
            ingest_report: dict[str, Any] = {
                "version": {
                    "dataset_version": (manifest or {}).get("dataset_version", ""),
                    "pipeline_version": (manifest or {}).get("pipeline_version", ""),
                    "source_sha256": (manifest or {}).get("source_sha256", ""),
                },
                "load_report": report.to_dict(),
                "type_enforcement_stats": {},
                "rename_audit": rename_audit,
                "rejects_summary": {"total": 0, "by_reason": {}},
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            persist_raw_outputs(
                pd.DataFrame(),
                pd.DataFrame(),
                ingest_report,
                manifest or {},
                raw_dir=RAW_OUTPUT_DIR,
                skip_clean=True,
            )
            raise

    # 5) Attach metadata to df_clean
    df_clean.attrs["load_report"] = report.to_dict()
    df_clean.attrs["type_enforcement_stats"] = stats
    df_clean.attrs["rename_audit"] = rename_audit
    if manifest:
        df_clean.attrs["dataset_version"] = manifest.get("dataset_version", "")

    # 6) Build ingest_report and run quality gates before persist
    ingest_report: dict[str, Any] = {
        "version": {
            "dataset_version": (manifest or {}).get("dataset_version", ""),
            "pipeline_version": (manifest or {}).get("pipeline_version", ""),
            "source_sha256": (manifest or {}).get("source_sha256", ""),
        },
        "load_report": report.to_dict(),
        "type_enforcement_stats": stats,
        "rename_audit": rename_audit,
        "rejects_summary": summarize_rejects(df_rejects),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    gate_ok, gate_errors, gate_stats = run_quality_gates(
        df_clean,
        key_fields=GATE_KEY_FIELDS,
        numeric_fields=GATE_NUMERIC_FIELDS,
    )
    ingest_report["gate_stats"] = gate_stats

    # 7) Persist: always write rejects + ingest_report; write data_raw.parquet only if gates pass
    if not gate_ok:
        persist_raw_outputs(
            df_clean,
            df_rejects,
            ingest_report,
            manifest or {},
            raw_dir=RAW_OUTPUT_DIR,
            skip_clean=True,
        )
        raise ValueError("Quality gates failed: " + "; ".join(gate_errors))

    persist_raw_outputs(
        df_clean,
        df_rejects,
        ingest_report,
        manifest or {},
        raw_dir=RAW_OUTPUT_DIR,
    )

    return df_clean, df_rejects, report
