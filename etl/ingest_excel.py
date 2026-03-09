from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_SHEETS = [
    "DATA RAW",
    "DATA SUMMARY",
    "DATA MAPPING",
    "ETF",
    "Executive Summary",
]
PIVOT_SHEETS = [
    "All AUM by...",
    "All by Date",
    "All by SubSegment",
    "All by Date&Ticker",
]

RAW_FALLBACK_CSV = "Distribution Analytics - Pretotype(DATA RAW).csv"
SUMMARY_FALLBACK_CSV = "Distribution Analytics - Pretotype(DATA SUMMARY).csv"
MAPPING_FALLBACK_CSV = "Distribution Analytics - Pretotype(DATA MAPPING).csv"
ETF_FALLBACK_CSV = "Distribution Analytics - Pretotype(ETF).csv"
EXEC_FALLBACK_CSV = "Distribution Analytics - Pretotype(Executive Summary).csv"

RAW_COLUMN_MAP = {
    "date": "month_end",
    "channel": "channel_raw",
    "standard_channel": "channel_standard",
    "best_of_source": "channel_best",
    "src_country": "src_country",
    "product_country": "product_country",
    "asset_under_management": "end_aum",
    "net_new_business": "nnb",
    "net_new_base_fees": "nnf",
    "display_firm": "display_firm",
    "product_ticker": "product_ticker",
    "segment": "segment",
    "sub_segment": "sub_segment",
    "master_custodian_firm": "master_custodian_firm",
}

NUMERIC_COLUMNS = ["end_aum", "nnb", "nnf", "channel_best"]


def _snake(name: str) -> str:
    txt = str(name or "").strip().lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    return txt.strip("_")


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str)


def _read_sheet_or_csv(excel_path: Path, sheet_name: str, fallback_csv: str) -> pd.DataFrame:
    if excel_path.exists():
        return pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str, engine="openpyxl")
    csv_path = excel_path.parent / fallback_csv
    if not csv_path.exists():
        csv_path = Path.cwd() / fallback_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source for {sheet_name}: {excel_path} and {fallback_csv}")
    return _read_csv_with_fallback(csv_path)


def _parse_number(value: Any) -> float:
    if value is None:
        return float("nan")
    txt = str(value).strip()
    if txt == "" or txt.lower() in {"nan", "none", "null"}:
        return float("nan")
    txt = txt.replace(",", "")
    neg = txt.startswith("(") and txt.endswith(")")
    if neg:
        txt = txt[1:-1]
    try:
        num = float(txt)
    except ValueError:
        return float("nan")
    return -num if neg else num


def _normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_snake(c) for c in out.columns]
    rename_map: dict[str, str] = {}
    for col in out.columns:
        mapped = RAW_COLUMN_MAP.get(col)
        if mapped:
            rename_map[col] = mapped
    out = out.rename(columns=rename_map)

    required = [
        "month_end",
        "channel_raw",
        "channel_standard",
        "channel_best",
        "src_country",
        "product_country",
        "end_aum",
        "nnb",
        "nnf",
        "product_ticker",
        "segment",
        "sub_segment",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"DATA RAW missing required canonical columns: {missing}")

    out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    for c in NUMERIC_COLUMNS:
        if c in out.columns:
            out[c] = out[c].map(_parse_number).astype("float64")

    for c in [
        "channel_raw",
        "channel_standard",
        "src_country",
        "product_country",
        "product_ticker",
        "segment",
        "sub_segment",
        "display_firm",
        "master_custodian_firm",
    ]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    if "channel_best" in out.columns:
        out["channel_best"] = out["channel_best"].astype(str).str.strip()

    return out


def _normalize_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_snake(c) for c in out.columns]
    cols = list(out.columns)
    if len(cols) >= 3:
        out = out.rename(
            columns={
                cols[0]: "source_table",
                cols[1]: "source_field",
                cols[2]: "target_field",
            }
        )
    return out


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_snake(c) for c in out.columns]
    if "unnamed_0" in out.columns:
        out = out.rename(columns={"unnamed_0": "month_end"})
    if "month_end" in out.columns:
        out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    for col in out.columns:
        if col == "month_end":
            continue
        out[col] = out[col].astype(str).str.replace("%", "", regex=False)
        out[col] = out[col].map(_parse_number) / 100.0
    return out


def run(excel_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _read_sheet_or_csv(excel_path, "DATA RAW", RAW_FALLBACK_CSV)
    summary_df = _read_sheet_or_csv(excel_path, "DATA SUMMARY", SUMMARY_FALLBACK_CSV)
    mapping_df = _read_sheet_or_csv(excel_path, "DATA MAPPING", MAPPING_FALLBACK_CSV)
    etf_df = _read_sheet_or_csv(excel_path, "ETF", ETF_FALLBACK_CSV)
    exec_df = _read_sheet_or_csv(excel_path, "Executive Summary", EXEC_FALLBACK_CSV)

    raw_norm = _normalize_raw(raw_df)
    summary_norm = _normalize_summary(summary_df)
    mapping_norm = _normalize_mapping(mapping_df)

    raw_norm.to_parquet(out_dir / "data_raw_normalized.parquet", index=False)
    summary_norm.to_parquet(out_dir / "data_summary_normalized.parquet", index=False)
    mapping_norm.to_parquet(out_dir / "data_mapping_normalized.parquet", index=False)
    etf_df.to_parquet(out_dir / "etf_raw.parquet", index=False)
    exec_df.to_parquet(out_dir / "executive_summary_raw.parquet", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Excel workbook into canonical normalized parquet files")
    parser.add_argument("--excel-path", default="data/input/source.xlsx")
    parser.add_argument("--out-dir", default="data/curated")
    args = parser.parse_args()
    run(Path(args.excel_path), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
