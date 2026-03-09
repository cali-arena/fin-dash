"""
QA gates for DuckDB analytics layer: rowcount parity (parquet vs DuckDB) and view time-budget checks.
Writes qa/duckdb_rowcount_parity.csv and qa/duckdb_view_latency.json. Exit 0 if all pass, 2 if any fail.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import duckdb

from legacy.legacy_pipelines.contracts.duckdb_policy_contract import (
    DuckDBPolicyError,
    load_and_validate_duckdb_policy,
)
from legacy.legacy_pipelines.duckdb.build_analytics_db import agg_source_to_table_name

DEFAULT_POLICY_PATH = "configs/duckdb_policy.yml"
QA_DIR_REL = "qa"
PARITY_CSV = "qa/duckdb_rowcount_parity.csv"
LATENCY_JSON = "qa/duckdb_view_latency.json"
EXIT_OK = 0
EXIT_FAIL = 2

DEFAULT_TIME_BUDGET_MS = {"firm": 200, "channel": 500, "ticker": 500}
VIEW_QUERIES = {
    "firm": "ORDER BY month_end DESC LIMIT 24",
    "channel": "ORDER BY month_end DESC LIMIT 2000",
    "ticker": "ORDER BY month_end DESC LIMIT 2000",
}


def _root_path(root: Path | None) -> Path:
    return Path(root) if root is not None else Path.cwd()


def _load_qa_config(policy_path: Path) -> dict[str, int]:
    """Load qa.time_budget_ms from policy YAML; merge with defaults."""
    try:
        import yaml
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_TIME_BUDGET_MS)
    qa = raw.get("qa") if isinstance(raw, dict) else None
    if not isinstance(qa, dict):
        return dict(DEFAULT_TIME_BUDGET_MS)
    tb = qa.get("time_budget_ms")
    if not isinstance(tb, dict):
        return dict(DEFAULT_TIME_BUDGET_MS)
    out = dict(DEFAULT_TIME_BUDGET_MS)
    for k, v in tb.items():
        if isinstance(v, (int, float)) and v >= 0:
            out[k] = int(v)
    return out


def _parquet_row_count(parquet_path: Path) -> int:
    """Fast row count via pyarrow or pandas."""
    try:
        import pyarrow.parquet as pq
        return pq.read_table(parquet_path).num_rows
    except ImportError:
        import pandas as pd
        return len(pd.read_parquet(parquet_path))


def run_rowcount_parity(root: Path, policy_path: Path, db_path: str, schema: str, agg_sources: dict[str, str]) -> list[dict]:
    """
    For each agg parquet, compare parquet row count to DuckDB table count. Return rows for CSV.
    """
    rows: list[dict] = []
    con = duckdb.connect(db_path, read_only=True)
    try:
        for source_name, rel_path in sorted(agg_sources.items()):
            full = root / rel_path
            table_name = agg_source_to_table_name(source_name)
            parquet_count = _parquet_row_count(full) if full.exists() else -1
            duckdb_count = -1
            try:
                r = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{table_name}"').fetchone()
                duckdb_count = int(r[0]) if r else -1
            except Exception:
                pass
            diff = (parquet_count - duckdb_count) if parquet_count >= 0 and duckdb_count >= 0 else None
            pass_ = diff is not None and diff == 0
            rows.append({
                "table": table_name,
                "parquet_path": rel_path,
                "parquet_count": parquet_count,
                "duckdb_count": duckdb_count,
                "diff": "" if diff is None else str(diff),
                "pass": pass_,
            })
    finally:
        con.close()
    return rows


def run_view_latency_checks(db_path: str, schema: str, time_budget_ms: dict[str, int]) -> tuple[list[dict], bool]:
    """
    Run view queries (firm, channel, ticker), measure elapsed ms, compare to budget. Return (results, all_passed).
    """
    results: list[dict] = []
    con = duckdb.connect(db_path, read_only=True)
    try:
        all_passed = True
        for view_key, suffix in VIEW_QUERIES.items():
            view_name = f"v_{view_key}_monthly"
            sql = f'SELECT * FROM "{schema}"."{view_name}" {suffix}'
            budget_ms = time_budget_ms.get(view_key, DEFAULT_TIME_BUDGET_MS.get(view_key, 500))
            t0 = time.perf_counter()
            try:
                con.execute(sql).fetchall()
            except Exception as e:
                results.append({
                    "view": view_name,
                    "query_ms": None,
                    "budget_ms": budget_ms,
                    "pass": False,
                    "error": str(e),
                })
                all_passed = False
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000
            pass_ = elapsed_ms < budget_ms
            if not pass_:
                all_passed = False
            results.append({
                "view": view_name,
                "query_ms": round(elapsed_ms, 2),
                "budget_ms": budget_ms,
                "pass": pass_,
                "error": None,
            })
    finally:
        con.close()
    return results, all_passed


def run(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    root: Path | None = None,
) -> int:
    """Run rowcount parity and view latency checks; write QA artifacts; return EXIT_OK or EXIT_FAIL."""
    root = _root_path(root)
    policy_path = Path(policy_path)
    policy = load_and_validate_duckdb_policy(policy_path)
    db_full = root / policy.db_path
    if not db_full.exists():
        raise FileNotFoundError(f"DuckDB not found: {db_full}. Run: python -m pipelines.duckdb.rebuild_analytics_layer")
    schema = policy.schema
    agg = policy.source_paths.get("agg") or {}
    time_budget_ms = _load_qa_config(policy_path)

    qa_dir = root / QA_DIR_REL
    qa_dir.mkdir(parents=True, exist_ok=True)

    parity_rows = run_rowcount_parity(root, policy_path, str(db_full.resolve()), schema, agg)
    parity_path = root / PARITY_CSV
    with open(parity_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["table", "parquet_path", "parquet_count", "duckdb_count", "diff", "pass"])
        w.writeheader()
        w.writerows(parity_rows)
    parity_ok = all(r["pass"] for r in parity_rows)

    latency_results, latency_ok = run_view_latency_checks(str(db_full.resolve()), schema, time_budget_ms)
    latency_path = root / LATENCY_JSON
    latency_path.write_text(
        json.dumps({"views": latency_results, "all_passed": latency_ok}, indent=2),
        encoding="utf-8",
    )

    if parity_ok and latency_ok:
        return EXIT_OK
    return EXIT_FAIL


def main() -> None:
    parser = argparse.ArgumentParser(description="QA gates: rowcount parity + view time budget. Exit 0 if all pass, 2 if any fail.")
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="Path to duckdb_policy.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root (default: cwd)")
    args = parser.parse_args()
    try:
        exit_code = run(policy_path=args.policy, root=args.root)
        raise SystemExit(exit_code)
    except DuckDBPolicyError as e:
        print(f"Policy error: {e}")
        raise SystemExit(EXIT_FAIL) from e
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(EXIT_FAIL) from e


if __name__ == "__main__":
    main()
