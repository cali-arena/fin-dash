"""
Single entrypoint: validate policy → build_analytics_db → create_views → smoke queries.
Deterministic, CI-friendly: exit 0 if OK, exit 2 on any failure.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from legacy.legacy_pipelines.contracts.duckdb_policy_contract import (
    DuckDBPolicyError,
    load_and_validate_duckdb_policy,
)

logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "configs/duckdb_policy.yml"
EXIT_OK = 0
EXIT_FAIL = 2


def _root_path(root: Path | None) -> Path:
    return Path(root) if root is not None else Path.cwd()


def _run_smoke_queries(db_path: str, schema: str) -> None:
    """
    Run smoke queries against v_* views. Raise on first failure.
    - Latest 5 months firm
    - Top 10 channels on latest month
    - Top 10 tickers on latest month
    """
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        qs = [
            (
                "latest_5_firm",
                f'SELECT month_end, end_aum FROM "{schema}"."v_firm_monthly" ORDER BY month_end DESC LIMIT 5',
            ),
            (
                "top10_channel_latest_month",
                f'''SELECT channel_l1, SUM(end_aum) AS total_aum FROM "{schema}"."v_channel_monthly"
WHERE month_end = (SELECT MAX(month_end) FROM "{schema}"."v_channel_monthly")
GROUP BY 1 ORDER BY 2 DESC LIMIT 10''',
            ),
            (
                "top10_ticker_latest_month",
                f'''SELECT product_ticker, SUM(end_aum) AS total_aum FROM "{schema}"."v_ticker_monthly"
WHERE month_end = (SELECT MAX(month_end) FROM "{schema}"."v_ticker_monthly")
GROUP BY 1 ORDER BY 2 DESC LIMIT 10''',
            ),
        ]
        for name, sql in qs:
            t0 = time.perf_counter()
            try:
                rows = con.execute(sql).fetchall()
            except Exception as e:
                raise RuntimeError(f"Smoke query {name!r} failed: {e}\nSQL: {sql}") from e
            elapsed = time.perf_counter() - t0
            logger.info("Smoke %s | rows=%s | %.3fs", name, len(rows), elapsed)
    finally:
        con.close()


def run(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    root: Path | None = None,
    force: bool = False,
) -> int:
    """
    Validate policy → build DB → create views → smoke. Returns EXIT_OK (0) or EXIT_FAIL (2).
    """
    root = _root_path(root)
    policy_path = Path(policy_path)

    # 1) Validate duckdb_policy
    logger.info("Stage 1: validate policy %s", policy_path)
    try:
        policy = load_and_validate_duckdb_policy(policy_path)
    except DuckDBPolicyError as e:
        logger.error("Policy validation failed: %s", e)
        return EXIT_FAIL

    # 2) Build analytics DB (rebuild unless cache hit or --force)
    logger.info("Stage 2: build_analytics_db (force=%s)", force)
    try:
        from legacy.legacy_pipelines.duckdb.build_analytics_db import run as build_run
        build_run(policy_path=policy_path, root=root, force=force)
    except Exception as e:
        logger.error("build_analytics_db failed: %s", e)
        return EXIT_FAIL

    # 3) Create views
    logger.info("Stage 3: create_views")
    try:
        from legacy.legacy_pipelines.duckdb.create_views import run as create_views_run
        create_views_run(policy_path=policy_path, root=root)
    except Exception as e:
        logger.error("create_views failed: %s", e)
        return EXIT_FAIL

    # 4) Smoke queries (views)
    db_full = root / policy.db_path
    if not db_full.exists():
        logger.error("DB not found after create_views: %s", db_full)
        return EXIT_FAIL
    logger.info("Stage 4: smoke queries (views)")
    try:
        _run_smoke_queries(str(db_full.resolve()), policy.schema)
    except Exception as e:
        logger.error("Smoke queries failed: %s", e)
        return EXIT_FAIL

    logger.info("Analytics layer ready.")
    return EXIT_OK


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Rebuild analytics layer: validate policy, build DB, create views, smoke. Exit 0 if OK, 2 on failure."
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="Path to duckdb_policy.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root (default: cwd)")
    parser.add_argument("--force", action="store_true", help="Force rebuild (skip manifest cache)")
    args = parser.parse_args()
    exit_code = run(policy_path=args.policy, root=args.root, force=args.force)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
