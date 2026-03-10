#!/usr/bin/env python3
"""
Guardrail: fail if Streamlit pages bypass the gateway (direct DuckDB/parquet access).
Scans app/pages/**/*.py and app/*.py; exits 2 on violations.
Allowlist: app/data_gateway.py, app/cache/*, app/observability/*, app/pages/Debug_Data.py, app/pages/debug_adhoc.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX_MSG = "Use app.data_gateway.run_query / run_chart only."

FORBIDDEN_PATTERNS = [
    (r"duckdb\.connect", "duckdb.connect"),
    (r"import\s+duckdb", "import duckdb"),
    (r"pd\.read_parquet", "pd.read_parquet"),
    (r"pyarrow\.parquet", "pyarrow.parquet"),
    (r"read_parquet\s*\(", "read_parquet("),
    (r"duckdb\.query", "duckdb.query"),
]

# Gateway infra + optional debug page. Extend for approved loaders (e.g. data_loader) as needed.
# app/guardrails.py defines deny patterns (strings); it does not use DuckDB/parquet.
ALLOWLIST = [
    "app/data_gateway.py",
    "app/agg_store.py",
    "app/duckdb_store.py",
    "app/data_loader.py",
    "app/data.py",
    "app/drill_paths.py",
    "app/guardrails.py",
    "app/pages/Debug_Data.py",
    "app/pages/debug_adhoc.py",
]


def _path_matches_allowlist(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    if norm in ALLOWLIST:
        return True
    if norm.startswith("app/cache/"):
        return True
    if norm.startswith("app/observability/"):
        return True
    if norm.startswith("app/data/"):
        return True
    return False


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_no, pattern_name, line_snippet) violations."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    violations: list[tuple[int, str, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pat, name in FORBIDDEN_PATTERNS:
            if re.search(pat, line):
                snippet = stripped[:80]
                if len(stripped) > 80:
                    snippet += "..."
                violations.append((i, name, snippet))
                break
    return violations


def main() -> int:
    app_dir = ROOT / "app"
    if not app_dir.exists():
        print("No app/ dir found.")
        return 0

    files_to_scan: list[Path] = []
    if (app_dir / "pages").exists():
        files_to_scan.extend((app_dir / "pages").rglob("*.py"))
    for py in app_dir.glob("*.py"):
        files_to_scan.append(py)

    all_violations: list[tuple[Path, int, str, str]] = []
    for path in sorted(set(files_to_scan)):
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            continue
        rel_str = str(rel).replace("\\", "/")
        if _path_matches_allowlist(rel_str):
            continue
        for line_no, pattern_name, snippet in scan_file(path):
            all_violations.append((path, line_no, pattern_name, snippet))

    if not all_violations:
        print("check_no_direct_data_access: clean.")
        return 0

    print("check_no_direct_data_access: VIOLATIONS (direct DuckDB/parquet access)\n")
    for path, line_no, pattern_name, snippet in all_violations:
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"  {rel}:{line_no}")
        print(f"    pattern: {pattern_name}")
        print(f"    snippet: {snippet}")
        print(f"    fix: {FIX_MSG}\n")

    return 2


if __name__ == "__main__":
    sys.exit(main())
