"""
Hard guardrails: no groupby/merge in UI pages; no DuckDB/parquet outside data_gateway.
Fail fast in dev-mode when violations are found.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# When "1", run pattern/import scans and raise on violations. Set APP_DEV_GUARDRAILS=0 to disable.
DEV_GUARDRAILS = os.environ.get("APP_DEV_GUARDRAILS", "1") == "1"

# Project root (app/guardrails.py -> app -> parent)
_APP_DIR = Path(__file__).resolve().parent
ROOT = _APP_DIR.parent

# Forbidden patterns in app/pages: no groupby/merge in UI; no direct DuckDB/parquet (all queries via gateway).
FORBIDDEN_PATTERNS_PAGES = (
    r"\.groupby\(",
    r"\.merge\(",
    r"duckdb\.connect\(",
    r"read_parquet\(",
    r"pd\.read_parquet\(",
)

# Paths excluded from forbidden-imports scan (gateway and infra may use DuckDB/parquet).
EXCLUDE_IMPORT_PATHS = (
    "app/data_gateway.py",
    "app/agg_store.py",
    "app/duckdb_store.py",
    "app/data_loader.py",
    "app/data.py",
    "app/cache/",
    "app/observability/",
)


def _relpath(path: Path) -> str:
    """Return path relative to ROOT, normalised with forward slashes."""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _skip_line(line: str) -> bool:
    """Skip empty and comment-only lines."""
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


@lru_cache(maxsize=4)
def scan_forbidden_patterns(
    base_dir: str = "app/pages",
    patterns: tuple[str, ...] = FORBIDDEN_PATTERNS_PAGES,
) -> list[dict[str, Any]]:
    """
    Scan Python files under base_dir (relative to ROOT) for forbidden patterns.
    Returns list of {file, line_no, line, pattern}.
    """
    violations: list[dict[str, Any]] = []
    root = ROOT
    dir_path = root / base_dir.replace("/", os.sep)
    if not dir_path.is_dir():
        return violations
    compiled = [re.compile(p) for p in patterns]
    for py_path in sorted(dir_path.rglob("*.py")):
        rel = _relpath(py_path)
        try:
            text = py_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if _skip_line(line):
                continue
            for pat in compiled:
                if pat.search(line):
                    violations.append({
                        "file": rel,
                        "line_no": i,
                        "line": line.strip()[:120],
                        "pattern": pat.pattern,
                    })
                    break
    return violations


@lru_cache(maxsize=2)
def scan_forbidden_imports(
    base_dir: str = "app",
    deny_imports: tuple[str, ...] = ("duckdb", "pyarrow.parquet", "pandas.read_parquet"),
) -> list[dict[str, Any]]:
    """
    Scan Python files under base_dir for forbidden imports.
    Excludes app/data_gateway.py and other EXCLUDE_IMPORT_PATHS.
    Returns list of {file, line_no, line, pattern}.
    """
    violations: list[dict[str, Any]] = []
    root = ROOT
    dir_path = root / base_dir.replace("/", os.sep)
    if not dir_path.is_dir():
        return violations
    # Build regex per deny item
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for imp in deny_imports:
        if imp == "duckdb":
            patterns.append((imp, re.compile(r"(?:import|from)\s+duckdb\b|\bduckdb\.(?:connect|query)\b")))
        elif "pyarrow" in imp:
            patterns.append((imp, re.compile(r"import\s+pyarrow\.parquet|pyarrow\.parquet\.|from\s+pyarrow")))
        elif "read_parquet" in imp:
            patterns.append((imp, re.compile(r"(?:pd|pandas)\.read_parquet\s*\(")))
        else:
            patterns.append((imp, re.compile(re.escape(imp))))
    for py_path in sorted(dir_path.rglob("*.py")):
        rel = _relpath(py_path)
        if not rel.replace("\\", "/").startswith("app/"):
            continue
        rel_norm = rel.replace("\\", "/")
        if any(rel_norm == p.rstrip("/") or rel_norm.startswith(p) for p in EXCLUDE_IMPORT_PATHS):
            continue
        try:
            text = py_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            for name, pat in patterns:
                if pat.search(line):
                    violations.append({
                        "file": rel,
                        "line_no": i,
                        "line": line.strip()[:120],
                        "pattern": name,
                    })
                    break
    return violations


def enforce_guardrails() -> None:
    """
    If DEV_GUARDRAILS is set and any violations exist, raise RuntimeError with a readable report.
    Otherwise no-op.
    """
    if not DEV_GUARDRAILS:
        return
    pattern_violations = scan_forbidden_patterns("app/pages", FORBIDDEN_PATTERNS_PAGES)
    import_violations = scan_forbidden_imports()
    all_violations: list[dict[str, Any]] = list(pattern_violations) + list(import_violations)
    if not all_violations:
        return
    lines = [
        "Guardrails failed: forbidden patterns or imports found.",
        "",
        "No .groupby(/.merge( in app/pages; no duckdb.connect(/read_parquet(/pd.read_parquet( outside gateway.",
        "",
    ]
    for v in all_violations:
        lines.append(f"  {v['file']}:{v['line_no']}  [{v.get('pattern', '')}]")
        lines.append(f"    {v['line']}")
        lines.append("")
    raise RuntimeError("\n".join(lines))
