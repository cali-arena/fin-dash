"""
UI contract loader and guardrail enforcement. Fail fast if pages violate contract.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = ROOT / "app" / "config" / "ui_contract.yml"

# Forbidden patterns: actual import or use, not docstrings/comments.
DEFAULT_DENY_PATTERNS = [
    r"(?:import|from)\s+duckdb\b|\bduckdb\.(?:connect|query)\b",
    r"import\s+pyarrow\.parquet|pyarrow\.parquet\.",
    r"(?:pd|pandas)\.read_parquet\s*\(",
]
# Gateway/cache/NLQ executor/contracts allowed to use duckdb/parquet; only pages + main are checked.
EXCLUDE_PREFIXES = (
    "app/data_gateway.py",
    "app/agg_store.py",
    "app/duckdb_store.py",
    "app/data_loader.py",
    "app/data.py",
    "app/drill_paths.py",
    "app/cache/",
    "app/contracts/",
    "app/observability/",
    "app/data/",
    "app/config/",
    "app/guardrails.py",
    "app/nlq/executor.py",
    "app/queries/",
)


def _load_ui_contract_impl(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return raw or {}
    except Exception:
        return {}


@lru_cache(maxsize=4)
def load_ui_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load UI contract YAML; cached (safe in Streamlit)."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    path_str = str(p.resolve())
    return _load_ui_contract_impl(path_str)


def _get_deny_patterns(contract: dict[str, Any]) -> list[str]:
    """Resolve deny_imports from contract or use defaults."""
    guardrails = contract.get("guardrails") or {}
    deny = guardrails.get("deny_imports")
    if not deny:
        return DEFAULT_DENY_PATTERNS
    out: list[str] = []
    for x in deny:
        s = str(x).strip()
        if s == "duckdb":
            out.append(r"(?:import|from)\s+duckdb\b|\bduckdb\.(?:connect|query)\b")
        elif "pyarrow" in s:
            out.append(r"pyarrow\.parquet|import\s+pyarrow")
        elif "read_parquet" in s:
            out.append(r"(?:pd|pandas)\.read_parquet\s*\(")
        else:
            out.append(re.escape(s))
    return out


def assert_contract_compliance(
    app_dir: Path | None = None,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> None:
    """
    Scan app/ for forbidden imports (from contract). Exclude data_gateway.py.
    Raises RuntimeError with file path + offending line if found.
    """
    app_dir = app_dir or (ROOT / "app")
    contract = load_ui_contract(str(contract_path))
    patterns = _get_deny_patterns(contract)
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    violations: list[tuple[Path, int, str]] = []

    for py_path in sorted(app_dir.rglob("*.py")):
        try:
            rel = py_path.relative_to(ROOT)
        except ValueError:
            continue
        rel_str = str(rel).replace("\\", "/")
        if any(rel_str == p or rel_str.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        try:
            text = py_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            for pat in compiled:
                if pat.search(line):
                    violations.append((py_path, i, line.strip()[:100]))
                    break

    if violations:
        path, line_no, snippet = violations[0]
        raise RuntimeError(
            f"Contract violation: forbidden import in {path}:{line_no} — {snippet!r}. "
            "Use app.data_gateway only."
        )


def ensure_contract_checked(session_key: str = "_contract_checked") -> None:
    """
    Run assert_contract_compliance once per Streamlit session (avoid rescan on every rerun).
    """
    try:
        import streamlit as st
        if session_key in st.session_state:
            return
        assert_contract_compliance()
        st.session_state[session_key] = True
    except ImportError:
        assert_contract_compliance()
