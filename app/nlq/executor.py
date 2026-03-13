"""
NLQ execution engine. Uses ONLY validated QuerySpec; no user input is ever concatenated into SQL.
WHERE clauses are parameterized. Hard safety gates: row caps, export mode, DuckDB timeout attempt, wall-clock guard.
DuckDB (preferred) and pandas fallback; identical result schema.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.nlq.governance import validate_queryspec
from models.query_spec import QuerySpec
from app.ui.formatters import fmt_currency, fmt_number, fmt_percent

try:
    import duckdb
    _DUCKDB_CONNECTION_TYPE = type(duckdb.connect(":memory:"))
except Exception:
    _DUCKDB_CONNECTION_TYPE = type(None)

__all__ = ["QueryResult", "execute_queryspec", "queryspec_hash"]

MAX_ROWS = 5000
DEFAULT_MAX_ROWS = 5000
EXPORT_MAX_ROWS = 50000
EXECUTOR_TIMEOUT_MS = 15000
DEFAULT_LIMIT_TOP = 20
FACT_VIEW_NAME = "fact"


def queryspec_hash(qs: QuerySpec) -> str:
    """Canonical JSON sha1 of QuerySpec for observability/cache keys."""
    payload = json.dumps(qs.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _metrics_by_id(metric_reg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build metric_id -> entry from registry."""
    metrics = metric_reg.get("metrics") or []
    if not isinstance(metrics, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for m in metrics:
        if isinstance(m, dict) and m.get("metric_id") is not None:
            by_id[str(m["metric_id"]).strip().lower()] = m
    return by_id


# ---- Safe expression: only +, -, *, / and identifiers; no eval of raw strings ----

_ALLOWED_ID_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _parse_expr_to_ast(expr_str: str) -> dict[str, Any]:
    """
    Parse a safe expression string into AST. Only +, -, *, / and identifiers.
    Returns e.g. {"op": "div", "lhs": {"op": "sub", "lhs": {"op": "col", "col": "end_aum"}, "rhs": {"op": "col", "col": "begin_aum"}}, "rhs": {"op": "col", "col": "begin_aum"}}.
    """
    s = expr_str.strip()
    if not s:
        raise ValueError("Empty expression")

    tokens: list[tuple[str, str]] = []  # ("id", "end_aum") or ("op", "+") or ("paren", "(")
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] in "()+*-/":
            if s[i] in "()":
                tokens.append(("paren", s[i]))
            else:
                tokens.append(("op", s[i]))
            i += 1
            continue
        m = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", s[i:])
        if m:
            ident = m.group(0)
            if not _ALLOWED_ID_RE.match(ident):
                raise ValueError(f"Invalid identifier: {ident}")
            tokens.append(("id", ident))
            i += m.end()
            continue
        raise ValueError(f"Unexpected character at position {i}: {s[i]!r}")

    pos = [0]

    def peek() -> tuple[str, str] | None:
        if pos[0] >= len(tokens):
            return None
        return tokens[pos[0]]

    def consume() -> tuple[str, str] | None:
        if pos[0] >= len(tokens):
            return None
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def parse_expr() -> dict[str, Any]:
        left = parse_term()
        while True:
            t = peek()
            if t and t[0] == "op" and t[1] in ("+", "-"):
                consume()
                right = parse_term()
                left = {"op": "add" if t[1] == "+" else "sub", "lhs": left, "rhs": right}
            else:
                break
        return left

    def parse_term() -> dict[str, Any]:
        left = parse_factor()
        while True:
            t = peek()
            if t and t[0] == "op" and t[1] in ("*", "/"):
                consume()
                right = parse_factor()
                left = {"op": "mul" if t[1] == "*" else "div", "lhs": left, "rhs": right}
            else:
                break
        return left

    def parse_factor() -> dict[str, Any]:
        t = consume()
        if not t:
            raise ValueError("Unexpected end of expression")
        if t[0] == "paren" and t[1] == "(":
            e = parse_expr()
            t2 = consume()
            if t2 != ("paren", ")"):
                raise ValueError("Missing closing parenthesis")
            return e
        if t[0] == "id":
            return {"op": "col", "col": t[1]}
        raise ValueError(f"Expected identifier or '(': got {t}")

    ast = parse_expr()
    if pos[0] != len(tokens):
        raise ValueError("Extra tokens after expression")
    return ast


def _ast_columns(ast: dict[str, Any]) -> set[str]:
    """Collect column names used in AST."""
    if ast.get("op") == "col":
        return {ast["col"]}
    out: set[str] = set()
    if "lhs" in ast:
        out |= _ast_columns(ast["lhs"])
    if "rhs" in ast:
        out |= _ast_columns(ast["rhs"])
    return out


def _ast_to_sql(ast: dict[str, Any], quote_id: str = '"') -> str:
    """Render AST as SQL expression (identifiers quoted). No user input in AST."""
    if ast.get("op") == "col":
        return f"{quote_id}{ast['col']}{quote_id}"
    op = ast.get("op")
    if op == "add":
        return f"({_ast_to_sql(ast['lhs'], quote_id)} + {_ast_to_sql(ast['rhs'], quote_id)})"
    if op == "sub":
        return f"({_ast_to_sql(ast['lhs'], quote_id)} - {_ast_to_sql(ast['rhs'], quote_id)})"
    if op == "mul":
        return f"({_ast_to_sql(ast['lhs'], quote_id)} * {_ast_to_sql(ast['rhs'], quote_id)})"
    if op == "div":
        return f"({_ast_to_sql(ast['lhs'], quote_id)} / NULLIF({_ast_to_sql(ast['rhs'], quote_id)}, 0))"
    raise ValueError(f"Unknown AST op: {op}")


def _ast_eval_pandas(ast: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """Compute AST over DataFrame columns. Safe: only column references and +,-,*,/."""
    if ast.get("op") == "col":
        c = ast["col"]
        if c not in df.columns:
            return pd.Series(dtype=float)
        return df[c].astype(float)
    op = ast.get("op")
    lhs = _ast_eval_pandas(ast["lhs"], df)
    rhs = _ast_eval_pandas(ast["rhs"], df)
    if op == "add":
        return lhs + rhs
    if op == "sub":
        return lhs - rhs
    if op == "mul":
        return lhs * rhs
    if op == "div":
        return lhs / rhs.replace(0, float("nan"))
    raise ValueError(f"Unknown AST op: {op}")


def _formula_to_metric_spec(formula: str) -> tuple[str, str | None, dict[str, Any] | None]:
    """
    Parse formula "column:<col>" or "expr:<...>". Returns (kind, col_or_none, ast_or_none).
    For expr: parses to AST (no eval of raw string).
    """
    if not formula or not isinstance(formula, str):
        return ("column", "end_aum", None)
    s = formula.strip()
    if s.startswith("column:"):
        col = s[7:].strip()
        if _ALLOWED_ID_RE.match(col):
            return ("column", col, None)
        return ("column", "end_aum", None)
    if s.startswith("expr:"):
        expr = s[5:].strip()
        try:
            ast = _parse_expr_to_ast(expr)
            return ("expr", None, ast)
        except ValueError:
            return ("column", "end_aum", None)
    return ("column", "end_aum", None)


@dataclass(frozen=True)
class QueryResult:
    data: pd.DataFrame
    numbers: dict  # headline values (optional)
    chart_spec: dict  # auto chart config derived from qs
    explain_context: dict  # facts only (no narrative)
    meta: dict  # timings, row caps, applied limits, warnings


def _normalize_allowlist(allowlist: dict[str, Any] | None) -> dict[str, Any]:
    """
    Normalize allowlist to: columns (frozenset), pii_columns (frozenset), max_rows (int).
    Accepts columns/pii_columns as set or list; legacy allowed/disallowed still supported.
    """
    if not allowlist or not isinstance(allowlist, dict):
        return {}
    out: dict[str, Any] = {}
    for key, legacy in [("columns", "allowed"), ("pii_columns", "disallowed")]:
        val = allowlist.get(key) or allowlist.get(legacy)
        if val is not None:
            if isinstance(val, (list, set)):
                out[key] = frozenset(str(c).strip() for c in val)
            else:
                out[key] = frozenset()
    if "max_rows" in allowlist and allowlist["max_rows"] is not None:
        try:
            out["max_rows"] = int(allowlist["max_rows"])
        except (TypeError, ValueError):
            out["max_rows"] = MAX_ROWS
    return out


def _output_columns_for_spec(qs: QuerySpec) -> set[str]:
    """Columns that will appear in the result: dimensions + metric."""
    return set(qs.dimensions) | {"metric"}


def _validate_allowlist_against_plan(qs: QuerySpec, allowlist_norm: dict[str, Any]) -> None:
    """Raise ValueError if plan selects blocked or disallowed columns."""
    out_cols = _output_columns_for_spec(qs)
    pii = allowlist_norm.get("pii_columns") or frozenset()
    if pii and (out_cols & set(pii)):
        blocked = out_cols & set(pii)
        raise ValueError(f"Plan selects blocked (PII) columns: {sorted(blocked)}")
    cols = allowlist_norm.get("columns")
    if cols and out_cols and (out_cols - set(cols)):
        disallowed = out_cols - set(cols)
        raise ValueError(f"Plan selects columns not in allowlist: {sorted(disallowed)}")


def _apply_allowlist(df: pd.DataFrame, allowlist: dict[str, Any]) -> pd.DataFrame:
    """
    Enforce allowlist: columns (only these), pii_columns (always blocked), max_rows applied earlier.
    """
    norm = _normalize_allowlist(allowlist)
    if not norm:
        return df
    pii = norm.get("pii_columns") or frozenset()
    if pii:
        cols = [c for c in df.columns if str(c).strip() not in pii]
        df = df[[c for c in cols if c in df.columns]] if cols else df
    allowed = norm.get("columns")
    if allowed and len(allowed) > 0:
        cols = [c for c in df.columns if str(c).strip() in allowed]
        df = df[[c for c in cols if c in df.columns]] if cols else df
    return df


def _chart_spec_from_qs(qs: QuerySpec) -> dict[str, Any]:
    """
    Chart config derived from qs.chart with resolved columns.
    Line: x=month_end, y=metric, series=optional dim. Bar: x=dim, y=metric. Table: type table.
    """
    c = qs.chart
    t = c.type or "table"
    if t == "line":
        return {
            "type": "line",
            "x": "month_end",
            "y": "metric",
            "series": (c.series and c.series.strip()) or (qs.dimensions[0] if len(qs.dimensions) == 1 else None),
        }
    if t == "bar":
        x_dim = (c.x and c.x.strip()) or (qs.dimensions[0] if qs.dimensions else None)
        return {"type": "bar", "x": x_dim, "y": "metric"}
    # Auto-visual fallback for dimensional queries when parser leaves chart as table.
    if qs.dimensions:
        non_time_dims = [d for d in qs.dimensions if d != "month_end"]
        if "month_end" in qs.dimensions:
            return {
                "type": "line",
                "x": "month_end",
                "y": "metric",
                "series": (non_time_dims[0] if non_time_dims else None),
            }
        return {"type": "bar", "x": qs.dimensions[0], "y": "metric"}
    return {"type": "table"}


def _format_metric_value(value: float, fmt: str | None) -> str:
    """Format scalar for display; facts only, no narrative. Uses app.ui.formatters."""
    if fmt == "percent":
        return fmt_percent(value, decimals=2, signed=False)
    if fmt == "currency":
        return fmt_currency(value, unit="auto", decimals=2)
    return fmt_number(value, decimals=2)


def _build_numbers(
    qs: QuerySpec,
    data: pd.DataFrame,
    metric_reg: dict[str, Any],
) -> dict[str, Any]:
    """
    numbers: if no dims -> metric_id, value, formatted; if trend (month_end series) add latest, change when computable.
    Facts only.
    """
    by_id = _metrics_by_id(metric_reg)
    mid = qs.metric_id.strip().lower()
    metric_entry = by_id.get(mid) or {}
    fmt = metric_entry.get("format") or "number"
    out: dict[str, Any] = {"metric_id": qs.metric_id, "value": None, "formatted": "—"}
    if data.empty or "metric" not in data.columns:
        return out
    if not qs.dimensions:
        scalar = data["metric"].iloc[0]
        out["value"] = float(scalar) if pd.notna(scalar) else None
        out["formatted"] = _format_metric_value(float(scalar) if pd.notna(scalar) else float("nan"), fmt)
    if "month_end" in data.columns and len(data) >= 2:
        try:
            vals = data["metric"].astype(float)
            out["latest"] = float(vals.iloc[-1]) if pd.notna(vals.iloc[-1]) else None
            first = float(vals.iloc[0]) if pd.notna(vals.iloc[0]) else None
            last = out.get("latest")
            if first is not None and last is not None:
                out["change"] = last - first
        except (TypeError, ValueError):
            pass
    return out


def _build_explain_context(
    qs: QuerySpec,
    data: pd.DataFrame,
    metric_reg: dict[str, Any],
) -> dict[str, Any]:
    """
    explain_context: facts only. metric_id, metric_label, agg_used, dims_used, time_range_resolved,
    filters_applied (canonical), row_count, top_entities (first 3 dim values). No prose.
    """
    by_id = _metrics_by_id(metric_reg)
    mid = qs.metric_id.strip().lower()
    metric_entry = by_id.get(mid) or {}
    formula = (metric_entry.get("formula") or "column:end_aum").strip()
    default_agg = (metric_entry.get("default_agg") or "last").strip().lower()
    tr = qs.time_range
    top_entities: list[Any] = []
    if qs.dimensions and qs.dimensions[0] in data.columns and len(data) > 0:
        top_entities = data[qs.dimensions[0]].head(3).astype(str).tolist()
    return {
        "metric_id": qs.metric_id,
        "metric_label": metric_entry.get("label") or qs.metric_id,
        "agg_used": default_agg,
        "dims_used": list(qs.dimensions),
        "time_range_resolved": {
            "start": tr.start.isoformat() if tr.start else None,
            "end": tr.end.isoformat() if tr.end else None,
        },
        "filters_applied": {k: list(v) for k, v in qs.filters.items()},
        "row_count": len(data),
        "top_entities": top_entities,
    }


def _ensure_fact_registered(conn: Any, base_df: pd.DataFrame) -> None:
    """Register 'fact' view once per connection. Uses conn attribute to avoid re-register."""
    if getattr(conn, "_nlq_fact_registered", False):
        return
    conn.register(FACT_VIEW_NAME, base_df)
    conn._nlq_fact_registered = True


def _build_where_and_params(
    qs: QuerySpec,
) -> tuple[list[str], list[Any]]:
    """Build WHERE clause fragments and param list from QuerySpec only. Parameterized."""
    parts: list[str] = []
    params: list[Any] = []
    tr = qs.time_range
    if tr.start is not None:
        parts.append('"month_end" >= ?')
        params.append(tr.start)
    if tr.end is not None:
        parts.append('"month_end" <= ?')
        params.append(tr.end)
    for dim, values in qs.filters.items():
        if not values:
            continue
        placeholders = ", ".join("?" for _ in values)
        parts.append(f'"{dim}" IN ({placeholders})')
        params.extend(values)
    return parts, params


def _build_duckdb_sql(
    qs: QuerySpec,
    metric_reg: dict[str, Any],
    effective_limit: int,
) -> tuple[str, list[Any], str]:
    """
    Build parameterized SQL from QuerySpec + metric_reg only. Returns (sql, params, sql_preview).
    effective_limit is already clamped to allowlist max_rows or MAX_ROWS.
    """
    by_id = _metrics_by_id(metric_reg)
    mid = qs.metric_id.strip().lower()
    metric_entry = by_id.get(mid) or {}
    formula = (metric_entry.get("formula") or "column:end_aum").strip()
    default_agg = (metric_entry.get("default_agg") or "last").strip().lower()
    kind, col, ast = _formula_to_metric_spec(formula)
    groupby_cols = [d for d in qs.dimensions if d]
    select_dims = [f'"{d}"' for d in groupby_cols]
    where_parts, params = _build_where_and_params(qs)
    where_sql = " AND ".join(where_parts) if where_parts else "1=1"
    order_col = (qs.sort.by or "metric").strip()
    order_dir = "DESC" if (qs.sort.order or "desc").strip().lower() == "desc" else "ASC"
    order_sql = f'"{order_col}"' if order_col != "metric" else "metric"
    limit_place = "?"
    params_limit = effective_limit

    if kind == "column" and col:
        metric_sel = f'SUM("{col}") AS metric' if default_agg == "sum" else f'ARG_MAX("{col}", "month_end") AS metric'
        select_list = ", ".join(select_dims + [metric_sel])
        group_sql = " GROUP BY " + ", ".join(select_dims) if groupby_cols else ""
        sql = f'SELECT {select_list} FROM {FACT_VIEW_NAME} WHERE {where_sql}{group_sql} ORDER BY {order_sql} {order_dir} LIMIT {limit_place}'
        params.append(params_limit)
        preview = f'SELECT {select_list} FROM {FACT_VIEW_NAME} WHERE {where_sql}{group_sql} ORDER BY {order_sql} {order_dir} LIMIT ?'
        return sql, params, preview
    if kind == "expr" and ast:
        cols = list(_ast_columns(ast))
        agg_parts = [f'ARG_MAX("{c}", "month_end") AS "{c}"' if default_agg != "sum" else f'SUM("{c}") AS "{c}"' for c in cols]
        select_inner = ", ".join(select_dims + agg_parts)
        group_sql = " GROUP BY " + ", ".join(select_dims) if groupby_cols else ""
        expr_sql = _ast_to_sql(ast)
        dims_sel = (", ".join(select_dims) + ", ") if select_dims else ""
        sql = f'WITH aggs AS (SELECT {select_inner} FROM {FACT_VIEW_NAME} WHERE {where_sql}{group_sql}) SELECT {dims_sel}{expr_sql} AS metric FROM aggs ORDER BY {order_sql} {order_dir} LIMIT {limit_place}'
        params.append(params_limit)
        preview = f'WITH aggs AS (SELECT ... FROM {FACT_VIEW_NAME} WHERE ...{group_sql}) SELECT ... AS metric FROM aggs ORDER BY {order_sql} {order_dir} LIMIT ?'
        return sql, params, preview
    metric_sel = 'SUM("end_aum") AS metric'
    select_list = ", ".join(select_dims + [metric_sel])
    group_sql = " GROUP BY " + ", ".join(select_dims) if groupby_cols else ""
    sql = f'SELECT {select_list} FROM {FACT_VIEW_NAME} WHERE {where_sql}{group_sql} ORDER BY {order_sql} {order_dir} LIMIT {limit_place}'
    params.append(params_limit)
    preview = sql.replace(limit_place, "?")
    return sql, params, preview


def _try_duckdb_timeout(engine: Any, timeout_ms: int) -> None:
    """If supported, set DuckDB statement timeout. Fail gracefully (no crash)."""
    timeout_sec = max(1, timeout_ms // 1000)
    try:
        engine.execute(f"SET statement_timeout = '{timeout_sec}s'")
    except Exception:
        try:
            engine.execute("SET timeout = ?", [f"{timeout_sec}s"])
        except Exception:
            pass


def _execute_against_duckdb(
    qs: QuerySpec,
    engine: Any,
    metric_reg: dict[str, Any],
    effective_limit: int,
    meta: dict[str, Any],
    base_df: pd.DataFrame | None,
    timeout_ms: int = EXECUTOR_TIMEOUT_MS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> pd.DataFrame:
    """Execute via DuckDB with parameter binding. Wall-clock guard; optional DB timeout."""
    if base_df is not None:
        _ensure_fact_registered(engine, base_df)
    _try_duckdb_timeout(engine, timeout_ms)
    sql, params, sql_preview = _build_duckdb_sql(qs, metric_reg, effective_limit)
    meta["sql_preview"] = sql_preview
    t0 = time.perf_counter()
    try:
        res = engine.execute(sql, params).df()
    except Exception:
        meta["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        return _empty_result_with_schema(qs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    meta["elapsed_ms"] = int(elapsed_ms)
    if elapsed_ms > timeout_ms:
        meta["status"] = "timeout"
        meta["message"] = "Query timed out. Try narrowing filters or time range."
        return _empty_result_with_schema(qs)
    if len(res) > max_rows:
        res = res.head(max_rows)
        meta["rows_capped"] = True
        meta.setdefault("warnings", []).append(f"Result capped to {max_rows} rows")
    else:
        meta["rows_capped"] = False
    meta["rows_returned"] = len(res)
    return res


def _execute_against_dataframe(
    qs: QuerySpec,
    df: pd.DataFrame,
    metric_reg: dict[str, Any],
    effective_limit: int,
    meta: dict[str, Any],
    timeout_ms: int = EXECUTOR_TIMEOUT_MS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> pd.DataFrame:
    """Pandas path: filters, groupby, aggregate (column or safe AST), sort, limit. Wall-clock guard."""
    def _pick_latest_index(frame: pd.DataFrame) -> Any | None:
        if frame is None or frame.empty:
            return None
        if "month_end" in frame.columns:
            me = pd.to_datetime(frame["month_end"], errors="coerce")
            valid = me.dropna()
            if not valid.empty:
                return valid.idxmax()
        return frame.index[0] if len(frame.index) else None

    def _pick_group_latest_indexes(frame: pd.DataFrame, dims: list[str]) -> pd.Index:
        if frame is None or frame.empty:
            return pd.Index([])
        grouped = frame.groupby(dims, dropna=False)
        if "month_end" not in frame.columns:
            return pd.Index(grouped.apply(lambda g: g.index[0]).tolist())
        me = pd.to_datetime(frame["month_end"], errors="coerce")
        work = frame.assign(_month_end_safe=me)
        grouped = work.groupby(dims, dropna=False)
        idx_values = grouped.apply(
            lambda g: g["_month_end_safe"].dropna().idxmax() if g["_month_end_safe"].notna().any() else g.index[0]
        )
        return pd.Index(idx_values.dropna().tolist())

    if df.empty:
        meta["elapsed_ms"] = 0
        meta["rows_returned"] = 0
        meta["rows_capped"] = False
        return _empty_result_with_schema(qs)
    out = df.copy()
    if "month_end" in out.columns:
        out["month_end"] = pd.to_datetime(out["month_end"])
    tr = qs.time_range
    if tr.start is not None and "month_end" in out.columns:
        out = out[out["month_end"].dt.date >= tr.start]
    if tr.end is not None and "month_end" in out.columns:
        out = out[out["month_end"].dt.date <= tr.end]
    for dim, values in qs.filters.items():
        if dim not in out.columns or not values:
            continue
        vals_lower = [v.strip().lower() for v in values]
        out = out[out[dim].astype(str).str.strip().str.lower().isin(vals_lower)]
    if out.empty:
        meta["elapsed_ms"] = 0
        meta["rows_returned"] = 0
        meta["rows_capped"] = False
        return _empty_result_with_schema(qs)
    by_id = _metrics_by_id(metric_reg)
    mid = qs.metric_id.strip().lower()
    metric_entry = by_id.get(mid) or {}
    formula = (metric_entry.get("formula") or "column:end_aum").strip()
    default_agg = (metric_entry.get("default_agg") or "last").strip().lower()
    kind, col, ast = _formula_to_metric_spec(formula)
    groupby_cols = [d for d in qs.dimensions if d and d in out.columns]
    t0 = time.perf_counter()

    if not groupby_cols:
        if kind == "column" and col and col in out.columns:
            if default_agg == "sum":
                val = out[col].sum()
            else:
                idx = _pick_latest_index(out)
                val = out.loc[idx, col] if len(out) else float("nan")
            agg_df = pd.DataFrame({"metric": [val]})
        elif kind == "expr" and ast:
            needed = list(_ast_columns(ast))
            if all(c in out.columns for c in needed):
                if default_agg == "sum":
                    row = out[needed].sum()
                else:
                    idx = _pick_latest_index(out)
                    row = out.loc[idx, needed]
                agg_df = row.to_frame().T
                agg_df["metric"] = _ast_eval_pandas(ast, agg_df).values
            else:
                agg_df = pd.DataFrame({"metric": [float("nan")]})
        else:
            agg_df = pd.DataFrame({"metric": [out["end_aum"].sum()] if "end_aum" in out.columns else [float("nan")]})
    else:
        if kind == "column" and col and col in out.columns:
            if default_agg == "sum":
                agg_df = out.groupby(groupby_cols, dropna=False)[col].sum().reset_index()
                agg_df = agg_df.rename(columns={col: "metric"})
            else:
                idx = _pick_group_latest_indexes(out, groupby_cols)
                agg_df = out.loc[idx, groupby_cols + [col]].copy()
                agg_df = agg_df.rename(columns={col: "metric"})
        elif kind == "expr" and ast:
            needed = list(_ast_columns(ast))
            if default_agg == "last" and "month_end" in out.columns and all(c in out.columns for c in needed):
                idx = _pick_group_latest_indexes(out, groupby_cols)
                sub = out.loc[idx, groupby_cols + needed].copy()
            else:
                agg_map = {c: "sum" if default_agg == "sum" else "last" for c in needed if c in out.columns}
                sub = out.groupby(groupby_cols, dropna=False).agg(agg_map).reset_index()
            if all(c in sub.columns for c in needed):
                sub["metric"] = _ast_eval_pandas(ast, sub)
                agg_df = sub[groupby_cols + ["metric"]].copy()
            else:
                agg_df = out.groupby(groupby_cols, dropna=False).agg({"end_aum": "sum"} if "end_aum" in out.columns else {}).reset_index()
                agg_df["metric"] = float("nan") if "end_aum" not in agg_df.columns else agg_df["end_aum"]
        else:
            agg_df = out.groupby(groupby_cols, dropna=False)["end_aum"].sum().reset_index() if "end_aum" in out.columns else out.groupby(groupby_cols, dropna=False).first().reset_index()
            agg_df = agg_df.rename(columns={"end_aum": "metric"}) if "end_aum" in agg_df.columns else agg_df.assign(metric=float("nan"))

    order_col = (qs.sort.by or "metric").strip()
    order_asc = (qs.sort.order or "desc").strip().lower() == "asc"
    if order_col in agg_df.columns:
        agg_df = agg_df.sort_values(order_col, ascending=order_asc, kind="mergesort")
    agg_df = agg_df.head(effective_limit)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    meta["elapsed_ms"] = int(elapsed_ms)
    if elapsed_ms > timeout_ms:
        meta["status"] = "timeout"
        meta["message"] = "Query timed out. Try narrowing filters or time range."
        return _empty_result_with_schema(qs)
    if len(agg_df) > max_rows:
        agg_df = agg_df.head(max_rows)
        meta["rows_capped"] = True
        meta.setdefault("warnings", []).append(f"Result capped to {max_rows} rows")
    else:
        meta["rows_capped"] = False
    meta["rows_returned"] = len(agg_df)
    return agg_df


def _empty_result_with_schema(qs: QuerySpec) -> pd.DataFrame:
    """Return DataFrame with same columns as non-empty result: dimensions + metric."""
    cols = [d for d in qs.dimensions if d]
    return pd.DataFrame(columns=cols + ["metric"])


def _is_duckdb_connection(engine: Any) -> bool:
    """True if engine is a duckdb.DuckDBPyConnection (preferred path)."""
    if _DUCKDB_CONNECTION_TYPE is type(None):
        return False
    return isinstance(engine, _DUCKDB_CONNECTION_TYPE)


def execute_queryspec(
    qs: QuerySpec,
    engine: Any,
    metric_reg: dict[str, Any],
    dim_reg: dict[str, Any],
    allowlist: dict[str, Any],
    base_df: pd.DataFrame | None = None,
    *,
    export_mode: bool = False,
) -> QueryResult:
    """
    Execute a validated QuerySpec. Last gate: validate_queryspec runs again before execution.
    engine: DuckDB connection (preferred) or pandas DataFrame. base_df: when engine is DuckDB,
    register as 'fact' once per connection.
    allowlist: columns (allowed set), pii_columns (always blocked), max_rows (default 5000).
    export_mode: if True, max_rows = EXPORT_MAX_ROWS (50000); else DEFAULT_MAX_ROWS (5000).
    Row cap enforced; DuckDB timeout attempted (graceful skip); wall-clock guard returns safe empty result.
    """
    validate_queryspec(qs, metric_reg, dim_reg)
    allow_norm = _normalize_allowlist(allowlist)
    _validate_allowlist_against_plan(qs, allow_norm)

    max_rows = EXPORT_MAX_ROWS if export_mode else (allow_norm.get("max_rows") or DEFAULT_MAX_ROWS)
    effective_limit = min(qs.limit, max_rows)
    warnings: list[str] = []
    limit_clamped = None
    if qs.limit > max_rows:
        limit_clamped = (qs.limit, max_rows)
        warnings.append(f"limit clamped from {qs.limit} to {max_rows}")

    meta: dict[str, Any] = {
        "row_cap": max_rows,
        "applied_limit": effective_limit,
        "warnings": warnings,
        "elapsed_ms": 0,
        "rows_returned": 0,
        "rows_capped": False,
        "sql_preview": None,
        "queryspec_hash": queryspec_hash(qs),
        "limit_clamped": limit_clamped,
    }

    if isinstance(engine, pd.DataFrame):
        data = _execute_against_dataframe(
            qs, engine, metric_reg, effective_limit, meta,
            timeout_ms=EXECUTOR_TIMEOUT_MS,
            max_rows=max_rows,
        )
    else:
        data = _execute_against_duckdb(
            qs, engine, metric_reg, effective_limit, meta, base_df,
            timeout_ms=EXECUTOR_TIMEOUT_MS,
            max_rows=max_rows,
        )

    if meta.get("status") == "timeout":
        meta["message"] = meta.get("message", "Query timed out. Try narrowing filters or time range.")
        return QueryResult(
            data=_empty_result_with_schema(qs),
            numbers={},
            chart_spec=_chart_spec_from_qs(qs),
            explain_context={},
            meta=meta,
        )

    data = _apply_allowlist(data, allowlist or {})
    if len(data) > max_rows:
        data = data.head(max_rows)
        meta["rows_capped"] = True
        meta["rows_returned"] = len(data)
        meta.setdefault("warnings", []).append(f"Result capped to {max_rows} rows")

    if meta.get("status") is None:
        meta["status"] = "capped" if meta.get("rows_capped") else "ok"
    meta["message"] = meta.get("message") or ("Result capped to max rows." if meta.get("rows_capped") else "OK.")
    chart_spec = _chart_spec_from_qs(qs)
    numbers = _build_numbers(qs, data, metric_reg)
    explain_context = _build_explain_context(qs, data, metric_reg)
    meta["exec_ms"] = meta.get("elapsed_ms", 0)

    return QueryResult(
        data=data,
        numbers=numbers,
        chart_spec=chart_spec,
        explain_context=explain_context,
        meta=meta,
    )
