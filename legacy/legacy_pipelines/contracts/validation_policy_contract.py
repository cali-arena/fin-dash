"""
Validation policy contract: strict config + schema validation for checksum-style workbook comparison.

Validates configs/validation_policy.yml and exposes:
- load_and_validate_validation_policy(path) -> ValidationPolicy
- required_expected_excel_columns(policy) -> set[str]
- summarize_policy(policy) -> dict (for metadata logging)
- policy_hash(policy) -> str (stable SHA-256 of canonical JSON; any policy change invalidates cached validation runs)

Invariants:
- expected_columns must include exactly: asset_growth_rate, organic_growth_rate, external_market_growth_rate
  (extra keys in YAML are allowed but warned).
- Duplicate mapping targets: if two canonical keys map to the same Excel column name → ValidationPolicyError.

Caching: Any policy change invalidates cached validation runs. Store policy_hash(policy) with validation
results and re-run validation when the hash changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_POLICY_PATH = "configs/validation_policy.yml"

# Canonical expected column keys (exactly these are required; extra in YAML allowed but warned).
CANONICAL_EXPECTED_KEYS = frozenset({
    "asset_growth_rate",
    "organic_growth_rate",
    "external_market_growth_rate",
})


class ValidationPolicyError(Exception):
    """Raised when validation_policy.yml is missing or invalid."""


@dataclass
class WorkbookConfig:
    path: str
    sheet: str
    month_column: str
    month_format: str | None


@dataclass
class ExpectedColumnsConfig:
    asset_growth_rate: str
    organic_growth_rate: str
    external_market_growth_rate: str


@dataclass
class NormalizationConfig:
    percent_to_decimal: bool
    percent_scale: float
    month_align: str
    timezone_naive: bool


@dataclass
class ToleranceConfig:
    abs_tol: float
    rel_tol: float


@dataclass
class FailFastConfig:
    max_mismatched_months: int
    max_deviation: float
    fail_on_missing_months: bool
    # Step 4 fail-fast gates (YAML: max_fail_months, max_abs_err, max_rel_err; fallback to above when missing)
    max_fail_months: int = 0
    max_abs_err: float = 0.0
    max_rel_err: float = 0.05


@dataclass
class HighlightedConfig:
    """Optional: which months are highlighted (drive fail-fast). mode: column | list | none."""
    mode: str
    column: str | None
    months: list[str] | None


DEFAULT_HIGHLIGHTED = HighlightedConfig(mode="none", column=None, months=None)


@dataclass
class ValidationPolicy:
    workbook: WorkbookConfig
    expected_columns: ExpectedColumnsConfig
    normalization: NormalizationConfig
    tolerance: ToleranceConfig
    fail_fast: FailFastConfig
    highlighted: HighlightedConfig


def _require_mapping(obj: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValidationPolicyError(f"{ctx} must be a mapping/object; got {type(obj).__name__}")
    return obj


def _require_str(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationPolicyError(f"{ctx} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, ctx: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationPolicyError(f"{ctx} must be a boolean")
    return value


def _require_float(value: Any, ctx: str, min_value: float | None = None) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValidationPolicyError(f"{ctx} must be a number") from None
    if min_value is not None and v < min_value:
        raise ValidationPolicyError(f"{ctx} must be >= {min_value}")
    return v


def _require_int(value: Any, ctx: str, min_value: int | None = None) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValidationPolicyError(f"{ctx} must be an integer") from None
    if min_value is not None and v < min_value:
        raise ValidationPolicyError(f"{ctx} must be >= {min_value}")
    return v


def _parse_workbook(cfg: Mapping[str, Any]) -> WorkbookConfig:
    wb = _require_mapping(cfg.get("workbook"), "validation.workbook")
    path = _require_str(wb.get("path"), "validation.workbook.path")
    sheet = _require_str(wb.get("sheet"), "validation.workbook.sheet")
    month_column = _require_str(wb.get("month_column"), "validation.workbook.month_column")
    month_format_raw = wb.get("month_format", None)
    if month_format_raw is not None and not isinstance(month_format_raw, str):
        raise ValidationPolicyError("validation.workbook.month_format must be a string or null")
    month_format = month_format_raw if isinstance(month_format_raw, str) and month_format_raw.strip() else None
    return WorkbookConfig(path=path, sheet=sheet, month_column=month_column, month_format=month_format)


def _parse_expected_columns(cfg: Mapping[str, Any]) -> ExpectedColumnsConfig:
    exp = _require_mapping(cfg.get("expected_columns"), "validation.expected_columns")
    missing = sorted(CANONICAL_EXPECTED_KEYS.difference(exp.keys()))
    if missing:
        raise ValidationPolicyError(f"validation.expected_columns missing keys: {missing}")
    extra = sorted(set(exp.keys()) - CANONICAL_EXPECTED_KEYS)
    if extra:
        logger.warning(
            "validation.expected_columns has extra keys (ignored): %s. Canonical set: %s.",
            extra,
            sorted(CANONICAL_EXPECTED_KEYS),
        )
    asset = _require_str(exp.get("asset_growth_rate"), "validation.expected_columns.asset_growth_rate")
    organic = _require_str(exp.get("organic_growth_rate"), "validation.expected_columns.organic_growth_rate")
    external = _require_str(
        exp.get("external_market_growth_rate"),
        "validation.expected_columns.external_market_growth_rate",
    )
    # Duplicate mapping targets: two canonical keys must not map to the same Excel column.
    excel_names = [("asset_growth_rate", asset), ("organic_growth_rate", organic), ("external_market_growth_rate", external)]
    seen: dict[str, str] = {}
    for canonical, excel_name in excel_names:
        if excel_name in seen:
            raise ValidationPolicyError(
                f"validation.expected_columns duplicate mapping: both {seen[excel_name]!r} and {canonical!r} "
                f"map to Excel column {excel_name!r}."
            )
        seen[excel_name] = canonical
    return ExpectedColumnsConfig(
        asset_growth_rate=asset,
        organic_growth_rate=organic,
        external_market_growth_rate=external,
    )


def _parse_normalization(cfg: Mapping[str, Any]) -> NormalizationConfig:
    norm = _require_mapping(cfg.get("normalization"), "validation.normalization")
    percent_to_decimal = _require_bool(norm.get("percent_to_decimal"), "validation.normalization.percent_to_decimal")
    percent_scale = _require_float(norm.get("percent_scale", 100), "validation.normalization.percent_scale")
    if percent_to_decimal and percent_scale <= 0:
        raise ValidationPolicyError(
            "validation.normalization.percent_scale must be > 0 when percent_to_decimal is true"
        )
    month_align = _require_str(norm.get("month_align"), "validation.normalization.month_align")
    timezone_naive = _require_bool(norm.get("timezone_naive"), "validation.normalization.timezone_naive")
    return NormalizationConfig(
        percent_to_decimal=percent_to_decimal,
        percent_scale=percent_scale,
        month_align=month_align,
        timezone_naive=timezone_naive,
    )


def _parse_tolerance(cfg: Mapping[str, Any]) -> ToleranceConfig:
    tol = _require_mapping(cfg.get("tolerance"), "validation.tolerance")
    abs_tol = _require_float(tol.get("abs_tol"), "validation.tolerance.abs_tol", min_value=0.0)
    rel_tol = _require_float(tol.get("rel_tol"), "validation.tolerance.rel_tol", min_value=0.0)
    return ToleranceConfig(abs_tol=abs_tol, rel_tol=rel_tol)


def _parse_fail_fast(cfg: Mapping[str, Any]) -> FailFastConfig:
    ff = _require_mapping(cfg.get("fail_fast"), "validation.fail_fast")
    # Prefer new names; fallback to legacy for backward compatibility
    max_fail_months_raw = ff.get("max_fail_months", ff.get("max_mismatched_months"))
    max_abs_err_raw = ff.get("max_abs_err", ff.get("max_deviation"))
    max_mismatched_months = _require_int(
        max_fail_months_raw,
        "validation.fail_fast.max_fail_months (or max_mismatched_months)",
        min_value=0,
    )
    max_deviation = _require_float(
        max_abs_err_raw,
        "validation.fail_fast.max_abs_err (or max_deviation)",
        min_value=0.0,
    )
    max_rel_err = _require_float(
        ff.get("max_rel_err", 0.05),
        "validation.fail_fast.max_rel_err",
        min_value=0.0,
    )
    fail_on_missing_months = _require_bool(
        ff.get("fail_on_missing_months"),
        "validation.fail_fast.fail_on_missing_months",
    )
    return FailFastConfig(
        max_mismatched_months=max_mismatched_months,
        max_deviation=max_deviation,
        fail_on_missing_months=fail_on_missing_months,
        max_fail_months=max_mismatched_months,
        max_abs_err=max_deviation,
        max_rel_err=max_rel_err,
    )


def _parse_highlighted(cfg: Mapping[str, Any]) -> HighlightedConfig:
    """Parse optional validation.highlighted. Default mode=none when missing."""
    hl = cfg.get("highlighted")
    if hl is None or not isinstance(hl, Mapping):
        return HighlightedConfig(mode="none", column=None, months=None)
    mode = (hl.get("mode") or "none").strip().lower()
    if mode not in ("column", "list", "none"):
        mode = "none"
    column = None
    if mode == "column":
        col_raw = hl.get("column")
        column = str(col_raw).strip() if col_raw is not None and str(col_raw).strip() else None
    months: list[str] | None = None
    if mode == "list":
        m_raw = hl.get("months")
        if isinstance(m_raw, list):
            months = [str(x).strip() for x in m_raw if str(x).strip()]
        else:
            months = []
    return HighlightedConfig(mode=mode, column=column, months=months or None)


def load_and_validate_validation_policy(
    path: str | Path = DEFAULT_VALIDATION_POLICY_PATH,
) -> ValidationPolicy:
    """
    Load configs/validation_policy.yml and return a validated ValidationPolicy.
    Raises ValidationPolicyError with a concise message on any schema/semantic issue.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationPolicyError(f"Validation policy config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise ValidationPolicyError(
            "PyYAML is required to load validation policy config. Install with: pip install pyyaml"
        ) from e

    raw_text = path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(raw_text)
    except Exception as e:
        raise ValidationPolicyError(f"Failed to parse YAML {path}: {e}") from e

    root = _require_mapping(raw, "root")
    cfg = root.get("validation")
    if cfg is None:
        raise ValidationPolicyError("Root config must contain 'validation' section")
    cfg = _require_mapping(cfg, "validation")

    workbook = _parse_workbook(cfg)
    expected_columns = _parse_expected_columns(cfg)
    normalization = _parse_normalization(cfg)
    tolerance = _parse_tolerance(cfg)
    fail_fast = _parse_fail_fast(cfg)
    highlighted = _parse_highlighted(cfg)

    return ValidationPolicy(
        workbook=workbook,
        expected_columns=expected_columns,
        normalization=normalization,
        tolerance=tolerance,
        fail_fast=fail_fast,
        highlighted=highlighted,
    )


def required_expected_excel_columns(policy: ValidationPolicy) -> set[str]:
    """
    Return the set of required column names that must be present in the Excel sheet:
    month_column plus all mapped expected rate columns.
    """
    cols: set[str] = {policy.workbook.month_column}
    cols.add(policy.expected_columns.asset_growth_rate)
    cols.add(policy.expected_columns.organic_growth_rate)
    cols.add(policy.expected_columns.external_market_growth_rate)
    return cols


def summarize_policy(policy: ValidationPolicy) -> dict[str, Any]:
    """
    Return a dict suitable for metadata logging. All values are JSON-serializable.
    Used by policy_hash; any change to this structure or policy values changes the hash.
    """
    wb = policy.workbook
    exp = policy.expected_columns
    norm = policy.normalization
    tol = policy.tolerance
    ff = policy.fail_fast
    return {
        "workbook_path": wb.path,
        "sheet": wb.sheet,
        "month_column": wb.month_column,
        "expected_columns": {
            "asset_growth_rate": exp.asset_growth_rate,
            "organic_growth_rate": exp.organic_growth_rate,
            "external_market_growth_rate": exp.external_market_growth_rate,
        },
        "normalization": {
            "percent_to_decimal": norm.percent_to_decimal,
            "percent_scale": norm.percent_scale,
            "month_align": norm.month_align,
            "timezone_naive": norm.timezone_naive,
        },
        "tolerance": {
            "abs_tol": tol.abs_tol,
            "rel_tol": tol.rel_tol,
        },
        "fail_fast": {
            "max_mismatched_months": ff.max_mismatched_months,
            "max_deviation": ff.max_deviation,
            "max_fail_months": ff.max_fail_months,
            "max_abs_err": ff.max_abs_err,
            "max_rel_err": ff.max_rel_err,
            "fail_on_missing_months": ff.fail_on_missing_months,
        },
        "highlighted": {
            "mode": policy.highlighted.mode,
            "column": policy.highlighted.column,
            "months": policy.highlighted.months,
        },
    }


def _canonical_json_dumps(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace. For hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def policy_hash(policy: ValidationPolicy) -> str:
    """
    Return SHA-256 hex digest of canonical JSON of summarize_policy(policy).
    Stable: same policy (same summarize_policy output) always yields the same hash.
    Note: Any policy change invalidates cached validation runs; consumers should store
    this hash with validation results and re-run when it changes.
    """
    summary = summarize_policy(policy)
    payload = _canonical_json_dumps(summary)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
