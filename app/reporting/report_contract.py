"""
Strict Report Contract loader and validator.
Enforces: every sentence maps to a computed number; placeholders match exactly.
"""
from __future__ import annotations

import re
from pathlib import Path
from string import Formatter
from typing import Any

REQUIRED_SECTION_ORDER = [
    "overview_firm",
    "channel_commentary",
    "product_etf_commentary",
    "geo_commentary",
    "anomalies_summary",
    "recommendations",
]


def _parse_placeholders_from_text(text: str) -> set[str]:
    """Extract {field_name} placeholders from template text using string.Formatter().parse()."""
    if not text:
        return set()
    seen: set[str] = set()
    for _literal, field_name, _format_spec, _conversion in Formatter().parse(text):
        if field_name is not None and str(field_name).strip():
            seen.add(field_name.strip())
    return seen


def load_report_contract(path: str | Path = "app/config/report_contract.yml") -> dict[str, Any]:
    """Load report contract YAML from path; return dict. Does not validate."""
    path = Path(path)
    if not path.is_absolute():
        base = Path(__file__).resolve().parents[2]
        path = base / path
    if not path.exists():
        raise FileNotFoundError(f"Report contract not found: {path}")
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Failed to load report contract from {path}: {e}") from e


def _allowed_metric_and_output_names(contract: dict[str, Any]) -> set[str]:
    """Set of all metric ids, section output_fields, and rule placeholders for rule 'when' validation."""
    allowed: set[str] = set()
    metrics = contract.get("metrics") or {}
    allowed.update(metrics.keys())
    for sec in contract.get("sections") or []:
        for f in sec.get("output_fields") or []:
            allowed.add(f)
    narrative = contract.get("narrative") or {}
    for r in narrative.get("rules") or []:
        for p in r.get("placeholders") or []:
            if p is not None and str(p).strip():
                allowed.add(str(p).strip())
    return allowed


def validate_report_contract(contract: dict[str, Any], strict: bool = True) -> None:
    """
    Validate contract structure and placeholder consistency. Hard fail with ValueError.
    strict=True: any mismatch raises ValueError (default).
    """
    if not contract:
        raise ValueError("Report contract is empty")

    # --- Sections exist and in required order
    sections = contract.get("sections")
    if not sections:
        raise ValueError("Contract has no 'sections'")
    if len(sections) != len(REQUIRED_SECTION_ORDER):
        raise ValueError(
            f"Section count mismatch: expected {len(REQUIRED_SECTION_ORDER)} sections, got {len(sections)}"
        )
    for i, expected_id in enumerate(REQUIRED_SECTION_ORDER):
        sec = sections[i] if i < len(sections) else None
        sec_id = (sec.get("id") if isinstance(sec, dict) else None) if sec else None
        if sec_id != expected_id:
            raise ValueError(
                f"Sections must be in fixed order: at index {i} expected id {expected_id!r}, got {sec_id!r}"
            )

    # --- Metrics exist
    metrics = contract.get("metrics") or {}
    if not metrics:
        raise ValueError("Contract has no 'metrics'")

    # --- Per-section: required_metrics in contract.metrics, templates
    narrative = contract.get("narrative") or {}
    templates_by_section = (narrative.get("templates") or {})

    for sec in sections:
        sec_id = sec.get("id", "")
        required_metrics = sec.get("required_metrics") or []
        for mid in required_metrics:
            if mid not in metrics:
                raise ValueError(
                    f"Section {sec_id!r} requires metric {mid!r} which is not defined in contract.metrics"
                )

        section_templates = templates_by_section.get(sec_id)
        if not section_templates:
            continue
        for t in section_templates:
            if not isinstance(t, dict):
                raise ValueError(f"Section {sec_id!r}: template must be a dict, got {type(t).__name__}")
            text = t.get("text") or ""
            placeholders_list = t.get("placeholders")
            if placeholders_list is None:
                raise ValueError(f"Section {sec_id!r} template {t.get('id', '?')!r}: missing 'placeholders'")
            if not isinstance(placeholders_list, (list, tuple)):
                raise ValueError(
                    f"Section {sec_id!r} template {t.get('id', '?')!r}: 'placeholders' must be a list"
                )
            declared = set(p for p in placeholders_list if p is not None and str(p).strip())
            if not declared and strict:
                raise ValueError(
                    f"Section {sec_id!r} template {t.get('id', '?')!r}: placeholders list is empty"
                )
            in_text = _parse_placeholders_from_text(text)
            if in_text != declared:
                only_in_text = in_text - declared
                only_in_list = declared - in_text
                if only_in_text:
                    raise ValueError(
                        f"Section {sec_id!r} template {t.get('id', '?')!r}: placeholders in text but not "
                        f"declared: {sorted(only_in_text)}"
                    )
                if only_in_list:
                    raise ValueError(
                        f"Section {sec_id!r} template {t.get('id', '?')!r}: placeholders declared but not "
                        f"used in text: {sorted(only_in_list)}"
                    )

    # --- Rules: when vars allowed, emit_template or own text+placeholders, text placeholders match
    rules = narrative.get("rules") or []
    allowed_when = _allowed_metric_and_output_names(contract)
    # Allow common numeric literal patterns in 'when'; extract identifiers
    when_var_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")

    for r in rules:
        if not isinstance(r, dict):
            raise ValueError(f"Rule must be a dict, got {type(r).__name__}")
        rid = r.get("id", "?")
        when = r.get("when")
        if when is None or not str(when).strip():
            raise ValueError(f"Rule {rid!r}: missing or empty 'when'")
        when_vars = set(when_var_pattern.findall(str(when)))
        # Remove Python keywords and numeric-looking tokens
        for v in list(when_vars):
            if v in ("and", "or", "not", "True", "False", "in", "is"):
                when_vars.discard(v)
            elif v.replace(".", "").replace("-", "").isdigit():
                when_vars.discard(v)
        unknown = when_vars - allowed_when
        if unknown and strict:
            raise ValueError(
                f"Rule {rid!r}: 'when' references names not in metrics or section output_fields: {sorted(unknown)}"
            )
        text = r.get("text") or ""
        placeholders_list = r.get("placeholders")
        if not text and not r.get("emit_template"):
            raise ValueError(f"Rule {rid!r}: must have 'text' and 'placeholders' or 'emit_template'")
        if text:
            if placeholders_list is None:
                raise ValueError(f"Rule {rid!r}: has 'text' but missing 'placeholders'")
            declared = set(p for p in (placeholders_list or []) if p is not None and str(p).strip())
            in_text = _parse_placeholders_from_text(text)
            if in_text != declared:
                only_in_text = in_text - declared
                only_in_list = declared - in_text
                if only_in_text:
                    raise ValueError(
                        f"Rule {rid!r}: placeholders in text but not declared: {sorted(only_in_text)}"
                    )
                if only_in_list:
                    raise ValueError(
                        f"Rule {rid!r}: placeholders declared but not used in text: {sorted(only_in_list)}"
                    )


def get_placeholders_in_text(text: str) -> list[str]:
    """Public helper: return ordered list of placeholder names found in text (using Formatter.parse)."""
    seen: list[str] = []
    for _literal, field_name, _format_spec, _conversion in Formatter().parse(text):
        if field_name is not None and str(field_name).strip() and field_name.strip() not in seen:
            seen.append(field_name.strip())
    return seen
