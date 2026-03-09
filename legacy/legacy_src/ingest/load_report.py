"""
Typed load report for robust ingestion. Matches load_report dict shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LoadReport:
    xlsx_path: str
    sheet_name: str
    rows_read: int
    columns_detected_original: list[str]
    columns_detected_normalized: list[str]
    expected_columns_contract: list[str]
    missing_columns: list[str]
    extra_columns: list[str]
    selected_columns_used: list[str]
    header_mapping_original_to_normalized: dict[str, str]
    header_mapping_normalized_to_original: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "xlsx_path": self.xlsx_path,
            "sheet_name": self.sheet_name,
            "rows_read": self.rows_read,
            "columns_detected_original": self.columns_detected_original,
            "columns_detected_normalized": self.columns_detected_normalized,
            "expected_columns_contract": self.expected_columns_contract,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "selected_columns_used": self.selected_columns_used,
            "header_mapping_original_to_normalized": self.header_mapping_original_to_normalized,
            "header_mapping_normalized_to_original": self.header_mapping_normalized_to_original,
        }

    def pretty(self) -> str:
        """Multiline human-readable report."""
        lines = [
            "LoadReport",
            f"  xlsx_path: {self.xlsx_path}",
            f"  sheet_name: {self.sheet_name}",
            f"  rows_read: {self.rows_read}",
            f"  columns_detected_original: {self.columns_detected_original}",
            f"  columns_detected_normalized: {self.columns_detected_normalized}",
            f"  expected_columns_contract: {self.expected_columns_contract}",
            f"  missing_columns: {self.missing_columns}",
            f"  extra_columns: {self.extra_columns}",
            f"  selected_columns_used: {self.selected_columns_used}",
        ]
        return "\n".join(lines)
