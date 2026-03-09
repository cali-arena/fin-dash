"""
Governed NLQ execution schema. Only QuerySpec drives execution.
User text is NEVER executed as SQL or arbitrary code; it is parsed into a QuerySpec
and execution uses solely this schema plus metric_registry and dim_registry.
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


TIME_DIM_KEYS = frozenset({"month_end"})


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: date | None = None
    end: date | None = None
    granularity: Literal["month"] = "month"


class SortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    by: str | None = None
    order: Literal["asc", "desc"] = "desc"


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["line", "bar", "table"] = "table"
    x: str | None = None
    y: str | None = None
    series: str | None = None


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric_id: str
    dimensions: list[str] = []
    filters: dict[str, list[str]] = {}
    time_range: TimeRange = TimeRange()
    sort: SortSpec = SortSpec()
    limit: int = 50
    chart: ChartSpec = ChartSpec()

    @field_validator("metric_id")
    @classmethod
    def metric_id_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("metric_id must be non-empty")
        return v.strip()

    @field_validator("dimensions")
    @classmethod
    def dimensions_unique_lowercase(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for d in v:
            if not d or not isinstance(d, str):
                raise ValueError("dimensions must be non-empty strings")
            key = d.strip().lower()
            if not key:
                raise ValueError("dimensions must be non-empty strings")
            if key in seen:
                raise ValueError("dimensions must be unique")
            seen.add(key)
            out.append(key)
        return out

    @field_validator("filters")
    @classmethod
    def filters_keys_valid(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        for k in v:
            if not k or not k.strip().lower():
                raise ValueError("filter keys must be non-empty dimension keys")
        return v

    @field_validator("limit")
    @classmethod
    def limit_range(cls, v: int) -> int:
        if v < 1 or v > 500:
            raise ValueError("limit must be between 1 and 500")
        return v

    @model_validator(mode="after")
    def filters_keys_in_dimensions_or_time(self) -> "QuerySpec":
        allowed = set(self.dimensions) | TIME_DIM_KEYS
        for key in self.filters:
            if key.strip().lower() not in allowed:
                raise ValueError(f"filter key '{key}' must be one of dimensions or time dims; allowed: {sorted(allowed)}")
        return self

    @model_validator(mode="after")
    def sort_by_allowed(self) -> "QuerySpec":
        if self.sort.by and self.sort.by.strip():
            by = self.sort.by.strip().lower()
            if by not in {"metric"} | set(self.dimensions):
                raise ValueError(f"sort.by must be 'metric' or one of dimensions; got '{self.sort.by}'")
        return self

    @model_validator(mode="after")
    def chart_rules(self) -> "QuerySpec":
        t = self.chart.type
        x = (self.chart.x or "").strip().lower() or None
        y = (self.chart.y or "").strip().lower() or None
        series = (self.chart.series or "").strip().lower() or None
        if t == "line":
            if x != "month_end":
                raise ValueError("chart type 'line' requires x='month_end'")
            if y != "metric":
                raise ValueError("chart type 'line' requires y='metric'")
            if series is not None and series not in self.dimensions:
                raise ValueError(f"chart.series for line must be one of dimensions: {self.dimensions}")
        elif t == "bar":
            if not x or x not in self.dimensions:
                raise ValueError(f"chart type 'bar' requires x to be one of dimensions: {self.dimensions}")
            if y != "metric":
                raise ValueError("chart type 'bar' requires y='metric'")
        return self
