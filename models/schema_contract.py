from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RawRowContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    month_end: datetime
    channel_raw: str
    channel_standard: str
    channel_best: str
    src_country: str
    product_country: str
    product_ticker: str
    segment: str
    sub_segment: str
    end_aum: float
    nnb: float
    nnf: Optional[float] = Field(default=None)


class MetricsRowContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    month_end: datetime
    channel: str
    product_ticker: str
    src_country: str
    segment: str
    sub_segment: str
    begin_aum: Optional[float] = None
    end_aum: float
    nnb: float
    nnf: Optional[float] = None
    ogr: Optional[float] = None
    market_impact: Optional[float] = None
    market_impact_rate: Optional[float] = None
    fee_yield: Optional[float] = None

