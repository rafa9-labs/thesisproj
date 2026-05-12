"""Pair and data schemas."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class PairInfo(BaseModel):
    symbol: str
    oanda_name: str
    pip_value: float
    lot_size: float
    base_currency: str
    quote_currency: str
    typical_spread_bps: float


class TimeframeData(BaseModel):
    timeframe: str
    rows: int
    start_date: Optional[str]
    end_date: Optional[str]


class PairDetail(BaseModel):
    pair: PairInfo
    timeframes: List[TimeframeData]


class PairListResponse(BaseModel):
    pairs: List[PairDetail]


class DataRangeResponse(BaseModel):
    symbol: str
    timeframes: List[TimeframeData]


class DownloadRequest(BaseModel):
    pair: str
    years: int = 10


class DownloadResponse(BaseModel):
    job_id: str
    pair: str
    status: str


class DefinePairRequest(BaseModel):
    symbol: str
    pip_value: float
    decimal_places: int = 4


class DataStatusSingle(BaseModel):
    available: bool
    start: Optional[str] = None
    end: Optional[str] = None
    bars: int = 0


class DataStatusResponse(BaseModel):
    symbol: str
    timeframes: Dict[str, DataStatusSingle]
    ready: bool
    missing: List[str] = []
