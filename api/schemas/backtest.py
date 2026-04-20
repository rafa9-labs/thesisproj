"""Backtest request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    pair: str = "EURUSD"
    models: List[str] = ["logistic"]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    trading_costs: bool = True
    months: int = 3
    repeats: int = 1
    config_overrides: Dict[str, Any] = Field(default_factory=dict)


class BacktestSubmitResponse(BaseModel):
    job_id: str
    status: str
    pair: str
    models: List[str]


class BacktestStatusResponse(BaseModel):
    job_id: str
    type: str
    status: str
    created_at: str
    updated_at: str
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class BacktestResultMetrics(BaseModel):
    model: str
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    total_return_pct: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    profit_factor: Optional[float] = None
    avg_trade: Optional[float] = None
    active_rate: Optional[float] = None
    directional_accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    equity_curve: Optional[List[float]] = None
    monthly_df: Optional[List[Dict[str, Any]]] = None


class BacktestResultsResponse(BaseModel):
    job_id: str
    pair: str
    models: List[str]
    metrics: List[BacktestResultMetrics]
    monthly_results: Optional[List[Dict[str, Any]]] = None


class BacktestListItem(BaseModel):
    job_id: str
    type: str
    status: str
    pair: str
    models: List[str]
    created_at: str


class BacktestListResponse(BaseModel):
    jobs: List[BacktestListItem]


class ModelInfo(BaseModel):
    name: str
    display_name: str
    category: str
    description: str


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
