"""Backtest request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


HPOIntensity = Literal["light", "quick", "standard", "deep"]

HPO_TRIAL_MAPS: Dict[str, Dict[str, Dict[str, int]]] = {
    "light": {
        "logistic": {"random": 1, "bayes": 1},
        "svm": {"random": 1, "bayes": 1},
        "decision_tree": {"random": 1, "bayes": 1},
        "random_forest": {"random": 1, "bayes": 1},
        "xgboost": {"random": 1, "bayes": 1},
        "lstm": {"random": 1, "bayes": 1},
        "cnn": {"random": 1, "bayes": 1},
        "transformer": {"random": 1, "bayes": 1},
        "ensemble_adaptive_regime": {"random": 1, "bayes": 1},
        "ensemble_cnn_lstm_xgboost": {"random": 1, "bayes": 1},
        "dqn": {"random": 1, "bayes": 1},
    },
    "quick": {
        "logistic": {"random": 2, "bayes": 2},
        "svm": {"random": 2, "bayes": 2},
        "decision_tree": {"random": 2, "bayes": 2},
        "random_forest": {"random": 2, "bayes": 2},
        "xgboost": {"random": 2, "bayes": 2},
        "lstm": {"random": 2, "bayes": 2},
        "cnn": {"random": 2, "bayes": 2},
        "transformer": {"random": 2, "bayes": 2},
        "ensemble_adaptive_regime": {"random": 2, "bayes": 2},
        "ensemble_cnn_lstm_xgboost": {"random": 2, "bayes": 2},
        "dqn": {"random": 1, "bayes": 1},
    },
    "standard": {
        "logistic": {"random": 5, "bayes": 5},
        "svm": {"random": 5, "bayes": 5},
        "decision_tree": {"random": 5, "bayes": 5},
        "random_forest": {"random": 5, "bayes": 10},
        "xgboost": {"random": 5, "bayes": 15},
        "lstm": {"random": 3, "bayes": 7},
        "cnn": {"random": 3, "bayes": 7},
        "transformer": {"random": 3, "bayes": 7},
        "ensemble_adaptive_regime": {"random": 2, "bayes": 3},
        "ensemble_cnn_lstm_xgboost": {"random": 2, "bayes": 3},
        "dqn": {"random": 2, "bayes": 3},
    },
    "deep": {
        "logistic": {"random": 10, "bayes": 10},
        "svm": {"random": 10, "bayes": 10},
        "decision_tree": {"random": 5, "bayes": 10},
        "random_forest": {"random": 10, "bayes": 20},
        "xgboost": {"random": 10, "bayes": 30},
        "lstm": {"random": 5, "bayes": 15},
        "cnn": {"random": 5, "bayes": 15},
        "transformer": {"random": 5, "bayes": 15},
        "ensemble_adaptive_regime": {"random": 3, "bayes": 7},
        "ensemble_cnn_lstm_xgboost": {"random": 3, "bayes": 7},
        "dqn": {"random": 3, "bayes": 5},
    },
}


class BacktestRequest(BaseModel):
    pair: str = "EURUSD"
    models: List[str] = ["logistic"]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    trading_costs: bool = True
    months: int = 3
    repeats: int = 1
    seed: int = 42
    hpo_intensity: HPOIntensity = "quick"
    config_overrides: Dict[str, Any] = Field(default_factory=dict)


CLASSIC_QUICK_TEST = {
    "pair": "EURUSD",
    "timeframe": "H1",
    "models": ["logistic", "cnn"],
    "start_date": None,
    "end_date": None,
    "months": 3,
    "repeats": 1,
    "seed": 42,
    "hpo_intensity": "quick",
    "trading_costs": True,
    "train_months": 12,
    "test_months": 1,
}

CLASSIC_VALIDATE = {
    "pair": "EURUSD",
    "timeframe": "H1",
    "models": ["logistic"],
    "start_date": None,
    "end_date": None,
    "months": 1,
    "repeats": 1,
    "seed": 42,
    "hpo_intensity": "light",
    "trading_costs": True,
    "train_months": 12,
    "test_months": 1,
}


class QuickTestPreset(BaseModel):
    name: str
    label: str
    description: str
    pair: str
    timeframe: str
    models: List[str]
    months: int
    hpo_intensity: HPOIntensity
    seed: int
    repeats: int
    trading_costs: bool


QUICK_TEST_PRESETS: List[QuickTestPreset] = [
    QuickTestPreset(
        name="validate",
        label="Validate",
        description="Smoke test -- 1 model, 1 month, minimal HPO. ~30s",
        pair="EURUSD",
        timeframe="H1",
        models=["logistic"],
        months=1,
        hpo_intensity="light",
        seed=42,
        repeats=1,
        trading_costs=True,
    ),
    QuickTestPreset(
        name="quick",
        label="Quick Test",
        description="Fast validation -- 2 models, 3 months, light HPO. ~3-5 min",
        pair="EURUSD",
        timeframe="H1",
        models=["logistic", "cnn"],
        months=3,
        hpo_intensity="quick",
        seed=42,
        repeats=1,
        trading_costs=True,
    ),
]


class DateRangePreset(BaseModel):
    key: str
    label: str
    start_date: str
    end_date: str


class DateRangeResponse(BaseModel):
    symbol: str
    timeframe: str
    data_start: str
    data_end: str
    presets: List[DateRangePreset]


class RuntimeEstimateRequest(BaseModel):
    models: List[str]
    months: int
    hpo_intensity: HPOIntensity = "quick"


class RuntimeEstimateResponse(BaseModel):
    models: List[str]
    months: int
    hpo_intensity: HPOIntensity
    total_trials: int
    estimated_minutes_low: float
    estimated_minutes_high: float


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


class BacktestEventsResponse(BaseModel):
    events: list
    total: int


class OverfittingCI(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None
    mean: Optional[float] = None


class OverfittingReport(BaseModel):
    overfit_score: float = 0.0
    risk_level: str = "low"
    risk_color: str = "green"
    train_oos_gap_pct: float = 0.0
    temporal_degradation_pct: float = 0.0
    sharpe_ci: Optional[OverfittingCI] = None
    return_ci: Optional[OverfittingCI] = None
    maxdd_ci: Optional[OverfittingCI] = None
    cv_sharpe_mean: Optional[float] = None
    cv_sharpe_std: Optional[float] = None
    cv_return_mean: Optional[float] = None
    cv_return_std: Optional[float] = None
    min_trl_trades: int = 10
    sufficient_trades: bool = False
    n_periods: int = 0
    n_signal_periods: int = 0
    signal_gap_pct: float = 0.0
    is_mean_sharpe: Optional[float] = None
    oos_mean_sharpe: Optional[float] = None


class WalkForwardPeriod(BaseModel):
    period_start: str = ""
    period_end: str = ""
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_sharpe: Optional[float] = None
    train_sharpe: Optional[float] = None
    strategy_return: Optional[float] = None
    bh_return: Optional[float] = None
    trades: int = 0
    signals_raw: int = 0
    signals_passed_gate: int = 0
    pct_sideways: Optional[float] = None
    pct_trend: Optional[float] = None
    pct_volatile: Optional[float] = None
    sharpe_gap_pct: Optional[float] = None
    return_gap_pct: Optional[float] = None


class FeatureImportanceEntry(BaseModel):
    feature: str
    importance: float


class PredictionHistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    bin_center: float
    count: int


class ConfusionMatrixData(BaseModel):
    matrix: Optional[List[List[int]]] = None
    labels: List[str] = ["Short", "Flat", "Long"]


class ConfidenceBand(BaseModel):
    band_min: float
    band_max: float
    count: int
    accuracy: float
    mean_return: float


class TrainingDiagnostics(BaseModel):
    feature_importance: Optional[List[FeatureImportanceEntry]] = None
    prediction_histogram: Optional[List[PredictionHistogramBin]] = None
    confusion_matrix: Optional[ConfusionMatrixData] = None
    confidence_bands: Optional[List[ConfidenceBand]] = None


class BacktestResultMetrics(BaseModel):
    model: str
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return_pct: Optional[float] = None
    cagr: Optional[float] = None
    calmar_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    profit_factor: Optional[float] = None
    avg_trade: Optional[float] = None
    active_rate: Optional[float] = None
    directional_accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    equity_curve: Optional[List[Dict[str, Any]]] = None
    buy_hold_curve: Optional[List[Dict[str, Any]]] = None
    drawdown_curve: Optional[List[Dict[str, Any]]] = None
    monthly_results: Optional[List[Dict[str, Any]]] = None
    trades: Optional[List[Dict[str, Any]]] = None
    hpo_param_importance: Optional[List[Dict[str, Any]]] = None
    hpo_trials: Optional[List[Dict[str, Any]]] = None
    overfitting: Optional[OverfittingReport] = None
    walkforward_periods: Optional[List[WalkForwardPeriod]] = None
    diagnostics: Optional[TrainingDiagnostics] = None
    summary_text: Optional[str] = None


class BacktestResultsResponse(BaseModel):
    job_id: str
    pair: str
    models: List[str]
    config: Optional[Dict[str, Any]] = None
    metrics: List[BacktestResultMetrics]


class BacktestListItem(BaseModel):
    job_id: str
    type: str
    status: str
    pair: str
    models: List[str]
    created_at: str


class BacktestListResponse(BaseModel):
    jobs: List[BacktestListItem]
    total: int = 0
    offset: int = 0
    limit: int = 50


class BacktestSummaryItem(BaseModel):
    job_id: str
    created_at: str
    pair: str
    timeframe: str = ""
    models: List[str]
    sharpe: Optional[float] = None
    total_return_pct: Optional[float] = None
    win_rate: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    total_trades: Optional[int] = None
    status: str = "completed"


class BacktestSummaryResponse(BaseModel):
    results: List[BacktestSummaryItem]
    total: int = 0
    offset: int = 0
    limit: int = 50


class ModelInfo(BaseModel):
    name: str
    display_name: str
    category: str
    description: str


class HeatmapCell(BaseModel):
    model: str
    pair: str
    sharpe: Optional[float] = None
    total_return_pct: Optional[float] = None
    win_rate: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: Optional[int] = None
    job_id: Optional[str] = None


class HeatmapResponse(BaseModel):
    models: List[str]
    pairs: List[str]
    cells: List[HeatmapCell]


class CrossPairCurve(BaseModel):
    model: str
    pair: str
    equity_curve: Optional[List[Dict[str, Any]]] = None


class CrossPairCurvesResponse(BaseModel):
    model: str
    curves: List[CrossPairCurve]


class ModelListResponse(BaseModel):
    models: List[ModelInfo]
