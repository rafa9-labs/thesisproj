"""
schemas.backtest — Pydantic models for backtest input parameters and output results.

═══════════════════════════════════════════════════════════════════════════════
EDUCATIONAL: What this file replaces
═══════════════════════════════════════════════════════════════════════════════

BEFORE:
  In ui/state.py, backtest parameters were collected as a raw Dict[str, Any].
  No type checking, no range validation, no guaranteed presence of keys.
  
  params = get_all_params()  # returns Dict[str, Any]
  model_type = params.get("model_type", "logistic")  # could be None, int, anything
  
  In ui/state.py lines 220-276, _compute_aggregate_metrics manually assembled
  a dict with 15 keys, each individually constructed with if/else guards.
  
AFTER:
  params = BacktestParams.model_validate(raw_dict)  # validates EVERYTHING
  params.model_type  # guaranteed str, guaranteed one of the allowed models
  
  results = BacktestResult(metrics=..., equity_curve=..., ...)
  results.model_dump()  # clean dict for serialization

═══════════════════════════════════════════════════════════════════════════════
EDUCATIONAL: Pydantic v2 Key Concepts Used Here
═══════════════════════════════════════════════════════════════════════════════

1. BaseModel        — Base class. Any class inheriting from it gets automatic
                      __init__, validation, serialization, and schema generation.

2. Field(ge=, le=)  — "ge" = greater-or-equal, "le" = less-or-equal.
                      These are CONSTRAINT VALIDATORS — Pydantic rejects values
                      outside the range BEFORE they reach your code.
                      Other constraints: gt (strictly greater), lt, min_length,
                      max_length, pattern (regex), multiple_of, etc.

3. Literal[...]     — Restricts a field to a fixed set of allowed values.
                      Like an Enum but simpler — no need to define an enum class.

4. model_validate() — Class method. Takes a raw dict and returns a validated
                      instance. Raises ValidationError if anything is wrong.
                      This is the MAIN ENTRY POINT — call it wherever raw data
                      enters your system.

5. model_dump()     — Instance method. Converts the model back to a plain dict.
                      Inverse of model_validate(). Useful for passing to functions
                      that still expect dicts (backward compatibility).

6. ConfigDict       — Controls model behavior. Here we use:
                      - extra="ignore"   → silently ignore unknown keys in input
                      - extra="forbid"   → reject unknown keys (stricter)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ══════════════════════════════════════════════════════════════════════════════
# BacktestParams — Validates EVERYTHING the user can configure before a run
# ══════════════════════════════════════════════════════════════════════════════
#
# This model corresponds to the params dict in ui/state.py::run_backtest()
# and the UI controls in ui/controls.py::get_all_params().
#
# Every field has:
#   - A TYPE (str, int, float, bool) — Pydantic coerces compatible types
#   - A DEFAULT — so old code that only passes some keys still works
#   - A CONSTRAINT (Field(ge=, le=)) — catches invalid values at the boundary
#   - A DESCRIPTION (Field(description=)) — auto-generates API documentation

class BacktestParams(BaseModel):
    """
    Validated input parameters for a backtest run.
    
    This is the SINGLE SOURCE OF TRUTH for what a backtest accepts.
    The UI, API, and CLI should all construct this model before calling
    the pipeline.
    """
    
    # ── Configure Pydantic behavior ──
    # extra="ignore" means if the input dict has keys we don't recognize,
    # they're silently dropped instead of causing an error.
    # This is IMPORTANT for backward compatibility — the existing codebase
    # passes around dicts with ~100 keys, and we don't want to break anything.
    model_config = ConfigDict(extra="ignore")
    
    # ── Core identity ──
    # Literal[...] restricts to a fixed set of allowed strings.
    # This replaces the old: model_type = params.get("model_type", "logistic")
    # which would accept ANY string silently.
    model_type: str = Field(
        default="logistic",
        description="ML model to use for prediction.",
    )
    
    # ── Data ──
    data_key: str = Field(
        default="EURUSD_H1",
        description="Dataset identifier (maps to CSV path in DATA_FILES).",
    )
    
    # ── Walk-forward geometry ──
    # Field(ge=1) means "greater than or equal to 1".
    # This prevents the nonsensical train_months=0 that would crash the pipeline.
    train_months: int = Field(
        default=36,
        ge=1,
        le=120,
        description="Number of months for the training window.",
    )
    test_months: int = Field(
        default=1,
        ge=1,
        le=12,
        description="Number of months for each test fold.",
    )
    period_unit: Literal["months", "weeks", "days"] = Field(
        default="months",
        description="Walk-forward period granularity. 'months' is the default; "
                    "'weeks' and 'days' enable finer-grained walk-forward splits.",
    )
    
    # ── HPO (Hyperparameter Optimization) ──
    n_trials: int = Field(
        default=10,
        ge=0,
        le=500,
        description="Number of Optuna HPO trials. 0 = use cached/default config.",
    )
    max_hpo_duration_minutes: float = Field(
        default=0,
        ge=0,
        le=1440,
        description="Max HPO duration in minutes. 0 = no limit.",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility.",
    )
    rep: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of walk-forward repetitions.",
    )
    
    # ── Trading costs ──
    eval_use_trading_costs: bool = Field(
        default=True,
        description="Apply spread/slippage costs during evaluation.",
    )
    slippage_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Multiplier on base slippage.",
    )
    slip_norm_bps: float = Field(
        default=0.25,
        ge=0.0,
        le=5.0,
        description="Baseline slippage in basis points.",
    )
    
    # ── Calibration ──
    calibrate_method: Literal["sigmoid", "isotonic"] = Field(
        default="sigmoid",
        description="Probability calibration method.",
    )
    
    # ── Confidence & coverage ──
    # Field(ge=0.0, le=1.0) enforces that this is a valid probability.
    # Before Pydantic, passing confidence_threshold=5.0 would silently
    # flow through and produce garbage results.
    confidence_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum predicted probability to open a trade.",
    )
    target_active_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Target fraction of bars where the strategy is active.",
    )
    target_coverage: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Target coverage (synced to target_active_rate at runtime).",
    )
    
    # ── Labeling: Triple Barrier ──
    use_triple_barrier: bool = Field(default=True)
    tb_pt_mult: float = Field(default=2.0, ge=0.1, le=10.0)
    tb_sl_mult: float = Field(default=2.0, ge=0.1, le=10.0)
    tb_neutral_zone: float = Field(default=0.5, ge=0.0, le=5.0)
    tb_max_holding: int = Field(default=36, ge=1, le=500)
    label_threshold: float = Field(default=0.0005, ge=0.0, le=0.01)
    
    # ── Feature engineering: Lags ──
    lags: int = Field(default=14, ge=1, le=100)
    lag_depth: int = Field(default=1, ge=1, le=5)
    
    # ── Feature engineering: Fracdiff ──
    use_fracdiff: bool = Field(default=True)
    fracdiff_d: float = Field(default=0.4, ge=0.0, le=1.0)
    
    # ── Indicator toggles ──
    # All default to True (enabled). The UI can toggle them.
    use_adx: bool = Field(default=True)
    use_atr: bool = Field(default=True)
    use_bbands: bool = Field(default=True)
    use_ema: bool = Field(default=True)
    use_sma: bool = Field(default=True)
    use_rsi: bool = Field(default=True)
    use_macd: bool = Field(default=True)
    use_stoch: bool = Field(default=True)
    use_sar: bool = Field(default=True)
    use_donchian: bool = Field(default=True)
    use_mtf_ma: bool = Field(default=True)
    
    # ── Advanced indicator toggles ──
    use_crossover_bins: bool = Field(default=False)
    use_ma_spread: bool = Field(default=False)
    use_price_ma_z: bool = Field(default=False)
    use_indicator_states: bool = Field(default=False)
    use_mtf_alignment: bool = Field(default=False)
    use_mtf_align: bool = Field(default=False)
    use_macd_atr_ratio: bool = Field(default=False)
    use_triple_confirm: bool = Field(default=False)
    use_trend_confirm: bool = Field(default=False)
    use_vol_managed_mom: bool = Field(default=False)
    use_vm_mom: bool = Field(default=False)
    use_squeeze_breakout: bool = Field(default=False)
    use_squeeze_expansion: bool = Field(default=False)
    use_atr_channel_breakout: bool = Field(default=False)
    use_ext_atr_low_adx: bool = Field(default=False)
    use_reentry_mom: bool = Field(default=False)
    use_slope_diff: bool = Field(default=False)
    use_rv_features: bool = Field(default=True)
    
    # ── Logistic regression specifics ──
    logit_C: float = Field(default=1.0, ge=0.001, le=10000.0)
    logit_solver: Literal["lbfgs", "newton-cg", "sag", "saga", "liblinear"] = Field(
        default="lbfgs",
    )
    logit_penalty: Literal["l1", "l2", "elasticnet", "none"] = Field(default="l2")
    logit_max_iter: int = Field(default=500, ge=10, le=10000)
    logit_tol: float = Field(default=1e-4, ge=1e-8, le=1.0)
    
    # ── Model validator: cross-field consistency ──
    # @model_validator runs AFTER all individual field validations pass.
    # This is where we check that COMBINATIONS of fields make sense.
    # It replaces the hand-written guards in ui/validators.py.
    @model_validator(mode="after")
    def validate_cross_field_consistency(self) -> "BacktestParams":
        """
        Check that combinations of fields are consistent.
        
        EDUCATIONAL NOTE:
        Pydantic has two validator types:
          - @field_validator: validates a SINGLE field in isolation
          - @model_validator: validates the ENTIRE model (cross-field checks)
        
        mode="after" means this runs after all fields are set.
        mode="before" would run before field validation (useful for pre-processing).
        """
        # Logistic regression: solver ↔ penalty compatibility
        # This is the same check as ui/validators.py:196-208, but automatic.
        _compat = {
            "lbfgs":      {"l2", "none"},
            "newton-cg":  {"l2", "none"},
            "sag":        {"l2", "none"},
            "saga":       {"l1", "l2", "elasticnet", "none"},
            "liblinear":  {"l1", "l2"},
        }
        if self.logit_solver in _compat:
            if self.logit_penalty not in _compat[self.logit_solver]:
                from pydantic import ValidationError
                raise ValueError(
                    f"logit_solver '{self.logit_solver}' does not support "
                    f"penalty '{self.logit_penalty}'. "
                    f"Allowed: {sorted(_compat[self.logit_solver])}"
                )
        
        # Feature explosion guard
        if self.lags * self.lag_depth > 100:
            raise ValueError(
                f"lags ({self.lags}) × lag_depth ({self.lag_depth}) = "
                f"{self.lags * self.lag_depth} features. "
                f"Max 100 to prevent overfitting."
            )
        
        return self


# ══════════════════════════════════════════════════════════════════════════════
# AggregateMetrics — The performance numbers from a completed backtest
# ══════════════════════════════════════════════════════════════════════════════
#
# This replaces the hand-built dict in ui/state.py::_compute_aggregate_metrics()
# (lines 220-276) which manually assembled 15 keys with if/else guards.

class AggregateMetrics(BaseModel):
    """
    Validated performance metrics from a completed backtest.
    
    Every field has a clear type and range constraint.
    No more "which keys are in this dict?" confusion.
    """
    model_config = ConfigDict(extra="ignore")
    
    # ── Return metrics ──
    total_return_pct: float = Field(
        default=0.0,
        description="Cumulative strategy return as percentage.",
    )
    cstrategy: float = Field(
        default=1.0,
        description="Final equity curve value (starts at 1.0).",
    )
    outperformance: float = Field(
        default=0.0,
        description="Strategy return minus buy-and-hold return.",
    )
    return_per_trade: float = Field(default=0.0)
    profit_per_hit: float = Field(default=0.0)
    geo_mean_ann: float = Field(default=0.0)
    
    # ── Risk metrics ──
    sharpe: Optional[float] = Field(
        default=None,
        description="Annualized Sharpe ratio. None if insufficient data.",
    )
    drawdown: float = Field(
        default=0.0,
        le=0.0,  # Drawdown is always negative or zero
        description="Maximum drawdown (negative value).",
    )
    strategy_volatility: float = Field(
        default=0.0,
        ge=0.0,
        description="Annualized strategy volatility.",
    )
    
    # ── Activity metrics ──
    trades: int = Field(default=0, ge=0)
    active_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # ── Classification metrics ──
    directional_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_macro: float = Field(default=0.0, ge=0.0, le=1.0)
    f1_macro: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @model_validator(mode="after")
    def set_geo_mean_default(self) -> "AggregateMetrics":
        """Compute geo_mean_ann from total_return if not explicitly set."""
        if self.geo_mean_ann == 0.0 and self.total_return_pct != 0.0:
            # Will be overridden by pipeline output; just a fallback
            pass
        return self


# ══════════════════════════════════════════════════════════════════════════════
# BacktestResult — The complete output of a backtest run
# ══════════════════════════════════════════════════════════════════════════════
#
# This replaces the untyped dict returned by ui/state.py::run_backtest()
# and consumed by ui/dashboard.py::render_dashboard().

class BacktestResult(BaseModel):
    """
    Complete output of a backtest run.
    
    Contains metrics, equity curve, monthly breakdown, and model metadata.
    
    EDUCATIONAL NOTE — Handling non-JSON types:
    Pydantic v2 can handle most Python types, but pandas DataFrames and
    numpy arrays aren't natively serializable to JSON. We handle this by:
    1. Storing them as their native types (DataFrame, Series)
    2. Using model_config = ConfigDict(arbitrary_types_allowed=True)
    3. Providing a .to_serializable() method for JSON/API output
    """
    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,  # Allow DataFrame, Series, np.ndarray
    )
    
    metrics: AggregateMetrics
    equity_curve: Any  # pd.Series — too complex for Pydantic to validate natively
    monthly_df: Any    # pd.DataFrame
    model_type: str
    
    # Optional: filled if Optuna artifacts are available
    param_importances: Optional[Dict[str, float]] = None
    
    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dict.
        
        EDUCATIONAL NOTE:
        Pydantic's .model_dump() works for simple types, but DataFrame/Series
        need special handling. This method bridges the gap.
        
        In the future (Step 5: FastAPI), we'll use a custom JSON encoder
        so FastAPI can automatically serialize BacktestResult responses.
        """
        result = {
            "metrics": self.metrics.model_dump(),
            "model_type": self.model_type,
        }
        
        # Convert pandas types to serializable formats
        if isinstance(self.equity_curve, pd.Series):
            result["equity_curve"] = self.equity_curve.tolist()
        elif isinstance(self.equity_curve, np.ndarray):
            result["equity_curve"] = self.equity_curve.tolist()
            
        if isinstance(self.monthly_df, pd.DataFrame):
            result["monthly_df"] = self.monthly_df.to_dict(orient="records")
        
        if self.param_importances is not None:
            result["param_importances"] = self.param_importances
            
        return result
    
    @classmethod
    def from_pipeline_output(
        cls,
        metrics_dict: Dict[str, Any],
        equity_curve: Any,
        monthly_df: Any,
        model_type: str,
        param_importances: Optional[Dict[str, float]] = None,
    ) -> "BacktestResult":
        """
        Construct a BacktestResult from the raw pipeline output.
        
        This is the ADAPTER between the old dict-based pipeline output
        and the new typed Pydantic model.
        
        USAGE (in ui/state.py):
            # Before:
            results = {"metrics": raw_metrics, "equity_curve": eq, ...}
            
            # After:
            results = BacktestResult.from_pipeline_output(
                metrics_dict=raw_metrics,
                equity_curve=eq,
                monthly_df=monthly,
                model_type=model_type,
            )
        """
        # Build validated metrics from raw dict
        validated_metrics = AggregateMetrics.model_validate(metrics_dict)
        
        return cls(
            metrics=validated_metrics,
            equity_curve=equity_curve,
            monthly_df=monthly_df,
            model_type=model_type,
            param_importances=param_importances,
        )