"""UserFixedConfig — immutable user choices that Optuna must NOT override.

Category A (User Domain): Feature toggles, execution settings, risk rules, study params.
Category B (Machine Domain): Model hyperparameters, indicator windows, continuous thresholds
                            (sampled conditionally based on Category A).

Contract: Optuna's trial.suggest_* must NEVER sample boolean feature flags.
          The user defines the thesis; the machine optimizes the thresholds.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UserFixedConfig:
    """Immutable configuration bundle from the user (frontend/API).

    These values are passed through the pipeline and are treated as FIXED
    by the Optuna sampler. The sampler reads them, never overwrites them,
    and only samples conditional thresholds for features the user enabled.
    """
    feature_toggles: dict[str, bool] = field(default_factory=dict)
    execution_settings: dict[str, Any] = field(default_factory=dict)
    risk_settings: dict[str, Any] = field(default_factory=dict)
    study_settings: dict[str, Any] = field(default_factory=dict)
    model_selection: list[str] = field(default_factory=list)
    asset: str = "EURUSD"
    timeframe: str = "H1"
    date_range: tuple[str | None, str | None] = (None, None)
    model_param_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_toggle_dict(self) -> dict[str, bool]:
        """Return {use_adx: True, use_rsi: False, ...} for injection into params."""
        return dict(self.feature_toggles)

    def is_enabled(self, feature_key: str) -> bool:
        """Check if a given feature toggle is True."""
        return bool(self.feature_toggles.get(feature_key, False))

    @classmethod
    def from_api_overrides(
        cls,
        config_overrides: dict[str, Any],
    ) -> "UserFixedConfig":
        """Build a UserFixedConfig from the API's config_overrides dictionary.

        The frontend sends keys like "use_adx", "eval_use_trading_costs",
        "sizing_method", "max_drawdown_pct", etc.
        """
        feature_toggles: dict[str, bool] = {}
        execution: dict[str, Any] = {}
        risk: dict[str, Any] = {}
        study: dict[str, Any] = {}

        # Feature toggle keys (match backend naming convention)
        _FEATURE_TOGGLE_KEYS = frozenset([
            "use_adx", "use_atr", "use_bbands", "use_ema", "use_sma",
            "use_rsi", "use_macd", "use_stoch", "use_sar", "use_donchian",
            "use_mtf_ma", "use_mtf_alignment", "use_mtf_align",
            "use_macd_atr_ratio", "use_triple_confirm", "use_trend_confirm",
            "use_vol_managed_mom", "use_vm_mom", "use_squeeze_breakout",
            "use_squeeze_expansion", "use_atr_channel_breakout",
            "use_ext_atr_low_adx", "use_reentry_mom", "use_slope_diff",
            "use_rv_features", "use_indicator_states", "use_fracdiff",
            "use_crossover_bins", "use_ma_spread", "use_price_ma_z",
            "use_news", "news_event_flags", "llm_sentiment_enabled",
        ])

        _EXECUTION_KEYS = frozenset([
            "eval_use_trading_costs", "sizing_method",
            "initial_equity", "max_leverage", "slip_norm_bps",
            "kelly_fraction", "kelly_min_trades", "risk_fraction",
            "atr_risk_pct", "atr_sl_mult",
        ])

        _RISK_KEYS = frozenset([
            "max_drawdown_pct", "max_consecutive_losses", "daily_loss_limit_pct",
        ])

        _STUDY_KEYS = frozenset([
            "train_months", "test_months", "hpo_intensity", "n_trials",
            "repeats", "seed", "optuna_direction", "max_hpo_duration_minutes",
            "hpo_sampler", "hpo_two_phase", "hpo_mode", "period_unit",
            "phase1_sampler", "phase1_trials", "phase2_trials",
            "phase2_top_n", "dynamic_hpo_trials",
            "wfo_train_periods", "wfo_test_periods",
            "target_active_rate", "target_coverage", "confidence_threshold",
            "calibrate_method", "label_threshold",
            "use_triple_barrier", "tb_pt_mult", "tb_sl_mult",
            "tb_max_holding", "tb_neutral_zone",
        ])

        for key, value in config_overrides.items():
            if key in _FEATURE_TOGGLE_KEYS:
                feature_toggles[key] = bool(value)
            elif key in _EXECUTION_KEYS:
                execution[key] = value
            elif key in _RISK_KEYS:
                risk[key] = value
            elif key in _STUDY_KEYS:
                study[key] = value

        # Per-param HPO range keys: model__param__hpo_range → [min, max]
        # Aliased through HYPERPARAM_ALIASES to match sampler's internal key convention.
        model_param_ranges: dict[str, tuple[float, float]] = {}
        from pipeline.hyperparam_aliases import HYPERPARAM_ALIASES as _ALIASES

        for key, value in config_overrides.items():
            if not (isinstance(key, str) and key.endswith("__hpo_range")):
                continue
            if not isinstance(value, list) or len(value) != 2:
                continue
            base_key = key[:-11]  # strip __hpo_range (11 chars)
            parts = base_key.split("__", 1)
            if len(parts) != 2:
                continue
            model, param = parts
            aliases = _ALIASES.get(model, {})
            internal_key = aliases.get(param, f"{model}_{param}")
            try:
                lo, hi = float(value[0]), float(value[1])
                model_param_ranges[internal_key] = (lo, hi)
            except (ValueError, TypeError):
                continue

        return cls(
            feature_toggles=feature_toggles,
            execution_settings=execution,
            risk_settings=risk,
            study_settings=study,
            model_param_ranges=model_param_ranges,
        )


# Category A: Boolean feature toggles — locked by user, never sampled by Optuna.
# These keys use the backend snake_case naming convention (as sent via
# config_overrides from the frontend API request).
USER_LOCKED_BOOLEAN_FEATURES = frozenset([
    "use_adx",
    "use_atr",
    "use_bbands",
    "use_ema",
    "use_sma",
    "use_rsi",
    "use_macd",
    "use_stoch",
    "use_sar",
    "use_donchian",
    "use_mtf_ma",
    "use_mtf_alignment",
    "use_mtf_align",
    "use_macd_atr_ratio",
    "use_triple_confirm",
    "use_trend_confirm",
    "use_vol_managed_mom",
    "use_vm_mom",
    "use_squeeze_breakout",
    "use_squeeze_expansion",
    "use_atr_channel_breakout",
    "use_ext_atr_low_adx",
    "use_reentry_mom",
    "use_slope_diff",
    "use_rv_features",
    "use_indicator_states",
    "use_fracdiff",
    "use_crossover_bins",
    "use_ma_spread",
    "use_price_ma_z",
    "use_news",
])

# Category A2: Execution/risk booleans — locked by user, never sampled.
USER_LOCKED_EXECUTION_BOOLEANS = frozenset([
    "eval_use_trading_costs",
    "use_triple_barrier",
])

# Category B: Continuous thresholds — sampled by Optuna IF parent toggle is ON.
THRESHOLD_PARAMS = frozenset([
    "fracdiff_d",
    "tb_pt_mult",
    "tb_sl_mult",
    "tb_max_holding",
    "tb_neutral_zone",
    "lags",
    "lag_depth",
    "label_threshold",
    "confidence_threshold",
    "llm_weight",
])

# Feature toggles from the frontend that map 1:1 to backend keys.
# (Same keys — the frontend sends camelCase keys and tasks.py converts them.)
FEATURE_TOGGLE_MAP: dict[str, str] = {k: k for k in USER_LOCKED_BOOLEAN_FEATURES}
