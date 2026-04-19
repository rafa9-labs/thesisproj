"""
Streamlit controls — icon nav bar at top, tab content in main body.

Vertical layout, helper text under selectors, backtest presets.
"""

import streamlit as st

# ── Tab definitions ─────────────────────────────────────────────────

TABS = [
    ("\U0001F4CA", "Data", "Select currency pair and preview data"),
    ("\U0001F916", "Model", "Choose ML model and hyperparameters"),
    ("\U0001F4C8", "Features", "Configure indicators and feature engineering"),
    ("\U0001F3AF", "Labels", "Set label thresholds and triple barrier"),
    ("\u2699\uFE0F", "HPO", "Walk-forward windows, coverage, and costs"),
    ("\U0001F4CB", "Results", "View backtest results and diagnostics"),
]

# ── Default parameter values ────────────────────────────────────────

DEFAULTS = {
    "model_type": "logistic",
    "calibrate_method": "sigmoid",
    "logit_C": 1.0, "logit_solver": "lbfgs", "logit_penalty": "l2",
    "logit_max_iter": 500, "logit_tol": 0.0001,
    "train_months": 36, "test_months": 1,
    "n_trials": 10, "seed": 42,
    "optuna_direction": "maximize",
    "eval_use_trading_costs": True, "slip_norm_bps": 0.25,
    "target_active_rate": 0.15, "target_coverage": 0.15,
    "confidence_threshold": 0.8,
    "use_triple_barrier": True,
    "tb_pt_mult": 2.0, "tb_sl_mult": 2.0,
    "tb_neutral_zone": 0.5, "tb_max_holding": 36,
    "label_threshold": 0.0005,
    "lags": 14, "lag_depth": 1,
    "use_fracdiff": True, "fracdiff_d": 0.4,
    "use_adx": True, "use_atr": True, "use_bbands": True,
    "use_ema": True, "use_sma": True, "use_rsi": True,
    "use_macd": True, "use_stoch": False, "use_sar": False,
    "use_donchian": True, "use_crossover_bins": True,
    "use_ma_spread": False, "use_price_ma_z": True,
    "use_indicator_states": False,
    "use_mtf_ma": True, "use_mtf_alignment": True,
    "use_mtf_align": True, "use_macd_atr_ratio": True,
    "use_triple_confirm": True, "use_trend_confirm": True,
    "use_vol_managed_mom": True, "use_vm_mom": True,
    "use_squeeze_breakout": False, "use_squeeze_expansion": False,
    "use_atr_channel_breakout": False, "use_ext_atr_low_adx": False,
    "use_reentry_mom": False, "use_slope_diff": False,
    "use_rv_features": False,
}

# ── Feature presets ─────────────────────────────────────────────────

_MINIMAL_INDICATORS = {
    "use_adx": True, "use_atr": True, "use_bbands": False,
    "use_ema": True, "use_sma": False, "use_rsi": True,
    "use_macd": False, "use_stoch": False, "use_sar": False,
    "use_donchian": False, "use_crossover_bins": False,
    "use_ma_spread": False, "use_price_ma_z": False,
    "use_indicator_states": False,
    "use_mtf_ma": False, "use_mtf_alignment": False,
    "use_mtf_align": False, "use_macd_atr_ratio": False,
    "use_triple_confirm": False, "use_trend_confirm": False,
    "use_vol_managed_mom": False, "use_vm_mom": False,
    "use_squeeze_breakout": False, "use_squeeze_expansion": False,
    "use_atr_channel_breakout": False, "use_ext_atr_low_adx": False,
    "use_reentry_mom": False, "use_slope_diff": False,
    "use_rv_features": False,
    "use_fracdiff": True, "fracdiff_d": 0.4,
    "lags": 8, "lag_depth": 1,
}

_ALL_INDICATORS = {
    "use_adx": True, "use_atr": True, "use_bbands": True,
    "use_ema": True, "use_sma": True, "use_rsi": True,
    "use_macd": True, "use_stoch": False, "use_sar": False,
    "use_donchian": True, "use_crossover_bins": True,
    "use_ma_spread": False, "use_price_ma_z": True,
    "use_indicator_states": False,
    "use_mtf_ma": True, "use_mtf_alignment": True,
    "use_mtf_align": True, "use_macd_atr_ratio": True,
    "use_triple_confirm": True, "use_trend_confirm": True,
    "use_vol_managed_mom": True, "use_vm_mom": True,
    "use_squeeze_breakout": False, "use_squeeze_expansion": False,
    "use_atr_channel_breakout": False, "use_ext_atr_low_adx": False,
    "use_reentry_mom": False, "use_slope_diff": False,
    "use_rv_features": False,
    "use_fracdiff": True, "fracdiff_d": 0.4,
    "lags": 14, "lag_depth": 1,
}

BACKTEST_PRESETS = [
    {
        "label": "\U0001F680 Quick Logistic (Minimal)",
        "desc": "Logistic regression with 4 core indicators. Fast baseline — ~2 min.",
        "model_type": "logistic",
        "n_trials": 5, "train_months": 24, "test_months": 1,
        "features": _MINIMAL_INDICATORS,
    },
    {
        "label": "\U0001F916 Quick CNN (Minimal)",
        "desc": "1D-CNN with minimal features. Tests deep learning baseline — ~10 min CPU.",
        "model_type": "cnn",
        "n_trials": 5, "train_months": 24, "test_months": 1,
        "features": _MINIMAL_INDICATORS,
    },
    {
        "label": "\U0001F4CA Full Logistic (All Features)",
        "desc": "Logistic with all indicators enabled. Maximum feature coverage — ~5 min.",
        "model_type": "logistic",
        "n_trials": 10, "train_months": 36, "test_months": 1,
        "features": _ALL_INDICATORS,
    },
    {
        "label": "\U0001F333 Full XGBoost (All Features)",
        "desc": "XGBoost with all indicators. Best tree-based benchmark — ~8 min.",
        "model_type": "xgboost",
        "n_trials": 10, "train_months": 36, "test_months": 1,
        "features": _ALL_INDICATORS,
    },
]


def _apply_preset(preset: dict):
    for key in (
        "model_type", "n_trials", "train_months", "test_months",
    ):
        st.session_state[key] = preset[key]
    for key, val in preset["features"].items():
        st.session_state[key] = val


def get_all_params() -> dict:
    params = {}
    for key, default in DEFAULTS.items():
        params[key] = st.session_state.get(key, default)
    return params


# ── Navigation Bar ──────────────────────────────────────────────────

def render_nav_bar() -> int:
    if "_nav_tab" not in st.session_state:
        st.session_state["_nav_tab"] = 0

    header_col, run_col = st.columns([4, 1])
    with header_col:
        st.markdown("# \U0001F4C8 FX ML Backtester")
    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("\U0001F680 Run Backtest", type="primary", width="stretch")

    cols = st.columns(len(TABS))
    for i, (icon, name, tooltip) in enumerate(TABS):
        with cols[i]:
            is_active = (st.session_state["_nav_tab"] == i)
            label = f"{icon} **{name}**" if is_active else f"{icon} {name}"
            if st.button(label, key=f"_nav_btn_{i}", width="stretch"):
                st.session_state["_nav_tab"] = i
                st.rerun()

    active_name = TABS[st.session_state["_nav_tab"]][1]
    st.markdown(f"**\u25b8 {active_name}**")
    st.divider()

    if run_clicked:
        st.session_state["_run_backtest"] = True
        st.session_state["_nav_tab"] = 5

    return st.session_state["_nav_tab"]


# ── Tab Content Router ──────────────────────────────────────────────

def render_tab_content(tab_index: int):
    if tab_index == 0:
        _render_data_tab()
    elif tab_index == 1:
        _render_model_tab()
    elif tab_index == 2:
        _render_features_tab()
    elif tab_index == 3:
        _render_labels_tab()
    elif tab_index == 4:
        _render_hpo_tab()
    elif tab_index == 5:
        _render_results_tab()


# ── Tab 0: Data Source ──────────────────────────────────────────────

def _render_data_tab():
    from ui.state import DATA_FILES, AppState

    st.subheader("\U0001F4CA Data Source")
    st.caption("Select a currency pair and timeframe. Each pair has ~10 years of OHLC data from OANDA.")
    st.markdown("")

    data_key = st.selectbox(
        "Currency Pair / Timeframe",
        options=list(DATA_FILES.keys()),
        index=0, key="data_key",
    )
    data_info = DATA_FILES[data_key]
    st.caption(f"File: `{data_info['path']}`  |  Timeframe: {data_info['tf']}")

    st.markdown("")

    if st.button("\U0001F4C2 Preview Data", key="preview_data_btn", width="stretch"):
        try:
            df = AppState.load_csv_data(data_info["path"])
            st.session_state["preview_df"] = df.head(20)
            st.session_state["preview_rows"] = len(df)
            st.session_state["preview_cols"] = list(df.columns)
        except Exception as e:
            st.error(f"Load error: {e}")

    if "preview_rows" in st.session_state:
        st.success(
            f"{st.session_state['preview_rows']:,} rows \u00d7 "
            f"{len(st.session_state.get('preview_cols', []))} cols"
        )

    st.markdown("")

    if "preview_df" in st.session_state:
        st.dataframe(st.session_state["preview_df"], width="stretch", height=350)
    else:
        st.info("Click **Preview Data** to see the first 20 rows.")


# ── Tab 1: Model ────────────────────────────────────────────────────

def _render_model_tab():
    from ui.state import AVAILABLE_MODELS

    st.subheader("\U0001F916 Model Configuration")
    st.caption("Choose the ML architecture and calibration method. Deep models (CNN/LSTM/Transformer) are auto-tuned via Optuna.")
    st.markdown("")

    model_type = st.selectbox(
        "Model Type",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index("logistic") if "logistic" in AVAILABLE_MODELS else 0,
        key="model_type",
    )
    st.caption(
        "Logistic/RF/XGBoost/SVM are fast on CPU. CNN/LSTM/Transformer benefit from GPU. "
        "Ensembles combine multiple architectures."
    )

    st.markdown("")

    st.selectbox(
        "Calibration Method",
        options=["sigmoid", "isotonic"],
        index=0, key="calibrate_method",
    )
    st.caption(
        "Sigmoid (Platt scaling) is faster and works well for most models. "
        "Isotonic is non-parametric and can handle skewed probability distributions."
    )

    st.markdown("")

    mt = model_type.lower() if isinstance(model_type, str) else str(model_type).lower()
    _GPU_HEAVY = {"cnn", "lstm", "transformer", "dqn"}
    if mt in _GPU_HEAVY:
        try:
            from pipeline.runtime import gpu_status
            gs = gpu_status()
            if not gs["available"]:
                st.warning(
                    f"\u26a0\ufe0f **{mt.upper()}** runs significantly faster on GPU. "
                    f"No GPU detected \u2014 running on CPU (may take 10-70+ minutes). "
                    f"Consider running via WSL with GPU enabled (`run_smoke_gpu.bat`)."
                )
            else:
                st.success(f"\U0001F3AE GPU acceleration active: {', '.join(gs['devices'])}")
        except Exception:
            pass
        st.markdown("")

    if mt == "logistic":
        st.markdown("#### Logistic Regression Hyperparameters")
        st.caption("Fine-tune the logistic regression solver, penalty, and regularization strength.")
        st.markdown("")

        st.selectbox(
            "Solver",
            options=["lbfgs", "newton-cg", "sag", "saga", "liblinear"],
            index=0, key="logit_solver",
        )
        st.caption("lbfgs is the default and works well for most cases. Use saga for L1 or elasticnet penalties.")

        st.markdown("")

        st.selectbox(
            "Penalty",
            options=["l2", "l1", "elasticnet", "none"],
            index=0, key="logit_penalty",
        )
        st.caption("L2 (Ridge) is standard. L1 (Lasso) can zero out features. Elasticnet combines both.")

        st.markdown("")

        st.number_input(
            "C (regularization inverse)",
            value=1.0, min_value=0.001, max_value=50000.0,
            step=10.0, format="%.3f", key="logit_C",
        )
        st.caption("Smaller C = stronger regularization (simpler model). Larger C = weaker regularization.")

        st.markdown("")

        st.number_input(
            "Max Iterations",
            value=500, min_value=50, max_value=5000, step=50,
            key="logit_max_iter",
        )
        st.caption("Increase if the solver doesn't converge (check warnings in logs).")

        st.markdown("")

        st.number_input(
            "Tolerance",
            value=0.0001, min_value=1e-6, max_value=0.1,
            step=0.0001, format="%.6f", key="logit_tol",
        )
        st.caption("Stopping criterion. Smaller values = more precise but slower convergence.")

    elif mt in ("cnn", "lstm", "transformer"):
        st.info(
            f"Deep model (**{mt.upper()}**) hyperparameters are tuned automatically via Optuna HPO. "
            f"No manual configuration needed \u2014 the tuner searches learning rate, layers, dropout, etc."
        )
    elif mt == "xgboost":
        st.info(
            "XGBoost hyperparameters are tuned automatically via Optuna HPO "
            "(max_depth, learning_rate, subsample, colsample, etc.)."
        )
    elif mt == "random_forest":
        st.info("Random Forest uses sensible defaults. The number of trees and depth are auto-configured.")
    elif mt == "svm":
        st.info("SVM uses RBF kernel by default with probability calibration. Tuned via Optuna.")
    elif mt == "dqn":
        st.info("Dueling DQN agent. Network architecture and RL hyperparameters are auto-tuned.")


# ── Tab 2: Features & Indicators ────────────────────────────────────

def _render_features_tab():
    st.subheader("\U0001F4C8 Indicators & Feature Engineering")
    st.caption("Toggle technical indicators on/off. Fewer indicators = faster training. More = richer signal but risk of overfitting.")
    st.markdown("")

    st.markdown("#### Core Indicators")
    st.caption("Standard technical analysis indicators. ADX, ATR, EMA and RSI are recommended minimums.")
    st.markdown("")

    cols = st.columns(3)
    indicators_core = [
        ("ADX", "use_adx", True, "Average Directional Index \u2014 trend strength"),
        ("ATR", "use_atr", True, "Average True Range \u2014 volatility measure"),
        ("EMA", "use_ema", True, "Exponential Moving Average \u2014 recent price trend"),
        ("SMA", "use_sma", True, "Simple Moving Average \u2014 smoothed price baseline"),
        ("RSI", "use_rsi", True, "Relative Strength Index \u2014 momentum oscillator"),
        ("MACD", "use_macd", True, "Moving Average Convergence/Divergence \u2014 trend momentum"),
        ("Bollinger Bands", "use_bbands", True, "Volatility envelope around SMA"),
        ("Donchian", "use_donchian", True, "Channel based on N-period highs/lows"),
        ("Stochastic", "use_stoch", False, "%K/%D oscillator \u2014 overbought/oversold"),
    ]
    for i, (label, key, default, desc) in enumerate(indicators_core):
        with cols[i % 3]:
            st.checkbox(label, value=default, key=key)
            st.caption(desc)

    st.divider()

    st.markdown("#### Feature Engineering")
    st.caption("Advanced feature transforms applied to price data before model training.")
    st.markdown("")

    st.checkbox("Fractional Differentiation (FracDiff)", value=True, key="use_fracdiff")
    st.caption(
        "Makes price series stationary while preserving memory. "
        "d=0 is raw price, d=1 is returns. 0.3\u20130.5 is typical for FX."
    )

    st.markdown("")

    if st.session_state.get("use_fracdiff", True):
        st.number_input("fracdiff d", value=0.4, min_value=0.0, max_value=1.0, step=0.05, key="fracdiff_d")
        st.caption("Lower d preserves more trend memory. Higher d makes data more stationary.")
        st.markdown("")

    st.checkbox("Crossover Bins", value=True, key="use_crossover_bins")
    st.caption("Binary features that flag when short MA crosses above/below long MA.")
    st.markdown("")

    st.checkbox("Price-MA Z-Score", value=True, key="use_price_ma_z")
    st.caption("Normalized distance between price and moving average in standard deviations.")
    st.markdown("")

    st.divider()

    st.markdown("#### Lag Features")
    st.caption("Lagged returns capture autocorrelation. More lags = longer lookback but more features.")
    st.markdown("")

    st.number_input("Number of Lags", value=14, min_value=1, max_value=60, step=1, key="lags")
    st.caption("14 lags on H1 = lookback of 14 hours. Increase for longer-term patterns.")
    st.markdown("")

    st.number_input(
        "Lag Depth",
        value=1, min_value=1, max_value=3, step=1, key="lag_depth",
    )
    st.caption("Number of lag differentiation orders. 1 = single lag. 2-3 adds higher-order lags.")
    st.markdown("")

    st.divider()

    with st.expander("\U0001F527 Advanced Toggles"):
        st.caption("Specialized features for specific market conditions. Enable selectively.")
        st.markdown("")

        adv_cols = st.columns(3)
        advanced = [
            ("MTF MA", "use_mtf_ma", True, "Multi-timeframe moving averages"),
            ("MTF Alignment", "use_mtf_alignment", True, "Multi-TF trend alignment score"),
            ("MACD/ATR Ratio", "use_macd_atr_ratio", True, "MACD normalized by ATR"),
            ("Triple Confirm", "use_triple_confirm", True, "ADX + trend + momentum agreement"),
            ("Trend Confirm", "use_trend_confirm", True, "Trend direction confirmation"),
            ("Vol-Managed Mom", "use_vol_managed_mom", True, "Momentum adjusted for volatility"),
            ("MA Spread", "use_ma_spread", False, "Distance between fast/slow MA"),
            ("Slope Diff", "use_slope_diff", False, "Rate of change between MA slopes"),
            ("SAR", "use_sar", False, "Parabolic SAR stop-and-reverse"),
            ("Squeeze Breakout", "use_squeeze_breakout", False, "TTM Squeeze breakout signal"),
            ("Squeeze Expansion", "use_squeeze_expansion", False, "TTM Squeeze expansion phase"),
            ("ATR Channel Break", "use_atr_channel_breakout", False, "Breakout from ATR channel"),
            ("Ext ATR Low ADX", "use_ext_atr_low_adx", False, "Extended ATR in low-AD regime"),
            ("Re-entry Momentum", "use_reentry_mom", False, "Re-entry after pullback signal"),
            ("RV Features", "use_rv_features", False, "Realized volatility features"),
            ("Indicator States", "use_indicator_states", False, "Binary indicator regime states"),
        ]
        for i, (label, key, default, desc) in enumerate(advanced):
            with adv_cols[i % 3]:
                st.checkbox(label, value=default, key=key)
                st.caption(desc)


# ── Tab 3: Labels & Triple Barrier ──────────────────────────────────

def _render_labels_tab():
    st.subheader("\U0001F3AF Labels & Triple Barrier")
    st.caption(
        "Labels define what the model learns to predict. The label threshold determines "
        "the minimum price move to classify as a directional signal."
    )
    st.markdown("")

    st.number_input(
        "Label Threshold",
        value=0.0005, min_value=0.0001, max_value=0.005,
        step=0.0001, format="%.6f", key="label_threshold",
    )
    st.caption(
        "0.0005 = 0.05% move (5 pips on EURUSD). Lower = more signals but noisier. "
        "Higher = fewer, cleaner signals."
    )
    st.markdown("")

    st.checkbox("Use Triple Barrier", value=True, key="use_triple_barrier")
    st.caption(
        "Triple Barrier Method assigns labels based on which price level is hit first: "
        "take-profit, stop-loss, or time timeout."
    )
    st.markdown("")

    if st.session_state.get("use_triple_barrier", True):
        st.markdown("#### Triple Barrier Multipliers")
        st.caption("All distances are multiples of ATR (Average True Range), so they adapt to current volatility.")
        st.markdown("")

        st.number_input(
            "Take-Profit Multiplier (PT)",
            value=2.0, min_value=0.5, max_value=4.0, step=0.25, key="tb_pt_mult",
        )
        st.caption("Higher PT = wider profit target = fewer but larger winners.")
        st.markdown("")

        st.number_input(
            "Stop-Loss Multiplier (SL)",
            value=2.0, min_value=0.5, max_value=4.0, step=0.25, key="tb_sl_mult",
        )
        st.caption("Higher SL = wider stop = more room for price to breathe before stopping out.")
        st.markdown("")

        st.number_input(
            "Max Holding Bars",
            value=36, min_value=4, max_value=72, step=4, key="tb_max_holding",
        )
        st.caption("Maximum bars before forced exit. On H1, 36 = 1.5 days. Assigns 'timeout' label if neither PT nor SL is hit.")
        st.markdown("")

        st.number_input(
            "Neutral Zone",
            value=0.5, min_value=0.0, max_value=2.0, step=0.25, key="tb_neutral_zone",
        )
        st.caption("Dead zone in ATR multiples around entry. Price moves within this zone are labeled as neutral (class 0).")


# ── Tab 4: HPO & CV ─────────────────────────────────────────────────

def _render_hpo_tab():
    st.subheader("\u2699\ufe0f HPO & Walk-Forward Configuration")
    st.caption(
        "Configure hyperparameter optimization (HPO) via Optuna, walk-forward window sizes, "
        "and trading cost assumptions."
    )
    st.markdown("")

    st.markdown("#### HPO Settings")
    st.caption("Optuna searches for the best hyperparameters using Bayesian optimization.")
    st.markdown("")

    st.number_input("HPO Trials", value=10, min_value=2, max_value=100, step=1, key="n_trials")
    st.caption("5\u201310 for quick tests. 50\u2013100 for production runs. More trials = better params but longer runtime.")
    st.markdown("")

    st.selectbox("Optimization Direction", options=["maximize", "minimize"], index=0, key="optuna_direction")
    st.caption("Maximize = higher Sharpe is better. Almost always use 'maximize'.")
    st.markdown("")

    st.number_input("Random Seed", value=42, min_value=0, max_value=9999, key="seed")
    st.caption("Ensures reproducibility. Same seed + same params = same results.")
    st.markdown("")

    st.divider()

    st.markdown("#### Walk-Forward Windows")
    st.caption("The model trains on N months, then predicts the next M months. This rolls forward monthly.")
    st.markdown("")

    st.number_input("Train Months", value=36, min_value=6, max_value=60, step=1, key="train_months")
    st.caption("36 months (3 years) is typical. Shorter windows adapt faster to regime changes.")
    st.markdown("")

    st.number_input("Test Months", value=1, min_value=1, max_value=6, step=1, key="test_months")
    st.caption("1 month is standard walk-forward. Increase for longer evaluation windows per fold.")
    st.markdown("")

    st.divider()

    st.markdown("#### Coverage & Confidence")
    st.caption("Control how aggressively the strategy takes trades.")
    st.markdown("")

    st.number_input(
        "Target Active Rate",
        value=0.15, min_value=0.05, max_value=0.30, step=0.01, format="%.3f",
        key="target_active_rate",
    )
    st.caption("Fraction of bars where the strategy opens a position. 0.15 = ~15% of time in market.")
    st.markdown("")

    st.number_input(
        "Target Coverage",
        value=0.15, min_value=0.05, max_value=0.30, step=0.01, format="%.3f",
        key="target_coverage",
    )
    st.caption("Should match target_active_rate. Used to calibrate the confidence threshold.")
    st.markdown("")

    st.number_input(
        "Confidence Threshold",
        value=0.8, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        key="confidence_threshold",
    )
    st.caption("Minimum predicted probability to trigger a trade. 0.0 = auto-calibrated. 0.8 = only high-confidence signals.")
    st.markdown("")

    st.divider()

    st.markdown("#### Trading Costs")
    st.caption("Realistic cost modeling is critical. Spread + slippage can erode returns significantly.")
    st.markdown("")

    st.checkbox("Include Trading Costs", value=True, key="eval_use_trading_costs")
    st.caption("Disabling costs shows gross returns. Enable for realistic net performance.")
    st.markdown("")

    if st.session_state.get("eval_use_trading_costs", True):
        st.number_input("Slippage (bps)", value=0.25, min_value=0.0, max_value=2.0, step=0.05, key="slip_norm_bps")
        st.caption("0.25 bps is typical for liquid FX pairs like EURUSD. Increase for exotic pairs.")


# ── Tab 5: Results ──────────────────────────────────────────────────

def _render_results_tab():
    if "_run_backtest" in st.session_state and st.session_state["_run_backtest"]:
        st.session_state["_run_backtest"] = False
        st.info("Phase 3 will execute the backtest and show results here.")
    elif "backtest_results" in st.session_state:
        from ui.results import render_results_tab as _rrt
        _rrt()
    else:
        st.subheader("\U0001F4CB Results")
        st.info("Run a backtest first (click \U0001F680 Run Backtest), or load previous results below.")

        st.markdown("")
        _render_presets()

        st.divider()

        _render_previous_runs()


def _render_presets():
    st.markdown("#### \u26a1 Quick-Start Presets")
    st.caption("Click a preset to configure all settings automatically, then hit Run Backtest.")
    st.markdown("")

    for i, preset in enumerate(BACKTEST_PRESETS):
        with st.container():
            pcol1, pcol2 = st.columns([3, 1])
            with pcol1:
                st.markdown(f"**{preset['label']}**")
                st.caption(preset["desc"])
            with pcol2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(
                    f"Apply",
                    key=f"_preset_btn_{i}",
                    type="primary" if i == 0 else "secondary",
                    width="stretch",
                ):
                    _apply_preset(preset)
                    st.toast(f"Applied preset: {preset['label']}", icon="\u2705")
                    st.rerun()
        if i < len(BACKTEST_PRESETS) - 1:
            st.markdown("")


def _render_previous_runs():
    import json
    from pathlib import Path

    st.markdown("#### Load Previous Run")
    st.caption("Load results from a previous Optuna or HPO run stored on disk.")
    st.markdown("")

    optuna_dir = Path("optuna_runs")
    hpo_dir = Path("hpo")
    runs = []
    if optuna_dir.exists():
        for d in sorted(optuna_dir.iterdir(), reverse=True):
            if d.is_dir():
                runs.append(("optuna", d.name, d))
    if hpo_dir.exists():
        for f in sorted(hpo_dir.glob("*.json"), reverse=True):
            runs.append(("hpo", f.name, f))

    if runs:
        run_labels = [f"{r[0]}: {r[1]}" for r in runs]
        selected = st.selectbox("Previous Runs", options=run_labels, index=0)
        st.caption(f"Found {len(runs)} previous runs on disk.")
        if st.button("Load Selected", key="load_prev_btn"):
            idx = run_labels.index(selected)
            rtype, rname, rpath = runs[idx]
            data = {"source": str(rpath)}
            best_file = rpath / "best_config.json" if rpath.is_dir() else None
            if best_file and best_file.exists():
                with open(best_file) as f:
                    data["best_config"] = json.load(f)
            elif rtype == "hpo" and rpath.suffix == ".json":
                with open(rpath) as f:
                    data["best_config"] = json.load(f)
            pi_file = rpath / "param_importances.json" if rpath.is_dir() else None
            if pi_file and pi_file.exists():
                with open(pi_file) as f:
                    data["param_importances"] = json.load(f)
            if data.get("best_config") or data.get("param_importances"):
                st.session_state["backtest_results"] = data
                st.rerun()
            else:
                st.warning("No usable data found in that run.")
    else:
        st.caption("No previous results found.")
