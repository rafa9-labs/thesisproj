"""
Streamlit controls — icon nav bar at top, tab content in main body.

Phase 2: Full tab content with all controls.
"""

import streamlit as st

# ── Tab definitions ─────────────────────────────────────────────────

TABS = [
    ("📊", "Data", "Select currency pair and preview data"),
    ("🤖", "Model", "Choose ML model and hyperparameters"),
    ("📈", "Features", "Configure indicators and feature engineering"),
    ("🎯", "Labels", "Set label thresholds and triple barrier"),
    ("⚙️", "HPO", "Walk-forward windows, coverage, and costs"),
    ("📋", "Results", "View backtest results and diagnostics"),
]

# ── Default parameter values ────────────────────────────────────────

DEFAULTS = {
    # Model
    "model_type": "logistic",
    "calibrate_method": "sigmoid",
    # Logistic HPs
    "logit_C": 1.0, "logit_solver": "lbfgs", "logit_penalty": "l2",
    "logit_max_iter": 500, "logit_tol": 0.0001,
    # Walk-forward
    "train_months": 36, "test_months": 1,
    "n_trials": 10, "seed": 42,
    "optuna_direction": "maximize",
    # Trading costs
    "eval_use_trading_costs": True, "slip_norm_bps": 0.25,
    # Coverage
    "target_active_rate": 0.15, "target_coverage": 0.15,
    # Confidence
    "confidence_threshold": 0.8,
    # Triple Barrier
    "use_triple_barrier": True,
    "tb_pt_mult": 2.0, "tb_sl_mult": 2.0,
    "tb_neutral_zone": 0.5, "tb_max_holding": 36,
    # Labels
    "label_threshold": 0.0005,
    # Lags
    "lags": 14, "lag_depth": 1,
    # Fracdiff
    "use_fracdiff": True, "fracdiff_d": 0.4,
    # Indicator toggles
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


def get_all_params() -> dict:
    """Collect all current UI parameter values from session_state."""
    params = {}
    for key, default in DEFAULTS.items():
        params[key] = st.session_state.get(key, default)
    return params


# ── Navigation Bar ──────────────────────────────────────────────────

def render_nav_bar() -> int:
    """Render the icon navigation bar at the top of the main area.

    Returns the index of the currently selected tab.
    """
    # Initialize tab state
    if "_nav_tab" not in st.session_state:
        st.session_state["_nav_tab"] = 0

    # Header row: title + run button
    header_col, run_col = st.columns([4, 1])
    with header_col:
        st.markdown("# 📈 FX ML Backtester")
    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    # Nav bar: icon buttons
    cols = st.columns(len(TABS))
    for i, (icon, name, tooltip) in enumerate(TABS):
        with cols[i]:
            is_active = (st.session_state["_nav_tab"] == i)
            label = f"{icon} **{name}**" if is_active else f"{icon} {name}"
            if st.button(
                label,
                key=f"_nav_btn_{i}",
                use_container_width=True,
            ):
                st.session_state["_nav_tab"] = i
                st.rerun()

    # Active tab indicator line
    active_name = TABS[st.session_state["_nav_tab"]][1]
    st.markdown(f"**▸ {active_name}**")
    st.divider()

    # Handle run button
    if run_clicked:
        st.session_state["_run_backtest"] = True
        st.session_state["_nav_tab"] = 5  # Switch to Results tab

    return st.session_state["_nav_tab"]


# ── Tab Content Router ──────────────────────────────────────────────

def render_tab_content(tab_index: int):
    """Render the selected tab's content in the main area."""
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

    st.subheader("📊 Data Source")

    c1, c2 = st.columns([1, 2])
    with c1:
        data_key = st.selectbox(
            "Currency Pair / Timeframe",
            options=list(DATA_FILES.keys()),
            index=0, key="data_key",
        )
        data_info = DATA_FILES[data_key]
        st.caption(f"File: `{data_info['path']}`  |  TF: {data_info['tf']}")

        if st.button("📂 Preview Data", key="preview_data_btn", use_container_width=True):
            try:
                df = AppState.load_csv_data(data_info["path"])
                st.session_state["preview_df"] = df.head(20)
                st.session_state["preview_rows"] = len(df)
                st.session_state["preview_cols"] = list(df.columns)
            except Exception as e:
                st.error(f"Load error: {e}")

        if "preview_rows" in st.session_state:
            st.success(
                f"{st.session_state['preview_rows']:,} rows × "
                f"{len(st.session_state.get('preview_cols', []))} cols"
            )

    with c2:
        if "preview_df" in st.session_state:
            st.dataframe(st.session_state["preview_df"], use_container_width=True, height=350)
        else:
            st.info("Click **Preview Data** to see the first 20 rows.")


# ── Tab 1: Model ────────────────────────────────────────────────────

def _render_model_tab():
    from ui.state import AVAILABLE_MODELS

    st.subheader("🤖 Model Configuration")

    c1, c2 = st.columns(2)
    with c1:
        model_type = st.selectbox(
            "Model Type", options=AVAILABLE_MODELS,
            index=0, key="model_type",
            help="Select the ML model architecture",
        )
    with c2:
        st.selectbox(
            "Calibration Method", options=["sigmoid", "isotonic"],
            index=0, key="calibrate_method",
            help="Probability calibration applied after model prediction",
        )

    # Model-specific HPs
    mt = model_type.lower() if isinstance(model_type, str) else str(model_type).lower()
    if mt == "logistic":
        st.markdown("#### Logistic Regression Hyperparameters")
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Solver", options=["lbfgs", "newton-cg", "sag", "saga", "liblinear"],
                          index=0, key="logit_solver")
            st.selectbox("Penalty", options=["l2", "l1", "elasticnet", "none"],
                          index=0, key="logit_penalty")
        with c2:
            st.number_input("C (regularization)", value=1.0, min_value=0.001,
                            max_value=50000.0, step=10.0, format="%.3f", key="logit_C")
            st.number_input("Max Iterations", value=500, min_value=50,
                            max_value=5000, step=50, key="logit_max_iter")
            st.number_input("Tolerance", value=0.0001, min_value=1e-6,
                            max_value=0.1, step=0.0001, format="%.6f", key="logit_tol")
    elif mt in ("cnn", "lstm", "transformer"):
        st.info(f"Deep model ({mt.upper()}) hyperparameters are tuned automatically via Optuna HPO.")
    elif mt == "xgboost":
        st.info("XGBoost hyperparameters are tuned automatically via Optuna HPO.")
    else:
        st.info(f"Model: {mt}")


# ── Tab 2: Features & Indicators ────────────────────────────────────

def _render_features_tab():
    st.subheader("📈 Indicators & Feature Engineering")

    # Core indicators
    st.markdown("#### Core Indicators")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("ADX", value=True, key="use_adx")
        st.checkbox("ATR", value=True, key="use_atr")
        st.checkbox("EMA", value=True, key="use_ema")
    with c2:
        st.checkbox("SMA", value=True, key="use_sma")
        st.checkbox("RSI", value=True, key="use_rsi")
        st.checkbox("MACD", value=True, key="use_macd")
    with c3:
        st.checkbox("Bollinger", value=True, key="use_bbands")
        st.checkbox("Donchian", value=True, key="use_donchian")
        st.checkbox("Stochastic", value=False, key="use_stoch")

    # Feature engineering
    st.markdown("#### Feature Engineering")
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox("FracDiff", value=True, key="use_fracdiff")
        if st.session_state.get("use_fracdiff", True):
            st.number_input("fracdiff d", value=0.4, min_value=0.0,
                           max_value=1.0, step=0.05, key="fracdiff_d")
    with c2:
        st.checkbox("Crossover Bins", value=True, key="use_crossover_bins")
        st.checkbox("Price-MA Z-Score", value=True, key="use_price_ma_z")

    # Lags
    st.markdown("#### Lag Features")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Lags", value=14, min_value=1, max_value=60, step=1, key="lags")
    with c2:
        st.number_input("Lag Depth", value=1, min_value=1, max_value=3, step=1, key="lag_depth",
                        help="Number of lag differentiation orders")

    # Advanced toggles
    with st.expander("🔧 Advanced Toggles"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.checkbox("MTF MA", value=True, key="use_mtf_ma")
            st.checkbox("MTF Alignment", value=True, key="use_mtf_alignment")
            st.checkbox("MACD/ATR Ratio", value=True, key="use_macd_atr_ratio")
        with c2:
            st.checkbox("Triple Confirm", value=True, key="use_triple_confirm")
            st.checkbox("Trend Confirm", value=True, key="use_trend_confirm")
            st.checkbox("Vol-Managed Mom", value=True, key="use_vol_managed_mom")
        with c3:
            st.checkbox("MA Spread", value=False, key="use_ma_spread")
            st.checkbox("Slope Diff", value=False, key="use_slope_diff")
            st.checkbox("SAR", value=False, key="use_sar")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.checkbox("Squeeze Breakout", value=False, key="use_squeeze_breakout")
            st.checkbox("Squeeze Expansion", value=False, key="use_squeeze_expansion")
        with c2:
            st.checkbox("ATR Channel Breakout", value=False, key="use_atr_channel_breakout")
            st.checkbox("Ext ATR Low ADX", value=False, key="use_ext_atr_low_adx")
        with c3:
            st.checkbox("Re-entry Momentum", value=False, key="use_reentry_mom")
            st.checkbox("RV Features", value=False, key="use_rv_features")
            st.checkbox("Indicator States", value=False, key="use_indicator_states")


# ── Tab 3: Labels & Triple Barrier ──────────────────────────────────

def _render_labels_tab():
    st.subheader("🎯 Labels & Triple Barrier")

    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Label Threshold",
            value=0.0005, min_value=0.0001, max_value=0.005,
            step=0.0001, format="%.6f", key="label_threshold",
            help="Minimum absolute return to assign a directional label. "
                 "Lower = more signals (noisier); Higher = fewer signals (cleaner).",
        )
    with c2:
        st.checkbox("Use Triple Barrier", value=True, key="use_triple_barrier")

    if st.session_state.get("use_triple_barrier", True):
        st.markdown("#### Triple Barrier Multipliers")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("PT Mult", value=2.0, min_value=0.5, max_value=4.0,
                           step=0.25, key="tb_pt_mult",
                           help="Take-profit distance as multiple of ATR")
        with c2:
            st.number_input("SL Mult", value=2.0, min_value=0.5, max_value=4.0,
                           step=0.25, key="tb_sl_mult",
                           help="Stop-loss distance as multiple of ATR")
        with c3:
            st.number_input("Max Holding", value=36, min_value=4, max_value=72,
                           step=4, key="tb_max_holding",
                           help="Maximum bars before forced exit (timeout label)")
        st.number_input("Neutral Zone", value=0.5, min_value=0.0, max_value=2.0,
                       step=0.25, key="tb_neutral_zone",
                       help="Dead zone around entry where no label is assigned")


# ── Tab 4: HPO & CV ─────────────────────────────────────────────────

def _render_hpo_tab():
    st.subheader("⚙️ HPO & Walk-Forward Configuration")

    st.markdown("#### HPO Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("HPO Trials", value=10, min_value=2, max_value=100,
                        step=1, key="n_trials")
    with c2:
        st.selectbox("Direction", options=["maximize", "minimize"],
                     index=0, key="optuna_direction")
    with c3:
        st.number_input("Random Seed", value=42, min_value=0, max_value=9999,
                        key="seed")

    st.markdown("#### Walk-Forward Windows")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Train Months", value=36, min_value=6, max_value=60,
                        step=1, key="train_months")
    with c2:
        st.number_input("Test Months", value=1, min_value=1, max_value=6,
                        step=1, key="test_months")

    st.markdown("#### Coverage & Confidence")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Target Active Rate", value=0.15, min_value=0.05,
                        max_value=0.30, step=0.01, format="%.3f",
                        key="target_active_rate",
                        help="Target fraction of bars where the strategy takes a trade")
    with c2:
        st.number_input("Target Coverage", value=0.15, min_value=0.05,
                        max_value=0.30, step=0.01, format="%.3f",
                        key="target_coverage",
                        help="Should match target_active_rate")
    with c3:
        st.number_input("Confidence Threshold", value=0.8, min_value=0.0,
                        max_value=1.0, step=0.05, format="%.2f",
                        key="confidence_threshold",
                        help="Minimum model confidence to trigger a trade (0 = auto-calibrated)")

    st.markdown("#### Trading Costs")
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox("Include Trading Costs", value=True, key="eval_use_trading_costs")
    with c2:
        if st.session_state.get("eval_use_trading_costs", True):
            st.number_input("Slippage (bps)", value=0.25, min_value=0.0,
                            max_value=2.0, step=0.05, key="slip_norm_bps")


# ── Tab 5: Results ──────────────────────────────────────────────────

def _render_results_tab():
    if "_run_backtest" in st.session_state and st.session_state["_run_backtest"]:
        st.session_state["_run_backtest"] = False
        st.info("Phase 3 will execute the backtest and show results here.")
    elif "backtest_results" in st.session_state:
        from ui.results import render_results_tab as _rrt
        _rrt()
    else:
        st.subheader("📋 Results")
        st.info("Run a backtest first (click 🚀 Run Backtest), or load previous results below.")

        # Discover previous runs
        import json
        from pathlib import Path
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
            st.markdown("#### Load Previous Run")
            run_labels = [f"{r[0]}: {r[1]}" for r in runs]
            selected = st.selectbox("Previous Runs", options=run_labels, index=0)
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