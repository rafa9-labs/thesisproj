"""
Modern tabbed control interface for Data & Features and Model & Tuning.

Replaces the cramped sidebar with spacious, full-width tabs using columns
and expandable sections for a professional SaaS ML platform design.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta

from src.core.config import AppConfig


def render_data_tab(config: AppConfig) -> Dict[str, Any]:
    """
    Render Data & Features tab with spacious layout.
    
    Uses st.columns for clean organization:
    - Instrument & Date Range (left column)
    - Granularity & Train Split (right column)
    - Indicator toggles (3 columns: Momentum, Trend, Volatility)
    
    Args:
        config: AppConfig instance
        
    Returns:
        Dictionary with data settings
    """
    st.header("⚙️ Data & Features Configuration")
    st.markdown("*Configure your data source, timeframe, and technical indicators*")
    st.markdown("---")
    
    # Row 1: Instrument and Settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Instrument & Timeframe")
        
        instrument = st.selectbox(
            "Currency Pair",
            ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD"],
            index=0,
            help="Select forex pair to backtest"
        )
        
        st.markdown("**Date Range**")
        # Default dates
        default_end = datetime.now().date()
        default_start = (datetime.now() - timedelta(days=365)).date()
        
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                help="Backtest start date"
            )
        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                help="Backtest end date"
            )
    
    with col2:
        st.subheader("⚙️ Backtest Settings")
        
        granularity = st.selectbox(
            "Timeframe",
            ["H1", "H4", "D"],
            index=0,
            help="Bar granularity (H1=1 hour, H4=4 hours, D=daily)"
        )
        
        train_split = st.slider(
            "Train/Test Split",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.05,
            help="Fraction of data for training (remaining for testing)"
        )
        
        # Visual feedback
        train_pct = int(train_split * 100)
        test_pct = 100 - train_pct
        st.info(f"📊 Split: {train_pct}% training / {test_pct}% testing")
    
    # Row 2: Technical Indicators
    st.markdown("---")
    st.subheader("📈 Technical Indicators")
    st.markdown("*Select indicators to include in feature engineering*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💪 Momentum**")
        use_rsi = st.checkbox(
            "RSI (Relative Strength Index)",
            value=config.features.use_rsi,
            help="Momentum oscillator (14-period default)"
        )
        use_macd = st.checkbox(
            "MACD (Moving Average Convergence Divergence)",
            value=config.features.use_macd,
            help="Trend-following momentum indicator"
        )
    
    with col2:
        st.markdown("**📈 Trend**")
        use_ema = st.checkbox(
            "EMA (Exponential Moving Average)",
            value=config.features.use_ema,
            help="Weighted moving average (20-period default)"
        )
        use_adx = st.checkbox(
            "ADX (Average Directional Index)",
            value=config.features.use_adx,
            help="Trend strength indicator (14-period default)"
        )
    
    with col3:
        st.markdown("**📊 Volatility**")
        use_bbands = st.checkbox(
            "Bollinger Bands",
            value=config.features.use_bbands,
            help="Volatility bands (20-period, 2 std dev default)"
        )
        use_atr = st.checkbox(
            "ATR (Average True Range)",
            value=config.features.use_atr,
            help="Volatility measure (14-period default)"
        )
    
    # Summary
    indicators_selected = sum([use_rsi, use_macd, use_ema, use_adx, use_bbands, use_atr])
    if indicators_selected > 0:
        st.success(f"✅ {indicators_selected} indicator(s) selected")
    else:
        st.warning("⚠️ No indicators selected - using default features only")
    
    return {
        "instrument": instrument,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "granularity": granularity,
        "train_split": train_split,
        "use_rsi": use_rsi,
        "use_macd": use_macd,
        "use_ema": use_ema,
        "use_bbands": use_bbands,
        "use_atr": use_atr,
        "use_adx": use_adx
    }


def render_model_tab(config: AppConfig, current_model_type: str = "XGBoost") -> Dict[str, Any]:
    """
    Render Model & Tuning tab with modern SaaS ML platform design.
    
    Uses columns and expanders for spacious, professional layout:
    - Model selection (prominent)
    - HPO configuration (expandable sections)
    - Dynamic parameter selection
    - Primary Run Backtest button
    
    Args:
        config: AppConfig instance
        current_model_type: Current model type from session state
        
    Returns:
        Dictionary with model and HPO settings
    """
    st.header("🧠 Model & Hyperparameter Tuning")
    st.markdown("*Configure your ML model and optimization strategy*")
    st.markdown("---")
    
    # Row 1: Model Selection
    st.subheader("🤖 Model Selection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Algorithm",
            ["XGBoost", "CNN", "LSTM", "Ensemble"],
            index=["XGBoost", "CNN", "LSTM", "Ensemble"].index(current_model_type) if current_model_type in ["XGBoost", "CNN", "LSTM", "Ensemble"] else 0,
            help="Choose machine learning model for predictions"
        )
    
    with col2:
        st.metric("Selected", model_type, delta=None)
    
    # Model description
    model_descriptions = {
        "XGBoost": "🌳 Gradient boosting - Fast, accurate, handles non-linearity well",
        "CNN": "🧠 Convolutional Neural Network - Captures local patterns in time series",
        "LSTM": "🔄 Long Short-Term Memory - Learns long-term dependencies",
        "Ensemble": "🎯 Combines XGBoost, CNN, and LSTM for robust predictions"
    }
    st.info(model_descriptions.get(model_type, ""))
    
    # Row 2: HPO Configuration
    st.markdown("---")
    st.subheader("⚙️ Hyperparameter Optimization")
    
    enable_hpo = st.toggle(
        "Enable HPO",
        value=False,
        help="Automatically tune model parameters using Optuna"
    )
    
    hpo_mode = "single_time"
    n_trials = 50
    active_params = []
    
    if enable_hpo:
        st.markdown("**Optimization Settings**")
        
        # HPO Settings in columns
        col1, col2 = st.columns(2)
        
        with col1:
            hpo_mode = st.selectbox(
                "Optimization Mode",
                ["single_time", "continuous_wfo", "mini_folds"],
                index=0,
                help=(
                    "• single_time: Optimize once, reuse params\n"
                    "• continuous_wfo: Optimize every fold\n"
                    "• mini_folds: Optimize when window expands"
                )
            )
        
        with col2:
            n_trials = st.number_input(
                "Number of Trials",
                min_value=1,  # SMOKE TEST: Allow single trial
                max_value=200,
                value=config.hpo.n_trials,  # Use config default (3)
                step=1,
                help="More trials = better optimization but slower"
            )
        
        # Mode description
        mode_descriptions = {
            "single_time": "⚡ Fast - Optimize once and reuse parameters",
            "continuous_wfo": "🔄 Thorough - Re-optimize for each fold",
            "mini_folds": "⚖️ Balanced - Optimize when training window expands"
        }
        st.info(mode_descriptions.get(hpo_mode, ""))
        
        # Parameter Selection (expandable)
        with st.expander("🎯 Parameter Selection", expanded=True):
            st.markdown("**Select parameters to optimize**")
            st.caption("Leave all unchecked to optimize ALL available parameters")
            
            # Define available parameters per model (from hpo.py)
            param_options = {
                "XGBoost": [
                    "xgb_n_estimators",
                    "xgb_max_depth",
                    "xgb_learning_rate",
                    "xgb_subsample",
                    "xgb_colsample_bytree",
                    "xgb_gamma",
                    "xgb_min_child_weight",
                    "xgb_reg_alpha",
                    "xgb_reg_lambda"
                ],
                "CNN": [
                    "cnn_filters1",
                    "cnn_filters2",
                    "cnn_kernel_size",
                    "cnn_dense_units",
                    "cnn_dropout_rate",
                    "cnn_learning_rate",
                    "cnn_padding_same",
                    "cnn_clipnorm",
                    "cnn_use_early_stopping",
                    "cnn_patience"
                ],
                "LSTM": [
                    "lstm_units1",
                    "lstm_units2",
                    "lstm_dense_units",
                    "lstm_dropout_rate",
                    "lstm_recurrent_dropout",
                    "lstm_learning_rate",
                    "lstm_clipnorm",
                    "lstm_use_early_stopping",
                    "lstm_patience"
                ],
                "Ensemble": [
                    "ensemble_voting_method",
                    "ensemble_weights",
                    "ensemble_calibrate",
                    "ensemble_fusion_alpha"
                ]
            }
            
            # Get available params for selected model_type
            available_params = param_options.get(model_type, [])
            
            # Use columns for parameter checkboxes (cleaner than multiselect)
            num_cols = 3
            cols = st.columns(num_cols)
            
            active_params = []
            for idx, param in enumerate(available_params):
                with cols[idx % num_cols]:
                    # Use session state key to maintain selection
                    if st.checkbox(
                        param.replace('_', ' ').title(),
                        key=f"param_{model_type}_{param}",
                        help=f"Optimize {param}"
                    ):
                        active_params.append(param)
            
            # Summary
            if not active_params:
                st.info("ℹ️ No parameters selected - will optimize ALL available parameters")
            else:
                st.success(f"✅ Optimizing {len(active_params)} parameter(s): {', '.join(active_params[:3])}{'...' if len(active_params) > 3 else ''}")
    
    # Row 3: WFO Configuration (NEW)
    st.markdown("---")
    st.subheader("📊 Walk-Forward Optimization")
    
    with st.expander("🔧 WFO Window Configuration", expanded=False):
        st.markdown("**Configure how training windows expand over time**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_window_type = st.selectbox(
                "Train Window Type",
                ["expanding", "rolling"],
                index=0,
                help=(
                    "• expanding: Train on all data from start (default)\n"
                    "• rolling: Train on fixed recent window"
                )
            )
            
            training_duration = st.number_input(
                "Training Duration (bars)",
                min_value=500,
                max_value=10000,
                value=2000,
                step=100,
                help="Number of bars to train on per fold"
            )
        
        with col2:
            test_period_duration = st.number_input(
                "Test Period Duration (bars)",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of bars to test per fold (walk-forward step size)"
            )
            
            n_mini_folds = st.number_input(
                "Number of Folds (optional)",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                help="Explicit fold count (0 = auto-generate all possible folds)"
            )
        
        hpo_validation_split = st.slider(
            "HPO Validation Split",
            min_value=0.10,
            max_value=0.30,
            value=0.20,
            step=0.05,
            help="Fraction of train block reserved for Optuna validation"
        )
        
        # Visual explanation
        if train_window_type == "expanding":
            st.info("📈 Expanding window: Each fold trains on all data from start to current position")
        else:
            st.info(f"📊 Rolling window: Each fold trains on last {training_duration} bars only")
    
    return {
        "model_type": model_type,
        "enable_hpo": enable_hpo,
        "hpo_mode": hpo_mode,
        "n_trials": n_trials,
        "active_params": active_params,
        "train_window_type": train_window_type,
        "training_duration": training_duration,
        "test_period_duration": test_period_duration,
        "n_mini_folds": n_mini_folds if n_mini_folds > 0 else None,
        "hpo_validation_split": hpo_validation_split
    }
