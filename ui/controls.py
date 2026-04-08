"""
Streamlit sidebar controls — adapted from init-proj to use our pipeline/ backend.
"""

import streamlit as st
from ui.state import AppState, AVAILABLE_MODELS, DATA_FILES


def render_data_tab():
    """Render data source selection in the sidebar."""
    st.sidebar.subheader("📊 Data Source")
    data_key = st.sidebar.selectbox(
        "Currency Pair / Timeframe",
        options=list(DATA_FILES.keys()),
        index=0,
        key="data_key",
    )
    data_info = DATA_FILES[data_key]
    st.sidebar.caption(f"File: `{data_info['path']}`  |  TF: {data_info['tf']}")

    if st.sidebar.button("Preview Data", key="preview_data_btn"):
        try:
            df = AppState.load_csv_data(data_info["path"])
            st.session_state["preview_df"] = df.head(20)
            st.session_state["preview_rows"] = len(df)
            st.session_state["preview_cols"] = list(df.columns)
        except Exception as e:
            st.sidebar.error(f"Load error: {e}")

    if "preview_df" in st.session_state:
        st.sidebar.success(
            f"{st.session_state['preview_rows']:,} rows × "
            f"{len(st.session_state['preview_cols'])} cols"
        )


def render_model_tab():
    """Render model configuration controls in the sidebar."""
    st.sidebar.subheader("🤖 Model Configuration")

    model_type = st.sidebar.selectbox(
        "Model Type",
        options=AVAILABLE_MODELS,
        index=0,
        key="model_type",
        help="Select the ML model architecture to backtest",
    )

    st.sidebar.subheader("⚙️ Backtest Settings")
    n_months = st.sidebar.slider(
        "Walk-Forward Months",
        min_value=1,
        max_value=36,
        value=6,
        step=1,
        key="n_months",
        help="Number of months for walk-forward validation",
    )
    n_trials = st.sidebar.slider(
        "HPO Trials per Window",
        min_value=2,
        max_value=50,
        value=10,
        step=1,
        key="n_trials",
        help="Optuna trials per walk-forward window",
    )
    seed = st.sidebar.number_input(
        "Random Seed",
        value=42,
        min_value=0,
        max_value=9999,
        key="seed",
    )
    trading_costs = st.sidebar.checkbox(
        "Include Trading Costs",
        value=True,
        key="trading_costs",
        help="Apply spread/commission costs to simulated trades",
    )
    return {
        "model_type": model_type,
        "n_months": n_months,
        "n_trials": n_trials,
        "seed": seed,
        "trading_costs": trading_costs,
    }