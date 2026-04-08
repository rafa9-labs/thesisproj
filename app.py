"""
FX ML Backtester — Streamlit Web Application Entry Point.

Launch with:  streamlit run app.py
"""

import streamlit as st
from ui.state import AppState, DATA_FILES
from ui.controls import render_data_tab, render_model_tab
from ui.dashboard import render_dashboard

st.set_page_config(
    page_title="FX ML Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Sidebar ----------
render_data_tab()
config = render_model_tab()

# Run button
run_clicked = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ---------- Main Area ----------
if run_clicked:
    data_key = st.session_state.get("data_key", "EURUSD_H1")
    data_info = DATA_FILES[data_key]
    csv_path = data_info["path"]

    progress = st.progress(0, text="Initializing backtest...")
    try:
        results = AppState.run_backtest(
            model_type=config["model_type"],
            csv_path=csv_path,
            n_months=config["n_months"],
            n_trials=config["n_trials"],
            seed=config["seed"],
            trading_costs=config["trading_costs"],
        )
        progress.progress(100, text="Done!")
        render_dashboard(results=results)
    except Exception as e:
        progress.empty()
        st.error(f"Backtest failed: {e}")
        with st.expander("Traceback"):
            import traceback
            st.code(traceback.format_exc())
else:
    preview_df = st.session_state.get("preview_df")
    render_dashboard(preview_df=preview_df)