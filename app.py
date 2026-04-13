"""
FX ML Backtester — Streamlit Web Application Entry Point.

Layout: Icon nav bar at top (main area) + tab content below. No sidebar navigation.
Launch with:  streamlit run app.py
"""

import logging
import streamlit as st
import traceback

# Suppress harmless "No runtime found" warning when importing outside streamlit run
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from ui.controls import render_nav_bar, render_tab_content, get_all_params
from ui.state import AppState, DATA_FILES


def main():
    """Main Streamlit app — only executed under ``streamlit run``."""
    st.set_page_config(
        page_title="FX ML Backtester",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ---------- Navigation Bar ----------
    current_tab = render_nav_bar()

    # ---------- Handle Run Backtest ----------
    if current_tab == 5 and st.session_state.get("_run_backtest", False):
        st.session_state["_run_backtest"] = False

        # Collect params and data path
        params = get_all_params()
        data_key = st.session_state.get("data_key", "EURUSD_H1")
        data_info = DATA_FILES.get(data_key, DATA_FILES["EURUSD_H1"])
        csv_path = data_info["path"]

        # Show run summary
        with st.expander("📋 Run Summary — parameters being used", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Data:** `{data_key}`")
                st.markdown(f"**Model:** `{params.get('model_type')}`")
                st.markdown(f"**Calibration:** `{params.get('calibrate_method')}`")
            with c2:
                st.markdown(f"**HPO Trials:** {params.get('n_trials')}")
                st.markdown(f"**Train/Test:** {params.get('train_months')}m / {params.get('test_months')}m")
                st.markdown(f"**Seed:** {params.get('seed')}")
            with c3:
                st.markdown(f"**Trading Costs:** {params.get('eval_use_trading_costs')}")
                st.markdown(f"**Slippage:** {params.get('slip_norm_bps')} bps")
                st.markdown(f"**Confidence:** {params.get('confidence_threshold')}")
            indicators_on = [k.replace("use_", "") for k, v in params.items()
                             if k.startswith("use_") and v is True]
            indicators_off = [k.replace("use_", "") for k, v in params.items()
                              if k.startswith("use_") and v is False]
            st.caption(f"Indicators ON: {', '.join(indicators_on)}")
            st.caption(f"Indicators OFF: {', '.join(indicators_off)}")

        progress = st.progress(0, text="Initializing backtest...")
        try:
            progress.progress(10, text="Loading data...")
            results = AppState.run_backtest(
                params=params,
                csv_path=csv_path,
            )
            st.session_state["backtest_results"] = results
            progress.progress(100, text="Done!")
            st.success("Backtest completed successfully!")
            # Render results dashboard
            from ui.dashboard import render_dashboard
            render_dashboard(results=results)
        except Exception as e:
            progress.empty()
            st.error(f"Backtest failed: {e}")
            with st.expander("Traceback", expanded=True):
                st.code(traceback.format_exc())

    # ---------- Tab Content ----------
    elif current_tab == 5 and "backtest_results" in st.session_state:
        # Show cached results
        from ui.dashboard import render_dashboard
        render_dashboard(results=st.session_state["backtest_results"])
    else:
        render_tab_content(current_tab)


# streamlit run executes the script as __main__; bare import does not
if __name__ == "__main__":
    main()
