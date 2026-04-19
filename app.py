"""
FX ML Backtester — Streamlit Web Application Entry Point.

Layout: Icon nav bar at top (main area) + tab content below. No sidebar navigation.
Launch with:  streamlit run app.py
"""

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import logging
import streamlit as st
import traceback

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

from ui.controls import render_nav_bar, render_tab_content, get_all_params
from ui.state import AppState, DATA_FILES


def main():
    st.set_page_config(
        page_title="FX ML Backtester",
        page_icon="\U0001F4C8",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    current_tab = render_nav_bar()

    if current_tab == 5 and st.session_state.get("_run_backtest", False):
        st.session_state["_run_backtest"] = False

        params = get_all_params()
        data_key = st.session_state.get("data_key", "EURUSD_H1")
        data_info = DATA_FILES.get(data_key, DATA_FILES["EURUSD_H1"])
        csv_path = data_info["path"]

        model_label = params.get("model_type", "logistic").upper()
        n_trials = params.get("n_trials", 10)
        train_m = params.get("train_months", 36)
        test_m = params.get("test_months", 1)

        with st.expander("\U0001F4CB Run Summary", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Data:** `{data_key}`")
                st.markdown(f"**Model:** `{model_label}`")
                st.markdown(f"**Calibration:** `{params.get('calibrate_method')}`")
            with c2:
                st.markdown(f"**HPO Trials:** {n_trials}")
                st.markdown(f"**Train/Test:** {train_m}m / {test_m}m")
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

        est_min = n_trials * test_m * 0.5
        est_max = n_trials * test_m * 2.0
        if model_label in ("CNN", "LSTM", "TRANSFORMER", "DQN"):
            est_min *= 5
            est_max *= 10

        with st.status(
            f"\U0001F504 Running backtest: {model_label} | {n_trials} trials | {train_m}m train / {test_m}m test",
            expanded=True,
        ) as status:
            st.caption(
                f"Estimated time: {est_min:.0f}\u2013{est_max:.0f} minutes. "
                f"The UI will appear frozen during computation \u2014 this is normal."
            )
            st.caption("Loading pipeline and computing features...")

            try:
                results = AppState.run_backtest(
                    params=params,
                    csv_path=csv_path,
                )
                st.session_state["backtest_results"] = results

                metrics = results.get("metrics", {})
                sharpe = metrics.get("sharpe", float("nan"))
                trades = metrics.get("trades", 0)
                total_ret = metrics.get("total_return_pct", 0)

                status.update(
                    label=f"\u2705 Backtest complete: {model_label} | Sharpe {sharpe:.2f} | {trades} trades | {total_ret:+.1f}% return",
                    state="complete",
                    expanded=False,
                )
                st.success(
                    f"**{model_label}** backtest finished \u2014 "
                    f"Sharpe: {sharpe:.2f} | Trades: {trades} | Return: {total_ret:+.1f}%"
                )

                from ui.dashboard import render_dashboard
                render_dashboard(results=results)

            except Exception as e:
                status.update(
                    label=f"\u274c Backtest failed: {e}",
                    state="error",
                    expanded=True,
                )
                st.error(f"Backtest failed: {e}")
                with st.expander("Traceback", expanded=True):
                    st.code(traceback.format_exc())

    elif current_tab == 5 and "backtest_results" in st.session_state:
        from ui.dashboard import render_dashboard
        render_dashboard(results=st.session_state["backtest_results"])
    else:
        render_tab_content(current_tab)


if __name__ == "__main__":
    main()
