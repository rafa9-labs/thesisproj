"""
Main dashboard rendering — delegates to results module for rich display,
retains data preview and landing page.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from ui.results import render_results_tab, _render_full_results


def render_dashboard(
    results: Optional[Dict[str, Any]] = None,
    preview_df: Optional[pd.DataFrame] = None,
):
    """Render the main dashboard area."""
    if results is None and preview_df is None:
        _render_landing()
        return

    if preview_df is not None:
        _render_data_preview(preview_df)

    if results is not None:
        _render_full_results(results)


def render_results_main():
    """Render results browser in the main area (for Results tab)."""
    render_results_tab()


def _render_landing():
    """Show welcome / instructions when no data is loaded."""
    st.markdown("## 📈 FX ML Backtester")
    st.markdown(
        """
        **Welcome!** Use the sidebar tabs to:

        1. **📊 Data** — pick a currency pair and timeframe
        2. **🤖 Model** — choose architecture and hyperparameters
        3. **📈 Features** — toggle indicators and feature engineering
        4. **🎯 Labels** — configure triple barrier and label thresholds
        5. **⚙️ HPO** — set walk-forward windows, coverage, and costs
        6. **📋 Results** — view current or load previous results

        Click **🚀 Run Backtest** at the bottom of the sidebar to start.

        The dashboard will display equity curves, performance metrics,
        HPO diagnostics, and trade-by-trade analysis once the backtest completes.
        """
    )


def _render_data_preview(df: pd.DataFrame):
    """Show a data preview table."""
    st.subheader("📊 Data Preview")
    st.dataframe(df, use_container_width=True, height=250)