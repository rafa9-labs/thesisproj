"""
Main dashboard rendering — equity curves, metrics tables, trade analysis.
Adapted from init-proj's dashboard to work with our pipeline backend.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional


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
        _render_backtest_results(results)


def _render_landing():
    """Show welcome / instructions when no data is loaded."""
    st.markdown("## 📈 FX ML Backtester")
    st.markdown(
        """
        **Welcome!** Use the sidebar to:
        1. **Select a data source** — pick a currency pair and timeframe
        2. **Configure your model** — choose architecture and hyperparameters
        3. **Run backtest** — click the Run button to start walk-forward validation

        The dashboard will display equity curves, performance metrics,
        and trade-by-trade analysis once the backtest completes.
        """
    )


def _render_data_preview(df: pd.DataFrame):
    """Show a data preview table."""
    st.subheader("📊 Data Preview")
    st.dataframe(df, use_container_width=True, height=250)


def _render_backtest_results(results: Dict[str, Any]):
    """Render full backtest results: metrics + charts."""
    metrics = results.get("metrics", {})
    equity_curve = results.get("equity_curve")
    monthly_df = results.get("monthly_df", pd.DataFrame())
    model_type = results.get("model_type", "unknown")

    st.subheader(f"🎯 Backtest Results — {model_type.upper()}")

    # --- KPI row ---
    _render_kpi_row(metrics)

    # --- Equity curve ---
    if equity_curve is not None and not equity_curve.empty:
        _render_equity_curve(equity_curve)

    # --- Monthly breakdown ---
    if not monthly_df.empty:
        _render_monthly_table(monthly_df)


def _render_kpi_row(metrics: Dict[str, Any]):
    """Top-level KPI cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sharpe = metrics.get("sharpe", np.nan)
        sharpe_str = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
        st.metric("Sharpe Ratio", sharpe_str)
    with col2:
        st.metric("Max Drawdown", f"{metrics.get('drawdown', 0):.1%}")
    with col3:
        st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.1f}%")
    with col4:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Total Trades", f"{metrics.get('trades', 0)}")
    with col6:
        st.metric("Directional Acc.", f"{metrics.get('directional_accuracy', 0):.1%}")
    with col7:
        st.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.3f}")
    with col8:
        st.metric("Outperformance", f"{metrics.get('outperformance', 0):.2%}")


def _render_equity_curve(equity_curve: pd.Series):
    """Plot the equity curve with drawdown shading."""
    st.subheader("📉 Equity Curve")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=equity_curve.values,
            mode="lines",
            name="Strategy",
            line=dict(color="#00d4aa", width=2),
        )
    )
    cum_max = np.maximum.accumulate(equity_curve.values)
    drawdown = (equity_curve.values - cum_max) / np.where(cum_max > 0, cum_max, 1.0)
    fig.add_trace(
        go.Scatter(
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="rgba(255,80,80,0.4)", width=1),
            yaxis="y2",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        yaxis=dict(title="Equity"),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right", range=[-0.5, 0]),
        margin=dict(l=50, r=50, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_monthly_table(monthly_df: pd.DataFrame):
    """Monthly performance breakdown table."""
    st.subheader("📅 Monthly Breakdown")
    display_cols = [c for c in [
        "month", "trades", "win_rate", "strategy_return",
        "directional_accuracy", "precision_macro", "f1_macro",
    ] if c in monthly_df.columns]
    if display_cols:
        formatted = monthly_df[display_cols].copy()
        for pct_col in ["win_rate", "directional_accuracy", "precision_macro"]:
            if pct_col in formatted.columns:
                formatted[pct_col] = formatted[pct_col].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "—"
                )
        for num_col in ["strategy_return", "f1_macro"]:
            if num_col in formatted.columns:
                formatted[num_col] = formatted[num_col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        st.dataframe(formatted, use_container_width=True, hide_index=True)
    else:
        st.dataframe(monthly_df, use_container_width=True)