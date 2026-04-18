"""
Plotly chart builders for the FX ML Backtester UI.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def equity_curve_chart(equity_curve: pd.Series, bh_curve: Optional[pd.Series] = None) -> go.Figure:
    """Equity curve with drawdown shading and optional buy-and-hold benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve.values,
        mode="lines", name="Strategy",
        line=dict(color="#00d4aa", width=2),
    ))
    if bh_curve is not None and not bh_curve.empty:
        fig.add_trace(go.Scatter(
            y=bh_curve.values, mode="lines", name="Buy & Hold",
            line=dict(color="#555555", width=1, dash="dash"),
        ))
    cum_max = np.maximum.accumulate(equity_curve.values)
    dd = (equity_curve.values - cum_max) / np.where(cum_max > 0, cum_max, 1.0)
    fig.add_trace(go.Scatter(
        y=dd, mode="lines", name="Drawdown", fill="tozeroy",
        line=dict(color="rgba(255,80,80,0.4)", width=1), yaxis="y2",
    ))
    fig.update_layout(
        template="plotly_dark", height=420,
        yaxis=dict(title="Equity"), margin=dict(l=50, r=50, t=30, b=30),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right", range=[-0.5, 0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def param_importance_chart(importance_dict: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of Optuna param importances."""
    if not importance_dict:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=300,
                          title="No param importance data available")
        return fig
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    params = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    fig = go.Figure(go.Bar(
        x=values, y=params, orientation="h",
        marker_color="#00d4aa",
    ))
    fig.update_layout(
        template="plotly_dark", height=max(300, len(params) * 22 + 60),
        title="📊 Parameter Importance (Top 20)",
        xaxis_title="Importance", margin=dict(l=140, r=30, t=50, b=30),
    )
    return fig


def optimization_trace_chart(trial_values: list) -> go.Figure:
    """Optuna optimization trace: best value vs trial number."""
    if not trial_values:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=300,
                          title="No trial history available")
        return fig
    running_best = np.maximum.accumulate(trial_values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(trial_values))), y=trial_values,
        mode="markers", name="Trial Score",
        marker=dict(color="#555555", size=6),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(running_best))), y=running_best,
        mode="lines", name="Best So Far",
        line=dict(color="#00d4aa", width=2),
    ))
    fig.update_layout(
        template="plotly_dark", height=350,
        title="🔬 Optimization Trace",
        xaxis_title="Trial #", yaxis_title="Score (Sharpe)",
        margin=dict(l=60, r=30, t=50, b=30),
    )
    return fig


def monthly_returns_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """Bar chart of monthly strategy returns."""
    if monthly_df.empty or "strategy_return" not in monthly_df.columns:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=300,
                          title="No monthly return data")
        return fig
    rets = monthly_df["strategy_return"].dropna()
    colors = ["#00d4aa" if r > 0 else "#ff5050" for r in rets.values]
    fig = go.Figure(go.Bar(
        x=list(range(len(rets))), y=rets.values,
        marker_color=colors,
    ))
    fig.update_layout(
        template="plotly_dark", height=300,
        title="Monthly Strategy Returns",
        xaxis_title="Month #", yaxis_title="Return",
        margin=dict(l=60, r=30, t=50, b=30),
    )
    return fig


def equity_curve_with_events(
    equity_curve: pd.Series,
    events: list | None = None,
    bh_curve: Optional[pd.Series] = None,
    show_events: bool = True,
) -> go.Figure:
    """Equity curve with optional economic event markers.

    Parameters
    ----------
    equity_curve : pd.Series
        Strategy equity curve.
    events : list[dict] or None
        Economic events: ``[{"date": datetime|str, "event": str, "impact": int}, ...]``.
    bh_curve : pd.Series or None
        Buy-and-hold benchmark.
    show_events : bool
        Whether to overlay event markers.

    Returns
    -------
    go.Figure
    """
    fig = equity_curve_chart(equity_curve, bh_curve)

    if not show_events or not events:
        return fig

    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        try:
            equity_index = pd.to_datetime(equity_curve.index)
        except Exception:
            return fig
    else:
        equity_index = equity_curve.index

    impact_colors = {3: "#ff4444", 2: "#ffaa00", 1: "#44aaff"}
    event_shown = set()

    for ev in events:
        ev_date = ev.get("date")
        ev_name = ev.get("event", "Event")
        ev_impact = ev.get("impact", 1)
        if ev_date is None:
            continue
        try:
            ts = pd.Timestamp(ev_date)
        except Exception:
            continue

        if ts not in equity_index:
            idx_near = equity_index.get_indexer([ts], method="nearest")
            if idx_near[0] < 0:
                continue
            ts = equity_index[idx_near[0]]

        if ts in event_shown:
            continue
        event_shown.add(ts)

        color = impact_colors.get(ev_impact, "#888888")

        fig.add_vline(
            x=ts,
            line_width=1,
            line_dash="dot",
            line_color=color,
            opacity=0.7,
        )

    if event_shown:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="#ff4444", symbol="diamond"),
            name="High-impact event",
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="#ffaa00", symbol="diamond"),
            name="Medium-impact event",
        ))

    return fig