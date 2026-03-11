"""
Dashboard view with Holy Trinity metrics and interactive equity curve.

Hyper-minimalist design optimized for speed with float32 equity curves.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
import time
import logging


def render_dashboard(
    metrics: Dict[str, Any],
    equity_curve: pd.Series,
    results_df: pd.DataFrame,
    fold_logs: list = None
):
    """
    Render minimalist dashboard with Holy Trinity metrics and equity curve.
    
    Args:
        metrics: Dictionary with 16 standard metrics
        equity_curve: Float32 equity curve (UI-optimized)
        results_df: Full backtest results
        fold_logs: Optional list of WFOFold objects with HPO results
    """
    # Title is now in app.py main(), not here
    
    # Holy Trinity: Sharpe, Drawdown, Win Rate
    render_holy_trinity(metrics)
    
    # Equity curve chart
    render_equity_curve(equity_curve)
    
    # Additional metrics (expandable)
    render_detailed_metrics(metrics)
    
    # HPO & WFO Logs (NEW)
    if fold_logs:
        render_hpo_fold_logs(fold_logs)
    
    # Trade statistics (expandable)
    render_trade_statistics(results_df)


def render_holy_trinity(metrics: Dict[str, Any]):
    """
    Render the 'Holy Trinity' metric row.
    
    CRITICAL: Uses st.columns(3) for clean layout.
    Displays Sharpe Ratio, Max Drawdown, and Win Rate.
    
    Args:
        metrics: Dictionary with all metrics
    """
    st.subheader("Key Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sharpe = metrics.get('sharpe', 0.0)
        
        # Handle NaN Sharpe (too few trades)
        if np.isnan(sharpe):
            st.metric(
                label="Sharpe Ratio",
                value="N/A",
                delta="Too few trades",
                delta_color="off"
            )
        else:
            delta_text = "Excellent" if sharpe > 2.0 else "Good" if sharpe > 1.0 else "Poor"
            delta_color = "normal" if sharpe > 1.0 else "inverse"
            
            st.metric(
                label="Sharpe Ratio",
                value=f"{sharpe:.2f}",
                delta=delta_text,
                delta_color=delta_color,
                help="Annualized risk-adjusted return (HAC-adjusted)"
            )
    
    with col2:
        drawdown = metrics.get('drawdown', 0.0)
        
        delta_text = "Low Risk" if drawdown > -0.10 else "Moderate" if drawdown > -0.20 else "High Risk"
        delta_color = "normal" if drawdown > -0.20 else "inverse"
        
        st.metric(
            label="Max Drawdown",
            value=f"{drawdown:.2%}",
            delta=delta_text,
            delta_color=delta_color,
            help="Maximum peak-to-trough decline"
        )
    
    with col3:
        win_rate = metrics.get('win_rate', 0.0)
        
        delta_text = "Excellent" if win_rate > 0.60 else "Good" if win_rate > 0.50 else "Poor"
        delta_color = "normal" if win_rate > 0.50 else "inverse"
        
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.2%}",
            delta=delta_text,
            delta_color=delta_color,
            help="Fraction of profitable trades"
        )


def render_equity_curve(equity_curve: pd.Series):
    """
    Render interactive equity curve using Plotly.
    
    CRITICAL: Uses float32 series for fast rendering.
    
    Args:
        equity_curve: Float32 equity curve from BacktestEngine
    """
    st.subheader("Equity Curve")
    
    # Verify float32 dtype
    if equity_curve.dtype != np.float32:
        st.warning(f"Equity curve dtype is {equity_curve.dtype}, expected float32")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Strategy equity line
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Strategy',
        line=dict(color='#00D9FF', width=2),
        hovertemplate='<b>%{x}</b><br>Equity: %{y:.4f}<extra></extra>',
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    
    # Add 1.0 baseline (initial equity)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="rgba(128, 128, 128, 0.5)",
        line_width=1,
        annotation_text="Initial Equity",
        annotation_position="right"
    )
    
    # Calculate final return
    final_equity = float(equity_curve.iloc[-1])
    total_return_pct = (final_equity - 1.0) * 100
    
    # Add final equity annotation
    fig.add_annotation(
        x=equity_curve.index[-1],
        y=final_equity,
        text=f"Final: {final_equity:.4f}<br>({total_return_pct:+.2f}%)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#00D9FF",
        ax=40,
        ay=-40,
        bgcolor="rgba(0, 0, 0, 0.8)",
        bordercolor="#00D9FF",
        borderwidth=1,
        font=dict(color="white", size=11)
    )
    
    # Clean, minimalist layout
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            title="",
            showgrid=False,
            zeroline=False,
            tickformat='%Y-%m-%d'
        ),
        yaxis=dict(
            title="Equity",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.15)',
            zeroline=False
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family="monospace", size=11, color="#E0E0E0"),
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch', key=f'equity_curve_chart_{int(time.time() * 1000)}')
    
    # Memory efficiency info
    memory_kb = equity_curve.memory_usage() / 1024
    st.caption(f"📊 {len(equity_curve)} bars | 💾 {memory_kb:.2f} KB (float32)")


def render_detailed_metrics(metrics: Dict[str, Any]):
    """
    Render detailed metrics in expandable section.
    
    Args:
        metrics: Dictionary with all metrics
    """
    with st.expander("📈 Detailed Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Returns**")
            st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
            st.metric("Cumulative Strategy", f"{metrics.get('cstrategy', 1.0):.4f}")
            st.metric("Geometric Mean (Ann.)", f"{metrics.get('geo_mean_ann', 0):.4f}")
            st.metric("Outperformance", f"{metrics.get('outperformance', 1.0):.4f}")
        
        with col2:
            st.markdown("**Trading Activity**")
            st.metric("Total Trades", f"{metrics.get('trades', 0)}")
            st.metric("Active Rate", f"{metrics.get('active_rate', 0):.2%}")
            st.metric("Return per Trade", f"{metrics.get('return_per_trade', 0):.6f}")
            st.metric("Profit per Hit", f"{metrics.get('profit_per_hit', 0):.6f}")
        
        with col3:
            st.markdown("**Accuracy**")
            st.metric("Directional Accuracy", f"{metrics.get('directional_accuracy', 0):.2%}")
            st.metric("Precision (Macro)", f"{metrics.get('precision_macro', 0):.2%}")
            st.metric("F1 Score (Macro)", f"{metrics.get('f1_macro', 0):.2%}")
            st.metric("Volatility", f"{metrics.get('strategy_volatility', 0):.6f}")


def render_trade_statistics(results_df: pd.DataFrame):
    """
    Render trade statistics in expandable section.
    
    Args:
        results_df: Full backtest results DataFrame
    """
    with st.expander("📋 Trade Statistics"):
        if 'position_exec' not in results_df.columns:
            st.info("No trade data available")
            return
        
        # Calculate trade statistics
        positions = results_df['position_exec'].values
        position_changes = np.diff(positions, prepend=0)
        trade_indices = np.where(position_changes != 0)[0]
        
        if len(trade_indices) == 0:
            st.info("No trades executed")
            return
        
        # Trade distribution
        st.markdown("**Position Distribution**")
        col1, col2, col3 = st.columns(3)
        
        long_bars = int((positions > 0).sum())
        short_bars = int((positions < 0).sum())
        flat_bars = int((positions == 0).sum())
        total_bars = len(positions)
        
        with col1:
            st.metric("Long", f"{long_bars} bars ({long_bars/total_bars:.1%})")
        with col2:
            st.metric("Short", f"{short_bars} bars ({short_bars/total_bars:.1%})")
        with col3:
            st.metric("Flat", f"{flat_bars} bars ({flat_bars/total_bars:.1%})")
        
        # Recent trades table
        st.markdown("**Recent Trades (Last 20)**")
        
        # Get last 20 trade changes
        recent_trade_indices = trade_indices[-20:] if len(trade_indices) > 20 else trade_indices
        
        trade_data = []
        for idx in recent_trade_indices:
            if idx < len(results_df):
                trade_data.append({
                    'Time': results_df.index[idx],
                    'Position': positions[idx],
                    'Equity': results_df['equity'].iloc[idx],
                    'Cost': results_df.get('costs', pd.Series([0]*len(results_df))).iloc[idx]
                })
        
        if trade_data:
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(
                trade_df,
                width='stretch',
                hide_index=True,
                key=f'trade_statistics_table_{int(time.time() * 1000)}'
            )
        else:
            st.info("No recent trades to display")


def render_hpo_fold_logs(fold_logs: list):
    """
    Render HPO & Walk-Forward fold logs as expandable table.
    
    Args:
        fold_logs: List of WFOFold objects with HPO results per fold
    """
    with st.expander("🧠 HPO & Walk-Forward Logs", expanded=False):
        st.markdown("**Fold-by-fold hyperparameter optimization results**")
        st.caption("View the best parameters selected by Optuna at each time step")
        
        if not fold_logs:
            st.info("No fold logs available (HPO may be disabled)")
            return
        
        # Convert fold logs to DataFrame
        fold_data = []
        for fold in fold_logs:
            # Extract key params for display
            params_str = ", ".join([
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in list(fold.hpo_best_params.items())[:5]
            ])
            if len(fold.hpo_best_params) > 5:
                params_str += "..."
            
            fold_data.append({
                'Fold': fold.fold_idx + 1,
                'Train Period': f"{fold.train_start} to {fold.train_end}",
                'Test Period': f"{fold.test_start} to {fold.test_end}",
                'Train Size': fold.train_size,
                'Test Size': fold.test_size,
                'HPO Trials': fold.hpo_n_trials,
                'HPO Best Score': f"{fold.hpo_best_score:.4f}" if not np.isnan(fold.hpo_best_score) else "N/A",
                'Best Params': params_str if params_str else "N/A",
                'Test Accuracy': f"{fold.test_metrics.get('accuracy', 0):.4f}",
                'Test Log Loss': f"{fold.test_metrics.get('log_loss', 0):.4f}"
            })
        
        df_folds = pd.DataFrame(fold_data)
        
        # Display as interactive table
        st.dataframe(
            df_folds,
            use_container_width=True,
            hide_index=True,
            key=f'hpo_fold_logs_{int(time.time() * 1000)}'
        )
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Folds", len(fold_logs))
        with col2:
            avg_score = np.mean([f.hpo_best_score for f in fold_logs if not np.isnan(f.hpo_best_score)])
            st.metric("Avg HPO Score", f"{avg_score:.4f}" if not np.isnan(avg_score) else "N/A")
        with col3:
            total_trials = sum(f.hpo_n_trials for f in fold_logs)
            st.metric("Total HPO Trials", total_trials)
