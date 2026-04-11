"""
Results display windows for the FX ML Backtester UI.

Shows KPIs, CV diagnostics, HPO analysis, trade logs, and loaded results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
from ui.charts import (
    equity_curve_chart, param_importance_chart,
    optimization_trace_chart, monthly_returns_chart,
)


def render_results_tab():
    """Render the Results tab — load previous results or show current run."""
    st.subheader("📋 Results Browser")

    # Option 1: Show current run results
    if "backtest_results" in st.session_state:
        st.markdown("#### Current Run Results")
        _render_full_results(st.session_state["backtest_results"])
        return

    # Option 2: Load previous results
    st.markdown("No current run results. Load a previous run below.")

    # Discover optuna runs
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

    if not runs:
        st.info("No previous results found. Run a backtest first.")
        return

    run_labels = [f"{r[0]}: {r[1]}" for r in runs]
    selected = st.selectbox("Load Previous Run", options=run_labels, index=0)
    if st.button("📂 Load Selected", key="load_prev_btn"):
        idx = run_labels.index(selected)
        rtype, rname, rpath = runs[idx]
        data = _load_previous_run(rtype, rpath)
        if data:
            st.session_state["backtest_results"] = data
            st.rerun()


def _load_previous_run(rtype: str, rpath: Path) -> Optional[Dict[str, Any]]:
    """Load results from optuna_runs or hpo directory."""
    data: Dict[str, Any] = {"source": str(rpath)}

    # Param importances
    pi_file = rpath / "param_importances.json" if rpath.is_dir() else None
    if pi_file and pi_file.exists():
        with open(pi_file) as f:
            data["param_importances"] = json.load(f)

    # Best config
    best_file = rpath / "best_config.json" if rpath.is_dir() else None
    if best_file and best_file.exists():
        with open(best_file) as f:
            data["best_config"] = json.load(f)
    elif rtype == "hpo" and rpath.suffix == ".json":
        with open(rpath) as f:
            data["best_config"] = json.load(f)

    # Trial time stats
    ts_file = rpath / "optuna_trial_time_stats.csv" if rpath.is_dir() else None
    if ts_file and ts_file.exists():
        try:
            data["trial_time_stats"] = pd.read_csv(ts_file)
        except Exception:
            pass

    return data if data.get("best_config") or data.get("param_importances") else None


def _render_full_results(results: Dict[str, Any]):
    """Render all result windows."""
    metrics = results.get("metrics", {})
    equity_curve = results.get("equity_curve")
    monthly_df = results.get("monthly_df", pd.DataFrame())
    model_type = results.get("model_type", "unknown")

    # ── Window A: KPI Summary ──
    _render_kpi_cards(metrics, model_type)

    # ── Export Buttons ──
    _render_export_bar(results, model_type)

    # ── Window B: Equity Curve ──
    if equity_curve is not None and not equity_curve.empty:
        eq_fig = equity_curve_chart(equity_curve)
        st.plotly_chart(eq_fig, width="stretch")

    # ── Window C: Monthly Breakdown ──
    if not monthly_df.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📅 Monthly Breakdown")
            _render_monthly_table(monthly_df)
        with col2:
            st.plotly_chart(monthly_returns_chart(monthly_df), width="stretch")

    # ── Window D: HPO Diagnostics ──
    _render_hpo_diagnostics(results)

    # ── Window E: Best Config ──
    best_cfg = results.get("best_config")
    if best_cfg:
        with st.expander("🔧 Best Configuration (JSON)", expanded=False):
            st.json(best_cfg)


def _render_kpi_cards(metrics: Dict[str, Any], model_type: str):
    """Top-level KPI cards in a grid."""
    st.subheader(f"🎯 Backtest Results — {model_type.upper()}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sharpe = metrics.get("sharpe", np.nan)
        s = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
        st.metric("Sharpe Ratio", s)
    with c2:
        st.metric("Max Drawdown", f"{metrics.get('drawdown', 0):.1%}")
    with c3:
        st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.1f}%")
    with c4:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Total Trades", f"{metrics.get('trades', 0)}")
    with c6:
        st.metric("Active Rate", f"{metrics.get('active_rate', 0):.1%}")
    with c7:
        st.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.3f}")
    with c8:
        of = metrics.get("outperformance", 0)
        st.metric("Outperformance", f"{of:+.2%}")


def _render_monthly_table(monthly_df: pd.DataFrame):
    """Formatted monthly performance table."""
    display_cols = [c for c in [
        "month", "trades", "win_rate", "strategy_return",
        "directional_accuracy", "precision_macro", "f1_macro",
        "active_rate", "sharpe",
    ] if c in monthly_df.columns]
    if display_cols:
        formatted = monthly_df[display_cols].copy()
        for pct_col in ["win_rate", "directional_accuracy", "precision_macro", "active_rate"]:
            if pct_col in formatted.columns:
                formatted[pct_col] = formatted[pct_col].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "—"
                )
        for num_col in ["strategy_return", "f1_macro", "sharpe"]:
            if num_col in formatted.columns:
                formatted[num_col] = formatted[num_col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )
        st.dataframe(formatted, use_container_width=True, hide_index=True)
    else:
        st.dataframe(monthly_df, use_container_width=True)


def _render_export_bar(results: Dict[str, Any], model_type: str):
    """Export buttons: download metrics CSV and equity curve PNG."""
    st.markdown("#### 📤 Export Results")
    c1, c2, c3 = st.columns(3)

    # --- CSV export: metrics + monthly breakdown ---
    with c1:
        metrics = results.get("metrics", {})
        monthly_df = results.get("monthly_df", pd.DataFrame())
        parts = []
        if metrics:
            row = pd.DataFrame([{k: v for k, v in metrics.items()}])
            parts.append(("metrics", row))
        if not monthly_df.empty:
            parts.append(("monthly", monthly_df))
        if parts:
            buf = io.StringIO()
            for name, df in parts:
                buf.write(f"# {name}\n")
                df.to_csv(buf, index=False)
                buf.write("\n")
            csv_bytes = buf.getvalue().encode()
            st.download_button(
                "📊 Download Metrics CSV",
                data=csv_bytes,
                file_name=f"backtest_{model_type}_metrics.csv",
                mime="text/csv",
                key="export_csv_btn",
            )

    # --- PNG export: equity curve chart ---
    with c2:
        equity_curve = results.get("equity_curve")
        if equity_curve is not None and not equity_curve.empty:
            fig = equity_curve_chart(equity_curve)
            img_bytes = fig.to_image(format="png", width=1200, height=500, scale=2)
            st.download_button(
                "📈 Download Equity Curve PNG",
                data=img_bytes,
                file_name=f"backtest_{model_type}_equity.png",
                mime="image/png",
                key="export_png_btn",
            )

    # --- JSON export: best config ---
    with c3:
        best_cfg = results.get("best_config")
        if best_cfg:
            json_bytes = json.dumps(best_cfg, indent=2).encode()
            st.download_button(
                "🔧 Download Config JSON",
                data=json_bytes,
                file_name=f"backtest_{model_type}_config.json",
                mime="application/json",
                key="export_json_btn",
            )


def _render_hpo_diagnostics(results: Dict[str, Any]):
    """HPO parameter importance and optimization trace."""
    pi = results.get("param_importances")
    if pi:
        st.subheader("🔬 HPO Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(param_importance_chart(pi), width="stretch")
        with col2:
            trial_vals = results.get("trial_values", [])
            if trial_vals:
                st.plotly_chart(optimization_trace_chart(trial_vals), width="stretch")
            else:
                st.info("No trial history available for this run.")
