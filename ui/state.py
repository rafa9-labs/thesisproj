"""
State management adapter — bridges Streamlit UI to pipeline/ backend.

Replaces init-proj's src.core.* imports with our config + MLBacktester.
Uses @st.cache_resource for singletons, @st.cache_data for data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Optional, Tuple, Dict, Any, List
from copy import deepcopy
from pathlib import Path

from config import Settings, get_settings, apply_global_env
from models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# Available models for UI (from registry)
AVAILABLE_MODELS = sorted(MODEL_REGISTRY.keys())

# CSV data files available
DATA_FILES = {
    "EURUSD_H1":  {"path": "csv_data/EURUSD_10_years_H1_OANDA.csv",  "tf": "H1"},
    "EURUSD_H4":  {"path": "csv_data/EURUSD_10_years_H4_OANDA.csv",  "tf": "H4"},
    "EURUSD_M30": {"path": "csv_data/EURUSD_10_years_M30_OANDA.csv", "tf": "M30"},
}


class AppState:
    """Centralized state management with Streamlit caching."""

    @staticmethod
    @st.cache_resource
    def get_settings() -> Settings:
        settings = get_settings()
        apply_global_env(settings)
        return settings

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner="Loading CSV data...")
    def load_csv_data(_csv_path: str) -> pd.DataFrame:
        logger.info(f"Loading CSV: {_csv_path}")
        csv_path = Path(_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        try:
            df = pd.read_csv(csv_path, engine="pyarrow")
        except Exception:
            df = pd.read_csv(csv_path, engine="c")
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        dt_col = None
        for col_name in ("date", "time", "datetime", "timestamp", "index"):
            if col_name in df.columns:
                dt_col = col_name
                break
        if dt_col:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
            df = df.set_index(dt_col).sort_index()
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.set_index(df.columns[0]).sort_index()
        # Map common OANDA column names to canonical names
        col_map = {"mid_open": "open", "mid_high": "high", "mid_low": "low", "mid_close": "close"}
        for original, canonical in col_map.items():
            if original in df.columns and canonical not in df.columns:
                df = df.rename(columns={original: canonical})
        price_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        for c in price_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "close" in df.columns and "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()
        df = df.dropna(subset=["close"])
        logger.info(f"Loaded {len(df)} bars from {csv_path.name}")
        return df

    @staticmethod
    def run_backtest(
        model_type: str,
        csv_path: str,
        n_months: int = 3,
        n_trials: int = 5,
        seed: int = 42,
        trading_costs: bool = True,
        trial_callback=None,
    ) -> Dict[str, Any]:
        """Run a full walk-forward backtest via MLBacktester."""
        from pipeline.backtester.composed import MLBacktester

        feat_path = Path("configs/feature_config.json")
        with open(feat_path, "r") as f:
            features_config = json.load(f)
        features_config["run_seed"] = seed
        features_config.pop("eval_seed_sets", None)
        features_config.pop("test_warmup_bars", None)
        features_config.setdefault("enforce_day1_start", True)
        if features_config.get("session_filter_mode") is None:
            features_config["session_filter_mode"] = "both"

        df_preview = AppState.load_csv_data(csv_path)
        start_date = str(df_preview.index[0])
        end_date = str(df_preview.index[-1])

        base_config = {
            "model_type": model_type,
            "rep": 1,
            "n_trials": n_trials,
            "n_startup_trials": max(1, n_trials // 2),
        }

        bt = MLBacktester(
            symbol="EURUSD",
            start=start_date,
            end=end_date,
            trading_costs=trading_costs,
            features_config=features_config,
        )
        try:
            df_sim = bt.real_trading_simulation(
                deepcopy(base_config),
                models_to_test=[model_type],
                months=n_months,
            )
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            raise

        results = _extract_backtest_results(bt, df_sim, model_type)
        try:
            if hasattr(bt, "free") and callable(bt.free):
                bt.free(release_data=True)
        except Exception:
            pass
        return results


def _extract_backtest_results(bt, df_sim: pd.DataFrame, model_type: str) -> Dict[str, Any]:
    monthly_df = df_sim if df_sim is not None and not df_sim.empty else pd.DataFrame()
    equity_curve = pd.Series(dtype=np.float32)
    bar_concat = getattr(bt, "bar_concat", None)
    if bar_concat is not None and not getattr(bar_concat, "empty", True):
        if "cstrategy_cont" in bar_concat.columns:
            equity_curve = bar_concat["cstrategy_cont"].astype(np.float32)
    metrics = _compute_aggregate_metrics(monthly_df, equity_curve)
    return {"metrics": metrics, "equity_curve": equity_curve, "monthly_df": monthly_df, "model_type": model_type}


def _compute_aggregate_metrics(monthly_df: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
    empty = {
        "sharpe": np.nan, "drawdown": 0.0, "win_rate": 0.0,
        "total_return_pct": 0.0, "trades": 0, "active_rate": 0.0,
        "directional_accuracy": 0.0, "precision_macro": 0.0,
        "f1_macro": 0.0, "strategy_volatility": 0.0,
        "cstrategy": 1.0, "geo_mean_ann": 0.0, "outperformance": 0.0,
        "return_per_trade": 0.0, "profit_per_hit": 0.0,
    }
    if monthly_df.empty:
        return empty
    m = {}
    if not equity_curve.empty and len(equity_curve) > 1:
        final_eq = float(equity_curve.iloc[-1])
        m["cstrategy"] = final_eq
        m["total_return_pct"] = (final_eq - 1.0) * 100.0
        cum_max = np.maximum.accumulate(equity_curve.values)
        drawdowns = (equity_curve.values - cum_max) / np.where(cum_max > 0, cum_max, 1.0)
        m["drawdown"] = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        if "strategy_return" in monthly_df.columns:
            rets = monthly_df["strategy_return"].dropna().values
            if len(rets) > 2:
                ann_ret = np.mean(rets) * 12
                ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
                m["sharpe"] = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
                m["strategy_volatility"] = float(ann_vol)
            else:
                m["sharpe"] = np.nan; m["strategy_volatility"] = 0.0
        else:
            m["sharpe"] = np.nan; m["strategy_volatility"] = 0.0
    else:
        m["cstrategy"] = 1.0; m["total_return_pct"] = 0.0; m["drawdown"] = 0.0
        m["sharpe"] = np.nan; m["strategy_volatility"] = 0.0

    if "strategy_return" in monthly_df.columns:
        rets = monthly_df["strategy_return"].dropna().values
        m["win_rate"] = float((rets > 0).mean()) if len(rets) > 0 else 0.0
    else:
        m["win_rate"] = 0.0

    for col, key in [("trades","trades"),("active_rate","active_rate"),("directional_accuracy","directional_accuracy"),
                     ("precision_macro","precision_macro"),("f1_macro","f1_macro"),("profit_per_hit","profit_per_hit")]:
        if col in monthly_df.columns:
            vals = monthly_df[col].dropna()
            m[key] = float(vals.mean()) if not vals.empty else 0.0
            if key == "trades": m[key] = int(vals.sum()) if not vals.empty else 0
        else:
            m[key] = 0

    m["return_per_trade"] = m["total_return_pct"] / m["trades"] if m.get("trades", 0) > 0 else 0.0
    if "creturns" in monthly_df.columns:
        bh = monthly_df["creturns"].dropna()
        bh_final = float(bh.iloc[-1]) if not bh.empty else 1.0
        m["outperformance"] = m.get("cstrategy", 1.0) - bh_final
    else:
        m["outperformance"] = 0.0
    return m