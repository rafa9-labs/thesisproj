"""Forward test engine — evaluate a saved model on any date range without retraining.

Loads a trained snapshot, computes features using the saved feature config,
predicts with the frozen model, simulates trades, and returns metrics.
No HPO, no training, no Optuna.

Also exports ``generate_forecast_errors()`` for future Diebold-Mariano testing.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from pipeline.models.model_persistence import load_snapshot, read_metadata
from pipeline.execution.position_sizing import (
    SizingConfig,
    SizingMethod,
    SizingState,
    compute_size,
    update_state,
)


# ──────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────
def _load_m30_data(pair: str, start: str, end: str, db_path: str = "data/forex.db", timeframe: str = "M30") -> pd.DataFrame:
    """Load base candles, remap columns to backtester format (price, high, low, etc.)."""
    from pipeline.data.data_sqlite import DataStore

    store = DataStore(db_path)
    raw = store.get_candles(pair.upper(), timeframe, start=start, end=end)
    if raw is None or raw.empty:
        raise FileNotFoundError(
            f"No {timeframe} candles for {pair} in [{start}, {end}]. Run data download first."
        )

    raw.rename(columns={
        "mid_close": "price",
        "mid_high": "high",
        "mid_low": "low",
    }, inplace=True)

    keep = [c for c in ["price", "high", "low", "spread"] if c in raw.columns]
    raw = raw[keep].copy()

    raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))
    for col in ("price", "high", "low", "spread", "returns"):
        if col in raw.columns:
            raw[col] = raw[col].astype("float32")

    if "time" in raw.columns:
        raw.set_index("time", inplace=True)
        raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")

    raw.dropna(inplace=True)
    if raw.empty:
        raise ValueError(f"No valid bars for {pair} {timeframe} in [{start}, {end}]")
    return raw


# ──────────────────────────────────────────────────────
#  FEATURE COMPUTATION
# ──────────────────────────────────────────────────────
def _compute_features_from_data(
    raw_data: pd.DataFrame,
    features_config: dict,
    pair: str = "EURUSD",
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    base_timeframe: str = "M30",
) -> pd.DataFrame:
    from pipeline.backtester.composed import MLBacktester

    merged_fc = {
        "use_adx": True, "use_atr": True, "use_bbands": True,
        "use_ema": True, "use_sma": True, "use_rsi": True, "use_macd": True,
        "use_donchian": True, "use_stoch": False, "use_sar": False,
        "use_fracdiff": True, "fracdiff_d": 0.4,
        "use_crossover_bins": True, "use_ma_spread": True,
        "use_price_ma_z": True, "use_indicator_states": False,
        "use_mtf_ma": True, "use_mtf_alignment": True,
        "use_macd_atr_ratio": True, "use_triple_confirm": True,
        "use_trend_confirm": True, "use_vol_managed_mom": True,
        "use_squeeze_breakout": True, "use_squeeze_expansion": True,
        "use_atr_channel_breakout": True, "use_ext_atr_low_adx": False,
        "use_reentry_mom": True, "use_slope_diff": True,
        "use_rv_features": True,
        "use_regime_features": False,
        "use_news": False,
        "lags": 14, "lag_depth": 1,
        "roll_windows_key": [5, 10, 20, 30, 60, 160],
    }
    merged_fc.update(features_config)
    merged_fc["use_news"] = False

    bt = MLBacktester(
        symbol=pair.upper(),
        start=start_date,
        end=end_date,
        trading_costs=False,
        model_type=merged_fc.get("model_type", "logistic"),
        features_config=dict(merged_fc),
        db_path="data/forex.db",
        base_timeframe=base_timeframe,
    )

    lags = int(merged_fc.get("lags", 10))
    lag_depth = int(merged_fc.get("lag_depth", 1))
    roll_windows = merged_fc.get("roll_windows_key", [5])
    if isinstance(roll_windows, (int, float)):
        roll_windows = [int(roll_windows)]
    elif not isinstance(roll_windows, list):
        roll_windows = [5]

    features_df, _feature_names_list = bt.prepare_features(
        df=raw_data.copy(),
        lags=lags,
        lag_depth=lag_depth,
        roll_windows=list(roll_windows) if roll_windows else [5],
        base_only=False,
    )

    if features_df is None or features_df.empty:
        raise ValueError("Feature computation returned empty DataFrame")
    return features_df, _feature_names_list


# ──────────────────────────────────────────────────────
#  EXECUTION SIMULATION
# ──────────────────────────────────────────────────────
def _simulate_execution(
    prediction_df: pd.DataFrame,
    initial_equity: float = 10_000.0,
    trading_costs: bool = True,
    sizing_method: str = "fixed",
    sizing_cfg: dict | None = None,
) -> dict:
    """Simple bar-by-bar execution loop with position sizing and trading costs."""
    n_bars = len(prediction_df)
    if n_bars == 0:
        return {
            "equity_curve": [],
            "trades": [],
            "sharpe": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

    sizing_config = SizingConfig(method=sizing_method or SizingMethod.FIXED)
    if sizing_cfg:
        for k, v in sizing_cfg.items():
            if hasattr(sizing_config, k):
                setattr(sizing_config, k, v)

    state = SizingState(equity=initial_equity)

    preds = prediction_df["pred"].fillna(0).astype(int).values
    rets = prediction_df["returns"].fillna(0.0).astype(float).values
    spread = (
        prediction_df["spread"].fillna(0).astype(float).values
        if "spread" in prediction_df.columns
        else np.zeros(n_bars)
    )

    confidence = (
        prediction_df["confidence"].fillna(0.5).astype(float).values
        if "confidence" in prediction_df.columns
        else np.full(n_bars, 0.5)
    )

    atr_vals = np.zeros(n_bars)
    bar_vol = np.zeros(n_bars)

    equity = float(initial_equity)
    equity_curve = [{"bar": 0, "equity": equity}]
    trades = []
    peak_equity = equity
    max_dd = 0.0
    in_position = 0  # 0 = flat, +1 = long, -1 = short

    for i in range(n_bars):
        sig = int(preds[i])
        bar_return = float(rets[i])
        bar_spread = float(spread[i])

        slippage = bar_spread * 0.5 if trading_costs and sig != 0 else 0.0
        size = float(compute_size(state, float(bar_vol[i]), float(atr_vals[i]), sizing_config))

        pnl = 0.0
        if in_position != 0 and sig != in_position:
            pnl = bar_return * in_position * size - slippage
            is_win = pnl > 0
            update_state(state, pnl, is_win)
            trades.append({
                "direction": in_position,
                "exit_bar": i,
                "pnl": round(pnl, 6),
                "is_win": is_win,
            })
            in_position = sig if sig != 0 else 0
        elif in_position != 0:
            pnl = bar_return * in_position * size
        elif sig != 0:
            in_position = sig

        equity += pnl
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / max(peak_equity, 1.0)
        max_dd = max(max_dd, dd)

        equity_curve.append({"bar": i + 1, "equity": equity})

    total_return = (equity - initial_equity) / initial_equity
    ret_series = pd.Series([
        equity_curve[j]["equity"] / max(equity_curve[j - 1]["equity"], 1.0) - 1.0
        for j in range(1, len(equity_curve))
    ]).dropna()

    if len(ret_series) > 1 and ret_series.std() > 0:
        # Annualize with the actual bar frequency (M30/H1/...), not daily 252.
        from pipeline.metrics.metrics_eval import estimate_frequency_per_year
        freq = float(estimate_frequency_per_year(prediction_df.index))
        sharpe = float(ret_series.mean() / ret_series.std() * np.sqrt(max(freq, 1.0)))
    else:
        sharpe = 0.0

    win_trades = [t for t in trades if t["is_win"]]
    win_rate = len(win_trades) / max(len(trades), 1)

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "sharpe": round(sharpe, 4),
        "total_return_pct": round(total_return * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 4),
        "total_trades": len(trades),
        "final_equity": round(equity, 2),
    }


# ──────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────
def run_forward_test(
    snapshot_path: str,
    pair: str,
    timeframe: str = "M30",
    start_date: str = "",
    end_date: str = "",
    position_sizing: str = "fixed",
    sizing_config: dict | None = None,
    trading_costs: bool = True,
) -> dict:
    """Load a saved model and run a forward-only simulation on a date range.

    Parameters
    ----------
    snapshot_path : str
        Path to snapshot directory (e.g. deployed_models/logistic_20260520T120000Z)
    pair : str
        Currency pair (EURUSD)
    timeframe : str
        Target timeframe (M15, M30, H1, H4). MTF features adapt based on this.
    start_date, end_date : str
        Date range for the forward test (e.g. "2026-05-01")
    position_sizing : str
        One of: fixed, fixed_fractional, kelly, atr, vol_target
    sizing_config : dict or None
        Optional SizingConfig field overrides
    trading_costs : bool
        Apply spread + slippage simulation

    Returns
    -------
    dict
        {model_type, snapshot_id, pair, timeframe, start_date, end_date,
         train_range, metrics: {sharpe, total_return_pct, max_drawdown_pct,
         win_rate, total_trades}, equity_curve, trades, diagnostics}
    """
    snapshot = load_snapshot(snapshot_path)
    model = snapshot.get("model")
    scaler = snapshot.get("scaler")
    imputer = snapshot.get("imputer")
    metadata = snapshot.get("metadata", {})

    if model is None:
        raise ValueError(f"No model found in snapshot: {snapshot_path}")

    model_type = metadata.get("model_type", "unknown")
    features_config = metadata.get("features_config", {})
    feature_names = metadata.get("feature_names", [])
    train_start = metadata.get("train_start", "?")
    train_end = metadata.get("train_end", "?")

    raw_data = _load_m30_data(pair, start_date, end_date, timeframe=timeframe)

    # Overlap guard: the forward window must start AFTER the snapshot's
    # training range (plus feature warm-up) or the test is not out-of-sample.
    if start_date and train_end and train_end != "?":
        try:
            train_end_ts = pd.to_datetime(train_end, utc=True)
            fwd_start_ts = pd.to_datetime(start_date, utc=True)
            if fwd_start_ts <= train_end_ts:
                import warnings
                warnings.warn(
                    f"Forward window starts at {start_date}, which overlaps the "
                    f"snapshot train range (train_end={train_end}). Results are "
                    "NOT out-of-sample."
                )
        except Exception:
            pass

    features_df, _computed_names = _compute_features_from_data(raw_data, features_config, pair=pair, start_date=start_date, end_date=end_date, base_timeframe=timeframe)

    exclude_cols = {"time", "target", "side", "returns", "spread", "label"}
    if feature_names:
        feature_cols = [c for c in feature_names if c in features_df.columns]
    else:
        feature_cols = []

    if not feature_cols or len(feature_cols) < max(3, len(feature_names) // 2):
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    if not feature_cols:
        raise RuntimeError("No feature columns found after computing features")

    if feature_names and len(feature_names) > 0:
        X_aligned = pd.DataFrame(0.0, index=features_df.index, columns=feature_names)
        for col in feature_names:
            if col in features_df.columns:
                X_aligned[col] = features_df[col].fillna(0.0)
        X = X_aligned.astype(np.float64)
    else:
        X = features_df[feature_cols].fillna(0.0).astype(np.float64)

    if imputer is not None:
        try:
            X = pd.DataFrame(
                imputer.transform(X.values), index=X.index, columns=X.columns
            )
        except Exception:
            pass

    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception:
            # NEVER fit a scaler on forward data (look-ahead). Fall back to
            # unscaled features with a loud warning.
            import warnings
            warnings.warn(
                "Saved scaler failed to transform forward features; using "
                "unscaled inputs. Predictions may be unreliable."
            )
            X_scaled = X.values
    else:
        X_scaled = X.values

    try:
        proba = model.predict_proba(X_scaled)
    except Exception:
        try:
            raw_preds = model.predict(X_scaled)
            proba = np.zeros((len(raw_preds), 3))
            for i, c in enumerate(raw_preds):
                cls = min(max(int(c) + 1, 0), 2)
                proba[i, cls] = 1.0
        except Exception:
            raise RuntimeError(
                f"Prediction failed for {model_type} on {X_scaled.shape} features"
            ) from None

    classes = np.argmax(proba, axis=1) - 1
    confidences = np.max(proba, axis=1)

    # Align predictions to raw_data index
    min_len = min(len(raw_data), len(classes))
    pred_df = raw_data.iloc[:min_len].copy()
    pred_df["pred"] = classes[:min_len]
    pred_df["confidence"] = confidences[:min_len]
    if "spread" not in pred_df.columns:
        pred_df["spread"] = 0.0

    sim_result = _simulate_execution(
        pred_df,
        initial_equity=10_000.0,
        trading_costs=trading_costs,
        sizing_method=position_sizing,
        sizing_cfg=sizing_config,
    )

    diagnostics: dict = {}
    try:
        diagnostics["prediction_histogram"] = [
            {"bin_start": round(i * 0.1, 1), "bin_end": round((i + 1) * 0.1, 1),
             "count": int(np.sum((confidences >= i * 0.1) & (confidences < (i + 1) * 0.1)))}
            for i in range(10)
        ]
    except Exception:
        pass

    snap_id = os.path.basename(snapshot_path.rstrip("/\\"))

    return {
        "model_type": model_type,
        "snapshot_id": snap_id,
        "pair": pair,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "train_range": f"{train_start} -> {train_end}" if train_start else "unknown",
        "metrics": {
            "sharpe": sim_result["sharpe"],
            "total_return_pct": sim_result["total_return_pct"],
            "max_drawdown_pct": sim_result["max_drawdown_pct"],
            "win_rate": sim_result["win_rate"],
            "total_trades": sim_result["total_trades"],
        },
        "equity_curve": sim_result["equity_curve"],
        "trades": sim_result["trades"],
        "diagnostics": diagnostics,
    }


def generate_forecast_errors(forward_test_result: dict) -> dict:
    """Extract per-bar forecast errors for DM/SPA testing.

    Returns {errors, benchmark_errors} as numpy arrays.
    """
    trades = forward_test_result.get("trades", [])
    if not trades:
        return {"errors": np.array([]), "benchmark_errors": np.array([])}

    errors = []
    bench_errors = []
    for t in trades:
        actual = float(t.get("pnl", 0.0))
        direction = int(t.get("direction", 0))
        err = (actual - direction * 0.0001) ** 2
        bench_err = actual ** 2
        errors.append(err)
        bench_errors.append(bench_err)

    return {
        "errors": np.array(errors, dtype=np.float64),
        "benchmark_errors": np.array(bench_errors, dtype=np.float64),
    }
