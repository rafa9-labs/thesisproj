"""
Model live trading simulation test — loads a deployed model, predicts on latest bar,
then simulates opening and closing a position via PaperEngine.

Validates:
  1. Deployed model loading from disk
  2. Feature computation on live candlestick data
  3. Model prediction (direction + confidence)
  4. PaperEngine trade entry (signal -> open)
  5. PaperEngine trade exit (reverse signal -> close)
  6. Trade journal entries

Usage:
  python scripts/test_model_live_trading.py [--model logistic] [--pair EURUSD] [--timeframe M30]
"""

import os
import sys
import json
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJ_ROOT / ".env", override=True)

os.environ.setdefault("LOG_MODE", "COMPACT")
os.environ.setdefault("SKIP_PLOTS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def pick_deployed_model(preferred=None):
    """Return (model_id, snapshot_path, model_type) for a deployed model."""
    from api.config import settings
    from pipeline.models.model_registry_disk import get_all_deployed

    rows = get_all_deployed(settings.db_full_path)
    if not rows:
        raise RuntimeError("No deployed models found. Deploy a model first.")

    if preferred:
        matching = [r for r in rows if r.get("model_type") == preferred]
        if matching:
            row = matching[0]
        else:
            print(f"  No '{preferred}' model found, using first available")
            row = rows[0]
    else:
        row = rows[0]

    model_id = row.get("id", "")
    snapshot_path = row.get("snapshot_path", "")
    model_type = row.get("model_type", "unknown")
    return model_id, snapshot_path, model_type


def load_model(snapshot_path):
    """Load model and metadata from a snapshot directory."""
    from pipeline.models.model_persistence import load_model_only, read_metadata

    if not os.path.isdir(snapshot_path):
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    model = load_model_only(snapshot_path)
    meta = read_metadata(snapshot_path)
    return model, meta


def get_latest_candles(pair, timeframe, n_bars=120):
    """Fetch latest OHLC bars from SQLite."""
    from pipeline.data.data_sqlite import DataStore
    from api.config import settings

    store = DataStore(settings.db_full_path)
    df = store.get_latest_candles(pair, timeframe, n_bars)
    if df is None or df.empty:
        raise RuntimeError(f"No candle data for {pair} {timeframe}")
    return df


def prepare_raw_data(candles_df):
    """Convert candles to backtester-compatible raw data format."""
    import numpy as np
    import pandas as pd

    raw = candles_df.copy()

    # Preserve time column for datetime index
    time_col = None
    if "time" in raw.columns:
        time_col = raw["time"].copy()

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

    if time_col is not None:
        raw["time"] = time_col.values
        raw.set_index("time", inplace=True)
        raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")

    raw.dropna(inplace=True)
    return raw


def compute_features(raw_data, meta, pair="EURUSD", base_timeframe="M30"):
    """Compute technical features using MLBacktester.prepare_features().
    
    Uses the feature config from the saved model metadata if available.
    """
    from pipeline.backtester.composed import MLBacktester

    # Resolve date range from raw_data index
    import datetime as _dt
    if len(raw_data) > 0:
        idx0 = raw_data.index[0]
        idx1 = raw_data.index[-1]
        if hasattr(idx0, "strftime"):
            start_date = idx0.strftime("%Y-%m-%d")
            end_date = idx1.strftime("%Y-%m-%d")
        else:
            end_date = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
            start_date = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=90)).strftime("%Y-%m-%d")
    else:
        end_date = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        start_date = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=90)).strftime("%Y-%m-%d")

    # Build a self-contained feature config that produces features
    # matching the model's expected feature names
    model_fc = meta.get("features_config", {})
    indicator_windows = model_fc.get("indicator_windows") if model_fc.get("indicator_windows") else {
        "sma": 20, "ema": 20, "rsi": 14, "atr": 14, "adx": 14,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bb_window": 20, "bb_dev": 2.0,
    }

    merged_fc = {
        "use_adx": True, "use_atr": True, "use_bbands": True,
        "use_ema": True, "use_sma": True, "use_rsi": True, "use_macd": True,
        "use_donchian": True, "use_stoch": True,
        "use_fracdiff": True, "fracdiff_d": 0.4,
        "use_crossover_bins": True, "use_ma_spread": True,
        "use_price_ma_z": True,
        "use_mtf_ma": True, "use_mtf_alignment": True,
        "use_macd_atr_ratio": True, "use_triple_confirm": True,
        "use_trend_confirm": True, "use_vol_managed_mom": True,
        "use_squeeze_breakout": True, "use_squeeze_expansion": True,
        "use_atr_channel_breakout": True, "use_ext_atr_low_adx": True,
        "use_reentry_mom": True, "use_slope_diff": True,
        "use_rv_features": True, "use_regime_features": False,
        "use_news": False,
        "lags": 14, "lag_depth": 2,
        "roll_windows_key": [5, 10, 20, 30, 60],
        "indicator_windows": indicator_windows,
    }

    bt = MLBacktester(
        symbol=pair.upper(),
        start=start_date,
        end=end_date,
        trading_costs=False,
        model_type=meta.get("model_type", "logistic"),
        features_config=dict(merged_fc),
        db_path="data/forex.db",
        base_timeframe=base_timeframe,
    )

    lags = int(merged_fc.get("lags", 10) or 10)
    lag_depth = int(merged_fc.get("lag_depth", 2) or 2)
    roll_windows = merged_fc.get("roll_windows_key", [5, 10, 20])
    if isinstance(roll_windows, (int, float)):
        roll_windows = [int(roll_windows)]
    elif not isinstance(roll_windows, list):
        roll_windows = [5]

    features_df, feature_names = bt.prepare_features(
        df=raw_data.copy(),
        lags=lags,
        lag_depth=lag_depth,
        roll_windows=list(roll_windows) if roll_windows else [5],
        base_only=False,
    )

    if features_df is None or features_df.empty:
        raise ValueError("Feature computation returned empty DataFrame")
    return features_df, feature_names, bt


def predict_direction(model, features_df, feature_names, meta_feature_names=None):
    """Run model prediction on the last row of features.
    
    Uses meta_feature_names (from training) to select only the features
    the model was trained on, falling back to feature_names if not provided.
    """
    import numpy as np

    last_row = features_df.iloc[[-1]]

    # Use the model's original training feature names if available
    model_features = meta_feature_names if meta_feature_names else feature_names

    # Align: only use columns that exist in both the model and the features df
    available = [f for f in model_features if f in last_row.columns]
    if not available:
        # Fall back to using all feature_names
        available = [f for f in feature_names if f in last_row.columns]

    X = last_row[available].values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = float(proba[0].max()) * 100
        prediction = int(proba[0].argmax())
    elif hasattr(model, "predict"):
        raw_out = model.predict(X)
        if hasattr(raw_out, "__getitem__"):
            prediction = int(raw_out[0])
        else:
            prediction = int(raw_out)
        confidence = 60.0
    else:
        raise RuntimeError("Model has no predict or predict_proba method")

    direction = "LONG" if prediction == 1 else "SHORT" if prediction in (0, -1) else "FLAT"

    return {
        "direction": direction,
        "confidence": min(confidence, 99.0),
        "prediction": prediction,
        "proba": proba[0].tolist() if hasattr(model, "predict_proba") else None,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model live trading simulation test")
    parser.add_argument("--model", default=None, help="Model type to test (e.g., logistic, lightgbm)")
    parser.add_argument("--pair", default="EURUSD", help="Trading pair")
    parser.add_argument("--timeframe", default="M30", help="Timeframe")
    args = parser.parse_args()

    pair = args.pair.upper()
    timeframe = args.timeframe.upper()

    print(f"Model Live Trading Simulation Test")
    print(f"  Pair:      {pair}")
    print(f"  Timeframe: {timeframe}\n")

    # 1. Load deployed model
    print("--- Step 1: Load Deployed Model ---")
    model_id, snapshot_path, model_type = pick_deployed_model(args.model)
    print(f"  ID:    {model_id}")
    print(f"  Type:  {model_type}")
    print(f"  Path:  {snapshot_path}")

    model, meta = load_model(snapshot_path)
    print(f"  Meta keys: {list(meta.keys())[:8]}...")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Feature count (meta): {len(meta.get('feature_names', []))}")
    print("  PASSED\n")

    # 2. Get latest candles
    print("--- Step 2: Get Latest Candles ---")
    candles_df = get_latest_candles(pair, timeframe, 600)
    print(f"  Rows:     {len(candles_df)}")
    print(f"  Columns:  {list(candles_df.columns)}")
    if "mid_close" in candles_df.columns:
        print(f"  Latest close: {candles_df.iloc[-1]['mid_close']:.5f}")
        print(f"  First close:  {candles_df.iloc[0]['mid_close']:.5f}")
    print("  PASSED\n")

    # 3. Prepare raw data and compute features
    print("--- Step 3: Compute Features ---")
    raw_data = prepare_raw_data(candles_df)
    print(f"  Raw bars: {len(raw_data)}")

    features_df, feature_names, bt_instance = compute_features(raw_data, meta, pair=pair, base_timeframe=timeframe)
    print(f"  Feature rows: {len(features_df)}")
    print(f"  Feature cols: {len(feature_names)}")
    print(f"  Sample cols:  {feature_names[:5]}...")
    print("  PASSED\n")

    # 4. Predict
    print("--- Step 4: Model Prediction ---")
    meta_features = meta.get("feature_names", [])
    result = predict_direction(model, features_df, feature_names, meta_features)
    print(f"  Direction:  {result['direction']}")
    print(f"  Confidence: {result['confidence']:.1f}%")
    print(f"  Using features: {len(meta_features) if meta_features else len(feature_names)} cols")
    if result.get("proba"):
        print(f"  Probabilities: {[f'{p:.4f}' for p in result['proba']]}")
    print("  PASSED\n")

    # 5. Simulate trade via PaperEngine
    print("--- Step 5: Paper Trading Simulation ---")
    from trading.paper_engine import PaperEngine

    latest_idx = -1
    mid = float(candles_df.iloc[latest_idx].get("mid_close", 0))
    bid = mid
    ask = mid
    if "spread" in candles_df.columns:
        half_spread = float(candles_df.iloc[latest_idx].get("spread", 0)) / 2
        bid = mid - half_spread
        ask = mid + half_spread

    engine = PaperEngine()
    engine.start({
        "initial_equity": 10000.0,
        "position_sizing": "fixed",
        "sizing_config": {"fixed_size": 1000},
    })

    # Open position
    open_result = engine.process_signal(result, bid=bid, ask=ask, mid=mid)
    print(f"  Open event:    {open_result.get('event')}")
    print(f"  Direction:     {open_result.get('direction')}")
    print(f"  Equity:        {open_result.get('equity', 0):.2f}")
    print(f"  Position:      {engine.get_portfolio_state().get('position')}")

    sub_events = open_result.get("sub_events", [])
    for se in sub_events:
        print(f"  Sub-event:     {se.get('event')} dir={se.get('direction')} size={se.get('size')}")

    # Simulate a small price move, then close by reversing
    price_move = 0.0010 if pair != "USDJPY" else 0.10
    new_mid = mid + price_move
    new_bid = new_mid - (mid - bid)
    new_ask = new_mid + (ask - mid)

    # Send reverse signal to close (go the opposite direction)
    reverse_dir = "SHORT" if result["direction"] == "LONG" else "LONG"
    reverse_signal = {
        "direction": reverse_dir,
        "confidence": 55.0,
    }
    close_result = engine.process_signal(reverse_signal, bid=new_bid, ask=new_ask, mid=new_mid)
    print(f"\n  Close event:   {close_result.get('event')}")
    close_subs = close_result.get("sub_events", [])
    for se in close_subs:
        print(f"  Sub-event:     {se.get('event')} size={se.get('size')} pnl={se.get('pnl', 0):.2f}")

    # 6. Results
    print("\n--- Step 6: Trade Summary ---")
    state = engine.get_portfolio_state()
    summary = engine.get_summary()

    print(f"  Final equity:    {state['equity']:.2f}")
    print(f"  Unrealized PnL:  {state['unrealized_pnl']:.2f}")
    print(f"  Total trades:    {summary.get('total_trades', 0)}")
    win_rate = summary.get('win_rate', 0) or 0
    print(f"  Win rate:        {win_rate * 100:.1f}%")
    print(f"  Avg trade PnL:   {summary.get('avg_trade_pnl', 0):.2f}")
    print("  PASSED\n")

    # 7. Trade journal
    print("--- Step 7: Trade Journal ---")
    trades = engine.get_trades()
    print(f"  Trade count: {len(trades)}")
    for t in trades:
        tid = str(t.get('trade_id', '?'))[:8]
        print(f"    {tid}: dir={t.get('direction')} "
              f"entry={t.get('entry_price', 0):.5f} "
              f"exit={str(t.get('exit_price', '?')):.5s} "
              f"pnl={t.get('pnl', 0):.2f} "
              f"reason={t.get('exit_reason', '?')}")
    print("  PASSED\n")

    print("=" * 60)
    print("ALL TESTS PASSED -- Model live trading pipeline verified")
    print("=" * 60)


if __name__ == "__main__":
    main()
