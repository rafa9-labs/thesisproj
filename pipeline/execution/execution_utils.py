"""Walk-forward execution helpers backported from utilsNoWFO.py.

Phase 4.2b -- test-bar alignment, warmup, trade logs, day-1 anchor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from pipeline.metrics.metrics_extra import compute_rolling_hit_rate


def first_tradable_test_bar(index, month_start_ts):
    """
    Return the first timestamp in `index` that is >= month_start_ts.
    If none exists, return None.
    """
    # Be tolerant: callers sometimes pass a DataFrame/Series by mistake.
    if hasattr(index, "index") and not isinstance(index, (pd.Index, pd.DatetimeIndex)):
        index = index.index

    if index is None:
        return None

    idx = pd.DatetimeIndex(index)
    ts = pd.to_datetime(month_start_ts)

    # TZ-safe comparison: align ts to idx timezone if needed
    if getattr(idx, "tz", None) is not None:
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(idx.tz)
        else:
            try:
                ts = ts.tz_convert(idx.tz)
            except Exception:
                # If tz_convert fails for any reason, fall back without conversion
                pass

    pos = idx.searchsorted(ts, side="left")
    if pos < len(idx):
        return idx[pos]

    return None


def compute_required_test_warmup_bars(params: dict) -> int:
    """
    Conservative pre-seed length from your tuning params.
    Considers lags, lag_depth, roll windows, long RV, rolling scaler, fracdiff, and DQN window.
    """
    p = params or {}
    lags = int(p.get("lags", p.get("lags_range", 0)) or 0)
    lag_depth = int(p.get("lag_depth", 1) or 1)
    need_lags = max(lags, lags * lag_depth)

    # roll_windows can be list or "5,10,20"
    rw = []
    if "roll_windows" in p and p.get("roll_windows"):
        if isinstance(p["roll_windows"], str):
            rw = [int(x) for x in p["roll_windows"].split(",") if str(x).strip().isdigit()]
        elif isinstance(p["roll_windows"], (list, tuple)):
            rw = [int(x) for x in p["roll_windows"] if not isinstance(x, (list, tuple))]
    elif "roll_windows_key" in p and p.get("roll_windows_key"):
        rw = [int(x) for x in str(p["roll_windows_key"]).split(",") if str(x).strip().isdigit()]
    need_roll = max(rw) if rw else 0

    rv_long = int(p.get("rv_window_long", 0) or 0)

    scaler_win = int(p.get("scaler_window", 0) or 0) if p.get("use_rolling_scaler") else 0

    fracdiff_warmup = int(p.get("fracdiff_warmup", 0) or 0)
    if not fracdiff_warmup and p.get("use_fracdiff") and float(p.get("fracdiff_d", 0)) > 0:
        fracdiff_warmup = 200  # safe default

    # deep windows / DQN
    mt = str(p.get("model_type", "")).lower()
    need_seq = 0
    if mt in ("cnn", "lstm", "transformer"):
        # For seq models, lags ~ effective window length
        need_seq = max(need_seq, lags)
    if mt == "dqn":
        dqn_cfg = p.get("dqn_config", {}) or {}
        need_seq = max(need_seq, int(dqn_cfg.get("window", lags) or 0))

    # --- NEW: generic window / lookback sweep ---
    generic_win = 0
    for k, v in p.items():
        key = str(k).lower()
        # Do NOT treat training-budget keys as feature warmup requirements
        if "train" in key and "window" in key:
            continue
        if any(tok in key for tok in ("window", "lookback", "period")):
            try:
                val = int(v)
            except Exception:
                continue
            generic_win = max(generic_win, val)

    # Final warm-up requirement = max of all known windows
    warm = max(
        need_lags,
        need_roll,
        rv_long,
        scaler_win,
        fracdiff_warmup,
        need_seq,
        generic_win,
    )

    if warm <= 0:
        return 0

    # Add a safety margin so month-start is definitely fully warmed
    margin = max(10, int(0.10 * warm))
    return int(warm + margin)


def enforce_day1_eval_anchor(index, month_start_ts):
    """
    Return the first index on the SAME calendar day as month_start_ts, after
    session filtering. If that day has no bars (rare), fall back to the first
    tradable bar >= month_start_ts.
    """
    import pandas as pd
    ts = pd.to_datetime(month_start_ts)
    # First bar on the same day (UTC) at/after ts
    same_day = index[(index >= ts) & (index.normalize() == ts.normalize())]
    if len(same_day) > 0:
        return same_day[0]
    # Fallback: first tradable bar >= ts (e.g., if the day had no bars)
    return first_tradable_test_bar(index, ts)


def find_hit_rate_switch_idx(df, window_bars: int, thr: float = 0.45, start_ts=None):
    """
    Return the FIRST timestamp >= start_ts where rolling hit-rate < thr.
    If none, return None.
    """
    import pandas as pd
    s = df if start_ts is None else df.loc[pd.to_datetime(start_ts):]
    hr = compute_rolling_hit_rate(s, int(window_bars), min_active=1).dropna()
    bad = hr[hr < float(thr)]
    return None if bad.empty else bad.index[0]


def build_trade_log_from_df(df, bar_minutes=None, price_col="close", pip_multiplier=10000):
    """
    Build a per-trade log from a per-bar results DataFrame.

    Expects df to contain:
      - index: datetime-like (bar timestamps),
      - 'position': net position (sign gives direction),
      - 'strategy': per-bar log-return (or simple return) of the strategy,
      - optionally a price column (default 'close') for entry/exit prices.

    Returns
    -------
    DataFrame with one row per trade:
      trade_id, entry_time, exit_time, side, side_sign,
      entry_bar, exit_bar, bars_held, holding_minutes,
      gross_log_return, pnl_pct, entry_price, exit_price, pips
    """
    import numpy as np
    import pandas as pd

    cols = [
        "trade_id",
        "entry_time",
        "exit_time",
        "side",
        "side_sign",
        "entry_bar",
        "exit_bar",
        "bars_held",
        "holding_minutes",
        "gross_log_return",
        "pnl_pct",
        "entry_price",
        "exit_price",
        "pips",
        "barrier_hit",
    ]

    _available_prices = None
    for _try_col in (price_col, "close", "Close", "mid_close", "price"):
        if _try_col is not None and _try_col in df.columns:
            _available_prices = pd.to_numeric(df[_try_col]).values.astype(float)
            break
    if _available_prices is not None:
        prices = _available_prices
    else:
        prices = None

    # Optional per-bar stop/TP fill prices (from the execution loop): when a
    # trade exits on a bar with a stop fill, the exit price is the actual
    # fill price, not the bar close.
    stop_fill_prices = None
    if "stop_fill_price" in df.columns:
        try:
            stop_fill_prices = pd.to_numeric(df["stop_fill_price"]).fillna(0.0).values.astype(float)
        except Exception:
            stop_fill_prices = None

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)

    if "strategy" not in df.columns:
        return pd.DataFrame(columns=cols)

    # Some evaluation routes store the executed position under 'position_exec'.
    pos_col = None
    for _c in ("position", "position_exec", "pos_exec", "pos"):
        if _c in df.columns:
            pos_col = _c
            break
    if pos_col is None:
        return pd.DataFrame(columns=cols)
 

    # Normalize inputs
    pos = np.sign(pd.to_numeric(df[pos_col]).fillna(0.0).values.astype(float))
    strat = pd.to_numeric(df["strategy"]).fillna(0.0).values.astype(float)
    idx = pd.to_datetime(df.index)

    # Infer bar length in minutes if not supplied
    if bar_minutes is None:
        try:
            if len(idx) >= 2:
                delta_min = (idx[1] - idx[0]).total_seconds() / 60.0
                bar_minutes = max(1, int(round(delta_min)))
            else:
                bar_minutes = 1
        except Exception:
            bar_minutes = 1

    trades = []
    current_side = 0.0
    entry_i = None

    def close_trade(exit_i: int, reason: str = "signal"):
        nonlocal current_side, entry_i
        if entry_i is None:
            return
        if exit_i < entry_i:
            exit_i = entry_i

        log_ret = float(np.nansum(strat[entry_i : exit_i + 1]))
        pnl_pct = float(np.exp(log_ret) - 1.0)
        bars_held = int(exit_i - entry_i + 1)

        entry_price = None
        exit_price = None
        pips = None
        barrier_reason = reason
        if prices is not None:
            try:
                entry_price = float(prices[entry_i])
                exit_price = float(prices[exit_i])
                if stop_fill_prices is not None and stop_fill_prices[exit_i] > 0.0:
                    exit_price = float(stop_fill_prices[exit_i])
                    barrier_reason = "stop"
                raw_pips = (exit_price - entry_price) * pip_multiplier
                pips = raw_pips if current_side > 0 else -raw_pips
            except Exception:
                entry_price = None
                exit_price = None
                pips = None

        trades.append(
            {
                "trade_id": len(trades),
                "entry_time": idx[entry_i],
                "exit_time": idx[exit_i],
                "side": "long" if current_side > 0 else "short",
                "side_sign": float(current_side),
                "entry_bar": int(entry_i),
                "exit_bar": int(exit_i),
                "bars_held": bars_held,
                "holding_minutes": int(bars_held * bar_minutes),
                "gross_log_return": log_ret,
                "pnl_pct": pnl_pct,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pips": pips,
                "barrier_hit": barrier_reason,
            }
        )
        current_side = 0.0
        entry_i = None

    # Walk the position series and detect opens / closes / flips
    for i, side in enumerate(pos):
        if current_side == 0.0 and side != 0.0:
            # Opening a new trade
            current_side = side
            entry_i = i
        elif current_side != 0.0:
            if side == 0.0:
                # Closing into flat -> close using this bar
                close_trade(i, "signal")
            elif side != current_side:
                # Flip: close old trade at i-1, open new one at i
                close_trade(i - 1, "signal")
                current_side = side
                entry_i = i
            else:
                # Same side, keep trade open
                pass

    # If still in a trade at the end, close at the last bar
    if current_side != 0.0 and entry_i is not None:
        close_trade(len(df) - 1, "eod")

    if not trades:
        return pd.DataFrame(columns=cols)

    tdf = pd.DataFrame(trades)
    return tdf[cols]

