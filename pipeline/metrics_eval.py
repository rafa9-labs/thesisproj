"""
Full evaluation metrics - compute_full_evaluation_metrics and helpers.

Extracted from utilsNoWFO.py (Phase 3, step 3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Helper functions - moved here from utilsNoWFO.py (Phase 3.1)
# These live in their final home so that metrics_eval.py is self-contained.
# ---------------------------------------------------------------------------

def _coerce_direction_labels(arr, labels=(-1, 0, 1), deadzone: float = 0.5):
    """Coerce predictions/targets into discrete {-1,0,+1} labels."""
    import numpy as _np
    a = _np.asarray(arr)
    if a.size == 0:
        return a.astype(int)
    if a.dtype.kind in ("f", "c"):
        a = _np.where(a > deadzone, 1, _np.where(a < -deadzone, -1, 0))
    else:
        try:
            a = a.astype(int, copy=False)
        except Exception:
            a = a.astype(float)
            a = _np.where(a > deadzone, 1, _np.where(a < -deadzone, -1, 0))
    valid = _np.isin(a, _np.array(labels, dtype=int))
    if not bool(_np.all(valid)):
        a = _np.where(valid, a, 0)
    return a.astype(int, copy=False)


def _auto_nw_lag(n: int, mode: str = "sqrt", x=None) -> int:
    """Newey-West lag selection: 'sqrt' (default) or 'andrews' plug-in."""
    import numpy as np
    n = int(max(1, n))
    m = (mode or "sqrt").lower()
    if m == "andrews":
        rho = 0.0
        if x is not None:
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            if x.size > 3:
                x0, x1 = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
                den = float((x0**2).sum()) or 1.0
                rho = float((x0 * x1).sum() / den)
                rho = float(np.clip(rho, -0.99, 0.99))
        c = 1.3221  # Bartlett kernel constant (Andrews 1991)
        q = int(max(1, round(c * (n ** 0.2))))
        return q
    return int(np.floor(np.sqrt(n)))


def hac_std(x, max_lag="auto") -> float:
    """Newey-West (HAC) standard deviation for a 1D array/Series."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 1:
        return 0.0
    x = x - np.mean(x)
    g0 = np.dot(x, x) / n
    if isinstance(max_lag, str):
        q = _auto_nw_lag(n, mode=max_lag, x=x)
    else:
        q = int(max(0, max_lag))
    if q == 0:
        var = g0
    else:
        var = g0
        for k in range(1, q + 1):
            w = 1.0 - k / (q + 1.0)  # Bartlett kernel
            gamma_k = np.dot(x[:-k], x[k:]) / n
            var += 2.0 * w * gamma_k
    return float(np.sqrt(max(var, 0.0)))


def estimate_frequency_per_year(index) -> float:
    """Estimate bars-per-year from a DateTimeIndex."""
    import numpy as np
    import pandas as pd
    if not hasattr(index, "tz"):
        try:
            index = pd.to_datetime(index, utc=True, errors="coerce")
        except Exception:
            return 252.0
    if len(index) < 3:
        return 252.0
    by_day = pd.Series(1.0, index=index).groupby(index.floor("D")).count()
    if by_day.empty:
        return 252.0
    bars_per_day = float(by_day.median())
    days = pd.Index(by_day.index)
    weekend_days = int(((days.dayofweek == 5) | (days.dayofweek == 6)).sum())
    frac_weekend = weekend_days / max(1, len(days))
    days_per_year = 365.0 if frac_weekend > 0.10 else 252.0
    return max(1.0, bars_per_day * days_per_year)


def compute_metrics(
    returns,
    positions,
    frequency_per_year=None,
    sharpe_cap=None,
    use_hac: bool = True,
    hac_max_lag="auto",
    min_active_obs: int = 25,
    std_floor: float = 1e-8,
):
    """Sharpe (annualized), max drawdown, trade count with HAC robust guards."""
    import os, numpy as np
    returns = returns.dropna()
    if frequency_per_year is None:
        try:
            frequency_per_year = float(estimate_frequency_per_year(returns.index))
        except Exception:
            frequency_per_year = 252.0
    ann_factor = float(np.sqrt(max(1.0, frequency_per_year)))
    active = returns[np.abs(returns) > 1e-12]
    n_active = int(active.size)
    if n_active < int(min_active_obs):
        sharpe = 0.0
    else:
        if use_hac:
            std = float(hac_std(active, max_lag=hac_max_lag))
        else:
            std = float(active.std(ddof=1))
        mean = float(active.mean())
        sharpe = (mean / std) * ann_factor if (np.isfinite(std) and std >= std_floor) else 0.0
    if sharpe_cap is None:
        try:
            cap_env = os.environ.get("SHARPE_CAP")
            if cap_env is not None:
                sharpe_cap = float(cap_env)
        except Exception:
            sharpe_cap = None
    if sharpe_cap is not None and sharpe_cap > 0:
        sharpe = float(np.clip(sharpe, -sharpe_cap, sharpe_cap))
    cum = returns.cumsum().apply(np.exp)
    drawdown = (cum / cum.cummax() - 1).min() if not cum.empty else 0.0
    try:
        p = positions
        if p is None:
            trades = 0
        else:
            if hasattr(p, "fillna"):
                p = p.fillna(0)
            p_arr = np.asarray(p, dtype=float)
            if p_arr.size <= 1:
                trades = 0
            else:
                p_dir = np.sign(p_arr)
                trades = int(np.sum(np.abs(np.diff(p_dir))))
    except Exception:
        trades = 0
    return round(sharpe, 2), round(drawdown, 4), trades


def compute_geometric_mean_annualized(returns):
    """Geometric mean annualized from per-period log returns."""
    import numpy as np
    n = len(returns)
    if n == 0:
        return np.nan
    compounded = np.exp(returns.sum())
    try:
        bars_per_year = float(estimate_frequency_per_year(returns.index))
    except Exception:
        bars_per_year = 252.0
    annual_factor = bars_per_year / max(1, n)
    return compounded ** annual_factor - 1


# ---------------------------------------------------------------------------
# End of helpers moved from utilsNoWFO.py
# ---------------------------------------------------------------------------

def _macro_prec_f1_from_confusion(y_true, y_pred, labels=(-1,0,1)):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    # precision per class: TP / (TP + FP)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec_per_class = np.diag(cm) / cm.sum(axis=0, where=np.ones_like(cm, dtype=bool))
        rec_per_class  = np.diag(cm) / cm.sum(axis=1, where=np.ones_like(cm, dtype=bool))
        f1_per_class = 2 * prec_per_class * rec_per_class / (prec_per_class + rec_per_class)
    prec = np.nanmean(np.where(np.isfinite(prec_per_class), prec_per_class, np.nan))
    f1   = np.nanmean(np.where(np.isfinite(f1_per_class), f1_per_class, np.nan))
    return float(np.nan_to_num(prec, nan=0.0)), float(np.nan_to_num(f1, nan=0.0)), cm

def compute_full_evaluation_metrics(
    df,
    trading_costs=False,
    slippage_factor=0.0,
    prev_position=None,      # carry previous month last position (-1/0/+1)
    prev_eq_strategy=None,   # running equity to rescale strategy curve
    prev_eq_bh=None,          # running equity to rescale buy&hold curve
    **kwargs,
):
    """
    Returns a fixed 16-tuple of MONTHLY metrics (factors from 1.0), and sets:
      df.attrs["last_position"], df.attrs["end_eq_strategy"], df.attrs["end_eq_bh"].
    Also adds continuous curves:
      df["cstrategy_cont"], df["creturns_cont"].

    Patch #1 (optional): volatility-targeted sizing (eval_use_vol_target)
    Patch #2 (optional): TP1 1/2 scale-out + breakeven + trailing stop (eval_use_scaleout_trail)
    Patch #3 (optional): impact-aware TWAP execution (eval_use_twap_execution)
    Patch #4 (optional): regime-adaptive TP1 & trailing (eval_use_regime_adaptive)
    Patch #5 (optional): daily/session kill-switch + cool-off (eval_use_kill_switch)
    """
    import os

    # --- Preconditions ---
    if "returns" not in df.columns:
        # Preserve carry state so downstream month stitching doesn't reset to 1.0
        try:
            if prev_position is not None:
                df.attrs["last_position"] = float(prev_position)
            if prev_eq_strategy is not None:
                df.attrs["end_eq_strategy"] = float(prev_eq_strategy)
            if prev_eq_bh is not None:
                df.attrs["end_eq_bh"] = float(prev_eq_bh)
        except Exception:
            pass
        return (np.nan,) * 16
    # Ensure pred exists AND is numeric (prevents object/str/NaN poisoning shift/diff)
    if "pred" not in df.columns:
        df["pred"] = 0.0
    try:
        df["pred"] = pd.to_numeric(df["pred"], errors="coerce").astype(float).fillna(0.0)
    except Exception:
        df["pred"] = 0.0
    # Guard against NaN/invalid spreads leaking into PnL math:
    # even (0.0 * NaN) will yield NaN and can poison curves/metrics.
    if "spread" not in df.columns:
        df["spread"] = 0.0
    else:
        try:
            df["spread"] = pd.to_numeric(df["spread"], errors="coerce").astype(float).fillna(0.0)
        except Exception:
            df["spread"] = 0.0

    # Same idea for slippage_bps if present; keep it numeric and finite.
    if "slippage_bps" in df.columns:
        try:
            df["slippage_bps"] = pd.to_numeric(df["slippage_bps"], errors="coerce").astype(float).fillna(0.0)
        except Exception:
            df["slippage_bps"] = 0.0

    # Config from attrs if caller injected CLASS_DEFAULTS upstream
    try:
        cfg_eval = dict(df.attrs.get("features_config", {}) or {})
    except Exception:
        cfg_eval = {}
    def _cfg(key, default):
        return cfg_eval.get(key, default)
    
    # Debug-only: cost breakdown (env var or caller-injected df.attrs).
    # This is intentionally side-effect free for non-debug runs.
    try:
        debug_costs = bool(os.getenv("MLB_DEBUG_COSTS", "0") == "1") or bool(df.attrs.get("debug_costs", False)) or bool(_cfg("eval_print_costs_debug", False))
    except Exception:
        debug_costs = bool(os.getenv("MLB_DEBUG_COSTS", "0") == "1")
        
    # Optional: caller label to disambiguate multiple eval passes (CV fold vs diagnostics vs real-sim stitching).
    # Purely for debug prints; does not affect PnL or metrics.
    eval_context = None
    try:
        eval_context = kwargs.get("eval_context", None)
        if eval_context is None:
           eval_context = df.attrs.get("eval_context", None)
    except Exception:
        eval_context = None
    _ctx = f" | ctx={eval_context}" if eval_context else ""
    
     
    # Cost-column sanity: if costs are enabled but columns are missing, warn loudly.
    # This prevents silent under-costing when a caller forgets to run `_ensure_cost_columns()`.
    try:
        warn_missing_costs = (bool(os.getenv("MLB_WARN_MISSING_COSTS", "0") == "1")
                              or bool(debug_costs)
                              or (eval_context and str(eval_context).startswith("real_sim")))
    except Exception:
        warn_missing_costs = bool(debug_costs)

    if trading_costs and warn_missing_costs:
        missing = []
        if "spread" not in df.columns:
            missing.append("spread")
        if "slippage_bps" not in df.columns:
            missing.append("slippage_bps")
        if missing:
            print(f"[Costs][Warn]{_ctx} missing cost columns: {missing}. Call _ensure_cost_columns() before evaluation.")


    # --- 1-bar delay & carry-in on first bar ---
    # IMPORTANT: compute_full_evaluation_metrics may be called more than once
    # on the *same* DataFrame (e.g., scalar metrics + continuous metrics).
    # We therefore always shift from the original, decision-time predictions
    # to avoid accidental double-shifting (2-bar delay).
    if "raw_pred" in df.columns:
        raw_pred = df["raw_pred"].copy()
    else:
        raw_pred = df["pred"].copy()
        df["raw_pred"] = raw_pred
        
    # Normalize missing decision-time preds (can appear after reindexing) to "flat"
    # so the 1-bar shift causality check remains well-defined.
    raw_pred = pd.to_numeric(raw_pred, errors="coerce").fillna(0.0).astype(float)
    df["raw_pred"] = raw_pred
    
    df["pred"] = raw_pred.shift(1)

    # If we have a previous position (carry-in from prior month),
    # inject it into the *first* bar only if that bar would otherwise be NaN.
    # Use positional access so we never get a Series even if the first index
    # label appears multiple times (e.g. after Top-N switches / concatenation).
    if prev_position is not None and not df.empty:
        first_pred = df["pred"].iloc[0]
        if pd.isna(first_pred):
            df.iloc[0, df.columns.get_loc("pred")] = prev_position

    df["pred"] = df["pred"].fillna(0.0)
    df["pred_exec"] = df["pred"].astype(float)
    
    # --- Optional causality checks / debug preview (no behavior change unless enabled) ---
    eval_assert_causality = bool(_cfg("eval_assert_causality", True))
    eval_print_causality  = bool(_cfg("eval_print_causality_debug", True))
    eval_preview_n        = int(_cfg("eval_causality_preview_n", 6))

    if eval_assert_causality:
        # Index sanity: evaluation assumes non-decreasing time order.
        if not getattr(df.index, "is_monotonic_increasing", False):
            raise AssertionError("[Eval][Causality] df.index is not monotonic increasing.")

        # 1-bar delay integrity: for i>=1, shifted pred must match raw_pred[i-1]
        if len(df) >= 2:
            a = df["pred"].iloc[1:].to_numpy(dtype=float)
            b = raw_pred.iloc[:-1].to_numpy(dtype=float)
            if not np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True):
                bad = np.where(~np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))[0]
                bad = bad[: min(5, bad.size)]
                raise AssertionError(
                    f"[Eval][Causality] pred shift mismatch at {bad.tolist()} "
                    f"(df.pred[i] should equal raw_pred[i-1] for i>=1)."
                )

        # First bar should be neutral unless explicitly carrying a prior position.
        if not df.empty:
            first = float(df["pred"].iloc[0])
            if prev_position is None and first != 0.0:
                raise AssertionError(
                    f"[Eval][Causality] First bar pred is {first}, expected 0.0 (no carry-in)."
                )

    if eval_print_causality and not df.empty:
        # Small head preview: raw_pred vs shifted pred (post-carry), plus returns/spread.
        preview = pd.DataFrame(index=df.index)
        preview["raw_pred"] = raw_pred
        preview["pred_exec"] = df["pred"]
        preview["returns"] = df["returns"]
        preview["spread"] = df["spread"]
        print("[Eval][Causality] Head preview (raw_pred vs executed pred):")
        print(preview.head(max(1, eval_preview_n)).to_string())
        print(f"[Eval][Causality] prev_position={prev_position}, trading_costs={trading_costs}, slippage_factor={slippage_factor}{_ctx}")
 

        # ---- Exec/trade counting audit (log-only; helps explain "signals exist but trades=0") ----
        try:
            _raw_s = pd.Series(raw_pred, index=df.index) if not isinstance(raw_pred, pd.Series) else raw_pred.reindex(df.index)
            nz_raw = int((_raw_s.fillna(0) != 0).sum())
            nz_exec = int((df["pred"].fillna(0) != 0).sum()) if "pred" in df.columns else -1

            # Position column name can differ by path; probe common candidates.
            pos_col = None
            # Prefer executed position streams when present.
            for _c in ("position_exec", "pos_exec", "position", "pos"):
                if _c in df.columns:
                    pos_col = _c
                    break

            # IMPORTANT: this is an early preview (executed position is finalized later).
            # Do NOT mutate df here; only read from it.
            if pos_col is not None:
                _pos = pd.to_numeric(df[pos_col], errors="coerce").fillna(0.0)
            else:
                if "pred_exec" in df.columns:
                    _pos = pd.to_numeric(df["pred_exec"], errors="coerce").fillna(0.0)
                    pos_col = "pred_exec"
                else:
                    _pos = pd.Series(0.0, index=df.index)
                    pos_col = "zeros"

            # Canonical directional trade counting (matches compute_metrics):
            # trades_dir = SUM |delta sign(position)|  (a flip +1->-1 counts as 2 fills)
            _pos_arr = _pos.astype(float).to_numpy(copy=False)
            _pos_dir = np.sign(_pos_arr)

            if _pos_dir.size <= 1:
                dir_changes = int(_pos_dir.size == 1 and _pos_dir[0] != 0)
                trades_dir = dir_changes
                entries = dir_changes
                size_changes = 0
            else:
                dir_changes = int(np.sum(_pos_dir[1:] != _pos_dir[:-1]))
                trades_dir = int(np.sum(np.abs(np.diff(_pos_dir))))
                entries = int(np.sum((_pos_dir[:-1] == 0) & (_pos_dir[1:] != 0)))
                # Any change in *magnitude* (TWAP ramps, vol-target sizing, scale-outs, etc.)
                size_changes = int(np.sum(np.abs(np.diff(_pos_arr)) > 1e-12))

            # Back-compat: keep the existing field name, but make it align with metrics "Trades"
            pos_changes = trades_dir
 
            # IMPORTANT: position series is computed later in this function.
            # Avoid logging a misleading pos_col=None in early-stage previews.
            if pos_col is not None:
                bars = int(len(df))
                sig_rate = float(nz_exec / bars) if bars > 0 and nz_exec >= 0 else float("nan")
                pos_rate = float((_pos_dir != 0).mean()) if bars > 0 else float("nan")
                print(
                    f"[ExecAudit]{_ctx} bars={bars} nz_raw={nz_raw} nz_exec={nz_exec} "
                    f"sig_rate={sig_rate:.3f} pos_rate={pos_rate:.3f} pos_col={pos_col} "
                    f"pos_changes={pos_changes} dir_changes={dir_changes} size_changes={size_changes} entries={entries}"
                )
        except Exception:
            pass


    # --- Gap detection (for kill-switch/session logic) ---
    # NOTE:
    # We still detect time gaps between bars (e.g. weekend / illiquid periods)
    # so that downstream logic (kill-switch, daily state) can use it *if needed*.
    # BUT we no longer force-flatten positions purely because we hit a gap:
    # no more "abrupt close at end of each session/day" here.
    gap_from_prev_bool = np.zeros(len(df), dtype=bool)
    if len(df.index) >= 2:
        idx_series = pd.Series(pd.to_datetime(df.index))
        diffs_forward = idx_series.shift(-1) - idx_series
        diffs_backward = idx_series - idx_series.shift(1)
        step_candidates = diffs_forward[diffs_forward > pd.Timedelta(0)]
        step = step_candidates.median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(minutes=15)

        gap_to_next = diffs_forward > (step * 1.5)     # big forward gap
        gap_from_prev = diffs_backward > (step * 1.5)  # big backward gap
        gap_to_next_bool   = gap_to_next.fillna(False).values
        gap_from_prev_bool = gap_from_prev.fillna(False).values

        # We KEEP gap_from_prev_bool for kill-switch/session logic later on,
        # but we do NOT zero df["pred"] here anymore.
        # Previously:
        #   - last bar before a gap had pred=0 -> flip_or_flat -> forced exit
        #   - first bar after gap also pred=0 -> forced start flat
        # Now:
        #   - decisions at gap edges follow the normal trading logic
        #   - if you keep a position, you effectively "carry across the gap".


    # --- Cadence & rolling bar-vol (sigma) for several patches ---
    try:
        deltas = df.index.to_series().diff().dropna().dt.total_seconds()
        sec_per_bar = int(np.median(deltas.values)) if len(deltas) else 1800
    except Exception:
        sec_per_bar = 1800
    bars_per_day = max(1, min(500, int(round(86400 / max(1, sec_per_bar)))))
    annual_bars = 252 * bars_per_day

    vol_floor = float(_cfg("eval_vol_floor", 1e-6))
    lookback_vol = int(_cfg("eval_vol_lookback", 48))
    df["_bar_vol"] = df["returns"].rolling(
        lookback_vol, min_periods=max(5, lookback_vol // 3)
    ).std().clip(lower=vol_floor)

    # Vol-target per-bar (Patch #1)
    use_vol_target = bool(_cfg("eval_use_vol_target", False))
    target_ann = float(_cfg("eval_vol_target_ann", 0.10))
    target_bar = target_ann / np.sqrt(max(1, annual_bars))
    max_lev = float(_cfg("eval_max_leverage", 1.50))

    # Patch #2 flags
    use_trail    = bool(_cfg("eval_use_scaleout_trail", False))
    tp1_z_base   = float(_cfg("eval_tp1_z", 1.0))
    trail_k_base = float(_cfg("eval_trail_k", 2.5))
    dyn_vol      = bool(_cfg("eval_trail_dynamic_vol", True))
    move_to_be   = bool(_cfg("eval_move_stop_to_be", True))
    # Max holding (timeout) used by execution / trailing logic.
    # Back-compat: if not explicitly provided, fall back to the triple-barrier horizon.
    max_hold_bars = int(_cfg("eval_max_holding_bars", 0))
    if max_hold_bars <= 0:
        max_hold_bars = int(_cfg("tb_max_holding", 0))
    min_hold_bars = int(_cfg("eval_min_holding_bars", 0))

    # Patch #3 flags (TWAP)
    use_twap    = bool(_cfg("eval_use_twap_execution", False))
    twap_span   = max(1, int(_cfg("eval_twap_span_bars", 2)))
    impact_eta  = float(_cfg("eval_impact_eta", 0.0))
    twap_freeze = bool(_cfg("eval_twap_freeze_size_at_entry", True))

    # Patch #4 flags (regime-adaptive)
    use_regime   = bool(_cfg("eval_use_regime_adaptive", False))
    regime_src   = str(_cfg("eval_regime_source", "sigma")).lower()
    q_low        = float(_cfg("eval_regime_q_low", 0.33))
    q_high       = float(_cfg("eval_regime_q_high", 0.66))

    tp1_z_calm     = _cfg("eval_tp1_z_calm", None)
    tp1_z_normal   = _cfg("eval_tp1_z_normal", None)
    tp1_z_volatile = _cfg("eval_tp1_z_volatile", None)
    trail_k_calm     = _cfg("eval_trail_k_calm", None)
    trail_k_normal   = _cfg("eval_trail_k_normal", None)
    trail_k_volatile = _cfg("eval_trail_k_volatile", None)

    # Patch #5 flags (kill-switch)
    use_kill     = bool(_cfg("eval_use_kill_switch", False))
    kill_mode    = str(_cfg("eval_kill_mode", "pct")).lower()  # "pct" or "sigma"
    kill_pct     = float(_cfg("eval_kill_limit_pct", 0.02))
    kill_sigma_k = float(_cfg("eval_kill_sigma", 3.0))
    kill_until_session_end = bool(_cfg("eval_kill_until_session_end", True))
    kill_cooloff_bars = int(_cfg("eval_cooloff_bars", 0))
    kill_print = bool(_cfg("eval_kill_print_debug", True))
    
    # --- SpreadGuard: block entries on extreme spread bars (allow exits) ---
    spread_cap = float(_cfg("eval_spread_cap", 0.00040))
    use_spread_guard = bool(_cfg("eval_use_spread_guard", True))

    # >>> SAFETY CLAMPS (add these) <<<
    _min_pct   = float(_cfg("eval_kill_min_limit_pct", 0.005))      # 0.5%
    _min_sigma = float(_cfg("eval_kill_min_sigma", 1.0))
    _max_sigma = float(_cfg("eval_kill_max_sigma", 6.0))
    _max_cool  = int(_cfg("eval_kill_max_cooloff_bars", 480))

    orig_kill_pct, orig_sigma, orig_cool = kill_pct, kill_sigma_k, kill_cooloff_bars

    if kill_mode == "pct":
        kill_pct = max(kill_pct, _min_pct)
    else:  # sigma
        kill_sigma_k = min(max(kill_sigma_k, _min_sigma), _max_sigma)

    kill_cooloff_bars = min(max(0, kill_cooloff_bars), _max_cool)

    if kill_print and use_kill:
        if kill_mode == "pct" and kill_pct != orig_kill_pct:
            print(f"[Eval] Kill clamp: pct {orig_kill_pct:.4%} -> {kill_pct:.4%}")
        if kill_mode == "sigma" and kill_sigma_k != orig_sigma:
            print(f"[Eval] Kill clamp: sigma k {orig_sigma:.2f} -> {kill_sigma_k:.2f}")
        if kill_cooloff_bars != orig_cool:
            print(f"[Eval] Kill clamp: cool-off {orig_cool} -> {kill_cooloff_bars} bars")

    # Build the regime metric series
    metric_series = None
    if use_regime and regime_src == "adx":
        if "adx" in df.columns:
            metric_series = df["adx"].astype(float)
        else:
            regime_src = "sigma"
    if (not use_regime) or regime_src == "sigma" or metric_series is None:
        metric_series = df["_bar_vol"]

    # Compute regime thresholds & codes (0 calm, 1 normal, 2 volatile)
    if use_regime:
        ql = float(metric_series.quantile(q_low))
        qh = float(metric_series.quantile(q_high))
        regime_code = np.where(metric_series <= ql, 0,
                        np.where(metric_series <= qh, 1, 2))
        if bool(_cfg("eval_print_regime_debug", True)):
            counts = pd.Series(regime_code).value_counts().to_dict()
            # NOTE: don't try to "pretty-print dict literals" with nested braces in f-strings
            # (it triggers "Invalid format specifier"). Just print a real dict.
            safe_counts = {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0)), 2: int(counts.get(2, 0))}
            print(f"[Eval] Regime adaptive ON (src={regime_src}, q_low={ql:.4g}, q_high={qh:.4g}) | counts={safe_counts}")
    else:
        regime_code = np.zeros(len(df), dtype=int)

    # Core arrays/state
    n = len(df)
    rets = df["returns"].values.astype(float)
    pred = df["pred"].values.astype(float)
    bar_vol = df["_bar_vol"].values.astype(float)

    # Optional session flag: 1 = inside trading session, 0 = outside.
    # If present, we will BLOCK *new entries* when session_flag == 0,
    # but we will NOT forcibly close existing positions on those bars.
    session_flag_arr = None
    if "session_flag" in df.columns:
        try:
            session_flag_arr = df["session_flag"].astype(int).values
        except Exception:
            session_flag_arr = None


    trades_sig = np.zeros(n, dtype=float)

    _is_cv = bool(eval_context) and (str(eval_context).startswith("cv:") or str(eval_context).startswith("hpo:"))

    if _is_cv:
        pos_actual = np.clip(pred, -1.0, 1.0)
        spread_arr = df["spread"].values.astype(float) if "spread" in df.columns else np.zeros(n)
        pos_diff = np.abs(np.diff(pos_actual, prepend=0.0))
        strat = (pos_actual * rets) - (spread_arr * pos_diff)
        tp1_hits = 0; stop_hits = 0; timeouts = 0; flips_exits = 0
        twap_events = 0; total_ramp_bars = 0
        total_impact_cost = 0.0; total_slippage_cost = 0.0
        spread_spike_blocked = 0; kills_triggered = 0; bars_flat_due_kill = 0
        record_cost_columns = False
    else:
        # --- Build PatchConfig and delegate to execution_patches ---
        from pipeline.backtester.execution_patches import PatchConfig, run_execution_loop

        cfg = PatchConfig(
            bars_per_day=bars_per_day,
            annual_bars=annual_bars,
            vol_floor=vol_floor,
            use_vol_target=use_vol_target,
            target_bar=target_bar,
            max_lev=max_lev,
            use_trail=use_trail,
            tp1_z_base=tp1_z_base,
            trail_k_base=trail_k_base,
            dyn_vol=dyn_vol,
            move_to_be=move_to_be,
            max_hold_bars=max_hold_bars,
            min_hold_bars=min_hold_bars,
            use_twap=use_twap,
            twap_span=twap_span,
            impact_eta=impact_eta,
            twap_freeze=twap_freeze,
            use_regime=use_regime,
            tp1_z_calm=tp1_z_calm,
            tp1_z_normal=tp1_z_normal,
            tp1_z_volatile=tp1_z_volatile,
            trail_k_calm=trail_k_calm,
            trail_k_normal=trail_k_normal,
            trail_k_volatile=trail_k_volatile,
            use_kill=use_kill,
            kill_mode=kill_mode,
            kill_pct=kill_pct,
            kill_sigma_k=kill_sigma_k,
            kill_until_session_end=kill_until_session_end,
            kill_cooloff_bars=kill_cooloff_bars,
            use_spread_guard=use_spread_guard,
            spread_cap=spread_cap,
            debug_costs=debug_costs,
            eval_context=eval_context,
            stop_pip_value=_cfg("stop_pip_value", 0.0001),
            trailing_pip_value=_cfg("trailing_pip_value", 0.0001),
            sizing_method=_cfg("sizing_method", "fixed"),
            sizing_risk_fraction=_cfg("sizing_risk_fraction", 0.02),
            sizing_kelly_fraction=_cfg("sizing_kelly_fraction", 0.5),
            sizing_kelly_min_trades=_cfg("sizing_kelly_min_trades", 10),
            sizing_atr_risk_pct=_cfg("sizing_atr_risk_pct", 0.02),
            sizing_atr_sl_mult=_cfg("sizing_atr_sl_mult", 2.0),
            sizing_initial_equity=_cfg("sizing_initial_equity", 10_000.0),
            sizing_max_leverage=_cfg("sizing_max_leverage", 5.0),
            sizing_contract_size=_cfg("sizing_contract_size", 100_000.0),
            stop_method=_cfg("stop_method", "none"),
            stop_sl_pips=_cfg("stop_sl_pips", 30.0),
            stop_tp_pips=_cfg("stop_tp_pips", 60.0),
            stop_sl_atr_mult=_cfg("stop_sl_atr_mult", 2.0),
            stop_tp_atr_mult=_cfg("stop_tp_atr_mult", 3.0),
            stop_sl_sigma_mult=_cfg("stop_sl_sigma_mult", 2.0),
            stop_tp_sigma_mult=_cfg("stop_tp_sigma_mult", 3.0),
            stop_use_be=_cfg("stop_use_be", False),
            stop_be_trigger_pips=_cfg("stop_be_trigger_pips", 20.0),
            stop_use_partial_close=_cfg("stop_use_partial_close", False),
            stop_tp1_ratio=_cfg("stop_tp1_ratio", 0.5),
            stop_tp1_pips=_cfg("stop_tp1_pips", 30.0),
            stop_tp2_pips=_cfg("stop_tp2_pips", 0.0),
            trailing_method=_cfg("trailing_method", "none"),
            trailing_pips=_cfg("trailing_pips", 30.0),
            trailing_atr_mult=_cfg("trailing_atr_mult", 3.0),
            trailing_chandelier_atr_mult=_cfg("trailing_chandelier_atr_mult", 3.0),
            trailing_chandelier_lookback=_cfg("trailing_chandelier_lookback", 22),
            trailing_activation_pips=_cfg("trailing_activation_pips", 10.0),
            risk_use_dd_breaker=_cfg("risk_use_dd_breaker", False),
            risk_max_drawdown_pct=_cfg("risk_max_drawdown_pct", 0.20),
            risk_dd_resume=_cfg("risk_dd_resume", "session_end"),
            risk_dd_cooloff_bars=_cfg("risk_dd_cooloff_bars", 48),
            risk_use_daily_loss=_cfg("risk_use_daily_loss", False),
            risk_max_daily_loss_pct=_cfg("risk_max_daily_loss_pct", 0.03),
            risk_max_daily_loss_sigma=_cfg("risk_max_daily_loss_sigma", 3.0),
            risk_daily_loss_mode=_cfg("risk_daily_loss_mode", "pct"),
            risk_use_consec_loss=_cfg("risk_use_consec_loss", False),
            risk_max_consecutive_losses=_cfg("risk_max_consecutive_losses", 5),
            risk_consec_resume=_cfg("risk_consec_resume", "session_end"),
            risk_consec_cooloff_bars=_cfg("risk_consec_cooloff_bars", 48),
            risk_initial_equity=_cfg("risk_initial_equity", 10_000.0),
            risk_max_open_positions=_cfg("risk_max_open_positions", 1),
        )

        record_cost_columns = bool(_cfg("eval_record_cost_columns", False)) or bool(debug_costs)

        result = run_execution_loop(
            df=df,
            pred=pred,
            rets=rets,
            bar_vol=bar_vol,
            gap_from_prev_bool=gap_from_prev_bool,
            regime_code=regime_code,
            cfg=cfg,
            trading_costs=trading_costs,
            slippage_factor=slippage_factor,
            session_flag_arr=session_flag_arr,
            record_cost_columns=record_cost_columns,
        )

        pos_actual = result.pos_actual
        strat = result.strat
        tp1_hits = result.tp1_hits
        stop_hits = result.stop_hits
        timeouts = result.timeouts
        flips_exits = result.flips_exits
        twap_events = result.twap_events
        total_ramp_bars = result.total_ramp_bars
        total_impact_cost = result.total_impact_cost
        total_slippage_cost = result.total_slippage_cost
        spread_spike_blocked = result.spread_spike_blocked
        kills_triggered = result.kills_triggered
        bars_flat_due_kill = result.bars_flat_due_kill
        if record_cost_columns:
            cost_spread_pf = result.cost_spread_pf
            cost_slip_pf = result.cost_slip_pf
            cost_impact_bar = result.cost_impact_bar
            cost_total_turn = result.cost_total_turn


    # Trades (reporting continuity)
    trades_sig[1:] = np.abs(np.sign(pred[1:]) - np.sign(pred[:-1])) / 2.0
    df["trades"] = pd.Series(trades_sig, index=df.index)

    # Final series (single source of truth)
    # IMPORTANT: Always overwrite position_exec so metrics/carry-out can't consume a stale stream
    df["position"] = pd.Series(pos_actual, index=df.index)
    df["position_exec"] = df["position"]
    df["strategy"] = pd.Series(strat, index=df.index)
    
    # Exec/trade audit (post-position) so month_eval logs reflect executed state.
    if eval_print_causality:
        try:
            _raw_s = pd.Series(raw_pred, index=df.index) if not isinstance(raw_pred, pd.Series) else raw_pred.reindex(df.index)
            nz_raw = int((_raw_s.fillna(0) != 0).sum())
            nz_exec = int((df["pred"].fillna(0) != 0).sum()) if "pred" in df.columns else -1
            _pos = pd.to_numeric(df["position_exec"], errors="coerce").fillna(0.0)
            _pos_arr = _pos.astype(float).to_numpy(copy=False)
            _pos_dir = np.sign(_pos_arr)

            if _pos_dir.size <= 1:
                dir_changes = int(_pos_dir.size == 1 and _pos_dir[0] != 0)
                trades_dir = dir_changes
                entries = dir_changes
                size_changes = 0
            else:
                dir_changes = int(np.sum(_pos_dir[1:] != _pos_dir[:-1]))
                trades_dir = int(np.sum(np.abs(np.diff(_pos_dir))))
                entries = int(np.sum((_pos_dir[:-1] == 0) & (_pos_dir[1:] != 0)))
                size_changes = int(np.sum(np.abs(np.diff(_pos_arr)) > 1e-12))

            # Back-compat: keep existing field name, but make it align with metrics "Trades"
            pos_changes = trades_dir
            
            
            bars = int(len(df))
            sig_rate = float(nz_exec / bars) if bars > 0 and nz_exec >= 0 else float("nan")
            
            pos_rate = float((_pos_dir != 0).mean()) if bars > 0 else float("nan")
            print(
                f"[ExecAudit]{_ctx} bars={bars} nz_raw={nz_raw} nz_exec={nz_exec} "
                f"sig_rate={sig_rate:.3f} pos_rate={pos_rate:.3f} pos_col=position_exec "
                f"pos_changes={pos_changes} dir_changes={dir_changes} size_changes={size_changes} entries={entries}"

            )
        except Exception:
            pass
        
        
    if record_cost_columns:
        df["cost_spread_per_fill"] = pd.Series(cost_spread_pf, index=df.index)
        df["cost_slip_per_fill"] = pd.Series(cost_slip_pf, index=df.index)
        df["cost_impact"] = pd.Series(cost_impact_bar, index=df.index)
        df["cost_total_turnover"] = pd.Series(cost_total_turn, index=df.index)


    # Debug-only cost breakdown
    if debug_costs:
        try:
            _total_cost = float(total_slippage_cost) + float(total_impact_cost)
            print(("[Eval][Costs] slippage_cost_total={:.6g} impact_cost_total={:.6g} total_cost={:.6g}" + _ctx).format(
                float(total_slippage_cost), float(total_impact_cost), float(_total_cost)
            ))
            
            # Extra audit-friendly stats (no behavior change)
            if "spread" in df.columns and (("price" in df.columns) or ("mid_close" in df.columns)):
                _p = df["price"] if "price" in df.columns else df["mid_close"]
                _p = pd.to_numeric(_p, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
                _s = pd.to_numeric(df["spread"], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
                _mask = (_p.notna()) & (_s.notna()) & (_p > 0.0)
                if _mask.any():
                    _full_spread_bps = (_s[_mask] / _p[_mask]) * 1e4
                    _half_spread_bps = 0.5 * _full_spread_bps
                    _med_full = float(_full_spread_bps.median())
                    _med_half = float(_half_spread_bps.median())
                    _med_slip = float(pd.to_numeric(df.get("slippage_bps", 0.0), errors="coerce").astype(float).fillna(0.0).median())
                    print("[Eval][Costs] median_full_spread_bps={:.4f} median_half_spread_bps={:.4f} median_slippage_bps={:.4f} slippage_factor={:.3f}".format(
                        _med_full, _med_half, _med_slip, float(slippage_factor)
                    ))
                    
            # SpreadGuard summary (audit)
            if use_spread_guard and ("spread" in df.columns):
                try:
                    _smax = float(pd.to_numeric(df["spread"], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).max())
                except Exception:
                    _smax = float("nan")
                print(f"[Eval][SpreadGuard] cap={spread_cap:.6f} spikes_blocked={int(spread_spike_blocked)} max_spread={_smax:.6f}{_ctx}")

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Spread/session audit (REAL-SIM only; prints even when debug_costs=False)
    # Purpose: make illiquid spread spikes visible in thesis logs and ensure
    # real-sim results can't be accused of microstructure distortion.
    # No behavior change.
    # ------------------------------------------------------------------
    try:
        _is_real_sim = bool(eval_context) and str(eval_context).startswith("real_sim")
    except Exception:
        _is_real_sim = False

    if _is_real_sim and bool(_cfg("eval_print_spread_audit", True)):
        try:
            _s = pd.to_numeric(df.get("spread", 0.0), errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            _max_s = float(np.nanmax(_s.to_numpy(dtype=float))) if len(_s) else float("nan")
            _p95_s = float(np.nanpercentile(_s.to_numpy(dtype=float), 95)) if len(_s) else float("nan")
            _spike_bars = int(np.nansum((_s.to_numpy(dtype=float) > float(spread_cap)).astype(int))) if len(_s) else 0

            # Session-aware stats if session_flag_arr exists
            _sess_share = None
            _spikes_in_sess = None
            _max_in_sess = None
            if "session_flag_arr" in locals() and session_flag_arr is not None and len(session_flag_arr) == len(df):
                _sess = np.asarray(session_flag_arr, dtype=int)
                _sess_share = float(np.mean(_sess == 1))
                _mask = (_sess == 1)
                if np.any(_mask):
                    _s_sess = _s.to_numpy(dtype=float)[_mask]
                    _max_in_sess = float(np.nanmax(_s_sess))
                    _spikes_in_sess = int(np.nansum((_s_sess > float(spread_cap)).astype(int)))

            # SpreadGuard blocked entries (if Patch 3 is installed)
            _blocked = int(locals().get("spread_spike_blocked", 0) or 0)

            msg = f"[Eval][SpreadAudit] cap={float(spread_cap):.6f} spike_bars={_spike_bars} blocked_entries={_blocked} max_spread={_max_s:.6f} p95_spread={_p95_s:.6f}{_ctx}"
            if _sess_share is not None:
                msg += f" | in_sess_share={_sess_share:.1%} spikes_in_sess={_spikes_in_sess} max_in_sess={_max_in_sess:.6f}"
            print(msg)
        except Exception:
            pass

    # Debug prints
    if bool(_cfg("eval_use_regime_adaptive", False)) and bool(_cfg("eval_print_regime_debug", True)):
        # Defensive: never let debug printing crash the fold
        try:
            rc = pd.Series(regime_code).value_counts(normalize=True).reindex([0, 1, 2]).fillna(0.0)
            def _pct(x):
                try: return float(x)
                except Exception: return 0.0
            calm = _pct(rc.get(0, 0.0)); normal = _pct(rc.get(1, 0.0)); volatile = _pct(rc.get(2, 0.0))
            print(f"[Eval] Regime shares: calm={calm:.1%}, normal={normal:.1%}, volatile={volatile:.1%}")
        except Exception as _e:
            try:
                print(f"[WARN] [Eval] Regime debug print failed: {type(_e).__name__}: {_e}")
            except Exception:
                pass
    if bool(_cfg("eval_print_trail_debug", True)) and bool(_cfg("eval_use_scaleout_trail", False)):
        print(
            f"[Eval] Trail/Scale-out ON: base tp1_z={tp1_z_base}, base k={trail_k_base}, "
            f"dyn_vol={dyn_vol}, max_hold={max_hold_bars}, "
            f"tp1_hits={tp1_hits}, stops={stop_hits}, flips_or_flat={flips_exits}, timeouts={timeouts}"
        )
    if bool(_cfg("eval_twap_print_debug", True)) and bool(_cfg("eval_use_twap_execution", False)):
        print(
            f"[Eval] TWAP ON: span={twap_span}, ramps={twap_events}, "
            f"ramp_bars={total_ramp_bars}, impact_eta={impact_eta}, "
            f"impact_cost_sum={total_impact_cost:.6f}"
        )
    if kill_print and use_kill:
        print(
            f"[Eval] Kill-switch ON: mode={kill_mode}, limit_pct={kill_pct:.2%}, "
            f"sigma_k={kill_sigma_k:.2f}, until_session_end={kill_until_session_end}, "
            f"cooloff_bars={kill_cooloff_bars}, triggers={kills_triggered}, "
            f"flat_bars={bars_flat_due_kill}"
        )

    # --- Cumulative curves ---
    df["creturns"]  = df["returns"].cumsum().apply(np.exp)
    df["cstrategy"] = df["strategy"].cumsum().apply(np.exp)

    # --- CONTINUOUS cums (carry rescale) ---
    start_eq_s = 1.0 if prev_eq_strategy is None else float(prev_eq_strategy)
    start_eq_b = 1.0 if prev_eq_bh       is None else float(prev_eq_bh)
    df["cstrategy_cont"] = df["cstrategy"] * start_eq_s
    df["creturns_cont"]  = df["creturns"]  * start_eq_b

    # --- Guards ---
    if df.empty or df["cstrategy"].dropna().empty or df["creturns"].dropna().empty:
        try:
            df.attrs["last_position"] = float(prev_position) if prev_position is not None else 0.0
            df.attrs["end_eq_strategy"] = float(prev_eq_strategy) if prev_eq_strategy is not None else 1.0
            df.attrs["end_eq_bh"] = float(prev_eq_bh) if prev_eq_bh is not None else 1.0
        except Exception:
            pass
        return (np.nan,) * 16

    # --- MONTHLY metrics ---
    perf      = float(df["cstrategy"].iloc[-1])
    creturns  = float(df["creturns"].iloc[-1])
    outperf   = perf - creturns

    bars_per_year = estimate_frequency_per_year(df.index)
    sharpe, drawdown, trades = compute_metrics(
        df["strategy"], df["position_exec"],
        frequency_per_year=bars_per_year,
        sharpe_cap=float(os.environ.get("SHARPE_CAP", 30.0)),
        use_hac=True, hac_max_lag="auto",
    )

    # --- Reliability guard (MinTRL-style) ---
    # If there are too few trades, Sharpe/PSR/DSR are not statistically reliable
    # (Bailey & Lopez de Prado, 2012, 2014). We keep the PnL but mark the
    # Sharpe as NaN so that higher-level stats don't over-interpret it.:contentReference[oaicite:3]{index=3}
    try:
        # IMPORTANT: default must not allow "hero Sharpe" from 1-5 trades.
        min_trades_rel = int(_cfg("min_trades_for_reliability", 10))
    except Exception:
        min_trades_rel = 30

    if trades is not None and trades < min_trades_rel:
        sharpe = float("nan")
        # optional: flag on the df for later debugging
        # Optional audit print (off by default to avoid spam in CV)
        if bool(_cfg("eval_print_reliability_debug", False)):
            print(f"[Eval][Reliability] trades={int(trades)} < min_trades_rel={int(min_trades_rel)} -> sharpe=NaN{_ctx}")

    geo_mean_ann = compute_geometric_mean_annualized(df["strategy"])

    df["true_direction"] = np.sign(df["returns"])
    df["pred_direction"] = df["pred"]

    # Deadzone for coercing continuous preds into {-1,0,+1}
    deadzone = float(_cfg("pred_dir_deadzone", 0.5))
    
    # Defensive: keep classification + trade-based metrics strictly discrete {-1,0,+1}
    y_true = _coerce_direction_labels(df["true_direction"].values, deadzone=deadzone)
    y_pred = _coerce_direction_labels(df["pred_direction"].values, deadzone=deadzone)
    pred_lbl = pd.Series(y_pred, index=df.index)
    precision_macro, f1_macro, _cm = _macro_prec_f1_from_confusion(y_true, y_pred)
    df.attrs["confusion_matrix"] = _cm
    
    # Trade-intent precision: P(correct direction | model chose to trade).
    # This avoids the "precision looks good because we stayed flat" trap.
    try:
        trade_mask = (y_pred != 0)
        n_trade_preds = int(np.sum(trade_mask))
        precision_trade = float(np.mean(y_pred[trade_mask] == y_true[trade_mask])) if n_trade_preds > 0 else 0.0
    except Exception:
        n_trade_preds = 0
        precision_trade = 0.0

    # Expose without changing the 16-metric tuple schema.
    try:
        df.attrs["precision_trade"] = float(precision_trade)
        df.attrs["n_trade_preds"] = int(n_trade_preds)
    except Exception:
        pass


    # Use discrete labels for hit-rate + profit-per-hit too (works with fractional positions)
    hit_mask = pd.Series((y_pred == y_true), index=df.index)
    directional_accuracy = float(hit_mask.mean()) if len(hit_mask) else 0.0

    # Activity metrics:
    #   - exec_active_rate: canonical "in market" share based on executed position
    #   - signal_coverage: secondary metric based on non-neutral model labels
    # NOTE: We keep the 16-metric tuple schema unchanged: METRIC_NAMES["active_rate"]
    # now corresponds to exec_active_rate.
    signal_coverage = float((pred_lbl != 0).mean())
    exec_active_rate = float((df["position_exec"].fillna(0).to_numpy() != 0).mean())
    active_rate = exec_active_rate

    # Expose both metrics for downstream logging/export without changing the 16-metric tuple.
    try:
        df.attrs["exec_active_rate"] = float(exec_active_rate)
        df.attrs["signal_coverage"] = float(signal_coverage)
    except Exception:
        pass

    correct_returns = float(df.loc[hit_mask, "strategy"].sum()) if len(df) else 0.0
    hits = int(hit_mask.sum()) if len(hit_mask) else 0
    profit_per_hit = float(correct_returns) / float(hits if hits > 0 else 1)

    return_per_trade = (perf - 1.0) / trades if trades > 0 else 0.0
    trade_edge = pred_lbl.diff().fillna(0) != 0
    trade_returns = df["strategy"][trade_edge]
    num_wins = int((trade_returns > 0).sum())
    win_rate = float(num_wins) / float(trades) if trades > 0 else 0.0

    volatility = float(np.std(df["strategy"]))
    excess_kurtosis = float(kurtosis(df["strategy"], fisher=True))

    # --- carry-out state (CONTINUOUS equities) ---
    # Carry the EXECUTED position stream when available.
    # 'position' is the (intent/derived) position series; 'position_exec' is what was actually held.
    _pos_col = "position_exec" if ("position_exec" in df.columns) else "position"
    try:
        _lp = df[_pos_col].iloc[-1]
        if pd.isna(_lp):
            _lp = df[_pos_col].ffill().iloc[-1]
        df.attrs["last_position"] = float(0.0 if pd.isna(_lp) else _lp)
    except Exception:
        # Last resort: default to flat
        df.attrs["last_position"] = 0.0
    df.attrs["end_eq_strategy"] = float(df["cstrategy_cont"].iloc[-1])
    df.attrs["end_eq_bh"]       = float(df["creturns_cont"].iloc[-1])

    return (
        round(perf, 6),                # 1  (monthly strategy factor)
        round(outperf, 6),             # 2
        round(creturns, 6),            # 3  (monthly B&H factor)
        sharpe,                        # 4
        drawdown,                      # 5
        trades,                        # 6
        round(geo_mean_ann, 6),        # 7
        round(directional_accuracy, 4),# 8
        round(precision_macro, 4),     # 9
        round(f1_macro, 4),            # 10
        round(active_rate, 4),         # 11
        round(profit_per_hit, 6),      # 12
        round(return_per_trade, 6),    # 13
        round(win_rate, 4),            # 14
        round(volatility, 6),          # 15
        round(excess_kurtosis, 4)      # 16
    )


