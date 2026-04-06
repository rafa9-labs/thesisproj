"""
Full evaluation metrics - compute_full_evaluation_metrics and helpers.

Extracted from utilsNoWFO.py (Phase 3, step 3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix

def _macro_prec_f1_from_confusion(y_true, y_pred, labels=(-1,0,1)):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    # precision per class: TP / (TP + FP)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec_per_class = np.diag(cm) / cm.sum(axis=0, where=np.ones_like(cm, dtype=bool))
        rec_per_class  = np.diag(cm) / cm.sum(axis=1, where=np.ones_like(cm, dtype=bool))
        f1_per_class = 2 * prec_per_class * rec_per_class / (prec_per_class + rec_per_class)
    prec = np.nanmean(np.where(np.isfinite(prec_per_class), prec_per_class, np.nan))
    f1   = np.nanmean(np.where(np.isfinite(f1_per_class), f1_per_class, np.nan))
    return float(np.nan_to_num(prec, nan=0.0)), float(np.nan_to_num(f1, nan=0.0))

def compute_full_evaluation_metrics(
    df,
    trading_costs=False,
    slippage_factor=0.0,
    prev_position=None,      # carry previous month last position (−1/0/+1)
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
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report  # retained for compatibility
    from scipy.stats import kurtosis

    # Lazy imports from utilsNoWFO to avoid circular dependency
    from utilsNoWFO import (
        compute_metrics,
        compute_geometric_mean_annualized,
        estimate_frequency_per_year,
        _coerce_direction_labels,
    )

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
            # trades_dir = Σ |Δ sign(position)|  (a flip +1→-1 counts as 2 fills)
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
        #   - last bar before a gap had pred=0 → flip_or_flat → forced exit
        #   - first bar after gap also pred=0 → forced start flat
        # Now:
        #   - decisions at gap edges follow the normal trading logic
        #   - if you keep a position, you effectively "carry across the gap".


    # --- Cadence & rolling bar-vol (σ) for several patches ---
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
            print(f"[Eval] Kill clamp: pct {orig_kill_pct:.4%} → {kill_pct:.4%}")
        if kill_mode == "sigma" and kill_sigma_k != orig_sigma:
            print(f"[Eval] Kill clamp: sigma k {orig_sigma:.2f} → {kill_sigma_k:.2f}")
        if kill_cooloff_bars != orig_cool:
            print(f"[Eval] Kill clamp: cool-off {orig_cool} → {kill_cooloff_bars} bars")

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
            # NOTE: don't try to “pretty-print dict literals” with nested braces in f-strings
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


    pos_actual = np.zeros(n, dtype=float)
    strat = np.zeros(n, dtype=float)
    trades_sig = np.zeros(n, dtype=float)

    in_pos = False
    dirn = 0.0
    size_entry = 0.0
    scaled_out = False
    cum_pl = 0.0
    mfe = 0.0
    be_floor = 0.0
    bars_held = 0
    sigma_entry = None

    pos_target = 0.0
    prev_pos_actual = 0.0

    # TWAP state
    ramp_active = False
    ramp_start = -1
    ramp_end = -1
    ramp_src = 0.0
    ramp_dst = 0.0

    # Kill-switch state
    kills_triggered = 0
    kill_active = False
    cooloff_remaining = 0
    day_pl = 0.0
    bars_flat_due_kill = 0

    # Diagnostics
    tp1_hits = 0
    stop_hits = 0
    timeouts = 0
    flips_exits = 0
    twap_events = 0
    total_ramp_bars = 0
    total_impact_cost = 0.0
    total_slippage_cost = 0.0
    spread_spike_blocked = 0
    
    # Optional: record per-bar cost components for auditing (no behavior change).
    # Enabled automatically in debug_costs or explicitly via eval_record_cost_columns.
    record_cost_columns = bool(_cfg("eval_record_cost_columns", False)) or bool(debug_costs)
    if record_cost_columns:
        cost_spread_pf = np.zeros(n, dtype=float)     # fractional per fill
        cost_slip_pf   = np.zeros(n, dtype=float)     # fractional per fill
        cost_impact_bar = np.zeros(n, dtype=float)    # per bar
        cost_total_turn = np.zeros(n, dtype=float)    # (slip_cost*delta_pos + impact) per bar


    # Helpers
    def start_ramp(i, new_target, current_actual):
        nonlocal ramp_active, ramp_start, ramp_end, ramp_src, ramp_dst, twap_events
        ramp_active = True
        ramp_start = i
        ramp_end = i + twap_span - 1
        ramp_src = current_actual
        ramp_dst = new_target
        twap_events += 1

    def ramped_position(i):
        if not ramp_active:
            return ramp_dst
        prog = min(1.0, (i - ramp_start + 1) / float(twap_span))
        return ramp_src + (ramp_dst - ramp_src) * prog

    def get_entry_size(i, sigma_ref):
        if use_vol_target:
            denom = max(sigma_ref, vol_floor)
            return min(max_lev, float(target_bar) / float(denom))
        return 1.0

    def regime_tp_trail(i, sigma_ref):
        # effective tp1_z and trail_k for bar i
        if not use_regime:
            return float(tp1_z_base), float(trail_k_base)
        code = regime_code[i]
        if code == 0:  # calm
            tp1 = tp1_z_calm if tp1_z_calm is not None else tp1_z_base
            tk  = trail_k_calm if trail_k_calm is not None else trail_k_base
        elif code == 2:  # volatile
            tp1 = tp1_z_volatile if tp1_z_volatile is not None else tp1_z_base
            tk  = trail_k_volatile if trail_k_volatile is not None else trail_k_base
        else:  # normal
            tp1 = tp1_z_normal if tp1_z_normal is not None else tp1_z_base
            tk  = trail_k_normal if trail_k_normal is not None else trail_k_base
        return float(tp1), float(tk)

    def reset_daily_state():
        nonlocal day_pl, kill_active, cooloff_remaining
        day_pl = 0.0
        kill_active = False
        cooloff_remaining = 0

    # Main loop
    prev_date = None
    for i in range(n):
        ts = df.index[i]
        cur_date = ts.date()
        new_day = (prev_date is None) or (cur_date != prev_date)
        prev_date = cur_date

        # Reset daily state on new calendar day
        if new_day:
            reset_daily_state()
        # Also reset if we prefer to end cool-off / kill at new session boundary
        if kill_until_session_end and gap_from_prev_bool[i]:
            reset_daily_state()

        # If kill active, suppress signals (force flat target)
        sig = np.sign(pred[i])

        # Session gating (optional):
        # - If we are OUTSIDE session and NOT currently in a position,
        #   we block new entries by zeroing the signal used for logic.
        # - If we are already in a position, we let the exit logic (TP/SL/trail)
        #   handle it normally, even outside the session.
        if session_flag_arr is not None and session_flag_arr[i] == 0 and (not in_pos):
            sig_session = 0.0
        else:
            sig_session = sig


        # SpreadGuard: block NEW entries on spike spreads (allow exits/holds)
        # Use prev_pos_actual (executed position) so it works for both trail and non-trail paths.
        if use_spread_guard:
            try:
                _spr_full = float(df["spread"].iloc[i]) if ("spread" in df.columns) else 0.0
            except Exception:
                _spr_full = 0.0

            # Only block *new entries* (flat -> nonzero). Do NOT interfere with exits.
            if (prev_pos_actual == 0.0) and (sig_session != 0.0) and np.isfinite(_spr_full) and (_spr_full > spread_cap):
                sig_session = 0.0
                spread_spike_blocked += 1

                # Optional per-spike print (keep it behind debug_costs so it doesn’t spam normal runs)
                if debug_costs:
                    print(f"[Eval][SpreadGuard] spike @ {df.index[i]} spread={_spr_full:.6f} > cap={spread_cap:.6f} → block entry{_ctx}")


        if use_kill:
            if kill_active:
                sig_for_logic = 0.0
                bars_flat_due_kill += 1
                if (not kill_until_session_end) and cooloff_remaining > 0:
                    cooloff_remaining -= 1
                if (not kill_until_session_end) and cooloff_remaining == 0:
                    # Cool-off expired → resume
                    kill_active = False
            else:
                sig_for_logic = sig_session
        else:
            sig_for_logic = sig_session

        # ----------------------
        # Trading logic branches
        # ----------------------
        if use_trail:
            # ---- Patch #2 path with (optional) regime-adaptive thresholds ----
            if not in_pos:
                if sig_for_logic != 0.0:
                    in_pos = True
                    dirn = sig_for_logic
                    sigma_entry = bar_vol[i]
                    size_entry = get_entry_size(i, sigma_entry)
                    scaled_out = False
                    cum_pl = 0.0
                    mfe = 0.0
                    be_floor = 0.0
                    bars_held = 0
                    pos_target = dirn * size_entry
                    if use_twap:
                        start_ramp(i, pos_target, prev_pos_actual)
                    else:
                        ramp_active = False
                        ramp_dst = pos_target
                else:
                    pos_target = 0.0
            else:
                # Update P&L state (executed pos used below for strat)
                bars_held += 1
                sigma_ref = bar_vol[i] if dyn_vol else sigma_entry
                sigma_ref = max(sigma_ref, vol_floor)
                cum_pl += dirn * rets[i]
                mfe = max(mfe, cum_pl)

                # Regime-adaptive TP1/trail
                tp1_z_eff, trail_k_eff = regime_tp_trail(i, sigma_ref)

                # TP1 → scale-out to 1/2
                if (not scaled_out) and (cum_pl >= tp1_z_eff * sigma_ref):
                    scaled_out = True
                    tp1_hits += 1
                    if move_to_be:
                        be_floor = 0.0
                    pos_target = dirn * (size_entry * 0.5)
                    if use_twap:
                        start_ramp(i, pos_target, prev_pos_actual)
                    else:
                        ramp_active = False
                        ramp_dst = pos_target

                # trailing stop
                trail_level = mfe - trail_k_eff * sigma_ref
                if move_to_be and scaled_out:
                    trail_level = max(trail_level, be_floor)

                # Min-hold logic: allow exit-to-flat anytime, but block flips before min_hold_bars
                want_flat = (sig_for_logic == 0.0)
                want_flip = (sig_for_logic == -dirn)

                if min_hold_bars > 0 and bars_held < min_hold_bars:
                    # Too early to flip; only allow exit to flat
                    allow_flip = False
                else:
                    allow_flip = True

                flip_or_flat = want_flat or (want_flip and allow_flip)
                max_hold_hit = (max_hold_bars > 0 and bars_held >= max_hold_bars)
                hit_stop = (cum_pl <= trail_level)

                if hit_stop or flip_or_flat or (use_kill and kill_active) or max_hold_hit:
                    if hit_stop:       stop_hits += 1
                    if flip_or_flat:   flips_exits += 1
                    if max_hold_hit:   timeouts += 1
                    in_pos = False
                    dirn = 0.0
                    size_entry = 0.0
                    scaled_out = False
                    pos_target = 0.0
                    if use_twap:
                        start_ramp(i, pos_target, prev_pos_actual)
                    else:
                        ramp_active = False
                        ramp_dst = pos_target
        else:
            # ---- Patch #1 / original path (no trailing) ----
            prev_sig = np.sign(pred[i-1]) if i > 0 else 0.0
            signal_change = (np.sign(sig_for_logic) != np.sign(prev_sig))
            if signal_change and sig_for_logic != 0.0:
                sigma_ref = bar_vol[i]
                size_entry = get_entry_size(i, sigma_ref)
                pos_target = sig_for_logic * (size_entry if (not use_twap or not twap_freeze) else size_entry)
                if use_twap:
                    start_ramp(i, pos_target, prev_pos_actual)
                else:
                    ramp_active = False
                    ramp_dst = pos_target
            elif signal_change and sig_for_logic == 0.0:
                pos_target = 0.0
                if use_twap:
                    start_ramp(i, pos_target, prev_pos_actual)
                else:
                    ramp_active = False
                    ramp_dst = pos_target
            else:
                if use_twap and twap_freeze:
                    pass
                else:
                    if use_vol_target and not use_twap:
                        denom = max(bar_vol[i], vol_floor)
                        size_now = min(max_lev, float(target_bar) / float(denom))
                        pos_target = np.sign(sig_for_logic) * size_now
                    else:
                        pos_target = np.sign(sig_for_logic) * (size_entry if size_entry else 1.0)

        # --- Execute via TWAP ramp or instant ---
        if use_twap:
            pos_exe = ramped_position(i)
            if ramp_active and i >= ramp_end:
                ramp_active = False
            if i >= ramp_start >= 0:
                total_ramp_bars += 1
        else:
            pos_exe = ramp_dst if ramp_dst is not None else pos_target

        # Costs & PnL this bar
        delta_pos = abs(pos_exe - prev_pos_actual)

        # CSV spread = ask_close - bid_close (FULL spread).
        # With mid-price returns, per-fill cost is HALF spread.
        spread_full = float(df["spread"].iloc[i]) if ("spread" in df.columns) else 0.0
        
        # IMPORTANT:
        # spread_full is in PRICE units (ask-bid). Strategy PnL is computed in (log-)return space.
        # Convert spread to a fractional drag by dividing by the current mid price (price column).
        # Fallback keeps legacy behavior only if no usable price is available.
        price_i = None
        if "price" in df.columns:
            try:
                price_i = float(df["price"].iloc[i])
            except Exception:
                price_i = None
        elif "mid_close" in df.columns:
            try:
                price_i = float(df["mid_close"].iloc[i])
            except Exception:
                price_i = None

        if price_i is not None and np.isfinite(price_i) and price_i > 0.0:
            spread_per_fill = 0.5 * (spread_full / price_i)   # fractional return drag per fill
        else:
            spread_per_fill = 0.5 * spread_full               # legacy fallback (assumes already normalized)

        # Vol-aware slippage from _ensure_cost_columns (bps -> fractional return)
        slip_bps = float(df["slippage_bps"].iloc[i]) if ("slippage_bps" in df.columns) else 0.0
        slip_per_fill = slip_bps / 1e4

        # Total per-fill drag, then multiply by turnover (delta_pos)
        slip_cost = (spread_per_fill + (float(slippage_factor) * slip_per_fill)) if trading_costs else 0.0

        impact = impact_eta * delta_pos if trading_costs else 0.0
        total_impact_cost += float(impact)

        if trading_costs:
            try:
                total_slippage_cost += float(slip_cost) * float(delta_pos)
            except Exception:
                pass
            
        if record_cost_columns:
            try:
                cost_spread_pf[i] = float(spread_per_fill)
                cost_slip_pf[i] = float(slip_per_fill)
                cost_impact_bar[i] = float(impact)
                cost_total_turn[i] = float(slip_cost) * float(delta_pos) + float(impact)
            except Exception:
                pass

        strat[i] = pos_exe * rets[i] - (slip_cost * delta_pos + impact)

        pos_actual[i] = pos_exe
        prev_pos_actual = pos_exe

        # --- Kill-switch check/update (after booking this bar's P&L) ---
        if use_kill:
            # Accumulate today's P&L in log-return space
            day_pl += strat[i]

            # Compute today's loss limit
            if kill_mode == "sigma":
                # daily σ ≈ bar σ × sqrt(bars/day)
                day_sigma_est = max(bar_vol[i], vol_floor) * np.sqrt(bars_per_day)
                loss_limit = kill_sigma_k * day_sigma_est
            else:
                loss_limit = kill_pct  # log-return approximation of % loss

            if (not kill_active) and (day_pl <= -float(loss_limit)):
                # trigger kill for the rest of session/day or for cool-off bars
                kills_triggered += 1
                kill_active = True
                if not kill_until_session_end and kill_cooloff_bars > 0:
                    cooloff_remaining = kill_cooloff_bars
                # Force target to flat going forward
                pos_target = 0.0
                if use_twap:
                    start_ramp(i, 0.0, prev_pos_actual)
                else:
                    ramp_active = False
                    ramp_dst = 0.0

    # Trades (reporting continuity)
    trades_sig[1:] = np.abs(np.sign(pred[1:]) - np.sign(pred[:-1])) / 2.0
    df["trades"] = pd.Series(trades_sig, index=df.index)

    # Final series (single source of truth)
    # IMPORTANT: Always overwrite position_exec so metrics/carry-out can’t consume a stale stream
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
    # real-sim results can’t be accused of microstructure distortion.
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
                print(f"⚠️ [Eval] Regime debug print failed: {type(_e).__name__}: {_e}")
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
    # (Bailey & López de Prado, 2012, 2014). We keep the PnL but mark the
    # Sharpe as NaN so that higher-level stats don't over-interpret it.:contentReference[oaicite:3]{index=3}
    try:
        # IMPORTANT: default must not allow "hero Sharpe" from 1–5 trades.
        min_trades_rel = int(_cfg("min_trades_for_reliability", 10))
    except Exception:
        min_trades_rel = 30

    if trades is not None and trades < min_trades_rel:
        sharpe = float("nan")
        # optional: flag on the df for later debugging
        # Optional audit print (off by default to avoid spam in CV)
        if bool(_cfg("eval_print_reliability_debug", False)):
            print(f"[Eval][Reliability] trades={int(trades)} < min_trades_rel={int(min_trades_rel)} → sharpe=NaN{_ctx}")

    geo_mean_ann = compute_geometric_mean_annualized(df["strategy"])

    df["true_direction"] = np.sign(df["returns"])
    df["pred_direction"] = df["pred"]

    # Deadzone for coercing continuous preds into {-1,0,+1}
    deadzone = float(_cfg("pred_dir_deadzone", 0.5))
    
    # Defensive: keep classification + trade-based metrics strictly discrete {-1,0,+1}
    y_true = _coerce_direction_labels(df["true_direction"].values, deadzone=deadzone)
    y_pred = _coerce_direction_labels(df["pred_direction"].values, deadzone=deadzone)
    pred_lbl = pd.Series(y_pred, index=df.index)
    precision_macro, f1_macro = _macro_prec_f1_from_confusion(y_true, y_pred)
    
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


