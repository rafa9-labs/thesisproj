"""
Execution patches for the backtest evaluation loop.

Extracted from pipeline/metrics_eval.py (Phase 3, step 3.1).

Contains:
- PatchConfig:   dataclass holding all patch configuration parsed from df.attrs
- LoopResult:    dataclass holding loop outputs (position, PnL arrays + diagnostics)
- run_execution_loop(): the main bar-by-bar execution loop with all patches

Patches:
  #0 -- Position sizer          (sizing_method: fixed/fractional/kelly/atr/vol_target)
  #1 -- Vol-target sizer        (eval_use_vol_target, legacy -- delegates to #0 when sizing_method="vol_target")
  #2 -- Trailing stop / scale-out (eval_use_scaleout_trail)
  #3 -- TWAP executor            (eval_use_twap_execution)
  #4 -- Regime adapter           (eval_use_regime_adaptive)
  #5 -- Kill switch              (eval_use_kill_switch)
  +  SpreadGuard                (eval_use_spread_guard)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pipeline.execution.position_sizing import (
    SizingConfig,
    SizingState,
    SizingMethod,
    compute_size as _compute_size,
    update_state as _update_sizing_state,
)

from pipeline.execution.stops import (
    StopConfig,
    StopLevels,
    StopMethod as _StopMethod,
    compute_stop_levels as _compute_stop_levels,
    check_stop_hit as _check_stop_hit,
    check_breakeven as _check_breakeven,
)

from pipeline.execution.trailing import (
    TrailingConfig,
    TrailingState,
    TrailingMethod as _TrailingMethod,
    update_trailing_state as _update_trailing_state,
    is_activated as _is_trailing_activated,
    compute_trailing_sl as _compute_trailing_sl,
)

from pipeline.execution.risk_manager import (
    RiskConfig,
    RiskState,
    should_suppress_entry as _risk_should_suppress,
    check_drawdown as _risk_check_drawdown,
    check_daily_loss as _risk_check_daily_loss,
    update_after_trade as _risk_update_after_trade,
    tick_cooloffs as _risk_tick_cooloffs,
    reset_daily as _risk_reset_daily,
    get_pause_reason as _risk_get_pause_reason,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _conviction_multiplier(confidence: float) -> float:
    """Tier-based conviction multiplier (mirrors ConvictionSizer's fallback)."""
    if confidence >= 0.80:
        return 1.5
    if confidence >= 0.65:
        return 1.0
    if confidence >= 0.55:
        return 0.5
    return 1.0


@dataclass
class PatchConfig:
    """All execution-patch configuration, parsed from df.attrs['features_config']."""

    # Cadence
    bars_per_day: int = 48
    annual_bars: int = 12096
    vol_floor: float = 1e-6

    # Patch #1 -- Vol-target sizer
    use_vol_target: bool = False
    target_bar: float = 0.0
    max_lev: float = 1.5

    # Conviction sizing (opt-in; mirrors live committee sizing)
    use_conviction_sizing: bool = False

    # Patch #2 -- Trailing stop / scale-out
    use_trail: bool = False
    tp1_z_base: float = 1.0
    trail_k_base: float = 2.5
    dyn_vol: bool = True
    move_to_be: bool = True
    max_hold_bars: int = 0
    min_hold_bars: int = 0

    # Patch #3 -- TWAP executor
    use_twap: bool = False
    twap_span: int = 2
    impact_eta: float = 0.0
    twap_freeze: bool = True

    # Patch #4 -- Regime adapter
    use_regime: bool = False
    tp1_z_calm: Optional[float] = None
    tp1_z_normal: Optional[float] = None
    tp1_z_volatile: Optional[float] = None
    trail_k_calm: Optional[float] = None
    trail_k_normal: Optional[float] = None
    trail_k_volatile: Optional[float] = None

    # Patch #5 -- Kill switch
    use_kill: bool = False
    kill_mode: str = "pct"          # "pct" or "sigma"
    kill_pct: float = 0.02
    kill_sigma_k: float = 3.0
    kill_until_session_end: bool = True
    kill_cooloff_bars: int = 0

    # SpreadGuard
    use_spread_guard: bool = True
    spread_cap: float = 0.00040

    # Patch #0 -- Position sizing (Sprint 2)
    sizing_method: str = "fixed"
    sizing_risk_fraction: float = 0.02
    sizing_kelly_fraction: float = 0.5
    sizing_kelly_min_trades: int = 10
    sizing_atr_risk_pct: float = 0.02
    sizing_atr_sl_mult: float = 2.0
    sizing_initial_equity: float = 10_000.0
    sizing_max_leverage: float = 5.0
    sizing_contract_size: float = 100_000.0

    # Patch #0b -- Stop-loss / take-profit (Sprint 2)
    stop_method: str = "none"
    stop_sl_pips: float = 30.0
    stop_tp_pips: float = 60.0
    stop_sl_atr_mult: float = 2.0
    stop_tp_atr_mult: float = 3.0
    stop_sl_sigma_mult: float = 2.0
    stop_tp_sigma_mult: float = 3.0
    stop_pip_value: float = 0.0001
    stop_use_be: bool = False
    stop_be_trigger_pips: float = 20.0
    stop_use_partial_close: bool = False
    stop_tp1_ratio: float = 0.5
    stop_tp1_pips: float = 30.0
    stop_tp2_pips: float = 0.0

    # Patch #0c -- Trailing stops (Sprint 2)
    trailing_method: str = "none"
    trailing_pips: float = 30.0
    trailing_atr_mult: float = 3.0
    trailing_chandelier_atr_mult: float = 3.0
    trailing_chandelier_lookback: int = 22
    trailing_activation_pips: float = 10.0
    trailing_pip_value: float = 0.0001

    # Patch #0d -- Risk management (Sprint 2)
    risk_use_dd_breaker: bool = False
    risk_max_drawdown_pct: float = 0.15
    risk_dd_resume: str = "session_end"
    risk_dd_cooloff_bars: int = 48
    risk_use_daily_loss: bool = False
    risk_max_daily_loss_pct: float = 0.05
    risk_max_daily_loss_sigma: float = 3.0
    risk_daily_loss_mode: str = "pct"
    risk_use_consec_loss: bool = False
    risk_max_consecutive_losses: int = 5
    risk_consec_resume: str = "session_end"
    risk_consec_cooloff_bars: int = 48
    risk_initial_equity: float = 10_000.0
    risk_max_open_positions: int = 1

    # Diagnostics
    debug_costs: bool = False
    eval_context: Optional[str] = None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    """Outputs from the execution loop."""

    # Core output arrays (length n)
    pos_actual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    strat: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Per-bar stop/TP fill price (0.0 where no stop fill occurred)
    stop_fill_price: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Diagnostic counters
    tp1_hits: int = 0
    stop_hits: int = 0
    timeouts: int = 0
    flips_exits: int = 0
    twap_events: int = 0
    total_ramp_bars: int = 0
    total_impact_cost: float = 0.0
    total_slippage_cost: float = 0.0
    spread_spike_blocked: int = 0
    kills_triggered: int = 0
    bars_flat_due_kill: int = 0

    # Position-sizing diagnostics
    sizing_method_used: str = "fixed"
    final_equity: float = 10_000.0

    # Stop/TP diagnostics
    sl_hits: int = 0
    tp_hits: int = 0
    tp1_partial_hits: int = 0
    tp2_full_hits: int = 0
    be_activations: int = 0
    stop_method_used: str = "none"

    # Trailing stop diagnostics
    trailing_activations: int = 0
    trailing_method_used: str = "none"

    # Risk manager diagnostics
    risk_dd_breaches: int = 0
    risk_daily_loss_breaches: int = 0
    risk_consec_loss_breaches: int = 0
    risk_bars_paused: int = 0
    risk_manager_active: bool = False

    # Optional per-bar cost audit arrays (None when not recording)
    cost_spread_pf: Optional[np.ndarray] = None
    cost_slip_pf: Optional[np.ndarray] = None
    cost_impact_bar: Optional[np.ndarray] = None
    cost_total_turn: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def run_execution_loop(
    df,
    pred: np.ndarray,
    rets: np.ndarray,
    bar_vol: np.ndarray,
    gap_from_prev_bool: np.ndarray,
    regime_code: np.ndarray,
    cfg: PatchConfig,
    trading_costs: bool,
    slippage_factor: float,
    session_flag_arr: Optional[np.ndarray] = None,
    record_cost_columns: bool = False,
    confidence: Optional[np.ndarray] = None,
) -> LoopResult:
    """Run the bar-by-bar execution loop with all patches applied.

    This is the tight inner loop extracted from
    ``compute_full_evaluation_metrics`` (lines 580-959 of the original).
    Stop/TP exits are booked at the stop price (clamped to the triggering
    bar's high/low range) instead of skipping the exit bar's return.
    """
    n = len(df)
    if n != len(pred) or n != len(rets) or n != len(bar_vol):
        raise ValueError(
            f"Array length mismatch: df={n}, pred={len(pred)}, "
            f"rets={len(rets)}, bar_vol={len(bar_vol)}"
        )
    _ctx = f" | ctx={cfg.eval_context}" if cfg.eval_context else ""

    # --- Output arrays ---
    pos_actual = np.zeros(n, dtype=float)
    strat = np.zeros(n, dtype=float)
    stop_fill_price_arr = np.zeros(n, dtype=float)

    # Optional cost-audit arrays
    cost_spread_pf = np.zeros(n, dtype=float) if record_cost_columns else None
    cost_slip_pf = np.zeros(n, dtype=float) if record_cost_columns else None
    cost_impact_bar = np.zeros(n, dtype=float) if record_cost_columns else None
    cost_total_turn = np.zeros(n, dtype=float) if record_cost_columns else None

    # --- Unpack config into locals for tight loop performance ---
    use_kill = cfg.use_kill
    kill_until_session_end = cfg.kill_until_session_end
    use_spread_guard = cfg.use_spread_guard
    spread_cap = cfg.spread_cap
    debug_costs = cfg.debug_costs
    use_trail = cfg.use_trail
    use_twap = cfg.use_twap
    twap_span = cfg.twap_span
    use_vol_target = cfg.use_vol_target
    use_conviction_sizing = bool(cfg.use_conviction_sizing) and confidence is not None
    vol_floor = cfg.vol_floor
    target_bar = cfg.target_bar
    max_lev = cfg.max_lev
    use_regime = cfg.use_regime
    tp1_z_calm = cfg.tp1_z_calm
    tp1_z_normal = cfg.tp1_z_normal
    tp1_z_volatile = cfg.tp1_z_volatile
    trail_k_calm = cfg.trail_k_calm
    trail_k_normal = cfg.trail_k_normal
    trail_k_volatile = cfg.trail_k_volatile
    tp1_z_base = cfg.tp1_z_base
    trail_k_base = cfg.trail_k_base
    dyn_vol = cfg.dyn_vol
    move_to_be = cfg.move_to_be
    min_hold_bars = cfg.min_hold_bars
    max_hold_bars = cfg.max_hold_bars
    impact_eta = cfg.impact_eta
    twap_freeze = cfg.twap_freeze
    kill_mode = cfg.kill_mode
    kill_pct = cfg.kill_pct
    kill_sigma_k = cfg.kill_sigma_k
    kill_cooloff_bars = cfg.kill_cooloff_bars
    bars_per_day = cfg.bars_per_day

    # --- Position state ---
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

    # --- Position-sizing setup (Sprint 2) ---
    sizing_method = cfg.sizing_method
    _has_atr_col = "atr_14" in df.columns
    sizing_cfg = SizingConfig(
        method=sizing_method,
        risk_fraction=cfg.sizing_risk_fraction,
        kelly_fraction=cfg.sizing_kelly_fraction,
        kelly_min_trades=cfg.sizing_kelly_min_trades,
        atr_risk_pct=cfg.sizing_atr_risk_pct,
        atr_sl_mult=cfg.sizing_atr_sl_mult,
        initial_equity=cfg.sizing_initial_equity,
        max_leverage=cfg.sizing_max_leverage,
        contract_size=cfg.sizing_contract_size,
        target_bar=target_bar,
        vol_floor=vol_floor,
        max_lev=max_lev,
    )
    sizing_state = SizingState(equity=cfg.sizing_initial_equity)
    trade_pnl_accum = 0.0

    # --- Stop-loss / take-profit setup (Sprint 2) ---
    stop_method = cfg.stop_method
    _has_high_low = ("high" in df.columns or "High" in df.columns) and (
        "low" in df.columns or "Low" in df.columns
    )
    stop_cfg = StopConfig(
        method=stop_method,
        sl_pips=cfg.stop_sl_pips,
        tp_pips=cfg.stop_tp_pips,
        sl_atr_mult=cfg.stop_sl_atr_mult,
        tp_atr_mult=cfg.stop_tp_atr_mult,
        sl_sigma_mult=cfg.stop_sl_sigma_mult,
        tp_sigma_mult=cfg.stop_tp_sigma_mult,
        pip_value=cfg.stop_pip_value,
        use_be=cfg.stop_use_be,
        be_trigger_pips=cfg.stop_be_trigger_pips,
        use_partial_close=cfg.stop_use_partial_close,
        tp1_ratio=cfg.stop_tp1_ratio,
        tp1_pips=cfg.stop_tp1_pips,
        tp2_pips=cfg.stop_tp2_pips,
    )
    active_stop_levels: Optional[StopLevels] = None
    entry_price: float = 0.0
    stop_sl_hits = 0
    stop_tp_hits = 0
    stop_tp1_partial_hits = 0
    stop_tp2_full_hits = 0
    stop_be_activations = 0
    partial_closed = False

    # --- Stop-fill accounting state (P3) ---
    # When a stop/TP is hit, the exit is booked at the stop price (clamped
    # to the bar's high/low range) instead of skipping the exit bar's move.
    stop_fill_ret: Optional[float] = None
    stop_fill_size: float = 0.0
    # Trades closed by signal/stops flush their accumulated PnL (including
    # the exit bar) into the sizing/risk state AFTER the bar is booked.
    deferred_trade_close = False

    # --- Trailing stop setup (Sprint 2) ---
    trailing_method = cfg.trailing_method
    trailing_cfg = TrailingConfig(
        method=trailing_method,
        trail_pips=cfg.trailing_pips,
        trail_atr_mult=cfg.trailing_atr_mult,
        chandelier_atr_mult=cfg.trailing_chandelier_atr_mult,
        chandelier_lookback=cfg.trailing_chandelier_lookback,
        activation_pips=cfg.trailing_activation_pips,
        pip_value=cfg.trailing_pip_value,
    )
    active_trailing_state: Optional[TrailingState] = None
    trailing_activations_count = 0

    # --- Risk management setup (Sprint 2) ---
    risk_any_active = (
        cfg.risk_use_dd_breaker or cfg.risk_use_daily_loss or cfg.risk_use_consec_loss
    )
    risk_cfg = RiskConfig(
        risk_use_dd_breaker=cfg.risk_use_dd_breaker,
        risk_max_drawdown_pct=cfg.risk_max_drawdown_pct,
        risk_dd_resume=cfg.risk_dd_resume,
        risk_dd_cooloff_bars=cfg.risk_dd_cooloff_bars,
        risk_use_daily_loss=cfg.risk_use_daily_loss,
        risk_max_daily_loss_pct=cfg.risk_max_daily_loss_pct,
        risk_max_daily_loss_sigma=cfg.risk_max_daily_loss_sigma,
        risk_daily_loss_mode=cfg.risk_daily_loss_mode,
        risk_use_consec_loss=cfg.risk_use_consec_loss,
        risk_max_consecutive_losses=cfg.risk_max_consecutive_losses,
        risk_consec_resume=cfg.risk_consec_resume,
        risk_consec_cooloff_bars=cfg.risk_consec_cooloff_bars,
        risk_initial_equity=cfg.risk_initial_equity,
        risk_max_open_positions=cfg.risk_max_open_positions,
    )
    risk_state = RiskState(equity_peak=cfg.risk_initial_equity)

    def _get_atr(i):
        if _has_atr_col:
            try:
                return float(df["atr_14"].iloc[i])
            except Exception:
                return 0.0
        return 0.0

    def _get_price(i):
        for col in ("close", "Close", "mid_close", "price"):
            if col in df.columns:
                try:
                    v = float(df[col].iloc[i])
                    if v > 0 and np.isfinite(v):
                        return v
                except Exception:
                    pass
        return 0.0

    def _get_high_low(i):
        h_col = "high" if "high" in df.columns else ("High" if "High" in df.columns else None)
        l_col = "low" if "low" in df.columns else ("Low" if "Low" in df.columns else None)
        hi = 0.0
        lo = 0.0
        if h_col:
            try:
                hi = float(df[h_col].iloc[i])
            except Exception:
                pass
        if l_col:
            try:
                lo = float(df[l_col].iloc[i])
            except Exception:
                pass
        return hi, lo

    # --- Inner helpers (closures over mutable state) ---

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
        size = 1.0
        if sizing_method == "fixed":
            if use_vol_target:
                denom = max(sigma_ref, vol_floor)
                size = min(max_lev, float(target_bar) / float(denom))
            else:
                size = 1.0
        else:
            atr_i = _get_atr(i)
            size = _compute_size(sizing_state, sigma_ref, atr_i, sizing_cfg)
        if use_conviction_sizing and confidence is not None and i < len(confidence):
            size = size * _conviction_multiplier(float(confidence[i]))
        return size

    def regime_tp_trail(i, sigma_ref):
        if not use_regime:
            return float(tp1_z_base), float(trail_k_base)
        code = regime_code[i]
        if code == 0:  # calm
            tp1 = tp1_z_calm if tp1_z_calm is not None else tp1_z_base
            tk = trail_k_calm if trail_k_calm is not None else trail_k_base
        elif code == 2:  # volatile
            tp1 = tp1_z_volatile if tp1_z_volatile is not None else tp1_z_base
            tk = trail_k_volatile if trail_k_volatile is not None else trail_k_base
        else:  # normal
            tp1 = tp1_z_normal if tp1_z_normal is not None else tp1_z_base
            tk = trail_k_normal if trail_k_normal is not None else trail_k_base
        return float(tp1), float(tk)

    def reset_daily_state():
        nonlocal day_pl, kill_active, cooloff_remaining
        day_pl = 0.0
        kill_active = False
        cooloff_remaining = 0

    # ==================================================================
    # Main loop
    # ==================================================================
    prev_date = None
    for i in range(n):
        ts = df.index[i]
        cur_date = ts.date()
        new_day = (prev_date is None) or (cur_date != prev_date)
        prev_date = cur_date

        # Reset daily state on new calendar day
        if new_day:
            reset_daily_state()
            if risk_any_active:
                _risk_reset_daily(risk_state, risk_cfg)
        # Also reset if we prefer to end cool-off / kill at new session boundary
        if kill_until_session_end and gap_from_prev_bool[i]:
            reset_daily_state()

        # If kill active, suppress signals (force flat target)
        sig = np.sign(pred[i])

        # Session gating (optional):
        if session_flag_arr is not None and session_flag_arr[i] == 0 and (not in_pos):
            sig_session = 0.0
        else:
            sig_session = sig

        # SpreadGuard: block NEW entries on spike spreads (allow exits/holds)
        if use_spread_guard:
            try:
                _spr_full = float(df["spread"].iloc[i]) if ("spread" in df.columns) else 0.0
            except Exception:
                _spr_full = 0.0

            if (prev_pos_actual == 0.0) and (sig_session != 0.0) and np.isfinite(_spr_full) and (_spr_full > spread_cap):
                sig_session = 0.0
                spread_spike_blocked += 1

                if debug_costs:
                    print(f"[Eval][SpreadGuard] spike @ {df.index[i]} spread={_spr_full:.6f} > cap={spread_cap:.6f} -> block entry{_ctx}")

        # Risk manager signal suppression (S2.4)
        if risk_any_active and _risk_should_suppress(risk_state):
            sig_session = 0.0
            _risk_tick_cooloffs(risk_cfg, risk_state)

        if use_kill:
            if kill_active:
                sig_for_logic = 0.0
                bars_flat_due_kill += 1
                if (not kill_until_session_end) and cooloff_remaining > 0:
                    cooloff_remaining -= 1
                if (not kill_until_session_end) and cooloff_remaining == 0:
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
                    partial_closed = False
                    entry_price = _get_price(i - 1) if i > 0 else _get_price(i)
                    if stop_method != "none" and entry_price > 0:
                        active_stop_levels = _compute_stop_levels(
                            stop_cfg, entry_price, dirn,
                            atr=_get_atr(i), bar_vol=bar_vol[i],
                        )
                    else:
                        active_stop_levels = None
                    if trailing_method != "none":
                        active_trailing_state = TrailingState()
                        _entry_px = _get_price(i - 1) if i > 0 else _get_price(i)
                        active_trailing_state.reset(trailing_cfg.chandelier_lookback, entry_price=_entry_px)
                    else:
                        active_trailing_state = None
                    pos_target = dirn * size_entry
                    if use_twap:
                        start_ramp(i, pos_target, prev_pos_actual)
                    else:
                        ramp_active = False
                        ramp_dst = pos_target
                else:
                    pos_target = 0.0
            else:
                bars_held += 1
                sigma_ref = bar_vol[i] if dyn_vol else sigma_entry
                sigma_ref = max(sigma_ref, vol_floor)
                cum_pl += dirn * rets[i]
                mfe = max(mfe, cum_pl)

                # --- Trailing stop update (S2.3) ---
                if active_trailing_state is not None and trailing_method != "none":
                    _tr_hi, _tr_lo = _get_high_low(i)
                    if _tr_hi > 0:
                        _update_trailing_state(trailing_cfg, active_trailing_state, _tr_hi, _tr_lo)
                    _tr_price = _get_price(i)
                    if _tr_price > 0 and _is_trailing_activated(
                        trailing_cfg, active_trailing_state,
                        entry_price, _tr_price, dirn,
                    ):
                        if not active_trailing_state.activated or active_trailing_state.activated:
                            if not active_trailing_state.activated:
                                trailing_activations_count += 1
                        _new_trail_sl = _compute_trailing_sl(
                            trailing_cfg, active_trailing_state, dirn, _get_atr(i),
                        )
                        if _new_trail_sl > 0 and active_stop_levels is not None:
                            if dirn > 0:
                                if _new_trail_sl > active_stop_levels.sl_price:
                                    active_stop_levels.sl_price = _new_trail_sl
                            else:
                                if active_stop_levels.sl_price == 0 or _new_trail_sl < active_stop_levels.sl_price:
                                    active_stop_levels.sl_price = _new_trail_sl

                # --- Stop/TP check (S2.2) ---
                stop_exit = False
                if active_stop_levels is not None and stop_method != "none":
                    _cur_price = _get_price(i)
                    _bar_hi, _bar_lo = _get_high_low(i)
                    if _cur_price > 0:
                        if stop_cfg.use_be and not partial_closed:
                            old_sl = active_stop_levels.sl_price
                            _check_breakeven(stop_cfg, active_stop_levels, entry_price, dirn, _cur_price)
                            if active_stop_levels.sl_price != old_sl:
                                stop_be_activations += 1
                        _hit, _hit_type = _check_stop_hit(active_stop_levels, _cur_price, dirn, _bar_hi, _bar_lo)
                        if _hit:
                            stop_fill_price = 0.0
                            if _hit_type == "sl":
                                stop_sl_hits += 1
                                stop_exit = True
                                stop_fill_price = active_stop_levels.sl_price
                            elif _hit_type == "tp":
                                stop_tp_hits += 1
                                stop_exit = True
                                stop_fill_price = active_stop_levels.tp_price
                            elif _hit_type == "tp1" and stop_cfg.use_partial_close and not partial_closed:
                                stop_tp1_partial_hits += 1
                                partial_closed = True
                                pos_target = dirn * (size_entry * (1.0 - stop_cfg.tp1_ratio))
                                if use_twap:
                                    start_ramp(i, pos_target, prev_pos_actual)
                                else:
                                    ramp_active = False
                                    ramp_dst = pos_target
                            elif _hit_type == "tp2":
                                stop_tp2_full_hits += 1
                                stop_exit = True
                                stop_fill_price = active_stop_levels.tp2_price

                            if stop_exit and stop_fill_price > 0.0 and dirn != 0.0:
                                # Book the exit at the stop/TP fill price. The
                                # directionless price move is clamped to the
                                # triggering bar's realized high/low range,
                                # then signed by the trade direction.
                                prev_close = _get_price(i - 1) if i > 0 else _get_price(i)
                                if prev_close > 0.0:
                                    price_move = float(np.log(stop_fill_price / prev_close))
                                    _b_hi, _b_lo = _get_high_low(i)
                                    if _b_hi > 0.0 and _b_lo > 0.0 and _b_lo < _b_hi:
                                        ret_hi = float(np.log(_b_hi / prev_close))
                                        ret_lo = float(np.log(_b_lo / prev_close))
                                        price_move = min(max(price_move, ret_lo), ret_hi)
                                    ret_stop = dirn * price_move
                                    if np.isfinite(ret_stop):
                                        stop_fill_ret = ret_stop
                                        stop_fill_size = float(abs(prev_pos_actual))
                                        stop_fill_price_arr[i] = float(stop_fill_price)

                if stop_exit:
                    # Exit booked at the stop fill price later in the bar-booking
                    # section; trade PnL is flushed into sizing/risk AFTER booking.
                    deferred_trade_close = True
                    in_pos = False
                    dirn = 0.0
                    size_entry = 0.0
                    scaled_out = False
                    active_stop_levels = None
                    active_trailing_state = None
                    pos_target = 0.0
                    # Stops execute as market orders: no TWAP ramp on stop bars.
                    ramp_active = False
                    ramp_dst = 0.0
                else:
                    # Regime-adaptive TP1/trail
                    tp1_z_eff, trail_k_eff = regime_tp_trail(i, sigma_ref)

                    # TP1 -> scale-out to 1/2
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

                    # Min-hold logic
                    want_flat = (sig_for_logic == 0.0)
                    want_flip = (sig_for_logic == -dirn)

                    if min_hold_bars > 0 and bars_held < min_hold_bars:
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
                        # Signal/trailing exits close at the bar close: defer the
                        # sizing/risk flush until the exit bar PnL is booked.
                        deferred_trade_close = True
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
            if signal_change and prev_sig != 0.0 and sizing_method != "fixed":
                _update_sizing_state(sizing_state, trade_pnl_accum, trade_pnl_accum > 0)
                if risk_any_active:
                    _risk_update_after_trade(risk_cfg, risk_state, trade_pnl_accum, trade_pnl_accum > 0)
                trade_pnl_accum = 0.0
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

        spread_full = float(df["spread"].iloc[i]) if ("spread" in df.columns) else 0.0

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
            spread_per_fill = 0.5 * (spread_full / price_i)
        else:
            spread_per_fill = 0.5 * spread_full

        slip_bps = float(df["slippage_bps"].iloc[i]) if ("slippage_bps" in df.columns) else 0.0
        slip_per_fill = slip_bps / 1e4

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

        # Stop/TP exits book the fill-price return for the pre-exit position
        # instead of skipping the exit bar's move.
        if stop_fill_ret is not None and stop_fill_size > 0.0:
            strat[i] = stop_fill_size * float(stop_fill_ret) - (slip_cost * delta_pos + impact)

        sizing_state.equity += strat[i]
        if in_pos:
            trade_pnl_accum += strat[i]

        # Flush the closed trade (including the exit bar's PnL) into the
        # sizing/risk statistics after the bar is booked.
        if deferred_trade_close:
            trade_pnl_accum += strat[i]
            if risk_any_active:
                _risk_update_after_trade(risk_cfg, risk_state, trade_pnl_accum, trade_pnl_accum > 0)
            if sizing_method != "fixed":
                _update_sizing_state(sizing_state, trade_pnl_accum, trade_pnl_accum > 0)
            trade_pnl_accum = 0.0
            deferred_trade_close = False
        stop_fill_ret = None
        stop_fill_size = 0.0

        pos_actual[i] = pos_exe
        prev_pos_actual = pos_exe

        # --- Kill-switch check/update (after booking this bar's P&L) ---
        if use_kill:
            day_pl += strat[i]

            if kill_mode == "sigma":
                day_sigma_est = max(bar_vol[i], vol_floor) * np.sqrt(bars_per_day)
                loss_limit = kill_sigma_k * day_sigma_est
            else:
                loss_limit = kill_pct

            if (not kill_active) and (day_pl <= -float(loss_limit)):
                kills_triggered += 1
                kill_active = True
                if not kill_until_session_end and kill_cooloff_bars > 0:
                    cooloff_remaining = kill_cooloff_bars
                pos_target = 0.0
                if use_twap:
                    start_ramp(i, 0.0, prev_pos_actual)
                else:
                    ramp_active = False
                    ramp_dst = 0.0

        # --- Risk manager bar-level checks (S2.4) ---
        if risk_any_active:
            _risk_check_drawdown(risk_cfg, risk_state, sizing_state.equity)
            _risk_check_daily_loss(risk_cfg, risk_state, strat[i], bar_vol[i], bars_per_day)

    # ==================================================================
    # Return results
    # ==================================================================
    return LoopResult(
        pos_actual=pos_actual,
        strat=strat,
        stop_fill_price=stop_fill_price_arr,
        tp1_hits=tp1_hits,
        stop_hits=stop_hits,
        timeouts=timeouts,
        flips_exits=flips_exits,
        twap_events=twap_events,
        total_ramp_bars=total_ramp_bars,
        total_impact_cost=total_impact_cost,
        total_slippage_cost=total_slippage_cost,
        spread_spike_blocked=spread_spike_blocked,
        kills_triggered=kills_triggered,
        bars_flat_due_kill=bars_flat_due_kill,
        sizing_method_used=sizing_method,
        final_equity=sizing_state.equity,
        sl_hits=stop_sl_hits,
        tp_hits=stop_tp_hits,
        tp1_partial_hits=stop_tp1_partial_hits,
        tp2_full_hits=stop_tp2_full_hits,
        be_activations=stop_be_activations,
        stop_method_used=stop_method,
        trailing_activations=trailing_activations_count,
        trailing_method_used=trailing_method,
        cost_spread_pf=cost_spread_pf,
        cost_slip_pf=cost_slip_pf,
        cost_impact_bar=cost_impact_bar,
        cost_total_turn=cost_total_turn,
        risk_dd_breaches=risk_state.dd_breaches,
        risk_daily_loss_breaches=risk_state.daily_loss_breaches,
        risk_consec_loss_breaches=risk_state.consec_loss_breaches,
        risk_bars_paused=risk_state.bars_paused,
        risk_manager_active=risk_any_active,
    )