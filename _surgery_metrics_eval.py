"""One-time surgery: replace execution loop in metrics_eval.py with call to execution_patches."""
import textwrap

with open('pipeline/metrics_eval.py', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original: {len(lines)} lines")

# Verify anchor lines (0-based indices)
assert 'pos_actual = np.zeros' in lines[596], f"Line 597 mismatch: {lines[596]!r}"
assert 'ramp_dst = 0.0' in lines[958], f"Line 959 mismatch: {lines[958]!r}"

# New glue code (replaces lines 597-959, 0-based 596-958)
new_block = textwrap.dedent("""\
    trades_sig = np.zeros(n, dtype=float)

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

    # Unpack results into local variables for post-loop code
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

""")

# Keep lines 0-595 (1-based: 1-596), replace 596-958 (1-based: 597-959), keep 959-end (1-based: 960-end)
new_lines = lines[:596] + [new_block] + lines[959:]

with open('pipeline/metrics_eval.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"New: {len(new_lines)} lines")
print(f"Removed lines 597-959 ({958-596+1} lines), inserted {len(new_block.splitlines())} new lines")