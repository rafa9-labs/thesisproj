"""Live trading risk controls — pre-trade gates, post-trade monitoring, infrastructure guards.

Four-layer risk architecture adapted from ``pipeline/execution/risk_manager.py``
but redesigned for real-time OANDA execution (time-based, not bar-based).

Layers
------
**Layer 1 — Pre-Trade Gates** (checked before every order reaches OANDA):
  G1  Max position size (% equity)
  G2  Max daily trades
  G3  Max trades per hour
  G4  Min signal confidence
  G5  Trade cooldown (min time between trades)
  G6  Trading schedule (weekend gap)
  G7  Account margin check

**Layer 2 — Post-Trade Monitoring** (checked after each trade closes):
  M1  Max drawdown kill switch
  M2  Daily loss limit
  M3  Max consecutive losses
  M4  Rolling Sharpe monitor (last N trades)
  M5  Win rate floor
  M6  Profit factor floor

**Layer 3 — Infrastructure Guards** (checked continuously):
  I1  Signal staleness
  I2  Order timeout
  I3  Stale session heartbeat
  I4  Max session lifetime
  I5  Consecutive API failures
  I6  Minimum account equity

**Layer 4 — Emergency Kill Switch**:
  K1  Manual emergency button
  K2  Auto-kill: margin call detected
  K3  Auto-kill: equity floor breached
  K4  Auto-kill: heartbeat timeout
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass
class LiveRiskConfig:
    """All live trading risk thresholds. Each gate is independently toggleable."""

    enabled: bool = True

    # ── G1: Max position size ─────────────────────────────────
    max_position_pct: float = 0.25

    # ── G2/G3: Trade frequency limits ─────────────────────────
    max_daily_trades: int = 20
    max_hourly_trades: int = 5

    # ── G4: Min signal confidence ──────────────────────────────
    min_confidence: float = 65.0

    # ── G5: Trade cooldown ─────────────────────────────────────
    trade_cooldown_sec: float = 120.0
    reversal_cooldown_sec: float = 60.0

    # ── G6: Trading schedule ───────────────────────────────────
    restrict_weekend: bool = True
    weekend_close_utc_hour: int = 21
    weekend_close_utc_day: int = 4
    weekend_open_utc_hour: int = 21
    weekend_open_utc_day: int = 6

    # ── G7: Account margin ─────────────────────────────────────
    max_margin_used_pct: float = 0.50

    # ── M1: Max drawdown kill switch ───────────────────────────
    use_dd_kill: bool = True
    max_drawdown_pct: float = 0.15

    # ── M2: Daily loss limit ───────────────────────────────────
    use_daily_loss: bool = True
    max_daily_loss_pct: float = 0.05

    # ── M3: Max consecutive losses ─────────────────────────────
    use_consec_loss: bool = True
    max_consecutive_losses: int = 5

    # ── M4: Rolling Sharpe monitor ─────────────────────────────
    use_rolling_sharpe: bool = True
    rolling_window_trades: int = 20
    min_rolling_sharpe: float = -1.0

    # ── M5: Win rate floor ─────────────────────────────────────
    use_win_rate_floor: bool = True
    min_rolling_win_rate: float = 0.25

    # ── M6: Profit factor floor ────────────────────────────────
    use_profit_factor_floor: bool = True
    min_profit_factor: float = 1.0

    # ── I1: Signal staleness ───────────────────────────────────
    max_signal_age_sec: float = 30.0

    # ── I2: Order timeout ──────────────────────────────────────
    order_timeout_sec: float = 10.0

    # ── I3: Heartbeat timeout ──────────────────────────────────
    heartbeat_timeout_sec: float = 300.0

    # ── I4: Max session lifetime ───────────────────────────────
    max_session_hours: float = 24.0

    # ── I5: Consecutive API failures ────────────────────────────
    max_consecutive_api_errors: int = 5

    # ── I6: Minimum account equity ─────────────────────────────
    min_equity_pct: float = 0.50

    initial_equity: float = 10000.0


@dataclass
class LiveRiskState:
    """Mutable risk state tracked across the session lifetime."""

    # Equity tracking
    equity_peak: float = 10000.0
    current_equity: float = 10000.0
    initial_equity: float = 10000.0
    daily_start_equity: float = 10000.0

    # Trade counters
    total_trades: int = 0
    total_wins: int = 0
    consecutive_losses: int = 0
    daily_trades: int = 0
    hourly_trades: int = 0
    current_hour_start: int = 0

    # Rolling trade window (for M4/M5/M6)
    trade_pnl_window: deque = field(default_factory=lambda: deque(maxlen=20))

    # Timestamps
    session_start: float = 0.0
    last_trade_time: float = 0.0
    last_heartbeat: float = 0.0
    last_signal_time: float = 0.0

    # API error tracking
    consecutive_api_errors: int = 0

    # Pause / kill state
    paused: bool = False
    pause_reason: str = ""
    paused_until: float = 0.0

    killed: bool = False
    kill_reason: str = ""
    kill_level: str = ""

    # Breach counters (for diagnostics)
    dd_breaches: int = 0
    daily_loss_breaches: int = 0
    consec_loss_breaches: int = 0
    confidence_rejections: int = 0
    cooldown_rejections: int = 0
    schedule_rejections: int = 0


# ═══════════════════════════════════════════════════════════════════
#  Layer 1 — Pre-Trade Gates
# ═══════════════════════════════════════════════════════════════════

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_GateResult = tuple[bool, str]


def gate_max_position_size(
    state: LiveRiskState, config: LiveRiskConfig, proposed_size: float
) -> _GateResult:
    if proposed_size > state.current_equity * config.max_position_pct:
        return False, f"position_size_exceeds_{config.max_position_pct*100:.0f}pct_equity"
    return True, ""


def gate_max_daily_trades(state: LiveRiskState, config: LiveRiskConfig) -> _GateResult:
    if state.daily_trades >= config.max_daily_trades:
        return False, f"max_daily_trades({config.max_daily_trades})_reached"
    return True, ""


def gate_max_hourly_trades(state: LiveRiskState, config: LiveRiskConfig) -> _GateResult:
    now_hour = int(time.time()) // 3600
    if now_hour != state.current_hour_start:
        state.hourly_trades = 0
        state.current_hour_start = now_hour
    if state.hourly_trades >= config.max_hourly_trades:
        return False, f"max_hourly_trades({config.max_hourly_trades})_reached"
    return True, ""


def gate_min_confidence(
    state: LiveRiskState, config: LiveRiskConfig, confidence: float
) -> _GateResult:
    if confidence < config.min_confidence:
        state.confidence_rejections += 1
        return False, f"confidence_{confidence:.1f}_below_min_{config.min_confidence:.0f}"
    return True, ""


def gate_trade_cooldown(
    state: LiveRiskState,
    config: LiveRiskConfig,
    current_direction: str,
    last_direction: str,
) -> _GateResult:
    if state.last_trade_time <= 0:
        return True, ""

    elapsed = time.time() - state.last_trade_time

    if current_direction == last_direction and last_direction != "FLAT":
        needed = config.trade_cooldown_sec
        if elapsed < needed:
            state.cooldown_rejections += 1
            return False, f"same_direction_cooldown_{needed - elapsed:.0f}s_remaining"
    elif (
        current_direction not in (last_direction, "FLAT")
        and last_direction != "FLAT"
    ):
        needed = config.reversal_cooldown_sec
        if elapsed < needed:
            state.cooldown_rejections += 1
            return False, f"reversal_cooldown_{needed - elapsed:.0f}s_remaining"

    return True, ""


def gate_trading_schedule(config: LiveRiskConfig) -> _GateResult:
    if not config.restrict_weekend:
        return True, ""

    now = _utc_now()
    wd = now.weekday()
    hh = now.hour
    wc_open = config.weekend_open_utc_day
    wc_open_h = config.weekend_open_utc_hour
    wc_close = config.weekend_close_utc_day
    wc_close_h = config.weekend_close_utc_hour

    if wc_close < 0 or wc_close > 6:
        return True, ""

    if wc_close <= wc_open:
        if wd > wc_close and wd < wc_open:
            return False, "weekend_market_closed"
        if wd == wc_close and hh >= wc_close_h:
            return False, "friday_market_closed"
        if wd == wc_open and hh < wc_open_h:
            return False, "sunday_market_not_open_yet"
    else:
        raise ValueError("invalid weekend schedule")

    return True, ""


def gate_margin_check(
    state: LiveRiskState, config: LiveRiskConfig, margin_used_pct: float
) -> _GateResult:
    if margin_used_pct >= config.max_margin_used_pct:
        return False, f"margin_used_{margin_used_pct*100:.0f}pct_exceeds_max_{config.max_margin_used_pct*100:.0f}pct"
    return True, ""


def check_all_pre_trade_gates(
    state: LiveRiskState,
    config: LiveRiskConfig,
    signal: dict,
    proposed_size: float,
    last_direction: str,
    margin_used_pct: float = 0.0,
) -> tuple[bool, str, list[str]]:
    """Run all Layer 1 pre-trade gates. Returns (allowed, first_reason, all_reasons)."""
    if not config.enabled:
        return True, "", []

    confidence = float(signal.get("confidence", 0.0))
    direction = signal.get("direction", "FLAT")

    gates: list[tuple[str, _GateResult]] = [
        ("G1", gate_max_position_size(state, config, proposed_size)),
        ("G2", gate_max_daily_trades(state, config)),
        ("G3", gate_max_hourly_trades(state, config)),
        ("G4", gate_min_confidence(state, config, confidence)),
        ("G5", gate_trade_cooldown(state, config, direction, last_direction)),
        ("G6", gate_trading_schedule(config)),
        ("G7", gate_margin_check(state, config, margin_used_pct)),
    ]

    blocked: list[str] = []
    for gate_id, (allowed, reason) in gates:
        if not allowed:
            blocked.append(f"{gate_id}:{reason}")

    if blocked:
        return False, blocked[0], blocked
    return True, "", []


# ═══════════════════════════════════════════════════════════════════
#  Layer 2 — Post-Trade Monitoring
# ═══════════════════════════════════════════════════════════════════

def _update_rolling_metrics(state: LiveRiskState, pnl: float, is_win: bool) -> None:
    state.trade_pnl_window.append(pnl)
    state.total_trades += 1
    if is_win:
        state.total_wins += 1


def check_drawdown_kill(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    if not config.use_dd_kill:
        return False, ""

    if state.current_equity > state.equity_peak:
        state.equity_peak = state.current_equity

    if state.equity_peak <= 0:
        return False, ""

    dd = (state.equity_peak - state.current_equity) / state.equity_peak
    if dd >= config.max_drawdown_pct:
        state.dd_breaches += 1
        return True, f"drawdown_{dd*100:.1f}pct_exceeds_{config.max_drawdown_pct*100:.0f}pct"
    return False, ""


def check_daily_loss(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    if not config.use_daily_loss:
        return False, ""

    daily_pnl = state.current_equity - state.daily_start_equity
    loss_pct = abs(daily_pnl) / max(state.daily_start_equity, 1.0)
    if daily_pnl < 0 and loss_pct >= config.max_daily_loss_pct:
        state.daily_loss_breaches += 1
        return True, f"daily_loss_{loss_pct*100:.1f}pct_exceeds_{config.max_daily_loss_pct*100:.0f}pct"
    return False, ""


def check_consecutive_losses(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    if not config.use_consec_loss:
        return False, ""

    if state.consecutive_losses >= config.max_consecutive_losses:
        state.consec_loss_breaches += 1
        return True, f"consecutive_losses_{state.consecutive_losses}_exceeds_{config.max_consecutive_losses}"
    return False, ""


def check_rolling_sharpe(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    if not config.use_rolling_sharpe:
        return False, ""

    window = list(state.trade_pnl_window)
    if len(window) < config.rolling_window_trades:
        return False, ""

    recent = window[-config.rolling_window_trades :]
    mean = np.mean(recent)
    std = np.std(recent, ddof=1) if len(recent) > 1 else 1e-8
    if std < 1e-8:
        sharpe = 0.0 if abs(mean) < 1e-8 else float(np.sign(mean) * 100)
    else:
        sharpe = float(mean / std * np.sqrt(len(recent)))

    if sharpe < config.min_rolling_sharpe:
        return True, f"rolling_sharpe_{sharpe:.2f}_below_min_{config.min_rolling_sharpe:.2f}"
    return False, ""


def check_win_rate_floor(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    if not config.use_win_rate_floor:
        return False, ""

    window = list(state.trade_pnl_window)
    if len(window) < config.rolling_window_trades:
        return False, ""

    recent = window[-config.rolling_window_trades :]
    wins = sum(1 for p in recent if p > 0)
    wr = wins / len(recent)
    if wr < config.min_rolling_win_rate:
        return True, f"rolling_win_rate_{wr:.2f}_below_min_{config.min_rolling_win_rate:.2f}"
    return False, ""


def check_profit_factor(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    if not config.use_profit_factor_floor:
        return False, ""

    window = list(state.trade_pnl_window)
    if len(window) < config.rolling_window_trades:
        return False, ""

    recent = window[-config.rolling_window_trades :]
    gross_win = sum(p for p in recent if p > 0)
    gross_loss = abs(sum(p for p in recent if p <= 0))
    pf = gross_win / max(gross_loss, 1e-8)
    if pf < config.min_profit_factor:
        return True, f"profit_factor_{pf:.2f}_below_min_{config.min_profit_factor:.2f}"
    return False, ""


def check_all_post_trade_gates(
    state: LiveRiskState, config: LiveRiskConfig, trade_pnl: float, is_win: bool
) -> tuple[bool, str, str]:
    """Update state with trade result, then run all Layer 2 monitors.

    Returns (killed, kill_reason, kill_level).
    kill_level: "M1" | "M2" | "M3" | "M4" | "M5" | "M6"
    """
    if not config.enabled:
        return False, "", ""

    _update_rolling_metrics(state, trade_pnl, is_win)

    if is_win:
        state.consecutive_losses = 0
    else:
        state.consecutive_losses += 1

    monitors: list[tuple[str, tuple[bool, str]]] = [
        ("M1", check_drawdown_kill(state, config)),
        ("M2", check_daily_loss(state, config)),
        ("M3", check_consecutive_losses(state, config)),
        ("M4", check_rolling_sharpe(state, config)),
        ("M5", check_win_rate_floor(state, config)),
        ("M6", check_profit_factor(state, config)),
    ]

    for level, (triggered, reason) in monitors:
        if triggered:
            return True, reason, level

    return False, "", ""


# ═══════════════════════════════════════════════════════════════════
#  Layer 3 — Infrastructure Guards
# ═══════════════════════════════════════════════════════════════════

def check_signal_staleness(
    state: LiveRiskState, config: LiveRiskConfig, signal_timestamp: float
) -> tuple[bool, str]:
    age = time.time() - signal_timestamp
    if age > config.max_signal_age_sec:
        return True, f"signal_stale_{age:.1f}s_exceeds_max_{config.max_signal_age_sec:.0f}s"
    return False, ""


def check_order_timeout(
    config: LiveRiskConfig, order_time: float
) -> _GateResult:
    elapsed = time.time() - order_time
    if elapsed > config.order_timeout_sec:
        return False, f"order_timeout_{elapsed:.1f}s_exceeds_{config.order_timeout_sec:.0f}s"
    return True, ""


def check_heartbeat(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    if state.last_heartbeat <= 0:
        return False, ""
    gap = time.time() - state.last_heartbeat
    if gap > config.heartbeat_timeout_sec:
        return True, f"heartbeat_lost_{gap:.0f}s_exceeds_max_{config.heartbeat_timeout_sec:.0f}s"
    return False, ""


def check_session_lifetime(state: LiveRiskState, config: LiveRiskConfig) -> tuple[bool, str]:
    if state.session_start <= 0:
        return False, ""
    elapsed_h = (time.time() - state.session_start) / 3600.0
    if elapsed_h > config.max_session_hours:
        return True, f"session_{elapsed_h:.1f}h_exceeds_max_{config.max_session_hours}h"
    return False, ""


def check_consecutive_api_errors(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    if state.consecutive_api_errors >= config.max_consecutive_api_errors:
        return True, f"api_errors_{state.consecutive_api_errors}_exceeds_max_{config.max_consecutive_api_errors}"
    return False, ""


def check_min_equity(
    state: LiveRiskState, config: LiveRiskConfig
) -> tuple[bool, str]:
    floor = config.initial_equity * config.min_equity_pct
    if state.current_equity < floor:
        return True, f"equity_{state.current_equity:.0f}_below_floor_{floor:.0f}"
    return False, ""


def check_all_infra_guards(
    state: LiveRiskState,
    config: LiveRiskConfig,
    signal_timestamp: float = 0.0,
) -> tuple[bool, str, str]:
    """Run all Layer 3 infrastructure checks.

    Returns (killed, reason, kill_level).
    kill_level: "I1" | "I2" | "I3" | "I4" | "I5" | "I6"
    """
    if not config.enabled:
        return False, "", ""

    guards: list[tuple[str, tuple[bool, str]]] = [
        ("I3", check_heartbeat(state, config)),
        ("I4", check_session_lifetime(state, config)),
        ("I5", check_consecutive_api_errors(state, config)),
        ("I6", check_min_equity(state, config)),
    ]

    if signal_timestamp > 0:
        guards.insert(0, ("I1", check_signal_staleness(state, config, signal_timestamp)))

    for level, (triggered, reason) in guards:
        if triggered:
            return True, reason, level

    return False, "", ""


# ═══════════════════════════════════════════════════════════════════
#  Layer 4 — Emergency Kill Switch
# ═══════════════════════════════════════════════════════════════════

def manual_kill(state: LiveRiskState, reason: str = "manual_kill_button") -> None:
    state.killed = True
    state.kill_reason = reason
    state.kill_level = "K1"


def auto_kill_margin_call(
    state: LiveRiskState, margin_used_pct: float, margin_closeout_pct: float = 1.0
) -> bool:
    if margin_used_pct >= margin_closeout_pct:
        state.killed = True
        state.kill_reason = f"margin_call_{margin_used_pct*100:.0f}pct"
        state.kill_level = "K2"
        return True
    return False


def auto_kill_equity_floor(
    state: LiveRiskState, config: LiveRiskConfig
) -> bool:
    floor = config.initial_equity * config.min_equity_pct
    if state.current_equity >= floor:
        return False
    state.killed = True
    state.kill_reason = f"equity_floor_{floor:.0f}"
    state.kill_level = "K3"
    return True


def auto_kill_heartbeat(
    state: LiveRiskState, config: LiveRiskConfig
) -> bool:
    triggered, reason = check_heartbeat(state, config)
    if triggered:
        state.killed = True
        state.kill_reason = reason
        state.kill_level = "K4"
        return True
    return False


def is_killed(state: LiveRiskState) -> bool:
    return state.killed


# ═══════════════════════════════════════════════════════════════════
#  State management helpers
# ═══════════════════════════════════════════════════════════════════

def reset_daily_counters(state: LiveRiskState) -> None:
    state.daily_trades = 0
    state.hourly_trades = 0
    state.current_hour_start = int(time.time()) // 3600
    state.daily_start_equity = state.current_equity


def record_api_error(state: LiveRiskState) -> None:
    state.consecutive_api_errors += 1


def record_api_success(state: LiveRiskState) -> None:
    state.consecutive_api_errors = 0


def record_trade_submission(state: LiveRiskState, direction: str) -> None:
    state.daily_trades += 1
    state.hourly_trades += 1
    state.last_trade_time = time.time()


def new_session_state(config: LiveRiskConfig) -> LiveRiskState:
    state = LiveRiskState(
        equity_peak=config.initial_equity,
        current_equity=config.initial_equity,
        initial_equity=config.initial_equity,
        daily_start_equity=config.initial_equity,
        session_start=time.time(),
        last_heartbeat=time.time(),
        current_hour_start=int(time.time()) // 3600,
    )
    return state
