"""Paper trading engine — simulate trades against live prices using trained models.

Processes signals (LONG/SHORT/FLAT) against bid/ask prices, tracks portfolio
equity, maintains a trade journal, and produces backtest-comparable metrics.

Design
------
- Single-position only (no pyramiding). A reversal signal closes the existing
  position first, then opens the opposite direction.
- Entry: LONG at ask, SHORT at bid (market crosses the spread).
- Exit:  LONG at bid, SHORT at ask.
- Position sizing via ``pipeline.execution.position_sizing`` (same as backtest).
- PnL:  ``direction * (exit - entry) * size``.
- Equity:  ``initial_equity + sum(realised_pnls) + unrealised_pnl``.
- Unrealised:  ``direction * (mid - entry) * size``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.execution.position_sizing import (
    SizingConfig,
    SizingMethod,
    SizingState,
    compute_size,
    update_state,
)

from pipeline.execution.stops import (
    StopConfig,
    StopLevels,
    StopMethod,
    compute_stop_levels,
    check_stop_hit,
)

from trading.vol_tracker import VolTracker

_DIRECTION_MAP = {"LONG": 1, "SHORT": -1, "FLAT": 0}
_DIRECTION_REV = {1: "LONG", -1: "SHORT", 0: "FLAT"}


@dataclass
class Trade:
    trade_id: str
    direction: int
    size: float
    entry_time: float
    entry_price: float
    exit_time: float | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    exit_reason: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def close(self, exit_price: float, reason: str, ts: float | None = None) -> float:
        self.exit_time = ts if ts is not None else time.time()
        self.exit_price = exit_price
        self.exit_reason = reason
        self.pnl = self.direction * (exit_price - self.entry_price) * self.size
        return self.pnl


@dataclass
class PaperPortfolio:
    initial_equity: float = 10000.0
    equity: float = 10000.0
    position: int = 0
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_sum: float = 0.0
    open_trade: Trade | None = None
    closed_trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    signal_count: int = 0


class PaperEngine:
    """Stateful paper trading simulator.

    Usage::

        engine = PaperEngine()
        engine.start({"initial_equity": 20000, "position_sizing": "fixed_fractional"})

        for signal in signal_stream:
            ev = engine.process_signal(signal, bid=1.08500, ask=1.08502, mid=1.08501)
            # ev contains {event, direction, equity, ...} for WebSocket push

        summary = engine.stop()
        print(summary["sharpe"], summary["total_return_pct"])
    """

    _SIZING_LOOKUP: dict[str, str] = {
        "fixed": SizingMethod.FIXED,
        "fixed_fractional": SizingMethod.FIXED_FRACTIONAL,
        "kelly": SizingMethod.KELLY,
        "atr": SizingMethod.ATR,
        "vol_target": SizingMethod.VOL_TARGET,
    }

    def __init__(self) -> None:
        self.portfolio: PaperPortfolio | None = None
        self._sizing_config: SizingConfig | None = None
        self._sizing_state: SizingState | None = None
        self._session_start: float = 0.0
        self._stopped: bool = False
        self._vol_tracker: VolTracker | None = None
        self._stop_cfg: StopConfig | None = None
        self._stop_levels: StopLevels | None = None

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self, config: dict) -> PaperPortfolio:
        initial_equity = float(config.get("initial_equity", 10000))
        method_key = config.get("position_sizing", "fixed")
        sizing_cfg_overrides = config.get("sizing_config", {})

        method = self._SIZING_LOOKUP.get(method_key, SizingMethod.FIXED)
        self._sizing_config = SizingConfig(method=method, initial_equity=initial_equity)
        for k, v in sizing_cfg_overrides.items():
            if hasattr(self._sizing_config, k):
                setattr(self._sizing_config, k, v)
        trust_mult = float(config.get("trust_multiplier", 1.0))
        if not (trust_mult >= 0 and trust_mult <= 1.0):
            trust_mult = 0.0
        self._sizing_config.trust_multiplier = trust_mult

        self._vol_tracker = VolTracker()

        stop_cfg_raw = config.get("stop_config") or {}
        self._stop_cfg = StopConfig(
            method=str(stop_cfg_raw.get("method", StopMethod.NONE)),
            sl_pips=float(stop_cfg_raw.get("sl_pips", 30.0)),
            tp_pips=float(stop_cfg_raw.get("tp_pips", 60.0)),
            sl_atr_mult=float(stop_cfg_raw.get("sl_atr_mult", 2.0)),
            tp_atr_mult=float(stop_cfg_raw.get("tp_atr_mult", 3.0)),
            sl_sigma_mult=float(stop_cfg_raw.get("sl_sigma_mult", 2.0)),
            tp_sigma_mult=float(stop_cfg_raw.get("tp_sigma_mult", 3.0)),
            pip_value=float(stop_cfg_raw.get("pip_value", 0.0001)),
        )
        self._stop_levels = None

        self._sizing_state = SizingState(equity=initial_equity)
        self._session_start = time.time()
        self._stopped = False

        self.portfolio = PaperPortfolio(initial_equity=initial_equity, equity=initial_equity)
        self.portfolio.equity_curve = [{"time": self._session_start, "equity": initial_equity}]
        return self.portfolio

    def stop(self, bid: float | None = None, ask: float | None = None) -> dict:
        self._stopped = True
        p = self.portfolio
        if p is None:
            return {"error": "no active session"}

        events: list[dict] = []
        now = time.time()

        if p.open_trade is not None and p.position != 0:
            exit_price = bid if p.position == 1 else ask
            if exit_price is None or exit_price <= 0:
                exit_price = p.open_trade.entry_price

            p.open_trade.close(exit_price, reason="stop", ts=now)
            p.realized_sum += p.open_trade.pnl
            p.closed_trades.append(p.open_trade)
            update_state(self._sizing_state, p.open_trade.pnl, p.open_trade.pnl > 0)

            events.append({
                "event": "trade_closed",
                "trade_id": p.open_trade.trade_id,
                "direction": _DIRECTION_REV[p.open_trade.direction],
                "entry_price": p.open_trade.entry_price,
                "exit_price": exit_price,
                "pnl": round(p.open_trade.pnl, 2),
                "is_win": p.open_trade.pnl > 0,
                "exit_reason": "stop",
                "time": now,
            })

            p.open_trade = None
            p.position = 0
            p.size = 0.0
            p.entry_price = 0.0
            p.unrealized_pnl = 0.0

        p.equity = p.initial_equity + p.realized_sum
        p.equity_curve.append({"time": now, "equity": p.equity})

        summary = self.get_summary()
        summary["events"] = events
        summary["stopped"] = True
        return summary

    # ── Core signal processing ─────────────────────────────────

    def _close_open_position(self, exit_price: float, reason: str, now: float, events: list) -> None:
        """Close the open trade at the given price and record the event."""
        p = self.portfolio
        if p is None or p.open_trade is None or p.position == 0:
            return
        p.open_trade.close(exit_price, reason=reason, ts=now)
        is_win = p.open_trade.pnl > 0
        p.realized_sum += p.open_trade.pnl
        p.closed_trades.append(p.open_trade)
        update_state(self._sizing_state, p.open_trade.pnl, is_win)

        events.append({
            "event": "trade_closed",
            "trade_id": p.open_trade.trade_id,
            "direction": _DIRECTION_REV[p.open_trade.direction],
            "entry_price": p.open_trade.entry_price,
            "exit_price": exit_price,
            "pnl": round(p.open_trade.pnl, 2),
            "is_win": is_win,
            "exit_reason": reason,
            "time": now,
        })

        p.open_trade = None
        p.position = 0
        p.size = 0.0
        p.entry_price = 0.0
        p.unrealized_pnl = 0.0
        self._stop_levels = None

    def process_signal(self, signal: dict, bid: float, ask: float, mid: float) -> dict:
        if self._stopped:
            return {"event": "already_stopped"}
        p = self.portfolio
        if p is None:
            return {"event": "error", "message": "engine not started"}

        direction_str = signal.get("direction", "FLAT")
        confidence = float(signal.get("confidence", 50.0))
        target = _DIRECTION_MAP.get(direction_str, 0)
        now = time.time()
        events: list[dict] = []

        # Track volatility/ATR so sizing and stops see real values (backtest-aligned)
        bar_vol, atr = self._vol_tracker.update(mid)

        # Stop/TP check against the exit-side quote (mirrors backtest stops)
        if p.position != 0 and self._stop_levels is not None and self._stop_cfg is not None:
            exit_quote = bid if p.position == 1 else ask
            hit, hit_type = check_stop_hit(
                self._stop_levels, exit_quote, float(p.position),
            )
            if hit:
                if hit_type == "sl":
                    fill_price = self._stop_levels.sl_price
                    reason = "stop_loss"
                else:
                    fill_price = (
                        self._stop_levels.tp2_price if hit_type == "tp2"
                        else self._stop_levels.tp_price
                    )
                    reason = "take_profit"
                self._close_open_position(fill_price, reason, now, events)
                p = self.portfolio  # state may have been updated

        if target == p.position and p.position != 0:
            p.unrealized_pnl = p.position * (mid - p.entry_price) * p.size
            p.equity = p.initial_equity + p.realized_sum + p.unrealized_pnl
            p.signal_count += 1
            p.equity_curve.append({"time": now, "equity": p.equity})
            return {
                "event": "hold",
                "direction": direction_str,
                "confidence": confidence,
                "mid_price": mid,
                "bid": bid,
                "ask": ask,
                "equity": round(p.equity, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "position": direction_str,
                "time": now,
            }

        if p.position != 0 and p.open_trade is not None:
            exit_price = bid if p.position == 1 else ask
            reason = "signal_reversal" if target != 0 else "signal_flat"
            self._close_open_position(exit_price, reason, now, events)

        if target != 0:
            entry_price = ask if target == 1 else bid
            size = float(compute_size(self._sizing_state, bar_vol, atr, self._sizing_config))
            if size <= 0:
                size = 1.0

            # Backtest-consistent stop levels (anchored at the entry price)
            self._stop_levels = None
            if self._stop_cfg is not None and self._stop_cfg.method != StopMethod.NONE:
                try:
                    self._stop_levels = compute_stop_levels(
                        self._stop_cfg, entry_price, float(target),
                        atr=atr, bar_vol=bar_vol,
                    )
                except Exception:
                    self._stop_levels = None

            trade = Trade(
                trade_id=self._next_trade_id(),
                direction=target,
                size=size,
                entry_time=now,
                entry_price=entry_price,
            )

            p.position = target
            p.size = size
            p.entry_price = entry_price
            p.open_trade = trade

            events.append({
                "event": "trade_opened",
                "trade_id": trade.trade_id,
                "direction": direction_str,
                "size": size,
                "entry_price": entry_price,
                "confidence": confidence,
                "time": now,
            })

        p.unrealized_pnl = (
            p.position * (mid - p.entry_price) * p.size if p.position != 0 else 0.0
        )
        p.equity = p.initial_equity + p.realized_sum + p.unrealized_pnl
        p.signal_count += 1
        p.equity_curve.append({"time": now, "equity": p.equity})

        result: dict = {
            "event": "signal",
            "direction": direction_str,
            "confidence": confidence,
            "mid_price": mid,
            "bid": bid,
            "ask": ask,
            "equity": round(p.equity, 2),
            "unrealized_pnl": round(p.unrealized_pnl, 2),
            "position": direction_str,
            "time": now,
        }
        if events:
            result["sub_events"] = events
        return result

    # ── Query methods ──────────────────────────────────────────

    def get_summary(self) -> dict:
        p = self.portfolio
        if p is None:
            return {}

        closed = p.closed_trades
        if not closed:
            return {
                "sharpe": None,
                "sortino": None,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
                "final_equity": p.equity,
                "signal_count": p.signal_count,
            }

        curve = p.equity_curve
        if curve and len(curve) > 1:
            eq = pd.Series([d["equity"] for d in curve], dtype=float)
            # Honest session metrics: daily-resampled returns, flat days included,
            # annualized with sqrt(252); None when the session is too short.
            try:
                from pipeline.metrics.metrics_eval import honest_session_metrics
                _hm = honest_session_metrics(curve, periods_per_year=252.0, min_samples=30)
                sharpe = _hm["sharpe"]
                sortino = _hm["sortino"]
            except Exception:
                _hm = {}
                sharpe, sortino = None, None

            peak = eq.cummax()
            max_dd = float(abs((eq / peak - 1).min()) * 100)
        else:
            sharpe, sortino, max_dd = None, None, 0.0

        total_return = (p.equity - p.initial_equity) / p.initial_equity * 100
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        win_rate = len(wins) / len(closed)
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_win / max(gross_loss, 1e-8)
        avg_trade_pnl = sum(t.pnl for t in closed) / len(closed)

        return {
            "sharpe": round(sharpe, 4) if sharpe is not None else None,
            "sortino": round(sortino, 4) if sortino is not None else None,
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate": round(win_rate, 4),
            "total_trades": len(closed),
            "profit_factor": round(profit_factor, 4),
            "avg_trade_pnl": round(avg_trade_pnl, 4),
            "final_equity": round(p.equity, 2),
            "signal_count": p.signal_count,
        }

    def compare_to_backtest(self, backtest_metrics: dict) -> dict:
        paper = self.get_summary()
        if not paper:
            return {}

        bm = backtest_metrics.get("metrics", backtest_metrics)
        comparison: dict[str, dict] = {}
        for key in ("sharpe", "total_return_pct", "max_drawdown_pct", "win_rate"):
            pv = paper.get(key, 0)
            bv = bm.get(key)
            if bv is not None and bv != 0:
                delta = round(pv - bv, 4) if pv is not None else None
                comparison[key] = {"paper": pv, "backtest": bv, "delta": delta}
            else:
                comparison[key] = {"paper": pv, "backtest": None, "delta": None}
        return comparison

    def get_portfolio_state(self) -> dict:
        p = self.portfolio
        if p is None:
            return {"position": "FLAT", "equity": 0}

        state: dict = {
            "position": _DIRECTION_REV.get(p.position, "FLAT"),
            "equity": round(p.equity, 2),
            "size": round(p.size, 4),
            "entry_price": round(p.entry_price, 5) if p.entry_price else None,
            "unrealized_pnl": round(p.unrealized_pnl, 2),
            "signal_count": p.signal_count,
            "total_trades_closed": len(p.closed_trades),
        }

        if p.open_trade:
            state["open_trade"] = {
                "trade_id": p.open_trade.trade_id,
                "direction": _DIRECTION_REV[p.open_trade.direction],
                "size": p.open_trade.size,
                "entry_price": p.open_trade.entry_price,
                "entry_time": p.open_trade.entry_time,
            }

        return state

    def get_trades(self, offset: int = 0, limit: int = 50) -> list[dict]:
        p = self.portfolio
        if p is None:
            return []

        page = p.closed_trades[::-1][offset : offset + limit]
        return [
            {
                "trade_id": t.trade_id,
                "direction": _DIRECTION_REV.get(t.direction, "FLAT"),
                "size": round(t.size, 4),
                "entry_time": t.entry_time,
                "entry_price": round(t.entry_price, 5),
                "exit_time": t.exit_time,
                "exit_price": round(t.exit_price, 5) if t.exit_price else None,
                "pnl": round(t.pnl, 4),
                "exit_reason": t.exit_reason,
            }
            for t in page
        ]

    # ── Internal helpers ───────────────────────────────────────

    def _next_trade_id(self) -> str:
        return uuid.uuid4().hex[:12]
