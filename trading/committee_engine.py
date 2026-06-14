"""Committee trading engine — execute committee signals with risk controls.

Wraps a LiveCommitteeRunner's signal output through the same 3-layer risk gate
pipeline as OandaTradingEngine. Supports paper (simulated fills) and live
(OANDA) modes.

API
---
- ``CommitteeTradingEngine`` — stateful engine, same lifecycle as OandaTradingEngine.
- ``start()`` / ``stop()`` / ``emergency_kill()``
- ``process_signal(signal, bid, ask, mid)`` → dict of events with committee_metadata
- ``get_portfolio_state()`` → dict for API/WebSocket
- ``get_summary()`` → dict (Sharpe, Sortino, return%, maxDD, win rate)

Usage::

    engine = CommitteeTradingEngine()
    engine.start({"pair": "EURUSD", "initial_equity": 10000, "mode": "paper"})

    for live_signal in runner_signals:
        ev = engine.process_signal(live_signal, bid=1.0850, ask=1.0852, mid=1.0851)
        # ev contains committee_metadata for WS rendering

    summary = engine.stop(bid=last_bid, ask=last_ask)
"""

from __future__ import annotations

import logging
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
from trading.risk_controls import (
    LiveRiskConfig,
    LiveRiskState,
    check_all_pre_trade_gates,
    check_all_post_trade_gates,
    check_all_infra_guards,
    record_api_error,
    record_api_success,
    record_trade_submission,
    new_session_state,
    is_killed,
    manual_kill,
)

logger = logging.getLogger(__name__)

_DIRECTION_MAP: dict[str, int] = {"LONG": 1, "SHORT": -1, "FLAT": 0}
_DIRECTION_REV: dict[int, str] = {1: "LONG", -1: "SHORT", 0: "FLAT"}

_SIZING_LOOKUP: dict[str, str] = {
    "fixed": SizingMethod.FIXED,
    "fixed_fractional": SizingMethod.FIXED_FRACTIONAL,
    "kelly": SizingMethod.KELLY,
    "atr": SizingMethod.ATR,
    "vol_target": SizingMethod.VOL_TARGET,
}


@dataclass
class CommitteeTrade:
    trade_id: str
    direction: int
    size: float
    entry_price: float
    entry_time: float
    exit_time: float | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    exit_reason: str = ""
    regime: str = ""
    confidence: float = 0.0
    active_models: list[str] = field(default_factory=list)

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
class CommitteePortfolio:
    initial_equity: float = 10000.0
    equity: float = 10000.0
    position: int = 0
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_sum: float = 0.0
    open_trade: CommitteeTrade | None = None
    closed_trades: list[CommitteeTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    signal_count: int = 0


class CommitteeTradingEngine:
    """Stateful committee trading engine with 3-layer risk gates.

    Accepts LiveSignal objects from LiveCommitteeRunner.process_bar() and runs
    them through the same risk pipeline as OandaTradingEngine. Emits events in
    the same format so the frontend WS handler works identically for both
    committee and single-model sessions.
    """

    _SIZING_LOOKUP = _SIZING_LOOKUP

    def __init__(self) -> None:
        self.portfolio: CommitteePortfolio | None = None
        self._sizing_config: SizingConfig | None = None
        self._sizing_state: SizingState | None = None
        self._risk_state: LiveRiskState | None = None
        self._risk_config: LiveRiskConfig | None = None
        self._oanda: object | None = None
        self._mode: str = "paper"
        self._pair: str = ""
        self._instrument: str = ""
        self._stopped: bool = False
        self._session_start: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────

    def start(
        self,
        config: dict,
        oanda_client: object | None = None,
    ) -> CommitteePortfolio:
        mode = config.get("mode", "paper")
        pair = config.get("pair", "EURUSD")
        self._pair = pair
        self._mode = mode
        self._instrument = self._pair_to_instrument(pair)

        initial_equity = float(config.get("initial_equity", 10000))
        sizing_key = config.get("position_sizing", "fixed")
        sizing_cfg_overrides = config.get("sizing_config", {})
        risk_overrides = config.get("risk_config", {})

        method = self._SIZING_LOOKUP.get(sizing_key, SizingMethod.FIXED)
        self._sizing_config = SizingConfig(method=method, initial_equity=initial_equity)
        for k, v in sizing_cfg_overrides.items():
            if hasattr(self._sizing_config, k):
                setattr(self._sizing_config, k, v)

        trust_mult = float(config.get("trust_multiplier", 1.0))
        self._sizing_config.trust_multiplier = max(0.0, min(1.0, trust_mult))

        self._sizing_state = SizingState(equity=initial_equity)

        self._risk_config = LiveRiskConfig(initial_equity=initial_equity)
        if not isinstance(risk_overrides, dict):
            risk_overrides = {}
        for k, v in risk_overrides.items():
            if hasattr(self._risk_config, k):
                setattr(self._risk_config, k, v)

        self._risk_state = new_session_state(self._risk_config)
        if oanda_client is not None:
            self._oanda = oanda_client
        self._stopped = False
        self._session_start = time.time()

        self.portfolio = CommitteePortfolio(initial_equity=initial_equity, equity=initial_equity)
        self.portfolio.equity_curve = [{"time": self._session_start, "equity": initial_equity}]
        return self.portfolio

    def stop(self, bid: float | None = None, ask: float | None = None) -> dict:
        self._stopped = True
        p = self.portfolio
        if p is None:
            return {"error": "no_active_session", "stopped": True}

        events: list[dict] = []
        now = time.time()

        if self._mode == "live" and self._oanda is not None and p.position != 0:
            try:
                result = self._oanda.close_position(self._instrument)
                events.append({
                    "event": "position_closed",
                    "instrument": self._instrument,
                    "result": str(result)[:200],
                })
            except Exception as exc:
                logger.exception("Failed to close position on stop")
                events.append({
                    "event": "close_failed",
                    "instrument": self._instrument,
                    "error": str(exc),
                })

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

    def emergency_kill(self, oanda_client: object | None = None) -> dict:
        client = oanda_client or self._oanda
        results = []
        if client is not None:
            try:
                results = client.close_all()
            except Exception as exc:
                logger.exception("Emergency kill failed")
                results = [{"action": "emergency_kill", "error": str(exc)}]

        if self._risk_state is not None:
            manual_kill(self._risk_state, "emergency_kill_button")

        return {
            "killed": True,
            "close_results": results,
            "stopped": True,
        }

    # ── Core signal processing ─────────────────────────────────

    def process_signal(
        self,
        signal: object,           # LiveSignal from LiveCommitteeRunner
        bid: float,
        ask: float,
        mid: float,
        oanda_client: object | None = None,
    ) -> dict:
        if self._stopped:
            return {"event": "already_stopped"}

        p = self.portfolio
        rs = self._risk_state
        rc = self._risk_config
        if p is None or rs is None or rc is None:
            return {"event": "error", "message": "engine_not_started"}

        signal_val = int(getattr(signal, "signal", 0) or 0)
        signal_val = max(-1, min(1, signal_val))

        confidence = float(getattr(signal, "confidence", 0.5) * 100)
        regime = str(getattr(signal, "regime", "unknown") or "unknown")
        active_models = list(getattr(signal, "active_models", []) or [])
        model_weights = list(getattr(signal, "model_weights", []) or [])
        blended_probs = getattr(signal, "blended_probs", {}) or {}
        regime_prob = float(getattr(signal, "regime_prob", 0.0) or 0.0)
        meta_override = bool(getattr(signal, "meta_override", False))
        is_healthy = bool(getattr(signal, "is_healthy", True))
        conviction_multiplier = float(getattr(signal, "conviction_multiplier", 1.0) or 1.0)
        conviction_multiplier = max(0.25, min(2.0, conviction_multiplier))

        direction_str = _DIRECTION_REV.get(signal_val, "FLAT")
        target = signal_val
        now = time.time()
        events: list[dict] = []

        rs.last_heartbeat = now
        rs.current_equity = p.equity

        kill, reason, level = check_all_infra_guards(
            rs, rc, signal_timestamp=now,
        )
        if kill:
            return self._emit_kill(reason, level, events)

        p.signal_count += 1

        if target == p.position and target != 0:
            p.unrealized_pnl = p.position * (mid - p.entry_price) * p.size
            p.equity = p.initial_equity + p.realized_sum + p.unrealized_pnl
            p.equity_curve.append({"time": now, "equity": p.equity})

            result: dict = {
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
                "committee_metadata": self._build_committee_metadata(signal),
            }
            return result

        if p.position != 0 and p.open_trade is not None:
            exit_price = bid if p.position == 1 else ask
            reason_close = "signal_reversal" if target != 0 else "signal_flat"
            closed_pnl = p.open_trade.close(exit_price, reason=reason_close, ts=now)
            closed_is_win = closed_pnl > 0
            p.realized_sum += closed_pnl
            p.closed_trades.append(p.open_trade)
            update_state(self._sizing_state, closed_pnl, closed_is_win)

            events.append({
                "event": "trade_closed",
                "trade_id": p.open_trade.trade_id,
                "direction": _DIRECTION_REV[p.open_trade.direction],
                "entry_price": p.open_trade.entry_price,
                "exit_price": exit_price,
                "pnl": round(closed_pnl, 2),
                "is_win": closed_is_win,
                "exit_reason": reason_close,
                "time": now,
            })

            p.open_trade = None
            p.position = 0
            p.size = 0.0
            p.entry_price = 0.0
            p.unrealized_pnl = 0.0

            kill, reason, level = check_all_post_trade_gates(
                rs, rc, closed_pnl, closed_is_win,
            )
            if kill:
                return self._emit_kill(reason, level, events)

        if is_killed(rs):
            return self._emit_kill(rs.kill_reason, rs.kill_level, events)

        if target == 0:
            p.equity_curve.append({"time": now, "equity": p.equity})

            result = {
                "event": "signal",
                "direction": "FLAT",
                "confidence": confidence,
                "mid_price": mid,
                "position": "FLAT",
                "equity": round(p.equity, 2),
                "time": now,
                "committee_metadata": self._build_committee_metadata(signal),
            }
            if events:
                result["sub_events"] = events
            return result

        size = float(compute_size(self._sizing_state, 0.0, 0.0, self._sizing_config))
        if size <= 0:
            size = 1.0
        size = size * conviction_multiplier

        if self._mode == "live" and oanda_client is not None:
            try:
                account = oanda_client.get_account_summary()
                margin_used_pct_raw = float(account.get("marginUsed", 0))
                balance = float(account.get("balance", p.equity))
                nav = float(account.get("NAV", p.equity))
                if balance > 0:
                    rs.current_equity = nav
                    p.equity = nav
                    margin_used_pct = margin_used_pct_raw / max(balance, 1.0)
                else:
                    margin_used_pct = 0.0
                record_api_success(rs)
            except Exception:
                margin_used_pct = 0.0
                record_api_error(rs)
        else:
            margin_used_pct = 0.0

        signal_dict = {"direction": direction_str, "confidence": confidence}
        allowed, blocked_reason, blocked_list = check_all_pre_trade_gates(
            rs, rc, signal_dict, size, direction_str, margin_used_pct,
        )

        if not allowed:
            p.equity_curve.append({"time": now, "equity": p.equity})

            result = {
                "event": "risk_blocked",
                "direction": direction_str,
                "confidence": confidence,
                "reason": blocked_reason,
                "all_reasons": blocked_list,
                "time": now,
                "committee_metadata": self._build_committee_metadata(signal),
            }
            return result

        if self._mode == "live" and oanda_client is not None:
            try:
                instrument = self._instrument
                units = max(1, int(round(size)))
                order_result = oanda_client.place_market_order(instrument, units)
                fill_price = 0.0
                result_body = order_result.get("orderFillTransaction", order_result)
                if isinstance(result_body, dict):
                    try:
                        fill_price = float(result_body.get("price", 0))
                    except (TypeError, ValueError):
                        fill_price = 0.0
                if fill_price <= 0:
                    fill_price = ask if target == 1 else bid

                entry_price = fill_price
                record_api_success(rs)
                record_trade_submission(rs, direction_str)

                events.append({
                    "event": "order_placed",
                    "direction": direction_str,
                    "instrument": instrument,
                    "units": units,
                    "price": entry_price,
                    "confidence": confidence,
                    "time": now,
                })
            except Exception as exc:
                record_api_error(rs)
                logger.exception("OANDA order failed for %s", self._pair)
                return {
                    "event": "order_failed",
                    "direction": direction_str,
                    "confidence": confidence,
                    "error": str(exc),
                    "time": now,
                }
        else:
            entry_price = ask if target == 1 else bid

        trade = CommitteeTrade(
            trade_id=self._next_trade_id(),
            direction=target,
            size=size,
            entry_price=entry_price,
            entry_time=now,
            regime=regime,
            confidence=confidence,
            active_models=list(active_models),
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
            "regime": regime,
            "active_models": list(active_models),
            "time": now,
        })

        p.unrealized_pnl = (
            p.position * (mid - p.entry_price) * p.size if p.position != 0 else 0.0
        )
        p.equity = p.initial_equity + p.realized_sum + p.unrealized_pnl
        p.equity_curve.append({"time": now, "equity": p.equity})

        result = {
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
            "committee_metadata": self._build_committee_metadata(signal),
        }
        if events:
            result["sub_events"] = events
        return result

    # ── Telemetry ──────────────────────────────────────────────

    def _build_committee_metadata(self, signal: object) -> dict:
        regime = str(getattr(signal, "regime", "unknown") or "unknown")
        regime_prob = float(getattr(signal, "regime_prob", 0.0) or 0.0)
        active_models = list(getattr(signal, "active_models", []) or [])
        model_weights = [float(w) for w in (getattr(signal, "model_weights", []) or [])]
        blended_probs = getattr(signal, "blended_probs", {}) or {}
        confidence = float(getattr(signal, "confidence", 0.5))
        conviction_multiplier = float(getattr(signal, "conviction_multiplier", 1.0) or 1.0)
        meta_override = bool(getattr(signal, "meta_override", False))
        is_healthy = bool(getattr(signal, "is_healthy", True))

        models_detail = []
        for i, name in enumerate(active_models):
            weight = model_weights[i] if i < len(model_weights) else 1.0
            prob = blended_probs.get(str(i), 0)
            models_detail.append({
                "name": name,
                "weight": round(weight, 4),
            })

        prob_list: list[float] = [0.0, 0.0, 0.0]
        if isinstance(blended_probs, dict):
            prob_list[0] = float(blended_probs.get("0", blended_probs.get("short", 0.0)))
            prob_list[1] = float(blended_probs.get("1", blended_probs.get("flat", 0.0)))
            prob_list[2] = float(blended_probs.get("2", blended_probs.get("long", 0.0)))

        return {
            "regime": regime,
            "regime_confidence": round(regime_prob, 4),
            "active_models": models_detail,
            "blended_probs": [round(p, 4) for p in prob_list],
            "conviction_multiplier": round(conviction_multiplier, 2),
            "meta_learner_active": False,
            "meta_learner_override": "overrode_committee" if meta_override else None,
            "is_healthy": is_healthy,
        }

    # ── Query methods ──────────────────────────────────────────

    def heartbeat(self) -> None:
        if self._risk_state is not None:
            self._risk_state.last_heartbeat = time.time()

    def get_summary(self) -> dict:
        p = self.portfolio
        if p is None:
            return {}

        closed = p.closed_trades
        if not closed:
            return {
                "sharpe": 0.0,
                "sortino": 0.0,
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
            rets = eq.pct_change().dropna()
            rets = rets[rets.abs() > 1e-12]
            if len(rets) > 1 and rets.std() > 1e-8:
                ann_factor = np.sqrt(min(252, len(rets)))
                sharpe = float(rets.mean() / rets.std() * ann_factor)
                downside = rets[rets < 0]
                sortino_std = float(downside.std()) if len(downside) > 1 else rets.std()
                sortino = float(rets.mean() / max(sortino_std, 1e-8) * ann_factor)
                peak = eq.cummax()
                max_dd = float(abs((eq / peak - 1).min()) * 100)
            else:
                sharpe, sortino, max_dd = 0.0, 0.0, 0.0
        else:
            sharpe, sortino, max_dd = 0.0, 0.0, 0.0

        total_return = (p.equity - p.initial_equity) / p.initial_equity * 100
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        win_rate = len(wins) / max(len(closed), 1)
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_win / max(gross_loss, 1e-8)
        avg_trade_pnl = sum(t.pnl for t in closed) / len(closed)

        return {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate": round(win_rate, 4),
            "total_trades": len(closed),
            "profit_factor": round(profit_factor, 4),
            "avg_trade_pnl": round(avg_trade_pnl, 4),
            "final_equity": round(p.equity, 2),
            "signal_count": p.signal_count,
        }

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
                "regime": p.open_trade.regime,
                "active_models": p.open_trade.active_models,
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
                "regime": t.regime,
                "confidence": round(t.confidence, 1),
                "active_models": t.active_models,
            }
            for t in page
        ]

    def get_risk_state(self) -> dict:
        rs = self._risk_state
        if rs is None:
            return {}

        return {
            "equity_peak": round(rs.equity_peak, 2),
            "current_equity": round(rs.current_equity, 2),
            "consecutive_losses": rs.consecutive_losses,
            "daily_trades": rs.daily_trades,
            "hourly_trades": rs.hourly_trades,
            "total_trades": rs.total_trades,
            "total_wins": rs.total_wins,
            "paused": rs.paused,
            "pause_reason": rs.pause_reason,
            "killed": rs.killed,
            "kill_reason": rs.kill_reason,
            "kill_level": rs.kill_level,
        }

    # ── Internal helpers ───────────────────────────────────────

    def _next_trade_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _emit_kill(self, reason: str, level: str, events: list[dict]) -> dict:
        result: dict = {
            "event": "kill",
            "reason": reason,
            "level": level,
            "time": time.time(),
        }
        if events:
            result["sub_events"] = events
        return result

    @staticmethod
    def _pair_to_instrument(pair: str) -> str:
        p = pair.upper().strip()
        if "_" not in p and len(p) == 6:
            p = p[:3] + "_" + p[3:]
        return p
