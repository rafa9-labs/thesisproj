"""Live trading engine — execute trained model signals through OANDA with risk controls.

Extends the paper engine pattern but connects to a real OANDA account.
Every signal passes through Layer 1 (pre-trade gates) before reaching
OANDA. After every trade close, Layer 2 (post-trade monitoring) checks
for kill conditions. Layer 3 (infrastructure) runs continuously.

API
---
- ``OandaTradingEngine`` — stateful engine with same lifecycle as PaperEngine.
- ``start()`` / ``stop()`` / ``emergency_kill()``
- ``process_signal(signal, bid, ask, mid, account)`` → dict of events
- ``get_session_state()`` → dict for WebSocket/API

Usage::

    engine = OandaTradingEngine()
    engine.start(config, oanda_client, sizing_config)

    for signal in signal_stream:
        ev = engine.process_signal(signal, bid, ask, mid, oanda_client)
        # ev may contain "risk_blocked", "order_placed", "order_filled"

    summary = engine.stop(oanda_client)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

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
class LiveJournalEntry:
    trade_id: str
    oanda_order_id: str = ""
    oanda_fill_id: str = ""
    direction: int = 0
    size: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    confidence: float = 50.0
    entry_time: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    is_win: bool = False
    risk_blocked: bool = False
    risk_reason: str = ""


@dataclass
class OandaSessionState:
    running: bool = False
    pair: str = ""
    instrument: str = ""
    mode: str = "demo"
    position: int = 0
    position_units: float = 0.0
    entry_price: float = 0.0
    equity: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used_pct: float = 0.0
    session_id: str = ""
    signal_count: int = 0
    last_direction: str = "FLAT"
    journal: list[LiveJournalEntry] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    killed: bool = False
    kill_reason: str = ""
    kill_level: str = ""


class OandaTradingEngine:
    """Stateful live trading engine with risk gates and OANDA execution."""

    def __init__(self) -> None:
        self.state: OandaSessionState | None = None
        self._risk_state: LiveRiskState | None = None
        self._risk_config: LiveRiskConfig | None = None
        self._sizing_config: SizingConfig | None = None
        self._sizing_state: SizingState | None = None
        self._oanda: object | None = None
        self._stopped: bool = False

    # ── Lifecycle ──────────────────────────────────────────────

    def start(
        self,
        config: dict,
        oanda_client: object,
    ) -> OandaSessionState:
        pair = config.get("pair", "EURUSD")
        instrument = self._pair_to_instrument(pair)
        initial_equity = float(config.get("initial_equity", 10000))
        mode = config.get("mode", "demo")
        sizing_key = config.get("position_sizing", "fixed")
        sizing_cfg_overrides = config.get("sizing_config", {})
        risk_overrides = config.get("risk_config", {})

        method = _SIZING_LOOKUP.get(sizing_key, SizingMethod.FIXED)
        self._sizing_config = SizingConfig(method=method, initial_equity=initial_equity)
        for k, v in sizing_cfg_overrides.items():
            if hasattr(self._sizing_config, k):
                setattr(self._sizing_config, k, v)
        trust_mult = float(config.get("trust_multiplier", 1.0))
        if not (trust_mult >= 0 and trust_mult <= 1.0):
            trust_mult = 0.0
        self._sizing_config.trust_multiplier = trust_mult

        self._sizing_state = SizingState(equity=initial_equity)

        self._risk_config = LiveRiskConfig(initial_equity=initial_equity)
        if not isinstance(risk_overrides, dict):
            risk_overrides = {}
        for k, v in risk_overrides.items():
            if hasattr(self._risk_config, k):
                setattr(self._risk_config, k, v)

        self._risk_state = new_session_state(self._risk_config)
        self._oanda = oanda_client
        self._stopped = False

        sid = uuid.uuid4().hex[:12]

        self.state = OandaSessionState(
            running=True,
            pair=pair,
            instrument=instrument,
            mode=mode,
            position=0,
            equity=initial_equity,
            session_id=sid,
            signal_count=0,
            last_direction="FLAT",
            equity_curve=[{"time": time.time(), "equity": initial_equity}],
        )

        return self.state

    def stop(self, oanda_client: object | None = None) -> dict:
        self._stopped = True
        st = self.state
        if st is None:
            return {"error": "no_active_session", "stopped": True}

        client = oanda_client or self._oanda
        events: list[dict] = []

        if getattr(st, "position", 0) != 0 and client is not None:
            try:
                result = client.close_position(st.instrument)
                events.append({
                    "event": "position_closed",
                    "instrument": st.instrument,
                    "result": str(result)[:200],
                })
            except Exception as exc:
                logger.exception("Failed to close position on stop")
                events.append({
                    "event": "close_failed",
                    "instrument": st.instrument,
                    "error": str(exc),
                })

        st.running = False
        st.position = 0
        st.position_units = 0.0

        return {
            "session_id": st.session_id,
            "stopped": True,
            "events": events,
            "journal_length": len(st.journal),
            "signal_count": st.signal_count,
            "equity": st.equity,
            "killed": st.killed,
        }

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

        st = self.state
        if st is not None:
            st.running = False
            st.killed = True
            st.kill_reason = "emergency_kill_button"
            st.kill_level = "K1"

        return {
            "session_id": st.session_id if st else "",
            "killed": True,
            "close_results": results,
            "stopped": True,
        }

    # ── Core signal processing ─────────────────────────────────

    def process_signal(
        self,
        signal: dict,
        bid: float,
        ask: float,
        mid: float,
        oanda_client: object,
    ) -> dict:
        if self._stopped:
            return {"event": "already_stopped"}

        st = self.state
        rs = self._risk_state
        rc = self._risk_config
        if st is None or rs is None or rc is None:
            return {"event": "error", "message": "engine_not_started"}

        direction_str = signal.get("direction", "FLAT")
        confidence = float(signal.get("confidence", 50.0))
        target = _DIRECTION_MAP.get(direction_str, 0)
        now = time.time()
        events: list[dict] = []

        rs.last_heartbeat = now
        rs.current_equity = st.equity

        kill, reason, level = check_all_infra_guards(
            rs, rc, signal_timestamp=signal.get("_timestamp", now)
        )
        if kill:
            return self._emit_kill(reason, level, events)

        st.signal_count += 1

        if target == st.position and target != 0:
            if mid > 0 and st.entry_price > 0:
                st.unrealized_pnl = target * (mid - st.entry_price) * st.position_units
            st.equity_curve.append({"time": now, "equity": st.equity})
            return {
                "event": "hold",
                "direction": direction_str,
                "confidence": confidence,
                "mid_price": mid,
                "position": direction_str,
                "equity": round(st.equity, 2),
                "unrealized_pnl": round(st.unrealized_pnl, 2),
                "time": now,
            }

        if st.position != 0:
            close_result = self._close_position(oanda_client, st, rs, rc)
            events.append(close_result)
            st.position = 0
            st.position_units = 0.0
            st.entry_price = 0.0
            st.unrealized_pnl = 0.0

        if is_killed(rs):
            return self._emit_kill(rs.kill_reason, rs.kill_level, events)

        if target == 0:
            st.last_direction = "FLAT"
            rs.current_equity = st.equity
            st.equity_curve.append({"time": now, "equity": st.equity})

            result: dict = {
                "event": "signal",
                "direction": "FLAT",
                "confidence": confidence,
                "mid_price": mid,
                "position": "FLAT",
                "equity": round(st.equity, 2),
                "time": now,
            }
            if events:
                result["sub_events"] = events
            return result

        size = float(compute_size(self._sizing_state, 0.0, 0.0, self._sizing_config))
        if size <= 0:
            size = 1.0

        try:
            account = oanda_client.get_account_summary()
            margin_used_pct_raw = float(account.get("marginUsed", 0))
            balance = float(account.get("balance", st.equity))
            nav = float(account.get("NAV", st.equity))
            if balance > 0:
                rs.current_equity = nav
                st.equity = nav
                margin_used_pct = margin_used_pct_raw / max(balance, 1.0)
            else:
                margin_used_pct = 0.0
            record_api_success(rs)
        except Exception:
            margin_used_pct = 0.0
            record_api_error(rs)

        allowed, blocked_reason, blocked_list = check_all_pre_trade_gates(
            rs, rc, signal, size, st.last_direction, margin_used_pct
        )

        if not allowed:
            st.last_direction = direction_str
            st.equity_curve.append({"time": now, "equity": st.equity})

            journal_entry = LiveJournalEntry(
                trade_id=uuid.uuid4().hex[:12],
                direction=target,
                size=size,
                entry_price=0.0,
                confidence=confidence,
                entry_time=now,
                risk_blocked=True,
                risk_reason=blocked_reason,
            )
            st.journal.append(journal_entry)

            return {
                "event": "risk_blocked",
                "direction": direction_str,
                "confidence": confidence,
                "reason": blocked_reason,
                "all_reasons": blocked_list,
                "time": now,
            }

        try:
            instrument = st.instrument
            units = max(1, int(round(size)))
            order_result = oanda_client.place_market_order(instrument, units)

            oanda_order_id = order_result.get("orderFillTransaction", {}).get("orderID", "")
            oanda_fill_id = order_result.get("orderFillTransaction", {}).get("id", "")
            fill_price = 0.0
            result_body: dict = order_result.get("orderFillTransaction", order_result)
            if isinstance(result_body, dict):
                try:
                    fill_price = float(result_body.get("price", 0))
                except (TypeError, ValueError):
                    fill_price = 0.0

            if fill_price <= 0:
                fill_price = ask if target == 1 else bid

            st.position = target
            st.position_units = float(units)
            st.entry_price = fill_price
            st.last_direction = direction_str

            record_api_success(rs)
            record_trade_submission(rs, direction_str)

            journal_entry = LiveJournalEntry(
                trade_id=uuid.uuid4().hex[:12],
                oanda_order_id=oanda_order_id,
                oanda_fill_id=oanda_fill_id,
                direction=target,
                size=float(units),
                entry_price=fill_price,
                confidence=confidence,
                entry_time=now,
            )
            st.journal.append(journal_entry)

            events.append({
                "event": "order_placed",
                "direction": direction_str,
                "instrument": instrument,
                "units": units,
                "price": fill_price,
                "oanda_order_id": oanda_order_id,
                "oanda_fill_id": oanda_fill_id,
                "confidence": confidence,
                "time": now,
            })

        except Exception as exc:
            record_api_error(rs)
            logger.exception("OANDA order failed for %s", st.pair)
            return {
                "event": "order_failed",
                "direction": direction_str,
                "confidence": confidence,
                "error": str(exc),
                "time": now,
            }

        st.unrealized_pnl = 0.0
        st.equity = rs.current_equity
        st.equity_curve.append({"time": now, "equity": st.equity})

        result = {
            "event": "signal",
            "direction": direction_str,
            "confidence": confidence,
            "mid_price": mid,
            "position": direction_str,
            "equity": round(st.equity, 2),
            "time": now,
        }
        if events:
            result["sub_events"] = events
        return result

    # ── Position close with risk monitoring ────────────────────

    def _close_position(
        self,
        oanda_client: object,
        st: OandaSessionState,
        rs: LiveRiskState,
        rc: LiveRiskConfig,
    ) -> dict:
        try:
            position_info = oanda_client.get_position(st.instrument)
        except Exception:
            record_api_error(rs)
            return {"event": "close_failed", "error": "failed_to_fetch_position"}

        if position_info is None or not isinstance(position_info, dict):
            st.position = 0
            return {"event": "position_flat"}

        long_pl = float(position_info.get("long", {}).get("pl", 0))
        short_pl = float(position_info.get("short", {}).get("pl", 0))
        trade_pnl = long_pl + short_pl
        is_win = trade_pnl > 0

        try:
            oanda_client.close_position(st.instrument)
            record_api_success(rs)
        except Exception as exc:
            record_api_error(rs)
            logger.exception("Failed to close position for %s", st.instrument)
            return {
                "event": "close_failed",
                "instrument": st.instrument,
                "error": str(exc),
            }

        entry_price = st.entry_price
        exit_price = 0.0
        long_avg = position_info.get("long", {}).get("averagePrice")
        short_avg = position_info.get("short", {}).get("averagePrice")
        if st.position == 1 and long_avg:
            exit_price = float(long_avg)
        elif st.position == -1 and short_avg:
            exit_price = float(short_avg)

        closed_journal = LiveJournalEntry(
            trade_id=uuid.uuid4().hex[:12],
            direction=st.position,
            size=st.position_units,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=trade_pnl,
            confidence=0.0,
            entry_time=0.0,
            exit_time=time.time(),
            exit_reason="signal_change",
            is_win=is_win,
        )
        st.journal.append(closed_journal)

        update_state(self._sizing_state, trade_pnl, is_win)

        kill, reason, level = check_all_post_trade_gates(rs, rc, trade_pnl, is_win)
        if kill:
            st.killed = True
            st.kill_reason = reason
            st.kill_level = level
            return {
                "event": "trade_closed_and_killed",
                "pnl": round(trade_pnl, 2),
                "is_win": is_win,
                "kill_reason": reason,
                "kill_level": level,
            }

        return {
            "event": "trade_closed",
            "pnl": round(trade_pnl, 2),
            "is_win": is_win,
        }

    # ── State & helpers ────────────────────────────────────────

    def heartbeat(self) -> None:
        if self._risk_state is not None:
            self._risk_state.last_heartbeat = time.time()

    def get_session_state(self) -> dict:
        st = self.state
        if st is None:
            return {"running": False}

        return {
            "running": st.running,
            "session_id": st.session_id,
            "pair": st.pair,
            "mode": st.mode,
            "position": _DIRECTION_REV.get(st.position, "FLAT"),
            "equity": round(st.equity, 2),
            "unrealized_pnl": round(st.unrealized_pnl, 2),
            "signal_count": st.signal_count,
            "total_journal_entries": len(st.journal),
            "killed": st.killed,
            "kill_reason": st.kill_reason,
            "kill_level": st.kill_level,
        }

    def get_journal(self, offset: int = 0, limit: int = 50) -> list[dict]:
        st = self.state
        if st is None:
            return []

        page = st.journal[::-1][offset : offset + limit]
        return [
            {
                "trade_id": e.trade_id,
                "oanda_order_id": e.oanda_order_id,
                "oanda_fill_id": e.oanda_fill_id,
                "direction": _DIRECTION_REV.get(e.direction, "FLAT"),
                "size": round(e.size, 4),
                "entry_price": round(e.entry_price, 5),
                "exit_price": round(e.exit_price, 5) if e.exit_price else None,
                "pnl": round(e.pnl, 4),
                "confidence": round(e.confidence, 1),
                "is_win": e.is_win,
                "exit_reason": e.exit_reason,
                "risk_blocked": e.risk_blocked,
                "risk_reason": e.risk_reason,
            }
            for e in page
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
            "consecutive_api_errors": rs.consecutive_api_errors,
            "signal_age_sec": round(time.time() - rs.last_signal_time, 1) if rs.last_signal_time else 0,
        }

    def _emit_kill(self, reason: str, level: str, events: list[dict]) -> dict:
        st = self.state
        if st is not None:
            st.running = False
            st.killed = True
            st.kill_reason = reason
            st.kill_level = level

        return {
            "event": "kill",
            "reason": reason,
            "level": level,
            "sub_events": events,
            "time": time.time(),
        }

    @staticmethod
    def _pair_to_instrument(pair: str) -> str:
        p = pair.upper().strip()
        if "_" not in p and len(p) == 6:
            p = p[:3] + "_" + p[3:]
        return p
