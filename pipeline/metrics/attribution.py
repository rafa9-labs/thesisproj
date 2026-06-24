"""
PnL Attribution — per-regime and per-model profit breakdown.

Answers "why did we make/lose money?" by tagging each trade
with its regime and attributing PnL to specific committee models.

Works with LiveJournalEntry from OandaTradingEngine, or with
approximate journal built from committee signal history.
"""

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RegimePnL:
    regime: str
    total_pnl: float = 0.0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    pnl_contribution_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "total_pnl": round(self.total_pnl, 4),
            "trade_count": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_pnl_per_trade": round(self.avg_pnl_per_trade, 4),
            "pnl_contribution_pct": round(self.pnl_contribution_pct, 2),
        }


@dataclass
class ModelContribution:
    model: str
    signals_produced: int = 0
    signals_correct: int = 0
    accuracy: float = 0.0
    contribution_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "signals_produced": self.signals_produced,
            "signals_correct": self.signals_correct,
            "accuracy": round(self.accuracy, 4),
            "contribution_score": round(self.contribution_score, 4),
        }


@dataclass
class SlippageReport:
    expected_pnl: float = 0.0
    actual_pnl: float = 0.0
    slippage_cost: float = 0.0
    slippage_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "expected_pnl": round(self.expected_pnl, 4),
            "actual_pnl": round(self.actual_pnl, 4),
            "slippage_cost": round(self.slippage_cost, 4),
            "slippage_pct": round(self.slippage_pct, 4),
        }


@dataclass
class AttributionReport:
    per_regime: Dict[str, RegimePnL] = field(default_factory=dict)
    per_model: Dict[str, ModelContribution] = field(default_factory=dict)
    slippage: SlippageReport = field(default_factory=SlippageReport)
    total_pnl: float = 0.0
    total_trades: int = 0
    status: str = "ok"

    def to_dict(self) -> dict:
        return {
            "total_pnl": round(self.total_pnl, 4),
            "total_trades": self.total_trades,
            "status": self.status,
            "per_regime": [v.to_dict() for v in self.per_regime.values()],
            "per_model": [v.to_dict() for v in self.per_model.values()],
            "slippage": self.slippage.to_dict(),
        }


class AttributionEngine:
    """Compute PnL attribution from trade journal and committee signal history.

    Parameters
    ----------
    journal : list[dict]
        Trade journal entries. Each entry must have: direction (-1/0/1),
        pnl (float), entry_time (float), is_win (bool).
    signal_history : list[dict]
        Committee signals with timestamps. Each signal must have:
        timestamp, signal, regime, active_models, model_weights.
    committee_config : dict
        Committee config with per-regime model assignments.
    """

    def __init__(
        self,
        journal: List[Dict[str, Any]],
        signal_history: List[Dict[str, Any]],
        committee_config: Optional[Dict[str, Any]] = None,
    ):
        self.journal = journal
        self.signal_history = signal_history
        self.committee_config = committee_config or {}

    def compute(self) -> AttributionReport:
        if not self.journal:
            return AttributionReport(status="insufficient_data")

        report = AttributionReport()
        report.total_trades = len(self.journal)

        # Tag each trade with its regime
        trade_regimes = self._tag_regimes()
        regime_pnl: Dict[str, RegimePnL] = {}

        for i, trade in enumerate(self.journal):
            pnl = float(trade.get("pnl", 0))
            is_win = trade.get("is_win", pnl > 0)
            regime = trade_regimes[i] if i < len(trade_regimes) else "unknown"

            report.total_pnl += pnl

            if regime not in regime_pnl:
                regime_pnl[regime] = RegimePnL(regime=regime)
            rp = regime_pnl[regime]
            rp.total_pnl += pnl
            rp.trade_count += 1
            if is_win:
                rp.wins += 1
            else:
                rp.losses += 1

        # Compute per-regime derived metrics
        for rp in regime_pnl.values():
            if rp.trade_count > 0:
                rp.win_rate = rp.wins / rp.trade_count
                rp.avg_pnl_per_trade = rp.total_pnl / rp.trade_count
            if abs(report.total_pnl) > 1e-8:
                rp.pnl_contribution_pct = (rp.total_pnl / report.total_pnl) * 100.0

        report.per_regime = regime_pnl

        # Per-model contribution
        if self.committee_config and self.signal_history:
            report.per_model = self._compute_model_contributions()

        # Slippage (from journal entries)
        report.slippage = self._compute_slippage()

        return report

    @staticmethod
    def _parse_ts(ts: Any) -> float:
        """Parse a timestamp to float epoch seconds. Returns -1 on failure."""
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(
                    ts.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, TypeError):
                return -1.0
        return -1.0

    def _tag_regimes(self) -> List[str]:
        """Match each trade to a regime from the nearest signal.

        Uses bisect for O(n log m) instead of brute-force O(n*m).
        """
        regimes: List[str] = []
        trade_times = [float(t.get("entry_time", 0)) for t in self.journal]

        if not self.signal_history:
            return ["unknown"] * len(self.journal)

        parsed = [
            (self._parse_ts(s.get("timestamp")), s.get("regime", "unknown"))
            for s in self.signal_history
        ]
        signal_times = [t for t, _ in parsed]
        signal_regimes = [r for _, r in parsed]

        valid = [(t, r) for t, r in zip(signal_times, signal_regimes) if t >= 0]
        if not valid:
            return ["unknown"] * len(self.journal)

        valid.sort(key=lambda x: x[0])
        sorted_times = [t for t, _ in valid]
        sorted_regimes = [r for _, r in valid]

        for trade_time in trade_times:
            idx = bisect.bisect_left(sorted_times, trade_time)
            best_idx = 0
            if idx == 0:
                best_idx = 0
            elif idx >= len(sorted_times):
                best_idx = len(sorted_times) - 1
            else:
                d_left = abs(trade_time - sorted_times[idx - 1])
                d_right = abs(sorted_times[idx] - trade_time)
                best_idx = idx if d_right <= d_left else idx - 1
            regimes.append(sorted_regimes[best_idx])

        return regimes

    def _build_signal_index(self):
        """Build sorted signal time index for binary search."""
        parsed = [
            (self._parse_ts(s.get("timestamp")), i)
            for i, s in enumerate(self.signal_history)
        ]
        valid = [(t, i) for t, i in parsed if t >= 0]
        valid.sort(key=lambda x: x[0])
        if not valid:
            return [], []
        times = [t for t, _ in valid]
        indices = [i for _, i in valid]
        return times, indices

    def _find_nearest_signal(
        self, trade_time: float, sorted_times: List[float],
    ) -> int:
        """Find index into self.signal_history of the nearest signal."""
        if not sorted_times:
            return -1
        idx = bisect.bisect_left(sorted_times, trade_time)
        if idx == 0:
            return 0
        if idx >= len(sorted_times):
            return len(sorted_times) - 1
        d_left = abs(trade_time - sorted_times[idx - 1])
        d_right = abs(sorted_times[idx] - trade_time)
        return idx if d_right <= d_left else idx - 1

    def _compute_model_contributions(self) -> Dict[str, ModelContribution]:
        models: Dict[str, ModelContribution] = {}

        for signal in self.signal_history:
            active = signal.get("active_models", [])
            weights = signal.get("model_weights", [])
            sig_val = int(signal.get("signal", 0))

            for i, model in enumerate(active):
                if model not in models:
                    models[model] = ModelContribution(model=model)
                mc = models[model]
                mc.signals_produced += 1
                weight = float(weights[i]) if i < len(weights) else 1.0
                mc.contribution_score += weight * (1.0 if sig_val != 0 else 0.0)

        if not models:
            return models

        sorted_times, sorted_indices = self._build_signal_index()

        for trade in self.journal:
            trade_time = float(trade.get("entry_time", 0))
            is_win = trade.get("is_win", trade.get("pnl", 0) > 0)
            sig_idx = self._find_nearest_signal(trade_time, sorted_times)
            if sig_idx < 0:
                continue
            signal = self.signal_history[sorted_indices[sig_idx]]
            sig_val = int(signal.get("signal", 0))
            direction = trade.get("direction", 0)
            if sig_val == 0 or direction == 0:
                continue
            signal_correct = (sig_val == direction)
            for model in signal.get("active_models", []):
                if model in models:
                    if signal_correct and is_win:
                        models[model].signals_correct += 1

        for mc in models.values():
            if mc.signals_produced > 0:
                mc.accuracy = mc.signals_correct / mc.signals_produced
                mc.contribution_score /= mc.signals_produced

        return models

    def _compute_slippage(self) -> SlippageReport:
        """Compute slippage from signal→execution price difference.

        Expected PnL = what the trade would have returned if filled at
        the signal price.  Actual PnL = what it returned at the real
        entry price.  Slippage = expected − actual (always ≥ 0 for
        winning trades, ≤ 0 for losing trades).
        """
        if not self.journal:
            return SlippageReport()

        sorted_times, sorted_indices = self._build_signal_index()

        actual_total = 0.0
        expected_total = 0.0

        for trade in self.journal:
            actual_pnl = float(trade.get("pnl", 0))
            direction = trade.get("direction", 0)
            size = float(trade.get("size", 1))
            entry_price = float(trade.get("entry_price", 0))
            trade_time = float(trade.get("entry_time", 0))

            actual_total += actual_pnl

            if entry_price <= 0 or direction == 0 or size <= 0:
                expected_total += actual_pnl
                continue

            sig_idx = self._find_nearest_signal(trade_time, sorted_times)
            if sig_idx < 0:
                expected_total += actual_pnl
                continue

            signal = self.signal_history[sorted_indices[sig_idx]]
            sig_price = float(signal.get("price", 0))
            if sig_price <= 0:
                expected_total += actual_pnl
                continue

            exit_price = entry_price + actual_pnl / (direction * size)
            expected_pnl = direction * (exit_price - sig_price) * size
            expected_total += expected_pnl

        slip_cost = expected_total - actual_total
        slip_pct = (
            (slip_cost / abs(actual_total) * 100.0)
            if abs(actual_total) > 1e-8 else 0.0
        )

        return SlippageReport(
            expected_pnl=expected_total,
            actual_pnl=actual_total,
            slippage_cost=slip_cost,
            slippage_pct=slip_pct,
        )


def compute_attribution_from_session(session: dict) -> AttributionReport:
    """Build attribution from a live trading session.

    Works with both OandaTradingEngine sessions and committee-only sessions.
    """
    journal: List[Dict[str, Any]] = []
    signal_history: List[Dict[str, Any]] = []
    committee_config = None

    # Try OandaTradingEngine journal first
    engine = session.get("engine")
    if engine and hasattr(engine, "get_journal"):
        journal = engine.get_journal(offset=0, limit=10_000)

    # Committee runner signals
    runner = session.get("runner")
    if runner and hasattr(runner, "get_recent_signals"):
        signal_history = runner.get_recent_signals(10_000)

    # Committee config
    cfg = session.get("committee_config")
    if cfg and hasattr(cfg, "to_dict"):
        committee_config = cfg.to_dict()

    engine = AttributionEngine(journal, signal_history, committee_config)
    return engine.compute()
