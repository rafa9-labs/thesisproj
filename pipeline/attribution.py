"""
PnL Attribution — per-regime and per-model profit breakdown.

Answers "why did we make/lose money?" by tagging each trade
with its regime and attributing PnL to specific committee models.

Works with LiveJournalEntry from OandaTradingEngine, or with
approximate journal built from committee signal history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

    def _tag_regimes(self) -> List[str]:
        """Match each trade to a regime from the nearest signal."""
        regimes: List[str] = []
        trade_times = [float(t.get("entry_time", 0)) for t in self.journal]

        if not self.signal_history:
            return ["unknown"] * len(self.journal)

        signal_times = []
        signal_regimes = []
        for s in self.signal_history:
            ts = s.get("timestamp")
            if isinstance(ts, str):
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = 0
            signal_times.append(float(ts))
            signal_regimes.append(s.get("regime", "unknown"))

        for trade_time in trade_times:
            best_idx = -1
            best_distance = float("inf")
            for i, st in enumerate(signal_times):
                dist = abs(trade_time - st)
                if dist < best_distance:
                    best_distance = dist
                    best_idx = i
            regimes.append(signal_regimes[best_idx] if best_idx >= 0 else "unknown")

        return regimes

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

        # Estimate signal correctness from win rate correlation
        total_wins = sum(1 for t in self.journal if t.get("is_win", False))
        total_entries = len(self.journal)
        base_accuracy = total_wins / max(1, total_entries)

        for mc in models.values():
            if mc.signals_produced > 0:
                mc.signals_correct = int(mc.signals_produced * base_accuracy)
                mc.accuracy = base_accuracy
                mc.contribution_score /= max(1, mc.signals_produced)

        return models

    def _compute_slippage(self) -> SlippageReport:
        actual = sum(float(t.get("pnl", 0)) for t in self.journal)
        expected = 0.0
        for t in self.journal:
            if t.get("is_win", False):
                expected += abs(float(t.get("pnl", 0)))
            else:
                expected -= abs(float(t.get("pnl", 0)))
        # Expected PnL from signal direction alone (simplified)
        # Actual slippage = entry/exit spread cost not captured in signal PnL
        slip_cost = actual - expected if abs(expected) > 1e-8 else 0.0
        slip_pct = (slip_cost / abs(actual) * 100.0) if abs(actual) > 1e-8 else 0.0

        return SlippageReport(
            expected_pnl=expected,
            actual_pnl=actual,
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
