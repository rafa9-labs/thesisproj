"""Alignment tests between live engines and the backtest execution policy (P5).

Covers:
- ATR / VOL_TARGET sizing with real volatility inputs (no silent fallbacks),
- OANDA unit scaling (lots x contract_size),
- paper-engine SL/TP fills at the stop level,
- LiveSignal bar_vol/atr from the runner bar buffer,
- opt-in conviction sizing in the backtest execution loop.
"""
from collections import deque

import numpy as np
import pytest

from pipeline.execution.position_sizing import (
    SizingConfig,
    SizingMethod,
    SizingState,
    compute_size,
)


class TestSizingWithRealVolatility:
    def test_atr_sizing_uses_atr_input(self):
        cfg = SizingConfig(method=SizingMethod.ATR, initial_equity=10_000.0,
                           atr_risk_pct=0.02, atr_sl_mult=2.0,
                           contract_size=100_000.0)
        state = SizingState(equity=10_000.0)
        atr = 0.0030
        size = compute_size(state, 0.0, atr, cfg)
        # dollar_risk = 0.02 * 10000 = 200; lots = 200 / (2*0.003*100000) = 0.3333
        expected = 200.0 / (2.0 * atr * 100_000.0)
        assert size == pytest.approx(expected, abs=1e-9)

    def test_atr_sizing_zero_atr_falls_back(self):
        cfg = SizingConfig(method=SizingMethod.ATR, initial_equity=10_000.0,
                           risk_fraction=0.02, atr_risk_pct=0.02,
                           atr_sl_mult=2.0, contract_size=100_000.0)
        state = SizingState(equity=10_000.0)
        size = compute_size(state, 0.0, 0.0, cfg)
        assert size == pytest.approx(0.02, abs=1e-9)  # fixed-fractional fallback

    def test_vol_target_uses_bar_vol(self):
        cfg = SizingConfig(method=SizingMethod.VOL_TARGET, target_bar=0.0001,
                           max_lev=1.5, vol_floor=1e-6)
        state = SizingState(equity=10_000.0)
        size = compute_size(state, 0.001, 0.0, cfg)
        assert size == pytest.approx(0.1, abs=1e-9)  # NOT max_lev


class TestOandaUnitScaling:
    def test_live_engine_scales_lots_to_units(self):
        from trading.live_engine import OandaTradingEngine

        class FakeOanda:
            def __init__(self):
                self.orders = []
                self.positions = []

            def place_market_order(self, instrument, units, *, stop_loss=None, take_profit=None):
                self.orders.append({
                    "instrument": instrument, "units": units,
                    "stop_loss": stop_loss, "take_profit": take_profit,
                })
                return {"orderFillTransaction": {"price": 1.1000, "orderID": "1", "id": "1"}}

            def get_account_summary(self):
                return {"marginUsed": 0, "balance": 10000, "NAV": 10000}

            def get_open_positions(self):
                return {"positions": []}

            def close_position(self, instrument):
                return {}

            def get_position_pnl(self, instrument):
                return 0.0

        client = FakeOanda()
        engine = OandaTradingEngine()
        engine.start(
            {
                "pair": "EURUSD", "mode": "demo",
                "position_sizing": "fixed_fractional",
                "sizing_config": {"risk_fraction": 0.02},
                "risk_config": {"max_position_pct": 1.0, "restrict_weekend": False},
                "stop_config": {"method": "fixed_pips", "sl_pips": 20.0, "tp_pips": 40.0},
            },
            client,
        )
        engine.process_signal({"direction": "LONG", "confidence": 80.0},
                              bid=1.0999, ask=1.1001, mid=1.1000,
                              oanda_client=client)
        assert client.orders, "order should have been placed"
        order = client.orders[0]
        # size = 0.02 lots -> 2000 OANDA units (contract_size=100_000)
        assert order["units"] == 2000
        # SL/TP levels anchored at the ask (1.1001)
        assert order["stop_loss"] == pytest.approx(1.1001 - 0.0020, abs=1e-6)
        assert order["take_profit"] == pytest.approx(1.1001 + 0.0040, abs=1e-6)


class TestPaperEngineStops:
    def test_long_sl_closes_at_stop_level(self):
        from trading.paper_engine import PaperEngine

        engine = PaperEngine()
        engine.start({
            "initial_equity": 10_000.0,
            "position_sizing": "fixed",
            "stop_config": {"method": "fixed_pips", "sl_pips": 20.0, "tp_pips": 60.0},
        })
        engine.process_signal({"direction": "LONG", "confidence": 80.0},
                              bid=1.0999, ask=1.1001, mid=1.1000)

        # Bid crosses below SL (entry ask 1.1001 - 0.0020 = 1.0981).
        # FLAT signal so the engine does not re-enter after the stop-out.
        result = engine.process_signal({"direction": "FLAT", "confidence": 50.0},
                                       bid=1.0979, ask=1.0981, mid=1.0980)
        assert any(e.get("event") == "trade_closed" and e.get("exit_reason") == "stop_loss"
                   for e in result.get("sub_events", []))
        p = engine.portfolio
        assert p.position == 0
        assert p.closed_trades[-1].exit_price == pytest.approx(1.0981, abs=1e-6)

    def test_long_tp_closes_at_tp_level(self):
        from trading.paper_engine import PaperEngine

        engine = PaperEngine()
        engine.start({
            "initial_equity": 10_000.0,
            "position_sizing": "fixed",
            "stop_config": {"method": "fixed_pips", "sl_pips": 100.0, "tp_pips": 20.0},
        })
        engine.process_signal({"direction": "LONG", "confidence": 80.0},
                              bid=1.0999, ask=1.1001, mid=1.1000)
        # Bid crosses above TP (1.1001 + 0.0020 = 1.1021).
        # FLAT signal so the engine does not re-enter after the TP close.
        result = engine.process_signal({"direction": "FLAT", "confidence": 50.0},
                                       bid=1.1022, ask=1.1024, mid=1.1023)
        assert any(e.get("event") == "trade_closed" and e.get("exit_reason") == "take_profit"
                   for e in result.get("sub_events", []))
        p = engine.portfolio
        assert p.position == 0
        assert p.closed_trades[-1].exit_price == pytest.approx(1.1021, abs=1e-6)


class TestRunnerVolatility:
    def test_compute_volatility_from_bar_buffer(self):
        from trading.live_committee_runner import LiveCommitteeRunner

        runner = object.__new__(LiveCommitteeRunner)
        closes = [1.1000 + 0.0005 * i + 0.0002 * np.sin(i) for i in range(60)]
        runner._bar_buffer = deque(
            [{"mid_c": c} for c in closes], maxlen=100,
        )
        bar_vol, atr = runner._compute_volatility()
        assert bar_vol > 0.0
        assert atr > 0.0
        assert np.isfinite(bar_vol) and np.isfinite(atr)

    def test_compute_volatility_short_buffer(self):
        from trading.live_committee_runner import LiveCommitteeRunner

        runner = object.__new__(LiveCommitteeRunner)
        runner._bar_buffer = deque([{"mid_c": 1.1}, {"mid_c": 1.1}], maxlen=100)
        bar_vol, atr = runner._compute_volatility()
        assert bar_vol == 0.0 and atr == 0.0


class TestBacktestConvictionSizing:
    def _run_loop(self, use_conviction, confidence):
        from pipeline.backtester.execution_patches import PatchConfig, run_execution_loop

        n = 8
        closes = np.full(n, 1.0000)
        df_idx = __import__("pandas").date_range("2025-01-01", periods=n, freq="30min")
        import pandas as pd
        df = pd.DataFrame({
            "close": closes, "high": closes, "low": closes, "spread": np.zeros(n),
        }, index=df_idx)
        rets = np.zeros(n)
        pred = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=float)
        cfg = PatchConfig(
            use_trail=False, use_twap=False, sizing_method="fixed",
            use_conviction_sizing=use_conviction,
            bars_per_day=48,
        )
        result = run_execution_loop(
            df=df, pred=pred, rets=rets,
            bar_vol=np.full(n, 0.001),
            gap_from_prev_bool=np.zeros(n, dtype=bool),
            regime_code=np.zeros(n, dtype=int),
            cfg=cfg, trading_costs=False, slippage_factor=0.5,
            confidence=np.asarray(confidence, dtype=float),
        )
        return result.pos_actual

    def test_conviction_off_uses_unit_size(self):
        pos = self._run_loop(False, [0.9] * 8)
        assert np.allclose(pos[2:], 1.0)

    def test_high_confidence_scales_up(self):
        pos = self._run_loop(True, [0.9] * 8)
        assert np.allclose(pos[2:], 1.5)

    def test_low_confidence_scales_down(self):
        pos = self._run_loop(True, [0.6] * 8)
        assert np.allclose(pos[2:], 0.5)
