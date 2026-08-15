"""Exact-arithmetic tests for stop/TP fill accounting (P3).

Verifies that:
- a stop/TP exit books the fill-price return (clamped to the triggering
  bar's high/low range) instead of skipping the exit bar's move,
- stop levels are anchored at the previous close (the fill price),
- the trade log stamps the stop fill price and barrier reason.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.backtester.execution_patches import PatchConfig, run_execution_loop


def _make_loop_inputs(closes, pred, sl_pips=20.0, tp_pips=1000.0, entry_anchor_check=False,
                      highs=None, lows=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    rets = np.zeros(n)
    rets[1:] = np.log(closes[1:] / closes[:-1])
    pred = np.asarray(pred, dtype=float)

    df = pd.DataFrame({
        "close": closes,
        "high": np.asarray(highs, dtype=float) if highs is not None else closes,
        "low": np.asarray(lows, dtype=float) if lows is not None else closes,
        "spread": np.zeros(n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="30min"))

    cfg = PatchConfig(
        use_trail=True,
        stop_method="fixed_pips",
        stop_sl_pips=sl_pips,
        stop_tp_pips=tp_pips,
        trailing_method="none",
        move_to_be=False,
        tp1_z_base=1.0,
        trail_k_base=2.5,
        use_twap=False,
        sizing_method="fixed",
        bars_per_day=48,
    )

    result = run_execution_loop(
        df=df,
        pred=pred,
        rets=rets,
        bar_vol=np.full(n, 0.001),
        gap_from_prev_bool=np.zeros(n, dtype=bool),
        regime_code=np.zeros(n, dtype=int),
        cfg=cfg,
        trading_costs=False,
        slippage_factor=0.5,
    )
    return df, rets, result


class TestStopFillAccounting:
    def test_long_sl_books_fill_price_return(self):
        """Long SL hit: exit bar books log(sl/prev_close), not zero."""
        closes = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9985, 0.9990, 0.9990, 0.9990, 0.9990]
        highs  = [1.0001] * 10
        lows   = [0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9970, 0.9990, 0.9990, 0.9990, 0.9990]
        pred   = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

        df, rets, result = _make_loop_inputs(closes, pred, sl_pips=20.0, highs=highs, lows=lows)

        # Entry at bar 2 (fill = prev close = 1.0000); SL = 1.0000 - 0.0020 = 0.9980.
        # Bar 5 low = 0.9970 <= 0.9980 -> SL hit.
        expected = np.log(0.9980 / 1.0000)
        assert result.sl_hits == 1
        assert result.stop_fill_price[5] == pytest.approx(0.9980, abs=1e-9)
        assert result.strat[5] == pytest.approx(expected, abs=1e-12)

    def test_breakeven_sl_fills_at_entry_price(self):
        """BE stop moved to entry: exit books log(entry_price/prev_close)."""
        closes = [1.0000, 1.0000, 1.0000, 1.0030, 1.0005, 1.0005, 1.0005, 1.0005, 1.0005, 1.0005]
        highs  = [1.0001, 1.0001, 1.0001, 1.0040, 1.0032, 1.0005, 1.0005, 1.0005, 1.0005, 1.0005]
        lows   = [0.9999, 0.9999, 0.9999, 1.0010, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998]
        pred   = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

        cfg_df, rets, result = None, None, None

        # Build inputs with breakeven enabled.
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        rets = np.zeros(n)
        rets[1:] = np.log(closes[1:] / closes[:-1])
        pred = np.asarray(pred, dtype=float)

        df = pd.DataFrame({
            "close": closes,
            "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float),
            "spread": np.zeros(n),
        }, index=pd.date_range("2025-01-01", periods=n, freq="30min"))

        cfg = PatchConfig(
            use_trail=True,
            stop_method="fixed_pips",
            stop_sl_pips=20.0,
            stop_tp_pips=1000.0,
            stop_use_be=True,
            stop_be_trigger_pips=20.0,
            trailing_method="none",
            move_to_be=False,
            tp1_z_base=100.0,  # disable the TP1 scale-out for this scenario
            trail_k_base=2.5,
            use_twap=False,
            sizing_method="fixed",
            bars_per_day=48,
        )
        result = run_execution_loop(
            df=df, pred=pred, rets=rets,
            bar_vol=np.full(n, 0.001),
            gap_from_prev_bool=np.zeros(n, dtype=bool),
            regime_code=np.zeros(n, dtype=int),
            cfg=cfg, trading_costs=False, slippage_factor=0.5,
        )

        # Entry at bar 2 at 1.0000. Bar 3 close 1.0030 (+30 pips) triggers BE:
        # SL moved to 1.0000. Bar 4 low 0.9998 <= 1.0000 -> BE SL hit,
        # filled at the entry price 1.0000 (prev close = close[3] = 1.0030).
        expected = np.log(1.0000 / 1.0030)
        assert result.sl_hits == 1
        assert result.stop_fill_price[4] == pytest.approx(1.0000, abs=1e-9)
        assert result.strat[4] == pytest.approx(expected, abs=1e-12)

    def test_long_tp_books_fill_price_return(self):
        """Long TP hit: exit bar books log(tp/prev_close)."""
        closes = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0015, 1.0015, 1.0015, 1.0015, 1.0015]
        highs  = [0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 1.0030, 1.0015, 1.0015, 1.0015, 1.0015]
        lows   = [0.9999] * 10
        pred   = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

        df, rets, result = _make_loop_inputs(closes, pred, sl_pips=100.0, tp_pips=10.0,
                                             highs=highs, lows=lows)

        # TP = 1.0000 + 0.0010 = 1.0010; bar 5 high = 1.0030 -> TP hit.
        expected = np.log(1.0010 / 1.0000)
        assert result.tp_hits == 1
        assert result.strat[5] == pytest.approx(expected, abs=1e-12)

    def test_short_sl_books_fill_price_return(self):
        """Short SL hit above entry: exit bar books -log(sl/prev_close)."""
        closes = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0015, 1.0010, 1.0010, 1.0010, 1.0010]
        highs  = [1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0030, 1.0010, 1.0010, 1.0010, 1.0010]
        lows   = [0.9999] * 10
        pred   = [0, 0, -1, -1, -1, -1, 0, 0, 0, 0]

        df, rets, result = _make_loop_inputs(closes, pred, sl_pips=20.0, highs=highs, lows=lows)

        # Short entry at 1.0000; SL = 1.0000 + 0.0020 = 1.0020.
        # Bar 5 high = 1.0030 -> SL hit; booked at the stop price.
        expected = -np.log(1.0020 / 1.0000)
        assert result.sl_hits == 1
        assert result.strat[5] == pytest.approx(expected, abs=1e-12)

    def test_entry_anchored_at_prev_close(self):
        """SL levels anchor at the previous close (fill price), not bar close."""
        closes = [1.0000, 1.0000, 1.0020, 1.0010, 1.0010, 1.0010, 1.0010, 1.0010, 1.0010, 1.0010]
        highs  = [1.0001, 1.0001, 1.0025, 1.0030, 1.0010, 1.0010, 1.0010, 1.0010, 1.0010, 1.0010]
        lows   = [0.9999, 0.9999, 0.9999, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995]
        pred   = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

        df, rets, result = _make_loop_inputs(closes, pred, sl_pips=20.0, highs=highs, lows=lows)

        # Anchored at prev close (1.0000): SL = 0.9980 -> bar 3 low (0.9995) does NOT hit.
        # Anchored at bar close (1.0020): SL = 1.0000 -> bar 3 low (0.9995) WOULD hit.
        assert result.sl_hits == 0
        assert result.strat[3] == pytest.approx(rets[3], abs=1e-12)


class TestTradeLogStopStamping:
    def test_trade_log_uses_stop_fill_price(self):
        from pipeline.execution.execution_utils import build_trade_log_from_df

        idx = pd.date_range("2025-01-01", periods=4, freq="30min")
        df = pd.DataFrame({
            "position": [0.0, 1.0, 1.0, 0.0],
            "strategy": [0.0, 0.0, 0.0, np.log(0.9980 / 1.0000)],
            "close": [1.0000, 1.0000, 1.0000, 0.9985],
            "stop_fill_price": [0.0, 0.0, 0.0, 0.9980],
        }, index=idx)

        log = build_trade_log_from_df(df)
        assert len(log) == 1
        row = log.iloc[0]
        assert row["exit_price"] == pytest.approx(0.9980, abs=1e-9)
        assert row["barrier_hit"] == "stop"

    def test_trade_log_uses_close_without_stop_fill(self):
        from pipeline.execution.execution_utils import build_trade_log_from_df

        idx = pd.date_range("2025-01-01", periods=4, freq="30min")
        df = pd.DataFrame({
            "position": [0.0, 1.0, 1.0, 0.0],
            "strategy": [0.0, 0.0, 0.0, 0.0005],
            "close": [1.0000, 1.0000, 1.0000, 1.0005],
        }, index=idx)

        log = build_trade_log_from_df(df)
        assert len(log) == 1
        row = log.iloc[0]
        assert row["exit_price"] == pytest.approx(1.0005, abs=1e-9)
        assert row["barrier_hit"] == "signal"
