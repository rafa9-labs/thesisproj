"""Tests for trading/paper_engine.py and api/routers/trading.py."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def paper_engine_config():
    return {"initial_equity": 20000, "position_sizing": "fixed"}


@pytest.fixture
def long_signal():
    return {"direction": "LONG", "confidence": 85.0}


@pytest.fixture
def short_signal():
    return {"direction": "SHORT", "confidence": 78.0}


@pytest.fixture
def flat_signal():
    return {"direction": "FLAT", "confidence": 50.0}


class TestPaperEngine:
    def test_portfolio_initial_state(self, paper_engine_config):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        p = engine.start(paper_engine_config)

        assert p.initial_equity == 20000
        assert p.equity == 20000
        assert p.position == 0
        assert p.signal_count == 0
        assert len(p.closed_trades) == 0
        assert p.open_trade is None
        assert len(p.equity_curve) == 1
        assert p.equity_curve[0]["equity"] == 20000

    def test_process_signal_enter_long(self, paper_engine_config, long_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        result = engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)

        assert result["event"] == "signal"
        assert result["position"] == "LONG"
        assert "sub_events" in result
        sub = result["sub_events"][0]
        assert sub["event"] == "trade_opened"
        assert sub["direction"] == "LONG"
        assert sub["entry_price"] == 1.08502

        pstate = engine.get_portfolio_state()
        assert pstate["position"] == "LONG"

    def test_process_signal_exit_long(self, paper_engine_config, long_signal, flat_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        result = engine.process_signal(flat_signal, bid=1.08600, ask=1.08602, mid=1.08601)

        assert result["position"] == "FLAT"
        assert "sub_events" in result
        closed = result["sub_events"][0]
        assert closed["event"] == "trade_closed"
        assert closed["exit_reason"] == "signal_flat"

        pstate = engine.get_portfolio_state()
        assert pstate["position"] == "FLAT"
        assert pstate["total_trades_closed"] == 1

    def test_process_signal_reversal(self, paper_engine_config, long_signal, short_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        result = engine.process_signal(short_signal, bid=1.08600, ask=1.08602, mid=1.08601)

        sub_events = result.get("sub_events", [])
        assert len(sub_events) == 2

        close_ev = sub_events[0]
        assert close_ev["event"] == "trade_closed"
        assert close_ev["exit_reason"] == "signal_reversal"

        open_ev = sub_events[1]
        assert open_ev["event"] == "trade_opened"
        assert open_ev["direction"] == "SHORT"
        assert open_ev["entry_price"] == 1.08600

        assert result["position"] == "SHORT"

    def test_equity_curve_tracking(self, paper_engine_config, long_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        engine.process_signal(long_signal, bid=1.08600, ask=1.08602, mid=1.08601)

        curve = engine.portfolio.equity_curve
        assert len(curve) >= 3

    def test_stop_closes_all_positions(self, paper_engine_config, long_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        summary = engine.stop(bid=1.08600, ask=1.08602)

        assert summary.get("stopped") is True
        pstate = engine.get_portfolio_state()
        assert pstate["position"] == "FLAT"

    def test_summary_metrics_empty(self, paper_engine_config):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)
        summary = engine.get_summary()

        assert summary["total_trades"] == 0
        # Honest reporting: no trades => Sharpe is None (insufficient data),
        # not a fabricated 0.0.
        assert summary["sharpe"] is None
        assert summary["sortino"] is None
        assert summary["total_return_pct"] == 0.0

    def test_summary_metrics_with_trades(self, paper_engine_config, long_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        engine.stop(bid=1.08600, ask=1.08602)

        summary = engine.get_summary()
        assert summary["total_trades"] == 1
        assert "sharpe" in summary
        assert "total_return_pct" in summary
        assert "max_drawdown_pct" in summary
        assert "win_rate" in summary
        assert "profit_factor" in summary
        assert "final_equity" in summary

    def test_compare_to_backtest(self, paper_engine_config, long_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        engine.stop(bid=1.08600, ask=1.08602)

        comparison = engine.compare_to_backtest({
            "metrics": {"sharpe": 1.5, "total_return_pct": 12.0, "max_drawdown_pct": 5.0, "win_rate": 0.6},
        })

        assert "sharpe" in comparison
        assert "total_return_pct" in comparison
        # Honest reporting: a 1-trade session has insufficient data -> paper
        # Sharpe is None rather than a fabricated number.
        assert comparison["sharpe"]["paper"] is None or isinstance(comparison["sharpe"]["paper"], (int, float))
        assert comparison["sharpe"]["backtest"] == 1.5

    def test_get_trades_pagination(self, paper_engine_config, long_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        engine.stop(bid=1.08600, ask=1.08602)

        trades = engine.get_trades(offset=0, limit=10)
        assert len(trades) >= 1
        assert "trade_id" in trades[0]
        assert "pnl" in trades[0]

        trades_page2 = engine.get_trades(offset=10, limit=10)
        assert trades_page2 == []

    def test_already_stopped_guard(self, paper_engine_config, long_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)
        engine.stop()

        result = engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        assert result["event"] == "already_stopped"

    def test_buy_at_ask_sell_at_bid(self, paper_engine_config, long_signal, short_signal):
        from trading.paper_engine import PaperEngine
        engine = PaperEngine()
        engine.start(paper_engine_config)

        r1 = engine.process_signal(long_signal, bid=1.08500, ask=1.08502, mid=1.08501)
        assert r1["sub_events"][0]["entry_price"] == 1.08502

        r2 = engine.process_signal(short_signal, bid=1.08600, ask=1.08602, mid=1.08601)
        close_ev = r2["sub_events"][0]
        assert close_ev["exit_price"] == 1.08600  # long closes at bid

        open_ev = r2["sub_events"][1]
        assert open_ev["entry_price"] == 1.08600  # short enters at bid


class TestTradingAPISchemas:
    def test_paper_session_info_fields(self):
        from api.routers.trading import PaperSessionInfo
        info = PaperSessionInfo(
            session_id="abc123",
            pair="EURUSD",
            model_type="logistic",
            timeframe="M30",
            status="running",
            equity=10000,
            position="FLAT",
            unrealized_pnl=0,
            total_trades=0,
            signal_count=0,
            created_at="2026-05-28T00:00:00Z",
        )
        assert info.session_id == "abc123"
        assert info.equity == 10000
        assert info.position == "FLAT"

    def test_deploy_paper_request_defaults(self):
        from api.routers.trading import DeployPaperRequest
        req = DeployPaperRequest(pair="EURUSD")
        assert req.pair == "EURUSD"
        assert req.model_type == "logistic"
        assert req.timeframe == "M30"
        assert req.initial_equity == 10000
        assert req.position_sizing == "fixed"
        assert req.sizing_config == {}
