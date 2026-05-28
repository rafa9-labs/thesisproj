"""Tests for trading module — OANDA client, paper engine, live engine."""
from __future__ import annotations

import time

import pytest


class TestOandaClient:
    """Unit tests for trading.oanda_client.OandaClient."""

    def test_instantiation_defaults(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001")
        assert client.environment == "practice"
        assert client.access_token == "token"
        assert client.account_id == "101-001-000000-001"
        assert client.max_requests_per_sec == 20.0

    def test_live_environment(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001", environment="live")
        assert client.environment == "live"

    def test_rate_limit_custom(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001",
                             max_requests_per_sec=5.0)
        assert client.max_requests_per_sec == 5.0

    def test_is_connected_false_with_bogus_creds(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        assert client.is_connected() is False

    def test_has_open_positions_false_with_bogus_creds(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        assert client.has_open_positions() is False

    def test_throttle_respects_rate_limit(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001",
                             max_requests_per_sec=100.0)
        t0 = time.monotonic()
        for _ in range(50):
            client._throttle()
        elapsed = time.monotonic() - t0
        expected_min = (50 - 1) / 100.0
        assert elapsed >= expected_min * 0.9

    @pytest.mark.parametrize("rate", [0.5, 1.0, 10.0, 50.0])
    def test_throttle_various_rates(self, rate):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001",
                             max_requests_per_sec=rate)
        t0 = time.monotonic()
        for _ in range(10):
            client._throttle()
        elapsed = time.monotonic() - t0
        expected = (10 - 1) / rate
        assert elapsed >= expected * 0.9


class TestOrderBody:
    """Tests for OandaClient._order_body() — no API calls."""

    def test_market_order_basic(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001")
        body = client._order_body("MARKET", "EUR_USD", 1000)
        assert body == {"order": {"type": "MARKET", "instrument": "EUR_USD", "units": "1000"}}

    def test_market_order_with_sl_tp(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001")
        body = client._order_body("MARKET", "EUR_USD", 1000,
                                  stop_loss=1.0850, take_profit=1.0950)
        order = body["order"]
        assert order["type"] == "MARKET"
        assert order["stopLossOnFill"] == {"price": "1.085"}
        assert order["takeProfitOnFill"] == {"price": "1.095"}

    def test_limit_order_with_price(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001")
        body = client._order_body("LIMIT", "EUR_USD", -500, price=1.09123)
        assert body["order"]["type"] == "LIMIT"
        assert body["order"]["price"] == "1.09123"
        assert body["order"]["units"] == "-500"

    def test_stop_order_with_sl(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001")
        body = client._order_body("STOP", "GBP_USD", 2000,
                                  price=1.30, stop_loss=1.2950)
        assert body["order"]["type"] == "STOP"
        assert body["order"]["price"] == "1.3"
        assert body["order"]["stopLossOnFill"] == {"price": "1.295"}
        assert "takeProfitOnFill" not in body["order"]

    def test_no_stop_loss_omits_key(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001")
        body = client._order_body("MARKET", "EUR_USD", 1000)
        assert "stopLossOnFill" not in body["order"]
        assert "takeProfitOnFill" not in body["order"]

    def test_negative_units(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("token", "101-001-000000-001")
        body = client._order_body("MARKET", "EUR_USD", -3000)
        assert body["order"]["units"] == "-3000"


class TestCloseAll:
    """Tests for OandaClient.close_all() under simulated failure."""

    def test_close_all_returns_list(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        results = client.close_all()
        assert isinstance(results, list)


class TestAPIErrors:
    """Tests for API error handling."""

    def test_get_position_bogus_returns_none(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        pos = client.get_position("EUR_USD")
        assert pos is None

    def test_get_positions_bogus_returns_empty(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        positions = client.get_positions()
        assert positions == []

    def test_place_market_order_bogus_raises(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        with pytest.raises(Exception):
            client.place_market_order("EUR_USD", 1000)

    def test_close_position_bogus_raises(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        with pytest.raises(Exception):
            client.close_position("EUR_USD")

    def test_cancel_order_bogus_raises(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        with pytest.raises(Exception):
            client.cancel_order("999")

    def test_get_transactions_bogus_raises(self):
        from trading.oanda_client import OandaClient
        client = OandaClient("bogus", "bogus")
        with pytest.raises(Exception):
            client.get_transactions(count=5)
