"""Tests for trading/lean_bridge.py — LEAN Bridge HTTP client."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from trading.lean_bridge import (
    LeanBridge,
    signal_to_insight,
    PortfolioState,
    PortfolioPosition,
    OrderResult,
    OrderStatus,
    LEAN_DEFAULT_URL,
)


def _run(coro):
    return asyncio.run(coro)


# ============================================================
# signal_to_insight
# ============================================================

class TestSignalToInsight:
    def test_long_signal(self):
        insight = signal_to_insight(1, 0.85, "EURUSD", tag="test_123")
        assert insight["Symbol"] == "EURUSD"
        assert insight["Direction"] == "Up"
        assert insight["Confidence"] == 0.85
        assert insight["SecurityType"] == "Forex"
        assert "Period" in insight
        assert insight["Tag"] == "test_123"

    def test_short_signal(self):
        insight = signal_to_insight(-1, 0.72, "GBPUSD")
        assert insight["Symbol"] == "GBPUSD"
        assert insight["Direction"] == "Down"

    def test_flat_signal_returns_empty(self):
        insight = signal_to_insight(0, 0.5)
        assert insight == {}

    def test_no_tag_generates_default(self):
        insight = signal_to_insight(1, 0.5)
        assert insight["Tag"].startswith("kodaquant_")


# ============================================================
# PortfolioState
# ============================================================

class TestPortfolioState:
    def test_from_lean_response_parses_correctly(self):
        data = {
            "Cash": {"Amount": 5000.0},
            "TotalPortfolioValue": 10250.0,
            "TotalMarginUsed": 1000.0,
            "BuyingPower": 9250.0,
            "Positions": {
                "Values": [
                    {
                        "Symbol": {"Value": "EURUSD"},
                        "Quantity": 10000.0,
                        "AveragePrice": 1.0850,
                        "MarketPrice": 1.0860,
                        "MarketValue": 500.0,
                        "UnrealizedProfit": 10.0,
                    }
                ]
            },
        }
        pf = PortfolioState.from_lean_response(data)
        assert pf.cash == 5000.0
        assert pf.equity == 10250.0
        assert pf.margin_used == 1000.0
        assert len(pf.positions) == 1
        assert pf.positions[0].symbol == "EURUSD"
        assert pf.positions[0].quantity == 10000.0

    def test_from_lean_response_empty_positions(self):
        data = {
            "Cash": {"Amount": 10000.0},
            "TotalPortfolioValue": 10000.0,
            "TotalMarginUsed": 0.0,
            "BuyingPower": 10000.0,
            "Positions": {"Values": []},
        }
        pf = PortfolioState.from_lean_response(data)
        assert pf.cash == 10000.0
        assert pf.positions == []


# ============================================================
# LeanBridge tests (sync with asyncio.run)
# ============================================================

def _make_resp(status=200, json_data=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = json_data or {}
    return resp


class TestLeanBridgeSubmitOrder:
    def test_submit_order_sends_correct_command(self):
        async def _test():
            bridge = LeanBridge(base_url="http://localhost:8888", timeout=5.0)
            mock_resp = _make_resp(200, {"CommandId": 42})
            bridge._request_with_retry = AsyncMock(return_value=mock_resp)

            result = await bridge.submit_order("EURUSD", 1, 10000, tag="test_tag")
            call_args = bridge._request_with_retry.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/live/commands"
            body = call_args[1]["json_body"]
            assert body["$type"] == "SubmitOrderCommand"
            assert body["ticker"] == "EURUSD"
            assert body["quantity"] == 10000
            assert body["orderType"] == "Market"
            assert body["tag"] == "test_tag"
            assert result.success
            assert result.order_id == 42

        _run(_test())

    def test_submit_order_flat_direction_noop(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            result = await bridge.submit_order("EURUSD", 0, 10000)
            assert result.success
            assert result.order_id is None

        _run(_test())

    def test_submit_order_limit_stop_prices(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            mock_resp = _make_resp(200, {"CommandId": 1})
            bridge._request_with_retry = AsyncMock(return_value=mock_resp)
            result = await bridge.submit_order(
                "EURUSD", 1, 5000, order_type="Limit",
                limit_price=1.0850, stop_price=1.0820,
            )
            body = bridge._request_with_retry.call_args[1]["json_body"]
            assert body["limitPrice"] == 1.0850
            assert body["stopPrice"] == 1.0820
            assert result.success

        _run(_test())

    def test_submit_order_lean_unreachable(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            bridge._request_with_retry = AsyncMock(return_value=None)
            result = await bridge.submit_order("EURUSD", 1, 10000)
            assert not result.success
            assert "unreachable" in result.error.lower()

        _run(_test())

    def test_submit_order_alert_on_unreachable(self):
        async def _test():
            alert_calls = []
            bridge = LeanBridge(
                timeout=5.0,
                alert_callback=AsyncMock(
                    side_effect=lambda t, m: alert_calls.append((t, m))
                ),
            )

            async def always_fails(*args, **kwargs):
                raise httpx.ConnectError("connection refused")

            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=always_fails)
            mock_client.request = AsyncMock(side_effect=always_fails)
            bridge._get_client = AsyncMock(return_value=mock_client)

            result = await bridge.submit_order("EURUSD", 1, 10000)
            assert not result.success
            assert len(alert_calls) >= 1
            assert alert_calls[0][0] == "lean_bridge_failure"

        _run(_test())


class TestLeanBridgeRetry:
    def test_retry_succeeds_on_second_attempt(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            call_count = [0]

            async def flaky_request(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise httpx.ConnectError("connection refused")
                resp = MagicMock(spec=httpx.Response)
                resp.status_code = 200
                resp.json.return_value = {"CommandId": 100}
                return resp

            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=flaky_request)
            bridge._get_client = AsyncMock(return_value=mock_client)

            resp = await bridge._request_with_retry("POST", "/live/commands", json_body={})
            assert resp is not None
            assert resp.status_code == 200
            assert call_count[0] == 3

        _run(_test())

    def test_retry_exhaustion_returns_none_and_alerts(self):
        async def _test():
            alert_calls = []
            bridge = LeanBridge(
                timeout=5.0,
                alert_callback=AsyncMock(
                    side_effect=lambda t, m: alert_calls.append((t, m))
                ),
            )

            async def always_fails(*args, **kwargs):
                raise httpx.ConnectError("connection refused")

            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=always_fails)
            mock_client.get = AsyncMock(side_effect=always_fails)
            mock_client.request = AsyncMock(side_effect=always_fails)
            bridge._get_client = AsyncMock(return_value=mock_client)

            resp = await bridge._request_with_retry("GET", "/live/portfolio")
            assert resp is None
            assert len(alert_calls) == 1
            assert alert_calls[0][0] == "lean_bridge_failure"

        _run(_test())


class TestLeanBridgeCancel:
    def test_cancel_order_sends_correct_command(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            mock_resp = _make_resp(200, {})
            bridge._request_with_retry = AsyncMock(return_value=mock_resp)
            result = await bridge.cancel_order(99)
            body = bridge._request_with_retry.call_args[1]["json_body"]
            assert body["$type"] == "CancelOrderCommand"
            assert body["orderId"] == 99
            assert result.success

        _run(_test())


class TestLeanBridgeLiquidate:
    def test_liquidate_all_sends_correct_command(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            mock_resp = _make_resp(200, {})
            bridge._request_with_retry = AsyncMock(return_value=mock_resp)
            result = await bridge.liquidate_all()
            body = bridge._request_with_retry.call_args[1]["json_body"]
            assert body["$type"] == "LiquidateCommand"
            assert body["liquidateAll"]
            assert result.success

        _run(_test())


class TestLeanBridgePortfolio:
    def test_get_portfolio_parses_response(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            data = {
                "Cash": {"Amount": 9500.0},
                "TotalPortfolioValue": 10500.0,
                "TotalMarginUsed": 0.0,
                "BuyingPower": 10500.0,
                "Positions": {"Values": []},
            }
            mock_resp = _make_resp(200, data)
            bridge._request_with_retry = AsyncMock(return_value=mock_resp)
            pf = await bridge.get_portfolio()
            assert pf is not None
            assert pf.cash == 9500.0
            assert pf.equity == 10500.0

        _run(_test())

    def test_get_portfolio_unreachable_returns_none(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            bridge._request_with_retry = AsyncMock(return_value=None)
            pf = await bridge.get_portfolio()
            assert pf is None

        _run(_test())


class TestLeanBridgeOrders:
    def test_get_orders_parses_response(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            data = {
                "Orders": [
                    {
                        "Id": 1,
                        "Symbol": {"Value": "EURUSD"},
                        "Direction": "Buy",
                        "Quantity": 10000.0,
                        "RemainingQuantity": 0.0,
                        "Status": "Filled",
                        "Price": 1.0850,
                    }
                ]
            }
            mock_resp = _make_resp(200, data)
            bridge._request_with_retry = AsyncMock(return_value=mock_resp)
            orders = await bridge.get_orders()
            assert len(orders) == 1
            assert orders[0].order_id == 1
            assert orders[0].symbol == "EURUSD"
            assert orders[0].status == "Filled"

        _run(_test())

    def test_get_orders_unreachable_returns_empty(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            bridge._request_with_retry = AsyncMock(return_value=None)
            orders = await bridge.get_orders()
            assert orders == []

        _run(_test())


class TestLeanBridgeHealthCheck:
    def test_health_check_ok(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            mock_resp = _make_resp(200)
            bridge._request_with_retry = AsyncMock(return_value=mock_resp)
            assert await bridge.health_check() is True

        _run(_test())

    def test_health_check_fail(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            bridge._request_with_retry = AsyncMock(return_value=None)
            assert await bridge.health_check() is False

        _run(_test())


class TestLeanBridgeClose:
    def test_close_cleans_up_client(self):
        async def _test():
            bridge = LeanBridge(timeout=5.0)
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()
            bridge._client = mock_client
            await bridge.close()
            mock_client.aclose.assert_awaited_once()
            assert bridge._client is None

        _run(_test())


# ============================================================
# Default URL / No CLI fallback
# ============================================================

class TestDefaultUrl:
    def test_default_url(self):
        assert LEAN_DEFAULT_URL == "http://localhost:8888"

    def test_no_cli_fallback_exists(self):
        source = open(__import__("trading.lean_bridge").__file__, encoding="utf-8").read()
        assert "subprocess" not in source.lower()
