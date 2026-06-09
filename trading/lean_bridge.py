"""
LEAN Bridge — HTTP client for KodaQuant → LEAN Engine communication.

Sends trade signals as Insight JSON commands to the LEAN Engine REST API.
LEAN runs as a separate Docker container with Interactive Brokers brokerage.
No CLI fallback — if the REST API is unreachable, signals are dropped with
critical alerts after 3 retries.

Architecture:
    KodaQuant signal loop → LeanBridge.submit_order()
        → HTTP POST http://lean-engine:8888/live/commands
            → LEAN Engine → IB Gateway → Interactive Brokers
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

LEAN_DEFAULT_HOST = os.environ.get("LEAN_HOST", "localhost")
LEAN_DEFAULT_PORT = int(os.environ.get("LEAN_REST_PORT", "8888"))
LEAN_DEFAULT_URL = f"http://{LEAN_DEFAULT_HOST}:{LEAN_DEFAULT_PORT}"

RETRY_BACKOFF_SECONDS = [1.0, 2.0, 4.0]
RETRY_MAX_ATTEMPTS = 3

AlertCallback = Callable[[str, str], Coroutine[Any, Any, None]]


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PortfolioPosition:
    symbol: str
    quantity: float
    average_price: float
    market_price: float
    market_value: float
    unrealized_pnl: float


@dataclass
class PortfolioState:
    cash: float
    equity: float
    margin_used: float
    buying_power: float
    total_portfolio_value: float
    positions: List[PortfolioPosition] = field(default_factory=list)

    @classmethod
    def from_lean_response(cls, data: dict) -> "PortfolioState":
        positions = []
        for pos in data.get("Positions", {}).get("Values", []):
            positions.append(PortfolioPosition(
                symbol=pos.get("Symbol", {}).get("Value", ""),
                quantity=float(pos.get("Quantity", 0)),
                average_price=float(pos.get("AveragePrice", 0)),
                market_price=float(pos.get("MarketPrice", 0)),
                market_value=float(pos.get("MarketValue", 0)),
                unrealized_pnl=float(pos.get("UnrealizedProfit", 0)),
            ))
        return cls(
            cash=float(data.get("Cash", {}).get("Amount", 0)),
            equity=float(data.get("TotalPortfolioValue", 0)),
            margin_used=float(data.get("TotalMarginUsed", 0)),
            buying_power=float(data.get("BuyingPower", 0)),
            total_portfolio_value=float(data.get("TotalPortfolioValue", 0)),
            positions=positions,
        )


@dataclass
class OrderResult:
    order_id: int | None
    success: bool
    error: str | None = None
    raw_response: dict | None = None


@dataclass
class OrderStatus:
    order_id: int
    symbol: str
    direction: str
    quantity: float
    filled_quantity: float
    status: str
    fill_price: float | None = None


# ═══════════════════════════════════════════════════════════════════
# Insight Builder
# ═══════════════════════════════════════════════════════════════════

def signal_to_insight(
    direction: int,
    confidence: float,
    symbol: str = "EURUSD",
    tag: str = "",
) -> dict:
    if direction == 0:
        return {}

    market = "Oanda"
    security_type = "Forex"

    insight = {
        "Symbol": symbol,
        "Market": market,
        "SecurityType": security_type,
    "Direction": "Up" if direction == 1 else "Down",
    "Period": _timedelta(hours=1),
    "Confidence": confidence,
    "Tag": tag or f"kodaquant_{int(time.time())}",
}

    return insight


def _timedelta(*, hours: int = 1) -> str:
    return f"{hours}.00:00:00"


# ═══════════════════════════════════════════════════════════════════
# LEAN Bridge Client
# ═══════════════════════════════════════════════════════════════════

class LeanBridge:
    def __init__(
        self,
        base_url: str = LEAN_DEFAULT_URL,
        timeout: float = 15.0,
        alert_callback: Optional[AlertCallback] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.alert_callback = alert_callback
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Retry logic ──────────────────────────────────────────────

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        json_body: Optional[dict] = None,
    ) -> Optional[httpx.Response]:
        client = await self._get_client()
        last_error = None

        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                if method == "POST":
                    resp = await client.post(path, json=json_body)
                elif method == "GET":
                    resp = await client.get(path)
                else:
                    resp = await client.request(method, path, json=json_body)

                if resp.status_code < 500:
                    return resp

                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
            except httpx.ConnectError as e:
                last_error = f"Connection refused: {e}"
            except Exception as e:
                last_error = f"Unexpected: {e}"

            if attempt < RETRY_MAX_ATTEMPTS - 1:
                backoff = RETRY_BACKOFF_SECONDS[attempt]
                logger.warning("LEAN request failed (attempt %d/%d), retrying in %.1fs: %s",
                               attempt + 1, RETRY_MAX_ATTEMPTS, backoff, last_error)
                await asyncio.sleep(backoff)

        logger.critical("LEAN bridge exhausted %d retries: %s", RETRY_MAX_ATTEMPTS, last_error)
        if self.alert_callback:
            try:
                await self.alert_callback(
                    "lean_bridge_failure",
                    f"LEAN bridge exhausted {RETRY_MAX_ATTEMPTS} retries: {last_error}",
                )
            except Exception:
                logger.exception("Failed to fire alert callback")
        return None

    # ── Order Submission ─────────────────────────────────────────

    async def submit_order(
        self,
        symbol: str,
        direction: int,
        quantity: float,
        order_type: str = "Market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tag: str = "",
    ) -> OrderResult:
        if direction == 0:
            return OrderResult(order_id=None, success=True, error=None)

        command = {
            "$type": "SubmitOrderCommand",
            "ticker": symbol,
            "market": "Oanda",
            "securityType": "Forex",
            "orderType": order_type,
            "quantity": quantity,
            "tag": tag or f"kodaquant_{int(time.time())}",
        }
        if limit_price is not None:
            command["limitPrice"] = limit_price
        if stop_price is not None:
            command["stopPrice"] = stop_price

        logger.info("LEAN submit_order: %s %s qty=%.0f type=%s tag=%s",
                     symbol, "BUY" if direction > 0 else "SELL", quantity, order_type, tag)

        resp = await self._request_with_retry("POST", "/live/commands", json_body=command)
        if resp is None:
            return OrderResult(order_id=None, success=False,
                               error="LEAN API unreachable after retries")
        try:
            data = resp.json()
            order_id = data.get("CommandId") or data.get("Id")
            return OrderResult(order_id=order_id, success=True, raw_response=data)
        except Exception as e:
            logger.error("Failed to parse LEAN submit_order response: %s", e)
            return OrderResult(order_id=None, success=False,
                               error=f"Response parse error: {e}")

    # ── Cancel Order ─────────────────────────────────────────────

    async def cancel_order(self, order_id: int) -> OrderResult:
        command = {
            "$type": "CancelOrderCommand",
            "orderId": order_id,
        }
        logger.info("LEAN cancel_order: id=%d", order_id)
        resp = await self._request_with_retry("POST", "/live/commands", json_body=command)
        if resp is None:
            return OrderResult(order_id=order_id, success=False,
                               error="LEAN API unreachable after retries")
        return OrderResult(order_id=order_id, success=resp.status_code < 400)

    # ── Liquidate All ────────────────────────────────────────────

    async def liquidate_all(self) -> OrderResult:
        command = {
            "$type": "LiquidateCommand",
            "liquidateAll": True,
        }
        logger.info("LEAN liquidate_all")
        resp = await self._request_with_retry("POST", "/live/commands", json_body=command)
        if resp is None:
            return OrderResult(order_id=None, success=False,
                               error="LEAN API unreachable after retries")
        return OrderResult(order_id=None, success=resp.status_code < 400)

    # ── Portfolio ────────────────────────────────────────────────

    async def get_portfolio(self) -> Optional[PortfolioState]:
        resp = await self._request_with_retry("GET", "/live/portfolio")
        if resp is None:
            return None
        try:
            return PortfolioState.from_lean_response(resp.json())
        except Exception as e:
            logger.error("Failed to parse LEAN portfolio response: %s", e)
            return None

    # ── Orders ───────────────────────────────────────────────────

    async def get_orders(self) -> List[OrderStatus]:
        resp = await self._request_with_retry("GET", "/live/orders")
        if resp is None:
            return []
        try:
            orders = resp.json().get("Orders", [])
            return [
                OrderStatus(
                    order_id=o.get("Id", 0),
                    symbol=o.get("Symbol", {}).get("Value", ""),
                    direction=o.get("Direction", "Hold"),
                    quantity=float(o.get("Quantity", 0)),
                    filled_quantity=float(o.get("Quantity", 0)) - float(o.get("RemainingQuantity", 0)),
                    status=o.get("Status", "Unknown"),
                    fill_price=float(o.get("Price", 0)) if o.get("Price") else None,
                )
                for o in orders
            ]
        except Exception as e:
            logger.error("Failed to parse LEAN orders response: %s", e)
            return []

    # ── Health Check ─────────────────────────────────────────────

    async def health_check(self) -> bool:
        resp = await self._request_with_retry("GET", "/live/portfolio")
        return resp is not None and resp.status_code < 500
