"""OANDA v20 REST API client — account, position, order, and trade management.

Design
------
- Single-purpose wrapper over ``oandapyV20`` (already a project dependency).
- All methods make exactly one rate-limited API call.
- ``close_all()`` is the emergency stop — closes every open position and
  cancels every pending order.
- ``environment`` controls the base URL: ``"practice"`` → demo, ``"live"`` → real.

Usage::

    from trading.oanda_client import OandaClient

    client = OandaClient(access_token="...", account_id="...")
    summary = client.get_account_summary()
    client.place_market_order("EUR_USD", 1000, stop_loss=1.0850)
    client.close_all()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from oandapyV20 import API

logger = logging.getLogger(__name__)


@dataclass
class OandaClient:
    """Low-level OANDA v20 REST API client."""

    access_token: str
    account_id: str
    environment: str = "practice"
    max_requests_per_sec: float = 20.0

    _api: API = field(init=False)

    def __post_init__(self) -> None:
        self._api = API(
            access_token=self.access_token,
            environment=self.environment,
        )

    # ── Rate Limiting ──────────────────────────────────────────

    def _throttle(self) -> None:
        interval = 1.0 / max(self.max_requests_per_sec, 0.1)
        elapsed = time.monotonic() - getattr(self, "_last_ts", 0.0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        object.__setattr__(self, "_last_ts", time.monotonic())

    def _req(self, endpoint: object) -> dict[str, Any]:
        self._throttle()
        try:
            self._api.request(endpoint)
        except Exception:
            logger.exception("OANDA API request failed")
            raise
        return endpoint.response  # type: ignore[attr-defined]

    # ── Account ─────────────────────────────────────────────────

    def get_account_summary(self) -> dict[str, Any]:
        from oandapyV20.endpoints.accounts import AccountSummary as EP
        return self._req(EP(self.account_id)).get("account", {})

    def get_account_details(self) -> dict[str, Any]:
        from oandapyV20.endpoints.accounts import AccountDetails as EP
        return self._req(EP(self.account_id)).get("account", {})

    # ── Positions ───────────────────────────────────────────────

    def get_positions(self) -> list[dict[str, Any]]:
        from oandapyV20.endpoints.positions import PositionList as EP
        try:
            return self._req(EP(self.account_id)).get("positions", [])
        except Exception:
            return []

    def get_position(self, instrument: str) -> dict[str, Any] | None:
        from oandapyV20.endpoints.positions import PositionDetails as EP
        try:
            return self._req(EP(self.account_id, instrument)).get("position")
        except Exception:
            return None

    def close_position(self, instrument: str) -> dict[str, Any]:
        from oandapyV20.endpoints.positions import PositionClose as EP
        data = {"longUnits": "ALL", "shortUnits": "ALL"}
        return self._req(EP(self.account_id, instrument, data))

    # ── Orders ──────────────────────────────────────────────────

    def _order_body(
        self,
        order_type: str,
        instrument: str,
        units: int,
        *,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": order_type,
            "instrument": instrument,
            "units": str(units),
        }
        if price is not None:
            body["price"] = str(round(price, 5))

        if stop_loss is not None:
            body["stopLossOnFill"] = {"price": str(round(stop_loss, 5))}

        if take_profit is not None:
            body["takeProfitOnFill"] = {"price": str(round(take_profit, 5))}

        return {"order": body}

    def place_market_order(
        self,
        instrument: str,
        units: int,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        from oandapyV20.endpoints.orders import OrderCreate as EP
        body = self._order_body("MARKET", instrument, units,
                                stop_loss=stop_loss, take_profit=take_profit)
        return self._req(EP(self.account_id, body))

    def place_limit_order(
        self,
        instrument: str,
        units: int,
        price: float,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        from oandapyV20.endpoints.orders import OrderCreate as EP
        body = self._order_body("LIMIT", instrument, units, price=price,
                                stop_loss=stop_loss, take_profit=take_profit)
        return self._req(EP(self.account_id, body))

    def place_stop_order(
        self,
        instrument: str,
        units: int,
        price: float,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        from oandapyV20.endpoints.orders import OrderCreate as EP
        body = self._order_body("STOP", instrument, units, price=price,
                                stop_loss=stop_loss, take_profit=take_profit)
        return self._req(EP(self.account_id, body))

    def get_pending_orders(self) -> list[dict[str, Any]]:
        from oandapyV20.endpoints.orders import OrdersPending as EP
        try:
            return self._req(EP(self.account_id)).get("orders", [])
        except Exception:
            return []

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        from oandapyV20.endpoints.orders import OrderCancel as EP
        return self._req(EP(self.account_id, order_id))

    # ── Trades ──────────────────────────────────────────────────

    def get_open_trades(self) -> list[dict[str, Any]]:
        from oandapyV20.endpoints.trades import TradesList as EP
        try:
            return self._req(EP(self.account_id)).get("trades", [])
        except Exception:
            return []

    def get_trade(self, trade_id: str) -> dict[str, Any]:
        from oandapyV20.endpoints.trades import TradeDetails as EP
        return self._req(EP(self.account_id, trade_id)).get("trade", {})

    def close_trade(self, trade_id: str) -> dict[str, Any]:
        from oandapyV20.endpoints.trades import TradeClose as EP
        return self._req(EP(self.account_id, trade_id))

    # ── Emergency Stop ──────────────────────────────────────────

    def close_all(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for pos in self.get_positions():
            inst = pos.get("instrument", "")
            if not inst:
                continue
            try:
                r = self.close_position(inst)
                results.append({"action": "close_position", "instrument": inst,
                                "status": "ok"})
            except Exception as exc:
                results.append({"action": "close_position", "instrument": inst,
                                "status": "error", "error": str(exc)})

        for order in self.get_pending_orders():
            oid = order.get("id", "")
            if not oid:
                continue
            try:
                self.cancel_order(oid)
                results.append({"action": "cancel_order", "order_id": oid,
                                "status": "ok"})
            except Exception as exc:
                results.append({"action": "cancel_order", "order_id": oid,
                                "status": "error", "error": str(exc)})

        return results

    # ── Transaction History ─────────────────────────────────────

    def get_transactions(
        self,
        since_id: str | None = None,
        count: int = 50,
    ) -> list[dict[str, Any]]:
        from oandapyV20.endpoints.transactions import TransactionList as EP
        params: dict[str, Any] = {"count": min(count, 500)}
        if since_id:
            params["from"] = since_id
        return self._req(EP(self.account_id, params=params)).get("transactions", [])

    # ── Convenience ─────────────────────────────────────────────

    def has_open_positions(self) -> bool:
        return len(self.get_positions()) > 0

    def is_connected(self) -> bool:
        try:
            self.get_account_summary()
            return True
        except Exception:
            return False
