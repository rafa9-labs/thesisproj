"""Async wrappers for OANDA operations — moves blocking HTTP off the asyncio event loop.

All methods use ``asyncio.to_thread()`` to run synchronous OandaClient calls
in the default thread pool, keeping the event loop free for other work.

Usage::

    from trading.async_oanda import AsyncOandaClient, async_fetch_prices
    client = AsyncOandaClient(access_token="...", account_id="...")
    summary = await client.get_account_summary()
    result = await client.place_market_order("EUR_USD", 1000)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"


class AsyncOandaClient:
    """Async wrapper around the synchronous OandaClient.

    Each method delegates to the sync client via ``asyncio.to_thread``,
    which runs the blocking call in a thread-pool thread.
    """

    def __init__(self, access_token: str, account_id: str, environment: str = "practice") -> None:
        self.access_token = access_token
        self.account_id = account_id
        self.environment = environment
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from trading.oanda_client import OandaClient
            self._client = OandaClient(
                access_token=self.access_token,
                account_id=self.account_id,
                environment=self.environment,
            )
        return self._client

    async def get_account_summary(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_client().get_account_summary)

    async def get_account_details(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_client().get_account_details)

    async def get_positions(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_client().get_positions)

    async def get_position(self, instrument: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_client().get_position, instrument)

    async def close_position(self, instrument: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_client().close_position, instrument)

    async def place_market_order(
        self,
        instrument: str,
        units: int,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._get_client().place_market_order,
            instrument,
            units,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    async def get_open_trades(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_client().get_open_trades)

    async def close_trade(self, trade_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_client().close_trade, trade_id)

    async def close_all(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_client().close_all)

    async def get_transactions(
        self, since_id: str | None = None, count: int = 50
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._get_client().get_transactions, since_id=since_id, count=count
        )

    async def is_connected(self) -> bool:
        return await asyncio.to_thread(self._get_client().is_connected)


async def async_fetch_prices(instruments: str, max_retries: int = 2):
    """Async wrapper for OANDA pricing API call.

    Returns (prices_list, source) where source is "oanda", "key_required", or "unavailable".
    """
    token = _get_credentials_from_env()
    if not token or not token[0]:
        return None, "key_required"

    access_token, account_id = token

    for attempt in range(max_retries):
        try:
            resp = await asyncio.to_thread(
                _sync_price_request, access_token, account_id, instruments
            )
            return resp, "oanda"
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(
                    "OANDA pricing call failed after %d attempts: %s", max_retries, e
                )
                return None, "unavailable"
            await asyncio.sleep(1)
    return None, "unavailable"


def _sync_price_request(access_token: str, account_id: str, instruments: str):
    import requests
    resp = requests.get(
        f"{_OANDA_PRACTICE_URL}/v3/accounts/{account_id}/pricing",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        params={"instruments": instruments},
        timeout=8,
    )
    resp.raise_for_status()
    return resp.json().get("prices", [])


def _get_credentials_from_env():
    token = None
    account_id = None
    try:
        from api.licensing.storage import SecureStorage
        secure = SecureStorage()
        token = secure.get_api_key("oanda")
        account_id = secure.get_kv("oanda_account_id")
    except Exception:
        pass
    token = token or os.environ.get("OANDA_ACCESS_TOKEN", "")
    account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "")
    t = token.strip() if token else None
    a = account_id.strip() if account_id else None
    if t and a:
        return t, a
    return None
