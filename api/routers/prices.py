"""Live price + candlestick data endpoints.

GET /prices/live?pairs=EURUSD,GBPUSD,USDJPY&lookback_bars=50
    Live bid/ask/mid from OANDA + sparkline from SQLite candles.

GET /candles/{pair}/{timeframe}?limit=200
    OHLC bars from SQLite candles table.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

import httpx
import numpy as np
import pandas as pd
from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query, Response
from httpx import Limits

from api.dependencies import get_data_store
from pipeline.pair_config import get_pair_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prices"])

OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"

# Shared async HTTP client for OANDA. Kept alive across requests so the
# highly-polled /prices/live route does not reopen TCP/TLS every 3 seconds.
_OANDA_ASYNC_CLIENT: httpx.AsyncClient | None = None

# 2-second in-memory cache for raw OANDA price responses. Multiple frontend
# widgets polling within the same window share this frame instead of hammering
# the external API.
_OANDA_PRICE_CACHE: TTLCache = TTLCache(maxsize=32, ttl=2)


def _get_oanda_credentials():
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
    return token.strip() if token else None, account_id.strip() if account_id else None


def _oanda_api_call(instruments: str, max_retries: int = 2):
    token, account_id = _get_oanda_credentials()
    if not token or not account_id:
        return None, "key_required"

    for attempt in range(max_retries):
        try:
            import requests
            resp = requests.get(
                f"{OANDA_PRACTICE_URL}/v3/accounts/{account_id}/pricing",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                params={"instruments": instruments},
                timeout=8,
            )
            resp.raise_for_status()
            return resp.json().get("prices", []), "oanda"
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning("OANDA pricing call failed after %d attempts: %s", max_retries, e)
                return None, "unavailable"
            time.sleep(1)
    return None, "unavailable"


async def _get_oanda_async_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient with keep-alive for OANDA."""
    global _OANDA_ASYNC_CLIENT
    if _OANDA_ASYNC_CLIENT is None:
        _OANDA_ASYNC_CLIENT = httpx.AsyncClient(
            base_url=OANDA_PRACTICE_URL,
            limits=Limits(max_connections=10, max_keepalive_connections=10),
            timeout=httpx.Timeout(8.0),
        )
    return _OANDA_ASYNC_CLIENT


async def _oanda_api_call_async(instruments: str, max_retries: int = 2):
    """Async OANDA pricing call using the shared client."""
    token, account_id = _get_oanda_credentials()
    if not token or not account_id:
        return None, "key_required"

    client = await _get_oanda_async_client()
    path = f"/v3/accounts/{account_id}/pricing"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    params = {"instruments": instruments}
    last_exc = None

    for attempt in range(max_retries):
        try:
            resp = await client.get(path, headers=headers, params=params)
            resp.raise_for_status()
            return resp.json().get("prices", []), "oanda"
        except Exception as e:
            last_exc = e
            logger.warning("OANDA async pricing call failed (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

    logger.warning("OANDA async pricing call failed after %d attempts: %s", max_retries, last_exc)
    return None, "unavailable"


async def _fetch_oanda_prices_cached(instruments: str):
    """Return (prices, source) reusing a 2-second in-memory cache."""
    cached = _OANDA_PRICE_CACHE.get(instruments)
    if cached is not None:
        return cached, True
    prices, source = await _oanda_api_call_async(instruments)
    if source == "oanda":
        _OANDA_PRICE_CACHE[instruments] = (prices, source)
    return (prices, source), False


@router.get("/prices/live")
async def get_live_prices(
    response: Response,
    pairs: str = Query("EURUSD,GBPUSD,USDJPY", description="Comma-separated pair symbols"),
    lookback_bars: int = Query(50, ge=1, le=500, description="Candles for sparkline"),
):
    pair_list = [p.strip().upper() for p in pairs.split(",") if p.strip() and len(p.strip()) >= 6][:5]
    if not pair_list:
        raise HTTPException(400, "At least one valid pair required (e.g. EURUSD)")

    instruments = []
    for sym in pair_list:
        try:
            cfg = get_pair_config(sym)
            instruments.append(cfg.oanda_name)
        except Exception:
            raise HTTPException(400, f"Unknown pair: {sym}")

    instrument_str = ",".join(instruments)
    (raw_prices, source), cached = await _fetch_oanda_prices_cached(instrument_str)

    response.headers["Cache-Control"] = "private, max-age=2"
    response.headers["X-Cache"] = "HIT" if cached else "MISS"

    if source == "key_required":
        return {"prices": [], "source": "key_required",
                "message": "Add your OANDA API key in Settings to see live prices."}

    if source == "unavailable":
        return {"prices": [], "source": "unavailable",
                "error": "OANDA API unreachable"}

    price_map: dict = {}
    for p in raw_prices:
        inst = p.get("instrument", "")
        symbol = inst.replace("_", "") if inst else ""
        bid = None
        ask = None
        bids = p.get("bids", [])
        asks = p.get("asks", [])
        if bids:
            try:
                bid = float(bids[0]["price"])
            except (ValueError, KeyError, TypeError):
                pass
        if asks:
            try:
                ask = float(asks[0]["price"])
            except (ValueError, KeyError, TypeError):
                pass
        mid = round((bid + ask) / 2, 10) if bid is not None and ask is not None else None
        spread_pips = None
        if bid is not None and ask is not None and bid > 0:
            try:
                cfg = get_pair_config(symbol)
                spread_pips = round((ask - bid) / cfg.pip_value, 2)
            except Exception:
                spread_pips = round((ask - bid) / 0.0001, 2)
        price_map[symbol] = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_pips": spread_pips,
        }

    store = get_data_store()
    result_prices = []
    for sym in pair_list:
        pdata = price_map.get(sym, {"bid": None, "ask": None, "mid": None, "spread_pips": None})

        sparkline = []
        try:
            for tf in ("M30", "H1", "H4"):
                df = store.get_latest_candles(sym, tf, lookback_bars)
                if not df.empty:
                    for _, row in df.iterrows():
                        t_val = row["time"]
                        t_epoch = int(t_val.timestamp()) if hasattr(t_val, "timestamp") else 0
                        sparkline.append({
                            "t": t_epoch,
                            "v": round(float(row["mid_close"]), 10),
                        })
                    break
        except Exception:
            pass

        change_pct = None
        if len(sparkline) >= 2:
            first_v = sparkline[0]["v"]
            last_v = sparkline[-1]["v"]
            if first_v and first_v != 0:
                change_pct = round((last_v - first_v) / first_v * 100, 2)

        result_prices.append({
            "symbol": sym,
            "bid": pdata["bid"],
            "ask": pdata["ask"],
            "mid": pdata["mid"],
            "spread_pips": pdata["spread_pips"],
            "change_pct": change_pct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sparkline": sparkline,
        })

    return {"prices": result_prices, "source": "oanda"}


@router.get("/candles/{pair}/{timeframe}")
def get_candles(
    pair: str,
    timeframe: str,
    limit: int = Query(200, ge=1, le=1000),
):
    pair = pair.upper()
    try:
        get_pair_config(pair)
    except Exception:
        raise HTTPException(404, f"Unknown pair: {pair}")

    valid_tfs = {"M15", "M30", "H1", "H2", "H4"}
    if timeframe not in valid_tfs:
        raise HTTPException(400, f"Invalid timeframe: {timeframe}. Valid: {sorted(valid_tfs)}")

    store = get_data_store()
    df = store.get_latest_candles(pair, timeframe, limit)

    candles = []
    if not df.empty:
        for _, row in df.iterrows():
            t_val = row["time"]
            t_epoch = int(t_val.timestamp()) if hasattr(t_val, "timestamp") else 0
            candles.append({
                "t": t_epoch,
                "o": round(float(row["mid_open"]), 10),
                "h": round(float(row["mid_high"]), 10),
                "l": round(float(row["mid_low"]), 10),
                "c": round(float(row["mid_close"]), 10),
                "volume": int(row.get("volume", 0) or 0),
            })

    return {"pair": pair, "timeframe": timeframe, "candles": candles}