"""Live price + candlestick data endpoints.

GET /prices/live?pairs=EURUSD,GBPUSD,USDJPY&lookback_bars=50
    Live bid/ask/mid from OANDA + sparkline from SQLite candles.

GET /candles/{pair}/{timeframe}?limit=200
    OHLC bars from SQLite candles table.

WS  /chart/{pair}/{timeframe}/ws
    Real-time chart stream — forming candles + new-bar events per tick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from api.dependencies import get_data_store
from pipeline.pair_config import get_pair_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prices"])

OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"


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


def _timeframe_minutes(timeframe: str) -> int:
    return {"M15": 15, "M30": 30, "H1": 60, "H4": 240}.get(timeframe, 60)


def _query_candles(
    store, pair: str, timeframe: str,
    start: Optional[int], end: Optional[int], limit: int,
) -> list[dict]:
    if start is not None or end is not None:
        ts_fmt = "%Y-%m-%d %H:%M:%S%z"
        start_iso = datetime.fromtimestamp(start, tz=timezone.utc).strftime(ts_fmt) if start is not None else None
        end_iso = datetime.fromtimestamp(end, tz=timezone.utc).strftime(ts_fmt) if end is not None else None
        df = store.get_candles(pair, timeframe, start=start_iso, end=end_iso)
        if limit and len(df) > limit:
            df = df.iloc[-limit:]
    else:
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
    return candles


def _backfill_candles(
    store, pair_cfg, pair: str, timeframe: str,
    last_ts: datetime, now: datetime, token: str, account_id: str,
) -> int:
    from oandapyV20 import API
    from pipeline.data_downloader import _fetch_candles, _df_to_rows

    oanda_env = os.environ.get("OANDA_ENV", "practice").strip().lower()
    client = API(access_token=token, environment=oanda_env)
    oanda_name = pair_cfg.oanda_name

    df = _fetch_candles(client, oanda_name, last_ts, now, timeframe, include_incomplete=True)
    if df.empty:
        return 0

    rows = _df_to_rows(df, pair, timeframe)
    store.insert_candles_batch(rows)
    return len(rows)


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


@router.get("/prices/live")
def get_live_prices(
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
    raw_prices, source = _oanda_api_call(instrument_str)

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
    start: int = Query(None, description="Start epoch seconds (inclusive)"),
    end: int = Query(None, description="End epoch seconds (inclusive)"),
):
    pair = pair.upper()
    try:
        pair_cfg = get_pair_config(pair)
    except Exception:
        raise HTTPException(404, f"Unknown pair: {pair}")

    valid_tfs = {"M15", "M30", "H1", "H2", "H4"}
    if timeframe not in valid_tfs:
        raise HTTPException(400, f"Invalid timeframe: {timeframe}. Valid: {sorted(valid_tfs)}")

    store = get_data_store()
    candles = _query_candles(store, pair, timeframe, start, end, limit)

    if not start and not end:
        now = datetime.now(timezone.utc)
        last_epoch = candles[-1]["t"] if candles else 0
        last_ts = datetime.fromtimestamp(last_epoch, tz=timezone.utc) if last_epoch else now
        tf_minutes = _timeframe_minutes(timeframe)
        gap_minutes = (now - last_ts).total_seconds() / 60

        if gap_minutes > tf_minutes:
            from pipeline.candle_syncer import _is_forex_weekend
            import pandas as pd
            if _is_forex_weekend(pd.Timestamp(now)):
                logger.debug(
                    "Skipping backfill for %s/%s — weekend gap (%d min -> %d candles), "
                    "market closed until Sun 22:00 UTC",
                    pair, timeframe, int(gap_minutes), int(gap_minutes / tf_minutes),
                )
            else:
                token, account_id = _get_oanda_credentials()
                if token and account_id:
                    try:
                        new_count = _backfill_candles(
                            store, pair_cfg, pair, timeframe,
                            last_ts, now, token, account_id,
                        )
                        if new_count > 0:
                            logger.info("Backfill %s/%s: %d candles inserted", pair, timeframe, new_count)
                            candles = _query_candles(store, pair, timeframe, None, None, limit)
                    except Exception as e:
                        logger.warning("Backfill failed for %s/%s: %s", pair, timeframe, e)
                else:
                    logger.info("Skipping backfill for %s/%s - no OANDA credentials", pair, timeframe)

    return {"pair": pair, "timeframe": timeframe, "candles": candles}


@router.websocket("/chart/{pair}/{timeframe}/ws")
async def chart_ws(websocket: WebSocket, pair: str, timeframe: str):
    """Real-time chart stream — forming candles + new-bar events per tick.

    Subscribe to the OANDA price stream for *pair* and forward
    ``price_tick`` (with ``forming_candle``) and ``new_bar_saved``
    events to the frontend.  No trading session required.
    """
    await websocket.accept()

    pair = pair.upper()
    valid_tfs = {"M15", "M30", "H1", "H2", "H4"}
    if timeframe not in valid_tfs:
        await websocket.send_text(json.dumps({"event": "error", "message": f"Invalid timeframe: {timeframe}"}))
        await websocket.close()
        return

    try:
        get_pair_config(pair)
    except Exception:
        await websocket.send_text(json.dumps({"event": "error", "message": f"Unknown pair: {pair}"}))
        await websocket.close()
        return

    from api.routers.price_stream import PriceStreamManager
    psm = PriceStreamManager.get()
    await psm.ensure_stream(pair)

    queue: asyncio.Queue = asyncio.Queue()
    psm.subscribe(pair, queue)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"event": "heartbeat", "time": int(time.time())}))
                continue

            if msg.get("pair") != pair or msg.get("timeframe") != timeframe:
                continue

            await websocket.send_text(json.dumps(msg, default=str))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Chart WS error for %s/%s", pair, timeframe)
    finally:
        psm.unsubscribe(pair, queue)
        await psm.release_stream(pair)