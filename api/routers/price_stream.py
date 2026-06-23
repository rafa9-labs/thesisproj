"""OANDA v20 Streaming API singleton — one connection per symbol.

Architecture:
  OANDA stream-fxpractice.oanda.com
        │  (one httpx stream per symbol)
        ▼
  _SymbolStream._read_loop()   <- asyncio Task, parses NDJSON lines
        │
        ├─► TickBuffer.append(timestamp, bid, ask)
        │       │
        │       ├─► Fan-out to subscribed WS clients as "price_tick" event
        │       │
        │       └─► _check_bar_close() — clock-aligned boundary detection
        │               │
        │               ├─► Compute OHLC from tick buffer
        │               ├─► store.upsert_candle()  <- DB write on ingestion thread
        │               └─► new_bar_event.set()     <- wakes signal engine
        │
        └─► HEARTBEAT -> ignored (OANDA keep-alive)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from api.dependencies import get_data_store
from api.routers.prices import _get_oanda_credentials

logger = logging.getLogger(__name__)

OANDA_STREAM_URL = "https://stream-fxpractice.oanda.com"

TF_SECONDS = {"M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}


def clock_bar_start(ts: datetime, timeframe: str) -> datetime:
    """Return the clock-aligned start of the bar containing *ts*.

    M30 -> floors to :00 or :30 of the hour.
    H1  -> floors to :00 of the hour.
    H4  -> floors to 00/04/08/12/16/20 of the day.
    """
    if timeframe == "M30":
        minute = 0 if ts.minute < 30 else 30
        return ts.replace(minute=minute, second=0, microsecond=0)
    if timeframe == "H1":
        return ts.replace(minute=0, second=0, microsecond=0)
    if timeframe == "H4":
        hour = (ts.hour // 4) * 4
        return ts.replace(hour=hour, minute=0, second=0, microsecond=0)
    seconds = TF_SECONDS.get(timeframe, 60)
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp((epoch // seconds) * seconds, tz=timezone.utc)


class TickBuffer:
    """Accumulates OANDA ticks for one bar, computes OHLC on close."""

    __slots__ = ("pair", "timeframe", "_ticks", "_bar_start",
                 "_open", "_high", "_low", "_close", "_first_tick")

    def __init__(self, pair: str, timeframe: str, bar_start: datetime):
        self.pair = pair
        self.timeframe = timeframe
        self._ticks: list[tuple[datetime, float]] = []
        self._bar_start = bar_start
        self._open = self._high = self._low = self._close = 0.0
        self._first_tick = True

    @property
    def bar_start(self) -> datetime:
        return self._bar_start

    @property
    def tick_count(self) -> int:
        return len(self._ticks)

    def append(self, ts: datetime, mid: float) -> None:
        self._ticks.append((ts, mid))
        if self._first_tick:
            self._open = mid
            self._high = mid
            self._low = mid
            self._first_tick = False
        else:
            if mid > self._high:
                self._high = mid
            if mid < self._low:
                self._low = mid
        self._close = mid

    def snapshot(self) -> dict:
        """Current forming-bar OHLC for pushing to frontend."""
        if self._first_tick:
            return {}
        return {
            "time": int(self._bar_start.timestamp()),
            "open": round(self._open, 10),
            "high": round(self._high, 10),
            "low": round(self._low, 10),
            "close": round(self._close, 10),
        }

    def finalize(self) -> dict:
        """Called on bar close — returns completed OHLC and resets."""
        result = self.snapshot()
        self._ticks.clear()
        self._first_tick = True
        return result


class _SymbolStream:
    """One httpx streaming connection to OANDA per symbol."""

    def __init__(self, pair: str, oanda_instrument: str,
                 token: str, account_id: str):
        self.pair = pair
        self._instrument = oanda_instrument
        self._token = token
        self._account_id = account_id

        self._client: Optional[httpx.AsyncClient] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        self._buffers: dict[str, TickBuffer] = {}

        self._ws_queues: list[asyncio.Queue] = []

        self._new_bar_events: dict[str, asyncio.Event] = {}

        self._ref_count = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))
        self._task = asyncio.create_task(self._read_loop())
        logger.info("Price stream started for %s", self.pair)

    async def stop(self) -> None:
        if not self.running:
            return
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Price stream stopped for %s", self.pair)

    def add_ref(self) -> None:
        self._ref_count += 1

    def remove_ref(self) -> None:
        self._ref_count = max(0, self._ref_count - 1)

    @property
    def ref_count(self) -> int:
        return self._ref_count

    def subscribe(self, queue: asyncio.Queue) -> None:
        if queue not in self._ws_queues:
            self._ws_queues.append(queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        try:
            self._ws_queues.remove(queue)
        except ValueError:
            pass

    def get_new_bar_event(self, timeframe: str) -> asyncio.Event:
        if timeframe not in self._new_bar_events:
            self._new_bar_events[timeframe] = asyncio.Event()
        return self._new_bar_events[timeframe]

    def _broadcast(self, msg: dict) -> None:
        for q in self._ws_queues:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    async def _read_loop(self) -> None:
        """Long-lived async generator reading OANDA pricing stream."""
        url = (f"{OANDA_STREAM_URL}/v3/accounts/{self._account_id}"
               f"/pricing/stream?instruments={self._instrument}")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        while not self._stop_event.is_set():
            try:
                async with self._client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.error("OANDA stream HTTP %s for %s: %s",
                                     resp.status_code, self.pair, body[:200])
                        await asyncio.sleep(5)
                        continue

                    async for line in resp.aiter_lines():
                        if self._stop_event.is_set():
                            break
                        if not line or not line.strip():
                            continue
                        self._process_line(line.strip())

            except httpx.RemoteProtocolError:
                logger.warning("OANDA stream disconnected for %s, reconnecting in 5s", self.pair)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("OANDA stream error for %s, reconnecting in 5s", self.pair)
                await asyncio.sleep(5)

    def _process_line(self, line: str) -> None:
        import json as _json
        try:
            msg = _json.loads(line)
        except _json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "HEARTBEAT":
            return

        if msg_type != "PRICE":
            logger.debug("OANDA stream unknown type=%s for %s", msg_type, self.pair)
            return

        try:
            time_str = msg.get("time", "")
            tick_ts = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])
            if not bids or not asks:
                return
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            mid = round((bid + ask) / 2.0, 10)
        except (ValueError, KeyError, IndexError) as e:
            logger.debug("OANDA stream parse error for %s: %s", self.pair, e)
            return

        for tf in ["M30", "H1", "H4"]:
            self._on_tick(tf, tick_ts, mid, bid, ask)

    def _on_tick(self, timeframe: str, ts: datetime,
                 mid: float, bid: float, ask: float) -> None:
        bar_start = clock_bar_start(ts, timeframe)

        if timeframe not in self._buffers:
            self._buffers[timeframe] = TickBuffer(self.pair, timeframe, bar_start)

        buf = self._buffers[timeframe]

        if bar_start != buf.bar_start:
            self._commit_bar(timeframe, buf)
            self._buffers[timeframe] = TickBuffer(self.pair, timeframe, bar_start)
            buf = self._buffers[timeframe]

        buf.append(ts, mid)

        forming = buf.snapshot()
        if forming:
            self._broadcast({
                "event": "price_tick",
                "pair": self.pair,
                "timeframe": timeframe,
                "live_price": round(mid, 10),
                "forming_candle": forming,
                "bid": round(bid, 10),
                "ask": round(ask, 10),
            })

    def _commit_bar(self, timeframe: str, buf: TickBuffer) -> None:
        """Compute OHLC from tick buffer -> upsert DB -> emit new_bar event."""
        if buf.tick_count == 0:
            return

        ohlc = buf.finalize()
        if not ohlc:
            return

        ts_str = buf.bar_start.strftime("%Y-%m-%d %H:%M:%S%z")
        pair = self.pair
        mid_o = ohlc["open"]
        mid_h = ohlc["high"]
        mid_l = ohlc["low"]
        mid_c = ohlc["close"]

        try:
            store = get_data_store()
            asyncio.create_task(
                asyncio.to_thread(
                    store.upsert_candle,
                    pair, timeframe, ts_str,
                    mid_o, mid_h, mid_l, mid_c,
                )
            )
            logger.debug("Upserted %s %s bar at %s", pair, timeframe, ts_str)
        except Exception:
            logger.exception("Failed to upsert candle for %s %s", pair, timeframe)

        event = self._new_bar_events.get(timeframe)
        if event:
            event.set()
            self._new_bar_events[timeframe] = asyncio.Event()

        self._broadcast({
            "event": "new_bar_saved",
            "pair": self.pair,
            "timeframe": timeframe,
            "candle": ohlc,
        })


class PriceStreamManager:
    """Global singleton that multiplexes one OANDA stream per symbol.

    Usage::

        manager = PriceStreamManager.get()
        await manager.ensure_stream("EURUSD")
        queue = asyncio.Queue()
        manager.subscribe("EURUSD", queue)

        # In signal engine:
        event = manager.get_new_bar_event("EURUSD", "M30")
        await event.wait()  # wakes when a new M30 bar is saved to DB
    """

    _instance: Optional["PriceStreamManager"] = None

    def __init__(self):
        self._streams: dict[str, _SymbolStream] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls) -> "PriceStreamManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def ensure_stream(self, pair: str) -> None:
        """Start the OANDA stream for *pair* if not already running."""
        pair = pair.upper()
        token, account_id = _get_oanda_credentials()
        if not token or not account_id:
            logger.warning("Cannot start price stream for %s — no OANDA credentials", pair)
            return

        async with self._lock:
            stream = self._streams.get(pair)
            if stream is None:
                from pipeline.pair_config import get_pair_config
                try:
                    cfg = get_pair_config(pair)
                    oanda_name = cfg.oanda_name
                except Exception:
                    logger.error("Unknown pair %s — cannot start stream", pair)
                    return

                stream = _SymbolStream(pair, oanda_name, token, account_id)
                self._streams[pair] = stream
            stream.add_ref()
            if not stream.running:
                await stream.start()
                logger.info("PriceStreamManager: started OANDA stream for %s", pair)

    async def release_stream(self, pair: str) -> None:
        """Decrement ref count; stop stream when no more subscribers."""
        pair = pair.upper()
        async with self._lock:
            stream = self._streams.get(pair)
            if stream is None:
                return
            stream.remove_ref()
            if stream.ref_count <= 0:
                await stream.stop()
                del self._streams[pair]

    def subscribe(self, pair: str, queue: asyncio.Queue) -> None:
        stream = self._streams.get(pair.upper())
        if stream:
            stream.subscribe(queue)

    def unsubscribe(self, pair: str, queue: asyncio.Queue) -> None:
        stream = self._streams.get(pair.upper())
        if stream:
            stream.unsubscribe(queue)

    def get_new_bar_event(self, pair: str, timeframe: str) -> Optional[asyncio.Event]:
        """Return an asyncio.Event that is set when a completed bar is saved to DB."""
        stream = self._streams.get(pair.upper())
        if stream is None:
            return None
        return stream.get_new_bar_event(timeframe)

    async def shutdown(self) -> None:
        async with self._lock:
            for pair, stream in list(self._streams.items()):
                await stream.stop()
            self._streams.clear()
        logger.info("PriceStreamManager: all streams shut down")
