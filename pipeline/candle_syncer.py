"""Incremental OANDA candle sync — keeps SQLite fresh for dashboard and live trading.

Design
------
- Stateless: queries latest timestamp from SQLite, fetches delta, UPSERTs.
  Survives server restarts with zero data gaps.
- Staggered intervals per timeframe to respect OANDA rate limits.
- Includes forming (incomplete) candles so the chart always has the
  current bar's latest OHLC.
- One asyncio.Task per (pair, timeframe) runs independently.

Usage::

    from pipeline.candle_syncer import CandleSyncer
    syncer = CandleSyncer(store, ["EUR_USD"], ["M30", "H1"])
    await syncer.start()
    # ... server running ...
    await syncer.stop()
"""

import asyncio
import logging
import os
from typing import Optional

import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints import instruments

from pipeline.data_sqlite import DataStore

logger = logging.getLogger(__name__)

OANDA_ENV = os.environ.get("OANDA_ENV", "practice").strip().lower()

INTERVALS: dict[str, float] = {
    "M1":  15,
    "M5":  30,
    "M15": 60,
    "M30": 90,
    "H1":  300,
    "H4":  900,
}

_SQLITE_CHUNK = 500


def _norm_pair(pair: str) -> str:
    return pair.upper().replace("/", "_")


def _norm_tf(timeframe: str) -> str:
    tf = timeframe.upper()
    if tf in ("1M", "1MIN"):
        return "M1"
    if tf in ("5M", "5MIN"):
        return "M5"
    if tf in ("15M", "15MIN"):
        return "M15"
    if tf in ("30M", "30MIN"):
        return "M30"
    if tf in ("1H", "1HR"):
        return "H1"
    if tf in ("4H", "4HR"):
        return "H4"
    return tf


def _normalize_ts(ts: str) -> str:
    """Convert OANDA RFC 3339 timestamp to DB format: YYYY-MM-DD HH:MM:SS+00:00."""
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return ts
    if pd.isna(t):
        return ts
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d %H:%M:%S+00:00")


def _granularity(timeframe: str) -> str:
    mapping = {
        "M1": "M1", "M5": "M5", "M15": "M15",
        "M30": "M30", "H1": "H1", "H4": "H4",
    }
    return mapping.get(timeframe, timeframe)


# Forex market: Sunday 22:00 UTC → Friday 22:00 UTC
# Weekend: Friday 22:00 UTC → Sunday 22:00 UTC
_FOREX_WEEKEND_START_UTC_HOUR = 22  # Friday 22:00
_FOREX_CLOSE_DAY = 4               # Friday (Monday=0)
_FOREX_OPEN_DAY = 6                # Sunday (Monday=0)


def _is_forex_weekend(dt: "pd.Timestamp") -> bool:
    """Return True if *dt* falls inside the Forex weekend (Fri 22:00–Sun 22:00 UTC)."""
    wd = dt.dayofweek   # Monday=0 ... Sunday=6
    hh = dt.hour
    if wd == _FOREX_CLOSE_DAY and hh >= _FOREX_WEEKEND_START_UTC_HOUR:
        return True
    if wd == 5:  # Saturday
        return True
    if wd == _FOREX_OPEN_DAY and hh < _FOREX_WEEKEND_START_UTC_HOUR:
        return True
    return False


def _validate_m30_sequence(rows: list[tuple], pair: str) -> None:
    """Log a warning if any M30 candle timestamp is not aligned to :00 or :30."""
    for row in rows:
        ts_str = row[2]  # ts column at index 2
        try:
            ts = pd.Timestamp(ts_str)
        except Exception:
            continue
        if ts.minute % 30 != 0 or ts.second != 0 or ts.microsecond != 0:
            logger.warning(
                "Non-aligned M30 candle for %s: %s (minute=%d, second=%d)",
                pair, ts_str, ts.minute, ts.second,
            )


_DEFAULT_TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4"]

_AUTH_BACKOFF = [300, 600, 1800, 3600]

_COLD_START_LOOKBACK: dict[str, int] = {
    "M1":  360,
    "M5":  360,
    "M15": 360,
    "M30": 360,
    "H1":  720,
    "H4":  2880,
}


def _resolve_credentials() -> tuple[str, str] | None:
    token = os.environ.get("OANDA_ACCESS_TOKEN", "").strip()
    account_id = os.environ.get("OANDA_ACCOUNT_ID", "").strip()
    if not token or not account_id:
        try:
            from api.licensing.storage import SecureStorage
            secure = SecureStorage()
            token = secure.get_api_key("oanda") or ""
            account_id = secure.get_kv("oanda_account_id") or ""
        except Exception:
            pass
    token = token.strip()
    account_id = account_id.strip()
    if token and account_id:
        return token, account_id
    return None


def _is_auth_error(exc: Exception) -> bool:
    try:
        from oandapyV20.exceptions import V20Error
        if isinstance(exc, V20Error):
            msg = str(exc).lower()
            return "insufficient authorization" in msg or "invalid token" in msg
    except ImportError:
        pass
    return False


class CandleSyncer:
    """Incremental OANDA candle sync engine."""

    def __init__(
        self,
        store: DataStore,
        active_pairs: Optional[list[str]] = None,
        timeframes: Optional[list[str]] = None,
    ):
        self._store = store
        self._pairs: list[str] = [_norm_pair(p) for p in (active_pairs or [])]
        self._timeframes: list[str] = [
            _norm_tf(t)
            for t in (timeframes or _DEFAULT_TIMEFRAMES)
        ]
        self._api: Optional[API] = None
        self._account_id: Optional[str] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ── public API ───────────────────────────────────────────────

    async def start(self) -> None:
        creds = _resolve_credentials()
        if not creds:
            logger.warning(
                "CandleSyncer: No OANDA credentials — sync disabled."
            )
            return
        token, account_id = creds
        self._api = API(access_token=token, environment=OANDA_ENV)
        self._account_id = account_id
        self._running = True

        for pair in self._pairs:
            for tf in self._timeframes:
                task = asyncio.create_task(
                    self._sync_loop(pair, tf)
                )
                self._tasks.append(task)
                await asyncio.sleep(_TASK_SPAWN_STAGGER)

        logger.info(
            "CandleSyncer started: %d pairs x %d timeframes = %d tasks",
            len(self._pairs), len(self._timeframes), len(self._tasks),
        )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("CandleSyncer stopped")

    async def sync_pair(self, pair: str, tf: str) -> int:
        pair = _norm_pair(pair)
        tf = _norm_tf(tf)
        return await self._sync_pair_impl(pair, tf)

    # ── internals ────────────────────────────────────────────────

    async def _sync_loop(self, pair: str, tf: str) -> None:
        interval = INTERVALS.get(tf, 60)
        consecutive_failures = 0
        while self._running:
            try:
                count = await self._sync_pair_impl(pair, tf)
                if count:
                    logger.debug(
                        "CandleSyncer %s/%s: %d new/updated candles",
                        pair, tf, count,
                    )
                consecutive_failures = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                consecutive_failures += 1
                backoff_idx = min(consecutive_failures - 1, len(_AUTH_BACKOFF) - 1)
                if _is_auth_error(exc):
                    if consecutive_failures <= 3:
                        logger.error(
                            "CandleSyncer %s/%s: OANDA authentication failed. "
                            "Check OANDA_ENV=%s (practice vs live). "
                            "Backing off %ds (failure #%d)",
                            pair, tf, OANDA_ENV, _AUTH_BACKOFF[backoff_idx],
                            consecutive_failures,
                        )
                    if consecutive_failures > 20:
                        logger.error(
                            "CandleSyncer %s/%s: authentication failed %d times — "
                            "permanently stopping sync for this pair/timeframe",
                            pair, tf, consecutive_failures,
                        )
                        break
                else:
                    logger.exception(
                        "CandleSyncer %s/%s: sync failed (attempt #%d)",
                        pair, tf, consecutive_failures,
                    )
                await asyncio.sleep(_AUTH_BACKOFF[backoff_idx])
                continue
            await asyncio.sleep(interval)

    async def _sync_pair_impl(self, pair: str, tf: str) -> int:
        if not self._api or not self._account_id:
            return 0

        db_pair = pair.upper().replace("_", "").replace("/", "")

        db_info = await asyncio.to_thread(self._store.get_pair, db_pair)
        if db_info and db_info.get("oanda_name"):
            oanda_instrument = db_info["oanda_name"]
        else:
            oanda_instrument = self._to_oanda_instrument(pair)

        last_ts = await asyncio.to_thread(self._get_latest_ts, db_pair, tf)

        if last_ts is None:
            lookback_bars = _COLD_START_LOOKBACK.get(tf, 360)
            granularity = _granularity(tf)
            bar_seconds = self._bar_seconds(tf)
            lookback = pd.Timestamp.utcnow() - pd.Timedelta(
                seconds=lookback_bars * bar_seconds,
            )
            last_ts = lookback.strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info(
                "CandleSyncer %s/%s: cold start — fetching %d bars (~%s)",
                pair, tf, lookback_bars, lookback.date(),
            )

        now = pd.Timestamp.utcnow()
        if _is_forex_weekend(now):
            if last_ts is not None:
                last_dt = pd.Timestamp(last_ts)
                if _is_forex_weekend(last_dt) or last_dt <= now:
                    return 0
            else:
                return 0

        all_rows: list[tuple] = []
        since = last_ts
        MAX_BATCH = 5000
        max_pages = 20

        for _page in range(max_pages):
            batch = await asyncio.to_thread(
                self._fetch_candles, oanda_instrument, tf, since,
            )
            if not batch:
                break

            for c in batch:
                mid = c.get("mid", {})
                bid = c.get("bid", {})
                ask = c.get("ask", {})
                try:
                    spread = round(
                        float(ask.get("c", 0) or 0) - float(bid.get("c", 0) or 0),
                        5,
                    )
                except (TypeError, ValueError):
                    spread = 0.0

                all_rows.append((
                    db_pair,
                    tf,
                    _normalize_ts(c["time"]),
                    float(mid.get("o", 0) or 0),
                    float(mid.get("h", 0) or 0),
                    float(mid.get("l", 0) or 0),
                    float(mid.get("c", 0) or 0),
                    float(bid.get("o", 0) or 0),
                    float(bid.get("c", 0) or 0),
                    float(ask.get("o", 0) or 0),
                    float(ask.get("c", 0) or 0),
                    float(spread),
                    int(c.get("volume", 0) or 0),
                ))

            last_candle_time = pd.Timestamp(batch[-1]["time"])
            since = (last_candle_time + pd.Timedelta(seconds=1)).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            )

            if len(all_rows) >= _SQLITE_CHUNK:
                await asyncio.to_thread(
                    self._store.insert_candles_batch, all_rows,
                )
                all_rows.clear()

            if len(batch) < MAX_BATCH:
                break

        if all_rows:
            await asyncio.to_thread(
                self._store.insert_candles_batch, all_rows,
            )
            if tf == "M30":
                _validate_m30_sequence(all_rows, db_pair)

        return len(all_rows)

    def _get_latest_ts(self, pair: str, tf: str) -> Optional[str]:
        df = self._store.get_latest_candles(pair, tf, 1)
        if df.empty:
            return None
        ts = pd.Timestamp(df.iloc[0]["time"])
        if pd.isna(ts):
            return None
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _to_oanda_instrument(pair: str) -> str:
        p = pair.upper()
        if len(p) != 6:
            return p
        return f"{p[:3]}_{p[3:]}"

    @staticmethod
    def _bar_seconds(tf: str) -> int:
        return {
            "M1": 60, "M5": 300, "M15": 900,
            "M30": 1800, "H1": 3600, "H4": 14400,
        }.get(tf, 60)

    def _fetch_candles(
        self,
        instrument: str,
        granularity: str,
        since: str,
    ) -> list[dict]:
        if self._api is None:
            return []

        params = {
            "from": since,
            "granularity": _granularity(granularity),
            "count": 5000,
            "price": "MBA",
            "includeFirst": True,
        }
        r = instruments.InstrumentsCandles(
            instrument=instrument, params=params,
        )
        self._api.request(r)
        if r.response is None:
            return []
        candles = r.response.get("candles")
        if not candles:
            return []
        return candles
