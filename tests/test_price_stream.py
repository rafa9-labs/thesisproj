"""Tests for Phase 1+2: price stream, clock-aligned bars, candles pagination."""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.api


# ═════════════════════════════════════════════════════════════════════
#  clock_bar_start
# ═════════════════════════════════════════════════════════════════════

class TestClockBarStart:
    """Clock-aligned bar boundary math."""

    @pytest.mark.parametrize("ts_minute,expected_minute", [
        (0, 0), (1, 0), (14, 0), (29, 0), (30, 30),
        (31, 30), (44, 30), (59, 30),
    ])
    def test_m30_boundaries(self, ts_minute, expected_minute):
        from api.routers.price_stream import clock_bar_start
        ts = datetime(2026, 6, 22, 10, ts_minute, 0, tzinfo=timezone.utc)
        result = clock_bar_start(ts, "M30")
        assert result.minute == expected_minute
        assert result.second == 0
        assert result.microsecond == 0
        assert result.hour == 10

    @pytest.mark.parametrize("ts_minute,expected_hour,expected_min", [
        (0, 10, 0), (30, 10, 0), (59, 10, 0),
    ])
    def test_h1_boundaries(self, ts_minute, expected_hour, expected_min):
        from api.routers.price_stream import clock_bar_start
        ts = datetime(2026, 6, 22, 10, ts_minute, 0, tzinfo=timezone.utc)
        result = clock_bar_start(ts, "H1")
        assert result.hour == expected_hour
        assert result.minute == expected_min
        assert result.second == 0

    @pytest.mark.parametrize("ts_hour,expected_hour", [
        (0, 0), (1, 0), (3, 0),
        (4, 4), (5, 4), (7, 4),
        (8, 8), (11, 8),
        (12, 12), (15, 12),
        (16, 16), (19, 16),
        (20, 20), (23, 20),
    ])
    def test_h4_boundaries(self, ts_hour, expected_hour):
        from api.routers.price_stream import clock_bar_start
        ts = datetime(2026, 6, 22, ts_hour, 0, 0, tzinfo=timezone.utc)
        result = clock_bar_start(ts, "H4")
        assert result.hour == expected_hour
        assert result.minute == 0
        assert result.second == 0

    def test_crosses_boundary_m30(self):
        """Tick at 10:30:01 means previous bar 10:00-10:30 closed."""
        from api.routers.price_stream import clock_bar_start
        old = clock_bar_start(datetime(2026, 6, 22, 10, 29, 59, tzinfo=timezone.utc), "M30")
        new = clock_bar_start(datetime(2026, 6, 22, 10, 30, 0, tzinfo=timezone.utc), "M30")
        assert old != new
        assert old.minute == 0
        assert new.minute == 30

    def test_fallback_tf(self):
        from api.routers.price_stream import clock_bar_start
        ts = datetime(2026, 6, 22, 10, 14, 0, tzinfo=timezone.utc)
        result = clock_bar_start(ts, "M15")
        assert result.second == 0
        assert result.minute in (0, 15)


# ═════════════════════════════════════════════════════════════════════
#  TickBuffer
# ═════════════════════════════════════════════════════════════════════

class TestTickBuffer:
    """Tick accumulation and OHLC computation."""

    def test_empty_snapshot_returns_empty_dict(self):
        from api.routers.price_stream import TickBuffer
        buf = TickBuffer("EURUSD", "M30", datetime(2026, 6, 22, 10, 0, 0, tzinfo=timezone.utc))
        assert buf.snapshot() == {}
        assert buf.tick_count == 0

    def test_single_tick_ohlc_all_same(self):
        from api.routers.price_stream import TickBuffer
        ts = datetime(2026, 6, 22, 10, 0, 1, tzinfo=timezone.utc)
        buf = TickBuffer("EURUSD", "M30", ts)
        buf.append(ts, 1.12345)
        snap = buf.snapshot()
        assert snap["open"] == snap["high"] == snap["low"] == snap["close"] == 1.12345
        assert snap["time"] == int(ts.timestamp())

    def test_multi_tick_tracks_ohlc(self):
        from api.routers.price_stream import TickBuffer
        bar_start = datetime(2026, 6, 22, 10, 0, 0, tzinfo=timezone.utc)
        buf = TickBuffer("EURUSD", "M30", bar_start)

        buf.append(datetime(2026, 6, 22, 10, 1, 0, tzinfo=timezone.utc), 1.12000)
        buf.append(datetime(2026, 6, 22, 10, 2, 0, tzinfo=timezone.utc), 1.12100)
        buf.append(datetime(2026, 6, 22, 10, 3, 0, tzinfo=timezone.utc), 1.11900)
        buf.append(datetime(2026, 6, 22, 10, 4, 0, tzinfo=timezone.utc), 1.11950)

        snap = buf.snapshot()
        assert snap["open"] == 1.12000
        assert snap["high"] == 1.12100
        assert snap["low"] == 1.11900
        assert snap["close"] == 1.11950
        assert buf.tick_count == 4

    def test_finalize_resets_buffer(self):
        from api.routers.price_stream import TickBuffer
        bar_start = datetime(2026, 6, 22, 10, 0, 0, tzinfo=timezone.utc)
        buf = TickBuffer("EURUSD", "M30", bar_start)
        buf.append(datetime(2026, 6, 22, 10, 1, 0, tzinfo=timezone.utc), 1.12000)
        buf.append(datetime(2026, 6, 22, 10, 2, 0, tzinfo=timezone.utc), 1.12100)

        ohlc = buf.finalize()
        assert ohlc["open"] == 1.12000
        assert ohlc["close"] == 1.12100
        assert buf.tick_count == 0
        assert buf.snapshot() == {}

    def test_finalize_then_new_ticks(self):
        """After finalize, new ticks start a fresh OHLC."""
        from api.routers.price_stream import TickBuffer
        bar_start = datetime(2026, 6, 22, 10, 0, 0, tzinfo=timezone.utc)
        buf = TickBuffer("EURUSD", "M30", bar_start)
        buf.append(datetime(2026, 6, 22, 10, 1, 0, tzinfo=timezone.utc), 1.12000)
        buf.finalize()

        buf.append(datetime(2026, 6, 22, 10, 5, 0, tzinfo=timezone.utc), 1.13000)
        snap = buf.snapshot()
        assert snap["open"] == 1.13000
        assert snap["high"] == 1.13000
        assert snap["low"] == 1.13000
        assert snap["close"] == 1.13000


# ═════════════════════════════════════════════════════════════════════
#  DataStore.upsert_candle
# ═════════════════════════════════════════════════════════════════════

class TestUpsertCandle:
    """Single-candle DB write."""

    @pytest.fixture
    def store(self, tmp_path):
        from pipeline.data.data_sqlite import DataStore
        db_path = str(tmp_path / "test_upsert.db")
        return DataStore(db_path)

    def test_upsert_inserts_new_candle(self, store):
        ts = "2026-06-22T10:00:00+00:00"
        store.upsert_candle("EURUSD", "M30", ts, 1.1000, 1.1100, 1.0950, 1.1050)

        df = store.get_latest_candles("EURUSD", "M30", 10)
        assert len(df) == 1
        assert float(df.iloc[0]["mid_open"]) == pytest.approx(1.1000, rel=1e-5)
        assert float(df.iloc[0]["mid_high"]) == pytest.approx(1.1100, rel=1e-5)
        assert float(df.iloc[0]["mid_low"]) == pytest.approx(1.0950, rel=1e-5)
        assert float(df.iloc[0]["mid_close"]) == pytest.approx(1.1050, rel=1e-5)

    def test_upsert_overwrites_existing_candle(self, store):
        ts = "2026-06-22T10:00:00+00:00"
        store.upsert_candle("EURUSD", "M30", ts, 1.1000, 1.1100, 1.0950, 1.1050)
        store.upsert_candle("EURUSD", "M30", ts, 1.2000, 1.2200, 1.1900, 1.2100)

        df = store.get_latest_candles("EURUSD", "M30", 10)
        assert len(df) == 1
        assert float(df.iloc[0]["mid_open"]) == pytest.approx(1.2000, rel=1e-5)

    def test_upsert_multiple_candles(self, store):
        store.upsert_candle("EURUSD", "M30", "2026-06-22T10:00:00+00:00",
                            1.1000, 1.1010, 1.1000, 1.1010)
        store.upsert_candle("EURUSD", "M30", "2026-06-22T10:30:00+00:00",
                            1.1010, 1.1020, 1.1010, 1.1020)
        store.upsert_candle("EURUSD", "M30", "2026-06-22T11:00:00+00:00",
                            1.1020, 1.1030, 1.1020, 1.1030)

        df = store.get_latest_candles("EURUSD", "M30", 10)
        assert len(df) == 3
        assert float(df.iloc[0]["mid_open"]) == pytest.approx(1.1000, rel=1e-5)
        assert float(df.iloc[2]["mid_open"]) == pytest.approx(1.1020, rel=1e-5)

    def test_upsert_different_pairs_independent(self, store):
        store.upsert_candle("EURUSD", "M30", "2026-06-22T10:00:00+00:00",
                            1.1000, 1.1100, 1.0950, 1.1050)
        store.upsert_candle("GBPUSD", "M30", "2026-06-22T10:00:00+00:00",
                            1.3000, 1.3100, 1.2950, 1.3050)

        eur = store.get_latest_candles("EURUSD", "M30", 10)
        gbp = store.get_latest_candles("GBPUSD", "M30", 10)
        assert len(eur) == 1
        assert len(gbp) == 1
        assert float(eur.iloc[0]["mid_open"]) == pytest.approx(1.1000, rel=1e-5)
        assert float(gbp.iloc[0]["mid_open"]) == pytest.approx(1.3000, rel=1e-5)


# ═════════════════════════════════════════════════════════════════════
#  Candles API pagination (start / end)
# ═════════════════════════════════════════════════════════════════════

class TestCandlesPagination:
    """GET /api/v1/candles/{pair}/{timeframe} with start/end params."""

    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_no_params_returns_latest(self, client):
        r = client.get("/api/v1/candles/EURUSD/M30?limit=5")
        assert r.status_code == 200
        data = r.json()
        assert data["pair"] == "EURUSD"
        assert data["timeframe"] == "M30"
        assert len(data["candles"]) == 5

    def test_start_param_returns_range(self, client):
        start_epoch = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp())
        r = client.get(f"/api/v1/candles/EURUSD/M30?start={start_epoch}")
        assert r.status_code == 200
        data = r.json()
        assert len(data["candles"]) > 0
        candle_times = [c["t"] for c in data["candles"]]
        assert all(t >= start_epoch for t in candle_times)

    def test_start_end_both_params(self, client):
        start_epoch = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp())
        end_epoch = int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp())
        r = client.get(
            f"/api/v1/candles/EURUSD/M30?start={start_epoch}&end={end_epoch}"
        )
        assert r.status_code == 200
        data = r.json()
        candle_times = [c["t"] for c in data["candles"]]
        assert all(t >= start_epoch and t <= end_epoch for t in candle_times)

    def test_candles_ascending_order(self, client):
        start_epoch = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp())
        end_epoch = int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp())
        r = client.get(
            f"/api/v1/candles/EURUSD/M30?start={start_epoch}&end={end_epoch}"
        )
        assert r.status_code == 200
        data = r.json()
        if len(data["candles"]) > 1:
            times = [c["t"] for c in data["candles"]]
            assert times == sorted(times), "Candles must be ascending"

    def test_candle_shape(self, client):
        r = client.get("/api/v1/candles/EURUSD/H1?limit=1")
        assert r.status_code == 200
        data = r.json()
        candle = data["candles"][0]
        for key in ("t", "o", "h", "l", "c"):
            assert key in candle, f"Missing key: {key}"
            assert isinstance(candle[key], (int, float)), f"{key} not numeric"

    def test_invalid_pair_404(self, client):
        r = client.get("/api/v1/candles/ZZZZZZ/M30")
        assert r.status_code == 404

    def test_invalid_timeframe_400(self, client):
        r = client.get("/api/v1/candles/EURUSD/W1")
        assert r.status_code == 400

    def test_limit_respected(self, client):
        r = client.get("/api/v1/candles/EURUSD/M30?limit=3")
        assert r.status_code == 200
        assert len(r.json()["candles"]) == 3


# ═════════════════════════════════════════════════════════════════════
#  PriceStreamManager singleton
# ═════════════════════════════════════════════════════════════════════

class TestPriceStreamManager:
    """Singleton lifecycle and ref-counting."""

    def test_singleton_returns_same_instance(self):
        from api.routers.price_stream import PriceStreamManager
        a = PriceStreamManager.get()
        b = PriceStreamManager.get()
        assert a is b

    def test_get_new_bar_event_returns_event(self):
        from api.routers.price_stream import PriceStreamManager
        psm = PriceStreamManager.get()
        event = psm.get_new_bar_event("NOSYMBOL", "M30")
        assert event is None  # no stream running for this symbol

    def test_subscribe_unsubscribe_noop_without_stream(self):
        from api.routers.price_stream import PriceStreamManager
        import asyncio
        psm = PriceStreamManager.get()
        q = asyncio.Queue()
        psm.subscribe("NOSYMBOL", q)
        psm.unsubscribe("NOSYMBOL", q)
        # Should not raise
        assert q.empty()

    def test_shutdown_idempotent(self):
        import asyncio
        from api.routers.price_stream import PriceStreamManager
        psm = PriceStreamManager.get()

        async def _run():
            await psm.shutdown()
            await psm.shutdown()  # Second shutdown is a no-op

        asyncio.run(_run())
