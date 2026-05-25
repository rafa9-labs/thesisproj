"""Shared fixtures for pipeline integrity tests."""
import gc
import os
import sys
import pytest

import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

os.environ.setdefault("SKLEARN_JOBS", "1")


@pytest.fixture(scope="session")
def pipeline_imports():
    """Import pipeline._imports once for the whole test session."""
    import pipeline._imports
    return pipeline._imports


@pytest.fixture(scope="session")
def ml_backtester_class():
    """Return the composed MLBacktester class."""
    from pipeline.backtester.composed import MLBacktester
    return MLBacktester


@pytest.fixture(scope="session")
def numpy_arr():
    """Small numpy array for metric tests."""
    import numpy as np
    return np.array


@pytest.fixture(autouse=True)
def _cleanup_between_tests():
    """Force garbage collection and clear TF session between tests.

    Prevents memory accumulation from NumPy arrays, TF sessions, and
    joblib worker pools that persist across test boundaries.
    """
    yield
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass
    except Exception:
        pass


@pytest.fixture(scope="session")
def _restrict_threading():
    """Limit thread pools across the entire test session.

    Sets OMP_NUM_THREADS and JOBLIB_NUM_CPUS to prevent joblib/mkl
    from spawning excessive worker threads in low-memory environments.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("JOBLIB_START_METHOD", "loky")
    yield


# ═════════════════════════════════════════════════════════════════════════
# Placeholder DB — synthetic EURUSD M30/H1/H4 for testing without OANDA
# ═════════════════════════════════════════════════════════════════════════

def _make_placeholder_ohlc(start: str = "2020-01-01", end: str = "2023-03-03"):
    """Generate a realistic random-walk EURUSD OHLC DataFrame.

    Columns match the DB schema: time, mid_o, mid_h, mid_l, mid_c,
    bid_o, bid_c, ask_o, ask_c, spread, volume.
    """
    dates = pd.date_range(start, end, freq="30min", tz="UTC")
    n = len(dates)
    mid_c = 1.1200 + np.cumsum(np.random.RandomState(42).randn(n) * 0.0004)
    mid_c = np.clip(mid_c, 1.05, 1.25)
    high_wiggle = np.abs(np.random.RandomState(43).randn(n)) * 0.0003
    low_wiggle = np.abs(np.random.RandomState(44).randn(n)) * 0.0003
    spread_arr = np.full(n, 0.00015)
    df = pd.DataFrame({
        "time": dates,
        "mid_o": np.roll(mid_c, 1),
        "mid_h": mid_c + high_wiggle,
        "mid_l": mid_c - low_wiggle,
        "mid_c": mid_c,
        "bid_o": np.roll(mid_c, 1) - spread_arr / 2,
        "bid_c": mid_c - spread_arr / 2,
        "ask_o": np.roll(mid_c, 1) + spread_arr / 2,
        "ask_c": mid_c + spread_arr / 2,
        "spread": spread_arr,
        "volume": np.random.RandomState(45).randint(1, 500, n),
    })
    df.loc[0, ["mid_o", "bid_o", "ask_o"]] = df.loc[0, ["mid_c", "bid_c", "ask_c"]].values
    return df


def _seed_placeholder_db_impl(db_path: str):
    """Create a SQLite DB with synthetic EURUSD M30/H1/H4 data."""
    from pipeline.data_sqlite import DataStore

    store = DataStore(db_path)

    store.insert_pairs([{
        "symbol": "EURUSD",
        "oanda_name": "EUR_USD",
        "pip_value": 0.0001,
        "lot_size": 100000,
        "base_currency": "EUR",
        "quote_currency": "USD",
        "typical_spread_bps": 1.5,
    }])

    df_m30 = _make_placeholder_ohlc()

    for timeframe, df in [("M30", df_m30), ("H1", df_m30.resample("1h", on="time").agg({
        "time": "first", "mid_o": "first", "mid_h": "max", "mid_l": "min",
        "mid_c": "last", "bid_o": "first", "bid_c": "last",
        "ask_o": "first", "ask_c": "last", "spread": "mean", "volume": "sum",
    }).reset_index(drop=True)), ("H4", df_m30.resample("4h", on="time").agg({
        "time": "first", "mid_o": "first", "mid_h": "max", "mid_l": "min",
        "mid_c": "last", "bid_o": "first", "bid_c": "last",
        "ask_o": "first", "ask_c": "last", "spread": "mean", "volume": "sum",
    }).reset_index(drop=True))]:
        rows = []
        for _, r in df.iterrows():
            rows.append((
                "EURUSD", timeframe, str(r["time"]),
                float(r["mid_o"]), float(r["mid_h"]), float(r["mid_l"]), float(r["mid_c"]),
                float(r["bid_o"]), float(r["bid_c"]),
                float(r["ask_o"]), float(r["ask_c"]),
                float(r["spread"]), int(r["volume"]),
            ))
        store.insert_candles_batch(rows)

    return db_path


@pytest.fixture(scope="session")
def seed_placeholder_db(tmp_path_factory):
    """Session-scoped synthetic EURUSD DB (M30/H1/H4, 2020-01-01 -> 2023-03-03).

    Returns the path to a temporary SQLite file ready for use by MLBacktester.
    """
    db_path = str(tmp_path_factory.mktemp("forex_test") / "forex.db")
    return _seed_placeholder_db_impl(db_path)