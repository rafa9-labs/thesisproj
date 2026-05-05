"""Merge news-derived features into OHLC DataFrames.

Provides ``merge_news_features()`` which performs a walk-forward-safe
left-join of aggregated sentiment + event flags onto price bars.

Walk-forward guarantee: only news articles with ``timestamp < bar_time``
are used for each row -- no future information leaks in.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "sentiment_score",
    "sentiment_magnitude",
    "news_volume_1h",
    "news_volume_24h",
]


def merge_news_features(
    df: pd.DataFrame,
    news_df: pd.DataFrame,
    events: List[Dict] | None = None,
    config: Dict | None = None,
    event_proximity_bars: int = 6,
) -> pd.DataFrame:
    """Merge news sentiment and event flags into an OHLC DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with DatetimeIndex.
    news_df : pd.DataFrame
        Output of ``SentimentAnalyzer.aggregate_to_df()``.
        Must have ``timestamp`` column or DatetimeIndex.
    events : list[dict] or None
        Economic calendar events. Each dict: ``{"date": datetime, "event": str, "impact": int}``.
    config : dict or None
        Feature toggles:
        - ``use_news`` (bool): enable news features (default True).
        - ``news_event_flags`` (bool): add event proximity flags (default True).
        - ``news_volume_windows`` (list[int]): rolling volume windows in bars (default [6, 24]).
    event_proximity_bars : int
        Number of bars before/after an event to set the flag (default 6).

    Returns
    -------
    pd.DataFrame
        Input df with added news feature columns.
    """
    cfg = config or {}
    if not cfg.get("use_news", True):
        return df

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            logger.warning("Cannot convert df index to DatetimeIndex -- skipping news features")
            return df

    # -- Sentiment merge ----------------------------------------------
    if news_df is not None and not news_df.empty:
        news_copy = news_df.copy()
        if "timestamp" in news_copy.columns:
            news_copy["timestamp"] = pd.to_datetime(news_copy["timestamp"], utc=True)
            news_copy = news_copy.set_index("timestamp")

        if not isinstance(news_copy.index, pd.DatetimeIndex):
            news_copy.index = pd.to_datetime(news_copy.index, utc=True)

        # Normalize timezone compatibility between news and OHLC indices
        ohlc_tz = df.index.tz
        news_tz = news_copy.index.tz
        if ohlc_tz is None and news_tz is not None:
            news_copy.index = news_copy.index.tz_convert("UTC").tz_localize(None)
        elif ohlc_tz is not None and news_tz is None:
            news_copy.index = news_copy.index.tz_localize(ohlc_tz)
        elif ohlc_tz is not None and news_tz is not None and ohlc_tz != news_tz:
            news_copy.index = news_copy.index.tz_convert(ohlc_tz)

        # Reindex to OHLC frequency (forward-fill, backward-looking)
        news_aligned = news_copy.reindex(
            df.index,
            method="ffill",
            limit=None,
        )

        df["sentiment_score"] = news_aligned.get("sentiment_score", np.nan).astype(np.float32)
        df["sentiment_magnitude"] = news_aligned.get("sentiment_magnitude", np.nan).astype(np.float32)

        # Rolling news volume windows
        vol_windows = cfg.get("news_volume_windows", [6, 24])
        raw_vol = news_aligned.get("news_volume", 0).fillna(0).astype(np.float32)
        for w in vol_windows:
            col = f"news_volume_{w}bars"
            df[col] = raw_vol.rolling(window=w, min_periods=1).sum().astype(np.float32)
    else:
        df["sentiment_score"] = np.float32(0.0)
        df["sentiment_magnitude"] = np.float32(0.0)
        vol_windows = cfg.get("news_volume_windows", [6, 24])
        for w in vol_windows:
            df[f"news_volume_{w}bars"] = np.float32(0.0)

    # -- Event flags --------------------------------------------------
    if cfg.get("news_event_flags", True) and events:
        df = _add_event_flags(df, events, event_proximity_bars)

    return df


def _add_event_flags(
    df: pd.DataFrame,
    events: List[Dict],
    proximity_bars: int = 6,
) -> pd.DataFrame:
    """Add binary event proximity flags for each economic event type.

    For each event, a column ``event_flag_<name>`` is set to 1.0 for
    ``proximity_bars`` bars before and after the event datetime.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    df_tz = df.index.tz

    freq = pd.infer_freq(df.index)
    if freq is None:
        try:
            median_delta = pd.Series(df.index).diff().median()
            bar_td = median_delta if pd.notna(median_delta) else pd.Timedelta(hours=1)
        except Exception:
            bar_td = pd.Timedelta(hours=1)
    else:
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
            bar_td = pd.Timedelta(offset.nanos / 1e9, unit="s") if hasattr(offset, "nanos") else pd.Timedelta(hours=1)
        except Exception:
            bar_td = pd.Timedelta(hours=1)

    proximity_td = bar_td * proximity_bars

    event_types = set()
    for ev in events:
        event_types.add(ev.get("event", "UNKNOWN"))

    for etype in sorted(event_types):
        col = f"event_flag_{etype.lower()}"
        flags = np.zeros(len(df), dtype=np.float32)

        for ev in events:
            if ev.get("event") != etype:
                continue
            ev_date = ev.get("date")
            if ev_date is None:
                continue
            if isinstance(ev_date, str):
                try:
                    ev_date = pd.Timestamp(ev_date)
                except Exception:
                    continue
            elif isinstance(ev_date, datetime):
                ev_date = pd.Timestamp(ev_date)

            if df_tz is not None and ev_date.tz is None:
                ev_date = ev_date.tz_localize(df_tz)
            elif df_tz is None and ev_date.tz is not None:
                ev_date = ev_date.tz_convert("UTC").tz_localize(None)

            start = ev_date - proximity_td
            end = ev_date + proximity_td
            mask = (df.index >= start) & (df.index <= end)
            flags[mask] = 1.0

        df[col] = flags

    return df


def get_news_feature_columns(config: Dict | None = None) -> List[str]:
    """Return the list of news feature column names that will be added.

    Useful for adding to the features list in ``prepare_features()``.
    """
    cfg = config or {}
    cols = ["sentiment_score", "sentiment_magnitude"]
    vol_windows = cfg.get("news_volume_windows", [6, 24])
    for w in vol_windows:
        cols.append(f"news_volume_{w}bars")
    return cols


LLM_FEATURE_COLUMNS = [
    "llm_sentiment",
    "llm_confidence",
    "llm_volatility",
]


def get_llm_feature_columns(config: Dict | None = None) -> List[str]:
    """Return the list of LLM feature column names that will be added.

    Includes rolling mean columns if configured.
    """
    cfg = config or {}
    cols = list(LLM_FEATURE_COLUMNS)
    ma_windows = cfg.get("llm_sentiment_ma_windows", [6, 24])
    for w in ma_windows:
        cols.append(f"llm_sentiment_ma_{w}")
    cols.append("blended_sentiment")
    return cols


def merge_llm_features(
    df: pd.DataFrame,
    llm_df: pd.DataFrame,
    config: Dict | None = None,
    llm_weight: float = 0.7,
) -> pd.DataFrame:
    """Merge LLM sentiment features into an OHLC DataFrame.

    Walk-forward safe: forward-fill only, no future data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with DatetimeIndex. Must already have VADER
        ``sentiment_score`` column (from ``merge_news_features``).
    llm_df : pd.DataFrame
        Output of ``LLMSentimentEngine.aggregate_to_df()``.
        Columns: timestamp, llm_sentiment, llm_confidence, llm_volatility, news_volume.
    config : dict or None
        Feature toggles:
        - ``llm_sentiment_enabled`` (bool): enable LLM features (default True).
        - ``llm_sentiment_ma_windows`` (list[int]): rolling MA windows (default [6, 24]).
        - ``llm_weight`` (float): blending weight for LLM vs VADER (default 0.7).
    llm_weight : float
        Backward-compat weight override (0.7 means 70% LLM, 30% VADER).

    Returns
    -------
    pd.DataFrame
        Input df with added LLM feature columns + blended_sentiment.
    """
    cfg = config or {}
    if not cfg.get("llm_sentiment_enabled", True):
        return df

    df = df.copy()
    w = float(cfg.get("llm_weight", llm_weight))

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            logger.warning("Cannot convert df index to DatetimeIndex -- skipping LLM features")
            return df

    if llm_df is not None and not llm_df.empty:
        llm_copy = llm_df.copy()
        if "timestamp" in llm_copy.columns:
            llm_copy["timestamp"] = pd.to_datetime(llm_copy["timestamp"], utc=True)
            llm_copy = llm_copy.set_index("timestamp")

        if not isinstance(llm_copy.index, pd.DatetimeIndex):
            llm_copy.index = pd.to_datetime(llm_copy.index, utc=True)

        ohlc_tz = df.index.tz
        llm_tz = llm_copy.index.tz
        if ohlc_tz is None and llm_tz is not None:
            llm_copy.index = llm_copy.index.tz_convert("UTC").tz_localize(None)
        elif ohlc_tz is not None and llm_tz is None:
            llm_copy.index = llm_copy.index.tz_localize(ohlc_tz)
        elif ohlc_tz is not None and llm_tz is not None and ohlc_tz != llm_tz:
            llm_copy.index = llm_copy.index.tz_convert(ohlc_tz)

        llm_aligned = llm_copy.reindex(df.index, method="ffill", limit=None)

        df["llm_sentiment"] = llm_aligned.get("llm_sentiment", np.float32(0)).astype(np.float32).fillna(0)
        df["llm_confidence"] = llm_aligned.get("llm_confidence", np.float32(0)).astype(np.float32).fillna(0)
        df["llm_volatility"] = llm_aligned.get("llm_volatility", np.float32(0)).astype(np.float32).fillna(0)

        ma_windows = cfg.get("llm_sentiment_ma_windows", [6, 24])
        for ma_w in ma_windows:
            col_name = f"llm_sentiment_ma_{ma_w}"
            df[col_name] = df["llm_sentiment"].rolling(window=ma_w, min_periods=1).mean().astype(np.float32)
    else:
        df["llm_sentiment"] = np.float32(0.0)
        df["llm_confidence"] = np.float32(0.0)
        df["llm_volatility"] = np.float32(0.0)
        ma_windows = cfg.get("llm_sentiment_ma_windows", [6, 24])
        for ma_w in ma_windows:
            df[f"llm_sentiment_ma_{ma_w}"] = np.float32(0.0)

    # Blended sentiment: llm_weight * llm + (1 - llm_weight) * vader
    vader_score = df.get("sentiment_score", pd.Series(0.0, index=df.index)).astype(np.float32)
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = vader_score
    df["blended_sentiment"] = (w * df["llm_sentiment"] + (1 - w) * vader_score).astype(np.float32)

    return df
