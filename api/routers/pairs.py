"""Pair registry and data endpoints."""
from typing import List

from fastapi import APIRouter, HTTPException

from api.dependencies import get_data_store
from api.schemas.pairs import (
    DataRangeResponse,
    PairDetail,
    PairInfo,
    PairListResponse,
    TimeframeData,
)
from pipeline.pair_config import VALID_PAIRS, get_pair_config

router = APIRouter(prefix="/pairs", tags=["pairs"])


@router.get("", response_model=PairListResponse)
def list_pairs():
    store = get_data_store()
    summary = store.get_pair_summary()

    pair_map: dict = {}
    for s in summary:
        p = s["pair"]
        if p not in pair_map:
            pair_map[p] = []
        pair_map[p].append(TimeframeData(
            timeframe=s["timeframe"],
            rows=s["rows"],
            start_date=s.get("start_date"),
            end_date=s.get("end_date"),
        ))

    pairs = []
    for symbol in VALID_PAIRS:
        try:
            cfg = get_pair_config(symbol)
        except ValueError:
            continue
        db_pair = store.get_pair(symbol)
        info = PairInfo(
            symbol=cfg.symbol,
            oanda_name=cfg.oanda_name,
            pip_value=cfg.pip_value,
            lot_size=cfg.lot_size,
            base_currency=cfg.base_currency,
            quote_currency=cfg.quote_currency,
            typical_spread_bps=cfg.typical_spread_bps,
        )
        pairs.append(PairDetail(
            pair=info,
            timeframes=pair_map.get(symbol, []),
        ))

    return PairListResponse(pairs=pairs)


@router.get("/{symbol}/data-range", response_model=DataRangeResponse)
def get_data_range(symbol: str):
    symbol = symbol.upper()
    store = get_data_store()

    try:
        get_pair_config(symbol)
    except ValueError:
        raise HTTPException(404, f"Unknown pair: {symbol}")

    tfs = store.list_timeframes(symbol)
    timeframe_data = []
    for tf in tfs:
        rng = store.get_date_range(symbol, tf)
        count = store.get_candle_count(symbol, tf)
        timeframe_data.append(TimeframeData(
            timeframe=tf,
            rows=count,
            start_date=rng[0] if rng else None,
            end_date=rng[1] if rng else None,
        ))

    return DataRangeResponse(symbol=symbol, timeframes=timeframe_data)
