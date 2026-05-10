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
from pipeline.pair_config import get_pair_config

router = APIRouter(prefix="/pairs", tags=["pairs"])


def _build_pair_info(symbol: str, store) -> PairInfo:
    """Build PairInfo from registry or from DB fallback."""
    try:
        cfg = get_pair_config(symbol)
        return PairInfo(
            symbol=cfg.symbol,
            oanda_name=cfg.oanda_name,
            pip_value=cfg.pip_value,
            lot_size=cfg.lot_size,
            base_currency=cfg.base_currency,
            quote_currency=cfg.quote_currency,
            typical_spread_bps=cfg.typical_spread_bps,
        )
    except ValueError:
        db_pair = store.get_pair(symbol)
        if db_pair:
            return PairInfo(
                symbol=db_pair["symbol"],
                oanda_name=db_pair.get("oanda_name", symbol[:3] + "_" + symbol[3:]),
                pip_value=db_pair.get("pip_value", 0.0001),
                lot_size=db_pair.get("lot_size", 100_000.0),
                base_currency=db_pair.get("base_currency", symbol[:3]),
                quote_currency=db_pair.get("quote_currency", symbol[3:]),
                typical_spread_bps=db_pair.get("typical_spread_bps", 1.0),
            )
        return PairInfo(
            symbol=symbol,
            oanda_name=symbol[:3] + "_" + symbol[3:],
            pip_value=0.0001,
            lot_size=100_000.0,
            base_currency=symbol[:3],
            quote_currency=symbol[3:],
            typical_spread_bps=1.0,
        )


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

    seen = set()
    pairs = []
    for symbol in pair_map.keys():
        if symbol in seen:
            continue
        seen.add(symbol)
        info = _build_pair_info(symbol, store)
        pairs.append(PairDetail(
            pair=info,
            timeframes=pair_map.get(symbol, []),
        ))

    return PairListResponse(pairs=pairs)


@router.get("/{symbol}/data-range", response_model=DataRangeResponse)
def get_data_range(symbol: str):
    symbol = symbol.upper()
    store = get_data_store()

    tfs = store.list_timeframes(symbol)
    if not tfs:
        # Allow unknown pairs if they have candles in the DB
        raise HTTPException(404, f"No data found for pair: {symbol}")

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
