"""Paper + Live trading API endpoints — async-safe, lock-guarded, kill-signalled.

POST /trading/paper/start        | WS /trading/paper/{id}/ws | POST /{id}/stop
GET  /trading/paper/{id}/status  | GET /{id}/trades           | GET /{id}/summary
GET  /trading/paper/sessions

POST /trading/live/start         | WS /trading/live/{id}/ws   | POST /{id}/stop
POST /trading/live/{id}/emergency | GET /{id}/status          | GET /{id}/journal
GET  /trading/live/{id}/risk      | GET /trading/live/sessions
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from api.dependencies import get_data_store
from api.routers.prices import _get_oanda_credentials, _oanda_api_call

logger = logging.getLogger(__name__)

# ── Routers ──────────────────────────────────────────────────────────
router = APIRouter(prefix="/trading/paper", tags=["trading-paper"])
live_router = APIRouter(prefix="/trading/live", tags=["trading-live"])

# ── Session stores (isolated per router) ────────────────────────────
active_sessions: Dict[str, dict] = {}
live_sessions: Dict[str, dict] = {}


# ═════════════════════════════════════════════════════════════════════
#  Pydantic models
# ═════════════════════════════════════════════════════════════════════

class DeployPaperRequest(BaseModel):
    pair: str
    model_id: str | None = None
    model_type: str = "logistic"
    timeframe: str = "M30"
    initial_equity: float = 10000.0
    position_sizing: str = "fixed"
    sizing_config: dict = {}


class PaperSessionInfo(BaseModel):
    session_id: str
    pair: str
    model_type: str
    timeframe: str
    status: str
    equity: float
    position: str
    unrealized_pnl: float
    total_trades: int
    signal_count: int
    created_at: str


class DeployLiveRequest(BaseModel):
    pair: str
    model_id: str | None = None
    model_type: str = "logistic"
    timeframe: str = "M30"
    initial_equity: float = 10000.0
    position_sizing: str = "fixed"
    sizing_config: dict = {}
    mode: str = "demo"
    risk_config: dict = {}


class LiveSessionInfo(BaseModel):
    session_id: str
    pair: str
    model_type: str
    timeframe: str
    mode: str
    status: str
    equity: float
    position: str
    unrealized_pnl: float
    signal_count: int
    killed: bool
    kill_reason: str


class DeployCommitteeRequest(BaseModel):
    pair: str
    timeframe: str = "H1"
    initial_equity: float = 10000.0
    full_cycle_job_id: str | None = None
    committee_config_path: str | None = None
    confidence_threshold: float = 0.55
    lookback_bars: int = 100
    position_sizing: str = "fixed"
    sizing_config: dict = {}
    risk_config: dict = {}
    mode: str = "paper"
    live_news_blend_enabled: bool = False
    live_news_blend_weight: float = 0.10


# ═════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═════════════════════════════════════════════════════════════════════

def _load_model_for_paper(model_id: str | None, model_type: str, pair: str, timeframe: str):
    model_obj = None
    bt = None

    if model_id:
        from pipeline.models.model_persistence import load_model_only, read_metadata
        from pipeline.models.model_registry_disk import get_all_deployed
        from api.config import settings

        rows = get_all_deployed(settings.db_full_path)
        snapshot_path = None
        for r in rows:
            if r.get("id") == model_id:
                snapshot_path = r.get("snapshot_path")
                break
        if not snapshot_path or not os.path.exists(snapshot_path):
            raise HTTPException(404, f"Snapshot not found for model {model_id}")
        try:
            model_obj = load_model_only(snapshot_path)
            meta = read_metadata(snapshot_path)
            actual_model_type = meta.get("model_type", model_type)
        except Exception:
            raise HTTPException(500, "Failed to load saved model")
    else:
        result = _run_backtest_for_model(pair, model_type, timeframe)
        if result:
            model_obj = result.get("model")
            bt = result.get("backtester")
            actual_model_type = result.get("config", {}).get("model_type", model_type)
        else:
            raise HTTPException(500, "Failed to train model")

    if model_obj is None:
        raise HTTPException(500, "Failed to obtain model")
    return model_obj, bt, actual_model_type


def _run_backtest_for_model(pair: str, model: str, timeframe: str):
    try:
        from pipeline.backtester.composed import MLBacktester
        from config import PIPELINE_CONSTANTS as _PC

        cfg = {
            "pair": pair, "timeframe": timeframe, "model_type": model,
            "n_months": 3, "use_WFO": False, "use_proba": True,
            "features_config": _PC.get("features_config", {}),
            "search_space": _PC.get("search_space", {}),
        }
        bt_instance = MLBacktester(config=cfg)
        bt_instance.load_data()
        result = bt_instance.run_strategy(config=cfg, models_to_test=[model], n_trials=1, n_startup_trials=1)
        if result is None or result[0] is None or result[0].empty:
            return None
        return {"model": getattr(bt_instance, "model", None),
                "config": getattr(bt_instance, "config", cfg),
                "backtester": bt_instance}
    except Exception:
        logger.exception("Backtest failed for %s/%s", pair, model)
        return None


async def _async_predict_signal(session: dict, candles_df):
    """Predict using model off the event loop (to_thread for ML inference)."""
    bt = session.get("backtester")
    trained_model = session.get("model_obj")
    if bt is None or trained_model is None:
        return None
    try:
        if candles_df is None or candles_df.empty:
            return None
        features_df = await asyncio.to_thread(bt._compute_features, candles_df)
        if features_df is None or features_df.empty:
            return None
        last_row = features_df.iloc[[-1]]
        feature_cols = [c for c in features_df.columns if c not in ("time", "target", "side")]

        def _predict():
            X = last_row[feature_cols].values
            if hasattr(trained_model, "predict_proba"):
                proba = trained_model.predict_proba(X)
                return proba
            elif hasattr(trained_model, "predict"):
                return trained_model.predict(X)
            return None
        raw = await asyncio.to_thread(_predict)
        if raw is None:
            return None
        if raw.shape[1] > 1:
            confidence = float(raw[0].max()) * 100
            prediction = int(raw[0].argmax())
        else:
            prediction = int(raw[0])
            confidence = 60.0
        direction = "LONG" if prediction == 1 else "SHORT" if prediction in (0, -1) else "FLAT"
        return {"direction": direction, "confidence": min(confidence, 99.0)}
    except Exception:
        logger.exception("Signal prediction failed %s", session.get("pair", ""))
        return None


async def _fetch_prices_async(pair: str):
    """Async version of _oanda_api_call (uses asyncio.to_thread)."""
    try:
        result = await asyncio.to_thread(_oanda_api_call, pair.replace("", ""))
        return result
    except Exception:
        return None, "unavailable"


# ═════════════════════════════════════════════════════════════════════
#  Paper Trading — Signal Loop (async-safe)
# ═════════════════════════════════════════════════════════════════════

async def _paper_signal_loop(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        return

    pair = session["pair"]
    timeframe = session["timeframe"]
    engine = session["engine"]
    store = get_data_store()
    lock: asyncio.Lock = session["_lock"]

    from api.routers.price_stream import PriceStreamManager
    psm = PriceStreamManager.get()

    await psm.ensure_stream(pair)

    price_queue: asyncio.Queue = asyncio.Queue()
    psm.subscribe(pair, price_queue)

    async def _forward_price_ticks():
        while not session["_kill_event"].is_set():
            try:
                msg = await asyncio.wait_for(price_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            _broadcast_ws(session, msg)

    forward_task = asyncio.create_task(_forward_price_ticks())

    try:
        while not session["_kill_event"].is_set():
            new_bar_event = psm.get_new_bar_event(pair, timeframe)
            if new_bar_event is None:
                await asyncio.sleep(60)
                continue

            try:
                await asyncio.wait_for(new_bar_event.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                logger.warning("No new bar for %s %s in 120s", pair, timeframe)
                continue

            candles = store.get_latest_candles(pair, timeframe, 60)
            if candles.empty:
                continue

            signal_result = await _async_predict_signal(session, candles)

            mid_price = float(candles.iloc[-1]["mid_close"])
            raw_prices, source = await _fetch_prices_async(pair)
            if source == "oanda" and raw_prices:
                mid_price = _extract_mid(raw_prices, source, pair, timeframe, store)
            bid, ask = _resolve_bid_ask(raw_prices, source, mid_price)

            async with lock:
                if session["_kill_event"].is_set():
                    break
                result = engine.process_signal(
                    signal_result or {"direction": "FLAT", "confidence": 50.0},
                    bid=bid, ask=ask, mid=mid_price,
                )
                last = candles.iloc[-1]
                result["candle"] = {
                    "time": int(last["time"].timestamp()),
                    "open": float(last["mid_open"]),
                    "high": float(last["mid_high"]),
                    "low": float(last["mid_low"]),
                    "close": float(last["mid_close"]),
                }
                result["live_price"] = mid_price
                _broadcast_ws(session, result)
                try:
                    eq = result.get("equity")
                    if eq is not None:
                        store.update_session_equity(session_id, float(eq), result.get("position", "FLAT"))
                except Exception:
                    pass

    except Exception:
        logger.exception("Paper signal loop error for session %s", session_id)
    finally:
        forward_task.cancel()
        try:
            await forward_task
        except asyncio.CancelledError:
            pass
        psm.unsubscribe(pair, price_queue)
        await psm.release_stream(pair)

    async with lock:
        session["status"] = "stopped"
    _broadcast_ws_stop(session, session_id)


# ═════════════════════════════════════════════════════════════════════
#  Live Trading — Signal Loop (async-safe + OANDA execution)
# ═════════════════════════════════════════════════════════════════════

async def _live_signal_loop(session_id: str):
    session = live_sessions.get(session_id)
    if not session:
        return

    pair = session["pair"]
    timeframe = session["timeframe"]
    engine = session["engine"]
    oanda = session["_oanda_client"]  # AsyncOandaClient
    store = get_data_store()
    lock: asyncio.Lock = session["_lock"]

    from api.routers.price_stream import PriceStreamManager
    psm = PriceStreamManager.get()

    await psm.ensure_stream(pair)

    price_queue: asyncio.Queue = asyncio.Queue()
    psm.subscribe(pair, price_queue)

    async def _forward_price_ticks():
        while not session["_kill_event"].is_set():
            try:
                msg = await asyncio.wait_for(price_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            _broadcast_ws(session, msg)

    forward_task = asyncio.create_task(_forward_price_ticks())

    try:
        while not session["_kill_event"].is_set():
            new_bar_event = psm.get_new_bar_event(pair, timeframe)
            if new_bar_event is None:
                await asyncio.sleep(60)
                continue

            try:
                await asyncio.wait_for(new_bar_event.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                logger.warning("No new bar for %s %s in 120s", pair, timeframe)
                continue

            candles = store.get_latest_candles(pair, timeframe, 60)
            if candles.empty:
                continue

            signal_result = await _async_predict_signal(session, candles)

            mid_price = float(candles.iloc[-1]["mid_close"])
            raw_prices, source = await _fetch_prices_async(pair)
            if source == "oanda" and raw_prices:
                mid_price = _extract_mid(raw_prices, source, pair, timeframe, store)
            bid, ask = _resolve_bid_ask(raw_prices, source, mid_price)

            async with lock:
                if session["_kill_event"].is_set():
                    break
                result = engine.process_signal(
                    signal_result or {"direction": "FLAT", "confidence": 50.0},
                    bid=bid, ask=ask, mid=mid_price,
                    oanda_client=oanda._get_client(),
                )
                engine.heartbeat()
                last = candles.iloc[-1]
                result["candle"] = {
                    "time": int(last["time"].timestamp()),
                    "open": float(last["mid_open"]),
                    "high": float(last["mid_high"]),
                    "low": float(last["mid_low"]),
                    "close": float(last["mid_close"]),
                }
                result["live_price"] = mid_price
                _broadcast_ws(session, result)
                try:
                    eq = result.get("equity")
                    if eq is not None:
                        store.update_session_equity(session_id, float(eq), result.get("position", "FLAT"))
                except Exception:
                    pass

            if result.get("event") == "kill":
                session["_kill_event"].set()

    except Exception:
        logger.exception("Live signal loop error for session %s", session_id)
    finally:
        forward_task.cancel()
        try:
            await forward_task
        except asyncio.CancelledError:
            pass
        psm.unsubscribe(pair, price_queue)
        await psm.release_stream(pair)

    async with lock:
        session["status"] = "stopped"
    _broadcast_ws_stop(session, session_id)


# ═════════════════════════════════════════════════════════════════════
#  Price helpers
# ═════════════════════════════════════════════════════════════════════

def _extract_mid(
    raw_prices, source: str, pair: str, timeframe: str, store
) -> float:
    if source == "oanda" and raw_prices:
        p0 = raw_prices[0]
        bids = p0.get("bids", [])
        asks = p0.get("asks", [])
        if bids and asks:
            return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0
    candles = store.get_latest_candles(pair, timeframe, 1)
    if not candles.empty:
        return float(candles.iloc[-1]["mid_close"])
    return 0.0


def _resolve_bid_ask(raw_prices, source: str, mid: float) -> tuple[float, float]:
    if source == "oanda" and raw_prices:
        p0 = raw_prices[0]
        bids = p0.get("bids", [])
        asks = p0.get("asks", [])
        if bids and asks:
            return float(bids[0]["price"]), float(asks[0]["price"])
    return mid, mid


# ═════════════════════════════════════════════════════════════════════
#  WebSocket broadcast (lock-free — asyncio.Queue is coroutine-safe)
# ═════════════════════════════════════════════════════════════════════

def _broadcast_ws(session: dict, msg: dict) -> None:
    for q in list(session.get("ws_queues", [])):
        try:
            q.put_nowait(msg)
        except Exception:
            pass
    for sub in msg.get("sub_events", []):
        for q in list(session.get("ws_queues", [])):
            try:
                q.put_nowait(sub)
            except Exception:
                pass


def _broadcast_ws_stop(session: dict, session_id: str) -> None:
    stop_msg = {"event": "stopped", "session_id": session_id}
    for q in list(session.get("ws_queues", [])):
        try:
            q.put_nowait(stop_msg)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════
#  Paper Trading — Endpoints
# ═════════════════════════════════════════════════════════════════════

@router.post("/start", response_model=PaperSessionInfo)
async def start_paper_session(req: DeployPaperRequest):
    pair = req.pair.upper().strip()
    token, account_id = _get_oanda_credentials()
    if not token or not account_id:
        logger.warning("OANDA credentials not configured — paper trading will use SQLite candles for prices")

    model_obj, bt, actual_model_type = _load_model_for_paper(
        req.model_id, req.model_type, pair, req.timeframe
    )

    session_id = str(uuid.uuid4())[:8]
    engine = __import__("trading.paper_engine", fromlist=["PaperEngine"]).PaperEngine()
    engine.start({
        "initial_equity": req.initial_equity,
        "position_sizing": req.position_sizing,
        "sizing_config": req.sizing_config,
    })

    session = {
        "session_id": session_id,
        "pair": pair,
        "model_type": actual_model_type,
        "timeframe": req.timeframe,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_obj": model_obj,
        "backtester": bt,
        "engine": engine,
        "ws_queues": [],
        "_lock": asyncio.Lock(),
        "_kill_event": asyncio.Event(),
        "_task": None,
    }
    active_sessions[session_id] = session
    task = asyncio.create_task(_paper_signal_loop(session_id))
    session["_task"] = task

    try:
        store = get_data_store()
        store.save_trading_session({
            "id": session_id, "mode": "paper", "pair": pair,
            "model_type": actual_model_type, "timeframe": req.timeframe,
            "status": "running", "initial_equity": req.initial_equity,
            "equity": req.initial_equity, "position": "FLAT",
            "model_id": req.model_id or "", "committee_name": "",
            "created_at": session["created_at"],
            "updated_at": session["created_at"],
        })
    except Exception:
        logger.exception("Failed to persist paper session %s", session_id)

    pstate = engine.get_portfolio_state()
    return PaperSessionInfo(
        session_id=session_id, pair=pair, model_type=actual_model_type,
        timeframe=req.timeframe, status="running",
        equity=pstate["equity"], position=pstate["position"],
        unrealized_pnl=pstate.get("unrealized_pnl", 0),
        total_trades=pstate.get("total_trades_closed", 0),
        signal_count=0, created_at=session["created_at"],
    )


@router.websocket("/{session_id}/ws")
async def paper_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_text(json.dumps({"event": "error", "message": "Session not found"}))
        await websocket.close()
        return

    q: asyncio.Queue = asyncio.Queue()
    session["ws_queues"].append(q)
    try:
        while not session["_kill_event"].is_set():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_text(json.dumps(msg, default=str))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"event": "heartbeat", "time": int(time.time())}))
            except asyncio.CancelledError:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error for paper session %s", session_id)
    finally:
        if q in session.get("ws_queues", []):
            session["ws_queues"].remove(q)


@router.post("/{session_id}/stop")
async def stop_paper_session(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    async with session["_lock"]:
        session["_kill_event"].set()
        if session["_task"]:
            session["_task"].cancel()
        engine = session["engine"]

    raw_prices, source = _oanda_api_call(session["pair"].replace("", ""))
    bid, ask = _resolve_bid_ask(raw_prices, source, 0.0)
    if bid <= 0:
        store = get_data_store()
        df = store.get_latest_candles(session["pair"], session["timeframe"], 1)
        if not df.empty:
            bid = ask = float(df.iloc[-1]["mid_close"])

    summary_raw = engine.stop(bid=bid if bid > 0 else None, ask=ask if ask > 0 else None)
    close_events = summary_raw.pop("events", [])

    _broadcast_ws(session, {"sub_events": close_events} if close_events else {})

    model_obj = session.get("model_obj")
    if model_obj is not None:
        try:
            if hasattr(model_obj, "close"):
                model_obj.close()
        except Exception:
            pass

    try:
        s = engine.get_summary()
        eq = float(s.get("equity", 0))
        store = get_data_store()
        store.update_session_status(session_id, "stopped", eq)
    except Exception:
        logger.exception("Failed to persist stop for session %s", session_id)

    return {
        "session_id": session_id, "status": "stopped",
        "summary": engine.get_summary(),
        "comparison": engine.compare_to_backtest({}),
    }


@router.get("/{session_id}/status", response_model=PaperSessionInfo)
async def get_paper_status(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    engine = session["engine"]
    pstate = engine.get_portfolio_state()
    return PaperSessionInfo(
        session_id=session["session_id"], pair=session["pair"],
        model_type=session["model_type"], timeframe=session["timeframe"],
        status=session["status"], equity=pstate["equity"],
        position=pstate["position"], unrealized_pnl=pstate.get("unrealized_pnl", 0),
        total_trades=pstate.get("total_trades_closed", 0),
        signal_count=pstate.get("signal_count", 0),
        created_at=session["created_at"],
    )


@router.get("/{session_id}/trades")
async def get_paper_trades(
    session_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500),
):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    engine = session["engine"]
    trades = engine.get_trades(offset=offset, limit=limit)
    return {"session_id": session_id, "trades": trades, "offset": offset, "limit": limit}


@router.get("/{session_id}/summary")
async def get_paper_summary(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    engine = session["engine"]
    return {
        "session_id": session_id, "summary": engine.get_summary(),
        "comparison": engine.compare_to_backtest({}),
    }


@router.get("/sessions")
async def list_paper_sessions():
    memory = [
        {"session_id": s["session_id"], "pair": s["pair"], "model_type": s["model_type"],
         "timeframe": s["timeframe"], "status": s["status"], "created_at": s["created_at"],
         "initial_equity": s.get("initial_equity", 10000)}
        for s in active_sessions.values()
    ]
    try:
        store = get_data_store()
        db = store.list_trading_sessions("paper")
        seen = {s["session_id"] for s in memory}
        for row in db:
            if row["id"] not in seen:
                memory.append({
                    "session_id": row["id"], "pair": row["pair"],
                    "model_type": row["model_type"], "timeframe": row["timeframe"],
                    "status": row["status"], "created_at": row["created_at"],
                    "initial_equity": row["initial_equity"],
                })
    except Exception:
        pass
    return memory


# ═════════════════════════════════════════════════════════════════════
#  Live Trading — Endpoints
# ═════════════════════════════════════════════════════════════════════

@live_router.post("/start", response_model=LiveSessionInfo)
async def start_live_session(req: DeployLiveRequest):
    pair = req.pair.upper().strip()

    from trading.async_oanda import AsyncOandaClient

    token, account_id = _get_oanda_credentials()
    if not token or not account_id:
        raise HTTPException(403, "OANDA API key not configured")

    oanda = AsyncOandaClient(access_token=token, account_id=account_id)
    try:
        acct = await oanda.get_account_summary()
        balance = float(acct.get("balance", 0))
    except Exception:
        raise HTTPException(500, "Failed to connect to OANDA")

    model_obj, bt, actual_model_type = _load_model_for_paper(
        req.model_id, req.model_type, pair, req.timeframe
    )

    session_id = str(uuid.uuid4())[:8]
    from trading.live_engine import OandaTradingEngine

    engine = OandaTradingEngine()
    engine.start({
        "pair": pair,
        "initial_equity": balance if balance > 0 else req.initial_equity,
        "mode": req.mode,
        "position_sizing": req.position_sizing,
        "sizing_config": req.sizing_config,
        "risk_config": req.risk_config,
    }, oanda._get_client())

    state = engine.get_session_state()
    session = {
        "session_id": session_id,
        "pair": pair,
        "model_type": actual_model_type,
        "timeframe": req.timeframe,
        "mode": req.mode,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_obj": model_obj,
        "backtester": bt,
        "engine": engine,
        "_oanda_client": oanda,
        "ws_queues": [],
        "_lock": asyncio.Lock(),
        "_kill_event": asyncio.Event(),
        "_task": None,
        "kill_reason": "",
    }
    live_sessions[session_id] = session
    task = asyncio.create_task(_live_signal_loop(session_id))
    session["_task"] = task

    try:
        store = get_data_store()
        store.save_trading_session({
            "id": session_id, "mode": "live", "pair": pair,
            "model_type": actual_model_type, "timeframe": req.timeframe,
            "status": "running", "initial_equity": state["equity"],
            "equity": state["equity"], "position": state["position"],
            "model_id": req.model_id or "", "committee_name": "",
            "created_at": session["created_at"],
            "updated_at": session["created_at"],
        })
    except Exception:
        logger.exception("Failed to persist live session %s", session_id)

    return LiveSessionInfo(
        session_id=session_id, pair=pair, model_type=actual_model_type,
        timeframe=req.timeframe, mode=req.mode, status="running",
        equity=state["equity"], position=state["position"],
        unrealized_pnl=state["unrealized_pnl"],
        signal_count=0, killed=False, kill_reason="",
    )


@live_router.websocket("/{session_id}/ws")
async def live_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = live_sessions.get(session_id)
    if not session:
        await websocket.send_text(json.dumps({"event": "error", "message": "Session not found"}))
        await websocket.close()
        return

    q: asyncio.Queue = asyncio.Queue()
    session["ws_queues"].append(q)
    try:
        while not session["_kill_event"].is_set():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_text(json.dumps(msg, default=str))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"event": "heartbeat", "time": int(time.time())}))
            except asyncio.CancelledError:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error for live session %s", session_id)
    finally:
        if q in session.get("ws_queues", []):
            session["ws_queues"].remove(q)


@live_router.post("/{session_id}/stop")
async def stop_live_session(session_id: str):
    session = live_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    async with session["_lock"]:
        session["_kill_event"].set()
        if session["_task"]:
            session["_task"].cancel()
        engine = session["engine"]

    oanda = session.get("_oanda_client")
    sync_client = oanda._get_client() if oanda else None
    result = engine.stop(oanda_client=sync_client)

    model_obj = session.get("model_obj")
    if model_obj is not None:
        try:
            if hasattr(model_obj, "close"):
                model_obj.close()
        except Exception:
            pass

    state = engine.get_session_state()
    try:
        store = get_data_store()
        store.update_session_status(session_id, "stopped", float(state["equity"]))
    except Exception:
        logger.exception("Failed to persist stop for session %s", session_id)
    return {
        "session_id": session_id, "status": "stopped",
        "equity": state["equity"],
        "events": result.get("events", []),
        "journal_length": result.get("journal_length", 0),
        "killed": state["killed"],
    }


@live_router.post("/{session_id}/emergency")
async def emergency_kill_session(session_id: str):
    session = live_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    async with session["_lock"]:
        session["_kill_event"].set()
        if session["_task"]:
            session["_task"].cancel()
        engine = session["engine"]

    oanda = session.get("_oanda_client")
    sync_client = oanda._get_client() if oanda else None
    result = engine.emergency_kill(oanda_client=sync_client)

    for q in list(session.get("ws_queues", [])):
        try:
            q.put_nowait({"event": "kill", "reason": "emergency_kill_button"})
        except Exception:
            pass

    return result


@live_router.get("/{session_id}/status", response_model=LiveSessionInfo)
async def get_live_status(session_id: str):
    session = live_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    engine = session["engine"]
    state = engine.get_session_state()
    return LiveSessionInfo(
        session_id=session["session_id"], pair=session["pair"],
        model_type=session["model_type"], timeframe=session["timeframe"],
        mode=session.get("mode", "demo"), status=session.get("status", "unknown"),
        equity=state["equity"], position=state["position"],
        unrealized_pnl=state["unrealized_pnl"], signal_count=state["signal_count"],
        killed=state["killed"], kill_reason=state["kill_reason"],
    )


@live_router.get("/{session_id}/journal")
async def get_live_journal(
    session_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500),
):
    session = live_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    engine = session["engine"]
    journal = engine.get_journal(offset=offset, limit=limit)
    return {"session_id": session_id, "journal": journal, "offset": offset, "limit": limit}


@live_router.get("/{session_id}/risk")
async def get_live_risk_state(session_id: str):
    session = live_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    engine = session["engine"]
    return {"session_id": session_id, "risk": engine.get_risk_state()}


@live_router.get("/{session_id}/attribution")
async def get_live_attribution(session_id: str):
    session = live_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    from pipeline.metrics.attribution import compute_attribution_from_session
    report = compute_attribution_from_session(session)
    return report.to_dict()


@live_router.get("/sessions")
async def list_live_sessions():
    memory = [
        {
            "session_id": s["session_id"], "pair": s["pair"],
            "model_type": s["model_type"], "timeframe": s["timeframe"],
            "mode": s.get("mode", "demo"), "status": s["status"],
            "killed": s.get("kill_reason", "") != "", "created_at": s["created_at"],
        }
        for s in live_sessions.values()
    ]
    try:
        store = get_data_store()
        db = store.list_trading_sessions("live")
        seen = {s["session_id"] for s in memory}
        for row in db:
            if row["id"] not in seen:
                memory.append({
                    "session_id": row["id"], "pair": row["pair"],
                    "model_type": row["model_type"], "timeframe": row["timeframe"],
                    "mode": row["mode"], "status": row["status"],
                    "killed": row["status"] == "killed", "created_at": row["created_at"],
                })
    except Exception:
        pass
    return memory


def build_committee_metrics_snapshot(session: dict) -> dict:
    """Build full metrics snapshot from a committee session for API/WS broadcast."""
    runner = session.get("runner")
    if not runner:
        return {"error": "no runner", "session_id": session.get("session_id", "")}

    health = runner.get_health_summary()
    regimes = runner.get_recent_regimes(100)
    regime_dist: Dict[str, float] = {}
    for r in regimes:
        regime_dist[r] = regime_dist.get(r, 0) + 1
    total_r = len(regimes)
    if total_r > 0:
        for r in regime_dist:
            regime_dist[r] = round(regime_dist[r] / total_r, 4)

    trust_data = session.get("trust_score") or {}

    return {
        "session_id": session["session_id"],
        "uptime_seconds": round(time.time() - runner._start_time.timestamp(),
                                1) if runner._start_time else 0,
        "bar_count": runner._bar_count,
        "signal_count": len(runner._signal_history),
        "non_zero_signals": sum(1 for s in runner._signal_history if s.signal != 0),
        "committee_healthy": runner._check_health(),
        "current_regime": regimes[-1] if regimes else "unknown",
        "regime_distribution": regime_dist,
        "trust_score": trust_data.get("trust_score"),
        "trust_multiplier": session.get("trust_multiplier", 1.0),
        "effective_multiplier": session.get("effective_multiplier", 1.0),
        "throttle_summary": session["regime_throttle"].to_dict() if session.get("regime_throttle") else {},
        "per_model_health": {
            m: {
                "rolling_sharpe": h["rolling_sharpe"],
                "rolling_hit_rate": h["rolling_hit_rate"],
                "total_signals": h["total_signals"],
                "wins": h["wins"],
                "losses": h["losses"],
                "status": (
                    "insufficient_data" if h.get("total_signals", 0) < 3
                    else "healthy" if h.get("is_healthy", True)
                    else "unhealthy"
                ),
            }
            for m, h in health.items()
        },
        "recent_signals": [s.to_dict() for s in runner.get_recent_signals(5)],
    }


# ═════════════════════════════════════════════════════════════════════
#  Committee Trading — Signal Loop + Endpoint
# ═════════════════════════════════════════════════════════════════════

async def _committee_trading_signal_loop(session_id: str):
    """Poll prices → build bar → runner.process_bar() → engine.process_signal() → WS."""
    session = live_sessions.get(session_id)
    if not session:
        return

    pair = session["pair"]
    timeframe = session["timeframe"]
    runner = session["runner"]
    engine = session["engine"]
    store = get_data_store()
    lock: asyncio.Lock = session["_lock"]

    tf_seconds = {"M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}
    poll_interval = min(tf_seconds.get(timeframe, 1800) // 4, 60)

    import numpy as np

    while not session["_kill_event"].is_set():
        try:
            raw_prices, source = await _fetch_prices_async(pair)
            mid_price = _extract_mid(raw_prices, source, pair, timeframe, store)
            if mid_price <= 0:
                await asyncio.sleep(poll_interval)
                continue

            candles_df = store.get_latest_candles(pair, timeframe, 100)
            if candles_df is None or candles_df.empty:
                await asyncio.sleep(poll_interval)
                continue

            bar = {
                "mid_c": float(candles_df.iloc[-1]["mid_close"])
                if "mid_close" in candles_df.columns
                else float(candles_df.iloc[-1].get("mid_c", mid_price)),
                "mid_h": float(candles_df.iloc[-1].get("mid_high",
                    candles_df.iloc[-1].get("mid_h", candles_df.iloc[-1].get("mid_c", mid_price)))),
                "mid_l": float(candles_df.iloc[-1].get("mid_low",
                    candles_df.iloc[-1].get("mid_l", candles_df.iloc[-1].get("mid_c", mid_price)))),
                "mid_o": float(candles_df.iloc[-1].get("mid_open",
                    candles_df.iloc[-1].get("mid_o", candles_df.iloc[-1].get("mid_c", mid_price)))),
                "spread": float(candles_df.iloc[-1].get("spread", 0.0001)),
                "returns": float(np.log(mid_price / session.get("last_mid", mid_price)))
                if session.get("last_mid") and mid_price else 0.0,
                "timestamp": int(time.time()),
            }

            live_signal = runner.process_bar(bar)

            bid, ask = _resolve_bid_ask(raw_prices, source, mid_price)

            async with lock:
                if session["_kill_event"].is_set():
                    break

                signal_dict = live_signal if live_signal is not None else None

                result = engine.process_signal(
                    signal_dict or None,
                    bid=bid, ask=ask, mid=mid_price,
                )
                engine.heartbeat()
                _broadcast_ws(session, result)
                try:
                    eq = result.get("equity")
                    if eq is not None:
                        store.update_session_equity(session_id, float(eq), result.get("position", "FLAT"))
                except Exception:
                    pass

                if result.get("event") == "kill":
                    session["_kill_event"].set()

        except Exception:
            logger.exception("Committee signal loop error for session %s", session_id)

        await asyncio.sleep(poll_interval)

    async with lock:
        session["status"] = "stopped"
    _broadcast_ws_stop(session, session_id)


@live_router.post("/committee/start")
async def start_committee_session(req: DeployCommitteeRequest):
    pair = req.pair.upper().strip()
    timeframe = req.timeframe

    # 1. Load committee config
    config_json = None
    parent_job_dir = None
    if req.full_cycle_job_id:
        parent_job_dir = Path("results/full_cycle") / req.full_cycle_job_id
        config_path = parent_job_dir / "committee_config_final.json"
        if not config_path.exists():
            config_path = parent_job_dir / "committee_config.json"
        if not config_path.exists():
            raise HTTPException(404, f"Full Cycle job {req.full_cycle_job_id} not found or has no config")
    elif req.committee_config_path:
        config_path = Path(req.committee_config_path)
    else:
        config_path = Path("results/full_cycle")
        candidates = sorted(config_path.glob("fullcycle_*"), reverse=True)
        found = False
        for c in candidates:
            final_cfg = c / "committee_config_final.json"
            if final_cfg.exists():
                config_path = final_cfg
                parent_job_dir = c
                found = True
                break
            base_cfg = c / "committee_config.json"
            if base_cfg.exists():
                config_path = base_cfg
                parent_job_dir = c
                found = True
                break
        if not found:
            raise HTTPException(404, "No committee config found. Run the Full Cycle first.")

    if not config_path or not config_path.exists():
        raise HTTPException(404, "No committee config found. Run the Full Cycle first.")

    with open(config_path) as f:
        config_json = json.load(f)

    from pipeline.committee.committee_builder import CommitteeConfig
    committee_config = CommitteeConfig.from_dict(config_json)
    model_params = committee_config.model_params or config_json.get("model_params", {})

    if parent_job_dir is None:
        parent_job_dir = config_path.parent if config_path.parent.name.startswith("fullcycle_") else None
    if parent_job_dir is None:
        parent_job_dir = config_path.parent.parent if "full_cycle" in str(config_path) else None

    # 2. Load trust score
    trust_multiplier = 1.0
    if parent_job_dir:
        trust_path = parent_job_dir / "trust_score.json"
        if trust_path.exists():
            try:
                with open(trust_path) as f:
                    trust_data = json.load(f)
                ts = trust_data.get("trust_score", 1.0)
                action = trust_data.get("action", "deploy")
                if action == "reject":
                    raise HTTPException(400,
                        f"Committee trust_score={ts:.2f} — deployment rejected.")
                trust_multiplier = max(0.0, min(1.0, ts))
            except HTTPException:
                raise
            except Exception:
                pass

    # 3. Load locked features
    locked_features = None
    locked_features_path = Path("results/locked_features.json")
    if locked_features_path.exists():
        try:
            from pipeline.features.feature_sweep import load_locked_features
            locked_features = load_locked_features(str(locked_features_path))
        except Exception:
            pass

    # 4. Get unique model types
    unique_models = list(set(committee_config.all_models()))
    if not unique_models:
        raise HTTPException(400, "Committee config has no models")

    trained_models: dict[str, Any] = {}
    feature_names = locked_features if locked_features else None

    # 5. Try loading saved snapshot (fast path)
    snapshot_loaded = False
    if parent_job_dir:
        snapshot_dir = parent_job_dir / "committee_snapshot"
        if snapshot_dir.exists() and (snapshot_dir / "manifest.json").exists():
            try:
                import joblib
                with open(snapshot_dir / "manifest.json") as f:
                    manifest = json.load(f)
                snapshot_feature_names = manifest.get("feature_names",
                    list(feature_names) if feature_names else [])
                for mtype in unique_models:
                    jl_path = snapshot_dir / f"{mtype}.joblib"
                    tf_path = snapshot_dir / f"{mtype}_tf"
                    if jl_path.exists():
                        trained_models[mtype] = joblib.load(str(jl_path))
                    elif tf_path.exists():
                        import tensorflow as tf
                        trained_models[mtype] = tf.keras.models.load_model(str(tf_path))
                if trained_models:
                    feature_names = snapshot_feature_names
                    snapshot_loaded = True
            except Exception:
                logger.exception("Snapshot load failed, falling back to fresh training")

    # 6. Train models (slow path)
    if not snapshot_loaded:
        trained_models, feature_names = _train_committee_models(
            pair, timeframe, unique_models,
            feature_names=locked_features,
            model_params=model_params,
        )

    if not trained_models:
        raise HTTPException(500, "Failed to obtain committee models")

    # 7. CommitteeMetaLearner retired — P1 MetaLabeler now handles trade filtering.
    #    The old meta-learner flipped signal direction (Long→Short), violating
    #    separation of concerns. A secondary model should never reverse the
    #    primary model's direction — only suppress to flat (which MetaLabeler does).

    # 7b. Load P1-P3 pipeline artifacts
    meta_labeler = None
    hmm_detector = None
    conviction_sizer = None
    if parent_job_dir:
        # P1: MetaLabeler
        ml_path = parent_job_dir / "meta_labeler.joblib"
        if ml_path.exists():
            try:
                from pipeline.models.meta_labeler import MetaLabeler
                meta_labeler = MetaLabeler.load(str(ml_path))
            except Exception:
                pass
        # P2: HMMRegimeDetector
        hmm_path = parent_job_dir / "hmm_detector.joblib"
        if hmm_path.exists():
            try:
                from pipeline.regime.hmm_regime import HMMRegimeDetector
                hmm_detector = HMMRegimeDetector.load(str(hmm_path))
            except Exception:
                pass
        # P3: ConvictionSizer
        cs_path = parent_job_dir / "conviction_sizer.json"
        if cs_path.exists():
            try:
                from pipeline.execution.conviction_sizer import ConvictionSizer
                conviction_sizer = ConvictionSizer.load(str(cs_path))
            except Exception:
                pass

    # 8. Create runner
    from trading.live_committee_runner import LiveCommitteeRunner
    from pipeline.regime.regime_utils import RegimeConfig

    runner = LiveCommitteeRunner(
        config=committee_config,
        models=trained_models,
        feature_names=list(feature_names) if feature_names else [],
        regime_cfg=RegimeConfig(),
        confidence_threshold=req.confidence_threshold,
        lookback_bars=req.lookback_bars,
        meta_labeler=meta_labeler,
        hmm_detector=hmm_detector,
        conviction_sizer=conviction_sizer,
    )
    runner.start()

    # 9. Create engine
    from trading.committee_engine import CommitteeTradingEngine

    engine = CommitteeTradingEngine()
    oanda_client = None
    if req.mode == "live":
        from trading.async_oanda import AsyncOandaClient
        token, account_id = _get_oanda_credentials()
        if token and account_id:
            oanda = AsyncOandaClient(access_token=token, account_id=account_id)
            oanda_client = oanda._get_client()
        else:
            raise HTTPException(403, "OANDA API key not configured")

    engine.start({
        "pair": pair,
        "initial_equity": req.initial_equity,
        "mode": req.mode,
        "position_sizing": req.position_sizing,
        "sizing_config": req.sizing_config,
        "risk_config": req.risk_config,
        "trust_multiplier": trust_multiplier,
    }, oanda_client=oanda_client)

    # 10. Register session
    session_id = str(uuid.uuid4())[:8]
    session = {
        "session_id": session_id,
        "pair": pair,
        "model_type": "committee",
        "timeframe": timeframe,
        "mode": req.mode,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": runner,
        "engine": engine,
        "ws_queues": [],
        "_lock": asyncio.Lock(),
        "_kill_event": asyncio.Event(),
        "_task": None,
        "last_mid": None,
        "kill_reason": "",
        "trained_models": trained_models,
        "committee_config": committee_config,
        "trust_multiplier": trust_multiplier,
        "live_news_blend_enabled": req.live_news_blend_enabled,
        "live_news_blend_weight": req.live_news_blend_weight,
        "_config_path": str(config_path),
        "_parent_job_dir": str(parent_job_dir) if parent_job_dir else None,
        "_retrain_task": None,
    }

    live_sessions[session_id] = session
    task = asyncio.create_task(_committee_trading_signal_loop(session_id))
    session["_task"] = task

    try:
        store = get_data_store()
        store.save_trading_session({
            "id": session_id, "mode": "committee", "pair": pair,
            "model_type": "committee", "timeframe": timeframe,
            "status": "running", "initial_equity": req.initial_equity,
            "equity": req.initial_equity, "position": "FLAT",
            "model_id": "", "committee_name": committee_name or "",
            "created_at": session["created_at"],
            "updated_at": session["created_at"],
        })
    except Exception:
        logger.exception("Failed to persist committee session %s", session_id)

    state = engine.get_portfolio_state()
    return {
        "session_id": session_id,
        "pair": pair,
        "model": "committee",
        "models": unique_models,
        "timeframe": timeframe,
        "mode": req.mode,
        "status": "running",
        "equity": state["equity"],
        "position": state["position"],
        "signal_count": 0,
        "feature_count": len(feature_names) if feature_names else 0,
        "lookback_bars": req.lookback_bars,
        "snapshot_loaded": snapshot_loaded,
        "model_params_count": len(model_params),
    }


# ── Fast Retrain endpoint (Fast Loop rolling refit) ──────────────

class RetrainRequest(BaseModel):
    lookback_bars: int = 20000
    oos_frac: float = 0.10


class RetrainStatusResponse(BaseModel):
    session_id: str
    status: str  # "idle" | "running" | "complete" | "failed"
    progress: float = 0.0
    current_phase: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    models_refitted: List[str] = []
    models_skipped: List[str] = []
    meta_labeler_refitted: bool = False
    meta_accuracy: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None


@live_router.post("/committee/{session_id}/retrain")
async def retrain_committee(session_id: str, req: RetrainRequest):
    session = live_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    if session.get("model_type") != "committee":
        raise HTTPException(400, "Retrain only available for committee sessions")
    if session["status"] != "running":
        raise HTTPException(400, f"Cannot retrain: session status is '{session['status']}'")

    existing_task = session.get("_retrain_task")
    if existing_task and not existing_task.done():
        raise HTTPException(409, "A retrain is already in progress")

    config_path = session.get("_config_path")
    if not config_path or not Path(config_path).exists():
        raise HTTPException(400, "Committee config path not available")

    parent_job_dir = session.get("_parent_job_dir")
    hmm_path = None
    meta_path = None
    if parent_job_dir:
        pjd = Path(parent_job_dir)
        hmm_candidate = pjd / "hmm_detector.joblib"
        if hmm_candidate.exists():
            hmm_path = str(hmm_candidate)
        meta_candidate = pjd / "meta_labeler.joblib"
        if meta_candidate.exists():
            meta_path = str(meta_candidate)

    if not hmm_path:
        raise HTTPException(400, "HMM artifact not found — cannot retrain")

    session["_retrain_status"] = {
        "status": "running",
        "progress": 0.0,
        "current_phase": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "models_refitted": [],
        "models_skipped": [],
        "error": None,
    }

    loop = asyncio.get_running_loop()

    def _retrain_sync():
        from pipeline.models.fast_retrain import FastRetrainer
        import shutil

        status = session["_retrain_status"]
        output_dir = Path("artifacts") / f"retrain_{session_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        try:
            status["current_phase"] = "loading"
            _broadcast_ws(session, {
                "event": "retrain_progress",
                "phase": "loading",
                "progress": 0.05,
            })

            retrainer = FastRetrainer(
                config_path=config_path,
                hmm_path=hmm_path,
                meta_path=meta_path,
                symbol=session["pair"],
                base_timeframe=session["timeframe"],
                lookback_bars=req.lookback_bars,
                oos_frac=req.oos_frac,
                output_dir=str(output_dir),
                seed=42,
            )
            retrainer._set_random_seed(42)

            status["current_phase"] = "loading_data"
            retrainer.load_data()

            status["current_phase"] = "computing_features"
            status["progress"] = 0.15
            _broadcast_ws(session, {
                "event": "retrain_progress", "phase": "features", "progress": 0.15,
            })
            X_full = retrainer.compute_features()
            retrainer.scale_features(X_full)
            retrainer.generate_labels()
            retrainer.tag_regimes()

            status["current_phase"] = "refitting_models"
            status["progress"] = 0.30
            _broadcast_ws(session, {
                "event": "retrain_progress", "phase": "refitting", "progress": 0.30,
            })
            refitted = retrainer.refit_primary_models()

            status["progress"] = 0.70
            _broadcast_ws(session, {
                "event": "retrain_progress", "phase": "meta_labeler", "progress": 0.70,
            })
            retrainer.refit_meta_labeler(refitted)

            status["current_phase"] = "saving"
            status["progress"] = 0.90
            retrainer.save_artifacts()

            manifest = retrainer.manifest

            # Hot-swap: update session models under lock
            async def _hot_swap():
                async with session["_lock"]:
                    new_models = {}
                    for key, path in refitted.items():
                        import joblib
                        try:
                            model = joblib.load(path)
                            model_name = key.split("/")[-1]
                            new_models[model_name] = model
                        except Exception as e:
                            logger.warning("Failed to load %s: %s", path, e)

                    if new_models:
                        session["trained_models"].update(new_models)
                        runner = session["runner"]
                        for name, model in new_models.items():
                            runner.rotate_model(name, name, model)

                    ml_path = output_dir / "meta_labeler.joblib"
                    if ml_path.exists() and manifest.get("meta_labeler_refitted"):
                        try:
                            from pipeline.models.meta_labeler import MetaLabeler
                            new_meta = MetaLabeler.load(str(ml_path))
                            session["runner"]._meta_labeler = new_meta
                        except Exception as e:
                            logger.warning("Failed to load new MetaLabeler: %s", e)

            asyncio.run_coroutine_threadsafe(_hot_swap(), loop)

            status["status"] = "complete"
            status["progress"] = 1.0
            status["completed_at"] = datetime.now(timezone.utc).isoformat()
            status["models_refitted"] = manifest.get("models_refitted", [])
            status["models_skipped"] = list(manifest.get("models_skipped", {}).keys())
            status["meta_labeler_refitted"] = manifest.get("meta_labeler_refitted", False)
            status["meta_accuracy"] = manifest.get("meta_labeler_accuracy")
            status["elapsed_seconds"] = manifest.get("elapsed_seconds")

            _broadcast_ws(session, {
                "event": "retrain_complete",
                "models_refitted": status["models_refitted"],
                "models_skipped": status["models_skipped"],
                "meta_accuracy": status.get("meta_accuracy"),
                "elapsed_seconds": status.get("elapsed_seconds"),
            })

        except Exception as e:
            logger.exception("Retrain failed for session %s", session_id)
            status["status"] = "failed"
            status["error"] = str(e)
            status["completed_at"] = datetime.now(timezone.utc).isoformat()
            _broadcast_ws(session, {
                "event": "retrain_failed",
                "error": str(e),
            })

    task = asyncio.get_running_loop().run_in_executor(None, _retrain_sync)
    session["_retrain_task"] = task

    return {
        "session_id": session_id,
        "status": "retraining",
        "started_at": session["_retrain_status"]["started_at"],
    }


@live_router.get("/committee/{session_id}/retrain/status")
async def retrain_status(session_id: str):
    session = live_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")

    status = session.get("_retrain_status")
    if status is None:
        return RetrainStatusResponse(session_id=session_id, status="idle")

    return RetrainStatusResponse(
        session_id=session_id,
        status=status.get("status", "idle"),
        progress=status.get("progress", 0.0),
        current_phase=status.get("current_phase", ""),
        started_at=status.get("started_at"),
        completed_at=status.get("completed_at"),
        models_refitted=status.get("models_refitted", []),
        models_skipped=status.get("models_skipped", []),
        meta_labeler_refitted=status.get("meta_labeler_refitted", False),
        meta_accuracy=status.get("meta_accuracy"),
        elapsed_seconds=status.get("elapsed_seconds"),
        error=status.get("error"),
    )


# ── Committee model training helper ───────────────────────────

def _train_committee_models(
    pair: str,
    timeframe: str,
    model_types: list[str],
    feature_names: list[str] | None = None,
    model_params: dict | None = None,
):
    """Train committee models on recent data (3 months)."""
    from pipeline.backtester.composed import MLBacktester
    from config import PIPELINE_CONSTANTS as _PC

    if model_params is None:
        model_params = {}

    cfg = {
        "pair": pair,
        "timeframe": timeframe,
        "model_type": model_types[0],
        "n_months": 3,
        "use_WFO": False,
        "use_proba": True,
        "features_config": _PC.get("features_config", {}),
        "search_space": _PC.get("search_space", {}),
    }

    trained: dict[str, Any] = {}
    final_features: list[str] = list(feature_names) if feature_names else []

    try:
        bt = MLBacktester(config=cfg)
        bt.load_data()

        for mtype in model_types:
            result = bt.run_strategy(
                config=cfg, models_to_test=[mtype],
                n_trials=1, n_startup_trials=1,
            )
            if result is not None and result[0] is not None and not result[0].empty:
                model_obj = getattr(bt, "model", None)
                if model_obj is not None:
                    trained[mtype] = model_obj
                if not final_features:
                    from pipeline.backtester.composed import MLBacktester as MBC
                    final_features = list(getattr(bt, "feature_names", []))

    except Exception:
        logger.exception("Failed to train committee models for %s/%s", pair, timeframe)

    return trained, final_features
