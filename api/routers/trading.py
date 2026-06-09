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
from typing import Dict

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


# ═════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═════════════════════════════════════════════════════════════════════

def _load_model_for_paper(model_id: str | None, model_type: str, pair: str, timeframe: str):
    model_obj = None
    bt = None

    if model_id:
        from pipeline.model_persistence import load_model_only, read_metadata
        from pipeline.model_registry_disk import get_all_deployed
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

    tf_seconds = {"M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}
    poll_interval = min(tf_seconds.get(timeframe, 1800) // 4, 60)

    while not session["_kill_event"].is_set():
        try:
            raw_prices, source = await _fetch_prices_async(pair)
            mid_price = _extract_mid(raw_prices, source, pair, timeframe, store)
            if mid_price <= 0:
                await asyncio.sleep(poll_interval)
                continue

            candles = store.get_latest_candles(pair, timeframe, 60)
            signal_result = await _async_predict_signal(session, candles)
            bid, ask = _resolve_bid_ask(raw_prices, source, mid_price)

            async with lock:
                if session["_kill_event"].is_set():
                    break
                result = engine.process_signal(
                    signal_result or {"direction": "FLAT", "confidence": 50.0},
                    bid=bid, ask=ask, mid=mid_price,
                )
                _broadcast_ws(session, result)

        except Exception:
            logger.exception("Paper signal loop error for session %s", session_id)

        await asyncio.sleep(poll_interval)

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

    tf_seconds = {"M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}
    poll_interval = min(tf_seconds.get(timeframe, 1800) // 4, 60)

    while not session["_kill_event"].is_set():
        try:
            raw_prices, source = await _fetch_prices_async(pair)
            mid_price = _extract_mid(raw_prices, source, pair, timeframe, store)
            if mid_price <= 0:
                await asyncio.sleep(poll_interval)
                continue

            candles = store.get_latest_candles(pair, timeframe, 60)
            signal_result = await _async_predict_signal(session, candles)
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
                _broadcast_ws(session, result)

            if result.get("event") == "kill":
                session["_kill_event"].set()

        except Exception:
            logger.exception("Live signal loop error for session %s", session_id)

        await asyncio.sleep(poll_interval)

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
        raise HTTPException(403, "OANDA API key not configured")

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
    return [
        {"session_id": s["session_id"], "pair": s["pair"], "model_type": s["model_type"],
         "timeframe": s["timeframe"], "status": s["status"], "created_at": s["created_at"]}
        for s in active_sessions.values()
    ]


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
    from pipeline.attribution import compute_attribution_from_session
    report = compute_attribution_from_session(session)
    return report.to_dict()


@live_router.get("/sessions")
async def list_live_sessions():
    return [
        {
            "session_id": s["session_id"], "pair": s["pair"],
            "model_type": s["model_type"], "timeframe": s["timeframe"],
            "mode": s.get("mode", "demo"), "status": s["status"],
            "killed": s.get("kill_reason", "") != "", "created_at": s["created_at"],
        }
        for s in live_sessions.values()
    ]
