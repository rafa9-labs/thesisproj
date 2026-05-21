"""Live trading deployment endpoints.

POST /live/deploy       - Start a live session (trains model on recent data, begins signal stream)
WS   /live/{session_id}/ws - Stream real-time signals
POST /live/{session_id}/stop - Stop a live session
GET  /live/{session_id}/status - Get session status
GET  /live/sessions     - List active sessions
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from api.dependencies import get_data_store
from api.routers.prices import _get_oanda_credentials, _oanda_api_call

logger = logging.getLogger(__name__)

router = APIRouter(tags=["live"])


class DeployRequest(BaseModel):
    pair: str
    model: str = "logistic"
    timeframe: str = "M30"
    initial_equity: float = 10000.0
    model_id: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    pair: str
    model: str
    timeframe: str
    status: str
    equity: float
    position: str
    signal_count: int
    created_at: str


active_sessions: Dict[str, dict] = {}


def _run_backtest_for_model(pair: str, model: str, timeframe: str):
    """Run a lightweight backtest on recent data to obtain a trained model.

    Returns the trained model object and config, or None if backtest fails.
    """
    try:
        from pipeline.backtester.composed import MLBacktester
        from config import PIPELINE_CONSTANTS as _PC

        cfg = {
            "pair": pair,
            "timeframe": timeframe,
            "model_type": model,
            "n_months": 3,
            "use_WFO": False,
            "use_proba": True,
            "features_config": _PC.get("features_config", {}),
            "search_space": _PC.get("search_space", {}),
        }

        bt = MLBacktester(config=cfg)
        bt.load_data()
        result = bt.run_strategy(config=cfg, models_to_test=[model], n_trials=1, n_startup_trials=1)

        if result is None or result[0] is None or result[0].empty:
            logger.warning("Backtest for %s/%s returned no results", pair, model)
            return None

        trained_model = getattr(bt, "model", None)
        bt_config = getattr(bt, "config", cfg)

        return {
            "model": trained_model,
            "config": bt_config,
            "backtester": bt,
        }
    except Exception:
        logger.exception("Failed to run backtest for %s/%s", pair, model)
        return None


def _predict_signal(session: dict, candles_df):
    """Generate a signal prediction from a trained model for the latest bar."""
    bt = session.get("backtester")
    trained_model = session.get("model_obj")

    if bt is None or trained_model is None:
        return None

    try:
        features_df = bt._compute_features(candles_df)
        if features_df is None or features_df.empty:
            return None

        last_row = features_df.iloc[[-1]]
        feature_cols = [c for c in features_df.columns if c not in ("time", "target", "side")]

        if hasattr(trained_model, "predict_proba"):
            proba = trained_model.predict_proba(last_row[feature_cols])
            confidence = float(proba[0].max()) * 100
            prediction = int(proba[0].argmax())
        elif hasattr(trained_model, "predict"):
            prediction = int(trained_model.predict(last_row[feature_cols])[0])
            confidence = 60.0
        else:
            return None

        if prediction == 1:
            direction = "LONG"
        elif prediction == 0 or prediction == -1:
            direction = "SHORT"
        else:
            direction = "FLAT"

        return {
            "direction": direction,
            "confidence": min(confidence, 99.0),
        }
    except Exception:
        logger.exception("Signal prediction failed")
        return None


async def _signal_loop(session_id: str):
    """Background loop that fetches live candles and runs predictions."""
    session = active_sessions.get(session_id)
    if not session:
        return

    pair = session["pair"]
    timeframe = session["timeframe"]
    model = session["model"]
    store = get_data_store()
    ws_queues = session.get("ws_queues", [])

    tf_seconds = {"M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400}
    poll_interval = min(tf_seconds.get(timeframe, 1800) // 4, 60)

    while session.get("status") == "running":
        try:
            raw_prices, source = _oanda_api_call(pair.replace("", ""))

            mid_price = None
            if source == "oanda" and raw_prices:
                p0 = raw_prices[0]
                bids = p0.get("bids", [])
                asks = p0.get("asks", [])
                if bids and asks:
                    bid = float(bids[0]["price"])
                    ask = float(asks[0]["price"])
                    mid_price = (bid + ask) / 2.0

            if mid_price is None:
                candles_df = store.get_latest_candles(pair, timeframe, 60)
                if not candles_df.empty:
                    mid_price = float(candles_df.iloc[-1]["mid_close"])

            if mid_price is None:
                await asyncio.sleep(poll_interval)
                continue

            signal_result = _predict_signal(session, store.get_latest_candles(pair, timeframe, 60))

            direction = signal_result["direction"] if signal_result else "FLAT"
            confidence = signal_result["confidence"] if signal_result else 50.0

            prev_equity = session["equity"]
            pos_multiplier = 1 if direction == "LONG" else -1 if direction == "SHORT" else 0
            if session["last_mid"] is not None and session["last_mid"] > 0:
                ret = (mid_price - session["last_mid"]) / session["last_mid"]
                pnl = pos_multiplier * ret * prev_equity
            else:
                pnl = 0.0

            new_equity = prev_equity + pnl
            now_epoch = int(time.time())

            signal_msg = {
                "event": "signal",
                "session_id": session_id,
                "time": now_epoch,
                "direction": direction,
                "confidence": round(confidence, 1),
                "price": round(mid_price, 10),
                "equity": round(new_equity, 2),
                "pnl": round(pnl, 2),
                "position": direction,
            }

            session["equity"] = new_equity
            session["position"] = direction
            session["last_mid"] = mid_price
            session["signal_count"] += 1

            for q in list(ws_queues):
                try:
                    q.put_nowait(signal_msg)
                except Exception:
                    pass

        except Exception:
            logger.exception("Signal loop error for session %s", session_id)

        await asyncio.sleep(poll_interval)

    session["status"] = "stopped"
    stop_msg = {
        "event": "stopped",
        "session_id": session_id,
        "equity": session["equity"],
        "signal_count": session["signal_count"],
    }
    for q in list(ws_queues):
        try:
            q.put_nowait(stop_msg)
        except Exception:
            pass


@router.post("/live/deploy", response_model=SessionInfo)
async def deploy_live_session(req: DeployRequest):
    pair = req.pair.upper().strip()

    token, account_id = _get_oanda_credentials()
    if not token or not account_id:
        raise HTTPException(403, "OANDA API key not configured. Add it in Settings.")

    session_id = str(uuid.uuid4())[:8]

    model_obj = None
    bt = None

    if req.model_id:
        # Load from saved snapshot
        from pipeline.model_persistence import load_snapshot, load_model_only, read_metadata
        from pipeline.model_registry_disk import get_all_deployed
        from api.config import settings

        rows = get_all_deployed(settings.db_full_path)
        snapshot_path = None
        for r in rows:
            if r.get("id") == req.model_id:
                snapshot_path = r.get("snapshot_path")
                break
        if not snapshot_path or not os.path.exists(snapshot_path):
            raise HTTPException(404, f"Snapshot not found for model {req.model_id}")

        try:
            model_obj = load_model_only(snapshot_path)
            model_type = read_metadata(snapshot_path).get("model_type", "unknown")
        except Exception:
            raise HTTPException(500, "Failed to load saved model")
    else:
        # Train fresh (existing behavior)
        bt_result = _run_backtest_for_model(pair, req.model, req.timeframe)
        if bt_result:
            model_obj = bt_result.get("model")
            bt = bt_result.get("backtester")
            model_type = bt_result.get("config", {}).get("model_type", req.model)

    if model_obj is None:
        raise HTTPException(500, "Failed to obtain model for live trading")

    session = {
        "session_id": session_id,
        "pair": pair,
        "model": req.model,
        "timeframe": req.timeframe,
        "status": "running",
        "equity": req.initial_equity,
        "position": "FLAT",
        "signal_count": 0,
        "last_mid": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_obj": model_obj,
        "backtester": bt,
        "ws_queues": [],
    }

    active_sessions[session_id] = session

    asyncio.create_task(_signal_loop(session_id))

    return SessionInfo(
        session_id=session_id,
        pair=pair,
        model=req.model,
        timeframe=req.timeframe,
        status="running",
        equity=session["equity"],
        position="FLAT",
        signal_count=0,
        created_at=session["created_at"],
    )


@router.websocket("/live/{session_id}/ws")
async def live_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_text(json.dumps({"event": "error", "message": "Session not found"}))
        await websocket.close()
        return

    q: asyncio.Queue = asyncio.Queue()
    session["ws_queues"].append(q)

    try:
        while session["status"] == "running":
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_text(json.dumps(msg))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"event": "heartbeat", "time": int(time.time())}))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error for session %s", session_id)
    finally:
        if q in session.get("ws_queues", []):
            session["ws_queues"].remove(q)


@router.post("/live/{session_id}/stop")
async def stop_live_session(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    session["status"] = "stopping"
    model_obj = session.get("model_obj")
    if model_obj is not None:
        try:
            if hasattr(model_obj, "close"):
                model_obj.close()
        except Exception:
            pass

    equity = session["equity"]
    signal_count = session["signal_count"]

    return {
        "session_id": session_id,
        "status": "stopped",
        "equity": equity,
        "signal_count": signal_count,
    }


@router.get("/live/{session_id}/status", response_model=SessionInfo)
async def get_session_status(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    return SessionInfo(
        session_id=session["session_id"],
        pair=session["pair"],
        model=session["model"],
        timeframe=session["timeframe"],
        status=session["status"],
        equity=session["equity"],
        position=session["position"],
        signal_count=session["signal_count"],
        created_at=session["created_at"],
    )


@router.get("/live/sessions")
async def list_sessions():
    return [
        {
            "session_id": s["session_id"],
            "pair": s["pair"],
            "model": s["model"],
            "timeframe": s["timeframe"],
            "status": s["status"],
            "equity": s["equity"],
            "signal_count": s["signal_count"],
            "created_at": s["created_at"],
        }
        for s in active_sessions.values()
    ]