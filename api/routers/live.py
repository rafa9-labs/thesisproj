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


class DeployCommitteeRequest(BaseModel):
    pair: str = "EURUSD"
    timeframe: str = "H1"
    initial_equity: float = 10000.0
    committee_config_path: str | None = None
    confidence_threshold: float = 0.55
    lookback_bars: int = 100


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


# ── Committee model training ────────────────────────────────────────

_LIVE_FEATURE_NAMES = [
    "mid_c", "mid_h", "mid_l", "sma_20", "ema_20", "rv_48",
    "rolling_std_20", "rsi_14", "macd_diff", "bb_upper", "bb_lower",
    "bb_pct", "bbw", "atr_14", "adx_14",
]


def _train_committee_models(
    pair: str, timeframe: str, model_types: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Train one model per unique type on full EURUSD history.

    Returns (trained_models: {type: model_obj}, feature_names: list[str]).
    The feature set matches what LiveCommitteeRunner._build_features() produces.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from models.registry import build_model

    csv_path = Path(f"csv_data/{pair}_10_years_{timeframe}_OANDA.csv")
    if not csv_path.exists():
        csv_path = Path("csv_data/EURUSD_10_years_H1_OANDA.csv")
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "mid_open": "mid_o", "mid_high": "mid_h",
        "mid_low": "mid_l", "mid_close": "mid_c",
    })

    price = df["mid_c"].astype(np.float64)
    n = len(df)

    features_df = pd.DataFrame(index=df.index)
    features_df["mid_c"] = price
    features_df["mid_h"] = df["mid_h"].astype(np.float64)
    features_df["mid_l"] = df["mid_l"].astype(np.float64)
    features_df["sma_20"] = price.rolling(20).mean()
    features_df["ema_20"] = price.ewm(span=20, adjust=False).mean()

    rets = np.log(price / price.shift(1)).fillna(0.0).astype(np.float64)
    features_df["rv_48"] = np.sqrt(rets.rolling(48).apply(lambda x: (x ** 2).sum(), raw=True))
    features_df["rolling_std_20"] = rets.rolling(20).std()

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    features_df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    features_df["macd_diff"] = ema12 - ema26

    bb_sma = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    features_df["bb_upper"] = bb_sma + 2.0 * bb_std
    features_df["bb_lower"] = bb_sma - 2.0 * bb_std
    features_df["bb_pct"] = (
        (price - features_df["bb_lower"]) /
        (features_df["bb_upper"] - features_df["bb_lower"] + 1e-10)
    )
    features_df["bbw"] = (
        (features_df["bb_upper"] - features_df["bb_lower"]) / (price + 1e-10)
    )

    hl = df["mid_h"] - df["mid_l"]
    features_df["atr_14"] = hl.rolling(14).mean()
    features_df["adx_14"] = _compute_simplified_adx_local(df)

    features_df = features_df.dropna()

    fwd_returns = rets.shift(-1).values
    labels = np.ones(len(features_df), dtype=np.int32)
    threshold = 0.0001
    labels[fwd_returns[:len(features_df)] > threshold] = 2
    labels[fwd_returns[:len(features_df)] < -threshold] = 0
    labels = labels[:len(features_df)]
    labels[-1] = 1

    X = features_df[_LIVE_FEATURE_NAMES].to_numpy(np.float32)
    y = labels

    valid = y != -1
    X, y = X[valid], y[valid]

    trained = {}
    for mtype in model_types:
        try:
            model = build_model(mtype, use_proba=True, n_features=X.shape[1])
            model.fit(X, y)
            trained[mtype] = model
            logger.info("Committee model %s trained on %d samples", mtype, len(X))
        except Exception:
            logger.exception("Failed to train committee model %s", mtype)

    return trained, _LIVE_FEATURE_NAMES


def _compute_simplified_adx_local(df: "pd.DataFrame") -> "pd.Series":
    import numpy as np
    import pandas as pd
    window = 14
    high = df["mid_h"].astype(np.float64)
    low = df["mid_l"].astype(np.float64)
    up_move = high.diff().clip(lower=0)
    down_move = (-low).diff().clip(lower=0)
    atr = (high - low).rolling(window, min_periods=1).mean().replace(0, np.nan)
    pdi = 100.0 * up_move.ewm(alpha=1.0 / window, adjust=False).mean() / atr
    mdi = 100.0 * down_move.ewm(alpha=1.0 / window, adjust=False).mean() / atr
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False).mean()


# ── Deploy committee endpoint ────────────────────────────────────────

@router.post("/live/deploy-committee")
async def deploy_committee_session(req: DeployCommitteeRequest):
    pair = req.pair.upper().strip()
    timeframe = req.timeframe

    # 1. Load committee config
    config_json = None
    if req.committee_config_path:
        config_path = Path(req.committee_config_path)
    else:
        config_path = Path("results/full_cycle")
        candidates = sorted(config_path.glob("fullcycle_*"), reverse=True)
        for c in candidates:
            final_cfg = c / "committee_config_final.json"
            if final_cfg.exists():
                config_path = final_cfg
                break
            base_cfg = c / "committee_config.json"
            if base_cfg.exists():
                config_path = base_cfg
                break

    if not config_path or not config_path.exists():
        raise HTTPException(404, "No committee config found. Run the Full Cycle first.")

    with open(config_path) as f:
        config_json = json.load(f)

    from pipeline.committee_builder import CommitteeConfig
    committee_config = CommitteeConfig.from_dict(config_json)

    # 2. Get unique model types
    unique_models = list(set(committee_config.all_models()))
    if not unique_models:
        raise HTTPException(400, "Committee config has no models")

    # 3. Train models
    trained_models, feature_names = _train_committee_models(
        pair, timeframe, unique_models,
    )
    if not trained_models:
        raise HTTPException(500, "Failed to train any committee models")

    # 4. Create runner
    from trading.live_committee_runner import LiveCommitteeRunner
    from pipeline.regime_utils import RegimeConfig

    runner = LiveCommitteeRunner(
        config=committee_config,
        models=trained_models,
        feature_names=feature_names,
        regime_cfg=RegimeConfig(),
        confidence_threshold=req.confidence_threshold,
        lookback_bars=req.lookback_bars,
    )
    runner.start()

    # 5. Create session
    session_id = str(uuid.uuid4())[:8]
    session = {
        "session_id": session_id,
        "pair": pair,
        "model": "committee",
        "timeframe": timeframe,
        "status": "running",
        "equity": req.initial_equity,
        "position": "FLAT",
        "signal_count": 0,
        "last_mid": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": runner,
        "trained_models": trained_models,
        "committee_config": committee_config,
        "backtester": None,
        "ws_queues": [],
        "model_obj": None,
    }

    active_sessions[session_id] = session
    asyncio.create_task(_committee_signal_loop(session_id))

    return {
        "session_id": session_id,
        "pair": pair,
        "model": "committee",
        "models": unique_models,
        "timeframe": timeframe,
        "status": "running",
        "equity": req.initial_equity,
        "signal_count": 0,
        "features": feature_names,
        "lookback_bars": req.lookback_bars,
    }


async def _committee_signal_loop(session_id: str):
    """Background loop: poll prices → build bar → runner.process_bar() → broadcast."""
    session = active_sessions.get(session_id)
    if not session:
        return

    pair = session["pair"]
    timeframe = session["timeframe"]
    runner = session.get("runner")
    store = get_data_store()
    ws_queues = session.get("ws_queues", [])

    tf_seconds = {"M15": 900, "M30": 1800, "H1": 3600, "H4": 14400}
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
                    mid_price = (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0

            candles_df = store.get_latest_candles(pair, timeframe, 100)
            if candles_df.empty:
                await asyncio.sleep(poll_interval)
                continue

            bar = {
                "mid_c": float(candles_df.iloc[-1]["mid_close"])
                if "mid_close" in candles_df.columns else (
                    float(candles_df.iloc[-1]["mid_c"]) if "mid_c" in candles_df.columns else mid_price or 1.10
                ),
                "mid_h": float(candles_df.iloc[-1].get("mid_high", candles_df.iloc[-1].get("mid_h", candles_df.iloc[-1].get("mid_c", 1.10)))),
                "mid_l": float(candles_df.iloc[-1].get("mid_low", candles_df.iloc[-1].get("mid_l", candles_df.iloc[-1].get("mid_c", 1.10)))),
                "mid_o": float(candles_df.iloc[-1].get("mid_open", candles_df.iloc[-1].get("mid_o", candles_df.iloc[-1].get("mid_c", 1.10)))),
                "spread": float(candles_df.iloc[-1].get("spread", 0.0001)),
                "returns": float(np.log(mid_price / session.get("last_mid", mid_price))) if session.get("last_mid") and mid_price else 0.0,
                "timestamp": int(time.time()),
            }

            live_signal = runner.process_bar(bar)

            direction = "FLAT"
            confidence = 50.0
            regime = "unknown"
            active_models = []

            if live_signal is not None:
                if live_signal.signal == 1:
                    direction = "LONG"
                elif live_signal.signal == -1:
                    direction = "SHORT"
                confidence = live_signal.confidence * 100
                regime = live_signal.regime
                active_models = live_signal.active_models

            prev_equity = session["equity"]
            pos = 1 if direction == "LONG" else -1 if direction == "SHORT" else 0
            if session["last_mid"] is not None and session["last_mid"] > 0 and mid_price:
                ret = (mid_price - session["last_mid"]) / session["last_mid"]
                pnl = pos * ret * prev_equity
            else:
                pnl = 0.0

            new_equity = prev_equity + pnl

            signal_msg = {
                "event": "signal",
                "session_id": session_id,
                "time": int(time.time()),
                "direction": direction,
                "confidence": round(confidence, 1),
                "price": round(mid_price, 10) if mid_price else 0,
                "equity": round(new_equity, 2),
                "pnl": round(pnl, 2),
                "position": direction,
                "regime": regime,
                "active_models": active_models,
            }

            session["equity"] = new_equity
            session["position"] = direction
            session["last_mid"] = mid_price
            session["signal_count"] = session.get("signal_count", 0) + 1

            for q in list(ws_queues):
                try:
                    q.put_nowait(signal_msg)
                except Exception:
                    pass

        except Exception:
            logger.exception("Committee signal loop error for session %s", session_id)

        await asyncio.sleep(poll_interval)

    session["status"] = "stopped"
    runner.stop()
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