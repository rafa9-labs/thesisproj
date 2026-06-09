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
    live_news_blend_enabled: bool = False
    live_news_blend_weight: float = 0.10


class DeployCommitteeRequest(BaseModel):
    pair: str = "EURUSD"
    timeframe: str = "H1"
    initial_equity: float = 10000.0
    committee_config_path: str | None = None
    full_cycle_job_id: str | None = None
    confidence_threshold: float = 0.55
    lookback_bars: int = 100
    live_news_blend_enabled: bool = False
    live_news_blend_weight: float = 0.10
    execution_mode: str = "paper"


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
    """Generate a signal prediction from a trained model for the latest bar.

    Applies news sentiment blending post-prediction if enabled in session config.
    """
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

        raw_signal = 1.0 if direction == "LONG" else -1.0 if direction == "SHORT" else 0.0

        blend_cfg = {
            "live_news_blend_enabled": session.get("live_news_blend_enabled", False),
            "live_news_blend_weight": session.get("live_news_blend_weight", 0.10),
        }
        if blend_cfg["live_news_blend_enabled"]:
            from pipeline.backtester.real_trading_mixin import RealTradingMixin
            blended = RealTradingMixin._blend_news_sentiment(raw_signal, session["pair"], blend_cfg)
            if blended > 0.3:
                direction = "LONG"
                confidence = max(confidence, abs(blended) * 100)
            elif blended < -0.3:
                direction = "SHORT"
                confidence = max(confidence, abs(blended) * 100)
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
        "live_news_blend_enabled": req.live_news_blend_enabled,
        "live_news_blend_weight": req.live_news_blend_weight,
    }

    if req.live_news_blend_enabled:
        try:
            from news.scraper import NewsScraper
            from news.sentiment import SentimentAnalyzer
            scraper = NewsScraper()
            articles = scraper.fetch_all()
            if articles:
                engine = SentimentAnalyzer(backend="vader")
                engine.score_articles(articles, pair=pair)
        except Exception:
            logger.warning("Failed to pre-warm news cache for session %s", session_id)

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


# ── Trust score → sizing multiplier ──────────────────────────────────────

def _trust_to_multiplier(trust_score: float) -> float:
    if trust_score >= 0.80: return 1.00
    if trust_score >= 0.70: return 0.80
    if trust_score >= 0.60: return 0.60
    if trust_score >= 0.50: return 0.40
    if trust_score >= 0.40: return 0.20
    return 0.00


# ── Committee model training ────────────────────────────────────────

_LIVE_FEATURE_NAMES = [
    "sma_20", "ema_20", "rv_48",
    "rsi_14", "macd_diff",
    "bb_upper_20", "bb_lower_20", "bb_pct_20", "bbw_20",
    "atr_14", "adx_14",
]


def _train_committee_models(
    pair: str, timeframe: str, model_types: list[str],
    feature_names: Optional[list[str]] = None,
    model_params: Optional[dict[str, dict]] = None,
) -> tuple[dict[str, Any], list[str]]:
    """Train one model per unique type on full history.

    Parameters
    ----------
    feature_names : list[str] or None
        Feature columns to train on. If None, defaults to _LIVE_FEATURE_NAMES.
    model_params : dict or None
        model_type -> {param: value} dict with unprefixed HPO-tuned
        hyperparameters. Applied at build time.

    Returns
    -------
    (trained_models: {type: model_obj}, feature_names: list[str])
    """
    from pathlib import Path
    from models.registry import build_model
    from pipeline.feature_sweep import compute_feature_matrix, FEATURE_NAMES
    from pipeline.expert_profiler import _reprefix_params
    import pandas as pd
    import numpy as np

    use_features = list(feature_names) if feature_names else list(_LIVE_FEATURE_NAMES)
    model_params = dict(model_params or {})

    csv_path = Path(f"csv_data/{pair}_10_years_{timeframe}_OANDA.csv")
    if not csv_path.exists():
        csv_path = Path("csv_data/EURUSD_10_years_H1_OANDA.csv")
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "mid_open": "mid_o", "mid_high": "mid_h",
        "mid_low": "mid_l", "mid_close": "mid_c",
    })

    feature_matrix = compute_feature_matrix(df, feature_names=use_features, include_ohlc=False)
    X = feature_matrix.to_numpy(np.float32)
    n_rows = X.shape[0]

    if "mid_c" in feature_matrix.columns:
        price = feature_matrix["mid_c"].to_numpy(dtype=np.float64).ravel()
    else:
        price = df["mid_c"].loc[feature_matrix.index].to_numpy(dtype=np.float64).ravel()
    rets_vals = np.zeros(n_rows, dtype=np.float64)
    rets_vals[1:] = np.log(price[1:] / price[:-1])

    labels = np.ones(n_rows, dtype=np.int32)
    threshold = 0.0001
    labels[rets_vals > threshold] = 2
    labels[rets_vals < -threshold] = 0
    labels[-1] = 1

    valid = (labels != -1)
    X, labels = X[valid], labels[valid]

    trained = {}
    for mtype in model_types:
        try:
            params = model_params.get(mtype, {})
            model = build_model(
                mtype, use_proba=True, n_features=X.shape[1],
                **_reprefix_params(params, mtype),
            )
            model.fit(X, labels)
            trained[mtype] = model
            logger.info("Committee model %s trained on %d samples with %d features",
                         mtype, len(X), X.shape[1])
        except Exception:
            logger.exception("Failed to train committee model %s", mtype)

    return trained, use_features


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
        for c in candidates:
            final_cfg = c / "committee_config_final.json"
            if final_cfg.exists():
                config_path = final_cfg
                break
            base_cfg = c / "committee_config.json"
            if base_cfg.exists():
                config_path = base_cfg
                break
        logger.warning("No full_cycle_job_id specified — auto-discovering latest config. This may deploy an unintended committee if pipeline runs overlap.")

    if not config_path or not config_path.exists():
        raise HTTPException(404, "No committee config found. Run the Full Cycle first.")

    with open(config_path) as f:
        config_json = json.load(f)

    from pipeline.committee_builder import CommitteeConfig
    committee_config = CommitteeConfig.from_dict(config_json)

    model_params = committee_config.model_params or config_json.get("model_params", {})

    # Detect the parent full-cycle job dir for locked features + snapshot + trust score
    if parent_job_dir is None:
        parent_job_dir = config_path.parent if config_path.parent.name.startswith("fullcycle_") else None
    if parent_job_dir is None:
        parent_job_dir = config_path.parent.parent if "full_cycle" in str(config_path) else None

    # Load trust score from the same full-cycle job
    trust_multiplier = 1.0
    trust_data = None
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
                        f"Committee trust_score={ts:.2f} — deployment rejected. "
                        "Re-run the Full Cycle with different models or parameters.")
                trust_multiplier = _trust_to_multiplier(ts)
                logger.info("Trust multiplier: %.2f (score=%.3f, action=%s)",
                             trust_multiplier, ts, action)
            except HTTPException:
                raise
            except Exception:
                logger.warning("Failed to load trust score — using full sizing")
                pass

    # 2. Load locked features if available
    locked_features = None
    locked_features_path = Path("results/locked_features.json")
    if locked_features_path.exists():
        try:
            from pipeline.feature_sweep import load_locked_features
            locked_features = load_locked_features(str(locked_features_path))
        except Exception:
            pass

    # 3. Get unique model types
    unique_models = list(set(committee_config.all_models()))
    if not unique_models:
        raise HTTPException(400, "Committee config has no models")

    trained_models: dict[str, Any] = {}
    feature_names = locked_features if locked_features else _LIVE_FEATURE_NAMES

    # 4. Try loading saved snapshot (fast path, byte-for-byte identical to validation)
    snapshot_loaded = False
    if parent_job_dir:
        snapshot_dir = parent_job_dir / "committee_snapshot"
        if snapshot_dir.exists() and (snapshot_dir / "manifest.json").exists():
            try:
                import joblib
                with open(snapshot_dir / "manifest.json") as f:
                    manifest = json.load(f)
                snapshot_feature_names = manifest.get("feature_names", list(feature_names))
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
                    logger.info("Loaded %d models from snapshot %s",
                                 len(trained_models), str(snapshot_dir))
            except Exception:
                logger.exception("Snapshot load failed, falling back to fresh training")

    # 5. Train models (slow path — only if no snapshot loaded)
    if not snapshot_loaded:
        trained_models, feature_names = _train_committee_models(
            pair, timeframe, unique_models,
            feature_names=locked_features if locked_features else _LIVE_FEATURE_NAMES,
            model_params=model_params,
        )
    if not trained_models:
        raise HTTPException(500, "Failed to obtain committee models")

    # 5.5 Load meta-learner if available
    meta_learner = None
    if parent_job_dir:
        meta_path = parent_job_dir / "committee_snapshot" / "meta" / "meta_model.joblib"
        if meta_path.exists():
            try:
                from pipeline.committee_meta import CommitteeMetaLearner
                meta_learner = CommitteeMetaLearner.load(str(meta_path))
                if meta_learner.is_trained:
                    logger.info("Meta-learner loaded from %s (accuracy=%.3f)",
                                 str(meta_path), meta_learner.accuracy)
                else:
                    meta_learner = None
            except Exception:
                logger.warning("Meta-learner load failed — continuing without it")

    # 6. Create runner
    from trading.live_committee_runner import LiveCommitteeRunner
    from pipeline.regime_utils import RegimeConfig

    runner = LiveCommitteeRunner(
        config=committee_config,
        models=trained_models,
        feature_names=list(feature_names),
        regime_cfg=RegimeConfig(),
        confidence_threshold=req.confidence_threshold,
        lookback_bars=req.lookback_bars,
        meta_learner=meta_learner,
    )
    runner.start()

    # 7. Create session
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
        "trust_multiplier": trust_multiplier,
        "trust_score": trust_data,
        "alert_manager": None,
        "regime_throttle": None,
        "rotation_scheduler": None,
        "effective_multiplier": trust_multiplier,
        "live_news_blend_enabled": req.live_news_blend_enabled,
        "live_news_blend_weight": req.live_news_blend_weight,
    }

    # Initialize alerting if configured
    if hasattr(req, "alert_config") and req.alert_config:
        try:
            from trading.alerting import AlertConfig, AlertManager
            alert_cfg = AlertConfig(**req.alert_config) if isinstance(req.alert_config, dict) else AlertConfig()
            alert_mgr = AlertManager(alert_cfg)
            session["alert_manager"] = alert_mgr
        except Exception:
            pass

    # Initialize regime throttle from Phase 5 coverage report
    if parent_job_dir:
        results_path = parent_job_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    full_results = json.load(f)
                coverage = full_results.get("phase3_regime_coverage", {})
                if coverage:
                    from trading.regime_throttle import RegimeThrottle
                    config_regimes = set(committee_config.regimes.keys())
                    session["regime_throttle"] = RegimeThrottle(coverage, config_regimes)
                    n_throttled = sum(1 for m in session["regime_throttle"]._throttles.values() if m < 1.0)
                    if n_throttled > 0:
                        logger.info("Regime throttle: %d/%d regimes throttled",
                                     n_throttled, len(coverage))
            except Exception:
                logger.warning("Failed to load regime throttle — all regimes trade at full size")
                pass

    # Initialize auto-rotation scheduler
    if snapshot_loaded and parent_job_dir:
        snapshot_dir = parent_job_dir / "committee_snapshot"
        matrix_path = parent_job_dir / "regime_matrix_tuned.json"
        if snapshot_dir.exists() and len(unique_models) > 1:
            try:
                from trading.rotation_scheduler import RotationScheduler
                session["rotation_scheduler"] = RotationScheduler(
                    runner=runner,
                    snapshot_dir=str(snapshot_dir),
                    matrix_path=str(matrix_path) if matrix_path.exists() else None,
                )
                logger.info("Auto-rotation scheduler initialized "
                             "(check interval: %sh, max rotations/24h: %d)",
                             session["rotation_scheduler"].check_interval_hours,
                             session["rotation_scheduler"].max_rotations_per_24h)
            except Exception:
                logger.warning("Failed to initialize rotation scheduler")

    # Initialize LEAN bridge (if execution_mode == "lean")
    session["execution_mode"] = req.execution_mode
    if req.execution_mode == "lean":
        try:
            from trading.lean_bridge import LeanBridge

            async def _lean_alert(alert_type: str, message: str):
                if mgr := session.get("alert_manager"):
                    try:
                        await mgr.send_alert(alert_type, message)
                    except Exception:
                        pass

            lean_url = os.environ.get("LEAN_BRIDGE_URL", "http://localhost:8888")
            session["lean_bridge"] = LeanBridge(
                base_url=lean_url,
                alert_callback=_lean_alert,
            )
            ok = await session["lean_bridge"].health_check()
            if not ok:
                logger.warning("LEAN bridge health check failed — is the LEAN engine running?")
            else:
                logger.info("LEAN bridge initialized at %s — health OK", lean_url)
                session["lean_portfolio_updated"] = 0.0
        except Exception:
            logger.exception("Failed to initialize LEAN bridge")

    if req.live_news_blend_enabled:
        try:
            from news.scraper import NewsScraper
            from news.sentiment import SentimentAnalyzer
            scraper = NewsScraper()
            articles = scraper.fetch_all()
            if articles:
                engine = SentimentAnalyzer(backend="vader")
                engine.score_articles(articles, pair=pair)
        except Exception:
            logger.warning("Failed to pre-warm news cache for committee session %s", session_id)

    active_sessions[session_id] = session
    asyncio.create_task(_committee_signal_loop(session_id))

    # Start rotation scheduler as background task
    if scheduler := session.get("rotation_scheduler"):
        async def _rotation_runner():
            await scheduler.run(
                lambda: session.get("status", "stopped"),
                session.get("alert_manager"),
            )
        asyncio.create_task(_rotation_runner())

    return {
        "session_id": session_id,
        "pair": pair,
        "model": "committee",
        "models": unique_models,
        "timeframe": timeframe,
        "status": "running",
        "equity": req.initial_equity,
        "signal_count": 0,
        "features": list(feature_names),
        "feature_count": len(feature_names),
        "lookback_bars": req.lookback_bars,
        "snapshot_loaded": snapshot_loaded,
        "model_params_count": len(model_params),
    }


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
    last_metrics_time = 0.0

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

            # Apply regime throttle
            effective_mult = session.get("trust_multiplier", 1.0)
            throttle_level = "full"
            if live_signal is not None:
                throttle = session.get("regime_throttle")
                if throttle is not None:
                    regime_mult = throttle.get_multiplier(live_signal.regime)
                    throttle_level = throttle.get_throttle_level(live_signal.regime)
                    effective_mult = effective_mult * regime_mult
                    if regime_mult == 0.0 and live_signal.signal != 0:
                        live_signal.signal = 0
            session["effective_multiplier"] = effective_mult

            direction = "FLAT"
            confidence = 50.0
            regime = "unknown"
            active_models = []

            if live_signal is not None:
                raw_signal_val = float(live_signal.signal)
                blend_cfg = {
                    "live_news_blend_enabled": session.get("live_news_blend_enabled", False),
                    "live_news_blend_weight": session.get("live_news_blend_weight", 0.10),
                }
                if blend_cfg["live_news_blend_enabled"]:
                    from pipeline.backtester.real_trading_mixin import RealTradingMixin
                    blended = RealTradingMixin._blend_news_sentiment(raw_signal_val, pair, blend_cfg)
                    if blended > 0.3:
                        live_signal.signal = 1
                    elif blended < -0.3:
                        live_signal.signal = -1
                    else:
                        live_signal.signal = 0

                if live_signal.signal == 1:
                    direction = "LONG"
                elif live_signal.signal == -1:
                    direction = "SHORT"
                confidence = live_signal.confidence * 100
                regime = live_signal.regime
                active_models = live_signal.active_models

            # ── LEAN Bridge Execution ───────────────────────────
            lean_bridge = session.get("lean_bridge")
            is_lean = session.get("execution_mode") == "lean" and lean_bridge is not None

            if is_lean and direction != "FLAT":
                try:
                    qty = round(effective_mult * 10000, 0)
                    if qty > 0:
                        order = await lean_bridge.submit_order(
                            symbol=pair,
                            direction=1 if direction == "LONG" else -1,
                            quantity=qty,
                            tag=f"kodaquant_{session_id}_{int(time.time())}",
                        )
                        if not order.success:
                            logger.error("LEAN order failed: %s", order.error)
                except Exception:
                    logger.exception("LEAN bridge submit_order failed")

            # ── LEAN Portfolio Sync (every 60s) ─────────────────
            if is_lean:
                now_sync = time.time()
                last_lean_sync = session.get("lean_portfolio_updated", 0.0)
                if now_sync - last_lean_sync >= 60:
                    try:
                        pf = await lean_bridge.get_portfolio()
                        if pf is not None:
                            session["equity"] = pf.total_portfolio_value
                            session["lean_portfolio_updated"] = now_sync
                            long_pos = [p for p in pf.positions if p.quantity > 0]
                            short_pos = [p for p in pf.positions if p.quantity < 0]
                            if long_pos and not short_pos:
                                session["position"] = "LONG"
                            elif short_pos and not long_pos:
                                session["position"] = "SHORT"
                            else:
                                session["position"] = "FLAT"
                    except Exception:
                        logger.exception("LEAN portfolio sync failed")

            # ── Equity / PnL Calculation ────────────────────────
            prev_equity = session["equity"]
            pos = 1 if direction == "LONG" else -1 if direction == "SHORT" else 0
            if session["last_mid"] is not None and session["last_mid"] > 0 and mid_price:
                ret = (mid_price - session["last_mid"]) / session["last_mid"]
                pnl = pos * ret * prev_equity
            else:
                pnl = 0.0

            if not is_lean:
                new_equity = prev_equity + pnl
                session["equity"] = new_equity
            else:
                new_equity = session["equity"]

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
                "throttle_level": throttle_level,
                "effective_multiplier": round(effective_mult, 2),
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

            # Broadcast metrics snapshot every 60 seconds
            now = time.time()
            if now - last_metrics_time >= 60:
                try:
                    metrics = build_committee_metrics_snapshot(session)
                    metrics_msg = {"event": "metrics_update", "timestamp": int(now), **metrics}
                    for q in list(ws_queues):
                        try:
                            q.put_nowait(metrics_msg)
                        except Exception:
                            pass
                    # Check alert thresholds
                    if alert_mgr := session.get("alert_manager"):
                        try:
                            alert_mgr.check_and_notify(metrics, session_id)
                        except Exception:
                            pass
                except Exception:
                    pass
                last_metrics_time = now

        except Exception:
            logger.exception("Committee signal loop error for session %s", session_id)

        await asyncio.sleep(poll_interval)

    session["status"] = "stopped"
    runner.stop()

    # Liquidate LEAN positions and close bridge
    if lean_bridge := session.get("lean_bridge"):
        try:
            await lean_bridge.liquidate_all()
        except Exception:
            logger.exception("LEAN liquidation failed during session stop")
        try:
            await lean_bridge.close()
        except Exception:
            pass

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

    # Liquidate LEAN positions on manual stop
    if lean_bridge := session.get("lean_bridge"):
        try:
            await lean_bridge.liquidate_all()
        except Exception:
            logger.exception("LEAN liquidation failed during stop_live_session")

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