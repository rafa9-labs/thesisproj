"""Live metrics endpoints for committee sessions.

GET  /live/committee/{session_id}/metrics     — metrics snapshot
WS   /live/committee/{session_id}/metrics/stream — streaming every 60s
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

logger = __import__("logging").getLogger(__name__)

router = APIRouter(tags=["live_metrics"])


def _get_committee_session(session_id: str) -> dict | None:
    """Import active_sessions from live.py — avoids circular imports."""
    from api.routers.live import active_sessions

    return active_sessions.get(session_id)


@router.get("/live/committee/{session_id}/metrics")
async def get_committee_metrics(session_id: str):
    """Get a full metrics snapshot for a committee live session."""
    session = _get_committee_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    from api.routers.live import build_committee_metrics_snapshot

    try:
        metrics = build_committee_metrics_snapshot(session)
    except Exception:
        logger.exception("Failed to build metrics for session %s", session_id)
        metrics = {
            "session_id": session_id,
            "error": "metrics unavailable",
        }

    return metrics


@router.websocket("/live/committee/{session_id}/metrics/stream")
async def metrics_stream(websocket: WebSocket, session_id: str):
    """WebSocket stream pushing metrics every 60 seconds."""
    await websocket.accept()

    session = _get_committee_session(session_id)
    if not session:
        await websocket.send_text(json.dumps({"event": "error", "message": "Session not found"}))
        await websocket.close()
        return

    from api.routers.live import build_committee_metrics_snapshot

    try:
        while session.get("status") == "running":
            try:
                metrics = build_committee_metrics_snapshot(session)
                await websocket.send_text(json.dumps({
                    "event": "metrics_update",
                    "timestamp": int(time.time()),
                    **metrics,
                }, default=str))
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Metrics WebSocket error for session %s", session_id)
    finally:
        try:
            if session.get("status") == "running":
                metrics = build_committee_metrics_snapshot(session)
                await websocket.send_text(json.dumps({
                    "event": "metrics_update",
                    "timestamp": int(time.time()),
                    "killed": True,
                    **metrics,
                }, default=str))
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
