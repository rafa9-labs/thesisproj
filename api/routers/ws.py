"""WebSocket endpoint for real-time job progress."""
import asyncio
import json
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.config import settings

router = APIRouter(tags=["websocket"])
_ws_connections: dict[str, set[WebSocket]] = defaultdict(set)


def _get_ws_connections(job_id: str) -> int:
    return len(_ws_connections.get(job_id, set()))

_DONE_EVENTS = {"job_complete", "job_failed", "download_complete", "download_failed"}


async def _ws_polling_loop(websocket: WebSocket, job_id: str):
    from api.tasks import get_job_events

    cursor = 0
    idle_rounds = 0
    done_sent = False
    while True:
        events = get_job_events(job_id, after=cursor)
        for evt in events:
            await websocket.send_text(json.dumps(evt, default=str))
            # Only exit if we just sent a *newly-fetched* done event, not a historical one
            if evt.get("event") in _DONE_EVENTS:
                done_sent = True
        if events:
            cursor += len(events)
            idle_rounds = 0
        else:
            idle_rounds += 1
            # If we already sent a done event and no new events arrived, the job is truly finished
            if done_sent and idle_rounds >= 2:
                return
            # After 30 idle rounds (~15s) with no events, check if job still running.
            if idle_rounds >= 30:
                from api.config import settings as s
                from api.services import JobManager as JM
                from pipeline.data_sqlite import DataStore as DS
                try:
                    jm = JM(DS(s.db_full_path))
                    job = jm.get_job(job_id)
                    if job and job.get("status") in ("pending", "running"):
                        idle_rounds = 0  # keep polling
                except Exception:
                    pass
        await asyncio.sleep(0.5)


@router.websocket("/backtest/{job_id}/ws")
async def backtest_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    _ws_connections[job_id].add(websocket)
    print(f"[WS-SRV] Client connected, job={job_id} (total: {len(_ws_connections[job_id])})", flush=True)

    try:
        await _ws_polling_loop(websocket, job_id)
    except WebSocketDisconnect:
        print(f"[WS-SRV] job={job_id} Client disconnected", flush=True)
    except Exception as e:
        print(f"[WS-SRV] job={job_id} Unexpected error: {type(e).__name__}: {e}", flush=True)
    finally:
        _ws_connections[job_id].discard(websocket)
        if not _ws_connections[job_id]:
            del _ws_connections[job_id]
