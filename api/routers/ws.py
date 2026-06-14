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
            if evt.get("event") in _DONE_EVENTS:
                done_sent = True
        if events:
            cursor += len(events)
            idle_rounds = 0
        else:
            idle_rounds += 1
            if done_sent and idle_rounds >= 2:
                return
            if idle_rounds >= 30:
                from api.config import settings as s
                from api.services import JobManager as JM
                from pipeline.data_sqlite import DataStore as DS
                try:
                    jm = JM(DS(s.db_full_path))
                    job = jm.get_job(job_id)
                    if job and job.get("status") in ("pending", "running"):
                        idle_rounds = 0
                except Exception:
                    pass
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return


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
            # Last client disconnected — abort running backtest to avoid orphaned runs
            try:
                from api.config import settings as s
                from api.services import JobManager as JM
                from pipeline.data_sqlite import DataStore as DS
                jm = JM(DS(s.db_full_path))
                job = jm.get_job(job_id)
                if job and job.get("status") in ("pending", "running"):
                    jm.force_stop_job(job_id)
                    from api.tasks import revoke_task
                    revoke_task(job_id)
                    print(f"[WS-SRV] job={job_id} Client disconnected, backtest aborted", flush=True)
            except Exception as e:
                print(f"[WS-SRV] job={job_id} abort failed: {e}", flush=True)
            del _ws_connections[job_id]


# ── News WebSocket ────────────────────────────────────────────────────

_news_ws_clients: set[WebSocket] = set()
_news_last_push: float = 0.0


async def broadcast_news_event(event_type: str, payload: dict):
    alive: set[WebSocket] = set()
    msg = json.dumps({"event": event_type, **payload}, default=str)
    for ws in _news_ws_clients:
        try:
            await ws.send_text(msg)
            alive.add(ws)
        except Exception:
            pass
    _news_ws_clients.clear()
    _news_ws_clients.update(alive)


@router.websocket("/news/ws")
async def news_ws(websocket: WebSocket):
    await websocket.accept()
    _news_ws_clients.add(websocket)
    print(f"[WS-NEWS] Client connected (total: {len(_news_ws_clients)})", flush=True)
    try:
        while True:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            try:
                from news.scraper import NewsScraper
                scraper = NewsScraper()
                articles = scraper.fetch_all()
                count = len(articles)
                await websocket.send_text(json.dumps({
                    "event": "news_sync",
                    "article_count": count,
                    "cached": count > 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, default=str))
            except Exception:
                pass
    except WebSocketDisconnect:
        print("[WS-NEWS] Client disconnected", flush=True)
    except Exception as e:
        print(f"[WS-NEWS] Unexpected error: {type(e).__name__}: {e}", flush=True)
    finally:
        _news_ws_clients.discard(websocket)


# Provide correct timezone import for the news ws endpoint
from datetime import timezone
