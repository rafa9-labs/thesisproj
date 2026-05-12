"""WebSocket endpoint for real-time job progress."""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.config import settings

router = APIRouter(tags=["websocket"])

_DONE_EVENTS = {"job_complete", "job_failed", "download_complete", "download_failed"}


async def _ws_redis_loop(websocket: WebSocket, job_id: str):
    import redis as _redis
    r = _redis.from_url(settings.redis_url)
    pubsub = r.pubsub()
    channel = f"job:{job_id}"
    pubsub.subscribe(channel)

    while True:
        msg = pubsub.get_message(timeout=2.0)
        if msg and msg["type"] == "message":
            data = msg["data"]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            parsed = json.loads(data)
            await websocket.send_text(data)
            if parsed.get("event") in _DONE_EVENTS:
                break
        await asyncio.sleep(0.1)

    pubsub.unsubscribe(channel)
    pubsub.close()


async def _ws_polling_loop(websocket: WebSocket, job_id: str):
    from api.tasks import get_job_events
    cursor = 0
    while True:
        events = get_job_events(job_id, after=cursor)
        for evt in events:
            await websocket.send_text(json.dumps(evt, default=str))
            if evt.get("event") in _DONE_EVENTS:
                return
        cursor += len(events)
        await asyncio.sleep(0.5)


@router.websocket("/backtest/{job_id}/ws")
async def backtest_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()

    try:
        try:
            import redis as _redis
            r_test = _redis.from_url(settings.redis_url)
            r_test.ping()
            r_test.close()
            await _ws_redis_loop(websocket, job_id)
        except Exception:
            await _ws_polling_loop(websocket, job_id)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass