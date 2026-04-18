"""WebSocket endpoint for real-time job progress."""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.config import settings

router = APIRouter(tags=["websocket"])


@router.websocket("/backtest/{job_id}/ws")
async def backtest_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()

    try:
        import redis as _redis

        r = _redis.from_url(settings.redis_url)
        pubsub = r.pubsub()
        channel = f"job:{job_id}"
        pubsub.subscribe(channel)

        done_events = {"job_complete", "job_failed", "download_complete", "download_failed"}

        while True:
            msg = pubsub.get_message(timeout=2.0)
            if msg and msg["type"] == "message":
                data = msg["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                parsed = json.loads(data)
                await websocket.send_text(data)

                if parsed.get("event") in done_events:
                    break

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            pubsub.unsubscribe(channel)
            pubsub.close()
        except Exception:
            pass
