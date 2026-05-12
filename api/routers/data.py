"""Data download and upload endpoints."""
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks

from api.dependencies import get_data_store
from api.schemas.pairs import DownloadRequest, DownloadResponse
from api.services import JobManager
from pipeline.pair_config import VALID_PAIRS, PAIR_REGISTRY, PairConfig

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/download", response_model=DownloadResponse, status_code=202)
def trigger_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    pair = req.pair.upper()

    store = get_data_store()
    jm = JobManager(store)

    # Allow registry pairs OR previously-defined custom pairs
    if pair not in VALID_PAIRS:
        db_pair = store.get_pair(pair)
        if db_pair is None:
            raise HTTPException(
                400,
                f"Unknown pair: '{pair}'. "
                f"Use /pairs/define to register it first, "
                f"or pick from: {VALID_PAIRS}"
            )

    job_id = str(uuid.uuid4())
    jm.create_job(job_id, "download", {"pair": pair, "years": req.years})

    # Run synchronously in-process (no Celery dependency)
    from api.tasks import _download_data_impl
    background_tasks.add_task(_download_data_impl, job_id, pair, req.years)

    return DownloadResponse(job_id=job_id, pair=pair, status="running")


@router.post("/upload")
def upload_csv(
    file: UploadFile = File(...),
    pair: str | None = None,
    timeframe: str | None = None,
):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    filename = file.filename
    inferred_pair = pair
    inferred_tf = timeframe
    if inferred_pair is None or inferred_tf is None:
        import re
        m = re.match(r"^([A-Z]{6})_(\d+)_years_(M\d+|H\d+)_OANDA\.csv$", filename)
        if m:
            inferred_pair = inferred_pair or m.group(1)
            inferred_tf = inferred_tf or m.group(3)
        else:
            if inferred_pair is None or inferred_tf is None:
                raise HTTPException(
                    400,
                    "Could not infer pair/timeframe from filename. "
                    "Please provide ?pair=EURUSD&timeframe=H1 query params "
                    "or name the file like EURUSD_10_years_H1_OANDA.csv",
                )

    pair = inferred_pair.upper()
    timeframe = inferred_tf.upper()

    import pandas as pd
    store = get_data_store()

    df = pd.read_csv(file.file)
    rows = []
    for _, r in df.iterrows():
        rows.append((
            pair,
            timeframe,
            str(r["time"]),
            float(r.get("mid_open", 0) or 0),
            float(r.get("mid_high", 0) or 0),
            float(r.get("mid_low", 0) or 0),
            float(r.get("mid_close", 0) or 0),
            float(r.get("bid_open", 0) or 0),
            float(r.get("bid_close", 0) or 0),
            float(r.get("ask_open", 0) or 0),
            float(r.get("ask_close", 0) or 0),
            float(r.get("spread", 0) or 0),
            int(r.get("volume", 0) or 0),
        ))

    BATCH_SIZE = 50_000
    for i in range(0, len(rows), BATCH_SIZE):
        store.insert_candles_batch(rows[i : i + BATCH_SIZE])

    return {
        "status": "ok",
        "pair": pair,
        "timeframe": timeframe,
        "rows": len(rows),
    }
