"""Data download endpoint."""
import uuid

from fastapi import APIRouter, HTTPException

from api.dependencies import get_data_store
from api.schemas.pairs import DownloadRequest, DownloadResponse
from api.services import JobManager
from api.tasks import download_data_task
from pipeline.pair_config import VALID_PAIRS

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/download", response_model=DownloadResponse, status_code=202)
def trigger_download(req: DownloadRequest):
    pair = req.pair.upper()
    if pair not in VALID_PAIRS:
        raise HTTPException(400, f"Unknown pair: {pair}. Available: {VALID_PAIRS}")

    store = get_data_store()
    jm = JobManager(store)

    job_id = str(uuid.uuid4())
    jm.create_job(job_id, "download", {"pair": pair, "years": req.years})

    download_data_task.delay(job_id, pair, req.years)

    return DownloadResponse(job_id=job_id, pair=pair, status="pending")
