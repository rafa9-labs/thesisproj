"""Data download and upload endpoints."""
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.dependencies import get_data_store
from api.schemas.pairs import DownloadRequest, DownloadResponse
from api.services import JobManager
from api.tasks import download_data_task
from pipeline.pair_config import VALID_PAIRS, PAIR_REGISTRY, PairConfig

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


@router.post("/upload")
def upload_csv(
    file: UploadFile = File(...),
    pair: str | None = None,
    timeframe: str | None = None,
):
    from api.config import settings
    from pipeline.data_migrator import migrate_pair

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    # Derive pair/timeframe from filename if not provided
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

    target_dir = Path(settings.csv_data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / filename
    with dest.open("wb") as f:
        f.write(file.file.read())

    store = get_data_store()

    # Register custom pair if unknown
    if pair not in PAIR_REGISTRY:
        from pipeline.pair_config import PAIR_REGISTRY as _reg, VALID_PAIRS as _vp
        _reg[pair] = PairConfig(
            symbol=pair,
            oanda_name=pair[:3] + "_" + pair[3:],
            pip_value=0.0001,
            lot_size=100_000.0,
            base_currency=pair[:3],
            quote_currency=pair[3:],
            typical_spread_bps=1.0,
        )
        if pair not in _vp:
            _vp.append(pair)
        store.insert_pairs([
            {
                "symbol": pair,
                "oanda_name": pair[:3] + "_" + pair[3:],
                "pip_value": 0.0001,
                "lot_size": 100_000.0,
                "base_currency": pair[:3],
                "quote_currency": pair[3:],
                "typical_spread_bps": 1.0,
            }
        ])

    migrate_pair(store, str(dest), pair, timeframe, force=True)
    return {
        "status": "ok",
        "filename": str(dest),
        "pair": pair,
        "timeframe": timeframe,
    }
