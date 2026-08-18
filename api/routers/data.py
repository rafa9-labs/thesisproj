"""Data download and upload endpoints."""
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks
from pydantic import BaseModel

from api.dependencies import get_data_store
from api.schemas.pairs import DownloadRequest, DownloadResponse
from api.services import JobManager
from pipeline.data.pair_config import VALID_PAIRS


class SeedDemoRequest(BaseModel):
    pairs: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None


class SeedDemoTimeframe(BaseModel):
    timeframe: str
    rows: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SeedDemoResponse(BaseModel):
    status: str
    pairs: Dict[str, List[SeedDemoTimeframe]]
    total_candles: int


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

    # Run synchronously in-process (no Celery dependency). The impl marks the
    # job failed on error; swallow the exception here so a failed download
    # never crashes the request cycle.
    from api.tasks import _download_data_impl

    def _run_download():
        try:
            _download_data_impl(job_id, pair, req.years, req.base_timeframe)
        except Exception:
            pass

    background_tasks.add_task(_run_download)

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


def _parse_csv_filename(filename: str) -> tuple[str, str] | None:
    """Extract (pair, timeframe) from a CSV filename like EURUSD_10_years_M30_OANDA.csv."""
    m = re.match(r"^([A-Z]{6})_(\d+)_years_(M\d+|H\d+)_OANDA\.csv$", filename)
    if m:
        return m.group(1), m.group(3)
    return None


def _seed_csv_to_store(
    csv_path: Path, store, pair: str, timeframe: str
) -> tuple[int, str, str]:
    """Read a CSV file and insert candles into the store. Returns (row_count, start_date, end_date)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
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
    start = str(df["time"].iloc[0])
    end = str(df["time"].iloc[-1])
    return len(rows), start, end


@router.post("/seed-demo", response_model=SeedDemoResponse)
def seed_demo(req: SeedDemoRequest):
    """Load demo data from bundled CSV files into the SQLite store.

    Scans ``csv_data/`` for files matching OANDA naming convention.
    Defaults to EURUSD M30 if no pairs/timeframes specified.
    """
    store = get_data_store()
    csv_dir = Path("csv_data")
    if not csv_dir.is_dir():
        raise HTTPException(500, "csv_data/ directory not found")

    wanted_pairs = {p.upper() for p in (req.pairs or ["EURUSD"])}
    wanted_tfs = {tf.upper() for tf in (req.timeframes or ["M30"])}

    # Map (pair, timeframe) → file path
    candidates: dict[tuple[str, str], Path] = {}
    for f in csv_dir.iterdir():
        if not f.name.endswith(".csv") or f.name == "README.txt":
            continue
        parsed = _parse_csv_filename(f.name)
        if parsed is None:
            continue
        pair, tf = parsed
        if pair in wanted_pairs and tf in wanted_tfs:
            candidates[(pair, tf)] = f

    if not candidates:
        raise HTTPException(
            400,
            f"No CSV files found for pairs={list(wanted_pairs)} timeframes={list(wanted_tfs)} "
            f"in csv_data/",
        )

    result: dict[str, list[SeedDemoTimeframe]] = {}
    total = 0
    for (pair, tf), csv_path in sorted(candidates.items()):
        rows, start, end = _seed_csv_to_store(csv_path, store, pair, tf)
        total += rows
        result.setdefault(pair, []).append(SeedDemoTimeframe(
            timeframe=tf,
            rows=rows,
            start_date=start,
            end_date=end,
        ))

    return SeedDemoResponse(
        status="ok",
        pairs=result,
        total_candles=total,
    )
