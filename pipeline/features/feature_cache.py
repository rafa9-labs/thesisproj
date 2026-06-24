"""Feature disk cache -- persist engineered features to Parquet for reuse.

Avoids recomputing indicators, lags, and composite features when the same
data file + feature config is used across runs.

Cache directory: ``.feature_cache/``  (gitignored)

Cache key = SHA256(csv_path + file_size + file_mtime + canonical feature config)
Each entry = ``<hash>.parquet`` + ``<hash>_features.json``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Default cache directory (project root)
_CACHE_DIR = Path(".feature_cache")


# -- Public helpers ------------------------------------------------------

def disk_cache_dir() -> Path:
    """Return the cache directory, creating it if needed."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def compute_disk_key(csv_path: str | Path, features_config: dict) -> str:
    """Build a deterministic SHA256 cache key from data source + config.

    Parameters
    ----------
    csv_path : str or Path
        Path to the source CSV file.
    features_config : dict
        The full ``features_config`` dict (toggles, windows, lags, etc.).

    Returns
    -------
    str
        16-char hex digest (truncated SHA256 for filename brevity).
    """
    csv = Path(csv_path)
    stat = csv.stat() if csv.exists() else None
    file_size = stat.st_size if stat else 0
    file_mtime = int(stat.st_mtime) if stat else 0

    # Canonical JSON of all feature config (sorted keys -> deterministic)
    config_str = json.dumps(features_config, sort_keys=True, default=str)

    raw = f"{csv}|{file_size}|{file_mtime}|{config_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_from_disk(cache_key: str) -> tuple[pd.DataFrame, list[str]] | None:
    """Load cached features from disk.

    Returns ``(df, features_list)`` on success, or ``None`` if not found
    or if loading fails (corrupt file, incompatible schema, etc.).
    """
    d = disk_cache_dir()
    pq = d / f"{cache_key}.parquet"
    js = d / f"{cache_key}_features.json"

    if not pq.exists() or not js.exists():
        return None

    try:
        df = pd.read_parquet(pq)
        with open(js) as f:
            features = json.load(f)
        logger.debug("[DISK_CACHE] LOAD key=%s rows=%d feats=%d", cache_key[:8], len(df), len(features))
        return df, features
    except Exception as exc:
        logger.warning("[DISK_CACHE] CORRUPT key=%s -- deleting; reason: %s", cache_key[:8], exc)
        # Remove corrupt entries so they get recomputed
        pq.unlink(missing_ok=True)
        js.unlink(missing_ok=True)
        return None


def save_to_disk(cache_key: str, df: pd.DataFrame, features: list[str]) -> None:
    """Persist engineered features to disk as Parquet + JSON sidecar."""
    d = disk_cache_dir()
    pq = d / f"{cache_key}.parquet"
    js = d / f"{cache_key}_features.json"

    try:
        df.to_parquet(pq, engine="pyarrow", compression="snappy")
        with open(js, "w") as f:
            json.dump(features, f)
        mb = pq.stat().st_size / 1024 / 1024
        logger.debug("[DISK_CACHE] SAVE key=%s rows=%d feats=%d %.1fMB", cache_key[:8], len(df), len(features), mb)
    except Exception as exc:
        logger.warning("[DISK_CACHE] SAVE FAILED key=%s -- reason: %s", cache_key[:8], exc)
        # Clean up partial writes
        pq.unlink(missing_ok=True)
        js.unlink(missing_ok=True)


def clear_disk_cache() -> int:
    """Delete all files in the cache directory. Returns count of files removed."""
    d = disk_cache_dir()
    count = 0
    for f in d.iterdir():
        if f.is_file():
            f.unlink()
            count += 1
    logger.info("[DISK_CACHE] CLEARED %d files", count)
    return count


def disk_cache_stats() -> dict[str, Any]:
    """Return stats about the disk cache (file count, total MB)."""
    d = disk_cache_dir()
    files = [f for f in d.iterdir() if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "path": str(d),
        "files": len(files),
        "mb": round(total_bytes / 1024 / 1024, 2),
    }