"""
Per-model budget cache with hardware-fingerprint invalidation.

Stores learned thread budgets from prior pipeline runs so subsequent
runs start at the 60% equilibrium point immediately, skipping the
exploration phase.

Cache location:
  Windows: %LOCALAPPDATA%/kodaquant/hw_profile.json
  Linux:   ~/.kodaquant/hw_profile.json
  macOS:   ~/Library/Application Support/kodaquant/hw_profile.json

Invalidation:
  - Hardware fingerprint mismatch (new CPU/GPU) -> entire cache discarded
  - Individual model profile > 30 days old -> stale, recomputed on next read
  - File deleted by user -> full re-profile on next run
"""
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, Optional, Any

CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days per model entry


def _cache_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    elif {"darwin", "macos"} & {platform.system().lower()}:
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))
    return Path(base) / "kodaquant"


def _cache_path() -> Path:
    return _cache_dir() / "hw_profile.json"


def load_cache(fingerprint: str) -> Dict[str, dict]:
    path = _cache_path()
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    cached_fp = data.get("fingerprint", "")
    if cached_fp != fingerprint:
        return {}

    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        return {}

    now = time.time()
    valid = {}
    for model_type, entry in profiles.items():
        if not isinstance(entry, dict):
            continue
        ts = entry.get("timestamp", 0)
        if now - float(ts) > CACHE_TTL_SECONDS:
            continue
        valid[model_type] = entry
    return valid


def save_cache(fingerprint: str, profiles: Dict[str, dict]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, dict):
                existing = raw.get("profiles", {})
                if not isinstance(existing, dict):
                    existing = {}
    except Exception:
        pass

    for model_type, entry in profiles.items():
        entry = dict(entry)
        entry["timestamp"] = time.time()
        existing[model_type] = entry

    payload = {
        "version": 1,
        "fingerprint": fingerprint,
        "profiles": existing,
    }

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    try:
        os.replace(tmp, path)
    except OSError:
        pass


def get_cached_budget(model_type: str, fingerprint: str) -> Optional[dict]:
    cache = load_cache(fingerprint)
    return cache.get(model_type)


def update_cached_budget(
    model_type: str,
    fingerprint: str,
    budget: Dict[str, Any],
) -> None:
    save_cache(fingerprint, {model_type: budget})
