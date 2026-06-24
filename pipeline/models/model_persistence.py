"""Model snapshot persistence — save/load complete trained models to disk.

Every snapshot is a directory containing:

    deployed_models/{model_type}_{timestamp}/
        model.joblib          trained estimator (joblib)
        metadata.json         config, env, lineage
        manifest.sha256       checksums of all artifacts

Optional (present when available):
    scaler.joblib            fitted StandardScaler or z-score (means, stds)
    imputer.joblib           fitted SimpleImputer
    calibration.joblib       fitted calibrator
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEPLOY_ROOT = "deployed_models"
ACTIVE_POINTER = ".active"


def _ensure_deploy_root() -> str:
    Path(DEPLOY_ROOT).mkdir(parents=True, exist_ok=True)
    return DEPLOY_ROOT


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ──────────────────────────────────────────────────────
#  SAVE
# ──────────────────────────────────────────────────────
def save_snapshot(
    *,
    model: Any,
    model_type: str,
    best_params: Dict[str, Any] | None = None,
    scaler: Any = None,
    imputer: Any = None,
    calibration: Any = None,
    coverage_conf_thr: float | None = None,
    feature_names: list | None = None,
    features_config: Dict[str, Any] | None = None,
    calibrate_method: str | None = None,
    input_shape: tuple | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    seed: int | None = None,
    pip_freeze: str | None = None,
    parent_job_id: str | None = None,
    metrics: Dict[str, Any] | None = None,
) -> str:
    """Save a complete model snapshot to ``deployed_models/``.

    Returns the snapshot directory path.
    """
    import joblib

    root = _ensure_deploy_root()
    snapshot_dir = os.path.join(root, f"{model_type}_{_timestamp()}")
    os.makedirs(snapshot_dir, exist_ok=True)

    checksums: Dict[str, str] = {}

    def _dump(obj, fname):
        p = os.path.join(snapshot_dir, fname)
        joblib.dump(obj, p)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        checksums[fname] = h.hexdigest()

    _dump(model, "model.joblib")
    if scaler is not None:
        _dump(scaler, "scaler.joblib")
    if imputer is not None:
        _dump(imputer, "imputer.joblib")
    if calibration is not None:
        _dump(calibration, "calibration.joblib")

    meta: Dict[str, Any] = {
        "schema_version": 1,
        "model_type": str(model_type),
        "best_params": best_params or {},
        "coverage_conf_thr": coverage_conf_thr,
        "calibrate_method": calibrate_method,
        "input_shape": list(input_shape) if input_shape else None,
        "feature_names": list(feature_names) if feature_names else [],
        "features_config": _safe_serialize(features_config or {}),
        "train_start": train_start,
        "train_end": train_end,
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_job_id": parent_job_id,
        "pip_freeze": pip_freeze or _capture_pip_freeze(),
        "metrics": _safe_metrics(metrics or {}),
    }
    with open(os.path.join(snapshot_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    with open(os.path.join(snapshot_dir, "manifest.sha256"), "w") as f:
        for name in sorted(checksums):
            f.write(f"{checksums[name]}  {name}\n")

    return snapshot_dir


# ──────────────────────────────────────────────────────
#  LOAD
# ──────────────────────────────────────────────────────
def load_snapshot(snapshot_path: str) -> Dict[str, Any]:
    """Load a complete model snapshot.

    Returns dict with keys: model, scaler, imputer, calibration, metadata.

    Raises FileNotFoundError if path missing, ValueError if manifest fails.
    """
    import joblib

    manifest = _verify_manifest(snapshot_path)
    if not manifest:
        raise ValueError(f"Manifest validation failed for {snapshot_path}")

    result: Dict[str, Any] = {}

    for fname in ["model.joblib", "scaler.joblib", "imputer.joblib", "calibration.joblib"]:
        p = os.path.join(snapshot_path, fname)
        if os.path.isfile(p):
            result[fname.replace(".joblib", "")] = joblib.load(p)
        else:
            result[fname.replace(".joblib", "")] = None

    meta_path = os.path.join(snapshot_path, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            result["metadata"] = json.load(f)
    else:
        result["metadata"] = {}

    return result


def load_model_only(snapshot_path: str):
    """Load just the model object (fast path for prediction)."""
    import joblib
    p = os.path.join(snapshot_path, "model.joblib")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"model.joblib missing in {snapshot_path}")
    return joblib.load(p)


def read_metadata(snapshot_path: str) -> Dict[str, Any]:
    """Read metadata.json without loading model weights."""
    p = os.path.join(snapshot_path, "metadata.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────
#  MANIFEST
# ──────────────────────────────────────────────────────
def _verify_manifest(snapshot_path: str) -> bool:
    mf = os.path.join(snapshot_path, "manifest.sha256")
    if not os.path.isfile(mf):
        return False
    expected: Dict[str, str] = {}
    with open(mf, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) == 2:
                expected[parts[1]] = parts[0]
    for fname, want in expected.items():
        p = os.path.join(snapshot_path, fname)
        if not os.path.isfile(p):
            return False
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            return False
    return True


def validate_snapshot(snapshot_path: str) -> Tuple[bool, str]:
    """Return (is_valid, reason). Checks manifest + metadata existence."""
    if not os.path.isdir(snapshot_path):
        return False, f"Directory not found: {snapshot_path}"
    model_f = os.path.join(snapshot_path, "model.joblib")
    if not os.path.isfile(model_f):
        return False, f"model.joblib missing in {snapshot_path}"
    if not _verify_manifest(snapshot_path):
        return False, "Manifest checksum mismatch"
    return True, "ok"


# ──────────────────────────────────────────────────────
#  ACTIVE MODEL POINTER (file-based)
# ──────────────────────────────────────────────────────
def _active_path() -> str:
    return os.path.join(_ensure_deploy_root(), ACTIVE_POINTER)


def _read_active() -> Dict[str, str]:
    """Read ``deployed_models/.active`` -> {model_type: snapshot_id, ...}."""
    p = _active_path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_active(data: Dict[str, str]) -> None:
    """Atomically write ``deployed_models/.active``."""
    dst = _active_path()
    fd, tmp = tempfile.mkstemp(dir=_ensure_deploy_root())
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, dst)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_active_model_id(model_type: str | None = None) -> Optional[str]:
    """Return the active model ID.

    With no arguments, returns the one global active model's ID (or None).
    With a type argument, returns the active model's ID only if it matches the given type.
    """
    data = _read_active()
    if model_type is not None:
        return data.get(model_type)
    for v in data.values():
        return v
    return None


def set_active_model_id(model_type: str, snapshot_id: str) -> None:
    """Set the single global active model. Replaces any previously active model."""
    _write_active({model_type: snapshot_id})


def clear_active_model_id() -> None:
    """Clear the global active model pointer."""
    _write_active({})


# ──────────────────────────────────────────────────────
#  UTILITY
# ──────────────────────────────────────────────────────
def _capture_pip_freeze() -> str:
    try:
        import subprocess, sys
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""


def _safe_serialize(obj: Any) -> Any:
    """Convert config values to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _safe_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("sharpe", "win_rate", "total_return_pct", "max_drawdown", "total_trades"):
        if k in m:
            out[k] = m[k]
    return out


def export_snapshot(snapshot_path: str, output: str | None = None) -> str:
    """Package snapshot directory as a .koda file (zip).

    Returns path to the .koda file.
    """
    import zipfile, glob as _glob

    root = _ensure_deploy_root()
    name = os.path.basename(snapshot_path.rstrip("/\\"))
    if output is None:
        output = os.path.join(root, name + ".koda")
    else:
        output = os.path.abspath(output)

    base = os.path.dirname(snapshot_path)
    tmp_name = name + ".zip"
    shutil.make_archive(
        os.path.join(root, name),
        "zip",
        root_dir=base,
        base_dir=name,
    )
    tmp_path = os.path.join(root, tmp_name)
    if not tmp_path.endswith(".koda"):
        if os.path.exists(output):
            os.remove(output)
        os.rename(tmp_path, output)
    return output


def import_snapshot(koda_file: str) -> str:
    """Import a .koda file into deployed_models/.

    Validates manifest, extracts, returns new snapshot path.
    """
    import zipfile

    root = _ensure_deploy_root()
    extract_to = os.path.join(root, "_import_tmp")
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(koda_file, "r") as zf:
        zf.extractall(extract_to)

    entries = os.listdir(extract_to)
    if len(entries) != 1:
        shutil.rmtree(extract_to, ignore_errors=True)
        raise ValueError("Expected exactly one directory in .koda archive")

    src = os.path.join(extract_to, entries[0])
    if not os.path.isdir(src):
        shutil.rmtree(extract_to, ignore_errors=True)
        raise ValueError("Expected directory in .koda archive, not a file")

    ok, reason = validate_snapshot(src)
    if not ok:
        shutil.rmtree(extract_to, ignore_errors=True)
        raise ValueError(f"Invalid snapshot: {reason}")

    meta = read_metadata(src)
    model_type = meta.get("model_type", "unknown")
    dst = os.path.join(root, f"{model_type}_imported_{_timestamp()}")
    shutil.move(src, dst)
    shutil.rmtree(extract_to, ignore_errors=True)
    return dst
