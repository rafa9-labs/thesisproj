"""
HPO configuration persistence (save/load to disk).

Extracted from MLBacktesterNoWFO.py lines 663-804.
"""

from pipeline._imports import *  # noqa: F401,F403
from pipeline.dqn_config import HPO_CONFIG_DIR  # noqa: F811

def _ensure_hpo_dir():
    try:
        os.makedirs(HPO_CONFIG_DIR, exist_ok=True)
    except Exception:
        pass

def save_hpo_config_to_disk(model_type: str, best_params: dict, topN_params=None):
    """
    Persist tuned hyperparameters for a given model_type so that they can be
    reused later (e.g. in real_trading_simulation) without re-running Optuna.
    """
    _ensure_hpo_dir()

    safe_best = _sanitize_for_json(best_params or {})
    safe_topN = _sanitize_for_json(topN_params) if topN_params else None

    payload = {
        "model_type": str(model_type),
        "best_params": safe_best,
    }
    if safe_topN:
        payload["topN_params"] = safe_topN

    path = os.path.join(HPO_CONFIG_DIR, f"model_{model_type}_hpo.json")
    try:
        with open(path, "w") as f:
            # default=str just in case something weird slips through (e.g. Timestamps)
            json.dump(payload, f, indent=2, default=str)
        print(f"[HPO] Saved config for {model_type} to {path}")
    except Exception as e:
        print(f"[HPO] Warning: could not save HPO config for {model_type}: {e}")


def _sanitize_for_json(obj):
    """
    Recursively replace NaN / +/-inf with None so that json.dump produces
    valid JSON. Leaves normal numbers/strings/bools untouched.
    """
    import math

    # Dict → sanitize values
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}

    # List / tuple → sanitize each element
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    # Plain Python floats/ints
    if isinstance(obj, (float, int)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    # Try to handle numpy scalar types if numpy is installed
    try:
        import numpy as _np  # type: ignore

        if isinstance(obj, (_np.floating, _np.integer)):
            v = float(obj)
            if not math.isfinite(v):
                return None
            return v
    except Exception:
        pass

    # Everything else (str, bool, None, etc.) → keep as is
    return obj

def load_hpo_config_from_disk(model_type: str):
    """
    Load previously tuned hyperparameters for model_type. Returns (best, topN).
    If nothing is found, returns (None, None).
    Compatibility notes:
        - Supports both file names:
            1) model_<model>_hpo.json   (MLBacktester save/load)
            2) <model>_best_config.json (utilsNoWFO save_hpo_config_to_disk)
        - Supports both schemas:
            { "best_params": {...}, "topN_params": [...] }
            { "best": {...},       "topN": [...] }
    """
    # Candidate paths (first match wins)
    candidates = []

    # Preferred: MLBacktester naming
    candidates.append(os.path.join(HPO_CONFIG_DIR, f"model_{model_type}_hpo.json"))

    # utilsNoWFO naming (safe-escaped)
    safe = str(model_type).replace("/", "_")
    candidates.append(os.path.join(HPO_CONFIG_DIR, f"{safe}_best_config.json"))
    candidates.append(os.path.join(HPO_CONFIG_DIR, f"{model_type}_best_config.json"))

    # If utilsNoWFO uses an absolute base dir and MLB_HPO_DIR isn't set, also check it.
    try:
        from utilsNoWFO import get_hpo_config_dir  # local import
        _base = str(get_hpo_config_dir())
        candidates.append(os.path.join(_base, f"{safe}_best_config.json"))
        candidates.append(os.path.join(_base, f"model_{model_type}_hpo.json"))
    except Exception:
        pass

    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not path:
        return None, None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[HPO] Warning: could not load HPO config for {model_type} from {path}: {e}")
        return None, None
    
    best = data.get("best_params") or data.get("best")
    topN = data.get("topN_params") or data.get("topN") or []

    # Fallback: if schema is flat (params at root), treat the whole dict (minus obvious metadata) as best.
    if not isinstance(best, dict) or not best:
        if isinstance(data, dict):
            drop_keys = {
                "model_type", "direction", "study_name", "schema_version", "generated_at_utc", "source_files",
                "best_params", "best", "topN_params", "topN", "trials",
            }
            best = {k: v for k, v in data.items() if (k not in drop_keys and not str(k).startswith("__"))}
        else:
            best = {}

    # Strip internal metadata keys that can leak into model constructors
    if isinstance(best, dict):
        best = {k: v for k, v in best.items() if not str(k).startswith("__")}

    # Attach a tiny committee pool for runtime consensus (Top-3 if available).
    # __* keys are stripped above, but consensus needs a pool on the evaluated params dict.
    try:
        if isinstance(best, dict) and isinstance(topN, list) and len(topN) >= 2:
            pool = [dict(x) for x in topN[:3] if isinstance(x, dict)]
            if len(pool) >= 2:
                best["__top3_params"] = pool
    except Exception:
        pass

    return best, topN

