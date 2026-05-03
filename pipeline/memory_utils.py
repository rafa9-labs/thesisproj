"""
Memory management utilities.

Extracted from MLBacktesterNoWFO.py lines 505-540.
"""

from pipeline._imports import *  # noqa: F401,F403

def _hard_free():
    try:
        import tensorflow as _tf
        _tf.keras.backend.clear_session()
    except Exception:
        pass
    import gc, time
    gc.collect()
    time.sleep(0.05)
    
def _apply_low_ram_overrides(cfg: dict) -> dict:
    """Shrink memory-heavy knobs when RAM is tight or MLB_LOW_RAM=1."""
    import os, psutil
    cfg = dict(cfg or {})
    avail = psutil.virtual_memory().available / (1024 ** 3)
    trigger = float(os.getenv("LOW_RAM_TRIGGER_GB", "1.25"))
    force = os.getenv("MLB_LOW_RAM", "0") in ("1","true","True")
    if not force and avail >= trigger:
        return cfg
    # caps that materially reduce memory:
    cfg["feature_top_k"] = int(min(cfg.get("feature_top_k", 192), 128))
    cfg["ensemble_deep_max_train_windows"] = int(min(cfg.get("ensemble_deep_max_train_windows", 10000), 10000))
    for k in ("cnn_batch_size","lstm_batch_size","transformer_batch_size"):
        if k in cfg: cfg[k] = int(min(64, int(cfg.get(k, 64))))
    for k in ("cnn_epochs","lstm_epochs","transformer_epochs"):
        if k in cfg: cfg[k] = int(min(15, int(cfg.get(k, 20))))
    # XGBoost memory savers
    cfg["xgb_tree_method"] = "hist"
    cfg["xgb_grow_policy"] = "lossguide"
    cfg["xgb_max_bin"] = 256
    cfg["xgb_n_estimators"] = int(min(int(cfg.get("xgb_n_estimators", 350)), 300))
    # Avoid extra scaling buffers
    cfg["use_rolling_scaler"] = False
    print(f"[COLD] LOW-RAM overrides applied (avail~={avail:.2f}GB).")
    return cfg

