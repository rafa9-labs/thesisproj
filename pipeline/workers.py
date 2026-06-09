"""
Worker functions for process-isolated deep model training.

These module-level functions are designed to be called via
concurrent.futures.ProcessPoolExecutor (spawn-safe on Windows).

The pattern:
  1. Main process writes .npy arrays + job.json to a temp dir
  2. Worker function is submitted to ProcessPoolExecutor
  3. Worker loads arrays, builds model, fits, predicts
  4. Worker writes proba_test.npy + out.json, returns metadata

Benefits over subprocess.run():
  - Proper Python exception propagation via Future.result()
  - Worker process can be reused (avoids repeated TF import ~15-30s)
  - Cleaner API (no manual exit code checking)
"""

import os
import sys
import json
import tempfile
import traceback

# Silence TF logs in worker
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _configure_tf_in_worker():
    """Configure TF for GPU-accelerated training in a worker process."""
    try:
        import tensorflow as tf
        import os
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        try:
            tf.config.set_soft_device_placement(True)
        except Exception:
            pass
        _threads = int(os.getenv("MLB_THREADS", str(max(1, (os.cpu_count() or 8) - 2))))
        try:
            tf.config.threading.set_intra_op_parallelism_threads(_threads)
            tf.config.threading.set_inter_op_parallelism_threads(max(2, _threads // 4))
        except Exception:
            pass
        if gpus:
            print(f"[worker] GPU: {len(gpus)} device(s) -- {gpus[0].name}, threads={_threads}")
    except Exception:
        pass


def _seed_everything(seed: int):
    """Set random seeds for reproducibility in the worker process."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def _build_model(model_type: str, input_shape: tuple, params: dict):
    """Build a compiled Keras model by type. Must match models/*.py builders."""
    mt = str(model_type).lower().strip()
    if mt == "cnn":
        from models.cnn import build_cnn
        return build_cnn(input_shape, config=params)
    elif mt == "lstm":
        from models.lstm import build_lstm
        return build_lstm(input_shape, config=params)
    elif mt == "transformer":
        from models.transformer import build_transformer
        return build_transformer(input_shape, config=params)
    else:
        raise ValueError(f"Unknown deep model type: {model_type}")


class _ThrottleCallback:
    """TF callback that inserts micro-sleeps between epochs when CPU/GPU is under pressure."""

    def on_epoch_end(self, epoch, logs=None):
        try:
            from pipeline.resource_monitor import get_throttle_signal
            sig = get_throttle_signal()
            if sig and sig.delay > 0:
                import time
                time.sleep(sig.delay)
        except Exception:
            pass


def _reshape_for_model(X_2d: "np.ndarray", mode: str, win: int):  # noqa: F821
    """Reshape 2D features into the shape expected by the model."""
    import numpy as np
    if mode == "3d":
        n = X_2d.shape[0]
        feats = X_2d.shape[1]
        # sliding window reshape
        if n >= win:
            X_3d = np.lib.stride_tricks.sliding_window_view(
                X_2d, window_shape=win, axis=0
            )
            # sliding_window_view adds a dim at the end; reshape to (n_windows, win, feats)
            X_3d = X_3d.reshape(X_3d.shape[0], win, feats)
            return X_3d
        else:
            return X_2d.reshape(1, n, X_2d.shape[1])
    else:
        # "seq" mode: return as-is (2D); model may handle its own reshaping
        return X_2d


def deep_fit_predict_worker(job_json_path: str) -> dict:
    """
    Main worker entry point for deep model fit + predict in an isolated process.

    Reads job.json for config, loads .npy arrays, builds model, fits, predicts,
    writes results to disk, and returns a summary dict.

    Parameters
    ----------
    job_json_path : str
        Path to the job.json written by the main process.

    Returns
    -------
    dict
        Keys: 'success' (bool), 'proba_path' (str), 'coverage_thr' (float|NaN),
              'error' (str, only on failure)
    """
    try:
        with open(job_json_path, "r", encoding="utf-8") as f:
            job = json.load(f)

        import numpy as np

        # Configure TF + seeds
        seed = int(job.get("seed", 11111))
        _configure_tf_in_worker()
        _seed_everything(seed)

        # Load arrays
        X_train = np.load(job["X_train_path"], allow_pickle=False)
        y_train = np.load(job["y_train_path"], allow_pickle=False)
        X_test = np.load(job["X_test_path"], allow_pickle=False)

        model_type = str(job["model_type"])
        mode = str(job.get("mode", "seq"))
        win = int(job.get("win", 0))
        batch_size = int(job.get("batch_size", 128))
        epochs = int(job.get("epochs", 20))
        params = dict(job.get("params", {}))
        features_config = dict(job.get("features_config", {}))

        # Reshape training data
        X_train_3d = _reshape_for_model(X_train, mode, win) if mode == "3d" else X_train
        if X_train_3d.ndim == 3:
            input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])
        else:
            input_shape = (X_train_3d.shape[1],)

        # Build model
        model = _build_model(model_type, input_shape, params)

        # Optional early stopping
        callbacks = []
        early_cb = getattr(model, "early_stop_callback", None)
        if early_cb is not None:
            callbacks.append(early_cb)
        callbacks.append(_ThrottleCallback())

        # Optional validation split (time-ordered tail)
        validation_split = 0.0
        if early_cb is not None:
            validation_split = float(features_config.get("deep_subprocess_val_split", 0.10))

        fit_kwargs = dict(
            x=X_train_3d, y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,  # time-series safe
        )

        if validation_split > 0 and early_cb is not None:
            n = len(X_train_3d)
            n_val = max(1, int(round(n * validation_split)))
            n_val = min(n_val, n - 1) if n > 1 else 1
            split = n - n_val
            fit_kwargs["x"] = X_train_3d[:split]
            fit_kwargs["y"] = y_train[:split]
            fit_kwargs["validation_data"] = (X_train_3d[split:], y_train[split:])

        model.fit(callbacks=callbacks, **fit_kwargs)

        # Predict on test data
        X_test_3d = _reshape_for_model(X_test, mode, win) if mode == "3d" else X_test
        pred_bs = int(features_config.get("deep_pred_batch_size", max(256, batch_size * 4)))
        pred_bs_cap = int(features_config.get("deep_pred_batch_size_cap", 2048))
        pred_bs = min(max(16, pred_bs), pred_bs_cap)

        proba = model.predict(X_test_3d, batch_size=pred_bs, verbose=0)

        # Ensure 2D
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)

        # Save proba
        proba_out = job["proba_test_out"]
        np.save(proba_out, proba.astype(np.float32))

        # Coverage threshold (simplified -- full calibration stays in main process)
        coverage_thr = float("nan")
        try:
            cfg_fc = features_config
            _mode = str(cfg_fc.get("gating_mode", cfg_fc.get("gate_mode", "threshold"))).lower()
            _tar = cfg_fc.get("target_active_rate", None)
            _use_cov = (_mode == "coverage") or (_tar is not None and float(_tar) > 0)
            if _use_cov and len(proba) > 100:
                max_conf = np.max(proba, axis=1)
                tgt = float(_tar) if (_tar is not None and float(_tar) > 0) else float(cfg_fc.get("target_coverage", 0.10))
                thr = float(np.quantile(max_conf, 1.0 - tgt))
                coverage_thr = thr
        except Exception:
            pass

        # Write output JSON
        out_json = job["out_json"]
        out = {"coverage_thr": coverage_thr, "n_test": int(proba.shape[0])}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f)

        # Cleanup: clear TF session to free GPU memory
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        return {"success": True, "proba_path": proba_out, "coverage_thr": coverage_thr}

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[DEEP_WORKER] Error: {e}\n{tb}")
        return {"success": False, "error": str(e), "traceback": tb}


def get_gpu_free_memory_mb() -> list:
    """
    Get free GPU memory in MiB per GPU.
    Cross-platform: works on Windows and Linux.
    Returns empty list if nvidia-smi is not available.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        ).decode("utf-8", errors="ignore").strip().splitlines()
        return [int(x.strip()) for x in out if x.strip().isdigit()]
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return []
    except Exception:
        return []