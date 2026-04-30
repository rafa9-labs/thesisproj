#!/usr/bin/env python3
"""
DEPRECATED: Use pipeline.workers.deep_fit_predict_worker via ProcessPoolExecutor instead.

This file is kept for backward compatibility with MLBacktesterNoWFO.py (monolith),
which calls it via subprocess.run(). The pipeline version uses ProcessPoolExecutor
for better performance (reused worker process, proper error propagation).

This worker:
  1. Reads job.json for config
  2. Loads .npy arrays from temp dir
  3. Builds, fits, and predicts with a deep model
  4. Writes proba_test.npy + out.json to temp dir
  5. Exits with code 0 (success) or 1 (failure)
"""

import os
import sys
import json
import argparse
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _configure_tf():
    """Configure TF for GPU-accelerated training."""
    try:
        import os
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        _threads = int(os.getenv("MLB_THREADS", str(max(1, (os.cpu_count() or 8) - 2))))
        try:
            tf.config.threading.set_intra_op_parallelism_threads(_threads)
            tf.config.threading.set_inter_op_parallelism_threads(max(2, _threads // 4))
        except Exception:
            pass
    except Exception:
        pass


def _seed_everything(seed: int):
    """Set random seeds for reproducibility."""
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
    """Build a compiled Keras model by type."""
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


def main():
    ap = argparse.ArgumentParser(description="Deep model fit+predict subprocess worker (DEPRECATED)")
    ap.add_argument("--job_json", required=True, help="Path to job.json")
    args = ap.parse_args()

    try:
        with open(args.job_json, "r", encoding="utf-8") as f:
            job = json.load(f)

        import numpy as np

        seed = int(job.get("seed", 11111))
        _configure_tf()
        _seed_everything(seed)

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

        # Reshape if needed
        if mode == "3d" and win > 0:
            n = X_train.shape[0]
            feats = X_train.shape[1]
            if n >= win:
                X_train = np.lib.stride_tricks.sliding_window_view(
                    X_train, window_shape=win, axis=0
                ).reshape(-1, win, feats)

        if X_train.ndim == 3:
            input_shape = (X_train.shape[1], X_train.shape[2])
        else:
            input_shape = (X_train.shape[1],)

        model = _build_model(model_type, input_shape, params)

        callbacks = []
        early_cb = getattr(model, "early_stop_callback", None)
        if early_cb is not None:
            callbacks.append(early_cb)

        validation_split = 0.0
        if early_cb is not None:
            validation_split = float(features_config.get("deep_subprocess_val_split", 0.10))

        fit_kwargs = dict(
            x=X_train, y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
        )

        if validation_split > 0 and early_cb is not None:
            n = len(X_train)
            n_val = max(1, int(round(n * validation_split)))
            n_val = min(n_val, n - 1) if n > 1 else 1
            split = n - n_val
            fit_kwargs["x"] = X_train[:split]
            fit_kwargs["y"] = y_train[:split]
            fit_kwargs["validation_data"] = (X_train[split:], y_train[split:])

        model.fit(callbacks=callbacks, **fit_kwargs)

        # Reshape test
        X_test_use = X_test
        if mode == "3d" and win > 0:
            n = X_test.shape[0]
            feats = X_test.shape[1]
            if n >= win:
                X_test_use = np.lib.stride_tricks.sliding_window_view(
                    X_test, window_shape=win, axis=0
                ).reshape(-1, win, feats)

        pred_bs = int(features_config.get("deep_pred_batch_size", max(256, batch_size * 4)))
        pred_bs_cap = int(features_config.get("deep_pred_batch_size_cap", 2048))
        pred_bs = min(max(16, pred_bs), pred_bs_cap)

        proba = model.predict(X_test_use, batch_size=pred_bs, verbose=0)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)

        proba_out = job["proba_test_out"]
        np.save(proba_out, proba.astype(np.float32))

        # Coverage threshold
        coverage_thr = float("nan")
        try:
            _mode = str(features_config.get("gating_mode", features_config.get("gate_mode", "threshold"))).lower()
            _tar = features_config.get("target_active_rate", None)
            _use_cov = (_mode == "coverage") or (_tar is not None and float(_tar) > 0)
            if _use_cov and len(proba) > 100:
                max_conf = np.max(proba, axis=1)
                tgt = float(_tar) if (_tar is not None and float(_tar) > 0) else float(features_config.get("target_coverage", 0.10))
                thr = float(np.quantile(max_conf, 1.0 - tgt))
                coverage_thr = thr
        except Exception:
            pass

        out_json = job["out_json"]
        out = {"coverage_thr": coverage_thr, "n_test": int(proba.shape[0])}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f)

        # Cleanup
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc
        gc.collect()

        sys.exit(0)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[DEEP_SUBPROC_WORKER] Error: {e}\n{tb}", file=sys.stderr)
        # Try to write error info
        try:
            with open(args.job_json, "r") as f:
                job = json.load(f)
            out_json = job.get("out_json", os.path.join(os.path.dirname(args.job_json), "out.json"))
            with open(out_json, "w") as f:
                json.dump({"error": str(e), "traceback": tb}, f)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()