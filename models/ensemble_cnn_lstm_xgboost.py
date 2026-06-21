import time
import numpy as np
from xgboost import XGBClassifier
from .cnn import build_cnn
from .lstm import build_lstm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

import xgboost as xgb  # for callback API

import tensorflow as tf
from tensorflow import keras
import gc as _gc
from contextlib import contextmanager as _contextmanager

# Keras callback aliases
EarlyStopping = keras.callbacks.EarlyStopping
Callback      = keras.callbacks.Callback


@_contextmanager
def _tf_session_scope():
    """Build/train/ predict inside; ensures graph/session cleanup afterward."""
    try:
        yield
    finally:
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        _gc.collect()

# --- PATCH: optional mixed precision (matches single-model path) ---
def _maybe_mixed_precision(enable: bool, tag: str = ""):
    if not enable:
        return
    try:
        from keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        if tag:
            print(f"[{tag}] Mixed precision enabled.")
    except Exception:
        pass

# --- Meta-feature transforms & calibration ---
import numpy as _np

def _logit_mc(p: _np.ndarray, eps: float = 1e-6) -> _np.ndarray:
    """Multi-class log-odds per class: log(p_k) - mean_j!=k log(p_j)."""
    p = _np.clip(p.astype(_np.float32), eps, 1.0 - eps)
    logp = _np.log(p)
    mean_others = (logp.sum(axis=1, keepdims=True) - logp) / _np.maximum(1, p.shape[1]-1)
    return (logp - mean_others).astype(_np.float32)

def _apply_temp(probs: _np.ndarray, T: float) -> _np.ndarray:
    if T is None or abs(float(T) - 1.0) < 1e-9:
        return probs
    z = _np.log(_np.clip(probs, 1e-12, 1.0)).astype(_np.float32) / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = _np.exp(z)
    return (e / (e.sum(axis=1, keepdims=True) + 1e-12)).astype(_np.float32)

def _purged_sequential_oof_splits(n, y, n_splits, purge_bars=12):
    """Purged expanding-window chronological splits for OOF prediction.

    Replaces shuffled KFold/StratifiedKFold with strictly time-ordered folds.
    Each fold trains on all data up to (val_block_start - purge_bars) and
    validates on the next consecutive block. Yields n_splits-1 folds.

    Parameters
    ----------
    n : int
        Total number of samples (chronologically ordered).
    y : np.ndarray
        Labels (unused; accepted for API compatibility).
    n_splits : int
        Requested number of folds (yields at most n_splits-1 valid folds).
    purge_bars : int
        Purge window between train end and validation start.

    Yields
    ------
    (train_indices, val_indices) : tuple of np.ndarray
    """
    if n < n_splits * 2 or n_splits < 2:
        return
    block_size = n // n_splits
    if block_size < 1:
        return
    for i in range(1, n_splits):
        val_start = i * block_size
        val_end = min((i + 1) * block_size, n) if i < n_splits - 1 else n
        train_end = max(0, val_start - purge_bars)
        if train_end < block_size:
            continue
        yield _np.arange(0, train_end, dtype=int), _np.arange(val_start, val_end, dtype=int)


def _learn_temperature(probs: _np.ndarray, y: _np.ndarray, iters: int = 200) -> float:
    """Guo et al. (ICML'17) temperature scaling: optimize T on val NLL."""
    y = y.astype(int)
    T = _np.array([1.0], dtype=_np.float32)
    lr = 0.05
    for _ in range(iters):
        pT = _apply_temp(probs, float(T[0]))
        # NLL grad wrt T (finite-diff to keep it robust/simple)
        f = _np.clip(1e-3, 1.0 - 1e-3, None)  # dummy to appease linters
        eps = 1e-3
        nll  = -_np.log(_np.take_along_axis(pT, y.reshape(-1,1), axis=1)+1e-12).mean()
        pT2  = _apply_temp(probs, float(T[0]+eps))
        nll2 = -_np.log(_np.take_along_axis(pT2, y.reshape(-1,1), axis=1)+1e-12).mean()
        g = (nll2 - nll) / eps
        T[0] = max(0.2, min(5.0, float(T[0] - lr*g)))
    return float(T[0])


# --- PATCH: lightweight wall-clock time cap for Keras training ---
import time as _time
import tensorflow as _tf

class TimeLimit(_tf.keras.callbacks.Callback):
    def __init__(self, seconds):
        super().__init__()
        try:
            s = float(seconds) if seconds is not None else None
        except Exception:
            s = None
        self.seconds = s  # None means "no cap"
        self._t0 = None

    def on_train_begin(self, logs=None):
        if self.seconds is not None:
            self._t0 = _time.time()

    def on_batch_end(self, batch, logs=None):
        if self.seconds is not None and (_time.time() - self._t0) >= self.seconds:
            self.model.stop_training = True



class EnsembleCNNLSTMXGBoost:
    """
    Local-to-Global Pattern Ensemble: Combines CNN and LSTM deep models as meta-feature generators
    and XGBoost as the final decision maker.

    Notes on speed controls (ensemble-only, safe defaults):
      - cnn_patience / lstm_patience (EarlyStopping)
      - cnn_time_limit_sec / lstm_time_limit_sec (hard wall time)
      - cnn_batch_size / lstm_batch_size (train & predict batch sizes)
      - early_stopping_rounds for XGB (via xgb_config)
    """

    def __init__(self, cnn_config=None, lstm_config=None, xgb_config=None, input_shape=None):
        self.cnn_config = dict(cnn_config or {})
        self.lstm_config = dict(lstm_config or {})
        self.xgb_config  = dict(xgb_config  or {})
        self.input_shape = input_shape
        
        # Meta-feature controls
        self.use_logit_meta = bool(self.xgb_config.get("use_logit_meta", True))
        self.calibrate_base_temps = bool(self.xgb_config.get("calibrate_base_temps", False))
        self._cnn_T = 1.0
        self._lstm_T = 1.0
        # --- OOF stacking controls (opt-in; keeps default behavior unchanged) ---
        self.use_oof_meta = bool(self.xgb_config.get("use_oof_meta", False))
        self.oof_splits   = int(self.xgb_config.get("oof_splits", 3))


        # Defaults to mirror single-model behavior
        self.cnn_config.setdefault("cnn_use_early_stopping", True)
        self.lstm_config.setdefault("lstm_use_early_stopping", True)

        # Validation only if ES is on
        self.cnn_val_split = float(self.cnn_config.get(
            "cnn_val_split",
            0.1 if self.cnn_config.get("cnn_use_early_stopping") else 0.0
        ))
        self.lstm_val_split = float(self.lstm_config.get(
            "lstm_val_split",
            0.1 if self.lstm_config.get("lstm_use_early_stopping") else 0.0
        ))

        # Time limits
        self.cnn_time_limit_sec  = int(self.cnn_config.get("cnn_time_limit_sec", 30))
        self.lstm_time_limit_sec = int(self.lstm_config.get("lstm_time_limit_sec", 45))

        # Mixed precision knobs
        self.cnn_mixed_precision  = bool(self.cnn_config.get("cnn_mixed_precision", False))
        self.lstm_mixed_precision = bool(self.lstm_config.get("lstm_mixed_precision", True))
        
        # If a GPU is available, force mixed precision for ensemble heads
        try:
            _gpus = tf.config.list_physical_devices("GPU")
        except Exception:
            _gpus = []
        if _gpus:
            self.cnn_mixed_precision = True
            self.lstm_mixed_precision = True

        # Optional safety cap (upstream already trims, this is a second guard)
        self.deep_max_train_windows = int(
            self.lstm_config.get("deep_max_train_windows",
            self.cnn_config.get("deep_max_train_windows", 0))
        )

        self.cnn = None
        self.lstm = None
        self.xgb = None
        self.scaler = None

        self.expected_dim = None
        self.last_meta_hash = None
        
        
    # --- Memory cleanup ---------------------------------------------------
    def free(self):
        """Release deep models/large state to avoid cross-fold RAM/VRAM growth."""
        try:
            self.cnn = None
        except Exception:
            pass
        try:
            self.lstm = None
        except Exception:
            pass
        try:
            self.xgb = None
        except Exception:
            pass
        try:
            self.scaler = None
        except Exception:
            pass
        # Clear TF backend & Python heap
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc as _gc; _gc.collect()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def fit(self, X_seq, X_flat, y):
        import inspect
        
        # --- Optional OOF meta training (leakage-free stacking) ---
        if getattr(self, "use_oof_meta", False):
            if bool(self.calibrate_base_temps):
                print("[OOF] calibrate_base_temps=True not supported in OOF path yet; falling back to standard fit().")
            else:
                return self.fit_with_oof_meta(X_seq, X_flat, y, n_splits=int(self.oof_splits))
            
        X_seq = np.asarray(X_seq, dtype=np.float32)
        if X_flat is not None:
            X_flat = np.asarray(X_flat, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)


        # --- Enable MP BEFORE building models ---
        _maybe_mixed_precision(self.cnn_mixed_precision or self.lstm_mixed_precision, "Ensemble-CNN/LSTM")

        # --- Guard CNN kernels vs. sequence length (avoids negative dims) ---
        lags = int(self.input_shape[0])
        ks1 = int(self.cnn_config.get("cnn_kernel_size", 3))
        k2_assumed = int(self.cnn_config.get("cnn_kernel_size2", 2))  # conv2 often uses 2
        ks1 = max(1, min(ks1, lags))
        if (lags - (ks1 - 1)) < k2_assumed:
            ks1 = max(1, lags - (k2_assumed - 1))
        self.cnn_config = dict(self.cnn_config)
        self.cnn_config["cnn_kernel_size"] = ks1

        # Basic safety checks
        if X_seq.shape[0] != len(y):
            raise ValueError(f"[[ERR]] X_seq rows ({X_seq.shape[0]}) != y length ({len(y)})")
        if X_flat is None or X_flat.shape[0] != len(y):
            raise ValueError(f"[[ERR]] X_flat rows must match y length ({len(y)}). Got {None if X_flat is None else X_flat.shape[0]}")

        # Optional cap on number of training windows (safety)
        if self.deep_max_train_windows > 0 and X_seq.shape[0] > self.deep_max_train_windows:
            X_seq = X_seq[-self.deep_max_train_windows:]
            y     = y[-self.deep_max_train_windows:]
            if X_flat is not None and len(X_flat) >= len(y):
                X_flat = X_flat[-self.deep_max_train_windows:]

        # Build models
        self.cnn  = build_cnn(self.input_shape, config=self.cnn_config)
        self.lstm = build_lstm(self.input_shape, config=self.lstm_config)

        # --- Keras EarlyStopping + optional time limits (ensemble-only) ---
        # Patience & wall-clock caps (CNN typically converges a bit faster than LSTM
        # in FX time-series tasks, so we use slightly shorter patience). 
        cnn_bs  = int(self.cnn_config.get("cnn_batch_size", 64))
        lstm_bs = int(self.lstm_config.get("lstm_batch_size", 64))

        cnn_pat  = int(self.cnn_config.get("cnn_patience", 8))
        lstm_pat = int(self.lstm_config.get("lstm_patience", 10))
        # If we actually reserve a validation split, monitor val_loss; otherwise fall back to loss.
        cnn_monitor  = "val_loss" if self.cnn_val_split > 0 else "loss"
        lstm_monitor = "val_loss" if self.lstm_val_split > 0 else "loss"

        cnn_es = EarlyStopping(
            monitor=cnn_monitor,
            patience=cnn_pat,
            restore_best_weights=True,
            min_delta=1e-4,
        ) if self.cnn_val_split > 0 else None

        lstm_es = EarlyStopping(
            monitor=lstm_monitor,
            patience=lstm_pat,
            restore_best_weights=True,
            min_delta=1e-4,
        ) if self.lstm_val_split > 0 else None

        cnn_callbacks = [cb for cb in (cnn_es,) if cb is not None]
        lstm_callbacks = [cb for cb in (lstm_es,) if cb is not None]

        # Serialize deep training on GPU; parallelize only on CPU.
        try:
            has_gpu = len(tf.config.list_physical_devices("GPU")) > 0
        except Exception:
            has_gpu = False

        # --- END PATCH ----------------------------------------------------------------


        if has_gpu:
            # <- GPU present: train sequentially to avoid VRAM contention/OOM
            self.cnn.fit(
                X_seq, y,
                epochs=int(self.cnn_config.get("cnn_epochs", 10)),
                batch_size=cnn_bs, verbose=0,
                validation_split=self.cnn_val_split if self.cnn_val_split > 0 else 0.0,
                callbacks=cnn_callbacks,
            )
            self.lstm.fit(
                X_seq, y,
                epochs=int(self.lstm_config.get("lstm_epochs", 10)),
                batch_size=lstm_bs, verbose=0,
                validation_split=self.lstm_val_split if self.lstm_val_split > 0 else 0.0,
                callbacks=lstm_callbacks,
            )
        else:
            # <- CPU only: keep your original parallelism for speed
            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(
                    self.cnn.fit, X_seq, y,
                    epochs=int(self.cnn_config.get("cnn_epochs", 10)),
                    batch_size=cnn_bs, verbose=0,
                    shuffle=False,
                    validation_split=self.cnn_val_split if self.cnn_val_split > 0 else 0.0,
                    callbacks=cnn_callbacks,
                )
                f2 = executor.submit(
                    self.lstm.fit, X_seq, y,
                    epochs=int(self.lstm_config.get("lstm_epochs", 10)),
                    batch_size=lstm_bs, verbose=0,
                    shuffle=False,
                    validation_split=self.lstm_val_split if self.lstm_val_split > 0 else 0.0,
                    callbacks=lstm_callbacks,
                )
                f1.result(); f2.result()



        # --- Meta-features (batched predict) ---
        cnn_probs  = self.cnn.predict(X_seq, verbose=0, batch_size=cnn_bs).astype(np.float32, copy=False)
        lstm_probs = self.lstm.predict(X_seq, verbose=0, batch_size=lstm_bs).astype(np.float32, copy=False)

        # Learn per-head temperatures on the *same* split we'll use for XGB
        eval_frac = float(self.xgb_config.get("xgb_eval_fraction", self.xgb_config.get("eval_fraction", 0.10)))

        idx = np.arange(len(y))
        y_arr = np.asarray(y).astype(int)

        # Time-series safe tail split (NO shuffle, NO stratify)
        eval_frac = 0.10 if not (0.0 < eval_frac < 0.5) else eval_frac
        n = len(idx)
        split = int(np.floor(n * (1.0 - eval_frac)))
        split = max(1, min(n - 1, split))
        idx_tr = idx[:split]
        idx_val = idx[split:]

        if self.calibrate_base_temps and 0.0 < eval_frac < 0.5 and len(idx_val) >= 200:
            self._cnn_T  = _learn_temperature(cnn_probs[idx_val],  y_arr[idx_val])
            self._lstm_T = _learn_temperature(lstm_probs[idx_val], y_arr[idx_val])
            cnn_probs  = _apply_temp(cnn_probs,  self._cnn_T)
            lstm_probs = _apply_temp(lstm_probs, self._lstm_T)


        # Optional: log-odds for stabler level-1 signal (stacking best practice)
        if self.use_logit_meta:
            cnn_feats  = _logit_mc(cnn_probs)
            lstm_feats = _logit_mc(lstm_probs)
        else:
            cnn_feats, lstm_feats = cnn_probs, lstm_probs

        meta_X = np.concatenate([X_flat, cnn_feats, lstm_feats], axis=1).astype(np.float32, copy=False)

        meta_X = meta_X.astype(np.float32, copy=False)

        eval_frac = float(self.xgb_config.get("xgb_eval_fraction",
                        self.xgb_config.get("eval_fraction", 0.10)))

        idx_all = np.arange(meta_X.shape[0])
        y_arr = np.asarray(y).astype(int)
        u_y, c_y = np.unique(y_arr, return_counts=True)
        can_stratify = len(u_y) >= 2 and (c_y.min() if len(c_y) else 0) >= 2

        # Time-series safe tail split (NO shuffle, NO stratify)
        eval_frac = 0.10 if not (0.0 < eval_frac < 0.5) else eval_frac
        n = len(idx_all)
        split = int(np.floor(n * (1.0 - eval_frac)))
        split = max(1, min(n - 1, split))
        idx_tr = idx_all[:split]
        idx_val = idx_all[split:]

        self.scaler = StandardScaler()
        self.scaler.fit(meta_X[idx_tr])  # fit ONLY on train fold

        meta_X_tr  = self.scaler.transform(meta_X[idx_tr]).astype(np.float32, copy=False)
        meta_X_val = self.scaler.transform(meta_X[idx_val]).astype(np.float32, copy=False)

        self.expected_dim = meta_X_tr.shape[1]
        self.last_meta_hash = hash((
            meta_X.shape,
            int(y[0]),
            int(y[-1]),
            float(meta_X.mean()),
            float(meta_X.std())
        ))

        # For convenience below
        X_tr, X_val, y_tr, y_val = meta_X_tr, meta_X_val, y[idx_tr], y[idx_val]
        
        # ------------------------------------------------------------
        # XGB label safety: ensure contiguous class ids in this split.
        #
        # XGBClassifier infers n_classes from unique(y_tr). If the neutral
        # class (1) is absent (common with TripleBarrier), y_tr may be {0,2}.
        # Then XGBoost expects labels {0,1} and raises:
        #   "Expected: [0 1], got [0 2]"
        #
        # We keep a stable 3-class space {0,1,2} by appending tiny-weight
        # dummy rows for any missing classes (derived from train fold only).
        # ------------------------------------------------------------
        _fit_sw = None
        try:
            _yuniq = set(np.unique(y_tr).tolist())
            _missing = [c for c in (0, 1, 2) if c not in _yuniq]
            if _missing:
                _x_mean = X_tr.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
                _x_pad  = np.repeat(_x_mean, repeats=len(_missing), axis=0)
                _y_pad  = np.asarray(_missing, dtype=np.int32)
                X_tr = np.vstack([X_tr, _x_pad]).astype(np.float32, copy=False)
                y_tr = np.concatenate([y_tr.astype(np.int32, copy=False), _y_pad])
                _fit_sw = np.ones(len(y_tr), dtype=np.float32)
                _fit_sw[-len(_missing):] = 1e-6
        except Exception:
            _fit_sw = None
        _fit_kwargs = {}
        if _fit_sw is not None:
            _fit_kwargs["sample_weight"] = _fit_sw



        # Merge config with fast, safe defaults (overridden by user config)
        xgb_params = dict(self.xgb_config or {})
        for _k in ("config","xgb_eval_fraction","eval_fraction","xgb_early_stopping_rounds",
           "early_stopping_rounds","oof_splits","use_oof_meta","es_val_fraction","oof_purge_bars"):
            xgb_params.pop(_k, None)
        xgb_params.pop("config", None)
        
        # IMPORTANT: force multiclass. See the note in fit().
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = 3
        xgb_params["eval_metric"] = "mlogloss"
        xgb_params.pop("scale_pos_weight", None)

        # Speed defaults (minimal impact, big speed)
        xgb_params.setdefault("subsample", 0.8)
        xgb_params.setdefault("colsample_bytree", 0.8)
        # xgb_params.setdefault("sampling_method", "gradient_based")  # GPU-friendly; ignored on CPU
        xgb_params.setdefault("max_depth", 6)
        xgb_params.setdefault("n_estimators", 400)
        xgb_params.setdefault("eta", 0.10)
        xgb_params.setdefault("min_child_weight", 1.0)

        # Regularization: prefer the modern "lambda" alias used elsewhere in the codebase.
        # If both are present, let "lambda" (e.g., tuned via xgb_lambda) win. 
        if "lambda" in xgb_params and "reg_lambda" in xgb_params:
            xgb_params.pop("reg_lambda")
        if "lambda" not in xgb_params and "reg_lambda" not in xgb_params:
            xgb_params["lambda"] = 1.0


        # Device-aware gating (match global project policy)
        # Use env var XGB_USE_GPU=1 to enable GPU; otherwise force CPU.
        import os as _os
        use_gpu = (_os.environ.get("XGB_USE_GPU", "0") == "1")
        xgb_params.setdefault("tree_method", "hist")

        # IMPORTANT: XGBoost "hist" only supports uniform sampling.
        # If a config / trial injects gradient_based, override to uniform to prevent hard-fail.
        sm = str(xgb_params.get("sampling_method", "uniform"))
        if xgb_params.get("tree_method") == "hist" and sm != "uniform":
            xgb_params["sampling_method"] = "uniform"
        else:
            xgb_params.setdefault("sampling_method", "uniform")
        xgb_params.pop("predictor", None)  # deprecated/ignored in 2.x
        if use_gpu:
            xgb_params["device"] = _os.environ.get("XGB_DEVICE", "cuda")
        else:
            xgb_params.pop("device", None)

        self.xgb = XGBClassifier(**xgb_params)

        # Prefer xgb_early_stopping_rounds but support legacy key too
        esr = int(self.xgb_config.get("xgb_early_stopping_rounds",
                  self.xgb_config.get("early_stopping_rounds", 50)))
        fit_sig = inspect.signature(self.xgb.fit)
        has_callbacks = "callbacks" in fit_sig.parameters
        has_esr = "early_stopping_rounds" in fit_sig.parameters

        try:
            if esr > 0 and has_callbacks:
                cbs = [xgb.callback.EarlyStopping(rounds=esr, save_best=True)]
                self.xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, callbacks=cbs)
            elif esr > 0 and has_esr:
                self.xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=esr)
            else:
                n_est = int(self.xgb_config.get("n_estimators", 300))
                self.xgb.set_params(n_estimators=min(n_est, 400))
                self.xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        except TypeError:
            n_est = int(self.xgb_config.get("n_estimators", 300))
            self.xgb.set_params(n_estimators=min(n_est, 400))
            self.xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
        del cnn_probs, lstm_probs, cnn_feats, lstm_feats
        del meta_X, meta_X_tr, meta_X_val, X_tr, X_val, y_tr, y_val, idx_tr, idx_val
        import gc as _gc
        _gc.collect()

        return self
    
    def fit_with_oof_meta(self, X_seq, X_flat, y, n_splits=3, random_state=42):
        _maybe_mixed_precision(
            self.cnn_mixed_precision or self.lstm_mixed_precision,
            "Ensemble-CNN/LSTM (OOF)"
        )

        import numpy as _np, inspect as _inspect, xgboost as _xgb
        from sklearn.preprocessing import StandardScaler
        from keras.callbacks import EarlyStopping

        # --- fast mode switch (set by caller in CV) ---
        _fast = (str(self.cnn_config.get("eval_mode", "")) == "cv_fast"
                or str(self.lstm_config.get("eval_mode", "")) == "cv_fast")

        # --- dtype cutover for GPU bandwidth ---
        _seq_dtype = _np.float16 if (self.cnn_mixed_precision or self.lstm_mixed_precision) else _np.float32
        X_seq = _np.asarray(X_seq, dtype=_seq_dtype)
        X_flat = _np.asarray(X_flat, dtype=_np.float32)
        y = _np.asarray(y).astype(int)
        n = X_seq.shape[0]

        # --- OOF folds (lighter in fast mode) ---
        try:
            n_splits = int(self.xgb_config.get("oof_splits", n_splits))
        except Exception:
            pass
        if _fast:
            n_splits = max(2, min(3, int(n_splits)))

        oof_cnn = _np.zeros((n, 3), dtype=_np.float32)
        oof_lstm = _np.zeros((n, 3), dtype=_np.float32)

        # Purged chronological OOF splits (no shuffle — preserves temporal structure)
        _n_splits_eff = int(max(2, n_splits))
        _purge = int(self.xgb_config.get("oof_purge_bars", 12))
        fold_iter = list(_purged_sequential_oof_splits(n, y, _n_splits_eff, purge_bars=_purge))
        if not fold_iter:
            print("[Ensemble] Warning: purged OOF produced 0 folds (n={}, n_splits={}, purge={}). "
                  "Skipping OOF meta-training.", n, _n_splits_eff, _purge)
            return self

        # --- callbacks/time limits ---
        _cnn_pat = int(self.cnn_config.get("cnn_patience", 8))
        _lstm_pat = int(self.lstm_config.get("lstm_patience", 10))

        # --- use consistent val splits everywhere (OOF + final heads) ---
        _cnn_val_split = float(self.cnn_config.get("cnn_val_split", getattr(self, "cnn_val_split", 0.10)))
        _lstm_val_split = float(self.lstm_config.get("lstm_val_split", getattr(self, "lstm_val_split", 0.10)))

        for tr_idx, val_idx in fold_iter:
            cnn = build_cnn(self.input_shape, config=self.cnn_config)
            lstm = build_lstm(self.input_shape, config=self.lstm_config)

            cb_cnn = [EarlyStopping(monitor="val_loss", patience=_cnn_pat, restore_best_weights=True)]
            cb_lstm = [EarlyStopping(monitor="val_loss", patience=_lstm_pat, restore_best_weights=True)]

            # Respect cnn_* and lstm_* overrides if present; otherwise fall back to generic keys.
            cnn_epochs = int(self.cnn_config.get(
                "cnn_epochs",
                self.cnn_config.get("epochs", 10 if _fast else 20),
            ))
            lstm_epochs = int(self.lstm_config.get(
                "lstm_epochs",
                self.lstm_config.get("epochs", 12 if _fast else 24),
            ))

            cnn_bs = int(self.cnn_config.get(
                "cnn_batch_size",
                self.cnn_config.get("batch_size", 64),
            ))
            lstm_bs = int(self.lstm_config.get(
                "lstm_batch_size",
                self.lstm_config.get("batch_size", 64),
            ))

            if _fast:
                cnn_bs = max(cnn_bs, 256)
                lstm_bs = max(lstm_bs, 256)

            cnn.fit(
                X_seq[tr_idx], y[tr_idx],
                epochs=cnn_epochs,
                batch_size=cnn_bs,
                verbose=0,
                shuffle=False,
                validation_split=_cnn_val_split,
                callbacks=cb_cnn,
            )

            lstm.fit(
                X_seq[tr_idx], y[tr_idx],
                epochs=lstm_epochs,
                batch_size=lstm_bs,
                verbose=0,
                shuffle=False,
                validation_split=_lstm_val_split,
                callbacks=cb_lstm,
            )

            oof_cnn[val_idx] = cnn.predict(X_seq[val_idx], verbose=0, batch_size=256)
            oof_lstm[val_idx] = lstm.predict(X_seq[val_idx], verbose=0, batch_size=256)

            # Cleanup per fold to avoid RAM/VRAM growth
            try:
                import tensorflow as _tf
                _tf.keras.backend.clear_session()
            except Exception:
                pass
            try:
                del cnn, lstm
            except Exception:
                pass
            import gc as _gc
            _gc.collect()

        def _logit_mc(p, eps=1e-6):
            p = _np.clip(p, eps, 1 - eps)
            z = _np.log(p)
            z -= z.mean(axis=1, keepdims=True)
            return z.astype(_np.float32)

        if getattr(self, "use_logit_meta", True):
            cnn_feats = _logit_mc(oof_cnn)
            lstm_feats = _logit_mc(oof_lstm)
        else:
            cnn_feats, lstm_feats = oof_cnn, oof_lstm

        meta_X = _np.concatenate([X_flat, cnn_feats, lstm_feats], axis=1).astype(_np.float32)

        # --- scaler on TRAIN ONLY ---
        idx_all = _np.arange(meta_X.shape[0])
        eval_frac = float(self.xgb_config.get("xgb_eval_fraction", self.xgb_config.get("eval_fraction", 0.10)))
        eval_frac = 0.10 if not (0.0 < eval_frac < 0.5) else eval_frac

        # Time-series safe tail split (NO shuffle, NO stratify)
        n = len(idx_all)
        split = int(_np.floor(n * (1.0 - eval_frac)))
        split = max(1, min(n - 1, split))
        idx_tr = idx_all[:split]
        idx_val = idx_all[split:]

        self.scaler = StandardScaler().fit(meta_X[idx_tr])
        meta_X_tr = self.scaler.transform(meta_X[idx_tr]).astype(_np.float32, copy=False)
        meta_X_val = self.scaler.transform(meta_X[idx_val]).astype(_np.float32, copy=False)
        y_tr, y_val = y[idx_tr], y[idx_val]

        # ------------------------------------------------------------
        # XGB label safety (OOF meta split):
        # Keep stable label space {0,1,2} by appending tiny-weight dummy rows
        # for any missing classes in y_tr.
        # ------------------------------------------------------------
        X_tr, X_val = meta_X_tr, meta_X_val
        _fit_sw = None
        try:
            _yuniq = set(_np.unique(y_tr).tolist())
            _missing = [c for c in (0, 1, 2) if c not in _yuniq]
            if _missing:
                _x_mean = X_tr.mean(axis=0, keepdims=True).astype(_np.float32, copy=False)
                _x_pad = _np.repeat(_x_mean, repeats=len(_missing), axis=0)
                _y_pad = _np.asarray(_missing, dtype=_np.int32)
                X_tr = _np.vstack([X_tr, _x_pad]).astype(_np.float32, copy=False)
                y_tr = _np.concatenate([_np.asarray(y_tr, dtype=_np.int32), _y_pad])
                _fit_sw = _np.ones(len(y_tr), dtype=_np.float32)
                _fit_sw[-len(_missing):] = 1e-6
        except Exception:
            _fit_sw = None

        _fit_kwargs = {}
        if _fit_sw is not None:
            _fit_kwargs["sample_weight"] = _fit_sw

        self.expected_dim = int(meta_X_tr.shape[1])

        # --- XGB params: sanitize + device semantics (2.x) ---
        xgb_params = dict(self.xgb_config or {})
        for _k in ("config", "xgb_eval_fraction", "eval_fraction", "xgb_early_stopping_rounds",
                "early_stopping_rounds", "oof_splits", "use_oof_meta", "es_val_fraction",
                "oof_purge_bars", "predictor", "tree_method"):
            xgb_params.pop(_k, None)

        # IMPORTANT: force multiclass config (still need dummy rows above!)
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = 3
        xgb_params["eval_metric"] = "mlogloss"
        xgb_params.pop("scale_pos_weight", None)

        import os as _os        
        use_gpu = (_os.environ.get("XGB_USE_GPU", "0") == "1")
        xgb_params.setdefault("tree_method", "hist")
        if use_gpu:
            xgb_params["device"] = _os.environ.get("XGB_DEVICE", "cuda")
        else:
            xgb_params.pop("device", None)

        self.xgb = XGBClassifier(**xgb_params)

        # --- Early stopping via callbacks if available ---
        esr = int(self.xgb_config.get(
            "xgb_early_stopping_rounds",
            self.xgb_config.get("early_stopping_rounds", 50)
        ))
        fit_sig = _inspect.signature(self.xgb.fit)
        _has_callbacks = "callbacks" in fit_sig.parameters
        _has_esr = "early_stopping_rounds" in fit_sig.parameters

        try:
            if esr > 0 and len(idx_val) >= 50:
                if _has_callbacks:
                    cbs = [_xgb.callback.EarlyStopping(rounds=esr, save_best=True)]
                    self.xgb.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                        callbacks=cbs,
                        **_fit_kwargs
                    )
                elif _has_esr:
                    self.xgb.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                        early_stopping_rounds=esr,
                        **_fit_kwargs
                    )
                else:
                    n_est = int((self.xgb_config.get("n_estimators") or 300))
                    self.xgb.set_params(n_estimators=min(n_est, 400))
                    self.xgb.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                        **_fit_kwargs
                    )
            else:
                n_est = int((self.xgb_config.get("n_estimators") or 300))
                self.xgb.set_params(n_estimators=min(n_est, 400))
                self.xgb.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    **_fit_kwargs
                )
        except TypeError:
            n_est = int((self.xgb_config.get("n_estimators") or 300))
            self.xgb.set_params(n_estimators=min(n_est, 400))
            self.xgb.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                **_fit_kwargs
            )

        # --- Final deployment heads on FULL train ---
        self.cnn = build_cnn(self.input_shape, config=self.cnn_config)
        self.lstm = build_lstm(self.input_shape, config=self.lstm_config)

        # Use the same epoch/batch overrides as in OOF, so deploy matches tuning intent.
        cnn_epochs_full = int(self.cnn_config.get(
            "cnn_epochs",
            self.cnn_config.get("epochs", 10 if _fast else 20),
        ))
        lstm_epochs_full = int(self.lstm_config.get(
            "lstm_epochs",
            self.lstm_config.get("epochs", 12 if _fast else 24),
        ))

        cnn_bs_full = int(self.cnn_config.get(
            "cnn_batch_size",
            self.cnn_config.get("batch_size", 64),
        ))
        lstm_bs_full = int(self.lstm_config.get(
            "lstm_batch_size",
            self.lstm_config.get("batch_size", 64),
        ))

        if _fast:
            cnn_bs_full = max(cnn_bs_full, 256)
            lstm_bs_full = max(lstm_bs_full, 256)

        self.cnn.fit(
            X_seq, y,
            epochs=cnn_epochs_full,
            batch_size=cnn_bs_full,
            verbose=0,
            shuffle=False,
            validation_split=_cnn_val_split,
        )
        self.lstm.fit(
            X_seq, y,
            epochs=lstm_epochs_full,
            batch_size=lstm_bs_full,
            verbose=0,
            shuffle=False,
            validation_split=_lstm_val_split,
        )

        return self

    
    def _build_meta_X_scaled(self, X_seq, X_flat):
        cnn_bs = int(self.cnn_config.get("cnn_batch_size", 64))
        lstm_bs = int(self.lstm_config.get("lstm_batch_size", 64))

        cnn_probs = self.cnn.predict(X_seq, verbose=0, batch_size=cnn_bs)
        lstm_probs = self.lstm.predict(X_seq, verbose=0, batch_size=lstm_bs)

        if self.calibrate_base_temps:
            cnn_probs  = _apply_temp(cnn_probs,  self._cnn_T)
            lstm_probs = _apply_temp(lstm_probs, self._lstm_T)
        if self.use_logit_meta:
            cnn_probs  = _logit_mc(cnn_probs)
            lstm_probs = _logit_mc(lstm_probs)

        meta_X = np.concatenate([X_flat, cnn_probs, lstm_probs], axis=1).astype(np.float32, copy=False)
        meta_X_scaled = self.scaler.transform(meta_X)

        if meta_X_scaled.shape[1] != self.expected_dim:
            raise ValueError(
                f"[[ERR]] Input shape {meta_X_scaled.shape[1]} does not match trained XGBoost shape {self.expected_dim}"
            )
        return meta_X_scaled


    def predict_proba(self, X_seq, X_flat):
        meta_X_scaled = self._build_meta_X_scaled(X_seq, X_flat)
        dm = xgb.DMatrix(meta_X_scaled)
        booster = self.xgb.get_booster()
        best_iter = getattr(self.xgb, "best_iteration", None)
        if best_iter is not None:
            proba = booster.predict(dm, iteration_range=(0, int(best_iter) + 1))
        else:
            proba = booster.predict(dm)
        return proba

    def predict(self, X_seq, X_flat):
        proba = self.predict_proba(X_seq, X_flat)
        return np.asarray(proba).argmax(axis=1)

