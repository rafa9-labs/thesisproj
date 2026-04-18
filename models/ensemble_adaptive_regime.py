# models/ensemble_adaptive_regime.py

import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
# Calibrated probabilities (Platt / isotonic) live under sklearn.calibration
try:
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    CalibratedClassifierCV = None  # fallback handled in fit()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .lstm import build_lstm
from utilsNoWFO import ensure_dict

import tensorflow as tf
from tensorflow import keras

Callback      = keras.callbacks.Callback
EarlyStopping = keras.callbacks.EarlyStopping
Adam          = keras.optimizers.Adam


# --- PATCH: optional mixed precision (consistency with single-model path) ---
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
# --- END PATCH ---

class _TimeLimit(Callback):
    def __init__(self, seconds=None):
        super().__init__()
        self.seconds = float(seconds) if seconds not in (None, 0, False) else None
        self._start = None

    def on_train_begin(self, logs=None):
        import time
        if self.seconds is not None:
            self._start = time.time()

    def on_batch_end(self, batch, logs=None):
        import time
        if self.seconds is not None and (time.time() - self._start) > self.seconds:
            self.model.stop_training = True

import os

def _safe_cpu_jobs(reserve: int = 4, cap: int = 20) -> int:
    """
    Leave a few cores free for OS/TF/XGB. Returns a stable default n_jobs.
    - reserve: how many logical CPUs to keep free
    - cap: upper bound even on very high-core hosts
    """
    n = os.cpu_count() or 8
    return min(cap, max(1, n - reserve))


class AdaptiveRegimeStrategy:
    """
    Adaptive Regime Strategy:
      - Detects regime (trend, volatile, sideways) from engineered features (ADX/vol).
      - Chooses model:
            'trend'    -> LSTM
            'volatile' -> RandomForest
            'sideways' -> Logistic Regression
      - RF & Logit train on the full data; LSTM can optionally train only on 'trend' windows.
    """

    def __init__(
        self,
        lstm_config=None,
        rf_config=None,
        logit_config=None,
        input_shape=None,
        adx_col="adx_14",
        vol_col="rolling_std_20",
        adx_thresh=25,
        vol_thresh=0.002,
        adx_thresh_q=0.70,
        train_lstm_on_trend_only=False,
        soft_regime_blend=False,
        soft_kappa=0.15,
        vol_thresh_q=0.80,
        use_learned_gate=False,
        gate_min_samples=1000,
    ):
        self.train_lstm_on_trend_only = bool(train_lstm_on_trend_only)

        self.lstm_config = ensure_dict(lstm_config)
        self.rf_config = ensure_dict(rf_config)
        self.logit_config = ensure_dict(logit_config)
        
        self.rf_calibrate = bool(self.rf_config.get("calibrate_proba", True))

        # If user didn't provide n_jobs, cap them to avoid oversubscription
        self.rf_config.setdefault("n_jobs", _safe_cpu_jobs())  # RF parallelizes over trees/features
        if str(self.logit_config.get("solver", "lbfgs")).lower() == "saga":
            # Only saga supports n_jobs; lbfgs/newton-cg ignore it
            self.logit_config.setdefault("n_jobs", _safe_cpu_jobs())

        self.input_shape = input_shape
        self.adx_col = adx_col
        self.vol_col = vol_col
        self.adx_thresh = adx_thresh
        self.vol_thresh = vol_thresh
        self.adx_thresh_q = adx_thresh_q
        
        # Soft/blended regime gating knobs
        self.soft_regime_blend = bool(soft_regime_blend)
        self.soft_kappa = float(soft_kappa)
        self.vol_thresh_q = vol_thresh_q

        # Learned gating (Mixture-of-Experts style) knobs
        self.use_learned_gate = bool(use_learned_gate)
        self.gate_min_samples = int(gate_min_samples)
        self.gate_model = None
        self.gate_feature_cols_ = None
        # Fixed mapping: 0=sideways, 1=volatile, 2=trend
        self.gate_regime_map_ = {"sideways": 0, "volatile": 1, "trend": 2}

        # LSTM defaults mirroring single-model path


        # LSTM defaults mirroring single-model path
        self.lstm_config.setdefault("lstm_use_early_stopping", True)
        self.lstm_val_split = float(self.lstm_config.get(
            "lstm_val_split",
            0.1 if self.lstm_config.get("lstm_use_early_stopping") else 0.0
        ))
        self.lstm_time_limit_sec = int(self.lstm_config.get("lstm_time_limit_sec", 45))
        self.lstm_mixed_precision = bool(self.lstm_config.get("lstm_mixed_precision", True))



        # models
        self.lstm = None
        self.rf = None
        self.logit = None

        # scalers / dims
        self.scaler = StandardScaler()
        self.expected_dim_rf = None
        self.expected_dim_logit = None

        # regime source kept after fit()
        self.X_flat_with_regime = None
        
        # caches
        self._cached_regimes = None
        self._cached_regime_source_id = None
        self._cached_regime_key = None  # (id(df), adx_col, vol_col, adx_thr, vol_thr)

    def _resolve_regime_columns(self, df: pd.DataFrame):
        """Resolve adx_col and vol_col to real columns in df when possible."""
        if df is None or not hasattr(df, "columns"):
            return self.adx_col, self.vol_col
        cols = list(df.columns)
        cols_low = [str(c).lower() for c in cols]

        def _extract_int(s: str):
            m = re.findall(r"\d+", str(s))
            return int(m[0]) if m else None

        def _best_match(requested: str, pool: list[str]):
            if requested in pool:
                return requested
            req_low = str(requested).lower()
            for c in pool:
                if str(c).lower() == req_low:
                    return c
            tgt = _extract_int(requested)
            if tgt is not None:
                scored = []
                for c in pool:
                    v = _extract_int(c)
                    scored.append((abs(v - tgt) if v is not None else 10**9, str(c)))
                scored.sort(key=lambda x: x[0])
                if scored:
                    best = scored[0][1]
                    for c in pool:
                        if str(c) == best:
                            return c
            return pool[0] if pool else requested

        # ADX
        old_adx = self.adx_col
        if old_adx not in cols:
            adx_pool = [c for c, cl in zip(cols, cols_low) if "adx" in cl]
            new_adx = _best_match(old_adx, adx_pool)
            if new_adx in cols and new_adx != old_adx:
                print(f"[REGIME] Resolved adx_col '{old_adx}' -> '{new_adx}'")
                self.adx_col = new_adx

        # VOL
        old_vol = self.vol_col
        if old_vol not in cols:
            vol_pool = [c for c, cl in zip(cols, cols_low) if "rolling_std" in cl]
            if not vol_pool:
                vol_pool = [c for c, cl in zip(cols, cols_low)
                            if ("realized" in cl and "vol" in cl) or cl.startswith("vol") or "_vol" in cl]
            if not vol_pool:
                vol_pool = [c for c, cl in zip(cols, cols_low) if cl.startswith("atr") or "_atr" in cl]
            new_vol = _best_match(old_vol, vol_pool)
            if new_vol in cols and new_vol != old_vol:
                print(f"[REGIME] Resolved vol_col '{old_vol}' -> '{new_vol}'")
                self.vol_col = new_vol

        return self.adx_col, self.vol_col


    # --- Memory cleanup ---------------------------------------------------
    def free(self):
        """Release submodels and large state; clear TF backend."""
        for attr in ("lstm", "rf", "logit", "scaler", "gate_model"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc as _gc; _gc.collect()

    def __del__(self):
        try: self.free()
        except Exception: pass

    # -----------------------------
    # Regime detection (with cache)
    # -----------------------------
    def detect_regimes(self, df: pd.DataFrame):
        try:
            self._resolve_regime_columns(df)
        except Exception:
            pass
        key = (id(df), str(self.adx_col), str(self.vol_col), float(self.adx_thresh), float(self.vol_thresh))
        if self._cached_regime_key == key and self._cached_regimes is not None:
            print("[CACHE HIT] Reusing regimes from previous detection.")
            return self._cached_regimes

        print("[CACHE MISS] Computing regimes from scratch.")
        self._cached_regime_source_id = id(df)
        
        self._cached_regime_key = key
        n = len(df)
        adx = df[self.adx_col].to_numpy(copy=False) if self.adx_col in df.columns else None
        vol = df[self.vol_col].to_numpy(copy=False) if self.vol_col in df.columns else None

        regimes = np.full(n, "sideways", dtype=object)

        # NOTE: If both are true (high ADX + high vol), treat as VOLATILE.
        # This avoids sending “wild trend” conditions to the LSTM expert.
        if adx is not None:
            adx = adx.astype(float, copy=False)
            regimes[(np.isfinite(adx)) & (adx > self.adx_thresh)] = "trend"
        if vol is not None:
            vol = vol.astype(float, copy=False)
            regimes[(np.isfinite(vol)) & (vol > self.vol_thresh)] = "volatile"
 
        if (adx is None) or (vol is None):
            missing = []
            if adx is None: missing.append(str(self.adx_col))
            if vol is None: missing.append(str(self.vol_col))
            print(f"[REGIME][WARN] Missing regime columns -> defaulting missing inputs to 'sideways': {missing}")
        self._cached_regimes = regimes
         
        return regimes
    
    def infer_regime_ids(self, df: pd.DataFrame) -> np.ndarray:
        """Return integer regime ids using the same rule/thresholds as `detect_regimes`.

        Mapping:
          0 = sideways
          1 = trend
          2 = volatile

        This is used for evaluation diagnostics so regime-sliced reporting matches the
        exact regime logic used by the strategy.
        """
        try:
            regimes = self.detect_regimes(df)
            if regimes is None or len(regimes) == 0:
                return np.zeros(len(df), dtype=np.int64)
            mapping = {"sideways": 0, "trend": 1, "volatile": 2}
            out = np.zeros(len(regimes), dtype=np.int64)
            for k, v in mapping.items():
                out[regimes == k] = v
            return out
        except Exception:
            return np.zeros(len(df), dtype=np.int64)

    def _proba_to_3class(self, proba: np.ndarray, model) -> np.ndarray:
        """
        Map arbitrary predict_proba output to a 3-column matrix over classes {0,1,2},
        using model.classes_ when available.
        """
        proba = np.asarray(proba, dtype=np.float32)
        n = proba.shape[0]
        out = np.full((n, 3), 1e-12, dtype=np.float32)

        classes_ = getattr(model, "classes_", None)
        if classes_ is None:
            # Fallback: infer from shape
            if proba.shape[1] == 3:
                return proba
            elif proba.shape[1] == 2:
                out[:, [0, 2]] = proba
                p0 = out[:, 0]; p2 = out[:, 2]
                out[:, 1] = np.maximum(out[:, 1], np.clip(1.0 - np.abs(p2 - p0), 0.0, 1.0))
            else:
                # Normalize and smear evenly if we really don't know
                s = proba.sum(axis=1, keepdims=True)
                s[s <= 0] = 1.0
                p = proba / s
                out[:, 1] = p.mean(axis=1)
        else:
            classes_ = np.asarray(classes_, dtype=int)
            for j, cls in enumerate(classes_):
                if 0 <= cls < 3:
                    out[:, int(cls)] = proba[:, j]
            if (1 not in classes_) and (0 in classes_) and (2 in classes_):
                p0 = out[:, 0]; p2 = out[:, 2]
                out[:, 1] = np.maximum(out[:, 1], np.clip(1.0 - np.abs(p2 - p0), 0.0, 1.0))

        # Renormalize rows
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        out /= row_sums
        return out


    def _fit_gate_model(self):
        """Train a light-weight gating classifier p(regime | features) for MoE blending.

        Uses ADX and volatility columns as inputs and the rule-based regimes as targets.
        Falls back silently to analytic gates if training is not possible.
        """
        # Reset gate by default
        self.gate_model = None
        self.gate_feature_cols_ = None

        if not getattr(self, "use_learned_gate", False):
            print("[GATE] use_learned_gate=False → skipping learned gate training.")
            return

        df = getattr(self, "X_flat_with_regime", None)
        if df is None:
            print("[GATE] No X_flat_with_regime available; skipping gate training.")
            return

        # Use ADX and vol columns if present
        cols = []
        for c in (self.adx_col, self.vol_col):
            if c in df.columns:
                cols.append(c)
        if len(cols) < 2:
            print(f"[GATE] Missing ADX/vol columns ({self.adx_col}, {self.vol_col}); skipping gate training.")
            return

        regimes = self.detect_regimes(df)
        regime_map = self.gate_regime_map_
        y_gate = np.array([regime_map.get(r, 0) for r in regimes], dtype=int)

        X_gate = df[cols].to_numpy(dtype=np.float32, copy=False)
        # Drop rows with NaNs in gate features
        mask = np.isfinite(X_gate).all(axis=1)
        if mask.sum() < int(self.gate_min_samples):
            print(f"[GATE] Only {mask.sum()} usable samples for gate (min={self.gate_min_samples}); skipping.")
            return
        X_gate = X_gate[mask]
        y_gate = y_gate[mask]

        try:
            gate = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=200,
                class_weight="balanced",
            )
            gate.fit(X_gate, y_gate)
        except Exception as e:
            print(f"[GATE] Failed to fit gate model: {e}; falling back to analytic gates.")
            return

        self.gate_model = gate
        self.gate_feature_cols_ = cols
        print(f"[GATE] Trained gating LogisticRegression on {X_gate.shape[0]} samples with features={cols}.")

    def _compute_gate_weights(self, regime_source: pd.DataFrame, n_samples: int):
        """Return (w_side, w_vol, w_trend) regime weights for each sample.

        If a learned gate_model is available and valid, use its probabilities.
        Otherwise, fall back to the original analytic ADX/vol logistic ramps.
        """
        # Default equal weights
        w_side = np.full(n_samples, 1.0 / 3.0, dtype=float)
        w_vol = np.full(n_samples, 1.0 / 3.0, dtype=float)
        w_trend = np.full(n_samples, 1.0 / 3.0, dtype=float)

        if regime_source is None:
            regime_source = self.X_flat_with_regime
        df = regime_source
        if df is None:
            return w_side, w_vol, w_trend

        # 1) Try learned gate
        gate = getattr(self, "gate_model", None)
        if getattr(self, "use_learned_gate", False) and gate is not None:
            try:
                cols = self.gate_feature_cols_ or [self.adx_col, self.vol_col]
                # Align to first n_samples rows of df (upstream caller must ensure alignment)
                sub = df[cols].iloc[:n_samples]
                X_gate = sub.to_numpy(dtype=np.float32, copy=False)
                mask = np.isfinite(X_gate).all(axis=1)
                proba_gate = gate.predict_proba(X_gate[mask])

                W = np.zeros((n_samples, 3), dtype=float)
                # Map logistic classes (0,1,2) -> (side, vol, trend)
                for j, cls in enumerate(gate.classes_):
                    cls = int(cls)
                    if 0 <= cls < 3:
                        W[mask, cls] = proba_gate[:, j]

                # Normalize rows with any positive mass
                row_sums = W.sum(axis=1, keepdims=True)
                valid = row_sums[:, 0] > 0
                W[valid] /= row_sums[valid]

                w_side, w_vol, w_trend = W[:, 0], W[:, 1], W[:, 2]
                return w_side, w_vol, w_trend
            except Exception as e:
                print(f"[GATE] Error while computing learned gate weights: {e}; falling back to analytic gates.")

        # 2) Fallback: original analytic gates from ADX/vol thresholds
        if self.adx_col in df.columns:
            adx = df[self.adx_col].to_numpy(copy=False)[:n_samples]
        else:
            adx = np.zeros(n_samples, dtype=float)
        if self.vol_col in df.columns:
            vol = df[self.vol_col].to_numpy(copy=False)[:n_samples]
        else:
            vol = np.zeros(n_samples, dtype=float)

        def _sig(x):
            return 1.0 / (1.0 + np.exp(-x))

        k = max(1e-6, float(self.soft_kappa))
        w_trend = _sig((adx - float(self.adx_thresh)) / (k * max(1.0, float(self.adx_thresh))))
        w_vol = _sig((vol - float(self.vol_thresh)) / (k * max(1e-9, float(self.vol_thresh))))
        w_side = np.clip(1.0 - (w_trend + w_vol), 0.0, 1.0)

        W = np.vstack([w_side, w_vol, w_trend]).T
        row_sums = W.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > 0
        W[valid] /= row_sums[valid]
        w_side, w_vol, w_trend = W[:, 0], W[:, 1], W[:, 2]
        return w_side, w_vol, w_trend

    # -----------
    # Train
    # -----------
    def fit(self, X_seq, X_flat, y, X_flat_with_regime=None, idx_end=None):
        """
        X_seq:  [m, t, f] sequence windows (end aligned with idx_end)
        X_flat: [m, f] row-aligned with idx_end (same m)
        y:      integer labels for the same m rows (class 0/1/2)
        X_flat_with_regime: full DataFrame on original timeline with ADX/vol columns
        idx_end: indices (into X_flat_with_regime) where each window ends (len m)
        """
        print("[DEBUG] Starting fit() for AdaptiveRegimeStrategy")

        # Basic safety casts
        X_seq = np.asarray(X_seq)
        X_flat = np.asarray(X_flat)
        y = np.asarray(y)

        m = len(y)
        if X_seq.shape[0] != m or X_flat.shape[0] != m:
            raise ValueError(f"[❌] Shape mismatch: len(y)={m}, X_seq={X_seq.shape}, X_flat={X_flat.shape}")

        # Quick label sanity (helps spot 'no neutral' conditions early)
        try:
            _u, _c = np.unique(y.astype(int), return_counts=True)
            _counts = {int(k): int(v) for k, v in zip(_u, _c)}
            print(f"[LABELS] class_counts={_counts}")
        except Exception:
            pass

        # Keep a regime source aligned to the original timeline for inference
        if X_flat_with_regime is not None:
            self.X_flat_with_regime = X_flat_with_regime.copy()
        else:
            # Fallback: create a dummy index-aligned frame of length m
            self.X_flat_with_regime = pd.DataFrame(index=np.arange(m))

        # Align/trim regime source to training horizon if needed
        m = len(y)
        if len(self.X_flat_with_regime) != m:
            # If a longer DataFrame was passed, use the *last m rows* so indices align
            self.X_flat_with_regime = self.X_flat_with_regime.tail(m).copy()

        # -------------------------------------------------------
        # Adaptive ADX / volatility thresholds via quantiles
        # -------------------------------------------------------
        df_reg = self.X_flat_with_regime

        # Ensure adx_col/vol_col point to real columns before computing quantiles
        try:
            self._resolve_regime_columns(df_reg)
        except Exception:
            pass

        if df_reg is not None and len(df_reg) > 0:
            # Adaptive ADX threshold
            if self.adx_thresh_q is not None and self.adx_col in df_reg.columns:
                adx_vals = df_reg[self.adx_col].to_numpy(dtype=float, copy=False)
                adx_vals = adx_vals[np.isfinite(adx_vals)]
                if adx_vals.size:
                    self.adx_thresh = float(np.nanquantile(adx_vals, self.adx_thresh_q))
                    print(
                        f"[REGIME] Using adaptive ADX threshold from q={self.adx_thresh_q:.3f}: "
                        f"{self.adx_thresh:.4f}"
                    )

            # Adaptive volatility threshold
            if self.vol_thresh_q is not None and self.vol_col in df_reg.columns:
                vol_vals = df_reg[self.vol_col].to_numpy(dtype=float, copy=False)
                vol_vals = vol_vals[np.isfinite(vol_vals)]
                if vol_vals.size:
                    self.vol_thresh = float(np.nanquantile(vol_vals, self.vol_thresh_q))
                    print(
                        f"[REGIME] Using adaptive vol threshold from q={self.vol_thresh_q:.3f}: "
                        f"{self.vol_thresh:.6f}"
                    )

        # Regime diagnostics on training span (acceptance check)
        try:
            _reg = self.detect_regimes(self.X_flat_with_regime)
            _u, _c = np.unique(_reg, return_counts=True)
            _counts = {str(k): int(v) for k, v in zip(_u, _c)}
            print(
                f"[REGIME] counts={_counts} adx_col={self.adx_col} vol_col={self.vol_col} "
                f"adx_thr={self.adx_thresh:.4f} vol_thr={self.vol_thresh:.6f}"
            )
        except Exception as _e:
            print(f"[REGIME][WARN] Failed to compute training regime counts: {_e}")

        _maybe_mixed_precision(self.lstm_mixed_precision, "Adaptive-LSTM")
        self.lstm = build_lstm(self.input_shape, config=self.lstm_config)

        epochs = int(self.lstm_config.get("epochs", self.lstm_config.get("lstm_epochs", 10)))
        batch_size = int(self.lstm_config.get("batch_size", self.lstm_config.get("lstm_batch_size", 64)))
        patience = int(self.lstm_config.get("patience", self.lstm_config.get("lstm_patience", 8)))
        time_limit = self.lstm_config.get("time_limit_sec", self.lstm_config.get("lstm_time_limit_sec", None))

        callbacks = []
        if getattr(self.lstm, "early_stop_callback", None) is not None:
            callbacks.append(self.lstm.early_stop_callback)
        else:
            monitor = "val_loss" if self.lstm_val_split > 0 else "loss"
            callbacks.append(
                EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, min_delta=1e-4)
            )
        if time_limit not in (None, 0, False):
            callbacks.append(_TimeLimit(time_limit))

        # Default: train on all windows
        X_seq_train, y_seq_train = X_seq, y

        # Optionally restrict LSTM to trend windows only
        if self.train_lstm_on_trend_only:
            regimes_train = self.detect_regimes(self.X_flat_with_regime)

            trend_mask = None
            if idx_end is not None:
                idx = np.asarray(idx_end, dtype=int)
                if idx.size and idx.max() < len(regimes_train):
                    trend_mask = (regimes_train[idx] == "trend")

            if trend_mask is None:
                # If we can't map via idx_end, try direct 1:1 alignment
                if len(regimes_train) == m:
                    trend_mask = (regimes_train == "trend")
                else:
                    print("[WARN] Regime mask length mismatch; using all windows for LSTM.")
                    trend_mask = np.ones(m, dtype=bool)

            # APPLY the trend-only mask (this was previously computed but unused)
            if isinstance(trend_mask, np.ndarray):
                if trend_mask.any():
                    X_seq_train = X_seq_train[trend_mask]
                    y_seq_train = y_seq_train[trend_mask]
                else:
                    print("[WARN] No 'trend' windows detected — training LSTM on all windows.")

        # Train LSTM
        self.lstm.fit(
            X_seq_train.astype(np.float32, copy=False),
            y_seq_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
            validation_split=self.lstm_val_split if self.lstm_val_split > 0 else 0.0,
            callbacks=callbacks,
        )
        print("[DEBUG] LSTM training complete")

        # ---------- Random Forest (volatile) ----------
        X_scaled = self.scaler.fit_transform(X_flat.astype(np.float32, copy=False))
        rf_cfg = dict(self.rf_config or {})
        if rf_cfg.get("max_features") == "auto":
            rf_cfg["max_features"] = "sqrt"
        rf_cfg.setdefault("n_jobs", -1)
        rf_cfg.setdefault("class_weight", "balanced_subsample")

        self.rf = RandomForestClassifier(**rf_cfg)
        self.rf.fit(X_scaled, y)
        self.expected_dim_rf = X_scaled.shape[1]

        if self.rf_calibrate and len(y) >= 1000:
            idx = np.arange(len(y))
            # Work on an explicit int array for safety
            y_arr = np.asarray(y).astype(int)

            # Time-series safe tail split (last 20% as calibration set)
            n = len(idx)
            split = int(np.floor(n * 0.8))
            split = max(1, min(n - 1, split))  # ensure both sides non-empty
            idx_val = idx[split:]

            if idx_val.size == 0:
                print("[WARN] RF calibration skipped — cannot create a valid calibration split.")
                idx_val = None

            if idx_val is not None:
                if CalibratedClassifierCV is not None:
                    # Keep your original heuristic: isotonic only if big enough
                    method = "isotonic" if len(idx_val) > 5000 else "sigmoid"
                    try:
                        # scikit-learn >= 1.6: use FrozenEstimator instead of cv="prefit"
                        from sklearn.frozen import FrozenEstimator
                        rf_cal = CalibratedClassifierCV(
                            FrozenEstimator(self.rf), cv=5, method=method
                        )
                    except ImportError:
                        # scikit-learn < 1.6: cv="prefit" still works
                        rf_cal = CalibratedClassifierCV(self.rf, cv="prefit", method=method)
                    rf_cal.fit(X_scaled[idx_val], y_arr[idx_val])
                    self.rf = rf_cal
                else:
                    print("[WARN] Skipping RF calibration: sklearn.calibration.CalibratedClassifierCV not available.")

        print(f"[DEBUG] RF training complete. Expected dim: {self.expected_dim_rf}")


        # ---------- Logistic Regression (sideways) ----------
        logit_cfg = dict(self.logit_config or {})

        # Map Optuna-style keys -> sklearn keys, and remove duplicates
        max_iter = int(logit_cfg.pop("max_iter", logit_cfg.pop("logit_max_iter", 1000)))
        solver = logit_cfg.pop("solver", logit_cfg.pop("logit_solver", "lbfgs"))

        # Move logit_C -> C (only if C not already provided)
        if "C" not in logit_cfg and "logit_C" in logit_cfg:
            logit_cfg["C"] = logit_cfg.pop("logit_C")

        # Sanitize class_weight
        cw = logit_cfg.pop("class_weight", logit_cfg.pop("logit_class_weight", None))
        try:
            if isinstance(cw, float) and (np.isnan(cw) or np.isinf(cw)):
                cw = None
        except Exception:
            pass
        if isinstance(cw, str):
            s = cw.strip().lower()
            if s in ("", "none", "null", "nan"):
                cw = None
            elif s == "balanced":
                cw = "balanced"
            else:
                try:
                    import json
                    cw = json.loads(s)
                except Exception:
                    cw = None
        elif (cw not in (None, "balanced")) and (not isinstance(cw, dict)):
            cw = None
        logit_cfg["class_weight"] = cw

        # Drop any leftover unknown logit_* keys to avoid sklearn TypeError
        for k in list(logit_cfg.keys()):
            if k.startswith("logit_"):
                logit_cfg.pop(k, None)

        # Parallelize if supported
        if solver == "saga":
            logit_cfg.setdefault("n_jobs", -1)

        # Fit Logit
        self.logit = LogisticRegression(max_iter=max_iter, solver=solver, **logit_cfg)
        self.logit.fit(X_scaled, y)
        self.expected_dim_logit = X_scaled.shape[1]
        print(f"[DEBUG] Logit training complete. Expected dim: {self.expected_dim_logit}")

        # Acceptance check: neutral should be reachable (even if class 1 is missing in training)
        try:
            n_chk = int(min(2000, X_scaled.shape[0]))
            if n_chk > 0:
                pr_rf = self._proba_to_3class(self.rf.predict_proba(X_scaled[:n_chk]), self.rf)
                pr_log = self._proba_to_3class(self.logit.predict_proba(X_scaled[:n_chk]), self.logit)

                def _cnt(p):
                    u, c = np.unique(np.argmax(p, axis=1), return_counts=True)
                    return {int(k): int(v) for k, v in zip(u, c)}

                print(
                    f"[ACCEPT][NEUTRAL] rf_argmax_counts={_cnt(pr_rf)} "
                    f"logit_argmax_counts={_cnt(pr_log)}"
                )
        except Exception as _e:
            print(f"[ACCEPT][NEUTRAL][WARN] neutral reachability check failed: {_e}")

        # Finally, (optionally) train a small gating model for MoE-style blending
        self._fit_gate_model()

    # -----------
    # Predict
    # -----------
    def predict(self, X_seq, X_flat, regime_source=None):
        """
        Returns final class label (-1, 0, 1) from argmax over [class0, class1, class2].
        """
        proba = self.predict_proba(X_seq, X_flat, regime_source)
        raw_preds = np.argmax(proba, axis=1)
        return np.where(raw_preds == 2, 1, np.where(raw_preds == 0, -1, 0))

    def predict_proba(self, X_seq, X_flat, regime_source=None):
        """
        Returns probability matrix (n, 3) for each sample according to active regime/model.
        """  
        # Hard safety: regime_source must match number of windows
        n_win = len(X_seq)
        if regime_source is not None:
            try:                
                if len(regime_source) != n_win:
                    raise ValueError(f"regime_source length ({len(regime_source)}) != n_windows ({n_win})")
            except TypeError:
                pass
            
        if self.soft_regime_blend:
            # Compute all three model probabilities for *all* rows
            seq = np.asarray(X_seq, dtype=np.float32, copy=False)
            flat = np.asarray(X_flat, dtype=np.float32, copy=False)

            prob_lstm = self.lstm.predict(seq, verbose=0).astype(np.float32, copy=False)
            xf = self.scaler.transform(flat)

            prob_rf_raw  = self.rf.predict_proba(xf)
            prob_log_raw = self.logit.predict_proba(xf)

            # Map to full 3-class matrix using model.classes_
            prob_rf  = self._proba_to_3class(prob_rf_raw,  self.rf)
            prob_log = self._proba_to_3class(prob_log_raw, self.logit)

            # Compute regime weights for each expert (sideways, volatile, trend)
            w_side, w_vol, w_trend = self._compute_gate_weights(
                regime_source or self.X_flat_with_regime,
                prob_lstm.shape[0],
            )

            # Combine experts: trend → LSTM, volatile → RF, sideways → Logit
            blend = (w_trend[:, None] * prob_lstm) \
                  + (w_side[:, None]  * prob_log)  \
                  + (w_vol[:, None]   * prob_rf)
            return blend

        # --- Hard regime switching ---


                # --- Hard regime switching ---
        if regime_source is None:
            regime_source = self.X_flat_with_regime

        regimes = self.detect_regimes(regime_source)

        trend_idx, vol_idx, side_idx = [], [], []
        for i, r in enumerate(regimes):
            if r == "trend":
                trend_idx.append(i)
            elif r == "volatile":
                vol_idx.append(i)
            else:
                side_idx.append(i)

        # Materialize inputs ONCE in compact dtype
        X_seq_arr = np.asarray(X_seq, dtype=np.float32)
        X_flat_arr = np.asarray(X_flat, dtype=np.float32)

        n = len(regimes)
        probas = np.zeros((n, 3), dtype=np.float32)

        # Pre-scale once if any RF/Logit rows are needed
        flat_scaled = None
        if vol_idx or side_idx:
            flat_scaled = self.scaler.transform(X_flat_arr)
            # Expected dims should match RF (and typically Logit) training dims
            if flat_scaled.shape[1] != getattr(self, "expected_dim_rf", flat_scaled.shape[1]):
                raise ValueError(
                    f"[❌] Scaled input dim {flat_scaled.shape[1]} != expected RF dim {self.expected_dim_rf}"
                )

        # LSTM (trend)
        if trend_idx:
            seq_batch = X_seq_arr[trend_idx]
            prob_trend = self.lstm.predict(seq_batch, verbose=0)
            # (LSTM should already be 3-class, but keep it safe)
            if prob_trend.shape[1] != 3:
                prob_trend = self._proba_to_3class(prob_trend, self.lstm)
            probas[trend_idx] = prob_trend

        # RandomForest (volatile)
        if vol_idx:
            x_batch = flat_scaled[vol_idx]
            proba_rf = self.rf.predict_proba(x_batch)
            proba_rf = self._proba_to_3class(proba_rf, self.rf)
            probas[vol_idx] = proba_rf

        # Logistic Regression (sideways)
        if side_idx:
            x_batch = flat_scaled[side_idx]
            if x_batch.shape[1] != getattr(self, "expected_dim_logit", x_batch.shape[1]):
                raise ValueError(
                    f"[❌] Logit input dim {x_batch.shape[1]} != expected {self.expected_dim_logit}"
                )
            proba_log = self.logit.predict_proba(x_batch)
            proba_log = self._proba_to_3class(proba_log, self.logit)
            probas[side_idx] = proba_log

        return probas
