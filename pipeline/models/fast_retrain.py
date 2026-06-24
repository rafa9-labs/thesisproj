"""
Fast Loop Rolling Refit — decouples weight fitting from HPO.

Freezes all structural choices (hyperparameters, HMM, feature config)
from the Slow Loop and only refits model weights on recent data.

Architecture (de Prado AFML Ch. 3, 5, 7, 15):
  Phase 1 — Data ingestion (last N bars)
  Phase 2 — Feature engineering with frozen fracdiff d and config
  Phase 3 — Frozen HMM regime tagging (predict only, NO .fit())
  Phase 4 — Primary committee refit (3D windows first, mask second)
  Phase 5 — Meta-Labeler refit (OOS time-series split)
  Phase 6 — Artifact export to /active_deployment directory

Usage:
    python pipeline/fast_retrain.py \
        --config-path results/committee_config.json \
        --hmm-path results/hmm_detector.joblib \
        --meta-path results/meta_labeler.joblib \
        --lookback-bars 20000 \
        --output-dir artifacts/fast_loop_update_20260620
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fast_retrain")


DEEP_MODELS = {"cnn", "lstm", "transformer", "gru", "gru_lstm"}
ENSEMBLE_MODELS = {
    "ensemble_adaptive_regime",
    "ensemble_cnn_lstm_xgboost",
    "stacking_ensemble",
    "meta_ensemble",
}
TABULAR_MODELS = {
    "logistic", "svm", "random_forest", "decision_tree",
    "xgboost", "lightgbm", "catboost", "regime_classifier",
}

REGIME_NAME_TO_ID: Dict[str, int] = {
    "quiet_squeeze": 0, "trend_up": 1, "trend_down": 2,
    "mean_reverting": 3, "breakout": 4, "high_volatile": 5, "sideways": 6,
}
REGIME_ID_TO_NAME: Dict[int, str] = {v: k for k, v in REGIME_NAME_TO_ID.items()}

MODEL_PREFIX_MAP: Dict[str, str] = {
    "logistic": "logistic_", "svm": "svm_", "random_forest": "rf_",
    "decision_tree": "dt_", "xgboost": "xgb_", "cnn": "cnn_",
    "lstm": "lstm_", "transformer": "transformer_", "dqn": "dqn_",
    "lightgbm": "lgbm_", "catboost": "cb_", "gru": "gru_",
    "gru_lstm": "gru_lstm_", "stacking_ensemble": "stacking_",
    "meta_ensemble": "meta_", "ensemble_adaptive_regime": "ens_ar_",
    "ensemble_cnn_lstm_xgboost": "ens_clx_", "regime_classifier": "rc_",
}


def _ensure_prefixed_params(model_name: str, params: Optional[Dict]) -> Dict:
    """Ensure hyperparameter keys carry their model prefix.

    Registry builders use filter_params(params, prefix) to extract their
    subset.  If the committee config stores stripped keys (e.g. 'max_depth'
    instead of 'xgb_max_depth'), this function adds the canonical prefix
    so build_model() works correctly.
    """
    if not params:
        return {}
    prefix = MODEL_PREFIX_MAP.get(model_name, "")
    if not prefix:
        return dict(params)

    out = {}
    for k, v in params.items():
        if k.startswith(prefix):
            out[k] = v
        else:
            out[f"{prefix}{k}"] = v
    return out


class FastRetrainer:
    """Orchestrates the Fast Loop rolling refit end-to-end."""

    def __init__(
        self,
        config_path: str,
        hmm_path: str,
        symbol: str = "EURUSD",
        base_timeframe: str = "M30",
        lookback_bars: int = 20000,
        oos_frac: float = 0.10,
        output_dir: Optional[str] = None,
        meta_path: Optional[str] = None,
        features_config_path: str = "configs/feature_config.json",
        db_path: str = "data/forex.db",
        seed: int = 42,
        window_type: str = "rolling",
    ):
        self.config_path = Path(config_path)
        self.hmm_path = Path(hmm_path)
        self.meta_path = Path(meta_path) if meta_path else None
        self.symbol = symbol.upper()
        self.base_timeframe = base_timeframe
        self.lookback_bars = lookback_bars
        self.oos_frac = float(oos_frac)
        self.window_type = window_type
        self.features_config_path = features_config_path
        self.db_path = db_path
        self.seed = seed

        self._set_random_seed()

        if output_dir is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = f"artifacts/fast_loop_update_{ts}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._store = None
        self._features_config: Dict[str, Any] = {}
        self._committee_config: Dict[str, Any] = {}
        self._hmm = None
        self._meta_override_threshold: float = 0.50

        self.df_raw: Optional[pd.DataFrame] = None
        self.df_features: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_oos: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.regime_ids_train: Optional[np.ndarray] = None
        self.split_idx: int = 0
        self.window_size: int = 50

        self.manifest: Dict[str, Any] = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_config": str(self.config_path),
            "symbol": self.symbol,
            "base_timeframe": self.base_timeframe,
            "lookback_bars": lookback_bars,
            "oos_frac": oos_frac,
            "models_refitted": [],
            "models_skipped": {},
            "meta_labeler_refitted": False,
        }

    @staticmethod
    def _set_random_seed(seed: int = 42):
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 1: Data Ingestion
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load the last lookback_bars of OHLCV data from SQLite.

        Replicates DataMixin.get_data() preprocessing exactly.
        """
        from pipeline.data.data_sqlite import DataStore

        self._store = DataStore(self.db_path)

        date_range = self._store.get_date_range(self.symbol, self.base_timeframe)
        if date_range is None:
            raise RuntimeError(
                f"No data for {self.symbol}/{self.base_timeframe} in {self.db_path}"
            )
        end_date = date_range[1]

        raw = self._store.get_candles(self.symbol, self.base_timeframe, None, end_date)
        if raw.empty:
            raise RuntimeError(
                f"No {self.base_timeframe} data for {self.symbol} up to {end_date}"
            )

        if len(raw) > self.lookback_bars:
            if self.window_type == "expanding":
                raw = raw.copy()
            else:
                raw = raw.tail(self.lookback_bars).copy()
        else:
            raw = raw.copy()

        raw.set_index("time", inplace=True)
        raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")

        rename_map = {}
        if "mid_close" in raw.columns:
            rename_map["mid_close"] = "price"
        if "mid_high" in raw.columns:
            rename_map["mid_high"] = "high"
        if "mid_low" in raw.columns:
            rename_map["mid_low"] = "low"
        raw.rename(columns=rename_map, inplace=True)

        keep_cols = [c for c in ["price", "high", "low", "spread"] if c in raw.columns]
        raw = raw[keep_cols]

        if "price" in raw.columns:
            raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))

        for col in raw.columns:
            if pd.api.types.is_numeric_dtype(raw[col]):
                raw[col] = raw[col].astype("float32")

        raw.dropna(how="all", inplace=True)
        self.df_raw = raw

        logger.info(
            "Loaded %d bars for %s/%s from %s to %s",
            len(raw), self.symbol, self.base_timeframe,
            str(raw.index[0]), str(raw.index[-1]),
        )
        self.manifest["data_start"] = str(raw.index[0])
        self.manifest["data_end"] = str(raw.index[-1])
        self.manifest["n_bars_loaded"] = len(raw)

        self._load_mtf_data(raw)
        return raw

    def _load_mtf_data(self, base_df: pd.DataFrame):
        """Load MTF data and precompute shifted MAs onto base_df."""
        from config import TIMEFRAME_HIERARCHY

        tf_h = TIMEFRAME_HIERARCHY.get(self.base_timeframe)
        if tf_h is None:
            logger.warning("No MTF hierarchy for %s — skipping MTF features", self.base_timeframe)
            return

        mtf_fast_tf = tf_h["mtf_fast"]
        mtf_slow_tf = tf_h["mtf_slow"]
        start = str(base_df.index[0])
        end = str(base_df.index[-1])

        ind_wins = self._features_config.get("indicator_windows", {}) or {}
        fast_w = int(ind_wins.get("mtf_ma_fast_window", 10))
        slow_w = int(ind_wins.get("mtf_ma_slow_window", 50))

        try:
            df_fast = self._store.get_candles(self.symbol, mtf_fast_tf, start, end)
            df_slow = self._store.get_candles(self.symbol, mtf_slow_tf, start, end)
        except Exception:
            logger.warning("MTF data unavailable — skipping")
            return

        for df_tf in (df_fast, df_slow):
            if df_tf.empty:
                continue
            df_tf.set_index("time", inplace=True)
            for col in df_tf.columns:
                if pd.api.types.is_numeric_dtype(df_tf[col]):
                    df_tf[col] = df_tf[col].astype("float32")

        close_col = "mid_close"
        if not df_fast.empty:
            if "mtf_ma_fast" not in df_fast.columns:
                col = close_col if close_col in df_fast else df_fast.columns[0]
                df_fast["mtf_ma_fast"] = (
                    df_fast[col].rolling(fast_w, min_periods=fast_w).mean().shift(1)
                )
            df_fast = df_fast[["mtf_ma_fast"]].reset_index()
            df_fast["time"] = pd.to_datetime(df_fast["time"], utc=True) + pd.Timedelta(minutes=1)
            base_reset = base_df.reset_index().rename(columns={"index": "time"})
            base_reset["time"] = pd.to_datetime(base_reset["time"], utc=True)
            merged = pd.merge_asof(
                base_reset.sort_values("time"),
                df_fast.sort_values("time"),
                on="time", direction="backward",
            ).set_index("time")["mtf_ma_fast"]
            base_df["mtf_ma_fast"] = merged.reindex(base_df.index).astype("float32")

        if not df_slow.empty:
            if "mtf_ma_slow" not in df_slow.columns:
                col = close_col if close_col in df_slow else df_slow.columns[0]
                df_slow["mtf_ma_slow"] = (
                    df_slow[col].rolling(slow_w, min_periods=slow_w).mean().shift(1)
                )
            df_slow = df_slow[["mtf_ma_slow"]].reset_index()
            df_slow["time"] = pd.to_datetime(df_slow["time"], utc=True) + pd.Timedelta(minutes=1)
            base_reset = base_df.reset_index().rename(columns={"index": "time"})
            base_reset["time"] = pd.to_datetime(base_reset["time"], utc=True)
            merged = pd.merge_asof(
                base_reset.sort_values("time"),
                df_slow.sort_values("time"),
                on="time", direction="backward",
            ).set_index("time")["mtf_ma_slow"]
            base_df["mtf_ma_slow"] = merged.reindex(base_df.index).astype("float32")

        logger.info("MTF features loaded (fast=%s w=%d, slow=%s w=%d)",
                     mtf_fast_tf, fast_w, mtf_slow_tf, slow_w)

    # ------------------------------------------------------------------
    # Phase 2: Feature Engineering
    # ------------------------------------------------------------------

    def _resolve_features_config(self) -> Dict[str, Any]:
        """Build features_config from committee metadata or fallback."""
        meta = self._committee_config.get("metadata", {})
        fc = meta.get("features_config")
        if fc and isinstance(fc, dict):
            logger.info("Using features_config from committee metadata")
            return deepcopy(fc)

        try:
            with open(self.features_config_path) as f:
                fc = json.load(f)
        except Exception:
            fc = {}

        from pipeline.metrics.metrics_tuples import CLASS_DEFAULTS
        defaults = deepcopy(CLASS_DEFAULTS.get("features", {}))
        defaults.update(fc)

        fracdiff_d = meta.get("fracdiff_d")
        if fracdiff_d is not None:
            defaults["fracdiff_d"] = float(fracdiff_d)
            defaults["use_fracdiff"] = True
            logger.info("fracdiff_d=%s from committee metadata", fracdiff_d)
        else:
            logger.warning("No fracdiff_d in committee metadata — using default")

        input_ws = meta.get("input_window_size")
        if input_ws is not None:
            defaults["lags_range"] = int(input_ws)

        return defaults

    def compute_features(self) -> np.ndarray:
        """Compute features on the full data window using MLBacktester.

        Returns the feature matrix as float32 array.  Also populates
        self.feature_names, self.df_features, and self.split_idx.
        """
        from pipeline.backtester.composed import MLBacktester

        self._features_config = self._resolve_features_config()

        bt = MLBacktester(
            symbol=self.symbol,
            start=str(self.df_raw.index[0]),
            end=str(self.df_raw.index[-1]),
            features_config=self._features_config,
            base_timeframe=self.base_timeframe,
            db_path=self.db_path,
        )
        bt.data = self.df_raw.copy()
        bt.features_config = self._features_config

        lags = int(self._features_config.get("lags_range", self._features_config.get("lags", 50)))
        lag_depth = int(self._features_config.get("lag_depth", 1))
        roll_windows = list(self._features_config.get("roll_windows", [5]))

        df_feat, feat_names = bt.prepare_features(
            bt.data, lags=lags, lag_depth=lag_depth, roll_windows=roll_windows,
        )
        df_feat.dropna(inplace=True)

        self.df_features = df_feat
        self.feature_names = list(feat_names)

        n_total = len(df_feat)
        self.split_idx = int(n_total * (1.0 - self.oos_frac))
        self.window_size = lags

        logger.info("Features computed: %d rows x %d cols (window_size=%d)",
                     n_total, len(feat_names), self.window_size)
        self.manifest["n_features"] = len(feat_names)
        self.manifest["window_size"] = self.window_size
        self.manifest["n_feature_rows"] = n_total
        self.manifest["train_bars"] = self.split_idx
        self.manifest["oos_bars"] = n_total - self.split_idx

        return df_feat[feat_names].to_numpy(dtype=np.float32, copy=False)

    def scale_features(self, X_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit StandardScaler on TRAIN portion only, transform all.

        Returns (X_train_scaled, X_oos_scaled).  Fitting on train only
        prevents microscopic leakage into the Meta-Labeler validation set.
        """
        self.scaler = StandardScaler()
        X_train_raw = X_full[:self.split_idx]
        X_oos_raw = X_full[self.split_idx:]

        self.X_train = self.scaler.fit_transform(X_train_raw).astype(np.float32)
        self.X_oos = self.scaler.transform(X_oos_raw).astype(np.float32)

        logger.info("Scaler fitted on %d train rows; OOS=%d rows",
                     self.split_idx, X_oos_raw.shape[0])
        return self.X_train, self.X_oos

    # ------------------------------------------------------------------
    # Phase 3: Labels
    # ------------------------------------------------------------------

    def generate_labels(self) -> np.ndarray:
        """Generate triple-barrier labels on the train portion."""
        from pipeline.features.feature_utils import triple_barrier_labels

        df = self.df_features.iloc[:self.split_idx]
        if "price" not in df.columns:
            raise KeyError("Column 'price' missing from feature df — cannot generate labels")

        y = triple_barrier_labels(
            close=df["price"],
            pt_mult=1.5,
            sl_mult=1.0,
            max_holding=48,
        )
        y_arr = y.to_numpy(dtype=np.int32, copy=False)
        self.y_train = y_arr
        logger.info("Triple-barrier labels: %d samples, classes=%s",
                     len(y_arr), dict(zip(*np.unique(y_arr, return_counts=True))))
        return y_arr

    # ------------------------------------------------------------------
    # Phase 4: Frozen HMM Regime Tagging
    # ------------------------------------------------------------------

    def tag_regimes(self):
        """Load frozen HMM, predict regime IDs on train data.  No .fit()."""
        from pipeline.regime.hmm_regime import HMMRegimeDetector

        self._hmm = HMMRegimeDetector.load(str(self.hmm_path))
        if not self._hmm.is_fitted:
            raise RuntimeError(f"HMM at {self.hmm_path} is not fitted")

        df_full = self.df_features
        regime_ids_all = self._hmm.predict_hard(df_full)
        self.regime_ids_train = regime_ids_all[:self.split_idx].copy()

        logger.info("HMM regime IDs predicted: %d bars (frozen, NOT refit)",
                     len(regime_ids_all))
        unique, counts = np.unique(self.regime_ids_train, return_counts=True)
        for rid, cnt in zip(unique, counts):
            logger.info("  regime %d (%s): %d bars", rid, REGIME_ID_TO_NAME.get(rid, "?"), cnt)

    # ------------------------------------------------------------------
    # Phase 5: Primary Committee Refit
    # ------------------------------------------------------------------

    def refit_primary_models(self) -> Dict[str, Any]:
        """Refit every model in the committee config on its assigned regime.

        3D windowing fix: create windows FIRST, then mask by regime of the
        END bar of each window.  This preserves chronological sequence
        integrity for deep models.
        """
        from models.registry import build_model, MODEL_REGISTRY

        regimes_cfg = self._committee_config.get("regimes", {})
        fallback_cfg = self._committee_config.get("fallback")
        model_params = self._committee_config.get("model_params", {})

        refitted: Dict[str, Any] = {}

        if not regimes_cfg:
            logger.warning("No regimes in committee config — nothing to refit")
            return refitted

        X_train_3d, y_train_win, idx_end = self._create_training_windows()
        regime_train_win = self.regime_ids_train[idx_end] if X_train_3d is not None else None

        for regime_name, assignment in regimes_cfg.items():
            regime_num = REGIME_NAME_TO_ID.get(regime_name)
            if regime_num is None:
                logger.warning("Unknown regime name '%s' — skipping", regime_name)
                continue

            for model_name in assignment.get("models", []):
                key = f"{regime_name}/{model_name}"
                try:
                    model = self._refit_one_model(
                        model_name, model_params, regime_name, regime_num,
                        X_train_3d, y_train_win, idx_end, regime_train_win,
                    )
                    if model is not None:
                        artifact_path = self.output_dir / f"{regime_name}_{model_name}.joblib"
                        joblib_dump(model, str(artifact_path))
                        refitted[key] = str(artifact_path)
                        self.manifest["models_refitted"].append(key)
                        logger.info("  [OK] %s -> %s", key, artifact_path.name)
                    else:
                        self.manifest["models_skipped"][key] = "insufficient_data"
                        logger.warning("  [SKIP] %s — insufficient regime data", key)
                except Exception as e:
                    self.manifest["models_skipped"][key] = str(e)
                    logger.error("  [FAIL] %s — %s", key, e)

        if fallback_cfg:
            for model_name in fallback_cfg.get("models", []):
                key = f"fallback/{model_name}"
                try:
                    model = self._refit_one_model_fallback(model_name, model_params)
                    if model is not None:
                        artifact_path = self.output_dir / f"fallback_{model_name}.joblib"
                        joblib_dump(model, str(artifact_path))
                        refitted[key] = str(artifact_path)
                        self.manifest["models_refitted"].append(key)
                        logger.info("  [OK] %s -> %s", key, artifact_path.name)
                except Exception as e:
                    self.manifest["models_skipped"][key] = str(e)
                    logger.error("  [FAIL] %s — %s", key, e)

        return refitted

    def _create_training_windows(self):
        """Create 3D windows from train data for deep and ensemble models.

        Returns (X_3d, y_win, idx_end) or (None, None, None) if too few rows.
        """
        if self.X_train is None or self.y_train is None:
            return None, None, None

        n = self.X_train.shape[0]
        w = self.window_size
        if n < w + 10:
            logger.warning("Train data too small for windows (n=%d < w=%d)", n, w)
            return None, None, None

        X_3d = sliding_window_view(self.X_train, window_shape=w, axis=0)
        X_3d = X_3d.reshape(-1, w, self.X_train.shape[1]).astype(np.float32, copy=False)
        idx_end = np.arange(w - 1, n, dtype=int)
        y_win = self.y_train[idx_end]

        return X_3d, y_win, idx_end

    def _refit_one_model(
        self, model_name: str, model_params: Dict, regime_name: str,
        regime_num: int, X_train_3d, y_train_win, idx_end, regime_train_win,
    ):
        """Refit a single model on regime-filtered data.

        Routes to tabular, deep, or ensemble path depending on model type.
        """
        locked_params = _ensure_prefixed_params(
            model_name, model_params.get(model_name, {})
        )
        locked_params["use_proba"] = True

        if model_name in DEEP_MODELS:
            return self._refit_deep(model_name, locked_params,
                                    X_train_3d, y_train_win, regime_train_win, regime_num)
        elif model_name in ENSEMBLE_MODELS:
            return self._refit_ensemble(model_name, locked_params,
                                        X_train_3d, y_train_win, idx_end, regime_train_win, regime_num)
        else:
            return self._refit_tabular(model_name, locked_params, regime_num)

    def _refit_tabular(self, model_name: str, locked_params: Dict, regime_num: int):
        """Fit sklearn-style model on regime-filtered 2D rows."""
        from models.registry import build_model

        mask = self.regime_ids_train == regime_num
        X_regime = self.X_train[mask]
        y_regime = self.y_train[mask]

        if len(X_regime) < 100:
            return None

        unique_cls = np.unique(y_regime)
        if len(unique_cls) < 2:
            return None

        model = build_model(model_name, **locked_params)
        if model is None:
            raise RuntimeError(f"build_model({model_name}) returned None")

        model.fit(X_regime, y_regime)
        return model

    def _refit_deep(
        self, model_name: str, locked_params: Dict,
        X_train_3d, y_train_win, regime_train_win, regime_num: int,
    ):
        """Fit Keras deep model on regime-filtered 3D windows.

        Windows are created first (preserving temporal order), then masked
        by the regime of the END bar of each window.
        """
        from models.registry import build_model

        if X_train_3d is None:
            return None

        mask = regime_train_win == regime_num
        X_model = X_train_3d[mask]
        y_model = y_train_win[mask]

        if len(X_model) < 50:
            return None
        unique_cls = np.unique(y_model)
        if len(unique_cls) < 2:
            return None

        locked_params["input_shape"] = (X_model.shape[1], X_model.shape[2])
        model = build_model(model_name, **locked_params)
        if model is None:
            raise RuntimeError(f"build_model({model_name}) returned None")

        n_cls = len(np.unique(y_model))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        val_frac = min(0.15, 100.0 / len(X_model))
        model.fit(
            X_model, y_model,
            epochs=min(8, max(2, int(locked_params.get(f"{MODEL_PREFIX_MAP.get(model_name, '')}epochs", 4)))),
            batch_size=min(256, len(X_model)),
            validation_split=val_frac,
            shuffle=False,
            verbose=0,
        )
        return model

    def _refit_ensemble(
        self, model_name: str, locked_params: Dict,
        X_train_3d, y_train_win, idx_end, regime_train_win, regime_num: int,
    ):
        """Fit ensemble model on regime-filtered windows.

        Ensemble models receive both X_seq (3D) and X_flat (2D end-bar features).
        For ensemble_adaptive_regime, passes X_flat_with_regime for internal routing.
        """
        from models.registry import build_model

        if X_train_3d is None:
            return None

        mask = regime_train_win == regime_num
        X_seq = X_train_3d[mask]
        y_model = y_train_win[mask]
        idx_end_masked = idx_end[mask]

        if len(X_seq) < 50:
            return None
        unique_cls = np.unique(y_model)
        if len(unique_cls) < 2:
            return None

        locked_params["input_shape"] = (X_seq.shape[1], X_seq.shape[2])

        X_flat = self.X_train[idx_end_masked]

        model = build_model(model_name, **locked_params)
        if model is None:
            raise RuntimeError(f"build_model({model_name}) returned None")

        if model_name == "ensemble_adaptive_regime":
            df_regime_source = self.df_features.iloc[:self.split_idx].copy()
            model.fit(X_seq, X_flat, y_model,
                      X_flat_with_regime=df_regime_source,
                      idx_end=idx_end_masked)
        else:
            model.fit(X_seq, X_flat, y_model)

        return model

    def _refit_one_model_fallback(self, model_name: str, model_params: Dict):
        """Refit on ALL train data (no regime mask) for the fallback."""
        from models.registry import build_model

        locked_params = _ensure_prefixed_params(
            model_name, model_params.get(model_name, {})
        )
        locked_params["use_proba"] = True

        if model_name in DEEP_MODELS:
            X_3d, y_win, _ = self._create_training_windows()
            if X_3d is None:
                return None
            locked_params["input_shape"] = (X_3d.shape[1], X_3d.shape[2])
            model = build_model(model_name, **locked_params)
            if model is None:
                return None
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            val_frac = min(0.15, 100.0 / len(X_3d))
            model.fit(X_3d, y_win, epochs=4, batch_size=min(256, len(X_3d)),
                      validation_split=val_frac, shuffle=False, verbose=0)
            return model

        elif model_name in ENSEMBLE_MODELS:
            X_3d, y_win, idx_end = self._create_training_windows()
            if X_3d is None:
                return None
            locked_params["input_shape"] = (X_3d.shape[1], X_3d.shape[2])
            X_flat = self.X_train[idx_end]
            model = build_model(model_name, **locked_params)
            if model is None:
                return None
            if model_name == "ensemble_adaptive_regime":
                df_regime_source = self.df_features.iloc[:self.split_idx].copy()
                model.fit(X_3d, X_flat, y_win,
                          X_flat_with_regime=df_regime_source, idx_end=idx_end)
            else:
                model.fit(X_3d, X_flat, y_win)
            return model

        else:
            mask = np.ones(len(self.y_train), dtype=bool)
            X_data = self.X_train[mask]
            y_data = self.y_train[mask]
            model = build_model(model_name, **locked_params)
            if model is None:
                return None
            model.fit(X_data, y_data)
            return model

    # ------------------------------------------------------------------
    # Phase 6: Meta-Labeler Refit
    # ------------------------------------------------------------------

    def refit_meta_labeler(self, refitted_models: Dict[str, Any]):
        """Refit MetaLabeler on OOS predictions from the freshly refitted committee.

        Uses a chronological time-series split: the trailing oos_frac of the
        window is held out.  Primary models predict on OOS data; MetaLabeler
        learns from the resulting per-bar predictions.
        """
        from pipeline.models.meta_labeler import MetaLabeler

        if self.X_oos is None or len(self.X_oos) < 50:
            logger.warning("OOS window too small (%d bars) — skipping MetaLabeler refit",
                           len(self.X_oos) if self.X_oos is not None else 0)
            self._copy_frozen_meta_labeler()
            return

        bar_predictions = self._predict_committee_oos(refitted_models)
        if len(bar_predictions) < 20:
            logger.warning("Too few OOS predictions (%d) — skipping MetaLabeler refit",
                           len(bar_predictions))
            self._copy_frozen_meta_labeler()
            return

        X_meta = MetaLabeler.build_features(bar_predictions)
        y_meta = MetaLabeler.build_targets(bar_predictions)

        if len(X_meta) < 20:
            logger.warning("Too few non-zero-signal bars (%d) — skipping MetaLabeler refit",
                           len(X_meta))
            self._copy_frozen_meta_labeler()
            return

        meta = MetaLabeler(override_threshold=self._meta_override_threshold)
        acc = meta.train(X_meta, y_meta)

        if meta.is_trained:
            out_path = self.output_dir / "meta_labeler.joblib"
            meta.save(str(out_path))
            self.manifest["meta_labeler_refitted"] = True
            self.manifest["meta_labeler_accuracy"] = float(acc)
            logger.info("MetaLabeler refitted: accuracy=%.3f on %d samples", acc, len(X_meta))
        else:
            logger.warning("MetaLabeler training failed — keeping frozen artifact")
            self._copy_frozen_meta_labeler()

    def _predict_committee_oos(self, refitted_models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate per-bar OOS predictions from the refitted committee.

        For tabular models: direct 2D prediction (vectorized).
        For deep models: sliding-window prediction aligned to output bars.
        For ensemble models: sliding-window prediction with both X_seq + X_flat.

        Returns list of bar_predictions dicts with keys:
          committee_signal, committee_prob_short, committee_prob_flat,
          committee_prob_long, committee_confidence, next_return, regime_id
        """
        regimes_cfg = self._committee_config.get("regimes", {})
        fallback_cfg = self._committee_config.get("fallback")

        n_oos = self.X_oos.shape[0]
        w = self.window_size

        probs = np.full((n_oos, 3), 0.5, dtype=np.float32)

        for regime_name, assignment in regimes_cfg.items():
            regime_num = REGIME_NAME_TO_ID.get(regime_name, -1)
            if regime_num < 0:
                continue

            regime_mask_oos = self.regime_ids_train[-n_oos:][:n_oos] == regime_num
            if not regime_mask_oos.any():
                continue

            for model_name in assignment.get("models", []):
                model_key = f"{regime_name}/{model_name}"
                if model_key not in refitted_models:
                    continue

                model = self._load_refitted_model(refitted_models[model_key])
                if model is None:
                    continue

                if model_name in DEEP_MODELS or model_name in ENSEMBLE_MODELS:
                    xs, idxs = self._build_oos_windows()
                    if xs is None:
                        continue
                    regime_win_mask = self.regime_ids_train[-n_oos:][idxs] == regime_num
                    xs_regime = xs[regime_win_mask]
                    if len(xs_regime) == 0:
                        continue
                    idxs_regime = idxs[regime_win_mask]

                    p = self._predict_seq_model(model, xs_regime, model_name)
                    for k, pi in zip(idxs_regime, p):
                        if k < n_oos and regime_mask_oos[k]:
                            probs[k] = pi.astype(np.float32)
                else:
                    idxs_regime = np.where(regime_mask_oos)[0]
                    p = model.predict_proba(self.X_oos[idxs_regime])
                    for k, pi in zip(idxs_regime, p):
                        probs[k] = pi.astype(np.float32)

        if fallback_cfg:
            fallback_mask = ~np.any(
                [self.regime_ids_train[-n_oos:][:n_oos] == REGIME_NAME_TO_ID.get(r, -1)
                 for r in regimes_cfg], axis=0
            )
            for model_name in fallback_cfg.get("models", []):
                model_key = f"fallback/{model_name}"
                if model_key not in refitted_models:
                    continue
                model = self._load_refitted_model(refitted_models[model_key])
                if model is None:
                    continue
                if model_name in DEEP_MODELS or model_name in ENSEMBLE_MODELS:
                    xs, idxs = self._build_oos_windows()
                    if xs is None:
                        continue
                    fb_idxs = idxs[fallback_mask[idxs]]
                    if len(fb_idxs) == 0:
                        continue
                    p = self._predict_seq_model(model, xs[fallback_mask[idxs]], model_name)
                    for k, pi in zip(fb_idxs, p):
                        if k < n_oos and fallback_mask[k]:
                            probs[k] = pi.astype(np.float32)
                else:
                    fb_idxs_regime = np.where(fallback_mask)[0]
                    p = model.predict_proba(self.X_oos[fb_idxs_regime])
                    for k, pi in zip(fb_idxs_regime, p):
                        probs[k] = pi.astype(np.float32)

        bar_predictions = []
        df_oos_price = self.df_features["price"].iloc[self.split_idx:].values
        df_oos_returns = self.df_features["returns"].iloc[self.split_idx:].values
        regime_oos = self.regime_ids_train[-n_oos:][:n_oos]

        for i in range(n_oos):
            p_short, p_flat, p_long = float(probs[i, 0]), float(probs[i, 1]), float(probs[i, 2])
            confidence = max(p_short, p_flat, p_long)
            signal_map = {0: -1, 1: 0, 2: 1}
            signal = signal_map.get(int(np.argmax([p_short, p_flat, p_long])), 0)

            next_return = float(df_oos_returns[i + 1]) if i + 1 < n_oos else 0.0

            bar_predictions.append({
                "committee_signal": signal,
                "committee_prob_short": p_short,
                "committee_prob_flat": p_flat,
                "committee_prob_long": p_long,
                "committee_confidence": confidence,
                "next_return": next_return,
                "regime_id": int(regime_oos[i]) if i < len(regime_oos) else 6,
            })

        return bar_predictions

    def _build_oos_windows(self):
        """Build sliding windows over OOS data + preceding warmup."""
        w = self.window_size
        n_oos = self.X_oos.shape[0]
        if n_oos < w:
            logger.warning("OOS too small for windows (n=%d < w=%d)", n_oos, w)
            return None, None

        X_full = np.vstack([self.X_train[-w:], self.X_oos]).astype(np.float32)
        X_3d = sliding_window_view(X_full, window_shape=w, axis=0)
        X_3d = X_3d.reshape(-1, w, X_full.shape[1]).astype(np.float32, copy=False)
        idx_end = np.arange(w - 1, X_full.shape[0], dtype=int)
        idx_oos = idx_end - w
        keep = idx_oos >= 0
        return X_3d[keep], idx_oos[keep]

    @staticmethod
    def _predict_seq_model(model, X_windows: np.ndarray, model_name: str) -> np.ndarray:
        """Predict probabilities from a seq or ensemble model."""
        if model_name in ENSEMBLE_MODELS:
            n_w = X_windows.shape[0]
            X_flat = X_windows[:, -1, :]
            return model.predict_proba(X_windows, X_flat)
        else:
            return model.predict(X_windows, verbose=0)

    @staticmethod
    def _load_refitted_model(artifact_path: str):
        """Load a .joblib artifact back from disk."""
        try:
            return joblib_load(artifact_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", artifact_path, e)
            return None

    def _copy_frozen_meta_labeler(self):
        """Copy the existing frozen MetaLabeler artifact if refit is skipped."""
        if self.meta_path and self.meta_path.exists():
            import shutil
            dest = self.output_dir / "meta_labeler.joblib"
            shutil.copy2(str(self.meta_path), str(dest))
            logger.info("Copied frozen MetaLabeler from %s", self.meta_path)

    # ------------------------------------------------------------------
    # Phase 7: Artifact Export
    # ------------------------------------------------------------------

    def save_artifacts(self):
        """Write all artifacts to self.output_dir."""
        if self.scaler is not None:
            joblib_dump(self.scaler, str(self.output_dir / "scaler.joblib"))
            logger.info("Scaler saved")

        with open(self.output_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f, indent=2)

        import shutil
        src_hmm = str(self.hmm_path)
        dest_hmm = str(self.output_dir / "hmm_detector.joblib")
        if os.path.abspath(src_hmm) != os.path.abspath(dest_hmm):
            shutil.copy2(src_hmm, dest_hmm)
        logger.info("HMM detector copied (frozen)")

        src_cfg = str(self.config_path)
        dest_cfg = str(self.output_dir / "committee_config.json")
        if os.path.abspath(src_cfg) != os.path.abspath(dest_cfg):
            shutil.copy2(src_cfg, dest_cfg)

        with open(self.output_dir / "fast_loop_manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

        logger.info("All artifacts saved to %s", self.output_dir)
        logger.info("Models refitted: %s", self.manifest["models_refitted"])
        logger.info("Models skipped: %s", list(self.manifest["models_skipped"].keys()))

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full Fast Loop pipeline."""
        t0 = time.time()

        logger.info("=" * 60)
        logger.info("Fast Loop Retrain — %s/%s", self.symbol, self.base_timeframe)
        logger.info("Config: %s", self.config_path)
        logger.info("HMM:    %s", self.hmm_path)
        logger.info("Meta:   %s", self.meta_path or "(none)")
        logger.info("Lookback: %d bars  |  OOS: %.0f%%  |  Seed: %d",
                     self.lookback_bars, self.oos_frac * 100, self.seed)
        logger.info("=" * 60)

        with open(self.config_path) as f:
            self._committee_config = json.load(f)

        meta_cfg = self._committee_config.get("metadata", {})
        self._meta_override_threshold = float(
            meta_cfg.get("meta_override_threshold", 0.50)
        )

        logger.info("[Phase 1] Loading data...")
        self.load_data()

        logger.info("[Phase 2] Computing features...")
        X_full = self.compute_features()

        logger.info("[Phase 3] Scaling (train-only fit)...")
        self.scale_features(X_full)

        logger.info("[Phase 4] Generating triple-barrier labels...")
        self.generate_labels()

        logger.info("[Phase 5] Frozen HMM regime tagging...")
        self.tag_regimes()

        logger.info("[Phase 6] Refitting primary committee models...")
        refitted = self.refit_primary_models()

        logger.info("[Phase 7] Refitting MetaLabeler...")
        self.refit_meta_labeler(refitted)

        logger.info("[Phase 8] Saving artifacts...")
        self.save_artifacts()

        elapsed = time.time() - t0
        self.manifest["elapsed_seconds"] = round(elapsed, 1)
        logger.info("Fast Loop complete in %.1f seconds", elapsed)

        return self.manifest


def main():
    parser = argparse.ArgumentParser(
        description="Fast Loop Rolling Refit — update model weights on recent data",
    )
    parser.add_argument(
        "--config-path", required=True,
        help="Path to committee_config.json (from Slow Loop HPO)",
    )
    parser.add_argument(
        "--hmm-path", required=True,
        help="Path to frozen HMM detector .joblib artifact",
    )
    parser.add_argument(
        "--meta-path", default=None,
        help="Path to frozen MetaLabeler .joblib (for override_threshold reference)",
    )
    parser.add_argument(
        "--lookback-bars", type=int, default=20000,
        help="Rolling window size in bars (default: 20000)",
    )
    parser.add_argument(
        "--oos-frac", type=float, default=0.10,
        help="Fraction held out for MetaLabeler OOS (default: 0.10)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output artifact directory (default: artifacts/fast_loop_update_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--symbol", default="EURUSD",
        help="Trading pair symbol (default: EURUSD)",
    )
    parser.add_argument(
        "--base-timeframe", default="M30",
        help="Base timeframe (default: M30)",
    )
    parser.add_argument(
        "--features-config", default="configs/feature_config.json",
        help="Fallback feature config JSON if not in committee metadata",
    )
    parser.add_argument(
        "--db-path", default="data/forex.db",
        help="Path to SQLite database (default: data/forex.db)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Reproducibility seed (default: 42)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    retrainer = FastRetrainer(
        config_path=args.config_path,
        hmm_path=args.hmm_path,
        meta_path=args.meta_path,
        symbol=args.symbol,
        base_timeframe=args.base_timeframe,
        lookback_bars=args.lookback_bars,
        oos_frac=args.oos_frac,
        output_dir=args.output_dir,
        features_config_path=args.features_config,
        db_path=args.db_path,
        seed=args.seed,
    )

    manifest = retrainer.run()

    print(f"\nFast Loop complete. Artifacts in: {retrainer.output_dir}")
    print(f"Models refitted: {len(manifest['models_refitted'])}")
    print(f"Models skipped:  {len(manifest['models_skipped'])}")
    print(f"MetaLabeler:     {'refitted' if manifest['meta_labeler_refitted'] else 'frozen copy'}")
    return 0 if manifest["models_refitted"] else 1


if __name__ == "__main__":
    sys.exit(main())
