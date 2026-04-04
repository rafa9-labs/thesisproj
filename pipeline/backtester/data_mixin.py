"""Auto-extracted mixin — see composed.py for the full MLBacktester."""
from pipeline._imports import *  # noqa: F401,F403


class DataMixin:
    """
    get_data, feature bank, regime

    Auto-extracted from MLBacktesterNoWFO.py lines 2197-2536.
    """
    def get_data(self) -> None:
        """
        Load and preprocess raw market data for the specified window.

        - 30m data → index=time (tz-aware), rename to price/high/low, compute log returns.
        - 1H / 4H data → loaded for multi-timeframe (MTF) features.
        - Precompute/Load 'mtf_ma_fast' (1H fast MA, shifted) and 'mtf_ma_slow' (4H slow MA, shifted).
        Uses tuned windows from features_config['indicator_windows'] and prefers precomputed columns if present.
        """
        # ---- 30m base data ----
        raw = _load_csv_cached(BASE_CSV, parse_dates=["time"], index_col="time")

        # normalize column names expected downstream
        raw.rename(columns={"mid_close": "price", "mid_high": "high", "mid_low": "low"}, inplace=True)
        raw = raw[["price", "high", "low", "spread"]]

        # compute log-returns
        raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))

        # 🔽 Downcast numeric columns to float32 to save RAM
        for col in ("price", "high", "low", "spread", "returns"):
            if col in raw.columns:
                raw[col] = raw[col].astype("float32")

        # ensure tz-aware index before slicing
        raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
        self.data = raw.loc[self.start:self.end].dropna()

        # Optionally attach macro features (daily / lower-frequency) to bar-level data
        cfg = self.features_config or {}
        if cfg.get("use_macro_features", False):
            macro_specs = cfg.get("macro_sources") or cfg.get("macro_csv_paths") or {}
            if macro_specs:
                try:
                    lag_days = int(cfg.get("macro_lag_days", 1))
                except Exception:
                    lag_days = 1
                try:
                    self.data = attach_macro_features(
                        self.data,
                        macro_specs=macro_specs,
                        lag_days=lag_days,
                    )
                except Exception as _e:
                    if self._is_debug():
                        print(f"⚠️ Failed to attach macro features: {_e}")

        # --- One-time NY session mask (02:00–13:00 NYT) ---
        try:
            _ny_times = self.data.index.tz_convert("America/New_York")
            _ny_active = (_ny_times.hour >= 2) & (_ny_times.hour <= 13)
            self._ny_mask = pd.Series(_ny_active, index=self.data.index)
        except Exception as _e:
            print(f"⚠️ Failed to precompute NY session mask: {_e}")
            self._ny_mask = pd.Series(True, index=self.data.index)  # safe fallback

        # ---- 1H and 4H for MTF features (cached) ----
        self.df_1h = _load_csv_cached(CSV_1H, parse_dates=["time"], index_col="time")
        self.df_4h = _load_csv_cached(CSV_4H, parse_dates=["time"], index_col="time")

        # 🔽 Downcast numeric columns in 1H / 4H to float32 as well
        for _df in (self.df_1h, self.df_4h):
            for _col in _df.columns:
                # only downcast numeric dtypes
                if pd.api.types.is_numeric_dtype(_df[_col].dtype):
                    _df[_col] = _df[_col].astype("float32")

        # ---- Precompute/Load MTF MAs on full history (shift(1) to avoid leakage) ----
        try:
            ind = (self.features_config or {}).get("indicator_windows", {}) or {}
            fast_w = int(ind.get("mtf_ma_fast_window", 10))   # default 10 (1H)
            slow_w = int(ind.get("mtf_ma_slow_window", 50))   # default 50 (4H)

            df1 = self.df_1h.copy()
            df4 = self.df_4h.copy()

            # Prefer precomputed columns if present; else compute from mid_close
            fast_candidates = [
                f"mtf_1h_ma{fast_w}", "mtf_1h_ma_fast", f"ma_1h_{fast_w}", f"ma_fast_{fast_w}"
            ]
            slow_candidates = [
                f"mtf_4h_ma{slow_w}", "mtf_4h_ma_slow", f"ma_4h_{slow_w}", f"ma_slow_{slow_w}"
            ]
            col_fast = next((c for c in fast_candidates if c in df1.columns), None)
            col_slow = next((c for c in slow_candidates if c in df4.columns), None)

            if col_fast is None:
                if "mid_close" not in df1:
                    raise KeyError("1H CSV missing 'mid_close' for MTF compute")
                df1["mtf_1h_ma_fast"] = (
                    df1["mid_close"]
                    .rolling(fast_w, min_periods=fast_w)
                    .mean()
                    .shift(1)
                )
                col_fast = "mtf_1h_ma_fast"

            if col_slow is None:
                if "mid_close" not in df4:
                    raise KeyError("4H CSV missing 'mid_close' for MTF compute")
                df4["mtf_4h_ma_slow"] = (
                    df4["mid_close"]
                    .rolling(slow_w, min_periods=slow_w)
                    .mean()
                    .shift(1)
                )
                col_slow = "mtf_4h_ma_slow"

            # Normalize names for merge
            df1 = df1[[col_fast]].reset_index().rename(columns={col_fast: "mtf_1h_ma_fast"})
            df4 = df4[[col_slow]].reset_index().rename(columns={col_slow: "mtf_4h_ma_slow"})

            # Align timestamps to minute grid so merge_asof matches 30m bars robustly
            df1["time"] = pd.to_datetime(df1["time"], utc=True) + pd.Timedelta(minutes=1)
            df4["time"] = pd.to_datetime(df4["time"], utc=True) + pd.Timedelta(minutes=1)

            # Merge onto current window
            base = self.data.reset_index().rename(columns={"index": "time"})
            base["time"] = pd.to_datetime(base["time"], utc=True)

            mtf_fast = pd.merge_asof(
                base.sort_values("time"), df1.sort_values("time"), on="time", direction="backward"
            ).set_index("time")["mtf_1h_ma_fast"]

            mtf_slow = pd.merge_asof(
                base.sort_values("time"), df4.sort_values("time"), on="time", direction="backward"
            ).set_index("time")["mtf_4h_ma_slow"]

            # assign to self.data aligned to index
            self.data["mtf_ma_fast"] = mtf_fast.reindex(self.data.index).astype("float32")
            self.data["mtf_ma_slow"] = mtf_slow.reindex(self.data.index).astype("float32")

            if self._is_debug():
                print(f"[MTF] fast_w={fast_w}, slow_w={slow_w} (mtf_ma_fast/slow ready)")

        except Exception as _e:
            print(f"⚠️ Precompute/Load MTF features failed: {_e}")

    @staticmethod
    def rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """
        Efficient O(n) rolling slope using cumulative sums.
        Much faster than per-window polyfit.
        """
        x = np.arange(window, dtype=float)
        Sx = x.sum()
        Sxx = (x * x).sum()
        n = window
        den = n * Sxx - Sx * Sx

        y = series.astype(float).to_numpy()
        # handle NaNs safely
        y_filled = np.where(np.isfinite(y), y, 0.0)

        csum_y = np.cumsum(y_filled)
        csum_xy = np.cumsum(y_filled * np.arange(len(y), dtype=float))

        Sy  = csum_y[window-1:] - np.concatenate(([0.0], csum_y[:-window]))
        Sxy = csum_xy[window-1:] - np.concatenate(([0.0], csum_xy[:-window]))

        # numerator for slope
        num = n * (Sxy - np.arange(window-1, len(y)) * Sy) - Sx * Sy
        slope = num / den

        out = np.full_like(y, np.nan, dtype=float)
        out[window-1:] = slope
        return pd.Series(out, index=series.index)
    
    def _ensure_feature_bank(self):
        """
        Build a simple FeatureBank of base indicators over the current `self.data`
        slice if not already present.

        Design:
        - Only base TA indicators + composite features are stored.
        - Lag/rolling expansions and raw returns_lag* are still done per-slice
          inside `prepare_features` to avoid RAM blow-up.
        - The bank is keyed by (data span + toggles + indicator_windows +
          RV/fracdiff settings + price_col). If that signature changes, the bank
          is rebuilt.
        """
        import pandas as pd
        import numpy as np
        import os, psutil

        # Prefer a stable source span if provided; else default to self.data
        src = getattr(self, "_feature_bank_src", None)
        if src is None:
            src = getattr(self, "data", None)
        if src is None or len(src) == 0:
            return

        cfg = self.features_config or {}
        ind_win = (cfg.get("indicator_windows", {}) or {})

        # Compute the desired key cheaply (even under low RAM) so we never reuse a stale bank.
        idx = pd.DatetimeIndex(src.index)
        first_idx = idx[0]
        last_idx  = idx[-1]
        toggles_on = tuple(sorted(k for k, v in cfg.items() if str(k).startswith("use_") and bool(v)))
        key = (
            first_idx,
            last_idx,
            int(len(idx)),
            toggles_on,
            tuple(sorted((k, str(v)) for k, v in ind_win.items())),
            bool(cfg.get("use_rv_features", False)),
            int(cfg.get("rv_window_short", 30)),
            int(cfg.get("rv_window_long", 120)),
            bool(cfg.get("use_fracdiff", False)),
            float(cfg.get("fracdiff_d", 0.4)),
            cfg.get("price_col", "price"),
        )

        # If existing bank matches this signature, keep it
        if getattr(self, "_feature_bank_full", None) is not None and getattr(self, "_feature_bank_key", None) == key:
            return

        # Low-RAM guard: if we skip building, CLEAR any mismatched/stale bank so it cannot be reused.
        avail_gb   = psutil.virtual_memory().available / (1024 ** 3)
        trigger_gb = float(os.getenv("LOW_RAM_TRIGGER_GB", "1.25"))
        force_off  = os.getenv("MLB_FEATUREBANK_OFF", "0") in ("1", "true", "True")
        # Optional: keep FeatureBank off during Optuna CV unless explicitly allowed
        in_cv = bool(getattr(self, "_in_optuna_cv", False))
        if in_cv and not bool(cfg.get("featurebank_in_cv", False)):
            force_off = True

        if force_off or avail_gb < trigger_gb:
            # ensure no stale reuse
            self._feature_bank_full = None
            self._feature_bank_meta = {}
            self._feature_bank_key  = None
            if self._is_debug():
                print(
                    f"[FeatureBank] Disabled/cleared (avail={avail_gb:.2f} GB, trigger={trigger_gb:.2f} GB, "
                    f"force_off={force_off}, in_cv={in_cv})"
                )
            return


        # Otherwise rebuild (clear old one first to free RAM ASAP)
        self._feature_bank_full = None
        self._feature_bank_meta = {}
        self._feature_bank_key  = None

        try:
            # We call prepare_features in "base_only" mode so it computes only
            # base indicators + composites, with NO lag/rolling expansion.
            lags_default = int(cfg.get("lags_range", cfg.get("lags", 10)))
            lag_depth    = int(cfg.get("lag_depth", 1))
            roll_windows = cfg.get("roll_windows", [5])
            if roll_windows is None:
                roll_windows = [5]

            df_feat, base_feats = self.prepare_features(
                src,
                lags=lags_default,
                lag_depth=lag_depth,
                roll_windows=roll_windows,
                base_only=True,        # <-- NEW flag
            )

            # Keep only numeric columns and downcast to float32 to control RAM
            fb = df_feat.select_dtypes(include=["number"]).astype("float32", copy=False)

            self._feature_bank_full = fb
            self._feature_bank_meta = {"base_features": list(base_feats)}
            self._feature_bank_key  = key

            if self._is_debug():
                print(
                    f"[FeatureBank] Built base-indicator bank: "
                    f"shape={fb.shape}, base_features={len(base_feats)}"
                )

        except Exception as e:
            # Fail silently (with debug print) and fall back to per-slice path
            if self._is_debug():
                print(f"⚠️ FeatureBank build failed; falling back to per-slice TA: {e}")
            self._feature_bank_full = None
            self._feature_bank_meta = {}
            self._feature_bank_key  = None

    def _attach_regime_columns(self, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """
        Add trend_score, vol_score, regime_id, and optional one-hot regime_* columns.
        Uses adx_col/vol_col + adx_thresh/vol_thresh from cfg or sane defaults.


        - 0 = SIDEWAYS
        - 1 = TREND
        - 2 = VOLATILE/CHOPPY
        """
        # Default ADX column used as trend proxy
        adx_col = cfg.get("adx_col") or "adx_14"

        # Default volatility proxy: realized volatility with the "short" window
        vol_col = cfg.get("vol_col")
        if not vol_col:
            rv_short = int(cfg.get("rv_window_short", 48))
            vol_col = f"rv_{rv_short}"

        adx_thr = float(cfg.get("adx_thresh", 20.0))
        # Prefer the train-anchored high-vol threshold already used by the cost model
        # (keeps regime segmentation consistent + avoids scale issues / collapse-to-volatile)
        if cfg.get("high_vol_thr") is not None:
            vol_thr = float(cfg.get("high_vol_thr"))
        else:
            vol_thr = float(cfg.get("vol_thresh", 0.001))  # fallback

        # 1) Guard: if these cols don’t exist, just skip and return df unmodified
        if adx_col not in df.columns or vol_col not in df.columns:
            # no regime annotation possible; keep compatibility
            df["regime_id"] = 1  # or SIDEWAYS default
            return df

        trend_score = df[adx_col].astype("float64")
        vol_score   = df[vol_col].astype("float64")

        # 2) Regime classification
        regime = np.full(len(df), 0, dtype="int8")   # 0 = SIDEWAYS

        trend_mask = trend_score >= adx_thr
        vol_high   = vol_score  >  vol_thr

        regime[trend_mask & ~vol_high] = 1  # TREND
        regime[~trend_mask & vol_high] = 2  # VOLATILE
        regime[trend_mask & vol_high]  = 2  # strong but wild → treat as VOLATILE for now

        df["trend_score"] = trend_score
        df["vol_score"]   = vol_score
        df["regime_id"]   = regime

        # Optional one-hots (helps classical models)
        df["regime_trend"]    = (regime == 1).astype("int8")
        df["regime_sideways"] = (regime == 0).astype("int8")
        df["regime_volatile"] = (regime == 2).astype("int8")
        
        return df

