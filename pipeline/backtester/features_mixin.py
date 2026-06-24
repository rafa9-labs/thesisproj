"""Auto-extracted mixin -- see composed.py for the full MLBacktester."""
from pipeline._imports import *  # noqa: F401,F403


class FeaturesMixin:
    """
    prepare_features, scale, labels

    Auto-extracted from MLBacktesterNoWFO.py lines 2537-3365.
    """
    def prepare_features(
        self,
        df: pd.DataFrame,
        lags: int,
        lag_depth: int = 1,
        roll_windows: list[int] = [5],
        base_only: bool = False,
    ):

        """
        Create feature matrix using toggles & windows in `self.features_config`.

        Pipeline
        --------
        1) Compute base indicators (as toggled in config).
        2) Add SAR unconditionally.
        3) Include MTF moving averages if present (computed in get_data()).
        4) Momentum extensions: EMA-SMA spread, price-MA z-scores, crossover bins, slope differential.
        5) Composite features: re-entry + momentum, extension/ATR with low ADX, squeeze->expansion, ATR channels,
        trend confirmation, MTF alignment, volatility-managed momentum, MACD/ATR ratio.
        6) Expand with lags/rolling stats; add hour features; drop NAs on active features.

        Notes
        -----
        - Adds columns to a copy of `df`; does not mutate input.
        - Respects indicator_windows and use_* toggles in `self.features_config`.
        - Higher-TF MAs must be precomputed in get_data() (shifted to avoid look-ahead).
        """

        # CLEANUP: cache debug flag once
        debug = bool(getattr(self, "_is_debug", lambda: False)())

        # CLEANUP: tiny local helper to avoid silent exception swallowing
        def _debug_once(tag: str, exc: Exception):
            # DEBUG: print a given exception at most once per function lifetime per instance
            if not debug:
                return
            attr = f"_prepare_features_exc_once__{tag}"
            if getattr(self, attr, False):
                return
            try:
                print(f"[WARN] [prepare_features][{tag}] {type(exc).__name__}: {exc}")
            except Exception:
                # Last-resort: never crash due to logging
                pass
            setattr(self, attr, True)

        # CLEANUP: tiny local helper to reduce repeated H/L/C extraction blocks
        def _get_hlc(_df: pd.DataFrame, _price_col: str):
            hi_ = _df.get("high", _df.get(_price_col))
            lo_ = _df.get("low", _df.get(_price_col))
            cl_ = _df.get("close", _df.get(_price_col))
            return hi_, lo_, cl_

        # Safety belt: normalize index to DatetimeIndex for stable caching / FeatureBank alignment
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
        except Exception as e:
            # CLEANUP/DEBUG: keep non-fatal behavior, but don't swallow silently in debug
            _debug_once("idx_to_datetime", e)

        # ---------- 0) Cache ----------
        if not hasattr(self, "_feat_cache"):
            self._feat_cache = {}

        # Telemetry-only: track approx bytes retained in the cache (truthful "current cache size")
        if not hasattr(self, "_feat_cache_bytes"):
            self._feat_cache_bytes = {}

        cfg = self.features_config or {}
        ind_win = (cfg.get("indicator_windows", {}) or {})

        # Feature-slice caching is now opt-in (default OFF) and always bypassed during Optuna CV.
        # Rationale: cache_key includes slice boundaries -> reuse is usually ~0 in walk-forward/monthly runs.
        in_cv = bool(getattr(self, "_in_optuna_cv", False))
        # Canonical flag is "slice_cache_enabled" (default OFF).
        # Back-compat: allow older configs that used "feat_cache_enabled".
        slice_cache_enabled = bool(cfg.get("slice_cache_enabled", cfg.get("feat_cache_enabled", False)))
        cache_enabled = slice_cache_enabled and (not in_cv) and (not base_only)

        # Emit cache mode once per run so it is always obvious whether we are caching or not.
        if (LOG_MODE in {"COMPACT", "DEBUG"}) and (not getattr(self, "_feat_cache_mode_logged", False)):
            msg = "[FEAT_CACHE] enabled (opt-in)" if slice_cache_enabled else "[FEAT_CACHE] disabled (default)"
            if slice_cache_enabled and in_cv:
                msg += " but BYPASSED during Optuna CV"
            print(msg)
            self._feat_cache_mode_logged = True  # CLEANUP: was False -> caused repeated prints forever

        # Optional safety net: cap number of cached engineered slices (default 0 = unlimited).
        # This is a *resource* guard only: it must not affect results because the cache is
        # only an optimization (equivalent to recomputing features).
        try:
            feat_cache_max_entries = int(cfg.get("feat_cache_max_entries", 0) or 0)
        except Exception:
            feat_cache_max_entries = 0

        #  --- Normalize roll_windows early (needed for cache_key safety) ---
        rw_cfg = cfg.get("roll_windows", roll_windows if roll_windows is not None else [5])
        if isinstance(rw_cfg, str):
            roll_windows = [int(x.strip()) for x in rw_cfg.split(",") if x.strip()]
        elif isinstance(rw_cfg, (list, tuple)):
            roll_windows = [int(x) for x in rw_cfg]
        else:
            roll_windows = [int(rw_cfg)]

        # Build a cache key that reflects settings affecting columns
        start_idx = df.index[0] if len(df.index) > 0 else None
        end_idx   = df.index[-1] if len(df.index) > 0 else None
        toggles_on = tuple(sorted(k for k, v in cfg.items() if str(k).startswith("use_") and v))
        cache_key = (
            start_idx, end_idx,
            int(cfg.get("lags_range", cfg.get("lags", lags if lags is not None else 10))),
            int(cfg.get("lag_depth", lag_depth if lag_depth is not None else 1)),
            tuple(roll_windows),
            tuple(sorted((k, str(v)) for k, v in ind_win.items())),
            toggles_on,
            bool(cfg.get("include_raw_lags", True)),
            bool(cfg.get("include_hour", True)),
            bool(cfg.get("use_rv_features", False)),
            int(cfg.get("rv_window_short", 30)),
            int(cfg.get("rv_window_long", 120)),
            bool(cfg.get("use_fracdiff", False)),
            float(cfg.get("fracdiff_d", 0.4)),
            bool(cfg.get("include_hour_cyclic", True)),
            cfg.get("price_col", "price"),
        )

        cached = (self._feat_cache.get(cache_key) if cache_enabled else None)
        if cached is not None:
            # Reuse previously engineered features for this exact slice/config combo.
            df_cached, feat_cached = cached

            # Diagnostics: cache hit
            self._feat_cache_hits = int(getattr(self, "_feat_cache_hits", 0)) + 1
            if LOG_MODE in {"COMPACT", "DEBUG"}:
                try:
                    n_entries = len(self._feat_cache) if isinstance(self._feat_cache, dict) else -1
                    n_feats = len(feat_cached)
                    hits = int(getattr(self, "_feat_cache_hits", 0))
                    misses = int(getattr(self, "_feat_cache_misses", 0))
                    denom = hits + misses
                    hit_rate = (hits / denom) if denom > 0 else 0.0
                    cur_bytes = int(getattr(self, "_feat_cache_cur_bytes", 0))
                    do_print = (LOG_MODE == "DEBUG") or (self._feat_cache_hits % 25 == 0)
                    if do_print and bool(int(os.getenv("KODAQUANT_VERBOSE", "0"))):
                        print(
                            f"[FEAT_CACHE] HIT  entries={n_entries} "
                            f"hits={hits} misses={misses} hit_rate={hit_rate:.2%} "
                            f"cache_mb={cur_bytes/1024/1024:.1f} feats={n_feats}"
                        )
                except Exception as e:
                    # DEBUG: don't spam, but don't hide forever
                    _debug_once("feat_cache_hit_diag", e)

            # Keep last-used features up to date for downstream logging.
            feat_cached = list(feat_cached)
            self._last_used_features = list(feat_cached)
            return df_cached, feat_cached  # (df_out, features)

        # --- Disk cache: try loading before recomputing ---
        if not base_only and not in_cv:
            try:
                from pipeline.features.feature_cache import load_from_disk
                import hashlib as _hashlib_dl, json as _json_dl
                _disk_key = _hashlib_dl.sha256(
                    _json_dl.dumps(cache_key, default=str).encode()
                ).hexdigest()[:16]
                _disk_cached = load_from_disk(_disk_key)
                if _disk_cached is not None:
                    df_out, features = _disk_cached
                    self._last_used_features = list(features)
                    # Also populate in-memory cache so subsequent calls in same run are fast
                    if cache_enabled:
                        self._feat_cache[cache_key] = (df_out, tuple(features))
                    if LOG_MODE in {"COMPACT", "DEBUG"}:
                        if bool(int(os.getenv("KODAQUANT_VERBOSE", "0"))):
                            print(f"[DISK_CACHE] HIT key={_disk_key[:8]} rows={len(df_out)} feats={len(features)}")
                    return df_out, list(features)
            except Exception as _e:
                if debug:
                    print(f"[DISK_CACHE] load failed: {_e}")

        # ---------- 1) Params & toggles ----------
        # Windows (accept *_window aliases)
        window_sma = int(ind_win.get("sma", ind_win.get("sma_window", 20)))
        window_ema = int(ind_win.get("ema", ind_win.get("ema_window", 20)))
        window_rsi = int(ind_win.get("rsi", ind_win.get("rsi_window", 14)))

        macd_fast   = int(ind_win.get("macd_fast", 12))
        macd_slow   = int(ind_win.get("macd_slow", 26))
        macd_signal = int(ind_win.get("macd_signal", 9))

        bb_window = int(ind_win.get("bb_window", 20))
        bb_dev    = float(ind_win.get("bb_dev", 2.0))

        atr_win = int(ind_win.get("atr", ind_win.get("atr_window", 14)))
        adx_win = int(ind_win.get("adx", ind_win.get("adx_window", 14)))

        stoch_k_win = int(ind_win.get("stoch_k", ind_win.get("stoch_k_window", 14)))
        stoch_d_win = int(ind_win.get("stoch_d", ind_win.get("stoch_d_window", 3)))

        # Feature toggles (defaults keep backward-compatibility)
        toggles = dict(
            use_sma   = cfg.get("use_sma", True),
            use_ema   = cfg.get("use_ema", True),
            use_rsi   = cfg.get("use_rsi", True),
            use_macd  = cfg.get("use_macd", True),
            use_bbands= cfg.get("use_bbands", True),
            use_atr   = cfg.get("use_atr", True),
            use_adx   = cfg.get("use_adx", True),
            use_stoch = cfg.get("use_stoch", True),
            use_mtf_ma= cfg.get("use_mtf_ma", True),
        )

        # Indicator state configuration (oscillator & volatility regimes)
        use_indicator_states    = bool(cfg.get("use_indicator_states", False))
        rsi_overbought_level    = float(cfg.get("rsi_overbought_level", 70))
        rsi_oversold_level      = float(cfg.get("rsi_oversold_level", 30))
        stoch_overbought_level  = float(cfg.get("stoch_overbought_level", 80))
        stoch_oversold_level    = float(cfg.get("stoch_oversold_level", 20))
        bbw_compress_threshold  = float(cfg.get("bbw_compress_threshold", 0.05))
        bbw_expand_threshold    = float(cfg.get("bbw_expand_threshold", 0.20))

        # Momentum extensions
        use_ma_spread      = bool(cfg.get("use_ma_spread", False))
        use_price_ma_z     = bool(cfg.get("use_price_ma_z", False))
        use_crossover_bins = bool(cfg.get("use_crossover_bins", False))
        use_slope_diff     = bool(cfg.get("use_slope_diff", False))

        # Composite toggles
        use_reentry_mom        = bool(cfg.get("use_reentry_mom", False))
        use_ext_atr_low_adx    = bool(cfg.get("use_ext_atr_low_adx", False))
        use_squeeze_expansion  = bool(cfg.get("use_squeeze_expansion", False))
        use_atr_channel_break  = bool(cfg.get("use_atr_channel_breakout", False))
        use_trend_confirm      = bool(cfg.get("use_trend_confirm", False))
        use_mtf_alignment      = bool(cfg.get("use_mtf_alignment", False))
        use_vol_managed_mom    = bool(cfg.get("use_vol_managed_mom", False))
        use_macd_atr_ratio     = bool(cfg.get("use_macd_atr_ratio", False))

        # Effective lags / depth / rolling windows
        num_lags = int(lags if lags is not None else 10)
        if "lags" in cfg:       num_lags = int(cfg["lags"])
        if "lags_range" in cfg: num_lags = int(cfg["lags_range"])

        lag_depth = int(cfg.get("lag_depth", lag_depth if lag_depth is not None else 1))

        # ---------- 2) Base indicators ----------
        price_col = cfg.get("price_col", "price")
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype("float32", copy=False)
        if price_col not in df.columns:
            price_col = "close" if "close" in df.columns else price_col

        base_cols: dict[str, pd.Series] = {}
        base_features: list[str] = []

        # CLEANUP: one canonical regime list used in two places later
        regime_cols = [
            "trend_score",
            "vol_score",
            "regime_id",
            "regime_trend",
            "regime_sideways",
            "regime_volatile",
        ]

        # Decide whether to reuse precomputed FeatureBank (only when not building it)
        use_fb = (
            not base_only
            and getattr(self, "_feature_bank_full", None) is not None
            and isinstance(getattr(self, "_feature_bank_meta", None), dict)
            and bool(getattr(self, "_feature_bank_meta", {}).get("base_features"))
        )

        if use_fb:
            try:
                fb = self._feature_bank_full
                meta = self._feature_bank_meta or {}
                base_features = list(meta.get("base_features", []))

                # Align FeatureBank slice to current df index
                fb_slice = fb.reindex(df.index)

                # Attach base/composite features from the bank
                df = pd.concat([df, fb_slice[base_features]], axis=1)
                df = df.loc[:, ~df.columns.duplicated(keep="last")]

            except Exception as e:
                if debug:
                    print(
                        f"[WARN] FeatureBank reuse failed in prepare_features; "
                        f"falling back to per-slice TA: {e}"
                    )
                use_fb = False
                base_features = []
                base_cols = {}

        if not use_fb:
            if "returns" in df:
                base_cols["rolling_std_20"] = df["returns"].rolling(20).std()
                base_features.append("rolling_std_20")

            # SMA / EMA
            if toggles["use_sma"] and price_col in df:
                name = f"sma_{window_sma}"
                base_cols[name] = ta.trend.sma_indicator(df[price_col], window=window_sma)
                base_features.append(name)

            if toggles["use_ema"] and price_col in df:
                name = f"ema_{window_ema}"
                base_cols[name] = ta.trend.ema_indicator(df[price_col], window=window_ema)
                base_features.append(name)

            # MACD (line, signal, diff)
            if toggles["use_macd"] and price_col in df:
                macd_obj = ta.trend.MACD(
                    df[price_col],
                    window_slow=macd_slow,
                    window_fast=macd_fast,
                    window_sign=macd_signal,
                )
                base_cols["macd_line"]   = macd_obj.macd()
                base_cols["macd_signal"] = macd_obj.macd_signal()
                base_cols["macd_diff"]   = macd_obj.macd_diff()
                base_features += ["macd_line", "macd_signal", "macd_diff"]

            # RSI
            if toggles["use_rsi"] and price_col in df:
                name = f"rsi_{window_rsi}"
                base_cols[name] = ta.momentum.RSIIndicator(df[price_col], window=window_rsi).rsi()
                base_features.append(name)

            # Bollinger Bands (+ width, %B)
            if toggles["use_bbands"] and price_col in df:
                bb = ta.volatility.BollingerBands(df[price_col], window=bb_window, window_dev=bb_dev)
                upper, lower = bb.bollinger_hband(), bb.bollinger_lband()
                base_cols["bb_upper"] = upper
                base_cols["bb_lower"] = lower
                base_cols["bb_pct"]   = (df[price_col] - lower) / (upper - lower)
                base_cols["bbw"]      = bb.bollinger_wband()
                base_features += ["bb_upper", "bb_lower", "bb_pct", "bbw"]

            # ATR
            if toggles["use_atr"]:
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    name = f"atr_{atr_win}"
                    base_cols[name] = ta.volatility.AverageTrueRange(hi, lo, cl, window=atr_win).average_true_range()
                    base_features.append(name)

                    # ATR-normalized spread (optional toggle)
                    if toggles.get("use_spread_over_atr", False) and ("spread" in df.columns):
                        eps = 1e-8
                        atr_series = base_cols[name].astype(float).replace(0.0, np.nan)
                        spread_series = df["spread"].astype(float)
                        spread_atr = (spread_series / atr_series).replace([np.inf, -np.inf], np.nan)
                        base_cols[f"spread_atr_{atr_win}"] = spread_atr
                        base_features.append(f"spread_atr_{atr_win}")

            # Donchian-style price channels (multi-horizon high/low bands)
            use_donchian = bool(cfg.get("use_donchian", False))
            if use_donchian:
                hi = df.get("high", df.get(price_col))
                lo = df.get("low", df.get(price_col))
                cl = df.get("close", df.get(price_col))
                if (hi is not None) and (lo is not None) and (cl is not None):
                    w_s = int(cfg.get("donchian_window_short", 20))
                    w_l = int(cfg.get("donchian_window_long", 60))
                    for w in sorted({w_s, w_l}):
                        dc_high = hi.rolling(w, min_periods=max(5, w // 3)).max()
                        dc_low  = lo.rolling(w, min_periods=max(5, w // 3)).min()
                        up_col  = f"donchian_up_{w}"
                        dn_col  = f"donchian_dn_{w}"
                        bu_col  = f"donchian_break_up_{w}"
                        bd_col  = f"donchian_break_dn_{w}"
                        base_cols[up_col] = dc_high
                        base_cols[dn_col] = dc_low
                        base_cols[bu_col] = (cl >= dc_high).astype("int8")
                        base_cols[bd_col] = (cl <= dc_low).astype("int8")
                        base_features += [up_col, dn_col, bu_col, bd_col]

            # ADX
            if toggles["use_adx"]:
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    name = f"adx_{adx_win}"
                    base_cols[name] = ta.trend.ADXIndicator(hi, lo, cl, window=adx_win).adx()
                    base_features.append(name)

            # Stochastic (K, D)
            if toggles.get("use_stoch", True):
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    k, d = int(stoch_k_win), int(stoch_d_win)
                    Hh = hi.rolling(k, min_periods=k).max()
                    Ll = lo.rolling(k, min_periods=k).min()
                    stoch_k = 100.0 * (cl - Ll) / (Hh - Ll).replace(0.0, np.nan)
                    base_cols["stoch_k"] = stoch_k
                    base_cols["stoch_d"] = stoch_k.rolling(d, min_periods=d).mean()
                    base_features += ["stoch_k", "stoch_d"]

            # Indicator states (oscillators & volatility compression/expansion)
            if use_indicator_states:
                if toggles.get("use_rsi", True):
                    rsi_col = f"rsi_{window_rsi}"
                    if rsi_col in base_cols:
                        rsi_ser = base_cols[rsi_col].astype(float)
                        rsi_state = pd.Series(0, index=rsi_ser.index, dtype="int8")
                        rsi_state[rsi_ser >= rsi_overbought_level] = 1
                        rsi_state[rsi_ser <= rsi_oversold_level] = -1
                        base_cols["rsi_state"] = rsi_state
                        if "rsi_state" not in base_features:
                            base_features.append("rsi_state")

                if toggles.get("use_stoch", True) and "stoch_k" in base_cols:
                    stoch_ser = base_cols["stoch_k"].astype(float)
                    st_state = pd.Series(0, index=stoch_ser.index, dtype="int8")
                    st_state[stoch_ser >= stoch_overbought_level] = 1
                    st_state[stoch_ser <= stoch_oversold_level] = -1
                    base_cols["stoch_state"] = st_state
                    if "stoch_state" not in base_features:
                        base_features.append("stoch_state")

                if toggles.get("use_bbands", True) and "bbw" in base_cols:
                    bbw_ser = base_cols["bbw"].astype(float)
                    vol_state = pd.Series(0, index=bbw_ser.index, dtype="int8")
                    vol_state[bbw_ser <= bbw_compress_threshold] = -1
                    vol_state[bbw_ser >= bbw_expand_threshold]   = 1
                    base_cols["vol_state_bbw"] = vol_state
                    if "vol_state_bbw" not in base_features:
                        base_features.append("vol_state_bbw")

            # SAR always on
            hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
            if (hi is not None) and (lo is not None) and (cl is not None):
                base_cols["sar"] = ta.trend.PSARIndicator(hi, lo, cl).psar()
                base_features.append("sar")

            # MTF MAs (provided by get_data)
            if toggles["use_mtf_ma"]:
                for c in ("mtf_ma_fast", "mtf_ma_slow"):
                    if c in df.columns and c not in base_features:
                        base_features.append(c)

            # Optional realized-volatility & bipower variation -- add before expansion so they get lags/rolls
            if cfg.get("use_rv_features", False) and "returns" in df:
                w_s = int(cfg.get("rv_window_short", 30))
                w_l = int(cfg.get("rv_window_long", 120))
                def _rv(s, w):
                    rv2 = s.pow(2).rolling(w, min_periods=max(5, w//3)).sum()
                    return np.sqrt(rv2)
                def _bpv(s, w):
                    abs_r = s.abs(); prod = abs_r * abs_r.shift(1)
                    bpv = (np.pi/2.0) * prod.rolling(w, min_periods=max(5, w//3)).sum()
                    return np.sqrt(bpv.clip(lower=0))
                base_cols[f"rv_{w_s}"]  = _rv(df["returns"], w_s)
                base_cols[f"rv_{w_l}"]  = _rv(df["returns"], w_l)
                base_cols[f"bpv_{w_s}"] = _bpv(df["returns"], w_s)
                base_cols[f"bpv_{w_l}"] = _bpv(df["returns"], w_l)
                base_cols[f"rv_roc_{w_s}"] = base_cols[f"rv_{w_s}"].pct_change()
                base_cols[f"rv_roc_{w_l}"] = base_cols[f"rv_{w_l}"].pct_change()
                base_features += [f"rv_{w_s}", f"rv_{w_l}", f"bpv_{w_s}", f"bpv_{w_l}", f"rv_roc_{w_s}", f"rv_roc_{w_l}"]

            # Optional fractional-diff seed (included in expansion)
            def _fracdiff_weights(d: float, size: int, thresh: float = 1e-4) -> np.ndarray:
                w = [1.0]
                for k in range(1, size):
                    w_k = -w[-1] * (d - (k - 1)) / k
                    if abs(w_k) < thresh:
                        break
                    w.append(w_k)
                return np.array(w, dtype="float64")
            def _fracdiff(series: pd.Series, d: float = 0.4, max_size: int = 2000, thresh: float = 1e-4) -> pd.Series:
                s = series.astype("float64")
                w = _fracdiff_weights(d, min(max_size, len(s)), thresh=thresh)
                out = np.full(len(s), np.nan, dtype="float64")
                kmax = len(w) - 1; vals = s.values
                for t in range(kmax, len(s)):
                    window = vals[t - kmax : t + 1]
                    out[t] = float(np.dot(w[::-1], window))
                return pd.Series(out, index=s.index, name=f"fd_{getattr(series, 'name','x')}_d{d:.2f}")
            if cfg.get("use_fracdiff", False) and price_col in df:
                d = float(cfg.get("fracdiff_d", 0.4))
                fd = _fracdiff(df[price_col], d=d)
                base_cols[fd.name] = fd
                base_features.append(fd.name)

            # ---------- 3) Momentum extensions (AFTER base indicators) ----------
            price_s = df.get(price_col)
            sma_col = f"sma_{window_sma}"
            ema_col = f"ema_{window_ema}"
            _eps = 1e-8

            if use_ma_spread and (ema_col in base_cols) and (sma_col in base_cols):
                base_cols["ema_sma_spread"] = (base_cols[ema_col] - base_cols[sma_col])
                base_features.append("ema_sma_spread")

            if use_price_ma_z and (price_s is not None):
                if sma_col in base_cols:
                    sd_sma = price_s.rolling(window_sma, min_periods=max(5, window_sma//3)).std(ddof=0)
                    base_cols[f"price_sma_z_{window_sma}"] = (price_s - base_cols[sma_col]) / (sd_sma + _eps)
                    base_features.append(f"price_sma_z_{window_sma}")
                if ema_col in base_cols:
                    sd_ema = price_s.rolling(window_ema, min_periods=max(5, window_ema//3)).std(ddof=0)
                    base_cols[f"price_ema_z_{window_ema}"] = (price_s - base_cols[ema_col]) / (sd_ema + _eps)
                    base_features.append(f"price_ema_z_{window_ema}")

            if use_crossover_bins:
                if (sma_col in base_cols) and (price_s is not None):
                    base_cols["price_gt_sma"] = (price_s > base_cols[sma_col]).astype(int)
                    base_features.append("price_gt_sma")
                if (ema_col in base_cols) and (price_s is not None):
                    base_cols["price_gt_ema"] = (price_s > base_cols[ema_col]).astype(int)
                    base_features.append("price_gt_ema")
                trend_proxy = base_cols.get("macd_diff")
                if trend_proxy is None and (ema_col in base_cols) and (sma_col in base_cols):
                    trend_proxy = (base_cols[ema_col] - base_cols[sma_col])
                if trend_proxy is not None:
                    base_cols["ma_cross_up"] = (trend_proxy > 0).astype(int)
                    base_cols["ma_cross_dn"] = (trend_proxy < 0).astype(int)
                    base_features += ["ma_cross_up", "ma_cross_dn"]

            if use_slope_diff:
                w_sd = max(5, min(window_ema, window_sma)//2)
                x = base_cols.get("macd_diff")
                if x is None and (ema_col in base_cols) and (sma_col in base_cols):
                    x = (base_cols[ema_col] - base_cols[sma_col])
                if x is not None:
                    base_cols[f"ma_spread_slope{w_sd}"] = self.rolling_slope(pd.Series(x).ffill(), w_sd)
                    base_features.append(f"ma_spread_slope{w_sd}")

            # ---------- 4) Composite features (built from existing columns) ----------
            ema_s = base_cols.get(ema_col)
            atr_s = base_cols.get(f"atr_{atr_win}") if f"atr_{atr_win}" in base_cols else None
            adx_s = base_cols.get(f"adx_{adx_win}") if f"adx_{adx_win}" in base_cols else None
            bbw_s = base_cols.get("bbw")
            macd_d = base_cols.get("macd_diff")
            rsi_s  = base_cols.get(f"rsi_{window_rsi}") if f"rsi_{window_rsi}" in base_cols else None

            if ema_s is None and (price_s is not None):
                ema_s = ta.trend.ema_indicator(price_s, window=window_ema)
            if atr_s is None:
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    atr_s = ta.volatility.AverageTrueRange(hi, lo, cl, window=atr_win).average_true_range()
            if adx_s is None:
                hi, lo, cl = _get_hlc(df, price_col)  # CLEANUP
                if (hi is not None) and (lo is not None) and (cl is not None):
                    adx_s = ta.trend.ADXIndicator(hi, lo, cl, window=adx_win).adx()
            if bbw_s is None and price_s is not None:
                bb_tmp = ta.volatility.BollingerBands(price_s, window=bb_window, window_dev=bb_dev)
                bbw_s = bb_tmp.bollinger_wband()

            if use_reentry_mom and (price_s is not None) and (rsi_s is not None):
                if "bb_pct" in base_cols:
                    bb_pct = base_cols["bb_pct"]
                else:
                    bb_tmp = ta.volatility.BollingerBands(price_s, window=bb_window, window_dev=bb_dev)
                    upper, lower = bb_tmp.bollinger_hband(), bb_tmp.bollinger_lband()
                    bb_pct = (price_s - lower) / (upper - lower + _eps)
                reenter = ((bb_pct.shift(1) < 0) & (bb_pct >= 0)).astype(float)
                rsi_slope = self.rolling_slope(rsi_s.ffill(), 5)
                base_cols["reentry_mom"] = reenter * rsi_slope.clip(lower=0.0)
                base_features.append("reentry_mom")

            if use_ext_atr_low_adx and (price_s is not None) and (ema_s is not None) and (atr_s is not None) and (adx_s is not None):
                ext_atr = (price_s - ema_s).abs() / (atr_s + _eps)
                adx_norm = (adx_s / 50.0).clip(0.0, 1.0)
                base_cols["ext_atr_low_adx"] = ext_atr * (1.0 - adx_norm)
                base_features.append("ext_atr_low_adx")

            if use_squeeze_expansion and (bbw_s is not None) and (adx_s is not None):
                w_sq = int(cfg.get("squeeze_window", 300))
                q    = float(cfg.get("squeeze_quantile", 0.10))
                def _pct_rank_last(x: np.ndarray) -> float:
                    s = pd.Series(x)
                    return float(s.rank(pct=True).iloc[-1]) if len(s) else np.nan
                bbw_rank = bbw_s.rolling(w_sq, min_periods=max(30, w_sq//5)).apply(_pct_rank_last, raw=True)
                adx_sl = self.rolling_slope(adx_s.ffill(), 5).clip(lower=0.0)
                base_cols["squeeze_expansion"] = ((q - bbw_rank).clip(lower=0.0)) * adx_sl
                base_features.append("squeeze_expansion")

            if use_atr_channel_break and (price_s is not None) and (ema_s is not None) and (atr_s is not None):
                m = float(cfg.get("atr_channel_mult", 1.5))
                base_cols["atr_ch_up"] = ((price_s - ema_s) / (atr_s + _eps)) - m
                base_cols["atr_ch_dn"] = ((ema_s - price_s) / (atr_s + _eps)) - m
                base_features += ["atr_ch_up", "atr_ch_dn"]

            if use_trend_confirm and (price_s is not None) and (ema_s is not None) and (adx_s is not None):
                adx_sl = self.rolling_slope(adx_s.ffill(), 5).clip(lower=0.0)
                macd_ok = (macd_d > 0).astype(float) if macd_d is not None else 1.0
                price_ok = (price_s > ema_s).astype(float)
                base_cols["trend_confirm"] = price_ok * macd_ok * adx_sl
                base_features.append("trend_confirm")

            if use_mtf_alignment and (price_s is not None) and (ema_s is not None) and ("mtf_ma_fast" in df):
                mtf_sl = self.rolling_slope(df["mtf_ma_fast"].ffill(), 5)
                base_cols["mtf_align"] = ((price_s > ema_s).astype(float)) * (mtf_sl > 0).astype(float)
                base_features.append("mtf_align")

            if use_vol_managed_mom and (price_s is not None) and (ema_s is not None) and (atr_s is not None):
                base_cols["mom_vmm"] = (price_s - ema_s) / (atr_s + _eps)
                base_features.append("mom_vmm")

            if use_macd_atr_ratio and (macd_d is not None) and (atr_s is not None):
                base_cols["macd_atr"] = macd_d / (atr_s + _eps)
                base_features.append("macd_atr")

            # ---------- 5) One-shot concat of base columns ----------
            if base_cols:
                for name, series in base_cols.items():
                    if hasattr(series, 'astype') and series.dtype != np.float32:
                        try:
                            base_cols[name] = series.astype("float32")
                        except (ValueError, TypeError):
                            pass
                base_cols_df = pd.DataFrame(base_cols, index=df.index)
                df = pd.concat([df, base_cols_df], axis=1)
                del base_cols, base_cols_df
                df = df.loc[:, ~df.columns.duplicated(keep="last")]

            # ---- Regime features (trend_score, vol_score, regime_id/one-hot) ----
            if bool(cfg.get("use_regime_features", True)):
                df = self._attach_regime_columns(df, cfg)
                for c in regime_cols:  # CLEANUP: single source of truth
                    if c in df.columns and c not in base_features:
                        base_features.append(c)

        # --- Base-only mode for FeatureBank build ----------------------------
        if base_only:
            self._last_used_features = list(base_features)
            return df.copy(), list(base_features)

        # ---------- 6) Lags and rolling expansions ----------
        new_cols = {}
        missing_for_expansion = []

        if cfg.get("include_raw_lags", True) and "returns" in df:
            for lag in range(1, num_lags + 1):
                new_cols[f"returns_lag{lag}"] = df["returns"].shift(lag).astype("float32")

        for feat in base_features:
            if feat not in df.columns:
                missing_for_expansion.append(feat)
                continue
            src = df[feat]
            if src.dtype != np.float32:
                src = src.astype("float32")
            for k in range(1, lag_depth + 1):
                new_cols[f"{feat}_lag{k}"] = src.shift(k)
            for w in roll_windows:
                new_cols[f"{feat}_rollmean{w}"]  = src.rolling(w).mean()
                new_cols[f"{feat}_rollstd{w}"]   = src.rolling(w).std()
                new_cols[f"{feat}_rollslope{w}"] = self.rolling_slope(src, w).astype("float32")

        # DEBUG: print once, not per feature
        if debug and missing_for_expansion:
            print(
                f"[WARN] Skipping lag/rolls for {len(missing_for_expansion)} base_features missing in df "
                f"(showing up to 8): {missing_for_expansion[:8]}"
            )

        # ---------- 7) Hour ----------
        if cfg.get("include_hour", True):
            try:
                new_cols["hour"] = df.index.hour
            except AttributeError:
                pass

        if cfg.get("include_hour_cyclic", True):
            try:
                hour_vals = df.index.hour.to_numpy(dtype="float32", copy=False)
            except Exception as e:
                hour_vals = None
                _debug_once("hour_to_numpy", e)

            if hour_vals is not None and len(hour_vals) > 0:
                hour_rad = 2.0 * np.pi * hour_vals / 24.0
                new_cols["hour_sin"] = np.sin(hour_rad)
                new_cols["hour_cos"] = np.cos(hour_rad)

        if new_cols:
            new_cols_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_cols_df], axis=1)
            del new_cols_df
        new_col_names = list(new_cols.keys())
        del new_cols
        df_out = df

        # ---------- 8) Finalize feature list, fill, dropna ----------
        features: list[str] = [f for f in (new_col_names + base_features) if f in df_out.columns]

        if bool(cfg.get("use_regime_features", True)):
            for c in regime_cols:  # CLEANUP: single list
                if c in df_out.columns and c not in features:
                    features.append(c)

        # Forward-fill MTF if requested
        mtf_fillna = cfg.get("mtf_fillna_method", None)
        if mtf_fillna == "ffill":
            for mtf_col in ("mtf_ma_fast", "mtf_ma_slow"):
                if mtf_col in df_out:
                    df_out[mtf_col] = df_out[mtf_col].ffill()
                    for col in df_out.columns:
                        if col.startswith(mtf_col + "_"):
                            df_out[col] = df_out[col].ffill()

        # ---------- 8b) News & Sentiment features (optional) ----------
        use_news = bool(cfg.get("use_news", True))
        if use_news and not base_only:
            try:
                from news.features import merge_news_features, get_news_feature_columns
                news_agg = getattr(self, "_news_aggregated", None)
                econ_events = getattr(self, "_news_economic_events", None)
                if news_agg is None:
                    if not getattr(self, "_news_warned", False):
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "use_news=True but no news data injected into backtester — "
                            "skipping news features. Ensure api/tasks.py fetches news before running."
                        )
                        self._news_warned = True
                else:
                    df_out = merge_news_features(
                        df_out, news_agg,
                        events=econ_events,
                        config=cfg,
                    )
                    news_feat_cols = get_news_feature_columns(cfg)
                    event_cols = [c for c in df_out.columns if c.startswith("event_flag_")]
                    for c in news_feat_cols + event_cols:
                        if c in df_out.columns and c not in features:
                            features.append(c)
            except Exception as _news_exc:
                _debug_once("news_features", _news_exc)

        # ---------- 8c) LLM Sentiment features (optional) ----------
        use_llm = bool(cfg.get("llm_sentiment_enabled", True))
        if use_llm and not base_only and use_news:
            try:
                from news.features import merge_llm_features, get_llm_feature_columns
                llm_agg = getattr(self, "_llm_aggregated", None)
                if llm_agg is None:
                    if not getattr(self, "_llm_warned", False):
                        import logging as _logging
                        _logging.getLogger(__name__).info(
                            "llm_sentiment_enabled=True but no LLM data injected — skipping LLM features."
                        )
                        self._llm_warned = True
                else:
                    df_out = merge_llm_features(df_out, llm_agg, config=cfg)
                    llm_feat_cols = get_llm_feature_columns(cfg)
                    for c in llm_feat_cols:
                        if c in df_out.columns and c not in features:
                            features.append(c)
            except Exception as _llm_exc:
                _debug_once("llm_features", _llm_exc)

        dropna_subset = [f for f in features if f in df_out.columns]
        if dropna_subset:
            df_out.dropna(subset=dropna_subset, inplace=True)
        else:
            if debug:
                print("[WARN] No valid features for dropna_subset, running full dropna().")
            df_out.dropna(inplace=True)

        if len(df_out) == 0:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "prepare_features: DataFrame empty after dropna -- "
                "returning early with no features (possible data issue)"
            )
            self._last_used_features = []
            return df_out, []

        # Deduplicate columns if any
        if df_out.columns.duplicated().any():
            dup_cols = df_out.columns[df_out.columns.duplicated()].tolist()
            if debug:
                print(f"[WARN] Duplicate columns detected and removed: {dup_cols}")
            else:
                # CLEANUP: compact one-liner in non-debug
                print(f"[WARN] Duplicate columns detected and removed: n={len(dup_cols)}")
            df_out = df_out.loc[:, ~df_out.columns.duplicated()]

        # Keep last-used feature list for logging/diagnostics
        features = list(features)
        self._last_used_features = list(features)

        # --- Disk cache: always persist (independent of in-memory toggle) ---
        if not base_only and not in_cv:
            try:
                from pipeline.features.feature_cache import compute_disk_key, save_to_disk
                import hashlib as _hashlib, json as _json
                # Derive a stable disk key from the in-memory cache_key tuple
                _disk_key = _hashlib.sha256(
                    _json.dumps(cache_key, default=str).encode()
                ).hexdigest()[:16]
                save_to_disk(_disk_key, df_out, features)
            except Exception as _e:
                # Disk cache is purely an optimization -- never crash
                if debug:
                    print(f"[DISK_CACHE] save failed: {_e}")

        # Store engineered slice in per-run cache so later calls can reuse it.
        # We store the features as an immutable tuple to avoid accidental mutation.
        if cache_enabled:
            self._feat_cache[cache_key] = (df_out, tuple(features))

            # ---- Patch: hard cap / eviction (optional) ----
            if feat_cache_max_entries > 0:
                try:
                    evicted = 0
                    while isinstance(self._feat_cache, dict) and len(self._feat_cache) > feat_cache_max_entries:
                        oldest_key = next(iter(self._feat_cache))
                        try:
                            self._feat_cache.pop(oldest_key, None)
                        except Exception:
                            break

                        try:
                            if hasattr(self, "_feat_cache_bytes") and isinstance(self._feat_cache_bytes, dict):
                                b = int(self._feat_cache_bytes.pop(oldest_key, 0) or 0)
                                if hasattr(self, "_feat_cache_cur_bytes"):
                                    self._feat_cache_cur_bytes = max(
                                        0, int(getattr(self, "_feat_cache_cur_bytes", 0)) - b
                                    )
                        except Exception as e:
                            _debug_once("feat_cache_eviction_bytes", e)

                        evicted += 1

                    if evicted:
                        self._feat_cache_evictions = int(getattr(self, "_feat_cache_evictions", 0)) + int(evicted)
                        if LOG_MODE in {"COMPACT", "DEBUG"} and bool(int(os.getenv("KODAQUANT_VERBOSE", "0"))):
                            try:
                                cur_bytes = int(getattr(self, "_feat_cache_cur_bytes", 0))
                                print(
                                    f"[FEAT_CACHE] EVICT evicted={evicted} cap={feat_cache_max_entries} "
                                    f"entries_now={len(self._feat_cache)} cache_mb={cur_bytes/1024/1024:.1f}"
                                )
                            except Exception as e:
                                _debug_once("feat_cache_evict_diag", e)
                except Exception as e:
                    _debug_once("feat_cache_eviction_outer", e)

            # Diagnostics: cache miss/store
            self._feat_cache_misses = int(getattr(self, "_feat_cache_misses", 0)) + 1
            if LOG_MODE in {"COMPACT", "DEBUG"}:
                try:
                    n_entries = len(self._feat_cache) if isinstance(self._feat_cache, dict) else -1
                    _deep_mem = bool(LOG_MODE == "DEBUG" or getattr(self, "debug", False))
                    try:
                        est_bytes = int(df_out.memory_usage(deep=_deep_mem).sum())
                    except Exception:
                        est_bytes = int(df_out.memory_usage(deep=False).sum())

                    prev = 0
                    try:
                        prev = int(self._feat_cache_bytes.get(cache_key, 0))
                    except Exception:
                        prev = 0
                    try:
                        self._feat_cache_bytes[cache_key] = est_bytes
                    except Exception as e:
                        _debug_once("feat_cache_bytes_set", e)

                    cur_bytes = int(getattr(self, "_feat_cache_cur_bytes", 0))
                    cur_bytes = max(0, int(cur_bytes) + int(est_bytes) - int(prev))
                    self._feat_cache_cur_bytes = int(cur_bytes)

                    self._feat_cache_est_bytes = int(getattr(self, "_feat_cache_est_bytes", 0)) + est_bytes

                    hits = int(getattr(self, "_feat_cache_hits", 0))
                    misses = int(getattr(self, "_feat_cache_misses", 0))
                    denom = hits + misses
                    hit_rate = (hits / denom) if denom > 0 else 0.0
                    do_print = (
                        (LOG_MODE == "DEBUG")
                        or (self._feat_cache_misses in {1, 2, 5, 10, 20, 50, 100})
                        or (self._feat_cache_misses % 25 == 0)
                    )
                    if do_print and bool(int(os.getenv("KODAQUANT_VERBOSE", "0"))):
                        print(
                            f"[FEAT_CACHE] MISS entries={n_entries} "
                            f"hits={hits} misses={misses} hit_rate={hit_rate:.2%} "
                            f"+{est_bytes/1024/1024:.1f}MB cache_mb={cur_bytes/1024/1024:.1f} "
                            f"feats={len(features)}"
                        )
                except Exception as e:
                    _debug_once("feat_cache_miss_diag", e)

        # ── Phase -1 locked features filter ──
        locked = cfg.get("locked_features")
        if locked:
            locked_set = set(locked)
            features = [f for f in features if f in locked_set]

        return df_out, features

    
    def scale_features(self, df, features, means=None, stds=None, log_id=None):
        """
        Standardizes feature columns and optionally logs mean/std per fold.

        Parameters:
        - df (pd.DataFrame): Data with features.
        - features (list): Columns to scale.
        - means/stds (pd.Series): Optional for applying saved scaling.
        - log_id (str): Optional string identifier to log per-fold stats.

        Returns:
        - df_scaled, means, stds
        """
        if means is None or stds is None:
            means = df[features].mean()
            stds  = df[features].std()

        # Avoid divide-by-zero
        stds = stds.where(stds != 0, 1e-8)

        df = df.copy()
        feat_block = df[features].astype("float32", copy=False)
        df[features] = (feat_block - means) / stds

        # Replace infs that can still appear from pathological inputs
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)

        return df, means, stds

    def label_with_neutral(self, returns, threshold):
        """
        Creates classification labels for ML based on return thresholds.

        Parameters:
        - returns (np.array or pd.Series): Returns series.
        - threshold (float): Absolute return threshold to define class boundaries.

        Returns:
        - np.array: Array of integer labels:
            - 2 (buy/long) if returns > threshold,
            - 0 (sell/short) if returns < -threshold,
            - 1 (neutral/hold) otherwise.
        """

        labels = np.where(returns > threshold, 2, np.where(returns < -threshold, 0, 1))
        # compact stats only
        unique, counts = np.unique(labels, return_counts=True)
        if self._is_debug():
            print("Label counts:", dict(zip(unique, counts)), f"| thr={threshold}")
            
        return labels

