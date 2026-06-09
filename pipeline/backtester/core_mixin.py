"""Auto-extracted mixin -- see composed.py for the full MLBacktester."""
from pipeline._imports import *  # noqa: F401,F403


class CoreMixin:
    """
    __init__, config helpers, repr

    Auto-extracted from MLBacktesterNoWFO.py lines 1461-2196.
    """
    def __init__(
        self,
        symbol,
        start,
        end,
        trading_costs: bool = True,
        use_extended_features: bool = True,
        model_type: str = "svm",
        slippage_factor: float = 0.5,
        features_config: dict | None = None,
        use_oof: bool = False,
        data_store = None,
        db_path: str = "data/forex.db",
        base_timeframe: str = "M30",
    ):
        """
        Initialize the backtester for a specific instrument and date range.

        Parameters
        ----------
        symbol : str
            Financial instrument (e.g., 'EURUSD').
        start, end : str | pd.Timestamp
            Backtest window (inclusive).
        trading_costs : bool
            If True, incorporate trading costs in evaluation.
        use_extended_features : bool
            If True, use engineered technical features.
        model_type : str
            Model identifier (e.g., 'svm', 'cnn', 'lstm', 'xgboost', etc.).
        slippage_factor : float
            Slippage coefficient to model execution friction.
        features_config : dict | None
            Configuration for feature generation (indicator windows, toggles, etc.).
        use_oof : bool
            If True, enables Out-of-Fold stacking (for ensemble models).
        data_store : DataStore, optional
            Pre-existing DataStore instance. Takes priority over db_path.
        db_path : str
            Path to SQLite database. Used to create DataStore if data_store not provided.
        base_timeframe : str
            Primary trading timeframe (M15, M30, H1, H4). MTF timeframes derived from hierarchy.
        """
        self.symbol = symbol
        self.start = start
        self.end = end

        # --- Resolve pair-specific config (S3.2) ---
        from pipeline.pair_config import get_pair_config
        from pipeline.data_sqlite import DataStore, DataNotAvailableError
        try:
            self._pair_config = get_pair_config(symbol)
        except ValueError:
            from pipeline.pair_config import PairConfig
            self._pair_config = PairConfig(
                symbol=symbol,
                oanda_name=symbol[:3] + "_" + symbol[3:] if len(symbol) == 6 else symbol,
                pip_value=0.0001,
            )

        # --- DataStore: use provided instance or create one ---
        if data_store is not None:
            self._store = data_store
        else:
            self._store = DataStore(db_path)

        # --- Resolve base_timeframe from features_config if not explicitly provided ---
        if base_timeframe == "M30" and isinstance(features_config, dict):
            base_timeframe = str(features_config.get("base_timeframe", "M30"))
        self.base_timeframe = base_timeframe

        # --- Resolve MTF timeframes from hierarchy ---
        from config import TIMEFRAME_HIERARCHY, DEFAULT_BASE_TIMEFRAME
        tf_h = TIMEFRAME_HIERARCHY.get(self.base_timeframe)
        if tf_h is None:
            print(f"[WARN] Unknown base_timeframe '{self.base_timeframe}', falling back to {DEFAULT_BASE_TIMEFRAME}")
            self.base_timeframe = DEFAULT_BASE_TIMEFRAME
            tf_h = TIMEFRAME_HIERARCHY[self.base_timeframe]
        self._mtf_fast_tf = tf_h["mtf_fast"]
        self._mtf_slow_tf = tf_h["mtf_slow"]

        # --- Validate required timeframes exist ---
        _tfs = self._store.list_timeframes(self.symbol)
        _required = {self.base_timeframe, self._mtf_fast_tf, self._mtf_slow_tf}
        _missing = _required - set(_tfs)
        if _missing:
            raise DataNotAvailableError(
                f"Pair '{self.symbol}' is missing required timeframes: {sorted(_missing)}. "
                "Download data before running backtests."
            )

        # If trading_costs is explicitly provided at construction, it must not be overwritten
        # by any loaded/merged config later (GlobalHPO reuse, etc.).
        self._trading_costs_locked = (trading_costs is not None)
        self.trading_costs = True if trading_costs is None else bool(trading_costs)

        self.use_extended_features = use_extended_features
        self.model_type = model_type
        self.slippage_factor = float(slippage_factor) if slippage_factor is not None else 1.0
        self.use_oof = use_oof  # control OOF stacking
        self.model = None
        self.results = None
        
        # CV diagnostics: last evaluated fold frame and per-CV-fold frames
        # Used only during Optuna-style CV runs (_in_optuna_cv True).
        self._cv_last_eval_df = None
        self._cv_fold_eval_frames: list = []
        
        # Accumulator for WFO/WFS monthly records (used by PBO/MCS analysis)
        self._wfo_monthly_records: list[dict] = []
        
        # Showing first bars of the trading month
        self._dbg_first_bars = False     # opt-in only
        self._in_cv = False              # set True inside CV wrappers
        self._in_real_sim = False        # set True inside real_trading_sim()


        # [OK] Instance-private copy so in-class mutations never leak outward
        self.features_config = deepcopy(features_config) if features_config else {}

        # --- Inject pair-specific pip_value (S3.2) ---
        if isinstance(self.features_config, dict):
            self.features_config["stop_pip_value"] = self._pair_config.pip_value
            self.features_config["trailing_pip_value"] = self._pair_config.pip_value
        
        # --- Resolve slippage_factor (explicit config > ctor arg). ---
        # Prevent silent 0.0 when trading_costs are enabled.
        try:
            if isinstance(self.features_config, dict) and ("slippage_factor" in self.features_config):
                self.slippage_factor = float(self.features_config.get("slippage_factor"))
            elif bool(self.trading_costs) and float(getattr(self, "slippage_factor", 0.0) or 0.0) == 0.0:
                # Legacy default was 0.0; treat as 'unset' unless explicitly provided in config.
                self.slippage_factor = 1.0
                if self._is_debug():
                    print("[Costs] slippage_factor missing; defaulting to 1.0 (set features_config['slippage_factor'] to override).")
        except Exception:
            # never fail init due to a bad config knob
            pass
        
        # Feature-slice cache is *off by default*.
        # Rationale: prepare_features() is usually invoked on unique slices
        # (train/test/month/fold), so caching retains large frames with ~0 reuse.
        if isinstance(self.features_config, dict):
            self.features_config.setdefault("feat_cache_enabled", False)
            
            # NOTE: "slice_cache_enabled" is the canonical flag (default OFF).
            # Back-compat: honor older configs that used "feat_cache_enabled".
            if "slice_cache_enabled" not in self.features_config and "feat_cache_enabled" in self.features_config:
                self.features_config["slice_cache_enabled"] = bool(self.features_config.get("feat_cache_enabled", False))
            self.features_config.setdefault("slice_cache_enabled", False)

        # --- Feature cache / FeatureBank (per-run, per-symbol) ---
        self._feat_cache: dict = {}
        self._feat_cache_hits = 0
        self._feat_cache_misses = 0
        self._feat_cache_est_bytes = 0
        self._feat_cache_mode_logged = False  # log cache mode once per run
        
        # Log slice-cache mode explicitly (once per phase) to avoid ambiguity.
        self._feat_cache_logged_cv = False
        self._feat_cache_logged_noncv = False
        
        self._feature_bank_full = None      # type: Optional[pd.DataFrame]
        self._feature_bank_meta = {}        # small dict with base feature names, etc.
        self._feature_bank_key  = None      # signature of data + config used to build the bank
        self._feature_bank_src  = None      # optional stable source df for bank builds (see set_feature_bank_source)


        # Will be populated in get_data()
        self.data = None
        self.df_1h = None
        self.df_4h = None


        # Load all required timeframes & compute base returns
        self.get_data()

    # --- Logging mode helper ---
    def _is_debug(self):
        # Respect module-level LOG_MODE default ("COMPACT") unless explicitly overridden.
        # Also allow an instance-level debug flag.
        try:
            if bool(getattr(self, "debug", False)):
                return True
        except Exception:
            pass
        return os.environ.get("LOG_MODE", LOG_MODE).upper() == "DEBUG"
    
    def _sanitize_runtime_coverage_nudge(self, band, step, *, ctx: str = ""):
        """Clamp and stabilize runtime active-rate 'coverage nudge' params.

        Ensures the nudge step is not larger than half the band (prevents flip-flop),
        applies a small minimum band when enabled, and stores the actually-used values
        for truthful fold/month logging.
        """
        try:
            band_f = float(band)
        except Exception:
            band_f = 0.0
        try:
            step_f = float(step)
        except Exception:
            step_f = 0.0

        if not np.isfinite(band_f):
            band_f = 0.0
        if not np.isfinite(step_f):
            step_f = 0.0

        band_old, step_old = band_f, step_f

        # Clamp to sane ranges (band==0 disables nudge)
        if band_f > 0.0:
            band_f = max(0.01, min(band_f, 0.25))
        else:
            band_f = 0.0

        if step_f > 0.0:
            step_f = min(step_f, 0.25)
        else:
            step_f = 0.0

        adjusted = False

        # Stability: step must not exceed half the band.
        if band_f > 0.0 and step_f > 0.0 and step_f > 0.5 * band_f:
            step_f = 0.5 * band_f
            adjusted = True

        # Persist actually-used values for fold/month summaries.
        try:
            self._last_runtime_active_band_used = float(band_f)
            self._last_runtime_conf_step_used = float(step_f)
        except Exception:
            pass

        if adjusted and self._is_debug():
            tag = f" ({ctx})" if ctx else ""
            print(
                f"[Gate[OK]] Coverage nudge params adjusted{tag}: "
                f"band {band_old:.4f}->{band_f:.4f}, step {step_old:.4f}->{step_f:.4f}"
            )

        return band_f, step_f

    
    def _safe_float(self, v, fallback_key: str | None = None) -> float:
        try:
            x = float(v)
            if np.isfinite(x):
                return x
        except Exception:
            pass
        if fallback_key:
            try:
                _attrs = getattr(getattr(self, "results", None), "attrs", {}) or {}
                x = float(_attrs.get(fallback_key, float("nan")))
                return x
            except Exception:
                return float("nan")
        return float("nan")
    
    def _tf_cleanup(self):
        try:
            import tensorflow as tf
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
        except Exception:
            pass
        try:
            _gc.collect()
        except Exception:
            pass


    def _safe_int(self, v, fallback_key: str | None = None) -> int:
        try:
            x = int(v)
            if x != 0:
                return x
        except Exception:
            pass
        if fallback_key:
            try:
                _attrs = getattr(getattr(self, "results", None), "attrs", {}) or {}
                return int(_attrs.get(fallback_key, 0) or 0)
            except Exception:
                return 0
        return 0

    
    # --- Calibration metrics helper (used by deep models to feed Patch #2 selection penalty) ---
    def _set_last_calib_metrics(self, proba_cal, y_cal, ctx: str = ""):
        """
        Compute and store calibration metrics (Brier, NLL, n) on a calibration slice.
        Exception-safe: if something fails, metrics become NaN and run continues.
        """
        try:
            from utilsNoWFO import compute_brier_and_nll
        except Exception:
            compute_brier_and_nll = None

        try:
            n = int(len(y_cal)) if y_cal is not None else 0
        except Exception:
            n = 0

        brier = float("nan")
        nll = float("nan")
        try:
            if compute_brier_and_nll is not None and n > 0:
                brier, nll = compute_brier_and_nll(proba_cal, y_cal)
        except Exception:
            brier, nll = float("nan"), float("nan")

        # Store for CV aggregation (tuningNoWFO Patch #2)
        try:
            setattr(self, "_last_calib_brier", float(brier))
            setattr(self, "_last_calib_nll", float(nll))
            setattr(self, "_last_calib_n", int(n))
        except Exception:
            pass

        # Optional audit log
        try:
            cfg = getattr(self, "features_config", {}) or {}
            if bool(cfg.get("print_cv_debug", False)) or bool(os.environ.get("HPO_SELECT_DEBUG", "0") == "1"):
                print(f"[Calib][Metrics] brier={float(brier):.6f} nll={float(nll):.6f} n={int(n)} ctx={ctx}")
        except Exception:
            pass
    
    def set_feature_bank_source(self, df):
        """Set a stable source DataFrame for FeatureBank base indicators.

        Why this exists:
        - In real_trading_simulation() the engine repeatedly overwrites `self.data`
        with month-sized slices.
        - Base indicators (ATR/ADX/RSI/etc.) are *causal per timestamp* and can be
        computed once over a larger, stable span and then reindexed to slices.

        This is a performance patch only: it does not change feature definitions.
        """
        self._feature_bank_src = df

        # Force rebuild next time _ensure_feature_bank() runs (span changed).
        self._feature_bank_full = None
        self._feature_bank_meta = {}
        self._feature_bank_key = None
    


    def _guard_label_mix_directional(
        self,
        y_train,
        label_threshold: float,
        context: str = "FOLD",
        min_dir_samples: int = 5,
    ) -> bool:
        """
        Sanity check on 3-class labels with convention:
          0 = SHORT, 1 = NEUTRAL, 2 = LONG.

        We only enforce minimum counts on *directional* classes (0, 2),
        and allow the neutral class to be arbitrarily large or small.

        Returns
        -------
        bool
            True  -> label mix is acceptable for training.
            False -> fold should be skipped as structurally degenerate.
        """
        # Empty labels -> nothing to train on
        if y_train is None or len(y_train) == 0:
            print(f"[WARN] [{context}] Skipping fold: empty label vector.")
            return False
        y_arr = np.asarray(y_train)
        u_tr, c_tr = np.unique(y_arr, return_counts=True)
        label_counts = dict(zip(u_tr, c_tr))

        if self._is_debug():
            print(f"[{context}] Label counts (train): {label_counts} | thr={label_threshold}")

        NEUTRAL_CLASS = 1
        dir_mask = (u_tr != NEUTRAL_CLASS)
        u_dir = u_tr[dir_mask]
        c_dir = c_tr[dir_mask]

        # No directional labels at all -> useless for trading
        if len(u_dir) == 0:
            print(f"[WARN] [{context}] Skipping fold: no directional labels in train {label_counts}")
            return False

        # Both SHORT and LONG present -> each must have at least min_dir_samples
        if len(u_dir) >= 2 and (c_dir.min() if len(c_dir) else 0) < min_dir_samples:
            print(f"[WARN] [{context}] Skipping fold: poor directional label mix in train {label_counts}")
            return False

        # Only one directional class present (e.g. only LONG) -> require enough events
        if len(u_dir) == 1 and c_dir[0] < min_dir_samples:
            print(f"[WARN] [{context}] Skipping fold: too few directional events in train {label_counts}")
            return False

        return True

    def _resolve_conf_thr(self, default_conf: float) -> float:
        """
        Decide the effective confidence threshold for this run.

        This is called for *every* trial, CV fold, WFO month, and
        real-trading simulation. Any tweaks here are baked in from the
        start - no feedback from test results.
        """
        cfg_f = getattr(self, "features_config", {}) or {}

        # 1) Base threshold: coverage-calibrated or manual
        cov_thr = getattr(self, "_coverage_conf_thr", None)
        thr = freeze_confidence_threshold(cfg_f, default_conf, cov_thr)
        
        
        # If coverage intent exists but we couldn't compute a calibrated threshold,
        # treat this as a *rare* fallback outside CV (CV should penalize).
        try:
            import numpy as _np
            _cov_intent = bool(is_coverage_intent(cfg_f))
            _thr_ok = _np.isfinite(float(thr))
        except Exception:
            _cov_intent, _thr_ok = False, True

        # Ensure diagnostics vars always exist (prevents UnboundLocalError in CV tripwire)
        cal_rows = int(getattr(self, "_last_cov_cal_rows", 0) or 0)

        if _cov_intent and (not _thr_ok):
            in_cv = bool(getattr(self, "_in_cv", False) or getattr(self, "_in_optuna_cv", False))
            if in_cv:
                # CV tripwire: never proceed with NaN thresholds (mask becomes a no-op).
                # Use a hard 0-trade gate so the fold is penalized deterministically.
                try:
                    max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))
                except Exception:
                    max_conf_thr = 0.90
                print(
                    f"[Calib][Coverage][TRIPWIRE][CV] conf_thr=nan cal_rows={cal_rows} "
                    f"reason=missing_coverage_thr -> forcing_conf_thr={max_conf_thr:.4f}"
                )
                thr = float(max_conf_thr)
            else:

                # deterministic fallback (explicit + auditable)
                fb = float(cfg_f.get("confidence_threshold", default_conf))
                ctx = "eval"
                try:
                    if bool(getattr(self, "_in_real_sim", False)):
                        mx = int(cfg_f.get("month_ix", getattr(self, "_rt_month_ix", 0) or 0))
                        ctx = f"real_m{mx}"
                except Exception:
                    pass
                try:
                    tar = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.0)) or 0.0)
                except Exception:
                    tar = 0.0
                cal_rows = int(getattr(self, "_last_cov_cal_rows", 0) or 0)
                print(
                    f"[Calib][Coverage][FALLBACK] conf_thr={fb:.6f} "
                    f"target_active_rate={tar:.6f} cal_rows={cal_rows} ctx={ctx} "
                    f"reason=missing_coverage_thr"
                )
                thr = fb

        # Book-keeping
        self._last_conf_thr_init = float(
            cfg_f.get("confidence_threshold", default_conf)
        )
        self._last_conf_thr_used = float(thr)

        return float(thr)

    def _emit_conf_gate_snapshot(
        self,
        *,
        model_type: str,
        eval_context: Optional[str],
        conf_requested: float,
        base_thr: float,
        thr_vec: "np.ndarray",
        eval_idx: Optional["np.ndarray"],
        dyn_abg: bool = True,
    ) -> None:
        
        cfg_f = getattr(self, "features_config", {}) or {}
        cov_intent = bool(is_coverage_intent(cfg_f))
        cov_thr = getattr(self, "_coverage_conf_thr", None)

        if cov_intent and cov_thr is not None:
            try:
                cov_thr_f = float(cov_thr)
            except Exception:
                cov_thr_f = float("nan")
        else:
            cov_thr_f = float("nan")

        if cov_intent:
            source = "coverage_calibrated" if np.isfinite(cov_thr_f) else "coverage_intent_missing"
        else:
            source = "static"

        try:
            if eval_idx is not None and hasattr(eval_idx, "size") and eval_idx.size > 0:
                used_m = float(np.nanmedian(thr_vec[eval_idx]))
            else:
                used_m = float(np.nanmedian(thr_vec))
        except Exception:
            used_m = float("nan")

        try:
            used_m = float(getattr(self, "_last_conf_thr_used", used_m))
        except Exception:
            pass

        cov_s = f"{cov_thr_f:.3f}" if np.isfinite(cov_thr_f) else "NA"
        ctx_s = str(eval_context or "")
        dyn_s = "on" if dyn_abg else "off"

        print(
            f"[LOCK] [ConfGate] model={model_type} ctx={ctx_s} "
            f"conf_requested={float(conf_requested):.3f} conf_base={float(base_thr):.3f} "
            f"conf_used_median={used_m:.3f} source={source} cov_thr={cov_s} dyn_abg={dyn_s}"
        )

    
    # === NEW: expose config merge as instance method (used by tuner & eval) ===
    def _merge_params_into_features_config(self, bp: dict, force_lags: int | None = None) -> dict:
        """
        Merge order:
          existing run config  << nested model sub-configs  << flat tuned keys.
        Then apply defaults ONLY to fill missing keys (defaults must not overwrite tuned keys).
        Returns the merged features_config dict.
        """
        try:
            bp = dict(bp or {})
            
            # --- Materialize derived keys for faithful replay ---------------------------------
            # Optuna trials often store selector keys (e.g., roll_windows_key_v2) and/or
            # per-indicator "*_window_core" primitives. During CV these are materialized
            # into concrete structures (roll_windows list, indicator_windows dict). If the
            # monthly real-trading loop replays a params dict missing these materialized
            # keys, apply_feature_defaults() can silently fall back to JSON defaults (e.g.,
            # roll_windows=[5,10,20]), producing different confidence distributions and
            # no-trade months. This block only fills *missing* derived fields.

            # 0) roll_windows: derive from roll_windows_key_v2/key if needed
            try:
                rk = bp.get("roll_windows_key_v2") or bp.get("roll_windows_key")
                if "roll_windows" not in bp and rk is not None:
                    bp["roll_windows"] = [int(x) for x in str(rk).split(",") if str(x).strip() != ""]
                    print(f"[HPOReplay] materialized roll_windows={bp.get('roll_windows')} from rk={rk}")
                # Drop selector aliases to avoid confusing downstream logs/snapshots
                bp.pop("roll_windows_key_v2", None)
                bp.pop("roll_windows_key", None)
            except Exception:
                pass

            # 1) indicator_windows: build from core window primitives if missing
            try:
                if "indicator_windows" not in bp or not isinstance(bp.get("indicator_windows"), dict):
                    iw = {}
                    if "sma_window_core" in bp: iw["sma"] = int(bp["sma_window_core"])
                    if "ema_window_core" in bp: iw["ema"] = int(bp["ema_window_core"])
                    if "rsi_window_core" in bp: iw["rsi"] = int(bp["rsi_window_core"])
                    if "atr_window_core" in bp: iw["atr"] = int(bp["atr_window_core"])
                    if "adx_window_core" in bp: iw["adx"] = int(bp["adx_window_core"])
                    if "bb_window_core" in bp:  iw["bb_window"] = int(bp["bb_window_core"])
                    if "bb_dev_core" in bp:     iw["bb_dev"] = float(bp["bb_dev_core"])

                    mv = bp.get("macd_core_variant", None)
                    if mv is not None and "macd_fast" not in iw:
                        try:
                            a, b, c = list(mv)
                            iw["macd_fast"] = int(a)
                            iw["macd_slow"] = int(b)
                            iw["macd_signal"] = int(c)
                        except Exception:
                            pass

                    if "mtf_ma_fast_window" in bp: iw["mtf_ma_fast_window"] = int(bp["mtf_ma_fast_window"])
                    if "mtf_ma_slow_window" in bp: iw["mtf_ma_slow_window"] = int(bp["mtf_ma_slow_window"])

                    if iw:
                        bp["indicator_windows"] = iw
            except Exception:
                pass


            # Drop CV-policy keys (never belong to features_config)
            for k in list(bp.keys()):
                if str(k).startswith("cv_"):
                    bp.pop(k, None)

            # Start from current features_config
            base = dict(self.features_config) if isinstance(self.features_config, dict) else {}

            # 1) Pull in namespaced sub-configs (cnn_config, lstm_config, transformer_config, xgb_config, dqn_config, rf_config, logit_config)
            for nested in ("cnn_config", "lstm_config", "transformer_config",
                           "xgb_config", "dqn_config", "rf_config", "logit_config"):
                val = bp.get(nested)
                if isinstance(val, dict):
                    base.update(val)

            # 2) Flat tuned keys win
            base.update(bp)

            # 3) Enforce specific keys if provided
            if force_lags is not None:
                base["lags"] = int(force_lags)
            if "use_fracdiff" in bp:
                base["use_fracdiff"] = bool(bp["use_fracdiff"])
            if "calibrate_method" in bp:
                base["calibrate_method"] = str(bp["calibrate_method"]).lower()

            # Materialize and let defaults fill only missing stuff
            self.features_config = base
            if hasattr(self, "apply_feature_defaults"):
                self.apply_feature_defaults()

            # Return a copy for external consumers (tuner)
            return dict(self.features_config)
        
        except Exception as e:
            print(f"[WARN] Could not merge best_params into features_config: {e}")
            return dict(self.features_config) if isinstance(self.features_config, dict) else {}

    def _short_param_string(self, p: dict) -> str:
        def _getint(k, default=0):
            try: return int(p.get(k, default))
            except (TypeError, ValueError): return default

        lags = _getint("lags", _getint("lags_range", 0))
        d    = _getint("lag_depth", 0)

        roll_a = str(p.get("roll_windows_key", "")).strip()
        roll_b = str(p.get("roll_windows_key_v2", "")).strip()
        # render like 5|10,20 if both present, else whichever exists
        roll = (roll_a + ("|" if (roll_a and roll_b) else "") + roll_b) if (roll_a or roll_b) else "-"

        strat = str(p.get("strategy_type", "-"))
        # include only the common, short, human-scan keys
        keys_by_strat = {
            "volatility": ["atr_window"],
            "confirmation": ["adx_window", "mtf_ma_fast_window", "mtf_ma_slow_window"],
            "contrarian": ["rsi_window","bb_window","bb_dev","stoch_k_window","stoch_d_window"],
            "momentum": ["ema_window","rsi_window"],
        }
        want = keys_by_strat.get(strat, [])
        bits = []
        for k in want:
            if k in p:
                label = k.replace("_window","").replace("mtf_ma_","ma_")
                bits.append(f"{label}={p[k]}")
        strat_str = f"{strat}({','.join(bits)})" if bits else strat

        tb_on = bool(p.get("use_triple_barrier", False))
        tb_bits = []
        if "tb_pt_mult" in p: tb_bits.append(f"pt={p['tb_pt_mult']}")
        if "tb_sl_mult" in p: tb_bits.append(f"sl={p['tb_sl_mult']}")
        if "tb_max_holding" in p: tb_bits.append(f"hold={p['tb_max_holding']}")
        tb_str = f"TB={'on' if tb_on else 'off'}" + ((" " + " ".join(tb_bits)) if tb_bits else "")

        return f"lags={lags} d={d} roll={roll}  strat={strat_str} {tb_str}"

    def _should_dump_decisions(self) -> bool:
        return (
            getattr(self, "_dbg_first_bars", False) and
            not getattr(self, "_in_cv", False) and
            getattr(self, "_in_real_sim", False)
        )

    def apply_feature_defaults(self, params: dict | None = None) -> dict:
        """Merge user/trial params over DEFAULT_FEATURES safely."""
        base = deepcopy(DEFAULT_FEATURES)
        if isinstance(getattr(self, "features_config", None), dict):
            base.update(self.features_config)   # class-level or previous
        if isinstance(params, dict):
            base.update(params)                 # trial-level wins
        self.features_config = base
        
        
        # ---------------------------------------------------------------
        # B2: Calibration safety clamps (stability; avoids degenerate cal windows)
        # ---------------------------------------------------------------
        try:
            def _clipf(x, lo, hi, default):
                try:
                    v = float(x)
                except Exception:
                    v = float(default)
                return float(max(lo, min(hi, v)))

            def _clipi(x, lo, hi, default):
                try:
                    v = int(x)
                except Exception:
                    v = int(default)
                return int(max(lo, min(hi, v)))

            # keep calibration fraction in a conservative band
            base["deep_calibration_frac"] = _clipf(base.get("deep_calibration_frac", 0.10), 0.08, 0.20, 0.10)
            base["classical_calibration_frac"] = _clipf(
                base.get("classical_calibration_frac", base.get("deep_calibration_frac", 0.10)), 0.08, 0.20, 0.10
            )
            # keep calibration min samples reasonable
            base["deep_calibration_min_samples"] = _clipi(base.get("deep_calibration_min_samples", 500), 500, 5000, 500)
            base["classical_calibration_min_samples"] = _clipi(
                base.get("classical_calibration_min_samples", base.get("deep_calibration_min_samples", 500)), 500, 5000, 500
            )
        except Exception:
            pass

        
        # ---------------------------------------------------------------
        # HARD DISABLE: CV thin-trades fallback (must never "invent" trades)
        # CV must match real_trading_simulation behavior.
        # ---------------------------------------------------------------
        try:
            in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
        except Exception:
            in_cv = False
        if in_cv and bool(base.get("allow_thin_trades_fallback", False)):
            base["allow_thin_trades_fallback"] = False
        # Warn once per instance to avoid spam
            if not bool(getattr(self, "_warned_thin_trades_disabled", False)):
                print("[DISABLED] [CV] allow_thin_trades_fallback is HARD-DISABLED (CV must not invent trades).")
                self._warned_thin_trades_disabled = True
                
        # --- B1 Policy: enforce GLOBAL target coverage (signal intent) for ALL models ---
        try:
            _mt = base.get("model_type", getattr(self, "model_type", None))
            enforce_target_coverage_policy(base, model_type=_mt)
            
            # Hard-assert (non-fatal) + one-time log after ALL merges.
            # This prevents silent drift from later overrides and gives a single
            # authoritative line you can trust in logs.
            try:
                tar = float(base.get("target_active_rate", base.get("target_coverage", 0.0)) or 0.0)
                exp = float(target_coverage_policy(_mt) or 0.0)
                if exp > 0.0 and abs(tar - exp) > 1e-9:
                    # Re-enforce (should be redundant); keep non-fatal to avoid breaking long runs.
                    print(f"[WARN] [CoveragePolicy][ASSERT] target_active_rate drifted to {tar:.6f}; re-enforcing policy={exp:.6f}")
                    enforce_target_coverage_policy(base, model_type=_mt)
                    tar = float(base.get("target_active_rate", base.get("target_coverage", 0.0)) or 0.0)

                if not bool(getattr(self, "_printed_coverage_policy", False)):
                    tc = float(base.get("target_coverage", tar) or tar)
                    gm = base.get("gating_mode", base.get("gate_mode", None))
                    in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                    in_real = bool(getattr(self, "_in_real_sim", False))
                    where = "CV" if in_cv else ("REAL" if in_real else "RUN")
                    print(
                        f"[TARGET] [CoveragePolicy][{where}] model_type={_mt} gating_mode={gm} "
                        f"target_active_rate={tar:.3f} target_coverage={tc:.3f} (policy-locked)"
                    )
                    self._printed_coverage_policy = True
            except Exception:
                pass

            
            self.features_config = base
        except Exception:
            pass
        return base

    def apply_cv_defaults(self, cv_cfg: dict | None = None) -> dict:
        """Merge user/runner CV cfg over DEFAULT_CV safely."""
        base = deepcopy(DEFAULT_CV)
        if isinstance(cv_cfg, dict):
            base.update(cv_cfg)
        return base


    @classmethod
    def set_global_defaults(cls, section: str, updates: dict):
        """Optional: tweak defaults globally at runtime, e.g., set_global_defaults('cv', {'cv_blocks': 5})."""
        if section in cls.CLASS_DEFAULTS and isinstance(updates, dict):
            cls.CLASS_DEFAULTS[section].update(updates)

        
    def __repr__(self) -> str:
        """Readable summary of key configuration."""
        return (
            f"MLBacktester(symbol={self.symbol}, start={self.start}, end={self.end}, "
            f"trading_costs={self.trading_costs}, use_extended_features={self.use_extended_features}, "
            f"model_type={self.model_type})"
        )


