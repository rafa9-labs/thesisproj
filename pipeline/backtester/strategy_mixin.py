"""Auto-extracted mixin -- see composed.py for the full MLBacktester."""
from config import PIPELINE_CONSTANTS as _PC
from pipeline._imports import *  # noqa: F401,F403


class StrategyMixin:
    """
    test_strategy

    Auto-extracted from MLBacktesterNoWFO.py lines 4226-7534.
    """
    def test_strategy(
        self,
        train_start,
        train_end,
        test_start,
        test_end,
        lags,
        confidence_threshold: float = 0.6,
        label_threshold: float = 0.0001,
        persist_results: bool = True,
        eval_context: str | None = None,
    ):
        """
        Train on [train_start, train_end], test on [test_start, test_end], and evaluate trading metrics.

        Returns
        -------
        tuple[float, ...]   # 16 metrics in fixed order
        """

        # CLEANUP: centralized debug/log helpers to prevent print storms (no logic changes)
        _DBG = bool(getattr(self, "debug", False)) or bool(getattr(self, "_debug", False))
        try:
            _DBG = bool(_DBG or (hasattr(self, "_is_debug") and self._is_debug()))
        except Exception:
            pass

        def _dprint(_msg: str):
            # DEBUG: only prints when debug is enabled
            if _DBG:
                print(_msg)

        def _print_once(_key: str, _msg: str, *, debug_only: bool = False):
            # DEBUG: print a message at most once per backtest instance
            if debug_only and not _DBG:
                return
            _flag = f"_ts_once_{_key}"
            if not getattr(self, _flag, False):
                print(_msg)
                setattr(self, _flag, True)

        def _dbg_exc(_label: str, _e: Exception):
            # DEBUG: never swallow exceptions silently
            if _DBG:
                print(f"[test_strategy][{_label}] {type(_e).__name__}: {_e}")
        with self._persist_results_guard(persist_results=persist_results):
        
            # Clear sticky feature-slice cache.
            # In practice cache keys are almost always unique (month-by-month + per-config),
            # so keeping these large frames across calls yields ~0 hits and rising RAM.
            self._clear_feature_cache()
         
            # [SHIELD] Set TF runtime knobs *before* importing tensorflow
            # Mirror global intra-trial knob
            _threads = int(os.getenv("BLAS_THREADS_PER_TRIAL", os.getenv("MLB_THREADS", str(max(1, (os.cpu_count() or 8) - 2)))))
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_threads))
            os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(max(2, min(4, _threads // 4))))
            for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS"):
                os.environ.setdefault(k, str(_threads))

            try:
                for _gpu in tf.config.list_physical_devices("GPU"):
                    try:
                        tf.config.experimental.set_memory_growth(_gpu, True)
                    except Exception:
                        pass
                tf.config.set_soft_device_placement(True)
            except Exception:
                pass


            # --- FIX: define cfg_f up-front so all branches can read it safely ---
            cfg_f = getattr(self, "features_config", {}) or {}
            # Default to coverage-anchored gating unless explicitly overridden
            if "gating_mode" not in cfg_f:
                cfg_f["gating_mode"] = "coverage"
                self.features_config = cfg_f
                
            # Prevent stale coverage thresholds leaking across runs/models.
            # If coverage intent is enabled but calibration fails, freeze_confidence_threshold()
            # should tripwire (NaN) rather than silently reusing an old threshold.
            try:
                if is_coverage_intent(cfg_f):
                    self._coverage_conf_thr = None
                    if hasattr(self, "_deep_coverage_thr"):
                        delattr(self, "_deep_coverage_thr")
            except Exception:
                pass
        
            
            in_cv = bool(getattr(self, "_in_optuna_cv", False))
            
            # CV memory hygiene: default to no TF cleanup unless a deep model is actually used.
            # Flags are consumed by _persist_results_guard() on exit.
            try:
                self._tf_cleanup_do = False
                self._tf_cleanup_del_model = False
            except Exception:
                pass

        

            # --- Costs knobs (respect constructor lock) ---
            try:
                if not getattr(self, "_trading_costs_locked", False):
                    if "eval_use_trading_costs" in cfg_f:
                        self.trading_costs = bool(cfg_f.get("eval_use_trading_costs", self.trading_costs))
                    elif "trading_costs" in cfg_f:
                        self.trading_costs = bool(cfg_f.get("trading_costs", self.trading_costs))
            except Exception:
                pass

            try:
                if "slippage_factor" in cfg_f:
                    self.slippage_factor = float(cfg_f.get("slippage_factor", self.slippage_factor))
            except Exception:
                pass
            if self._is_debug() and bool(getattr(self, "trading_costs", True)):
                try:
                    _sf = float(getattr(self, "slippage_factor", 0.0) or 0.0)
                    if _sf == 0.0:
                        _print_once("costs_slip_disabled", "[Costs][Warn] trading_costs=True but slippage_factor=0.0. Slippage disabled.")  # CLEANUP
                except Exception:
                    pass
        
            # --- Real-trading guard: if a target_active_rate is set, ensure coverage mode
            # so the existing train-anchored coverage threshold fitting can run.
            # This prevents the system from staying stuck at confidence_threshold=0.8
            # and then getting bumped higher by alphabetagamma, which can easily yield 0 trades.
            try:
                in_real = bool(getattr(self, "_in_real_sim", False))
                gmode = str(cfg_f.get("gating_mode", cfg_f.get("gate_mode", "threshold"))).lower()
                tgt = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.0)) or 0.0)
                if in_real and (not in_cv) and gmode in ("threshold", "", "none") and tgt > 0.0:
                    cfg_f["gating_mode"] = "coverage"
                    self.features_config = cfg_f
                    if self._is_debug():
                        _print_once("gate_auto_cov_real", f"[Gate] Auto-enabled gating_mode='coverage' for real-sim (target_active_rate={tgt:.2f}).")  # CLEANUP
            except Exception:
                pass
        
            self.model = None
            self._diagnostics_feature_importance = []

            # Clear any sticky feature cache from previous evals
            self._clear_feature_cache()

            # Build / refresh FeatureBank for current data + feature config
            try:
                self._ensure_feature_bank()
            except Exception as e:
                if self._is_debug():
                    _dbg_exc("_ensure_feature_bank", e)  # CLEANUP

            # ---- RAM USAGE: Print at the very start ----
            mem_gb_start = psutil.virtual_memory().used / (1024**3)

            # if self._is_debug():
            #     print(f"[RAM] Start of test_strategy: {mem_gb_start:.2f} GB used")
            _dprint(f"[RAM] Start of test_strategy: {mem_gb_start:.2f} GB used")  # CLEANUP
            # ----------------------------
            # 1) Train/Test slicing (+ NY session on test)
            # ----------------------------
            full_data  = self.data
            train_data = full_data.loc[train_start:train_end]


            # --- warm-up aware test selection (pre-roll before test_start) ---
            true_test_start = pd.to_datetime(test_start)
            test_end        = pd.to_datetime(test_end)
            model_label     = str(self.features_config.get("model_type", self.model_type))
            warmup_need     = int(compute_required_test_warmup_bars({**self.features_config, "model_type": model_label}))
        
            # account for final embargo so pre-roll remains outside test month
            embargo_n = int(self.features_config.get("final_embargo_bars", 0) or 0)
            _total_warmup_need = max(0, warmup_need + embargo_n)

            def _slice_with_warmup(n_extra: int):
                if n_extra <= 0:
                    return full_data.loc[true_test_start:test_end]
                idx_before = full_data.index[full_data.index < true_test_start]
                if len(idx_before) == 0:
                    return full_data.loc[true_test_start:test_end]
                start_pos = max(0, len(idx_before) - n_extra)
                warmup_start = idx_before[start_pos]
                return full_data.loc[warmup_start:test_end]


            # initial pre-roll (build test_data before any filtering/embargo)
            test_data = _slice_with_warmup(_total_warmup_need)

            sess_mode = str(self.features_config.get("session_filter_mode", "both")).lower()

            if not hasattr(self, "_ny_mask") or self._ny_mask is None:
                try:
                    full_idx = pd.to_datetime(self.data.index, utc=True, errors="coerce")
                    _ny_times = full_idx.tz_convert("America/New_York")
                    self._ny_mask = pd.Series((_ny_times.hour >= 2) & (_ny_times.hour <= 13), index=full_idx)
                except Exception as _e:
                    _dbg_exc("ny_mask", _e)  # CLEANUP
                    self._ny_mask = pd.Series(True, index=self.data.index)

            # NEW semantics:
            # - "both":        filter train + test
            # - "test_only":   filter test only
            # - "train_only":  filter train only
            if sess_mode in ("test_only", "both"):
                test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]
            if sess_mode in ("train_only", "both"):
                train_data = train_data.loc[self._ny_mask.reindex(train_data.index, fill_value=False)]

        
            # ensure we still have enough warm-up after session filter
            if warmup_need > 0 and len(test_data) > 0:
                have = int((test_data.index < true_test_start).sum())
                if have < _total_warmup_need:
                    # fetch more history and reapply the session filter
                    need_more = _total_warmup_need - have
                    test_data = _slice_with_warmup(_total_warmup_need + need_more)
                    if sess_mode in ("test_only", "both"):
                        test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]
                    
            # Final embargo to avoid bleed -- but NEVER eat CV mini-fold heads
            try:
                embargo_n = int(self.features_config.get("final_embargo_bars", 0))
                if bool(getattr(self, "_in_optuna_cv", False)):
                    embargo_n = 0  # [LEFT] disable head-drop during CV mini-folds
                if embargo_n > 0 and len(test_data) > embargo_n:
                    test_data = test_data.iloc[embargo_n:].copy()
                    if self._is_debug():
                        print(f"[Embargo] Dropped first {embargo_n} bars from TEST (non-CV only).")

            except Exception as e:
                _dbg_exc("final_embargo_bars", e)  # CLEANUP

            # Evaluation anchor:
            # In real trading we start *after* embargo; in CV we start EXACTLY at the fold start.
            use_strict_day1 = bool(self.features_config.get("enforce_day1_start", True))

            if getattr(self, "_in_real_sim", False):
                use_strict_day1 = True

            first_eval_ts = (
                pd.to_datetime(true_test_start)
                if bool(getattr(self, "_in_optuna_cv", False))
                else (
                    enforce_day1_eval_anchor(test_data.index, true_test_start)
                    if use_strict_day1 else
                    first_tradable_test_bar(test_data.index, true_test_start)
                )
            )

        
            if bool(getattr(self, "_in_optuna_cv", False)):
                if self._is_debug():
                    print(f"[CV/CLASSICAL] Eval anchor forced to fold start: {first_eval_ts} | test_len={len(test_data)} | warmup_need={_total_warmup_need}")

            if first_eval_ts is None:
                print("[ERR] No tradable bar found in test window.")

                 # IMPORTANT: never persist heavy frames during Optuna CV.
                if in_cv:
                    self.results = None
                    self.results_full = None
                    self._cv_last_eval_df = None
                else:
                    self.results = pd.DataFrame()
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:no_tradable_test_bar")
            self._expected_eval_start = first_eval_ts
        
            # ------------------------------------------------------------------
            # Patch D: Eligibility diagnostics (CV vs real comparability; no logic change)
            # Records: eligible_bars, embargo_dropped, warmup_need, eval_anchor_ts, etc.
            # ------------------------------------------------------------------
            try:
                _in_cv   = bool(getattr(self, "_in_optuna_cv", False))
                _in_real = bool(getattr(self, "_in_real_sim", False))

                # Month-only slice (no warmup) for "how many bars exist this month?"
                _month_raw = full_data.loc[true_test_start:test_end]
                _raw_n = int(len(_month_raw))
                # Apply the same NY session mask to the month slice (test-side semantics)
                _month_sess = _month_raw
                if sess_mode in ("test_only", "both"):
                    _month_sess = _month_raw.loc[
                        self._ny_mask.reindex(_month_raw.index, fill_value=False)
                    ]
                _sess_n = int(len(_month_sess))
                _sess_drop = int(max(0, _raw_n - _sess_n))

                # Embargo is disabled for CV heads (your existing behavior)
                _embargo_used = int(self.features_config.get("final_embargo_bars", 0) or 0)
                if _in_cv:
                    _embargo_used = 0
                _emb_drop = int(min(_embargo_used, _sess_n))

                _after_emb = _month_sess.iloc[_emb_drop:] if _emb_drop > 0 else _month_sess
                _post_emb_n = int(len(_after_emb))

                # Eligibility after the evaluation anchor (day-1 anchor / fold start)
                _anchor_ts = first_eval_ts
                if _anchor_ts is not None and _post_emb_n > 0:
                    _eligible_n = int(len(_after_emb.loc[_anchor_ts:]))
                else:
                    _eligible_n = int(_post_emb_n)
                _anchor_drop = int(max(0, _post_emb_n - _eligible_n))

                self._last_eligibility_diag = {
                    "in_cv": _in_cv,
                    "in_real": _in_real,
                    "sess_mode": sess_mode,
                    "raw_month_bars": _raw_n,
                    "session_month_bars": _sess_n,
                    "session_dropped": _sess_drop,
                    "final_embargo_bars_used": _embargo_used,
                    "embargo_dropped": _emb_drop,
                    "post_embargo_bars": _post_emb_n,
                    # Additive denominator for GateSummary: total bars on eval grid
                    # (after session filter + final embargo, before anchor selection).
                    "bars_total": _post_emb_n,
                    "warmup_need": int(warmup_need),
                    "warmup_plus_embargo_need": int(_total_warmup_need),
                    "eval_anchor_ts": str(_anchor_ts) if _anchor_ts is not None else None,
                    "eligible_bars": _eligible_n,
                    "anchor_dropped": _anchor_drop,
                }
            except Exception:
                self._last_eligibility_diag = {}

            cfg = self.apply_feature_defaults()
            lag_depth    = cfg.get("lag_depth", 1)
            roll_windows = cfg.get("roll_windows", [5])
            lags_eff = int(cfg.get("lags_range", cfg.get("lags", lags)))
            if self._is_debug():
                print(f"[TEST] effective_lags={lags_eff} (cfg-precedence)")

            # === Feature engineering (TRAIN) ===
            train_data, features = self.prepare_features(
                train_data, lags_eff, lag_depth=lag_depth, roll_windows=roll_windows
            )

            # if self._is_debug():
            #     print("Train data length after prepare_features:", len(train_data))
            #     print("Train data columns after prepare_features:", train_data.columns.tolist())

            if len(train_data) < 100:
                raise ValueError("Training data too short.")
            train_data = train_data.loc[:, ~train_data.columns.duplicated()]

            # --- robust feature prefilter on TRAIN only (near-constant -> corr -> MI) ---
            keep = None  # ensure defined even if an exception occurs
            if bool((self.features_config or {}).get("use_prefilter", True)) and features:
                try:
                    # # Build a provisional label on the training slice (no leakage into TEST)
                    # ret_fwd_pre = train_data["returns"].shift(-1)
                    # thr = float((self.features_config or {}).get("label_threshold", 1e-4))

                    # # label_with_neutral is a class method; align to train_data index
                    # y_pre = pd.Series(
                    #     self.label_with_neutral(ret_fwd_pre, threshold=thr),
                    #     index=train_data.index
                    # ).astype(int)
                
                    # Build a provisional label on the training slice (no leakage into TEST)
                    cfg = getattr(self, "features_config", {}) or {}

                    tb_on = bool(cfg.get("use_triple_barrier", False))
                    if tb_on:
                        y_pre = triple_barrier_labels(
                            close=train_data["price"],
                            pt_mult=float(cfg.get("tb_pt_mult", 1.5)),
                            sl_mult=float(cfg.get("tb_sl_mult", 1.0)),
                            max_holding=int(cfg.get("tb_max_holding", 48)),
                            neutral_zone=float(cfg.get("tb_neutral_zone", 0.0)),
                            neutral_zone_is_sigma=bool(cfg.get("tb_neutral_zone_is_sigma", False)),
                        ).astype(int)
                    else:
                        ret_fwd_pre = train_data["returns"].shift(-1)
                        thr = float(cfg.get("label_threshold", 1e-4))
                        y_pre = pd.Series(self.label_with_neutral(ret_fwd_pre, threshold=thr),
                                        index=train_data.index).astype(int)


                    # Explicit index intersection for robustness
                    common_idx = train_data.index.intersection(y_pre.index)
                    X_pref = train_data.loc[common_idx, features]
                    y_pref = y_pre.loc[common_idx]

                    # 3-stage prefilter (near-constant -> high-corr collapse -> MI top-K)
                    keep = prefilter_features_train(
                        X=X_pref,
                        y=y_pref,
                        cfg=(self.features_config or {}),
                    )
                except Exception as e:
                    print(f"[WARN] Prefilter skipped (non-fatal): {e}")
            else:
                if self._is_debug():
                    print("[Prefilter] disabled via config or empty feature list.")

            # Apply the reduced feature set only if it truly shrank
            if keep and len(keep) < len(features):
                if self._is_debug():
                    print(f"[Prefilter] Kept {len(keep)}/{len(features)} features.")
                features = [f for f in features if f in set(keep)]
            else:
                if self._is_debug():
                    print("[Prefilter] No change to feature set.")

            # Impute on TRAIN, then apply to both TRAIN and TEST
            imputer = SimpleImputer(strategy="mean")
            train_imputed = pd.DataFrame(
                imputer.fit_transform(train_data[features]),
                index=train_data.index, columns=features
            )

            # (Test imputation happens later once test_data is prepared)
            train_data_scaled, means, stds = self.scale_features(
                pd.concat([train_data.drop(columns=features), train_imputed], axis=1),
                features, log_id=f"test_train_{train_start.date()}_{train_end.date()}"
            )

            # Drop rows with remaining NaNs (train)
            orig_train_len = len(train_data_scaled)
            if train_data_scaled[features].isna().any().any():
                n_dropped = train_data_scaled[features].isna().any(axis=1).sum()
                print(
                    f"[WARN] Dropping {n_dropped} ({n_dropped/orig_train_len:.2%}) train rows with NaN after impute+scale (test_strategy).",
                    train_data_scaled[features].isna().sum()[train_data_scaled[features].isna().sum() > 0],
                )
                train_data_scaled = train_data_scaled[~train_data_scaled[features].isna().any(axis=1)]
                if len(train_data_scaled) == 0:
                    print("[WARN] All train rows dropped after impute+scale. Skipping fold.")
                    return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:train_all_rows_dropped")

                        # -------------------------------------------------------------
            # Labels (train) -- unified regime
            # If TripleBarrier is enabled, all supervised families (classical + deep)
            # train on TB event labels. Otherwise fall back to next-bar (T+1) labels.
            # -------------------------------------------------------------
            cfg_lbl = getattr(self, "features_config", {}) or {}
            tb_on_lbl = bool(cfg_lbl.get("use_triple_barrier", False))

            def _resolve_price_col(_df: "pd.DataFrame") -> str | None:
                """Resolve a close-like series for TB labels without assuming column names."""
                try:
                    if _df is None or len(_df) == 0:
                        return None
                    if "price" in _df.columns:
                        return "price"
                    if "mid_close" in _df.columns:
                        return "mid_close"
                    if "close" in _df.columns:
                        return "close"
                    if {"ask_close", "bid_close"}.issubset(_df.columns):
                        _df["__mid_close__"] = (_df["ask_close"] + _df["bid_close"]) / 2.0
                        return "__mid_close__"
                except Exception:
                    return None
                return None

            _pcol_tr = _resolve_price_col(train_data_scaled)
            if tb_on_lbl and _pcol_tr is None:
                if self._is_debug():
                    print("[WARN] TripleBarrier enabled but no price column; falling back to return-based labels (train).")
                tb_on_lbl = False

            if tb_on_lbl:
                y_train = triple_barrier_labels(
                    close=train_data_scaled[_pcol_tr].astype(float),
                    pt_mult=float(cfg_lbl.get("tb_pt_mult", 1.5)),
                    sl_mult=float(cfg_lbl.get("tb_sl_mult", 1.0)),
                    max_holding=int(cfg_lbl.get("tb_max_holding", 48)),
                    neutral_zone=float(cfg_lbl.get("tb_neutral_zone", 0.0)),
                    neutral_zone_is_sigma=bool(cfg_lbl.get("tb_neutral_zone_is_sigma", False)),
                ).astype(int)
            else:
                # Next-bar returns (T+1)
                _returns_fwd = train_data_scaled["returns"].shift(-1)
                # drop last row with NaN forward return to keep X and y aligned
                train_data_scaled = train_data_scaled.loc[_returns_fwd.notna()].copy()
                y_train = self.label_with_neutral(
                    _returns_fwd.loc[train_data_scaled.index],
                    threshold=label_threshold,
                )


            # Features (aligned to y_train)
            X_train = train_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
            if y_train is None or len(y_train) == 0:
                print("[WARN] No labels generated (all NaN or below threshold). Skipping fold.")
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:no_labels")
            y_train = y_train.astype(int)
            if self._is_debug():
                print("Label distribution in training set:", pd.Series(y_train).value_counts())


            # Make labels available to windowing helper (transformer/LSTM/CNN)
            train_data_scaled["label"] = y_train

            # -----------------------------------------------------------------
            # Train-anchored high-volatility threshold for cost regime switching
            # -----------------------------------------------------------------
            high_vol_thr_train = None
            def _push_thr_to_attrs(_df, thr):
                try:
                    if _df is None or len(_df) == 0 or thr is None:
                        return
                    fc = dict(_df.attrs.get("features_config", {}) or {})
                    fc["high_vol_thr"] = float(thr)
                    _df.attrs["features_config"] = fc
                except Exception:
                    pass
            
            try:
                from utilsNoWFO import realized_vol as _rv_fn
                _vol_w = int(cfg_f.get("vol_window_bars", _PC["vol_window_bars"]))
                _qhi   = float(cfg_f.get("high_vol_q", _PC["high_vol_q"]))
                _rv  = _rv_fn(train_data_scaled["returns"].astype(float), window=_vol_w)
                _thr = float(_rv.quantile(_qhi))
                if np.isfinite(_thr):
                    high_vol_thr_train = _thr
                    if self._is_debug():
                        print(f"[Costs] Train-anchored high_vol_thr={high_vol_thr_train:.8f} (q={_qhi:.2f}, vol_w={_vol_w})")

                    # 1) config path (used by _ensure_cost_columns when config is passed)
                    try:
                        if not isinstance(config, dict):
                            config = {}
                    except Exception:
                        config = {}
                        
                    # Cache on instance for downstream consumers (e.g., Top-N consensus)
                    # when config/attrs propagation is temporarily missing.
                    try:
                        self._last_high_vol_thr_train = float(high_vol_thr_train)
                    except Exception:
                        pass

                    # Also mirror into DataFrame attrs (best-effort).
                    try:
                        _push_thr_to_attrs(train_data_scaled, high_vol_thr_train)
                    except Exception:
                        pass

            except Exception as _e:
                if self._is_debug():
                    print(f"[Costs] Failed to compute train-anchored high_vol_thr: {_e}")
                        

            # Guard: minimum per-class and at least 2 classes
            # Require at least 2 classes and a minimum count per class.
            # In Optuna CV, optionally prune immediately when labels collapse.
            _cv_cfg = getattr(self, "_cv_config_current", None) or getattr(self, "cv_config", None) or {}
            try:
                MIN_CLASS_SAMPLES = int(_cv_cfg.get("cv_min_class_samples", 5))
            except Exception:
                MIN_CLASS_SAMPLES = 5
            unique, counts = np.unique(y_train, return_counts=True)
            class_counts = dict(zip(unique, counts))
            too_few = [cls for cls, count in class_counts.items() if count < MIN_CLASS_SAMPLES]
            if len(too_few) > 0 or len(class_counts) < 2:
                msg = (f"[WARN] Skipping fold: Not enough samples for classes {too_few} "
                       f"or only one class present: {class_counts}")
                print(msg)

                # Early prune (CV only): don't waste compute on a broken label regime
                _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                _prune_lbl  = bool(_cv_cfg.get("cv_prune_on_label_collapse", True))
                if _in_cv_mode and _prune_lbl:
                    try:
                        import optuna as _opt
                        raise _opt.TrialPruned(msg)
                    except Exception:
                        # If Optuna isn't available for some reason, fall back to invalid metrics.
                        pass

                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:insufficient_class_support")

            # === Feature engineering (TEST) ===
            test_data, _ = self.prepare_features(
                test_data, lags_eff, lag_depth=lag_depth, roll_windows=roll_windows
            )

            if test_data is None or test_data.empty:
                print(f"[ERROR] test_data is empty after prepare_features for test period {test_start} - {test_end}")
                # IMPORTANT: never persist heavy frames during Optuna CV.
                if in_cv:
                     self.results = None
                     self.results_full = None
                     self._cv_last_eval_df = None
                else:
                     self.results = pd.DataFrame()
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:empty_test_data")

            test_data_raw_with_extras = test_data.copy()
            if self._is_debug():
                print("Test data length after prepare_features:", len(test_data))

            if len(test_data) < 20:
                raise ValueError("Test data too short.")
            test_data = test_data.loc[:, ~test_data.columns.duplicated()]

            # Align + scale test with train stats
            test_imputed = pd.DataFrame(
                imputer.transform(test_data[features]),
                index=test_data.index, columns=features
            )
            test_data_scaled, _, _ = self.scale_features(
                pd.concat([test_data.drop(columns=features), test_imputed], axis=1),
                features, means, stds, log_id=f"test_eval_{test_start.date()}_{test_end.date()}"
            )

            # Force same columns/order as training
            test_data_scaled = test_data_scaled.reindex(columns=features)

            # Drop rows with NaNs (test)
            orig_test_len = len(test_data_scaled)
            if test_data_scaled[features].isna().any().any():
                n_dropped = test_data_scaled[features].isna().any(axis=1).sum()
                print(
                    f"[WARN] Dropping {n_dropped} ({n_dropped/orig_test_len:.2%}) test rows with NaN after impute+scale (test_strategy).",
                    test_data_scaled[features].isna().sum()[test_data_scaled[features].isna().sum() > 0],
                )
                test_data_scaled = test_data_scaled[~test_data_scaled[features].isna().any(axis=1)]

            test_data_scaled = test_data_scaled.copy().reindex(columns=features)
            X_test = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)

            if len(X_test) == 0:
                print("[ERR] [ABORT] Empty X_test after scaling/alignment.")
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:empty_X_test")
        
            # evaluation starts at the first tradable bar of the month (after session filter)
            eval_anchor_ts = getattr(self, "_expected_eval_start", pd.to_datetime(test_start))
            eval_mask = test_data_scaled.index >= eval_anchor_ts
        
            # ------------------------------------------------------------------
            # Patch D (final write): eligibility diagnostics on the POST-FEATURE grid
            # Grid must match the exact dataframe used for gating/evaluation.
            # ------------------------------------------------------------------
            try:
                eval_anchor_ts = getattr(self, "_expected_eval_start", pd.to_datetime(test_start))
                gating_df = test_data_scaled  # exact post-feature, post-dropna, post-scale grid
                bars_total = int(len(gating_df))

                # Eligible bars are those in the evaluation window (>= anchor)
                eligible_mask_pf = gating_df.index >= eval_anchor_ts
                eligible_bars_pf = int(np.sum(eligible_mask_pf))

                # Optional split: bars between true month start and anchor (on the same grid)
                try:
                    _tts = pd.to_datetime(true_test_start)
                except Exception:
                    _tts = pd.to_datetime(test_start)
                anchor_dropped_pf = int(np.sum((gating_df.index >= _tts) & (gating_df.index < eval_anchor_ts)))

                # Warmup bars are everything before true month start (on the same grid)
                warmup_dropped_pf = int(bars_total - eligible_bars_pf - anchor_dropped_pf)
                if warmup_dropped_pf < 0:
                        warmup_dropped_pf = 0

                # Update (not replace) so pre-feature month diagnostics remain available
                _diag_pf = dict(getattr(self, "_last_eligibility_diag", {}) or {})
                _diag_pf.update({
                    "post_feature_grid": True,
                    "bars_total": bars_total,
                    "gating_df_len": bars_total,
                    "eligible_bars": eligible_bars_pf,
                    "warmup_dropped": warmup_dropped_pf,
                    "anchor_dropped": anchor_dropped_pf,
                    "eval_anchor_ts": str(eval_anchor_ts) if eval_anchor_ts is not None else None,
                    "true_test_start": str(_tts) if _tts is not None else None,
                })

                # Fail-loud invariants (debug-only): prevent mixed denominators in GateSummary
                if self._is_debug():
                    _lhs = int(_diag_pf.get("warmup_dropped", 0) or 0) + int(_diag_pf.get("anchor_dropped", 0) or 0) + int(_diag_pf.get("eligible_bars", 0) or 0)
                    if bars_total and _lhs and _lhs != bars_total:
                        print(
                            f"[WARN] [EligDiag] Invariant mismatch on post-feature grid: "
                            f"warm({warmup_dropped_pf})+anch({anchor_dropped_pf})+elig({eligible_bars_pf})={_lhs} vs bars_total={bars_total}"
                        )

                self._last_eligibility_diag = _diag_pf
                # ------------------------------------------------------------
                # Explicit eligibility audit log (no behavior change)
                # Confirms: post-feature denominator, warmup dropped, eval anchor.
                # ------------------------------------------------------------
                try:
                    _ctx = "eval"
                    if bool(getattr(self, "_in_optuna_cv", False)):
                        _ctx = "cv"
                    elif bool(getattr(self, "_in_real_sim", False)):
                        _mx = getattr(self, "_rt_month_idx", None)
                        _ctx = f"real_m{int(_mx)}" if _mx is not None else "real"

                    print(
                        f"[Eligibility] post_feature_bars_total={int(bars_total)} "
                        f"eligible={int(eligible_bars_pf)} "
                        f"warmup_dropped={int(warmup_dropped_pf)} "
                        f"anchor={str(eval_anchor_ts)} "
                        f"ctx={_ctx}"
                    )
                except Exception:
                    pass
            except Exception as _e:
                if self._is_debug():
                    print(f"[WARN] [EligDiag] Post-feature eligibility diag failed (non-fatal): {_e}")

            # Patch D (eligibility diagnostics, post-feature):
            # Recompute on the SAME bar grid used for gating/eval (test_data_scaled),
            # so GateSummary doesn't mix pre-feature monthly counts with post-feature counts.
            try:
                _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                _idx = test_data_scaled.index
                _month_start_ts = pd.to_datetime(true_test_start)

                _bars_total = int(len(_idx))
                _warm_drop = int(np.sum(_idx < _month_start_ts))
            
                _anchor_ts = pd.to_datetime(eval_anchor_ts) if eval_anchor_ts is not None else None
                _eligible = int(np.sum(_idx >= _anchor_ts)) if _anchor_ts is not None else _bars_total
                _anch_drop = int(np.sum((_idx >= _month_start_ts) & (_idx < _anchor_ts))) if _anchor_ts is not None else 0
            
                _diag.update(
                    {
                        "bars_total": _bars_total,
                        "eligible_bars": _eligible,
                        "warmup_dropped": _warm_drop,
                        "anchor_dropped": _anch_drop,
                        "eval_anchor_ts": str(_anchor_ts) if _anchor_ts is not None else _diag.get("eval_anchor_ts"),
                    }
                )
                # Back-compat: keep this aligned with gating grid so denom matches what we trade/gate on
                _diag["raw_month_bars"] = _bars_total
                self._last_eligibility_diag = _diag
            except Exception:
                pass


            # ----------------------------
            # 3) Build & fit model
            # ----------------------------
            deep_models = ["cnn", "lstm", "transformer"]
            ensemble_models = [
                "ensemble_cnn_lstm_xgboost",
                "ensemble_adaptive_regime",
            ]

            params = self.features_config.copy()
            model_type = params.pop("model_type", self.model_type)
            # Keep internal tag in sync so gating/threshold logic sees the right model
            self.model_type = model_type
            self._maybe_configure_tf_runtime_once(model_type)
            
            # Optuna CV: release TF graph/session + model refs after this fold.
            # (Runs via _persist_results_guard() even on early returns.)
            try:
                if bool(in_cv) and str(model_type) in set(deep_models):
                    self._tf_cleanup_do = True
                    self._tf_cleanup_del_model = True
            except Exception:
                pass
            params.pop("input_shape", None)

            mem_gb_pre_fit = psutil.virtual_memory().used / (1024**3)
            if self._is_debug():
                print(f"[RAM] Before model fit: {mem_gb_pre_fit:.2f} GB used")

            class _TimeLimit(Callback):
                """Hard wall-clock cap for deep trainings."""
                def __init__(self, seconds):
                    super().__init__()
                    self.seconds = float(seconds) if seconds is not None else None
                    self._start = None
                def on_train_begin(self, logs=None):
                    if self.seconds is not None:
                        self._start = time.time()
                def on_batch_end(self, batch, logs=None):
                    if self.seconds is not None and (time.time() - self._start) > self.seconds:
                        self.model.stop_training = True

            def _maybe_mixed_precision(enable: bool, tag: str):
                if not enable:
                    return
                try:
                    mixed_precision.set_global_policy("mixed_float16")
                    print(f"[{tag}] Mixed precision enabled.")
                except Exception:
                    pass

            def _make_windows_fast(X2d: np.ndarray, win: int, stride: int = 1, labels_1d=None):
                """
                Vectorized sliding windows.
                X2d: (n, f) float32  ->  (m, win, f), y_seq (m,) if labels_1d provided, idx_end (m,)
                """
                n = X2d.shape[0]
                if n < win:
                    return None, None, None
                Xv = sliding_window_view(X2d, window_shape=win, axis=0)  # (n-win+1, win, f)
                if stride > 1:
                    Xv = Xv[::stride]
                m = Xv.shape[0]
                idx_end = np.arange(win - 1, win - 1 + m * stride, stride, dtype=int)
                yv = labels_1d[idx_end] if labels_1d is not None else None
                return Xv, yv, idx_end
            
            def _start_idx_for_last_strided_windows(n_rows: int, win: int, stride: int, max_windows: int) -> int:
                """
                Compute the *exact* starting row index so that:
                  make_windows(stride) then take [-max_windows:]
                is identical to:
                  slice df[start_idx:] then make_windows(stride) (and optionally still slice)
                This avoids building huge intermediate arrays during CV/HPO.
                """
                try:
                    n_rows = int(n_rows)
                    win = max(1, int(win))
                    stride = max(1, int(stride))
                    max_windows = int(max_windows) if max_windows is not None else 0
                    if max_windows <= 0:
                        return 0
                    total = n_rows - win + 1  # number of raw windows before stride
                    if total <= 0:
                        return 0
                    m = (total + stride - 1) // stride  # number of windows after stride
                    if m <= max_windows:
                        return 0
                    k0 = m - max_windows
                    return int(k0 * stride)
                except Exception:
                    return 0

            def _start_idx_for_last_stride_rows(n_rows: int, stride: int, max_rows: int) -> int:
                """
                For 3D-feed paths:
                  X[::stride] then take [-max_rows:]
                -> compute start row so slicing first preserves exact same sampled rows.
                """
                try:
                    n_rows = int(n_rows)
                    stride = max(1, int(stride))
                    max_rows = int(max_rows) if max_rows is not None else 0
                    if max_rows <= 0:
                        return 0
                    m = (n_rows + stride - 1) // stride
                    if m <= max_rows:
                        return 0
                    k0 = m - max_rows
                    return int(k0 * stride)
                except Exception:
                    return 0

        
            # ---- branch: deep / ensemble / classical ----
            if model_type in deep_models:

                if model_type == "transformer":
                    # knobs
                    train_stride   = int(params.get("transformer_train_stride", 2))
                    batch_size     = int(params.get("transformer_batch_size", 128))
                    epochs         = int(params.get("transformer_epochs", 20))
                    use_mixed_prec = bool(params.get("transformer_mixed_precision", True)) and bool(tf.config.list_physical_devices("GPU"))

                    in_cv = bool(getattr(self, "_in_optuna_cv", False))
                    if in_cv:
                        # Multi-fidelity: coarser stride + fewer windows during CV.
                        # Wu et al. (2020) and Won et al. (2025) use sample/epoch
                        # count as natural fidelities in deep HPO.
                        cfg = getattr(self, "features_config", {}) or {}
                        stride_min = int(cfg.get("transformer_cv_train_stride", train_stride))
                        train_stride = max(train_stride, stride_min)  # higher stride = fewer windows (cheaper)
                        
                    _maybe_mixed_precision(use_mixed_prec, "Transformer")


                    win = max(2, int(lags_eff))
                    
                    # Cap training windows if requested (compute BEFORE materializing arrays)
                    max_train_windows = int(params.get("deep_max_train_windows", 10000))
                    if in_cv:
                        cfg = getattr(self, "features_config", {}) or {}
                        max_cap = int(cfg.get("transformer_cv_max_train_windows", max_train_windows))
                        max_train_windows = min(max_train_windows, max_cap)
                        
                    use_subproc = bool((getattr(self, "features_config", {}) or {}).get("deep_use_subprocess", False)) \
                        or str(os.getenv("MLB_DEEP_SUBPROCESS", "0")).lower() in ("1", "true", "yes")
                    if use_subproc and (not in_cv):
                        X2d_train = train_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
                        y1d_train = train_data_scaled["label"].to_numpy(dtype=np.int32, copy=False)
                        X2d_test  = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)

                        proba_sub, thr_sub = self._deep_fit_predict_subprocess(
                            model_type="transformer",
                            mode="seq",
                            X_train_2d=X2d_train,
                            y_train_1d=y1d_train,
                            X_test_2d=X2d_test,
                            win=win,
                            train_stride=train_stride,
                            max_train_windows=max_train_windows,
                            batch_size=batch_size,
                            epochs=epochs,
                            params=dict(params),
                        )
                        if proba_sub is not None:
                            self._deep_subproc_proba = proba_sub
                            if thr_sub is not None and np.isfinite(thr_sub):
                                self._coverage_conf_thr = float(thr_sub)
                                self._deep_coverage_thr = float(thr_sub)
                            # Skip in-proc TF build/fit entirely
                            self.model = None
                            # jump to prediction section
                            goto_predict = True
                        else:
                            goto_predict = False
                    else:
                        goto_predict = False

                    if not goto_predict:
                        # Pre-slice DF so that the resulting (strided) windows are identical to the
                        # previous approach (build-all -> stride -> take last max_train_windows).
                        _si = _start_idx_for_last_strided_windows(
                            len(train_data_scaled), win, train_stride, max_train_windows
                        )
                        _df_tr = train_data_scaled.iloc[_si:] if _si > 0 else train_data_scaled
                        X2d_train = _df_tr[features].to_numpy(dtype=np.float32, copy=False)
                        y1d_train = _df_tr["label"].to_numpy(dtype=np.int32, copy=False)


                        X_seq_train, y_seq_train, _ = _make_windows_fast(
                            X2d_train, win=win, stride=max(1, train_stride), labels_1d=y1d_train
                        )
                        if X_seq_train is None or len(X_seq_train) == 0:
                            print("[ERR] [ABORT] Empty training sequences for transformer.")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:transformer_empty_train_seq")

                        if X_seq_train.shape[0] > max_train_windows:
                            X_seq_train = X_seq_train[-max_train_windows:]
                            y_seq_train = y_seq_train[-max_train_windows:]

                        params["input_shape"] = (X_seq_train.shape[1], X_seq_train.shape[2])
                        self.model = self.get_model(model_type, **params)
                        # Tag for per-model CV caps (used in _fit_keras_with_cv_controls).
                        setattr(self.model, "_mlb_model_tag", "transformer")

                        # callbacks
                        cb = getattr(self.model, "early_stop_callback", None)


                        print(f"[DEEP] model={model_type} | seq_windows={X_seq_train.shape[0]} "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}")

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_seq_train, y_seq_train,
                            X_val=None, y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )
                    
                        # Unified deep calibration + coverage threshold (works in CV too)
                        X_cal = X_seq_train
                        y_cal = y_seq_train

                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)
                        try:
                            if X_cal is not None and callable(pred_fn):
                                self._fit_deep_calibration_and_coverage(
                                    X_cal=X_cal, y_cal=y_cal, pred_fn=pred_fn,
                                    model_type=model_type, in_cv=in_cv
                                )
                                
                            # IMPORTANT: X_seq_train is a view over X2d_train (sliding_window_view),
                            # so free the base arrays too once training+calibration is done.
                            try:
                                del X2d_train, y1d_train, _df_tr
                            except Exception:
                                pass
                        finally:
                            del X_seq_train, y_seq_train, X_cal, y_cal, pred_fn
                        _gc.collect()

                elif model_type == "lstm":
                    lstm_use_seq   = bool(params.get("lstm_use_seq_windows", True))
                    train_stride   = int(params.get("lstm_train_stride", 2))
                    batch_size     = int(params.get("lstm_batch_size", 128))
                    epochs         = int(params.get("lstm_epochs", 20))
                    use_mixed_prec = bool(params.get("lstm_mixed_precision", True)) and bool(
                        tf.config.list_physical_devices("GPU")
                    )

                    in_cv = bool(getattr(self, "_in_optuna_cv", False))
                    if in_cv:
                        cfg = getattr(self, "features_config", {}) or {}
                        stride_min = int(cfg.get("lstm_cv_train_stride", train_stride))
                        train_stride = max(train_stride, stride_min)


                    _maybe_mixed_precision(use_mixed_prec, "LSTM")

                    # ---------------------------------------------
                    # A) LSTM with sequence windows (X_seq_train)
                    # ---------------------------------------------
                    if lstm_use_seq:
                        win = max(2, int(lags_eff))

                        max_train_windows = int(params.get("deep_max_train_windows", 10000))
                        if in_cv:
                            cfg = getattr(self, "features_config", {}) or {}
                            max_cap = int(cfg.get("lstm_cv_max_train_windows", max_train_windows))
                            max_train_windows = min(max_train_windows, max_cap)

                        _si = _start_idx_for_last_strided_windows(len(train_data_scaled), win, train_stride, max_train_windows)
                        _df_tr = train_data_scaled.iloc[_si:] if _si > 0 else train_data_scaled
                        X2d = _df_tr[features].to_numpy(dtype=np.float32, copy=False)
                        y1d = _df_tr["label"].to_numpy(dtype=np.int32, copy=False)

                        X_seq_train, y_seq_train, _ = _make_windows_fast(
                            X2d,
                            win=win,
                            stride=max(1, train_stride),
                            labels_1d=y1d,
                        )
                        if X_seq_train is None or len(X_seq_train) == 0:
                            print("[ERR] [ABORT] Empty training sequences for LSTM (seq mode).")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:lstm_empty_train_seq")
                    

                        if X_seq_train.shape[0] > max_train_windows:
                            X_seq_train = X_seq_train[-max_train_windows:]
                            y_seq_train = y_seq_train[-max_train_windows:]

                        params.setdefault("lstm_use_early_stopping", True)
                        params["input_shape"] = (X_seq_train.shape[1], X_seq_train.shape[2])

                        self.model = self.get_model(model_type, **params)
                        # Tag for per-model CV caps.
                        setattr(self.model, "_mlb_model_tag", "lstm")
                        cb = getattr(self.model, "early_stop_callback", None)


                        print(
                            f"[DEEP] model={model_type} | seq_windows={X_seq_train.shape[0]} "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}"
                        )

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_seq_train,
                            y_seq_train,
                            X_val=None,
                            y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )

                        X_cal = X_seq_train
                        y_cal = y_seq_train

                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)
                        try:
                            if X_cal is not None and callable(pred_fn):
                                self._fit_deep_calibration_and_coverage(
                                    X_cal=X_cal, y_cal=y_cal, pred_fn=pred_fn,
                                    model_type=model_type, in_cv=in_cv
                                )
                                
                            # cleanup (seq only)
                            # IMPORTANT: X_seq_train is a view over X2d (sliding_window_view).
                            # Free base arrays/DF slice too to reduce RSS growth.
                            try:
                                del X2d, y1d, _df_tr
                            except Exception:
                                pass
                        finally:
                            del X_seq_train, y_seq_train, X_cal, y_cal, pred_fn
                        _gc.collect()

                    # ---------------------------------------------
                    # B) LSTM with simple 3D feed (no seq windows)
                    # ---------------------------------------------
                    else:
                        params["input_shape"] = (X_train.shape[1], 1)
                        self.model = self.get_model(model_type, **params)
                        setattr(self.model, "_mlb_model_tag", "lstm")

                        max_train_windows = int(params.get("deep_max_train_windows", 10000))

                        # Preserve exact rows vs old path: (X_train[::stride] then tail-slice)
                        _si = _start_idx_for_last_stride_rows(X_train.shape[0], train_stride, max_train_windows)
                        if _si > 0:
                            X_tr2 = X_train[_si:]
                            y_tr2 = y_train[_si:]
                        else:
                            X_tr2 = X_train
                            y_tr2 = y_train

                        # (N, features, 1)
                        X_train_3d = X_tr2.astype(np.float32).reshape((X_tr2.shape[0], X_tr2.shape[1], 1))

                        # Optional stride-based downsampling
                        if train_stride > 1:
                            X_train_3d = X_train_3d[::train_stride]
                            y_train_eff = y_tr2[::train_stride]
                        else:
                            y_train_eff = y_tr2

                        # Tail cap (apply to both)
                        if X_train_3d.shape[0] > max_train_windows:
                            X_train_3d  = X_train_3d[-max_train_windows:]
                            y_train_eff = y_train_eff[-max_train_windows:]

                        # Hard guard (prevents silent garbage)
                        if int(X_train_3d.shape[0]) != int(len(y_train_eff)):
                            raise ValueError(
                                f"LSTM 3D-feed X/y mismatch: X={X_train_3d.shape[0]} y={len(y_train_eff)} "
                                f"(train_stride={train_stride}, max_train_windows={max_train_windows}, _si={_si})"
                            )

                        cb = getattr(self.model, "early_stop_callback", None)
                        print(f"[DEEP] model={model_type} | seq_windows=NA(3D-feed) "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}")

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_train_3d, y_train_eff,
                            X_val=None, y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )

                        # Calibration inputs for shared block
                        X_cal  = X_train_3d
                        y_cal  = y_train_eff
                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)
                        try:
                            if X_cal is not None and callable(pred_fn):
                                self._fit_deep_calibration_and_coverage(
                                    X_cal=X_cal, y_cal=y_cal, pred_fn=pred_fn,
                                    model_type=model_type, in_cv=in_cv
                                )
                        finally:
                            del X_cal, y_cal, pred_fn
                        
                        # Free intermediate bases too (can be large)
                        try:
                            del X_tr2, y_tr2
                        except Exception:
                            pass
                        del X_train_3d, y_train_eff
                        
                        _gc.collect()

                else:  # CNN
                    cnn_use_seq    = bool(params.get("cnn_use_seq_windows", True))
                    train_stride   = max(1, int(params.get("cnn_train_stride", 3)))
                    batch_size     = int(params.get("cnn_batch_size", 128))
                    epochs         = min(int(params.get("cnn_epochs", 20)), 40)
                    use_mixed_prec = bool(params.get("cnn_mixed_precision", True)) and bool(tf.config.list_physical_devices("GPU"))
                    in_cv = bool(getattr(self, "_in_optuna_cv", False))
                    if in_cv:
                        cfg = getattr(self, "features_config", {}) or {}
                        stride_min = int(cfg.get("cnn_cv_train_stride", train_stride))
                        train_stride = max(train_stride, stride_min)

                    max_train_windows = int(params.get("deep_max_train_windows", 10000))
                    if in_cv:
                        cfg = getattr(self, "features_config", {}) or {}
                        max_cap = int(cfg.get("cnn_cv_max_train_windows", max_train_windows))
                        max_train_windows = min(max_train_windows, max_cap)


                    _maybe_mixed_precision(use_mixed_prec, "CNN")
                    params.setdefault("cnn_use_early_stopping", True)

                    # We'll set these after the fit, so the calibration block is shared
                    X_cal = None
                    y_cal = None
                    pred_fn = None

                    if cnn_use_seq:
                        # ---- Sequence windowing path ----
                        win = max(2, int(lags_eff))
                        
                        # Pre-slice DF so windows match build-all->stride->tail-slice
                        _si = _start_idx_for_last_strided_windows(len(train_data_scaled), win, train_stride, max_train_windows)
                        _df_tr = train_data_scaled.iloc[_si:] if _si > 0 else train_data_scaled
                        X2d = _df_tr[features].to_numpy(dtype=np.float32, copy=False)
                        y1d = _df_tr["label"].to_numpy(dtype=np.int32, copy=False)
 

                        X_seq_train, y_seq_train, _ = _make_windows_fast(
                            X2d, win=win, stride=train_stride, labels_1d=y1d
                        )
                        if X_seq_train is None or len(X_seq_train) == 0:
                            print("[ERR] [ABORT] Empty training sequences for CNN (seq mode).")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_empty_train_seq")

                        if X_seq_train.shape[0] > max_train_windows:
                            X_seq_train = X_seq_train[-max_train_windows:]
                            y_seq_train = y_seq_train[-max_train_windows:]

                        params["input_shape"] = (X_seq_train.shape[1], X_seq_train.shape[2])
                        self.model = self.get_model(model_type, **params)
                        # Tag for per-model CV caps.
                        setattr(self.model, "_mlb_model_tag", "cnn")

                        cb = getattr(self.model, "early_stop_callback", None)


                        print(f"[DEEP] model={model_type} | seq_windows={X_seq_train.shape[0]} "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}")

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_seq_train, y_seq_train,
                            X_val=None, y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )

                        # Calibration inputs for shared block
                        X_cal  = X_seq_train
                        y_cal  = y_seq_train
                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)

                    else:
                        # ---- 3D "image-like" feed path ----
                        params["input_shape"] = (X_train.shape[1], 1)
                        self.model = self.get_model(model_type, **params)

                        # Preserve exact rows vs old path: (X_train[::stride] then tail-slice)
                        _si = _start_idx_for_last_stride_rows(X_train.shape[0], train_stride, max_train_windows)
                        if _si > 0:
                            X_tr2 = X_train[_si:]
                            y_tr2 = y_train[_si:]
                        else:
                            X_tr2 = X_train
                            y_tr2 = y_train

                        X_train_3d = X_tr2.astype(np.float32).reshape((X_tr2.shape[0], X_tr2.shape[1], 1))
                        
                        if train_stride > 1:
                            X_train_3d = X_train_3d[::train_stride]
                            y_train_eff = y_tr2[::train_stride]
                        else:
                            y_train_eff = y_tr2
 

                        if X_train_3d.shape[0] > max_train_windows:
                            X_train_3d  = X_train_3d[-max_train_windows:]
                            y_train_eff = y_train_eff[-max_train_windows:]

                        cb = getattr(self.model, "early_stop_callback", None)

                        print(f"[DEEP] model={model_type} | seq_windows=NA(3D-feed) "
                            f"| early_stopping={bool(cb)} | max_train_windows={max_train_windows}")

                        self._fit_keras_with_cv_controls(
                            self.model,
                            X_train_3d, y_train_eff,
                            X_val=None, y_val=None,
                            base_epochs=epochs,
                            base_batch=batch_size,
                            verbose=0,
                            validation_split_if_needed=0.10,
                            extra_callbacks=None,
                        )

                        # Calibration inputs for shared block
                        X_cal  = X_train_3d
                        y_cal  = y_train_eff
                        pred_fn = lambda X: self.model.predict(X, verbose=0, batch_size=batch_size)


                    try:
                        if X_cal is not None and callable(pred_fn):
                            self._fit_deep_calibration_and_coverage(
                                X_cal=X_cal, y_cal=y_cal, pred_fn=pred_fn,
                                model_type=model_type, in_cv=in_cv
                            )
                    finally:
                        try:
                            if cnn_use_seq:
                                try:
                                    del X2d, y1d, _df_tr
                                except Exception:
                                    pass
                                del X_seq_train, y_seq_train
                            else:
                                try:
                                    del X_tr2, y_tr2
                                except Exception:
                                    pass
                                del X_train_3d, y_train_eff
                        except Exception:
                            pass
                        try:
                            del X_cal, y_cal, pred_fn
                        except Exception:
                            pass
                    _gc.collect()


            elif model_type in ensemble_models:
                raise RuntimeError(
                    f"{model_type} should not be trained via test_strategy(). "
                    "Use the dedicated ensemble handler functions."
                )
            else:
            
                # Classical ML (logistic/logistic_ovr/svm/rf/xgb/...)
                self.model = self.get_model(model_type, **params)
            
                # Fit FIRST (required for sklearn Pipelines / predict_proba).
                self.model.fit(X_train, y_train)

                # --- S16.3: Capture feature importance for classical models ---
                try:
                    from pipeline.diagnostics import compute_feature_importance
                    _feat_names = list(features) if isinstance(features, (list, tuple)) else None
                    _fi = compute_feature_importance(self.model, model_type, feature_names=_feat_names)
                    if _fi:
                        self._diagnostics_feature_importance = [(e.feature, e.importance) for e in _fi]
                    else:
                        self._diagnostics_feature_importance = []
                except Exception:
                    self._diagnostics_feature_importance = []

                # ------------------------------------------------------------
                # Classical coverage-threshold (train-anchored, causal)
                # IMPORTANT: must run AFTER fit, otherwise sklearn Pipelines
                # can raise "Pipeline is not fitted yet."
                # ------------------------------------------------------------
                try:
                    cfg = getattr(self, "features_config", {}) or {}
                    tgt = float(cfg.get("target_active_rate", cfg.get("target_coverage", 0.0)) or 0.0)
                    if tgt > 0.0 and hasattr(self.model, "predict_proba"):
                        frac = float(cfg.get("classical_calibration_frac", cfg.get("deep_calibration_frac", 0.15)))
                        nmin = int(cfg.get("classical_calibration_min_samples", cfg.get("deep_calibration_min_samples", 500)))
                        nwin = int(X_train.shape[0]) if hasattr(X_train, "shape") else 0
                        ncal = max(nmin, int(round(nwin * frac))) if nwin > 0 else 0
                        ncal = min(ncal, nwin - 1) if nwin > 1 else 0
                        if ncal >= 50:
                            X_cal = X_train[-ncal:]
                            p_cal = sanitize_proba(self.model.predict_proba(X_cal))
                            self._coverage_conf_thr = float(fit_coverage_threshold_on_calibration(p_cal, tgt))
                            setattr(self, "_cv_cov_thr_last", float(self._coverage_conf_thr))
                            try:
                                setattr(self, "_last_cov_cal_rows", int(ncal))
                            except Exception:
                                pass
                            _in_cv = False
                            try:
                                _in_cv = bool(getattr(self, "_in_cv", False) or getattr(self, "_in_optuna_cv", False))
                            except Exception:
                                _in_cv = False

                            _ctx = "cv" if _in_cv else "eval"

                            # Only label as real_mX when NOT in CV
                            if not _in_cv:
                                try:
                                    if bool(getattr(self, "_in_real_sim", False)):
                                        mx = int(cfg.get("month_ix", getattr(self, "_rt_month_ix", 0) or 0))
                                        _ctx = f"real_m{mx}"
                                except Exception:
                                    pass
                            print(
                                f"[Calib][Coverage] conf_thr={float(self._coverage_conf_thr):.6f} "
                                f"target_active_rate={float(tgt):.6f} cal_rows={int(ncal)} ctx={_ctx}"
                            )
                except Exception as _e:
                    if self._is_debug():
                        print(f"[WARN] [Calib][Classical] Coverage fit skipped: {_e}")


            # ---- RAM (soft guard) BEFORE prediction ----
            try:
                import gc, time
                free_gb = psutil.virtual_memory().available / (1024**3)
                used_gb = psutil.virtual_memory().used / (1024**3)
                if self._is_debug():
                    print(f"[RAM] Before model predict: used={used_gb:.2f} GB | free={free_gb:.2f} GB | floor={float(os.environ.get('MLB_MIN_FREE_GB','2.5')):.2f} GB")

                # Soft guard: try local cleanup if free RAM is below floor; never raise
                if free_gb < float(os.environ.get("MLB_MIN_FREE_GB", "2.5")):
                    _gc.collect(); time.sleep(0.05)
                    free_retry = psutil.virtual_memory().available / (1024**3)
                    if self._is_debug():
                        print(f"[RAM] After cleanup: free={free_retry:.2f} GB")
                    if free_retry < float(os.environ.get("MLB_MIN_FREE_GB", "2.5")):
                        print(f"[WARN] Low free RAM persists ({free_retry:.2f} GB); continuing without raising.")
            except Exception:
                pass

        
            # ----------------------------
            # 4) Predict (branch specific)
            # ----------------------------
            test_data_for_eval = None

            if model_type in deep_models:
                # Build sliding windows over TEST (stride=1) when in seq mode
                if model_type == "transformer":
                    win = max(2, int(lags_eff))

                    X2d_test = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
                    if X2d_test.shape[0] < win:
                        print("[ERR] [ABORT] Test set shorter than window size for transformer.")
                        return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:transformer_test_too_short")

                    n_win = int(X2d_test.shape[0] - win + 1)
                    idx_end = np.arange(win - 1, win - 1 + n_win, 1, dtype=int)

                    proba = getattr(self, "_deep_subproc_proba", None)
                    if proba is not None:
                        # shape guard: if stale/wrong, ignore and fallback
                        try:
                            if int(proba.shape[0]) != int(n_win):
                                proba = None
                        except Exception:
                            proba = None
                        # consume once
                        self._deep_subproc_proba = None

                    if proba is None:
                        bs = int(params.get("transformer_batch_size", 128))
                        free_gb_pred = psutil.virtual_memory().available / (1024**3)
                        floor_gb = float(os.environ.get("MLB_MIN_FREE_GB", "2.5"))
                        force_chunk = bool(int(os.environ.get("MLB_CHUNK_SEQ_PRED", "0")))
                        use_chunk = force_chunk or (free_gb_pred < floor_gb)

                        if use_chunk:
                            chunk_windows = int(os.environ.get("MLB_PRED_CHUNK_WINDOWS", "4096"))
                            print(f"[INFO] Low-RAM predict: chunking windows (chunk_windows={chunk_windows}).")
                            proba = self._predict_seq_windows_chunked(
                                self.model, X2d_test, win=win, batch_size=bs, chunk_windows=chunk_windows
                            )
                        else:
                            Xv = sliding_window_view(X2d_test, window_shape=win, axis=0)
                            proba = self.model.predict(Xv, verbose=0, batch_size=bs)
                            
                    proba = sanitize_proba(proba)

                    # apply learned temperature if available
                    if hasattr(self, "_deep_temp_T"):
                        try:
                            proba = _apply_temperature_to_proba(proba, float(self._deep_temp_T))
                        except Exception:
                            pass

                    if proba.shape[1] >= 3:
                        p_short = proba[:, 0]
                        p_long  = proba[:, 2]
                        max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                        decoded_raw = np.asarray(
                            np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0)),
                            dtype=np.int8
                        )
                        raw_classes = np.asarray(
                            np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)),
                            dtype=np.int8
                        )
                    else:
                        raw = np.argmax(proba, axis=1)
                        max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                        raw_classes = np.asarray(raw, dtype=np.int8)
                        decoded_raw = np.asarray(np.where(raw_classes == 1, 1, -1), dtype=np.int8)



                    # --- Edge-vs-Cost gating (dynamic; align on window-end idx) ---
                    cfg_f = getattr(self, "features_config", {}) or {}
                    base_thr = float(self._resolve_conf_thr(confidence_threshold))
                    self._last_conf_thr_init = float(cfg_f.get("confidence_threshold", confidence_threshold))

                    _cfg_cost = dict(cfg_f)
                    if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                        _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                    _td_cost = test_data[["returns"]] if (test_data is not None and "returns" in test_data.columns) else test_data.loc[:, []]
                    _cost_src = self._ensure_cost_columns(_td_cost, _cfg_cost)


                    # Build drivers over all test rows, then sample by idx_end
                    _all_idx = test_data_scaled.index
                    rets_all, sprd_all, slip_all = self._get_cost_arrays_aligned(_cost_src, _all_idx)

                    # -------------------------------
                    # Causal volatility scaling patch
                    # -------------------------------
                    vol_w = int(cfg_f.get("vol_window_bars", _PC["vol_window_bars"]))

                    # 1) Compute volatility scale + denom floor from TRAIN (causal)
                    rv_m_tr, rv_s_tr, den_floor_tr = float("nan"), float("nan"), float("nan")
                    try:
                        _tr_cost = train_data[["returns"]] if (train_data is not None and "returns" in train_data.columns) else train_data.loc[:, []]
                        _cost_train = self._ensure_cost_columns(_tr_cost, _cfg_cost)
                        rets_tr = _cost_train["returns"].astype(float)
                        rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                        rv_m_tr, rv_s_tr = float(np.nanmean(rv_tr)), float(np.nanstd(rv_tr))
                        _pos = rv_tr[np.isfinite(rv_tr) & (rv_tr > 0)]
                        if _pos.size > 0:
                            den_floor_tr = float(np.nanmedian(_pos))
                    except Exception:
                        pass

                    # 2) Compute realized vol on TEST, but reuse TRAIN stats
                    rv_all = realized_vol(rets_all, window=vol_w).to_numpy(dtype=np.float32)

                    # Final z-score: bar-by-bar TEST vol vs TRAIN-based scale
                    if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                        vol_z_all = (rv_all - rv_m_tr) / rv_s_tr
                    else:
                        # Degenerate train stats -> neutral vol term (no hidden test-fit fallback).
                        vol_z_all = np.zeros_like(rv_all, dtype=np.float32)

                    # Normalised spread vs vol: use TRAIN-derived floor (or constant) -- never test-wide median.
                    den_floor = den_floor_tr if (np.isfinite(den_floor_tr) and den_floor_tr > 1e-8) else 1e-6
                    den_all = np.where(rv_all > 1e-8, rv_all, den_floor).astype(np.float32)
                    spread_norm_all = np.divide(
                        sprd_all,
                        den_all,
                        out=np.zeros_like(sprd_all, dtype=np.float32),
                        where=np.isfinite(den_all),
                    )

                    # alphabetagamma coefficients unchanged
                    a = float(cfg_f.get("alpha_vol_z", 0.01))
                    b = float(cfg_f.get("beta_spread_norm", _PC["beta_spread_norm"]))
                    g = float(cfg_f.get("gamma_slip_norm", _PC["gamma_slip_norm"]))
                    slip_norm_bps = float(cfg_f.get("slip_norm_bps", _PC["slip_norm_bps"]))
                    max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))

                    thr_full = (
                        base_thr
                        + a * vol_z_all
                        + b * spread_norm_all
                        + g * (slip_all / max(1e-9, slip_norm_bps))
                    )
                    thr_full = np.clip(thr_full, 0.0, max_conf_thr).astype(np.float32)

                    # Align thresholds to window ends / eval bars (unchanged)
                    try:
                        idx_arr = np.asarray(idx_end, dtype=int)
                    except NameError:
                        idx_arr = np.arange(len(test_data_scaled), dtype=int)
                        idx_end = idx_arr.tolist()

                    thr_vec = thr_full[idx_arr]

                    if self._is_debug():
                        print(
                            f"[Gate[OK]] Dynamic alphabetagamma active | base={base_thr:.3f} alpha={a:.3f} beta={b:.3f} gamma={g:.3f} "
                            f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}"
                        )

                    # ------------------------------------------------------------
                    # IMPORTANT: define the TRUE evaluation universe for seq models
                    # (window-ends that land on eligible eval bars AND are >= anchor)
                    # ------------------------------------------------------------
                    idx_end_arr = np.asarray(idx_end, dtype=int)
                    keep_win = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                    # eval_mask is bar-level (session/warmup/eligibility); map to window ends
                    try:
                        _em = np.asarray(eval_mask, dtype=bool)
                        if _em.size == len(test_data_scaled):
                            keep_win &= _em[idx_end_arr]
                    except Exception:
                        pass
                    _eval_idx = np.flatnonzero(keep_win)

                    # --- Soft coverage-drift nudge (regime-aware, non-forcing) ---
                    try:
                        tgt   = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))
                        band  = float(cfg_f.get("runtime_active_band_margin", _PC["runtime_active_band_margin"]))
                        win_k = int(cfg_f.get("runtime_coverage_window", 96))
                        step  = float(cfg_f.get("runtime_conf_nudge", 0.01))

                        # Stability guard: if step is too large relative to the band, the nudge can ping-pong.
                        # Enforce step <= 0.5 * band (monotone adjustment within a symmetric deadband).
                        try:
                            band = float(band)
                            step = float(step)
                        except Exception:
                            band, step = 0.0, 0.0
                        band = max(0.0, band)
                        step = abs(step)
                        if band > 0.0 and step > 0.5 * band:
                            _step_old = step
                            step = max(1e-6, 0.5 * band)
                            try:
                                if bool(getattr(self, "debug", False)):
                                    log_print(
                                        f"[CoverageNudge][Stability] step={_step_old:.5f} > 0.5*band={(0.5*band):.5f}; clamped to {step:.5f}",
                                        level="COMPACT",
                                    )
                            except Exception:
                                pass

                        n = int(_eval_idx.size)
                        
                        # --- Rolling-quantile cap (prevents "bunched confidence" => near-zero trades) ---
                        # If current activity is below the lower band, cap thresholds DOWN to the rolling
                        # (1 - tgt) quantile of past confidences so "top tgt%" remains achievable.
                        _low = max(0.0, tgt - band)
                        allow_qcap = bool(cfg_f.get("runtime_allow_rolling_qcap", True))
                        if allow_qcap and win_k > 1 and n >= win_k:
                            try:
                                _dr = decoded_raw[_eval_idx]
                                _mc = max_conf[_eval_idx]
                                _tv = thr_vec[_eval_idx]
                                _act0 = ((_dr != 0) & (_mc >= _tv)).astype(np.float32)
                                if float(np.nanmean(_act0)) < _low:
                                    _q = (
                                        pd.Series(_mc)
                                        .rolling(win_k, min_periods=win_k)
                                        .quantile(1.0 - tgt)
                                        .shift(1)  # causal: use past only
                                        .to_numpy(dtype=np.float32)
                                    )
                                    _m = np.isfinite(_q)
                                    if _m.any():
                                        # cap thresholds only on eval windows
                                        _tv[_m] = np.minimum(_tv[_m], _q[_m])
                                        thr_vec[_eval_idx] = _tv
                                        if self._is_debug():
                                            print(
                                                f"[Gate[OK]] Rolling-quantile cap active | q={1.0 - tgt:.3f} "
                                                f"win={win_k} | thr_med={float(np.nanmedian(_tv)):.3f}"
                                            )
                            except Exception:
                                pass
                        
                        # preliminary decisions with alphabetagamma only (causal)
                        if n > 0:
                            _dr = decoded_raw[_eval_idx].copy()
                            _mc = max_conf[_eval_idx]
                            _tv = thr_vec[_eval_idx]
                            _mask0 = (_mc < _tv)
                            _dr[_mask0] = 0
                            _act = (_dr != 0).astype(np.float32)
                        else:
                            _act = np.asarray([], dtype=np.float32)
                        if win_k > 1 and n > 0:
                            _cs = np.cumsum(np.insert(_act, 0, 0.0))
                            _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                            _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                            _low, _high = max(0.0, tgt - band), min(1.0, tgt + band)
                            _drift = np.where(_roll < _low, -step, np.where(_roll > _high, step, 0.0)).astype(np.float32)
                            _drift = np.nan_to_num(_drift, nan=0.0)
                            min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                            max_conf_thr = float((cfg_f or {}).get("max_conf_thr", 0.90))

                            _drift = np.clip(_drift, -abs(step), abs(step)).astype(np.float32)
                            thr_vec[_eval_idx] = np.clip(
                                thr_vec[_eval_idx] + _drift,
                                min_conf_thr,
                                max_conf_thr
                            ).astype(np.float32)
                            # ----------------------

                    except Exception as _e:
                        print(f"[Gate] Coverage nudge skipped (transformer-seq): {_e}")
                    self._last_conf_thr_used = float(np.nanmedian(thr_vec[_eval_idx])) if _eval_idx.size > 0 else float(np.nanmedian(thr_vec))
                
                    # (D1) Snapshot distributions for post-mortem debug (no look-ahead; uses eval arrays only)
                    try:
                        _nq = int(min(len(max_conf), len(thr_vec)))
                        if _eval_idx.size > 0:
                            _mc = max_conf[_eval_idx]
                            _tv = thr_vec[_eval_idx]
                            self._last_lstm_conf_q = tuple(np.nanquantile(_mc, [0.50, 0.75, 0.90]).astype(float).tolist())
                            self._last_lstm_thr_q  = tuple(np.nanquantile(_tv, [0.50, 0.75, 0.90]).astype(float).tolist())
                    except Exception:
                        pass

                    # Apply confidence filter ONLY on eligible eval windows (others forced flat)
                    final_preds = np.zeros_like(decoded_raw, dtype=int)
                    if _eval_idx.size > 0:
                        final_preds[_eval_idx] = decoded_raw[_eval_idx]
                        _mask = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                        final_preds[_eval_idx[_mask]] = 0

                    # No-trade month -> invalid fold (let CV aggregator/Optuna handle)
                
                    if self._is_debug():
                        try:
                            _rawc   = pd.Series(decoded_raw[keep_win]).value_counts().to_dict()
                            _finalc = pd.Series(final_preds[keep_win]).value_counts().to_dict()
                            print(f"[DeepGate][Dist][transformer-seq] raw={_rawc} | final={_finalc}")
                        except Exception:
                            pass
                    if (final_preds != 0).sum() == 0:
                        return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:transformer_seq_no_trades")
                
                    # Build evaluation frame aligned to end-of-window indices
                    idx_end_kept = idx_end_arr[keep_win]
                    keep = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                    idx_end_kept = idx_end_arr[keep]
                    if idx_end_kept.size == 0:
                        return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:transformer_seq_no_eval_windows")
                    eval_index = test_data_scaled.index[idx_end_kept]
                    final_preds_kept = np.asarray(final_preds, dtype=int)[keep_win]

                    test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                    test_data_for_eval["pred"] = final_preds_kept  # CLEANUP: keep-index aligned

                    # Stats for CV / summaries (normalize to -1/0/+1 so the table isn't garbage)
                    try:
                        raw_counts = _norm_class_counts(
                            pd.Series(decoded_raw[keep_win]).value_counts().to_dict()
                        )
                        final_counts = _norm_class_counts(
                            pd.Series(final_preds[keep_win]).value_counts().to_dict()
                        )
                        self._last_class_dists = {"raw": raw_counts, "final": final_counts}
                        self._last_conf_stats_label = str(model_type)
                        self._last_conf_stats_max_conf = np.asarray(max_conf[keep_win], dtype=np.float32)
                    except Exception:
                        self._last_class_dists = {"raw": {}, "final": {}}
                        self._last_conf_stats_label = str(model_type)
                        self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)

                elif model_type == "lstm":
                    lstm_use_seq = bool(params.get("lstm_use_seq_windows", True))


                    # ==================================================================
                    #  LSTM - SEQ MODE (sliding windows + idx_end)
                    # ==================================================================
                    if lstm_use_seq:
                        win = max(2, int(lags_eff))

                        X2d_test = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
                        if X2d_test.shape[0] < win:
                            print("[ERR] [ABORT] Test set shorter than window size for LSTM (seq mode).")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:lstm_seq_test_too_short")
                        n_win = int(X2d_test.shape[0] - win + 1)
                        idx_end = np.arange(win - 1, win - 1 + n_win, 1, dtype=int)

                        bs = int(params.get("lstm_batch_size", 128))
                        free_gb_pred = psutil.virtual_memory().available / (1024**3)
                        floor_gb = float(os.environ.get("MLB_MIN_FREE_GB", "2.5"))
                        force_chunk = bool(int(os.environ.get("MLB_CHUNK_SEQ_PRED", "0")))
                        use_chunk = force_chunk or (free_gb_pred < floor_gb)

                        if use_chunk:
                            chunk_windows = int(os.environ.get("MLB_PRED_CHUNK_WINDOWS", "4096"))
                            print(f"[INFO] Low-RAM predict: chunking windows (chunk_windows={chunk_windows}).")
                            proba = self._predict_seq_windows_chunked(
                                self.model, X2d_test, win=win, batch_size=bs, chunk_windows=chunk_windows
                            )
                        else:
                            Xv = sliding_window_view(X2d_test, window_shape=win, axis=0)
                            proba = self.model.predict(Xv, verbose=0, batch_size=bs)

                        proba = sanitize_proba(proba)

                        # apply learned temperature if available
                        if hasattr(self, "_deep_temp_T"):
                            try:
                                proba = _apply_temperature_to_proba(proba, float(self._deep_temp_T))
                            except Exception:
                                pass

                        if proba.shape[1] >= 3:
                            p_short = proba[:, 0]
                            p_long  = proba[:, 2]
                            max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                        else:
                            raw = np.argmax(proba, axis=1)
                            max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                            raw_classes = np.asarray(raw, dtype=int)
                            decoded_raw = np.where(raw_classes == 1, 1, -1)

                        # --- Edge-vs-Cost gating (dynamic; align on window-end idx) ---
                        _global_f = (CLASS_DEFAULTS.get("features", {}) if "CLASS_DEFAULTS" in globals() else {})
                        _cfg_raw  = (getattr(self, "features_config", {}) or {})
                        cfg_f     = {**_global_f, **_cfg_raw}

                        # 1) Base confidence threshold (from CV / coverage fit / user override)
                        base_thr = float(self._resolve_conf_thr(confidence_threshold))
                        self._last_conf_thr_init = float(
                            cfg_f.get("confidence_threshold", confidence_threshold)
                        )

                        # 2) Build cost & volatility drivers on the full test index
                        _cfg_cost = dict(cfg_f)
                        if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                            _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                        _td_cost = test_data[["returns"]] if (test_data is not None and "returns" in test_data.columns) else test_data.loc[:, []]
                        _cost_src = self._ensure_cost_columns(_td_cost, _cfg_cost)

                        _all_idx = test_data_scaled.index
                        rets_all, sprd_all, slip_all = self._get_cost_arrays_aligned(_cost_src, _all_idx)

                        # Volatility z-score (realized vol over window)
                        vol_w = int(cfg_f.get(
                            "vol_window_bars",
                            _global_f.get("vol_window_bars", _PC["vol_window_bars"])
                        ))

                        # --- Train-anchored vol scaling (avoid ex-post test-month stats) ---
                        rv_m_tr, rv_s_tr, den_floor_tr = np.nan, np.nan, np.nan
                        try:
                            rets_tr = train_data_scaled["returns"].astype(float)
                            rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                            rv_m_tr = float(np.nanmean(rv_tr))
                            rv_s_tr = float(np.nanstd(rv_tr))
                            _pos = rv_tr[np.isfinite(rv_tr) & (rv_tr > 0)]
                            if _pos.size > 0:
                                den_floor_tr = float(np.nanmedian(_pos))
                        except Exception:
                            pass

                        rv_all = realized_vol(rets_all, window=vol_w).to_numpy(dtype=np.float32)
                        if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                            vol_z_all = (rv_all - rv_m_tr) / rv_s_tr
                        else:
                            vol_z_all = np.zeros_like(rv_all, dtype=np.float32)

                        den_floor = den_floor_tr if (np.isfinite(den_floor_tr) and den_floor_tr > 1e-8) else 1e-6
                        den_all = np.where(rv_all > 1e-8, rv_all, den_floor).astype(np.float32)
                        spread_norm_all = np.divide(
                            sprd_all,
                            den_all,
                            out=np.zeros_like(sprd_all, dtype=np.float32),
                            where=np.isfinite(den_all)
                        )

                        # 3) alphabetagamma: volatility-, spread-, and slippage-aware threshold bump
                        a = float(cfg_f.get("alpha_vol_z", _global_f.get("alpha_vol_z", 0.004)))
                        b = float(cfg_f.get("beta_spread_norm", _global_f.get("beta_spread_norm", _PC["beta_spread_norm"])))
                        g = float(cfg_f.get("gamma_slip_norm", _global_f.get("gamma_slip_norm", _PC["gamma_slip_norm"])))
                        slip_norm_bps = float(cfg_f.get("slip_norm_bps", _global_f.get("slip_norm_bps", _PC["slip_norm_bps"])))
                        min_slip_norm_bps = float(cfg_f.get("min_slip_norm_bps", _global_f.get("min_slip_norm_bps", _PC["min_slip_norm_bps"])))
                        slip_norm_bps = max(slip_norm_bps, min_slip_norm_bps, 1e-6)

                        vol_z_cap = float(cfg_f.get("vol_z_cap", _global_f.get("vol_z_cap", 6.0)))
                        spread_norm_cap = float(cfg_f.get("spread_norm_cap", _global_f.get("spread_norm_cap", 5.0)))
                        slip_ratio_cap = float(cfg_f.get("slip_ratio_cap", _global_f.get("slip_ratio_cap", 6.0)))
                        max_conf_thr = float(cfg_f.get("max_conf_thr", _global_f.get("max_conf_thr", 0.90)))
                        
                        try:
                            idx_arr = np.asarray(idx_end, dtype=int)
                        except NameError:
                            idx_arr = np.arange(len(test_data_scaled), dtype=int)
                            idx_end = idx_arr.tolist()

                        tgt_ar = cfg_f.get("target_active_rate", cfg_f.get("target_coverage", None))
                        base_thr_vec = float(base_thr)  # default: scalar
                        if tgt_ar is not None:
                            try:
                                W = int(cfg_f.get(
                                    "coverage_rolling_window",
                                    _global_f.get("coverage_rolling_window", 48)
                                ))
                                _minp = max(10, W // 3)
                                # Build full-length confidence aligned to bar index, then roll causally.
                                conf_full = np.full(len(test_data_scaled), np.nan, dtype=np.float32)
                                _nfill = int(min(len(max_conf), len(idx_arr)))
                                if _nfill > 0:
                                    conf_full[idx_arr[:_nfill]] = np.asarray(max_conf[:_nfill], dtype=np.float32)
                                conf_s = pd.Series(conf_full, index=test_data_scaled.index).astype(float)
                                thr_roll = (
                                    conf_s.rolling(W, min_periods=_minp)
                                          .quantile(1.0 - float(tgt_ar))
                                          .shift(1)
                                )
                                base_thr_vec = thr_roll.fillna(float(base_thr)).to_numpy(dtype=np.float32)
                                if self._is_debug():
                                    print(
                                        f"[Gate[OK]][RollingQuantile] W={W} target={float(tgt_ar):.3f} "
                                        f"base_med={float(np.nanmedian(base_thr_vec)):.3f}"
                                    )
                            except Exception as _e:
                                base_thr_vec = float(base_thr)


                        slip_ratio = np.clip(slip_all / max(1e-9, slip_norm_bps), 0.0, slip_ratio_cap)
                        vol_z_all = np.clip(vol_z_all, -vol_z_cap, vol_z_cap)
                        spread_norm_all = np.clip(spread_norm_all, 0.0, spread_norm_cap)

                        thr_full = np.clip(
                            base_thr_vec
                            + a * vol_z_all
                            + b * spread_norm_all
                            + g * slip_ratio,
                            0.0, max_conf_thr
                        ).astype(np.float32)

                        thr_vec = thr_full[idx_arr]

                        print(
                            "[Gate[OK]] Dynamic alphabetagamma active | "
                            f"base={base_thr:.3f} alpha={a:.3f} beta={b:.3f} gamma={g:.3f} "
                            f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}"
                        )
                        
                        # ------------------------------------------------------------
                        # IMPORTANT: define eval-universe for seq models (LSTM-seq)
                        # windows whose END index is eligible AND on/after anchor
                        # ------------------------------------------------------------
                        idx_end_arr = np.asarray(idx_end, dtype=int)
                        keep_win = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                        try:
                            _em = np.asarray(eval_mask, dtype=bool)
                            if _em.size == len(test_data_scaled):
                                keep_win &= _em[idx_end_arr]
                        except Exception:
                            pass
                        _eval_idx = np.flatnonzero(keep_win)

                        # 4) Soft coverage-drift nudge (regime-aware, non-forcing)
                        try:
                            # Anchor: Optuna-tuned or config-provided target_active_rate / target_coverage
                            tgt = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))

                            band = float(cfg_f.get(
                                "runtime_active_band_margin",
                                _global_f.get("runtime_active_band_margin", _PC["runtime_active_band_margin"])
                            ))
                            win_k = int(cfg_f.get(
                                "runtime_coverage_window",
                                _global_f.get("runtime_coverage_window", 96)
                            ))
                            step = float(cfg_f.get(
                                "runtime_conf_nudge",
                                _global_f.get("runtime_conf_nudge", 0.01)
                            ))


                            # Stabilize runtime nudge params (avoid flip-flop when step > band/2)
                            band, step = self._sanitize_runtime_coverage_nudge(band, step, ctx="runtime")

                            # ------------------------------------------------------------
                            # IMPORTANT: restrict nudging + filtering to TRUE eval universe
                            # windows whose END index is eligible AND on/after anchor
                            # ------------------------------------------------------------
                            idx_end_arr = np.asarray(idx_end, dtype=int)
                            keep_win = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                            try:
                                _em = np.asarray(eval_mask, dtype=bool)
                                if _em.size == len(test_data_scaled):
                                    keep_win &= _em[idx_end_arr]
                            except Exception:
                                pass
                            _eval_idx = np.flatnonzero(keep_win)

                            n = int(_eval_idx.size)
                            if n > 0 and win_k > 1 and step > 0.0:
                                # Decisions using alphabetagamma-threshold only (on eval windows)
                                _dr = decoded_raw[_eval_idx].copy()
                                _mc = max_conf[_eval_idx]
                                _tv = thr_vec[_eval_idx]
                                _dr[_mc < _tv] = 0
                                _act = (_dr != 0).astype(np.float32)

                                # Rolling active rate (simple moving average) on eval windows
                                roll = np.full_like(_act, np.nan, dtype=np.float32)
                                if n >= win_k:
                                    csum = np.cumsum(_act, dtype=float)
                                    roll[win_k - 1:] = (
                                        csum[win_k - 1:] -
                                        np.concatenate(([0.0], csum[:-win_k]))
                                    ) / float(win_k)

                                low, high = tgt - band, tgt + band
                                low = max(0.0, low)
                                high = min(1.0, high)

                                sel = np.isfinite(roll)
                                below = sel & (roll < low)
                                above = sel & (roll > high)

                                drift = np.zeros(n, dtype=np.float32)
                                drift[below] = -step   # too quiet -> lower threshold -> more trades
                                drift[above] = step    # too active -> raise threshold -> fewer trades

                                min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                                # Apply drift to thresholds on eval windows (THIS WAS MISSING BEFORE)
                                thr_vec[_eval_idx] = np.clip(
                                    thr_vec[_eval_idx] + drift,
                                    min_conf_thr,
                                    max_conf_thr
                                ).astype(np.float32)

                            self._last_conf_thr_used = float(np.nanmedian(thr_vec[_eval_idx])) if _eval_idx.size > 0 else float(np.nanmedian(thr_vec))
                                                    
                            if self._is_debug():
                                print(
                                    "[Gate[OK]] Coverage nudge active | "
                                    f"target={tgt:.2f} band=+/-{band:.2f} "
                                    f"step={step:.3f} | median_used={self._last_conf_thr_used:.3f}"
                                )

                        except Exception as _ee:
                            # Fail-safe: keep alphabetagamma-only thr if nudging breaks
                            self._last_conf_thr_used = float(np.nanmedian(thr_vec))
                            print(f"[Gate] Coverage nudge skipped: {type(_ee).__name__}: {_ee}")


                        # (D1) Snapshot distributions for post-mortem debug (no look-ahead; uses eval arrays only)
                        try:
                            if _eval_idx.size > 0:
                                _mc = max_conf[_eval_idx]
                                _tv = thr_vec[_eval_idx]
                                self._last_lstm_conf_q = tuple(np.nanquantile(_mc, [0.50, 0.75, 0.90]).astype(float).tolist())
                                self._last_lstm_thr_q  = tuple(np.nanquantile(_tv, [0.50, 0.75, 0.90]).astype(float).tolist())
                        except Exception:
                            pass

                        # 5) Apply gating to predictions (seq mode) -- only on eval windows, force flat elsewhere
                        final_preds = np.zeros_like(decoded_raw, dtype=int)
                        if _eval_idx.size > 0:
                            final_preds[_eval_idx] = decoded_raw[_eval_idx]
                            _mask = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                            final_preds[_eval_idx[_mask]] = 0

                        # No-trade month -> invalid fold (let CV aggregator/Optuna handle)
                        if self._is_debug():
                            try:
                                _rawc   = pd.Series(decoded_raw[keep_win]).value_counts().to_dict()
                                _finalc = pd.Series(final_preds[keep_win]).value_counts().to_dict()
                                print(f"[DeepGate][Dist][lstm-seq] raw={_rawc} | final={_finalc}")
                            except Exception:
                                pass

                        if (final_preds != 0).sum() == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:lstm_seq_no_trades")

                        # keep only windows ending on/after the first eval bar AND eligible by eval_mask
                        idx_end_kept = idx_end_arr[keep_win]
                        if idx_end_kept.size == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:lstm_seq_no_eval_windows")

                        eval_index = test_data_scaled.index[idx_end_kept]
                        final_preds_kept = np.asarray(final_preds, dtype=int)[keep_win]

                        test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                        test_data_for_eval["pred"] = pd.Series(final_preds_kept, index=eval_index).values

                        # Stats for CV / summaries (normalize to -1/0/+1 so fold table prints correctly)
                        try:
                            raw_counts = _norm_class_counts(
                                pd.Series(decoded_raw[keep_win]).value_counts(dropna=False).to_dict()
                            )
                            final_counts = _norm_class_counts(
                                pd.Series(final_preds[keep_win]).value_counts(dropna=False).to_dict()
                            )

                            self._last_class_dists = {
                                "raw": raw_counts,
                                "final": final_counts,
                            }
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray(max_conf[keep_win], dtype=np.float32)
                        except Exception:
                            self._last_class_dists = {"raw": {}, "final": {}}
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)


                    # ==================================================================
                    #  LSTM - 3D-FEED MODE (flat windows, no idx_end)
                    # ==================================================================
                    else:
                        X_test_3d = X_test.astype(np.float32).reshape(
                            (X_test.shape[0], X_test.shape[1], 1)
                        )
                        proba = self.model.predict(
                            X_test_3d, verbose=0,
                            batch_size=int(params.get("lstm_batch_size", 128))
                        )
                        proba = sanitize_proba(proba)

                        if proba.shape[1] >= 3:
                            p_short = proba[:, 0]
                            p_long  = proba[:, 2]
                            max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                        else:
                            raw = np.argmax(proba, axis=1)
                            max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                            raw_classes = np.asarray(raw, dtype=int)
                            decoded_raw = np.where(raw_classes == 1, 1, -1)

                        # non-seq branch retains backoff/coverage logic unchanged
                        cfg_f  = getattr(self, "features_config", {}) or {}
                        cv_mode = bool(getattr(self, "_in_optuna_cv", False))

                        # Use unified resolver so:
                        #  - coverage-based threshold is respected (if fitted),
                        #  - LSTM gets its per-model relaxation (lstm_conf_relax / floor).
                        base_conf = float(cfg_f.get("confidence_threshold", confidence_threshold))
                        conf0     = float(self._resolve_conf_thr(base_conf))

                        q75, q50, q90 = np.quantile(max_conf, [0.75, 0.50, 0.90])

                        # (D1) Snapshot confidence distribution quantiles (3D has no thr_vec)
                        try:
                            self._last_lstm_conf_q = (float(q50), float(q75), float(q90))
                        except Exception:
                            pass

                        # --- Debug: threshold vs confidence distribution (eval/WFO only) ---
                        if not cv_mode:
                            n_conf0 = int((max_conf >= conf0).sum())
                            _mtag = getattr(self, "model_type", str(model_type))
                            print(
                                f"[Deep3D][GateDebug] model={_mtag} "
                                f"| conf0={conf0:.3f} q75={q75:.3f} q90={q90:.3f} "
                                f"| n_conf0={n_conf0}/{len(max_conf)}"
                            )

                        allow_cv_backoff   = bool(cfg_f.get("allow_conf_backoff_cv", False))
                        allow_eval_backoff = bool(cfg_f.get("allow_conf_backoff_eval", False))
                        floor_cv   = float(cfg_f.get("conf_backoff_floor_cv", 0.33))
                        floor_eval = float(cfg_f.get("conf_backoff_floor_eval", 0.33))

                        candidates = [conf0]
                        if cv_mode:
                            if allow_cv_backoff:
                                candidates = [conf0, min(conf0, q90), q75]
                                candidates = [max(floor_cv, c) for c in candidates]
                        else:
                            if allow_eval_backoff:
                                candidates = [conf0, min(conf0, q90), q75, 0.33, 0.25]
                                candidates = [max(floor_eval, c) for c in candidates]

                        _seen = set()
                        candidates = [
                            x for x in candidates
                            if (round(x, 6) not in _seen and not _seen.add(round(x, 6)))
                        ]

                        final_preds = None
                        self._last_conf_thr_init = float(conf0)
                        self._last_max_conf_q75  = float(q75)
                        self._last_max_conf_q90  = float(q90)
                        self._last_conf_backoff_steps = 0

                        for thr in candidates:
                            preds_try = np.asarray(decoded_raw, dtype=np.int8).copy()

                            preds_try[max_conf < thr] = 0
                            if np.count_nonzero(preds_try) > 0:
                                if abs(thr - conf0) > 1e-9:
                                    print(
                                        f"[WARN] Confidence threshold relaxed {conf0:.3f} -> "
                                        f"{thr:.3f} to avoid 0 trades."
                                    )
                                    self._last_conf_backoff_steps = 1
                                final_preds = preds_try
                                self._last_conf_thr_used = float(thr)
                                try:
                                    # 3D uses a scalar threshold; store as a "degenerate" quantile tuple
                                    self._last_lstm_thr_q = (float(thr), float(thr), float(thr))
                                except Exception:
                                    pass
                            
                            
                                break
                        
                        # HARD guard: thin-trades fallback must never be used in CV.
                        if cv_mode and bool(cfg_f.get("allow_thin_trades_fallback", False)):
                            raise RuntimeError(
                                "CV thin-trades fallback is disabled: remove allow_thin_trades_fallback "
                                "(CV must not invent trades that real months will not take)."
                            )


                        if final_preds is None or np.count_nonzero(final_preds) == 0:
                            # Penalize no-trade configs during Optuna/CV (helps search),
                            # but NEVER poison real_trading_simulation with NaNs.
                            in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                            in_real = bool(getattr(self, "_in_real_sim", False))
                            if in_cv and not in_real:
                                if self._is_debug():
                                    print("[ERR] No trades predicted after filtering -- penalizing this parameter set.")
                                    try:
                                        _rawc = pd.Series(decoded_raw).value_counts().to_dict()
                                        _finalc = {} if final_preds is None else pd.Series(final_preds).value_counts().to_dict()
                                        print(f"[DeepGate][Dist][deep3d] raw={_rawc} | final={_finalc}")
                                    except Exception:
                                        pass
                                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:no_trades_cv")
                            if self._is_debug():
                                print("[FLAT] No trades predicted after filtering -- keeping 0-trade evaluation (real-sim / non-CV).")
                                
                        # FAIL-SAFE: if still None here, force HOLD vector so we don't crash on eval_mask slicing.
                        if final_preds is None:
                            final_preds = np.zeros_like(decoded_raw, dtype=int)
                            try:
                                self._last_conf_thr_used = float(conf0)
                            except Exception:
                                pass
                    
                        # --- Debug: final trade count after eval_mask (eval/WFO only) ---
                        if not cv_mode:
                            n_trades = int(np.count_nonzero(final_preds))
                            _mtag = getattr(self, "model_type", str(model_type))
                            print(
                                f"[Deep3D][GateDebug] model={_mtag} "
                                f"| trades_after_mask={n_trades}"
                            )
 

                        # 3D path uses eval_mask (already computed outside this block)
                        eval_index = test_data_scaled.index[eval_mask]
                        final_preds = final_preds[eval_mask]

                        test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                        test_data_for_eval["pred"] = pd.Series(
                            final_preds, index=eval_index
                        ).values

                        try:
                            raw_counts = pd.Series(raw_classes).value_counts().to_dict()
                            final_counts = pd.Series(final_preds).value_counts().to_dict()

                            self._last_class_dists = {
                                "raw": raw_counts,
                                "final": final_counts,
                            }
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray(max_conf, dtype=np.float32)
                        except Exception:
                            self._last_class_dists = {"raw": {}, "final": {}}
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)

                else:  # CNN
                    cnn_use_seq = bool(params.get("cnn_use_seq_windows", True))

                    if cnn_use_seq:
                        win = max(2, int(lags_eff))

                        X2d_test = test_data_scaled[features].to_numpy(dtype=np.float32, copy=False)
                        if X2d_test.shape[0] < win:
                            print("[ERR] [ABORT] Test set shorter than window size for CNN (seq mode).")
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_seq_test_too_short")

                        n_win = int(X2d_test.shape[0] - win + 1)
                        idx_end = np.arange(win - 1, win - 1 + n_win, 1, dtype=int)

                        bs = int(params.get("cnn_batch_size", 128))
                        free_gb_pred = psutil.virtual_memory().available / (1024**3)
                        floor_gb = float(os.environ.get("MLB_MIN_FREE_GB", "2.5"))
                        force_chunk = bool(int(os.environ.get("MLB_CHUNK_SEQ_PRED", "0")))
                        use_chunk = force_chunk or (free_gb_pred < floor_gb)

                        if use_chunk:
                            chunk_windows = int(os.environ.get("MLB_PRED_CHUNK_WINDOWS", "4096"))
                            print(f"[INFO] Low-RAM predict: chunking windows (chunk_windows={chunk_windows}).")
                            proba = self._predict_seq_windows_chunked(
                                self.model, X2d_test, win=win, batch_size=bs, chunk_windows=chunk_windows
                            )
                        else:
                            Xv = sliding_window_view(X2d_test, window_shape=win, axis=0)
                            proba = self.model.predict(Xv, verbose=0, batch_size=bs)

                        proba = sanitize_proba(proba)
                        if proba.shape[1] >= 3:
                            p_short = proba[:, 0]
                            p_long  = proba[:, 2]
                            max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                        else:
                            raw = np.argmax(proba, axis=1)
                            max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                            raw_classes = np.asarray(raw, dtype=int)
                            decoded_raw = np.where(raw_classes == 1, 1, -1)
                            
                            
                        # --- Edge-vs-Cost gating (dynamic; align on window-end idx) ---
                        cfg_f = getattr(self, "features_config", {}) or {}
                        base_thr = float(self._resolve_conf_thr(confidence_threshold))
                        self._last_conf_thr_init = float(cfg_f.get("confidence_threshold", confidence_threshold))

                        _cfg_cost = dict(cfg_f)
                        if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                            _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                        _cost_src = self._ensure_cost_columns(test_data, _cfg_cost)

                        _all_idx = test_data_scaled.index
                        rets_all, sprd_all, slip_all = self._get_cost_arrays_aligned(_cost_src, _all_idx)

                        vol_w = int(cfg_f.get("vol_window_bars", _PC["vol_window_bars"]))
                    
                        rv_m_tr, rv_s_tr, den_floor_tr = np.nan, np.nan, np.nan
                        try:
                            rets_tr = train_data_scaled["returns"].astype(float)
                            rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                            rv_m_tr = float(np.nanmean(rv_tr))
                            rv_s_tr = float(np.nanstd(rv_tr))
                            _pos = rv_tr[np.isfinite(rv_tr) & (rv_tr > 0)]
                            if _pos.size > 0:
                                den_floor_tr = float(np.nanmedian(_pos))
                        except Exception:
                            pass

                        rv_all = realized_vol(rets_all, window=vol_w).to_numpy(dtype=np.float32)
                        if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                            vol_z_all = (rv_all - rv_m_tr) / rv_s_tr
                        else:
                            vol_z_all = np.zeros_like(rv_all, dtype=np.float32)

                        den_floor = den_floor_tr if (np.isfinite(den_floor_tr) and den_floor_tr > 1e-8) else 1e-6
                        den_all = np.where(rv_all > 1e-8, rv_all, den_floor).astype(np.float32)
                        spread_norm_all = np.divide(sprd_all, den_all, out=np.zeros_like(sprd_all, dtype=np.float32), where=np.isfinite(den_all))

                        # Dynamic alphabetagamma coefficients (small by default) and slippage scaling in bps
                        a = float(cfg_f.get("alpha_vol_z", 0.01))
                        b = float(cfg_f.get("beta_spread_norm", _PC["beta_spread_norm"]))
                        g = float(cfg_f.get("gamma_slip_norm", _PC["gamma_slip_norm"]))
                        slip_norm_bps = float(cfg_f.get("slip_norm_bps", _PC["slip_norm_bps"]))
                        max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))
                    
                        thr_full = (
                            base_thr
                            + a * vol_z_all
                            + b * spread_norm_all
                            + g * (slip_all / max(1e-9, slip_norm_bps))
                        )
                        thr_full = np.clip(thr_full, 0.0, max_conf_thr).astype(np.float32)

                        # [OK] Align thresholds with window-end indices (seq) or fallback to full rows
                        try:
                            idx_arr = np.asarray(idx_end, dtype=int)
                        except NameError:
                            idx_arr = np.arange(len(test_data_scaled), dtype=int)
                            idx_end = idx_arr.tolist()

                        thr_vec = thr_full[idx_arr]

                        if self._is_debug():
                            print(
                                f"[Gate[OK]] Dynamic alphabetagamma active | base={base_thr:.3f} alpha={a:.3f} beta={b:.3f} gamma={g:.3f} "
                                f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}"
                            )
                            
                        # ------------------------------------------------------------
                        # IMPORTANT: define eval-universe for seq models (CNN-seq)
                        # windows whose END index is eligible AND on/after anchor
                        # ------------------------------------------------------------
                        idx_end_arr = np.asarray(idx_end, dtype=int)
                        keep_win = (test_data_scaled.index[idx_end_arr] >= self._expected_eval_start)
                        try:
                            _em = np.asarray(eval_mask, dtype=bool)
                            if _em.size == len(test_data_scaled):
                                keep_win &= _em[idx_end_arr]
                        except Exception:
                            pass
                        _eval_idx = np.flatnonzero(keep_win)

                        # --- Soft coverage-drift nudge (regime-aware, non-forcing) ---
                        try:
                            tgt   = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))
                            band  = float(cfg_f.get("runtime_active_band_margin", _PC["runtime_active_band_margin"]))
                            win_k = int(cfg_f.get("runtime_coverage_window", 96))
                            step  = float(cfg_f.get("runtime_conf_nudge", 0.01))

                            # Stability guard: if step is too large relative to the band, the nudge can ping-pong.
                            # Enforce step <= 0.5 * band (monotone adjustment within a symmetric deadband).
                            try:
                                band = float(band)
                                step = float(step)
                            except Exception:
                                band, step = 0.0, 0.0
                            band = max(0.0, band)
                            step = abs(step)
                            if band > 0.0 and step > 0.5 * band:
                                _step_old = step
                                step = max(1e-6, 0.5 * band)
                                try:
                                    if bool(getattr(self, "debug", False)):
                                        log_print(
                                            f"[CoverageNudge][Stability] step={_step_old:.5f} > 0.5*band={(0.5*band):.5f}; clamped to {step:.5f}",
                                            level="COMPACT",
                                        )
                                except Exception:
                                    pass

                            n = int(_eval_idx.size)
                            if n > 0:
                                _pre = decoded_raw[_eval_idx].copy()
                                _mask0 = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                                _pre[_mask0] = 0
                                _act = (_pre != 0).astype(np.float32)
                            if win_k > 1 and n > 0:
                                _cs = np.cumsum(np.insert(_act, 0, 0.0))
                                _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                                _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                                _low, _high = max(0.0, tgt - band), min(1.0, tgt + band)
                                _drift = np.where(_roll < _low, -step, np.where(_roll > _high, step, 0.0)).astype(np.float32)
                                _drift = np.nan_to_num(_drift, nan=0.0)
                                min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                                max_conf_thr = float((cfg_f or {}).get("max_conf_thr", 0.90))

                                _drift = np.clip(_drift, -abs(step), abs(step)).astype(np.float32)
                                thr_vec[_eval_idx] = np.clip(
                                    thr_vec[_eval_idx] + _drift,
                                    min_conf_thr,
                                    max_conf_thr
                                ).astype(np.float32)

                                if self._is_debug():
                                    print(f"[Gate[OK]] Coverage nudge active | target={tgt:.2f} band=+/-{band:.2f} step={step:.3f}")
                        except Exception as _e:
                            print(f"[Gate] Coverage nudge skipped (cnn-seq): {_e}")

                        self._last_conf_thr_used = float(np.nanmedian(thr_vec[_eval_idx])) if _eval_idx.size > 0 else float(np.nanmedian(thr_vec))
                    

                    
                        # (D1) Snapshot distributions for post-mortem debug (no look-ahead; uses eval arrays only)
                        try:
                            if _eval_idx.size > 0:
                                _mc = max_conf[_eval_idx]
                                _tv = thr_vec[_eval_idx]
                                self._last_lstm_conf_q = tuple(np.nanquantile(_mc, [0.50, 0.75, 0.90]).astype(float).tolist())
                                self._last_lstm_thr_q  = tuple(np.nanquantile(_tv, [0.50, 0.75, 0.90]).astype(float).tolist())
                        except Exception:
                            pass
                        
                        # Apply confidence filter ONLY on eligible eval windows (others forced flat)
                        final_preds = np.zeros_like(decoded_raw, dtype=int)
                        if _eval_idx.size > 0:
                            final_preds[_eval_idx] = decoded_raw[_eval_idx]
                            _mask = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                            final_preds[_eval_idx[_mask]] = 0
                    
                        if self._is_debug():
                            try:
                                _rawc   = pd.Series(decoded_raw[keep_win]).value_counts().to_dict()
                                _finalc = pd.Series(final_preds[keep_win]).value_counts().to_dict()
                                print(f"[DeepGate][Dist][cnn-seq] raw={_rawc} | final={_finalc}")
                            except Exception:
                                pass

                        if final_preds is None or (final_preds != 0).sum() == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_seq_no_trades")
                    
                        # keep only windows ending on/after the first eval bar
                        idx_end_kept = idx_end_arr[keep_win]
                        if idx_end_kept.size == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_seq_no_eval_windows")
                    
                        eval_index = test_data_scaled.index[idx_end_kept]
                        final_preds_kept = np.asarray(final_preds, dtype=int)[keep_win]

                        test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                        test_data_for_eval["pred"] = pd.Series(final_preds_kept, index=eval_index).values

                        try:
                            raw_counts = _norm_class_counts(
                                pd.Series(decoded_raw[keep_win]).value_counts(dropna=False).to_dict()
                            )
                            final_counts = _norm_class_counts(
                                pd.Series(final_preds[keep_win]).value_counts(dropna=False).to_dict()
                            )

                            # Store for CV / summary
                            self._last_class_dists = {
                                "raw": raw_counts,
                                "final": final_counts,
                            }

                            # Store confidence stats so CV can aggregate
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray(max_conf[keep_win], dtype=np.float32)
                        except Exception:
                            self._last_class_dists = {"raw": {}, "final": {}}
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)

                    else:
                        X_test_3d = X_test.astype(np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))
                        proba = self.model.predict(
                            X_test_3d, verbose=0, batch_size=int(params.get("cnn_batch_size", 128))
                        )
                        proba = sanitize_proba(proba)
                        if proba.shape[1] >= 3:
                            p_short = proba[:, 0]
                            p_long  = proba[:, 2]
                            max_conf = np.asarray(np.maximum(p_short, p_long), dtype=np.float32)
                            decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                            raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                        else:
                            raw = np.argmax(proba, axis=1)
                            max_conf = np.asarray(proba.max(axis=1), dtype=np.float32)
                            raw_classes = np.asarray(raw, dtype=int)
                            decoded_raw = np.where(raw_classes == 1, 1, -1)

                        # --- Edge-vs-Cost gating (flat 3D variant) ---
                        cfg_f = getattr(self, "features_config", {}) or {}
                        base_thr = float(self._resolve_conf_thr(confidence_threshold))
                        self._last_conf_thr_init = float(cfg_f.get("confidence_threshold", confidence_threshold))

                        _cfg_cost = dict(cfg_f)
                        if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                            _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                        _cost_src = self._ensure_cost_columns(test_data, _cfg_cost)

                        _eval_idx = test_data_scaled.index
                        rets, sprd, slip = self._get_cost_arrays_aligned(_cost_src, _eval_idx)

                        vol_w = int(cfg_f.get("vol_window_bars", _PC["vol_window_bars"]))
                        rv = realized_vol(rets, window=vol_w).to_numpy(dtype=np.float32)
                    
                        # --- Causal scaling: compute mu/sigma (and a safe floor) from TRAIN only ---
                        # Avoid using full-test-month statistics to set live thresholds.
                        try:
                            rets_tr = train_data_scaled["returns"].astype(float)
                            rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                            rv_m_tr = float(np.nanmean(rv_tr))
                            rv_s_tr = float(np.nanstd(rv_tr))
                            rv_floor_tr = float(np.nanmedian(rv_tr[rv_tr > 0])) if np.any(rv_tr > 0) else float("nan")
                        except Exception:
                            rv_m_tr, rv_s_tr, rv_floor_tr = float("nan"), float("nan"), float("nan")
        
                        if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                            vol_z = (rv - rv_m_tr) / rv_s_tr
                        else:
                            # Degenerate train stats -> neutral vol term (no hidden test-fit fallback).
                            vol_z = np.zeros_like(rv, dtype=np.float32)
        
                        # Normalized spread vs vol: use TRAIN-derived floor (or constant) -- never test-wide median.
                        den_floor = rv_floor_tr if (np.isfinite(rv_floor_tr) and rv_floor_tr > 1e-8) else 1e-6
                        den = np.where(rv > 1e-8, rv, den_floor).astype(np.float32)
                        spread_norm = np.divide(sprd, den, out=np.zeros_like(sprd, dtype=np.float32), where=np.isfinite(den))

                        a = float(cfg_f.get("alpha_vol_z", 0.004))
                        b = float(cfg_f.get("beta_spread_norm", _PC["beta_spread_norm"]))
                        g = float(cfg_f.get("gamma_slip_norm", _PC["gamma_slip_norm"]))
                        slip_norm_bps = float(cfg_f.get("slip_norm_bps", _PC["slip_norm_bps"]))
                        min_slip_norm_bps = float(cfg_f.get("min_slip_norm_bps", _PC["min_slip_norm_bps"]))
                        slip_norm_bps = max(slip_norm_bps, min_slip_norm_bps, 1e-6)

                        vol_z_cap = float(cfg_f.get("vol_z_cap", 6.0))
                        spread_norm_cap = float(cfg_f.get("spread_norm_cap", 5.0))
                        slip_ratio_cap = float(cfg_f.get("slip_ratio_cap", 6.0))
                        max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))

                        vol_z = np.clip(vol_z, -vol_z_cap, vol_z_cap)
                        spread_norm = np.clip(spread_norm, 0.0, spread_norm_cap)
                        slip_norm = np.clip(slip / slip_norm_bps, 0.0, slip_ratio_cap)

                        thr_vec = np.clip(base_thr + a*vol_z + b*spread_norm + g*slip_norm, 0.0, max_conf_thr).astype(np.float32)
                        
                        # --- Soft coverage-drift nudge (regime-aware, non-forcing) ---
                        try:
                            tgt   = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))
                            band  = float(cfg_f.get("runtime_active_band_margin", _PC["runtime_active_band_margin"]))
                            win_k = int(cfg_f.get("runtime_coverage_window", 96))
                            step  = float(cfg_f.get("runtime_conf_nudge", 0.01))

                            # Stability guard: if step is too large relative to the band, the nudge can ping-pong.
                            # Enforce step <= 0.5 * band (monotone adjustment within a symmetric deadband).
                            try:
                                band = float(band)
                                step = float(step)
                            except Exception:
                                band, step = 0.0, 0.0
                            band = max(0.0, band)
                            step = abs(step)
                            if band > 0.0 and step > 0.5 * band:
                                _step_old = step
                                step = max(1e-6, 0.5 * band)
                                try:
                                    if bool(getattr(self, "debug", False)):
                                        log_print(
                                            f"[CoverageNudge][Stability] step={_step_old:.5f} > 0.5*band={(0.5*band):.5f}; clamped to {step:.5f}",
                                            level="COMPACT",
                                        )
                                except Exception:
                                    pass
                                                    
                            n = min(len(decoded_raw), len(thr_vec))
                            
                            # preliminary gating with alphabetagamma only
                            _pre = decoded_raw.copy()
                            _mask0 = (max_conf[:n] < thr_vec[:n])
                            np.putmask(_pre[:n], _mask0, 0)
                            # causal rolling active rate on preliminary decisions
                            _act = (_pre[:n] != 0).astype(np.float32)
                            if win_k > 1 and n >= win_k:
                                _cs = np.cumsum(np.insert(_act, 0, 0.0))
                                _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                                # pad to length n
                                _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                                _low, _high = max(0.0, tgt - band), min(1.0, tgt + band)
                                _drift = np.where(_roll < _low, -step, np.where(_roll > _high, step, 0.0)).astype(np.float32)
                                _drift = np.nan_to_num(_drift, nan=0.0)
                                min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                                max_conf_thr = float((cfg_f or {}).get("max_conf_thr", 0.90))

                                _drift = np.clip(_drift, -abs(step), abs(step)).astype(np.float32)
                                thr_vec[:n] = np.clip(
                                    thr_vec[:n] + _drift[:n],
                                    min_conf_thr,
                                    max_conf_thr
                                ).astype(np.float32)

                                if self._is_debug():
                                    print(f"[Gate[OK]] Coverage nudge active | target={tgt:.2f} band=+/-{band:.2f} step={step:.3f}")
                        except Exception as _e:
                            print(f"[Gate] Coverage nudge skipped: {_e}")
                        if self._is_debug():
                            print(f"[Gate[OK]] Dynamic alphabetagamma active | base={base_thr:.3f} alpha={a:.3f} beta={b:.3f} gamma={g:.3f} "
                                f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}")

                        self._last_conf_thr_used = float(np.nanmedian(thr_vec))

                        final_preds = decoded_raw.copy()
                        n = min(len(final_preds), len(thr_vec))
                        mask = (max_conf[:n] < thr_vec[:n])
                        np.putmask(final_preds[:n], mask, 0)

                        if final_preds is None or (final_preds != 0).sum() == 0:
                            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:cnn_seq_no_trades")

                        eval_index = test_data_scaled.index[eval_mask]
                        final_preds = final_preds[eval_mask]
                        test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                        test_data_for_eval["pred"] = pd.Series(final_preds, index=eval_index).values

                        try:
                            raw_counts = pd.Series(raw_classes).value_counts().to_dict()
                            final_counts = pd.Series(final_preds).value_counts().to_dict()

                            # Store for CV / summary
                            self._last_class_dists = {
                                "raw": raw_counts,
                                "final": final_counts,
                            }

                            # Store confidence stats so CV can aggregate
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray(max_conf, dtype=np.float32)
                        except Exception:
                            self._last_class_dists = {"raw": {}, "final": {}}
                            self._last_conf_stats_label = str(model_type)
                            self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)

            else:
                # Classical ML prediction path (with CV backoff + thin-trades fallback)
                # 1) Get class probabilities (calibrated if requested)
                try:
                    proba = self.model.predict_proba(X_test)
                    try:
                        cal_method = str((params.get("calibrate_method")
                                        or (self.features_config or {}).get("calibrate_method")
                                        or "")).lower()
                    except Exception:
                        cal_method = ""
                    if cal_method in ("isotonic", "sigmoid") and str(model_type).lower() != "svm":
                        try:
                            n = int(getattr(X_train, "shape", [0])[0])
                            if n > 0:
                                n_cal = max(200, min(n // 10, 2000))
                                cal_X = X_train[-n_cal:] if n_cal < n else X_train
                                cal_y = y_train[-n_cal:] if n_cal < n else y_train
                                proba, _ = calibrate_prefit_and_predict_proba(
                                    self.model, cal_X, cal_y, X_test, method=cal_method
                                )
                                proba = sanitize_proba(proba)
                        except Exception as _e:
                            print(f"[WARN] Calibration failed ({cal_method}): {_e}")
                except Exception:
                    try:
                        scores = self.model.decision_function(X_test)
                    except Exception as e:
                        raise RuntimeError(f"Model does not support predict_proba/decision_function: {e}")
                    scores = np.atleast_2d(scores)
                    if scores.ndim == 2 and scores.shape[1] == 1:
                        scores = np.column_stack([-scores, scores])
                    scores = scores - np.max(scores, axis=1, keepdims=True)
                    exp = np.exp(scores)
                    proba = exp / np.sum(exp, axis=1, keepdims=True)

                proba = sanitize_proba(proba)

                # 2) Map to 3-class format (short/flat/long)
                classes_attr = getattr(self.model, "classes_", None)
                proba = np.asarray(proba, dtype=np.float32)
                n_rows = proba.shape[0]
                proba3 = np.full((n_rows, 3), 1e-12, dtype=np.float32)

                if classes_attr is not None and proba.ndim == 2 and proba.shape[1] == len(classes_attr):
                    for j, cls in enumerate(classes_attr):
                        if cls in (0, 1, 2):
                            proba3[:, int(cls)] = proba[:, j]
                else:
                    if proba.ndim == 1:
                        p_long = np.clip(proba, 0.0, 1.0)
                        proba3[:, 2] = p_long
                        proba3[:, 0] = 1.0 - p_long
                    elif proba.ndim == 2 and proba.shape[1] == 2:
                        proba3[:, 0] = np.clip(proba[:, 0], 0.0, 1.0)
                        proba3[:, 2] = np.clip(proba[:, 1], 0.0, 1.0)
                    elif proba.ndim == 2 and proba.shape[1] >= 3:
                        proba3 = np.clip(proba[:, :3], 1e-12, 1.0).astype(np.float32)

                proba3 = np.nan_to_num(proba3, nan=1e-12, posinf=1.0, neginf=1e-12)
                proba3 /= np.maximum(proba3.sum(axis=1, keepdims=True), 1.0)
            
                # --- CV-only calibration metrics (Brier / NLL on test window) ---
                try:
                    if getattr(self, "_in_optuna_cv", False):
                        cfg_eval = getattr(self, "features_config", {}) or {}

                        # Use the same label logic as plain thresholding: sign of forward returns
                        thr = float(cfg_eval.get("label_threshold", label_threshold))

                        # Forward one-bar return on the TEST index used for X_test
                        rets = self.data["returns"].reindex(test_data_scaled.index).astype(float)
                        ret_fwd = rets.shift(-1)

                        mask_valid = np.isfinite(ret_fwd.to_numpy())
                        if mask_valid.any():
                            # label_with_neutral -> {0:short, 1:flat, 2:long}
                            y_cal = self.label_with_neutral(ret_fwd[mask_valid], thr).astype(int)
                            proba_cal = proba[mask_valid]

                            brier, nll = compute_brier_and_nll(proba_cal, y_cal)

                            # Expose to the CV aggregator (_single_study_cv)
                            self._last_calib_brier = float(brier)
                            self._last_calib_nll   = float(nll)
                            self._last_calib_n     = int(len(y_cal))

                            if bool(cfg_eval.get("print_cv_debug", False)):
                                print(
                                    f"[CV-Calib/test_strategy] "
                                    f"brier={brier:.6f} | nll={nll:.6f} | n={len(y_cal)}"
                                )
                except Exception as _e:
                    # Never break the evaluation path because of calibration
                    cfg_eval = getattr(self, "features_config", {}) or {}
                    if bool(cfg_eval.get("print_cv_debug", False)):
                        print(f"[CV-Calib/test_strategy] Calibration metric failed: {_e}")

                proba = proba3

                # 3) Optional: fit coverage threshold
                try:
                    cfg_f = getattr(self, "features_config", {}) or {}
                    if is_coverage_intent(cfg_f):
                        # Learn a per-fold coverage->threshold mapping on the calibration tail of TRAIN (CV-safe)
                        try:
                            # target coverage knob (accept both names)
                            _tgt = float(cfg_f.get("target_active_rate",
                                           cfg_f.get("target_coverage",
                                           params.get("target_coverage", params.get("target_active_rate", 0.10)))))
                            # build a small tail from the training matrix used above
                            n_tr = int(getattr(X_train, "shape", [0])[0])
                            ncal = max(200, min(n_tr // 10, 2000)) if n_tr > 0 else 0
                            cal_X = X_train[-ncal:] if (ncal and ncal < n_tr) else X_train
                            cal_y = y_train[-ncal:] if (ncal and ncal < n_tr) else y_train

                            # compute calibrated probabilities on the calibration tail if a method was chosen
                            try:
                                _cal_m = str((params.get("calibrate_method") or cfg_f.get("calibrate_method") or "")).lower()
                            except Exception:
                                _cal_m = ""
                            # NOTE: Don't "calibrate again" here for SVM / already-calibrated estimators.
                           # Coverage threshold only needs stable probs on TRAIN tail.
                            _already_calibrated = False
                            try:
                                from sklearn.calibration import CalibratedClassifierCV
                                _already_calibrated = isinstance(self.model, CalibratedClassifierCV) or hasattr(self.model, "calibrated_classifiers_")
                            except Exception:
                                _already_calibrated = hasattr(self.model, "calibrated_classifiers_")

                            if (_cal_m in ("isotonic", "sigmoid")) and (str(model_type).lower() != "svm") and (not _already_calibrated):
                                p_cal, _ = calibrate_prefit_and_predict_proba(
                                    self.model, cal_X, cal_y, cal_X, method=_cal_m
                                )
                                p_cal = sanitize_proba(p_cal)
                            else:
                                p_cal = sanitize_proba(self.model.predict_proba(cal_X))


                            # map coverage -> threshold on this run and stash for aggregation (CV only)
                            coverage_thr = float(fit_coverage_threshold_on_calibration(p_cal, _tgt))
                            self._coverage_conf_thr = float(coverage_thr)

                            # keep last for CV collector (only when actually in CV)
                            _in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                            if _in_cv:
                                setattr(self, "_cv_cov_thr_last", float(coverage_thr))
                            try:
                                setattr(self, "_last_cov_cal_rows", int(getattr(cal_X, "shape", [0])[0]))
                            except Exception:
                                pass
                            
                            # ctx label must reflect the actual run context (cv vs real_mX vs eval)
                            _ctx = "cv" if _in_cv else "eval"
                            if not _in_cv:
                                try:
                                    if bool(getattr(self, "_in_real_sim", False)):
                                        mx = int(cfg_f.get("month_ix", getattr(self, "_rt_month_ix", 0) or 0))
                                        _ctx = f"real_m{mx}"
                                except Exception:
                                    pass
                            print(
                                f"[Calib][Coverage] conf_thr={float(coverage_thr):.6f} "
                                f"target_active_rate={float(_tgt):.6f} "
                                f"cal_rows={int(getattr(cal_X, 'shape', [0])[0])} ctx={_ctx}"
                            )
                        except Exception as _ee:
                            print(f"[WARN] Coverage threshold fit skipped in CV: {_ee}")
                except Exception as _e:
                    print(f"[Calib] Classical coverage threshold skipped: {_e}")

                # Coverage should be based on **trade intent**, not certainty about "flat".
                if proba.shape[1] >= 3:
                    p_short = proba[:, 0]
                    p_long  = proba[:, 2]
                    max_conf = np.maximum(p_short, p_long)
                    decoded_raw = np.where(p_long > p_short, 1, np.where(p_short > p_long, -1, 0))
                    raw_classes = np.where(p_long > p_short, 2, np.where(p_short > p_long, 0, 1)).astype(int)
                else:
                    max_conf    = proba.max(axis=1)
                    raw_classes = proba.argmax(axis=1)
                    decoded_raw = np.where(raw_classes == 1, 1, -1)  # best-effort for 2-class

                cfg_f = getattr(self, "features_config", {}) or {}
                conf0 = float(cfg_f.get("confidence_threshold", confidence_threshold))
                try:
                    if is_coverage_intent(cfg_f) and hasattr(self, "_coverage_conf_thr"):
                        conf0 = float(getattr(self, "_coverage_conf_thr"))
                except Exception:
                    pass

                # --- Edge-vs-Cost gating (dynamic threshold) ---
                base_thr = float(self._resolve_conf_thr(conf0))
                self._last_conf_thr_init = float(conf0)

                print(f"[DEBUG][Costs] high_vol_thr_train={high_vol_thr_train} | cfg_high_vol_thr={cfg_f.get('high_vol_thr')}")
                
                # Persist for *all* downstream paths this month (TopN, consensus, cont-metrics)
                try:
                    if high_vol_thr_train is not None:
                        if not hasattr(self, "features_config") or not isinstance(self.features_config, dict):
                            self.features_config = {}
                        self.features_config["high_vol_thr"] = float(high_vol_thr_train)
                        self._high_vol_thr_train = float(high_vol_thr_train)
                except Exception:
                    pass
                
                _cfg_cost = dict(cfg_f)
                if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
                    _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                _cost_src = self._ensure_cost_columns(test_data, _cfg_cost)

                _eval_idx = test_data_scaled.index
                rets, sprd, slip = self._get_cost_arrays_aligned(_cost_src, _eval_idx)

                vol_w = int(cfg_f.get("vol_window_bars", _PC["vol_window_bars"]))
                # --- Train-anchored vol scaling (avoid ex-post test-month stats) ---
                rv_m_tr, rv_s_tr, den_floor_tr = np.nan, np.nan, np.nan
                try:
                    rets_tr = train_data_scaled["returns"].astype(float)
                    rv_tr = realized_vol(rets_tr, window=vol_w).to_numpy(dtype=np.float32)
                    rv_m_tr = float(np.nanmean(rv_tr))
                    rv_s_tr = float(np.nanstd(rv_tr))
                    _pos = rv_tr[np.isfinite(rv_tr) & (rv_tr > 0)]
                    if _pos.size > 0:
                        den_floor_tr = float(np.nanmedian(_pos))
                except Exception:
                    pass

                rv = realized_vol(rets, window=vol_w).to_numpy(dtype=np.float32)
                if np.isfinite(rv_s_tr) and rv_s_tr > 0:
                    vol_z = (rv - rv_m_tr) / rv_s_tr
                else:
                    vol_z = np.zeros_like(rv, dtype=np.float32)

                den_floor = den_floor_tr if (np.isfinite(den_floor_tr) and den_floor_tr > 1e-8) else 1e-6
                den = np.where(rv > 1e-8, rv, den_floor).astype(np.float32)
                spread_norm = np.divide(sprd, den, out=np.zeros_like(sprd, dtype=np.float32), where=np.isfinite(den))

                min_conf_thr = float(cfg_f.get("min_conf_thr", 0.33))
                max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))

                # safer alphabetagamma defaults (small nudges, not giant jumps)
                a = float(cfg_f.get("alpha_vol_z", 0.004))
                b = float(cfg_f.get("beta_spread_norm", _PC["beta_spread_norm"]))
                g = float(cfg_f.get("gamma_slip_norm", _PC["gamma_slip_norm"]))

                # cap the drivers (prevents spikes from blowing up thr)
                vol_z_cap = float(cfg_f.get("vol_z_cap", 6.0))
                spread_norm_cap = float(cfg_f.get("spread_norm_cap", 5.0))
                slip_ratio_cap = float(cfg_f.get("slip_ratio_cap", 6.0))

                vol_z = np.clip(vol_z, -vol_z_cap, vol_z_cap).astype(np.float32)
                spread_norm = np.clip(spread_norm, 0.0, spread_norm_cap).astype(np.float32)

                # slippage normalization (make denominator never tiny, and cap ratio)
                slip_norm_bps = float(cfg_f.get("slip_norm_bps", _PC["slip_norm_bps"]))
                min_slip_norm_bps = float(cfg_f.get("min_slip_norm_bps", _PC["min_slip_norm_bps"]))
                slip_norm_bps = max(slip_norm_bps, min_slip_norm_bps, 1e-6)

                slip_norm = np.clip(slip / slip_norm_bps, 0.0, slip_ratio_cap).astype(np.float32)

                thr_vec = np.clip(
                    base_thr + a*vol_z + b*spread_norm + g*slip_norm,
                    min_conf_thr,
                    max_conf_thr
                ).astype(np.float32)

                if self._is_debug():
                    print(f"[Gate[OK]] Dynamic alphabetagamma active | base={base_thr:.3f} alpha={a:.3f} beta={b:.3f} gamma={g:.3f} "
                        f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}")

            
                # --- IMPORTANT: restrict gating to ELIGIBLE bars only ---
                # Otherwise you "target coverage" on bars you later drop (warmup/session/anchor),
                # which can yield 0 trades in real-sim despite nonzero signals pre-mask.
                try:
                    _eval_mask = np.asarray(eval_mask, dtype=bool)
                    if _eval_mask.size != decoded_raw.size:
                        _eval_mask = np.ones(decoded_raw.size, dtype=bool)
                except Exception:
                    _eval_mask = np.ones(decoded_raw.size, dtype=bool)
                _eval_idx = np.flatnonzero(_eval_mask)
                            
                # --- Soft coverage-drift nudge (regime-aware, non-forcing) ---
                try:
                    tgt   = float(cfg_f.get("target_active_rate", cfg_f.get("target_coverage", 0.10)))
                    band  = float(cfg_f.get("runtime_active_band_margin", _PC["runtime_active_band_margin"]))
                    win_k = int(cfg_f.get("runtime_coverage_window", 96))
                    step  = float(cfg_f.get("runtime_conf_nudge", 0.01))

                    # Stability guard: if step is too large relative to the band, the nudge can ping-pong.
                    # Enforce step <= 0.5 * band (monotone adjustment within a symmetric deadband).
                    try:
                        band = float(band)
                        step = float(step)
                    except Exception:
                        band, step = 0.0, 0.0
                    band = max(0.0, band)
                    step = abs(step)
                    if band > 0.0 and step > 0.5 * band:
                        _step_old = step
                        step = max(1e-6, 0.5 * band)
                        try:
                            if bool(getattr(self, "debug", False)):
                                log_print(
                                    f"[CoverageNudge][Stability] step={_step_old:.5f} > 0.5*band={(0.5*band):.5f}; clamped to {step:.5f}",
                                    level="COMPACT",
                                )
                        except Exception:
                            pass

                    n = int(_eval_idx.size)
                    if win_k > 1 and n > 0:
                        _pre = decoded_raw[_eval_idx].copy()
                        _mask0 = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                        _pre[_mask0] = 0
                        _act = (_pre != 0).astype(np.float32)

                        if n >= win_k:
                            _cs = np.cumsum(np.insert(_act, 0, 0.0))
                            _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                            _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                        else:
                            _roll = np.full(n, np.nan, dtype=np.float32)

                        _low, _high = max(0.0, tgt - band), min(1.0, tgt + band)
                        _drift = np.where(_roll < _low, -step, np.where(_roll > _high, step, 0.0)).astype(np.float32)
                        _drift = np.nan_to_num(_drift, nan=0.0)
                        min_conf_thr = float((cfg_f or {}).get("min_conf_thr", 0.33))
                        max_conf_thr = float((cfg_f or {}).get("max_conf_thr", 0.90))

                        _drift = np.clip(_drift, -abs(step), abs(step)).astype(np.float32)
                        thr_vec[_eval_idx] = np.clip(
                            thr_vec[_eval_idx] + _drift,
                            min_conf_thr,
                            max_conf_thr
                        ).astype(np.float32)
                        if self._is_debug():
                            print(f"[Gate[OK]] Coverage nudge active | target={tgt:.2f} band=+/-{band:.2f} step={step:.3f}")
                except Exception as _e:
                    print(f"[Gate] Coverage nudge skipped (deep-3D): {_e}")


                self._last_conf_thr_used = float(np.nanmedian(thr_vec))

                # Apply confidence filter ONLY on eligible bars (others are forced flat)
                final_preds = np.zeros_like(decoded_raw)
                if _eval_idx.size > 0:
                    final_preds[_eval_idx] = decoded_raw[_eval_idx]
                    _mask = (max_conf[_eval_idx] < thr_vec[_eval_idx])
                    final_preds[_eval_idx[_mask]] = 0

                if final_preds is None or (final_preds != 0).sum() == 0:
                    if self._is_debug():
                        print("[ERR] No trades predicted after filtering -- penalizing this parameter set.")
                    if in_cv:
                        return _safe_metrics_return(
                            (np.nan,) * N_METRICS,
                            context="test_ensemble_strategy:no_trades_cv",
                        )
                    final_preds = np.zeros_like(decoded_raw, dtype=int)

                eval_index = test_data_scaled.index[eval_mask]
                final_preds = final_preds[eval_mask]
                test_data_for_eval = test_data_raw_with_extras.loc[eval_index].copy()
                test_data_for_eval["pred"] = pd.Series(final_preds, index=eval_index).values

                try:
                    if self._is_debug():
                        print("Raw prediction distribution:", pd.Series(raw_classes).value_counts())
                    decoded = decoded_raw
                    if self._is_debug():
                        print("Decoded preds (before confidence filter):", pd.Series(decoded).value_counts())
                        print("Final preds (after confidence filter):", pd.Series(final_preds).value_counts())

                    # --- Store class distributions for CV / mini-block summaries ---
                    raw_counts = _norm_class_counts(pd.Series(raw_classes).value_counts(dropna=False).to_dict())
                    final_counts = _norm_class_counts(pd.Series(final_preds).value_counts(dropna=False).to_dict())
                    self._last_class_dists = {"raw": raw_counts, "final": final_counts}

                    # --- Store confidence stats for aggregated diagnostics ---
                    self._last_conf_stats_label = str(model_type)
                    self._last_conf_stats_max_conf = np.asarray(max_conf, dtype=np.float32)

                    # Print concise confidence summary (unchanged behaviour)
                    _in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                    _in_real = bool(getattr(self, "_in_real_sim", False))
                    _rt_mx = int(getattr(self, "_rt_month_ix", 0) or 0)
                    _ctx_label = "cv" if _in_cv else (f"real_m{_rt_mx}" if _in_real else "eval")

                    if self._is_debug():
                        print_conf_stats(
                            max_conf,
                            label=f"{str(model_type)}@{_ctx_label}",
                            median_thr=float(getattr(self, "_last_conf_thr_used", float("nan"))),
                        )
                except Exception:
                    # Defensive defaults so summaries never break
                    self._last_class_dists = {"raw": {}, "final": {}}
                    self._last_conf_stats_label = str(model_type)
                    self._last_conf_stats_max_conf = np.asarray([], dtype=np.float32)


                    # --- Ensure 'returns' exists and is aligned for ALL branches ---
            # IMPORTANT: do NOT shift returns here. compute_full_evaluation_metrics()
            # applies the one-bar execution delay by shifting the predictions (pred.shift(1)).
            # Keeping returns un-shifted (original series) avoids double-lag.
            test_data_for_eval["returns"] = (
                self.data["returns"].reindex(test_data_for_eval.index).astype(float)
            )
            if test_data_for_eval["returns"].isna().any():
                # Drop rows where we truly have missing returns (end-of-data)
                test_data_for_eval = test_data_for_eval.dropna(subset=["returns"])

            # --- Optional session_flag for evaluation (1 = inside NY session, 0 = outside) ---
            # Right now, because test_data_for_eval is already session-filtered when
            # session_filter_mode includes "test", this will be a column of ones.
            # It becomes meaningful as soon as you stop filtering test data but still
            # want to block NEW entries outside session in compute_full_evaluation_metrics.
            try:
                if hasattr(self, "_ny_mask") and self._ny_mask is not None:
                    _sess = self._ny_mask.reindex(test_data_for_eval.index, fill_value=False)
                    test_data_for_eval["session_flag"] = _sess.astype(int)
                else:
                    # No precomputed mask: treat all bars as tradable (session_flag=1)
                    test_data_for_eval["session_flag"] = 1
            except Exception:
                # Fail-soft: if anything goes wrong (index mismatch, etc.), we just
                # skip session gating in the evaluator.
                test_data_for_eval["session_flag"] = 1

            # --- Edge-bar guard: require the next in-filter bar to be contiguous (no overnight open) ---
            _idx = test_data_for_eval.index
            if len(_idx) >= 2:
                gaps = pd.Series(_idx[1:] - _idx[:-1], index=_idx[:-1])
                exp  = gaps.median()  # ~= base bar length (auto-infers 15m)
                is_edge = gaps > (exp * 1.5)

                # Audit (debug-only): check whether the edge-bar guard is killing sparse signals
                if self._is_debug():
                    try:
                        _ctx = "cv" if bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False)) else ("real_sim" if bool(getattr(self, "_in_real_sim", False)) else "run")
                        _edge_idx = is_edge.index[is_edge]
                        _pred_ser = test_data_for_eval["pred"]
                        _nz_before = int((_pred_ser != 0).sum())
                        _nz_edge = int((_pred_ser.reindex(_edge_idx).fillna(0) != 0).sum()) if len(_edge_idx) else 0
                        _nz_last = int(bool(_pred_ser.iloc[-1] != 0))
                        print(f"[EdgeGuardAudit][{model_type}] ctx={_ctx} exp={exp} edge_bars={int(is_edge.sum())} nz_before={_nz_before} nz_on_edge={_nz_edge} nz_last={_nz_last}")
                    except Exception:
                        pass

                # Zero out initiations on edge bars and on the very last row
                test_data_for_eval.loc[is_edge.index[is_edge], "pred"] = 0
                test_data_for_eval.iloc[-1, test_data_for_eval.columns.get_loc("pred")] = 0

                if self._is_debug():
                    try:
                        _nz_after = int((test_data_for_eval["pred"] != 0).sum())
                        print(f"[EdgeGuardAudit][{model_type}] nz_after={_nz_after}")
                    except Exception:
                        pass


            # ----------------------------
            # 5) Evaluation + storage
            # ----------------------------
            # The new dynamic edge-vs-cost gating already adjusts thresholds per-bar
            # using alpha*vol_z + beta*spread_norm + gamma*slip_norm, so no extra quantile bump is needed.

            cfg_adj = dict(getattr(self, "features_config", {}) or {})
        
            # Propagate train-anchored high-vol threshold into the evaluation cost layer
            # (prevents LeakageGuard / ex-post thresholding on the eval window)
            try:
                if high_vol_thr_train is not None and cfg_adj.get("high_vol_thr") is None:
                    cfg_adj["high_vol_thr"] = float(high_vol_thr_train)
            except Exception:
                pass


            # 1) Ensure real per-bar costs are attached (no synthetic means)
            try:
                if bool(getattr(self, "trading_costs", True)):
                    _eval_df = locals().get("test_eval_df", None) or locals().get("test_data_for_eval", None)
                    if _eval_df is not None:
                        _eval_df_refreshed = self._ensure_cost_columns(_eval_df, cfg_adj)
                        if _eval_df_refreshed is not None:
                            if _eval_df is locals().get("test_eval_df", None):
                                test_eval_df = _eval_df_refreshed
                            else:
                                test_data_for_eval = _eval_df_refreshed
            except Exception:
                pass

            # 2) Compute after-cost metrics
            # Robust handles: avoid NameError if a partial refactor/merge left variables uninitialized.
            _eval_df = locals().get("test_eval_df", None)
            if _eval_df is None:
                _eval_df = locals().get("test_data_for_eval", None)

            _full_df = locals().get("test_data", None)

            # If we still don't have an eval frame, bail safely with the fixed 16-metric contract.
            if _eval_df is None:
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:no_eval_frame")
        
            try:
                if _eval_df is not None:
                    _eval_df.attrs["features_config"] = cfg_adj
                    _eval_df.attrs["debug_costs"] = bool(self._is_debug())
            except Exception:
                pass

            _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
            if _in_cv_mode:
                _eval_ctx = "cv:fold_or_month_eval:test_strategy"
            elif bool(getattr(self, "_in_real_sim", False)):
                _eval_ctx = "real_sim:month_eval:test_strategy"
            else:
                _eval_ctx = "eval:test_strategy"
                
            # Telemetry-only: persist the true evaluated bar-grid length.
            # This should match ExecAudit bars (position_exec) for this evaluation context.
            try:
                self._last_eval_bars = int(len(_eval_df)) if _eval_df is not None else 0
            except Exception:
                pass

            metrics = compute_full_evaluation_metrics(
                df=_eval_df,
                trading_costs=self.trading_costs,
                slippage_factor=self.slippage_factor,
                eval_context=_eval_ctx,
            )
            
            # Capture trade-intent precision from evaluator (cheap scalar; safe in CV).
            try:
                _attrs = getattr(_eval_df, "attrs", {}) or {}
                self._last_precision_trade = float(_attrs.get("precision_trade", float("nan")))
                self._last_n_trade_preds = int(_attrs.get("n_trade_preds", 0) or 0)
            except Exception:
                self._last_precision_trade = float("nan")
                self._last_n_trade_preds = 0


            if not _in_cv_mode:
                # normal run -- keep final month-level results
                # Safety: keep canonical executed position in `position` (downstream expects it).
                if _eval_df is not None and "position_exec" in _eval_df.columns:
                    try:
                        pos = _eval_df.get("position", None)
                        posx = _eval_df["position_exec"]
                        if pos is None:
                            _eval_df["position"] = posx
                        else:
                            same = np.allclose(
                                pos.fillna(0).to_numpy(dtype=float),
                                posx.fillna(0).to_numpy(dtype=float),
                                atol=1e-12, rtol=0.0
                            )
                            if not same:
                                _eval_df["position"] = posx
                    except Exception:
                        _eval_df["position"] = _eval_df["position_exec"]
                self.results = _eval_df.copy() if _eval_df is not None else None
                self.results_full = _full_df.copy() if _full_df is not None else None
                # clear any CV scratch storage
                self._cv_last_eval_df = None

                # Free deep model/TF-graph immediately after results are captured,
                # before the caller proceeds to store large frames like all_dfs / trade_log.
                if model_type in deep_models:
                    try:
                        if getattr(self, "model", None) is not None:
                            self.model = None
                    except Exception:
                        pass
                    try:
                        tf.keras.backend.clear_session()
                    except Exception:
                        pass
                    try:
                        self._clear_feature_cache()
                    except Exception:
                        pass
            else:
                # CV run -- expose a lightweight copy for the tuner/CV aggregator
                # Keep only execution + PnL columns to avoid retaining the full feature matrix in RAM.
                if _eval_df is not None and not _eval_df.empty:
                    _keep = [
                        c for c in (
                            "timestamp", "time", "price", "close",
                            "pred", "position", "position_exec",
                            "returns", "strategy", "strategy_exec",
                            "cstrategy", "creturns", "cstrategy_cont", "creturns_cont",
                            "regime_id", "regime_id_diag",
                        )
                        if c in _eval_df.columns
                    ]
                    # If we didn't match any of the keep-cols, keep the full eval df (otherwise diagnostics become all-NaN).
                    self._cv_last_eval_df = _eval_df[_keep].copy() if _keep else _eval_df.copy()
                else:
                    self._cv_last_eval_df = None
                # accumulate per-fold frames (small list kept on the instance only during this CV run)
                try:
                   if self._cv_last_eval_df is not None and self._is_debug():
                        _cap = int(os.environ.get("CV_MAX_EVAL_FRAMES", "5"))
                        if _cap > 0 and len(self._cv_fold_eval_frames) < _cap:
                            self._cv_fold_eval_frames.append(self._cv_last_eval_df.copy())
                except Exception:
                    # defensive: ensure _cv_fold_eval_frames exists and is list-like
                    try:
                        if self._cv_last_eval_df is not None and self._is_debug():
                            self._cv_fold_eval_frames = [self._cv_last_eval_df.copy()]
                        else:
                            self._cv_fold_eval_frames = []
                    except Exception:
                        self._cv_fold_eval_frames = []
                # do not persist fold outputs into the long-lived results (CV-only)
                self.results = None
                self.results_full = None

            # Aggressively free memory
            try:
                del X_test
            except Exception:
                pass
            
            # Drop deep-model tensors/arrays if they exist (no-op for classical models).
            try:
                del X_seq_train
            except Exception:
                pass
            try:
                del y_seq_train
            except Exception:
                pass
            try:
                del X_seq_test  # noqa: F821
            except Exception:
                pass
            try:
                del y_seq_test  # noqa: F821
            except Exception:
                pass

            # Drop model reference before clearing TF session to improve release behavior.
            try:
                if getattr(self, "model", None) is not None:
                    self.model = None
            except Exception:
                pass
        
            # Release large engineered feature frames ASAP (train/test df_out stored in _feat_cache).
            # We already clear at function start, but clearing here avoids holding those frames
            # until the *next* call and helps reduce RAM high-water across long runs.
            try:
                self._clear_feature_cache()
            except Exception:
                pass

            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            _gc.collect()

            # [OK] Return standardized, validated metrics
            metrics = _safe_metrics_return(metrics, context="test_strategy")
            return metrics


