"""Auto-extracted mixin — see composed.py for the full MLBacktester."""
from pipeline._imports import *  # noqa: F401,F403


class DQNMixin:
    """
    DQN strategy

    Auto-extracted from MLBacktesterNoWFO.py lines 9523-10205.
    """
    def test_dqn_strategy(self, train_start, train_end, test_start, test_end, lags, dqn_config: dict | None = None):
        """
        Train a DQN agent on the training window and evaluate on the test window.
        Returns a fixed-length metrics tuple from `compute_full_evaluation_metrics`.
        """

        # 1) Lazy-load feature config for DQN path ONLY if no features_config was provided already
        if not hasattr(self, "_dqn_features_config_set"):
            if not isinstance(getattr(self, "features_config", None), dict) or not self.features_config:
                with open(FEATURES_PATH, "r") as f:
                    self.features_config = json.load(f)
            self._dqn_features_config_set = True

        # B1: ensure DEFAULT_FEATURES fill missing execution knobs for DQN path too
        # (compute_full_evaluation_metrics reads from df.attrs["features_config"])
        try:
            self.apply_feature_defaults()
        except Exception:
            pass

        # Clear any sticky feature cache between DQN runs
        self._clear_feature_cache()

        # 2) Prepare data (apply same session filter + embargo used elsewhere)
        full_data  = self.data
        train_data = full_data.loc[train_start:train_end]

        true_test_start = pd.to_datetime(test_start)
        test_end        = pd.to_datetime(test_end)
        warmup_need     = int(compute_required_test_warmup_bars({
            **self.features_config, "model_type": "dqn", "dqn_config": (dqn_config or {})
        }))
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
                print(f"⚠️ Lazy NY mask build failed: {_e}")
                self._ny_mask = pd.Series(True, index=self.data.index)

        # NEW semantics:
        # - "both":        filter train + test
        # - "test_only":   filter test only
        # - "train_only":  filter train only
        if sess_mode in ("test_only", "both"):
            test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]
        if sess_mode in ("train_only", "both"):
            train_data = train_data.loc[self._ny_mask.reindex(train_data.index, fill_value=False)]

        if warmup_need > 0 and len(test_data) > 0:
            have = int((test_data.index < true_test_start).sum())
            if have < _total_warmup_need:
                need_more = _total_warmup_need - have
                test_data = _slice_with_warmup(_total_warmup_need + need_more)
                if sess_mode in ("test_only", "both"):
                    test_data = test_data.loc[self._ny_mask.reindex(test_data.index, fill_value=False)]

        try:
            embargo_n = int(self.features_config.get("final_embargo_bars", 0))
            if bool(getattr(self, "_in_optuna_cv", False)):
                embargo_n = 0
            if embargo_n > 0 and len(test_data) > embargo_n:
                test_data = test_data.iloc[embargo_n:].copy()
                print(f"[Embargo] Dropped first {embargo_n} test bars (DQN, non-CV).")
        except Exception as e:
            print(f"⚠️ final_embargo_bars handling failed (DQN): {e}")

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
            print(f"[CV/DQN] Eval anchor forced to fold start: {first_eval_ts} | test_len={len(test_data)} | warmup_need={_total_warmup_need}")

        if first_eval_ts is None:
            print("❌ No tradable bar found in test window (DQN).")
            if bool(getattr(self, "_in_optuna_cv", False)):
                self.results = None
                self.results_full = None
                self._cv_last_eval_df = None
            else:
                self.results = pd.DataFrame()
                self.results_full = self.results
                self._cv_last_eval_df = None
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_dqn_strategy:no_tradable_bar")

        self._expected_eval_start = first_eval_ts

        # Feature knobs
        cfg          = self.features_config
        lag_depth    = cfg.get("lag_depth", 1)
        roll_windows = cfg.get("roll_windows", [5])
        
        # DQN env knobs live in features_config by default.
        # Allow safe overrides from dqn_config for reward processing + DQN env costs.
        cfg_env = dict(cfg or {})
        try:
            _dcfg = dict(dqn_config or {})
            for _k in (
                "env_reward_clip", "env_reward_tanh_k", "env_reward_clip_range",
                "env_reward_norm", "env_reward_norm_beta",
                "env_cost_scale_dqn", "env_turnover_penalty_dqn",
            ):
                if _k in _dcfg:
                    cfg_env[_k] = _dcfg[_k]
        except Exception:
            cfg_env = dict(cfg or {})


        # Build features (train): persist the exact feature list for DQN consistency
        train_data, features = self.prepare_features(train_data, int(lags), lag_depth=lag_depth, roll_windows=roll_windows)
        train_data = train_data.loc[:, ~train_data.columns.duplicated()]
        if features:
            train_data = train_data.dropna(subset=features).copy()

        self.dqn_feature_list = features

        high_vol_thr_train = None
        try:
            if "returns" in train_data.columns:
                _cfg_cost_src = getattr(self, "features_config", {}) or {}
                vol_w = int(_cfg_cost_src.get("vol_window_bars", 48))
                qhi   = float(_cfg_cost_src.get("high_vol_q", 0.75))
                _rv_tr = realized_vol(train_data["returns"].astype(float), window=vol_w)
                _rv_tr = _rv_tr.dropna()
                if len(_rv_tr) > 0:
                    high_vol_thr_train = float(_rv_tr.quantile(qhi))
        except Exception:
            high_vol_thr_train = None

        # Attach cost columns on the TRAIN slice (no leakage; rewards only)
        try:
            if bool(getattr(self, "trading_costs", True)):
                cfg_cost = dict(getattr(self, "features_config", {}) or {})
                if high_vol_thr_train is not None and cfg_cost.get("high_vol_thr") is None:
                    cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
                train_data = self._ensure_cost_columns(train_data.copy(), cfg_cost)
        except Exception as _e:
            print(f"[DQN-cost] Skipped adding cost columns on train_data: {_e}")

        # Build features (test)
        test_data_full, _ = self.prepare_features(test_data, int(lags), lag_depth=lag_depth, roll_windows=roll_windows)
        test_data_full = test_data_full.loc[:, ~test_data_full.columns.duplicated()]
        if features:
            test_data_full = test_data_full.dropna(subset=features).copy()

        test_data_features = test_data_full[self.dqn_feature_list].copy()
        
        # Local DQN config (must be applied to env wrappers + training)
        try:
            cfg_local = dict(dqn_config or {})
        except Exception:
            cfg_local = {}


        # 3) Initialize environment (window=lags, state_size=len(features))
        # IMPORTANT (validity): if we attach CostAwareWrapper (spread/slippage arrays),
        # disable the env's internal fixed slippage penalty to avoid double-charging costs.
        spr = None
        slp = None
        _will_use_cost_wrapper = False
        try:
            if bool(getattr(self, "trading_costs", True)):
                spr = train_data.get("spread", None)
                slp = train_data.get("slippage_bps", None)
                _will_use_cost_wrapper = (spr is not None) and (slp is not None)
        except Exception:
            spr = None
            slp = None
            _will_use_cost_wrapper = False

        _env_slippage = 0.0 if _will_use_cost_wrapper else float(getattr(self, "slippage_factor", 0.0))
        env = TradingEnv(train_data, features, slippage=_env_slippage, window=int(lags))

        # Optional: cost-aware reward using spread/slippage arrays from the TRAIN slice
        if _will_use_cost_wrapper:
            try:
                spr_arr = spr.to_numpy(dtype=np.float32, copy=False)
                slp_arr = slp.to_numpy(dtype=np.float32, copy=False)

                cfg_cost_env = getattr(self, "features_config", {}) or {}
                cost_scale = float(cfg_local.get("env_cost_scale_dqn", cfg_cost_env.get("env_cost_scale_dqn", 1.5)))
                turnover_penalty = float(cfg_local.get("env_turnover_penalty_dqn", cfg_cost_env.get("env_turnover_penalty_dqn", 0.0)))

                # Provide mid-price for spread->return conversion if available
                px_arr = None
                try:
                    for _c in ("mid_close", "close", "price"):
                        if _c in train_data.columns:
                            px_arr = train_data[_c].to_numpy(dtype=np.float32, copy=False)
                            break
                except Exception:
                    px_arr = None
                    
                env = CostAwareWrapper(
                    env,
                    spread=spr_arr,
                    slippage_bps=slp_arr,
                    mid_price=px_arr,
                    cost_scale=cost_scale,
                    turnover_penalty=turnover_penalty,
                )
                if self._is_debug():
                    print(
                        f"[DQN-cost] CostAwareWrapper attached (n={spr_arr.shape[0]}), "
                        f"cost_scale={cost_scale}, turnover_penalty={turnover_penalty}."
                    )
            except Exception as _e:
                print(f"[DQN-cost] Skipped CostAwareWrapper: {_e}")

        # Prefer dqn_config, fall back to features_config
        cfg_env = dict(getattr(self, "features_config", {}) or {})
        cfg_env.update(cfg_local or {})

        rw_clip = cfg_env.get("env_reward_clip", None)              # None|"tanh"|"range"
        rw_tk   = float(cfg_env.get("env_reward_tanh_k", 3.0))
        rw_rng  = tuple(cfg_env.get("env_reward_clip_range", (-1.0, 1.0)))
        rw_norm = bool(cfg_env.get("env_reward_norm", True))
        rw_b    = float(cfg_env.get("env_reward_norm_beta", 0.99))

        if self._is_debug():
            print(f"[DQN-reward] clip={rw_clip} tanh_k={rw_tk} range={rw_rng} norm={rw_norm} beta={rw_b}")

        if rw_clip or rw_norm:
            env = RewardProcessWrapper(
                env,
                clip_mode=rw_clip,
                tanh_k=rw_tk,
                clip_range=rw_rng,
                norm=rw_norm,
                norm_beta=rw_b,
            )

        if not hasattr(env, "reset"):
            raise AttributeError("TradingEnv has no reset(). Update rl/environment.py or import the correct class.")

        print(f"[DQN] Env ready | window={env.window} | feature_dim={len(features)}")

        input_shape = (len(features),)
        # Validity: agent window must match env.window (lags). Override any mismatch.
        _w_cfg = cfg_local.get("window", None)
        cfg_local["window"] = int(lags)
        if _w_cfg is not None:
            try:
                if int(_w_cfg) != int(lags) and self._is_debug():
                    print(f"[DQN] Overriding dqn_config.window={_w_cfg} -> {int(lags)} to match lags/env.window.")
            except Exception:
                pass
        cfg_local.setdefault("reward_switch_penalty", 0.007)

        use_pretrained = bool(cfg.get("dqn_use_pretrained", cfg_local.get("use_pretrained", False)))
        in_real_sim = bool(getattr(self, "_in_real_sim", False))
        
        # Run-scoped pretrained paths (avoid accidental cross-run reuse)
        dqn_model_path = MODEL_DQN_PATH
        dqn_cfg_path   = DQN_AGENT_CONFIG_PATH
        try:
            _run_dir = os.environ.get("RESULTS_RUN_DIR", "") or ""
            if _run_dir.strip():
                os.makedirs(_run_dir, exist_ok=True)
                dqn_model_path = os.path.join(_run_dir, os.path.basename(MODEL_DQN_PATH))
                dqn_cfg_path   = os.path.join(_run_dir, os.path.basename(DQN_AGENT_CONFIG_PATH))
        except Exception:
            pass

        loaded_from_disk = False
        if use_pretrained and in_real_sim:
            try:
                if os.path.exists(dqn_model_path) and os.path.exists(dqn_cfg_path):
                    if self._is_debug():
                        print(f"[DQN] Loading pretrained agent from {dqn_model_path}")
                    from rl.dqn_agent import DQNAgent
                    self.model = DQNAgent.load(dqn_model_path, dqn_cfg_path)
                    loaded_from_disk = True
                else:
                    if self._is_debug():
                        print("[DQN] No pretrained DQN files found; will train from scratch.")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained DQNAgent; training from scratch instead: {e}")
                self.model = None
                loaded_from_disk = False

        if not loaded_from_disk:
            self.model = self.get_model("dqn", input_shape=input_shape, dqn_config=cfg_local, lags=int(lags))
            self.model.fit(env)

            if use_pretrained and in_real_sim:
                try:
                    self.model.save(dqn_model_path, dqn_cfg_path)
                    print(f"💾 DQN model trained and saved to {dqn_model_path} / {dqn_cfg_path}.")

                except Exception as e:
                    print(f"⚠️ Could not save DQN model: {e}")

        # 5) Prediction on test (windowed) + supervised-style gating compatibility
        feats = self.dqn_feature_list
        X_test = test_data_features.to_numpy(dtype=np.float32, copy=False)

        if self.model is None:
            raise RuntimeError("DQN model is not initialized (NoneType)!")
        if X_test.shape[1] != self.model.state_size:
            raise ValueError(
                f"DQN feature mismatch: model expects {self.model.state_size}, got {X_test.shape[1]}.\n"
                f"Train features: {self.dqn_feature_list}\n"
                f"Test features:  {list(test_data_features.columns)}"
            )

        # Keep eval window consistent with env.window (lags), even for loaded agents.
        lags_eff = int(lags)
        try:
            if hasattr(self.model, "window") and int(getattr(self.model, "window")) != int(lags_eff):
                if self._is_debug():
                    print(f"[DQN] Forcing loaded agent window={getattr(self.model, 'window')} -> {lags_eff} to match lags.")
                setattr(self.model, "window", lags_eff)
                if hasattr(self.model, "config") and isinstance(getattr(self.model, "config"), dict):
                    self.model.config["window"] = lags_eff
        except Exception:
            pass

        # ---- Build a gating config that behaves like test_strategy (but without redesign) ----
        cfg_gate = dict(getattr(self, "features_config", {}) or {})
        # Allow a small whitelist of overrides from dqn_config (do NOT allow coverage overrides)
        try:
            for _k in (
                "gating_mode", "gate_mode",
                "target_active_rate", "target_coverage",
                "confidence_threshold",
                "deep_calibration_frac", "deep_calibration_min_samples",
            ):
                if _k in (cfg_local or {}):
                    cfg_gate[_k] = cfg_local[_k]
        except Exception:
            pass
        cfg_gate.setdefault("model_type", "dqn")
        
        # --- B1 Policy: GLOBAL target coverage locked for DQN as well (ignore local overrides) ---
        try:
            enforce_target_coverage_policy(cfg_gate, model_type='dqn')
        except Exception:
            pass

        # Coverage intent + calibration on TRAIN-only tail windows (prevents leakage)
        default_conf = float(cfg_gate.get("confidence_threshold", 0.50))
        cov_intent = bool(is_coverage_intent(cfg_gate))
        rate = float(cfg_gate.get("target_active_rate", cfg_gate.get("target_coverage", 0.0)) or 0.0)

        coverage_thr = None
        if cov_intent and rate > 0.0:
            try:
                frac = float(cfg_gate.get("deep_calibration_frac", 0.10))
                nmin = int(cfg_gate.get("deep_calibration_min_samples", 500))

                X_tr = train_data[self.dqn_feature_list].to_numpy(dtype=np.float32, copy=False)
                ntr = int(X_tr.shape[0])
                if ntr >= lags_eff and (ntr - lags_eff + 1) >= 50:
                    nwin_tr = ntr - lags_eff + 1
                    ncal = max(nmin, int(round(nwin_tr * max(0.01, min(frac, 0.99)))))
                    ncal = min(ncal, nwin_tr)
                    if ncal >= 50:
                        start = nwin_tr - ncal
                        states_cal = np.empty((ncal, lags_eff, X_tr.shape[1]), dtype=np.float32)
                        for j in range(ncal):
                            k = start + j
                            states_cal[j] = X_tr[k:k + lags_eff]

                        if not hasattr(self.model, "predict_proba"):
                            raise AttributeError("DQNAgent missing predict_proba(); add softmax(Q) helper or expose it.")

                        p_cal = sanitize_proba(self.model.predict_proba(states_cal))
                        # V3 (DQN): under coverage intent, calibrate threshold on TRADE confidence only (ignore HOLD)
                        p_use = p_cal[:, [0, 2]] if bool(cov_intent) and p_cal.ndim == 2 and p_cal.shape[1] >= 3 else p_cal
                        coverage_thr = float(fit_coverage_threshold_on_calibration(p_use, rate))

                        if self._is_debug():
                            _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                            _ctx = "cv" if _in_cv_mode else ("real_sim" if bool(getattr(self, "_in_real_sim", False)) else "eval")
                            print(f"[Calib][Coverage][DQN] conf_thr={coverage_thr:.6f} target_active_rate={rate:.6f} cal_windows={int(ncal)} ctx={_ctx}")
            except Exception as _e:
                coverage_thr = None
                if self._is_debug():
                    print(f"[Calib][Coverage][DQN] skipped: {_e}")

        conf_thr = float(freeze_confidence_threshold(cfg_gate, default_conf, coverage_conf_thr=coverage_thr))

        # Tripwire consistency: coverage intent but no calibrated threshold => invalid (nan) metrics
        if cov_intent and (not np.isfinite(conf_thr)):
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_dqn_strategy:coverage_intent_missing_thr")

        # ---- Batch predict windows on TEST: proba -> action + confidence ----
        n = int(X_test.shape[0])
        raw_actions = np.ones(n, dtype=int)          # default HOLD (1)
        max_conf    = np.zeros(n, dtype=np.float32)  # default low confidence

        if n >= lags_eff:
            nwin = n - lags_eff + 1
            states = np.empty((nwin, lags_eff, X_test.shape[1]), dtype=np.float32)
            for j in range(nwin):
                states[j] = X_test[j:j + lags_eff]

            if not hasattr(self.model, "predict_proba"):
                raise AttributeError("DQNAgent missing predict_proba(); add softmax(Q) helper or expose it.")

            proba = sanitize_proba(self.model.predict_proba(states))

            # IMPORTANT (coverage intent):
            # DQN has an explicit HOLD action. If we take argmax over {sell,hold,buy},
            # target_active_rate/coverage gating cannot "pull trades up" when HOLD dominates.
            # Under coverage intent, compute confidence over TRADE actions only (sell/buy),
            # then apply conf_thr (and αβγ dynamic thr_vec) to that trade confidence.
            if bool(cov_intent):
                # trade_p: [sell, buy] only
                trade_p = proba[:, [0, 2]]
                trade_conf = trade_p.max(axis=1)
                trade_dir = np.asarray(np.argmax(trade_p, axis=1), dtype=int)  # 0=sell, 1=buy
                # Map back to action space: 0 -> SELL(0), 1 -> BUY(2)
                a = np.where(trade_dir == 0, 0, 2).astype(int)
                c = trade_conf
            else:
                a = np.asarray(np.argmax(proba, axis=1), dtype=int)
                c = proba.max(axis=1)

            # IMPORTANT: write outputs for BOTH paths
            raw_actions[lags_eff - 1:] = a
            max_conf[lags_eff - 1:]    = c

        # Apply the SAME style of confidence gating as supervised models:
        # below threshold => force HOLD/neutral (action=1)
        if np.isfinite(conf_thr):
            raw_actions[max_conf < float(conf_thr)] = 1

        # --- 5b) Trade-frequency control: minimum bars between position switches ---
        min_switch = int(cfg_local.get("dqn_min_switch_interval", 0))
        if min_switch > 1 and len(raw_actions) > 0:
            filtered = []
            current = int(raw_actions[0])
            last_switch_idx = 0

            for i, a in enumerate(raw_actions):
                if i == 0:
                    filtered.append(current)
                    continue
                if a != current and (i - last_switch_idx) < min_switch:
                    filtered.append(current)
                else:
                    filtered.append(a)
                    if a != current:
                        current = a
                        last_switch_idx = i

            raw_actions = np.asarray(filtered, dtype=int)

        # --- 5c) Map discrete actions to trading signals {-1,0,+1} ---
        action_to_signal = np.array([-1.0, 0.0, 1.0], dtype=float)
        preds = action_to_signal[raw_actions]

        # Assemble aligned eval frame (safe)
        result_df = test_data_full.copy()
        result_df["pred"] = preds

        # Expose confidence stats for real-sim GateDiag (debug-only consumer reads these attrs).
        try:
            self._last_conf_thr_used = float(conf_thr) if np.isfinite(conf_thr) else None
            self._last_conf_stats_max_conf = np.asarray(max_conf, dtype=np.float32)
        except Exception:
            pass


        # Keep confidence around for debug / audits (no harm; evaluator ignores it)
        try:
            result_df["max_conf"] = pd.Series(max_conf, index=test_data_full.index).astype(float).values
        except Exception:
            pass

        if "returns" not in result_df.columns:
            result_df["returns"] = self.data["returns"].reindex(result_df.index).astype(float)

        result_df = result_df.dropna(subset=["returns"])
        if result_df.index.tz is None:
            result_df.index = result_df.index.tz_localize("UTC")

        # Align DQN evaluation window to the expected start (avoid warmup leakage)
        eval_start = getattr(self, "_expected_eval_start", None)
        if eval_start is not None:
            result_df = result_df.loc[result_df.index >= eval_start].copy()

        # --- Edge-bar guard for DQN (avoid big session gaps / last bar) ---
        _idx = result_df.index
        if len(_idx) >= 2:
            gaps = pd.Series(_idx[1:] - _idx[:-1], index=_idx[:-1])
            exp = gaps.median()
            is_edge = gaps > (exp * 1.5)

            if self._is_debug():
                try:
                    _ctx = "cv" if bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False)) else ("real_sim" if bool(getattr(self, "_in_real_sim", False)) else "run")
                    _edge_idx = is_edge.index[is_edge]
                    _pred_ser = result_df["pred"]
                    _nz_before = int((_pred_ser != 0).sum())
                    _nz_edge = int((_pred_ser.reindex(_edge_idx).fillna(0) != 0).sum()) if len(_edge_idx) else 0
                    _nz_last = int(bool(_pred_ser.iloc[-1] != 0))
                    print(f"[EdgeGuardAudit][dqn] ctx={_ctx} exp={exp} edge_bars={int(is_edge.sum())} nz_before={_nz_before} nz_on_edge={_nz_edge} nz_last={_nz_last}")
                except Exception:
                    pass

            result_df.loc[is_edge.index[is_edge], "pred"] = 0
            result_df.iloc[-1, result_df.columns.get_loc("pred")] = 0

            if self._is_debug():
                try:
                    _nz_after = int((result_df["pred"] != 0).sum())
                    print(f"[EdgeGuardAudit][dqn] nz_after={_nz_after}")
                except Exception:
                    pass

        # First-10 trace for DQN (now has proba-derived confidence)
        if self._should_dump_decisions():
            try:
                _mc = None
                if "max_conf" in result_df.columns:
                    _mc = result_df["max_conf"].to_numpy(dtype=np.float32, copy=False)
                self._debug_dump_first_bars(
                    result_df.index,
                    raw_classes=None,
                    max_conf=_mc,
                    final_preds=result_df["pred"].values,
                    n=10,
                    label="dqn",
                )
            except Exception:
                pass

        if len(result_df) == 0:
            print("❌ No tradable rows left in DQN result after start cut.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_dqn_strategy:empty_result_after_cut")

        if (result_df["pred"] != 0).sum() == 0:
            print("ℹ️ DQN produced no trades in this window.")

        # Keep zero-lag here; compute_full_evaluation_metrics applies the 1-bar execution delay.
        result_df["pred"] = result_df["pred"].fillna(0.0)

        # 6) Evaluation (unified)
        try:
            cfg_adj = dict(cfg_gate)
            # persist the effective threshold so overlays/audits can see it
            cfg_adj["confidence_threshold"] = float(conf_thr)
            if coverage_thr is not None and np.isfinite(float(coverage_thr)):
                cfg_adj["_coverage_conf_thr"] = float(coverage_thr)

            if high_vol_thr_train is not None and cfg_adj.get("high_vol_thr") is None:
                cfg_adj["high_vol_thr"] = float(high_vol_thr_train)

            # Guarantee a valid spread series aligned to evaluated index
            try:
                if "spread" in getattr(self, "data", pd.DataFrame()).columns:
                    result_df["spread"] = self.data["spread"].reindex(result_df.index).astype(float).fillna(0.0)
                else:
                    result_df["spread"] = 0.0
            except Exception:
                result_df["spread"] = 0.0

            # Attach config for execution overlays (TWAP / kill-switch / etc.)
            try:
                result_df.attrs["features_config"] = dict(cfg_adj)
                result_df.attrs["debug_costs"] = bool(self._is_debug())
            except Exception:
                pass

            if bool(getattr(self, "trading_costs", True)):
                result_df = self._ensure_cost_columns(result_df, cfg_adj)
        except Exception:
            pass

        _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
        if _in_cv_mode:
            _eval_ctx = "cv:test_dqn_strategy"
        elif bool(getattr(self, "_in_real_sim", False)):
            _eval_ctx = "real_sim:test_dqn_strategy"
        else:
            _eval_ctx = "eval:test_dqn_strategy"

        metrics = compute_full_evaluation_metrics(
            df=result_df,
            trading_costs=self.trading_costs,
            slippage_factor=self.slippage_factor,
            eval_context=_eval_ctx,
        )
        
        # Capture trade-intent precision from evaluator (cheap scalar; safe in CV).
        try:
            _attrs = getattr(result_df, "attrs", {}) or {}
            self._last_precision_trade = float(_attrs.get("precision_trade", float("nan")))
            self._last_n_trade_preds = int(_attrs.get("n_trade_preds", 0) or 0)
        except Exception:
            self._last_precision_trade = float("nan")
            self._last_n_trade_preds = 0


        # Optional: annotate raw test slice (legacy, enabled only in DEBUG mode)
        if self._is_debug():
            metric_names = [
                "cstrategy", "outperformance", "creturns", "sharpe", "drawdown", "trades",
                "geo_mean_ann", "directional_accuracy", "precision_macro", "f1_macro",
                "active_rate", "profit_per_hit", "return_per_trade", "win_rate",
                "strategy_volatility", "kurtosis",
            ]
            for name, value in zip(metric_names, metrics):
                test_data[name] = value  # raw slice (for parity)

        # ----------------------------
        # Results storage (R4 + R3 cap)
        # ----------------------------
        if not getattr(self, "_in_optuna_cv", False):
            self.results = result_df.copy() if result_df is not None else None
            try:
                _es = getattr(self, "_expected_eval_start", None)
                self.results_full = (
                    test_data_full.loc[test_data_full.index >= _es].copy()
                    if (_es is not None and test_data_full is not None)
                    else (test_data_full.copy() if test_data_full is not None else None)
                )
            except Exception:
                self.results_full = test_data_full.copy() if test_data_full is not None else None
            self._cv_last_eval_df = None
        else:
            self._cv_last_eval_df = (result_df.copy() if (result_df is not None and not result_df.empty) else None)

            try:
                if self._cv_last_eval_df is not None and self._is_debug():
                    _cap = int(os.environ.get("CV_MAX_EVAL_FRAMES", "5"))
                    if _cap > 0 and len(self._cv_fold_eval_frames) < _cap:
                        self._cv_fold_eval_frames.append(self._cv_last_eval_df.copy())
                    try:
                        _max_keep = int((getattr(self, "config", {}) or {}).get("cv_max_fold_eval_frames", 3) or 3)
                    except Exception:
                        _max_keep = 3
                if _max_keep > 0 and len(self._cv_fold_eval_frames) > _max_keep:
                    self._cv_fold_eval_frames = self._cv_fold_eval_frames[-_max_keep:]
            except Exception:
                try:
                    if self._cv_last_eval_df is not None and self._is_debug():
                        self._cv_fold_eval_frames = [self._cv_last_eval_df.copy()]
                    else:
                        self._cv_fold_eval_frames = []
                except Exception:
                    self._cv_fold_eval_frames = []

            self.results = None
            self.results_full = None

        metrics = _safe_metrics_return(metrics, context="eval_block_2")
        return metrics

    

