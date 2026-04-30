"""Auto-extracted mixin — see composed.py for the full MLBacktester."""
from config import PIPELINE_CONSTANTS as _PC
from pipeline._imports import *  # noqa: F401,F403


class EnsembleMixin:
    """
    ensemble, adaptive, free

    Auto-extracted from MLBacktesterNoWFO.py lines 7535-9522.
    """
    def evaluate_strategy_adaptive_top3(
        self,
        best_params: dict,
        train_start,
        train_end,
        test_start,
        test_end,
        hit_thr: float = 0.45,
        window_days: int = 5,
    ):
        """
        Adaptive run over a single test month using Top-3 parameter sets:
        - Start with Top-1 (best_params).
        - If last 5 days' rolling hit-rate < hit_thr, switch to Top-2 from next bar.
        - After >=5 further days, if it fails again, switch to Top-3.
        Skips for DQN + Transformer-XGB-DQN ensemble (evaluate once).
        Prints/logs switch events if features_config['log_switch'] is True (default).
        """
        import pandas as pd
        from copy import deepcopy

        log_switch = bool(getattr(self, "features_config", {}).get("log_switch", True))

        model_type = (best_params or {}).get("model_type", getattr(self, "model_type", ""))
        if model_type in {"dqn"}:
            # excluded families for now → plain full-month eval
            return self.evaluate_strategy(best_params, train_start, train_end, test_start, test_end)

        # Pull Top-3 alternates (back-compat with old __top5_params files)
        top_alts = list(best_params.get("__top3_params") or best_params.get("__top5_params") or [])
        top_alts = top_alts[:2]  # only need 2 alternates (ranks 2 & 3)

        # Normalize alternates: ensure required keys survive the merge
        REQUIRED = ["model_type", "use_extended_features", "lags", "label_threshold", "confidence_threshold"]
        def _merge_params(base, alt):
            p = dict(base); p.update(deepcopy(alt or {}))
            for k in REQUIRED:
                if k not in p and k in base:
                    p[k] = base[k]
            return p

        # ---------- Leg #1: run Top-1 on full month ----------
        m1 = self.evaluate_strategy(best_params, train_start, train_end, test_start, test_end)
        df1 = getattr(self, "results", pd.DataFrame()).copy()
        if df1 is None or df1.empty:
            return m1

        # Eval anchor for monitoring (no look-ahead)
        first_eval_ts = getattr(self, "_expected_eval_start", None) or df1.index[0]

        # Rolling window in bars = window_days × bars_per_day (estimated on actual index)
        bpd = max(1, estimate_bars_per_day(df1.index))
        nwin = int(max(1, window_days) * bpd)

        # Precompute rolling hit-rate series & window counts (for readable prints)
        # ANCHOR: # Precompute rolling hit-rate series & window counts (for readable prints)
        # df1["pred"] is already the executed (shifted) series after compute_full_evaluation_metrics().
        pred_exec1 = df1["pred"]
        active1 = (pred_exec1 != 0).astype(int)
        correct1 = ((pred_exec1 * df1["returns"]) > 0).astype(int) * active1
        act_roll1 = active1.rolling(nwin, min_periods=1).sum()
        hit_roll1 = correct1.rolling(nwin, min_periods=1).sum()
        hr_series1 = (hit_roll1 / act_roll1.replace(0, pd.NA))

        # Find first switch timestamp t1 (if any), using only info up to each bar
        t1 = find_hit_rate_switch_idx(df1.loc[first_eval_ts:], nwin, thr=float(hit_thr), start_ts=first_eval_ts)

        if t1 is None:
            # No switch → keep Top-1 for whole month
            self._last_switch_log = []
            if log_switch:
                print(f"✅ No switch: Top-1 held entire month "
                    f"(window={window_days}d, bars/day≈{bpd}).")
            return m1

        # Compute resume timestamp (next bar after t1)
        idx = df1.index
        try:
            pos = idx.get_loc(pd.to_datetime(t1))
        except Exception:
            pos = max(0, idx.searchsorted(pd.to_datetime(t1)))
        if pos >= len(idx) - 1:
            # Trigger at final bar → effectively no room to switch
            self._last_switch_log = [{"at": str(pd.to_datetime(t1)), "to_rank": 2, "note": "triggered_at_end"}]
            if log_switch:
                # window stats at t1
                t1_ts = pd.to_datetime(t1)
                trades_win = int(act_roll1.loc[:t1_ts].iloc[-1]) if len(act_roll1.loc[:t1_ts]) else 0
                hits_win = int(hit_roll1.loc[:t1_ts].iloc[-1]) if len(hit_roll1.loc[:t1_ts]) else 0
                hr_val = float(hr_series1.loc[:t1_ts].iloc[-1]) if len(hr_series1.loc[:t1_ts]) else float("nan")
                print(f"⚠️ Switch triggered at end-of-month ({t1_ts}) but no bars remain. "
                    f"Window={window_days}d | hit-rate={hr_val:.2%} on {trades_win} trades | hits={hits_win}.")
            self.results = df1
            return m1

        start2 = idx[pos + 1]
        # Stats at t1 for logging
        t1_ts = pd.to_datetime(t1)
        trades_win = int(act_roll1.loc[:t1_ts].iloc[-1]) if len(act_roll1.loc[:t1_ts]) else 0
        hits_win = int(hit_roll1.loc[:t1_ts].iloc[-1]) if len(hit_roll1.loc[:t1_ts]) else 0
        hr_val = float(hr_series1.loc[:t1_ts].iloc[-1]) if len(hr_series1.loc[:t1_ts]) else float("nan")

        if log_switch:
            print(f"🔁 [Switch #1] {t1_ts} → switching to Top-2 "
                f"(window={window_days}d | hit-rate={hr_val:.2%} on {trades_win} active trades "
                f"< {hit_thr:.0%}); retrain_end={t1_ts}, resume={start2}.")

        # ---------- Leg #2: train Top-2 up to t1; test from start2..end ----------
        p2 = _merge_params(best_params, (top_alts[0] if len(top_alts) >= 1 else {}))
        _ = self.evaluate_strategy(p2, train_start, pd.to_datetime(t1), pd.to_datetime(start2), test_end)
        df2 = getattr(self, "results", pd.DataFrame()).copy()
        self._last_switch_log = [{
            "at": str(t1_ts),
            "to_rank": 2,
            "window_days": int(window_days),
            "hit_rate": float(hr_val) if pd.notna(hr_val) else None,
            "trades_in_window": int(trades_win),
            "hits_in_window": int(hits_win),
            "resume_ts": str(pd.to_datetime(start2)),
        }]

        # Decide on second switch (to Top-3) — only after another full window of Top-2 data
        t2 = None
        if len(top_alts) >= 2 and not df2.empty:
            bpd2 = max(1, estimate_bars_per_day(df2.index))
            nwin2 = int(max(1, window_days) * bpd2)

            # Precompute Top-2 rolling stats for readable prints
            pred_exec2 = df2["pred"]
            active2 = (pred_exec2 != 0).astype(int)
            correct2 = ((pred_exec2 * df2["returns"]) > 0).astype(int) * active2
            act_roll2 = active2.rolling(nwin2, min_periods=1).sum()
            hit_roll2 = correct2.rolling(nwin2, min_periods=1).sum()
            hr_series2 = (hit_roll2 / act_roll2.replace(0, pd.NA))

            # enforce cooldown/evidence: require at least one full window before checking
            df2_chk = df2.iloc[nwin2 - 1 :] if len(df2) >= nwin2 else df2.iloc[0:0]
            t2 = find_hit_rate_switch_idx(
                df2_chk,
                nwin2,
                thr=float(hit_thr),
                start_ts=(df2_chk.index[0] if len(df2_chk) else None),
            )

        # Build combined and hard-reset execution at the switch bar(s) to avoid carry-over via pred.shift(1)
        if t2 is None:
            combined = pd.concat([df1.loc[:t1_ts], df2], axis=0).sort_index()
            # reset pred at t1 so first bar of df2 executes with 0 prev position
            if t1_ts in combined.index:
                if "raw_pred" in combined.columns:
                    combined.loc[t1_ts, "raw_pred"] = 0
                combined.loc[t1_ts, "pred"] = 0
            self.results = combined
            return compute_full_evaluation_metrics(
                df=combined,
                trading_costs=self.trading_costs,
                slippage_factor=self.slippage_factor,
            )

        # ---------- Leg #3: train Top-3 up to t2; test from next bar ----------
        idx2 = df2.index
        try:
            pos2 = idx2.get_loc(pd.to_datetime(t2))
        except Exception:
            pos2 = max(0, idx2.searchsorted(pd.to_datetime(t2)))
        if pos2 >= len(idx2) - 1:
            combined = pd.concat([df1.loc[:t1_ts], df2], axis=0).sort_index()
            if t1_ts in combined.index:
                if "raw_pred" in combined.columns:
                    combined.loc[t1_ts, "raw_pred"] = 0
                combined.loc[t1_ts, "pred"] = 0
            self.results = combined
            self._last_switch_log.append({"at": str(pd.to_datetime(t2)), "to_rank": 3, "note": "triggered_at_end"})
            if log_switch:
                t2_ts = pd.to_datetime(t2)
                # window stats at t2 on Top-2 series if available
                if 'hr_series2' in locals() and t2_ts in hr_series2.index:
                    trades2 = int(act_roll2.loc[:t2_ts].iloc[-1])
                    hits2 = int(hit_roll2.loc[:t2_ts].iloc[-1])
                    hr2 = float(hr_series2.loc[:t2_ts].iloc[-1])
                    print(f"⚠️ Switch #2 triggered at end-of-month ({t2_ts}) but no bars remain. "
                        f"Window={window_days}d | hit-rate={hr2:.2%} on {trades2} trades | hits={hits2}.")
            return compute_full_evaluation_metrics(
                df=combined,
                trading_costs=self.trading_costs,
                slippage_factor=self.slippage_factor,
            )

        start3 = idx2[pos2 + 1]
        t2_ts = pd.to_datetime(t2)

        if log_switch:
            # pretty window stats on Top-2 at t2
            if 'hr_series2' in locals() and t2_ts in hr_series2.index:
                trades2 = int(act_roll2.loc[:t2_ts].iloc[-1])
                hits2 = int(hit_roll2.loc[:t2_ts].iloc[-1])
                hr2 = float(hr_series2.loc[:t2_ts].iloc[-1])
            else:
                trades2, hits2, hr2 = 0, 0, float("nan")
            print(f"🔁 [Switch #2] {t2_ts} → switching to Top-3 "
                f"(window={window_days}d | hit-rate={hr2:.2%} on {trades2} active trades "
                f"< {hit_thr:.0%}); retrain_end={t2_ts}, resume={start3}.")

        p3 = _merge_params(best_params, (top_alts[1] if len(top_alts) >= 2 else {}))
        _ = self.evaluate_strategy(p3, train_start, pd.to_datetime(t2), pd.to_datetime(start3), test_end)
        df3 = getattr(self, "results", pd.DataFrame()).copy()
        self._last_switch_log.append({
            "at": str(t2_ts),
            "to_rank": 3,
            "window_days": int(window_days),
            "hit_rate": float(hr2) if pd.notna(hr2) else None,
            "trades_in_window": int(trades2),
            "hits_in_window": int(hits2),
            "resume_ts": str(pd.to_datetime(start3)),
        })

        combined = pd.concat([df1.loc[:t1_ts], df2.loc[:t2_ts], df3], axis=0).sort_index()
        # reset pred at switch bars to prevent execution carry-over across legs
        if t1_ts in combined.index:
            if "raw_pred" in combined.columns:
                combined.loc[t1_ts, "raw_pred"] = 0
            combined.loc[t1_ts, "pred"] = 0
            
        if t2_ts in combined.index:
            if "raw_pred" in combined.columns:
                combined.loc[t2_ts, "raw_pred"] = 0
            combined.loc[t2_ts, "pred"] = 0

        self.results = combined
        return compute_full_evaluation_metrics(
            df=combined,
            trading_costs=self.trading_costs,
            slippage_factor=self.slippage_factor,
        )
        
    def free(self, release_data: bool = False):
        """Aggressively release memory held by this backtester instance.

        - During Optuna CV / repeated trials, call free(release_data=False) so self.data survives.
        - After finishing a model / repeat (when the instance won't be reused), call free(release_data=True).
        """

        # 0) Drop TF/Keras models first (graphs/buffers)
        for _attr in ("model", "cnn", "lstm", "transformer"):
            try:
                if hasattr(self, _attr):
                    delattr(self, _attr)
            except Exception:
                pass

        # 1) Drop run artifacts (safe in CV)
        for _attr in (
            "results", "results_full", "bar_concat", "trade_log",
            "_cv_last_eval_df", "_cv_fold_eval_frames",
            "_ensemble_win_cache", "_seq_cache",
            "_optuna_best_for_wfo", "_optuna_top5_for_wfo", "_optuna_consensus_pool_for_wfo",

        ):
            try:
                if hasattr(self, _attr):
                    delattr(self, _attr)
            except Exception:
                pass

        # 2) Clear per-run caches
        try:
            if hasattr(self, "_feat_cache") and isinstance(self._feat_cache, dict):
                self._feat_cache.clear()
        except Exception:
            pass

        # FeatureBank: safe to clear in CV (it will rebuild)
        for _attr in ("_feature_bank_full", "_feature_bank_meta", "_feature_bank_key", "_feature_bank_src"):
            try:
                if hasattr(self, _attr):
                    delattr(self, _attr)
            except Exception:
                pass

        # 3) Only release the dataset when explicitly requested
        if release_data:
            for _attr in ("data", "data_raw", "raw_data", "df_1h", "df_4h"):
                try:
                    if hasattr(self, _attr):
                        # Prefer setting None to keep attribute shape predictable if anything touches it later
                        setattr(self, _attr, None)
                except Exception:
                    pass

        # 4) Kill cached joblib/loky workers (optional; can be heavy per-trial)
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        except Exception:
            pass

        # 5) Close matplotlib figures
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass

        # 6) Clear DL backend + GC
        _hard_free()



        
        
    def _clear_feature_cache(self):
        """Clear per-run feature cache; keep FeatureBank (it is self-keyed)."""
        try:
            # Main engineered-slice cache
            if hasattr(self, "_feat_cache") and isinstance(self._feat_cache, dict):
                self._feat_cache.clear()

            # Patch 2: per-entry byte accounting (telemetry only)
            if hasattr(self, "_feat_cache_bytes") and isinstance(getattr(self, "_feat_cache_bytes", None), dict):
                self._feat_cache_bytes.clear()

            # Reset optional stats used by [FEAT_CACHE] logging (if present)
            for _k in (
                "_feat_cache_hits",
                "_feat_cache_misses",
                "_feat_cache_est_bytes",      # legacy cumulative counter (if present)
                "_feat_cache_cur_bytes",      # truthful: current retained bytes
                "_feat_cache_evictions",      # Patch 3: eviction counter (if present)
            ):
                if hasattr(self, _k):
                    setattr(self, _k, 0)

        except Exception:
            pass


    def test_ensemble_strategy(
        self,
        train_start,
        train_end,
        test_start,
        test_end,
        lags,
        label_threshold,
        ensemble_config,
        model_type,
    ):
        """
        Wrapper around the ensemble backtest to prevent TensorFlow/Keras
        graph/session accumulation across:
          - Optuna CV folds/trials
          - real_trading_simulation months

        Ensembles train deep heads (CNN/LSTM/Transformer) and must receive the
        same cleanup treatment as standalone deep models.
        """
        try:
            # Always safe: cleanup runs on exit via _persist_results_guard()
            # and does not change outputs, only releases memory.
            self._tf_cleanup_do = True
            self._tf_cleanup_del_model = True
        except Exception:
            pass

        # Ensure cleanup runs even on early returns/exceptions inside the core.
        with self._persist_results_guard(persist_results=True):
            return self._test_ensemble_strategy_core(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                lags=lags,
                label_threshold=label_threshold,
                ensemble_config=ensemble_config,
                model_type=model_type,
            )


    def _test_ensemble_strategy_core(
        self,
        train_start,
        train_end,
        test_start,
        test_end,
        lags,
        label_threshold,
        ensemble_config,
        model_type,
    ):
        """
        Universal backtest for ensemble models (CNN+LSTM+XGB, Adaptive Regime).
        Handles: feature prep, scaling, labels, windowing, fitting, prediction, metrics.

        Returns
        -------
        tuple[float, ...]   # 16 metrics in fixed order; NaNs/-9999 on bailouts
        """
        # Clear any sticky feature cache from previous evals (ensemble path)
        self._clear_feature_cache()
        
        # ---- tiny local helper if not imported elsewhere ----
        def filter_params(d: dict, prefix: str) -> dict:
            if not isinstance(d, dict):
                return {}
            L = len(prefix)
            return {k[L:]: v for k, v in d.items() if isinstance(k, str) and k.startswith(prefix)}

        # ----------------------------
        # Data selection & preparation
        # ----------------------------
        full_data  = self.data
        train_data = full_data.loc[train_start:train_end]

        true_test_start = pd.to_datetime(test_start)
        test_end        = pd.to_datetime(test_end)
        model_label     = str(model_type or "ensemble")
        
        self._proba_came_from_dqn_fusion = False

        # merge flags/knobs from features_config and ensemble_config (ensemble overrides features)
        cfg_f  = (getattr(self, "features_config", {}) or {}).copy()
        ens_cf = (ensemble_config or {}).copy()
        merged = {**cfg_f, **ens_cf}
        
        # Always use fused path when DQN is present
        merged.setdefault("use_dqn_fusion", True)
        

        # Trace context early (avoid NameError in downstream guardrails/logs)
        in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
        
        # ------------------------------------------------------------
        # Mirror test_strategy() defaults/normalization for fairness:
        # 1) Default gating_mode to coverage unless explicitly set
        # 2) In real-sim, if target_active_rate is set but gating_mode is threshold,
        #    auto-switch to coverage so activity-rate tuning actually applies
        # 3) Mirror eval cost knobs (eval_use_trading_costs / trading_costs, slippage_factor)
        # ------------------------------------------------------------
        if "gating_mode" not in merged:
            merged["gating_mode"] = "coverage"


        # Prevent stale coverage thresholds leaking across models/months.
        # Ensures coverage intent without a fresh calibration trips the NaN tripwire.
        try:
            if is_coverage_intent(merged):
                self._coverage_conf_thr = None
                if hasattr(self, "_deep_coverage_thr"):
                    delattr(self, "_deep_coverage_thr")
        except Exception:
            pass

        try:
            in_real = bool(getattr(self, "_in_real_sim", False))
            gmode = str(merged.get("gating_mode", merged.get("gate_mode", "threshold"))).lower()
            tgt = float(merged.get("target_active_rate", merged.get("target_coverage", 0.0)) or 0.0)
            if in_real and (not in_cv) and gmode in ("threshold", "", "none") and tgt > 0.0:
                merged["gating_mode"] = "coverage"
                if self._is_debug():
                    print(f"[Gate] Auto-enabled gating_mode='coverage' for ensemble real-sim (target_active_rate={tgt:.2f}).")
        except Exception:
            pass

        try:
            if not getattr(self, "_trading_costs_locked", False):
                if "eval_use_trading_costs" in merged:
                    self.trading_costs = bool(merged.get("eval_use_trading_costs", self.trading_costs))
                elif "trading_costs" in merged:
                    self.trading_costs = bool(merged.get("trading_costs", self.trading_costs))
        except Exception:
            pass

        try:
            if "slippage_factor" in merged:
                self.slippage_factor = float(merged.get("slippage_factor", self.slippage_factor))
        except Exception:
            pass
        
        # --- Confidence threshold handling (avoid pre-calibration tripwire in CV) ---
        # In ensemble CV, the coverage threshold is computed AFTER fitting (train-tail calibration).
        # So we *do not* call _resolve_conf_thr() here (it would run before calibration and spam
        # TRIPWIRE logs / potentially force a placeholder cap). Instead, we carry the requested
        # default through the pipeline and let the later gating stage prefer the calibrated
        # self._coverage_conf_thr when gating_mode='coverage'.
        default_conf = float(ens_cf.get("confidence_threshold", cfg_f.get("confidence_threshold", 0.50)))
        try:
            merged["confidence_threshold"] = float(default_conf)
        except Exception:
            pass
        try:
            # keep a trace of the requested init threshold for diagnostics
            self._last_conf_thr_init = float(default_conf)
        except Exception:
            pass
        try:
            merged.setdefault("model_type", model_label)
        except Exception:
            pass

        
        # --- Warm-up bars: use the SAME feature config as other models ---
        # For classical/CNN/LSTM/Transformer we base warm-up on the feature pipeline
        # (lags_range, lag_depth, triple-barrier horizon, etc.). Use that here too,
        # and only tag the model_type so compute_required_test_warmup_bars() knows
        # which branch to use.
        cfg_for_warmup = dict(self.features_config or {})
        cfg_for_warmup["model_type"] = model_label

        warmup_need = int(compute_required_test_warmup_bars(cfg_for_warmup))

        # account for final embargo so pre-roll remains outside test month
        embargo_n = int(cfg_f.get("final_embargo_bars", 0) or 0)
        _total_warmup_need = max(0, warmup_need + embargo_n)

        # account for final embargo so pre-roll remains outside test month
        embargo_n = int(cfg_f.get("final_embargo_bars", 0) or 0)
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

        sess_mode = str(cfg_f.get("session_filter_mode", "both")).lower()

        if not hasattr(self, "_ny_mask") or self._ny_mask is None:
            try:
                full_idx = pd.to_datetime(self.data.index, utc=True, errors="coerce")
                _ny_times = full_idx.tz_convert("America/New_York")
                self._ny_mask = pd.Series((_ny_times.hour >= 2) & (_ny_times.hour <= 13), index=full_idx)
            except Exception as _e:
                print(f"⚠️ Lazy NY mask build failed in ensemble path: {_e}")
                self._ny_mask = pd.Series(True, index=self.data.index)

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

        # Optional final embargo — disable for CV mini-folds
        try:
            embargo_n = int(cfg_f.get("final_embargo_bars", 0))
            if in_cv:
                embargo_n = 0
            if embargo_n > 0 and len(test_data) > embargo_n:
                test_data = test_data.iloc[embargo_n:].copy()
                print(f"[Embargo] Dropped first {embargo_n} test bars (ensemble, non-CV).")
        except Exception as e:
            print(f"⚠️ final_embargo_bars handling failed (ensemble): {e}")

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
        
        if in_cv:
            print(f"[CV/ENSEMBLE] Eval anchor forced to fold start: {first_eval_ts} | ...")

        if in_cv:
            print(f"[CV/ENSEMBLE] Eval anchor forced to fold start: {first_eval_ts} | test_len={len(test_data)} | warmup_need={_total_warmup_need}")

        if first_eval_ts is None:
            print("❌ No tradable bar found in test window (ensemble).")
            # Bail safely with fixed 16-metric contract.
            # IMPORTANT: never persist heavy frames during Optuna CV.
            if in_cv:
                self.results = None
                self.results_full = None
                self._cv_last_eval_df = None
            else:
                self.results = pd.DataFrame()
                self.results_full = self.results
                self._cv_last_eval_df = None
            return _safe_metrics_return(
                (np.nan,) * N_METRICS,
                context="test_ensemble_strategy:no_tradable",
            )
        self._expected_eval_start = first_eval_ts

        # ----------------------------
        # Feature engineering + scaling
        # ----------------------------
        cfg = self.apply_feature_defaults()
        lag_depth    = cfg.get("lag_depth", 1)
        roll_windows = cfg.get("roll_windows", [5])
        lags_eff = int(cfg.get("lags_range", cfg.get("lags", lags)))
        if getattr(self, "_is_debug", lambda: False)():
            print(f"[ENSEMBLE] effective_lags={lags_eff} (cfg-precedence)")

        train_data, features = self.prepare_features(
            train_data, lags_eff, lag_depth=lag_depth, roll_windows=roll_windows
        )
        test_data, _ = self.prepare_features(
            test_data, lags_eff, lag_depth=lag_depth, roll_windows=roll_windows
        )

        # De-dup columns
        train_data = train_data.loc[:, ~train_data.columns.duplicated()]
        test_data  = test_data.loc[:,  ~test_data.columns.duplicated()]

        # Replace infs & drop NaNs in *active* features to keep indices aligned
        for df in (train_data, test_data):
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            if features:
                df.dropna(subset=features, inplace=True)

        # Scale (fit on train → apply to test)
        train_data, means, stds = self.scale_features(train_data, features)
        test_data,  _,    _     = self.scale_features(test_data,  features, means, stds)

        # ----------------------------
        # Labels (T+1) and alignment
        # ----------------------------
        # _ret_fwd_tr = train_data["returns"].shift(-1)
        # train_data  = train_data.loc[_ret_fwd_tr.notna()].copy()
        # y_train     = self.label_with_neutral(_ret_fwd_tr.loc[train_data.index], threshold=float(label_threshold))

        # _ret_fwd_te = test_data["returns"].shift(-1)
        # test_data   = test_data.loc[_ret_fwd_te.notna()].copy()
        # y_test      = self.label_with_neutral(_ret_fwd_te.loc[test_data.index],  threshold=float(label_threshold))
        # Labels and alignment
        
        # ----------------------------
        # Labels and alignment
        # ----------------------------
        cfg_lbl = dict(merged)

        # pick a close price column for triple-barrier
        if "price" in train_data.columns:
            _price_col = "price"
        elif "mid_close" in train_data.columns:
            _price_col = "mid_close"
        elif "close" in train_data.columns:
            _price_col = "close"
        elif {"ask_close", "bid_close"}.issubset(train_data.columns):
            # build a mid-close if only bid/ask are present
            train_data = train_data.copy()
            test_data  = test_data.copy()
            train_data["__mid_close__"] = (train_data["ask_close"] + train_data["bid_close"]) / 2.0
            test_data["__mid_close__"]  = (test_data["ask_close"]  + test_data["bid_close"])  / 2.0
            _price_col = "__mid_close__"
        else:
            _price_col = None  # only used if triple-barrier is on

        tb_on_lbl = bool(cfg_lbl.get("use_triple_barrier", False))

        # If triple-barrier is requested but we can't resolve a price series, fall back safely.
        if tb_on_lbl and (_price_col is None or (_price_col not in train_data.columns) or (_price_col not in test_data.columns)):
            print("⚠️ TripleBarrier enabled but no price column found; falling back to return-based labels.")
            tb_on_lbl = False

        if tb_on_lbl:
            y_train = triple_barrier_labels(
                close=train_data[_price_col],
                pt_mult=float(cfg_lbl.get("tb_pt_mult", 1.5)),
                sl_mult=float(cfg_lbl.get("tb_sl_mult", 1.0)),
                max_holding=int(cfg_lbl.get("tb_max_holding", 48)),
                neutral_zone=float(cfg_lbl.get("tb_neutral_zone", 0.0)),
                neutral_zone_is_sigma=bool(cfg_lbl.get("tb_neutral_zone_is_sigma", False)),
            ).astype(int)

            y_test = triple_barrier_labels(
                close=test_data[_price_col],
                pt_mult=float(cfg_lbl.get("tb_pt_mult", 1.5)),
                sl_mult=float(cfg_lbl.get("tb_sl_mult", 1.0)),
                max_holding=int(cfg_lbl.get("tb_max_holding", 48)),
                neutral_zone=float(cfg_lbl.get("tb_neutral_zone", 0.0)),
                neutral_zone_is_sigma=bool(cfg_lbl.get("tb_neutral_zone_is_sigma", False)),
            ).astype(int)

            # --- Debug: confirm triple-barrier config & class balance (safe: no `params`) ---
            if bool(cfg_lbl.get("print_labeling_debug", False)):
                from collections import Counter
                _pt = float(cfg_lbl.get("tb_pt_mult", 1.5))
                _sl = float(cfg_lbl.get("tb_sl_mult", 1.0))
                _mh = int(cfg_lbl.get("tb_max_holding", 48))
                _nz = float(cfg_lbl.get("tb_neutral_zone", 0.0))
                print(f"[Labeling] TripleBarrier ON | pt={_pt}×σ sl={_sl}×σ hold={_mh} bars neutral={_nz}")
                print(f"[Labeling] Train counts={dict(Counter(y_train))} | Test counts={dict(Counter(y_test))}")
        else:
            _ret_fwd_tr = train_data["returns"].shift(-1)
            train_data  = train_data.loc[_ret_fwd_tr.notna()].copy()
            y_train     = self.label_with_neutral(_ret_fwd_tr.loc[train_data.index], threshold=float(cfg_lbl.get("label_threshold", 0.0)))

            _ret_fwd_te = test_data["returns"].shift(-1)
            test_data   = test_data.loc[_ret_fwd_te.notna()].copy()
            y_test      = self.label_with_neutral(_ret_fwd_te.loc[test_data.index],  threshold=float(cfg_lbl.get("label_threshold", 0.0)))

        if y_train is None or y_test is None or len(y_train) == 0 or len(y_test) == 0:
            print("⚠️ Labels empty in ensemble strategy. Skipping fold.")
            # Avoid returning stale frames from previous months/folds.
            if in_cv:
                self.results = None
                self.results_full = None
                self._cv_last_eval_df = None
            else:
                self.results = pd.DataFrame()
                self.results_full = self.results
                self._cv_last_eval_df = None
            return _safe_metrics_return(
                (np.nan,) * N_METRICS,
                context="test_ensemble_strategy:empty_labels",
            )

        # Basic label diagnostics
        u_tr, c_tr = np.unique(y_train, return_counts=True)
        u_te, c_te = np.unique(y_test,  return_counts=True)
        if self._is_debug():
            print("Label counts (train):", dict(zip(u_tr, c_tr)), f"| thr={label_threshold}")
            print("Label counts (test): ", dict(zip(u_te, c_te)))

        # Directional-only label mix guard for ensemble folds
        if not self._guard_label_mix_directional(
            y_train,
            label_threshold=label_threshold,
            context="ENSEMBLE_FOLD",
            min_dir_samples=5,
        ):
            # Avoid returning stale frames from previous months/folds.
            if in_cv:
                self.results = None
                self.results_full = None
                self._cv_last_eval_df = None
            else:
                self.results = pd.DataFrame()
                self.results_full = self.results
                self._cv_last_eval_df = None
            return _safe_metrics_return(
                (np.nan,) * N_METRICS,
                context="test_ensemble_strategy:label_mix_guard",
            )

        # Defragment before attaching labels to reduce pandas fragmentation warnings
        train_data = train_data.copy()
        test_data = test_data.copy()
        train_data["label"] = y_train.astype(int)
        test_data["label"] = y_test.astype(int)

        input_shape = (int(lags_eff), len(features))
        
        high_vol_thr_train = None
        try:
            if "returns" in train_data.columns:
                _cfg_cost_src = getattr(self, "features_config", {}) or {}
                vol_w = int(_cfg_cost_src.get("vol_window_bars", _PC["vol_window_bars"]))
                qhi   = float(_cfg_cost_src.get("high_vol_q", _PC["high_vol_q"]))
                _rv_tr = realized_vol(train_data["returns"].astype(float), window=vol_w)
                _rv_tr = _rv_tr.dropna()
                if len(_rv_tr) > 0:
                    high_vol_thr_train = float(_rv_tr.quantile(qhi))
        except Exception:
            high_vol_thr_train = None



        # -------------------------------------
        # Windowing helpers (vectorized & cache)
        # -------------------------------------
        def _fallback_make_windows(df, feats, window):
            X_seq, _, idx = self._create_sliding_windows(df, feats, window_size=int(window))
            X_flat = df[feats].iloc[idx].values
            y_arr  = df["label"].iloc[idx].values
            return (
                X_seq.astype(np.float32, copy=False),
                y_arr.astype(np.int64,  copy=False),
                idx,
                X_flat.astype(np.float32, copy=False),
            )

        if not hasattr(self, "_ensemble_win_cache"):
            self._ensemble_win_cache = {}

        maker = getattr(self, "_ensemble_make_windows", None)
        make_windows = (lambda df: maker(df, features, int(lags_eff))) if callable(maker) \
                    else (lambda df: _fallback_make_windows(df, features, int(lags_eff)))

        # Deep windowing (train / test)
        #
        # IMPORTANT (OOM fix):
        # Apply stride/cap on the *raw bars* before creating sliding windows.
        # Caching full train windows across months is also disabled here (it explodes RAM).
        max_train_windows = ens_cf.get("ensemble_deep_max_train_windows", 10000)
        train_stride = ens_cf.get("ensemble_deep_train_stride", ens_cf.get("ensemble_train_stride", 3))

        try:
            if max_train_windows is not None:
                max_train_windows = int(max_train_windows)
        except Exception:
            max_train_windows = 10000
            
        try:
            train_stride = int(train_stride) if train_stride is not None else 1
        except Exception:
            train_stride = 1
        if train_stride < 1:
            train_stride = 1

        def _start_idx_for_last_strided_windows(n_rows, win, stride, max_windows):
            if max_windows is None:
                return 0
            try:
                max_windows = int(max_windows)
            except Exception:
                return 0
            if max_windows <= 0:
                return 0
            if n_rows <= win:
                return 0
            total_windows = n_rows - win + 1
            if stride <= 1:
                need = min(total_windows, max_windows)
                start_window = total_windows - need
            else:
                total_strided = (total_windows + stride - 1) // stride
                need = min(total_strided, max_windows)
                start_window = (total_strided - need) * stride
            return max(0, int(start_window))

        _start_idx = _start_idx_for_last_strided_windows(len(train_data), int(lags_eff), int(train_stride), max_train_windows)
        if _start_idx > 0:
            train_data = train_data.iloc[_start_idx:].copy()
 
        X_seq_train, y_train_win, idx_train, X_flat_train = make_windows(train_data)
        X_seq_test,  y_test_win,  idx_test,  X_flat_test  = make_windows(test_data)
 

        if X_seq_train.shape[0] == 0 or X_seq_test.shape[0] == 0:
            print("⚠️ Ensemble produced zero windows. Skipping fold.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:zero_windows")

        if self._is_debug():
            print(f"[ENSEMBLE-CV] X_seq_test={getattr(X_seq_test,'shape',None)} "
                f"X_flat_test={getattr(X_flat_test,'shape',None)} y_test_win={getattr(y_test_win,'shape',None)}")

        # Extra safety: apply stride/cap on the already-windowed arrays (train only)
        if train_stride > 1:
            X_seq_train  = X_seq_train[::train_stride]
            X_flat_train = X_flat_train[::train_stride]
            y_train_win  = y_train_win[::train_stride]
        if X_seq_train.shape[0] > max_train_windows:
            X_seq_train  = X_seq_train[-max_train_windows:]
            X_flat_train = X_flat_train[-max_train_windows:]
            y_train_win  = y_train_win[-max_train_windows:]

        # ==============================
        # ENSEMBLE: CNN + LSTM + XGBoost
        # ==============================
        if model_type == "ensemble_cnn_lstm_xgboost":
            cnn_cfg  = filter_params(ens_cf, "cnn_")
            lstm_cfg = filter_params(ens_cf, "lstm_")
            xgb_cfg  = filter_params(ens_cf, "xgb_")

            # CV-friendly caps (fast mode for heads + sane XGB defaults)
            if in_cv:
                # Global deep caps (fallback)
                cv_epochs_global = int(cfg_f.get("deep_cv_max_epochs", 12))
                cv_bs_global     = int(cfg_f.get("deep_cv_batch_size", 256))
                cv_pat_global    = int(cfg_f.get("deep_cv_patience", 6))

                # Extra-strict caps for ensemble heads
                cv_epochs_cnn  = int(cfg_f.get("cnn_ens_cv_max_epochs",  cv_epochs_global))
                cv_epochs_lstm = int(cfg_f.get("lstm_ens_cv_max_epochs", cv_epochs_global))
                cv_bs_cnn      = cv_bs_global
                cv_bs_lstm     = cv_bs_global

                # Base window cap for ensemble heads
                base_max_win = int(ens_cf.get("ensemble_deep_max_train_windows", 10000))
                max_win_cnn  = int(cfg_f.get("cnn_ens_cv_max_train_windows",  base_max_win))
                max_win_lstm = int(cfg_f.get("lstm_ens_cv_max_train_windows", base_max_win))

                # Coarser stride only for CV
                ens_cf.setdefault("ensemble_train_stride", 3)

                # Heads (enable real ES + time caps in CV)
                cnn_cfg = dict(cnn_cfg)
                cnn_cfg.setdefault("eval_mode", "cv_fast")
                cnn_cfg.setdefault("cnn_use_early_stopping", True)
                # clip trial-level epochs/batch to ensemble caps
                cnn_cfg["cnn_epochs"] = min(int(cnn_cfg.get("cnn_epochs", cv_epochs_cnn)), cv_epochs_cnn)
                cnn_cfg["cnn_batch_size"] = min(int(cnn_cfg.get("cnn_batch_size", cv_bs_cnn)), cv_bs_cnn)
                cnn_cfg.setdefault("cnn_patience", cv_pat_global)
                cnn_cfg["deep_max_train_windows"] = min(
                    int(cnn_cfg.get("deep_max_train_windows", base_max_win)),
                    max_win_cnn,
                )

                lstm_cfg = dict(lstm_cfg)
                lstm_cfg.setdefault("eval_mode", "cv_fast")
                lstm_cfg.setdefault("lstm_use_early_stopping", True)
                lstm_cfg["lstm_epochs"] = min(int(lstm_cfg.get("lstm_epochs", cv_epochs_lstm)), cv_epochs_lstm)
                lstm_cfg["lstm_batch_size"] = min(int(lstm_cfg.get("lstm_batch_size", cv_bs_lstm)), cv_bs_lstm)
                lstm_cfg.setdefault("lstm_patience", cv_pat_global)
                lstm_cfg["deep_max_train_windows"] = min(
                    int(lstm_cfg.get("deep_max_train_windows", base_max_win)),
                    max_win_lstm,
                )

                # XGB (modern device semantics + ES)
                xgb_cfg = dict(xgb_cfg)
                xgb_cfg.setdefault("n_estimators", min(int(xgb_cfg.get("n_estimators", 400)), 400))
                xgb_cfg.setdefault("n_jobs", int(xgb_cfg.get("n_jobs", 3)))
                xgb_cfg.setdefault("xgb_eval_fraction", float(xgb_cfg.get("xgb_eval_fraction", 0.10)))
                xgb_cfg.setdefault("xgb_early_stopping_rounds", int(xgb_cfg.get("xgb_early_stopping_rounds", 50)))
                xgb_cfg.setdefault("use_oof_meta", False)
                xgb_cfg.setdefault("oof_splits", 3)

                # Match global XGB policy: env-gated GPU (XGB_USE_GPU=1) else CPU.
                use_gpu = (os.environ.get("XGB_USE_GPU", "0") == "1")
                xgb_cfg.setdefault("tree_method", "hist")
                xgb_cfg.pop("predictor", None)
                if use_gpu:
                    xgb_cfg["device"] = os.environ.get("XGB_DEVICE", "cuda")
                else:
                    xgb_cfg.pop("device", None)
            else:
                # 🔹 Non-CV path (refit + final evaluation): ALWAYS use OOF stacking
                xgb_cfg = dict(xgb_cfg)
                xgb_cfg.setdefault("n_estimators", min(int(xgb_cfg.get("n_estimators", 400)), 400))
                xgb_cfg.setdefault("n_jobs", int(xgb_cfg.get("n_jobs", 3)))
                xgb_cfg.setdefault("xgb_eval_fraction", float(xgb_cfg.get("xgb_eval_fraction", 0.10)))
                xgb_cfg.setdefault("xgb_early_stopping_rounds", int(xgb_cfg.get("xgb_early_stopping_rounds", 50)))
                
                # 🔹 Force OOF ON here regardless of tuned value
                xgb_cfg.setdefault("use_oof_meta", False)
                xgb_cfg.setdefault("oof_splits", 3)
                use_gpu = (os.environ.get("XGB_USE_GPU", "0") == "1")
                xgb_cfg.setdefault("tree_method", "hist")
                xgb_cfg.pop("predictor", None)
                if use_gpu:
                    xgb_cfg["device"] = os.environ.get("XGB_DEVICE", "cuda")
                else:
                    xgb_cfg.pop("device", None)

            # --- Train throttling (applies in both CV and final): stride + tail cap ---
            try:
                max_win = int(ens_cf.get("ensemble_deep_max_train_windows", 10000))
            except Exception:
                max_win = 10000
            try:
                train_stride = max(1, int(ens_cf.get("ensemble_train_stride", 1)))
            except Exception:
                train_stride = 1

            if train_stride > 1:
                X_seq_train  = X_seq_train[::train_stride]
                y_train_win  = y_train_win[::train_stride]
                if X_flat_train is not None:
                    X_flat_train = X_flat_train[::train_stride]

            if X_seq_train.shape[0] > max_win:
                X_seq_train  = X_seq_train[-max_win:]
                y_train_win  = y_train_win[-max_win:]
                if X_flat_train is not None and len(X_flat_train) >= len(y_train_win):
                    X_flat_train = X_flat_train[-max_win:]

            lags_eff_local = max(int(lags_eff), 3)

            self.model = EnsembleCNNLSTMXGBoost(
                cnn_config=cnn_cfg,
                lstm_config=lstm_cfg,
                xgb_config=xgb_cfg,
                input_shape=(lags_eff_local, len(features)),
            )

            try:
                self.model.fit(X_seq_train, X_flat_train, y_train_win)

                # --- Calibration on train-tail (no leakage), do it ONCE per fold ---
                try:
                    # merge feature + ensemble cfg (ens_cf overrides)
                    cfg = {**(getattr(self, "features_config", {}) or {}), **(ens_cf or {})}
                    use_temp = bool(cfg.get("deep_calibrate", False)) and \
                               str(cfg.get("deep_calibration_method", "")).lower() == "temperature"
                    need_cov = is_coverage_intent(cfg)

                    # IMPORTANT: set CV calibration defaults on *cfg* (ens_cf.setdefault here was too late)
                    if in_cv:
                        cfg.setdefault("deep_calibration_frac", 0.05)
                        cfg.setdefault("deep_calibration_min_samples", 300)

                    # Only compute if missing (reuse across blocks)
                    in_cv_flag = bool(getattr(self, "_in_optuna_cv", False))
                    # Temperature scaling: keep OFF in CV (extra overhead, not needed for activity control).
                    # In eval (real testing), recompute each window after retrain to avoid stale state.
                    must_cal_temp = (not in_cv_flag) and bool(use_temp)
                    # Coverage threshold: MUST be available in CV or Optuna's target_active_rate is ignored.
                    # Also recompute per evaluation window (train window changes month-to-month).
                    must_cal_cov  = bool(need_cov)

                    if must_cal_temp or must_cal_cov:
                        frac = float(cfg.get("deep_calibration_frac", 0.10))
                        nmin = int(cfg.get("deep_calibration_min_samples", 500))
                        nwin = int(X_seq_train.shape[0])
                        ncal = min(
                            max(nmin, int(round(nwin * max(0.01, min(frac, 0.99))))),
                            (nwin - 1)
                        ) if nwin > 1 else 0

                        if ncal >= 50:
                            X_tail_seq  = X_seq_train[-ncal:]
                            X_tail_flat = X_flat_train[-ncal:] if X_flat_train is not None else None
                            y_tail      = y_train_win[-ncal:].astype(int)
                            p_tail = self.model.predict_proba(X_tail_seq, X_tail_flat)
                            p_tail = sanitize_proba(p_tail)

                            if must_cal_temp:
                                self._deep_temp_T = float(fit_temperature_from_proba(p_tail, y_tail))
                                if self._is_debug():
                                    print(f"[Calib] Ensemble (CNN+LSTM+XGB) T={self._deep_temp_T:.3f} on {len(y_tail)} cal windows.")
                                p_tail = apply_temperature_to_proba(p_tail, self._deep_temp_T)
                                
                            if must_cal_cov:
                                _tgt = float(cfg.get("target_active_rate", cfg.get("target_coverage", 0.10)))
                                try:
                                    _mc = np.asarray(p_tail, dtype=float).max(axis=1)
                                    _mc = _mc[np.isfinite(_mc)]
                                    _q = np.quantile(_mc, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
                                    _in_cv = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                                    _ctx = "cv" if _in_cv else (
                                        ("real_m" + str(int(getattr(self, "_rt_month_ix", 0) or 0)))
                                        if bool(getattr(self, "_in_real_sim", False))
                                        else "eval"
                                    )
                                    print(
                                        f"[ConfDist][CalTail][{_ctx}][CLX] std={np.std(_mc):.4f} iqr={(_q[2]-_q[0]):.4f} "
                                        f"p50={_q[1]:.3f} p75={_q[2]:.3f} p90={_q[3]:.3f} p95={_q[4]:.3f} p99={_q[5]:.3f}"
                                    )
                                except Exception:
                                    pass

                                self._coverage_conf_thr = float(fit_coverage_threshold_on_calibration(p_tail, _tgt))

                                # Always log (COMPACT-safe), not only debug.
                                print(
                                    f"[Calib][Coverage][CLX] conf_thr={float(self._coverage_conf_thr):.6f} "
                                    f"target_active_rate={float(_tgt):.6f} cal_rows={int(len(p_tail))} ctx={_ctx}"
                                )
                                setattr(self, "_deep_coverage_thr", float(self._coverage_conf_thr))


                except Exception as _e:
                    print(f"[Calib] Ensemble (CNN+LSTM+XGB) calibration skipped: {_e}")

                # Predict on test
                proba = self.model.predict_proba(X_seq_test, X_flat_test)

            finally:
                try:
                    if hasattr(self.model, "free"):
                        self.model.free()
                except Exception:
                    pass
                try:
                    import gc as _gc
                    tf.keras.backend.clear_session()
                    _gc.collect()
                except Exception:
                    pass

        # ===========================
        # ENSEMBLE: ADAPTIVE REGIME
        # ===========================
        elif model_type == "ensemble_adaptive_regime":
            # Extract configs for the adaptive regime ensemble
            lstm_cfg   = filter_params(ens_cf, "lstm_")
            rf_cfg     = filter_params(ens_cf, "rf_")
            logit_cfg  = filter_params(ens_cf, "logit_")

            if in_cv:
                # Global deep CV caps (fallback)
                cv_epochs_global = int(cfg_f.get("deep_cv_max_epochs", 12))
                cv_bs_global     = int(cfg_f.get("deep_cv_batch_size", 256))
                cv_pat_global    = int(cfg_f.get("deep_cv_patience", 6))

                # Extra-strict caps for the LSTM head inside the adaptive ensemble.
                lstm_ens_epochs_cap = int(cfg_f.get("lstm_ens_cv_max_epochs", cv_epochs_global))
                base_max_win        = int(ens_cf.get("ensemble_deep_max_train_windows", 10000))
                lstm_ens_win_cap    = int(cfg_f.get("lstm_ens_cv_max_train_windows", base_max_win))

                # Coarser stride in CV only (final WFO retrain ignores in_cv).
                ens_cf.setdefault("ensemble_train_stride", 3)

                # LSTM head (clip epochs / batch / windows to ensemble-specific caps)
                lstm_cfg = dict(lstm_cfg)
                lstm_cfg.setdefault("lstm_use_early_stopping", True)
                lstm_cfg["lstm_epochs"] = min(
                    int(lstm_cfg.get("lstm_epochs", lstm_ens_epochs_cap)),
                    lstm_ens_epochs_cap,
                )
                lstm_cfg["lstm_batch_size"] = min(
                    int(lstm_cfg.get("lstm_batch_size", cv_bs_global)),
                    cv_bs_global,
                )
                lstm_cfg.setdefault("lstm_patience", cv_pat_global)
                lstm_cfg["deep_max_train_windows"] = min(
                    int(lstm_cfg.get("deep_max_train_windows", base_max_win)),
                    lstm_ens_win_cap,
                )

                # RF and Logistic heads are cheap; keep their existing CV caps/logic.
                rf_cfg = dict(rf_cfg)
                rf_cfg.setdefault("n_estimators", min(int(rf_cfg.get("n_estimators", 400)), 400))
                rf_cfg.setdefault("max_depth", 8)
                rf_cfg.setdefault("min_samples_leaf", 20)
                rf_cfg.setdefault("n_jobs", int(rf_cfg.get("n_jobs", 3)))
                rf_cfg.setdefault("random_state", int(rf_cfg.get("random_state", 42)))

                logit_cfg = dict(logit_cfg)
                logit_cfg.setdefault("penalty", "l2")
                logit_cfg.setdefault("C", 1.0)
                logit_cfg.setdefault("solver", "lbfgs")
                logit_cfg.setdefault("max_iter", 200)
                logit_cfg.setdefault("n_jobs", int(logit_cfg.get("n_jobs", 3)))

            # sanitize class_weight for LogisticRegression
            cw = logit_cfg.get("class_weight", logit_cfg.get("logit_class_weight", None))
            try:
                if isinstance(cw, float) and (np.isnan(cw) or np.isinf(cw)):
                    cw = None
            except Exception:
                pass
            if isinstance(cw, str):
                s = cw.strip().lower()
                cw = None if s in ("", "none", "null", "nan") else ("balanced" if s == "balanced" else None)
            elif (cw not in (None, "balanced")) and (not isinstance(cw, dict)):
                cw = None
            if cw is not None:
                logit_cfg["class_weight"] = cw
            else:
                logit_cfg.pop("class_weight", None)

            if not hasattr(self, "_ensemble_win_cache"):
                self._ensemble_win_cache = {}
            maker = getattr(self, "_ensemble_make_windows", None)
            make_windows = (lambda df: maker(df, features, int(lags_eff))) if callable(maker) \
                        else (lambda df: _fallback_make_windows(df, features, int(lags_eff)))

            # Deep windowing (train / test)
            #
            # IMPORTANT (OOM fix):
            # Apply stride/cap on the *raw bars* before creating sliding windows,
            # and do NOT cache full train windows across folds/months.
            max_train_windows = ens_cf.get("ensemble_deep_max_train_windows", 10000)
            train_stride = ens_cf.get("ensemble_deep_train_stride", ens_cf.get("ensemble_train_stride", 3))

            try:
                if max_train_windows is not None:
                    max_train_windows = int(max_train_windows)
            except Exception:
                max_train_windows = 10000
                
            try:
                train_stride = int(train_stride) if train_stride is not None else 1
            except Exception:
                train_stride = 1
            if train_stride < 1:
                train_stride = 1

            def _start_idx_for_last_strided_windows(n_rows, win, stride, max_windows):
                if max_windows is None:
                    return 0
                try:
                    max_windows = int(max_windows)
                except Exception:
                    return 0
                if max_windows <= 0:
                    return 0
                if n_rows <= win:
                    return 0
                total_windows = n_rows - win + 1
                if stride <= 1:
                    need = min(total_windows, max_windows)
                    start_window = total_windows - need
                else:
                    total_strided = (total_windows + stride - 1) // stride
                    need = min(total_strided, max_windows)
                    start_window = (total_strided - need) * stride
                return max(0, int(start_window))

            _start_idx = _start_idx_for_last_strided_windows(len(train_data), int(lags_eff), int(train_stride), max_train_windows)
            if _start_idx > 0:
                train_data = train_data.iloc[_start_idx:].copy()

            X_seq_train, y_train_win, idx_train, X_flat_train = make_windows(train_data)
            X_seq_test,  y_test_win,  idx_test,  X_flat_test  = make_windows(test_data)

            if X_seq_train.shape[0] == 0 or X_seq_test.shape[0] == 0:
                print("⚠️ Ensemble (adaptive) produced zero windows. Skipping fold.")
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:adaptive_zero_windows")

            # Extra safety: apply stride/cap on the already-windowed arrays (train only)
            idx_train_np = np.asarray(idx_train, dtype=int)

            if train_stride > 1:
                idx_stride = np.arange(0, X_seq_train.shape[0], train_stride, dtype=int)
                y_stride   = y_train_win[idx_stride]
                u_s, c_s   = np.unique(y_stride, return_counts=True)

                try:
                    MIN_CLASS_SAMPLES_ENSEMBLE = int(
                        getattr(self, "ensemble_min_class_samples", 3)
                    )
                except Exception:
                    MIN_CLASS_SAMPLES_ENSEMBLE = 3

                if (len(u_s) < 2) or (c_s.min() < MIN_CLASS_SAMPLES_ENSEMBLE):
                    # Class-aware stride downsampling:
                    # keep all minority-class windows, stride only the majority,
                    # and top-up any class that would otherwise vanish.
                    try:
                        y_full = np.asarray(y_train_win)
                        u_f, c_f = np.unique(y_full, return_counts=True)
                        counts_full = {int(k): int(v) for k, v in zip(u_f, c_f)}

                        # Classes that are "rare" in the full set (we keep all of them)
                        rare_classes = [int(k) for k, v in zip(u_f, c_f) if int(v) < MIN_CLASS_SAMPLES_ENSEMBLE]
                        if rare_classes:
                            idx_keep_rare = np.nonzero(np.isin(y_full, rare_classes))[0].astype(int)
                            idx_sel = np.unique(np.concatenate([idx_stride, idx_keep_rare]).astype(int))
                        else:
                            idx_sel = np.unique(idx_stride.astype(int))
                        idx_sel.sort()

                        # Top-up any class so it doesn't drop below min(desired, available)
                        sel_set = set(idx_sel.tolist())
                        y_sel = y_full[idx_sel] if idx_sel.size else np.asarray([], dtype=y_full.dtype)
                        for cls, full_cnt in zip(u_f, c_f):
                            cls_i = int(cls)
                            full_cnt_i = int(full_cnt)
                            desired = min(int(MIN_CLASS_SAMPLES_ENSEMBLE), full_cnt_i)
                            cur = int(np.sum(y_sel == cls_i)) if y_sel.size else 0
                            if cur < desired:
                                need = desired - cur
                                candidates = np.nonzero(y_full == cls_i)[0].astype(int)
                                added = []
                                for j in candidates:
                                    if int(j) not in sel_set:
                                        sel_set.add(int(j))
                                        added.append(int(j))
                                        if len(added) >= need:
                                            break
                                if added:
                                    idx_sel = np.unique(np.concatenate([idx_sel, np.asarray(added, dtype=int)]))
                                    idx_sel.sort()
                                    y_sel = y_full[idx_sel]

                        # Only accept if we did not lose any class that exists in full y
                        u_post, c_post = np.unique(y_sel, return_counts=True) if y_sel.size else (np.array([]), np.array([]))
                        if (len(u_post) == len(u_f)) and (len(u_post) >= 2):
                            counts_post = {int(k): int(v) for k, v in zip(u_post, c_post)}
                            print(
                                f"[Ensemble-Adapt][Stride] Applied class-aware stride downsampling "
                                f"(stride={train_stride}) kept_rare={rare_classes} "
                                f"full={counts_full} post={counts_post} sel={len(idx_sel)}/{len(y_full)}"
                            )
                            X_seq_train  = X_seq_train[idx_sel]
                            X_flat_train = X_flat_train[idx_sel]
                            y_train_win  = y_sel
                            idx_train_np = idx_train_np[idx_sel]
                        else:
                            counts_s = {int(k): int(v) for k, v in zip(u_s, c_s)}
                            print(
                                f"[Ensemble-Adapt][Stride] Disabled stride downsampling (stride={train_stride}) "
                                f"full={counts_full} post_stride={counts_s}"
                            )
                    except Exception:
                        counts_s = {int(k): int(v) for k, v in zip(u_s, c_s)}
                        print(
                            f"[Ensemble-Adapt][Stride] Disabled stride downsampling (stride={train_stride}) "
                            f"post_stride={counts_s}"
                        )
                else:
                    X_seq_train  = X_seq_train[idx_stride]
                    X_flat_train = X_flat_train[idx_stride]
                    y_train_win  = y_stride
                    idx_train_np = idx_train_np[idx_stride]


            if X_seq_train.shape[0] > max_train_windows:
                X_seq_train  = X_seq_train[-max_train_windows:]
                X_flat_train = X_flat_train[-max_train_windows:]
                y_train_win  = y_train_win[-max_train_windows:]
                idx_train_np = idx_train_np[-max_train_windows:]

            # Regime features
            adx_col_req = str(ens_cf.get("adx_col", "adx_14"))
            vol_col_req = str(ens_cf.get("vol_col", "rolling_std_20"))

            def _resolve_regime_col(req, df_a, df_b, kind):
                # Pick a regime feature column that exists in BOTH train and test frames.
                if (req in df_a.columns) and (req in df_b.columns):
                    return req

                inter = [c for c in df_a.columns if c in df_b.columns]
                low = {str(c).lower(): c for c in inter}
                req_low = str(req).lower()
                if req_low in low:
                    return low[req_low]

                if kind == "adx":
                    pool = [c for c in inter if "adx" in str(c).lower()]
                else:
                    pool = [c for c in inter if "rolling_std" in str(c).lower()]
                    if not pool:
                        pool = [c for c in inter if ("realized" in str(c).lower() and "vol" in str(c).lower())
                                or str(c).lower().startswith("vol") or "_vol" in str(c).lower()]
                    if not pool:
                        pool = [c for c in inter if str(c).lower().startswith("atr") or "_atr" in str(c).lower()]

                if pool:
                    import re as _re
                    def _num(s):
                        m = _re.findall(r"\d+", str(s))
                        return int(m[0]) if m else None
                    tgt = _num(req)
                    if tgt is not None:
                        scored = []
                        for c in pool:
                            v = _num(c)
                            scored.append((abs(v - tgt) if v is not None else 10**9, str(c)))
                        scored.sort(key=lambda x: x[0])
                        best = scored[0][1]
                        for c in pool:
                            if str(c) == best:
                                return c
                    return pool[0]

                return req

            adx_col = _resolve_regime_col(adx_col_req, train_data, test_data, "adx")
            vol_col = _resolve_regime_col(vol_col_req, train_data, test_data, "vol")

            if (adx_col != adx_col_req) or (vol_col != vol_col_req):
                print(
                    f"[Ensemble-Adapt][RegimeCols] adx_col={adx_col} (req={adx_col_req}) "
                    f"vol_col={vol_col} (req={vol_col_req})"
                )

            missing = []
            if (adx_col not in train_data.columns) or (adx_col not in test_data.columns):
                missing.append(adx_col)
            if (vol_col not in train_data.columns) or (vol_col not in test_data.columns):
                missing.append(vol_col)
            if missing:
                print(
                    f"[Ensemble-Adapt][REGIME][WARN] Missing regime cols {missing} in train/test; "
                    "regime switching will default to 'sideways'."
                )
                # Last resort: create constant columns to keep the pipeline running.
                for col in missing:
                    if col not in train_data.columns:
                        train_data[col] = 0.0
                    if col not in test_data.columns:
                        test_data[col] = 0.0


            regime_source_train = train_data[[adx_col, vol_col]].iloc[idx_train_np]
            regime_source_test  = test_data[[adx_col, vol_col]].iloc[idx_test]

            self.model = AdaptiveRegimeStrategy(
                lstm_config=lstm_cfg,
                rf_config=rf_cfg,
                logit_config=logit_cfg,
                input_shape=input_shape,
                adx_col=adx_col,
                vol_col=vol_col,
                adx_thresh=float(ens_cf.get("adx_thresh", 25)),
                vol_thresh=float(ens_cf.get("vol_thresh", 0.002)),
                adx_thresh_q=(
                    float(ens_cf.get("adx_thresh_q", 0.70))
                    if bool(ens_cf.get("train_lstm_on_trend_only", True))
                    else None
                ),
                train_lstm_on_trend_only=bool(ens_cf.get("train_lstm_on_trend_only", True)),
            )
            try:
                idx_end_pos = np.arange(len(X_seq_train), dtype=int)
                self.model.fit(
                    X_seq_train, X_flat_train, y_train_win,
                    X_flat_with_regime=regime_source_train,
                    idx_end=idx_end_pos,
                )
                
                
                # Coverage calibration for AdaptiveRegime ensemble (mirrors test_strategy coverage gating)
                try:
                    cfg_cal = dict(merged)
                    need_cov_cal = is_coverage_intent(cfg_cal)
                    if in_cv:
                        cfg_cal.setdefault("deep_calibration_frac", 0.05)
                        cfg_cal.setdefault("deep_calibration_min_samples", 300)
                    if need_cov_cal:
                        frac = float(cfg_cal.get("deep_calibration_frac", 0.10))
                        nmin = int(cfg_cal.get("deep_calibration_min_samples", 500))
                        nwin = int(X_seq_train.shape[0])
                        ncal = min(
                            max(nmin, int(round(nwin * max(0.01, min(frac, 0.99))))),
                            (nwin - 1)
                        ) if nwin > 1 else 0
                        if ncal >= 50:
                            X_tail_seq  = X_seq_train[-ncal:]
                            X_tail_flat = X_flat_train[-ncal:] if X_flat_train is not None else None
                            y_tail      = y_train_win[-ncal:].astype(int)
                            rs_tail = None
                            try:
                                rs_tail = regime_source_train.iloc[-ncal:]
                            except Exception:
                                rs_tail = None
                            p_tail = self.model.predict_proba(X_tail_seq, X_tail_flat, regime_source=rs_tail)
                            p_tail = sanitize_proba(p_tail)
                            _tgt = float(cfg_cal.get("target_active_rate", cfg_cal.get("target_coverage", 0.10)))
                            self._coverage_conf_thr = float(fit_coverage_threshold_on_calibration(p_tail, _tgt))
                            setattr(self, "_deep_coverage_thr", float(self._coverage_conf_thr))
                            try:
                                setattr(self, "_last_cov_cal_rows", int(len(p_tail)))
                            except Exception:
                                pass
                            _in_cv2 = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                            if _in_cv2:
                                _ctx = "cv"
                            elif bool(getattr(self, "_in_real_sim", False)):
                                _ctx = "real_m" + str(int(getattr(self, "_rt_month_ix", 0) or 0))
                            else:
                                _ctx = "eval"
                            print(
                                f"[Calib][Coverage][AR] conf_thr={float(self._coverage_conf_thr):.6f} "
                                f"target_active_rate={float(_tgt):.6f} cal_rows={int(len(p_tail))} ctx={_ctx}"
                            )
                except Exception as _e:
                    print(f"[Calib] Ensemble (adaptive) coverage calibration skipped: {_e}")

                
                proba = self.model.predict_proba(
                    X_seq_test, X_flat_test, regime_source=regime_source_test
                )
            finally:
                try:
                    import gc as _gc
                    tf.keras.backend.clear_session()
                    _gc.collect()
                except Exception:
                    pass

        else:
            raise ValueError(f"Unknown ensemble model_type: {model_type}")

        # ---------------------------------------
        # Generic postprocessing for all ensembles
        # ---------------------------------------
        if proba is None or (hasattr(proba, "__len__") and len(proba) == 0):
            print("❌ Ensemble produced no probabilities.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:no_probabilities")


        proba = np.asarray(proba, dtype=np.float32)
        if proba.ndim == 1:
            proba = np.stack([1.0 - proba, np.zeros_like(proba), proba], axis=1)
        elif proba.shape[1] == 2:
            neutral = 1.0 - proba.sum(axis=1, keepdims=True)
            proba = np.hstack([proba[:, :1], neutral, proba[:, 1:2]])

        proba = np.nan_to_num(proba, nan=1e-6, posinf=1.0, neginf=0.0)
        row_sums = np.clip(proba.sum(axis=1, keepdims=True), 1e-9, None)
        proba = proba / row_sums
        
        # Apply temperature only if NOT fused with DQN
        if hasattr(self, "_deep_temp_T") and not bool(getattr(self, "_proba_came_from_dqn_fusion", False)):
            try:
                proba = apply_temperature_to_proba(proba, float(getattr(self, "_deep_temp_T")))
            except Exception:
                pass

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

        # --- Edge-vs-Cost gating (dynamic; align on ensemble window ends = idx_test) ---
        cfg_gate = dict(merged or {})
        # If using coverage gating, start from the calibrated coverage threshold (fold-safe).
        _gmode = str(cfg_gate.get("gating_mode", cfg_gate.get("gate_mode", "threshold"))).lower()
        base_thr = None
        if _gmode == "coverage":
            try:
                _bt = getattr(self, "_coverage_conf_thr", None)
                if _bt is not None and np.isfinite(float(_bt)):
                    base_thr = float(_bt)
            except Exception:
                base_thr = None
        if base_thr is None:
            base_thr = float(self._resolve_conf_thr(
                float(cfg_gate.get("confidence_threshold", 0.0))
            ))
        _cfg_cost = dict(cfg_gate)
        if high_vol_thr_train is not None and _cfg_cost.get("high_vol_thr") is None:
            _cfg_cost["high_vol_thr"] = float(high_vol_thr_train)
        _cost_src = self._ensure_cost_columns(test_data, _cfg_cost)
        _all_idx = test_data.index
        
        rets_all, sprd_all, slip_all = self._get_cost_arrays_aligned(_cost_src, _all_idx)
        
        vol_w = int(cfg_gate.get("vol_window_bars", _PC["vol_window_bars"]))
        rv_all = realized_vol(rets_all, window=vol_w).to_numpy(dtype=np.float32)

        rv_m, rv_s = np.nan, np.nan
        den_floor = np.nan
        try:
            if "returns" in train_data.columns:
                _rv_tr = realized_vol(train_data["returns"].astype(float), window=vol_w).to_numpy(dtype=np.float32)
                _rv_tr_f = _rv_tr[np.isfinite(_rv_tr)]
                if _rv_tr_f.size >= 50:
                    rv_m = float(np.nanmean(_rv_tr_f))
                    rv_s = float(np.nanstd(_rv_tr_f))
                    _pos = _rv_tr_f[_rv_tr_f > 0]
                    if _pos.size > 0:
                        den_floor = float(np.nanmedian(_pos))
        except Exception:
            pass
        
        # IMPORTANT (causality): do NOT fall back to test-window stats.
        # If TRAIN stats are unusable, neutralize the volatility term and use a constant denom floor.
        if (not np.isfinite(rv_m)) or (not np.isfinite(rv_s)) or (rv_s <= 0):
            vol_z_all = np.zeros_like(rv_all, dtype=np.float32)
        else:
            vol_z_all = ((rv_all - rv_m) / rv_s).astype(np.float32)

        den_floor = den_floor if (np.isfinite(den_floor) and den_floor > 1e-8) else 1e-6
        den_all = np.where(rv_all > 1e-8, rv_all, den_floor).astype(np.float32)
            
        spread_norm_all = np.divide(sprd_all, den_all, out=np.zeros_like(sprd_all, dtype=np.float32), where=np.isfinite(den_all))
        
        a = float(cfg_gate.get("alpha_vol_z", 0.004))
        b = float(cfg_gate.get("beta_spread_norm", _PC["beta_spread_norm"]))
        g = float(cfg_gate.get("gamma_slip_norm", _PC["gamma_slip_norm"]))
        slip_norm_bps = float(cfg_gate.get("slip_norm_bps", _PC["slip_norm_bps"]))
        min_slip_norm_bps = float(cfg_gate.get("min_slip_norm_bps", _PC["min_slip_norm_bps"]))
        slip_norm_bps = max(slip_norm_bps, min_slip_norm_bps, 1e-6)

        vol_z_cap = float(cfg_gate.get("vol_z_cap", 6.0))
        spread_norm_cap = float(cfg_gate.get("spread_norm_cap", 5.0))
        slip_ratio_cap = float(cfg_gate.get("slip_ratio_cap", 6.0))
        max_conf_thr = float(cfg_gate.get("max_conf_thr", 0.90))

        vol_z_all = np.clip(vol_z_all, -vol_z_cap, vol_z_cap)
        spread_norm_all = np.clip(spread_norm_all, 0.0, spread_norm_cap)
        slip_ratio = np.clip(slip_all / slip_norm_bps, 0.0, slip_ratio_cap)

        thr_full = np.clip(base_thr + a*vol_z_all + b*spread_norm_all + g*slip_ratio, 0.0, max_conf_thr).astype(np.float32)
        idx_test_arr = np.asarray(idx_test, dtype=int)
        thr_vec = thr_full[idx_test_arr]
        if self._is_debug():
            print(f"[Gate✔] Dynamic αβγ active | base={base_thr:.3f} α={a:.3f} β={b:.3f} γ={g:.3f} "
                        f"| median_thr={np.nanmedian(thr_vec):.3f} | bars={len(thr_vec)}")
            
        # ------------------------------------------------------------
        # IMPORTANT: define the TRUE eval-universe for windows
        # (windows whose END time is on/after the eval anchor).
        # The nudge/gating must only use this universe; otherwise it
        # adapts using warmup windows that are later discarded.
        # ------------------------------------------------------------
        try:
            keep_win = (test_data.index[idx_test_arr] >= self._expected_eval_start)
            _eval_idx = np.flatnonzero(keep_win)
        except Exception:
            keep_win = np.zeros(len(idx_test_arr), dtype=bool)
            _eval_idx = np.asarray([], dtype=int)

        if _eval_idx.size == 0:
            print("❌ No tradable test windows in ensemble after start cut.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:no_eval_windows")

        # --- Soft coverage-drift nudge (mirrors test_strategy runtime control) ---
        try:
            tgt   = float(merged.get("target_active_rate", merged.get("target_coverage", 0.10)))
            band  = float(merged.get("runtime_active_band_margin", _PC["runtime_active_band_margin"]))
            win_k = int(merged.get("runtime_coverage_window", 96))
            step  = float(merged.get("runtime_conf_nudge", 0.01))
            n_nudge = int(_eval_idx.size)
            if n_nudge <= 0:
                raise ValueError("no_eval_windows_for_nudge")


            # Rolling-quantile cap (prevents "bunched confidence" → near-zero trades)
            _low = max(0.0, tgt - band)
            allow_qcap = bool(merged.get("runtime_allow_rolling_qcap", True))
            if allow_qcap and win_k > 1 and n_nudge >= win_k:
                try:
                    _dr = np.asarray(decoded_raw, dtype=int)[_eval_idx]
                    _mc = np.asarray(max_conf, dtype=np.float32)[_eval_idx]
                    _tv = np.asarray(thr_vec, dtype=np.float32)[_eval_idx].copy()
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
                            _tv[_m] = np.minimum(_tv[_m], _q[_m])
                            thr_vec[_eval_idx] = _tv
                            if self._is_debug():
                                print(
                                    f"[Gate✔] Rolling-quantile cap active | q={1.0 - tgt:.3f} "
                                    f"win={win_k} | thr_med={float(np.nanmedian(thr_vec[_eval_idx])):.3f}"
                                )
                except Exception:
                    pass

            # Rolling activity drift control: nudge thresholds up/down to stay inside band
            if win_k > 1 and n_nudge >= win_k:
                _dr = np.asarray(decoded_raw, dtype=int)[_eval_idx]
                _mc = np.asarray(max_conf, dtype=np.float32)[_eval_idx]
                _tv = np.asarray(thr_vec, dtype=np.float32)[_eval_idx]
                _act = ((_dr != 0) & (_mc >= _tv)).astype(np.float32)
                _cs = np.cumsum(np.insert(_act, 0, 0.0))
                _roll = (_cs[win_k:] - _cs[:-win_k]) / float(win_k)
                _roll = np.concatenate([np.full(win_k-1, np.nan, dtype=np.float32), _roll]).astype(np.float32)
                _low2, _high2 = max(0.0, tgt - band), min(1.0, tgt + band)
                _drift = np.where(_roll < _low2, -step, np.where(_roll > _high2, step, 0.0)).astype(np.float32)
                _drift = np.nan_to_num(_drift, nan=0.0)
                max_conf_thr = float(cfg_f.get("max_conf_thr", 0.90))
                min_conf_thr = float(cfg_f.get("min_conf_thr", 0.33))

                thr_vec[:n_nudge] = np.clip(
                    thr_vec[:n_nudge] + _drift[:n_nudge],
                    min_conf_thr,
                    max_conf_thr
                ).astype(np.float32)
        except Exception as _e:
            if self._is_debug():
                print(f"[Gate] Coverage nudge skipped (ensemble): {_e}")

        # Save median threshold for reporting / safety-ladder reference
        conf_thr_final = float(np.nanmedian(thr_vec[_eval_idx])) if _eval_idx.size > 0 else float(np.nanmedian(thr_vec))

        self._last_conf_thr_used = conf_thr_final
        # Apply gating ONLY on eval windows; force flat elsewhere
        final_preds = np.zeros_like(decoded_raw, dtype=int)
        if _eval_idx.size > 0:
            final_preds[_eval_idx] = np.asarray(decoded_raw, dtype=int)[_eval_idx]
            _mask = (np.asarray(max_conf, dtype=np.float32)[_eval_idx] < np.asarray(thr_vec, dtype=np.float32)[_eval_idx])
            final_preds[_eval_idx[_mask]] = 0

        # safety: if gating nuked all trades, step down the threshold
        _allow_backoff = bool(
            merged.get(
                "allow_conf_backoff_cv" if in_cv else "allow_conf_backoff_eval",
                False
            )
        )
        if (final_preds != 0).sum() == 0 and _allow_backoff:
            for t in [0.50, 0.33, 0.20, 0.10, 0.00]:
                if t >= self._last_conf_thr_used:
                    continue
                tmp = np.zeros_like(decoded_raw, dtype=int)
                if _eval_idx.size > 0:
                    tmp[_eval_idx] = np.asarray(decoded_raw, dtype=int)[_eval_idx]
                    tmp[_eval_idx[np.asarray(max_conf, dtype=np.float32)[_eval_idx] < t]] = 0
                if (tmp != 0).sum() > 0:
                    print(f"[Backoff] lowered conf_thr → {t:.2f}; active_rate={np.mean(tmp!=0):.3f}")
                    final_preds = tmp
                    self._last_conf_thr_used = float(t)
                    conf_thr_final = float(t)
                    break
        # quick trace
        print(f"[ENSEMBLE-CV] trades={int((final_preds != 0).sum() if final_preds is not None else 0)} "
            f"at thr={float(getattr(self, '_last_conf_thr_used', 0.0)):.4f}")

        if final_preds is None or (final_preds != 0).sum() == 0:
            if in_cv:
                return _safe_metrics_return(
                    (np.nan,) * N_METRICS,
                    context="test_ensemble_strategy:no_trades_cv",
                )
            final_preds = np.zeros_like(decoded_raw, dtype=int)

        # -------- Start-cut before building result_df (FIX: index/pred alignment) --------
        _mask_keep = np.asarray(keep_win, dtype=bool)
        if not _mask_keep.any():
            print("❌ No tradable test windows in ensemble after start cut.")
            return _safe_metrics_return((np.nan,) * N_METRICS, context="test_strategy:deep2d_no_trades")

        # apply mask first, then build aligned result df
        idx_test_masked = idx_test_arr[_mask_keep]
        final_preds     = final_preds[_mask_keep]
        try:
            proba = proba[_mask_keep]
        except Exception:
            pass

        test_index = test_data.index[idx_test_masked]
        result_df  = test_data.loc[test_index].copy()
        result_df["pred"] = final_preds

        # ------------------------------------------------------
        # EXPAND RESULT FRAME TO THE FULL EVAL WINDOW (like others)
        # ------------------------------------------------------
        try:
            # Evaluation should start at _expected_eval_start (day-1 anchor) and
            # end at test_end, same as in test_strategy. Build that index from
            # the master data so we don't silently skip early-month bars.
            full_eval_index = (
                self.data.loc[self._expected_eval_start:test_end].index
            )

            # Keep a copy of the narrow frame to preserve attrs, then reindex.
            base_result = result_df.copy()
            result_df = base_result.reindex(full_eval_index)

            # Where the ensemble did not produce a window / prediction,
            # treat it as "no position" (0) rather than dropping the bar.
            result_df["pred"] = result_df.get("pred", 0).fillna(0).astype(int)

            # Preserve any attrs (features_config, etc.) that were on the
            # narrower frame before reindexing.
            for k, v in getattr(base_result, "attrs", {}).items():
                result_df.attrs.setdefault(k, v)
                
            # Attach model-consistent regimes for diagnostics (primarily used
            # by AdaptiveRegimeStrategy). This ensures the per-regime CV table
            # reflects the same regime logic used inside the strategy rather
            # than heuristic `regime_id_diag` reconstruction.
            try:
                _m = getattr(self, "model", None)
                if "regime_id" not in result_df.columns and (_m is not None) and hasattr(_m, "infer_regime_ids"):
                    _cols = []
                    _adx_col = getattr(_m, "adx_col", None)
                    _vol_col = getattr(_m, "vol_col", None)
                    if _adx_col and _adx_col in self.data.columns:
                        _cols.append(_adx_col)
                    if _vol_col and _vol_col in self.data.columns:
                        _cols.append(_vol_col)
                    if _cols:
                        _rs = self.data[_cols].reindex(result_df.index)
                        result_df["regime_id"] = _m.infer_regime_ids(_rs)
                        print(f"[RegimeDiag] Attached regime_id for diagnostics | cols={_cols} | rows={len(result_df)}")
            except Exception:
                pass
        except Exception as _e:
            print(f"⚠️ Ensemble reindex to full eval window failed, using narrow frame: {_e}")
            # fall back to the original result_df


        # Ensure required columns exist/aligned
        if "spread" not in result_df.columns:
            result_df["spread"] = 0.0

        if "returns" not in result_df.columns:
            # try align from full dataset; else compute from price/close
            if hasattr(self, "data") and isinstance(getattr(self, "data"), pd.DataFrame) and "returns" in self.data.columns:
                result_df["returns"] = self.data["returns"].reindex(result_df.index).astype(float)
            else:
                # last-ditch: build simple returns from price/close if available
                px = None
                for cand in ("price", "close", "Price", "Close"):
                    if cand in result_df.columns:
                        px = result_df[cand].astype(float)
                        break
                if px is not None:
                    result_df["returns"] = px.pct_change().fillna(0.0)
                else:
                    # nothing to do; place zeros so metrics are defined
                    result_df["returns"] = 0.0
                    
        # --- Edge-bar guard for ensembles ---
        _idx = result_df.index
        if len(_idx) >= 2:
            gaps = pd.Series(_idx[1:] - _idx[:-1], index=_idx[:-1])
            exp  = gaps.median()
            is_edge = gaps > (exp * 1.5)

            if self._is_debug():
                try:
                    _ctx = "cv" if bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False)) else ("real_sim" if bool(getattr(self, "_in_real_sim", False)) else "run")
                    _edge_idx = is_edge.index[is_edge]
                    _pred_ser = result_df["pred"]
                    _nz_before = int((_pred_ser != 0).sum())
                    _nz_edge = int((_pred_ser.reindex(_edge_idx).fillna(0) != 0).sum()) if len(_edge_idx) else 0
                    _nz_last = int(bool(_pred_ser.iloc[-1] != 0))
                    print(f"[EdgeGuardAudit][{model_type}] ctx={_ctx} exp={exp} edge_bars={int(is_edge.sum())} nz_before={_nz_before} nz_on_edge={_nz_edge} nz_last={_nz_last}")
                except Exception:
                    pass

            result_df.loc[is_edge.index[is_edge], "pred"] = 0
            result_df.iloc[-1, result_df.columns.get_loc("pred")] = 0

            if self._is_debug():
                try:
                    _nz_after = int((result_df["pred"] != 0).sum())
                    print(f"[EdgeGuardAudit][{model_type}] nz_after={_nz_after}")
                except Exception:
                    pass



        # Drop rows without returns
        result_df = result_df.dropna(subset=["returns"])

        # If no rows or no active signals, return a clean no-trades tuple (non-CV),
        # but keep CV behavior so folds can be pruned upstream.
        def _no_trades_tuple():
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    float(result_df["returns"].std(ddof=0)) if len(result_df) else 0.0,
                    0.0)

        if len(result_df) == 0:
            if getattr(self, "_in_optuna_cv", False):
                return _safe_metrics_return((np.nan,) * N_METRICS, context="test_ensemble_strategy:empty_result_df_cv")

            print("ℹ️ Ensemble evaluation window empty after cleaning. Returning no-trades metrics.")
            return _no_trades_tuple()

        # Heuristic for “no activity”: all zeros (or single class that maps to flat)
        # If your pipeline uses {0,1,2}, adjust this if 1/2 map to long/short.
        has_activity = (result_df["pred"] != 0).any()
        if not has_activity:
            if in_cv:
                # Let CV callers penalize this split; they check trades == 0 and score -9999.
                return _safe_metrics_return(
                    (np.nan,) * N_METRICS,
                    context="test_ensemble_strategy:no_activity_cv",
                )

            # Non-CV: keep going so compute_full_evaluation_metrics produces flat curves + attrs.
            if self._is_debug():
                print("ℹ️ [Ensemble] No trades in this window; computing flat metrics.")
            return _no_trades_tuple()

        # ----------------------------
        # Evaluation + return
        # ----------------------------
        
        # Attach per-bar trading-cost columns on ensembles too
        try:
            if bool(getattr(self, "trading_costs", True)):
                _cfg_cost2 = dict(merged)
                if high_vol_thr_train is not None and _cfg_cost2.get("high_vol_thr") is None:
                    _cfg_cost2["high_vol_thr"] = float(high_vol_thr_train)
                result_df = self._ensure_cost_columns(result_df, _cfg_cost2)
        except Exception:
            pass
        
        # Ensure ensemble evaluation can use the same execution overlays (TWAP / kill-switch / etc.)
        # as single-model evaluation (they are driven by df.attrs["features_config"]).
        try:
            
            # Make the stored snapshot truthful: record the final operative threshold actually used.
            try:
                if 'conf_thr_final' in locals() and np.isfinite(float(conf_thr_final)):
                    merged["confidence_threshold"] = float(conf_thr_final)
                    merged["confidence_threshold_used"] = float(conf_thr_final)
            except Exception:
                pass
            result_df.attrs["features_config"] = dict(merged)
            result_df.attrs["debug_costs"] = bool(self._is_debug())
        except Exception:
            pass

        _in_cv_mode = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
        if _in_cv_mode:
            _eval_ctx = "cv:fold_or_month_eval:test_ensemble_strategy"
        elif bool(getattr(self, "_in_real_sim", False)):
            _eval_ctx = "real_sim:month_eval:test_ensemble_strategy"
        else:
            _eval_ctx = "eval:test_ensemble_strategy"

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


        # Keep canonical executed position in `position` (downstream expects it).
        try:
            if result_df is not None and "position_exec" in result_df.columns:
                result_df["position"] = result_df["position_exec"]
        except Exception:
            pass
        
        # --- Proactive cleanup to avoid cumulative RAM growth across CV folds ---
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc, time
        gc.collect()
        time.sleep(0.05)

        
        if getattr(self, "_in_optuna_cv", False):
            self._cv_last_eval_df = (
                result_df.copy() if result_df is not None and not result_df.empty else None
            )

            # Only accumulate per-fold frames in debug mode.
            # In normal runs we avoid keeping all folds in memory to prevent
            # RAM drift across Optuna trials.
            try:
                if self._cv_last_eval_df is not None and self._is_debug():
                    try:
                        self._cv_fold_eval_frames.append(self._cv_last_eval_df.copy())
                        #   R3: hard-cap stored CV fold frames to prevent RAM drift in debug runs
                        try:
                            _max_keep = int((getattr(self, 'config', {}) or {}).get('cv_max_fold_eval_frames', 3) or 3)
                        except Exception:
                            _max_keep = 3
                        if _max_keep > 0 and len(self._cv_fold_eval_frames) > _max_keep:
                            self._cv_fold_eval_frames = self._cv_fold_eval_frames[-_max_keep:]
                    except AttributeError:
                        # First time in this process: create the list
                        self._cv_fold_eval_frames = [self._cv_last_eval_df.copy()]
            except Exception:
                if self.debug:
                    self._log("⚠️ Failed to append CV fold eval frame", level="warning")
                    
            self.results = None
            self.results_full = None
        else:
            # Non-CV: persist evaluated frame for downstream plotting/exports.
            self.results = result_df.copy() if result_df is not None else None
            self.results_full = self.results
            self._cv_last_eval_df = None


        # Best-effort TF/GC cleanup between runs (non-invasive)
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()

        metrics = _safe_metrics_return(metrics, context="eval_block_1")
        return metrics


