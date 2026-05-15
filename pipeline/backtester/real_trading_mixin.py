"""Auto-extracted mixin -- see composed.py for the full MLBacktester."""
from config import PIPELINE_CONSTANTS as _PC
from pipeline._imports import *  # noqa: F401,F403
from pipeline.dqn_config import HPO_CONFIG_DIR  # noqa: F811


class RealTradingMixin:
    """
    real_trading_simulation

    Auto-extracted from MLBacktesterNoWFO.py lines 15359-18440.
    """
    def real_trading_simulation(self, config, models_to_test=None, months=1):
        """
        Simulate sequential walk-forward live trading:
        - per period: tune (or short-circuit for DQN), evaluate, log metrics,
        and carry continuous equity to the next period.

        The `months` parameter defines the total walk-forward span in months.
        When period_unit is "weeks" or "days", the span is converted to the
        equivalent number of periods for the iteration loop.

        Returns
        -------
        pd.DataFrame
            One row per successfully evaluated month.
        """      
        _prev_real = getattr(self, "_in_real_sim", False)
        _prev_dbg  = getattr(self, "_dbg_first_bars", False)
        # Real-trading sim must not inherit Optuna CV mode.
        _prev_optuna_cv = getattr(self, "_in_optuna_cv", False)

        self.bar_concat = pd.DataFrame()
        self.eq_concat  = pd.DataFrame()
        self.trade_log  = pd.DataFrame()
        self._in_real_sim = True
        self._in_optuna_cv = False

        # ------------------------------------------------------------------
        # SAVE_* toggle dicts -- control which per-month artifacts are written.
        # These were referenced but never defined (caused NameError at line 2604).
        # Defaults: enable trade CSVs, disable heavy per-month PNGs/CSVs.
        # Override via config["save_trades"] / config["save_metrics"] / etc.
        # ------------------------------------------------------------------
        SAVE_TRADES = dict(config.get("save_trades", {})) if isinstance(config.get("save_trades"), dict) else {}
        SAVE_METRICS = dict(config.get("save_metrics", {})) if isinstance(config.get("save_metrics"), dict) else {}
        SAVE_FEATURES = dict(config.get("save_features", {})) if isinstance(config.get("save_features"), dict) else {}
        SAVE_EQUITY = dict(config.get("save_equity", {})) if isinstance(config.get("save_equity"), dict) else {}
        
        # ------------------------------------------------------------
        # Freeze the baseline feature config at entry to real-trading sim
        # so monthly evaluation cannot drift due to prior trial/Month state.
        # ------------------------------------------------------------
        try:
            self._rt_sim_base_features_config = deepcopy(getattr(self, 'features_config', {}) or {})
        except Exception:
            self._rt_sim_base_features_config = {}

        # --- helper: make any timestamp tz-aware (UTC) safely ---
        def _ensure_dt(ts):
            """Convert naive Timestamp to UTC, pass aware ones through."""
            return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')

        # repetition index (for file naming); defaults to 1 if not provided
        rep_idx = int(config.get("rep", 1))

        log_print(
            "IN REAL_TRADING_SIMULATION, model_type is: " + str(config.get("model_type")),
            level="DEBUG",
        )


        # keep evaluation costs consistent across all paths
        # BUT: do not override constructor choice if it was explicitly set.
        if not getattr(self, "_trading_costs_locked", False):
            if "eval_use_trading_costs" in config:
                self.trading_costs = bool(config.get("eval_use_trading_costs"))
            elif "trading_costs" in config:
                self.trading_costs = bool(config.get("trading_costs"))
        else:
            # Optional: leave a breadcrumb in debug logs if you want
            # if debug(): print(f"[Costs] Constructor lock active -> ignoring config trading_costs override.")
            pass

            
        self.slippage_factor = float(config.get("slippage_factor", self.slippage_factor))

        def _log_flat_month_fallback(
            period_idx,
            train_start,
            train_end,
            test_start,
            test_end,
            model_type,
            full_data,
            prev_position,
            prev_eq_strategy,
            prev_eq_bh,
        ):
            """
            Log a flat no-trades month when we have no usable WFO combo
            or no valid metrics. Returns updated (prev_eq_strategy, prev_eq_bh).
            """
            cfg_f       = getattr(self, "features_config", {}) or {}
            sess_mode   = str(cfg_f.get("session_filter_mode", "both")).lower()
            use_strict  = bool(cfg_f.get("enforce_day1_start", True))

            # In real_trading_simulation always use strict day-1 anchor
            if getattr(self, "_in_real_sim", False):
                use_strict = True

            # Start from raw month slice
            test_bars = full_data.loc[test_start:test_end].copy()

            # Apply NY session filter if used during testing
            if sess_mode in ("test_only", "both"):
                if not hasattr(self, "_ny_mask") or self._ny_mask is None:
                    try:
                        full_idx  = pd.to_datetime(full_data.index, utc=True, errors="coerce")
                        _ny_times = full_idx.tz_convert("America/New_York")
                        # 02:00-13:00 NY
                        self._ny_mask = pd.Series(
                            (_ny_times.hour >= 2) & (_ny_times.hour <= 13),
                            index=full_idx,
                        )
                    except Exception as _e:
                        print(f"[WARN] Lazy NY mask build failed in flat-month fallback: {_e}")
                        self._ny_mask = pd.Series(True, index=full_data.index)
                test_bars = test_bars.loc[self._ny_mask.reindex(test_bars.index, fill_value=False)]
                
            # Tag session_flag for this month slice as well (mainly for future use
            # if we ever feed these bars into compute_full_evaluation_metrics).
            try:
                if hasattr(self, "_ny_mask") and self._ny_mask is not None:
                    _sess_month = self._ny_mask.reindex(test_bars.index, fill_value=False)
                    test_bars["session_flag"] = _sess_month.astype(int)
                else:
                    test_bars["session_flag"] = 1
            except Exception:
                test_bars["session_flag"] = 1


            # Enforce day-1 anchor + warm-up, consistent with evaluation
            month_start_dt = _ensure_dt(test_start)
            if use_strict and not test_bars.empty:
                try:
                    first_tradable = first_tradable_test_bar(test_bars.index, month_start_dt)
                except Exception as _e:
                    print(f"[WARN] first_tradable_test_bar failed in flat-month fallback: {_e}")
                    first_tradable = None

                try:
                    needed = compute_required_test_warmup_bars(
                        {**cfg_f, "model_type": model_type, "lags": int(cfg_f.get("lags", 8))}
                    )
                except Exception:
                    needed = 0

                warmups    = max(int(needed), 0)
                min_anchor = month_start_dt
                if first_tradable is not None and first_tradable > min_anchor:
                    min_anchor = first_tradable

                try:
                    anchor_idx = test_bars.index.get_loc(min_anchor, method="bfill")
                except Exception:
                    anchor_idx = 0

                if warmups > 0:
                    anchor_idx = min(anchor_idx + warmups, max(len(test_bars) - 1, 0))
                test_bars = test_bars.iloc[anchor_idx:]

            # If there are literally no bars after filters, keep both equities flat
            if test_bars.empty:
                df_flat = test_bars.copy()
                try:
                    df_flat.attrs["signal_coverage"] = 0.0
                    df_flat.attrs["end_eq_strategy"] = float(prev_eq_strategy)
                    df_flat.attrs["end_eq_bh"] = float(prev_eq_bh)
                    df_flat.attrs["last_position"] = float(prev_position)
                except Exception:
                    pass
                prev_position_out = float(prev_position)
                monthly_bh_factor      = 1.0
                equity_strategy        = float(prev_eq_strategy)
                equity_bh              = float(prev_eq_bh)
                perf                   = 1.0
                creturns               = monthly_bh_factor
                outperf                = perf - creturns
                sharpe                 = 0.0
                drawdown               = 0.0
                trades                 = 0
                geo_mean_ann           = 0.0
                directional_accuracy   = 0.0
                precision_macro        = 0.0
                f1_macro               = 0.0
                active_rate            = 0.0
                profit_per_hit         = 0.0
                return_per_trade       = 0.0
                win_rate               = 0.0
                strategy_volatility    = 0.0
                kurtosis               = 0.0
            else:
                # Build a minimal flat-strategy df and run it through the SAME engine
                df_flat = test_bars.copy()

                # Shared baseline returns: always from self.data
                df_flat["returns"] = (
                    self.data["returns"].reindex(df_flat.index).astype(float)
                )

                # Strategy is flat all month
                # Real-sim continuity: if we come in holding a position, keep it unless a model says otherwise.
                # This prevents "teleporting to flat" just because WFO returned nothing.
                df_flat["pred"] = float(prev_position)

                # Minimal spread; full cost model will read more columns if present
                if "spread" not in df_flat.columns:
                    df_flat["spread"] = 0.0

                # ------------------------------------------------------------
                # Propagate (or compute) train-anchored high_vol_thr for costs
                # so fallback never triggers LeakageGuard.
                # ------------------------------------------------------------
                cfg_cost = dict(getattr(self, "features_config", {}) or {})

                # 1) If global config (closure) already has a threshold, reuse it.
                try:
                    _thr_cfg = cfg_cost.get("high_vol_thr", None)
                    if _thr_cfg is None and isinstance(config, dict):
                        _thr_cfg = config.get("high_vol_thr", None)
                    _thr_cfg = float(_thr_cfg) if _thr_cfg is not None else None
                    if _thr_cfg is not None and np.isfinite(_thr_cfg):
                        cfg_cost["high_vol_thr"] = float(_thr_cfg)
                except Exception:
                    pass

                # 2) If still missing, compute from TRAIN window only (no leakage).
                try:
                    if cfg_cost.get("high_vol_thr", None) is None:
                        from utilsNoWFO import realized_vol as _rv_fn
                        _vol_w = int(cfg_cost.get("vol_window_bars", _PC["vol_window_bars"]))
                        _qhi   = float(cfg_cost.get("high_vol_q", _PC["high_vol_q"]))

                        train_bars = full_data.loc[train_start:train_end].copy()

                        # Apply NY session filter to TRAIN if your pipeline uses it there.
                        if sess_mode in ("train_only", "both"):
                            if hasattr(self, "_ny_mask") and self._ny_mask is not None:
                                train_bars = train_bars.loc[
                                    self._ny_mask.reindex(train_bars.index, fill_value=False)
                                ]

                        if "returns" in train_bars.columns and len(train_bars) > 0:
                            _rv_tr = _rv_fn(train_bars["returns"].astype(float), window=_vol_w)
                            _thr_tr = float(_rv_tr.quantile(_qhi))
                            if np.isfinite(_thr_tr):
                                cfg_cost["high_vol_thr"] = float(_thr_tr)
                except Exception:
                    pass

                # Ensure the evaluator/cost layer can see the config via attrs as well.
                try:
                    df_flat.attrs["features_config"] = cfg_cost
                    df_flat.attrs["debug_costs"] = bool(self._is_debug())
                except Exception:
                    pass

                # Align cost columns if trading_costs are enabled
                if bool(getattr(self, "trading_costs", True)):
                    # Ensure we pass a TRAIN-anchored high_vol_thr into the cost layer,
                    # even in flat-month fallback, to avoid LeakageGuard forcing HIGH slippage.
                    cfg_cost = dict(getattr(self, "features_config", {}) or {})
                    try:
                        _thr = cfg_cost.get("high_vol_thr", None)
                        if _thr is None:
                            from utilsNoWFO import realized_vol as _rv_fn
                            vol_w = int(cfg_cost.get("vol_window_bars", _PC["vol_window_bars"]))
                            qhi   = float(cfg_cost.get("high_vol_q", _PC["high_vol_q"]))
                            _train = self.data.loc[train_start:train_end]
                            if (
                                _train is not None
                                and hasattr(_train, "columns")
                                and "returns" in _train.columns
                                and len(_train) > max(vol_w, 5)
                            ):
                                _rv = _rv_fn(_train["returns"].astype(float), window=vol_w)
                                _thr = float(_rv.quantile(qhi))
                                if _thr is not None and np.isfinite(_thr):
                                    cfg_cost["high_vol_thr"] = float(_thr)
                    except Exception:
                        pass
                    try:
                        df_flat = self._ensure_cost_columns(
                            df_flat, cfg_cost
                        )
                    except Exception as _e:
                        print(f"[WARN] _ensure_cost_columns failed in flat-month fallback: {_e}")

                cont_metrics = compute_full_evaluation_metrics(
                    df_flat,
                    trading_costs=self.trading_costs,
                    slippage_factor=self.slippage_factor,
                    prev_position=float(prev_position),
                    prev_eq_strategy=prev_eq_strategy,
                    prev_eq_bh=prev_eq_bh,
                    eval_context="real_sim:flat_month_fallback",
                )
                
                from utilsNoWFO import validate_metrics_shape
                validate_metrics_shape(cont_metrics, context="real_sim:cont_metrics")

                (
                    perf,
                    outperf,
                    creturns,
                    sharpe,
                    drawdown,
                    trades,
                    geo_mean_ann,
                    directional_accuracy,
                    precision_macro,
                    f1_macro,
                    active_rate,
                    profit_per_hit,
                    return_per_trade,
                    win_rate,
                    strategy_volatility,
                    kurtosis,
                ) = cont_metrics

                # Pull continuous end-of-month equities from engine attrs
                equity_strategy    = float(df_flat.attrs.get("end_eq_strategy", prev_eq_strategy))
                equity_bh          = float(df_flat.attrs.get("end_eq_bh", prev_eq_bh))
                prev_position_out = float(df_flat.attrs.get("last_position", prev_position))
                monthly_bh_factor  = float(creturns)

            result = {
                "month": period_idx,
                "model": model_type,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "cstrategy": perf,
                "creturns": creturns,
                "outperformance": outperf,
                "sharpe": sharpe,
                "drawdown": drawdown,
                "trades": trades,
                "geo_mean_ann": geo_mean_ann,
                "directional_accuracy": directional_accuracy,
                "precision_macro": precision_macro,
                "f1_macro": f1_macro,
                "active_rate": active_rate,
                "signal_coverage": float(df_flat.attrs.get("signal_coverage", float("nan"))),
                "profit_per_hit": profit_per_hit,
                "return_per_trade": return_per_trade,
                "win_rate": win_rate,
                "strategy_volatility": strategy_volatility,
                "kurtosis": kurtosis,
            }

            model_name = friendly_model_name(model_type)
            model_base_dir = os.path.join(out_dir, model_name)
            month_dirs = month_dir_path(model_base_dir, period_idx)
            csv_path   = os.path.join(month_dirs["csv"], f"csv_month_{period_idx}.csv")

            self.log_simulation_result(
                i=period_idx - 1,
                test_start=test_start,
                test_end=test_end,
                perf=float(result["cstrategy"]),         # 1.0 for a flat month
                creturns=float(result["creturns"]),      # BH factor
                sharpe=float(result["sharpe"]),          # 0.0
                trades=int(result["trades"]),            # 0
                drawdown=float(result["drawdown"]),      # 0.0
                cumsum=float(equity_strategy),           # carried strategy equity
                result=result,
                csv_path=csv_path,
                directional_accuracy=float(result["directional_accuracy"]),
                precision_macro=float(result["precision_macro"]),
                f1_macro=float(result["f1_macro"]),
                active_rate=float(result["active_rate"]),
                profit_per_hit=float(result["profit_per_hit"]),
                equity_bh=float(equity_bh),
            )

            results.append(result)
            # Keep the cross-month curve continuous even in a flat-month fallback.
            # NOTE: test_bars is raw market data; we want the evaluated *_cont curves.
            try:
                _month_df = df_flat[["cstrategy_cont", "creturns_cont"]].copy()
                if self._monthly_all_dfs_concat is None:
                    self._monthly_all_dfs_concat = _month_df
                else:
                    self._monthly_all_dfs_concat = pd.concat([self._monthly_all_dfs_concat, _month_df])
                del _month_df
            except Exception:
                pass

            try:
                _month_trade = build_trade_log_from_df(df_flat)
                if self._monthly_trade_dfs_concat is None:
                    self._monthly_trade_dfs_concat = _month_trade
                else:
                    self._monthly_trade_dfs_concat = pd.concat([self._monthly_trade_dfs_concat, _month_trade], ignore_index=True)
                del _month_trade
            except Exception:
                pass

            print(
                f"[FLAT] Month {period_idx}: logged FLAT month "
                f"(no usable config / no valid metrics)."
            )

            prev_eq_strategy = equity_strategy
            prev_eq_bh       = equity_bh
            
            return prev_eq_strategy, prev_eq_bh, prev_position_out



        def get_first(val, default):
            
            if isinstance(val, (list, tuple)):
                return val[0]
            return val if val is not None else default

        def _is_valid_metrics_tuple(t):
            """Expect a 16-tuple. Reject NaN/Inf/sentinel and no-trade configs."""
            if t is None or not isinstance(t, (list, tuple)) or len(t) < 16:
                return False
            perf, outperf, creturns, sharpe, drawdown, trades, geo_mean_ann, \
            directional_accuracy, precision_macro, f1_macro, active_rate, profit_per_hit, \
            return_per_trade, win_rate, volatility, excess_kurtosis = t

            # In small-sample months (or very flat paths), Sharpe/outperf may be NaN.
            # That should NOT zero out the entire month or prune an Optuna trial.
            # We require the equity results themselves to be finite, and treat
            # non-finite diagnostics as 0.0 for validity purposes.
            try:
                sharpe_v = 0.0 if sharpe is None else float(sharpe)
            except Exception:
                sharpe_v = 0.0
            if not np.isfinite(sharpe_v):
                sharpe_v = 0.0

            try:
                outperf_v = float(outperf)
            except Exception:
                outperf_v = 0.0
            if not np.isfinite(outperf_v):
                outperf_v = 0.0

            arr = np.array([perf, creturns])
            if np.any(~np.isfinite(arr)):
                return False

            if perf <= -9999 or creturns <= -9999:
                return False
            try:
                active_v = float(active_rate)
            except Exception:
                active_v = -1.0
            if not np.isfinite(active_v) or active_v <= 0:
                return False

            try:
                trades_v = float(trades)
            except Exception:
                trades_v = -1.0
            if not np.isfinite(trades_v) or trades_v < 0:
                return False

            return True

        model_type = config.get("model_type", "svm")
        log_print(f"\n[ANNOUNCE] Starting Real Trading Simulation for {months} month(s)", level="COMPACT")
        log_print(
            f"[STRATEGY] Strategy: {model_type.upper()} | Logging results per month...",
            level="COMPACT",
        )

        out_dir, _stamp = make_results_run_dir() 
        full_data = self.data.copy()
        for col in full_data.columns:
            if pd.api.types.is_float_dtype(full_data[col]):
                full_data[col] = full_data[col].astype("float32", copy=False)
        
        # FeatureBank: keep a stable source across month slices so base indicators
        # are computed once and then reindexed to each month.
        try:
            self.set_feature_bank_source(full_data)
        except Exception:
            # Hard-fail would be silly; the system can always fall back to per-slice TA.
            pass

        # Pre-compute BH curve and fire simulation_started BEFORE HPO
        # so the frontend can show the buy-and-hold line immediately
        try:
            _mt_for_bh = model_type
            _train_m = get_first(config.get("train_months"), TRAIN_TEST_MONTHS[_mt_for_bh]["train"][0])
            _test_m = get_first(config.get("test_months"), TRAIN_TEST_MONTHS[_mt_for_bh]["test"][0])
            _pu_bh = config.get("period_unit", "months")
            from config import convert_month_count_to_periods as _cvt_bh, period_offset as _po_bh
            _np_bh = _cvt_bh(months, _pu_bh)
            if _np_bh > 0:
                bh_curve = []
                _start_dt_bh = pd.to_datetime(self.start) if self.start is not None else self.data.index[0]
                _warmup_bh = _cvt_bh(37, _pu_bh)  # same conversion as month loop
                _test_p_bh = _cvt_bh(_test_m, _pu_bh)
                cum_bh = 1.0
                for i in range(_np_bh):
                    test_s = _start_dt_bh + _po_bh(_warmup_bh + i, unit=_pu_bh)
                    test_e = test_s + _po_bh(_test_p_bh, unit=_pu_bh)
                    td = full_data.loc[(full_data.index >= test_s) & (full_data.index < test_e)]
                    if len(td) > 1 and "mid_close" in td.columns:
                        cum_bh *= float((1 + td["mid_close"].pct_change().dropna()).prod())
                    elif len(td) > 1 and "close" in td.columns:
                        cum_bh *= float((1 + td["close"].pct_change().dropna()).prod())
                    bh_curve.append({"period": i + 1, "bh": round(float(cum_bh), 6)})
                _progress_cb_pre = getattr(self, "_progress_callback", None)
                if _progress_cb_pre:
                    _progress_cb_pre("simulation_started", _mt_for_bh, {
                        "n_periods": _np_bh,
                        "bh_curve": bh_curve,
                    })
                if _np_bh > 0:
                    log_print(
                        f"[BH] Pre-computed {_np_bh} BH points for {_mt_for_bh} "
                        f"(last_bh={bh_curve[-1]['bh']})",
                        level="COMPACT",
                    )
        except Exception as _bh_err:
            log_print(f"[BH] Pre-computation skipped for {model_type}: {_bh_err}", level="COMPACT")

        # --- Global HPO: tune once per run, reuse hyperparameters every month (non-DQN only) ---
        global_hpo_best = None
        global_hpo_topN = None

        if model_type != "dqn":
            # Decide whether to reuse a cached config or force a fresh study.
            # Default behaviour (use_cached_global_hpo=False) is to always run a
            # new global HPO for this run and overwrite any stale cache.
            use_cached_global_hpo = bool(
                config.get(
                    "use_cached_global_hpo",
                    DEFAULT_CV.get("use_cached_global_hpo", False),
                )
            )

            # If the user requested n_trials <= 0, we must load a cached HPO config
            # and skip Optuna entirely (otherwise the tuner will have zero trials and crash).
            _req_trials = int(config.get("n_trials", 0) or 0)
            _force_cached_hpo = (_req_trials <= 0)
            if _force_cached_hpo:
                use_cached_global_hpo = True

            # 1) Optionally try to load from disk (if reuse is allowed)
            if use_cached_global_hpo:
                try:
                    global_hpo_best, global_hpo_topN = load_hpo_config_from_disk(model_type)
                except Exception:
                    global_hpo_best, global_hpo_topN = None, None
                    
            # If we explicitly forced cached HPO (n_trials <= 0), missing cache falls back to defaults.
            if _force_cached_hpo:
                if (not isinstance(global_hpo_best, dict)) or (not global_hpo_best):
                    log_print(
                        f"[HPO] n_trials=0 and no cached HPO config found for {model_type}. "
                        f"Searched: {HPO_CONFIG_DIR} (set MLB_HPO_DIR to override). "
                        f"Falling back to default parameters.",
                        level="COMPACT",
                    )
                    global_hpo_best = None
                else:
                    log_print(
                        f"[HPO] Using cached global HPO for {model_type} (n_trials=0).",
                        level="COMPACT",
                    )

            # 2) If cache is disabled or missing/invalid, run a single Optuna study now
            if (not _force_cached_hpo) and ((not use_cached_global_hpo) or (not isinstance(global_hpo_best, dict) or not global_hpo_best)):
                log_print(
                    f"[HPO] Running ONE global Optuna study for {model_type} "
                    f"(use_cached_global_hpo={use_cached_global_hpo})...",
                    level="COMPACT",
                )
                hpo_cfg = dict(config)  # shallow copy is enough

                _progress_cb = getattr(self, "_progress_callback", None)
                if _progress_cb and isinstance(hpo_cfg, dict):
                    hpo_cfg["_progress_callback"] = _progress_cb

                # Ensure HPO-only flags
                hpo_cfg["hpo_only"] = True
                hpo_cfg.setdefault("hpo_save_to_disk", True)

                # Use TRIAL_COUNTS to configure n_trials / n_startup_trials if not already set
                tc = TRIAL_COUNTS.get(model_type, {})
                default_random = int(tc.get("random", hpo_cfg.get("n_startup_trials", 10)))
                default_bayes = int(tc.get("bayes", max(hpo_cfg.get("n_trials", 30) - default_random, 0)))

                if "n_startup_trials" not in hpo_cfg:
                    hpo_cfg["n_startup_trials"] = default_random
                if "n_trials" not in hpo_cfg:
                    hpo_cfg["n_trials"] = default_random + default_bayes

                # Fix 3: enforce minimum trial counts to survive a few failures
                _min_total = 10
                if int(hpo_cfg.get("n_trials", 0)) < _min_total:
                    hpo_cfg["n_trials"] = _min_total
                if int(hpo_cfg.get("n_startup_trials", 0)) < 5:
                    hpo_cfg["n_startup_trials"] = 5

                # IMPORTANT: for HPO we want the full dataset available, not the per-month slice
                if hasattr(self, "data"):
                    del self.data
                self.data = full_data.copy()

                # We only need params for this model_type
                try:
                    res_hpo = self.run_strategy(
                        hpo_cfg,
                        models_to_test=[model_type],
                        n_trials=hpo_cfg["n_trials"],
                        n_startup_trials=hpo_cfg["n_startup_trials"],
                    )
                except RuntimeError as _hpo_err:
                    _hpo_msg = str(_hpo_err)
                    if "No completed Optuna trials" in _hpo_msg:
                        log_print(
                            f"[WARN] Global HPO failed: {_hpo_msg[:200]}... "
                            f"Falling back to per-month WFO tuning.",
                            level="COMPACT",
                        )
                        global_hpo_best = None
                        self._global_hpo_best = None
                        self._global_hpo_topN = []
                        # Skip the rest of the HPO block
                        _hpo_failed = True
                    else:
                        raise
                else:
                    _hpo_failed = False
                    # In hpo_only mode we return (None, best_params)
                    if isinstance(res_hpo, tuple) and len(res_hpo) >= 2 and isinstance(res_hpo[1], dict):
                        global_hpo_best = res_hpo[1]
                    else:
                        global_hpo_best = getattr(self, "_optuna_best_for_wfo", None)

                # If the study persisted a Top-N pool, load it back from disk; otherwise
                # fall back to any in-memory Top-5 captured during the study.
                try:
                    _best_tmp, global_hpo_topN = load_hpo_config_from_disk(model_type)
                except Exception:
                    try:
                        global_hpo_topN = getattr(self, "_optuna_top5_for_wfo", None)
                    except Exception:
                        global_hpo_topN = None

            # Cache on the instance for the monthly loop
            self._global_hpo_best = global_hpo_best
            self._global_hpo_topN = global_hpo_topN or []
        else:
            # DQN: no global HPO, keep behaviour unchanged there
            self._global_hpo_best = None
            self._global_hpo_topN = []



        results = []
        all_dfs = []
        trade_dfs = []   # per-month trade DataFrames (aligned with all_dfs / results)
        self._monthly_all_dfs_concat = None
        self._monthly_trade_dfs_concat = None
        _month_ix = 0

        _progress_cb = getattr(self, "_progress_callback", None)

        if _progress_cb and model_type != "dqn":
            _progress_cb("hpo", model_type, {"n_trials": int(config.get("n_trials", 6)), "cv_blocks": 5})

        if _progress_cb:
            _progress_cb("model_phase", model_type, {"phase": "simulation"})

        # Reset per-run PBO/MCS accumulator
        self._wfo_monthly_records = []

        train_months = get_first(
            config.get("train_months"),
            TRAIN_TEST_MONTHS[model_type]["train"][0]
        )
        test_months = get_first(
            config.get("test_months"),
            TRAIN_TEST_MONTHS[model_type]["test"][0]
        )
        period_unit = config.get("period_unit", "months")

        # Convert total walk-forward span (in months) to number of periods
        from config import convert_month_count_to_periods as _cvt_periods
        n_periods = _cvt_periods(months, period_unit)

        if period_unit != "months" and n_periods > 365:
            log_print(
                f"[WARN] period_unit={period_unit} with months={months} produces "
                f"{n_periods} iterations -- this may be very slow. Consider reducing months.",
                level="COMPACT",
            )

        # ---- carry continuous state across periods ----
        prev_eq_strategy = 1.0
        prev_eq_bh = 1.0
        prev_position = 0.0
        
        # --- Build model output tree (Months/Final) -- do NOT rely on a global RUN_DIR ---
        disp_name = friendly_model_name(model_type)
        
        # --- helper: map model_type -> family folder name
        def _infer_family(m: str) -> str:
            m = (m or "").lower()
            classical = {"logistic", "svm", "decision_tree", "random_forest", "xgboost"}
            rl        = {"cnn", "lstm", "transformer"}      # deep models only
            dqn       = {"dqn"}
            ensembles = {"ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost"}
            if m in classical:  return "Classical"
            if m in rl:         return "RL"
            if m in dqn:        return "DQN"
            if m in ensembles:  return "Ensembles"
            return "Classical"

        # --- Determine Top-N size by model family for second-study fallback ---
        mt_local = str(config.get("model_type", model_type)).lower()
        fam = _infer_family(mt_local)
        cfg_local = getattr(self, "features_config", {}) or {}
        if fam == "Classical":
            rt_n = int(cfg_local.get("topN_classical", 4))
        elif fam == "RL":
            rt_n = int(cfg_local.get("topN_deep", 3))
        elif fam == "Ensembles":
            rt_n = int(cfg_local.get("topN_ensemble", 2))
        elif fam == "DQN":
            rt_n = int(cfg_local.get("topN_dqn", 2))
        else:
            rt_n = int(cfg_local.get("topN_default", 3))
        rt_n = max(1, rt_n)

        # Prefer a run dir injected by main(), else the env var, else fall back to this method's out_dir
        RUN_DIR_LOCAL = config.get("_run_dir") or os.environ.get("RESULTS_RUN_DIR") or out_dir
        os.makedirs(RUN_DIR_LOCAL, exist_ok=True)
        
        buckets = comparison_dirs(RUN_DIR_LOCAL) 

        # Compat: allow ensure_model_dirs to return (base, dict) OR a dict
        ret = ensure_model_dirs(RUN_DIR_LOCAL, model_type, disp_name)        
        
        
        if isinstance(ret, tuple):
            model_base_dir, model_dirs_meta = ret
        elif isinstance(ret, dict):
            model_dirs_meta = ret
            # try common keys for the base path
            model_base_dir = (model_dirs_meta.get("base")
                            or model_dirs_meta.get("root")
                            or model_dirs_meta.get("dir")
                            or model_dirs_meta.get("path"))
            if not model_base_dir:
                raise RuntimeError("ensure_model_dirs returned dict without a base path key")
        else:
            raise TypeError(f"Unexpected return from ensure_model_dirs: {type(ret)}")

        final_dirs = model_dirs_meta["final"]
        
        os.makedirs(final_dirs["csv"], exist_ok=True)
        csv_path = os.path.join(final_dirs["csv"], f"real_trading_simulation_{model_type}.csv")

        # --- DQN periodic retraining settings (ignored for other models) ---
        if model_type == "dqn":
            dqn_retrain_period_months = int(config.get("dqn_retrain_period_months", 12))
            dqn_retrain_period = _cvt_periods(dqn_retrain_period_months, period_unit)
            dqn_period_counter = 0
        else:
            dqn_retrain_period = None
            dqn_period_counter = 0
            
        # --- helper: deterministic month config fingerprint (auditable) ---
        def _month_cfg_fingerprint(_cfg: dict):
            """Return (short_sha1, compact_dict) for a stable month-config snapshot."""
            try:
                _keys = [
                    "model_type",
                    "gating_mode",
                    "confidence_threshold",
                    "target_active_rate",
                    "calibrate_method",
                    "label_threshold",
                    "lags",
                    "lags_range",
                    "lag_depth",
                    "roll_windows_key",
                    "roll_windows_key_v2",
                    "runtime_active_band_margin",
                    "runtime_conf_nudge",
                    "runtime_coverage_window",
                    "alpha_vol_z",
                    "beta_spread_norm",
                    "gamma_slip_norm",
                    "vol_window_bars",
                    "high_vol_q",
                    "slip_norm_bps",
                    "eval_use_trading_costs",
                ]
                _snap = {k: _cfg.get(k, None) for k in _keys if k in _cfg}
                _ser = json.dumps(_snap, sort_keys=True, default=str)
                _fp = hashlib.sha1(_ser.encode("utf-8")).hexdigest()[:10]
                return _fp, _snap
            except Exception:
                return "na", {}

        for i in range(n_periods):
            period_idx = i + 1
            
            # V2 export safety: always define, then overwrite if evaluator provides it
            _signal_coverage_month = float("nan")
            
            # ------------------------------------------------------------
            # Allows ctx=real_mX tagging in test_strategy() eligibility prints.
            # ------------------------------------------------------------
            try:
                # NOTE: keep both spellings; downstream log/ctx code reads _rt_month_ix.
                setattr(self, "_rt_month_idx", int(period_idx))
                setattr(self, "_rt_month_ix", int(period_idx))

            except Exception:
                pass
            try:
                _start_dt = pd.to_datetime(self.start) if self.start is not None else self.data.index[0]
                from config import period_offset, convert_month_count_to_periods as _cvt
                _pu = period_unit
                _warmup = _cvt(37, _pu)
                _train_p = _cvt(train_months, _pu)
                _test_p = _cvt(test_months, _pu)
                _pad_p = _cvt(1, _pu)
                test_start_naive  = _start_dt + period_offset(_warmup + i, unit=_pu)
                test_end_naive    = test_start_naive + period_offset(_test_p, unit=_pu)
                train_start_naive = test_start_naive - period_offset(_train_p, unit=_pu)
                train_end_naive   = test_start_naive - pd.Timedelta(minutes=30)

                # make the slices tz-aware (UTC) -- uses _ensure_dt defined at method top
                test_start  = _ensure_dt(test_start_naive)
                test_end    = _ensure_dt(test_end_naive)
                train_start = _ensure_dt(train_start_naive)
                train_end   = _ensure_dt(train_end_naive)

                if train_end >= test_start:
                    print(f"[ERR] Sanity check failed: train_end ({train_end}) is not before test_start ({test_start})")
                    continue

                train_start_nominal = test_start - period_offset(_train_p, unit=_pu)
                train_start = max(full_data.index[0], train_start_nominal)
                if train_start > train_start_nominal:
                    log_print(
                        f"[INFO] Train window truncated to data start: "
                        f"{train_start_nominal.date()} -> {train_start.date()}",
                        level="COMPACT",
                    )
                log_print(
                    f"[DATE] Month {i+1}/{months}: "
                    f"Tuning on {train_start.date()} -> {train_end.date()} | "
                    f"Testing on {test_start.date()} -> {test_end.date()}",
                    level="COMPACT",
                )


                data_slice = full_data.loc[train_start - period_offset(_pad_p, unit=_pu):test_end].copy()
                log_print(
                    f"[DATA] Data shape for training/testing window: {data_slice.shape}",
                    level="DEBUG",
                )
                
                # ------------------------------------------------------------
                # Compute ONCE per month from the TRAIN window and cache it so
                # every evaluation path (single model, Top-N, consensus) uses
                # the same volatility regime split without leakage.
                # ------------------------------------------------------------
                try:
                    from utilsNoWFO import realized_vol as _realized_vol
                    _thr_month = None

                    _train_for_thr = full_data.loc[train_start:train_end]

                    # Optional: match train-side session filtering if enabled
                    try:
                        _fc = getattr(self, "features_config", {}) or {}
                        _sf_mode = str(_fc.get("session_filter_mode", "none")).lower()
                        _sf_on_train = bool(_fc.get("session_filter_on_train", False))
                        if _sf_on_train and _sf_mode in ("train", "both") and hasattr(self, "_ny_mask"):
                            _m = getattr(self, "_ny_mask")
                            _train_for_thr = _train_for_thr.loc[_m.reindex(_train_for_thr.index).fillna(False).values]
                    except Exception:
                        pass

                    if isinstance(_train_for_thr, pd.DataFrame) and "returns" in _train_for_thr.columns and len(_train_for_thr) > 10:
                        _vol_w = int((config or {}).get("vol_window_bars", (getattr(self, "features_config", {}) or {}).get("vol_window_bars", _PC["vol_window_bars"])))
                        _qhi   = float((config or {}).get("high_vol_q", (getattr(self, "features_config", {}) or {}).get("high_vol_q", _PC["high_vol_q"])))
                        _rv = _realized_vol(pd.to_numeric(_train_for_thr["returns"], errors="coerce").astype(float), window=_vol_w)
                        _rv = _rv.dropna()
                        if len(_rv) > 0:
                            _thr_month = float(_rv.quantile(_qhi))

                    setattr(self, "_last_high_vol_thr_train", _thr_month)

                    # Mirror into configs used downstream (safe; only affects cost regime split)
                    try:
                        if isinstance(getattr(self, "features_config", None), dict):
                            if _thr_month is not None:
                                self.features_config["high_vol_thr"] = float(_thr_month)
                            else:
                                self.features_config.pop("high_vol_thr", None)
                    except Exception:
                        pass
                    try:
                        if isinstance(config, dict):
                            if _thr_month is not None:
                                config["high_vol_thr"] = float(_thr_month)
                            else:
                                config.pop("high_vol_thr", None)
                    except Exception:
                        pass

                    if bool(getattr(self, "debug", False)):
                        log_print(f"[DEBUG][Costs] high_vol_thr_train={_thr_month} | ctx=real_sim:m{i+1}", level="DEBUG")
                except Exception:
                    # If anything goes wrong, leave the cache unset; downstream will use BASE slippage.
                    try:
                        setattr(self, "_last_high_vol_thr_train", None)
                    except Exception:
                        pass

                    
                # -- DQN short-circuit: evaluate directly (no Optuna) --
                if model_type == "dqn":
                    try:
                        del self.data
                    except Exception:
                        pass
                    self.data = data_slice

                    lags_val = int(
                        config.get(
                            "lags",
                            getattr(self, "features_config", {}).get(
                                "lags",
                                getattr(self, "features_config", {}).get(
                                    "lags_range", 10
                                ),
                            ),
                        )
                    )

                    # ---- DQN periodic retrain logic ----
                    # We always run with use_pretrained=True so that:
                    # - If no model files exist for this month -> train + save new policy.
                    # - If files exist -> load and reuse the saved policy (no retrain).
                    if dqn_retrain_period:
                        # At the START of each retrain block (every N months),
                        # delete only the *model* so the next evaluation will train a new one.
                        # KEEP the JSON config so your manual settings persist.
                        if (dqn_period_counter % dqn_retrain_period) == 0:
                            try:
                                if os.path.exists(MODEL_DQN_PATH):
                                    os.remove(MODEL_DQN_PATH)
                                    
                                # IMPORTANT: do NOT remove DQN_AGENT_CONFIG_PATH here (keeps pretrained config alongside weights).
                                log_print(
                                    f"[DQN] Starting new retrain block at month {i + 1}; "
                                    f"removed old pretrained DQN model (kept config).",
                                    level="COMPACT",
                                )
                            except Exception as e:
                                log_print(
                                    f"[WARN] Could not remove old DQN model file: {e}",
                                    level="COMPACT",
                                )

                    raw_dqn_cfg = config.get("dqn_config", None)
                    if isinstance(raw_dqn_cfg, dict) and len(raw_dqn_cfg) > 0:
                         dqn_cfg = dict(raw_dqn_cfg)
                         cfg_source = "inline"
                    else:
                         dqn_cfg = _load_default_dqn_cfg(DQN_GRID_CONFIG_PATH)
                         cfg_source = "grid"
                    print(f"[DQN][CFG] source={cfg_source} episodes={dqn_cfg.get('episodes')} path={DQN_GRID_CONFIG_PATH}")

                    # Ensure DQN path uses the pretrained behaviour in test_dqn_strategy
                    dqn_cfg.setdefault("use_pretrained", True)

                    best_combo = {
                        "model_type": "dqn",
                        "lags": lags_val,
                        "use_extended_features": config.get(
                            "use_extended_features", True
                        ),
                        "dqn_config": dqn_cfg,
                    }

                    try:
                        metrics = self.evaluate_strategy(
                            best_combo, train_start, train_end, test_start, test_end
                        )
                        # Increment DQN month counter only on successful evaluation
                        if dqn_retrain_period:
                            dqn_period_counter += 1
                    except Exception as e:
                        print(f"[ERR] DQN evaluation failed for month {i + 1}: {e}")
                        metrics = None

                else:
                    # ----------- Non-DQN monthly evaluation -----------
                    # If global_hpo_once is enabled and we have a cached global
                    # config, reuse it here and *do not* run Optuna/WFO again.
                    use_global_hpo = bool(config.get("global_hpo_once", True))

                    df_wfo = None
                    best_combo = None
                    wfo_ok = False

                    if use_global_hpo:
                        # Try to reuse the globally tuned parameters from the header
                        best_combo = getattr(self, "_global_hpo_best", None)
                        topN_from_header = getattr(self, "_global_hpo_topN", None) or []

                        if isinstance(best_combo, dict) and best_combo:
                            # If Top-N list exists but is not attached to the config dict,
                            # attach it so consensus logic works.
                            if topN_from_header and "__top5_params" not in best_combo:
                                best_combo["__top5_params"] = topN_from_header

                            wfo_ok = True
                            print(
                                f"[GlobalHPO] Month {i+1}: reusing globally tuned params for "
                                f"{model_type} (no Optuna/WFO in monthly loop)."
                            )
                        else:
                            print(
                                f"[GlobalHPO] Month {i+1}: no cached global HPO config found "
                                f"for {model_type}; falling back to per-month WFO tuning."
                            )

                    if not wfo_ok:
                        # ----------- Legacy Optuna WFO fallback -----------
                        try:
                            try:
                                del self.data
                            except Exception:
                                pass
                            self.data = data_slice
                            config["use_proba"] = config.get("use_proba", True)

                            month_dirs = month_dir_path(model_base_dir, i + 1)

                            config["month_ix"] = int(i + 1)
                            config["month_graphs_dir"] = month_dirs["graphs"]

                            # [WARN] run_strategy may return None (e.g. ensembles with no valid trials)
                            res = self.run_strategy(
                                config,
                                models_to_test=models_to_test,
                                n_trials=config.get("n_trials", 30),
                                n_startup_trials=config.get("n_startup_trials", 10),
                            )
                            if isinstance(res, tuple) and len(res) >= 2:
                                df_wfo, best_combo = res[0], res[1]
                            else:
                                df_wfo, best_combo = None, None

                            # --- Inject Top-5 params into best_combo for Top-N consensus ---
                            try:
                                top5_params_rt = getattr(self, "_optuna_top5_for_wfo", None) or []
                            except Exception:
                                top5_params_rt = []

                            if (
                                isinstance(best_combo, dict)
                                and top5_params_rt
                                and "__top5_params" not in best_combo
                            ):
                                best_combo["__top5_params"] = top5_params_rt
                                print(
                                    f"[TopN] Attached {len(top5_params_rt)} tuned configs "
                                    "to best_combo for real-trading consensus."
                                )

                        except Exception as e:
                            # Hard fail: do NOT try to restart Optuna here.
                            # We want to see the real error and stop the study.
                            print(f"[ERR] run_strategy failed in primary WFO: {e}")
                            raise

                        # --- WFO result can be empty if every fold is 0-trade or pruned ---
                        wfo_ok = isinstance(df_wfo, pd.DataFrame) and isinstance(best_combo, dict)
                        if not wfo_ok:
                            # Do NOT crash: this just means no usable config was found.
                            # We let the downstream flat-month fallback handle it.
                            print(
                                f"[WARN] WFO returned no usable result for month {i + 1} "
                                f"(df_wfo={type(df_wfo)}, best_combo={type(best_combo)}). "
                                "Skipping model evaluation and logging a flat no-trades month."
                            )

                    # Restore full_data slice before either evaluation or flat-month fallback
                    try:
                        del self.data
                    except Exception:
                        pass
                    self.data = full_data.loc[train_start - period_offset(_pad_p, unit=_pu):test_end].copy()

                    # Metrics placeholder for this month/model (filled by consensus / Top-3 / single-best)
                    metrics = None


                    # Only build / run Top-N + adaptive Top-3 if WFO actually produced a config
                    if wfo_ok:
                             
                        def _evaluate_with_topn_consensus(base_params):
                            """
                            Build a small Top-N ensemble over the tuned parameter set.

                            High-level intent:
                            - Start from a pool of tuned candidates (Optuna Top-N trials per model type).
                            - Filter for (optionally) style coherence + local similarity (geometry ball) + perf floor.
                            - Form a small committee: base + (N_target-1) best neighbours (by CV objective value).
                            - Evaluate each candidate on the SAME month, align on common index, majority-vote preds in {-1,0,+1}.
                            - Compute metrics for the consensus pred using compute_full_evaluation_metrics.

                            Return:
                            metrics tuple (same format as your evaluation path) or None (caller falls back to single-model logic).
                            """
                            import numpy as _np
                            import pandas as _pd
                            import json as _json
                            import math as _math
                            
                            # --- Robust unwrap: accept either {"best_params": {...}} or the inner dict ---
                            # If caller accidentally passes the outer JSON (model_type/best_params/topN_params),
                            # consensus keys live under best_params, so unwrap here.
                            if (
                                isinstance(base_params, dict)
                                and isinstance(base_params.get("best_params"), dict)
                            ):
                                base_params = base_params["best_params"]


                            cfg_local = getattr(self, "features_config", {}) or {}
                            if not bool(cfg_local.get("deploy_topN_consensus", True)):
                                return None

                            mt_local = str(base_params.get("model_type", getattr(self, "model_type", ""))).lower()
                            # keep behaviour unchanged for RL/DQN (and anything you explicitly want excluded)
                            if mt_local in {"dqn"}:
                                return None

                            # ---------- committee size ----------
                            try:
                                from utilsNoWFO import _infer_family
                                family = _infer_family(mt_local)
                            except Exception:
                                family = "Unknown"

                            if family == "Classical":
                                N_target = int(cfg_local.get("topN_classical", 3))
                            elif family in {"RL"}:  # your code uses "RL" bucket for deep supervised
                                N_target = int(cfg_local.get("topN_deep", 2))
                            elif family == "Ensembles":
                                N_target = int(cfg_local.get("topN_ensemble", 2))
                            elif family == "DQN":
                                N_target = int(cfg_local.get("topN_dqn", 2))
                            else:
                                N_target = int(cfg_local.get("topN_default", 2))

                            N_target = max(2, int(N_target))

                            debug_topn    = bool(cfg_local.get("print_topN_debug", True))
                            style_lock    = bool(cfg_local.get("topN_style_lock", True))
                            geom_radius   = float(cfg_local.get("topN_geom_radius", 0.30))
                            min_perf_frac = float(cfg_local.get("topN_min_perf_frac", 0.60))
                            max_corr      = float(cfg_local.get("topN_max_corr", 0.95))

                            # ---------- 1) pool ----------
                            raw_pool = (
                                base_params.get("__committee_fixed")
                                or base_params.get("__consensus_pool")
                                or base_params.get("__top5_params")
                                or base_params.get("__top3_params")
                                or []
                            )

                            pool_src = (
                                "committee_fixed"
                                if isinstance(base_params.get("__committee_fixed"), list) and len(base_params.get("__committee_fixed")) > 0
                                else ("consensus_pool" if "__consensus_pool" in base_params
                                    else ("top5_params" if "__top5_params" in base_params else "top3_params"))
                            )
                            if not raw_pool:
                                return None

                            raw_topk = []
                            for x in (raw_pool or []):
                                try:
                                    raw_topk.append(dict(x or {}))
                                except Exception:
                                    continue
                            if not raw_topk:
                                return None

                            # ---------- 2) CV meta (direction + values) ----------
                            top_info        = base_params.get("__top5_info") or {}
                            trials_info     = list((top_info.get("trials") or [])) if isinstance(top_info, dict) else []
                            dir_str         = str(top_info.get("direction", "maximize")).lower() if isinstance(top_info, dict) else "maximize"
                            is_minimize     = dir_str.startswith("min")

                            # keep pool + trials_info aligned ONLY for Top-K pools that are index-aligned to trials_info.
                            # For consensus/frozen committees, ordering can differ and per-item metadata should be trusted.
                            pool_src = (
                                "committee_fixed" if isinstance(base_params.get("__committee_fixed"), list) and len(base_params.get("__committee_fixed")) > 0
                                else ("consensus_pool" if isinstance(base_params.get("__consensus_pool"), list) and len(base_params.get("__consensus_pool")) > 0
                                      else "top5_params")
                            )

                            if pool_src != "committee_fixed":
                                # trials_info alignment is only valid for top5_params (same ordering).
                                if pool_src == "top5_params":
                                    if trials_info and len(raw_topk) > len(trials_info):
                                        raw_topk = raw_topk[:len(trials_info)]
                            def _meta_for_pool(idx, alt_dict):
                                """Extract trial_number + objective value for this pool row (robust across formats)."""
                                meta = {}
                                try:
                                    # 1) Prefer metadata stored on the pool dict itself (authoritative)
                                    if isinstance(alt_dict, dict):
                                        tn = alt_dict.get("__trial_number", alt_dict.get("trial_number", None))
                                        vv = alt_dict.get("__cv_value", alt_dict.get("cv_value", alt_dict.get("value", None)))
                                        if tn is not None:
                                            meta["trial_number"] = int(tn)
                                        if vv is not None:
                                            meta["value"] = float(vv)

                                    if meta:
                                        return meta

                                    # 2) Fallback: trials_info aligned by index (less reliable)
                                    if trials_info and idx < len(trials_info):
                                        row = trials_info[idx] or {}
                                        tn = row.get("number", row.get("trial_number", None))
                                        vv = row.get("value", row.get("cv_value", row.get("cv", None)))
                                        if tn is not None:
                                            meta["trial_number"] = int(tn)
                                        if vv is not None:
                                            meta["value"] = float(vv)
                                except Exception:
                                    meta = {}
                                return meta


                            def _meta_value(meta):
                                try:
                                    v = float(meta.get("value")) if isinstance(meta, dict) and meta.get("value") is not None else None
                                    return v if (v is not None and _np.isfinite(v)) else None
                                except Exception:
                                    return None

                            def _meta_trial(meta):
                                try:
                                    t = meta.get("trial_number") if isinstance(meta, dict) else None
                                    return int(t) if t is not None else None
                                except Exception:
                                    return None

                            # compute best_val for perf floor
                            best_val = None
                            vals = []

                            if trials_info:
                                for row in trials_info:
                                    try:
                                        v = float((row or {}).get("value"))
                                        if _np.isfinite(v):
                                            vals.append(v)
                                    except Exception:
                                        pass
                            if not vals:
                                for j, alt in enumerate(raw_topk):
                                    mv = _meta_value(_meta_for_pool(j, alt))
                                    if mv is not None:
                                        vals.append(mv)
                            if vals:
                                best_val = float(_np.min(vals) if is_minimize else _np.max(vals))

                            def _passes_perf_floor(v):
                                # Keep only candidates reasonably close to the best objective value.
                                # IMPORTANT: objectives can be negative in trading; ratio comparisons break on negatives.
                                try:
                                    if best_val is None or v is None:
                                        return True
                                    if not (_np.isfinite(float(best_val)) and _np.isfinite(float(v))):
                                        return False
                                    bv = float(best_val)
                                    vv = float(v)
                                    frac = float(min_perf_frac)
                                    if frac <= 0.0:
                                        return True
                                    if frac >= 1.0:
                                        return (vv <= bv) if is_minimize else (vv >= bv)
                                    tol = abs(bv) * (1.0 - frac)
                                    # minimize: allow up to +tol above best; maximize: allow down to -tol below best
                                    return (vv <= bv + tol) if is_minimize else (vv >= bv - tol)
                                except Exception:
                                    return True

                            # ---------- 3) geometry function (DEFINED BEFORE USE) ----------
                            # Ranges aligned with your tuning ranges (roughly); used only for normalization.
                            HP_RANGES = {
                                "lags_range":         (8.0, 40.0),
                                "lag_depth":          (1.0, 4.0),
                                "target_active_rate": (0.15, 0.35),
                                "label_threshold":    (5e-5, 5e-3),
                                "alpha_vol_z":        (0.0, 0.03),
                                "beta_spread_norm":   (0.0, 0.08),
                                "gamma_slip_norm":    (0.0, 0.08),
                            }

                            def _norm_dist(a, b, key):
                                lo, hi = HP_RANGES.get(key, (None, None))
                                if lo is None or hi is None or hi <= lo:
                                    return _math.inf
                                if a is None or b is None:
                                    return _math.inf
                                try:
                                    return abs(float(a) - float(b)) / (hi - lo)
                                except Exception:
                                    return _math.inf

                            def _within_geom_ball(base_cfg, alt_cfg, radius):
                                """Normalized max-distance ball in key structural/cost-sensitive knobs."""
                                try:
                                    r = float(radius or 0.0)
                                    if r <= 0:
                                        return True

                                    base_vals = {
                                        "lags_range":         base_cfg.get("lags_range", base_cfg.get("lags", None)),
                                        "lag_depth":          base_cfg.get("lag_depth", None),
                                        "target_active_rate": base_cfg.get("target_active_rate", base_cfg.get("target_coverage", None)),
                                        "label_threshold":    base_cfg.get("label_threshold", None),
                                        "alpha_vol_z":        base_cfg.get("alpha_vol_z", None),
                                        "beta_spread_norm":   base_cfg.get("beta_spread_norm", None),
                                        "gamma_slip_norm":    base_cfg.get("gamma_slip_norm", None),
                                    }
                                    alt_vals = {
                                        "lags_range":         alt_cfg.get("lags_range", alt_cfg.get("lags", base_vals["lags_range"])),
                                        "lag_depth":          alt_cfg.get("lag_depth", base_vals["lag_depth"]),
                                        "target_active_rate": alt_cfg.get("target_active_rate", alt_cfg.get("target_coverage", base_vals["target_active_rate"])),
                                        "label_threshold":    alt_cfg.get("label_threshold", None),
                                        "alpha_vol_z":        alt_cfg.get("alpha_vol_z", None),
                                        "beta_spread_norm":   alt_cfg.get("beta_spread_norm", None),
                                        "gamma_slip_norm":    alt_cfg.get("gamma_slip_norm", None),
                                    }

                                    max_d = 0.0
                                    any_finite = False
                                    for k in HP_RANGES.keys():
                                        d = _norm_dist(base_vals.get(k), alt_vals.get(k), k)
                                        if not _math.isfinite(d):
                                            continue
                                        any_finite = True
                                        if d > max_d:
                                            max_d = d

                                    # If no comparable dims, don't block by geometry.
                                    return True if (not any_finite) else (max_d <= r)
                                except Exception:
                                    return True

                            # ---------- 4) filter pool -> eligible neighbours ----------
                            base_style = base_params.get("strategy_type", None)

                            eligible_alts = []
                            eligible_meta = []
                            eligible_audit = []
                            rejected_audit = []

                            for idx, alt_dict in enumerate(raw_topk):
                                meta = _meta_for_pool(idx, alt_dict)
                                reasons = []

                                if style_lock:
                                    alt_style = alt_dict.get("strategy_type", None)
                                    if (base_style is not None) and (alt_style is not None) and (str(base_style) != str(alt_style)):
                                        reasons.append("STYLE_MISMATCH")

                                if not _within_geom_ball(base_params, alt_dict, geom_radius):
                                    reasons.append("OUTSIDE_GEOM_RADIUS")

                                mv = _meta_value(meta)
                                if not _passes_perf_floor(mv):
                                    reasons.append("BELOW_PERF_FLOOR")

                                if reasons:
                                    if debug_topn:
                                        rejected_audit.append({
                                            "idx": idx,
                                            "trial": meta.get("trial_number"),
                                            "value": meta.get("value"),
                                            "reasons": reasons,
                                        })
                                    continue

                                eligible_alts.append(alt_dict)
                                eligible_meta.append(meta)
                                if debug_topn:
                                    eligible_audit.append({
                                        "idx": idx,
                                        "trial": meta.get("trial_number"),
                                        "value": meta.get("value"),
                                    })

                            # ---------- 5) build candidates list (BASE FIRST, ALWAYS) ----------
                            base_core = dict(base_params)
                            # strip helper keys so dict equality/dedup is meaningful
                            for k_rm in ("__top5_params", "__top5_info", "__top5_path", "__consensus_pool"):
                                base_core.pop(k_rm, None)

                            # base meta: try winner row if possible, else fallback to best_val
                            base_meta = {"trial_number": None, "value": best_val}
                            try:
                                winner_idx = int(base_params.get("__winner_index", 0))
                            except Exception:
                                winner_idx = 0
                            try:
                                if trials_info and 0 <= winner_idx < len(trials_info):
                                    row = trials_info[winner_idx] or {}
                                    tn = row.get("number", row.get("trial_number", None))
                                    vv = row.get("value", row.get("cv_value", row.get("cv", None)))
                                    if tn is not None:
                                        base_meta["trial_number"] = int(tn)
                                    if vv is not None:
                                        base_meta["value"] = float(vv)
                            except Exception:
                                pass

                            candidates = [base_core]
                            candidate_meta = [base_meta]

                            # merge each eligible alt onto base_core (non-None overrides)
                            for alt_dict, meta in zip(eligible_alts, eligible_meta):
                                merged = dict(base_core)
                                try:
                                    merged.update({k: v for k, v in (alt_dict or {}).items() if v is not None})
                                except Exception:
                                    pass
                                candidates.append(merged)
                                candidate_meta.append(meta or {})

                            # if we somehow ended up with only base, no consensus possible
                            if len(candidates) < 2:
                                return None

                            # ---------- 6) committee size trim: base + best neighbours ----------
                            selected_trials_pre_dedup = []

                            if len(candidates) > N_target:
                                base_cand = candidates[0]
                                base_m    = candidate_meta[0] if candidate_meta else {}

                                alt_pairs = list(zip(candidates[1:], candidate_meta[1:]))

                                # rank neighbours by objective value (finite only); if missing values, they go last
                                finite = [(c, m, _meta_value(m)) for (c, m) in alt_pairs]
                                finite_sorted = [x for x in finite if x[2] is not None]
                                none_sorted   = [x for x in finite if x[2] is None]

                                finite_sorted.sort(key=lambda x: x[2], reverse=(not is_minimize))

                                k_keep = max(0, int(N_target) - 1)
                                picked = finite_sorted[:k_keep]

                                # if not enough finite-valued neighbours, pad with unknown-valued ones (stable order)
                                if len(picked) < k_keep:
                                    picked += none_sorted[:(k_keep - len(picked))]

                                candidates     = [base_cand] + [c for (c, _, _) in picked]
                                candidate_meta = [base_m]    + [m for (_, m, _) in picked]

                            # snapshot pre-dedup trial ids
                            try:
                                for m in (candidate_meta or []):
                                    t = _meta_trial(m)
                                    if t is not None:
                                        selected_trials_pre_dedup.append(t)
                            except Exception:
                                selected_trials_pre_dedup = []

                            # ---------- 7) de-dup candidates (keep meta aligned) ----------
                            seen = set()
                            uniq_cands = []
                            uniq_meta  = []
                            for cand, meta in zip(candidates, candidate_meta):
                                key = _json.dumps({k: v for k, v in (cand or {}).items() if not str(k).startswith("__")},
                                                sort_keys=True, default=str)
                                if key in seen:
                                    continue
                                seen.add(key)
                                uniq_cands.append(cand)
                                uniq_meta.append(meta)

                            candidates = uniq_cands
                            candidate_meta = uniq_meta

                            # collapse pseudo-committee if all from same trial_number (and not missing)
                            try:
                                tids = []
                                for m in (candidate_meta or []):
                                    t = _meta_trial(m)
                                    tids.append(t if t is not None else "__missing__")
                                uniq = set(tids)
                                if len(candidates) > 1 and len(uniq) == 1 and list(uniq)[0] not in (None, "__missing__"):
                                    if debug_topn:
                                        print(f"[TopN] All committee members share trial_number={list(uniq)[0]}; collapsing to Top-1.")
                                    candidates = candidates[:1]
                                    candidate_meta = candidate_meta[:1]
                            except Exception:
                                pass

                            if len(candidates) < 2:
                                if debug_topn:
                                    print("[TopN] <=1 distinct config after size/dedup; skipping consensus.")
                                return None
                            
                            # Cache the final committee (post filter/trim/dedup) so it stays fixed across months
                            try:
                                base_params["__consensus_committee_cache"] = deepcopy(candidates)
                                base_params["__consensus_committee_meta"]  = deepcopy(candidate_meta)
                                if debug_topn:
                                    print(f"[TopN] Cached committee for reuse across months (k={len(candidates)}) src={pool_src}.")
                            except Exception:
                                pass

                            # ---------- 8) debug: committee table (ONE place, no duplicates) ----------
                            if debug_topn:
                                try:
                                    base = candidates[0]
                                    base_style_dbg  = base.get("strategy_type")
                                    base_lags_dbg   = base.get("lags_range", base.get("lags"))
                                    base_depth_dbg  = base.get("lag_depth")
                                    base_target_dbg = base.get("target_active_rate", base.get("target_coverage"))

                                    print(
                                        f"[TopN] Committee ({len(candidates)} configs) | model={mt_local} "
                                        f"| strategy_type={base_style_dbg} | lags={base_lags_dbg} | depth={base_depth_dbg} | target_active={base_target_dbg}"
                                    )

                                    def _fmt(v, nd=4):
                                        try:
                                            if v is None:
                                                return "--"
                                            if isinstance(v, bool):
                                                return str(v)
                                            if isinstance(v, int):
                                                return str(v)
                                            fv = float(v)
                                            if not _np.isfinite(fv):
                                                return "--"
                                            return f"{fv:.{nd}g}"
                                        except Exception:
                                            return str(v)

                                    # model-family specific extras (keep short)
                                    extra_keys = []
                                    if mt_local == "logistic":
                                        extra_keys = ["logit_C", "logit_penalty", "logit_class_weight"]
                                    elif mt_local == "svm":
                                        extra_keys = ["svm_c", "svm_gamma", "svm_kernel", "svm_degree"]
                                    elif mt_local == "xgboost":
                                        extra_keys = ["xgb_eta", "xgb_max_depth", "xgb_subsample", "xgb_colsample_bytree", "xgb_min_child_weight"]
                                    elif mt_local in {"random_forest", "decision_tree"}:
                                        extra_keys = ["max_depth", "min_samples_leaf", "min_samples_split"]
                                    elif mt_local == "cnn":
                                        extra_keys = ["cnn_num_filters", "cnn_num_layers", "cnn_kernel_size", "cnn_dropout_rate"]
                                    elif mt_local == "lstm":
                                        extra_keys = ["lstm_units", "lstm_num_layers", "lstm_dropout_rate"]
                                    elif mt_local == "transformer":
                                        extra_keys = ["transformer_d_model", "transformer_n_heads", "transformer_num_layers", "transformer_dropout_rate"]
                                    elif mt_local.startswith("ensemble"):
                                        extra_keys = ["ensemble_weight_cnn", "ensemble_weight_lstm", "ensemble_weight_xgb", "ensemble_weight_meta"]

                                    headers = ["id", "trial", "value", "lags", "depth", "target", "label_thr", "conf_thr", "alpha", "beta", "gamma", "extra"]
                                    rows = []
                                    for i_c, (cand, meta) in enumerate(zip(candidates, candidate_meta), start=1):
                                        member_id = "base" if i_c == 1 else str(i_c)
                                        extra = []
                                        for k in extra_keys:
                                            if k in cand:
                                                extra.append(f"{k}={_fmt(cand.get(k), nd=4)}")
                                        rows.append({
                                            "id":        member_id,
                                            "trial":     _fmt(_meta_trial(meta), nd=0),
                                            "value":     _fmt(_meta_value(meta), nd=4),
                                            "lags":      _fmt(cand.get("lags_range", cand.get("lags")), nd=0),
                                            "depth":     _fmt(cand.get("lag_depth"), nd=0),
                                            "target":    _fmt(cand.get("target_active_rate", cand.get("target_coverage")), nd=4),
                                            "label_thr": _fmt(cand.get("label_threshold"), nd=4),
                                            "conf_thr":  _fmt(cand.get("confidence_threshold"), nd=4),
                                            "alpha":     _fmt(cand.get("alpha_vol_z"), nd=3),
                                            "beta":      _fmt(cand.get("beta_spread_norm"), nd=3),
                                            "gamma":     _fmt(cand.get("gamma_slip_norm"), nd=3),
                                            "extra":     ", ".join(extra) if extra else "--",
                                        })

                                    colw = {h: len(h) for h in headers}
                                    for h in headers:
                                        for r in rows:
                                            colw[h] = max(colw[h], len(str(r.get(h, ""))))

                                    def _line(d):
                                        return " | ".join(str(d.get(h, "")).ljust(colw[h]) for h in headers)

                                    print("      " + _line({h: h for h in headers}))
                                    print("      " + " | ".join("-" * colw[h] for h in headers))
                                    for r in rows:
                                        print("      " + _line(r))

                                    print(
                                        f"[TopN][Audit] pool_raw={len(raw_topk)} eligible={len(eligible_audit)} rejected={len(rejected_audit)} "
                                        f"selected_pre_dedup_trials={selected_trials_pre_dedup}"
                                    )
                                    if rejected_audit:
                                        rej = [(x.get("trial"), x.get("value"), ",".join(x.get("reasons") or [])) for x in rejected_audit[:40]]
                                        print(f"[TopN][Audit] rejected(trial,value,why)={rej}")

                                except Exception as _e:
                                    print(f"[TopN] (debug: failed committee print -> {_e})")
                            else:
                                # optional single-line info (non-debug)
                                if bool(cfg_local.get("topN_deploy", False)):
                                    try:
                                        print(f"[TopN] committee_size={len(candidates)}/{int(N_target)} geom={geom_radius:.3g} floor={min_perf_frac:.3g}")
                                    except Exception:
                                        pass

                            # ---------- 9) evaluate each candidate & collect per-bar pred/returns ----------
                            bar_dfs = []
                            for idx_c, cand in enumerate(candidates, start=1):
                                try:
                                    # evaluate_strategy is assumed to set self.results to a DF for this candidate
                                    _ = self.evaluate_strategy(cand, train_start, train_end, test_start, test_end)
                                    df_c = getattr(self, "results", None)

                                    if df_c is None or getattr(df_c, "empty", True):
                                        continue
                                    if ("pred" not in df_c.columns) or ("returns" not in df_c.columns):
                                        continue

                                    # ANCHOR: df_loc = df_c[["pred", "returns"]].copy()
                                    # IMPORTANT: df_c["pred"] is executed-time (already shifted by compute_full_evaluation_metrics).
                                    # For committee voting we need decision-time signals. Prefer raw_pred if available.
                                    if "raw_pred" in df_c.columns:
                                        _sig = df_c["raw_pred"]
                                    else:
                                        # Best-effort reconstruction: executed pred -> decision-time (undo 1-bar delay)
                                        _sig = df_c["pred"].shift(-1)

                                    df_loc = _pd.DataFrame(
                                        {
                                            "raw_pred": _pd.to_numeric(_sig, errors="coerce").fillna(0.0).astype(float),
                                            "returns":  _pd.to_numeric(df_c["returns"], errors="coerce").fillna(0.0).astype(float),
                                        },
                                        index=df_c.index,
                                    )
                                    df_loc.index = _pd.to_datetime(df_loc.index, utc=True, errors="coerce")
                                    df_loc = df_loc[~df_loc.index.isna()]
                                    if df_loc.empty:
                                        continue
                                    bar_dfs.append(df_loc)

                                except Exception as _e:
                                    if debug_topn:
                                        print(f"[TopN] Candidate {idx_c} failed during consensus eval: {_e}")

                            if len(bar_dfs) < 2:
                                if debug_topn:
                                    print("[TopN] Need >=2 valid candidates with pred+returns; skipping.")
                                return None

                            # ---------- 10) align on common index ----------
                            common_idx = bar_dfs[0].index
                            for df_c in bar_dfs[1:]:
                                common_idx = common_idx.intersection(df_c.index)
                            common_idx = common_idx.sort_values()

                            if len(common_idx) == 0:
                                if debug_topn:
                                    print("[TopN] No overlapping bars across candidates; skipping.")
                                return None

                            aligned = [df_c.reindex(common_idx) for df_c in bar_dfs]

                            # drop bars where returns missing in any member
                            mask = _np.ones(len(common_idx), dtype=bool)
                            for df_c in aligned:
                                mask &= df_c["returns"].notna().to_numpy()
                            if not mask.any():
                                if debug_topn:
                                    print("[TopN] All overlapping bars invalid after NA filter; skipping.")
                                return None

                            common_idx = common_idx[mask]
                            aligned = [df_c.loc[common_idx] for df_c in aligned]

                            # ---------- 11) optional correlation diversity filter (on RETURNS series) ----------
                            if len(aligned) > 2 and (max_corr < 1.0):
                                try:
                                    base_sig = _pd.to_numeric(aligned[0]["raw_pred"], errors="coerce").fillna(0.0).astype(float)
                                    keep = [True]
                                    for df_c in aligned[1:]:
                                        sig_c = _pd.to_numeric(df_c["raw_pred"], errors="coerce").fillna(0.0).astype(float)
                                        corr = base_sig.corr(sig_c)
                                        # If corr is NaN/None (e.g., constant signals), don't block diversity.
                                        if corr is None or (not _np.isfinite(float(corr))) or abs(float(corr)) <= max_corr:
                                            keep.append(True)
                                        else:
                                            keep.append(False)

                                    if sum(keep) < 2:
                                        if debug_topn:
                                            print("[TopN] Corr-diversity filter left <2 members; skipping.")
                                        return None

                                    aligned = [df for df, k in zip(aligned, keep) if k]
                                except Exception:
                                    pass

                            # ---------- 12) majority vote + metric eval ----------
                            base_df = aligned[0].copy()
                            preds = _np.stack([df_c["raw_pred"].astype(float).to_numpy() for df_c in aligned], axis=0)

                            # Majority vote on {-1,0,+1} (ties go to 0 because sign(0)=0)
                            consensus_raw = _np.sign(preds.sum(axis=0))
                            # Feed decision-time signals; evaluator applies the 1-bar delay exactly once.
                            base_df["raw_pred"] = consensus_raw
                            # Keep pred numeric for preconditions; evaluator overwrites pred from raw_pred anyway.
                            base_df["pred"] = 0.0
                            
                            # Ensure the evaluator sees the same config-driven execution overlays
                            # (TWAP / kill-switch / gating diagnostics) as the single-model path.
                            # Preserve train-anchored high_vol_thr (prevents leakage-guard HIGH slippage)
                            _fc = dict(cfg_local)
                            try:
                                if _fc.get("high_vol_thr") is None:
                                    _thr_prev = (base_df.attrs.get("features_config", {}) or {}).get("high_vol_thr", None)
                                    if _thr_prev is None:
                                        _thr_prev = getattr(self, "_last_high_vol_thr_train", None)
                                    if _thr_prev is not None:
                                        _fc["high_vol_thr"] = float(_thr_prev)
                            except Exception:
                                pass
                            try:
                                base_df.attrs["features_config"] = dict(_fc)
                                base_df.attrs["debug_costs"] = bool(self._is_debug())
                                base_df.attrs["eval_context"] = "real_sim:topN_consensus"
                            except Exception:
                                pass

                            # Ensure cost columns consistent with your single-model evaluation path (best-effort)
                            try:
                                if bool(getattr(self, "trading_costs", False)):
                                    base_df = self._ensure_cost_columns(base_df, _fc)
                            except Exception:
                                pass

                            # Carry state is only meaningful in real_sim; guard in case this helper is ever re-used elsewhere.
                            try:
                                _pp   = prev_position
                                _peqs = prev_eq_strategy
                                _peqb = prev_eq_bh
                            except Exception:
                                _pp = _peqs = _peqb = None

                            metrics_cons = compute_full_evaluation_metrics(
                                df=base_df,
                                trading_costs=self.trading_costs,
                                slippage_factor=self.slippage_factor,
                                prev_position=_pp,
                                prev_eq_strategy=_peqs,
                                prev_eq_bh=_peqb,
                                eval_context="real_sim:topN_consensus",
                            )

                            # expose for downstream logging/plots
                            self.results = base_df

                            try:
                                metrics_cons = _safe_metrics_return(metrics_cons, context="topN_consensus")
                            except Exception:
                                pass

                            return metrics_cons


                    # ------------------------------------------------------------
                    # Month CONFIG (single source of truth)
                    # ------------------------------------------------------------
                    try:
                        import hashlib as _hashlib, json as _json
                        from copy import deepcopy

                        # 1) deterministic month baseline
                        _month_base = deepcopy(DEFAULT_FEATURES)

                        # restore frozen base (if you set it once at sim start)
                        _month_base.update(deepcopy(getattr(self, "_rt_sim_base_features_config", {}) or {}))

                        # IMPORTANT: allow run-level keys that affect gating/calibration/execution
                        _month_allow = set(DEFAULT_FEATURES.keys()) | {
                            "model_type",
                            "gating_mode",
                            "confidence_threshold",
                            "target_active_rate",
                            "calibrate_method",
                            "label_threshold",
                            "lags", "lags_range", "lag_depth",
                            "roll_windows_key", "roll_windows_key_v2",
                            "runtime_active_band_margin", "runtime_conf_nudge", "runtime_coverage_window",
                            "alpha_vol_z", "beta_spread_norm", "gamma_slip_norm",  "real_sim_target_active_mult",
                            "allow_real_sim_target_active_mult",
                            # cost regime (train-anchored)
                            "vol_window_bars", "high_vol_q", "high_vol_thr",
                            "eval_slip_bps_lo", "eval_slip_bps_hi",
                        }
                        
                        if isinstance(config, dict):
                            _month_base.update({k: deepcopy(v) for k, v in config.items() if k in _month_allow})
                            
                        # Ensure train-anchored high-vol threshold is present for the cost model
                        try:
                            _thr_m = getattr(self, "_last_high_vol_thr_train", None)
                            if _thr_m is not None:
                                _month_base["high_vol_thr"] = float(_thr_m)
                            else:
                                _month_base.pop("high_vol_thr", None)
                        except Exception:
                            pass

                        # ensure model_type exists in baseline (do NOT rely on DEFAULT_FEATURES)
                        _month_base["model_type"] = str(_month_base.get("model_type") or model_type)

                        # Store baseline for auditing (do NOT overwrite self.features_config here)
                        setattr(self, "_month_base_features_config", deepcopy(_month_base))

                        # 2) effective month config = base + tuned params (best_combo)
                        _params = best_combo if isinstance(best_combo, dict) else {}
                        # Preserve internal helper keys (e.g., __top5_params / __consensus_pool) for
                        # downstream Top-N consensus logic, while keeping the *effective* month config
                        # free of internal metadata.
                        _params_internal = {k: deepcopy(v) for k, v in _params.items() if str(k).startswith("__")}
                        _params_clean = {k: v for k, v in _params.items() if not str(k).startswith("__")}

                        _effective = deepcopy(_month_base)
                        _effective.update(deepcopy(_params_clean))
                        
                        # Keep the calibration mapping consistent with the CV-selected params.
                        # (Confidence thresholds are only comparable under the same calibration method.)
                        if _params_clean.get("calibrate_method") is not None:
                            _cal_raw = str(_params_clean.get("calibrate_method") or "").strip().lower()
                            _effective["calibrate_method"] = _cal_raw if _cal_raw in ("sigmoid", "isotonic") else "sigmoid"
                            
                        # --- Real-sim only: bump target_active_rate to offset downstream gates (does NOT affect CV) ---
                        try:
                            _mult_raw = _effective.get('real_sim_target_active_mult', None)
                            _allow_mult = bool(_effective.get('allow_real_sim_target_active_mult', False))
                            if _allow_mult and _mult_raw is not None and _effective.get('target_active_rate') is not None:
                                _mult = float(_mult_raw)
                                if _mult != 1.0:
                                    _tar0 = float(_effective.get('target_active_rate'))

                                    # 1) Clamp multiplier (prevents typos like 10.0)
                                    _mult = float(max(0.80, min(1.30, _mult)))

                                    # 2) Cap effective TAR (prevents "trade-all-bars" behavior)
                                    _tar_cap = float(_effective.get('real_sim_target_active_cap', 0.25))

                                    _effective['target_active_rate'] = float(
                                        max(0.0, min(_tar_cap, _tar0 * _mult))
)
                                    log_print(f"[RealSim][Coverage] m{period_idx} target_active_rate base={_tar0:.3f} mult={_mult:.3f} effective={_effective['target_active_rate']:.3f}", level="COMPACT")
                        except Exception:
                            pass

                        if "lags" not in _effective and "lags_range" in _effective:
                            _effective["lags"] = _effective.get("lags_range")
                        _fp_view = {
                            "model_type": str(_effective.get("model_type", "")),
                            "lags": _effective.get("lags"),
                            "lag_depth": _effective.get("lag_depth"),

                            # NEW: what the engine actually uses
                            "roll_windows": (
                                _effective.get("roll_windows")                  # e.g. [10, 30, 60]
                                or _effective.get("roll_windows_key_v2")         # e.g. "10,30,60"
                                or _effective.get("roll_windows_key")            # e.g. "20,60"
                            ),

                            # keep the raw keys too (for traceability)
                            "roll_windows_key": _effective.get("roll_windows_key"),
                            "roll_windows_key_v2": _effective.get("roll_windows_key_v2"),

                            "target_active_rate": _effective.get("target_active_rate"),
                            "confidence_threshold": _effective.get("confidence_threshold"),
                            "calibrate_method": _effective.get("calibrate_method"),
                            "runtime_active_band_margin": _effective.get("runtime_active_band_margin"),
                            "runtime_coverage_window": _effective.get("runtime_coverage_window"),
                            "runtime_conf_nudge": _effective.get("runtime_conf_nudge"),
                            "alpha_vol_z": _effective.get("alpha_vol_z"),
                            "beta_spread_norm": _effective.get("beta_spread_norm"),
                            "gamma_slip_norm": _effective.get("gamma_slip_norm"),
                        }


                        _fp = _hashlib.sha1(_json.dumps(_fp_view, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
                        log_print(
                            f"[CONFIG-FINGERPRINT] m{period_idx} sha1={_fp} | "
                            f"tar={_fp_view['target_active_rate']} conf={_fp_view['confidence_threshold']} cal={_fp_view['calibrate_method']} | "
                            f"lags={_fp_view['lags']} ld={_fp_view['lag_depth']} | "
                            f"rwk={_fp_view['roll_windows_key']} rwk2={_fp_view['roll_windows_key_v2']} | "
                            f"roll_windows={_fp_view['roll_windows']} | "
                            f"rwin={_fp_view['runtime_coverage_window']} nudge={_fp_view['runtime_conf_nudge']} band={_fp_view['runtime_active_band_margin']}",
                            level="COMPACT",
                        )

                        # Guardrail: drift detection for target_active_rate
                        if "target_active_rate" in _params_clean:
                            _tuned = float(_params_clean.get("target_active_rate"))
                            _eff   = float(_effective.get("target_active_rate"))
                            if abs(_tuned - _eff) > 1e-9:
                                _base = float(_month_base.get("target_active_rate", _tuned))
                                if bool(getattr(self, "_in_real_sim", False)):
                                    log_print(
                                        f"[CONFIG] m{period_idx} target_active_rate base={_base} tuned={_tuned} effective={_eff} (real-sim override)",
                                        level="COMPACT",
                                    )
                                else:
                                    log_print(
                                        f"[WARN] [CONFIG-DRIFT] m{period_idx} target_active_rate tuned={_tuned} effective={_eff}",
                                        level="COMPACT",
                                    )

                        # OPTIONAL BUT STRONGLY RECOMMENDED:
                        # make downstream use the *effective* dict, not the pre-cleaned best_combo
                        best_combo = deepcopy(_effective)
                        # Re-attach internal metadata (Top-N pools, audit helpers, etc.)
                        if isinstance(_params_internal, dict) and _params_internal:
                            try:
                                best_combo.update(deepcopy(_params_internal))
                            except Exception:
                                best_combo.update(_params_internal)

                    except Exception as _e_cfg:
                        log_print(f"[WARN] [CONFIG-FINGERPRINT] m{period_idx} failed: {type(_e_cfg).__name__}: {_e_cfg}", level="COMPACT")
                        
                    # ------------------------------------------------------------
                    # Patch 1: Re-fingerprint immediately before each evaluation call
                    # and after a Top-N candidate is accepted (audit determinism).
                    # This rebuilds an effective config from _month_base_features_config
                    # so the fingerprint matches what evaluate_strategy() will see.
                    # ------------------------------------------------------------
                    def _rt_print_config_fingerprint(_params_in, _tag="pre-eval"):
                        try:
                            import hashlib as __hashlib, json as __json
                            from copy import deepcopy as __deepcopy

                            _base = __deepcopy(getattr(self, "_month_base_features_config", {}) or {})
                            _p = _params_in if isinstance(_params_in, dict) else {}
                            _p_clean = {k: v for k, v in _p.items() if not str(k).startswith("__")}

                            _eff = __deepcopy(_base)
                            _eff.update(__deepcopy(_p_clean))
                            if "lags" not in _eff and "lags_range" in _eff:
                                _eff["lags"] = _eff.get("lags_range")

                            _roll = _eff.get("roll_windows") or _eff.get("roll_windows_key") or _eff.get("roll_windows_key_v2")
                            _conf = _eff.get("confidence_threshold")
                            if _conf is None:
                                _conf = 0.0
                                
                            _view = {
                                "model_type": str(_eff.get("model_type", "")),
                                "lags": _eff.get("lags"),
                                "lag_depth": _eff.get("lag_depth"),
                                "roll_windows": _roll,
                                "target_active_rate": _eff.get("target_active_rate"),
                                "confidence_threshold": _conf,
                                "calibrate_method": _eff.get("calibrate_method"),
                                "runtime_active_band_margin": _eff.get("runtime_active_band_margin"),
                                "runtime_coverage_window": _eff.get("runtime_coverage_window"),                                
                                "runtime_conf_nudge": _eff.get("runtime_conf_nudge"),
                                "alpha_vol_z": _eff.get("alpha_vol_z"),
                                "beta_spread_norm": _eff.get("beta_spread_norm"),
                                "gamma_slip_norm": _eff.get("gamma_slip_norm"),
                            }

                            _sha = __hashlib.sha1(
                                __json.dumps(_view, sort_keys=True, default=str).encode("utf-8")
                            ).hexdigest()[:10]

                            log_print(
                                f"[CONFIG-FINGERPRINT] m{period_idx} {_tag} sha1={_sha} | "
                                f"tar={_view['target_active_rate']} conf={_view['confidence_threshold']} cal={_view['calibrate_method']} | "
                                f"lags={_view['lags']} ld={_view['lag_depth']} | "
                                f"| roll_windows={_view['roll_windows']} | "
                                f"rwin={_view['runtime_coverage_window']} nudge={_view['runtime_conf_nudge']} band={_view['runtime_active_band_margin']}",
                                level="COMPACT",
                            )
                            return _eff
                        except Exception as __e:
                            log_print(
                                f"[WARN] [CONFIG-FINGERPRINT] m{period_idx} {_tag} failed: {type(__e).__name__}: {__e}",
                               level="COMPACT",
                            )
                            return _params_in
                    
                    # ---------- Primary evaluation step (consensus -> adaptive Top-3 -> single best) ----------
                    # For all non-DQN / non-TF-XGB-DQN models, force the Top-N consensus path
                    # to run, even if a previous refit already produced metrics.

                    params_safe = best_combo if isinstance(best_combo, dict) else {}
                    
                    # model type for gating / debug (must exist even if signal_coverage attrs read fails)
                    mt_eval = str(params_safe.get("model_type", getattr(self, "model_type", ""))).lower()

                    # Secondary activity metric from evaluator (non-neutral label coverage)
                    try:
                        _signal_coverage_month = float(eval_df_cont.attrs.get("signal_coverage", float("nan")))
                    except Exception:
                        _signal_coverage_month = float("nan")


                    # Debug-only: explain whether Top-N consensus will run (logging only; no behaviour change)
                    try:
                        _cfg = getattr(self, "features_config", {}) or {}
                        if bool(_cfg.get("print_topN_debug", False)):
                            _has_pool = bool(
                                params_safe.get("__top5_params") or
                                params_safe.get("__consensus_pool") or
                                params_safe.get("__top3_params")
                            )                            
                            print(f"[TopN][Precheck] metrics_is_none={metrics is None} has_pool={_has_pool} deploy={bool(_cfg.get('deploy_topN_consensus', True))} model={mt_eval}")
                    except Exception:
                        pass

                    # Run Top-N consensus only when explicitly enabled.
                    _deploy_topn = True
                    try:
                        _cfg = getattr(self, "features_config", {}) or {}
                        _deploy_topn = bool(_cfg.get("deploy_topN_consensus", True))
                    except Exception:
                        _deploy_topn = True

                    if _deploy_topn and mt_eval not in {"dqn"} and params_safe:
                        _m_cons = _evaluate_with_topn_consensus(params_safe)
                        if _m_cons is not None:
                            metrics = _m_cons
                    # 2) If consensus is disabled or failed, fall back to previous behavior
                    if metrics is None:
                        try:
                            # ADAPTIVE Top-3 (skip DQN and Transformer-XGB-DQN ensemble)
                            has_top3 = bool(
                                best_combo.get("__top3_params") or best_combo.get("__top5_params")
                            )
                            mt = best_combo.get("model_type", getattr(self, "model_type", ""))

                            # New: flag from CLASS_DEFAULTS["features"]
                            use_adaptive_top3 = bool(
                                (self.features_config or {}).get(
                                    "use_adaptive_top3_for_main_results", False
                                )
                            )

                            if has_top3 and mt not in {"dqn"} and use_adaptive_top3:
                                thr = float(self.features_config.get("switch_hit_rate_thr", 0.45))
                                wnd = int(self.features_config.get("switch_window_days", 5))
                                metrics = self.evaluate_strategy_adaptive_top3(
                                    best_combo,
                                    train_start,
                                    train_end,
                                    test_start,
                                    test_end,
                                    hit_thr=thr,
                                    window_days=wnd,
                                )
                            else:
                                metrics = self.evaluate_strategy(
                                    best_combo,
                                    train_start,
                                    train_end,
                                    test_start,
                                    test_end,
                                )
                        except Exception as e:
                            print(f"[WARN] evaluate_strategy failed (primary): {e}")
                            metrics = None

                    # ---------- Top-N fallbacks if needed ----------
                    def _safe_build_topn_candidates(base_params):
                        
                        if isinstance(base_params, dict) and isinstance(base_params.get("best_params"), dict):
                            base_params = base_params["best_params"]
                        
                        base = dict(base_params)
                        raw_topk = base_params.get("__top5_params") or []
                        cands = [base] + [{**base, **deepcopy(alt)} for alt in raw_topk]

                        REQUIRED = [
                            "model_type",
                            "use_extended_features",
                            "lags",
                            "label_threshold",
                            "confidence_threshold",
                        ]
                        for c in cands:
                            for k in REQUIRED:
                                if k not in c and k in base:
                                    c[k] = base[k]

                        return cands


                    # If Top-N fallbacks are disabled, keep the primary result even if invalid.
                    if (not _is_valid_metrics_tuple(metrics)) and (
                        not bool(self.features_config.get("allow_param_fallback", False))
                    ):
                        print("[LOCK] Realism ON: skipping Top-N fallbacks; keeping primary result (may be NaN/0-trade).")
                    elif not _is_valid_metrics_tuple(metrics) and isinstance(best_combo, dict):
                        print("[WARN] Best combo invalid -- trying Top-N fallbacks...")

                        for idx, params_try in enumerate(_safe_build_topn_candidates(best_combo), start=1):
                            try:
                                params_eval = _rt_print_config_fingerprint(params_try, _tag=f"top{idx}-pre")
                                alt_metrics = self.evaluate_strategy(
                                    params_eval, train_start, train_end, test_start, test_end
                                )
                                if _is_valid_metrics_tuple(alt_metrics):
                                    _internal = {k: v for k, v in (params_try or {}).items() if str(k).startswith("__")}
                                    best_combo = dict(params_eval) if isinstance(params_eval, dict) else dict(params_try)
                                    best_combo.update(_internal)
                                    _rt_print_config_fingerprint(best_combo, _tag=f"top{idx}-ACCEPT")
                                    best_combo = params_try
                                    metrics = alt_metrics
                                    print(f"    [OK] Using Top-{idx} candidate (non-degenerate result).")
                                    break
                                else:
                                    print(f"    [WARN] Top-{idx} candidate degenerate (e.g., 0 trades). Trying next...")
                            except Exception as e:
                                print(f"    [X] Top-{idx} candidate crashed: {e}")
                                continue

                    # If Top-N fallbacks are disabled, keep the primary result even if invalid.
                    if (not _is_valid_metrics_tuple(metrics)) and (
                        not bool(self.features_config.get("allow_param_fallback", False))
                    ):
                        print(
                            "[LOCK] Realism ON: skipping Top-N fallbacks; keeping primary result "
                            "(may be NaN/0-trade)."
                        )

                    elif not _is_valid_metrics_tuple(metrics):
                        print("[WARN] Best combo invalid -- trying Top-N fallbacks...")

                        for idx, params_try in enumerate(_safe_build_topn_candidates(best_combo), start=1):
                            try:
                                params_eval = _rt_print_config_fingerprint(params_try, _tag=f"top{idx}-pre")
                                alt_metrics = self.evaluate_strategy(
                                    params_eval, train_start, train_end, test_start, test_end
                                )
                                if _is_valid_metrics_tuple(alt_metrics):
                                    _internal = {k: v for k, v in (params_try or {}).items() if str(k).startswith("__")}
                                    best_combo = dict(params_eval) if isinstance(params_eval, dict) else dict(params_try)
                                    best_combo.update(_internal)
                                    _rt_print_config_fingerprint(best_combo, _tag=f"top{idx}-ACCEPT")
                                    print(f"    [OK] Using Top-{idx} candidate (non-degenerate result).")
                                    break
                                else:
                                    print(
                                        f"    [WARN] Top-{idx} candidate degenerate (e.g., 0 trades). "
                                        "Trying next..."
                                    )
                            except Exception as e:
                                print(f"    [X] Top-{idx} candidate crashed: {e}")
                                continue

                        # If still invalid after trying Top-N, just fall through.
                        if not _is_valid_metrics_tuple(metrics):
                            print(
                                "[WARN] Top-N fallbacks exhausted; keeping primary/degenerate "
                                "result for this month."
                            )
                            
                # ------------------------------------------------------------
                # Gate diagnostic (debug-only):
                # Print median threshold actually used vs max_conf distribution
                # so CV vs real-sim mismatches cannot hide silently.
                # ------------------------------------------------------------
                try:
                    from utilsNoWFO import print_conf_stats
                    _thr_med = getattr(self, "_last_conf_thr_used", None)
                    _conf    = getattr(self, "_last_conf_stats_max_conf", None)
                    if self._is_debug():
                        print_conf_stats(_conf, label=f"real_m{period_idx}", thr=_thr_med)
                except Exception:
                    pass
                
                
                # --- Patch B: real-sim metric sanitization (avoid false "no valid trades") ---
                # Some months can produce a few trades but a secondary metric (e.g., Sharpe/PSR)
                # becomes non-finite due to tiny-sample variance. In real-sim we still want to
                # log the month (equity curve/trade stats) rather than force a flat month.
                try:
                    _in_real = bool(getattr(self, "_in_real_sim", False))
                    _in_cv   = bool(getattr(self, "_in_optuna_cv", False) or getattr(self, "_in_cv", False))
                    if _in_real and (not _in_cv) and isinstance(metrics, tuple) and len(metrics) >= 16:
                        _mm = list(metrics)
                        _repl = []
                        for _i in (4, 12, 13, 14):  # sharpe, psr, dsr, calmar
                            try:
                                if not np.isfinite(float(_mm[_i])):
                                    _mm[_i] = 0.0
                                    _repl.append(_i)
                            except Exception:
                                _mm[_i] = 0.0
                                _repl.append(_i)
                        if _repl:
                            print(f"[RealSim][MetricsSanitize] m{period_idx} replaced non-finite metric(s) at idx={_repl}")
                            metrics = tuple(_mm)
                except Exception:
                    pass

                # Skip if still invalid
                # If still invalid (e.g., no trades), log a flat month instead of skipping
                if not _is_valid_metrics_tuple(metrics):
                    prev_eq_strategy, prev_eq_bh, prev_position = _log_flat_month_fallback(
                        period_idx=period_idx,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        model_type=model_type,
                        full_data=full_data,
                        prev_position=prev_position,
                        prev_eq_strategy=prev_eq_strategy,
                        prev_eq_bh=prev_eq_bh,
                    )
                    _progress_cb = getattr(self, "_progress_callback", None)
                    if _progress_cb:
                        phase = "month" if period_unit == "months" else "period"
                        _progress_cb(phase, model_type, {
                            "period": period_idx + 1,
                            "total_periods": n_periods,
                            "sharpe": 0.0,
                            "trades": 0,
                            "equity_strategy": prev_eq_strategy,
                            "equity_bh": prev_eq_bh,
                            "drawdown": 0.0,
                            "win_rate": 0.0,
                            "precision_macro": 0.0,
                            "f1_macro": 0.0,
                            "return_pct": 0.0,
                            "directional_accuracy": 0.0,
                            "active_rate": 0.0,
                            "flat": True,
                        })
                    continue



                # -------- Build continuous-month DF + carry state --------
                (perf, outperf, ret, sharpe, drawdown, trades,
                geo_mean_ann, directional_accuracy, precision_macro, f1_macro,
                active_rate, profit_per_hit, return_per_trade, win_rate,
                strategy_volatility, kurtosis_val) = metrics

                if hasattr(self, "results") and isinstance(self.results, pd.DataFrame):
                    # Base df with whatever the model produced
                    test_df = self.results.loc[
                        (self.results.index >= test_start) & (self.results.index <= test_end)
                    ].copy()

                    # --- Build a canonical evaluation index for this month ---
                    # 1) Raw month slice from full_data
                    cfg_f = getattr(self, "features_config", {}) or {}
                    sess_mode = str(cfg_f.get("session_filter_mode", "both")).lower()
                    use_strict = bool(cfg_f.get("enforce_day1_start", True))
                    if getattr(self, "_in_real_sim", False):
                        use_strict = True

                    test_bars = full_data.loc[test_start:test_end].copy()

                    # 2) Apply the same NY session filter used during testing
                    if sess_mode in ("test_only", "both"):
                        if not hasattr(self, "_ny_mask") or self._ny_mask is None:
                            try:
                                full_idx = pd.to_datetime(full_data.index, utc=True, errors="coerce")
                                _ny_times = full_idx.tz_convert("America/New_York")
                                # 02:00-13:00 NY
                                self._ny_mask = pd.Series(
                                    (_ny_times.hour >= 2) & (_ny_times.hour <= 13),
                                    index=full_idx,
                                )
                            except Exception as _e:
                                print(f"[WARN] Lazy NY mask build failed in real-trading eval: {_e}")
                                self._ny_mask = pd.Series(True, index=full_data.index)
                        test_bars = test_bars.loc[
                            self._ny_mask.reindex(test_bars.index, fill_value=False)
                        ]

                    # 3) Enforce day-1 calendar anchor (same rule for ALL models)
                    eval_index = test_bars.index
                    if use_strict and not test_bars.empty:
                        month_start_dt = _ensure_dt(test_start)
                        try:
                            first_eval_ts = enforce_day1_eval_anchor(test_bars.index, month_start_dt)
                            eval_index = test_bars.loc[first_eval_ts:].index
                        except Exception as _e:
                            print(f"[WARN] enforce_day1_eval_anchor failed in real-trading eval: {_e}")
                            # fallback: keep full test_bars index

                    # 4) Reindex model outputs onto canonical index
                    if not test_df.empty and len(eval_index) > 0:
                        # Align to the canonical monthly timeline
                        test_df = test_df.reindex(eval_index)

                        # Shared buy-and-hold baseline: always from the same returns stream
                        test_df["returns"] = (
                            self.data["returns"].reindex(test_df.index).astype(float)
                        )

                        # If the model never produced preds for some bars, treat them as flat
                        if "pred" not in test_df.columns:
                            test_df["pred"] = 0.0
                        test_df["pred"] = test_df["pred"].fillna(0.0)
                        
                        # If raw_pred exists (decision-time preds), reindexing can introduce NaNs.
                        # Treat those as flat so causality/shift logic remains consistent.
                        if "raw_pred" in test_df.columns:
                            test_df["raw_pred"] = pd.to_numeric(test_df["raw_pred"], errors="coerce").fillna(0.0)

                        # Ensure evaluator has consistent cost columns (spread, price/mid_close, slippage_bps)
                        # so cont_metrics uses the same cost model as the single-model evaluation path.
                        try:
                            cfg_cost = {}
                            try:
                                cfg_cost = dict((test_df.attrs.get("features_config", {}) or {}))
                            except Exception:
                                cfg_cost = {}

                            # cfg_local (this model/month config) takes precedence when available
                            try:
                                if isinstance(locals().get("cfg_local", None), dict):
                                    cfg_cost = dict(cfg_cost)
                                    cfg_cost.update(dict(locals().get("cfg_local") or {}))
                            except Exception:
                                pass

                            try:
                                test_df.attrs["features_config"] = dict(cfg_cost)
                                test_df.attrs["debug_costs"] = bool(self._is_debug())
                                test_df.attrs["eval_context"] = "real_sim:month_eval:cont_metrics:prep"
                            except Exception:
                                pass

                            if bool(getattr(self, "trading_costs", False)):
                                # Avoid copying the full feature frame when attaching cost columns.
                                _td_cost = test_df[["returns"]] if ("returns" in test_df.columns) else test_df.loc[:, []]
                                _cost_df = self._ensure_cost_columns(_td_cost, cfg_cost)
                                for _c in ("spread", "slippage_bps"):
                                    if _c in _cost_df.columns:
                                        test_df[_c] = _cost_df[_c].reindex(test_df.index)
                                if self._is_debug():
                                    if ("spread" not in test_df.columns) or ("slippage_bps" not in test_df.columns):
                                        print("[Costs][Warn] cont_metrics frame missing spread/slippage_bps after _ensure_cost_columns.")
                        except Exception as _e:
                            if self._is_debug():
                                print(f"[WARN] Cost-column prep failed in real_sim cont_metrics: {_e}")




                        # test_df coming from test_strategy has already been evaluated once
                        # (it already contains continuous curves), meaning its 'pred' is in executed-time.
                        # Reconstruct decision-time 'pred' so cont_metrics applies exactly ONE shift.
                        df_for_cont = test_df
                        if ("cstrategy_cont" in df_for_cont.columns) and ("pred" in df_for_cont.columns):
                            df_for_cont = df_for_cont.copy()
                            df_for_cont["pred"] = df_for_cont["pred"].shift(-1).fillna(0.0)

                        cont_metrics = compute_full_evaluation_metrics(
                            df_for_cont,
                            trading_costs=self.trading_costs,
                            slippage_factor=self.slippage_factor,
                            prev_position=prev_position,
                            prev_eq_strategy=prev_eq_strategy,
                            prev_eq_bh=prev_eq_bh,
                            eval_context="real_sim:month_eval:cont_metrics",
                        )
                        
                        from utilsNoWFO import validate_metrics_shape
                        validate_metrics_shape(cont_metrics, context="real_sim:cont_metrics")

                        (perf, outperf, ret, sharpe, drawdown, trades,
                        geo_mean_ann, directional_accuracy, precision_macro, f1_macro,
                        active_rate, profit_per_hit, return_per_trade, win_rate,
                        strategy_volatility, kurtosis_val) = cont_metrics

                        # IMPORTANT: use the *post-cont_metrics* frame for carry + plots.
                        # test_df came from test_strategy (monthly-rebased). df_for_cont has
                        # been re-evaluated with prev_eq_* so its *_cont curves are truly continuous.
                        eval_df_cont = df_for_cont
                        

                        # V2: secondary activity metric (signal coverage) from evaluator attrs
                        try:
                            _signal_coverage_month = float(eval_df_cont.attrs.get("signal_coverage", float("nan")))
                        except Exception:
                            _signal_coverage_month = float("nan")

                        # carry-out for next month (continuous equities)
                        prev_position    = float(eval_df_cont.attrs.get("last_position", prev_position))
                        prev_eq_strategy = float(eval_df_cont.attrs.get("end_eq_strategy", prev_eq_strategy))
                        prev_eq_bh       = float(eval_df_cont.attrs.get("end_eq_bh", prev_eq_bh))

                        # per-trade log for this month (built from the *continuous* evaluated df)
                        try:
                            trade_df_month = build_trade_log_from_df(eval_df_cont)
                        except Exception as _e:
                            print(f"[WARN] Could not build trade log for month {i + 1}: {_e}")
                            trade_df_month = None

                        # save continuous curves for the cross-month plot (incremental to avoid list growth)
                        _month_df = eval_df_cont[["cstrategy_cont", "creturns_cont"]].copy()
                        try:
                            _month_df.attrs = {}
                        except Exception:
                            pass
                        if self._monthly_all_dfs_concat is None:
                            self._monthly_all_dfs_concat = _month_df
                        else:
                            self._monthly_all_dfs_concat = pd.concat([self._monthly_all_dfs_concat, _month_df])
                        del _month_df

                        if trade_df_month is not None:
                            if self._monthly_trade_dfs_concat is None:
                                self._monthly_trade_dfs_concat = trade_df_month
                            else:
                                try:
                                    trade_df_month.attrs = {}
                                except Exception:
                                    pass
                                self._monthly_trade_dfs_concat = pd.concat(
                                    [self._monthly_trade_dfs_concat, trade_df_month], ignore_index=True
                                )
                            del trade_df_month
                        
                    else:
                        print("[WARN] results DataFrame missing required columns -- skipping bar concat.")
                else:
                    print("[WARN] No self.results to build bar DF from.")


                # Carry-over equities are already updated just above (prev_eq_strategy / prev_eq_bh)
                monthly_bh_factor = float(ret)                  # BH factor this month (continuous)
                equity_strategy   = float(prev_eq_strategy)     # carried strategy equity
                equity_bh         = float(prev_eq_bh)           # carried BH equity

                # Safely capture optional fields for downstream plotting/reconstruction
                features_used = list(getattr(self, "_last_used_features", []))

                _ct_init = best_combo.get("confidence_threshold_init")
                if _ct_init is None:
                    _ct_init = getattr(self, "_last_conf_thr_init", float("nan"))

                _ct_used = getattr(self, "_last_conf_thr_used", None)
                if _ct_used is None:
                    _ct_used = best_combo.get("confidence_threshold")
                if _ct_used is None:
                    _ct_used = float("nan")

                _backoff = getattr(self, "_last_conf_backoff_steps", 0) or 0
                _max_q75 = getattr(self, "_last_max_conf_q75", float("nan"))
                _max_q90 = getattr(self, "_last_max_conf_q90", float("nan"))
                
                # ------------------------------------------------------------------
                # Patch C: Compact per-month gating summary (real-sim; no behavior change)
                # Prints: active_rate, trades, conf_init/used, eligible bars, anchor,
                # and top 3 "filter" contributors if available.
                # ------------------------------------------------------------------
                try:
                    _diag = getattr(self, "_last_eligibility_diag", {}) or {}

                    # Pull components FIRST (avoid UnboundLocalError / silent swallow)
                    _elig_n = int(_diag.get("eligible_bars", 0) or 0)
                    _sess_d = int(_diag.get("session_dropped", 0) or 0)
                    _emb_d  = int(_diag.get("embargo_dropped", 0) or 0)
                    _warm_d = int(_diag.get("warmup_dropped", 0) or 0)
                    _anch_d = int(_diag.get("anchor_dropped", 0) or 0)
                    _anchor = _diag.get("eval_anchor_ts", None)

                    _sum_parts = _elig_n + _sess_d + _emb_d + _warm_d + _anch_d

                    # bars_total must be additive on the eval grid; warn if inconsistent.
                    _bars_total = int(_diag.get("bars_total", 0) or 0)
                    _post_emb_n = int(_diag.get("post_embargo_bars", 0) or 0)

                    # Ensure denominator is ALWAYS defined for summary formatting.
                    # Preference order:
                    #  1) explicit bars_total (additive eval grid),
                    #  2) post_embargo_bars (same meaning in older diags),
                    #  3) eligible + anchor_dropped (also additive within post-embargo bars),
                    #  4) last-resort fallback to any positive count.
                    if _bars_total > 0:
                        _total_n = int(_bars_total)
                    elif _post_emb_n > 0:
                        _total_n = int(_post_emb_n)
                    else:
                        _ea = int(_elig_n + _anch_d)
                        _total_n = int(max(_ea, _elig_n, _sum_parts, 0))


                    # NOTE: these diagnostic counts are NOT mutually exclusive.
                    # Example: `eligible` is (by construction) a subset of `session`,
                    # so summing will usually exceed `bars_total`. Only warn on impossible
                    # accounting (a component exceeds total bars).
                    _parts = {
                        "eligible": int(_elig_n),
                        "session": int(_sess_d),
                        "embargo": int(_emb_d),
                        "warmup": int(_warm_d),
                        "anchor": int(_anch_d),
                    }
                    _max_part = max(_parts.values()) if _parts else 0
                    if _bars_total > 0 and _max_part > _bars_total:
                        print(f"[WARN] [GateSummary][WARN] impossible eligibility counts: bars_total={_bars_total} max_part={_max_part} parts={_parts}")
                    elif bool(getattr(self, "debug", False)) and _bars_total > 0 and _sum_parts > 0 and _bars_total != _sum_parts:
                        # Overlap is expected; keep this at debug-level only.
                        try:
                            log_print(
                                f"[GateSummary][DEBUG] overlapping eligibility counts (expected): "
                                f"bars_total={_bars_total} sum={_sum_parts} parts={_parts}",
                                level="DEBUG",
                            )
                        except Exception:
                            pass

                    # Conf gate filtered count (uses last max_conf snapshot if present)
                    _conf = getattr(self, "_last_conf_stats_max_conf", None)
                    _conf_filt = None
                    try:
                        # _ct_used is expected to exist in this scope (computed upstream)
                        if _conf is not None and np.size(_conf) > 0 and (_ct_used == _ct_used):
                            _conf_arr = np.asarray(_conf, dtype=float)
                            _conf_filt = int(np.sum(_conf_arr < float(_ct_used)))
                    except Exception:
                        _conf_filt = None

                    _reasons = []
                    if _conf_filt is not None:
                        _reasons.append(("conf", _conf_filt))
                    if _sess_d:
                        _reasons.append(("session", _sess_d))
                    if _emb_d:
                        _reasons.append(("embargo", _emb_d))
                    if _anch_d:
                        _reasons.append(("anchor", _anch_d))

                    _reasons = sorted(_reasons, key=lambda kv: kv[1], reverse=True)[:3]
                    _top = ", ".join([f"{k}:{int(v)}" for k, v in _reasons]) if _reasons else "n/a"

                    # Robust formatting (avoid exceptions if upstream values are nan/missing)
                    _ar = float(active_rate) if active_rate == active_rate else float("nan")
                    _tr = int(trades) if trades == trades else 0
                    _ci = float(_ct_init) if _ct_init == _ct_init else float("nan")
                    _cu = float(_ct_used) if _ct_used == _ct_used else float("nan")

                    log_print(
                        f"[GateSummary][M{i+1}] "
                        f"ar={_ar:.3f} trades={_tr} "
                        f"conf_init={_ci:.3f} conf_used={_cu:.3f} "
                        f"eligible={_elig_n}/{_total_n} anchor={_anchor} "
                        f"drops(sess={_sess_d}, emb={_emb_d}, warm={_warm_d}, anch={_anch_d}) "
                        f"top=[{_top}]",
                        level="COMPACT",
                    )
                except Exception as _e:
                    # No silent pass: if summary fails, it MUST be visible during audits.
                    log_print(f"[WARN] [GateSummary][WARN] failed to print gating summary: {_e}", level="COMPACT")



                # Safely read from best_combo; if it's not a dict, fall back to {}
                params_safe = best_combo if isinstance(best_combo, dict) else {}
                
                 
                # Effective confidence threshold for this month (USED by runtime gating/backoff).
                # Schema rule: monthly CSV column `confidence_threshold` must reflect the effective used value.
                _ct_param = params_safe.get("confidence_threshold", None)
                _ct_eff = self._safe_float(_ct_used)
                if not np.isfinite(_ct_eff):
                    _ct_eff = self._safe_float(_ct_param)


                result = {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_months": train_months,
                    "test_months": test_months,
                    
                    # Patch D (persist eligibility deltas for auditability)
                    "elig_raw_month_bars": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("raw_month_bars", 0) or 0),
                    "elig_session_dropped": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("session_dropped", 0) or 0),
                    "elig_embargo_dropped": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("embargo_dropped", 0) or 0),
                    "elig_anchor_dropped": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("anchor_dropped", 0) or 0),
                    "elig_warmup_need": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("warmup_need", 0) or 0),
                    "elig_eval_anchor_ts": (getattr(self, "_last_eligibility_diag", {}) or {}).get("eval_anchor_ts", None),
                    "elig_eligible_bars": int((getattr(self, "_last_eligibility_diag", {}) or {}).get("eligible_bars", 0) or 0),

                    # factors/returns (continuous month)
                    "cum_return":          round(monthly_bh_factor - 1.0, 6),
                    "strategy_return":     round(perf - 1.0, 6),

                    # core knobs used by reconstructor
                    "lags":                 params_safe.get("lags"),
                    "lags_range":           params_safe.get("lags_range"),
                    "lag_depth":            params_safe.get("lag_depth"),
                    "roll_windows":         params_safe.get("roll_windows"),
                    "include_raw_lags":     params_safe.get("include_raw_lags"),

                    # thresholds / switches
                    "label_threshold":      params_safe.get("label_threshold"),
                    "confidence_threshold": _ct_eff,
                    "confidence_threshold_param": _ct_param,
                    "confidence_threshold_init": float(_ct_init),
                    "confidence_threshold_used": float(_ct_used),
                    "conf_backoff_steps":        int(_backoff),
                    "max_conf_q75":              float(_max_q75),
                    "max_conf_q90":              float(_max_q90),

                    "use_extended_features": params_safe.get("use_extended_features"),
                    "use_proba":             params_safe.get("use_proba"),
                    "strategy_type":         params_safe.get("strategy_type"),

                    # per-indicator toggles
                    "use_sma":    params_safe.get("use_sma"),
                    "use_ema":    params_safe.get("use_ema"),
                    "use_macd":   params_safe.get("use_macd"),
                    "use_rsi":    params_safe.get("use_rsi"),
                    "use_bbands": params_safe.get("use_bbands"),
                    "use_atr":    params_safe.get("use_atr"),
                    "use_stoch":  params_safe.get("use_stoch"),
                    "use_adx":    params_safe.get("use_adx"),
                    "use_mtf_ma": params_safe.get("use_mtf_ma"),

                    "indicator_windows": params_safe.get("indicator_windows"),

                    # model metadata
                    "model_type": model_type,

                    # performance snapshot (continuous month + carried equities)
                    "cstrategy":           round(float(perf), 6),
                    "creturns":            round(float(monthly_bh_factor), 6),
                    "outperformance":      round(float(perf) - float(monthly_bh_factor), 6),
                    "equity_strategy":     round(equity_strategy, 6),
                    "equity_bh":           round(equity_bh, 6),
                    "equity_outperformance": round(equity_strategy - equity_bh, 6),

                    # detailed metrics
                    "sharpe":              float(sharpe),
                    "drawdown":            float(drawdown),
                    "trades":              int(trades) if trades == trades else 0,
                    "directional_accuracy":float(directional_accuracy),
                    "precision_macro":     float(precision_macro),
                    "f1_macro":            float(f1_macro),
                    "active_rate":         float(active_rate),
                    "signal_coverage":     float(_signal_coverage_month),
                    "profit_per_hit":      float(profit_per_hit),
                    "return_per_trade":    float(return_per_trade),
                    "win_rate":            float(win_rate),
                    "strategy_volatility": float(strategy_volatility),
                    "kurtosis":            float(kurtosis_val),

                    # trace features actually used this month
                    "features_used": features_used,
                }

                # --- S16.2: Walk-forward transparency data ---
                # Signal counts: how many bars had predictions vs passed confidence gate
                _signals_raw = 0
                _signals_passed = 0
                try:
                    _res_df = getattr(self, "results", None)
                    if _res_df is not None and isinstance(_res_df, pd.DataFrame) and not _res_df.empty:
                        if "pred" in _res_df.columns:
                            _preds = _res_df["pred"].dropna()
                            _signals_raw = int((_preds != 0).sum())
                        if "position_exec" in _res_df.columns:
                            _positions = _res_df["position_exec"].dropna()
                            _signals_passed = int((_positions != 0).sum())
                except Exception:
                    pass
                result["signals_raw"] = _signals_raw
                result["signals_passed_gate"] = _signals_passed

                # Regime distribution: percentage of bars in sideways/trend/volatile
                _pct_sideways = float("nan")
                _pct_trend = float("nan")
                _pct_volatile = float("nan")
                try:
                    _res_df = getattr(self, "results", None)
                    if _res_df is not None and isinstance(_res_df, pd.DataFrame) and not _res_df.empty and "regime_id" in _res_df.columns:
                        _rid = _res_df["regime_id"].dropna()
                        if len(_rid) > 0:
                            _vc = _rid.value_counts(normalize=True)
                            _pct_sideways = float(_vc.get(0, 0.0))
                            _pct_trend = float(_vc.get(1, 0.0))
                            _pct_volatile = float(_vc.get(2, 0.0))
                except Exception:
                    pass
                result["pct_sideways"] = _pct_sideways
                result["pct_trend"] = _pct_trend
                result["pct_volatile"] = _pct_volatile

                # Only try to serialize sub-configs if best_combo is a dict
                if isinstance(best_combo, dict):
                    for key in ["cnn_config", "lstm_config", "transf_config", "dqn_config",
                                "xgb_config", "rf_config", "logit_config"]:
                        if key in best_combo:
                            result[key] = json.dumps(best_combo[key], sort_keys=True)

                # Call stays with your current signature
                self.log_simulation_result(
                    i=i,
                    test_start=test_start, test_end=test_end,
                    perf=float(perf),
                    creturns=float(monthly_bh_factor),
                    sharpe=float(sharpe), trades=int(trades) if trades == trades else 0,
                    drawdown=float(drawdown), cumsum=float(equity_strategy),  # pass strategy equity here
                    result=result, csv_path=csv_path,
                    directional_accuracy=float(directional_accuracy),
                    precision_macro=float(precision_macro),
                    f1_macro=float(f1_macro), active_rate=float(active_rate),
                    profit_per_hit=float(profit_per_hit),
                    equity_bh=float(equity_bh)
                )

                results.append(result)

                _progress_cb = getattr(self, "_progress_callback", None)
                if _progress_cb:
                    phase = "month" if period_unit == "months" else "period"
                    _progress_cb(phase, model_type, {
                        "period": i + 1,
                        "total_periods": n_periods,
                        "sharpe": result.get("sharpe"),
                        "trades": result.get("trades"),
                        "equity_strategy": result.get("equity_strategy"),
                        "equity_bh": result.get("equity_bh"),
                        "drawdown": result.get("drawdown"),
                        "win_rate": result.get("win_rate"),
                        "precision_macro": result.get("precision_macro"),
                        "f1_macro": result.get("f1_macro"),
                        "return_pct": result.get("cstrategy"),
                        "directional_accuracy": result.get("directional_accuracy"),
                        "active_rate": result.get("active_rate"),
                    })
                
                # PBO/MCS monthly bookkeeping (does not affect trading logic)
                try:
                    self._record_wfo_monthly_result(result)
                except Exception as _e:
                    if self._is_debug():
                        print(f"[PBO/MCS] Failed to record monthly result: {_e}")
                        
                time.sleep(1)
            finally:
                _hard_free()
                import gc as _gc
                _gc.collect()

                # Also clear feature cache after each month to avoid accumulation
                self._clear_feature_cache()

                # Periodic deep-pool restart to prevent worker RSS from growing across months.
                # The ProcessPoolExecutor worker builds TF models each month and accumulates
                # residual allocations even after clear_session(). Restarting every 6 months
                # (configurable via MLB_DEEP_POOL_RESTART_INTERVAL) gives the worker a clean
                # process memory footprint.
                _restart_interval = int(os.environ.get("MLB_DEEP_POOL_RESTART_INTERVAL", "6"))
                if _restart_interval > 0 and (_month_ix > 0) and (_month_ix % _restart_interval == 0):
                    try:
                        from pipeline.backtester.deep_mixin import DeepMixin
                        DeepMixin._shutdown_deep_pool()
                    except Exception:
                        pass
                _month_ix += 1
        
        # ---------------------------------------------------------------------
        # Wrap-up: aggregate results, save artifacts into this run's out_dir
        # ---------------------------------------------------------------------
        df_months = pd.DataFrame(results)
        if not df_months.empty:
            print(f"\n[OK] Real Trading Simulation Complete ({months} Months)")

            # df_months contains only the valid months; all_dfs has one per-bar DF per valid month in order.
            df_months_reset = df_months.reset_index(drop=True)
            df_rows = df_months_reset.to_dict(orient="records")

            # Month-level artifacts (csv_month_k, featuresconfigused_k.txt,
            # monthly_equity_k.png, feature_heatmap_k.png) are now handled
            # by the bar_concat-based block further below, gated by SAVE_* flags.


            # -------------------------------------------------------------
            # Monthly trade summary for this model & repetition
            # -------------------------------------------------------------
            if SAVE_TRADES.get("monthly_summary_per_rep_csv", True):
                try:
                    import pandas as _pd
                    import os as _os

                    monthly_trade_summaries = []

                    # df_months_reset and trade_dfs are in the same order of valid months
                    for idx, row in df_months_reset.iterrows():
                        if idx >= len(trade_dfs):
                            continue
                        tdf = trade_dfs[idx]
                        if tdf is None or tdf.empty:
                            continue

                        period_idx = int(row.get("period_idx", idx + 1))
                        n_trades = int(len(tdf))

                        if n_trades > 0:
                            wins = tdf["pnl_pct"] > 0
                            win_rate = float(wins.mean())

                            avg_pnl = float(tdf["pnl_pct"].mean())
                            med_pnl = float(tdf["pnl_pct"].median())
                            std_pnl = float(tdf["pnl_pct"].std(ddof=0))

                            avg_hold = float(tdf["holding_minutes"].mean())
                            med_hold = float(tdf["holding_minutes"].median())
                        else:
                            win_rate = avg_pnl = med_pnl = std_pnl = 0.0
                            avg_hold = med_hold = 0.0

                        monthly_trade_summaries.append(
                            {
                                "run_id": _os.path.basename(RUN_DIR_LOCAL),
                                "model_type": model_type,
                                "repetition": int(rep_idx),
                                "period_idx": period_idx,
                                "test_start": row.get("test_start"),
                                "test_end": row.get("test_end"),
                                "n_trades": n_trades,
                                "win_rate": win_rate,
                                "avg_pnl_pct": avg_pnl,
                                "median_pnl_pct": med_pnl,
                                "std_pnl_pct": std_pnl,
                                "avg_holding_minutes": avg_hold,
                                "median_holding_minutes": med_hold,
                            }
                        )

                    if monthly_trade_summaries:
                        monthly_df = _pd.DataFrame(monthly_trade_summaries)
                        monthly_path = _os.path.join(
                            final_dirs["csv"],
                            f"monthly_trade_summary_rep{rep_idx}.csv",
                        )
                        monthly_df.to_csv(
                            monthly_path,
                            index=False,
                            float_format="%.10f",
                        )
                        print(f"[OK] Saved monthly trade summary for rep {rep_idx} -> {monthly_path}")
                    else:
                        print("[INFO] No trades recorded; skipping monthly trade summary.")

                except Exception as _e:
                    print(f"[WARN] Could not build monthly trade summary: {_e}")
            else:
                if self._is_debug():
                    print("[INFO] Monthly trade summary disabled via SAVE_TRADES['monthly_summary_per_rep_csv'].")

            # -------------------------------------------------------------
            # Per-trade BH vs model comparison at entry/exit
            # -------------------------------------------------------------
            if SAVE_TRADES.get("trade_entry_exit_compare_csv", True):
                try:
                    import pandas as _pd
                    import os as _os
                    import math as _math

                    trade_compare_rows = []

                    # df_months_reset, all_dfs and trade_dfs share the same valid-month ordering
                    for idx, row in df_months_reset.iterrows():
                        if idx >= len(trade_dfs) or idx >= len(all_dfs):
                            continue

                        tdf = trade_dfs[idx]
                        mdf = all_dfs[idx]

                        if tdf is None or tdf.empty or mdf is None or mdf.empty:
                            continue
                        if not {"cstrategy_cont", "creturns_cont"} <= set(mdf.columns):
                            continue

                        eq_index = mdf.index

                        for _, tr in tdf.iterrows():
                            try:
                                entry_i = int(tr.get("entry_bar"))
                                exit_i = int(tr.get("exit_bar"))
                            except Exception:
                                continue

                            if entry_i < 0 or exit_i < 0:
                                continue
                            if entry_i >= len(mdf) or exit_i >= len(mdf):
                                continue

                            # ensure order
                            if exit_i < entry_i:
                                entry_i, exit_i = exit_i, entry_i

                            entry_time = eq_index[entry_i]
                            exit_time = eq_index[exit_i]

                            strat_start = float(mdf["cstrategy_cont"].iloc[entry_i])
                            strat_end   = float(mdf["cstrategy_cont"].iloc[exit_i])
                            bh_start    = float(mdf["creturns_cont"].iloc[entry_i])
                            bh_end      = float(mdf["creturns_cont"].iloc[exit_i])

                            def _rel_ret(end, start):
                                try:
                                    if start == 0.0 or not _math.isfinite(start) or not _math.isfinite(end):
                                        return float("nan")
                                except Exception:
                                    return float("nan")
                                return float(end / start - 1.0)

                            bh_ret = _rel_ret(bh_end, bh_start)
                            model_curve_ret = _rel_ret(strat_end, strat_start)

                            pnl_pct = float(tr.get("pnl_pct", float("nan")))
                            edge_vs_bh = float("nan")
                            if _math.isfinite(bh_ret) and _math.isfinite(pnl_pct):
                                edge_vs_bh = float(pnl_pct - bh_ret)

                            trade_compare_rows.append(
                                {
                                    "run_id": _os.path.basename(RUN_DIR_LOCAL),
                                    "model_type": model_type,
                                    "repetition": int(rep_idx),
                                    "period_idx": int(row.get("period_idx", idx + 1)),
                                    "trade_id": tr.get("trade_id"),
                                    "side": tr.get("side"),
                                    "side_sign": tr.get("side_sign"),
                                    "entry_bar": entry_i,
                                    "exit_bar": exit_i,
                                    "entry_time": entry_time,
                                    "exit_time": exit_time,
                                    "bars_held": tr.get("bars_held"),
                                    "holding_minutes": tr.get("holding_minutes"),
                                    "pnl_pct": pnl_pct,
                                    "model_curve_return_pct": model_curve_ret,
                                    "bh_return_pct": bh_ret,
                                    "edge_vs_bh_pct": edge_vs_bh,
                                }
                            )

                    if trade_compare_rows:
                        compare_df = _pd.DataFrame(trade_compare_rows)
                        compare_path = _os.path.join(
                            final_dirs["csv"],
                            f"trade_entry_exit_compare_rep{rep_idx}.csv",
                        )
                        compare_df.to_csv(
                            compare_path,
                            index=False,
                            float_format="%.10f",
                        )
                        print(
                            f"[OK] Saved trade entry/exit BH comparison for rep {rep_idx} -> "
                            f"{compare_path}"
                        )
                    else:
                        if self._is_debug():
                            print(
                                "[INFO] No trades with usable equity curves for trade_entry_exit_compare; skipping CSV."
                            )

                except Exception as _e:
                    print(f"[WARN] Could not build trade entry/exit comparison CSV: {_e}")
            else:
                if self._is_debug():
                    print(
                        "[INFO] Trade entry/exit comparison disabled via "
                        "SAVE_TRADES['trade_entry_exit_compare_csv']."
                    )

            

        # 2) Final (model-level) artifacts
        #    - feature_heatmap_final.png over all months of _this_ model
        #      HEAVY -> gated by config + SKIP_PLOTS
        try:
            cfg_local = getattr(self, "features_config", {}) or {}
            do_feat_freq = bool(cfg_local.get("deploy_feature_freq", True))
            if do_feat_freq and not SKIP_PLOTS:
                save_feature_frequency_from_monthly_results(
                    df_months,
                    base_features=[],
                    out_png=os.path.join(final_dirs["graphs"], "feature_heatmap_final.png"),
                    top_k=30,
                    style="nature",
                    palette="okabe_ito_no_black",
                    exclude_prefixes=("returns_lag", "hour"),
                    collapse_raw_lags=True,
                    out_csv=os.path.join(final_dirs["csv"], "feature_frequency_monthly.csv"),
                )
        except Exception as _e:
            if self._is_debug():
                print(f"[WARN] Feature-frequency (model-level) heatmap skipped: {_e}")



        #    - csv over all months of this model
        _csv_exclude = [
            "features_used",
            "dqn_config", "cnn_config", "lstm_config", "transformer_config", "xgb_config", "rf_config", "logit_config",
        ]
        df_months.drop(columns=[c for c in _csv_exclude if c in df_months.columns], errors="ignore") \
            .to_csv(os.path.join(final_dirs["csv"], f"real_trading_simulation_{model_type}.csv"),
                    index=False, float_format="%.10f")

                # (optional) one consolidated TXT with per-month feature/config refs
        try:
            agg_path = os.path.join(final_dirs["csv"], "featuresconfigused_all.txt")
            with open(agg_path, "w", encoding="utf-8") as f:
                for idx, row in df_months.reset_index(drop=True).iterrows():
                    k = idx + 1
                    f.write(f"=== Month {k} ===\n")
                    feats = row.get("features_used", [])
                    f.write("features_used:\n")
                    if isinstance(feats, str):
                        f.write(feats + "\n")
                    else:
                        # Normalize feats: non-iterables (NaN, scalars) -> empty list
                        try:
                            from collections.abc import Iterable
                            if not isinstance(feats, Iterable):
                                feats = []
                        except Exception:
                            feats = []
                        for ft in (feats or []):
                            f.write(str(ft) + "\n")
                    for cfg_key in (
                        "cnn_config", "lstm_config", "transformer_config",
                        "xgb_config", "rf_config", "logit_config", "dqn_config"
                    ):
                        if cfg_key in row:
                            f.write(f"\n{cfg_key}:\n{row[cfg_key]}\n")
                    f.write("\n")
        except Exception as _e:
            print(f"[WARN] Could not write aggregated features/config dump: {_e}")

        # Do NOT create a new timestamped folder here; reuse the one from earlier.
        # Just make sure it exists.
        os.makedirs(out_dir, exist_ok=True)

        # One-shot monthly feature-frequency heatmap (across all months in this repeat) -> All/
        # derive the repeat id directly from the data (written in main(): df_sim["rep"] = rep)
        
        rep = int(df_months['rep'].dropna().iloc[0]) if ('rep' in df_months.columns and df_months['rep'].notna().any()) else 1

        try:
            save_feature_frequency_from_monthly_results(
                df_months,
                base_features=[],
                out_png=os.path.join(buckets["All"]["heatmaps"], f"feature_frequency_monthly_rep{rep}.png"),
                top_k=30,
                style="nature",
                palette="okabe_ito_no_black",
                exclude_prefixes=("returns_lag", "hour"),
                collapse_raw_lags=True,
                out_csv=os.path.join(buckets["All"]["csv"], f"feature_frequency_monthly_rep{rep}.csv"),
            )
        except Exception as _e:
            print(f"[WARN] Could not save monthly feature frequency heatmap: {_e}")

        # Per-bar comparison CSV/PNG -- run ONCE (single model vs BH)
        try:
            _bar_concat = getattr(self, "_monthly_all_dfs_concat", None)
            if _bar_concat is not None and not getattr(_bar_concat, "empty", True):
                bar_concat = _bar_concat.sort_index()
                bar_concat.columns = ["cstrategy_cont", "creturns_cont"]
                self.bar_concat = bar_concat

                _trades = getattr(self, "_monthly_trade_dfs_concat", None)
                if _trades is not None and not getattr(_trades, "empty", True):
                    self.trade_log = _trades
                else:
                    self.trade_log = pd.DataFrame()

                bt_dict = {
                    "BH": bar_concat["creturns_cont"],
                    f"{model_type}_equity": bar_concat["cstrategy_cont"],
                }
                cfg_local = getattr(self, "features_config", {}) or {}
                light_output = bool(cfg_local.get("light_output", False))
                if not (SKIP_PLOTS or light_output):
                    save_model_bar_comparison_outputs(
                        bt_dict,
                        csv_dir=final_dirs["csv"],
                        png_dir=final_dirs["graphs"],
                        style="nature",
                        palette="okabe_ito_no_black",
                        bh_color="#666666",
                        n_time_parts=10,
                        dpi=300,
                        line_width=1.2,
                        annotate_coverage=False,
                    )

        except Exception as e:
            print(f"[WARN] Per-bar comparison (single model) failed: {e}")

        # --- Now that bar_concat exists, write month PNGs by slicing it ---
        try:
            cfg_local = getattr(self, "features_config", {}) or {}

            # Combine config flags with SAVE_* toggles
            do_csv = bool(SAVE_METRICS.get("per_month_metrics_csv", False))
            do_feat_txt = bool(SAVE_FEATURES.get("featuresconfig_txt", False))
            do_equity = (
                bool(SAVE_EQUITY.get("per_month_equity_png", False))
                and bool(cfg_local.get("save_monthly_equity_plots", False))
            )
            do_heatmap = (
                bool(SAVE_FEATURES.get("monthly_heatmap_png", False))
                and bool(cfg_local.get("save_monthly_feature_heatmaps", False))
            )

            # If nothing is enabled, skip the whole loop (but keep monthly stats / PBO below)
            if not (do_csv or do_feat_txt or do_equity or do_heatmap):
                if self._is_debug():
                    print(
                        "[INFO] All SAVE_* per-month artifacts disabled; "
                        "skipping month-level file writes."
                    )
            else:
                for idx, row in df_months.iterrows():
                    month_ix = int(row.get("period_idx", idx + 1))
                    mdirs = month_dir_path(model_base_dir, month_ix)

                    # (a) CSV with only that month row (clean)
                    if do_csv:
                        _csv_exclude = {
                            "features_used",
                            "dqn_config", "cnn_config", "lstm_config", "transformer_config",
                            "xgb_config", "rf_config", "logit_config",
                        }
                        row_csv = {kk: vv for kk, vv in row.items() if kk not in _csv_exclude}

                        pd.DataFrame([row_csv]).to_csv(
                            os.path.join(mdirs["csv"], f"csv_month_{month_ix}.csv"),
                            index=False,
                            float_format="%.10f",
                        )

                    # (a2) Dump features/configs to a TXT file (only if enabled)
                    if do_feat_txt:
                        try:
                            dump_path = os.path.join(
                                mdirs["csv"],
                                f"featuresconfigused_{month_ix}.txt",
                            )
                            with open(dump_path, "w", encoding="utf-8") as f:
                                f.write(f"Month: {month_ix}\n")
                                f.write("\nfeatures_used:\n")
                                feats = row.get("features_used", [])
                                if isinstance(feats, str):
                                    f.write(feats + "\n")
                                else:
                                    # Normalize feats: non-iterables (NaN, scalars) -> empty list
                                    try:
                                        from collections.abc import Iterable
                                        if not isinstance(feats, Iterable):
                                            feats = []
                                    except Exception:
                                        feats = []
                                    for ft in (feats or []):
                                        f.write(str(ft) + "\n")
                                for cfg_key in (
                                    "cnn_config", "lstm_config", "transformer_config",
                                    "xgb_config", "rf_config", "logit_config", "dqn_config",
                                ):
                                    if cfg_key in row:
                                        f.write(f"\n{cfg_key}:\n{row[cfg_key]}\n")
                        except Exception as _e:
                            print(
                                f"[WARN] Could not write features/config dump for month {month_ix}: {_e}"
                            )

                    # (b) slice per-bar equity for that month and plot (HEAVY -> gated)
                    if do_equity:
                        ts, te = row.get("test_start"), row.get("test_end")
                        if pd.notna(ts) and pd.notna(te):
                            mdf = self.bar_concat.loc[ts:te].copy()
                        else:
                            mdf = self.bar_concat.copy()

                        if (
                            mdf is not None
                            and not mdf.empty
                            and all(
                                c in mdf.columns
                                for c in ("cstrategy_cont", "creturns_cont")
                            )
                        ):
                            save_month_equity_graph(
                                mdf,
                                out_csv=None,
                                out_png=os.path.join(
                                    mdirs["graphs"],
                                    f"monthly_equity_{month_ix}.png",
                                ),
                                label_model=disp_name,
                                title=f"{disp_name} -- Month {month_ix}",
                                dpi=300,
                            )

                    # (c) Month-only feature heatmap (HEAVY -> gated)
                    if do_heatmap:
                        save_feature_heatmap_for_single_month(
                            pd.DataFrame([row]),
                            out_png=os.path.join(
                                mdirs["heatmaps"],
                                f"feature_heatmap_{month_ix}.png",
                            ),
                        )

        except Exception as e:
            print(f"[WARN] Month artifact export failed: {e}")

        # Persist the monthly stats table for the whole simulation period
        try:
            if not df_months.empty:
                # Use the shared run directory so all models land in the same master CSV
                buckets = comparison_dirs(RUN_DIR_LOCAL)
                save_monthly_model_stats(df_months, buckets["All"]["csv"], filename="monthly_model_stats.csv")

        except Exception as e:
            print(f"[WARN] Saving monthly model stats CSV failed: {e}")
            
        
        # Optional post-hoc PBO/MCS analysis (read-only)
        try:
            cfg = getattr(self, "features_config", {}) or {}
            if bool(cfg.get("enable_pbo_mcs_analysis", False)):
                pbo_mcs_result = self.run_pbo_mcs_analysis()
                # keep for external inspection
                self._pbo_mcs_result = pbo_mcs_result
        except Exception as e:
            print(f"[PBO/MCS] Analysis failed: {e}")


        # Restore caller flags to prevent mode leakage across runs (CV vs real-sim)
        try:
            self._in_real_sim = _prev_real
            self._dbg_first_bars = _prev_dbg
        except Exception:
            pass

        # Restore mode flags (keeps CV-only behavior from leaking into subsequent runs).
        try:
            self._in_optuna_cv = bool(_prev_optuna_cv)
        except Exception:
            pass
        try:
            self._in_real_sim = bool(_prev_real)
        except Exception:
            pass
        try:
            self._dbg_first_bars = bool(_prev_dbg)
        except Exception:
            pass

        return df_months
    
    
    
# TRIAL_COUNTS_FULL = {
#     "logistic":       {"random": 15, "bayes": 45},
#     "svm":            {"random": 15, "bayes": 45},
#     "decision_tree":  {"random": 15, "bayes": 45},
#     "random_forest":  {"random": 15, "bayes": 45},
#     "xgboost":        {"random": 15, "bayes": 45},
#     "lstm":           {"random": 20, "bayes": 50},
#     "cnn":            {"random": 0,  "bayes": 0},
#     "transformer":    {"random": 20, "bayes": 50},
#     "ensemble_adaptive_regime":     {"random": 20, "bayes": 50},
#     "ensemble_cnn_lstm_xgboost":    {"random": 20, "bayes": 50},
#     "dqn":            {"random": 0,  "bayes": 0},
# }


# Used for thesis
# TRIAL_COUNTS = {
#     "logistic":       {"random": 20, "bayes": 40},
#     "svm":            {"random": 20, "bayes": 40},
#     "decision_tree":  {"random": 20, "bayes": 40},
#     "random_forest":  {"random": 20, "bayes": 40},
#     "xgboost":        {"random": 20, "bayes": 40},
#     "lstm":           {"random": 20, "bayes": 40},
#     "cnn":            {"random": 20, "bayes": 40},
#     "transformer":    {"random": 20, "bayes": 40},
#     "ensemble_adaptive_regime":     {"random": 20, "bayes": 40},
#     "ensemble_cnn_lstm_xgboost":    {"random": 20, "bayes": 40},
#     "dqn":            {"random": 0,  "bayes": 0},
# }

# For quick system check.
TRIAL_COUNTS = {
    "logistic":       {"random": 3, "bayes": 3},
    "svm":            {"random": 3, "bayes": 3},
    "decision_tree":  {"random": 3, "bayes": 3},
    "random_forest":  {"random": 3, "bayes": 3},
    "xgboost":        {"random": 3, "bayes": 3},
    "lstm":           {"random": 3, "bayes": 3},
    "cnn":            {"random": 3, "bayes": 3},
    "transformer":    {"random": 3, "bayes": 3},
    "ensemble_adaptive_regime":     {"random": 3, "bayes": 3},
    "ensemble_cnn_lstm_xgboost":    {"random": 3, "bayes": 3},
    "dqn":            {"random": 3,  "bayes": 3},
}



