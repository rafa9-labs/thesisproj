"""Auto-extracted mixin -- see composed.py for the full MLBacktester."""
from pipeline._imports import *  # noqa: F401,F403


class EvaluationMixin:
    """
    evaluate_strategy, walk-forward, logging

    Auto-extracted from MLBacktesterNoWFO.py lines 10708-11190.
    """
    def get_walk_forward_splits(self, walk_data, train_months_list, test_months_list, max_end,
                                   period_unit="months"):
        """
        Generate WFO splits, shrinking train_months if necessary so that at least
        one split is produced (when data is limited).

        period_unit: "months" (default), "weeks", or "days" -- controls walk-forward granularity.
        """
        from config import period_offset, periods_between, convert_month_count_to_periods

        tasks = []
        first = walk_data.index[0]
        avail_periods = periods_between(first, max_end, unit=period_unit)

        for train_months in train_months_list:
            for test_months in test_months_list:
                train_periods = convert_month_count_to_periods(train_months, period_unit)
                test_periods = convert_month_count_to_periods(test_months, period_unit)
                req = train_periods + test_periods
                if req > avail_periods:
                    min_train = convert_month_count_to_periods(6, period_unit)
                    best_train = max(min_train, avail_periods - test_periods)
                    if best_train < min_train:
                        print(f"[WFO] No feasible split: need {req} {period_unit}, have {avail_periods}. Skipping ({train_periods}/{test_periods}).")
                        continue
                    print(f"[WFO] Shrinking train {train_periods}->{best_train} {period_unit} due to limited history ({avail_periods} {period_unit}).")
                    train_periods_eff = int(best_train)
                else:
                    train_periods_eff = int(train_periods)

                start_date = first
                end_needed = start_date + period_offset(train_periods_eff + test_periods, unit=period_unit)
                if end_needed > max_end:
                    start_date = max_end - period_offset(train_periods_eff + test_periods, unit=period_unit)

                while True:
                    if start_date + period_offset(train_periods_eff + test_periods, unit=period_unit) > max_end:
                        break
                    tasks.append((start_date, train_periods_eff, test_periods, period_unit))
                    start_date += period_offset(test_periods, unit=period_unit)

        try:
            print(f"[WFO] available_{period_unit}={avail_periods} | requested train/test={train_months_list}/{test_months_list} | splits={len(tasks)}")
        except Exception:
            pass
        return tasks

    
        
    def evaluate_strategy(self, best_params, train_start, train_end, test_start, test_end):
        """
        Route a single train/test evaluation to the correct routine based on `model_type`.

        Notes
        -----
        - Avoids pre-building models here; the called test/eval functions build what they need.
        - Accepts legacy grids that used `lags_range` instead of `lags`.

        Returns
        -------
        tuple[float, ...]
            The standard 16-tuple of metrics produced by the test/eval functions.
        """

        # CLEANUP: local debug + print-once helpers (no algorithm change)
        _dbg = bool(getattr(self, "_is_debug", lambda: False)()) or bool(getattr(self, "debug", False))

        def _dprint(msg: str):
            # DEBUG: quiet unless debug
            if _dbg:
                print(msg)

        def _print_once(key: str, msg: str, debug_only: bool = False):
            # CLEANUP: prevent print storms across CV / real-sim loops
            if debug_only and (not _dbg):
                return
            attr = f"_eval_strategy_once__{key}"
            if getattr(self, attr, False):
                return
            print(msg)
            setattr(self, attr, True)

        def _safe_len(x):
            try:
                return len(x)
            except Exception:
                return None

        # ---- Backward-compat for 'lags_range' ----
        if "lags" not in best_params and "lags_range" in best_params:
            # CLEANUP: keep the audit, avoid spam
            _print_once("lags_range_bc", "[WARN] 'lags' not in best_params, using 'lags_range'.")
            best_params["lags"] = int(best_params["lags_range"])

        # In real_trading_simulation, force each evaluation attempt (Top-N, consensus, etc.)
        # to start from the same deterministic month baseline to avoid config drift.
        in_real_sim = bool(getattr(self, "_in_real_sim", False))
        if in_real_sim:
            _base = getattr(self, "_month_base_features_config", None)
            if isinstance(_base, dict) and _base:
                try:
                    self.features_config = deepcopy(_base)
                except Exception as e:
                    # DEBUG: don't swallow silently
                    _dprint(f"[WARN] [evaluate_strategy] deepcopy(_month_base_features_config) failed: {e}")

        # ---- Basic coercions / defaults ----
        model_type = best_params["model_type"]

        # Respect user-provided toggle if present; otherwise keep current instance setting
        self.use_extended_features = best_params.get(
            "use_extended_features",
            getattr(self, "use_extended_features", True)
        )

        # Safe defaults (avoid KeyError)
        label_threshold = float(best_params.get("label_threshold", 1e-4))

        # IMPORTANT: do not hard-default confidence_threshold here.
        # Resolve it AFTER merging params into features_config so CV and real-sim match.
        confidence_threshold = best_params.get("confidence_threshold", None)
        lags = int(best_params.get("lags", 8))

        # CLEANUP: one unified snapshot printer (logging-only)
        def _print_eval_snapshot(_model: str, _cfg: dict, _lags: int, _conf_thr, _calib):
            if _dbg or in_real_sim:
                print(
                    f"[EVAL-SNAPSHOT] model={_model} | "
                    f"lags={_lags} | "
                    f"lag_depth={_cfg.get('lag_depth')} | "
                    f"roll_windows={_cfg.get('roll_windows') or _cfg.get('roll_windows_key') or _cfg.get('roll_windows_key_v2')} | "
                    f"use_fracdiff={_cfg.get('use_fracdiff')} | "
                    f"confidence_threshold={_conf_thr} | "
                    f"calibrate_method={_calib}"
                )

        # ANCHOR: # ---------- Pure Transformer + XGB (explicit, no DQN) ----------
        # Transformer+XGB path has been retired in this project build.
        # Fail fast if the model_type accidentally appears in a grid / Top-N pool.
        if model_type in {"transformer_xgb", "transformer_xgb_only"}:
            raise ValueError(
                "[ERROR] model_type=transformer_xgb(_only) is not supported in this build. "
                "Remove it from configs / Top-N pools."
            )
            
        if model_type == "dqn":
            # ANCHOR: # ---------- DQN only ----------
            # DQN is routed differently than supervised models; keep the dedicated handler.
            # But still merge best_params into features_config so gating knobs (coverage intent etc.)
            # are visible during this evaluation, then restore config to avoid cross-month drift.
            _cfg_snapshot = deepcopy(self.features_config)
            try:
                self._merge_params_into_features_config(best_params, force_lags=lags)
                metrics = self.test_dqn_strategy(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    lags=lags,
                    dqn_config=best_params.get("dqn_config", {}),
                )
            finally:
                self.features_config = _cfg_snapshot

            if not isinstance(metrics, tuple) or len(metrics) != 16:
                raise ValueError("[ERROR] dqn path did not return 16 metrics")
            return metrics
 

        # ---------- Ensembles: CNN+LSTM+XGB or Adaptive Regime ----------
        elif model_type in {"ensemble_cnn_lstm_xgboost", "ensemble_adaptive_regime"}:
            _cfg_snapshot = deepcopy(self.features_config)
            try:
                # Ensure tuned feature toggles and per-model knobs are visible to the pipeline
                self._merge_params_into_features_config(best_params, force_lags=lags)

                # Build/normalize ensemble_config
                ens_cfg = dict(best_params.get("ensemble_config", {}))

                # Allow passing sub-configs directly in best_params; copy them into ensemble_config if missing
                for sub in ("cnn_config", "lstm_config", "transformer_config", "xgb_config"):
                    if sub not in ens_cfg and sub in best_params and isinstance(best_params[sub], dict):
                        ens_cfg[sub] = dict(best_params[sub])

                # Do not propagate confidence_threshold here; coverage/backtests will compute it after
                if "confidence_threshold" in ens_cfg:
                    ens_cfg.pop("confidence_threshold", None)
                if "calibrate_method" in best_params and "calibrate_method" not in ens_cfg:
                    ens_cfg["calibrate_method"] = str(best_params["calibrate_method"]).lower()

                cfg = self.features_config or {}

                # CLEANUP: snapshot line was unconditional; now debug/real-sim only.
                # Also show the *source* of confidence_threshold since ens_cfg removes it intentionally.
                _print_eval_snapshot(
                    _model=model_type,
                    _cfg=cfg,
                    _lags=lags,
                    _conf_thr=best_params.get("confidence_threshold", cfg.get("confidence_threshold")),
                    _calib=ens_cfg.get("calibrate_method") or cfg.get("calibrate_method"),
                )

                metrics = self.test_ensemble_strategy(
                    train_start=train_start, train_end=train_end,
                    test_start=test_start,  test_end=test_end,
                    lags=lags,
                    label_threshold=label_threshold,
                    ensemble_config=ens_cfg,
                    model_type=model_type,
                )
            finally:
                # Prevent param bleed to subsequent runs
                self.features_config = _cfg_snapshot

            # Optional: fail-fast if your pipeline reports effective lags different from tuned
            if hasattr(self, "_effective_lags_last"):
                eff = int(getattr(self, "_effective_lags_last"))
                if eff != lags:
                    raise RuntimeError(
                        f"[ABORT] Effective lags={eff} differs from tuned lags={lags}. "
                        f"Refuse to evaluate with a silently-shrunk spec."
                    )

            if (not isinstance(metrics, tuple)) or (len(metrics) != 16):
                n = _safe_len(metrics)
                raise ValueError(f"[ERROR] test_ensemble_strategy() returned {n} values -- expected 16")
            return metrics

        # ---------- CNN / LSTM / Transformer / Classical ML ----------
        else:
            # Merge tuned trial params into a TEMP config that is visible to prepare_features(),
            # run the evaluation, then restore whatever the backtester had before.
            _cfg_snapshot = deepcopy(self.features_config)
            try:
                self._merge_params_into_features_config(best_params, force_lags=lags)
                cfg = self.features_config or {}

                # Resolve confidence_threshold consistently with CV:
                # - use tuned param if present
                # - else use merged cfg if present
                # - else fallback: deep models -> 0.0 (no silent no-trade), others -> 0.80
                if confidence_threshold is None:
                    if model_type in {"cnn", "lstm", "transformer"}:
                        confidence_threshold = float(cfg.get("confidence_threshold", 0.0))
                    else:
                        confidence_threshold = float(cfg.get("confidence_threshold", 0.80))
                else:
                    confidence_threshold = float(confidence_threshold)

                # CLEANUP: unify snapshot printing; keep GateInfo line as audit
                _print_eval_snapshot(
                    _model=(cfg.get("model_type") or model_type),
                    _cfg=cfg,
                    _lags=lags,
                    _conf_thr=confidence_threshold,
                    _calib=cfg.get("calibrate_method"),
                )
                if (_dbg or in_real_sim) and (cfg.get("target_active_rate", None) is not None):
                    _print_once(
                        "gateinfo_target_active_rate",
                        "[GateInfo] target_active_rate is set -> coverage-calibrated threshold is used; "
                        "fixed confidence_threshold is ignored.",
                    )

                metrics = self.test_strategy(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    lags=lags,
                    confidence_threshold=confidence_threshold,
                    label_threshold=label_threshold,
                )
            finally:
                # Always restore caller state so Top-N tries and later folds don't leak config
                self.features_config = _cfg_snapshot

            # Optional: fail-fast on silent lags shrink (if your pipeline sets this attribute)
            if hasattr(self, "_effective_lags_last"):
                eff = int(getattr(self, "_effective_lags_last"))
                if eff != lags:
                    raise RuntimeError(
                        f"[ABORT] Effective lags={eff} differs from tuned lags={lags}. "
                        f"Refuse to evaluate with a silently-shrunk spec."
                    )

            # Validate the fixed-length contract
            if (not isinstance(metrics, tuple)) or (len(metrics) != 16):
                n = _safe_len(metrics)
                raise ValueError(f"[ERROR] test_strategy() returned {n} values -- expected 16")

            return metrics


    def _record_wfo_monthly_result(self, result: dict) -> None:
        """
        Append a compact monthly record for PBO/MCS and walk-forward analysis.

        Parameters
        ----------
        result : dict
            The per-month result dict built inside real_trading_simulation.
            Expected keys (if present): 'model_type', 'strategy_type',
            'test_start', 'test_end', 'strategy_return', 'cum_return',
            'sharpe', 'trades'. Optional: 'train_sharpe', 'train_start',
            'train_end', 'signals_raw', 'signals_passed_gate',
            'pct_sideways', 'pct_trend', 'pct_volatile'.
        """
        try:
            import pandas as _pd
            mt = result.get("model_type", getattr(self, "model_type", ""))
            st = result.get("strategy_type", None)
            sid = f"{mt}:{st}" if st is not None else str(mt)

            rec = {
                "strategy_id": sid,
                "model_type": mt,
                "strategy_type": st,
                "test_start": _pd.to_datetime(result.get("test_start")),
                "test_end": _pd.to_datetime(result.get("test_end")),
                "train_start": _pd.to_datetime(result.get("train_start")) if result.get("train_start") is not None else float("nan"),
                "train_end": _pd.to_datetime(result.get("train_end")) if result.get("train_end") is not None else float("nan"),
                "strategy_return": float(result.get("strategy_return", float("nan"))),
                "bh_return": float(result.get("cum_return", float("nan"))),
                "sharpe": float(result.get("sharpe", float("nan"))),
                "trades": int(result.get("trades", 0) or 0),
                "train_sharpe": float(result.get("train_sharpe", float("nan"))),
                "signals_raw": int(result.get("signals_raw", 0) or 0),
                "signals_passed_gate": int(result.get("signals_passed_gate", 0) or 0),
                "pct_sideways": float(result.get("pct_sideways", float("nan"))),
                "pct_trend": float(result.get("pct_trend", float("nan"))),
                "pct_volatile": float(result.get("pct_volatile", float("nan"))),
            }
        except Exception as _e:
            if self._is_debug():
                print(f"[PBO/MCS] Failed to build monthly record: {_e}")
            return

        self._wfo_monthly_records.append(rec)
        

    def log_simulation_result(
        self,
        i: int,
        test_start,
        test_end,
        perf: float,
        creturns: float,
        sharpe: float,
        trades: int,
        drawdown: float,
        cumsum: float,
        result: dict,
        csv_path: str,
        directional_accuracy: float,
        precision_macro: float,
        f1_macro: float,
        active_rate: float,
        profit_per_hit: float,
        equity_bh: float | None = None,
    ):
        """
        Append a single fold's summary metrics to a CSV and print a concise log line.

        Parameters
        ----------
        i : int
            Fold index (0-based) -- will be logged as month = i+1.
        perf, creturns : float
            Monthly equity factors for strategy and buy&hold (e.g., 0.995, 1.012).
        cumsum : float
            Strategy continuous equity for the month (will be reported as equity - 1 in 'cumsum').
        equity_bh : float | None
            Buy&Hold continuous equity for the month (optional).
        """
        # ensure the output directory exists
        out_dir = os.path.dirname(csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        equity_strategy = float(cumsum)
        eq_bh = float(equity_bh) if equity_bh is not None else np.nan
        
        # Secondary activity metric (signal coverage).
        try:
            _signal_coverage = float(result.get("signal_coverage", np.nan))
        except Exception:
            _signal_coverage = np.nan

        result.update({
            "month": i + 1,
            "test_start": test_start,
            "test_end": test_end,

            # Monthly factors
            "cstrategy": perf,
            "creturns": creturns,
            "outperformance": round(perf - creturns, 6),

            "sharpe": sharpe,
            "drawdown": drawdown,
            "trades": trades,

            # Keep legacy 'cumsum' as cumulative strategy return (equity - 1)
            "cumsum": round(equity_strategy - 1.0, 6),

            # Explicit continuous equities
            "equity_strategy": round(equity_strategy, 6),
            "equity_bh": round(eq_bh, 6) if np.isfinite(eq_bh) else np.nan,
            "equity_outperformance": round(
                (equity_strategy - eq_bh) if np.isfinite(eq_bh) else np.nan, 6
            ),

            # Helpful monthly add-ons
            "strategy_return": round(perf - 1.0, 6),
            "bh_return": round(creturns - 1.0, 6),

            "directional_accuracy": directional_accuracy,
            "precision_macro": precision_macro,
            "f1_macro": f1_macro,
            
            # Trade-intent precision (post-gating, causally aligned). We try result first,
            # then fall back to evaluated df attrs (self.results) if available.
            "precision_intent": self._safe_float(
                result.get("precision_intent", float("nan")),
                fallback_key="precision_intent",
            ),
            "intent_bars": self._safe_int(
                result.get("intent_bars", 0),
                fallback_key="intent_bars",
            ),

            
            # Activity (canonical vs secondary)
            "exec_active_rate": active_rate,
            "signal_coverage": _signal_coverage,
            "profit_per_hit": profit_per_hit,
        })
        
        # --- Schema guard: ensure effective confidence threshold is present under
        # the canonical column name expected by downstream ranking.
        try:
            _ct = self._safe_float(result.get("confidence_threshold", np.nan))
            if not np.isfinite(_ct):
                _ctu = self._safe_float(result.get("confidence_threshold_used", np.nan))
                if np.isfinite(_ctu):
                    result["confidence_threshold"] = float(_ctu)
        except Exception:
            pass

        # Drop heavy config blobs from the main CSV; keep them only in sidecar dumps.
        _cfg_keys = {
            "cnn_config", "lstm_config", "transformer_config",
            "xgb_config", "rf_config", "logit_config", "dqn_config"
        }
        _row = {k: v for k, v in result.items() if k not in _cfg_keys}

        pd.DataFrame([_row]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=not os.path.exists(csv_path)
        )

        # Compact monthly line (Europe/Lisbon assumed externally)
        try:
            _ret_m  = float(perf)
            _bh_m   = float(creturns)
            _earned_s = float(equity_strategy) - 1.0
            _earned_b = float(eq_bh) - 1.0 if not np.isnan(eq_bh) else float("nan")
            _start = str(getattr(test_start, "date", lambda: test_start)())
            _end   = str(getattr(test_end, "date", lambda: test_end)())
            print(
                f"[CHART] M{i+1} {_start}->{_end} | "
                f"month_factor: Strat {_ret_m:.5f} vs BH {_bh_m:.5f} | "
                f"cum_equity: Strat {float(equity_strategy):.5f} vs BH {float(eq_bh):.5f} | "
                f"cum_pnl: Strat {_earned_s:+.2%} vs BH {_earned_b:+.2%} | "
                f"Sharpe {float(sharpe):+.2f} | Trades {int(trades)} | DD {float(drawdown):.2%}"
              )
        except Exception:
            # Fallback to legacy print if anything goes wrong
            print(
                f"\n[CHART] Month {i + 1} Results: Strat(m): {perf:.5f} | BH(m): {creturns:.5f} | "
                f"EqStrat: {equity_strategy:.5f} | EqBH: {eq_bh:.5f} | "
                f"Sharpe: {sharpe:.2f} | Trades: {trades} | DD: {drawdown:.2%}"
            )

