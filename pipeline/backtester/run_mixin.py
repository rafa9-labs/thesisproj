"""Auto-extracted mixin -- see composed.py for the full MLBacktester."""
from config import PIPELINE_CONSTANTS as _PC
from pipeline._imports import *  # noqa: F401,F403
from pipeline.printer import HPOProgress

import logging as _logging
try:
    _optuna_logger = _logging.getLogger("optuna")
    _optuna_logger.setLevel(_logging.WARNING)
except Exception:
    pass


class RunMixin:
    """
    run_strategy (HPO loop)

    Auto-extracted from MLBacktesterNoWFO.py lines 11191-15358.
    """

    def _store_fold_cv_data(self, final_score, block_scores, valid_mask, coverage, calib_brier_sum, calib_n_samples, block_cov_thr):
        """Store fold SRs and CV result on self for progress tracking."""
        try:
            dummy = block_scores  # noqa: F841 — verify in scope
        except NameError:
            self._last_fold_srs = None
            self._last_cv_result = None
            return
        try:
            fold_srs = []
            for i, sc in enumerate(block_scores):
                if i < len(valid_mask) and valid_mask[i] and np.isfinite(sc):
                    fold_srs.append(float(sc))
                else:
                    fold_srs.append(float("nan"))
            self._last_fold_srs = fold_srs
            brier = (calib_brier_sum / calib_n_samples) if calib_n_samples > 0 else float("nan")
            self._last_cv_result = {
                "sr": float(final_score) if np.isfinite(final_score) else float("nan"),
                "sr_std": float(np.nanstd(np.asarray(block_scores, dtype=float)[valid_mask])) if np.any(valid_mask) else float("nan"),
                "brier": float(brier) if np.isfinite(brier) else float("nan"),
                "coverage": float(coverage) if np.isfinite(coverage) else 0.0,
            }
        except Exception:
            self._last_fold_srs = None
            self._last_cv_result = None

    def run_strategy(self, config, models_to_test=None, n_trials=30, n_startup_trials=10): 
        """
        Run walk-forward optimization (WFO): for each split, tune with Optuna (sliding CV or mini-block CV),
        refit/evaluate on the held-out test window, and aggregate results.

        Parameters
        ----------
        config : dict
            Experiment configuration (model_type, search spaces, months, etc.).
        models_to_test : list[str] | None
            Subset of model types to consider. If None, uses `config['model_type']`.
        n_trials, n_startup_trials : int
            Optuna trial counts (forwarded via `config` to the tuner).

        Returns
        -------
        (pd.DataFrame, dict | None)
            DataFrame of fold results and the best aggregated parameter combo (or None).
        """
        
        # --- Per-run CV geometry cache (safe: only small integers, no DataFrames) ---
        # This is used inside _single_study_cv to avoid recomputing identical
        # Mini-Block geometry (k_blocks, embargo_bars, etc.) for every Optuna trial.
        # It is reset on each run_strategy call, so it cannot accumulate across runs.
        self._cv_geom_cache = {}

        # ---- Config defaults (centralized) ----
        self.apply_cv_defaults(config)
        
        cfg_f = getattr(self, "features_config", {}) or {}

        # Limit the model set (skip DQN here; it has its own path)
        if models_to_test is None:
            models_to_test = [config.get("model_type","xgboost")]
        
        # Only exclude standalone DQN here;
        models_to_test = [m for m in models_to_test if m != "dqn"]

        if not models_to_test:
            # print("[DEBUG] run_strategy called with no models to test (DQN or empty). Skipping Optuna!")
            return None, None

        log_print(f"Models to test in this WFO: {models_to_test}", level="COMPACT")

        self._progress = HPOProgress()
        try:
            self._progress.draw_header(
                model=config.get("model_type", "?"),
                pair=getattr(self, "symbol", "?"),
                timeframe=config.get("timeframe", config.get("tf", "?")),
                date_range=(str(walk_data.index[0])[:10], str(walk_data.index[-1])[:10]),
                cv_info=f"mode={config.get('cv_mode', 'mini_block')}"
            )
        except Exception:
            pass

        model_type = config.get("model_type", "svm")
        use_proba = config.get("use_proba", True)  # currently unused here, kept for parity

        full_data = self.data

        # --- Session filter (NY hours) with tz-safety ---
        walk_limit_start = full_data.index[0]
        walk_data = full_data.loc[walk_limit_start:]

        # Optional: only filter for sessions if you explicitly want to plan WFO on a reduced clock.
        if bool(config.get("wfo_session_filter", False)):
            try:
                if not hasattr(self, "_ny_mask") or self._ny_mask is None:
                    full_idx = pd.to_datetime(self.data.index, utc=True, errors="coerce")
                    _ny_times = full_idx.tz_convert("America/New_York")
                    self._ny_mask = pd.Series((_ny_times.hour >= 2) & (_ny_times.hour <= 13), index=full_idx)
                walk_data = walk_data.loc[self._ny_mask.reindex(walk_data.index, fill_value=False)]
            except Exception as e:
                log_print(f"[WARN] WFO session filter (NY) failed: {e} -- proceeding without it.", level="COMPACT")
        max_end = walk_data.index[-1]

        if self._is_debug():
            log_print(
                f"Walk data range after filtering: {walk_data.index[0]} to {walk_data.index[-1]}",
                level="DEBUG",
            )


        def ensure_list(x):
            if isinstance(x, (list, tuple)):
                return list(x)
            if x is None:
                return []
            return [x]

        # --- Respect tuned months from Optuna or defaults ---
        wfo_train = config.get("wfo_train_periods", None)
        wfo_test  = config.get("wfo_test_periods", None)
        if wfo_train is not None:
            train_months_list = ensure_list(int(wfo_train))
        else:
            train_months_list = ensure_list(
                config.get("train_months", TRAIN_TEST_MONTHS[model_type]["train"][0])
            )
        if wfo_test is not None:
            test_months_list = ensure_list(int(wfo_test))
        else:
            test_months_list = ensure_list(
                config.get("test_months", TRAIN_TEST_MONTHS[model_type]["test"][0])
            )
        period_unit = config.get("period_unit", "months")

        tasks = self.get_walk_forward_splits(
            walk_data, train_months_list, test_months_list, max_end,
            period_unit=period_unit,
        )
        if self._is_debug():
            log_print(f"Number of walk-forward splits: {len(tasks)}", level="DEBUG")

        # === ONE-TIME OPTUNA STUDY (before any parallel folds) ======================
        # Use the first fold's train window for tuning; cache Top-5 on self.
        if not tasks:
            log_print("[ERR] No WFO tasks generated.", level="COMPACT")
            return None, None

        first_start, first_train_months, first_test_months, _first_pu = tasks[0]
        from config import period_offset
        first_train_end = first_start + period_offset(first_train_months, unit=_first_pu)
        
        # IMPORTANT: training must end strictly BEFORE the first test month begins
        # to avoid boundary leakage (pandas .loc is inclusive on endpoints).
        first_test_start = first_train_end
        idx = walk_data.index

        cv_mode_req = str(config.get("cv_mode", "mini_block")).lower()
        monthly_req = cv_mode_req in {"monthly_roll", "monthly", "month", "month_roll", "rolling_month"}

        if monthly_req:
            # Ensure Optuna's tuning sample contains enough calendar history to support:
            # - rolling train_months per fold (same as real trading), plus
            # - K monthly validation blocks (default 5 months), all strictly before first_test_start.
            try:
                k_blocks = int(config.get("cv_blocks", 5))
            except Exception:
                k_blocks = 5
            try:
                val_months_eff = max(1, int(round(float(config.get("cv_val_months", 1.0)))))
            except Exception:
                val_months_eff = 1

            # Match real trading train window length (WFO uses first_train_months here)
            train_months_eff = int(first_train_months)

            need_months = train_months_eff + (k_blocks * val_months_eff)
            optuna_start = first_test_start - period_offset(int(need_months), unit=_first_pu)

            # Clamp to available data start
            if len(idx) > 0:
                optuna_start = max(optuna_start, idx[0])

            first_train_df = walk_data[(idx >= optuna_start) & (idx < first_test_start)]
        else:
            # Keep legacy behaviour for mini_block, but still end-exclusive for leakage safety
            first_train_df = walk_data[(idx >= first_start) & (idx < first_test_start)]

        
        # --- Transparency log: what exactly is the HPO tuning span? ---
        try:
            if bool(config.get("print_cv_debug", False)) or str(config.get("logmode", "")).lower() in {"compact","verbose"}:
                if len(first_train_df) > 0:
                    _hpo_s = first_train_df.index[0]
                    _hpo_e = first_train_df.index[-1]
                    _nbar  = len(first_train_df)
                    _mode  = "monthly_roll" if monthly_req else "mini_block"
                    log_print(
                        f"[HPO] mode={_mode} | tuning_span={_hpo_s} -> {_hpo_e} "
                        f"({_nbar} bars) | boundary(first_test_start)={first_test_start} (end-exclusive)"
                    , level="COMPACT")
                else:
                    log_print(f"[HPO] tuning_span is empty | boundary(first_test_start)={first_test_start} (end-exclusive)", level="COMPACT")
        except Exception:
            pass

        if len(first_train_df) < 150:
            print("[ERR] Not enough data in the first fold to run tuning.")
            return None, None

        # Base features for the tuning fold (exclude leakage/targets)
        base_features_first = [
            c for c in first_train_df.columns
            if c not in ("returns", "price", "spread", "high", "low", "label", "time")
        ]

        # Coarse windows for legacy sliding fallback (mini-block CV sizes itself)
        min_train_window_first = int(len(first_train_df) * 0.75)
        val_window_first       = max(1, int(len(first_train_df) * 0.25))
        if min_train_window_first + val_window_first > len(first_train_df):
            val_window_first = len(first_train_df) - min_train_window_first
        cv_config_first = {"min_train_window": min_train_window_first, "val_window": val_window_first}
        if isinstance(config, dict) and "_progress_callback" in config:
            cv_config_first["_progress_callback"] = config["_progress_callback"]
        _cv_jobs_raw = (os.environ.get("CV_JOBS", "") or "").strip()
        try:
            cv_config_first["cv_n_jobs"] = int(_cv_jobs_raw) if _cv_jobs_raw else (os.cpu_count() or 1)
        except (ValueError, TypeError):
            cv_config_first["cv_n_jobs"] = os.cpu_count() or 1
        cv_config_first["score_for_no_trades"] = -1.0

        # [POINT] Make CV knobs visible to the nested _single_study_cv via self.config
        #    (that function reads getattr(self, "config", {}) then apply_cv_defaults(...))
        try:
            self.config = {**getattr(self, "config", {}), **dict(cv_config_first)}
        except Exception:
            self.config = dict(cv_config_first)


        def _single_study_cv(train_data, params, min_train_window, val_window, trial=None, cv_config_override=None):
            """
            Mini-block cross-validation driver for a single Optuna trial.
            Computes the objective J over K folds.
            """
    
            import numpy as np
            
            # ============================================================
            # CV fold-row alignment (kills "random" row drift)
            # - Always append exactly ONE row per fold (OK or invalid)
            # - Overview/pruning/coverage read from fold_rows only
            # ============================================================
            fold_rows: list[dict] = []

            def _base_row(fold_id: int, val_start, val_end, train_rows=None, val_rows=None) -> dict:
                return {
                    "fold_id": int(fold_id),
                    "val_start": val_start,
                    "val_end": val_end,
                    "train_rows": int(train_rows) if train_rows is not None else 0,
                    "val_rows": int(val_rows) if val_rows is not None else 0,
                    "trades": 0,
                    "active": 0.0,
                    "sr": None,
                    "psr": None,
                    "status": "[BLOCK] UNSET",
                    "reason": "",
                }

            def _safe_float(x, default=None):
                try:
                    v = float(x)
                    return v if np.isfinite(v) else default
                except Exception:
                    return default

            def _safe_int(x, default=0):
                try:
                    return int(x)
                except Exception:
                    return default

            def _finalize_row(row: dict, *, trades=None, active=None, sr=None, psr=None, status=None, reason=None):
                if trades is not None: row["trades"] = _safe_int(trades, 0)
                if active is not None: row["active"] = _safe_float(active, 0.0) or 0.0
                if sr is not None: row["sr"] = _safe_float(sr, None)
                if psr is not None: row["psr"] = _safe_float(psr, None)
                if status is not None: row["status"] = str(status)
                if reason is not None: row["reason"] = str(reason)
                return row

            
            
            # -- Reset small per-fold CV diagnostics at the start of each trial --
            # Without this, _cv_fold_eval_frames accumulates copies of every fold
            # DataFrame across all Optuna trials, causing a slow RAM drift.
            try:
                self._cv_fold_eval_frames = []
            except AttributeError:
                # First CV call in this process: create the attribute
                self._cv_fold_eval_frames = []

            # Suppress verbose per-fold block summaries unless KODAQUANT_VERBOSE
            _block_verbose = getattr(self, "_progress", None) and self._progress.verbose
            if _block_verbose:
                _print_pruned_summary = print_pruned_block_summary
                _print_block_summary = print_block_summary
            else:
                _print_pruned_summary = lambda *a, **kw: None
                _print_block_summary = lambda *a, **kw: None

            # ---- Minimal pretty table helper ----
            def _fmt_table(headers, rows, title=None):
                col_w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)] if rows else [len(str(h)) for h in headers]
                sep = "+".join("-" * (w + 2) for w in col_w)
                def _row(cells):
                    pads = (cells + [""] * (len(col_w) - len(cells))) if len(cells) < len(col_w) else cells
                    return "|".join(" " + str(c).ljust(w) + " " for c, w in zip(pads, col_w))
                lines = []
                if title:
                    lines.append(f"\n{title}")
                lines.append(sep)
                lines.append(_row(headers))
                lines.append(sep)
                for r in rows:
                    lines.append(_row(r))
                lines.append(sep)
                return "\n".join(lines)

            # (Used only if trial is provided; harmless otherwise)
            def _quantifiable_items(d: dict):
                keep_names = {"model_type","strategy_type","feature_selection","calibrate_method",
                            "roll_windows_key","roll_windows_key_v2"}
                out = []
                for k, v in sorted(d.items()):
                    if k.startswith("_"):
                        continue
                    if isinstance(v, (int, float, bool)):
                        out.append((k, v))
                    elif isinstance(v, str) and (k in keep_names or len(v) <= 16):
                        out.append((k, v))
                return out

            def _print_trial_header_table(params, cfg, trial):
                pass  # Suppressed: was dumping 50+ params per trial, creating noise
                    
            _prev_cv  = getattr(self, "_in_cv", False)
            _prev_dbg = getattr(self, "_dbg_first_bars", False)
            self._in_cv = True
            self._dbg_first_bars = False
            _old_cv_flag = getattr(self, "_in_optuna_cv", False)
            
            # --- Calibration accumulators across folds ---
            calib_brier_sum = 0.0
            calib_nll_sum   = 0.0
            calib_n_samples = 0
            
            setattr(self, "_in_optuna_cv", True)
            for _k in ("_deep_temp_T", "_coverage_conf_thr"):
                if hasattr(self, _k):
                    try: delattr(self, _k)
                    except Exception: pass
                    
            # CV memory control: feature-slice caching is bypassed in CV mode, but
            # any previously cached engineered frames (from non-CV runs) can still
            # linger on the instance and bloat RAM. Clear them eagerly at CV entry.
            try:
                self._clear_feature_cache()
            except Exception:
                pass

            # Pull config safely and apply CV defaults if available
            config = getattr(self, "config", {}) if hasattr(self, "config") else {}
            try:
                config = self.apply_cv_defaults(dict(config))
            except Exception:
                config = dict(config)
                
            # without relying on evaluate_cv_func being a bound method.
            if cv_config_override:
                try:
                    config.update(dict(cv_config_override))
                except Exception:
                    pass

            # -------------------------------
            # CV config normalization (single source of truth)
            # -------------------------------
            cv_config = dict(config.get("cv_config", {}) or {})
            for _k in (
                "cv_prune_precision_intent",
                "cv_prune_min_precision_intent",
                "cv_prune_min_intent_bars_fold",
                "cv_prune_min_intent_bars",
            ):
                if _k in config:
                    try:
                        cv_config[_k] = config.get(_k)
                    except Exception:
                        pass


            # CV table behavior (single source of truth)
            table_mode          = str(config.get("cv_table_mode", "compact")).lower()   # "compact" | "verbose" | "full" | "off"
            table_verbose       = bool(config.get("cv_table_verbose", False)) or (table_mode in {"verbose","full"})
            table_only_failures = bool(config.get("cv_table_only_failures", False))

                        # Global pruning relaxation knob
            cv_relax = float(config.get("cv_prune_relax", 1.0))
            cv_relax = max(0.0, min(cv_relax, 1.0))

            # Base gates
            _M_gate_base = int(config.get("cv_min_trades_per_block", 5))
            _r_min_base  = float(config.get("cv_gate_min_active_rate", 0.02))
            _L_gate_base = int(config.get("cv_gate_min_folds", 3))

            # Effective gates after relaxation
            if cv_relax <= 0.0:
                # Fully relaxed: do not gate/prune via these thresholds
                _M_gate_eff = 0
                _r_min_eff  = 0.0
                _L_gate_eff = 1
            else:
                # Larger base thresholds = stricter -> scale down by cv_relax
                _M_gate_eff = max(1, int(round(_M_gate_base * cv_relax)))
                _r_min_eff  = max(0.0, _r_min_base * cv_relax)
                _L_gate_eff = max(1, int(round(_L_gate_base * cv_relax)))


            # Unified penalty logger
            def _cv_penalty(reason: str, **kv):
                if bool(config.get("print_cv_debug", False)):
                    extras = " | ".join(f"{k}={v}" for k, v in kv.items())
                    log_print(
                        f"[CV-PENALTY] {reason}" + (f" | {extras}" if extras else ""),
                        level="DEBUG",
                    )


            if bool(config.get("print_cv_debug", False)):
                print(f"[CV] mode={str(config.get('cv_mode','mini_block')).lower()} | model={params.get('model_type','?')}")

            setattr(self, "_in_optuna_cv", True)
            try:
                total_len = len(train_data)

                # Sliding stride hints (only for legacy path; we keep mini-block)
                target_folds = int(config.get("cv_target_folds", 5))
                cv_val_months = float(config.get("cv_val_months", 1.0))
                bars_per_month_hint = int(config.get("bars_per_month_hint", 1000))
                if val_window is None or val_window <= 0:
                    val_window = max(1, int(round(cv_val_months * bars_per_month_hint)))

                embargo = int(config.get("cv_embargo_bars", 0))
                avail = max(0, int(min_train_window) - int(val_window) - embargo)
                forced_stride_frac = config.get("cv_sliding_stride_frac", None)
                if forced_stride_frac is not None:
                    step = max(1, int(round(val_window * float(forced_stride_frac))))
                else:
                    denom = max(1, target_folds - 1)
                    step = max(1, int(avail // denom)) if avail > 0 else max(1, int(val_window // 2))
                step = max(1, min(step, max(1, int(val_window))))

                if min_train_window + val_window > total_len:
                    _cv_penalty("Insufficient data for requested CV window",
                                min_train_window=min_train_window, val_window=val_window, total_len=total_len)
                    return float("nan")

                # Merge per-trial feature config
                self.features_config.update(params)
                cfg = self.apply_feature_defaults()
                
                self._optuna_locked_keys = set(params.keys())
                

                self.features_config = cfg
                # Tripwire: warn if CV/time-caps changed Optuna keys
                if getattr(self, "_optuna_locked_keys", None):
                    _cl = {k for k in self._optuna_locked_keys
                        if k in self.features_config and self.features_config[k] != params.get(k)}
                    if _cl:
                        print(f"[WARN] Optuna keys were changed by CV/time-caps: {sorted(_cl)}")

                

                lags     = int(params.get("lags", params.get("lags_range", 5)))
                conf_thr = float(params.get("confidence_threshold", 0.0))
                model_type_local = params["model_type"]
                # cv_mode_local = str(config.get("cv_mode", "mini_block")).lower()
                
                #  # Month-aligned CV request (Patch M1 builds fold boundaries; Patch M2 will wire
                # # them into the training/validation slices). For now, we *prepare* the fold plan
                # # and keep evaluation on mini_block so behaviour stays stable until M2 lands.
                # monthly_roll_requested = cv_mode_local in {
                #     "monthly_roll", "monthly", "month", "month_roll", "rolling_month"
                # }
                # cv_mode_effective = "mini_block" if monthly_roll_requested else cv_mode_local
                
                # # Accept a future month-aligned CV mode name without breaking older runs.
                # # (Patch M1/M2 will implement this fully; for now we fall back to mini_block.)
                # if cv_mode_local in {"monthly_roll", "monthly", "month", "month_roll", "rolling_month"}:
                #     if bool(config.get("print_cv_debug", False)):
                #         print("[CV] monthly_roll requested but not yet implemented; falling back to mini_block.")
                #     cv_mode_local = "mini_block"
                cv_mode_local = str(config.get("cv_mode", "mini_block")).lower()

                # Month-aligned CV request ("monthly_roll" and aliases).
                # Implemented via month-aligned folds inside the CV loop (see _use_monthly below).
                monthly_roll_requested = cv_mode_local in {
                    "monthly_roll", "monthly", "month", "month_roll", "rolling_month"
                }

                # Normalize aliases so logs and downstream checks are unambiguous.
                cv_mode_effective = "monthly_roll" if monthly_roll_requested else cv_mode_local

                is_dqn_like = model_type_local in {"dqn"}

                # Header (if Optuna)
                try:
                    _print_trial_header_table(params, cfg, trial)
                    if bool(config.get("print_cv_debug", False)):
                        if monthly_roll_requested:
                            print(
                                f"[SEARCH] CV geometry: mode={cv_mode_effective} "
                                f"| cv_blocks={int(config.get('cv_blocks', 5))} "
                                f"| val_months={float(config.get('cv_val_months', 1.0)):.2f} "
                                f"| cv_train_months={config.get('cv_train_months', None)}"
                        )
                        else:
                            print(
                                f"[SEARCH] CV geometry: mode={cv_mode_effective} | K={int(config.get('cv_blocks', 5))} "
                                f"| embargo={int(config.get('cv_embargo_bars', 0))} "
                                f"| val_frac={float(config.get('cv_val_frac', 0.09)):.3f} "
                                f"| min_train_frac={float(config.get('cv_min_train_frac', 0.80)):.3f}"
                            )
                        print(f"[LOCK] Confidence threshold (requested): {conf_thr:.3f} "
                            f"| backoff_cv={bool(cfg.get('allow_conf_backoff_cv', False))} "
                            f"| floor_cv={float(cfg.get('conf_backoff_floor_cv', 0.33)):.2f}")
                except Exception:
                    pass
                

                # --- Month-aligned fold plan (prepared only; will be used in Patch M2) ---
                if monthly_roll_requested and not is_dqn_like:
                    def _safe_int_months(v, default=1):
                        try:
                            if v is None:
                                return int(default)
                            if isinstance(v, (list, tuple)):
                                v = v[0]
                            return max(1, int(round(float(v))))
                        except Exception:
                            return int(default)

                    def _build_monthly_roll_folds(df, k_blocks, train_months_eff, val_months_eff, embargo_bars_eff):
                        # "\"\"Return a fold plan as iloc ranges that align to calendar months.\"\"\"
                        if df is None or len(df) < 10:
                            return []
                        idx = df.index
                        if not hasattr(idx, "to_period"):
                            return []
                        from config import to_period_freq as _tpf
                        _freq = _tpf(period_unit)
                        months = pd.Index(idx.to_period(_freq)).unique().sort_values()
                        if len(months) == 0:
                            return []
                        k_use = min(int(k_blocks), len(months))
                        val_months = months[-k_use:]
                        folds = []
                        idx_values = idx.values
                        for j, m in enumerate(val_months, start=1):
                            val_start = m.start_time
                            val_next  = (m + val_months_eff).start_time
                            vs = int(np.searchsorted(idx_values, np.array(val_start, dtype=idx_values.dtype), side="left"))
                            ve = int(np.searchsorted(idx_values, np.array(val_next,  dtype=idx_values.dtype), side="left"))
                            if ve <= vs:
                                continue
                            tr_end = max(0, vs - int(embargo_bars_eff))
                            if tr_end <= 0:
                                continue
                            tr_end_time = idx[tr_end - 1]
                            tr_start_time = tr_end_time - period_offset(int(train_months_eff), unit=period_unit)
                            ts = int(np.searchsorted(idx_values, np.array(tr_start_time, dtype=idx_values.dtype), side="left"))
                            ts = max(0, min(ts, tr_end))
                            folds.append({
                                "fold": j,
                                "train_iloc": (ts, tr_end),
                                "val_iloc": (vs, ve),
                                "train_start": idx[ts] if ts < len(idx) else None,
                                "train_end": idx[tr_end - 1] if tr_end - 1 < len(idx) else None,
                                "val_start": idx[vs] if vs < len(idx) else None,
                                "val_end": idx[ve - 1] if ve - 1 < len(idx) else None,
                                "val_month": str(m),
                            })
                        return folds

                    k_blocks_cfg = int(config.get("cv_blocks", 5))
                    val_months_eff = _safe_int_months(config.get("cv_val_months", 1.0), default=1)
                    cv_train_months_cfg = config.get("cv_train_months", None)
                    if cv_train_months_cfg is None:
                        cv_train_months_cfg = config.get("train_months", None)
                    if isinstance(cv_train_months_cfg, (list, tuple)):
                        cv_train_months_cfg = cv_train_months_cfg[0]
                    if cv_train_months_cfg is None:
                        try:
                            cv_train_months_cfg = TRAIN_TEST_MONTHS[model_type_local]["train"][0]
                        except Exception:
                            cv_train_months_cfg = 12
                    train_months_eff = _safe_int_months(cv_train_months_cfg, default=12)
                    tb_max_holding_local = int(self.features_config.get("tb_max_holding", int(config.get("tb_max_holding", 0))))
                    embargo_bars_eff = max(int(config.get("cv_embargo_bars", 0)), tb_max_holding_local)
                    monthly_folds = _build_monthly_roll_folds(train_data, k_blocks_cfg, train_months_eff, val_months_eff, embargo_bars_eff)
                    setattr(self, "_cv_monthly_fold_plan", monthly_folds)
                    if bool(config.get("print_cv_debug", False)) and monthly_folds:
                        rows = []
                        for f in monthly_folds:
                            ts, te = f["train_iloc"]; vs, ve = f["val_iloc"]
                            rows.append([f["fold"], f["val_month"], f"{ts}:{te} ({max(0, te-ts)} bars)", f"{vs}:{ve} ({max(0, ve-vs)} bars)"])
                        print(_fmt_table(["Fold","ValMonth","Train(iloc)","Val(iloc)"], rows,
                                         title="[DATE]  Monthly-roll CV fold plan"))

                # -------------------------------
                # Mini-block CV (preferred path)
                # -------------------------------
                if (cv_mode_effective in {"mini_block", "mini", "monthly_roll"}) and not is_dqn_like:

                    # ---- Tiny CV geometry cache (per-run, per-geometry) ----
                    # We only cache small integers (no DataFrames) so this cannot
                    # blow up RAM. Keyed by geometry-only knobs: data length,
                    # cv_blocks, val/min_train fractions and embargo.
                    geom_cache = getattr(self, "_cv_geom_cache", None)
                    if geom_cache is None:
                        geom_cache = {}
                        setattr(self, "_cv_geom_cache", geom_cache)

                    total_len = len(train_data)

                    # Base fractional geometry (depends only on config + total_len)
                    val_frac  = float(config.get("cv_val_frac", 0.09))
                    min_frac  = float(config.get("cv_min_train_frac", 0.80))
                    # A: Adaptive K — data-aware block count [3, 10], overridable via cv_blocks
                    _adaptive_k = max(3, min(10, total_len // 4000))
                    k_blocks_cfg = int(config.get("cv_blocks", _adaptive_k))

                    # Stable identifiers for this train_data (month)
                    try:
                        idx0 = train_data.index[0]
                        idx1 = train_data.index[-1]
                    except Exception:
                        idx0 = ("len", total_len)
                        idx1 = None

                    cv_key = (
                        "mini_block_geom",
                        k_blocks_cfg,
                        val_frac,
                        min_frac,
                        int(config.get("cv_embargo_bars", 0)),
                        int(getattr(self, "features_config", {}).get(
                            "tb_max_holding",
                            int(config.get("tb_max_holding", 0)),
                        )),
                        int(total_len),
                        str(idx0),
                        str(idx1),
                    )

                    cached = geom_cache.get(cv_key)
                    if cached is not None:
                        # Fast path: reuse integers from previous trial
                        (
                            k_blocks,
                            tb_max_holding_local,
                            embargo_bars,
                            val_window_local,
                            min_train_local,
                            smin,
                            smax,
                        ) = cached
                    else:
                        # Slow path: compute geometry as before (only once per run)
                        k_blocks = k_blocks_cfg
                        tb_max_holding_local = int(
                            self.features_config.get(
                                "tb_max_holding",
                                int(config.get("tb_max_holding", 0)),
                            )
                        )
                        embargo_bars = max(
                            int(config.get("cv_embargo_bars", 0)),
                            tb_max_holding_local,
                        )

                        # B: soft floor — min(120, 3% of data) for statistical significance
                        _soft_floor = min(120, max(30, int(round(0.03 * total_len))))
                        val_window_local = max(
                            _soft_floor,
                            int(round(val_frac * total_len)),
                        )
                        min_train_local = int(round(min_frac * total_len))

                        smin = max(0, min_train_local + int(embargo_bars))
                        smax = total_len - val_window_local

                        # Cap K to what geometry actually supports (before shrink loop refines further)
                        _fit_usable = max(1, int(smax) - int(smin))
                        _max_fit_k = max(1, int(_fit_usable // max(1, val_window_local)) + 1)
                        k_blocks = min(k_blocks, _max_fit_k)

                        # Store only small ints; no DataFrames are cached.
                        geom_cache[cv_key] = (
                            int(k_blocks),
                            int(tb_max_holding_local),
                            int(embargo_bars),
                            int(val_window_local),
                            int(min_train_local),
                            int(smin),
                            int(smax),
                        )

                    if smax <= smin:
                        _cv_penalty("MiniBlockCV invalid split geometry", smax=smax, smin=smin)
                        import optuna as _opt
                        raise _opt.TrialPruned("Broken CV geometry: cannot form blocks (no valid bars)")


                    # Exact-fit mode to preserve requested K blocks
                    exact = bool(config.get("cv_fit_blocks_exact", True))
                    if exact:
                        usable = (smax - smin)
                        if usable <= 0:
                            _cv_penalty("MiniBlockCV invalid geometry (usable<=0)", usable=usable, smax=smax, smin=smin)
                            import optuna as _opt
                            raise _opt.TrialPruned("Broken CV geometry: usable<=0")
                        needed = int(max(0, (k_blocks - 1)) * val_window_local)
                        if usable < needed:
                            new_val = max(30, usable // max(1, k_blocks))
                            if new_val < 30:
                                _cv_penalty("MiniBlockCV cannot shrink below 30 bars", new_val=new_val, K=k_blocks, usable=usable)
                                import optuna as _opt
                                raise _opt.TrialPruned("Broken CV geometry: val_window < 30")
                            if bool(config.get("print_cv_debug", False)):
                                print(f"[MiniBlockCV] Shrinking val_window from {val_window_local} -> {new_val} to keep K={k_blocks}")
                            val_window_local = new_val
                            smax = total_len - val_window_local

                    # Build split starts (tail-anchored optional)
                    K = int(k_blocks)
                    while K > 1 and (smax - smin) < ((K - 1) * val_window_local):
                        K -= 1
                    if K < 1:
                        for k_try in range(min(k_blocks, 5), 1, -1):
                            K_try = (smax - smin) - (k_try - 1) * val_window_local
                            K_try = K_try // max(30, val_window_local)
                            if K_try >= 1:
                                k_blocks = k_try
                                K = K_try
                                break
                    if K < 1:
                        K = 1
                        k_blocks = 2
                        val_window_local = max(val_window_local, (smax - smin - embargo_bars) // 2)

                    tail_anchor = bool(config.get("cv_tail_anchor", True))
                    if (not tail_anchor) or (K == 0):
                        slack = (smax - smin) - (K * val_window_local)
                        gap   = 0 if K == 1 else max(0, slack // (K - 1))
                        splits = []
                        cursor = smin
                        for _ in range(K):
                            splits.append(cursor)
                            cursor += val_window_local + gap
                    else:
                        if K == 1:
                            splits = [smax]
                        else:
                            early = K - 1
                            avail_early = (smax - smin) - (early * val_window_local)
                            gap_early = 0 if early == 1 else max(0, avail_early // (early - 1))
                            splits = []
                            cursor = smin
                            for _ in range(early):
                                splits.append(cursor)
                                cursor += val_window_local + gap_early
                            splits.append(smax)

                    # Early-stopping knobs for deep models (treat all deep + ensemble as deep)
                    is_ensemble = isinstance(model_type_local, str) and model_type_local.startswith("ensemble_")
                    is_deep = (model_type_local in {"cnn","lstm","transformer","gru","gru_lstm"}) or is_ensemble

                    # =========================
                    # CV pre-setup (deep caps)
                    # =========================
                    if is_deep:
                        cfg = self.apply_feature_defaults()
                        # Remember which keys Optuna actually sampled this trial
                        try:
                            self._optuna_locked_keys = set(params.keys())
                        except Exception:
                            self._optuna_locked_keys = set()

                        cfg["deep_eval_mode"] = "cv_fast"
                        cfg.setdefault("deep_cv_batch_size", 256)

                        # Per-family CV caps — tuned to convergence speed of each arch
                        if model_type_local == "transformer":
                            cfg.setdefault("deep_cv_max_epochs", 6)
                            cfg.setdefault("deep_cv_patience", 4)
                            cfg.setdefault("transformer_use_early_stopping", True)
                            cfg.setdefault("transformer_patience", 4)
                        elif model_type_local in ("gru", "gru_lstm"):
                            cfg.setdefault("deep_cv_max_epochs", 12)
                            cfg.setdefault("deep_cv_patience", 8)
                            cfg.setdefault("gru_use_early_stopping", True)
                            cfg.setdefault("gru_lstm_use_early_stopping", True)
                            cfg.setdefault(f"{model_type_local}_patience", 8)
                        elif model_type_local == "cnn":
                            cfg.setdefault("deep_cv_max_epochs", 10)
                            cfg.setdefault("deep_cv_patience", 6)
                            cfg.setdefault("cnn_use_early_stopping", True)
                            cfg.setdefault("cnn_patience", 6)
                        elif model_type_local == "lstm":
                            cfg.setdefault("deep_cv_max_epochs", 10)
                            cfg.setdefault("deep_cv_patience", 7)
                            cfg.setdefault("lstm_use_early_stopping", True)
                            cfg.setdefault("lstm_patience", 7)
                        else:
                            # ensemble / other: keep current defaults
                            cfg.setdefault("deep_cv_max_epochs", 8)
                            cfg.setdefault("deep_cv_patience", 6)
                            cfg.setdefault("cnn_use_early_stopping", True)
                            cfg.setdefault("lstm_use_early_stopping", True)
                            cfg.setdefault("transformer_use_early_stopping", True)
                            cfg.setdefault("gru_use_early_stopping", True)
                            cfg.setdefault("gru_lstm_use_early_stopping", True)
                            cv_pat = int(cfg.get("deep_cv_patience", 5))
                            cfg.setdefault("cnn_patience", cv_pat)
                            cfg.setdefault("lstm_patience", cv_pat)
                            cfg.setdefault("transformer_patience", cv_pat)
                            cfg.setdefault("gru_patience", cv_pat)
                            cfg.setdefault("gru_lstm_patience", cv_pat)

                        cfg["skip_perm_importance"] = True
                        self.features_config = cfg

                        # Print if anything Optuna chose got clobbered by CV caps
                        if getattr(self, "_optuna_locked_keys", None):
                            clobbered = {
                                k for k in self._optuna_locked_keys
                                if k in self.features_config and self.features_config[k] != params.get(k)
                            }
                            if clobbered:
                                print(f"[WARN] Optuna keys were changed by CV/time-caps: {sorted(clobbered)}")

                    # Timestamps for split starts (debug)
                    val_starts_ts = []
                    try:
                        for i in splits:
                            if 0 <= int(i) < len(train_data):
                                val_starts_ts.append(train_data.index[int(i)])
                    except Exception:
                        val_starts_ts = []

                    if bool(config.get("print_cv_debug", False)):
                        try:
                            _spl = list(map(int, splits))
                        except Exception:
                            _spl = splits
                        print(f"[MiniBlockCV] Using k={K} blocks | val_window={val_window_local} rows | "
                            f"embargo_bars={embargo_bars} | min_train_local={min_train_local} | "
                            f"splits(starts)={_spl} | lags={lags} | deep_fast_cv={is_deep}")
                        try:
                            _ts = [str(t) for t in val_starts_ts]
                            print(f"[MiniBlockCV] val_starts_ts={_ts}")
                        except Exception:
                            pass

                    # ------------------------------------------------------
                    # helpers for status + dict fmt (used by table printing)
                    # ------------------------------------------------------
                    # Read the generic CV gates once for the status helper
                    _M_gate = int(config.get("cv_min_trades_per_block", 5))
                    _r_min  = float(config.get("cv_gate_min_active_rate", 0.02))
                    
                    _relax = float(config.get("cv_prune_relax", 1.0))
                    _relax = max(0.0, min(_relax, 1.0))
                    
                    if _relax <= 0.0:
                        # Disable Thin-gating: we still flag blatant errors (NoTrades, NaN, etc.)
                        _M_gate_eff = 0
                        _r_min_eff = 0.0
                    else:
                        # Larger thresholds = stricter => scale down to relax
                        _M_gate_eff = max(0, int(round(_M_gate * _relax)))
                        _r_min_eff  = max(0.0, _r_min * _relax)

                    def _status_for_block(trades, sr, active, all_hold=False, pruned=False):
                        if pruned:
                            return "[PRUNE] prune"
                        if (trades is None) or (int(trades) <= 0) or all_hold:
                            return "[BLOCK] NoTrades"
                        if (sr is None) or (not np.isfinite(sr)):
                            return "[BLOCK] SRNaN"
                        if (int(trades) < _M_gate_eff) or (
                            active is not None and np.isfinite(active) and float(active) < _r_min_eff
                        ):
                            return "[WARN] Thin"
                        if float(sr) < 0:
                            return "[BAD] Bad"
                        return "[OK] OK"

                    def _fmt_dict(d):
                        try:
                            if isinstance(d, dict):
                                return "{" + ", ".join(f"{k}: {d[k]}" for k in sorted(d)) + "}"
                        except Exception:
                            pass
                        return str(d)
                    
                    def _early_structural_prune_if_hopeless():
                        """
                        Early-stop a trial during MiniBlockCV when:
                        (A) The first few folds are all invalid / no-trades (degenerate config).
                        (B) Even in the best case, we cannot hit the required active-fold coverage.

                        (B) is aligned with the final coverage gate:
                        we only trigger when this trial would be hopeless anyway under the
                        cv_min_coverage target. (A) is a pragmatic speed-up for configs that
                        clearly produce 0 trades everywhere.
                        """
                        # Need Optuna + numpy; otherwise do nothing.
                        try:
                            import numpy as _np
                            import optuna as _opt
                        except Exception:
                            return

                        # How many folds are planned
                        try:
                            K_plan = len(splits)
                        except Exception:
                            K_plan = int(config.get("cv_blocks", 5)) or 1
                        if K_plan <= 1:
                            return

                        processed = len(block_scores)
                        if processed <= 0:
                            return

                        arr = _np.asarray(block_scores[:processed], dtype=float)
                        k_valid = int(_np.isfinite(arr).sum())
                        remaining = max(0, K_plan - processed)

                        # -- (A) Degenerate no-trades heuristic -----------------------------
                        # If we've already evaluated N folds and NONE produced a valid score,
                        # this config is effectively dead: thresholds/gating too strict.
                        # Default N=2; can be tuned via cv_early_all_invalid_patience.
                        patience = int(config.get("cv_early_all_invalid_patience", 2))
                        if processed >= patience and k_valid == 0:
                            msg = (f"[MiniBlockCV:EARLY_DEGENERATE] "
                                   f"{processed} folds, all invalid/no-trades -> prune trial")
                            if bool(config.get("print_cv_debug", False)):
                                print(msg)
                            raise _opt.TrialPruned(msg)

                        # -- (B) Structural coverage hopelessness ---------------------------
                        # Use cv_min_coverage as the design target. We only cut when even if
                        # all remaining folds were perfect, we cannot reach that coverage.
                        min_cov_base = float(config.get("cv_min_coverage", 0.80))
                        min_cov_base = max(0.0, min(1.0, min_cov_base))

                        # For early-hopeless logic we *do not* weaken this with cv_prune_relax.
                        # If you want it tied to relax, replace min_cov_base with
                        # (min_cov_base * cv_relax).
                        max_possible_valid = k_valid + remaining
                        required_valid = min_cov_base * K_plan

                        if max_possible_valid < required_valid:
                            msg = (f"[MiniBlockCV:EARLY_COVERAGE_PRUNE] "
                                   f"k_valid={k_valid}, remaining={remaining}, "
                                   f"K_plan={K_plan}, min_cov={min_cov_base:.2f}")
                            if bool(config.get("print_cv_debug", False)):
                                print(msg)
                            raise _opt.TrialPruned(msg)

                        # -- (C) Early-abort hopeless Sharpe ----------------------------------
                        # After cv_early_sharpe_folds (default 3) valid folds, if the mean
                        # Sharpe is below the threshold (default -1.0), prune the trial.
                        # No amount of additional folds will salvage a config this bad.
                        _early_sharpe_folds = int(config.get("cv_early_sharpe_folds", 3))
                        _early_sharpe_thr   = float(config.get("cv_early_sharpe_threshold", -1.0))
                        if (_relax > 0.0 and processed >= _early_sharpe_folds
                                and k_valid >= _early_sharpe_folds):
                            _valid_scores = arr[_np.isfinite(arr)]
                            if len(_valid_scores) >= _early_sharpe_folds:
                                _mean_sharpe = float(_np.mean(_valid_scores))
                                if _mean_sharpe < _early_sharpe_thr * _relax:
                                    msg = (f"[MiniBlockCV:EARLY_HOPELESS] "
                                           f"mean Sharpe={_mean_sharpe:.3f} "
                                           f"< {_early_sharpe_thr:.0f} "
                                           f"after {_early_sharpe_folds} folds -> prune trial")
                                    if bool(config.get("print_cv_debug", False)):
                                        print(msg)
                                    raise _opt.TrialPruned(msg)


                    # --------------------------------------------
                    # Collectors BEFORE the mini-block evaluation
                    # --------------------------------------------
                    block_scores, block_active_rates, block_trades = [], [], []
                    block_eff_conf, block_rows, block_pruned, block_all_hold = [], [], [], []
                    block_train_rows = []      # NEW: train rows per fold (for tables)
                    block_precision_intent = []
                    block_intent_bars = []


                    block_reasons = []          # human-readable gating reason per block
                    pred_cards, val_ends_ts = [], []
                    val_starts_ts_cv = []
                    val_ends_ts_cv = []
                    
                    block_psr, block_neff = [], []
                    block_sharpe = []           # raw after-cost Sharpe per block
                    block_cov_thr = []          # per-block coverage thresholds (base)
                    
                    # Per-regime accumulators (0=SIDEWAYS,1=TREND,2=VOLATILE)
                    regime_stats = {
                        0: {"sum_ret": 0.0, "sum_ret_sq": 0.0, "trades": 0, "bars": 0},
                        1: {"sum_ret": 0.0, "sum_ret_sq": 0.0, "trades": 0, "bars": 0},
                        2: {"sum_ret": 0.0, "sum_ret_sq": 0.0, "trades": 0, "bars": 0},
                    }
                    
                    # NEW: per-fold eval frame collector for CV diagnostics
                    fold_eval_frames: list = []        # each fold's evaluation DataFrame (if available)
                    per_fold_regime_trades = {0: [], 1: [], 2: []}
                    per_fold_regime_active = {0: [], 1: [], 2: []}
                    per_fold_regime_sharpe = {0: [], 1: [], 2: []}

                    # -------------------------
                    # Evaluate each mini-block
                    # -------------------------
                    # If monthly-roll CV was requested and a fold plan exists (prepared in Patch M1),
                    # use it. Otherwise, default to the existing expanding-window mini-block logic.
                    _cv_mode_req = str(config.get("cv_mode", "mini_block")).lower()
                    _monthly_req = _cv_mode_req in {"monthly_roll","monthly","month","month_roll","rolling_month"}
                    monthly_folds = getattr(self, "_cv_monthly_fold_plan", None) if _monthly_req else None
                    _use_monthly = bool(monthly_folds) and isinstance(monthly_folds, list)
                    
                    # --- Guardrails for monthly fold geometry (Patch M3) ---
                    if _use_monthly:
                        bpm = int(config.get("bars_per_month_hint", 1000))
                        val_m = float(config.get("cv_val_months", 1.0))
                        exp_val_bars = max(10, int(round(bpm * val_m)))

                        # Defaults chosen to be conservative and stable:
                        # - train must be at least ~3 months worth of bars
                        # - val must be at least 60% of expected month bars (handles missing days / session filters)
                        min_train_months = float(config.get("cv_min_train_months_monthly", 3.0))
                        min_val_frac     = float(config.get("cv_min_val_frac_monthly", 0.60))
                        min_valid_folds  = int(config.get("cv_min_valid_folds_monthly", 3))

                        min_train_bars = max(50, int(round(bpm * min_train_months)))
                        min_val_bars   = max(10, int(round(exp_val_bars * min_val_frac)))

                        filtered = []
                        dropped_rows = []
                        for f in monthly_folds:
                            try:
                                ts, te = f.get("train_iloc", (0, 0))
                                vs, ve = f.get("val_iloc", (0, 0))
                                ts, te, vs, ve = int(ts), int(te), int(vs), int(ve)
                            except Exception:
                                dropped_rows.append([str(f.get("fold","?")), str(f.get("val_month","?")), "bad iloc parse"])
                                continue

                            if not (0 <= ts < te <= len(train_data) and 0 <= vs < ve <= len(train_data)):
                                dropped_rows.append([str(f.get("fold","?")), str(f.get("val_month","?")), "iloc out of range"])
                                continue

                            train_n = te - ts
                            val_n   = ve - vs

                            if train_n < min_train_bars:
                                dropped_rows.append([str(f.get("fold","?")), str(f.get("val_month","?")), f"train too short ({train_n} < {min_train_bars})"])
                                continue
                            if val_n < min_val_bars:
                                dropped_rows.append([str(f.get("fold","?")), str(f.get("val_month","?")), f"val too short ({val_n} < {min_val_bars})"])
                                continue

                            filtered.append(f)

                        if bool(config.get("print_cv_debug", False)) and dropped_rows:
                            print(_fmt_table(["Fold","ValMonth","DroppedReason"], dropped_rows,
                                             title="[WARN] Monthly-roll CV: dropped folds (guardrails)"))

                        if len(filtered) >= min_valid_folds:
                            monthly_folds = filtered
                            _use_monthly = True
                        else:
                            if bool(config.get("print_cv_debug", False)):
                                print(f"[CV] Monthly-roll folds valid={len(filtered)} < {min_valid_folds}; falling back to mini_block for this trial.")
                            _use_monthly = False


                    if _use_monthly and bool(config.get("print_cv_debug", False)):
                        print(f"[CV] Using monthly-roll fold slicing for this trial (K={len(monthly_folds)})")
                        
                   # Remember what we actually used this trial so the summary table title matches reality
                    try:
                        setattr(self, "_cv_used_monthly_last", bool(_use_monthly))
                    except Exception:
                        pass

                    fold_iter = monthly_folds if _use_monthly else splits
                    fold_label = "CV Fold" if _use_monthly else "Mini-Block Fold"

                    # (B1) Stable per-fold record slots (prevents index drift when folds are pruned/skipped).
                    # NOTE: We keep all existing parallel arrays for compatibility in this patch.
                    fold_records = [None] * int(len(fold_iter))
                    if bool(config.get("print_cv_debug", False)):
                        print(f"[CV][FoldRecords] init slots={len(fold_records)}")
                        
                        
                    # ------------------------------------------------------------------
                    # Single source of truth: normalize cv_config ONCE, before fold loop.
                    # Baseline: nested config["cv_config"]; allow a few flat keys in
                    # `config` to override for backward compatibility.
                    # ------------------------------------------------------------------
                    cv_config = dict(config.get("cv_config", {}) or {})
                    for _k in (
                        "cv_prune_precision_intent",
                        "cv_prune_min_precision_intent",
                        "cv_prune_min_intent_bars_fold",
                        "cv_prune_min_intent_bars",
                    ):
                        if _k in config:
                            cv_config[_k] = config.get(_k)
                            
                    # ----------------------------------------------------------------
                    #   Snapshot/restore all _last_* and _cv_last_* attrs + results fields
                    #   around any diagnostic pass, and scream if the fold guard changes.
                    # ------------------------------------------------------------------
                    def _cv__clone_state(v):
                        try:
                            import pandas as _pd
                            import numpy as _np
                        except Exception:
                            _pd = None
                            _np = None
                        try:
                            from copy import deepcopy as _deepcopy
                        except Exception:
                            _deepcopy = None
                        try:
                            if _pd is not None and isinstance(v, _pd.DataFrame):
                                return v.copy(deep=True)
                            if _pd is not None and isinstance(v, _pd.Series):
                                return v.copy(deep=True)
                            if _np is not None and isinstance(v, _np.ndarray):
                                return v.copy()
                            if isinstance(v, (dict, list, tuple)) and _deepcopy is not None:
                                return _deepcopy(v)
                        except Exception:
                            pass
                        return v

                    def _cv__snapshot_state_for_diagnostics():
                        keys = {"results", "results_full", "_cv_last_eval_df", "_last_eligibility_diag"}
                        try:
                            for k in list(getattr(self, "__dict__", {}).keys()):
                                if k.startswith("_last_") or k.startswith("_cv_last_"):
                                    keys.add(k)
                        except Exception:
                            pass
                        snap, present = {}, {}
                        try:
                            d = getattr(self, "__dict__", {})
                            for k in keys:
                                present[k] = (k in d)
                                if present[k]:
                                    snap[k] = _cv__clone_state(d.get(k))
                        except Exception:
                            pass
                        return snap, present

                    def _cv__restore_state_after_diagnostics(snap, present):
                        for k, was_present in (present or {}).items():
                            if was_present:
                                try:
                                    setattr(self, k, snap.get(k))
                                except Exception:
                                    pass
                            else:
                                if hasattr(self, k):
                                    try:
                                        delattr(self, k)
                                    except Exception:
                                        pass
                        try:
                            d_now = getattr(self, "__dict__", {})
                            for k in list(d_now.keys()):
                                if (k.startswith("_last_") or k.startswith("_cv_last_")) and (k not in (present or {})):
                                    try:
                                        delattr(self, k)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    @contextmanager
                    def _cv_diagnostic_guard(ctx="cv:diagnostic"):
                        snap, present = _cv__snapshot_state_for_diagnostics()
                        g_uuid   = getattr(self, "_cv_fold_guard_uuid", None)
                        g_trades = getattr(self, "_cv_fold_guard_trades_main", None)
                        try:
                            yield
                        finally:
                            try:
                                a_uuid   = getattr(self, "_cv_fold_guard_uuid", None)
                                a_trades = getattr(self, "_cv_fold_guard_trades_main", None)
                                if (a_uuid != g_uuid) or (a_trades != g_trades):
                                    print(
                                        f"[ALERT] [CV][Patch7] Diagnostic mutated fold guard | ctx={ctx} | "
                                        f"uuid {g_uuid}->{a_uuid} | trades {g_trades}->{a_trades}"
                                    )
                            except Exception:
                                pass
                            _cv__restore_state_after_diagnostics(snap, present)
                            try:
                                self._cv_fold_guard_uuid = g_uuid
                                self._cv_fold_guard_trades_main = g_trades
                            except Exception:
                                pass


                    for j, fold in enumerate(fold_iter, start=1):
                        if _use_monthly:
                            # fold is a dict like: {"train_iloc": (ts, te), "val_iloc": (vs, ve), ...}
                            try:
                                ts, te = fold.get("train_iloc", (0, 0))
                                vs, ve = fold.get("val_iloc", (0, 0))
                            except Exception:
                                ts = te = vs = ve = 0
                            tr  = train_data.iloc[int(ts):int(te)]
                            val = train_data.iloc[int(vs):int(ve)]
                            split = int(vs)  # for logging/penalty context below
                        else:
                            split = fold
                            tr_end_idx = max(0, split - embargo_bars)
                            tr         = train_data.iloc[:tr_end_idx]
                            val        = train_data.iloc[split : split + val_window_local]

                        # record rows + val_end ts
                        # (NEW) also record train rows + true val_start ts for accurate tables
                        try:
                            block_train_rows.append(int(len(tr)))
                        except Exception:
                            block_train_rows.append(0)

                        rows_i = len(val)
                        block_rows.append(rows_i)
                        try:
                            vstart_ts = val.index[0] if rows_i > 0 else None
                        except Exception:
                            vstart_ts = None
                        val_starts_ts_cv.append(vstart_ts)
                        try:
                            vend_ts = val.index[-1] if rows_i > 0 else None
                        except Exception:
                            vend_ts = None

                        val_ends_ts.append(vend_ts)
                        val_ends_ts_cv.append(vend_ts)

                        # Size sanity
                        if len(tr) < max(100, lags + 5) or len(val) < max(20, lags + 1):
                            _cv_penalty(
                                "MiniBlockCV reject split (insufficient rows)",
                                split_start=int(split),
                                len_tr=len(tr),
                                min_len_tr=max(100, lags + 5),
                                len_val=len(val),
                                min_len_val=max(20, lags + 1),
                            )

                            block_scores.append(float("nan"))
                            block_active_rates.append(0.0)
                            block_trades.append(0)
                            block_eff_conf.append(float("nan"))
                            block_sharpe.append(float("nan"))
                            block_psr.append(float("nan"))
                            block_neff.append(float("nan"))
                            block_pruned.append(False)
                            block_all_hold.append(True)
                            block_reasons.append("TooShort")

                            # (B1/B2-minimal) Commit a fold record into the fixed slot BEFORE continue.
                            # This is the key to preventing "parallel array drift" for this early-exit branch.
                            try:
                                fold_records[int(j) - 1] = {
                                    "fold_idx": int(j),
                                    "train_rows": int(len(tr)),
                                    "val_rows": int(rows_i),
                                    "vstart": vstart_ts,     # prefer these keys everywhere
                                    "vend": vend_ts,

                                    # denoms (safe defaults -- eligibility not computed in this early exit)
                                    "post_feature_bars_total": int(rows_i),
                                    "post_feature_eligible": int(rows_i),
                                    "eval_bars": int(rows_i),

                                    "score": float("nan"),
                                    "sharpe": float("nan"),
                                    "psr": float("nan"),
                                    "trades": 0,
                                    "active_rate": 0.0,
                                    "precision_trade": float("nan"),
                                    "n_trade_preds": 0,
                                    "eff_conf": float("nan"),
                                    "reason": "TooShort",
                                    "pruned": False,
                                    "all_hold": True,
                                    "status": "skipped",
                                }
                            except Exception:
                                pass

                            # Pretty debug card for this invalid fold
                            _print_pruned_summary(
                                block_id=j,
                                reason="MiniBlockCV reject split (insufficient rows)",
                                rows=rows_i,
                                trades=0,
                                active_rate=0.0,
                                sharpe=float("nan"),
                                fold_label=fold_label,
                            )

                            _early_structural_prune_if_hopeless()
                            continue

                        # =========================
                        # Evaluate this mini-block
                        # =========================
                        try:
                            # Reset per-block 'used' nudge params (avoids log bleed from prior blocks)
                            try:
                                self._last_runtime_active_band_used = None
                                self._last_runtime_conf_step_used = None
                            except Exception:
                                pass
                            
                            # Evaluate block (ensemble routing — deep ensembles only)
                            if (
                                isinstance(model_type_local, str)
                                and model_type_local.startswith("ensemble_")
                            ):
                                metrics = self.test_ensemble_strategy(
                                    train_start=tr.index[0],
                                    train_end=tr.index[-1],
                                    test_start=val.index[0],
                                    test_end=val.index[-1],
                                    lags=lags,
                                    label_threshold=params.get(
                                        "label_threshold", 0.0
                                    ),
                                    ensemble_config=params,
                                    model_type=model_type_local,
                                )
                            else:
                                metrics = self.test_strategy(
                                    train_start=tr.index[0],
                                    train_end=tr.index[-1],
                                    test_start=val.index[0],
                                    test_end=val.index[-1],
                                    lags=lags,
                                    confidence_threshold=conf_thr,
                                    label_threshold=params.get(
                                        "label_threshold", 0.0
                                    )
                                )

                            # Require 16-tuple
                            if (not isinstance(metrics, tuple)) or len(metrics) != 16:
                                _cv_penalty(
                                    "MiniBlockCV metrics malformed",
                                    mtype=type(metrics).__name__,
                                    mlen=(
                                        len(metrics)
                                        if hasattr(metrics, "__len__")
                                        else "NA"
                                    ),
                                )

                                block_scores.append(float("nan"))
                                block_active_rates.append(0.0)
                                block_trades.append(0)
                                block_eff_conf.append(float("nan"))
                                block_sharpe.append(float("nan"))
                                block_psr.append(float("nan"))
                                block_neff.append(float("nan"))
                                block_pruned.append(False)
                                block_all_hold.append(True)
                                block_reasons.append("BadMetrics")

                                _print_pruned_summary(
                                    block_id=j,
                                    reason="MiniBlockCV metrics malformed",
                                    rows=rows_i,
                                    fold_label=fold_label,
                                )
                                
                                try:
                                    _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                                    _pf_total = int(_diag.get("bars_total", rows_i) or rows_i)
                                    _pf_elig  = int(_diag.get("eligible_bars", _pf_total) or _pf_total)
                                    _eval_bars = int(getattr(self, "_last_eval_bars", _pf_elig) or _pf_elig)
                                    fold_records[int(j) - 1] = {
                                        "vstart": vstart_ts,
                                        "vend": vend_ts,
                                        "train_rows": int(len(tr)),
                                        "val_rows": int(rows_i),
                                        "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                        "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                        "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(rows_i),
                                        "psr": float("nan"),
                                        "trades": 0,
                                        "active_rate": 0.0,
                                        "precision_trade": float("nan"),
                                        "n_trade_preds": 0,
                                        "sharpe": float("nan"),
                                        "reason": "BadMetrics",
                                        "pruned": False,
                                        "status": "invalid",
                                    }
                                except Exception:
                                    pass

                                
                                _early_structural_prune_if_hopeless()
                                continue


                            # Unpack metrics
                            (
                                perf,
                                outperf,
                                creturns,
                                sharpe,
                                drawdown,
                                trades,
                                _,
                                _,
                                _,
                                _,
                                active_rate,
                                *_rest,
                            ) = metrics
                            
                            # ============================================================
                            # - Prevents stale reuse of _cv_last_eval_df from prior fold
                            # - Ensures per-regime outputs for invalid folds are NaN/empty
                            # ============================================================
                            _diag_valid = True
                            try:
                                _diag_valid = (int(trades) > 0) and np.isfinite(float(sharpe))
                            except Exception:
                                _diag_valid = False

                            if not _diag_valid:
                                # ------------------------------------------------------------
                                # (e.g., 0 trades after filtering). Without this, the monthly
                                # overview later shows "NO DATA / PRUNED" despite the fold
                                # having run, causing the historic index-drift bug.
                                # ------------------------------------------------------------
                                try:
                                    trades_int = int(trades) if trades is not None else 0
                                except Exception:
                                    trades_int = 0

                                try:
                                    ar_pr = float(active_rate) if (active_rate is not None and np.isfinite(float(active_rate))) else 0.0
                                except Exception:
                                    ar_pr = 0.0

                                reason_tag = "NoTrades" if trades_int <= 0 else "InvalidSR"
                                
                                try:
                                    _print_pruned_summary(
                                        block_id=j,
                                        reason=reason_tag,
                                        rows=rows_i,
                                        trades=trades_int,
                                        active_rate=float(ar_pr),
                                        sharpe=float(sharpe) if (sharpe is not None and np.isfinite(float(sharpe))) else float("nan"),
                                        fold_label=fold_label,
                                    )
                                except Exception:
                                    pass

                                # Append fold-aligned placeholders (so len(block_*) increments)
                                block_scores.append(float("nan"))
                                block_active_rates.append(float(ar_pr))
                                block_trades.append(int(trades_int))
                                block_eff_conf.append(float("nan"))
                                block_sharpe.append(float("nan"))
                                block_psr.append(float("nan"))
                                block_neff.append(float("nan"))
                                block_pruned.append(False)
                                block_all_hold.append(True)
                                pred_cards.append({})
                                block_reasons.append(reason_tag)

                                # Commit fold-local record into the fixed slot
                                try:
                                    fold_records[int(j) - 1] = {
                                        "vstart": vstart_ts,
                                        "vend": vend_ts,
                                        "train_rows": int(len(tr)),
                                        "val_rows": int(rows_i),
                                        "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                        "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                        "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(rows_i),
                                        "psr": float("nan"),
                                        "trades": int(trades_int),
                                        "active_rate": float(ar_pr),
                                        "precision_trade": float("nan"),
                                        "n_trade_preds": 0,
                                        "sharpe": float("nan"),
                                        "reason": reason_tag,
                                        "pruned": False,
                                        "status": "invalid",
                                    }
                                except Exception:
                                    pass

                                # Hard-stop any possibility of stale fold reuse.
                                try:
                                    self._cv_last_eval_df = None
                                except Exception:
                                    pass

                                # Keep fold alignment: add an empty placeholder frame
                                # so later per-regime table prints NaNs for this fold.
                                try:
                                    _empty = pd.DataFrame(
                                        {
                                            "regime_id": pd.Series(dtype=int),
                                            "pred": pd.Series(dtype=float),
                                        }
                                    )
                                    fold_eval_frames.append(_empty)
                                except Exception:
                                    pass

                                # Also append fold-aligned per-regime placeholders.
                                try:
                                    for rid in (0, 1, 2):
                                        per_fold_regime_trades[rid].append(0)
                                        per_fold_regime_active[rid].append(float("nan"))
                                        per_fold_regime_sharpe[rid].append(float("nan"))
                                except Exception:
                                    pass

                                _early_structural_prune_if_hopeless()

                                # Skip diagnostic recomputation entirely for this fold.
                                continue

                            
                            # store per-fold evaluation frame for downstream per-regime CV diagnostics
                            # (fold-local copy: diagnostics must never mutate the fold's official result)
                            try:
                                _df0 = getattr(self, "_cv_last_eval_df", None)
                                df_fold = _df0.copy(deep=True) if isinstance(_df0, pd.DataFrame) else None
                                if df_fold is not None and (not df_fold.empty):
                                    # already copied above
                                    fold_eval_frames.append(df_fold)

                                    # best-effort per-regime simple stats for this fold (non-mutating)
                                    df_r = df_fold
                                    if "regime_id" not in df_r.columns:
                                        df_r = df_r.copy()
                                        df_r["regime_id"] = 1
                                        
                                    # ------------------------------------------------------------
                                    # DIAGNOSTIC-ONLY regime id (does NOT affect training/features)
                                    # Use cfg_high_vol_thr when available to avoid "all volatile"
                                    # collapse in regime logs/tables.
                                    # ------------------------------------------------------------
                                    try:
                                        if "regime_id_diag" not in df_r.columns:
                                            _cfgd = {}
                                            try:
                                                _cfgd = dict(getattr(df_r, "attrs", {}).get("features_config", {}) or {})
                                            except Exception:
                                                _cfgd = dict(getattr(self, "features_config", {}) or {})

                                            # choose columns
                                            adx_col = "adx"
                                            if adx_col not in df_r.columns:
                                                adx_w = int(_cfgd.get("adx_window_core", 14))
                                                if f"adx_{adx_w}" in df_r.columns:
                                                    adx_col = f"adx_{adx_w}"
                                            vol_col = None
                                            rv_w = int(_cfgd.get("rv_window_short", 48))
                                            if f"rv_{rv_w}" in df_r.columns:
                                                vol_col = f"rv_{rv_w}"
                                            elif "rv" in df_r.columns:
                                                vol_col = "rv"

                                            adx_thr = float(_cfgd.get("adx_thresh", 20.0))
                                            # prefer train-anchored threshold printed in logs
                                            vol_thr = _cfgd.get("high_vol_thr", None)
                                            if vol_thr is None:
                                                vol_thr = float(_cfgd.get("vol_thresh", 0.001))
                                            else:
                                                vol_thr = float(vol_thr)

                                            if vol_col is not None and adx_col in df_r.columns:
                                                _adx = df_r[adx_col].astype(float).fillna(0.0)
                                                _vol = df_r[vol_col].astype(float).fillna(0.0)
                                                vol_high = (_vol > vol_thr)
                                                trend = (_adx > adx_thr)
                                                # 2=volatile, 1=trend, 0=sideways
                                                df_r = df_r.copy()
                                                df_r["regime_id_diag"] = np.where(vol_high, 2, np.where(trend, 1, 0)).astype(int)
                                    except Exception:
                                        pass

                                    _rid_col = "regime_id_diag" if "regime_id_diag" in df_r.columns else "regime_id"


                                    for rid in (0, 1, 2):
                                        sub = df_r[df_r[_rid_col] == rid]
                                        if sub is None or len(sub) == 0:
                                            continue

                                        # Use whichever prediction column exists (prefer executed "pred")
                                        pred_col = None
                                        for _c in ("pred", "pred_exec", "final_pred", "prediction"):
                                            if _c in sub.columns:
                                                pred_col = _c
                                                break
                                        if pred_col is None:
                                            continue

                                        s = sub[pred_col].fillna(0)
                                        trades_i = int((s != 0).sum())
                                        active_i = float((s != 0).mean())

                                        sharpe_i = float("nan")
                                        try:
                                            _sub_eval = sub.copy()
                                            try:
                                                if bool(getattr(self, "trading_costs", False)):
                                                    _cfg_cost = {}
                                                    try:
                                                        _cfg_cost = dict(getattr(df_r, "attrs", {}).get("features_config", {}) or {})
                                                    except Exception:
                                                        _cfg_cost = dict(getattr(self, "features_config", {}) or {})
                                                    try:
                                                        _sub_eval.attrs["features_config"] = dict(_cfg_cost)
                                                    except Exception:
                                                        pass
                                                    _sub_eval = self._ensure_cost_columns(_sub_eval, _cfg_cost)
                                            except Exception:
                                                pass

                                            _m = compute_full_evaluation_metrics(
                                                df=_sub_eval,
                                                trading_costs=self.trading_costs,
                                                slippage_factor=self.slippage_factor,
                                                eval_context=f"cv:diagnostic:per_regime_metrics:rid={rid}",
                                            )
                                            sharpe_i = float(_m[3]) if _m is not None else float("nan")
                                        except Exception:
                                            pass

                                        per_fold_regime_trades[rid].append(trades_i)
                                        per_fold_regime_active[rid].append(active_i)
                                        per_fold_regime_sharpe[rid].append(sharpe_i)
                            except Exception:
                                pass

                        except Exception as e:
                            # If an inner component requested pruning (Optuna), propagate it.
                            try:
                                import optuna as _opt
                                if isinstance(e, _opt.TrialPruned):
                                    raise
                            except Exception:
                                pass
                            _cv_penalty(
                                "MiniBlockCV exception during block evaluation",
                                split_start=int(split),
                                error=str(e),
                            )
                            if bool(config.get("print_cv_debug", False)):
                                traceback.print_exc()

                            # Detect Optuna structural prune vs generic exception
                            _is_pruned = False
                            try:
                                import optuna as _opt
                                _is_pruned = isinstance(e, _opt.TrialPruned)
                            except Exception:
                                pass

                            reason = f"Pruned: {e}" if _is_pruned else f"Exception: {e}"

                            block_scores.append(float("nan"))
                            block_sharpe.append(float("nan"))
                            block_active_rates.append(0.0)
                            block_trades.append(0)
                            block_eff_conf.append(float("nan"))
                            block_pruned.append(_is_pruned)
                            block_all_hold.append(False)
                            pred_cards.append({})
                            block_reasons.append(reason)
                            block_psr.append(float("nan"))
                            block_neff.append(float("nan"))

                            # Pretty card for this invalid/pruned fold
                            _print_pruned_summary(
                                block_id=j,
                                reason=reason,
                                rows=rows_i,
                                fold_label=fold_label,
                            )

                            continue

                        try:
                            # These will be set inside test_strategy/test_ensemble_strategy
                            brier_f = float(getattr(self, "_last_calib_brier", float("nan")))
                            nll_f   = float(getattr(self, "_last_calib_nll",   float("nan")))
                            n_f     = int(getattr(self, "_last_calib_n",       0))
                            if (
                                n_f > 0
                                and np.isfinite(brier_f)
                                and np.isfinite(nll_f)
                            ):
                                calib_brier_sum += brier_f * n_f
                                calib_nll_sum   += nll_f   * n_f
                                calib_n_samples += n_f
                                if bool(config.get("print_cv_debug", False)):
                                    print(
                                        f"[CV-Calib-Fold] brier={brier_f:.6f} | "
                                        f"nll={nll_f:.6f} | n={n_f}"
                                    )
                        except Exception as _e:
                            if bool(config.get("print_cv_debug", False)):
                                print(f"[CV-Calib-Fold] Failed to accumulate calibration: {_e}")
                        
                        # Basic validity after successful metrics retrieval
                        _invalid_sharpe = (
                            sharpe is None
                            or (not np.isfinite(sharpe))
                            or (float(sharpe) <= -9998.0)
                        )

                        _no_trades = (
                            trades is None
                            or (not np.isfinite(trades))
                            or (int(trades) <= 0)
                        )
                        if _invalid_sharpe or _no_trades:
                            # mark invalid block uniformly (no-trade or broken)
                            # IMPORTANT: append to each block_* list EXACTLY ONCE
                            try:
                                trades_int = int(trades) if (trades is not None and np.isfinite(trades)) else 0
                            except Exception:
                                trades_int = 0
                                
                            try:
                                ar_pr = float(active_rate) if (active_rate is not None and np.isfinite(active_rate)) else 0.0
                            except Exception:
                                ar_pr = 0.0

                            block_scores.append(float("nan"))
                            block_active_rates.append(float(ar_pr))
                            block_trades.append(int(trades_int))
                            block_eff_conf.append(float("nan"))
                            block_pruned.append(False)

                            _final_after = getattr(
                                self, "_last_final_preds_dist", None
                            )
                            _all_hold = False
                            try:
                                if isinstance(_final_after, dict):
                                    non_hold = sum(
                                        v
                                        for k, v in _final_after.items()
                                        if str(k) != "0"
                                    )
                                    _all_hold = (int(non_hold) == 0)
                            except Exception:
                                pass

                            block_all_hold.append(bool(_all_hold))
                            pred_cards.append(
                                {
                                    "label_counts": getattr(
                                        self, "_last_label_counts", None
                                    ),
                                    "thr": getattr(
                                        self,
                                        "_last_label_threshold",
                                        params.get("label_threshold", None),
                                    ),
                                    "test_len": getattr(
                                        self, "_last_test_len", rows_i
                                    ),
                                    "raw_preds": getattr(
                                        self,
                                        "_last_raw_pred_dist",
                                        None,
                                    ),
                                    "decoded_before": getattr(
                                        self,
                                        "_last_decoded_preds_before",
                                        None,
                                    ),
                                    "final_after_thr": getattr(
                                        self,
                                        "_last_final_preds_dist",
                                        None,
                                    ),
                                }
                            )
                            
                            reason_tag = "NoTrades" if _no_trades else "InvalidSR"
                            block_reasons.append(reason_tag)
                            

                            _print_pruned_summary(
                                block_id=j,
                                reason=reason_tag,
                                rows=rows_i,
                                trades=trades_int,
                                active_rate=float(ar_pr),
                                fold_label=fold_label,
                            )

                            try:
                                fold_records[int(j) - 1] = {
                                    "vstart": vstart_ts,
                                    "vend": vend_ts,
                                    "train_rows": int(len(tr)),
                                    "val_rows": int(rows_i),
                                    "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                    "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                    "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(rows_i),
                                    "psr": float("nan"),
                                    "trades": int(trades_int),
                                    "active_rate": float(ar_pr),
                                    "precision_trade": float(getattr(self, "_last_precision_trade", float("nan"))),
                                    "n_trade_preds": int(getattr(self, "_last_n_trade_preds", 0) or 0),
                                    "sharpe": float("nan"),
                                    "reason": reason_tag,
                                    "pruned": False,
                                    "status": "invalid",
                                }
                            except Exception:
                                pass


                            block_sharpe.append(float("nan"))
                            block_psr.append(float("nan"))
                            block_neff.append(float("nan"))
                            
                            _early_structural_prune_if_hopeless()
                            continue



                        # --- Research-aligned gates & penalties ---
                        trades_i = int(trades)
                        ar = (
                            float(active_rate)
                            if (active_rate is not None and np.isfinite(active_rate))
                            else 0.0
                        )

                        # Estimate average holding in bars and #independent bets (Lopez de Prado style)
                        # active_rate ~= (trades * avg_hold) / rows  => avg_hold ~= (ar * rows) / trades
                        avg_hold_bars = (
                            float("inf")
                            if trades_i <= 0
                            else (ar * float(rows_i)) / float(trades_i)
                        )
                        indep_bets_est = (
                            0.0
                            if (not np.isfinite(avg_hold_bars) or avg_hold_bars <= 0)
                            else (float(rows_i) / (2.0 * avg_hold_bars))
                        )
                        indep_bets_est = (
                            float(min(float(trades_i), indep_bets_est))
                            if np.isfinite(indep_bets_est)
                            else 0.0
                        )

                        # Activity band around per-trial target
                        # Center the CV active-rate band on the *per-trial* (per-family) target.
                        # NOTE: `config` here is CV config (self.config). The target is a feature-level policy.
                        _f_cfg = getattr(self, "features_config", {}) or {}
                        ar_target = float(_f_cfg.get("target_active_rate", config.get("target_active_rate", 0.10)))
                        ar_margin = float(
                            config.get("cv_active_rate_margin", 0.12)
                        )  # +/- absolute
                        ar_low = float(
                            config.get(
                                "cv_active_rate_low",
                                max(0.01, ar_target - ar_margin),
                            )
                        )
                        ar_high = float(
                            config.get(
                                "cv_active_rate_high",
                                min(0.95, ar_target + ar_margin),
                            )
                        )

                        # Dynamic "too-many-trades" cap
                        T_cap_hard = int(
                            config.get(
                                "cv_max_trades_per_block",
                                DEFAULT_CV["cv_max_trades_per_block"],
                            )
                        )
                        cap_frac = float(
                            config.get(
                                "cv_dynamic_trades_cap_frac",
                                DEFAULT_CV["cv_dynamic_trades_cap_frac"],
                            )
                        )
                        T_cap_dyn = int(max(50, cap_frac * float(rows_i)))
                        T_cap = int(min(T_cap_hard, T_cap_dyn))

                        # Reliability gates defaults
                        from math import sqrt
                        try:
                            from scipy.stats import norm
                        except Exception:
                            norm = None

                        def _psr(sr, n_eff, sr_bench=0.0, skew=0.0, kurt=3.0):
                            """Probabilistic Sharpe Ratio: P(SR > sr_bench)."""
                            if n_eff is None or n_eff < 2 or not (sr == sr):
                                return 0.0
                            num = (sr - sr_bench) * sqrt(max(n_eff - 1, 1))
                            den = sqrt(
                                max(
                                    1e-12,
                                    1
                                    - skew * sr
                                    + (kurt - 1.0) * (sr ** 2) / 4.0,
                                )
                            )
                            z = num / den
                            if norm is None:
                                import math

                                return 0.5 * (1.0 + math.erf(z / sqrt(2)))
                            return float(norm.cdf(z))

                        def _dsr_sign(sr, n_eff, sr_max=0.0):
                            """Simple DSR sign proxy."""
                            if n_eff is None or n_eff < 2 or not (sr == sr):
                                return -1.0
                            return (sr - sr_max) * sqrt(max(n_eff, 1))

                        if bool(config.get("print_cv_debug", False)):
                            print(
                                f"[Block {j}] rows={rows_i} trades={trades_i} ar={ar:.3f}"
                            )

                        defaults_features = deepcopy(DEFAULT_FEATURES)
                        defaults_cv = deepcopy(DEFAULT_CV)

                        gating_mode = config.get(
                            "gating_mode",
                            defaults_features.get("gating_mode", "bets_psr"),
                        )
                        min_trades_block = int(
                            config.get(
                                "cv_min_trades_per_block",
                                defaults_cv.get(
                                    "cv_min_trades_per_block",
                                    defaults_features.get(
                                        "min_trades_per_block", 20
                                    ),
                                ),
                            )
                        )
                        min_indep_bets = int(
                            config.get(
                                "cv_min_indep_bets_per_block",
                                defaults_features.get(
                                    "min_independent_bets", 10
                                ),
                            )
                        )
                        psr_alpha = float(
                            config.get(
                                "psr_alpha",
                                defaults_features.get("psr_alpha", 0.10),
                            )
                        )  # PSR cutoff = 1 - psr_alpha
                        dsr_prune = bool(
                            config.get(
                                "dsr_prune",
                                defaults_features.get("dsr_prune", False),
                            )
                        )
                        floor_cv_final = float(
                            config.get(
                                "floor_cv_final",
                                defaults_features.get("floor_cv_final", -4.0),
                            )
                        )

                        if config.get("print_cv_debug", False):
                            print(
                                "[Debug] Reliability gates -> "
                                f"psr_alpha={psr_alpha:.2f} (cutoff={1.0 - psr_alpha:.2f}) | "
                                f"min_trades={min_trades_block} | "
                                f"min_indep={min_indep_bets} | "
                                f"dsr_prune={dsr_prune}"
                            )

                        # --- Effective independent bets & PSR for this block (if possible) ---
                        indep_bets = float("nan")
                        psr_block = float("nan")
                        try:
                            if trades_i > 0 and np.isfinite(sharpe):
                                avg_hold_safe = (
                                    float(avg_hold_bars)
                                    if np.isfinite(avg_hold_bars)
                                    and avg_hold_bars > 0
                                    else 1.0
                                )
                                n_eff = trades_i / avg_hold_safe
                                n_eff = max(min_indep_bets, n_eff)
                                indep_bets = float(n_eff)
                                psr_block = float(
                                    _psr(
                                        float(sharpe),
                                        int(round(n_eff)),
                                        sr_bench=0.0,
                                    )
                                )
                        except Exception:
                            indep_bets = float("nan")
                            psr_block = float("nan")

                        # -- Reliability decision: HARD vs SOFT vs OK --
                        reason = None
                        hard_reject = False

                        # 1) Truly broken -> HARD
                        if np.isfinite(T_cap) and np.isfinite(trades_i) and float(trades_i) > float(T_cap):
                            reason = f"OvertradeCap(trades={int(trades_i)} > cap={int(T_cap)})"
                            hard_reject = True
                        elif trades_i <= 0 or not np.isfinite(sharpe):
                            reason = (
                                f"NoTradesOrNaN(trades={trades_i}, sr={sharpe})"
                            )
                            hard_reject = True

                        # 2) Too few trades -> SOFT
                        elif trades_i < min_trades_block:
                            reason = (
                                f"TooFewTrades({trades_i}<{min_trades_block})"
                            )

                        # 3) PSR / DSR / Sharpe checks -> SOFT (informative)
                        else:
                            try:
                                n_eff_int = int(
                                    max(
                                        min_indep_bets,
                                        trades_i
                                        / max(1.0, float(avg_hold_bars)),
                                    )
                                )
                            except Exception:
                                n_eff_int = int(
                                    max(min_indep_bets, trades_i)
                                )

                            indep_bets = float(n_eff_int)
                            psr = _psr(float(sharpe), n_eff_int)
                            dsr = _dsr_sign(float(sharpe), n_eff_int)
                            psr_block = float(psr)

                            if psr < (1.0 - psr_alpha):
                                reason = (
                                    f"PSR<{1.0 - psr_alpha:.2f} ({psr:.3f})"
                                )
                            elif dsr_prune and dsr <= 0.0:
                                reason = f"DSR<=0 ({dsr:.3f})"
                            elif float(sharpe) <= float(floor_cv_final):
                                reason = (
                                    f"Sharpe<=floor "
                                    f"({float(sharpe):.2f} <= {float(floor_cv_final):.2f})"
                                )

                        if reason is not None:
                            block_reasons.append(reason)

                            if hard_reject:
                                # Hard fail -> no score; mark NaNs and continue
                                eff_conf_local = float(
                                    getattr(
                                        self,
                                        "_last_conf_thr_used",
                                        conf_thr,
                                    )
                                )
                                block_scores.append(float("nan"))
                                block_active_rates.append(float(ar))
                                block_trades.append(int(trades_i))
                                block_eff_conf.append(eff_conf_local)
                                block_sharpe.append(float("nan"))
                                block_psr.append(float("nan"))
                                block_neff.append(float("nan"))
                                block_pruned.append(True)
                                block_all_hold.append(False)
                                pred_cards.append(
                                    {
                                        "label_counts": getattr(
                                            self,
                                            "_last_label_counts",
                                            None,
                                        ),
                                        "thr": getattr(
                                            self,
                                            "_last_label_threshold",
                                            params.get(
                                                "label_threshold", None
                                            ),
                                        ),
                                        "test_len": rows_i,
                                        "raw_preds": getattr(
                                            self,
                                            "_last_raw_pred_dist",
                                            None,
                                        ),
                                        "decoded_before": getattr(
                                            self,
                                            "_last_decoded_preds_before",
                                            None,
                                        ),
                                        "final_after_thr": getattr(
                                            self,
                                            "_last_final_preds_dist",
                                            None,
                                        ),
                                        "eff_conf": eff_conf_local,
                                        "avg_hold_bars": (
                                            float(avg_hold_bars)
                                            if np.isfinite(
                                                avg_hold_bars
                                            )
                                            else "--"
                                        ),
                                        "indep_bets": "--",
                                        "psr": "--",
                                    }
                                )

                                # Pretty card explaining *why* this fold was hard-pruned
                                _print_pruned_summary(
                                    block_id=j,
                                    reason=reason,
                                    rows=rows_i,
                                    trades=int(trades_i),
                                    active_rate=float(ar),
                                    sharpe=float(sharpe) if np.isfinite(sharpe) else float("nan"),
                                    fold_label=fold_label,
                                )

                                _early_structural_prune_if_hopeless()
                                continue  # skip scoring for this block

                            # Soft fail -> informative only; still score block
                            block_pruned.append(False)
                        else:
                            block_reasons.append("")
                            block_pruned.append(False)


                        # --- Soft activity regularization around [ar_low, ar_high] ---

                        _cd = (
                            CLASS_DEFAULTS
                            if "CLASS_DEFAULTS" in globals()
                            else {}
                        )
                        _cd_cv = _cd.get("cv", {})

                        lam_turn = float(
                            config.get(
                                "turnover_penalty_lambda",
                                _cd_cv.get(
                                    "turnover_penalty_lambda", 0.0
                                ),
                            )
                        )
                        lam_low = float(
                            config.get(
                                "cv_soft_active_low_lambda",
                                _cd_cv.get(
                                    "cv_soft_active_low_lambda", 0.0
                                ),
                            )
                        )
                        lam_high = float(
                            config.get(
                                "cv_soft_active_high_lambda",
                                _cd_cv.get(
                                    "cv_soft_active_high_lambda", 0.0
                                ),
                            )
                        )

                        pen_low = max(0.0, float(ar_low) - float(ar))
                        pen_high = max(0.0, float(ar) - float(ar_high))
                        turnover = float(trades_i) / float(max(1, rows_i))
                        
                                                # --- Soft turnover band penalties (model-family aware) ---
                        model_type_local = str(
                            params.get(
                                "model_type",
                                _f_cfg.get("model_type", getattr(self, "model_type", "")),
                            )
                        )
                        _turn_bands = {
                            # Classical ML
                            "logistic": (0.03, 0.18),
                            "svm": (0.03, 0.18),
                            "decision_tree": (0.03, 0.18),
                            "random_forest": (0.03, 0.18),
                            "xgboost": (0.03, 0.18),
                            # Deep supervised
                            "cnn": (0.02, 0.15),
                            "lstm": (0.02, 0.15),
                            "transformer": (0.02, 0.15),
                            # Ensembles
                            "ensemble_cnn_lstm_xgboost": (0.02, 0.14),
                            "ensemble_adaptive_regime": (0.01, 0.12),
                            # RL
                            "dqn": (0.05, 0.25),
                        }

                        # Allow explicit override from CV config; otherwise use family band.
                        _tlow_cfg = config.get("cv_turnover_low", _cd_cv.get("cv_turnover_low", None))
                        _thigh_cfg = config.get("cv_turnover_high", _cd_cv.get("cv_turnover_high", None))
                        try:
                            turn_low = float(_tlow_cfg) if _tlow_cfg is not None else float(_turn_bands.get(model_type_local, (0.02, 0.18))[0])
                        except Exception:
                            turn_low = float(_turn_bands.get(model_type_local, (0.02, 0.18))[0])
                        try:
                            turn_high = float(_thigh_cfg) if _thigh_cfg is not None else float(_turn_bands.get(model_type_local, (0.02, 0.18))[1])
                        except Exception:
                            turn_high = float(_turn_bands.get(model_type_local, (0.02, 0.18))[1])

                        lam_tlow = float(
                            config.get(
                                "cv_turnover_low_lambda",
                                _cd_cv.get("cv_turnover_low_lambda", 0.0),
                            )
                        )
                        lam_thigh = float(
                            config.get(
                                "cv_turnover_high_lambda",
                                _cd_cv.get(
                                    "cv_turnover_high_lambda",
                                    0.0,
                                ),
                            )
                        )

                        # Normalize penalties so units are comparable across bands.
                        _tl_den = max(1e-12, abs(float(turn_low)))
                        _th_den = max(1e-12, abs(float(turn_high)))
                        pen_turn_low = max(0.0, float(turn_low) - float(turnover)) / _tl_den
                        pen_turn_high = max(0.0, float(turnover) - float(turn_high)) / _th_den


                        # Penalized CV score
                        score_penalized = (
                            float(sharpe)
                            - lam_turn * turnover
                            - (lam_low * pen_low + lam_high * pen_high)
                            - (lam_tlow * pen_turn_low + lam_thigh * pen_turn_high)
                        )
                        
                        if config.get("print_cv_debug", False) and (
                            (pen_low > 0.0) or (pen_high > 0.0) or (pen_turn_low > 0.0) or (pen_turn_high > 0.0)
                        ):
                            print(
                                f"[CV][Penalty] model={model_type_local} sr={float(sharpe):.3f} score={float(score_penalized):.3f} "
                                f"ar={ar:.3f} band=[{ar_low:.3f},{ar_high:.3f}] "
                                f"turn={turnover:.4f} band=[{turn_low:.4f},{turn_high:.4f}] "
                                f"pen_ar=({pen_low:.3f},{pen_high:.3f}) pen_turn=({pen_turn_low:.3f},{pen_turn_high:.3f})"
                            )

                        # --- Record metrics for this block ---

                        # -------------------------------
                        # Fold-level intent-precision gate
                        # -------------------------------
                        _pgate_on = bool(cv_config.get("cv_prune_precision_intent", False))
                        _p_thr = float(cv_config.get("cv_prune_min_precision_intent", 0.38))
                        _p_nmin_fold = int(cv_config.get("cv_prune_min_intent_bars_fold", 30))

                        # Pull fold-local intent precision (post-confidence gating).
                        # Prefer the fold eval df attrs; fall back to self._last_* mirrors.
                        _p_int = float("nan")
                        _n_int = 0
                        try:
                            _eval_df = getattr(self, "_cv_last_eval_df", None)
                            _attrs = getattr(_eval_df, "attrs", {}) or {}
                            _p_int = float(_attrs.get("precision_intent", float("nan")))
                            _n_int = int(_attrs.get("intent_bars", 0) or 0)
                        except Exception:
                            pass
                        try:
                            if not np.isfinite(_p_int):
                                _p_int = float(getattr(self, "_last_precision_intent", float("nan")))
                        except Exception:
                            pass
                        try:
                            if int(_n_int) <= 0:
                                _n_int = int(getattr(self, "_last_intent_bars", 0) or 0)
                        except Exception:
                            pass
                        
                        _reason = ""  # default: no fold invalidation reason
                        # Debug visibility (only when print_cv_debug=True)
                        # Print right before the fold gate can actually trigger (on + eligible).
                        if bool(config.get("print_cv_debug", False)) and _pgate_on and (_n_int >= int(_p_nmin_fold)) and np.isfinite(_p_int):
                            print(f"[CV][IntentGate] on={_pgate_on} thr={_p_thr} nmin_fold={_p_nmin_fold} p={_p_int} n={_n_int}")

                        # If enabled: discard fold like other invalid folds (score -> NaN, reason tagged)
                        if _pgate_on and (_n_int >= int(_p_nmin_fold)) and np.isfinite(_p_int) and (float(_p_int) < float(_p_thr)):

                            # Keep ALL per-fold lists aligned with the normal (non-invalid) append path
                            block_scores.append(float("nan"))
                            block_active_rates.append(float(ar))
                            block_trades.append(int(trades_i))

                            # These exist in your normal path; add placeholders here too
                            try:
                                eff_conf_local = float(getattr(self, "_last_conf_thr_used", conf_thr))
                            except Exception:
                                eff_conf_local = float("nan")
                            try:
                                block_eff_conf.append(float(eff_conf_local))
                            except Exception:
                                pass
                            try:
                                block_sharpe.append(float("nan"))
                            except Exception:
                                pass
                            try:
                                block_psr.append(float("nan"))
                            except Exception:
                                pass
                            try:
                                block_neff.append(float("nan"))
                            except Exception:
                                pass
                            try:
                                block_all_hold.append(False)
                            except Exception:
                                pass
                            try:
                                pred_cards.append(None)
                            except Exception:
                                pass
                            # Optional: if you maintain cov-threshold per fold, keep it aligned too
                            try:
                                block_cov_thr.append(float(getattr(self, "_coverage_conf_thr", float("nan"))))
                            except Exception:
                                pass

                            # Keep intent arrays aligned
                            block_precision_intent.append(float(_p_int))
                            block_intent_bars.append(int(_n_int))

                            try:
                                _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                                _pf_total = int(_diag.get("bars_total", rows_i) or rows_i)
                                _pf_elig  = int(_diag.get("eligible_bars", _pf_total) or _pf_total)
                                _eval_bars = int(getattr(self, "_last_eval_bars", _pf_elig) or _pf_elig)
                                fold_records[int(j) - 1] = {
                                    "vstart": vstart_ts,
                                    "vend": vend_ts,
                                    "train_rows": int(len(tr)),
                                    "val_rows": int(rows_i),
                
                                    "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                    "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                    "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(rows_i),
                                    "psr": float(psr_block) if np.isfinite(psr_block) else float("nan"),
                                    "trades": int(trades_i),
                                    "active_rate": float(ar),
                                    "precision_trade": float(getattr(self, "_last_precision_trade", float("nan"))),
                                    "n_trade_preds": int(getattr(self, "_last_n_trade_preds", 0) or 0),
                                    "precision_intent": float(_p_int),
                                    "intent_bars": int(_n_int),
                                    "sharpe": float("nan"),
                                    "reason": _reason,
                                    "pruned": False,
                                    "status": "invalid",
                                }
                            except Exception:
                                pass

                            continue

                        eff_conf_local = float(getattr(self, "_last_conf_thr_used", conf_thr))

                        try:
                            _thr_base = float(getattr(self, "_coverage_conf_thr"))
                            block_cov_thr.append(_thr_base)
                        except Exception:
                            pass

                        _final_after = getattr(self, "_last_final_preds_dist", None)
                        _all_hold = False
                        try:
                            if isinstance(_final_after, dict):
                                non_hold = sum(v for k, v in _final_after.items() if str(k) != "0")
                                _all_hold = (int(non_hold) == 0)
                        except Exception:
                            _all_hold = False

                        block_scores.append(float(score_penalized))
                        block_active_rates.append(float(ar))
                        block_trades.append(int(trades_i))
                        
                        block_eff_conf.append(float("nan"))
                        block_sharpe.append(float("nan"))
                        block_psr.append(float("nan"))
                        block_neff.append(float("nan"))
                        block_pruned.append(False)
                        block_all_hold.append(True)
                        block_reasons.append(str(_reason))
                        pred_cards.append({})


                        # Keep intent arrays aligned
                        block_precision_intent.append(float(_p_int) if np.isfinite(_p_int) else float("nan"))
                        block_intent_bars.append(int(_n_int))

                        pred_cards.append({
                            "label_counts": getattr(self, "_last_label_counts", None),
                            "thr": getattr(self, "_last_label_threshold", params.get("label_threshold", None)),
                            "test_len": getattr(self, "_last_test_len", rows_i),
                            "final_after_thr": _final_after,
                            "eff_conf": eff_conf_local,
                            "avg_hold_bars": (float(avg_hold_bars) if np.isfinite(avg_hold_bars) else "--"),
                            "indep_bets": (float(indep_bets) if np.isfinite(indep_bets) else "--"),
                            "psr": (float(psr_block) if np.isfinite(psr_block) else "--"),
                            "turnover": float(turnover),
                        })

                        # fold-local record (canonical source for overview table)
                        try:
                            fold_records[int(j) - 1] = {
                                "vstart": vstart_ts,
                                "vend": vend_ts,
                                "train_rows": int(len(tr)),
                                "val_rows": int(rows_i),
                                "post_feature_bars_total": int(_pf_total) if "_pf_total" in locals() and _pf_total else int(rows_i),
                                "post_feature_eligible": int(_pf_elig) if "_pf_elig" in locals() and _pf_elig else int(rows_i),
                                "eval_bars": int(_eval_bars) if "_eval_bars" in locals() and _eval_bars else int(getattr(self, "_last_eval_bars", 0) or 0),
                                "psr": float(psr_block) if np.isfinite(psr_block) else float("nan"),
                                "trades": int(trades_i),
                                "active_rate": float(ar),
                                "precision_trade": float(getattr(self, "_last_precision_trade", float("nan"))),
                                "n_trade_preds": int(getattr(self, "_last_n_trade_preds", 0) or 0),
                                "precision_intent": float(_p_int) if np.isfinite(_p_int) else float("nan"),
                                "intent_bars": int(_n_int),
                                "sharpe": float(sharpe) if np.isfinite(sharpe) else float("nan"),
                                "reason": "",
                                "pruned": False,
                                "status": "ok",
                            }
                        except Exception:
                            pass


                        # --- Compact per-fold summary (Mini-Block Fold #j) ---
                        try:
                            cfg_f = getattr(self, "features_config", {}) or {}
                            _cd = CLASS_DEFAULTS.get("features", {}) if "CLASS_DEFAULTS" in globals() else {}

                            # Coverage / calibration
                            target_cov = float(
                                cfg_f.get(
                                    "target_active_rate",
                                    cfg_f.get("target_coverage", 0.10),
                                )
                            )
                            try:
                                base_conf = float(getattr(self, "_coverage_conf_thr"))
                            except Exception:
                                base_conf = float(
                                    cfg_f.get(
                                        "confidence_threshold",
                                        _cd.get("confidence_threshold", 0.0),
                                    )
                                )

                            calib_info = {
                                "target": target_cov,
                                "conf_thr": base_conf,
                                "bars": int(rows_i),
                            }
                            
                            # Reporting denominators (telemetry only):
                            # - bars_total: post-feature eval grid (Eligibility bars_total)
                            # - bars_eligible: actual evaluated bars (matches ExecAudit bars)
                            try:
                                _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                                _bars_total = int(_diag.get("bars_total", rows_i) or rows_i)
                            except Exception:
                                _bars_total = int(rows_i)
                            try:
                                _bars_eval = int(getattr(self, "_last_eval_bars", 0) or 0)
                                if _bars_eval <= 0:
                                    _bars_eval = int(_diag.get("eligible_bars", _bars_total) or _bars_total)
                            except Exception:
                                _bars_eval = int(_bars_total)

                            calib_info["bars_total"] = int(_bars_total)
                            calib_info["bars_eligible"] = int(_bars_eval)
                            
                            # --- Patch 2: fold reporting denominators (telemetry only) ---
                            # Use the same bar-grid universe as gating/execution (post-feature),
                            # and the same evaluated bars as Gate[OK]/ExecAudit.
                            try:
                                _diag = getattr(self, "_last_eligibility_diag", {}) or {}
                                _bars_postfeat_total = int(_diag.get("bars_total", rows_i) or rows_i)
                                _bars_postfeat_elig  = int(_diag.get("eligible_bars", _bars_postfeat_total) or _bars_postfeat_total)
                            except Exception:
                                _bars_postfeat_total = int(rows_i)
                                _bars_postfeat_elig  = int(rows_i)
                            try:
                                _bars_eval = int(getattr(self, "_last_eval_bars", _bars_postfeat_elig) or _bars_postfeat_elig)
                            except Exception:
                                _bars_eval = int(_bars_postfeat_elig)

                            calib_info["bars_total"] = int(_bars_postfeat_total)
                            calib_info["bars_eligible"] = int(_bars_eval)

                            # Dynamic alphabetagamma and coverage nudge
                            alpha = float(cfg_f.get("alpha_vol_z", _cd.get("alpha_vol_z", 0.0)))
                            beta  = float(cfg_f.get("beta_spread_norm", _cd.get("beta_spread_norm", _PC["beta_spread_norm"])))
                            gamma = float(cfg_f.get("gamma_slip_norm", _cd.get("gamma_slip_norm", _PC["gamma_slip_norm"])))
                            band  = float(cfg_f.get("runtime_active_band_margin", _cd.get("runtime_active_band_margin", _PC["runtime_active_band_margin"])))
                            step  = float(cfg_f.get("runtime_conf_nudge", _cd.get("runtime_conf_nudge", 0.01)))
                            
                            # Prefer actually-used (sanitized) nudge params from this block, when available.
                            try:
                                _band_used = getattr(self, "_last_runtime_active_band_used", None)
                                _step_used = getattr(self, "_last_runtime_conf_step_used", None)
                                if _band_used is not None and np.isfinite(_band_used):
                                    band = float(_band_used)
                                if _step_used is not None and np.isfinite(_step_used):
                                    step = float(_step_used)
                            except Exception:
                                pass

                            gate_info = {
                                "base": base_conf,
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "median_thr": float(
                                    getattr(self, "_last_conf_thr_used", base_conf)
                                ),
                                "band": band,
                                "step": step,
                            }

                            # Reliability gate parameters
                            # IMPORTANT: the summary must reflect the SAME knobs used by the actual gate.
                            # Prefer the canonical keys used elsewhere in test_strategy:
                            #   psr_alpha, floor_cv_final, cv_min_indep_bets_per_block
                            psr_alpha = float(
                                config.get(
                                    "psr_alpha",
                                    config.get(
                                        "cv_psr_alpha",
                                        cfg_f.get("psr_alpha", _cd.get("psr_alpha", 0.10)),
                                    ),
                                )
                            )

                            # Clamp to sane range so cutoff computation can't go weird.
                            if (not np.isfinite(psr_alpha)) or (psr_alpha <= 0.0) or (psr_alpha >= 1.0):
                                if config.get("print_cv_debug", False):
                                    print(f"[WARN] [CV][Reliability] psr_alpha out of range ({psr_alpha}); reset -> 0.10")
                                psr_alpha = 0.10

                            cutoff = float(
                                config.get(
                                    "floor_cv_final",
                                    config.get(
                                        "cv_sharpe_floor_final",
                                        cfg_f.get("floor_cv_final", _cd.get("floor_cv_final", -4.0)),
                                    ),
                                )
                            )

                            min_trades_block = int(config.get("cv_min_trades_per_block", 5))
                            min_indep_bets = int(
                                config.get(
                                    "cv_min_indep_bets_per_block",
                                    config.get("cv_min_indep_bets", 12),
                                )
                            )

                            reliability = {
                                "psr_alpha": psr_alpha,
                                "cutoff": cutoff,
                                "min_trades": min_trades_block,
                                "min_indep": min_indep_bets,
                            }


                            # Class distributions captured earlier in test_strategy
                            class_dists = getattr(
                                self, "_last_class_dists", {"raw": {}, "final": {}}
                            )

                            block_stats = {
                                # Back-compat key
                                "rows": int(_bars_eval),
                                # New: show raw val slice vs evaluated bar-grid rows
                                "rows_total": int(rows_i),
                                "rows_eligible": int(_bars_postfeat_elig),
                                "trades": int(trades),
                                "ar": float(active_rate),
                                "sr": float(sharpe),
                                "precision_intent": float(getattr(self, "_last_precision_intent", float("nan"))),
                                "intent_bars": int(getattr(self, "_last_intent_bars", 0) or 0),
                            }
 

                            if getattr(self, "_progress", None) and self._progress.verbose:
                                _print_block_summary(
                                    block_id=j,
                                    calib_info=calib_info,
                                    gate_info=gate_info,
                                    reliability=reliability,
                                    class_dists=class_dists,
                                    block_stats=block_stats,
                                )
                        except Exception:
                            # Summary printing should never break CV
                            pass


                        # --- Interim Optuna reporting & pruning ---
                        try:
                            valid_now = [
                                s for s in block_scores if np.isfinite(s)
                            ]
                            if trial is not None and valid_now:
                                arr_t = np.asarray(valid_now, dtype=float)

                                # Recency-tilted interim mean
                                tail_t = float(
                                    config.get(
                                        "cv_tail_weight", 1.35
                                    )
                                )
                                if arr_t.size > 1:
                                    w_rec = np.array(
                                        [
                                            1.0
                                            + i
                                            * (
                                                (tail_t - 1.0)
                                                / max(
                                                    1,
                                                    arr_t.size - 1,
                                                )
                                            )
                                            for i in range(
                                                arr_t.size
                                            )
                                        ],
                                        dtype=float,
                                    )
                                else:
                                    w_rec = np.ones_like(arr_t)

                                interim = float(
                                    np.average(
                                        arr_t, weights=w_rec
                                    )
                                )

                                step_idx = len(block_scores)
                                trial.report(interim, step=step_idx)

                                if (
                                    os.getenv(
                                        "MLB_DISABLE_OPTUNA_PRUNING",
                                        "0",
                                    )
                                    != "1"
                                ):
                                    relax = float(
                                        config.get(
                                            "cv_prune_relax", 1.0
                                        )
                                    )
                                    relax = max(
                                        0.0, min(relax, 1.0)
                                    )

                                    if relax > 0.0:
                                        base_min_k = float(
                                            config.get(
                                                "prune_min_folds",
                                                2,
                                            )
                                        )
                                        base_abs_fl = float(
                                            config.get(
                                                "prune_abs_floor_sr",
                                                -8.0,
                                            )
                                        )
                                        base_iqr_m = float(
                                            config.get(
                                                "prune_iqr_mult",
                                                0.75,
                                            )
                                        )

                                        k_done = int(
                                            arr_t.size
                                        )
                                        min_k = max(
                                            1,
                                            int(
                                                round(
                                                    base_min_k
                                                    * relax
                                                )
                                            ),
                                        )

                                        if base_abs_fl < 0:
                                            abs_fl = (
                                                base_abs_fl
                                                / max(
                                                    relax,
                                                    1e-6,
                                                )
                                            )
                                        else:
                                            abs_fl = (
                                                base_abs_fl
                                                * relax
                                            )

                                        iqr_m = (
                                            base_iqr_m
                                            * relax
                                        )

                                        if k_done >= 3:
                                            q1, q3 = (
                                                np.percentile(
                                                    arr_t,
                                                    [25, 75],
                                                )
                                            )
                                        else:
                                            q1 = float(
                                                np.min(
                                                    arr_t
                                                )
                                            )
                                            q3 = float(
                                                np.max(
                                                    arr_t
                                                )
                                            )

                                        iqr = max(
                                            1e-12,
                                            (float(q3) - float(q1)),
                                        )
                                        rel_fl = float(
                                            np.median(
                                                arr_t
                                            )
                                        ) - iqr_m * iqr
                                        gate = max(
                                            abs_fl, rel_fl
                                        )

                                        if (
                                            k_done
                                            >= min_k
                                        ) and (
                                            interim
                                            < gate
                                        ):
                                            import optuna as _opt
                                            if bool(config.get("cv_strict_pruning", False)):
                                                # Keep legacy behavior (abort whole trial)
                                                raise _opt.TrialPruned(
                                                    "Pruned early: "
                                                    f"interim={interim:.4f} "
                                                    f"< gate={gate:.4f} "
                                                    f"at step={step_idx}"
                                                )
                                            else:
                                                # Downgrade to fold-level invalidation; the outer
                                                # exception handler will convert this to a NaN score
                                                # and keep evaluating remaining blocks.
                                                raise RuntimeError(
                                                    "FoldPrunedByGate: "
                                                    f"interim={interim:.4f} "
                                                    f"< gate={gate:.4f} "
                                                    f"at step={step_idx}"
                                                )

                                # Also honor Optuna's own pruner
                                import optuna as _opt

                                if trial.should_prune():
                                    raise _opt.TrialPruned(
                                        "Pruned by scheduler "
                                        f"at step={step_idx} "
                                        f"with interim={interim:.4f}"
                                    )

                        except Exception as e:
                            # Propagate real TrialPruned
                            try:
                                import optuna as _opt
                                if isinstance(e, _opt.TrialPruned):
                                    raise
                            except Exception:
                                pass

                            # Normal block error -> mark invalid and continue
                            _cv_penalty(
                                "MiniBlockCV exception during block evaluation",
                                split_start=int(split),
                                error=str(e),
                            )
                            block_scores.append(float("nan"))
                            block_active_rates.append(0.0)
                            block_trades.append(0)
                            block_eff_conf.append(float("nan"))
                            block_sharpe.append(float("nan"))
                            block_psr.append(float("nan"))
                            block_neff.append(float("nan"))
                            _is_pruned = ("TrialPruned" in type(e).__name__) or ("Pruned" in str(e))
                            block_pruned.append(_is_pruned)
                            block_all_hold.append(False)
                            pred_cards.append({})
                            block_reasons.append("Pruned" if _is_pruned else "Exception")
                            _early_structural_prune_if_hopeless()
                            continue
                        
                        
                    # -------------------------------
                    # ALIGNMENT TRIPWIRE (debug only)
                    # -------------------------------
                    if bool(config.get("print_cv_debug", False)):
                        L = len(block_scores)
                        for name, lst in [
                            ("block_active_rates", block_active_rates),
                            ("block_trades", block_trades),
                            ("block_rows", block_rows),
                            ("block_sharpe", block_sharpe),
                            ("block_psr", block_psr),
                            ("block_neff", block_neff),
                            ("block_pruned", block_pruned),
                            ("block_all_hold", block_all_hold),
                            ("block_reasons", block_reasons),
                            ("pred_cards", pred_cards),
                        ]:
                            if len(lst) != L:
                                raise RuntimeError(
                                    f"[CV][ALIGNMENT_BUG] {name} len={len(lst)} != block_scores len={L}"
                                )


                    # --------------------------------------------
                    # Active folds & coverage (pre-aggregator)
                    # --------------------------------------------
                    M_gate = _M_gate_eff
                    L_gate = _L_gate_eff
                    r_min  = _r_min_eff

                    K_plan = int(config.get("cv_blocks", len(block_scores) or 5))
                    N_use  = min(len(block_scores), K_plan)

                    active_mask = [
                        (i < len(block_trades))
                        and (int(block_trades[i]) >= M_gate)
                        and (float(block_active_rates[i]) >= r_min)
                        for i in range(N_use)
                    ]
                    active_folds = int(sum(1 for a in active_mask if a))

                    # -------------------------------
                    # Single place to print mini-table
                    # -------------------------------
                    def _print_mini_tables():
                        """
                        Compact MiniBlockCV overview:
                        - Always 1 row per planned fold.
                        - Uses tracked raw Sharpe (block_sharpe) and PSR (block_psr).
                        - Penalized scores (block_scores) are internal to Optuna.
                        """
                        if table_mode == "off":
                            return

                        # How many folds we intended to have
                        planned_k = int(config.get("cv_blocks", 0)) or len(val_ends_ts) or len(block_scores) or 5
                        N_tbl = planned_k

                        rows_over = []

                        for i in range(N_tbl):

                            # Prefer fold_records (slot-stable) over parallel arrays.
                            fr = None
                            try:
                                fr = fold_records[i] if (i < len(fold_records)) else None
                            except Exception:
                                fr = None

                            # IMPORTANT: "has_data" must reflect whether this fold actually ran,
                            # not whether a parallel array happened to have an entry.
                            if isinstance(fr, dict):
                                rows_i_probe = fr.get("val_rows", 0) or 0
                                tr_rows_probe = fr.get("train_rows", 0) or 0
                                pruned_probe = bool(fr.get("pruned", False))
                                # Treat as "has data" if it produced rows or was explicitly pruned with a reason.
                                has_data = (int(rows_i_probe) > 0) or (int(tr_rows_probe) > 0) or pruned_probe or bool(fr.get("reason", ""))
                            else:
                                has_data = (i < len(block_scores))

                            if isinstance(fr, dict):
                                # IMPORTANT: when fold_records exists, the overview must be driven ONLY by it.
                                # Never index parallel arrays here (that's the historic drift bug).
                                sc = fr.get("score", float("nan"))
                                tr = fr.get("trades", float("nan"))
                                ar = fr.get("active_rate", float("nan"))
                                sh = fr.get("sharpe", float("nan"))
                                ps = fr.get("psr", float("nan"))
                                rows_i = fr.get("val_rows", 0) or 0
                                tr_rows = fr.get("train_rows", 0) or 0

                                # NEW: denominators (must exist in both branches)
                                pf_total  = int(fr.get("post_feature_bars_total", rows_i) or rows_i)
                                pf_elig   = int(fr.get("post_feature_eligible", pf_total) or pf_total)
                                eval_bars = int(fr.get("eval_bars", pf_elig) or pf_elig)

                                # NEW: intent precision + intent bars
                                pint = fr.get("precision_intent", float("nan"))
                                nint = fr.get("intent_bars", 0) or 0

                                # tolerate older/newer key variants
                                vstart = fr.get("vstart", fr.get("val_start", None))
                                vend   = fr.get("vend",   fr.get("val_end", None))

                                reason = fr.get("reason", "") or ""
                                pruned = bool(fr.get("pruned", False))

                            else:
                                sc = block_scores[i]        if has_data and i < len(block_scores)       else float("nan")
                                tr = block_trades[i]        if has_data and i < len(block_trades)       else float("nan")
                                ar = block_active_rates[i]  if has_data and i < len(block_active_rates) else float("nan")
                                sh = block_sharpe[i]        if has_data and i < len(block_sharpe)       else float("nan")
                                ps = block_psr[i]           if has_data and i < len(block_psr)          else float("nan")
                                rows_i = block_rows[i]      if has_data and i < len(block_rows)         else 0
                                tr_rows = block_train_rows[i] if has_data and i < len(block_train_rows) else 0

                                # Defaults (fallback when fold_records[i] is missing / non-dict)
                                pf_total  = int(rows_i) if rows_i else 0
                                pf_elig   = int(pf_total)
                                eval_bars = int(pf_elig)

                                # NEW: intent precision fallback arrays
                                pint = block_precision_intent[i] if has_data and i < len(block_precision_intent) else float("nan")
                                nint = block_intent_bars[i] if has_data and i < len(block_intent_bars) else 0

                                # val_start / val_end: use captured fold timestamps for correct alignment
                                vstart = val_starts_ts_cv[i] if i < len(val_starts_ts_cv) else None
                                vend   = val_ends_ts_cv[i]   if i < len(val_ends_ts_cv)   else (val_ends_ts[i] if i < len(val_ends_ts) else None)

                                # Human-readable status
                                reason = block_reasons[i] if i < len(block_reasons) else ""
                                pruned = block_pruned[i] if i < len(block_pruned) else False

                            if not has_data:
                                st = "[BLOCK] NO DATA / PRUNED"
                            elif pruned:
                                st = f"[BLOCK] {reason or 'Pruned'}"
                            elif reason:
                                # soft issues / diagnostics
                                if "Bad" in reason or "SRNaN" in reason:
                                    st = f"[BAD] {reason}"
                                else:
                                    st = f"[BLOCK] {reason}"
                            else:
                                st = (f"[WARN] {reason}" if reason else "[OK] OK")

                            # PSR column: use stored block_psr (based on raw Sharpe & n_eff)
                            if has_data and ps is not None and np.isfinite(ps):
                                psr_str = f"{float(ps):.3f}"
                            else:
                                psr_str = "--"

                            rows_over.append([
                                i + 1,
                                (str(vstart) if vstart is not None else "--"),
                                (str(vend)   if vend   is not None else "--"),
                                int(tr_rows) if tr_rows else 0,
                                int(rows_i) if rows_i else 0,
                                int(pf_total) if pf_total else 0,
                                int(pf_elig) if pf_elig else 0,
                                int(eval_bars) if eval_bars else 0,
                                int(tr) if tr is not None and np.isfinite(tr) else 0,
                                (f"{float(ar):.3f}" if ar is not None and np.isfinite(ar) else "--"),
                                (f"{float(pint):.3f}" if pint is not None and np.isfinite(pint) else "--"),
                                (int(nint) if nint else 0),
                                (f"{float(sh):.3f}" if sh is not None and np.isfinite(sh) else "--"),  # SR = raw Sharpe
                                psr_str,                                                                # PSR(raw SR)
                                st,
                            ])

                        if rows_over:
                            _title = "[DATE]  Monthly-roll CV overview" if bool(getattr(self, "_cv_used_monthly_last", False)) else "[TEST] Mini-block overview"
                            log_print(
                                _fmt_table(
                                    ["#", "val_start", "val_end", "train_rows", "val_rows", "pf_total", "pf_elig", "eval_bars", "trades", "active", "PrecInt", "nInt", "SR", "PSR", "status"],
                                    rows_over,
                                    title=_title
                                ),
                                level="COMPACT",
                            )

                        if table_verbose:
                            for i in range(N_tbl):
                                # IMPORTANT: if fold_records exists for this slot, drive the verbose
                                # card from it (never from parallel arrays).
                                fr = None
                                try:
                                    fr = fold_records[i] if (i < len(fold_records)) else None
                                except Exception:
                                    fr = None

                                if isinstance(fr, dict):
                                    sc = fr.get("score", float("nan"))
                                    tr = fr.get("trades", float("nan"))
                                    ar = fr.get("active_rate", float("nan"))
                                    all_hold_i = bool(fr.get("all_hold", False))
                                    pruned_i = bool(fr.get("pruned", False))
                                    reason_i = str(fr.get("reason", "") or "")

                                    # tolerate older/newer key variants
                                    vstart = fr.get("vstart", fr.get("val_start", None))
                                    vend   = fr.get("vend",   fr.get("val_end", None))
                                else:
                                    sc = block_scores[i] if i < len(block_scores) else float("nan")
                                    tr = block_trades[i] if i < len(block_trades) else 0
                                    ar = block_active_rates[i] if i < len(block_active_rates) else float("nan")
                                    all_hold_i = (block_all_hold[i] if i < len(block_all_hold) else False)
                                    pruned_i = (bool(block_pruned[i]) if i < len(block_pruned) else False)
                                    reason_i = (block_reasons[i] if i < len(block_reasons) else "")

                                    try:
                                        vstart = (val_starts_ts_cv[i] if i < len(val_starts_ts_cv) else val_starts_ts[i])
                                    except Exception:
                                        vstart = None
                                    try:
                                        vend = (val_ends_ts_cv[i] if i < len(val_ends_ts_cv) else val_ends_ts[i])
                                    except Exception:
                                        vend = None

                                st = _status_for_block(
                                    tr,
                                    sc,
                                    ar,
                                    all_hold=all_hold_i,
                                    pruned=pruned_i,
                                )
                                try:
                                    if reason_i:
                                        st = f"[BLOCK] {reason_i}"
                                except Exception:
                                    pass

                                if table_only_failures and st == "[OK] OK":
                                    continue

                                card = pred_cards[i] if i < len(pred_cards) else {}

                                try:
                                    vstart = str(vstart).split("+")[0].replace("T", " ") if vstart is not None else ""
                                except Exception:
                                    vstart = ""
                                try:
                                    vend = str(vend).split("+")[0].replace("T", " ") if vend is not None else ""
                                except Exception:
                                    vend = ""

                                # PSR for verbose card: same logic as above
                                psr_card = card.get("psr", None)
                                psr_str = "--"
                                try:
                                    if psr_card is not None and psr_card != "--" and np.isfinite(float(psr_card)):
                                        psr_str = f"{float(psr_card):.3f}"
                                    else:
                                        indep = card.get("indep_bets", None)
                                        if (
                                            sc is not None
                                            and np.isfinite(sc)
                                            and indep not in (None, "--")
                                        ):
                                            n_eff = int(max(2, round(float(indep))))
                                            psr_val = _psr(float(sc), n_eff, sr_bench=0.0)
                                            if np.isfinite(psr_val):
                                                psr_str = f"{psr_val:.3f}"
                                except Exception:
                                    psr_str = "--"

                                rows_card = [
                                    ["val_start",        vstart],
                                    ["val_end",          vend],
                                    ["label_counts",     _fmt_dict(card.get("label_counts"))],
                                    ["thr",              card.get("thr", params.get("label_threshold", None))],
                                    ["test_len",         card.get("test_len", (fr.get("val_rows", "") if isinstance(fr, dict) else (block_rows[i] if i < len(block_rows) else "")))],
                                    ["raw_preds",        _fmt_dict(card.get("raw_preds"))],
                                    ["decoded_before",   _fmt_dict(card.get("decoded_before"))],
                                    ["final_after_thr",  _fmt_dict(card.get("final_after_thr"))],
                                    ["eff_conf",         (f"{float(block_eff_conf[i]):.3f}" if i < len(block_eff_conf) and np.isfinite(block_eff_conf[i]) else "--")],
                                    ["indep_bets",       card.get("indep_bets", "--")],
                                    ["psr",              psr_str],
                                    ["status",           st],
                                ]

                                print(
                                    _fmt_table(
                                        ["field", "value"],
                                        rows_card,
                                        title=f"[TARGET] Predictions -- Block {i+1:02d}",
                                    )
                                )

                    # Print once before exit
                    _print_mini_tables()

                    # Gate on active folds:
                    # - If literally ZERO valid folds -> structural failure -> prune.
                    # - Otherwise: KEEP the trial; few-fold coverage is treated as "very noisy/bad" but informative.
                    if cv_relax > 0.0 and active_folds == 0:
                        msg = (f"[MiniBlockCV:GATE_FAIL] active_folds=0/{K_plan} | "
                            f"M={M_gate}, r_min={r_min:.3f}")
                        if bool(config.get("print_cv_debug", False)):
                            print(msg + " -> TrialPruned (no usable folds)")
                        import optuna as _opt
                        raise _opt.TrialPruned(msg)


                    # Soft gate: if too few active folds, just warn; Optuna will downweight this trial.
                    if active_folds < L_gate:
                        if bool(config.get("print_cv_debug", False)):
                            print(f"[MiniBlockCV:GATE_SOFT] active_folds={active_folds}/{K_plan} "
                                  f"<{L_gate} -> keeping trial with weak evidence.")


                    # --- Aggregate fold scores (configurable) ---
                    # -------------------------------
                    # CV-level intent precision aggregation (for Optuna pruning / logs)
                    # -------------------------------
                    try:
                        _p_arr = np.asarray(block_precision_intent[:K_plan], dtype=float)
                        _n_arr = np.asarray(block_intent_bars[:K_plan], dtype=float)
                        _m = np.isfinite(_p_arr) & (_n_arr > 0)
                        _n_sum = float(np.sum(_n_arr[_m])) if _m.size else 0.0
                        if _n_sum > 0:
                            precision_intent_cv = float(np.sum(_p_arr[_m] * _n_arr[_m]) / _n_sum)
                            intent_bars_cv = int(np.sum(_n_arr[_m]))
                        else:
                            precision_intent_cv = float("nan")
                            intent_bars_cv = 0
                        setattr(self, "_last_precision_intent_cv", precision_intent_cv)
                        setattr(self, "_last_intent_bars_cv", intent_bars_cv)
                        if trial is not None:
                            try:
                                trial.set_user_attr("precision_intent_cv", precision_intent_cv)
                                trial.set_user_attr("intent_bars_cv", intent_bars_cv)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    
                    arr_all    = np.asarray(block_scores[:K_plan], dtype=float)
                    valid_mask = np.isfinite(arr_all)
                    K_all      = int(arr_all.size)
                    k_valid    = int(valid_mask.sum())
                    coverage   = float(k_valid / max(1, K_all))
                    
                    # --- Hard requirement: minimum valid folds (prevents fold-gaming) ---
                    # Prefer an integer gate over fraction for small K (e.g., K=4).
                    # If set, this is an unrelaxed structural requirement.
                    try:
                        _used_monthly = bool(getattr(self, "_cv_used_monthly_last", False))
                    except Exception:
                        _used_monthly = False

                    min_valid_folds = config.get(
                        "cv_min_valid_folds_monthly" if _used_monthly else "cv_min_valid_folds",
                        None
                    )
                    prune_on_min_valid_folds = bool(config.get("cv_prune_on_min_valid_folds", True))
                    if min_valid_folds is not None:
                        try:
                            min_valid_folds = int(min_valid_folds)
                        except Exception:
                            min_valid_folds = None

                    if min_valid_folds is not None and min_valid_folds > 0:
                        if k_valid < min_valid_folds:
                            msg = (f"[MiniBlockCV:GATE_MIN_VALID_FOLDS] valid={k_valid}/{K_all} "
                                   f"< min_folds={min_valid_folds} -> PRUNE")
                            if bool(config.get("print_cv_debug", False)):
                                print(msg)
                            if prune_on_min_valid_folds:
                                import optuna as _opt
                                raise _opt.TrialPruned(msg)
                            else:
                                return float("nan")
                    
                    
                    # --- Hard requirement: minimum valid-fraction (unrelaxed) ---
                    min_valid_frac = float(config.get("cv_min_valid_fraction", 0.80))
                    prune_low_valid = bool(config.get("cv_prune_on_low_valid_fraction", True))
                    if np.isfinite(min_valid_frac) and min_valid_frac > 0.0:
                        if coverage < min_valid_frac:
                            if prune_low_valid:
                                msg = (f"[MiniBlockCV:GATE_MIN_VALID] valid={k_valid}/{K_all} "
                                       f"(cov={coverage:.2f}) < min={min_valid_frac:.2f} -> PRUNE")
                                if bool(config.get("print_cv_debug", False)):
                                    print(msg)
                                try:
                                    import optuna as _opt
                                    raise _opt.TrialPruned(msg)
                                except Exception:
                                    return float("nan")
                            else:
                                # We'll keep the trial but apply a penalty later (after aggregation).
                                if bool(config.get("print_cv_debug", False)):
                                    print(f"[MiniBlockCV:GATE_MIN_VALID] cov={coverage:.2f} < {min_valid_frac:.2f} -> "
                                          f"KEEP & PENALIZE (cv_invalid_share_penalty x invalid_folds)")

                    min_cov_base = float(config.get("cv_min_coverage", 0.80))

                    if cv_relax <= 0.0:
                        # Fully relaxed: do NOT hard-prune on coverage.
                        # We let NaNs / low coverage propagate as "bad scores" instead.
                        min_cov = 0.0
                    else:
                        # Larger base threshold = stricter; scale down by relaxation.
                        min_cov = max(0.0, min_cov_base * cv_relax)

                    if cv_relax > 0.0:
                        # Normal behavior: if nothing usable OR too little coverage -> prune the trial.
                        if k_valid == 0 or coverage < min_cov:
                            msg = f"[MiniBlockCV] coverage={coverage:.2f} < min_cov={min_cov:.2f}"
                            if bool(config.get("print_cv_debug", False)):
                                print(msg + " -> TrialPruned")
                            if optuna is not None:
                                raise optuna.TrialPruned(msg)
                            # If Optuna is not available, fall back to "hopeless" score.
                            return float("nan")
                    else:
                        # cv_prune_relax == 0.0 -> no coverage-based pruning.
                        # If k_valid == 0, we just log and continue; downstream agg will yield NaN.
                        if k_valid == 0 and bool(config.get("print_cv_debug", False)):
                            print("[MiniBlockCV] cv_prune_relax=0.0 & k_valid=0 -> no hard prune (returning NaN later).")

                    vals = np.sort(arr_all[valid_mask])
                    trim_frac = float(config.get("cv_trim_frac", 0.0))
                    trim_n = int(round(k_valid * trim_frac))
                    if trim_n > 0 and (2 * trim_n) < k_valid:
                        vals = vals[trim_n:-trim_n]


                    agg_mode = str(config.get("cv_agg_mode", "tanh_mean")).lower()
                    if agg_mode == "tanh_mean":
                        s0 = float(config.get("cv_tanh_s", 10.0))
                        if (not np.isfinite(s0)) or (s0 <= 0):
                            s0 = 0.0
                        elif s0 < 1.0:
                            print(f"[CVCombine] cv_tanh_s too small ({s0}); clamping to 1.0")
                            s0 = 1.0                    
                        if s0 > 0:
                            vals = s0 * np.tanh(vals / s0)
                        final_score = float(np.nanmean(vals)) if vals.size else float("nan")
                    elif agg_mode == "mean":
                        final_score = float(np.nanmean(vals)) if vals.size else float("nan")
                    elif agg_mode == "median":
                        final_score = float(np.nanmedian(vals)) if vals.size else float("nan")
                    elif agg_mode == "psr_weighted_tanh_mean":
                        # Values (already tanh-capped below if s0>0 like in tanh_mean)
                        s0 = float(config.get("cv_tanh_s", 10.0))
                        if (not np.isfinite(s0)) or (s0 <= 0):
                            s0 = 0.0
                        elif s0 < 1.0:
                            print(f"[CVCombine] cv_tanh_s too small ({s0}); clamping to 1.0")
                            s0 = 1.0
                        if (not np.isfinite(s0)) or (s0 <= 0):
                            s0 = 0.0
                        elif s0 < 1.0:
                            print(f"[CVCombine] cv_tanh_s too small ({s0}); clamping to 1.0")
                            s0 = 1.0
                        vals_full = arr_all.copy()
                        if s0 > 0:
                            vals_full = s0 * np.tanh(vals_full / s0)
                        vals = vals_full[valid_mask]

                        # Build weights = PSR^power  * optional recency tilt
                        psr_power = float(config.get("cv_psr_power", 1.0))
                        # Align PSR weights with the ACTUAL number of scored folds (K_all),
                        # not the planned K_plan. This avoids shape mismatches when some
                        # folds failed before producing PSR.
                        psr_arr_all = np.asarray(
                            (block_psr[:K_all] + [np.nan] * max(0, K_all - len(block_psr))),
                            dtype=float,
                        )

                        w_psr = np.power(np.clip(psr_arr_all, 0.0, 1.0), psr_power)

                        # Defensive alignment: if something is off, truncate to the common length.
                        if w_psr.shape[0] != valid_mask.shape[0]:
                            common = min(w_psr.shape[0], valid_mask.shape[0])
                            w_psr = w_psr[:common]
                            valid_mask = valid_mask[:common]
                            vals = vals_full[:common][valid_mask]

                        w_psr = w_psr[valid_mask]

                        # Optional recency tilt (monotone increasing weights)
                        use_rec = bool(config.get("cv_use_recency_weight", True))
                        if use_rec and vals.size > 1:
                            rec_pow = float(config.get("cv_recency_power", 1.40))
                            w_rec = np.array(
                                [1.0 + i * ((rec_pow - 1.0) / (vals.size - 1)) for i in range(vals.size)],
                                dtype=float
                            )
                        else:
                            w_rec = np.ones_like(vals)

                        # Combine & guard
                        w = w_psr * w_rec
                        if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0:
                            final_score = float(np.nanmean(vals)) if vals.size else float("nan")
                        else:
                            final_score = float(np.average(vals, weights=w)) if vals.size else float("nan")
                    else:
                        final_score = float(np.nanmean(vals)) if vals.size else float("nan")

                    # --- Optional invalid-share penalty (only if we didn't prune earlier) ---
                    try:
                        min_valid_frac = float(config.get("cv_min_valid_fraction", 0.80))
                        prune_low_valid = bool(config.get("cv_prune_on_low_valid_fraction", True))
                        if not prune_low_valid and np.isfinite(final_score):
                            invalid_folds = int(max(0, K_all - k_valid))
                            if k_valid < int(math.ceil(min_valid_frac * max(1, K_all))):
                                per_fold_pen = float(config.get("cv_invalid_share_penalty", 0.5))
                                penalty = per_fold_pen * float(invalid_folds)
                                final_score = float(final_score - penalty)
                                if bool(config.get("print_cv_debug", False)):
                                    print(f"[MiniBlockCV:INVALID_PEN] invalid={invalid_folds} x {per_fold_pen:.2f} "
                                        f"-> -{penalty:.2f} -> final={final_score:.4f}")
                    except Exception:
                        pass

                    # --- CSCV / PBO temporal stability penalty ---
                    pbo_weight = float(config.get("cv_cscv_penalty_weight", 0.2))
                    if pbo_weight > 0.0:
                        v = np.asarray(arr_all[valid_mask], dtype=float)
                        if v.size >= 3 and np.all(np.isfinite(v)):
                            # rank correlation between fold performance and time (proxy for CSCV/PBO stability)
                            ranks = np.argsort(np.argsort(v))
                            t_idx = np.arange(v.size)
                            corr = np.corrcoef(ranks, t_idx)[0, 1]

                            # Optional disqualification (very unstable through time)
                            min_corr = float(config.get("cv_cscv_min_rank_corr", np.nan))
                            if np.isfinite(min_corr) and (float(corr) < min_corr) and bool(config.get("cv_strict_pruning", False)):
                                if bool(config.get("print_cv_debug", False)):
                                    print(f"[CSCV-PBO] corr={corr:.3f} < min={min_corr:.2f} -> DISQUALIFY (strict)")
                                import optuna as _opt
                                raise _opt.TrialPruned("CSCV/PBO: rank-corr below minimum (strict)")

                            else:

                                # Direction-safe penalty:
                                # - maximize: always DECREASE score by subtracting abs(score)*penalty
                                # - minimize: always INCREASE score by adding abs(score)*penalty (optional; or disable)
                                direction = str(config.get("optuna_direction", "maximize")).lower().strip()
                                is_max = (direction != "minimize")

                                try:
                                    base_before_pbo = float(final_score)
                                except Exception:
                                    base_before_pbo = None

                                # Robust corr handling
                                try:
                                    corr_f = float(corr)
                                except Exception:
                                    corr_f = float("nan")
                                if not np.isfinite(corr_f):
                                    corr_f = 1.0  # corr unavailable => skip CSCV/PBO penalty (pbo_proxy=0)

                                pbo_proxy = max(0.0, 1.0 - corr_f)
                                pen_frac = float(pbo_weight) * float(pbo_proxy)

                                # clip to [0, 1] to avoid pathological amplification
                                if pen_frac < 0.0:
                                    pen_frac = 0.0
                                elif pen_frac > 1.0:
                                    pen_frac = 1.0

                                pen_amt = 0.0
                                if (base_before_pbo is not None) and np.isfinite(base_before_pbo) and (pen_frac > 0.0):
                                    pen_amt = abs(base_before_pbo) * pen_frac
                                    if is_max:
                                        final_score = float(base_before_pbo - pen_amt)
                                    else:
                                        final_score = float(base_before_pbo + pen_amt)  # or: final_score = base_before_pbo
                                # else: leave final_score unchanged

                                # Persist audit attrs (best-effort; never crash objective)
                                try:
                                    if trial is not None:
                                        trial.set_user_attr("cv_cscv_rank_corr", float(corr_f))
                                        trial.set_user_attr("cv_cscv_pbo_proxy", float(pbo_proxy))
                                        trial.set_user_attr("cv_cscv_pen_frac", float(pen_frac))
                                        trial.set_user_attr("cv_cscv_pen_amount", float(pen_amt))
                                        trial.set_user_attr("cv_cscv_pen_direction", direction)
                                except Exception:
                                    pass

                                if bool(config.get("print_cv_debug", False)):
                                    try:
                                        b = float(base_before_pbo)
                                    except Exception:
                                        b = float("nan")
                                    print(
                                        f"[CSCV-PBO] corr={corr_f:.3f} proxy={pbo_proxy:.3f} "
                                        f"pen_frac={pen_frac:.3f} pen_amt={pen_amt:.4f} "
                                        f"dir={direction} base={b:.4f} -> final={float(final_score):.4f}"
                                    )

                    if bool(config.get("print_cv_fold_scores", False)) or bool(config.get("print_cv_debug", False)):
                        _prec = int(config.get("cv_log_precision", 8))
                        _raw  = np.round(arr_all[valid_mask], 4).tolist()
                        _fin  = f"{final_score:.{_prec}f}"
                        prefix = "CVCombine" if bool(getattr(self, "_cv_used_monthly_last", False)) else "MiniBlockCV"
                        kept_ids = [i + 1 for i, ok in enumerate(list(valid_mask)) if bool(ok)]
                        print(
                            f"[{prefix}:{agg_mode}] kept_folds={kept_ids}/{K_all} folds={_raw} "
                            f"| k={k_valid}/{K_all} (cov={coverage:.2f}) "
                            f"| trim_frac={trim_frac:.2f} -> final={_fin}"
                        )

                # --- Store diagnostics (robust to partial folds) ---
                try:
                    if trial is not None:
                        # Build per-regime table: per-fold rows + median aggregate
                        try:
                            import numpy as _np
                            names = {0: "sideways", 1: "trend", 2: "volatile"}
                            per_fold_rows = []
                            for fidx, df_f in enumerate(fold_eval_frames, start=1):
                                row = {"FOLD": fidx}
                                _cols = getattr(df_f, "columns", [])
                                _rid_col = "regime_id" if ("regime_id" in _cols) else ("regime_id_diag" if ("regime_id_diag" in _cols) else None)
                                if (_rid_col is None) and fidx == 1:
                                    try:
                                        log_print("[MiniBlock][Diag] 'regime_id' not found in fold eval frames; per-regime stats assume all TREND (rid=1).", level="COMPACT")
                                    except Exception:
                                        pass

                                for rid, rname in names.items():
                                    if _rid_col is not None:
                                        sub = df_f[df_f[_rid_col] == rid]
                                    else:
                                        sub = df_f if rid == 1 else df_f.iloc[0:0]

                                    if len(sub) == 0:
                                        row[rname] = {"cstrategy": float("nan"), "sharpe": float("nan"), "trades": 0, "active_rate": float("nan")}
                                    else:
                                        try:
                                            _sub_eval = sub.copy()
                                            try:
                                                if bool(getattr(self, "trading_costs", False)):
                                                    try:
                                                        _sub_eval.attrs["features_config"] = dict(getattr(df_f, "attrs", {}).get("features_config", {}) or {})
                                                    except Exception:
                                                        _sub_eval.attrs["features_config"] = dict(getattr(self, "features_config", {}) or {})
                                                    _sub_eval = self._ensure_cost_columns(_sub_eval, _sub_eval.attrs.get("features_config", {}))
                                            except Exception:
                                                pass

                                            m = compute_full_evaluation_metrics(
                                                df=_sub_eval,
                                                trading_costs=self.trading_costs,
                                                slippage_factor=self.slippage_factor,
                                                eval_context=f"cv:diagnostic:per_regime_metrics:rid={rid}",
                                            )
                                            cstr = float(m[0]) if m is not None else float("nan")
                                            sr = float(m[3]) if m is not None else float("nan")

                                            if (m is not None) and (len(m) > 5) and (m[5] is not None):
                                                tr = int(m[5])
                                            elif "pred" in sub.columns:
                                                tr = int((sub["pred"].fillna(0) != 0).sum())
                                            elif "position_exec" in sub.columns:
                                                tr = int((sub["position_exec"].fillna(0) != 0).sum())
                                            else:
                                                tr = 0

                                            if "pred" in sub.columns:
                                                ar = float((sub["pred"].fillna(0) != 0).mean())
                                            elif "position_exec" in sub.columns:
                                                ar = float((sub["position_exec"].fillna(0) != 0).mean())
                                            else:
                                                ar = float("nan")

                                            row[rname] = {"cstrategy": cstr, "sharpe": sr, "trades": tr, "active_rate": ar}
                                        except Exception:
                                            if "pred" in sub.columns:
                                                tr = int((sub["pred"].fillna(0) != 0).sum())
                                            elif "position_exec" in sub.columns:
                                                tr = int((sub["position_exec"].fillna(0) != 0).sum())
                                            else:
                                                tr = 0
                                            row[rname] = {"cstrategy": float("nan"), "sharpe": float("nan"), "trades": tr, "active_rate": float("nan")}

                                per_fold_rows.append(row)

                            # median-aggregate across folds
                            agg = {}
                            for rid, rname in names.items():
                                vals = {"cstrategy": [], "sharpe": [], "trades": [], "active_rate": []}
                                for r in per_fold_rows:
                                    d = r[rname]
                                    vals["cstrategy"].append(d["cstrategy"])
                                    vals["sharpe"].append(d["sharpe"])
                                    vals["trades"].append(d["trades"])
                                    vals["active_rate"].append(d["active_rate"])
                                def _safe_med(lst):
                                    a = _np.asarray(lst, dtype=float)
                                    a = a[_np.isfinite(a)]
                                    return float(_np.nanmedian(a)) if a.size else float("nan")
                                agg[rname] = {
                                    "cstrategy": _safe_med(vals["cstrategy"]),
                                    "sharpe": _safe_med(vals["sharpe"]),
                                    "trades": int(_np.nanmedian(_np.asarray(vals["trades"], dtype=float))) if vals["trades"] else 0,
                                    "active_rate": _safe_med(vals["active_rate"]),
                                }

                            trial.set_user_attr("per_regime_cv_per_fold", per_fold_rows)
                            trial.set_user_attr("per_regime_cv_median", agg)

                            # Print per-regime CV table only in verbose mode
                            try:
                                if getattr(self, "_progress", None) and self._progress.verbose:
                                    log_print("\nPer-regime CV table (per-fold rows + median):", level="COMPACT")
                                    header = f"{'FOLD':>4} {'REGIME':<10} {'CSTRAT':>10} {'SHARPE':>8} {'TRADES':>8} {'ACTIVE%':>8}"
                                    log_print(header)
                                    for r in per_fold_rows:
                                        fidx = r["fold"]
                                        for rn in ("sideways", "trend", "volatile"):
                                            v = r[rn]
                                            ar_pct = (v["active_rate"] * 100) if (v["active_rate"] == v["active_rate"]) else float("nan")
                                            log_print(f"{fidx:4d} {rn:<10} {v['cstrategy']:10.4f} {v['sharpe']:8.3f} {int(v['trades']):8d} {ar_pct:8.2f}")
                                    log_print("-" * len(header))
                                    for rn in ("sideways", "trend", "volatile"):
                                        a = agg[rn]
                                        ar_pct = (a["active_rate"] * 100) if (a["active_rate"] == a["active_rate"]) else float("nan")
                                        log_print(f"{'MED':>4} {rn:<10} {a['cstrategy']:10.4f} {a['sharpe']:8.3f} {int(a['trades']):8d} {ar_pct:8.2f}")
                                    log_print("")
                            except Exception:
                                pass
                        except Exception as e:
                            try:
                                if LOG_MODE in {"COMPACT", "DEBUG"} or getattr(self, "debug", False):
                                    print(f"[MiniBlockCV] per-regime CV table failed: {type(e).__name__}: {e}")
                            except Exception:
                                pass


                        arr_tr = np.asarray(block_trades[:K_plan], dtype=float) if 'block_trades' in locals() else np.array([])
                        arr_ar = np.asarray(block_active_rates[:K_plan], dtype=float) if 'block_active_rates' in locals() else np.array([])

                        # Align masks defensively
                        if arr_tr.size and valid_mask.size:
                            m = min(arr_tr.size, valid_mask.size)
                            mask_tr = valid_mask[:m] & np.isfinite(arr_tr[:m])
                            med_tr = float(np.nanmedian(arr_tr[:m][mask_tr])) if mask_tr.any() else float('nan')
                        else:
                            med_tr = float('nan')

                        if arr_ar.size and valid_mask.size:
                            m = min(arr_ar.size, valid_mask.size)
                            mask_ar = valid_mask[:m] & np.isfinite(arr_ar[:m])
                            med_ar = float(np.nanmedian(arr_ar[:m][mask_ar])) if mask_ar.any() else float('nan')
                        else:
                            med_ar = float('nan')

                        trial.set_user_attr("trades_cv", med_tr)
                        trial.set_user_attr("active_rate_cv", med_ar)
                        
                        # Trade-intent precision (median across valid folds).
                        # Defined as P(correct direction | model chose to trade) on each fold.
                        try:
                            _prec_vals = []
                            for _fr in (fold_records or []):
                                if isinstance(_fr, dict):
                                    _prec_vals.append(float(_fr.get("precision_trade", float("nan"))))
                                else:
                                    _prec_vals.append(float("nan"))
                            _arr_p = np.asarray(_prec_vals, dtype=float)
                            if _arr_p.size and valid_mask.size:
                                _m = min(_arr_p.size, valid_mask.size)
                                _mask_p = valid_mask[:_m] & np.isfinite(_arr_p[:_m])
                                med_p = float(np.nanmedian(_arr_p[:_m][_mask_p])) if _mask_p.any() else float("nan")
                            else:
                                med_p = float("nan")
                        except Exception:
                            med_p = float("nan")
                        trial.set_user_attr("precision_trade_cv", med_p)

                        trial.set_user_attr("cv_k_valid", int(k_valid))
                        
                        # Attach intent-precision aggregates for Optuna pruning
                        try:
                            _pi = np.asarray(block_precision_intent, dtype=float)
                            _ni = np.asarray(block_intent_bars, dtype=float)
                            if _pi.size and valid_mask.size:
                                m = min(_pi.size, valid_mask.size)
                                _vm = valid_mask[:m]
                                mask_pi = _vm & np.isfinite(_pi[:m]) & np.isfinite(_ni[:m]) & (_ni[:m] > 0)
                                pi_cv = float(np.nanmedian(_pi[:m][mask_pi])) if mask_pi.any() else float("nan")
                                mask_ni = _vm & np.isfinite(_ni[:m]) & (_ni[:m] > 0)
                                ni_cv = int(np.nansum(_ni[:m][mask_ni])) if mask_ni.any() else 0
                            else:
                                pi_cv = float("nan")
                                ni_cv = 0
                            trial.set_user_attr("precision_intent_cv", float(pi_cv))
                            trial.set_user_attr("intent_bars_cv", int(ni_cv))
                        except Exception:
                            pass
                        # Diagnostics-only: aggregate PSR across CV blocks for Top-N payload
                        try:
                            _psr_vals = np.asarray(block_psr, dtype=float)
                            _psr_vals = _psr_vals[np.isfinite(_psr_vals)]
                            trial.set_user_attr("psr", float(np.nanmedian(_psr_vals)) if _psr_vals.size else float("nan"))
                        except Exception:
                            pass
                except Exception as ex:
                    try:
                        import optuna as _opt
                        if isinstance(ex, _opt.TrialPruned):
                            raise
                    except Exception:
                        pass

                    self._store_fold_cv_data(final_score, block_scores, valid_mask, coverage, calib_brier_sum, calib_n_samples, block_cov_thr)
                    return final_score
                
                # --- Aggregate CV-derived coverage thresholds across blocks (median) ---
                try:
                    if block_cov_thr:
                        _agg_thr = float(np.nanmedian(np.asarray(block_cov_thr, dtype=float)))
                        setattr(self, "_cv_coverage_thr_agg", _agg_thr)
                        if self._progress.verbose:
                            print(
                                f"[CV] Aggregated coverage conf_thr (median of blocks) = {_agg_thr:.3f} | "
                                f"_cv_coverage_thr_agg={_agg_thr:.6f}"
                            )

                except Exception as _ee:
                    if self._progress.verbose:
                        print(f"[CV] Coverage threshold aggregation skipped: {_ee}")

                # --- Attach calibration metrics to Optuna trial 
                if trial is not None and calib_n_samples > 0:
                    try:
                        brier_avg = calib_brier_sum / calib_n_samples
                        nll_avg   = calib_nll_sum   / calib_n_samples
                        trial.set_user_attr("brier", float(brier_avg))
                        trial.set_user_attr("nll",   float(nll_avg))
                        if self._progress.verbose:
                            print(
                                f"[CV-Calib] Trial {trial.number}: "
                                f"brier={brier_avg:.6f} | nll={nll_avg:.6f} "
                                f"| n={calib_n_samples}"
                            )
                    except Exception as _e:
                        if self._progress.verbose:
                            print(f"[CV-Calib] Failed to attach calibration metrics to trial: {_e}")

                # Successful mini-block CV: return the aggregated score
                self._store_fold_cv_data(final_score, block_scores, valid_mask, coverage, calib_brier_sum, calib_n_samples, block_cov_thr)
                return float(final_score)

            finally:
                try:
                    self.free(release_data=False)
                except Exception:
                    pass
                setattr(self, "_in_optuna_cv", _old_cv_flag)
                # Restore CV/debug flags to prevent CV-mode leakage into real_trading_simulation
                try:
                    self._in_cv = _prev_cv
                    self._dbg_first_bars = _prev_dbg
                except Exception:
                    pass



        # Run a single study now and cache Top-5 on self
        month_ix_local = int(config.get("month_ix", 1))
        month_graphs_dir_local = config.get("month_graphs_dir", None) or None

        # Call tuner. If the tuner hasn't been patched to accept month_out_dir/month_ix yet,
        # this will gracefully fall back to the old signature.
        try:
            from inspect import signature as _sig
            _sig_params = set(_sig(run_optuna_tuning).parameters.keys())
        except Exception:
            _sig_params = set()

        # --- Determine Top-N size by model family (Classical/Deep/Ensemble/DQN) ---
        mt_local = str(config.get("model_type", getattr(self, "model_type", ""))).lower()
        cfg_local = getattr(self, "features_config", {}) or {}

        classical = {"logistic", "svm", "decision_tree", "random_forest", "xgboost"}
        rl        = {"cnn", "lstm", "transformer"}
        dqn       = {"dqn"}
        ensembles = {"ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost"}

        if mt_local in classical:
            rt_n = int(cfg_local.get("topN_classical", 4))
        elif mt_local in rl:
            rt_n = int(cfg_local.get("topN_deep", 3))
        elif mt_local in ensembles:
            rt_n = int(cfg_local.get("topN_ensemble", 2))
        elif mt_local in dqn:
            rt_n = int(cfg_local.get("topN_dqn", 2))
        else:
            rt_n = int(cfg_local.get("topN_default", 3))
        rt_n = max(1, rt_n)

        self._progress.set_n_trials(int(config.get("n_trials", 1)))

        # Wrap objective function for per-trial progress tracking
        _cv_objective = _single_study_cv
        def _progress_tracking_objective(train_data, params, min_train_window, val_window, trial=None, cv_config_override=None):
            result = _cv_objective(train_data, params, min_train_window, val_window, trial=trial, cv_config_override=cv_config_override)
            try:
                if trial is not None and hasattr(self, "_progress"):
                    best_val = float("nan")
                    try:
                        best_val = float(trial.study.best_value)
                    except Exception:
                        if result is not None and (isinstance(result, (int, float)) and math.isfinite(result)):
                            best_val = float(result)
                    fold_srs = getattr(self, "_last_fold_srs", None)
                    cv_res = getattr(self, "_last_cv_result", None)
                    self._progress.update_trial(trial.number + 1, best_val, fold_srs, cv_res)
            except Exception:
                pass
            return result

        _common_kwargs = dict(
            train_data=first_train_df,
            base_features=base_features_first,
            evaluate_cv_func=_progress_tracking_objective,
            cv_config=cv_config_first,
            models_to_test=models_to_test,
            n_trials=int(config.get("n_trials", 1)),
            return_top_n=rt_n,
            study=None,
            sampler_seed=int(self.features_config.get("run_seed", 0)) or None,
            max_hpo_duration_minutes=float(config.get("max_hpo_duration_minutes", 0)),
            sampler_method=str(config.get("hpo_sampler", "tpe")),
        )

        # Auto-enable two-phase HPO for models with 4+ tunable hyperparameters
        _auto_two_phase = False
        if model_type_local is not None and model_type_local != "":
            _n_tunable = sum(1 for v in SEARCH_SPACE.get(model_type_local, {}).values()
                             if isinstance(v, (list, tuple)))
            _auto_two_phase = _n_tunable >= 4

        if bool(config.get("hpo_two_phase", _auto_two_phase)):
            # ── TWO-PHASE HPO ──
            phase1_sampler = str(config.get("phase1_sampler", "cmaes"))
            phase1_trials  = int(config.get("phase1_trials", 30))
            phase2_trials  = int(config.get("phase2_trials", 15))
            phase2_top_n   = int(config.get("phase2_top_n", 5))

            kw1 = {**_common_kwargs, "sampler_method": phase1_sampler,
                   "n_trials": phase1_trials}
            if {"month_out_dir", "month_ix"} <= _sig_params:
                best1, score1, top5_1, study1, pool1 = run_optuna_tuning(
                    **kw1, month_out_dir=month_graphs_dir_local, month_ix=month_ix_local)
            else:
                best1, score1, top5_1, study1, pool1 = run_optuna_tuning(**kw1)

            from optuna.trial import TrialState
            valid = [t for t in (study1.trials or [])
                    if getattr(t, "value", None) is not None
                    and getattr(t, "state", None) == TrialState.COMPLETE]
            topN = sorted(valid, key=lambda t: t.value, reverse=True)[:phase2_top_n]

            if topN:
                from optuna.samplers import TPESampler as _TPE2
                _s2 = _TPE2(seed=kw1["sampler_seed"],
                           n_startup_trials=min(5, max(1, phase2_trials // 3)))
                study2 = optuna.create_study(direction="maximize", sampler=_s2)
                for t in topN:
                    study2.enqueue_trial(t.params)

                kw2 = {**_common_kwargs, "sampler_method": "tpe",
                       "n_trials": phase2_trials, "study": study2}
                if {"month_out_dir", "month_ix"} <= _sig_params:
                    best2, score2, top5_2, study2, pool2 = run_optuna_tuning(
                        **kw2, month_out_dir=month_graphs_dir_local, month_ix=month_ix_local)
                else:
                    best2, score2, top5_2, study2, pool2 = run_optuna_tuning(**kw2)

                best_params_once, best_score_once, top5_once, study_obj, consensus_pool_once = \
                    best2, score2, top5_2, study2, pool2
            else:
                best_params_once, best_score_once, top5_once, study_obj, consensus_pool_once = \
                    best1, score1, top5_1, study1, pool1
        else:
            # ── SINGLE-PHASE HPO ──
            if {"month_out_dir", "month_ix"} <= _sig_params:
                best_params_once, best_score_once, top5_once, study_obj, consensus_pool_once = run_optuna_tuning(
                    **_common_kwargs,
                    month_out_dir=month_graphs_dir_local,
                    month_ix=month_ix_local,
                )
            else:
                best_params_once, best_score_once, top5_once, study_obj, consensus_pool_once = run_optuna_tuning(
                    **_common_kwargs
                )

        log_print(
            f"[DONE] Optuna best trial: #{study_obj.best_trial.number} value={study_obj.best_value:.6f}",
            level="COMPACT",
        )

        
        # Attach Top-5 and consensus pool metadata for downstream use
        if top5_once:
            best_params_once["__top5_params"] = top5_once

        # Normalise consensus pool to a list (may be empty)
        consensus_pool_once = consensus_pool_once or []
        best_params_once["__consensus_pool"] = consensus_pool_once

        # --------------------------------------------------------------
        # Freeze a Top-N committee once after global HPO.
        # This prevents re-selecting the committee each month and avoids
        # any pool/trials_info alignment quirks later.
        # --------------------------------------------------------------
        try:
            from utilsNoWFO import _infer_family  # local import to avoid any cyclical surprises
            fam = _infer_family(str(best_params_once.get("model_type", "")).lower()).strip()
        except Exception:
            fam = "Unknown"

        try:
            cfg_local = getattr(self, "features_config", {}) or {}
            if fam == "Classical":
                N_target = int(cfg_local.get("topN_classical", 3))
            elif fam == "Ensembles":
                N_target = int(cfg_local.get("topN_ensemble", 2))
            elif fam == "RL":
                N_target = int(cfg_local.get("topN_deep", 2))
            else:
                N_target = int(cfg_local.get("topN_default", 2))
            N_target = max(2, int(N_target))
        except Exception:
            N_target = 2

        def _pool_val(d):
            try:
                v = d.get("__cv_value", d.get("cv_value", d.get("value", None)))
                return float(v) if v is not None else float("-inf")
            except Exception:
                return float("-inf")

        try:
            _pool = [dict(x) for x in (consensus_pool_once or []) if isinstance(x, dict)]
            _committee = sorted(_pool, key=_pool_val, reverse=True)[: max(1, min(N_target, len(_pool)))]
            best_params_once["__committee_fixed"] = _committee
            best_params_once["__committee_fixed_n"] = int(len(_committee))
        except Exception:
            # Don't break training if anything unexpected happens here.
            pass


        # Cache on the backtester for WFO / real_trading_simulation helpers
        self._optuna_best_for_wfo = best_params_once
        self._optuna_top5_for_wfo = top5_once or []
        self._optuna_consensus_pool_for_wfo = consensus_pool_once
        
        log_print(
            f"[DATA] Stored Top-{len(self._optuna_top5_for_wfo or ['best'])} params "
            f"and {len(self._optuna_consensus_pool_for_wfo)} consensus candidates for fallback/consensus use." 
        , level="COMPACT")
        # ============================================================================
        
                # --- HPO-only mode: persist tuned hyperparameters and skip WFO evaluation ---
        if bool(config.get("hpo_only", False)):
            try:
                if bool(config.get("hpo_save_to_disk", False)):
                    # mt_local was defined above when we chose Top-N defaults
                    save_hpo_config_to_disk(mt_local, best_params_once, top5_once)
            except Exception as e:
                print(f"[HPO] Warning: failed to save HPO config for {mt_local}: {e}")
            # In HPO-only mode we return a (None, best_params) tuple so callers
            # can grab the tuned config and run their own evaluation logic.
            return None, best_params_once


        def evaluate_fold(start_date, train_months, test_months, pu="months"):

            train_end = start_date + period_offset(train_months, unit=pu)
            test_end  = train_end + period_offset(test_months, unit=pu)
            if test_end > max_end:
                return None

            # IMPORTANT: end-exclusive slicing to avoid boundary leakage.
            # pandas .loc is inclusive; using < train_end keeps train strictly before test.
            idx_w = walk_data.index
            train_data = walk_data[(idx_w >= start_date) & (idx_w < train_end)]
            test_data  = walk_data[(idx_w >= train_end) & (idx_w < test_end)]
            if len(train_data) < 150 or len(test_data) < 30:
                log_print(
                    f"[WARN] Skipping fold: train={len(train_data)}, test={len(test_data)} too small",
                    level="COMPACT",
                )
                return None

            # Base features (exclude leakage/targets)
            base_features = [
                c for c in train_data.columns
                if c not in ("returns", "price", "spread", "high", "low", "label", "time")
            ]

            # Coarse windows for legacy sliding fallback (mini-block CV has its own sizing)
            min_train_window = int(len(train_data) * 0.75)
            val_window       = int(len(train_data) * 0.25)
            if min_train_window + val_window > len(train_data):
                val_window = len(train_data) - min_train_window
            cv_config = {"min_train_window": min_train_window, "val_window": val_window}
            if isinstance(config, dict) and "_progress_callback" in config:
                cv_config["_progress_callback"] = config["_progress_callback"]
            cv_config["score_for_no_trades"] = -1.0

            # Per-trial CV objective (kept here so second-study fallback can reuse it)
            def evaluate_cv_func(train_data, params, min_train_window, val_window, trial=None):
                # (identical body as above; omitted here for brevity in this snippet)
                return _single_study_cv(train_data, params, min_train_window, val_window, trial=trial)

            # --- NO per-fold Optuna here. Reuse the single study's Top-5 sequentially ---
            best_params = getattr(self, "_optuna_best_for_wfo", None) or {}

            # Seed runtime base threshold from CV aggregation, if available
            try:
                _cv_thr = getattr(self, "_cv_coverage_thr_agg", None)
                if _cv_thr is not None and float(_cv_thr) == float(_cv_thr):  # isfinite without np
                    self._coverage_conf_thr = float(_cv_thr)
                    if self._is_debug():
                        log_print(
                            f"[CV->Runtime] Using aggregated coverage conf_thr={self._coverage_conf_thr:.3f} as base.",
                            level="DEBUG",
                        )
            except Exception as _e:
                if self._is_debug():
                    log_print(f"[CV->Runtime] Coverage conf_thr aggregation not available: {_e}", level="DEBUG")

            top5_params_list = getattr(self, "_optuna_top5_for_wfo", None) or []
            if top5_params_list and "__top5_params" not in best_params:
                best_params["__top5_params"] = top5_params_list

            # --- Prepare Top-5 candidate evaluation (no extra deep refit here) ---
            perf_tuple  = None
            valid_found = False
            selected_params = None  # track chosen candidate for final deep refit


            self.features_config.update(best_params)
            self._optuna_locked_keys = set(best_params.keys())
            base     = dict(best_params)
            raw_topk = best_params.get("__top5_params") or []
            candidates = [base] + [{**base, **deepcopy(alt)} for alt in raw_topk]

            # --- NEW: realism mode uses only the pre-committed CV winner for the candidate list ---
            if not bool(self.features_config.get("allow_param_fallback", False)):
                widx = int(best_params.get("__winner_index", 0))
                if widx <= 0:
                    candidates = [base]
                else:
                    try:
                        chosen = {**base, **deepcopy(raw_topk[widx])}
                        candidates = [chosen]
                    except Exception:
                        candidates = [base]

            REQUIRED_KEYS = ["model_type", "use_extended_features", "lags", "label_threshold"]
            for c in candidates:
                for k in REQUIRED_KEYS:
                    if k not in c and k in base:
                        c[k] = base[k]

            # === Try Top-K candidates sequentially until one meets the WFO trade rule ===
            # NOTE: This now runs even when allow_param_fallback=False, but in that
            #       case `candidates` contains only the chosen CV winner.
            if not valid_found:
                # Configurable minimum trades for *runtime* WFO.
                # Default: 0 -> allows "flat but valid" months.
                cfg_f = getattr(self, "features_config", {}) or {}
                min_trades_wfo = int(cfg_f.get("min_trades_for_wfo", 0))

                for idx, params_try in enumerate(candidates):
                    try:
                        self.features_config.update(params_try)
                        perf_tuple = self.evaluate_strategy(
                            params_try,
                            train_start=start_date,
                            train_end=train_end,
                            test_start=train_end,
                            test_end=test_end,
                        )
                        (
                            perf, outperf, creturns,
                            sharpe, drawdown, trades,
                            geo_mean_ann, directional_accuracy, precision_macro,
                            f1_macro, active_rate, profit_per_hit,
                            return_per_trade, win_rate,
                            strategy_volatility, kurtosis
                        ) = perf_tuple

                        try:
                            trades_int = int(trades) if (trades is not None and trades == trades) else 0
                        except Exception:
                            trades_int = 0

                        if trades_int >= min_trades_wfo:
                            print(f"[OK] Using Top-{idx+1} candidate (trades={trades_int})")
                            selected_params = dict(params_try)
                            valid_found = True
                            break
                        else:
                            print(
                                f"[WARN] Top-{idx+1} candidate produced {trades_int} trades; "
                                "trying next."
                            )
                    except Exception as e:
                        print(f"[ERR] Error with Top-{idx+1} candidate: {e}")
                        continue


            # --- Optional SECOND Optuna study (only when allow_param_fallback=True) ---
            if (not valid_found or perf_tuple is None) and bool(self.features_config.get("allow_param_fallback", False)):
                print("[WARN] Top-5 produced no valid result -> starting a SECOND Optuna study now (sequential).")

                try:
                    best2, score2, top5_2, _ = run_optuna_tuning(
                        train_data=train_data,
                        base_features=base_features,
                        evaluate_cv_func=evaluate_cv_func,
                        cv_config=cv_config,
                        models_to_test=models_to_test,
                        n_trials=int(config.get("retry_extra_trials", 20)),
                        return_top_n=rt_n,
                        study=None,
                        sampler_seed=int(self.features_config.get("run_seed", 0)) or None,
                        max_hpo_duration_minutes=float(config.get("max_hpo_duration_minutes", 0)),
                        sampler_method=str(config.get("hpo_sampler", "tpe")),
                    )
                    if top5_2:
                        best2["__top5_params"] = top5_2

                    base2 = dict(best2)
                    raw2 = best2.get("__top5_params") or []
                    candidates2 = [base2] + [{**base2, **deepcopy(alt)} for alt in raw2]

                    for c in candidates2:
                        for k in REQUIRED_KEYS:
                            if k not in c and k in base2:
                                c[k] = base2[k]

                    for idx, params_try in enumerate(candidates2):
                        try:
                            self.features_config.update(params_try)
                            perf_tuple = self.evaluate_strategy(
                                params_try,
                                train_start=start_date, train_end=train_end,
                                test_start=train_end,  test_end=test_end,
                            )
                            (
                                perf, outperf, creturns, sharpe, drawdown, trades,
                                geo_mean_ann, directional_accuracy, precision_macro,
                                f1_macro, active_rate, profit_per_hit,
                                return_per_trade, win_rate,
                                strategy_volatility, kurtosis
                            ) = perf_tuple

                            cfg_f = getattr(self, "features_config", {}) or {}
                            min_trades_wfo = int(cfg_f.get("min_trades_for_wfo", 0))

                            try:
                                trades_int = int(trades) if (trades is not None and trades == trades) else 0
                            except Exception:
                                trades_int = 0

                            if trades_int >= min_trades_wfo:
                                print(
                                    f"[OK] Using SECOND-study Top-{idx+1} candidate "
                                    f"(trades={trades_int})"
                                )
                                selected_params = dict(params_try)
                                valid_found = True
                                break
                            else:
                                print(
                                    f"[WARN] SECOND-study Top-{idx+1} candidate produced {trades_int} "
                                    "trades; trying next."
                                )
                        except Exception as e:
                            print(f"[ERR] Error with SECOND-study Top-{idx+1} candidate: {e}")
                            continue


                except Exception as e:
                    print(f"[ERR] SECOND Optuna study failed: {e}")

                if not valid_found or perf_tuple is None:
                    print("[ERR] No valid configuration was found in either study.")
                    return None

            # --- FINAL GUARD: no usable metrics -> skip this fold (WFO will log flat month upstream) ---
            if (not valid_found) or (perf_tuple is None):
                print("[ERR] evaluate_fold: no valid metrics produced for this fold; skipping.")
                return None

            # ------------------------------------------------------------
            # Optuna uses capped, compute-saving CV. After a candidate is
            # selected for this fold/month, run a *single* uncapped refit
            # (stride=1, deep_max_train_windows~=inf, etc.) and report those
            # metrics. This preserves compute during search while ensuring
            # final reported results match the deployment training regime.
            # ------------------------------------------------------------
            try:
                _cfg_f = getattr(self, "features_config", {}) or {}
                _do_refit = bool(_cfg_f.get("final_refit_enabled", True))
                _mt = str((selected_params or {}).get("model_type", "")).strip().lower()
                _is_deep_like = _mt in {
                    "cnn", "lstm", "transformer",
                    "ensemble_cnn_lstm_xgboost", "ensemble_adaptive_regime",
                }
                if _do_refit and _is_deep_like and (selected_params is not None):
                    perf_tuple = final_refit_if_deep(
                        backtester=self,
                        best_params=selected_params,
                        train_start=start_date, train_end=train_end,
                        test_start=train_end,  test_end=test_end,
                        overrides={},
                    )
            except Exception as _e:
                try:
                    print(f"[WARN] Final refit skipped/failed; using original metrics. err={_e}")
                except Exception:
                    pass

            # Ensure metrics have the correct arity / structure
            perf_tuple = _safe_metrics_return(perf_tuple, context="wfo_fold")

            # === Save and return results for this fold ===
            (perf, outperf, creturns, sharpe, drawdown, trades,
             geo_mean_ann, directional_accuracy, precision_macro, f1_macro,
             active_rate, profit_per_hit, return_per_trade, win_rate,
             strategy_volatility, kurtosis) = perf_tuple

            return {
                "type": "walk_forward",
                "train_start": start_date,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "performance": perf,
                "outperformance": outperf,
                "return": creturns,
                "sharpe": sharpe,
                "drawdown": drawdown,
                "trades": trades,
                "geo_mean_ann": geo_mean_ann,
                "directional_accuracy": directional_accuracy,
                "precision_macro": precision_macro,
                "f1_macro": f1_macro,
                "active_rate": active_rate,
                "profit_per_hit": profit_per_hit,
                "return_per_trade": return_per_trade,
                "win_rate": win_rate,
                "strategy_volatility": strategy_volatility,
                "kurtosis": kurtosis,
                **best_params,
            }


        start = time.time()
        GPU_MODELS = {
            "cnn", "lstm", "transformer",
            "ensemble_cnn_lstm_xgboost",
            "ensemble_adaptive_regime",
            "xgboost",  # XGBoost with device="cuda" also shares GPU; avoid loky
        }
        is_gpu_model = model_type in GPU_MODELS
        backend = "threading" if is_gpu_model else "loky"

        if model_type in GPU_MODELS:
            if self._progress and self._progress.verbose:
                print("[WARN] Serializing TF-based trials to avoid GPU contention.")
            n_jobs_actual = 1
        else:
            # Use our unified CPU-centric thread knob
            n_jobs_actual = int(
                os.getenv("MLB_THREADS", os.getenv("BLAS_THREADS_PER_TRIAL", "8"))
            )


        hpo_mode = str(config.get("hpo_mode", "static")).lower()

        if hpo_mode == "dynamic" and len(tasks) > 1:
            from pipeline.tuning.runner import run_optuna_tuning as _dyn_run

            _dyn_trials = int(config.get("dynamic_hpo_trials", int(config.get("n_trials", 10))))
            print(f"[HPO] Dynamic mode: re-running HPO for each of {len(tasks)} walk-forward steps "
                  f"({_dyn_trials} trials each)")
            all_results = []
            for s, trn, tst, pu in tqdm(tasks, desc="Walk-forward splits (dynamic HPO)"):
                train_end = s + period_offset(trn, unit=pu)
                fold_train = walk_data[(walk_data.index >= s) & (walk_data.index < train_end)]

                if len(fold_train) >= 150:
                    fold_features = [c for c in fold_train.columns
                                   if c not in ("returns", "price", "spread",
                                                "high", "low", "label", "time")]
                    d_min = int(len(fold_train) * 0.75)
                    d_val = max(1, int(len(fold_train) * 0.25))
                    if d_min + d_val > len(fold_train):
                        d_val = len(fold_train) - d_min
                    fold_cv = {"min_train_window": d_min, "val_window": d_val,
                              "cv_blocks": int(config.get("cv_blocks", 5)),
                              "score_for_no_trades": -1.0}
                    if "_progress_callback" in (config if isinstance(config, dict) else {}):
                        fold_cv["_progress_callback"] = config["_progress_callback"]

                    best_p, _best_s, top5, _st, _pool = _dyn_run(
                        train_data=fold_train,
                        base_features=fold_features,
                        evaluate_cv_func=_single_study_cv,
                        cv_config=fold_cv,
                        models_to_test=models_to_test,
                        n_trials=_dyn_trials,
                        n_startup_trials=int(config.get("n_startup_trials", 10)),
                        return_top_n=rt_n,
                        study=None,
                        sampler_seed=int(self.features_config.get("run_seed", 0)) or None,
                        sampler_method=str(config.get("hpo_sampler", "tpe")),
                    )
                    self._optuna_best_for_wfo = best_p
                    self._optuna_top5_for_wfo = top5 or []

                result = evaluate_fold(s, trn, tst, pu)
                all_results.append(result)
        else:
            all_results = Parallel(n_jobs=n_jobs_actual, backend=backend)(
                delayed(evaluate_fold)(s, trn, tst, pu) for s, trn, tst, pu in tqdm(tasks, desc="Walk-forward splits")
            )


        elapsed_wfo = time.time() - start
        print(f"[OK] Parallel walk-forward completed in {elapsed_wfo:.2f} seconds.")

        all_results = [r for r in all_results if r is not None]
        if not all_results:
            print("[ERR] WFO failed completely.")
            try:
                getattr(self, "_progress", None) and self._progress.draw_final(
                    elapsed=time.time() - start, trades=0, sr=0.0, sharpe=0.0, dd=0.0,
                )
            except Exception:
                pass
            return None, None

        # Serialize config dicts for grouping
        for r in all_results:
            for key in [
                "cnn_config", "lstm_config", "transformer_config", "xgb_config",
                "dqn_config", "rf_config", "logit_config", "indicator_windows",
            ]:
                if key in r and isinstance(r[key], dict):
                    r[key] = json.dumps(r[key], sort_keys=True)

        df_wfo = pd.DataFrame(all_results)

        # Normalize potential leftover dicts to JSON strings
        for col in [
            "cnn_config", "lstm_config", "transformer_config", "xgb_config",
            "dqn_config", "rf_config", "logit_config",
        ]:
            if col in df_wfo.columns:
                df_wfo[col] = df_wfo[col].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else x)

        metric_cols = {
            "type", "train_start", "train_end", "test_start", "test_end",
            "performance", "return", "sharpe", "drawdown", "trades",
            "geo_mean_ann", "directional_accuracy", "precision_macro",
            "f1_macro", "active_rate", "profit_per_hit",
            "return_per_trade", "win_rate", "strategy_volatility", "kurtosis",
        }
        candidates = [c for c in df_wfo.columns if c not in metric_cols and not str(c).startswith("__top5")]

        def _is_scalar_series(s):
            return s.map(lambda x: isinstance(x, (type(None), bool, int, float, str))).all()

        param_cols = [c for c in candidates if _is_scalar_series(df_wfo[c])]
        if not param_cols:
            print("[WARN] No hashable parameter columns to group by; cannot select best combo.")
            return df_wfo, None

        grouped = (
            df_wfo.groupby(param_cols, dropna=False)["performance"]
            .mean().sort_values(ascending=False)
        )
        if grouped.empty:
            print("[WARN] WFO produced no valid rows to rank; cannot select best combo.")
            return df_wfo, None

        topk_df = grouped.reset_index()
        best_combo = topk_df.iloc[0].to_dict()

        # Carry __top5_* helpers through (don't group on them)
        for helper in ["__top5_info", "__top5_params", "__top5_path"]:
            if helper in df_wfo.columns:
                first_nonnull = df_wfo[helper].dropna()
                if not first_nonnull.empty:
                    best_combo[helper] = first_nonnull.iloc[0]

        # --- Fallback: if for some reason __top5_params did not survive
        # into df_wfo, pull directly from the backtester attribute set
        # by run_optuna_tuning() earlier.
        try:
            topN_fallback = getattr(self, "_optuna_top5_for_wfo", None) or []
        except Exception:
            topN_fallback = []

        if topN_fallback and not best_combo.get("__top5_params"):
            best_combo["__top5_params"] = topN_fallback
            print(f"[TopN] Attached {len(topN_fallback)} tuned configs to best_combo for real-trading consensus.")

        # Deserialize JSON config columns back to dicts, if any
        for key in ["cnn_config", "lstm_config", "transformer_config", "xgb_config", "dqn_config", "rf_config", "logit_config"]:
            if key in best_combo and isinstance(best_combo[key], str):
                try:
                    best_combo[key] = json.loads(best_combo[key])
                except Exception:
                    pass

        # Draw final summary
        try:
            _sharpe = float(best_combo.get("sharpe", 0))
            _dd = float(best_combo.get("drawdown", 0))
            _trades = int(best_combo.get("trades", 0))
            getattr(self, "_progress", None) and self._progress.draw_final(
                elapsed=elapsed_wfo, trades=_trades,
                sr=_sharpe, sharpe=_sharpe, dd=_dd,
            )
        except Exception:
            pass

        return df_wfo, best_combo

