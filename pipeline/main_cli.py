"""
CLI entry point — main() function.

Extracted from MLBacktesterNoWFO.py lines 18441-end.
"""

from pipeline._imports import *  # noqa: F401,F403
from pipeline.metrics_tuples import CLASS_DEFAULTS, _safe_metrics_return, _empty_metrics  # noqa: F811
from pipeline.backtester.composed import MLBacktester  # noqa: F811
from pipeline.memory_utils import _apply_low_ram_overrides  # noqa: F811

# ── CLI-level constants (from original MLBacktesterNoWFO.py) ──
SAVE_METRICS = {
    "per_month_metrics_csv": True,
    "monthly_results_all_csv": True,
    "monthly_results_per_rep_csv": True,
}

# Trial counts per model type (from original MLBacktesterNoWFO.py)
TRIAL_COUNTS = {
    "logistic":                    {"random": 5,  "bayes": 5},
    "svm":                         {"random": 5,  "bayes": 5},
    "random_forest":               {"random": 5,  "bayes": 10},
    "decision_tree":               {"random": 5,  "bayes": 5},
    "xgboost":                     {"random": 5,  "bayes": 15},
    "lstm":                        {"random": 3,  "bayes": 7},
    "cnn":                         {"random": 3,  "bayes": 7},
    "transformer":                 {"random": 3,  "bayes": 7},
    "dqn":                         {"random": 2,  "bayes": 3},
    "ensemble_cnn_lstm_xgboost":   {"random": 2,  "bayes": 3},
    "ensemble_adaptive_regime":    {"random": 2,  "bayes": 3},
}

def main() ->  None:
    """
    Run a 36-month real-trading simulation across selected models,
    repeat it N times with different seeds, and rank models across repeats.
    """
    
    # --- local memory cleanup helper (no external deps) ---
    import time, gc
    def _hard_free_local():
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.05)

    # ------------------------------------------------------------
    # FINAL EXPERIMENT: fixed per-run seeds (one full pipeline per seed)
    # This replaces the old "fresh os.urandom seed per repeat" behavior.
    # ------------------------------------------------------------

    
    # SEEDS = [11111, 22222, 33333]
    # SEEDS = [22222, 33333]
    
    # ── Configurable via environment variables ──
    # SEEDS: comma-separated list, e.g. "11111,22222,33333"
    _env_seeds = os.environ.get("SEEDS", "")
    SEEDS = [int(s.strip()) for s in _env_seeds.split(",") if s.strip()] if _env_seeds else [33333]
    REPEATS = int(os.environ.get("REPEATS", "1"))
    PAIR = os.environ.get("PAIR", "EURUSD").upper()

    # ── Smoke-test mode: override config for fast validation ──
    _SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

    if _SMOKE:
        global TRIAL_COUNTS
        TRIAL_COUNTS = {k: {"random": 1, "bayes": 1} for k in TRIAL_COUNTS}

    N_REAL_MONTHS = int(os.environ.get("N_MONTHS", "1" if _SMOKE else "3"))  # default 36 for full run
    END_DATE = "2025-12-01 00:00:00"   # end-of-Aug 2025, inclusive-ish for bar data

    # 1) Load feature configuration
    with open("configs/feature_config.json", "r") as f:
        features_config = json.load(f)
        
    # 🔒 Research-grade defaults for fairness:
    # - All models share the same strict day-1 calendar anchor.
    # - All models share the same NY session filtering rule.
    #   (Change "both" to "test_only" here if you want, but KEEP IT GLOBAL.)
    features_config.setdefault("enforce_day1_start", True)
    if features_config.get("session_filter_mode") is None:
        features_config["session_filter_mode"] = "both"

    # 1.1) EXPERIMENT LOCK: enforce CLASS_DEFAULTS over JSON for reproducibility.
    # JSON can still add extra keys, but it cannot override the experiment defaults.
    for _k, _v in CLASS_DEFAULTS["features"].items():
        features_config[_k] = deepcopy(_v) if isinstance(_v, (dict, list)) else _v

    # 1.5) Create one study run folder and make it global for this process
    RUN_DIR, _ = make_results_run_dir()
    # Softer Optuna RAM defaults: keep a small absolute floor, no percent-of-total gate.
    # The RAM guard is now *soft* (warns + GC) instead of pruning trials.
    need_default = float(os.environ.get("OPTUNA_MIN_FREE_GB", "0.35"))
    # Clamp to a maximum default; user can override via env var if they want stricter limits.
    if need_default > 0.35:
        os.environ["OPTUNA_MIN_FREE_GB"] = "0.35"

    # IMPORTANT: disable the percent-of-total rule by default.
    # This used to force need_gb ~= 3% of total RAM (≈0.76GB on your machine),
    # which caused "low RAM prune" even when ~0.75GB was free.
    os.environ.setdefault("OPTUNA_MIN_FREE_GB_PERCENT", "0.0")

    # Keep a mild relax/floor in case you later want stricter behaviour.
    os.environ.setdefault("OPTUNA_MIN_FREE_GB_RELAX", "0.6")
    os.environ.setdefault("OPTUNA_MIN_FREE_GB_FLOOR", "0.20")


    # If a CUDA GPU is present, also demand some VRAM headroom so we prune instead of OOMing
    try:
        import tensorflow as _tf
        if _tf.config.list_physical_devices("GPU"):
            # ask for at least ~1.0 GB free VRAM before starting a trial
            os.environ.setdefault("OPTUNA_MIN_FREE_VRAM_GB", "1.0")
    except Exception:
        pass

    # Keep BLAS inside each trial reasonable; if user already set
    # BLAS_THREADS_PER_TRIAL (e.g. via .env), respect it. Otherwise
    # fall back to (cores - 2) as a sensible default.
    _safe = max(1, (os.cpu_count() or 8) - 2)
    os.environ.setdefault("BLAS_THREADS_PER_TRIAL", str(_safe))

    # Keep BLAS/OpenMP stacks consistent with the chosen BLAS_THREADS_PER_TRIAL
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, os.environ["BLAS_THREADS_PER_TRIAL"])

    init_study_tree(RUN_DIR)

    os.environ["RESULTS_RUN_DIR"] = RUN_DIR  # ensures internal calls reuse the same folder
    
    # Persistent joblib memmap root for this run (prevents racey temp deletion)
    JOBLIB_ROOT = os.path.abspath("./joblib_tmp")
    os.makedirs(JOBLIB_ROOT, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = JOBLIB_ROOT
    
    log_print(f"🗂️ JOBLIB_TEMP_FOLDER={JOBLIB_ROOT}", level="COMPACT")
    log_print(f"\n📁 Study folder: {RUN_DIR}", level="COMPACT")


    # 2) Choose models — configurable via MODEL_LIST env var
    # Examples:
    #   MODEL_LIST=logistic             (single model)
    #   MODEL_LIST=logistic,xgboost,cnn (multi-model)
    #   unset → uses hardcoded defaults below
    _env_models = os.environ.get("MODEL_LIST", "")
    if _env_models:
        MODEL_LIST = [m.strip() for m in _env_models.split(",") if m.strip()]
    elif _SMOKE:
        MODEL_LIST = ["logistic"]  # fastest model, just 1 for smoke
    else:
        MODEL_LIST = [
            # Linear / margin baselines (shallow, low-capacity)
            "logistic", 
            # "svm", 

            # # Tree-based classical ML (nonlinear tabular learners)
            # "random_forest", 
            # "decision_tree",  
            "xgboost",  
        
        # # Deep supervised sequence models (learn temporal structure directly)
        # "lstm", 
        # "cnn", 
        # "transformer", 
        
        # Reinforcement learning (policy/Q-learning)
        # "dqn",  
        
        # Hybrid ensembles (explicit fusion / regime routing)
        # "ensemble_cnn_lstm_xgboost",  
        # "ensemble_adaptive_regime", 
    ]
    
    print(f"\n🧪 Models for real trading simulation: {MODEL_LIST}")

    all_reps = []  # collect combined monthly results across all repeats
    eq_by_model: dict[str, list[pd.DataFrame]] = {}  # collect per-rep equity paths

    for rep in range(1, REPEATS + 1):

        # ------------------------------------------------------------------
        # Per-repetition run directory: <RUN_DIR>/repetition_1, repetition_2, ...
        # Everything for this repetition (per-model, per-month) is routed here.
        # ------------------------------------------------------------------
        rep_run_dir = os.path.join(RUN_DIR, f"repetition_{rep}")
        os.makedirs(rep_run_dir, exist_ok=True)

        # Let downstream helpers know where to write results for this repeat
        os.environ["RESULTS_RUN_DIR"] = rep_run_dir

        # ---- Fixed seed per repeat (research reproducibility) ----
        run_seed = int(SEEDS[rep - 1])
        set_global_determinism(seed=run_seed)
        
        # ---- Seed sanity trace (should match across same-seed reruns) ----
        try:
            import random as _py_random
            log_print(
                f"[SEED-SANITY] seed={run_seed} py={_py_random.random():.12f} np={np.random.randint(0, 2**31-1)}",
                level="COMPACT",
            )
        except Exception:
            pass

        # ✅ Fresh, isolated config for THIS repeat (no cross-repeat bleed)
        features_config_rep = deepcopy(features_config)
        features_config_rep["run_seed"] = int(run_seed)
        # Drop any sticky derived fields that prior runs may have injected
        features_config_rep.pop("eval_seed_sets", None)
        features_config_rep.pop("test_warmup_bars", None)

        log_print(
            f"\n========== 🔁 REPEAT {rep}/{REPEATS} — seed={run_seed} =========="
            f"\n📁 repetition_run_dir = {rep_run_dir}",
            level="COMPACT",
        )

        rep_results: dict[str, pd.DataFrame] = {}
        bt_by_model = {}

        # 3) Simulate N walk-forward months per model
        for model_type in MODEL_LIST:
            log_print(
                f"\n🚦 Running real trading simulation for model: {model_type}",
                level="COMPACT",
            )
            
            # --- CPU/GPU perf profile per category ---
            try:
                cat = model_category(model_type)
            except Exception:
                cat = "unknown"

            # One Optuna trial at a time for all families (n_jobs=1), but let that trial use many cores
            # Respect user-provided MLB_THREADS; otherwise fall back conservatively
            cpu_total  = max(1, (os.cpu_count() or 8))
            safe_cores = int(os.environ.get("MLB_THREADS", str(max(1, cpu_total - 2))))

            os.environ.setdefault("CV_JOBS", str(safe_cores))
            os.environ.setdefault("OPTUNA_N_JOBS", "1")  # keep trials sequential
            os.environ.setdefault("BLAS_THREADS_PER_TRIAL", str(safe_cores))

            for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(k, str(safe_cores))

            # keep these aligned if not set
            os.environ.setdefault("SKLEARN_JOBS", str(safe_cores))
            os.environ.setdefault("XGB_JOBS", str(safe_cores))
            os.environ.setdefault("RF_JOBS", str(safe_cores))



            # Also align common BLAS envs so numpy/scipy/OpenMP agree
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[k] = os.environ["BLAS_THREADS_PER_TRIAL"]

            print(f"⚙️ Perf [{cat}] → BLAS_THREADS_PER_TRIAL={os.environ['BLAS_THREADS_PER_TRIAL']} "
                f"| OPTUNA_N_JOBS={os.environ['OPTUNA_N_JOBS']}")

            
            try:
                trial_cfg = TRIAL_COUNTS.get(model_type, {"random": 5, "bayes": 5})
                base_config = {
                    "model_type": model_type,
                    "rep": rep,  # repetition index used for trade-log file names
                    "n_trials": trial_cfg["random"] + trial_cfg["bayes"],
                    "n_startup_trials": trial_cfg["random"],
                    # Route all artifacts for this repetition into its own subfolder:
                    # e.g. <DATE>/rep_1/logistic/..., <DATE>/rep_1/xgboost/...
                    "_run_dir": rep_run_dir,
                }

                # Apply low-RAM overrides on a *copy* so the original config remains intact
                model_features_cfg = deepcopy(features_config_rep)
                if os.environ.get("MLB_DISABLE_LOW_RAM_OVERRIDES", "0") != "1":
                    model_features_cfg = _apply_low_ram_overrides(model_features_cfg)

                # Instantiate a fresh backtester for this model
                bt = MLBacktester(
                    symbol=PAIR,
                    start="2019-10-01 00:00:00",
                    end=END_DATE,
                    trading_costs=False,
                    features_config=model_features_cfg,
                    use_oof=("ensemble" in model_type),
                )

                try:
                    df_sim = bt.real_trading_simulation(
                        deepcopy(base_config),
                        models_to_test=[model_type],
                        months=N_REAL_MONTHS,
                    )
                    if df_sim is not None and not df_sim.empty:
                        df_sim = df_sim.copy()
                        df_sim["model_type"] = model_type
                        df_sim["rep"] = rep
                        df_sim["run_seed"] = run_seed
                        rep_results[model_type] = df_sim
                        # Keep only the per-bar equity curves for cross-model plots.
                        # Storing the full backtester keeps large frames/caches alive.
                        from types import SimpleNamespace
                        bc = getattr(bt, "bar_concat", None)
                        if bc is not None and not getattr(bc, "empty", True):
                            _cols = [c for c in ("cstrategy_cont", "creturns_cont") if c in bc.columns]
                            bc_small = bc[_cols].copy() if _cols else bc.iloc[:, :0].copy()
                        else:
                            bc_small = pd.DataFrame()
                        bt_by_model[model_type] = SimpleNamespace(bar_concat=bc_small)


                        # Collect full-horizon equity path for mean-over-reps (this model, this rep)
                        try:
                            bc = getattr(bt, "bar_concat", None)
                            if bc is not None and not getattr(bc, "empty", True):
                                eq_df = bc.copy()

                                # Ensure expected columns
                                cols = list(eq_df.columns)
                                if "cstrategy_cont" not in cols or "creturns_cont" not in cols:
                                    if len(cols) >= 2:
                                        eq_df = eq_df.copy()
                                        eq_df.columns = ["cstrategy_cont", "creturns_cont"][:len(cols)]

                                if "cstrategy_cont" in eq_df.columns:
                                    tmp = eq_df[["cstrategy_cont"]].copy()
                                    tmp["rep"] = int(rep)
                                    tmp["ts"] = tmp.index
                                    # Append to accumulator
                                    eq_by_model.setdefault(model_type, []).append(tmp)
                        except Exception as _e:
                            print(f"⚠️ Could not collect equity path for model {model_type}, rep {rep}: {_e}")
                    else:
                        print(f"⚠️ No rows returned for model {model_type}.")

                except Exception as e:
                    print(f"❌ Simulation failed for {model_type}: {e}")
                    traceback.print_exc()
                finally:
                    # Release model resources ASAP
                    try:
                        if hasattr(bt, "free") and callable(getattr(bt, "free")):
                            bt.free(release_data=True)

                    except Exception:
                        pass
            finally:
                # Hard cleanup between models
                try:
                    _hard_free_local()
                except Exception:
                    pass
                try:
                    import tensorflow as _tf
                    _tf.keras.backend.clear_session()
                except Exception:
                    pass
                import gc, time
                gc.collect()
                time.sleep(0.05)

        # 4) Per-repeat cross-model outputs + save combined monthly
        if rep_results:
            dfs = [df for df in rep_results.values() if not df.empty]
            combined_rep = pd.concat(dfs, ignore_index=True)
            # print("\n📊 Combined monthly results (this repeat):")
            # print(combined_rep.to_string(index=False))

            # === Write this-repeat combined monthly table into the repetition-local 'ALL/csv' ===
            rep_buckets = comparison_dirs(rep_run_dir)
            os.makedirs(rep_buckets["All"]["csv"], exist_ok=True)
            combined_rep_path = os.path.join(
                rep_buckets["All"]["csv"],
                f"combined_monthly_rep{rep}.csv",
            )
            combined_rep.to_csv(combined_rep_path, index=False)
            print(f"✅ Saved per-repeat monthly CSV → {combined_rep_path}")

            # --- Per-repeat ranking across models (this repeat only) ---
            try:
                rep_rank_df = build_model_ranking(combined_rep, min_months=1)
                if rep_rank_df is not None and not rep_rank_df.empty:
                    rep_rank_path = save_model_ranking_csv(
                        rep_rank_df,
                        rep_buckets["All"]["csv"],
                        filename=f"csv_ranking_rep{rep}.csv",
                    )
                    print(f"✅ Saved per-repeat ranking → {rep_rank_path}")
            except Exception as _e:
                print(f"[tables] ranking for repeat {rep} skipped: {_e}")

            # Generate cross-model artifacts only if 2+ models produced per-bar curve
            if len(bt_by_model) > 1:
                try:
                    bt_dict_for_compare = _build_bar_compare_dict(bt_by_model)
                except Exception as e:
                    log_print(
                        f"⚠️ Failed to build bar comparison dict: {e}",
                        level="DEBUG",
                    )
                    bt_dict_for_compare = {}

                if bt_dict_for_compare:
                    # Which models actually produced results this repeat?
                    avail_all = sorted([
                        k.replace("_equity", "")
                        for k in (bt_dict_for_compare or {}).keys()
                        if isinstance(k, str) and k != "BH" and k.endswith("_equity")
                    ])

                    if avail_all:
                        # Create bar-comparison + risk outputs in this repetition's ALL/ bucket.
                        if not SKIP_PLOTS:
                            # Per-bar cumulative equity (existing)
                            save_model_bar_comparison_outputs(
                                bt_dict_for_compare,
                                models=avail_all,
                                csv_dir=rep_buckets["All"]["csv"],   # CSV suppressed inside helper
                                png_dir=rep_buckets["All"]["graphs"],
                                style="nature",
                                palette="okabe_ito_no_black",
                                bh_color="#666666",
                                n_time_parts=10,
                                dpi=300,
                                line_width=1.2,
                                annotate_coverage=False,
                                save_csv=False,
                            )

                            # NEW: multi-model underwater / drawdown curve
                            save_model_underwater_outputs(
                                bt_dict_for_compare,
                                models=avail_all,
                                csv_dir=rep_buckets["All"]["csv"],
                                png_dir=rep_buckets["All"]["graphs"],
                                style="nature",
                                palette="okabe_ito_no_black",
                                bh_color="#666666",
                                n_time_parts=10,
                                dpi=300,
                                line_width=1.2,
                            )

                            # NEW: multi-model rolling Sharpe curve
                            save_model_rolling_performance_outputs(
                                bt_dict_for_compare,
                                models=avail_all,
                                csv_dir=rep_buckets["All"]["csv"],
                                png_dir=rep_buckets["All"]["graphs"],
                                style="nature",
                                palette="okabe_ito_no_black",
                                bh_color="#666666",
                                n_time_parts=10,
                                dpi=300,
                                line_width=1.2,
                                window_bars=None,  # auto ~1-month from frequency
                            )

                        # Rename outputs so each repetition has its own files
                        try:
                            # Bar compare
                            default_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                "model_bar_compare.csv",
                            )
                            default_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                "model_bar_compare_bars.png",
                            )
                            rep_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                f"bar_compare_models_rep{rep}.csv",
                            )
                            rep_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                f"bar_compare_models_rep{rep}.png",
                            )

                            # Underwater
                            under_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                "model_bar_underwater.csv",
                            )
                            under_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                "model_bar_underwater.png",
                            )
                            rep_under_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                f"underwater_models_rep{rep}.csv",
                            )
                            rep_under_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                f"underwater_models_rep{rep}.png",
                            )

                            # Rolling Sharpe
                            roll_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                "model_rolling_sharpe.csv",
                            )
                            roll_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                "model_rolling_sharpe.png",
                            )
                            rep_roll_csv = os.path.join(
                                rep_buckets["All"]["csv"],
                                f"rolling_sharpe_models_rep{rep}.csv",
                            )
                            rep_roll_png = os.path.join(
                                rep_buckets["All"]["graphs"],
                                f"rolling_sharpe_models_rep{rep}.png",
                            )

                            # Rename if the defaults exist
                            for src, dst in [
                                (default_csv, rep_csv),
                                (default_png, rep_png),
                                (under_csv, rep_under_csv),
                                (under_png, rep_under_png),
                                (roll_csv, rep_roll_csv),
                                (roll_png, rep_roll_png),
                            ]:
                                if os.path.exists(src):
                                    os.replace(src, dst)

                            print(
                                f"✅ Saved bar & risk comparison for rep {rep} → "
                                f"{rep_csv}, {rep_png}"
                            )
                        except Exception as _e:
                            print(
                                f"⚠️ Could not rename comparison outputs for rep {rep}: {_e}"
                            )
                    else:
                        print("⚠️ No models available for 'All' comparison.")
                else:
                    print("⚠️ No bt_dict_for_compare to plot.")
            else:
                print("ℹ️ Single-model (or no per-bar) in this repeat: skipping cross-model plots.")

            all_reps.append(combined_rep)
        else:
            print("\n❌ No valid simulation results produced in this repeat.")

        # --- end-of-repeat cleanup (keep RSS stable across repetitions) ---
        try:
            rep_results.clear()
        except Exception:
            pass
        try:
            bt_by_model.clear()
        except Exception:
            pass
        try:
            _hard_free_local()
        except Exception:
            pass


    if not all_reps:
        print("\n❌ No results in any repeat — nothing to rank.")
        return

    # -------------------------------------------------------------
    # 5) Final: aggregate across repeats (all_reps) and route tables
    # -------------------------------------------------------------
    combined_all = pd.concat(all_reps, ignore_index=True)

    # Global combined monthly table (all repeats, all models)
    #   <RUN_DIR>/combined_monthly_all.csv
    try:
        combined_all_path = os.path.join(RUN_DIR, "combined_monthly_all.csv")
        combined_all.to_csv(
            combined_all_path,
            index=False,
            float_format="%.10f",
        )
        print(f"✅ Saved combined monthly (all repeats) → {combined_all_path}")
    except Exception as e:
        print(f"[tables] combined_monthly_all.csv skipped: {e}")
        
    # Global equity curves by model (all months × repeats), consistent
    # with the ranking table. For each (month, model) we aggregate the
    # monthly 'cstrategy' factors across repeats via a geometric mean,
    # then compound those factors in chronological order starting from 1.0.
    try:
        from utilsNoWFO import (
            build_model_monthly_pivots,
            save_group_equity_curves,
        )  # local import is cheap and robust; comparison_dirs is global

        equity_pivot, returns_pivot, bh_equity = build_model_monthly_pivots(combined_all)
        if equity_pivot is not None and not equity_pivot.empty:
            # use the globally imported comparison_dirs
            global_buckets = comparison_dirs(RUN_DIR)
            out_png = os.path.join(
                global_buckets["All"]["graphs"],
                "model_equity_all_months.png",
            )

            save_group_equity_curves(
                equity_pivot,
                bh_equity,
                out_png=out_png,
                title="Equity by Model (all months × repeats)",
                include_bh=True,
            )
            print(f"✅ Saved global equity curves (all months × repeats) → {out_png}")
        else:
            print("⚠️ Global equity curves skipped: empty equity_pivot.")
    except Exception as e:
        print(f"⚠️ Global equity curve plot skipped: {e}")

    # -------------------------------------------------------------
    # Per-model monthly results
    #   • monthly_results_all_<model>.csv      → <RUN_DIR>/model_stats/
    #   • monthly_results_rep<k>_<model>.csv  → <RUN_DIR>/repetition_k/<Model>/csv/
    # -------------------------------------------------------------
    if not (
        SAVE_METRICS.get("monthly_results_all_csv", True)
        or SAVE_METRICS.get("monthly_results_per_rep_csv", True)
    ):
        print("[tables] Per-model monthly results saving is disabled via SAVE_METRICS.")
    else:
        try:
            import os as _os
            import pandas as _pd
            from utilsNoWFO import friendly_model_name

            # Root-level folder for "all repeats" per-model tables
            model_stats_dir = _os.path.join(RUN_DIR, "model_stats")
            _os.makedirs(model_stats_dir, exist_ok=True)

            model_col = combined_all.get("model_type")
            if model_col is None:
                model_col = combined_all.get("model")

            if model_col is None:
                print("[tables] No 'model_type'/'model' column in combined_all; skipping per-model monthly tables.")
            else:
                models_present = [
                    m for m in model_col.dropna().unique().tolist()
                    if isinstance(m, str) and m.strip()
                ]
                models_present = sorted(models_present)

                for m in models_present:
                    model_df = combined_all[model_col == m].copy()
                    if model_df.empty:
                        continue

                    # (a) All repeats, this model → <RUN_DIR>/model_stats/
                    if SAVE_METRICS.get("monthly_results_all_csv", True):
                        all_path = _os.path.join(
                            model_stats_dir,
                            f"monthly_results_all_{m}.csv",
                        )
                        model_df.to_csv(
                            all_path,
                            index=False,
                            float_format="%.10f",
                        )
                        if SAVE_METRICS.get("verbose", False):
                            print(f"    ↳ Saved all-reps monthly results for {m} → {all_path}")

                    # (b) Per repetition, this model → <RUN_DIR>/repetition_k/<Model>/csv/
                    if SAVE_METRICS.get("monthly_results_per_rep_csv", True) and "rep" in model_df.columns:
                        reps_present = (
                            model_df["rep"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                        for rep_val in sorted(reps_present):
                            try:
                                rep_int = int(rep_val)
                            except Exception:
                                continue

                            rep_df = model_df[model_df["rep"] == rep_val].copy()
                            if rep_df.empty:
                                continue

                            rep_dir = _os.path.join(RUN_DIR, f"repetition_{rep_int}")
                            model_folder = friendly_model_name(m)
                            model_base_dir = _os.path.join(rep_dir, model_folder)
                            csv_dir = _os.path.join(model_base_dir, "csv")
                            _os.makedirs(csv_dir, exist_ok=True)

                            rep_path = _os.path.join(
                                csv_dir,
                                f"monthly_results_rep{rep_int}_{m}.csv",
                            )
                            rep_df.to_csv(
                                rep_path,
                                index=False,
                                float_format="%.10f",
                            )
                            if SAVE_METRICS.get("verbose", False):
                                print(f"    ↳ Saved rep-{rep_int} monthly results for {m} → {rep_path}")

                print("✅ Saved per-model monthly_results_all_* and monthly_results_rep*_*.csv in new layout.")
        except Exception as e:
            print(f"[tables] Per-model monthly results skipped due to error: {e}")

    # -------------------------------------------------------------
    # NOTE: We intentionally do NOT save mean equity curves over
    #       repetitions anymore (full_equity_mean_over_reps_*).
    #       Only per-rep equity plots, monthly tables, and global
    #       feature heatmaps are produced at run level.
    # -------------------------------------------------------------

    # 6) Global feature heatmaps across ALL repetitions
    #    • one "all models" heatmap
    #    • one per-model heatmap
    #    Location: <RUN_DIR>/heatmaps/
    if not SKIP_PLOTS:
        try:
            global_heat_dir = os.path.join(RUN_DIR, "heatmaps")
            os.makedirs(global_heat_dir, exist_ok=True)

            # (a) All models together
            try:
                save_feature_frequency_from_monthly_results(
                    combined_all,
                    base_features=[],
                    out_png=os.path.join(
                        global_heat_dir,
                        "feature_heatmap_all_models_all_reps.png",
                    ),
                    top_k=30,
                    top_percent=1.0,
                    weight_by_score=False,
                    minimize_objective=False,
                    style="nature",
                    palette="okabe_ito_no_black",
                    exclude_prefixes=("returns_lag", "hour"),
                    collapse_raw_lags=True,
                    out_csv=os.path.join(
                        global_heat_dir,
                        "feature_heatmap_all_models_all_reps.csv",
                    ),
                )
                print(
                    f"✅ Saved global feature heatmap (all models, all reps) → {global_heat_dir}"
                )
            except Exception as _e:
                print(f"⚠️ Global all-model heatmap skipped: {_e}")

            # (b) Per-model heatmaps
            model_col = combined_all.get("model_type")
            if model_col is None:
                model_col = combined_all.get("model")

            if model_col is not None:
                models_present = sorted(
                    [
                        m for m in model_col.dropna().unique().tolist()
                        if isinstance(m, str) and m.strip()
                    ]
                )
                for m in models_present:
                    df_m = combined_all[model_col == m]
                    if df_m.empty:
                        continue
                    try:
                        save_feature_frequency_from_monthly_results(
                            df_m,
                            base_features=[],
                            out_png=os.path.join(
                                global_heat_dir,
                                f"feature_heatmap_{m}_all_reps.png",
                            ),
                            top_k=30,
                            top_percent=1.0,
                            weight_by_score=False,
                            minimize_objective=False,
                            style="nature",
                            palette="okabe_ito_no_black",
                            exclude_prefixes=("returns_lag", "hour"),
                            collapse_raw_lags=True,
                            out_csv=os.path.join(
                                global_heat_dir,
                                f"feature_heatmap_{m}_all_reps.csv",
                            ),
                        )
                        print(
                            f"✅ Saved global feature heatmap for model={m} → {global_heat_dir}"
                        )
                    except Exception as _e:
                        print(
                            f"⚠️ Global per-model heatmap for {m} skipped: {_e}"
                        )
            else:
                print(
                    "ℹ️ No model_type/model column; skipping global per-model heatmaps."
                )

        except Exception as e:
            print(f"⚠️ Global heatmap generation skipped: {e}")


    # Build & save definitive model ranking across X months (and repeats)
    try:
        rank_df = build_model_ranking(combined_all, min_months=1)

        # Save CSV at run root (next to repetition_1/, repetition_2/, model_stats/, etc.)
        try:
            global_rank_path = os.path.join(RUN_DIR, "csv_ranking_FINAL.csv")
            rank_df.to_csv(
                global_rank_path,
                index=False,
                float_format="%.10f",
            )
            print(f"✅ Saved global ranking across repeats → {global_rank_path}")
        except Exception as _e:
            print(f"[tables] ranking CSV skipped: {_e}")

        # Pretty ASCII table, consistent with other outputs
        cols = [
            "rank","model","months","trades","active","SR","PSR","DSR","Calmar",
            "AnnRet","FinalEq","DA","Prec","F1","Profit/Hit","LabelThr","EffConf","lags"
        ]
        rows = []
        if rank_df is not None and not rank_df.empty:
            for _, r in rank_df.iterrows():
                rows.append([
                    int(r.get("rank", 0)),
                    str(r.get("model", "")),
                    int(r.get("months", 0)) if pd.notna(r.get("months", None)) else "—",
                    int(r.get("trades", 0)) if pd.notna(r.get("trades", None)) else "—",
                    (f"{float(r.get('active', float('nan'))):.5f}"     if pd.notna(r.get("active", None)) else "—"),
                    (f"{float(r.get('SR', float('nan'))):.3f}"         if pd.notna(r.get("SR", None)) else "—"),
                    (f"{float(r.get('PSR', float('nan'))):.3f}"        if pd.notna(r.get("PSR", None)) else "—"),
                    (f"{float(r.get('DSR', float('nan'))):.3f}"        if pd.notna(r.get("DSR", None)) else "—"),
                    (f"{float(r.get('Calmar', float('nan'))):.3f}"     if pd.notna(r.get("Calmar", None)) else "—"),
                    (f"{float(r.get('AnnRet', float('nan'))):.5f}"     if pd.notna(r.get("AnnRet", None)) else "—"),
                    (f"{float(r.get('FinalEq', float('nan'))):.5f}"    if pd.notna(r.get("FinalEq", None)) else "—"),
                    (f"{float(r.get('DA', float('nan'))):.5f}"         if pd.notna(r.get("DA", None)) else "—"),
                    (f"{float(r.get('Prec', float('nan'))):.5f}"       if pd.notna(r.get("Prec", None)) else "—"),
                    (f"{float(r.get('F1', float('nan'))):.5f}"         if pd.notna(r.get("F1", None)) else "—"),
                    (f"{float(r.get('Profit/Hit', float('nan'))):.6f}" if pd.notna(r.get("Profit/Hit", None)) else "—"),
                    (f"{float(r.get('LabelThr', float('nan'))):.6f}"   if pd.notna(r.get("LabelThr", None)) else "—"),
                    (f"{float(r.get('EffConf', float('nan'))):.3f}"    if pd.notna(r.get("EffConf", None)) else "—"),
                    (int(r.get("lags", 0)) if pd.notna(r.get("lags", None)) else "—"),
                ])
        _fmt_table_ascii(
            cols,
            rows,
            title="🏁 Model Ranking (all months × repeats; equity compounded monthly)",
        )

    except Exception as e:
        print(f"[tables] ranking build skipped: {e}")


    # ---------------- Local toggles for this section (no global `config` required) ----------
    PRINT_EVAL_TABLES      = True    # master switch: prints ONLY the complex per-run table
    EVAL_TABLE_MAX_ROWS    = 40      # max rows for the per-run table


    def _safe_first(group, cols, default=None):
        for c in cols:
            if c in group and not group[c].dropna().empty:
                return group[c].dropna().iloc[0]
        return default

    def _safe_mean(group, cols):
        import numpy as _np
        for c in cols:
            if c in group and group[c].notna().any():
                v = _np.asarray(group[c].values, dtype=float)
                v = v[_np.isfinite(v)]
                if v.size:
                    return float(_np.mean(v))
        return float("nan")

    def _safe_sum(group, cols):
        import numpy as _np
        for c in cols:
            if c in group and group[c].notna().any():
                v = _np.asarray(group[c].values, dtype=float)
                v = v[_np.isfinite(v)]
                if v.size:
                    return float(_np.sum(v))
        return float("nan")

    # ---- Ranking (simple, used to compute per_run_summary only) ----------------------------
    def _rank_from_combined_simple(df_all: pd.DataFrame):
        import numpy as _np
        df = df_all.copy()
        if "strategy_return" not in df.columns and "cstrategy" in df.columns:
            df["strategy_return"] = df["cstrategy"] - 1.0
        df = df.sort_values(["model_type", "rep", "test_end"])

        def _agg_one(g: pd.DataFrame) -> pd.Series:
            n = g.shape[0]
            if "equity_strategy" in g.columns and not g["equity_strategy"].dropna().empty:
                final_eq = float(g["equity_strategy"].dropna().iloc[-1])
                eq_series = g["equity_strategy"].astype(float).values
            else:
                eq_series = _np.cumprod(1.0 + g["strategy_return"].astype(float).values)
                final_eq = float(eq_series[-1]) if n else _np.nan
            mret = g["strategy_return"].astype(float).values
            ann_ret = (final_eq ** (12.0 / n) - 1.0) if n > 0 else _np.nan
            m_active = mret[_np.abs(mret) > 1e-12]
            if m_active.size < 3:
                ann_vol = _np.nan; sharpe = _np.nan
            else:
                vol_m = float(_np.std(m_active, ddof=1))
                ann_vol = float(vol_m * _np.sqrt(12.0))
                sharpe = float(ann_ret / ann_vol) if _np.isfinite(ann_vol) and ann_vol > 0 else _np.nan
            dd = float(_np.min(eq_series / _np.maximum.accumulate(eq_series) - 1.0)) if n else _np.nan
            calmar = (ann_ret / abs(dd)) if (isinstance(dd, float) and dd < 0) else _np.nan
            win_rate = float((mret > 0).mean()) if n else _np.nan
            return pd.Series({
                "months": n, "final_equity": final_eq, "ann_return": ann_ret,
                "ann_vol": ann_vol, "sharpe": sharpe, "calmar": calmar, "win_rate": win_rate
            })

        per_run = (
            df.groupby(["model_type","rep","run_seed"], dropna=False)
            .apply(_agg_one, include_groups=False)
            .reset_index()
        )

        # keep ranking_df available for CSV saving (not printed)
        ranking = (
            per_run.groupby("model_type", as_index=False)
                .agg(runs=("rep","nunique"),
                     months_mean=("months","mean"),
                     sharpe_mean=("sharpe","mean"),
                     calmar_mean=("calmar","mean"),
                     ann_return_mean=("ann_return","mean"),
                     final_equity_median=("final_equity","median"))
        )
        ranking["composite"] = (
            0.50*ranking["sharpe_mean"] + 0.30*ranking["calmar_mean"] + 0.20*ranking["ann_return_mean"]
        )
        ranking = ranking.sort_values(["composite","final_equity_median"], ascending=[False, False]).reset_index(drop=True)
        ranking["rank"] = _np.arange(1, len(ranking) + 1)
        return per_run, ranking

    per_run_summary, ranking_df = _rank_from_combined_simple(combined_all)

    # === Wide per-run detailed table (ONLY table we print) ===================================
    if PRINT_EVAL_TABLES:
        try:
            # Build a rich, one-row-per-(model,rep,seed) table
            gcols = [c for c in ["model_type","rep","run_seed"] if c in combined_all.columns]
            if not gcols:
                print("⚠️ Cannot render per-run table: missing model/rep/seed columns.")
            else:
                rows = []
                # stable order: sort by sharpe desc then model
                _order = (per_run_summary.sort_values(["sharpe","model_type","rep"], ascending=[False, True, True])
                                    if {"sharpe","model_type","rep"} <= set(per_run_summary.columns)
                                    else per_run_summary)
                seen = set()
                for _, rr in _order.iterrows():
                    key = (rr.get("model_type",""), int(rr.get("rep",0)),
                           int(rr.get("run_seed",0) if pd.notna(rr.get("run_seed",None)) else 0))
                    if key in seen:
                        continue
                    seen.add(key)
                    model, rep, seed = key
                    grp = combined_all[(combined_all.get("model_type")==model) &
                                       (combined_all.get("rep")==rep) &
                                       (combined_all.get("run_seed")==seed)]
                    if grp.empty:
                        continue

                    test_start = _safe_first(grp, ["test_start","val_start","start_ts"])
                    test_end   = _safe_first(grp.iloc[[-1]], ["test_end","val_end","end_ts"])
                    months     = int(rr.get("months", grp.shape[0]) if pd.notna(rr.get("months", None)) else grp.shape[0])

                    final_eq   = float(rr.get("final_equity", float("nan")))
                    ann_ret    = float(rr.get("ann_return", float("nan")))
                    ann_vol    = float(rr.get("ann_vol", float("nan")))
                    sharpe     = float(rr.get("sharpe", float("nan")))
                    calmar     = float(rr.get("calmar", float("nan")))
                    win_rate   = float(rr.get("win_rate", float("nan")))

                    trades     = _safe_sum(grp, ["trades","n_trades","positions_opened"])
                    active     = _safe_mean(grp, ["active_rate","coverage","coverage_rate"])
                    da         = _safe_mean(grp, ["directional_accuracy","hit_rate","win_rate"])
                    prec       = _safe_mean(grp, ["precision_macro","precision"])
                    f1_        = _safe_mean(grp, ["f1_macro","f1"])
                    pph        = _safe_mean(grp, ["profit_per_hit","avg_profit_per_trade"])

                    label_thr  = _safe_first(grp, ["label_threshold","thr","threshold"])

                    lags_used  = _safe_first(grp, ["lags","lags_range"])
                    
                    # Effective confidence threshold (per-run summary):
                    # Use median across months. Prefer canonical column name(s).
                    eff_conf = float("nan")
                    try:
                        if "confidence_threshold" in grp.columns:
                            eff_conf = float(pd.to_numeric(grp["confidence_threshold"], errors="coerce").median())
                        if (not np.isfinite(eff_conf)) and ("confidence_threshold_used" in grp.columns):
                            eff_conf = float(pd.to_numeric(grp["confidence_threshold_used"], errors="coerce").median())
                        # Backward compat for older schemas (best-effort)
                        if (not np.isfinite(eff_conf)):
                            for _c in ["eff_conf", "confidence_used", "conf_threshold_used", "conf_threshold"]:
                                if _c in grp.columns:
                                    eff_conf = float(pd.to_numeric(grp[_c], errors="coerce").median())
                                    if np.isfinite(eff_conf):
                                        break
                    except Exception:
                        pass

                    rows.append([
                        str(model),
                        int(rep),
                        (int(seed) if pd.notna(seed) else ""),
                        str(test_start).split("+")[0] if test_start is not None else "",
                        str(test_end).split("+")[0]   if test_end   is not None else "",
                        months,
                        (f"{trades:.0f}"  if pd.notna(trades)   else "—"),
                        (f"{active:.5f}"  if pd.notna(active)   else "—"),
                        (f"{sharpe:.3f}"  if pd.notna(sharpe)   else "—"),
                        (f"{calmar:.3f}"  if pd.notna(calmar)   else "—"),
                        (f"{ann_ret:.5f}" if pd.notna(ann_ret)  else "—"),
                        (f"{final_eq:.5f}"if pd.notna(final_eq) else "—"),
                        (f"{da:.5f}"      if pd.notna(da)       else "—"),
                        (f"{prec:.5f}"    if pd.notna(prec)     else "—"),
                        (f"{f1_:.5f}"     if pd.notna(f1_)      else "—"),
                        (f"{pph:.6f}"     if pd.notna(pph)      else "—"),
                        (f"{float(label_thr):.6f}" if pd.notna(label_thr) else "—"),
                        (f"{float(eff_conf):.3f}"  if pd.notna(eff_conf)  else "—"),
                        (int(lags_used) if pd.notna(lags_used) else "—"),
                    ])
                    if len(rows) >= EVAL_TABLE_MAX_ROWS:
                        break

                _fmt_table_ascii(
                    ["model","rep","seed","test_start","test_end","months","trades","active",
                     "SR","Calmar","AnnRet","FinalEq","DA","Prec","F1",
                     "Profit/Hit","LabelThr","EffConf","lags"],
                    rows,
                    title="📈 Real-trading per-run summary (detailed, precise, top by Sharpe)"
                )
        except Exception as _e:
            print(f"[tables] detailed per-run table skipped: {_e}")

if __name__ == "__main__":
    main()
