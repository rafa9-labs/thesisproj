"""Main Optuna tuning runner: run_optuna_tuning."""
import datetime
import gc
import json
import math
import os
import threading
import time
import traceback
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy

import numpy as np
import optuna
from joblib import parallel_backend
from threadpoolctl import threadpool_limits

from utilsNoWFO import (
    TRAIN_TEST_MONTHS,
    TRAIN_TEST_MONTHS_DEBUG,
    _bad_objective_for_direction,
    _norm_optuna_direction,
    compute_required_test_warmup_bars,
    log_print,
    save_optuna_progress_from_study,
    target_coverage_policy,
)
from pipeline.tuning.helpers import DISABLE_OPTUNA_PRUNING

TRAIN_TEST_DEBUG_MODE = False


from pipeline.tuning.objective import optuna_objective

def run_optuna_tuning(
        train_data, base_features, evaluate_cv_func, cv_config, models_to_test,
        n_trials=1, n_startup_trials=10, return_top_n=3, study=None, sampler_seed=None,
        month_out_dir: str | None = None, month_ix: int | None = None,
        max_hpo_duration_minutes: float = 0,
        sampler_method: str = "tpe"):

    """
    Runs Optuna tuning for a given model configuration, evaluates via CV,
    and returns the best parameter set along with embedded Top-N trial info.

    Returns:
        best_params (dict): Best trial parameters with extra Top-N metadata.
        best_score (float): study.best_value (TRUE Sharpe).
        topN_params (list[dict]): List of Top-N param dicts (ranked best->worse).
        study (optuna.study.Study): the Optuna study.
        consensus_pool (list[dict]): Small pool of candidate configs (all valid trials) for consensus selection.
     """
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner
    import os, json, datetime

    def _create_sampler(method, seed, n_startup, model_name=""):
        tpe_ei = int(os.environ.get("TPE_EI_CANDIDATES", "64"))
        if method == "tpe":
            s = TPESampler(
                n_startup_trials=n_startup, multivariate=True, group=True,
                seed=seed, n_ei_candidates=tpe_ei,
            )
        elif method == "random":
            s = RandomSampler(seed=seed)
        elif method == "cmaes":
            s = CmaEsSampler(seed=seed, warn_independent_sampling=False)
        else:
            s = TPESampler(
                n_startup_trials=n_startup, multivariate=True, group=True,
                seed=seed, n_ei_candidates=tpe_ei,
            )
        tag = f" model={model_name}" if model_name else ""
        print(f"[Optuna] {type(s).__name__} seed={seed}{tag}"
              + (f" n_startup_trials={n_startup}" if method == "tpe" else ""))
        return s

    # Reset per-run hyperparameter boundary diagnostics
    global HP_BOUNDARY_HITS, HP_BOUNDARY_HITS_MIN, HP_BOUNDARY_HITS_MAX, HP_BOUNDARY_RANGES
    HP_BOUNDARY_HITS = {}
    HP_BOUNDARY_HITS_MIN = {}
    HP_BOUNDARY_HITS_MAX = {}
    HP_BOUNDARY_RANGES = {}

    # Lazy imports (avoid circulars)
    try:
        from utilsNoWFO import save_optuna_progress_from_study
    except Exception:
        save_optuna_progress_from_study = None

    try:
        from utilsNoWFO import save_feature_frequency_from_trials
    except Exception:
        save_feature_frequency_from_trials = None
        
    try:
        from utilsNoWFO import save_optuna_learning_summary
    except Exception:
        save_optuna_learning_summary = None
        
    try:
        from utilsNoWFO import save_hpo_config_to_disk, get_hpo_config_dir
    except Exception:
        save_hpo_config_to_disk = None
        get_hpo_config_dir = None

    try:
        from pipeline._imports import SKIP_PLOTS, SAVE_TRIAL_FEATURE_FREQ
    except Exception:
        SKIP_PLOTS = True
        SAVE_TRIAL_FEATURE_FREQ = False
        
    # Work on a single consolidated copy of the DF across all trials
    train_data = train_data.copy()
    try:
        # Defragment once so Pandas storage is compact
        train_data._consolidate_inplace()
    except Exception:
        pass

    # Defensive
    models_to_test = sorted(list(models_to_test))
    
    
    # ------------------------------------------------------------
    # Model-family detection (shared by sampler + pruner)
    # ------------------------------------------------------------
    try:
        _model_name = models_to_test[0] if isinstance(models_to_test, (list, tuple)) else models_to_test
    except Exception:
        _model_name = models_to_test
    _model_name = str(_model_name).lower()
    _is_deep_family = (
        _model_name in {"cnn", "lstm", "transformer", "gru", "gru_lstm", "dqn"} or _model_name.startswith("ensemble_")
    )
    
    # Ensure CV parallelism is wired even if caller omitted it
    cv_config = dict(cv_config or {})
    if "cv_n_jobs" not in cv_config:
        import os
        import multiprocessing
        _cv_jobs_raw = (os.getenv("CV_JOBS", "") or "").strip()
        try:
            cv_config["cv_n_jobs"] = int(_cv_jobs_raw) if _cv_jobs_raw else (multiprocessing.cpu_count() or 16)
        except (ValueError, TypeError):
            cv_config["cv_n_jobs"] = multiprocessing.cpu_count() or 16
        
    # ------------------------------------------------------------
    # Fast path: n_trials <= 0 => load cached HPO config (no Optuna)
    # ------------------------------------------------------------
    try:
        _ntr = int(n_trials or 0)
    except Exception:
        _ntr = 0

    if _ntr <= 0:
        import os, json, math

        base = os.environ.get(
            "MLB_HPO_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpo"),
        )

        safe = str(_model_name).replace("/", "_")
        candidates = [
            os.path.join(base, f"model_{_model_name}_hpo.json"),   # MLBacktester schema
            os.path.join(base, f"{safe}_best_config.json"),        # utilsNoWFO schema
            os.path.join(base, f"{_model_name}_best_config.json"), # legacy
        ]

        data = None
        used_path = None
        for p in candidates:
            if p and os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                    used_path = p
                    break
                except Exception:
                    data = None
                    used_path = None

        best = None
        topN = []
        if isinstance(data, dict):
            best = data.get("best_params") or data.get("best") or data
            topN = data.get("topN_params") or data.get("topN") or []

        if not isinstance(best, dict) or not best:
            raise RuntimeError(
                f"n_trials=0 but no cached HPO config found for '{_model_name}'. "
                f"Looked in: {candidates}. Set MLB_HPO_DIR to override."
            )

        # Preserve a score if present (used only for logging)
        try:
            _score = float(best.get('__cv_value', best.get('cv_value', best.get('value', float('nan')))))
        except Exception:
            _score = float('nan')

        # Strip internal metadata keys that can leak into model constructors
        best = {k: v for k, v in best.items() if not str(k).startswith("__")}
        best.setdefault("model_type", _model_name)

        # Minimal dummy study so downstream logging doesn't crash
        class _DummyTrial:
            number = -1

        class _DummyStudy:
            best_trial = _DummyTrial()
            best_value = _score
            trials = []

        study = _DummyStudy()
        print(f"[HPO] n_trials=0 -> loaded cached config for {_model_name} from {used_path}")
        return best, study.best_value, list(topN or []), study, list(topN or [])

        
    
    # Consensus pool knobs (used later for Top-N consensus selection)
    consensus_pool_max_trials = int(cv_config.get("consensus_pool_max_trials", 0))
    consensus_pool_min_perf_frac = float(
        cv_config.get("consensus_pool_min_perf_frac", cv_config.get("topN_min_perf_frac", 0.60))
    )
    
    # --- Precompute volatility stats once for label_threshold scaling ---
    vol_stats: dict = {}
    if (
        train_data is not None
        and hasattr(train_data, "columns")
        and "returns" in train_data.columns
    ):
        import numpy as _np
        r = train_data["returns"].astype("float64").dropna()
        if r.size > 0:
            sigma = float(r.rolling(48).std().median())
            sigma = float(_np.clip(sigma, 1e-5, 5e-3))
            vol_stats["sigma48"] = sigma

    # ------------------------------------------------------------
    # Patch: model-aware startup trials
    # If caller passes default 10 -> treat as "auto".
    # If caller passes anything else -> respect it.
    # ------------------------------------------------------------
    try:
        _n_startup_arg = int(n_startup_trials)
    except Exception:
        _n_startup_arg = 10

    if _n_startup_arg == 10:
        if _is_deep_family:
            _n_startup = int(cv_config.get("n_startup_trials_deep", 25))
        else:
            _n_startup = int(cv_config.get("n_startup_trials_classical", 15))
    else:
        _n_startup = _n_startup_arg

    # Clamp startup trials to total trials — avoid 15 startup + 10 total = all random
    try:
        _total = int(n_trials)
        if _n_startup > _total:
            _n_startup = max(1, _total // 2)
    except Exception:
        pass

    sampler = _create_sampler(sampler_method, sampler_seed, _n_startup, _model_name)
    
    from optuna.pruners import SuccessiveHalvingPruner, NopPruner

    if _is_deep_family:
        _pruner_min_resource = int(cv_config.get("pruner_min_resource_deep", 4))
        _pruner_reduction_factor = int(cv_config.get("pruner_reduction_factor_deep", 3))
    else:
        _pruner_min_resource = int(cv_config.get("pruner_min_resource_classical", 2))
        _pruner_reduction_factor = int(cv_config.get("pruner_reduction_factor_classical", 2))

    # Cap min_resource to n_trials so low-trial runs don't get fully pruned
    if n_trials is not None and _pruner_min_resource > n_trials:
        _pruner_min_resource = max(1, n_trials)

    # ASHA-style pruning (default), or no pruning if disabled
    if DISABLE_OPTUNA_PRUNING:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=_pruner_min_resource,       # "min_folds before prune"
            reduction_factor=_pruner_reduction_factor,
            bootstrap_count=0,
            min_early_stopping_rate=0                # allow pruning once min_resource is reached
        )
        print(f"[Optuna] SuccessiveHalvingPruner(min_resource={_pruner_min_resource}, "
              f"reduction_factor={_pruner_reduction_factor}) model={_model_name}")



    if study is None:

        sampler = _create_sampler(sampler_method, sampler_seed, _n_startup, _model_name)

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    _progress_cb = cv_config.get("_progress_callback", None) if isinstance(cv_config, dict) else None
    _model_name_for_cb = str(models_to_test[0]) if isinstance(models_to_test, (list, tuple)) and models_to_test else str(models_to_test)
    _n_trials_for_cb = int(n_trials) if n_trials is not None else 0
    _cv_blocks_for_cb = int(cv_config.get("cv_blocks", 5)) if isinstance(cv_config, dict) else 5

    _trial_counter = [0]

    func = lambda trial: optuna_objective(
        trial,
        train_data,
        base_features,
        evaluate_cv_func,
        cv_config,
        models_to_test,
        vol_stats=vol_stats,
    )

    def _func_with_progress(trial):
        _trial_start_time[0] = time.time()
        _trial_counter[0] += 1
        print(f"[HEARTBEAT] Starting trial {_trial_counter[0]}/{_n_trials_for_cb}")
        pruned = None
        _prune_reason = None
        _exc = None
        result = None
        try:
            result = func(trial)
        except optuna.TrialPruned as _e:
            pruned = "PRUNED"
            _prune_reason = str(_e)[:120]
            _exc = _e
        except Exception as _e:
            pruned = "FAIL"
            _prune_reason = str(_e)[:120]
            _exc = _e

        if _progress_cb:
            try:
                best_so_far = None
                if pruned is None:
                    try:
                        best_so_far = float(study.best_value)
                    except Exception:
                        pass
                trial_state = str(trial.state) if hasattr(trial, "state") else "COMPLETE"
                if _prune_reason:
                    trial_state = f"{pruned}:{_prune_reason}"
                _progress_cb("hpo_trial", _model_name_for_cb, {
                    "trial": _trial_counter[0],
                    "total_trials": _n_trials_for_cb,
                    "cv_blocks": _cv_blocks_for_cb,
                    "score": result,
                    "params": dict(trial.params) if trial.params else {},
                    "best_score_so_far": best_so_far,
                    "trial_state": trial_state,
                })
            except Exception:
                pass

        if _exc is not None:
            raise _exc
        return result


    # --- CPU/BLAS parallelism controls (single wide trial) ---
    import os, multiprocessing
    from threadpoolctl import threadpool_limits

    # One Optuna worker only (sequential trials)
    n_jobs = int(os.getenv("OPTUNA_N_JOBS", "1"))  # keep = 1

    # Wide intra-trial parallelism via BLAS threads:
    _bl_env = os.getenv("BLAS_THREADS_PER_TRIAL", "").strip()
    if _bl_env:
        blas_threads = max(1, int(_bl_env))
    else:
        # CPU-centric fallback: ~75% of logical cores, leave 2 for OS
        _cpu = os.cpu_count() or multiprocessing.cpu_count() or 8
        blas_threads = max(2, min(_cpu - 2, int(round(_cpu * 0.75))))

    print(f"[Optuna] sequential n_jobs={n_jobs} | BLAS_THREADS_PER_TRIAL={blas_threads} | CV_JOBS={os.getenv('CV_JOBS', '?')}")

    _hpo_deadline = None
    if max_hpo_duration_minutes and max_hpo_duration_minutes > 0:
        _hpo_deadline = time.time() + max_hpo_duration_minutes * 60
        print(f"[Optuna] Time budget: {max_hpo_duration_minutes:.1f} min (deadline {_hpo_deadline:.0f})")

    def _hpo_timeout_callback(study, trial):
        if _hpo_deadline is not None and time.time() >= _hpo_deadline:
            study.stop()

    def _throttle_callback(study, trial):
        try:
            from pipeline.resource_monitor import get_throttle_signal
            sig = get_throttle_signal()
            if sig and sig.delay > 0:
                time.sleep(sig.delay)
        except Exception:
            pass

    study.set_user_attr("max_hpo_duration_minutes", max_hpo_duration_minutes)

    _trial_start_time = [time.time()]
    _per_trial_timeout = int(os.environ.get("OPTUNA_PER_TRIAL_TIMEOUT", "1800"))
    _watchdog_stop = threading.Event()

    def _per_trial_watchdog(study_ref, trial_start_ref, timeout_s, stop_event):
        while not stop_event.is_set():
            elapsed = time.time() - trial_start_ref[0]
            if elapsed > timeout_s:
                mins = elapsed / 60
                print(f"[TIMEOUT] Single trial running {mins:.1f} min > {timeout_s//60} min -- stopping study")
                try:
                    study_ref.stop()
                except Exception:
                    pass
                break
            stop_event.wait(60)

    _wd = threading.Thread(target=_per_trial_watchdog,
                           args=(study, _trial_start_time, _per_trial_timeout, _watchdog_stop),
                           daemon=True)
    _wd.start()

    # Cap NumPy/SciPy/Sklearn/XGB BLAS threads inside the trial
    with threadpool_limits(limits=blas_threads):
        # ------------------------------------------------------------
        # Patch: plateau early-stop (optional)
        # Stop if best_value hasn't improved by >= plateau_delta for
        # plateau_patience consecutive trials.
        # ------------------------------------------------------------
        plateau_patience = int(cv_config.get("plateau_patience", 15) or 15)
        plateau_delta = float(cv_config.get("plateau_delta", 0.0) or 0.0)
        plateau_min_trials = int(cv_config.get("plateau_min_trials", 0) or 0)

        if plateau_patience <= 0:
            _cb = [_throttle_callback]
            if _hpo_deadline:
                _cb.append(_hpo_timeout_callback)
            study.optimize(_func_with_progress, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True,
                           callbacks=_cb)
        else:
            print(f"[Optuna] Plateau stop enabled: patience={plateau_patience} "
                  f"delta={plateau_delta} min_trials={plateau_min_trials}")

            _target_trials = int(n_trials) if n_trials is not None else 0
            _target_trials = max(0, _target_trials)

            # Support resumed studies (if study already has trials)
            try:
                _best = float(getattr(study, "best_value", None))
            except Exception:
                _best = None
            _no_improve = 0

            for _i in range(_target_trials):
                study.optimize(_func_with_progress, n_trials=1, n_jobs=n_jobs, gc_after_trial=True,
                               callbacks=[_throttle_callback])

                try:
                    _after = study.best_value
                except (AttributeError, ValueError):
                    _after = None
                if _after is None:
                    continue

                try:
                    _after_f = float(_after)
                except Exception:
                    # If comparison fails, don't early-stop
                    _no_improve = 0
                    continue

                if _best is None:
                    _best = _after_f
                    _no_improve = 0
                elif _after_f >= (_best + plateau_delta):
                    _best = _after_f
                    _no_improve = 0
                else:
                    _no_improve += 1

                _done = _i + 1
                _min_ok = (_done >= plateau_min_trials) if plateau_min_trials > 0 else True
                if _min_ok and (_no_improve >= plateau_patience):
                    print(f"[Optuna] Plateau stop: no improvement >= {plateau_delta} "
                          f"for {plateau_patience} trials (best={_best}). "
                          f"Stopped at {_done}/{_target_trials} trials.")
                    break


    _watchdog_stop.set()
    if _wd.is_alive():
        _wd.join(timeout=5)

    # --- post-study cleanup (runs once per study) ---
    try:
        import tensorflow as _tf
        _tf.keras.backend.clear_session()
    except Exception:
        pass

    # Optional: only helps if loky reusable executor was created somewhere.
    # Use timeout to prevent deadlock if workers are stuck in native code.
    try:
        from joblib.externals.loky import get_reusable_executor
        try:
            get_reusable_executor().shutdown(wait=True, timeout=10)
        except TypeError:
            get_reusable_executor().shutdown(wait=False)
    except Exception:
        pass

    import gc as _gc
    _gc.collect()

    # Log hyperparameter boundary hits (how often we sampled near the bounds)
    try:
        if HP_BOUNDARY_HITS:
            log_print("[Optuna] Hyperparameter boundary hits across all trials:", level="COMPACT")
            for _name, _count in sorted(HP_BOUNDARY_HITS.items(), key=lambda kv: (-kv[1], kv[0])):
                log_print(f"  - {_name}: {_count}", level="COMPACT")
                
            # ------------------------------------------------------------
            # Split min/max edge pressure and recommend small range expansion.
            # Does NOT change search ranges automatically.
            # ------------------------------------------------------------
            try:
                _n_trials_total = int(len(getattr(study, "trials", []) or []))
                _n_trials_total = max(1, _n_trials_total)
                _ratio_thr = float((cv_config or {}).get("range_suggest_ratio_thr", 0.25))
                _expand = float((cv_config or {}).get("range_suggest_expand_frac", 0.25))

                _items = []
                for _p in set(list(HP_BOUNDARY_HITS_MIN.keys()) + list(HP_BOUNDARY_HITS_MAX.keys()) + list(HP_BOUNDARY_RANGES.keys())):
                    _hm = int(HP_BOUNDARY_HITS_MIN.get(_p, 0) or 0)
                    _hM = int(HP_BOUNDARY_HITS_MAX.get(_p, 0) or 0)
                    if _hm <= 0 and _hM <= 0:
                        continue
                    _items.append((_p, _hm, _hM))

                def _is_loglike(_name: str, _low: float, _high: float) -> bool:
                    _n = str(_name).lower()
                    if any(k in _n for k in ["lr", "learning_rate", "label_threshold", "alpha", "beta", "gamma"]):
                        return True
                    try:
                        if _low > 0 and (_high / _low) >= 10.0:
                            return True
                    except Exception:
                        pass
                    return False

                _printed = 0
                for _p, _hm, _hM in sorted(_items, key=lambda x: (-(max(x[1], x[2])), x[0])):
                    _rng = HP_BOUNDARY_RANGES.get(_p, None)
                    if not _rng:
                        continue
                    _low, _high = float(_rng[0]), float(_rng[1])
                    if not (_high > _low):
                        continue
                    _span = _high - _low

                    _rmin = _hm / float(_n_trials_total)
                    _rmax = _hM / float(_n_trials_total)
                    if _rmin < _ratio_thr and _rmax < _ratio_thr:
                        continue

                    _loglike = _is_loglike(_p, _low, _high)
                    _sug_low = None
                    _sug_high = None

                    if _rmin >= _ratio_thr:
                        if _loglike and _low > 0:
                            _sug_low = _low / (1.0 + _expand)
                        else:
                            _sug_low = _low - (_expand * _span)
                        if _low >= 0.0:
                            _sug_low = max(0.0, float(_sug_low))

                    if _rmax >= _ratio_thr:
                        if _loglike and _high > 0:
                            _sug_high = _high * (1.0 + _expand)
                        else:
                            _sug_high = _high + (_expand * _span)

                    if _printed == 0:
                        log_print("[HPO][RANGE-SUGGEST] boundary pressure detected; consider adjusting ranges:", level="COMPACT")

                    _min_txt = f"{_hm}/{_n_trials_total}"
                    _max_txt = f"{_hM}/{_n_trials_total}"
                    _sl = "NA" if _sug_low is None else f"{float(_sug_low):.6g}"
                    _sh = "NA" if _sug_high is None else f"{float(_sug_high):.6g}"
                    log_print(
                        f"[HPO][RANGE-SUGGEST] param={_p} hits_min={_min_txt} hits_max={_max_txt} "
                        f"low={_low:.6g} high={_high:.6g} suggest_low={_sl} suggest_high={_sh}",
                        level="COMPACT",
                    )
                    _printed += 1
                    if _printed >= int((cv_config or {}).get("range_suggest_max_params", 12)):
                        break
            except Exception:
                pass
    except Exception:
        # Diagnostics-only; never break tuning if logging fails.
        pass
    
    # Completed trials only
    from optuna.trial import TrialState
    
    TOPN_FOR_WFO = 5
    
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]

    if not completed:
        failed  = [t for t in study.trials if t.state == TrialState.FAIL]
        pruned  = [t for t in study.trials if t.state == TrialState.PRUNED]
        _msgs = []
        for t in failed[-5:]:
            _err = "unknown"
            try:
                _err = t.user_attrs.get("error", t.user_attrs.get("exception", "unknown"))
            except Exception:
                pass
            _msgs.append(f"  Trial {t.number}: {_err}")
        _summary = (
            f"No completed Optuna trials "
            f"(completed=0, failed={len(failed)}, pruned={len(pruned)}, total={len(study.trials)}). "
        )
        if _msgs:
            _summary += "Last failures:\n" + "\n".join(_msgs)
        else:
            _summary += "No failure info available (all pruned?)."
        raise RuntimeError(_summary)

    # [POINT] Re-rank by Deflated Sharpe proxy (DSR)
    try:
        from utilsNoWFO import compute_dsr_scores
        _scores = [float(t.value) for t in completed]
        _dsr    = compute_dsr_scores(_scores)
        for t, d in zip(completed, _dsr):
            try:
                t.set_user_attr("dsr", float(d))
            except Exception:
                pass

        # Sort by DSR descending (more conservative than raw Sharpe)
        # Optional tie-breaker: if a calibration metric (brier or nll) is present
        # in trial.user_attrs, prefer better-calibrated configs among similar DSR.
        def _rank_key(t):
            dsr = float(t.user_attrs.get("dsr", 0.0))
            # Lower is better for Brier/NLL; default 0.0 means "no info"
            brier = t.user_attrs.get("brier", None)
            nll   = t.user_attrs.get("nll", None)
            if brier is not None:
                return (dsr, -float(brier))
            if nll is not None:
                return (dsr, -float(nll))
            return (dsr, 0.0)

        completed.sort(key=_rank_key, reverse=True)

        
    except Exception:
        # Fallback: raw value
        completed.sort(key=lambda t: t.value, reverse=True)
        
    # --- Diversity-aware Top-N helper ---
    def _build_diverse_top_trials(trials, top_n: int):
        """
        Select up to top_n trials, preserving performance order, but avoiding
        near-duplicate configs in a normalized hyperparameter space.

        Two trials are treated as near-duplicates if they:
          - share the same model_type and strategy_type,
          - use the same active feature subset (use_* flags),
          - and lie within a small normalized radius (GEOM_RADIUS) for key knobs:
            lags_range, lag_depth, target_active_rate, label_threshold,
            alpha_vol_z, beta_spread_norm, gamma_slip_norm.

        This approximates a "robust region" around the best trial instead of
        picking many almost-identical spike solutions.
        """
        import math

        selected   = []
        signatures = []

        # Hyperparameter ranges used only for distance normalization.
        # Keep these aligned with the suggest_* ranges above.
        HP_RANGES = {
            "lags_range":         (8.0, 40.0),   # 8-24 (ensembles) or 12-40 (others) -> global span
            "lag_depth":          (1.0, 4.0),    # 1-3 or 2-4
            "target_active_rate": (0.24, 0.26),  # as in trial.suggest_float(...)
            "label_threshold":    (5e-5, 5e-3),  # covers dynamic sigma-based bounds
            "alpha_vol_z":        (0.0, 0.03),
            "beta_spread_norm":   (0.0, 0.08),
            "gamma_slip_norm":    (0.0, 0.08),
        }

        # Radius in normalized space (~=10-15% of search span).
        GEOM_RADIUS = 0.10

        def _norm_dist(v1, v2, key):
            lo, hi = HP_RANGES.get(key, (None, None))
            if lo is None or hi is None or hi <= lo:
                return math.inf
            if v1 is None or v2 is None:
                return math.inf
            try:
                return abs(float(v1) - float(v2)) / (hi - lo)
            except Exception:
                return math.inf

        def _trial_signature(t):
            """
            Compact signature capturing:
              - model_type, strategy_type
              - active feature subset (use_* flags)
              - the raw values of the key hyperparameters we distance-check.
            """
            p = getattr(t, "params", {}) or {}
            model = p.get("model_type", None)
            strat = p.get("strategy_type", None)

            # Feature subset: names of toggles that are True.
            active_feats = tuple(
                sorted(k for k, v in p.items() if k.startswith("use_") and bool(v))
            )

            sig_vals = {k: p.get(k, None) for k in HP_RANGES.keys()}

            return (model, strat, active_feats, sig_vals)

        def _too_similar(sig, others):
            model, strat, feats, vals = sig
            if strat is None:
                # If no strategy_type, don't try to merge aggressively.
                return False

            for o_model, o_strat, o_feats, o_vals in others:
                # Only compare inside the same family + strategy + feature subset.
                if model != o_model:
                    continue
                if o_strat is None or strat != o_strat:
                    continue
                if feats != o_feats:
                    continue

                # Compute max normalized distance across the tracked knobs.
                max_d      = 0.0
                any_finite = False
                for k in HP_RANGES.keys():
                    d = _norm_dist(vals.get(k), o_vals.get(k), k)
                    if not math.isfinite(d):
                        continue
                    any_finite = True
                    if d > max_d:
                        max_d = d

                # If we had at least one comparable dimension and everything is within
                # GEOM_RADIUS, treat as "same robust region" -> near-duplicate.
                if any_finite and max_d <= GEOM_RADIUS:
                    return True

            return False

        for t in trials:
            sig = _trial_signature(t)
            if _too_similar(sig, signatures):
                # Skip very similar config; we already have a representative.
                continue
            selected.append(t)
            signatures.append(sig)
            if len(selected) >= top_n:
                break

        if not selected:
            # Fallback: original behavior
            return trials[:top_n]
        return selected

    # Top-N (diversity-aware)
    top_n = max(1, int(return_top_n))
    top_trials = _build_diverse_top_trials(completed, top_n)
    
    def _merged_trial_params(_t):
        """Merge trial.params with user_attrs["full_params"] without losing keys.

        Some runs store derived/expanded keys (e.g., resolved roll windows) in full_params,
        but in a few cases full_params can be partial. We therefore merge on top of
        trial.params and refuse to clobber a non-empty value with an empty placeholder.
        """
        base = dict(getattr(_t, "params", {}) or {})
        fp = None
        try:
            fp = _t.user_attrs.get("full_params", None)
        except Exception:
            fp = None
        if isinstance(fp, dict) and fp:
            for k, v in fp.items():
                # Do not overwrite a non-empty base value with an empty/None placeholder.
                if v is None:
                    base.setdefault(k, v)
                    continue
                if isinstance(v, str) and v.strip() == "":
                    base.setdefault(k, v)
                    continue
                if isinstance(v, (list, tuple, set, dict)) and len(v) == 0:
                    base.setdefault(k, v)
                    continue
                base[k] = v
        return base

    top_params = [_merged_trial_params(t) for t in top_trials]

    
    # Decide output location:
    # - If month_out_dir is provided -> save ONLY the per-month plot there with the project's filename.
    # - Else (legacy) -> create an optuna_runs/<id> audit folder with the usual artifacts.
    legacy_optuna_dir = None
    if month_out_dir:
        out_dir = month_out_dir
    else:
        run_id  = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        legacy_optuna_dir = os.path.join("optuna_runs", run_id)
        out_dir = legacy_optuna_dir
    os.makedirs(out_dir, exist_ok=True)

    top_path = os.path.join(out_dir, f"top{top_n}.json")
    
    if legacy_optuna_dir and (save_optuna_learning_summary is not None):
        try:
            save_optuna_learning_summary(
                study,
                os.path.join(out_dir, "learning_summary.json"),
                n_startup=int(n_startup_trials),
            )

        except Exception as _e:
            print(f"[WARN] Could not save learning summary: {_e}")

    if legacy_optuna_dir:
        try:
            from optuna.importance import get_param_importances
            imps = get_param_importances(study)  # uses study.best_value as target by default
            import json as _json, collections as _c
            imps_path = os.path.join(out_dir, "param_importances.json")

            # Convert OrderedDict/Keys to plain dict for JSON
            imps = {str(k): float(v) for k, v in imps.items()}
            with open(imps_path, "w") as f:
                _json.dump(imps, f, indent=2, sort_keys=True)
            print(f"[OK] Saved Optuna param importances -> {imps_path}")
        except Exception as _e:
            print(f"[WARN] Could not compute/save param importances: {_e}")

    # --- Build Top-N, normalize, and save audit JSON ---

    from optuna.trial import TrialState
    from optuna.study import StudyDirection

    # Local helpers
    def _normalize_roll_windows_inplace(p: dict) -> None:
        """Ensure 'roll_windows' exists and drop legacy/versioned selector keys."""
        rk = p.get("roll_windows_key_v2") or p.get("roll_windows_key")
        if "roll_windows" not in p and rk is not None:
            p["roll_windows"] = [int(x) for x in str(rk).split(",") if str(x).strip() != ""]
        p.pop("roll_windows_key_v2", None)
        p.pop("roll_windows_key", None)

    def _json_sanitize_inplace(p: dict) -> None:
        """Cast numpy scalars to native types + sanitize lists for JSON."""
        import numpy as np
        for k, v in list(p.items()):
            if isinstance(v, np.generic):
                p[k] = v.item()
            elif isinstance(v, (list, tuple)):
                p[k] = [x.item() if isinstance(x, np.generic) else x for x in v]
                
    def _ensure_lags_inplace(p: dict) -> None:
        """Ensure 'lags' exists to avoid warnings later."""
        if "lags" not in p and "lags_range" in p:
            try:
                p["lags"] = int(p.get("lags_range"))
            except Exception:
                pass

    def _pick_preselected_winner(trials, min_trades_cv: float = 5.0, min_active_cv: float = 0.02) -> int:
        """Choose first Top-N trial whose CV metrics meet basic gates; else 0."""
        import math
        for j, t in enumerate(trials):
            try:
                tr = float(t.user_attrs.get("trades_cv", float("nan")))
                ar = float(t.user_attrs.get("active_rate_cv", float("nan")))
                if (not math.isnan(tr)) and (not math.isnan(ar)) and (tr >= float(min_trades_cv)) and (ar >= float(min_active_cv)):                    
                    return j
            except Exception:
                continue
        return 0

    # Build consensus pool from all completed trials (for Top-N consensus selection)
    # New behaviour:
    #   - Ignore any performance fraction filter.
    #   - Take up to 'consensus_pool_max_trials' best VALID trials
    #     in the DSR-ranked 'completed' list.
    #   - "Valid" = basic trades / active-rate gates.
    #   - Similarity vs. the CV winner (not equal / not too different) is
    #     enforced later in MLBacktester._evaluate_with_topn_consensus via
    #     style + geometry filters.
    consensus_pool: list[dict] = []
    import math as _math

    def _trial_is_valid_for_consensus(_t, min_trades: float = 5.0, min_active: float = 0.02) -> bool:
        """
        Basic validity gate for consensus use:
          - enough trades in CV,
          - non-tiny active rate (avoid degenerate near-always-neutral configs).
        """
        try:
            tr = float(_t.user_attrs.get("trades_cv", float("nan")))
            ar = float(_t.user_attrs.get("active_rate_cv", float("nan")))
        except Exception:
            return False
        if _math.isnan(tr) or _math.isnan(ar):
            return False
        return (tr >= float(min_trades)) and (ar >= float(min_active))

    if completed:
        # 'completed' is already DSR-ranked above (or raw-value sorted as fallback),
        # so we just walk it in that order and collect valid trials.
        valid_trials = [t for t in completed if _trial_is_valid_for_consensus(t)]

        # If nothing passes the validity gate, fall back to all completed trials
        # so that consensus still has something to work with.
        if not valid_trials:
            valid_trials = completed

        for _t in valid_trials[: int(consensus_pool_max_trials)]:
            try:
                v = float(_t.value)
            except Exception:
                continue
            if not _math.isfinite(v):
                continue

            # Build param dict in same style as the Top-N payload
            p = dict(_t.params)
            _normalize_roll_windows_inplace(p)
            _ensure_lags_inplace(p)
            _json_sanitize_inplace(p)

            # Attach CV metrics for diagnostics / runtime selection
            try:
                p["__cv_value"] = float(v)
            except Exception:
                pass
            try:
                p["__cv_psr"] = float(_t.user_attrs.get("psr", float("nan")))
            except Exception:
                pass
            try:
                p["__cv_dsr"] = float(_t.user_attrs.get("dsr", float("nan")))
            except Exception:
                pass
            try:
                p["__trades_cv"] = float(_t.user_attrs.get("trades_cv", float("nan")))
                p["__active_cv"] = float(_t.user_attrs.get("active_rate_cv", float("nan")))
            except Exception:
                pass
            try:
                p["__trial_number"] = int(getattr(_t, "number", -1))
            except Exception:
                pass

            consensus_pool.append(p)
    # Logging-only: summarize the consensus pool built from completed trials
    try:
        _trials = [int(p.get("__trial_number", -1)) for p in (consensus_pool or []) if isinstance(p, dict)]
        _vals   = [float(p.get("__cv_value", float("nan"))) for p in (consensus_pool or []) if isinstance(p, dict)]
        print(f"[TopN][BuiltPool] size={len(consensus_pool)} trials={_trials} cv_values={_vals}")
    except Exception:
        pass

    # 1) Compute Top-N trials if not already available
    try:
        top_params  # noqa: F401
        top_trials  # noqa: F401
    except NameError:
        
        # MAXIMIZE-ONLY safety: never allow silent direction inversions.
        if study.direction != StudyDirection.MAXIMIZE:
            raise RuntimeError(f"Only MAXIMIZE supported, got {study.direction}")
        completed = [
            t for t in study.get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE and t.value is not None
        ]
        if completed:
            # MAXIMIZE: higher objective value = better trial
            completed.sort(key=lambda t: float(t.value), reverse=True)
            top_trials = completed[:max(int(top_n), 0)]
        else:
            top_trials = []
        top_params = [_merged_trial_params(t) for t in top_trials]

    # 2) Build/normalize best params

    # Prefer the *materialized* params recorded during the objective (includes
    # derived keys like roll_windows / indicator_windows). Falling back to
    # trial.params can cause replay drift in real_trading_simulation.
    _bt = study.best_trial
    _bp = None
    try:
        _bp = _bt.user_attrs.get("full_params", None)
    except Exception:
        _bp = None
    best_params = _merged_trial_params(_bt)
    _normalize_roll_windows_inplace(best_params)
    _json_sanitize_inplace(best_params)
    
    # Normalize 'lags' everywhere to remove eval warnings
    _ensure_lags_inplace(best_params)
    for p in top_params:
        _ensure_lags_inplace(p)

    # Pre-commit a CV winner index (0 = top by DSR/value order)
    winner_index = _pick_preselected_winner(top_trials, min_trades_cv=5.0, min_active_cv=0.02)
    best_params["__winner_index"] = int(winner_index)

    # (Optional) annotate CV rank on the Top-N payload (for audit)
    for i, t in enumerate(top_trials):
        try:
            t.set_user_attr("__cv_rank", int(i))
        except Exception:
            pass

    # 3) Normalize Top-N param dicts in-place
    for p in top_params:
        _normalize_roll_windows_inplace(p)
        _json_sanitize_inplace(p)

    # 4) Rebuild the "params-only" view AFTER normalization
    top_params_only = [{k: v for k, v in p.items() if not str(k).startswith("__")} for p in top_params]

    # 5) Prepare/normalize top_payload (create if absent)
    try:
        top_payload  # noqa: F401
    except NameError:
        top_payload = {
            "study_name": study.study_name,
            "direction": study.direction.name.lower(),
            "top_n": int(top_n),
            "trials": [
                {"number": t.number, "value": float(t.value), "params": dict(t.params)}
                for t in top_trials
            ],
        }
    # If it has embedded params, normalize them too
    if isinstance(top_payload, dict):
        if "top_params" in top_payload and isinstance(top_payload["top_params"], list):
            top_payload["top_params"] = top_params_only
        elif "trials" in top_payload and isinstance(top_payload["trials"], list):
            for t in top_payload["trials"]:
                if isinstance(t, dict) and "params" in t and isinstance(t["params"], dict):
                    _normalize_roll_windows_inplace(t["params"])
                    _json_sanitize_inplace(t["params"])

    # 6) Save Top-N JSON (audit)
    try:
        with open(top_path, "w") as f:
            json.dump(top_payload, f, indent=2, sort_keys=True)
    except Exception as _e:
        print(f"[WARN] Failed to write Top-{top_n} JSON: {_e}")

    # 7) Embed Top-N pointers into best_params (for downstream refit)
    best_params["__top5_params"] = top_params_only
    best_params["__top5_path"]   = top_path
    best_params["__top5_info"]   = top_payload
    
    # ------------------------------------------------------------------
    # Optional single-shot consensus finalization (freeze committee ONCE)
    #
    # If enabled via cv_config["use_consensus"], we build a fixed committee
    # *here* (end of global HPO) and store it into best_params so downstream
    # month-by-month simulation can reuse the same committee without
    # re-selecting neighbours each month.
    # ------------------------------------------------------------------
    try:
        _use_consensus = bool(cv_config.get("use_consensus", False))
    except Exception:
        _use_consensus = False

    if _use_consensus:
        try:
            from utilsNoWFO import _infer_family
            _family = _infer_family(str(best_params.get("model_type", "")))
        except Exception:
            _family = "Unknown"

        # committee size mirrors MLBacktester._evaluate_with_topn_consensus
        try:
            if _family == "Classical":
                _N_target = int(cv_config.get("topN_classical", 3))
            elif _family in {"RL"}:
                _N_target = int(cv_config.get("topN_deep", 2))
            elif _family in {"Ensembles"}:
                _N_target = int(cv_config.get("topN_ensemble", 2))
            elif _family in {"DQN"}:
                _N_target = int(cv_config.get("topN_dqn", 2))
            else:
                _N_target = int(cv_config.get("topN_default", 2))
        except Exception:
            _N_target = 2
        _N_target = max(2, int(_N_target))

        # Base params (strip helper keys)
        _base_core = {k: v for k, v in (best_params or {}).items() if not str(k).startswith("__")}

        def _params_key(d: dict) -> str:
            try:
                return json.dumps({k: v for k, v in (d or {}).items() if not str(k).startswith("__")}, sort_keys=True, default=str)
            except Exception:
                return str(d)

        _seen = set()
        _committee = []
        _committee.append(dict(_base_core))
        _seen.add(_params_key(_base_core))

        # Sort pool by stored CV objective value (respect study direction)
        _is_min = False
        try:
            _is_min = str(getattr(study, "direction", "maximize")).lower().startswith("min")
        except Exception:
            _is_min = False

        def _pool_value(p: dict):
            try:
                v = p.get("__cv_value", p.get("cv_value", p.get("value", None)))
                return float(v) if v is not None else float("nan")
            except Exception:
                return float("nan")

        _pool_sorted = list(consensus_pool or [])
        try:
            _pool_sorted.sort(key=_pool_value, reverse=(not _is_min))
        except Exception:
            pass

        _selected_trials = []
        for _p in _pool_sorted:
            if not isinstance(_p, dict):
                continue
            # Keep params, plus lightweight meta keys that Top-N can read.
            _cand = {k: v for k, v in _p.items() if not str(k).startswith("__")}
            try:
                if "trial_number" not in _cand and _p.get("__trial_number", None) is not None:
                    _cand["trial_number"] = int(_p.get("__trial_number"))
            except Exception:
                pass
            try:
                if "value" not in _cand and _p.get("__cv_value", None) is not None:
                    _cand["value"] = float(_p.get("__cv_value"))
            except Exception:
                pass

            _k = _params_key(_cand)
            if _k in _seen:
                continue
            _seen.add(_k)
            _committee.append(_cand)

            try:
                if _cand.get("trial_number", None) is not None:
                    _selected_trials.append(int(_cand.get("trial_number")))
            except Exception:
                pass

            if len(_committee) >= _N_target:
                break

        # Persist on the returned best_params (downstream will reuse)
        best_params["__committee_fixed"] = _committee
        best_params["__committee_fixed_info"] = {
            "enabled": True,
            "N_target": int(_N_target),
            "family": str(_family),
            "selected_trial_numbers": _selected_trials,
            "pool_size": int(len(consensus_pool or [])),
        }

        # Write an audit artifact alongside Top-N JSON
        try:
            _cons_path = os.path.join(out_dir, "consensus_frozen.json")
            with open(_cons_path, "w") as f:
                json.dump(
                    {
                        "info": best_params.get("__committee_fixed_info", {}),
                        "committee": best_params.get("__committee_fixed", []),
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            best_params["__committee_fixed_path"] = _cons_path
        except Exception:
            pass
        
    try:
        _do_robust = bool((cv_config or {}).get("robustness_eval", False))
        _robust_src = "cv_config" if (isinstance(cv_config, dict) and ("robustness_eval" in cv_config)) else "default_off"
    except Exception:
        _do_robust = False
        _robust_src = "error"
        
    if _do_robust:
        # Determinism helper (already exists in your codebase)
        try:
            from utilsNoWFO import set_global_determinism
        except Exception:
            set_global_determinism = None

        class _DummyTrial:
            """Minimal Optuna-trial-like object to capture CV user_attrs without pruning."""
            def __init__(self):
                self.user_attrs = {}
            def set_user_attr(self, k, v):
                self.user_attrs[str(k)] = v
            def report(self, value, step):
                return
            def should_prune(self):
                return False

        def _iqr(x):
            xs = [float(v) for v in x if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
            if len(xs) == 0:
                return float("nan")
            xs.sort()
            def _q(p):
                if len(xs) == 1:
                    return xs[0]
                i = p * (len(xs) - 1)
                lo = int(math.floor(i))
                hi = int(math.ceil(i))
                if lo == hi:
                    return xs[lo]
                w = i - lo
                return xs[lo] * (1 - w) + xs[hi] * w
            return _q(0.75) - _q(0.25)

        def _median(x):
            xs = [float(v) for v in x if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
            if len(xs) == 0:
                return float("nan")
            xs.sort()
            n = len(xs)
            if n % 2 == 1:
                return xs[n // 2]
            return 0.5 * (xs[n // 2 - 1] + xs[n // 2])

        # Candidate set: best + optional top-K
        _topk = int(cv_config.get("robustness_top_k", 1))
        _topk = max(1, min(5, _topk))
        _cands = []
        try:
            _cands.append(dict(best_params))
        except Exception:
            _cands.append(best_params)
        try:
            for p in (top_params_only or []):
                if len(_cands) >= _topk:
                    break
                _cands.append(dict(p))
        except Exception:
            pass

        # Seeds: default 3, configurable to 5
        _seeds = cv_config.get("robustness_seeds", None)
        if not isinstance(_seeds, (list, tuple)) or len(_seeds) == 0:
            _seeds = [101, 202, 303]
        _seeds = list(_seeds)[: int(cv_config.get("robustness_max_seeds", 5))]

        # Conservative rejection rule (configurable)
        # - PASS if: median_SR >= min_median_sr
        #           and median_trades >= min_median_trades
        #           and IQR_SR <= max_iqr_sr
        #           and worst_SR >= min_worst_sr
        _min_med_sr = float(cv_config.get("robust_min_median_sr", 0.10))
        _min_med_tr = int(cv_config.get("robust_min_median_trades", 8))
        _max_iqr_sr = float(cv_config.get("robust_max_iqr_sr", 0.60))
        _min_worst_sr = float(cv_config.get("robust_min_worst_sr", -0.50))

        def _rule_text():
            return (f"median_SR>={_min_med_sr:.2f} "
                    f"median_trades>={_min_med_tr} "
                    f"IQR_SR<={_max_iqr_sr:.2f} "
                    f"worst_SR>={_min_worst_sr:.2f}")

        # Evaluate each candidate across seeds
        for i, cand in enumerate(_cands):
            sr_list, tr_list, dd_list, eq_list = [], [], [], []
            worst_sr = float("inf")
            for sd in _seeds:
                try:
                    if set_global_determinism is not None:
                        set_global_determinism(int(sd))
                except Exception:
                    pass

                _trial = _DummyTrial()
                try:
                    min_tw = int(cv_config.get("min_train_window", 28032))
                    val_w  = int(cv_config.get("val_window", 1475))
                    
                    # Evaluate via the existing CV function.
                    # Score is treated as Sharpe proxy (your objective is Sharpe-like)
                    _score = evaluate_cv_func(
                        train_data,
                        cand,
                        min_train_window=min_tw,
                        val_window=val_w,
                        trial=_trial,
                        cv_config_override=cv_config,
                    )
                    _sr = float(_score)
                except Exception as _e:
                    _sr = float("nan")

                sr_list.append(_sr)
                if math.isfinite(_sr):
                    worst_sr = min(worst_sr, _sr)

                # Trades from existing CV attrs (already set in MLBacktesterNoWFO)
                try:
                    tr_list.append(float(_trial.user_attrs.get("trades_cv", float("nan"))))
                except Exception:
                    tr_list.append(float("nan"))

                # Optional (will become real once Patch 6 adds these attrs)
                try:
                    dd_list.append(float(_trial.user_attrs.get("dd_cv", float("nan"))))
                except Exception:
                    dd_list.append(float("nan"))
                try:
                    eq_list.append(float(_trial.user_attrs.get("eq_end_cv", float("nan"))))
                except Exception:
                    eq_list.append(float("nan"))

            med_sr = _median(sr_list)
            iqr_sr = _iqr(sr_list)
            med_tr = _median(tr_list)
            med_dd = _median(dd_list)
            med_eq = _median(eq_list)

            passed = True
            if not (math.isfinite(med_sr) and med_sr >= _min_med_sr):
                passed = False
            if not (math.isfinite(med_tr) and med_tr >= float(_min_med_tr)):
                passed = False
            if math.isfinite(iqr_sr) and iqr_sr > _max_iqr_sr:
                passed = False
            if worst_sr is float("inf") or (math.isfinite(worst_sr) and worst_sr < _min_worst_sr):
                passed = False

            dd_txt = "NA" if (not math.isfinite(med_dd)) else f"{med_dd:.4f}"
            eq_txt = "NA" if (not math.isfinite(med_eq)) else f"{med_eq:.4f}"

            log_print(
                f"[ROBUST] cand={i} seeds={_seeds} median_SR={med_sr:.4f} IQR_SR={iqr_sr:.4f} "
                f"median_trades={med_tr:.1f} median_DD={dd_txt} median_eq={eq_txt} "
                f"rule='{_rule_text()}' {'PASS' if passed else 'FAIL'}",
                level="COMPACT",
            )

            # Store for downstream reporting
            try:
                if i == 0:
                    best_params["__robust_seeds"] = list(_seeds)
                    best_params["__robust_median_sr"] = float(med_sr) if math.isfinite(med_sr) else None
                    best_params["__robust_iqr_sr"] = float(iqr_sr) if math.isfinite(iqr_sr) else None
                    best_params["__robust_median_trades"] = float(med_tr) if math.isfinite(med_tr) else None
                    best_params["__robust_pass"] = bool(passed)
                    best_params["__robust_rule"] = _rule_text()
            except Exception:
                pass

            # Optional hard reject: if best cand fails, downgrade to next passing cand
            if i == 0 and (not passed) and bool(cv_config.get("robust_fail_downgrade_to_next", True)):
                # try to find a passing candidate among the remaining ones
                continue
            if i == 0:
                # keep going but do not auto-swap unless explicitly requested later
                pass

        # If requested, enforce that the top config must pass robustness
        if bool(cv_config.get("robust_require_pass", False)):
            if not bool(best_params.get("__robust_pass", False)):
                raise RuntimeError("[ROBUST] Best candidate FAILED robustness and robust_require_pass=True")
            
    # ------------------------------------------------------------------
    # Purpose:
    #   - Use fast mini-block CV for screening during HPO
    #   - Then confirm ONLY Top-N with a more realistic monthly-roll CV
    #
    # Controls:
    #   cv_config["verify_topn_monthly_roll"] = True/False (explicit)
    #   cv_config["verify_topn_count"]        = N (default 5)
    #   cv_config["verify_cv_blocks"]         = folds for monthly-roll (default 5)
    #   cv_config["verify_cv_val_months"]     = val months (default 1.0)
    # ------------------------------------------------------------------
    best_score_override = None
    try:
        _do_verify = bool((cv_config or {}).get("verify_topn_monthly_roll", False))
        _verify_src = "cv_config" if (isinstance(cv_config, dict) and ("verify_topn_monthly_roll" in cv_config)) else "default_off"
    except Exception:
        _do_verify = False
        _verify_src = "error"

    if _do_verify and train_data is not None and callable(evaluate_cv_func):
        try:
            _verify_n = int(cv_config.get("verify_topn_count", 5))
        except Exception:
            _verify_n = 5
        _verify_n = max(1, _verify_n)

        # Build a monthly-roll CV override for verification (final exam)
        verify_cv_config = dict(cv_config) if isinstance(cv_config, dict) else {}
        verify_cv_config["cv_mode"] = "monthly_roll"
        try:
            verify_blocks = int(verify_cv_config.get("verify_cv_blocks", cv_config.get("verify_cv_blocks", 5)))
        except Exception:
            verify_blocks = 5
        verify_blocks = max(2, verify_blocks)
        verify_cv_config["cv_blocks"] = verify_blocks
        verify_cv_config["cv_target_folds"] = int(verify_cv_config.get("cv_target_folds", verify_blocks))
        verify_cv_config["cv_tail_anchor"] = True
        try:
            verify_cv_config["cv_val_months"] = float(verify_cv_config.get("verify_cv_val_months", cv_config.get("verify_cv_val_months", 1.0)))
        except Exception:
            verify_cv_config["cv_val_months"] = 1.0

        # Evaluate only Top-N candidates
        candidates = []
        try:
            # top_params_only is the normalized params-only list
            candidates = list(top_params_only[:_verify_n]) if top_params_only else []
        except Exception:
            candidates = []

        # Ensure current best is included (front-load it for logging clarity)
        try:
            if isinstance(best_params, dict):
                _bp_clean = {k: v for k, v in best_params.items() if not str(k).startswith("__")}
                if _bp_clean and all((_bp_clean != c) for c in candidates if isinstance(c, dict)):
                    candidates = [_bp_clean] + candidates
        except Exception:
            pass

        # Minimal trial-like shim so evaluate_cv_func can write attrs safely
        class _DummyTrialVerify:
            def __init__(self):
                self.user_attrs = {}
            def set_user_attr(self, k, v):
                self.user_attrs[str(k)] = v
            def report(self, value, step):
                return
            def should_prune(self):
                return False

        try:
            min_tw = int(cv_config.get("min_train_window", 28032))
        except Exception:
            min_tw = 28032
        try:
            val_w = int(cv_config.get("val_window", 1475))
        except Exception:
            val_w = 1475

        verify_scores = []
        best_v_score = float("-inf")
        best_v_params = None
        best_v_idx = -1

        log_print(f"[VERIFY] Top-N monthly-roll verification enabled ({_verify_src}) "
                  f"model={_model_name} N={len(candidates)} folds={verify_blocks} val_months={verify_cv_config.get('cv_val_months')}",
                  level="COMPACT")

        for i, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                continue
            _trial = _DummyTrialVerify()
            try:
                _score = evaluate_cv_func(
                    train_data,
                    cand,
                    min_train_window=min_tw,
                    val_window=val_w,
                    trial=_trial,
                    cv_config_override=verify_cv_config,
                )
                _sr = float(_score)
            except Exception as e:
                _sr = float("-inf")
                try:
                    log_print(f"[VERIFY] Candidate {i} failed: {type(e).__name__}: {e}", level="DEBUG")
                except Exception:
                    pass

            verify_scores.append(_sr)
            if math.isfinite(_sr) and _sr > best_v_score:
                best_v_score = _sr
                best_v_params = dict(cand)
                best_v_idx = i

        # If verification produced a winner, promote it to final best_params
        if best_v_params is not None and math.isfinite(best_v_score):
            try:
                _hpo_best = float(getattr(study, "best_value", float("nan")))
            except Exception:
                _hpo_best = float("nan")

            # Preserve meta keys from current best_params (e.g., __top5_params, __winner_index)
            _meta = {}
            try:
                _meta = {k: v for k, v in best_params.items() if str(k).startswith("__")}
            except Exception:
                _meta = {}

            best_params = dict(best_v_params)
            best_params.update(_meta)
            best_params["__winner_index_preverify"] = int(best_params.get("__winner_index", -1))
            best_params["__verified_winner_index"] = int(best_v_idx)
            best_params["__verify_mode"] = "monthly_roll"
            best_params["__verify_topn_count"] = int(len(candidates))
            best_params["__hpo_best_score"] = _hpo_best
            best_params["__verify_best_score"] = float(best_v_score)

            # Override the returned best_score to match the verified winner
            best_score_override = float(best_v_score)

            log_print(f"[VERIFY] Promoted verified winner idx={best_v_idx} "
                      f"verify_score={best_v_score:.4f} (hpo_best={_hpo_best:.4f})",
                      level="COMPACT")


    
    

    # --- Patch B: persist tuned execution defaults (subset) ---
    TUNED_KEYS = [
        # labeling
        "use_triple_barrier","tb_pt_mult","tb_sl_mult","tb_max_holding","tb_neutral_zone",
        # calibration & features
        "calibrate_method","use_fracdiff","fracdiff_d","use_rv_features","rv_window_short","rv_window_long"
    ]

    try:
        import json, os
        tuned_subset = {k: best_params[k] for k in TUNED_KEYS if k in best_params}
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/best_exec_defaults.json", "w") as f:
            json.dump(tuned_subset, f, indent=2, sort_keys=True)
        print("[Tuning] Wrote tuned execution defaults -> artifacts/best_exec_defaults.json")
    except Exception as e:
        print(f"[Tuning] Could not write tuned defaults: {e}")

    # If monthly-roll verification ran, return the verified score; otherwise keep study.best_value
    try:
        if "best_score_override" in locals() and (best_score_override is not None):
            best_score = float(best_score_override)
        else:
            best_score = float(study.best_value)
    except Exception:
        best_score = study.best_value

    # ---- Plots (only what's needed) ----
    # Only save Optuna progress for the *first* trading month.
    # Later months reuse tuned params, so we don't spam extra plots.
    if (save_optuna_progress_from_study is not None) and not SKIP_PLOTS:
        try:
            try:
                month_idx_int = int(month_ix) if month_ix is not None else 1
            except Exception:
                month_idx_int = 1

            if month_idx_int == 1:
                if month_out_dir:
                    # month_out_dir should now map to this model's graphs directory
                    base = os.path.join(month_out_dir, "optuna_scores_1")
                    save_optuna_progress_from_study(
                        study,
                        out_prefix=base,
                        metric_name="Sharpe",
                        style="nature",
                        palette="okabe_ito_no_black",
                    )
                else:
                    # Fallback: generic progress file in out_dir
                    save_optuna_progress_from_study(
                        study,
                        out_prefix=os.path.join(out_dir, "optuna_progress"),
                        metric_name="Sharpe",
                        style="nature",
                        palette="okabe_ito_no_black",
                    )
            else:
                print(
                    f"[INFO] Skipping Optuna progress plot for month {month_idx_int} "
                    f"(only month 1 is plotted)."
                )
        except Exception as _e:
            print(f"[WARN] Failed to save Optuna progress: {_e}")

    # 2) Trial-level feature-frequency heatmap (top 20% trials, Sharpe-weighted)
    if legacy_optuna_dir and SAVE_TRIAL_FEATURE_FREQ and (save_feature_frequency_from_trials is not None) and not SKIP_PLOTS:

        try:
            save_feature_frequency_from_trials(
                study_or_trials=study,
                base_features=[],  # only engineered features
                out_png=os.path.join(out_dir, "feature_frequency_trials.png"),
                top_k=30,
                top_percent=0.20,
                weight_by_score=True,
                minimize_objective=False,              # TRUE Sharpe (maximize)
                style="nature",
                palette="okabe_ito_no_black",
                exclude_prefixes=("returns_lag","hour"),
                collapse_raw_lags=True,
            )
        except Exception as _e:
            print(f"[WARN] Failed to save trial-level feature frequency: {_e}")

    # 3) Trial-duration stats (per study/run)
    try:
        import csv
        import numpy as _np

        durations = []
        for t in study.trials:
            dur = getattr(t, "duration", None)
            if dur is not None:
                try:
                    durations.append(float(dur.total_seconds()))
                except Exception:
                    continue

        if durations:
            avg_sec = float(_np.mean(durations))
            med_sec = float(_np.median(durations))
            min_sec = float(_np.min(durations))
            max_sec = float(_np.max(durations))

            # month_ix is optional - best-effort cast
            try:
                m_ix = int(month_ix) if month_ix is not None else ""
            except Exception:
                m_ix = ""

            row = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "month_ix": m_ix,
                "n_trials": int(len(durations)),
                "avg_sec": avg_sec,
                "median_sec": med_sec,
                "min_sec": min_sec,
                "max_sec": max_sec,
                "models": ",".join(sorted(set(models_to_test))) if models_to_test else "",
            }

            stats_csv = os.path.join(out_dir, "optuna_trial_time_stats.csv")
            os.makedirs(out_dir, exist_ok=True)
            file_exists = os.path.exists(stats_csv)

            with open(stats_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            print(
                f"[TIME] Trial-time stats: avg={avg_sec:.2f}s "
                f"(n={len(durations)}) -> {stats_csv}"
            )
        else:
            print("[TIME] No trial durations available to log.")
    except Exception as _e:
        print(f"[WARN] Could not save trial-time stats: {_e}")

    import gc as _gc
    _gc.collect()

    # ------------------------------------------------------------------
    # Persist best config / Top-N (or consensus pool) for later reuse
    # ------------------------------------------------------------------
    if save_hpo_config_to_disk is not None:
        try:
            # Best model_type should always be present in best_params
            model_type = str(best_params.get("model_type", "unknown"))

            # Minimal metadata about the study
            try:
                direction = study.direction.name if study is not None else None
                n_trials = len(study.trials) if study is not None else None
            except Exception:
                direction = None
                n_trials = None

            study_meta = {
                "best_score": float(best_score),
                "direction": direction,
                "n_trials": n_trials,
                "saved_at_utc": datetime.datetime.utcnow().isoformat(),
            }

            # Decide which configs to actually persist:
            # - If we built a consensus_pool, that is our "used configs" set.
            # - Else fall back to the Top-N list.
            # - As a last resort, persist just the single best config.
            try:
                if consensus_pool:
                    configs_to_persist = list(consensus_pool)
                elif top_params_only:
                    configs_to_persist = list(top_params_only)
                else:
                    configs_to_persist = [dict(best_params)]
            except Exception:
                configs_to_persist = [dict(best_params)]

            save_hpo_config_to_disk(
                model_type=model_type,
                best_params=best_params,
                topN_params=configs_to_persist,
                study_meta=study_meta,
            )
        except Exception as e:
            # Do not crash the tuning run just because persistence failed
            print(f"[HPO] Warning: failed to persist best config: {e}")

    return best_params, best_score, top_params_only, study, consensus_pool

