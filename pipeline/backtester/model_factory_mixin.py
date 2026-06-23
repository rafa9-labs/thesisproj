"""Auto-extracted mixin -- see composed.py for the full MLBacktester."""
from pipeline._imports import *  # noqa: F401,F403


def filter_params(d: dict, prefix: str) -> dict:
    """Filter dict keys by prefix, stripping the prefix from returned keys."""
    if not isinstance(d, dict):
        return {}
    L = len(prefix)
    return {k[L:]: v for k, v in d.items() if isinstance(k, str) and k.startswith(prefix)}


def ensure_dict(obj):
    """Coerce obj to dict; return empty dict for None/non-dict input."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception:
        return {}


class ModelFactoryMixin:
    """
    get_model, windows, predict chunked

    Auto-extracted from MLBacktesterNoWFO.py lines 10206-10707.
    """
    def _print_thread_budget(self, tag: str = ""):
        if self._is_debug():
            try:
                from threadpoolctl import threadpool_info
                pools = threadpool_info()
            except Exception:
                pools = []
            envs = {k: os.getenv(k) for k in [
                "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS","SKLEARN_JOBS","RF_JOBS","XGB_JOBS",
                "TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS"
            ]}
            print(f"[THREAD] Threads[{tag}] env={envs} pools={pools}")


    def get_model(self, model_type, use_proba: bool = True, warm_start_from=None, **params):
        """
        Construct and return an instance of the specified model.
        Trials stay serialized; each fit uses many threads/GPU.

        Tries the model registry first; falls back to the legacy
        inline if/elif chain for backwards compatibility.

        warm_start_from: for deep models, a path to a .weights.h5 file to
        load_weights() from after construction. For tree models, the previous
        model object is passed directly to .fit() -- no change here.
        """
        self._print_thread_budget(tag=model_type)  # show effective threads for this build

        # Shared per-repeat seed
        seed = None
        try:
            seed = int((getattr(self, "features_config", {}) or {}).get("run_seed", 0)) or None
        except Exception:
            seed = None

        # ---------- Registry path (preferred) ----------
        try:
            from models.registry import build_model as _build
            params.setdefault("seed", seed)
            model = _build(model_type, use_proba=use_proba, **params)
            if model is not None:
                if warm_start_from and hasattr(model, 'load_weights'):
                    import os as _os
                    if _os.path.exists(str(warm_start_from)):
                        try:
                            model.load_weights(str(warm_start_from))
                        except Exception:
                            pass
                return model
        except Exception:
            pass  # fall through to legacy path

        # ---------- Legacy inline path (fallback) ----------
        # --- sklearn version detection for deprecation-safe param handling ---
        try:
            import sklearn as _sk
            _SK_GE_18 = tuple(int(x) for x in _sk.__version__.split('.')[:2]) >= (1, 8)
        except Exception:
            _SK_GE_18 = False

        # --- local helper: guard invalid sklearn solver/penalty combos ---
        def _sanitize_logit_params(d: dict, *, ovr: bool = False) -> dict:
            p = dict(d or {})
            p.pop("multi_class", None)
            solver  = str(p.get("solver", "saga")).strip().lower()
            penalty = str(p.get("penalty", "l2")).strip().lower()
            allowed_solvers = {"lbfgs", "newton-cg", "liblinear", "sag", "saga"}
            allowed_penalty = {"l2", "l1", "elasticnet", "none"}
            if solver not in allowed_solvers: solver = "saga"
            if penalty not in allowed_penalty: penalty = "l2"

            if penalty == "l1":
                if solver in {"lbfgs","newton-cg","sag"}: solver = "saga"
                if not ovr and solver == "liblinear":    solver = "saga"
            elif penalty == "elasticnet":
                solver = "saga"
                try:
                    p["l1_ratio"] = min(1.0, max(0.0, float(p.get("l1_ratio", 0.5))))
                except Exception:
                    p["l1_ratio"] = 0.5
            elif penalty in {"none", "", None}:
                penalty = "none"
                if solver == "liblinear": solver = "lbfgs"

            if bool(p.get("dual", False)) and not (solver == "liblinear" and penalty == "l2"):
                p["dual"] = False
            try:
                C = float(p.get("C", 1.0));    p["C"]   = C if C > 0 else 1.0
            except Exception:
                p["C"] = 1.0
            try:
                tol = float(p.get("tol", 1e-3)); p["tol"] = tol if tol > 0 else 1e-3
            except Exception:
                p["tol"] = 1e-3
            try:
                mi = int(p.get("max_iter", 2000)); p["max_iter"] = mi if mi > 0 else 2000
            except Exception:
                p["max_iter"] = 2000
            p["solver"]  = solver

            # sklearn >=1.8 deprecated `penalty` param in favour of `l1_ratio` / C
            if _SK_GE_18:
                p.pop("penalty", None)          # remove to avoid FutureWarning
                if penalty == "l2":
                    p.setdefault("l1_ratio", 0)   # equivalent to penalty='l2'
                elif penalty == "l1":
                    p["l1_ratio"] = 1.0
                # elasticnet keeps its own l1_ratio from above
                # none -> set C=inf (sklearn recommendation)
                elif penalty == "none":
                    p["C"] = float("inf")
                # n_jobs has no effect since 1.8 -- remove to avoid warning
                p.pop("n_jobs", None)
            else:
                p["penalty"] = penalty

            return p

        # ========== DEEP MODELS (GPU; light CPU threads for input pipeline) ==========
        if model_type == "cnn":
            cfg = filter_params(params, "cnn_")
            if seed is not None: cfg.setdefault("seed", seed)
            input_shape = params["input_shape"]
            # Match TF thread knobs to env (intra/inter)
            try:
                import tensorflow as _tf
                intra = int(os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (os.cpu_count() or 8) - 2)
                inter = int(os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
                _tf.config.threading.set_intra_op_parallelism_threads(intra)
                _tf.config.threading.set_inter_op_parallelism_threads(inter)
            except Exception:
                pass
            model = build_cnn(input_shape=input_shape, config=cfg)

        elif model_type == "lstm":
            cfg = filter_params(params, "lstm_")
            if seed is not None: cfg.setdefault("seed", seed)
            input_shape = params.get("input_shape")
            if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
                raise ValueError(f"Invalid input_shape for LSTM: {input_shape}")
            try:
                import tensorflow as _tf
                intra = int(os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (os.cpu_count() or 8) - 2)
                inter = int(os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
                _tf.config.threading.set_intra_op_parallelism_threads(intra)
                _tf.config.threading.set_inter_op_parallelism_threads(inter)
            except Exception:
                pass
            model = build_lstm(input_shape=input_shape, config=cfg)
            if model is None:
                raise RuntimeError("build_lstm returned None. Check model config or input shape.")

        elif model_type == "transformer":
            cfg = filter_params(params, "transformer_")
            if seed is not None: cfg.setdefault("seed", seed)
            input_shape = params["input_shape"]
            try:
                import tensorflow as _tf
                intra = int(os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (os.cpu_count() or 8) - 2)
                inter = int(os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
                _tf.config.threading.set_intra_op_parallelism_threads(intra)
                _tf.config.threading.set_inter_op_parallelism_threads(inter)
            except Exception:
                pass
            model = build_transformer(input_shape=input_shape, config=cfg)
        elif model_type == "dqn":
            input_shape = params["input_shape"]
            dqn_cfg = params.get("dqn_config", {}) or filter_params(params, "dqn_")
            dqn_cfg = filter_dqn_config(dqn_cfg or {})
            dqn_cfg["state_size"] = int(input_shape[0])
            if "window" not in dqn_cfg and "lags" in params:
                dqn_cfg["window"] = int(params["lags"])
            if seed is not None:
                dqn_cfg.setdefault("seed", seed)

            # NEW: thread tuning, like other TF models
            try:
                import tensorflow as _tf, os as _os
                intra = int(_os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (_os.cpu_count() or 8) - 2)
                inter = int(_os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
                _tf.config.threading.set_intra_op_parallelism_threads(intra)
                _tf.config.threading.set_inter_op_parallelism_threads(inter)
            except Exception:
                pass

            dqn_cfg = _coerce_dqn_cfg(dqn_cfg)
            model = DQNAgent(**dqn_cfg)

        # ========================== CLASSICAL (CPU) ==========================
        elif model_type == "svm":
            import numpy as _np
            svm_params = filter_params(params, "svm_")
            kernel = str(svm_params.get("kernel", "rbf")).lower()

            cw = svm_params.get("class_weight", None)
            if isinstance(cw, float) and (_np.isnan(cw) or _np.isinf(cw)):
                cw = None
            elif isinstance(cw, str):
                _cws = cw.strip().lower()
                cw = "balanced" if _cws == "balanced" else (None if _cws in ("", "nan", "none", "null") else None)

            gamma = svm_params.get("gamma", "scale")
            if isinstance(gamma, float):
                gamma = "scale" if (_np.isnan(gamma) or _np.isinf(gamma)) else gamma
            elif isinstance(gamma, str):
                g = gamma.strip().lower()
                if g not in ("scale", "auto"):
                    try:
                        gv = float(gamma); gamma = "scale" if (_np.isnan(gv) or _np.isinf(gv)) else gv
                    except Exception:
                        gamma = "scale"
                else:
                    gamma = g

            def _to_float(x, default):
                try: 
                    v = float(x); 
                    return float(default) if (_np.isnan(v) or _np.isinf(v)) else v
                except Exception:
                    return float(default)
            def _to_int(x, default):
                try: return int(x)
                except Exception: return int(default)

            C         = _to_float(svm_params.get("C", 1.0), 1.0)
            degree    = _to_int(svm_params.get("degree", 3), 3) if kernel == "poly" else 3
            max_iter  = _to_int(svm_params.get("max_iter", 200_000), 200_000)
            tol       = _to_float(svm_params.get("tol", 1e-2), 1e-2)
            shrinking = bool(svm_params.get("shrinking", True))
            cache_sz  = _to_float(svm_params.get("cache_size", 2048.0), 2048.0)

            svc = SVC(
                C=C, gamma=gamma, kernel=kernel, degree=degree,
                class_weight=cw,
                probability=False,  # calibrate below
                max_iter=max_iter, tol=tol, shrinking=shrinking, cache_size=cache_sz,
                decision_function_shape="ovr",
                random_state=seed,
            )

            # Features are already standardized by the global scale_features() path.
            # We therefore pass the bare SVC into CalibratedClassifierCV instead of
            # adding another StandardScaler inside a Pipeline.
            calibrate_method = params.get("calibrate_method", None)
            if calibrate_method not in ("sigmoid", "isotonic"):
                calibrate_method = "isotonic"
            cal_jobs = int(params.get("svm_calib_n_jobs", 0)) or int(os.environ.get("SKLEARN_JOBS", -1))
            cal_jobs = max(1, min(cal_jobs, 2))
            model = CalibratedClassifierCV(estimator=svc, cv=3, method=calibrate_method, n_jobs=cal_jobs)

        elif model_type == "random_forest":
            rf_params = ensure_dict(filter_params(params, "rf_"))
            rf_params.setdefault("n_estimators", 300)
            
            # Safety clamp: protects against accidentally loaded giant configs.
            # No effect for normal Optuna ranges; does not increase capacity.
            try:
                rf_params["n_estimators"] = int(rf_params.get("n_estimators", 300))
            except Exception:
                rf_params["n_estimators"] = 300
            rf_params["n_estimators"] = max(1, min(rf_params["n_estimators"], 1200))
                
            rf_params.setdefault("max_depth", 18)
            rf_params.setdefault("min_samples_leaf", 10)
            rf_params.setdefault("max_features", "sqrt")
            rf_params.setdefault("class_weight", "balanced_subsample")

            # --- OOB vs bootstrap guard ---
            # In sklearn, oob_score is only valid if bootstrap=True. If bootstrap is
            # tuned to False, we must disable oob_score to avoid errors.
            bootstrap_flag = rf_params.get("bootstrap", True)
            if not bootstrap_flag:
                # force OOB off when no bootstrap sampling is used
                rf_params["oob_score"] = False
            else:
                # leave OOB on by default when using bootstrap
                rf_params.setdefault("oob_score", True)

            # ------------------------------------------------------------------
            # Threading safety: avoid rf n_jobs=-1 by default (can hard-crash some
            # native stacks under repeated CV/Optuna evaluation).
            # - Default RF_JOBS=1
            # - Treat -1/0 as "use safe default"
            # - Clamp to [1, cpu_count]
            # ------------------------------------------------------------------
            try:
                _safe_rf_jobs = int(os.environ.get("RF_JOBS", "1") or 1)
            except Exception:
                _safe_rf_jobs = 1
            if _safe_rf_jobs in (-1, 0):
                _safe_rf_jobs = max(1, (os.cpu_count() or 1) - 1)

            _rf_n_jobs = rf_params.get("n_jobs", _safe_rf_jobs)
            try:
                _rf_n_jobs = int(_rf_n_jobs)
            except Exception:
                _rf_n_jobs = _safe_rf_jobs
            if _rf_n_jobs in (-1, 0):
                _rf_n_jobs = _safe_rf_jobs
            _rf_n_jobs = max(1, min(_rf_n_jobs, (os.cpu_count() or 1)))
            rf_params["n_jobs"] = _rf_n_jobs
            
            if seed is not None:
                rf_params.setdefault("random_state", seed)
            model = RandomForestClassifier(**rf_params)


        elif model_type == "logistic":
            # single-estimator multinomial; rely on OpenMP inside solver; no joblib nesting
            _raw_logit = filter_params(params, "logit_") or {}
            logit_params = ensure_dict(_raw_logit)
            logit_params = _sanitize_logit_params(logit_params, ovr=False)
            logit_params.setdefault("solver", "saga")
            logit_params.setdefault("max_iter", 2000)
            logit_params.setdefault("tol", 1e-3)
            logit_params.setdefault("class_weight", "balanced")
            # Only set penalty/n_jobs for sklearn <1.8 (sanitizer already handled >=1.8)
            if not _SK_GE_18:
                logit_params.setdefault("penalty", "l2")
                logit_params.setdefault("n_jobs", int(os.environ.get("SKLEARN_JOBS", max(1, (os.cpu_count() or 2) - 1))))
            if seed is not None: logit_params.setdefault("random_state", seed)
            model = Pipeline([("std", StandardScaler()), ("logit", LogisticRegression(**logit_params))])
            
        elif model_type == "decision_tree":
            dt_params = ensure_dict(filter_params(params, "dt_"))

            # Moderately regularised defaults; Optuna-tuned runs override via dt_* keys.
            # - max_depth: shallow-to-medium tree to avoid extreme overfitting.
            # - min_samples_split/leaf: ensure each leaf has enough samples to be stable.
            dt_params.setdefault("max_depth", 12)
            dt_params.setdefault("min_samples_split", 2)
            dt_params.setdefault("min_samples_leaf", 10)

            # FX labels are often imbalanced -> use balanced class weights by default.
            dt_params.setdefault("class_weight", "balanced")

            # Mild cost-complexity pruning; Optuna can override via dt_ccp_alpha.
            dt_params.setdefault("ccp_alpha", 1e-4)

            if seed is not None:
                dt_params.setdefault("random_state", seed)

            model = DecisionTreeClassifier(**dt_params)

        elif model_type == "xgboost":
            xgb_params = ensure_dict(filter_params(params, "xgb_"))
            # multiclass objective + good defaults
            xgb_params.setdefault("objective", "multi:softprob")
            xgb_params.setdefault("num_class", 3)
            xgb_params.setdefault("eval_metric", "mlogloss")
            xgb_params.setdefault("importance_type", "gain")
            xgb_params.setdefault("subsample", 0.8)
            xgb_params.setdefault("colsample_bytree", 0.8)
            xgb_params.setdefault("max_depth", 6)
            xgb_params.setdefault("n_estimators", 400)
            
            # Safety clamp: protects against accidentally loaded giant configs.
            # No effect for normal Optuna ranges; does not increase capacity.
            try:
                xgb_params["n_estimators"] = int(xgb_params.get("n_estimators", 400))
            except Exception:
                xgb_params["n_estimators"] = 400
            xgb_params["n_estimators"] = max(1, min(xgb_params["n_estimators"], 1500))
            xgb_params.setdefault("min_child_weight", 1.0)

            # ---- REGULARIZATION KEYS (L2) ----
            # Optuna gives us xgb_lambda -> "lambda" after filter_params.
            # XGBClassifier expects "reg_lambda" (sklearn-style name).
            if "lambda" in xgb_params and "reg_lambda" not in xgb_params:
                xgb_params["reg_lambda"] = xgb_params.pop("lambda")
            # Safe default if nothing was supplied
            xgb_params.setdefault("reg_lambda", 1.0)
            # Just in case, drop any stray "lambda"
            xgb_params.pop("lambda", None)

            # ---- THREADS ----
            xgb_params.setdefault(
                "n_jobs",
                int(os.environ.get("XGB_JOBS", max(1, (os.cpu_count() or 2) - 1)))
            )

            # ---- GPU CONTROL (XGBoost >= 2.0 style) ----
            # Auto-detect GPU; env var XGB_USE_GPU can override (0=force CPU, 1=force GPU)
            _xgb_env_gpu = os.environ.get("XGB_USE_GPU")
            if _xgb_env_gpu is not None:
                use_gpu = _xgb_env_gpu == "1"
            else:
                try:
                    from pipeline.runtime import gpu_status
                    use_gpu = gpu_status().get("available", False)
                except Exception:
                    use_gpu = False

            if use_gpu:
                # New-style: tree_method + device
                # https://xgboost.readthedocs.io/en/stable/gpu/
                xgb_params.setdefault("tree_method", "hist")
                xgb_params["device"] = os.environ.get("XGB_DEVICE", "cuda")
                # Let XGBoost pick the right GPU predictor internally
                xgb_params.pop("predictor", None)
            else:
                # Pure CPU
                xgb_params.setdefault("tree_method", "hist")
                xgb_params.pop("device", None)
                xgb_params.pop("predictor", None)

            if seed is not None:
                xgb_params.setdefault("random_state", seed)

            # Try GPU, fall back to CPU if GPU config explodes
            try:
                model = XGBClassifier(**xgb_params)
            except Exception as e:
                if use_gpu:
                    print(f"[XGBoost] GPU init failed ({e}); falling back to CPU.")
                    # Strip GPU-specific keys and retry on CPU
                    xgb_params.pop("device", None)
                    xgb_params["tree_method"] = "hist"
                    model = XGBClassifier(**xgb_params)
                else:
                    raise



        elif model_type == "ensemble_adaptive_regime":
            lstm_config = filter_params(params, "lstm_")
            rf_config   = filter_params(params, "rf_")
            logit_config= filter_params(params, "logit_")
            if seed is not None:
                lstm_config.setdefault("seed", seed)
                rf_config.setdefault("random_state", seed)
                logit_config.setdefault("random_state", seed)
            input_shape = params["input_shape"]
            model = AdaptiveRegimeStrategy(
                lstm_config=lstm_config,
                rf_config=rf_config,
                logit_config=logit_config,
                input_shape=input_shape,
                adx_col=params.get("adx_col", "adx_14"),
                vol_col=params.get("vol_col", "rolling_std_20"),
                adx_thresh=params.get("adx_thresh", 25),
                vol_thresh=params.get("vol_thresh", 0.002),
                adx_thresh_q=(
                    float(params.get("adx_thresh_q", 0.70))
                    if bool(params.get("train_lstm_on_trend_only", True))
                    else None
                ),
            )

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if model is None:
            raise ValueError(f"Model creation failed for type {model_type}")
        if warm_start_from and hasattr(model, 'load_weights'):
            import os as _os
            if _os.path.exists(str(warm_start_from)):
                try:
                    model.load_weights(str(warm_start_from))
                except Exception:
                    pass
        return model



    def _create_sliding_windows(self, df, features, window_size):
        """
        Vectorized fixed-length sliding windows over time.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain `features` columns and optionally a 'label' column.
        features : list[str]
            Feature columns to window.
        window_size : int
            Length of each temporal window.

        Returns
        -------
        Xv : np.ndarray       shape (n_windows, window_size, n_features)   dtype float32
        yv : np.ndarray       shape (n_windows,)                           dtype int32 (zeros if no 'label' col)
        idx : list[int]       end indices (in `df`) corresponding to each window
        """
        from numpy.lib.stride_tricks import sliding_window_view

        X2d = df[features].to_numpy(dtype=np.float32, copy=False)
        n   = X2d.shape[0]
        w   = int(window_size)
        if n < w:
            return np.empty((0, w, X2d.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32), []

        Xv = sliding_window_view(X2d, window_shape=w, axis=0)  # (n-w+1, w, f)
        Xv = Xv.reshape(-1, w, X2d.shape[1])

        idx = list(range(w - 1, n))
        if "label" in df.columns:
            yv = df["label"].to_numpy(dtype=np.int32, copy=False)[idx]
        else:
            yv = np.zeros((len(idx),), dtype=np.int32)

        return Xv, yv, idx
    
    def _predict_seq_windows_chunked(self, model, X2d: np.ndarray, win: int, batch_size: int, chunk_windows: int = 4096):
        """
        Memory-stable prediction for seq models (CNN/LSTM/Transformer).

        Instead of passing a massive sliding_window_view into Keras in one go
        (which can force large contiguous copies), we generate windows in chunks
        and predict chunk-by-chunk.

        Parameters
        ----------
        model : tf.keras.Model
        X2d : np.ndarray
            Shape (n_rows, n_features), float32 preferred.
        win : int
            Window length.
        batch_size : int
            Keras predict batch size.
        chunk_windows : int
            Number of windows per chunk (not rows). Lower = less peak RAM.

        Returns
        -------
        proba : np.ndarray
            Concatenated model outputs for all windows, shape (n_windows, n_classes).
        """
        from numpy.lib.stride_tricks import sliding_window_view

        try:
            n = int(X2d.shape[0])
            win = max(1, int(win))
            m = n - win + 1
            if m <= 0:
                return np.empty((0, 0), dtype=np.float32)

            chunk_windows = int(chunk_windows) if chunk_windows is not None else 0
            if chunk_windows <= 0:
                chunk_windows = 4096

            outs = []
            for s in range(0, m, chunk_windows):
                e = min(m, s + chunk_windows)

                # Need rows [s : e+win-1] to build exactly (e-s) windows
                X_slice = X2d[s : (e + win - 1)]
                Xv = sliding_window_view(X_slice, window_shape=win, axis=0)  # (e-s, win, f)

                p = model.predict(Xv, verbose=0, batch_size=int(batch_size))
                outs.append(p)

            proba = np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
            # Shape sanity: should match m windows
            if int(getattr(proba, "shape", [0])[0]) != int(m):
                raise ValueError(f"chunked predict produced wrong length: got {proba.shape[0]}, expected {m}")
            return proba

        except Exception as _e:
            # Fallback to one-shot predict (may be memory heavy, but keeps semantics)
            try:
                Xv = sliding_window_view(X2d, window_shape=win, axis=0)
                return model.predict(Xv, verbose=0, batch_size=int(batch_size))
            except Exception:
                raise _e


