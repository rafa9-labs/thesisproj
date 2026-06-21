"""
Model registry -- maps model type names -> builder callables.

Usage:
    from models.registry import build_model, MODEL_REGISTRY, register_model

    model = build_model("logistic", input_shape=(50,), use_proba=True, ...)

To register a new model:
    @register_model("my_model")
    def _build_my_model(*, seed=None, use_proba=True, **params):
        return MyModel(...)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a builder function under ``name``."""
    def decorator(fn):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def build_model(model_type: str, *, use_proba: bool = True, **params) -> Any:
    """Look up *model_type* in the registry and call its builder.

    Falls back to ``None`` (the caller should raise) if unknown.
    """
    builder = MODEL_REGISTRY.get(model_type)
    if builder is None:
        return None
    return builder(use_proba=use_proba, **params)


# ---------------------------------------------------------------------------
# Helpers shared across builders
# ---------------------------------------------------------------------------

def _extract_seed(params: dict) -> Optional[int]:
    """Try to pull a reproducible seed from params."""
    try:
        s = int(params.get("seed", 0))
        return s if s else None
    except Exception:
        return None


def _configure_tf_threads():
    """Set TF intra/inter op threads from env (idempotent)."""
    try:
        import tensorflow as _tf
        import os as _os
        intra = int(_os.getenv("TF_NUM_INTRAOP_THREADS", "0")) or max(1, (_os.cpu_count() or 8) - 2)
        inter = int(_os.getenv("TF_NUM_INTEROP_THREADS", "0")) or max(1, intra // 2)
        _tf.config.threading.set_intra_op_parallelism_threads(intra)
        _tf.config.threading.set_inter_op_parallelism_threads(inter)
    except Exception:
        pass


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


# ---------------------------------------------------------------------------
# Builder registrations
# ---------------------------------------------------------------------------

@register_model("logistic")
def _build_logistic(*, use_proba=True, **params):
    """LogisticRegression wrapped in a StandardScaler Pipeline."""
    import os
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    seed = _extract_seed(params)
    _raw = filter_params(params, "logit_") or {}
    logit_params = ensure_dict(_raw)

    # --- sanitize solver/penalty combos ---
    solver  = str(logit_params.get("solver", "saga")).strip().lower()
    penalty = str(logit_params.get("penalty", "l2")).strip().lower()
    allowed_solvers = {"lbfgs", "newton-cg", "liblinear", "sag", "saga"}
    allowed_penalty = {"l2", "l1", "elasticnet", "none"}
    if solver not in allowed_solvers: solver = "saga"
    if penalty not in allowed_penalty: penalty = "l2"

    if penalty == "l1":
        if solver in {"lbfgs", "newton-cg", "sag"}: solver = "saga"
    elif penalty == "elasticnet":
        solver = "saga"
        try:
            logit_params["l1_ratio"] = min(1.0, max(0.0, float(logit_params.get("l1_ratio", 0.5))))
        except Exception:
            logit_params["l1_ratio"] = 0.5
    elif penalty in {"none", "", None}:
        penalty = "none"
        if solver == "liblinear": solver = "lbfgs"

    if bool(logit_params.get("dual", False)) and not (solver == "liblinear" and penalty == "l2"):
        logit_params["dual"] = False

    try:
        C = float(logit_params.get("C", 1.0));    logit_params["C"] = C if C > 0 else 1.0
    except Exception:
        logit_params["C"] = 1.0
    try:
        tol = float(logit_params.get("tol", 1e-3)); logit_params["tol"] = tol if tol > 0 else 1e-3
    except Exception:
        logit_params["tol"] = 1e-3
    try:
        mi = int(logit_params.get("max_iter", 2000)); logit_params["max_iter"] = mi if mi > 0 else 2000
    except Exception:
        logit_params["max_iter"] = 2000

    logit_params["solver"] = solver

    # sklearn >=1.8 deprecated `penalty` and `n_jobs`
    try:
        import sklearn as _sk
        _sk_ge_18 = tuple(int(x) for x in _sk.__version__.split('.')[:2]) >= (1, 8)
    except Exception:
        _sk_ge_18 = False

    if _sk_ge_18:
        logit_params.pop("penalty", None)
        if penalty == "l2":
            logit_params.setdefault("l1_ratio", 0)
        elif penalty == "l1":
            logit_params["l1_ratio"] = 1.0
        elif penalty == "none":
            logit_params["C"] = float("inf")
        # elasticnet keeps its own l1_ratio from above
        logit_params.pop("n_jobs", None)
    else:
        logit_params["penalty"] = penalty
        logit_params.setdefault("n_jobs", int(os.environ.get("SKLEARN_JOBS", max(1, (os.cpu_count() or 2) - 1))))

    logit_params.setdefault("class_weight", "balanced")
    if seed is not None:
        logit_params.setdefault("random_state", seed)

    return Pipeline([("std", StandardScaler()), ("logit", LogisticRegression(**logit_params))])


@register_model("svm")
def _build_svm(*, use_proba=True, **params):
    """SVC wrapped in CalibratedClassifierCV for probability estimates."""
    import os
    import numpy as _np
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV

    seed = _extract_seed(params)
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
            v = float(x)
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

    max_iter = min(max_iter, 100_000)
    cache_sz = min(cache_sz, 1024.0)

    svc = SVC(
        C=C, gamma=gamma, kernel=kernel, degree=degree,
        class_weight=cw, probability=False,
        max_iter=max_iter, tol=tol, shrinking=shrinking, cache_size=cache_sz,
        decision_function_shape="ovr", random_state=seed,
    )

    calibrate_method = params.get("calibrate_method", None)
    if calibrate_method not in ("sigmoid", "isotonic"):
        calibrate_method = "isotonic"
    cal_jobs = int(params.get("svm_calib_n_jobs", 0)) or int(os.environ.get("SKLEARN_JOBS", -1))
    cal_jobs = max(1, min(cal_jobs, 2))
    return CalibratedClassifierCV(estimator=svc, cv=2, method=calibrate_method, n_jobs=cal_jobs)


@register_model("random_forest")
def _build_random_forest(*, use_proba=True, **params):
    """RandomForestClassifier with safe threading defaults."""
    import os
    from sklearn.ensemble import RandomForestClassifier

    seed = _extract_seed(params)
    rf_params = ensure_dict(filter_params(params, "rf_"))
    rf_params.setdefault("n_estimators", 300)
    try:
        rf_params["n_estimators"] = int(rf_params.get("n_estimators", 300))
    except Exception:
        rf_params["n_estimators"] = 300
    rf_params["n_estimators"] = max(1, min(rf_params["n_estimators"], 1200))
    rf_params.setdefault("max_depth", 18)
    rf_params.setdefault("min_samples_leaf", 10)
    rf_params.setdefault("max_features", "sqrt")
    rf_params.setdefault("class_weight", "balanced_subsample")

    bootstrap_flag = rf_params.get("bootstrap", True)
    if not bootstrap_flag:
        rf_params["oob_score"] = False
    else:
        rf_params.setdefault("oob_score", True)

    try:
        _safe_rf_jobs = int(os.environ.get("RF_JOBS", "1") or 1)
    except Exception:
        _safe_rf_jobs = 1
    if _safe_rf_jobs in (-1, 0):
        _safe_rf_jobs = max(1, (os.cpu_count() or 1) - 1)
    _rf_n_jobs = rf_params.get("n_jobs", _safe_rf_jobs)
    try: _rf_n_jobs = int(_rf_n_jobs)
    except Exception: _rf_n_jobs = _safe_rf_jobs
    if _rf_n_jobs in (-1, 0): _rf_n_jobs = _safe_rf_jobs
    _rf_n_jobs = max(1, min(_rf_n_jobs, (os.cpu_count() or 1)))
    rf_params["n_jobs"] = _rf_n_jobs

    if seed is not None:
        rf_params.setdefault("random_state", seed)
    return RandomForestClassifier(**rf_params)


@register_model("decision_tree")
def _build_decision_tree(*, use_proba=True, **params):
    """DecisionTreeClassifier with FX-friendly defaults."""
    from sklearn.tree import DecisionTreeClassifier

    seed = _extract_seed(params)
    dt_params = ensure_dict(filter_params(params, "dt_"))
    dt_params.setdefault("max_depth", 12)
    dt_params.setdefault("min_samples_split", 2)
    dt_params.setdefault("min_samples_leaf", 10)
    dt_params.setdefault("class_weight", "balanced")
    dt_params.setdefault("ccp_alpha", 1e-4)
    if seed is not None:
        dt_params.setdefault("random_state", seed)
    return DecisionTreeClassifier(**dt_params)


@register_model("xgboost")
def _build_xgboost(*, use_proba=True, **params):
    """XGBClassifier with GPU fallback."""
    import os
    from xgboost import XGBClassifier

    seed = _extract_seed(params)
    xgb_params = ensure_dict(filter_params(params, "xgb_"))
    xgb_params.setdefault("objective", "multi:softprob")
    xgb_params.setdefault("num_class", 3)
    xgb_params.setdefault("eval_metric", "mlogloss")
    xgb_params.setdefault("importance_type", "gain")
    xgb_params.setdefault("subsample", 0.8)
    xgb_params.setdefault("colsample_bytree", 0.8)
    xgb_params.setdefault("max_depth", 6)
    xgb_params.setdefault("n_estimators", 400)
    try:
        xgb_params["n_estimators"] = int(xgb_params.get("n_estimators", 400))
    except Exception:
        xgb_params["n_estimators"] = 400
    xgb_params["n_estimators"] = max(1, min(xgb_params["n_estimators"], 1500))
    xgb_params.setdefault("min_child_weight", 1.0)

    if "lambda" in xgb_params and "reg_lambda" not in xgb_params:
        xgb_params["reg_lambda"] = xgb_params.pop("lambda")
    xgb_params.setdefault("reg_lambda", 1.0)
    xgb_params.pop("lambda", None)

    xgb_params.setdefault("n_jobs", int(os.environ.get("XGB_JOBS", max(1, (os.cpu_count() or 2) - 1))))

    use_gpu = os.environ.get("XGB_USE_GPU", "0") == "1"
    if use_gpu:
        xgb_params.setdefault("tree_method", "hist")
        xgb_params["device"] = os.environ.get("XGB_DEVICE", "cuda")
        xgb_params.pop("predictor", None)
    else:
        xgb_params.setdefault("tree_method", "hist")
        xgb_params.pop("device", None)
        xgb_params.pop("predictor", None)

    if seed is not None:
        xgb_params.setdefault("random_state", seed)

    try:
        return XGBClassifier(**xgb_params)
    except Exception as e:
        if use_gpu:
            xgb_params.pop("device", None)
            xgb_params["tree_method"] = "hist"
            return XGBClassifier(**xgb_params)
        raise


@register_model("cnn")
def _build_cnn(*, use_proba=True, **params):
    """1D-CNN via Keras. Requires input_shape in params."""
    from models.cnn import build_cnn
    _configure_tf_threads()
    seed = _extract_seed(params)
    cfg = filter_params(params, "cnn_")
    if seed is not None: cfg.setdefault("seed", seed)
    input_shape = params["input_shape"]
    return build_cnn(input_shape=input_shape, config=cfg)


@register_model("lstm")
def _build_lstm(*, use_proba=True, **params):
    """Stacked LSTM via Keras. Requires input_shape in params."""
    from models.lstm import build_lstm
    _configure_tf_threads()
    seed = _extract_seed(params)
    cfg = filter_params(params, "lstm_")
    if seed is not None: cfg.setdefault("seed", seed)
    input_shape = params.get("input_shape")
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
        raise ValueError(f"Invalid input_shape for LSTM: {input_shape}")
    model = build_lstm(input_shape=input_shape, config=cfg)
    if model is None:
        raise RuntimeError("build_lstm returned None. Check model config or input shape.")
    return model


@register_model("transformer")
def _build_transformer(*, use_proba=True, **params):
    """Transformer via Keras. Requires input_shape in params."""
    from models.transformer import build_transformer
    _configure_tf_threads()
    seed = _extract_seed(params)
    cfg = filter_params(params, "transformer_")
    if seed is not None: cfg.setdefault("seed", seed)
    input_shape = params["input_shape"]
    return build_transformer(input_shape=input_shape, config=cfg)


@register_model("dqn")
def _build_dqn(*, use_proba=True, **params):
    """Dueling DQN agent. Requires input_shape in params."""
    from rl.dqn_agent import DQNAgent
    from pipeline.dqn_config import _coerce_dqn_cfg
    from rl.dqn_agent import filter_dqn_config

    _configure_tf_threads()
    seed = _extract_seed(params)
    input_shape = params["input_shape"]
    dqn_cfg = params.get("dqn_config", {}) or filter_params(params, "dqn_")
    dqn_cfg = filter_dqn_config(dqn_cfg or {})
    dqn_cfg["state_size"] = int(input_shape[0])
    if "window" not in dqn_cfg and "lags" in params:
        dqn_cfg["window"] = int(params["lags"])
    if seed is not None:
        dqn_cfg.setdefault("seed", seed)
    dqn_cfg = _coerce_dqn_cfg(dqn_cfg)
    return DQNAgent(**dqn_cfg)


@register_model("ensemble_adaptive_regime")
def _build_ensemble_adaptive_regime(*, use_proba=True, **params):
    """Adaptive regime-based Mixture-of-Experts ensemble."""
    from models.ensemble_adaptive_regime import AdaptiveRegimeStrategy

    seed = _extract_seed(params)
    lstm_config  = filter_params(params, "lstm_")
    rf_config    = filter_params(params, "rf_")
    logit_config = filter_params(params, "logit_")
    if seed is not None:
        lstm_config.setdefault("seed", seed)
        rf_config.setdefault("random_state", seed)
        logit_config.setdefault("random_state", seed)

    input_shape = params["input_shape"]
    return AdaptiveRegimeStrategy(
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


@register_model("lightgbm")
def _build_lightgbm(*, use_proba=True, **params):
    """LGBMClassifier — histogram-based gradient boosting (Microsoft).

    Literature: SSRN (2023) "Performance Analysis of Gradient Boosting Models
    for Forex Market Prediction" — LightGBM fastest and among most accurate
    vs XGBoost/CatBoost on FX data.
    """
    import os
    from lightgbm import LGBMClassifier

    seed = _extract_seed(params)
    lgb_params = ensure_dict(filter_params(params, "lgbm_"))
    lgb_params.setdefault("objective", "multiclass")
    lgb_params.setdefault("num_class", 3)
    lgb_params.setdefault("boosting_type", "gbdt")
    lgb_params.setdefault("n_estimators", 400)
    try:
        lgb_params["n_estimators"] = int(lgb_params.get("n_estimators", 400))
    except Exception:
        lgb_params["n_estimators"] = 400
    lgb_params["n_estimators"] = max(1, min(lgb_params["n_estimators"], 1500))
    lgb_params.setdefault("max_depth", 6)
    lgb_params.setdefault("num_leaves", 31)
    lgb_params.setdefault("learning_rate", 0.1)
    lgb_params.setdefault("subsample", 0.8)
    lgb_params.setdefault("colsample_bytree", 0.8)
    lgb_params.setdefault("reg_lambda", 1.0)
    lgb_params.setdefault("reg_alpha", 0.0)
    lgb_params.setdefault("min_child_samples", 20)
    lgb_params.setdefault("class_weight", "balanced")
    lgb_params.setdefault("n_jobs", int(os.environ.get("RF_JOBS", max(1, (os.cpu_count() or 2) - 1))))
    lgb_params.setdefault("verbosity", -1)
    if seed is not None:
        lgb_params.setdefault("random_state", seed)
    return LGBMClassifier(**lgb_params)


@register_model("catboost")
def _build_catboost(*, use_proba=True, **params):
    """CatBoostClassifier — ordered boosting (Yandex).

    Literature: SSRN (2023) forex comparison shows CatBoost often best with
    minimal tuning (handles categorical features natively).
    """
    import os
    from catboost import CatBoostClassifier

    seed = _extract_seed(params)
    cb_params = ensure_dict(filter_params(params, "cb_"))
    cb_params.setdefault("loss_function", "MultiClass")
    cb_params.setdefault("eval_metric", "MultiClass")
    cb_params.setdefault("iterations", 400)
    try:
        cb_params["iterations"] = int(cb_params.get("iterations", 400))
    except Exception:
        cb_params["iterations"] = 400
    cb_params["iterations"] = max(1, min(cb_params["iterations"], 1500))
    cb_params.setdefault("depth", 6)
    cb_params.setdefault("learning_rate", 0.1)
    cb_params.setdefault("subsample", 0.8)
    if float(cb_params.get("subsample", 1.0)) < 1.0:
        cb_params["bootstrap_type"] = "Bernoulli"
    cb_params.setdefault("l2_leaf_reg", 3.0)
    cb_params.setdefault("border_count", 128)
    cb_params.setdefault("thread_count", int(os.environ.get("RF_JOBS", max(1, (os.cpu_count() or 2) - 1))))
    cb_params.setdefault("logging_level", "Silent")
    cb_params.setdefault("allow_writing_files", False)
    if seed is not None:
        cb_params.setdefault("random_seed", seed)
    return CatBoostClassifier(**cb_params)


@register_model("gru")
def _build_gru(*, use_proba=True, **params):
    """Gated Recurrent Unit via Keras. Requires input_shape in params.

    Literature: Springer Digital Finance (2020) — GRU simpler, faster,
    statistically competitive or superior to LSTM across 4 FX pairs.
    """
    from models.gru import build_gru
    _configure_tf_threads()
    seed = _extract_seed(params)
    cfg = filter_params(params, "gru_")
    if seed is not None:
        cfg.setdefault("seed", seed)
    input_shape = params.get("input_shape")
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
        raise ValueError(f"Invalid input_shape for GRU: {input_shape}")
    model = build_gru(input_shape=input_shape, config=cfg)
    if model is None:
        raise RuntimeError("build_gru returned None. Check model config or input shape.")
    return model


@register_model("gru_lstm")
def _build_gru_lstm(*, use_proba=True, **params):
    """GRU-LSTM hybrid via Keras. Requires input_shape in params.

    Literature: Nature Scientific Reports (2025), ScienceDirect (2020) —
    outperforms standalone GRU, LSTM, and SMA on FX currency pairs.
    """
    from models.gru_lstm import build_gru_lstm
    _configure_tf_threads()
    seed = _extract_seed(params)
    cfg = filter_params(params, "gru_lstm_")
    if seed is not None:
        cfg.setdefault("seed", seed)
    input_shape = params.get("input_shape")
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
        raise ValueError(f"Invalid input_shape for GRU-LSTM: {input_shape}")
    model = build_gru_lstm(input_shape=input_shape, config=cfg)
    if model is None:
        raise RuntimeError("build_gru_lstm returned None. Check model config or input shape.")
    return model


@register_model("stacking_ensemble")
def _build_stacking_ensemble(*, use_proba=True, **params):
    """Stacking ensemble with OOF meta-learner via sklearn.StackingClassifier.

    Unlike MetaEnsemble (voting), this trains a LogisticRegression meta-learner
    on out-of-fold predictions from all base models.

    Literature: ResearchGate (2025) — stacking outperforms individual models
    on financial prediction tasks.
    """
    from models.stacking_ensemble import StackingEnsemble

    sub_types = params.get("stack_sub_models", ["logistic", "xgboost", "lightgbm"])
    if isinstance(sub_types, str):
        sub_types = [t.strip() for t in sub_types.split(",") if t.strip()]
    cv = int(params.get("stack_cv", 5))
    method = str(params.get("stack_method", "auto")).lower()

    base_models = []
    for t in sub_types:
        m = build_model(t, use_proba=True, **params)
        if m is not None:
            base_models.append(m)

    if len(base_models) < 2:
        raise ValueError(f"StackingEnsemble requires >=2 base models, got {len(base_models)} from types: {sub_types}")

    seed = _extract_seed(params)
    return StackingEnsemble(
        base_models=base_models,
        cv=cv,
        method=method,
        seed=seed,
        stack_sub_models=sub_types,
    )


@register_model("meta_ensemble")
def _build_meta_ensemble(*, use_proba=True, **params):
    """Signal committee: wraps N model types, combines via voting."""
    from models.meta_ensemble import MetaEnsemble

    sub_types = params.get("meta_sub_models", ["logistic", "xgboost"])
    if isinstance(sub_types, str):
        sub_types = [t.strip() for t in sub_types.split(",") if t.strip()]
    method = str(params.get("meta_combination_method", "majority")).lower()
    weights_raw = params.get("meta_weights")
    weights = [float(w) for w in weights_raw] if weights_raw and isinstance(weights_raw, list) else None

    sub_models = []
    for t in sub_types:
        m = build_model(t, use_proba=True, **params)
        if m is not None:
            sub_models.append(m)

    if not sub_models:
        raise ValueError(f"No valid sub-models built for types: {sub_types}")

    return MetaEnsemble(
        sub_models=sub_models,
        method=method,
        weights=weights,
        meta_sub_models=sub_types,
    )


# ---------------------------------------------------------------------------
# CNN-LSTM-XGBoost Ensemble (deep ensemble — trained via ensemble mixin)
# ---------------------------------------------------------------------------

@register_model("ensemble_cnn_lstm_xgboost")
def _build_ensemble_cnn_lstm_xgboost(*, seed=None, use_proba=True, **params):
    """Build a CNN-LSTM-XGBoost ensemble.

    This model is trained via the ensemble mixin path (test_ensemble_strategy).
    The factory exists so the registry can enumerate and validate it.
    """
    from .ensemble_cnn_lstm_xgboost import EnsembleCNNLSTMXGBoost

    cnn_config = {k: params[k] for k in ("cnn_filters", "cnn_kernel_size",
                 "cnn_pool_size", "cnn_dropout", "cnn_l2", "cnn_use_early_stopping",
                 "cnn_patience", "cnn_time_limit_sec", "cnn_batch_size") if k in params}
    lstm_config = {k: params[k] for k in ("lstm_units", "lstm_dropout",
                   "lstm_recurrent_dropout", "lstm_l2", "lstm_use_early_stopping",
                   "lstm_patience", "lstm_time_limit_sec", "lstm_batch_size") if k in params}
    xgb_config = {k: params[k] for k in ("xgb_max_depth", "xgb_learning_rate",
                  "xgb_n_estimators", "xgb_subsample", "xgb_colsample_bytree",
                  "xgb_min_child_weight", "xgb_gamma", "xgb_reg_alpha", "xgb_reg_lambda",
                   "use_logit_meta", "calibrate_base_temps", "use_oof_meta",
                   "oof_splits", "oof_purge_bars") if k in params}

    return EnsembleCNNLSTMXGBoost(
        cnn_config=cnn_config,
        lstm_config=lstm_config,
        xgb_config=xgb_config,
    )


# ---------------------------------------------------------------------------
# Regime Classifier (meta-model — used by exploration/committee layer)
# ---------------------------------------------------------------------------

@register_model("regime_classifier")
def _build_regime_classifier(*, seed=None, use_proba=True, **params):
    """Build a RegimeClassifier (Random Forest for 7-class market regime labeling).

    Not a trading-signal model. Used by the exploration agent, ExpertProfiler,
    and committee builder to classify market state and route to specialist models.
    """
    from models.regime_classifier import RegimeClassifier

    n_estimators = int(params.get("n_estimators", 100))
    max_depth_val = params.get("max_depth")
    max_depth = int(max_depth_val) if max_depth_val is not None else 8
    min_samples_leaf = int(params.get("min_samples_leaf", 50))
    class_weight = str(params.get("class_weight", "balanced_subsample"))
    random_state = int(params.get("random_state", seed or 42))
    feature_columns = params.get("feature_columns", None)

    from pipeline.regime_utils import RegimeConfig
    regime_cfg = RegimeConfig(**{k: v for k, v in params.items()
                                  if k in ("adx_thresh", "rsi_high", "rsi_low",
                                           "bbpct_high", "bbpct_low",
                                           "atr_high_quantile", "bbw_high_quantile",
                                           "bbw_low_quantile", "rv_low_quantile")})

    return RegimeClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        feature_columns=feature_columns,
        regime_cfg=regime_cfg,
    )