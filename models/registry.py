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
    return CalibratedClassifierCV(estimator=svc, cv=3, method=calibrate_method, n_jobs=cal_jobs)


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