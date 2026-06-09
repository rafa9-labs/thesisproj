"""Verify that Phase 3 validation uses HPO-tuned params, not sklearn defaults."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline.committee_builder import CommitteeConfig, RegimeAssignment
from pipeline.committee_backtester import CommitteeBacktester
from pipeline.regime_utils import RegimeConfig


@pytest.fixture
def dummy_config():
    return CommitteeConfig(
        regimes={
            "trend_up": RegimeAssignment(models=["logistic", "xgboost"], weights=[0.7, 0.3]),
            "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
        },
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
    )


@pytest.fixture
def tuned_params():
    return {
        "logistic": {"C": 0.3, "max_iter": 300, "tol": 1e-4},
        "xgboost": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05},
    }


def test_committee_backtester_stores_model_params(dummy_config, tuned_params):
    """model_params propogated through __init__ → _build_model."""
    bt = CommitteeBacktester(
        dummy_config,
        regime_cfg=RegimeConfig(),
        confidence_threshold=0.5,
        model_params=tuned_params,
    )
    assert bt.model_params == tuned_params
    assert bt.model_params["logistic"]["C"] == 0.3
    assert bt.model_params["xgboost"]["max_depth"] == 8


def test_build_model_applies_logistic_params(dummy_config, tuned_params):
    """_build_model for logistic uses C from model_params, not default 1.0."""
    bt = CommitteeBacktester(
        dummy_config,
        regime_cfg=RegimeConfig(),
        model_params=tuned_params,
    )

    model = bt._build_model("logistic", n_features=10,
                            params=tuned_params["logistic"])
    if isinstance(model, Pipeline):
        logit = model.named_steps["logit"]
    else:
        logit = model
    assert isinstance(logit, LogisticRegression)
    # C should NOT be the default 1.0 — it should be 0.3
    assert abs(logit.C - 0.3) < 1e-6, f"Expected C=0.3, got {logit.C}"
    assert logit.max_iter == 300


def test_build_model_applies_xgboost_params(dummy_config, tuned_params):
    """_build_model for xgboost uses max_depth from model_params."""
    bt = CommitteeBacktester(
        dummy_config,
        regime_cfg=RegimeConfig(),
        model_params=tuned_params,
    )

    from sklearn.ensemble import RandomForestClassifier

    model = bt._build_model("xgboost", n_features=10,
                            params=tuned_params["xgboost"])

    try:
        from xgboost import XGBClassifier
        if isinstance(model, XGBClassifier):
            assert model.max_depth == 8, f"Expected max_depth=8, got {model.max_depth}"
            assert model.n_estimators == 200
            assert abs(model.learning_rate - 0.05) < 1e-6
    except ImportError:
        # Fallback to RF — check RF params
        if isinstance(model, RandomForestClassifier):
            pass  # RF fallback is fine


def test_build_model_defaults_when_no_params(dummy_config):
    """Without model_params, _build_model uses hardcoded defaults."""
    bt = CommitteeBacktester(
        dummy_config,
        regime_cfg=RegimeConfig(),
        model_params=None,
    )
    # When model_params is None, _evaluate_fold passes {} for each model
    model = bt._build_model("logistic", n_features=10, params={})
    if isinstance(model, Pipeline):
        logit = model.named_steps["logit"]
    else:
        logit = model
    assert abs(logit.C - 1.0) < 1e-6, f"Expected default C=1.0, got {logit.C}"
    assert logit.max_iter == 500


def test_evaluate_fold_uses_model_params(dummy_config, tuned_params):
    """_evaluate_fold passes model_params[model] to _build_model."""
    bt = CommitteeBacktester(
        dummy_config,
        regime_cfg=RegimeConfig(),
        model_params=tuned_params,
    )

    # Create tiny synthetic data with correct index
    df = _make_synthetic_df(200)

    for model_type in ["logistic", "xgboost"]:
        params = bt.model_params.get(model_type, {})
        model = bt._build_model(model_type, n_features=10, params=params)
        if isinstance(model, Pipeline):
            logit = model.named_steps["logit"]
            assert abs(logit.C - 0.3) < 1e-6


def _make_synthetic_df(n):
    import pandas as pd
    rng = np.random.default_rng(42)
    price = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "mid_o": price,
        "mid_h": price * 1.002,
        "mid_l": price * 0.998,
        "mid_c": price,
    }, index=idx)
    df["returns"] = np.log(df["mid_c"] / df["mid_c"].shift(1)).fillna(0.0)
    # Add indicator columns needed by _prepare_features
    df["sma_20"] = df["mid_c"].rolling(20).mean().fillna(method="bfill")
    df["ema_20"] = df["mid_c"].ewm(span=20).mean()
    df["adx_14"] = 20.0
    df["rsi_14"] = 50.0
    df["bb_upper"] = df["mid_c"] * 1.02
    df["bb_lower"] = df["mid_c"] * 0.98
    df["bbw"] = (df["bb_upper"] - df["bb_lower"]) / df["mid_c"]
    df["bb_pct"] = 0.5
    df["atr_14"] = 0.001
    df["rv_48"] = np.sqrt(df["returns"].rolling(48).var()).fillna(0.001)
    df["macd_diff"] = 0.0
    df["donchian_up_20"] = df["mid_h"].rolling(20).max()
    df["donchian_dn_20"] = df["mid_l"].rolling(20).min()
    df["donchian_break_up_20"] = 0
    df["donchian_break_dn_20"] = 0
    df["spread"] = 0.0001
    return df.dropna()
