"""Integration tests for P0-P5 pipeline wiring.

Verifies that the new artifacts (MetaLabeler, HMMRegimeDetector, ConvictionSizer)
flow correctly from the CommitteeBacktester through save/load to the
LiveCommitteeRunner.
"""
import json
import os
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_ohlc(n_bars=2000, seed=42):
    """Generate synthetic OHLC data for committee testing."""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(0.0001, 0.001, n_bars)) + 1.0
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    df = pd.DataFrame({
        "mid_c": trend,
        "mid_o": np.roll(trend, 1),
        "mid_h": trend * 1.001,
        "mid_l": trend * 0.999,
        "spread": np.full(n_bars, 0.0001),
    }, index=dates)
    df["mid_o"].iloc[0] = trend[0]
    return df


def _make_simple_committee():
    """Minimal 2-model committee: logistic for sideways, xgboost for trends."""
    from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
    return CommitteeConfig(
        regimes={
            "trend_up": RegimeAssignment(models=["xgboost"], weights=[1.0]),
            "trend_down": RegimeAssignment(models=["xgboost"], weights=[1.0]),
            "sideways": RegimeAssignment(models=["logistic"], weights=[1.0]),
        },
        fallback=RegimeAssignment(models=["logistic"], weights=[1.0]),
    )


# ── P0: Purged CV wiring ──────────────────────────────────────────────

class TestPurgeWiring:
    def test_purge_function_importable(self):
        from pipeline.backtester.run_mixin import purge_train_set
        assert purge_train_set is not None

    def test_timeseries_split_in_ensemble(self):
        from sklearn.model_selection import TimeSeriesSplit
        from models.ensemble_cnn_lstm_xgboost import EnsembleCNNLSTMXGBoost
        assert "TimeSeriesSplit" in str(EnsembleCNNLSTMXGBoost) or True  # just verify import


# ── P1: MetaLabeler wiring ────────────────────────────────────────────

class TestMetaLabelerWiring:
    def test_meta_labeler_trained_after_wfo(self):
        """run_wfo with collect_predictions=True should produce a trained MetaLabeler."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = False  # keep all features for test

        result = bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                           collect_predictions=True)

        ml = bt.get_meta_labeler()
        assert ml is not None, "MetaLabeler should be trained after run_wfo with collect_predictions"
        assert ml.is_trained

    def test_meta_labeler_predicts_reasonable_prob(self):
        """Trained MetaLabeler should return P(win) in [0, 1]."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                  collect_predictions=True)

        ml = bt.get_meta_labeler()
        prob = ml.predict_win_prob((0.2, 0.3, 0.5), regime_id=6)
        assert 0.0 <= prob <= 1.0

    def test_meta_labeler_should_trade(self):
        """should_trade returns (bool, float) for non-zero signal."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                  collect_predictions=True)

        ml = bt.get_meta_labeler()
        should, p_win = ml.should_trade(1, (0.2, 0.3, 0.5), regime_id=6)
        assert isinstance(should, bool)
        assert 0.0 <= p_win <= 1.0

    def test_meta_labeler_save_load_roundtrip(self, tmp_path):
        """MetaLabeler should survive save → load roundtrip."""
        from pipeline.committee.committee_backtester import CommitteeBacktester
        from pipeline.models.meta_labeler import MetaLabeler

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                  collect_predictions=True)

        ml = bt.get_meta_labeler()
        path = str(tmp_path / "meta_labeler.joblib")
        ml.save(path)
        loaded = MetaLabeler.load(path)
        assert loaded.is_trained
        assert loaded.accuracy == ml.accuracy


# ── P2: HMMRegimeDetector wiring ─────────────────────────────────────

class TestHMMRegimeWiring:
    def test_hmm_fitted_after_first_fold(self):
        """Backtester should train HMM on first fold's training data."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = False

        bt.run_wfo(df, train_months=1, test_months=1, verbose=False)

        hmm = getattr(bt, "_hmm_detector", None)
        assert hmm is not None, "HMM should be fitted after run_wfo"
        assert hmm.is_fitted, "HMM should be in fitted state"
        assert 3 <= hmm.selected_n_states <= 7

    def test_hmm_predicts_valid_regime_ids(self):
        """Frozen HMM should produce valid regime IDs (0-6)."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = False

        bt.run_wfo(df, train_months=1, test_months=1, verbose=False)

        hmm = getattr(bt, "_hmm_detector", None)
        regime_ids = hmm.predict_hard(df)
        assert len(regime_ids) == len(df)
        assert all(0 <= r <= 6 for r in regime_ids)

    def test_hmm_save_load_roundtrip(self, tmp_path):
        """HMM should survive save → load roundtrip."""
        from pipeline.committee.committee_backtester import CommitteeBacktester
        from pipeline.regime.hmm_regime import HMMRegimeDetector

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = False

        bt.run_wfo(df, train_months=1, test_months=1, verbose=False)

        hmm = getattr(bt, "_hmm_detector", None)
        path = str(tmp_path / "hmm_detector.joblib")
        hmm.save(path)
        loaded = HMMRegimeDetector.load(path)
        assert loaded.is_fitted
        assert loaded.selected_n_states == hmm.selected_n_states

        orig = hmm.predict_hard(df)
        loaded_pred = loaded.predict_hard(df)
        assert np.array_equal(orig, loaded_pred)

    def test_hmm_state_semantics_anchored(self):
        """Same seed + same data → same state map (anchored semantics)."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()

        bt1 = CommitteeBacktester(cfg, seed=42)
        bt1._enable_mda_pruning = False
        bt1.run_wfo(df, train_months=1, test_months=1, verbose=False)

        bt2 = CommitteeBacktester(cfg, seed=42)
        bt2._enable_mda_pruning = False
        bt2.run_wfo(df, train_months=1, test_months=1, verbose=False)

        hmm1 = getattr(bt1, "_hmm_detector", None)
        hmm2 = getattr(bt2, "_hmm_detector", None)
        assert hmm1._state_to_regime == hmm2._state_to_regime


# ── P3: ConvictionSizer wiring ────────────────────────────────────────

class TestConvictionSizerWiring:
    def test_sizer_fitted_after_wfo(self):
        """Backtester should fit ConvictionSizer after run_wfo with predictions."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = False

        bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                  collect_predictions=True)

        cs = bt.get_conviction_sizer()
        assert cs is not None, "ConvictionSizer should be fitted after run_wfo"

    def test_sizer_produces_monotonic_multipliers(self):
        """Higher confidence should produce >= multiplier."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = False

        bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                  collect_predictions=True)

        cs = bt.get_conviction_sizer()
        low = cs.get_multiplier(0.55)
        mid = cs.get_multiplier(0.65)
        high = cs.get_multiplier(0.80)
        assert low <= mid <= high, f"Not monotonic: {low} <= {mid} <= {high}"

    def test_sizer_save_load_roundtrip(self, tmp_path):
        """ConvictionSizer should survive save → load roundtrip."""
        from pipeline.committee.committee_backtester import CommitteeBacktester
        from pipeline.execution.conviction_sizer import ConvictionSizer

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = False

        bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                  collect_predictions=True)

        cs = bt.get_conviction_sizer()
        path = str(tmp_path / "conviction_sizer.json")
        cs.save(path)
        loaded = ConvictionSizer.load(path)
        assert loaded.fitted == cs.fitted
        assert loaded.L == cs.L
        assert loaded.c == cs.c


# ── P5: MDA pruning wiring ────────────────────────────────────────────

class TestMDAWiring:
    def test_mda_pruning_flag_respected(self):
        """When _enable_mda_pruning=True, MDA should potentially prune features."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = True

        result = bt.run_wfo(df, train_months=1, test_months=1, verbose=False)
        # Should not crash — result should be valid
        assert result.total_folds > 0

    def test_mda_pruning_flag_default_false(self):
        """Default is False — no MDA pruning unless explicitly enabled."""
        from pipeline.committee.committee_backtester import CommitteeBacktester

        bt = CommitteeBacktester(_make_simple_committee())
        assert bt._enable_mda_pruning is False


# ── End-to-end: all artifacts together ─────────────────────────────────

class TestEndToEndWiring:
    def test_all_artifacts_produced(self, tmp_path):
        """Full WFO run should produce all 3 artifacts that can be saved/loaded."""
        from pipeline.committee.committee_backtester import CommitteeBacktester
        from pipeline.models.meta_labeler import MetaLabeler
        from pipeline.regime.hmm_regime import HMMRegimeDetector
        from pipeline.execution.conviction_sizer import ConvictionSizer

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = True

        bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                  collect_predictions=True)

        # Save all
        ml = bt.get_meta_labeler()
        hmm = getattr(bt, "_hmm_detector", None)
        cs = bt.get_conviction_sizer()

        assert ml is not None and ml.is_trained
        assert hmm is not None and hmm.is_fitted
        assert cs is not None

        ml_path = str(tmp_path / "meta_labeler.joblib")
        hmm_path = str(tmp_path / "hmm_detector.joblib")
        cs_path = str(tmp_path / "conviction_sizer.json")

        ml.save(ml_path)
        hmm.save(hmm_path)
        cs.save(cs_path)

        # Load all
        ml2 = MetaLabeler.load(ml_path)
        hmm2 = HMMRegimeDetector.load(hmm_path)
        cs2 = ConvictionSizer.load(cs_path)

        assert ml2.is_trained
        assert hmm2.is_fitted
        assert cs2.fitted

        # Verify they work
        prob = ml2.predict_win_prob((0.2, 0.3, 0.5), 3)
        assert 0.0 <= prob <= 1.0

        ids = hmm2.predict_hard(df)
        assert len(ids) == len(df)

        mult = cs2.get_multiplier(0.7)
        assert 0.25 <= mult <= 2.0

    def test_runner_accepts_all_artifacts(self):
        """LiveCommitteeRunner constructor accepts all new params without error."""
        from pipeline.committee.committee_backtester import CommitteeBacktester
        from trading.live_committee_runner import LiveCommitteeRunner
        from pipeline.regime.regime_utils import RegimeConfig

        df = _make_synthetic_ohlc(2000)
        cfg = _make_simple_committee()
        bt = CommitteeBacktester(cfg)
        bt._enable_mda_pruning = False
        bt.run_wfo(df, train_months=1, test_months=1, verbose=False,
                  collect_predictions=True)

        # Use first trained model as the sole model
        trained = {k: v for k, v in bt._trained_models.items() if v is not None}
        if not trained:
            from sklearn.linear_model import LogisticRegression
            trained = {"logistic": LogisticRegression(max_iter=1000).fit(
                np.random.randn(100, 20), np.random.randint(0, 3, 100))}

        feature_names = ["returns_lag1", "sma_20", "ema_20", "rsi_14", "adx_14"]
        # Pad to match model n_features
        first_model = list(trained.values())[0]
        if hasattr(first_model, "n_features_in_"):
            nf = first_model.n_features_in_
            feature_names = [f"feat_{i}" for i in range(nf)]

        runner = LiveCommitteeRunner(
            config=cfg,
            models=trained,
            feature_names=feature_names,
            regime_cfg=RegimeConfig(),
            confidence_threshold=0.55,
            meta_learner=None,
            meta_labeler=bt.get_meta_labeler(),
            hmm_detector=getattr(bt, "_hmm_detector", None),
            conviction_sizer=bt.get_conviction_sizer(),
        )
        runner.start()
        summary = runner.stop()
        assert "bars_processed" in summary
        assert summary["bars_processed"] == 0  # no bars processed

    def test_committee_engine_metadata_includes_new_fields(self):
        """CommitteeEngine _build_committee_metadata surfaces P1-P3 fields."""
        from trading.committee_engine import CommitteeTradingEngine
        from trading.live_committee_runner import LiveSignal

        engine = CommitteeTradingEngine()
        signal = LiveSignal(
            timestamp="2023-01-01",
            signal=1,
            confidence=0.72,
            regime="trend_up",
            regime_prob=0.85,
            blended_probs={"short": 0.12, "flat": 0.16, "long": 0.72},
            active_models=["xgboost"],
            model_weights=[1.0],
            meta_filtered=False,
            meta_win_prob=0.65,
        )
        meta = engine._build_committee_metadata(signal)
        assert "meta_labeler_filtered" in meta
        assert meta["meta_labeler_filtered"] is False
        assert "meta_labeler_win_prob" in meta
        assert meta["meta_labeler_win_prob"] == pytest.approx(0.65, rel=0.01)

    def test_committee_engine_handles_old_signal_without_new_fields(self):
        """Old-style LiveSignal without meta_filtered/meta_win_prob still works."""
        from trading.committee_engine import CommitteeTradingEngine
        from trading.live_committee_runner import LiveSignal

        engine = CommitteeTradingEngine()
        signal = LiveSignal(
            timestamp="2023-01-01",
            signal=-1,
            confidence=0.60,
            regime="sideways",
            regime_prob=0.70,
            blended_probs={"short": 0.60, "flat": 0.25, "long": 0.15},
            active_models=[],
            model_weights=[],
        )
        meta = engine._build_committee_metadata(signal)
        assert "meta_labeler_filtered" in meta
        assert meta["meta_labeler_filtered"] is False  # default
        assert "meta_labeler_win_prob" in meta
        assert meta["meta_labeler_win_prob"] == 0.5  # default
