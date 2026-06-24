"""Tests for HMMRegimeDetector — P2 probabilistic regime detection."""
import numpy as np
import pandas as pd
import pytest

from pipeline.regime.hmm_regime import HMMRegimeDetector


def _make_ohlc_df(n_bars=500, seed=42):
    """Generate synthetic OHLC data with distinct regimes."""
    rng = np.random.default_rng(seed)
    n1 = n_bars // 3
    n2 = n_bars // 3
    n3 = n_bars - n1 - n2

    # Regime 1: uptrend with low vol
    trend_up = np.cumsum(rng.normal(0.0003, 0.001, n1)) + 1.0
    # Regime 2: downtrend with moderate vol
    trend_down = np.cumsum(rng.normal(-0.0003, 0.0015, n2)) + trend_up[-1]
    # Regime 3: high volatility chop
    high_vol = np.cumsum(rng.normal(0.0, 0.003, n3)) + trend_down[-1]

    mid_c = np.concatenate([trend_up, trend_down, high_vol])
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    df = pd.DataFrame({
        "mid_c": mid_c,
        "mid_o": np.roll(mid_c, 1),
        "mid_h": mid_c * 1.001,
        "mid_l": mid_c * 0.999,
        "spread": np.full(n_bars, 0.0001),
    }, index=dates)
    df["mid_o"].iloc[0] = mid_c[0]
    return df


class TestHMMRegimeDetectorComputeFeatures:
    def test_feature_shape(self):
        df = _make_ohlc_df(200)
        feats = HMMRegimeDetector.compute_features(df)
        assert feats.shape[0] == 200
        assert "ret_mean_20" in feats.columns
        assert "ret_std_20" in feats.columns
        assert "ret_ac1_5" in feats.columns
        assert "spread_ratio" in feats.columns

    def test_features_finite(self):
        df = _make_ohlc_df(200)
        feats = HMMRegimeDetector.compute_features(df)
        valid = feats.dropna()
        assert np.isfinite(valid.values).all()

    def test_no_spread_column(self):
        df = _make_ohlc_df(200)
        del df["spread"]
        feats = HMMRegimeDetector.compute_features(df)
        assert feats.shape[0] == 200


class TestHMMRegimeDetectorFit:
    def test_fit_with_auto_n_states(self):
        df = _make_ohlc_df(500)
        det = HMMRegimeDetector(random_state=42)
        det.fit(df)
        assert det.is_fitted
        assert 3 <= det.selected_n_states <= 7

    def test_fit_with_fixed_n_states(self):
        df = _make_ohlc_df(500)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.fit(df)
        assert det.is_fitted
        assert det.selected_n_states == 3

    def test_fit_preserves_state_map(self):
        df = _make_ohlc_df(500)
        det = HMMRegimeDetector(random_state=42)
        det.fit(df)
        assert len(det._state_to_regime) == det.selected_n_states
        # All regime IDs should be in valid range
        for regime_id in det._state_to_regime.values():
            assert 0 <= regime_id <= 6

    def test_reproducible_with_seed(self):
        df = _make_ohlc_df(500)
        det1 = HMMRegimeDetector(random_state=42)
        det1.fit(df)
        det2 = HMMRegimeDetector(random_state=42)
        det2.fit(df)
        assert det1.selected_n_states == det2.selected_n_states
        assert det1._state_to_regime == det2._state_to_regime

    def test_too_few_bars_skips_fit(self):
        df = _make_ohlc_df(10)
        det = HMMRegimeDetector(random_state=42)
        det.fit(df)
        assert not det.is_fitted


class TestHMMRegimeDetectorPredict:
    def test_predict_hard_returns_valid_ids(self):
        df = _make_ohlc_df(500)
        det = HMMRegimeDetector(random_state=42)
        det.fit(df)
        regime_ids = det.predict_hard(df)
        assert len(regime_ids) == 500
        assert all(0 <= r <= 6 for r in regime_ids)
        assert regime_ids.dtype == np.int8

    def test_predict_regime_probs_shapes(self):
        df = _make_ohlc_df(500)
        det = HMMRegimeDetector(random_state=42)
        det.fit(df)
        probs = det.predict_regime_probs(df)
        assert probs.shape == (500, 7)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.01)

    def test_predict_unfitted_returns_sideways(self):
        df = _make_ohlc_df(100)
        det = HMMRegimeDetector(random_state=42)
        regime_ids = det.predict_hard(df)
        assert all(r == 6 for r in regime_ids)  # all sideways

    def test_predict_returns_correct_types(self):
        df = _make_ohlc_df(500)
        det = HMMRegimeDetector(random_state=42)
        det.fit(df)
        regime_ids, hmm_probs = det.predict(df)
        assert isinstance(regime_ids, np.ndarray)
        assert isinstance(hmm_probs, np.ndarray)
        assert regime_ids.dtype == np.int8


class TestHMMRegimeDetectorPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        df = _make_ohlc_df(500)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.fit(df)

        path = str(tmp_path / "hmm_detector.joblib")
        det.save(path)
        loaded = HMMRegimeDetector.load(path)
        assert loaded.is_fitted
        assert loaded.selected_n_states == det.selected_n_states
        assert loaded._state_to_regime == det._state_to_regime

        # Predictions should match
        orig_ids = det.predict_hard(df)
        loaded_ids = loaded.predict_hard(df)
        assert np.array_equal(orig_ids, loaded_ids)


class TestHMMRegimeDetectorRegimeDiversity:
    def test_multiple_regimes_detected(self):
        """Synthetic data with 3 regimes should produce at least 2 distinct regime IDs."""
        df = _make_ohlc_df(500)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.fit(df)
        regime_ids = det.predict_hard(df)
        unique = set(regime_ids)
        assert len(unique) >= 2, f"Only found regimes: {unique}"
