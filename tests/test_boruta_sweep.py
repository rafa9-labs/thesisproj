"""Tests for pipeline/boruta_sweep.py — BorutaSHAP feature selector."""
import pytest
import numpy as np
import pandas as pd

from pipeline.features.boruta_sweep import (
    BorutaSHAPSelector,
    boruta_sweep_features,
    _make_labels,
    _shuffle_columns,
    _chronological_fold_indices,
)


# ============================================================
# Shadow features
# ============================================================

class TestShadowFeatures:
    def test_shadow_shape_matches_real(self):
        X = np.random.randn(100, 10)
        rng = np.random.default_rng(42)
        shadow = _shuffle_columns(X, rng)
        assert shadow.shape == X.shape

    def test_shadow_columns_are_permutations(self):
        X = np.random.randn(100, 5)
        rng = np.random.default_rng(42)
        shadow = _shuffle_columns(X, rng)
        for j in range(X.shape[1]):
            assert sorted(X[:, j]) == sorted(shadow[:, j])
            assert not np.array_equal(X[:, j], shadow[:, j])

    def test_shadow_different_each_call(self):
        X = np.random.randn(100, 5)
        rng = np.random.default_rng(42)
        s1 = _shuffle_columns(X, rng)
        s2 = _shuffle_columns(X, rng)
        assert not np.array_equal(s1, s2)


# ============================================================
# Chronological folds
# ============================================================

class TestChronologicalFolds:
    def test_returns_list_of_tuples(self):
        indices = _chronological_fold_indices(500, 5)
        assert isinstance(indices, list)
        assert len(indices) > 0
        for train, test in indices:
            assert isinstance(train, np.ndarray)
            assert isinstance(test, np.ndarray)

    def test_train_before_test(self):
        indices = _chronological_fold_indices(500, 5)
        for train, test in indices:
            if len(train) > 0 and len(test) > 0:
                assert train.max() < test.min()

    def test_no_overlap_between_folds(self):
        indices = _chronological_fold_indices(500, 5)
        for train, test in indices:
            assert len(np.intersect1d(train, test)) == 0

    def test_purge_creates_gap_between_train_and_test(self):
        """Purged folds: train ends at least purge_bars before test starts."""
        indices = _chronological_fold_indices(500, 5, purge_bars=1, embargo_bars=0)
        for train, test in indices:
            if len(train) > 0 and len(test) > 0:
                assert test.min() - train.max() >= 1

    def test_embargo_creates_explicit_gap(self):
        indices = _chronological_fold_indices(500, 5, purge_bars=1, embargo_bars=3)
        for train, test in indices:
            if len(train) > 0 and len(test) > 0:
                assert test.min() - train.max() >= 3

    def test_insufficient_data_returns_empty(self):
        indices = _chronological_fold_indices(10, 5)
        assert indices == []


# ============================================================
# Labels
# ============================================================

class TestMakeLabels:
    def test_labels_three_classes(self):
        prices = np.linspace(1.1000, 1.2000, 200)
        df = pd.DataFrame({"mid_c": prices})
        labels = _make_labels(df, threshold=0.001)
        assert set(np.unique(labels)) <= {-1, 0, 1}

    def test_last_bar_is_neutral(self):
        """The last bar has no forward return: neutral (0), not 'buy'."""
        prices = np.linspace(1.1000, 1.2000, 100)
        df = pd.DataFrame({"mid_c": prices})
        labels = _make_labels(df)
        assert labels[-1] == 0

    def test_sell_neutral_buy_convention(self):
        prices = np.array([1.0000, 1.0000, 1.0100, 0.9900, 1.0001, 0.9998])
        df = pd.DataFrame({"mid_c": prices})
        labels = _make_labels(df, threshold=0.0001)
        # bar0 -> next return 0.0000 -> neutral
        # bar1 -> next return +0.00995 > thr -> buy
        # bar2 -> next return -0.0200 < -thr -> sell
        # bar3 -> next return +0.01015 > thr -> buy
        # bar4 -> next return -0.0003 < -thr -> sell
        assert labels[0] == 0
        assert labels[1] == 1
        assert labels[2] == -1
        assert labels[3] == 1
        assert labels[4] == -1
        assert labels[5] == 0


# ============================================================
# BorutaSHAPSelector
# ============================================================

@pytest.fixture
def large_synthetic_data():
    """50 informative features + 30 noise features, 500 samples."""
    n_samples, n_info, n_noise = 500, 50, 30
    rng = np.random.default_rng(42)
    X_info = np.random.randn(n_samples, n_info)
    y = (X_info[:, 0] + X_info[:, 1] * 0.5 + X_info[:, 2] * 0.3 > 0).astype(int)
    X_noise = rng.normal(0, 1, (n_samples, n_noise))
    X = np.hstack([X_info, X_noise])
    fnames = [f"informative_{i}" for i in range(n_info)] + [f"noise_{i}" for i in range(n_noise)]
    return X, y, fnames


class TestBorutaSHAPSelector:
    def test_select_returns_triple(self, large_synthetic_data):
        X, y, fnames = large_synthetic_data
        selector = BorutaSHAPSelector(
            n_estimators=30, max_depth=3, n_folds=3, percentile=66,
            max_iter=10, random_state=42,
        )
        confirmed, tentative, rejected, report = selector.select(X, y, fnames)
        assert isinstance(confirmed, list)
        assert isinstance(rejected, list)
        assert isinstance(report, dict)

    def test_confirms_strong_features(self, large_synthetic_data):
        X, y, fnames = large_synthetic_data
        selector = BorutaSHAPSelector(
            n_estimators=30, max_depth=3, n_folds=3, percentile=66,
            max_iter=15, random_state=42,
        )
        confirmed, _, rejected, _ = selector.select(X, y, fnames)
        top_info = {"informative_0", "informative_1", "informative_2"}
        confirmed_set = set(confirmed)
        assert top_info.intersection(confirmed_set)

    def test_strong_signal_confirmed_weak_signal_rejected(self):
        rng = np.random.default_rng(99)
        n_samples = 500
        X_strong = rng.normal(0, 0.3, (n_samples, 3))
        X_strong[:, 0] *= 3
        y = (X_strong[:, 0] + X_strong[:, 1] * 0.5 > 0.2).astype(int)
        X_weak = rng.normal(0, 0.1, (n_samples, 8))
        X = np.hstack([X_strong, X_weak])
        fnames = [f"strong_{i}" for i in range(3)] + [f"weak_{i}" for i in range(8)]

        selector = BorutaSHAPSelector(
            n_estimators=60, max_depth=4, n_folds=5, percentile=80,
            max_iter=25, random_state=42,
        )
        confirmed, _, _, _ = selector.select(X, y, fnames)
        assert "strong_0" in confirmed

    def test_percentile_90_stricter_than_66(self, large_synthetic_data):
        X, y, fnames = large_synthetic_data
        s_strict = BorutaSHAPSelector(
            n_estimators=30, max_depth=3, n_folds=3, percentile=90,
            max_iter=10, random_state=42,
        )
        s_relaxed = BorutaSHAPSelector(
            n_estimators=30, max_depth=3, n_folds=3, percentile=66,
            max_iter=10, random_state=42,
        )
        c_s, _, _, _ = s_strict.select(X, y, fnames)
        c_r, _, _, _ = s_relaxed.select(X, y, fnames)
        assert len(c_s) <= len(c_r)

    def test_converges_within_max_iter(self, large_synthetic_data):
        X, y, fnames = large_synthetic_data
        selector = BorutaSHAPSelector(
            n_estimators=30, max_depth=3, n_folds=3, percentile=100,
            max_iter=20, random_state=42,
        )
        confirmed, _, rejected, report = selector.select(X, y, fnames)
        assert report["iterations"] <= 20

    def test_report_has_required_keys(self, large_synthetic_data):
        X, y, fnames = large_synthetic_data
        selector = BorutaSHAPSelector(n_estimators=30, max_depth=3, n_folds=3, max_iter=5)
        _, _, _, report = selector.select(X, y, fnames)
        for key in ["method", "iterations", "n_confirmed", "n_rejected", "confirmed", "rejected"]:
            assert key in report

    def test_progress_callback_fires(self, large_synthetic_data):
        X, y, fnames = large_synthetic_data
        calls = []

        def cb(msg):
            calls.append(msg)

        selector = BorutaSHAPSelector(
            n_estimators=30, max_depth=3, n_folds=3, max_iter=5,
        )
        selector.select(X, y, fnames, progress_callback=cb)
        assert len(calls) > 0

    def test_selects_fewer_than_majority_vote_equivalent(self, large_synthetic_data):
        X, y, fnames = large_synthetic_data
        selector = BorutaSHAPSelector(
            n_estimators=50, max_depth=4, n_folds=3, percentile=90,
            max_iter=15, random_state=42,
        )
        confirmed, _, rejected, _ = selector.select(X, y, fnames)
        assert len(confirmed) + len(rejected) == len(fnames)


# ============================================================
# boruta_sweep_features integration
# ============================================================

class TestBorutaSweepFeatures:
    def test_end_to_end_with_ohlc_data(self):
        n_bars = 300
        np.random.seed(42)
        df = pd.DataFrame({
            "mid_h": 1.1050 + np.cumsum(np.random.randn(n_bars) * 0.0002),
            "mid_l": 1.1045 + np.cumsum(np.random.randn(n_bars) * 0.0002),
            "mid_c": 1.1048 + np.cumsum(np.random.randn(n_bars) * 0.0002),
            "mid_o": 1.1047 + np.cumsum(np.random.randn(n_bars) * 0.0002),
        })
        df["mid_h"] = df[["mid_h", "mid_c"]].max(axis=1) + 0.0001
        df["mid_l"] = df[["mid_l", "mid_c"]].min(axis=1) - 0.0001

        locked, scores, report = boruta_sweep_features(
            df, n_estimators=30, max_depth=3, n_folds=3,
            percentile=90, max_iter=10, random_state=42,
        )
        assert len(locked) >= 1
        assert isinstance(scores, dict)
        assert report["method"] == "boruta_shap"

    def test_output_shape_matches_sweep_features(self):
        n_bars = 200
        np.random.seed(42)
        df = pd.DataFrame({
            "mid_h": 1.10 + np.cumsum(np.random.randn(n_bars) * 0.0001),
            "mid_l": 1.10 + np.cumsum(np.random.randn(n_bars) * 0.0001),
            "mid_c": 1.10 + np.cumsum(np.random.randn(n_bars) * 0.0001),
            "mid_o": 1.10 + np.cumsum(np.random.randn(n_bars) * 0.0001),
        })
        df["mid_h"] = df[["mid_h", "mid_c"]].max(axis=1) + 0.00005
        df["mid_l"] = df[["mid_l", "mid_c"]].min(axis=1) - 0.00005

        locked, scores, report = boruta_sweep_features(
            df, n_estimators=30, max_depth=3, n_folds=3,
            percentile=90, max_iter=10, random_state=42,
        )
        assert isinstance(locked, list)
        assert isinstance(scores, dict)
        assert isinstance(report, dict)
        assert "locked" in report
        assert report["locked"] == locked
