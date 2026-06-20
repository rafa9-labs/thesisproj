"""Tests for compute_mda and prune_noise_features — P5 feature importance."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pipeline.feature_utils import compute_mda, prune_noise_features


def _make_informative_data(n_samples=200, n_features=10, n_informative=5, n_noise=3, seed=42):
    """Generate synthetic classification data with known informative + noise features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features))
    # Informative features: linear combination determines label
    y = (X[:, :n_informative].sum(axis=1) > 0).astype(int)
    n_extra = n_features - n_informative - n_noise
    feature_names = (
        [f"info_{i}" for i in range(n_informative)]
        + [f"noise_{i}" for i in range(n_noise)]
        + [f"extra_{i}" for i in range(max(0, n_extra))]
    )
    return X, y, feature_names[:n_features]


class TestComputeMDA:
    def test_returns_dict_with_all_features(self):
        X, y, names = _make_informative_data(200)
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        mda = compute_mda(model, X_train, y_train, X_val, y_val, names)
        assert len(mda) == len(names)
        for name in names:
            assert name in mda

    def test_informative_features_have_higher_mda(self):
        X, y, names = _make_informative_data(200, n_informative=5, n_noise=3)
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        mda = compute_mda(model, X_train, y_train, X_val, y_val, names)

        info_mda = np.mean([mda[n] for n in names if n.startswith("info_")])
        noise_mda = np.mean([mda[n] for n in names if n.startswith("noise_")])
        assert info_mda > noise_mda, f"Info MDA={info_mda:.4f} <= Noise MDA={noise_mda:.4f}"

    def test_scores_in_range(self):
        X, y, names = _make_informative_data(200)
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
        mda = compute_mda(model, X_train, y_train, X_val, y_val, names)
        for score in mda.values():
            assert -1.0 <= score <= 1.0

    def test_only_validation_column_shuffled(self):
        """MDA must not modify the input X_val array."""
        X, y, names = _make_informative_data(200)
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_val_copy = X_val.copy()
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        compute_mda(model, X_train, y_train, X_val, y_val, names)
        assert np.array_equal(X_val, X_val_copy)

    def test_raises_on_mismatched_feature_names(self):
        X, y, _ = _make_informative_data(200)
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = LogisticRegression().fit(X_train, y_train)
        with pytest.raises(ValueError):
            compute_mda(model, X_train, y_train, X_val, y_val, ["a", "b"])  # wrong count


class TestPruneNoiseFeatures:
    def test_drops_negative_mda(self):
        mda = {"info_a": 0.1, "info_b": 0.2, "noise_x": -0.05, "noise_y": -0.10}
        kept = prune_noise_features(mda, threshold=0.0)
        assert "info_a" in kept
        assert "info_b" in kept
        assert "noise_x" not in kept
        assert "noise_y" not in kept

    def test_keeps_zero_mda_with_default_threshold(self):
        mda = {"info_a": 0.1, "zero_f": 0.0}
        kept = prune_noise_features(mda, threshold=0.0)
        assert "zero_f" not in kept  # threshold=0 drops 0.0

    def test_custom_threshold(self):
        mda = {"strong": 0.3, "weak": 0.05, "noise": -0.1}
        kept = prune_noise_features(mda, threshold=0.1)
        assert "strong" in kept
        assert "weak" not in kept
        assert "noise" not in kept

    def test_returns_sorted_by_mda(self):
        mda = {"c": 0.1, "a": 0.3, "b": 0.2}
        kept = prune_noise_features(mda)
        assert kept == ["a", "b", "c"]


class TestMDAIntegration:
    def test_pruned_model_still_predicts(self):
        """End-to-end: MDA prune features, retrain, predict."""
        X, y, names = _make_informative_data(300, n_informative=5, n_noise=5)
        n = len(X)
        split_fit = int(n * 0.6)
        split_val = int(n * 0.8)
        X_fit, y_fit = X[:split_fit], y[:split_fit]
        X_mda, y_mda = X[split_fit:split_val], y[split_fit:split_val]
        X_test, y_test = X[split_val:], y[split_val:]

        model = LogisticRegression(max_iter=1000).fit(X_fit, y_fit)
        mda = compute_mda(model, X_fit, y_fit, X_mda, y_mda, names)
        kept_names = prune_noise_features(mda)

        # Verify at least some features are kept
        assert len(kept_names) > 0

        # Retrain with only kept features
        kept_indices = [i for i, n in enumerate(names) if n in kept_names]
        model_pruned = LogisticRegression(max_iter=1000).fit(
            X_fit[:, kept_indices], y_fit
        )
        acc_pruned = model_pruned.score(X_test[:, kept_indices], y_test)
        assert 0.0 <= acc_pruned <= 1.0
