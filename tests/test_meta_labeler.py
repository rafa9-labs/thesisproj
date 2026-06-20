"""Tests for the MetaLabeler — binary P(win) secondary model (P1)."""
import numpy as np
import pytest

from pipeline.meta_labeler import MetaLabeler


def _make_bar_predictions(signal, next_return, prob_short=0.2, prob_flat=0.3, prob_long=0.5, regime_id=3):
    return [{
        "committee_signal": signal,
        "next_return": next_return,
        "committee_prob_short": prob_short,
        "committee_prob_flat": prob_flat,
        "committee_prob_long": prob_long,
        "committee_confidence": max(prob_short, prob_flat, prob_long),
        "regime_id": regime_id,
    }]


class TestMetaLabelerBuildTargets:
    def test_long_win_becomes_1(self):
        bars = _make_bar_predictions(signal=1, next_return=0.001)
        y = MetaLabeler.build_targets(bars)
        assert y[0] == 1

    def test_long_loss_becomes_0(self):
        bars = _make_bar_predictions(signal=1, next_return=-0.001)
        y = MetaLabeler.build_targets(bars)
        assert y[0] == 0

    def test_short_win_becomes_1(self):
        bars = _make_bar_predictions(signal=-1, next_return=-0.001)
        y = MetaLabeler.build_targets(bars)
        assert y[0] == 1

    def test_short_loss_becomes_0(self):
        bars = _make_bar_predictions(signal=-1, next_return=0.001)
        y = MetaLabeler.build_targets(bars)
        assert y[0] == 0

    def test_flat_signal_skipped(self):
        bars = _make_bar_predictions(signal=0, next_return=0.001)
        y = MetaLabeler.build_targets(bars)
        assert len(y) == 0  # no trade → no meta-label

    def test_mixed_signals(self):
        bars = [
            _make_bar_predictions(1, 0.001)[0],   # win
            _make_bar_predictions(1, -0.001)[0],  # loss
            _make_bar_predictions(-1, -0.001)[0], # win
            _make_bar_predictions(-1, 0.001)[0],  # loss
            _make_bar_predictions(0, 0.001)[0],   # skip
        ]
        y = MetaLabeler.build_targets(bars)
        assert list(y) == [1, 0, 1, 0]


class TestMetaLabelerBuildFeatures:
    def test_feature_shape(self):
        bars = [
            _make_bar_predictions(1, 0.001)[0],
            _make_bar_predictions(-1, -0.001)[0],
        ]
        X = MetaLabeler.build_features(bars)
        assert X.shape == (2, 7)  # 7 feature columns

    def test_flat_skipped(self):
        bars = _make_bar_predictions(0, 0.001)
        X = MetaLabeler.build_features(bars)
        assert len(X) == 0

    def test_conviction_spread(self):
        bars = _make_bar_predictions(1, 0.001, prob_short=0.1, prob_long=0.7)
        X = MetaLabeler.build_features(bars)
        assert X[0, 4] == pytest.approx(0.6)  # 0.7 - 0.1

    def test_directional_commitment(self):
        bars = _make_bar_predictions(1, 0.001, prob_flat=0.2)
        X = MetaLabeler.build_features(bars)
        assert X[0, 5] == pytest.approx(0.8)  # 1 - 0.2


class TestMetaLabelerTrain:
    def test_train_on_balanced_data(self):
        bars = []
        for _ in range(50):
            bars.append(_make_bar_predictions(1, 0.001)[0])   # win
            bars.append(_make_bar_predictions(1, -0.001)[0])  # loss
        X = MetaLabeler.build_features(bars)
        y = MetaLabeler.build_targets(bars)
        meta = MetaLabeler()
        acc = meta.train(X, y)
        assert meta.is_trained
        assert 0.0 <= acc <= 1.0

    def test_train_too_few_samples(self):
        bars = [_make_bar_predictions(1, 0.001)[0]]
        X = MetaLabeler.build_features(bars)
        y = MetaLabeler.build_targets(bars)
        meta = MetaLabeler()
        acc = meta.train(X, y)
        assert not meta.is_trained
        assert acc == 0.0

    def test_train_all_wins(self):
        bars = [_make_bar_predictions(1, 0.001)[0] for _ in range(5)]
        X = MetaLabeler.build_features(bars)
        y = MetaLabeler.build_targets(bars)
        meta = MetaLabeler()
        acc = meta.train(X, y)
        assert not meta.is_trained  # class imbalance prevents training


class TestMetaLabelerInference:
    def test_should_trade_returns_true_when_untrained(self):
        meta = MetaLabeler()
        should, p_win = meta.should_trade(1, (0.2, 0.3, 0.5), 3)
        assert should is True   # pass-through
        assert p_win == 0.5

    def test_should_trade_returns_false_for_flat(self):
        meta = MetaLabeler()
        should, p_win = meta.should_trade(0, (0.3, 0.4, 0.3), 3)
        assert should is False
        assert p_win == 0.0

    def test_predict_win_prob_untrained_returns_neutral(self):
        meta = MetaLabeler()
        prob = meta.predict_win_prob((0.2, 0.3, 0.5), 3)
        assert prob == 0.5

    def test_trained_model_predicts_reasonable(self):
        bars = []
        for _ in range(30):
            bars.append(_make_bar_predictions(1, 0.001, prob_long=0.7)[0])
            bars.append(_make_bar_predictions(1, -0.001, prob_long=0.35)[0])
        X = MetaLabeler.build_features(bars)
        y = MetaLabeler.build_targets(bars)
        meta = MetaLabeler()
        meta.train(X, y)
        assert meta.is_trained

        prob = meta.predict_win_prob((0.2, 0.3, 0.5), 3)
        assert 0.0 <= prob <= 1.0


class TestMetaLabelerPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        bars = []
        for _ in range(30):
            bars.append(_make_bar_predictions(1, 0.001, prob_long=0.7)[0])
            bars.append(_make_bar_predictions(1, -0.001, prob_long=0.35)[0])
        X = MetaLabeler.build_features(bars)
        y = MetaLabeler.build_targets(bars)
        meta = MetaLabeler(override_threshold=0.55)
        meta.train(X, y)

        path = str(tmp_path / "meta_labeler.joblib")
        meta.save(path)
        loaded = MetaLabeler.load(path)
        assert loaded.is_trained
        assert loaded.override_threshold == 0.55
        assert loaded.accuracy == meta.accuracy
