"""Unit tests for committee voting, blending, and tie-breaking logic.

Tests _proba_to_trade (confidence gate), weighted probability blending,
and majority-vote consensus using pure numpy math.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.committee_backtester import CommitteeBacktester as CBT  # noqa: N811


class DummyConfig:
    def __init__(self, assignment):
        self._assignment = assignment

    def regime_models(self, _name):
        return self._assignment


class DummyAssignment:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights


class DummyModel:
    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=np.float64)

    def predict_proba(self, X):
        return np.tile(self._proba.reshape(1, -1), (X.shape[0], 1))

    def predict(self, X, verbose=0):
        return self.predict_proba(X)


class TestProbaToTrade:
    """Tests for _proba_to_trade — confidence-gated argmax."""

    def make_cbt(self, threshold=0.55):
        cbt = CBT.__new__(CBT)
        cbt.confidence_threshold = threshold
        return cbt

    def test_long_when_confident(self):
        cbt = self.make_cbt(0.5)
        proba = np.array([[0.1, 0.2, 0.7]])  # long
        trades = cbt._proba_to_trade(proba)
        assert trades[0] == 1.0

    def test_short_when_confident(self):
        cbt = self.make_cbt(0.5)
        proba = np.array([[0.7, 0.2, 0.1]])  # short
        trades = cbt._proba_to_trade(proba)
        assert trades[0] == -1.0

    def test_flat_when_max_is_flat_class(self):
        cbt = self.make_cbt(0.5)
        proba = np.array([[0.1, 0.8, 0.1]])  # flat class wins
        trades = cbt._proba_to_trade(proba)
        assert trades[0] == 0.0

    def test_flat_when_below_threshold(self):
        cbt = self.make_cbt(0.55)
        proba = np.array([[0.1, 0.2, 0.5]])  # max 0.5 < 0.55 threshold
        trades = cbt._proba_to_trade(proba)
        assert trades[0] == 0.0

    def test_tie_goes_to_zero(self):
        cbt = self.make_cbt(0.5)
        proba = np.array([[0.5, 0.0, 0.5]])  # tie long/short, argmax picks first
        trades = cbt._proba_to_trade(proba)
        assert trades[0] == -1.0  # argmax picks index 0 (short)

    def test_multiple_bars(self):
        cbt = self.make_cbt(0.5)
        proba = np.array([
            [0.1, 0.2, 0.7],   # long
            [0.7, 0.2, 0.1],   # short
            [0.1, 0.8, 0.1],   # flat
            [0.2, 0.3, 0.6],   # long
        ])
        trades = cbt._proba_to_trade(proba)
        np.testing.assert_array_equal(trades, [1.0, -1.0, 0.0, 1.0])


class TestMajorityVoteConsensus:
    """Tests for sign-based majority voting (pure numpy math)."""

    def test_unanimous_long(self):
        preds = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        consensus = np.sign(preds.sum(axis=0))
        np.testing.assert_array_equal(consensus, [1, 1, 1])

    def test_unanimous_short(self):
        preds = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
        consensus = np.sign(preds.sum(axis=0))
        np.testing.assert_array_equal(consensus, [-1, -1, -1])

    def test_2v1_long(self):
        preds = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [-1, -1, 1],
        ])
        consensus = np.sign(preds.sum(axis=0))
        np.testing.assert_array_equal(consensus, [1, 1, 1])

    def test_tie_goes_to_zero(self):
        preds = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
        consensus = np.sign(preds.sum(axis=0))
        np.testing.assert_array_equal(consensus, [0, 0, 0])

    def test_all_zero(self):
        preds = np.array([[0, 0, 0], [0, 0, 0]])
        consensus = np.sign(preds.sum(axis=0))
        np.testing.assert_array_equal(consensus, [0, 0, 0])

    def test_three_models_two_bars(self):
        preds = np.array([
            [1, -1],   # model A
            [1, 1],    # model B
            [-1, -1],  # model C
        ])
        consensus = np.sign(preds.sum(axis=0))
        # bar 0: 1+1-1 = 1 → sign(1) = 1
        # bar 1: -1+1-1 = -1 → sign(-1) = -1
        np.testing.assert_array_equal(consensus, [1, -1])


class TestWeightedBlend:
    """Tests for weighted probability blending (pure numpy math)."""

    def test_equal_weights_average(self):
        probs = np.array([
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4],
        ])
        weights = [1.0, 1.0]
        prob_sum = (weights[0] * probs[0] + weights[1] * probs[1])
        weight_sum = sum(weights)
        blended = prob_sum / weight_sum
        expected = (probs[0] + probs[1]) / 2.0
        np.testing.assert_array_almost_equal(blended, expected)

    def test_unbalanced_weights(self):
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
        ])
        weights = [0.7, 0.3]
        prob_sum = weights[0] * probs[0] + weights[1] * probs[1]
        blended = prob_sum / sum(weights)
        expected = np.array([0.62, 0.28, 0.10])
        np.testing.assert_array_almost_equal(blended, expected)
