"""Tests for the corrected Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)."""
import math

import numpy as np
from scipy.stats import norm

from pipeline.metrics.dsr import deflated_sharpe_ratio, expected_max_sharpe


class TestExpectedMaxSharpe:
    def test_single_trial_has_no_selection_bias(self):
        assert expected_max_sharpe(1) == 0.0
        assert expected_max_sharpe(0) == 0.0
        assert expected_max_sharpe(-5) == 0.0

    def test_grows_with_trials(self):
        assert expected_max_sharpe(10) < expected_max_sharpe(100) < expected_max_sharpe(1000)

    def test_positive_for_multiple_trials(self):
        assert expected_max_sharpe(2) > 0.0


class TestDeflatedSharpeRatio:
    def test_guards(self):
        assert deflated_sharpe_ratio(float("nan"), 100, 10) == 0.0
        assert deflated_sharpe_ratio(1.0, 1, 10) == 0.0
        assert deflated_sharpe_ratio(0.5, 100, 10, sr_star=1.0) == 0.0

    def test_sample_length_matters(self):
        """Regression test: T must deflate the DSR (old impl cancelled T out)."""
        d_short = deflated_sharpe_ratio(1.5, T=50, N_trials=1)
        d_long = deflated_sharpe_ratio(1.5, T=5000, N_trials=1)
        assert d_long > d_short

    def test_more_trials_deflates(self):
        d_1 = deflated_sharpe_ratio(1.0, T=1000, N_trials=1)
        d_10 = deflated_sharpe_ratio(1.0, T=1000, N_trials=10)
        d_100 = deflated_sharpe_ratio(1.0, T=1000, N_trials=100)
        assert d_1 >= d_10 >= d_100

    def test_fat_tails_deflate(self):
        base = deflated_sharpe_ratio(0.8, T=100, N_trials=10, skew=0.0, kurt=3.0)
        fat = deflated_sharpe_ratio(0.8, T=100, N_trials=10, skew=0.0, kurt=10.0)
        assert fat < base

    def test_negative_skew_deflates(self):
        base = deflated_sharpe_ratio(0.8, T=100, N_trials=10, skew=0.0, kurt=3.0)
        neg = deflated_sharpe_ratio(0.8, T=100, N_trials=10, skew=-1.0, kurt=3.0)
        assert neg < base

    def test_periods_per_year_invariance(self):
        """Annualized SR with ppy must equal per-period SR without ppy."""
        sr_ann = 1.2
        T = 6048
        a = deflated_sharpe_ratio(sr_ann, T=T, N_trials=5, periods_per_year=6048)
        b = deflated_sharpe_ratio(sr_ann / math.sqrt(6048), T=T, N_trials=5)
        assert abs(a - b) < 1e-9

    def test_matches_paper_formula(self):
        """Closed-form check against the paper's equations."""
        sr_hat, T, N = 1.5, 1000, 10
        skew, kurt, sr_star = -0.5, 6.0, 0.0
        got = deflated_sharpe_ratio(sr_hat, T=T, N_trials=N,
                                    skew=skew, kurt=kurt, sr_star=sr_star)
        e_max = expected_max_sharpe(N)
        var_sr = (1.0 - skew * sr_hat + (kurt - 1.0) * sr_hat ** 2 / 4.0) / (T - 1)
        sigma = math.sqrt(var_sr)
        z = (sr_hat - sr_star - e_max * sigma) / sigma
        expected = float(norm.cdf(z))
        assert abs(got - expected) < 1e-9

    def test_high_sr_long_sample_single_trial_keeps(self):
        d = deflated_sharpe_ratio(1.5, T=10000, N_trials=1)
        assert d > 0.95

    def test_none_handling(self):
        assert deflated_sharpe_ratio(None, 100, 10) == 0.0
        assert deflated_sharpe_ratio("abc", 100, 10) == 0.0
