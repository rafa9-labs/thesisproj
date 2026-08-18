"""Tests for the corrected CSCV PBO (Bailey et al. 2016)."""
import numpy as np

from pipeline.metrics.pbo import compute_pbo, sharpe_ratio


class TestSharpeHelper:
    def test_basic(self):
        r = np.array([0.01, -0.005, 0.02, 0.0])
        s = sharpe_ratio(r)
        assert s == float(np.mean(r) / np.std(r, ddof=1))

    def test_constant_zero(self):
        assert sharpe_ratio(np.zeros(10)) == 0.0

    def test_short_series_nan(self):
        assert np.isnan(sharpe_ratio(np.array([0.01])))


class TestComputePBO:
    def test_shape_guards(self):
        assert np.isnan(compute_pbo(np.zeros((1, 100))))
        assert np.isnan(compute_pbo(np.zeros((3, 3))))
        assert np.isnan(compute_pbo(np.zeros(10)))
        m = np.zeros((4, 50))
        m[0, 0] = np.nan
        assert np.isnan(compute_pbo(m))

    def test_truly_best_low_pbo(self):
        """A config that dominates in-sample and out-of-sample -> low PBO."""
        rng = np.random.default_rng(7)
        M = rng.normal(0, 0.01, size=(6, 500))
        M[0] += 0.003
        pbo = compute_pbo(M, S=16)
        assert pbo <= 0.1

    def test_is_best_is_oos_worst_high_pbo(self):
        """Config wins in-sample but crashes out-of-sample -> high PBO."""
        rng = np.random.default_rng(11)
        T = 64
        M = rng.normal(0, 0.01, size=(6, T)) + 0.001
        M[0, : T // 2] += 0.02
        M[0, T // 2 :] -= 0.02
        pbo = compute_pbo(M, S=16)
        assert pbo >= 0.5

    def test_noise_not_more_robust_than_genuine(self):
        """PBO must rank pure noise worse than a genuinely good config."""
        rng = np.random.default_rng(0)
        M_noise = rng.normal(0, 0.01, size=(6, 500)) + 0.0005
        pbo_noise = compute_pbo(M_noise, S=16)

        rng = np.random.default_rng(0)
        M_best = rng.normal(0, 0.01, size=(6, 500))
        M_best[0] += 0.003
        pbo_best = compute_pbo(M_best, S=16)

        assert pbo_noise > pbo_best

    def test_deterministic(self):
        rng = np.random.default_rng(5)
        M = rng.normal(0, 0.01, size=(4, 60))
        a = compute_pbo(M, S=8)
        b = compute_pbo(M, S=8)
        assert a == b

    def test_more_configs_no_crash(self):
        rng = np.random.default_rng(9)
        M = rng.normal(0, 0.01, size=(20, 40))
        pbo = compute_pbo(M, S=8)
        assert 0.0 <= pbo <= 1.0

    def test_subset_sampling_path(self):
        """Large S exercises the max_combos deterministic subsampling branch."""
        rng = np.random.default_rng(13)
        M = rng.normal(0, 0.01, size=(4, 60))
        pbo = compute_pbo(M, S=16, max_combos=50)
        assert 0.0 <= pbo <= 1.0
