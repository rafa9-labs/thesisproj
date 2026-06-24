"""Tests for ConvictionSizer — P3 sigmoid conviction sizing."""
import numpy as np
import pytest

from pipeline.execution.conviction_sizer import ConvictionSizer, logistic


class TestLogisticFunction:
    def test_midpoint_returns_half_L(self):
        result = logistic(np.array([0.65]), L=1.5, k=10.0, c=0.65)
        assert result[0] == pytest.approx(0.75, rel=0.01)  # L/2

    def test_high_input_approaches_L(self):
        result = logistic(np.array([0.95]), L=1.5, k=10.0, c=0.65)
        assert result[0] > 1.3

    def test_low_input_approaches_zero(self):
        result = logistic(np.array([0.35]), L=1.5, k=10.0, c=0.65)
        assert result[0] < 0.2

    def test_vectorized(self):
        x = np.array([0.5, 0.6, 0.7, 0.8])
        result = logistic(x, L=1.5, k=10.0, c=0.65)
        assert len(result) == 4
        assert (np.diff(result) > 0).all()  # monotonically increasing


class TestConvictionSizerDefaults:
    def test_get_multiplier_unfitted_uses_tiers(self):
        sizer = ConvictionSizer()
        assert not sizer.fitted
        assert sizer.get_multiplier(0.85) == 1.5
        assert sizer.get_multiplier(0.70) == 1.0
        assert sizer.get_multiplier(0.60) == 0.5
        assert sizer.get_multiplier(0.50) == 1.0  # below threshold, flat

    def test_default_params_stored(self):
        sizer = ConvictionSizer(L=1.8, k=15.0, c=0.70)
        assert sizer.L == 1.8
        assert sizer.k == 15.0
        assert sizer.c == 0.70


class TestConvictionSizerFit:
    def _make_trades(self, n_per_bucket=20):
        """Create synthetic trades where higher confidence → higher PnL."""
        bars = []
        rng = np.random.default_rng(42)
        for conf_bucket in np.linspace(0.5, 0.95, 10):
            for _ in range(n_per_bucket):
                conf = conf_bucket + rng.uniform(-0.03, 0.03)
                # Higher confidence → higher chance of winning
                win_prob = 0.3 + 0.6 * conf_bucket
                pnl = 0.0002 if rng.random() < win_prob else -0.0001
                bars.append({
                    "committee_signal": 1,
                    "committee_confidence": float(np.clip(conf, 0.0, 1.0)),
                    "next_return": pnl,
                })
        return bars

    def test_fit_on_synthetic_data(self):
        bars = self._make_trades(20)
        sizer = ConvictionSizer()
        sizer.fit(bars, n_bins=10)
        assert sizer.fitted
        assert sizer.n_trades >= 100
        assert 0.5 <= sizer.L <= 2.0
        assert 1.0 <= sizer.k <= 30.0
        assert 0.50 <= sizer.c <= 0.85

    def test_fit_too_few_trades_uses_defaults(self):
        bars = []
        for _ in range(10):
            bars.append({
                "committee_signal": 1,
                "committee_confidence": 0.6,
                "next_return": 0.001,
            })
        sizer = ConvictionSizer()
        sizer.fit(bars)
        assert not sizer.fitted  # too few trades

    def test_get_multiplier_monotonic_after_fit(self):
        bars = self._make_trades(30)
        sizer = ConvictionSizer()
        sizer.fit(bars)
        assert sizer.fitted
        mults = [sizer.get_multiplier(c) for c in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]]
        assert all(mults[i] <= mults[i + 1] for i in range(len(mults) - 1)), f"Not monotonic: {mults}"

    def test_fit_all_flat_pnl(self):
        bars = []
        for conf in np.linspace(0.5, 0.9, 10):
            for _ in range(10):
                bars.append({
                    "committee_signal": 1,
                    "committee_confidence": float(conf),
                    "next_return": 0.001,
                })
        sizer = ConvictionSizer()
        sizer.fit(bars)
        assert sizer.fitted
        assert sizer.L == 1.0  # flat PnL → flat multiplier
        assert sizer.k == 0.0

    def test_skips_flat_signals(self):
        bars = []
        for _ in range(100):
            bars.append({"committee_signal": 0, "committee_confidence": 0.6, "next_return": 0.001})
        sizer = ConvictionSizer()
        sizer.fit(bars)
        assert not sizer.fitted  # no trades to fit on


class TestConvictionSizerPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        bars = []
        for _ in range(100):
            bars.append({
                "committee_signal": 1,
                "committee_confidence": 0.6,
                "next_return": 0.001,
            })
        sizer = ConvictionSizer(L=1.8, k=12.0, c=0.68)
        sizer.fit(bars)

        path = str(tmp_path / "conviction_sizer.json")
        sizer.save(path)
        loaded = ConvictionSizer.load(path)
        assert loaded.fitted == sizer.fitted
        assert loaded.L == sizer.L
        assert loaded.k == sizer.k
        assert loaded.c == sizer.c
        assert loaded.n_trades == sizer.n_trades
