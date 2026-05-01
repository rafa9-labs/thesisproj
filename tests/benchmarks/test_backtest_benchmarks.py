"""Backtest duration and memory benchmarks.

Each classical model must train+predict within a reasonable time.
Deep model benchmarks are marked @pytest.mark.slow (excluded from CI).

Thresholds are generous to avoid flaky failures on CI runners.
"""
import time
import pytest
import numpy as np

os = __import__("os")

_CLASSICAL_TIME_LIMIT_S = 5.0
_DEEP_TIME_LIMIT_S = 60.0

slow = pytest.mark.slow


class TestClassicalModelBenchmarks:
    @pytest.fixture(autouse=True)
    def _setup(self, flat_synthetic):
        self.X, self.y = flat_synthetic

    def test_logistic_build_train_predict_under_limit(self):
        from models.registry import build_model
        m = build_model("logistic", seed=42)
        t0 = time.perf_counter()
        m.fit(self.X, self.y)
        preds = m.predict(self.X[:20])
        elapsed = time.perf_counter() - t0
        assert elapsed < _CLASSICAL_TIME_LIMIT_S, f"logistic took {elapsed:.2f}s > {_CLASSICAL_TIME_LIMIT_S}s"
        assert preds.shape == (20,)

    def test_svm_build_train_predict_under_limit(self):
        from models.registry import build_model
        m = build_model("svm", seed=42, svm_C=1.0)
        t0 = time.perf_counter()
        m.fit(self.X, self.y)
        preds = m.predict(self.X[:20])
        elapsed = time.perf_counter() - t0
        assert elapsed < _CLASSICAL_TIME_LIMIT_S, f"svm took {elapsed:.2f}s > {_CLASSICAL_TIME_LIMIT_S}s"
        assert preds.shape == (20,)

    def test_random_forest_build_train_predict_under_limit(self):
        from models.registry import build_model
        m = build_model("random_forest", seed=42, rf_n_estimators=50)
        t0 = time.perf_counter()
        m.fit(self.X, self.y)
        preds = m.predict(self.X[:20])
        elapsed = time.perf_counter() - t0
        assert elapsed < _CLASSICAL_TIME_LIMIT_S, f"rf took {elapsed:.2f}s > {_CLASSICAL_TIME_LIMIT_S}s"
        assert preds.shape == (20,)

    def test_decision_tree_build_train_predict_under_limit(self):
        from models.registry import build_model
        m = build_model("decision_tree", seed=42)
        t0 = time.perf_counter()
        m.fit(self.X, self.y)
        preds = m.predict(self.X[:20])
        elapsed = time.perf_counter() - t0
        assert elapsed < _CLASSICAL_TIME_LIMIT_S, f"dt took {elapsed:.2f}s > {_CLASSICAL_TIME_LIMIT_S}s"
        assert preds.shape == (20,)

    def test_xgboost_build_train_predict_under_limit(self):
        from models.registry import build_model
        m = build_model("xgboost", seed=42, xgb_n_estimators=50)
        t0 = time.perf_counter()
        m.fit(self.X, self.y)
        preds = m.predict(self.X[:20])
        elapsed = time.perf_counter() - t0
        assert elapsed < _CLASSICAL_TIME_LIMIT_S, f"xgb took {elapsed:.2f}s > {_CLASSICAL_TIME_LIMIT_S}s"
        assert preds.shape == (20,)


class TestClassicalMemoryBenchmarks:
    @pytest.fixture(autouse=True)
    def _setup(self, flat_synthetic):
        self.X, self.y = flat_synthetic

    def test_logistic_memory_reasonable(self):
        import tracemalloc
        from models.registry import build_model
        tracemalloc.start()
        m = build_model("logistic", seed=42)
        m.fit(self.X, self.y)
        _ = m.predict(self.X[:20])
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 100, f"logistic peak memory {peak_mb:.1f}MB > 100MB"

    def test_xgboost_memory_reasonable(self):
        import tracemalloc
        from models.registry import build_model
        tracemalloc.start()
        m = build_model("xgboost", seed=42, xgb_n_estimators=50)
        m.fit(self.X, self.y)
        _ = m.predict(self.X[:20])
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 100, f"xgb peak memory {peak_mb:.1f}MB > 100MB"


@slow
class TestDeepModelBenchmarks:
    @pytest.fixture(autouse=True)
    def _setup(self, seq_synthetic):
        self.X_seq, self.X_flat, self.y = seq_synthetic

    def test_cnn_build_train_predict_under_limit(self):
        from models.registry import build_model
        m = build_model("cnn", seed=42, input_shape=(self.X_seq.shape[1], self.X_seq.shape[2]),
                         cnn_filters1=16, cnn_epochs=3, cnn_use_early_stopping=False)
        t0 = time.perf_counter()
        m.fit(self.X_seq, self.y, epochs=3, batch_size=32, verbose=0)
        preds = m.predict(self.X_seq[:20], verbose=0)
        elapsed = time.perf_counter() - t0
        assert elapsed < _DEEP_TIME_LIMIT_S, f"cnn took {elapsed:.2f}s > {_DEEP_TIME_LIMIT_S}s"
        assert preds.shape == (20, 3)

    def test_lstm_build_train_predict_under_limit(self):
        from models.registry import build_model
        m = build_model("lstm", seed=42, input_shape=(self.X_seq.shape[1], self.X_seq.shape[2]),
                         lstm_units=32, lstm_num_layers=1, lstm_epochs=3,
                         lstm_use_early_stopping=False)
        t0 = time.perf_counter()
        m.fit(self.X_seq, self.y, epochs=3, batch_size=32, verbose=0)
        preds = m.predict(self.X_seq[:20], verbose=0)
        elapsed = time.perf_counter() - t0
        assert elapsed < _DEEP_TIME_LIMIT_S, f"lstm took {elapsed:.2f}s > {_DEEP_TIME_LIMIT_S}s"
        assert preds.shape == (20, 3)

    def test_transformer_build_train_predict_under_limit(self):
        from models.registry import build_model
        m = build_model("transformer", seed=42,
                         input_shape=(self.X_seq.shape[1], self.X_seq.shape[2]),
                         transformer_d_model=32, transformer_num_heads=2,
                         transformer_num_blocks=1, transformer_epochs=3,
                         transformer_use_early_stopping=False)
        t0 = time.perf_counter()
        m.fit(self.X_seq, self.y, epochs=3, batch_size=32, verbose=0)
        preds = m.predict(self.X_seq[:20], verbose=0)
        elapsed = time.perf_counter() - t0
        assert elapsed < _DEEP_TIME_LIMIT_S, f"transformer took {elapsed:.2f}s > {_DEEP_TIME_LIMIT_S}s"
        assert preds.shape == (20, 3)

    def test_ensemble_cnn_lstm_xgboost_under_limit(self):
        from models.ensemble_cnn_lstm_xgboost import EnsembleCNNLSTMXGBoost
        m = EnsembleCNNLSTMXGBoost(
            input_shape=(self.X_seq.shape[1], self.X_seq.shape[2]),
            cnn_config={"filters1": 16, "filters2": 32, "epochs": 2, "use_early_stopping": False},
            lstm_config={"units": 16, "epochs": 2, "use_early_stopping": False},
            xgb_config={"n_estimators": 10},
        )
        t0 = time.perf_counter()
        m.fit(self.X_seq, self.X_flat, self.y)
        preds = m.predict(self.X_seq[:20], self.X_flat[:20])
        elapsed = time.perf_counter() - t0
        assert elapsed < _DEEP_TIME_LIMIT_S, f"ensemble_cnn_lstm_xgboost took {elapsed:.2f}s > {_DEEP_TIME_LIMIT_S}s"
        assert preds.shape == (20,)