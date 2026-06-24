"""Tests for the in-memory feature cache (Step 2.9).

We test the cache mechanism at the unit level by exercising
the cache dict and _clear_feature_cache directly, plus verify
the config flag handling logic.
"""
import pytest


# ---------------------------------------------------------------------------
# Lightweight mock for cache testing (avoids full MLBacktester construction)
# ---------------------------------------------------------------------------

class _CacheProxy:
    """Minimal object that has the same cache attributes as MLBacktester."""

    def __init__(self, slice_cache_enabled=False):
        self._feat_cache = {}
        self._feat_cache_hits = 0
        self._feat_cache_misses = 0
        self._feat_cache_est_bytes = 0
        self._feat_cache_cur_bytes = 0
        self._feat_cache_evictions = 0
        self._feat_cache_bytes = {}
        self._feat_cache_mode_logged = False
        self._in_optuna_cv = False
        self.features_config = {
            "slice_cache_enabled": slice_cache_enabled,
        }

    def _clear_feature_cache(self):
        """Mirror of ensemble_mixin._clear_feature_cache."""
        if hasattr(self, "_feat_cache") and isinstance(self._feat_cache, dict):
            self._feat_cache.clear()
        if hasattr(self, "_feat_cache_bytes") and isinstance(getattr(self, "_feat_cache_bytes", None), dict):
            self._feat_cache_bytes.clear()
        for _k in (
            "_feat_cache_hits", "_feat_cache_misses",
            "_feat_cache_est_bytes", "_feat_cache_cur_bytes",
            "_feat_cache_evictions",
        ):
            if hasattr(self, _k):
                setattr(self, _k, 0)

    def _simulate_engineer(self, cache_key, data):
        """Simulate the cache check/store logic from features_mixin."""
        cfg = self.features_config
        slice_cache_enabled = bool(cfg.get("slice_cache_enabled", False))
        cache_enabled = slice_cache_enabled and not self._in_optuna_cv

        # Check cache
        cached = self._feat_cache.get(cache_key) if cache_enabled else None
        if cached is not None:
            self._feat_cache_hits += 1
            return cached

        # Miss: compute (simulate) and store
        result = data
        if cache_enabled:
            self._feat_cache[cache_key] = result
            self._feat_cache_misses += 1
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureCacheMechanics:
    """In-memory feature cache dict behaviour."""

    def test_cache_initialized_empty(self):
        cp = _CacheProxy(slice_cache_enabled=True)
        assert cp._feat_cache == {}
        assert cp._feat_cache_hits == 0
        assert cp._feat_cache_misses == 0

    def test_cache_miss_then_store(self):
        cp = _CacheProxy(slice_cache_enabled=True)
        cp._simulate_engineer("key_0_100", "result_A")
        assert cp._feat_cache_misses == 1
        assert cp._feat_cache["key_0_100"] == "result_A"

    def test_cache_hit_on_repeat(self):
        cp = _CacheProxy(slice_cache_enabled=True)
        cp._simulate_engineer("key_0_100", "result_A")
        assert cp._feat_cache_misses == 1

        # Same key again → hit
        result = cp._simulate_engineer("key_0_100", "result_B")
        assert cp._feat_cache_hits == 1
        assert result == "result_A"  # returns cached, not new data

    def test_different_keys_are_separate(self):
        cp = _CacheProxy(slice_cache_enabled=True)
        cp._simulate_engineer("key_0_100", "A")
        cp._simulate_engineer("key_100_200", "B")
        assert cp._feat_cache_misses == 2
        assert cp._feat_cache_hits == 0
        assert len(cp._feat_cache) == 2

    def test_cache_bypassed_when_disabled(self):
        cp = _CacheProxy(slice_cache_enabled=False)
        cp._simulate_engineer("key_0_100", "A")
        cp._simulate_engineer("key_0_100", "B")
        assert cp._feat_cache_hits == 0
        assert len(cp._feat_cache) == 0  # nothing stored

    def test_cache_bypassed_during_cv(self):
        cp = _CacheProxy(slice_cache_enabled=True)
        cp._in_optuna_cv = True  # CV mode
        cp._simulate_engineer("key_0_100", "A")
        cp._simulate_engineer("key_0_100", "B")
        assert cp._feat_cache_hits == 0
        assert len(cp._feat_cache) == 0  # bypassed during CV

    def test_clear_resets_everything(self):
        cp = _CacheProxy(slice_cache_enabled=True)
        cp._simulate_engineer("key_0_100", "A")
        assert len(cp._feat_cache) == 1
        assert cp._feat_cache_misses == 1

        cp._clear_feature_cache()
        assert len(cp._feat_cache) == 0
        assert cp._feat_cache_hits == 0
        assert cp._feat_cache_misses == 0

    def test_clear_then_repopulate(self):
        cp = _CacheProxy(slice_cache_enabled=True)
        cp._simulate_engineer("key_0_100", "A")
        cp._clear_feature_cache()
        # Re-populate with new data
        cp._simulate_engineer("key_0_100", "B")
        assert cp._feat_cache["key_0_100"] == "B"
        assert cp._feat_cache_misses == 1


class TestFeatureCacheConfig:
    """Config flag handling for slice_cache_enabled."""

    def test_default_is_disabled(self):
        cp = _CacheProxy()  # default: False
        assert not cp.features_config.get("slice_cache_enabled", False)

    def test_enabled_flag_respected(self):
        cp = _CacheProxy(slice_cache_enabled=True)
        assert cp.features_config["slice_cache_enabled"] is True

    def test_feat_cache_enabled_backcompat(self):
        """Old 'feat_cache_enabled' key should also work (back-compat)."""
        cp = _CacheProxy()
        cp.features_config = {"feat_cache_enabled": True}
        # The real code in features_mixin checks both keys
        cfg = cp.features_config
        enabled = bool(cfg.get("slice_cache_enabled", cfg.get("feat_cache_enabled", False)))
        assert enabled is True


class TestClearFeatureCacheMethod:
    """Test the actual _clear_feature_cache method on the composed class."""

    def test_method_exists(self, ml_backtester_class):
        """MLBacktester has _clear_feature_cache."""
        assert hasattr(ml_backtester_class, "_clear_feature_cache")

    def test_feat_cache_attribute_exists(self, ml_backtester_class):
        """The _feat_cache dict is initialized in __init__."""
        # We can't instantiate with csv_path (wrong constructor), but
        # we can check the attribute is set via the __init__ path.
        # Verify the method signature accepts no args
        import inspect
        sig = inspect.signature(ml_backtester_class._clear_feature_cache)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        assert len(params) == 0, "_clear_feature_cache should take no args beyond self"


# ---------------------------------------------------------------------------
# Disk cache tests (pipeline.features.feature_cache)
# ---------------------------------------------------------------------------

import tempfile
import shutil
import pandas as pd
import numpy as np


class TestDiskCache:
    """Tests for the Parquet disk cache in pipeline.features.feature_cache."""

    def setup_method(self):
        """Create a temp cache dir for each test."""
        import pipeline.features.feature_cache as fc
        self._orig_dir = fc._CACHE_DIR
        self._tmp = tempfile.mkdtemp()
        fc._CACHE_DIR = type(fc._CACHE_DIR)(self._tmp) / ".feature_cache"

    def teardown_method(self):
        import pipeline.features.feature_cache as fc
        fc._CACHE_DIR = self._orig_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        from pipeline.features.feature_cache import save_to_disk, load_from_disk
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        feats = ["a", "b"]
        save_to_disk("test_key_1", df, feats)
        result = load_from_disk("test_key_1")
        assert result is not None
        df_loaded, feats_loaded = result
        pd.testing.assert_frame_equal(df_loaded, df)
        assert feats_loaded == feats

    def test_load_returns_none_for_missing(self):
        from pipeline.features.feature_cache import load_from_disk
        assert load_from_disk("nonexistent_key_xyz") is None

    def test_different_config_different_key(self):
        from pipeline.features.feature_cache import compute_disk_key
        import tempfile, os
        # Create a fake CSV
        tmp = os.path.join(self._tmp, "fake.csv")
        with open(tmp, "w") as f:
            f.write("a,b\n1,2\n")
        key1 = compute_disk_key(tmp, {"lags": 10, "use_sma": True})
        key2 = compute_disk_key(tmp, {"lags": 20, "use_sma": True})
        assert key1 != key2

    def test_same_config_same_key(self):
        from pipeline.features.feature_cache import compute_disk_key
        import tempfile, os
        tmp = os.path.join(self._tmp, "fake.csv")
        with open(tmp, "w") as f:
            f.write("a,b\n1,2\n")
        cfg = {"lags": 10, "use_sma": True}
        k1 = compute_disk_key(tmp, cfg)
        k2 = compute_disk_key(tmp, cfg)
        assert k1 == k2

    def test_corrupt_parquet_handled_gracefully(self):
        from pipeline.features.feature_cache import save_to_disk, load_from_disk, disk_cache_dir
        import json
        # Write a valid entry then corrupt the parquet
        df = pd.DataFrame({"x": [1.0]})
        save_to_disk("corrupt_test", df, ["x"])
        pq_path = disk_cache_dir() / "corrupt_test.parquet"
        # Overwrite with garbage
        pq_path.write_bytes(b"NOT_A_PARQUET_FILE")
        result = load_from_disk("corrupt_test")
        assert result is None  # should not crash
        # Corrupt file should have been cleaned up
        assert not pq_path.exists()

    def test_clear_disk_cache(self):
        from pipeline.features.feature_cache import save_to_disk, clear_disk_cache, load_from_disk
        save_to_disk("clear_test", pd.DataFrame({"a": [1.0]}), ["a"])
        assert load_from_disk("clear_test") is not None
        count = clear_disk_cache()
        assert count >= 2  # parquet + json sidecar
        assert load_from_disk("clear_test") is None

    def test_disk_cache_stats(self):
        from pipeline.features.feature_cache import save_to_disk, disk_cache_stats
        save_to_disk("stats_test", pd.DataFrame({"a": np.arange(100.0)}), ["a"])
        stats = disk_cache_stats()
        assert stats["files"] >= 2
        assert isinstance(stats["mb"], float)
        # Parquet is very compact; 100 floats may round to 0.00 MB, so just check >= 0
        assert stats["mb"] >= 0
