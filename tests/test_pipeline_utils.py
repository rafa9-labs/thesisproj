"""Test 3: Standalone utility functions work correctly."""
import pytest
import numpy as np
import pandas as pd


def test_norm_class_counts_basic():
    """_norm_class_counts normalizes dict keys to plain ints."""
    from pipeline.standalone_utils import _norm_class_counts
    raw = {np.int64(-1): 10, np.int64(0): 20, np.int64(1): 30}
    out = _norm_class_counts(raw)
    assert out == {-1: 10, 0: 20, 1: 30}
    assert all(isinstance(k, int) for k in out.keys())


def test_norm_class_counts_empty():
    """_norm_class_counts returns empty dict for non-dict input."""
    from pipeline.standalone_utils import _norm_class_counts
    assert _norm_class_counts(None) == {}
    assert _norm_class_counts("bad") == {}
    assert _norm_class_counts([]) == {}


def test_norm_class_counts_skips_bad_keys():
    """_norm_class_counts skips keys that can't be converted to int."""
    from pipeline.standalone_utils import _norm_class_counts
    raw = {1: 5, "bad": 10, 3.0: 15}
    out = _norm_class_counts(raw)
    assert 1 in out
    assert 3 in out
    assert "bad" not in out


def test_load_csv_cached_returns_dataframe(tmp_path):
    """_load_csv_cached loads a CSV and returns a DataFrame."""
    from pipeline.standalone_utils import _load_csv_cached, _DATA_CACHE
    # Clear cache for this test
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    df.to_csv(csv_file, index=False)
    # Clear any cached entry
    keys_to_del = [k for k in _DATA_CACHE if str(k[0]).endswith("test_data.csv")]
    for k in keys_to_del:
        del _DATA_CACHE[k]
    result = _load_csv_cached(str(csv_file))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]


def test_load_csv_cached_caches_result(tmp_path):
    """_load_csv_cached returns a copy (not the same object)."""
    from pipeline.standalone_utils import _load_csv_cached, _DATA_CACHE
    csv_file = tmp_path / "test_cache.csv"
    pd.DataFrame({"x": [1]}).to_csv(csv_file, index=False)
    keys_to_del = [k for k in _DATA_CACHE if str(k[0]).endswith("test_cache.csv")]
    for k in keys_to_del:
        del _DATA_CACHE[k]
    r1 = _load_csv_cached(str(csv_file))
    r2 = _load_csv_cached(str(csv_file))
    assert r1 is not r2  # must be copies


def test_print_block_summary_runs(capfd):
    """print_block_summary must not crash with valid inputs."""
    from pipeline.standalone_utils import print_block_summary
    calib_info = {"target": 0.15, "conf_thr": 0.8, "bars_total": 1000, "bars_eligible": 900}
    gate_info = {"base": 0.5, "alpha": 0.01, "beta": 0.02, "gamma": 0.03, "median_thr": 0.75, "band": 0.08, "step": 0.005}
    reliability = {"psr_alpha": 0.05, "cutoff": 0.95, "min_trades": 10, "min_indep": 20}
    class_dists = {"raw": {-1: 5, 0: 80, 1: 15}, "final": {-1: 4, 0: 60, 1: 12}}
    block_stats = {"rows_total": 100, "rows_eligible": 90, "rows": 90, "trades": 20, "ar": 0.15, "sr": 1.5}
    # Should not raise
    print_block_summary(1, calib_info, gate_info, reliability, class_dists, block_stats)


def test_print_pruned_block_summary_runs():
    """print_pruned_block_summary must not crash."""
    from pipeline.standalone_utils import print_pruned_block_summary
    print_pruned_block_summary(1, "Low trades", rows=100, trades=2, active_rate=0.01, sharpe=-0.5)


def test_hard_free_does_not_crash():
    """_hard_free must not crash even without TF models."""
    from pipeline.memory_utils import _hard_free
    _hard_free()  # should succeed silently