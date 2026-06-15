"""Tests for LTTB downsampling and multi-series alignment."""
from __future__ import annotations

import numpy as np
import pytest

from api.utils.downsampling import (
    SKIP_THRESHOLD,
    DEFAULT_TARGET,
    downsample_aligned,
    lttb_downsample,
)


def _make_curve(n: int, start: float = 100.0) -> list[dict]:
    """Create a wavy equity curve with n points."""
    rng = np.random.default_rng(42)
    values = [start]
    for _ in range(1, n):
        values.append(values[-1] + rng.normal(0, 1))
    return [{"time": i, "value": float(v)} for i, v in enumerate(values)]


def test_lttb_preserves_small_curve():
    curve = _make_curve(100)
    result = lttb_downsample(curve, DEFAULT_TARGET)
    assert result == curve


def test_lttb_reduces_large_curve():
    curve = _make_curve(5000)
    result = lttb_downsample(curve, DEFAULT_TARGET)
    assert len(result) == DEFAULT_TARGET
    assert result[0] == curve[0]
    assert result[-1] == curve[-1]


def test_lttb_handles_number_list():
    data = [10.0, 11.0, 9.0, 12.0, 8.0]
    result = lttb_downsample(data, 3)
    assert result == [{"time": 0, "value": 10.0}, {"time": 3, "value": 12.0}, {"time": 4, "value": 8.0}]


def test_lttb_handles_tuple_list():
    data = [(0, 10.0), (1, 11.0), (2, 9.0), (3, 12.0), (4, 8.0)]
    result = lttb_downsample(data, 3)
    assert result == [{"time": 0, "value": 10.0}, {"time": 3, "value": 12.0}, {"time": 4, "value": 8.0}]


def test_lttb_preserves_structural_peaks():
    """A synthetic V-shaped curve should retain the bottom point."""
    data = [{"time": i, "value": float(abs(i - 500))} for i in range(1001)]
    result = lttb_downsample(data, 10)
    times = [p["time"] for p in result]
    assert 500 in times


def test_downsample_aligned_skips_below_threshold():
    primary = _make_curve(1000)
    secondary = [{"time": p["time"], "value": p["value"] * 2} for p in primary]
    ec, sec = downsample_aligned(primary, secondary)
    assert len(ec) == 1000
    assert len(sec) == 1000


def test_downsample_aligned_synchronizes_timestamps():
    primary = _make_curve(5000)
    secondary = [{"time": p["time"], "value": p["value"] * 2} for p in primary]
    ec, sec = downsample_aligned(primary, secondary)
    assert len(ec) == DEFAULT_TARGET
    assert len(sec) == DEFAULT_TARGET
    ec_times = [p["time"] for p in ec]
    sec_times = [p["time"] for p in sec]
    assert ec_times == sec_times


def test_downsample_aligned_handles_none_secondary():
    primary = _make_curve(5000)
    ec, sec = downsample_aligned(primary, None)
    assert len(ec) == DEFAULT_TARGET
    assert sec == []


def test_skip_threshold_constant():
    assert SKIP_THRESHOLD == 1500
