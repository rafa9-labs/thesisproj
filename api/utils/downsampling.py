"""Largest Triangle Three Buckets (LTTB) downsampling utilities.

Used to shrink large equity/drawdown curves before JSON serialization so the
Electron frontend does not freeze rendering millions of invisible sub-pixel
points. Supports multi-series alignment: only the primary curve runs the LTTB
math; secondary curves are sampled at the exact timestamps retained by the
primary curve so all lines share a synchronized X-axis.
"""
from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SKIP_THRESHOLD = 1500
DEFAULT_TARGET = 1000


def _normalize_points(data: Sequence[Any] | None) -> List[Dict[str, Any]]:
    """Convert list-of-numbers, list-of-tuples, or list-of-dicts to [{time, value}]."""
    if not data:
        return []
    points: List[Dict[str, Any]] = []
    first = data[0]
    if isinstance(first, dict):
        for p in data:
            points.append({"time": p["time"], "value": float(p["value"])})
    elif isinstance(first, (list, tuple)):
        for t, v in data:
            points.append({"time": t, "value": float(v)})
    else:
        for i, v in enumerate(data):
            points.append({"time": i, "value": float(v)})
    return points


def lttb_downsample(data: Sequence[Any] | None, threshold: int = DEFAULT_TARGET) -> List[Dict[str, Any]]:
    """Downsample a time-series to ``threshold`` points using LTTB.

    The first and last points are always preserved. The algorithm divides the
    interior points into buckets and selects the point that maximizes the
    triangle area formed with the previously selected point and the average of
    the next bucket.
    """
    points = _normalize_points(data)
    n = len(points)
    if n <= threshold:
        return points

    x = np.array([p["time"] for p in points], dtype=np.float64)
    y = np.array([p["value"] for p in points], dtype=np.float64)

    sampled_idx = np.zeros(threshold, dtype=np.int64)
    sampled_idx[0] = 0
    sampled_idx[-1] = n - 1

    bucket_size = (n - 2) / (threshold - 2)
    a_x, a_y = x[0], y[0]

    for i in range(1, threshold - 1):
        start = int((i - 1) * bucket_size) + 1
        end = int(i * bucket_size) + 1
        if end > n:
            end = n

        next_start = end
        next_end = int((i + 1) * bucket_size) + 1
        if next_end > n:
            next_end = n
        if next_start >= n:
            next_start = n - 1

        avg_x = float(x[next_start:next_end].mean())
        avg_y = float(y[next_start:next_end].mean())

        bx = x[start:end]
        by = y[start:end]
        areas = np.abs((a_x - avg_x) * (by - a_y) - (a_x - bx) * (avg_y - a_y))
        local_idx = int(np.argmax(areas))
        idx = start + local_idx

        sampled_idx[i] = idx
        a_x, a_y = x[idx], y[idx]

    return [points[int(idx)] for idx in sampled_idx]


def _sample_by_timestamps(
    source_curve: Sequence[Any] | None,
    target_timestamps: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Return points from ``source_curve`` matching ``target_timestamps`` in order.

    If an exact timestamp is missing, the nearest neighbor in the source curve
    is returned so secondary series never drift from the primary X-axis.
    """
    points = _normalize_points(source_curve)
    if not points:
        return []

    times = [p["time"] for p in points]
    mapping = {p["time"]: p for p in points}

    result: List[Dict[str, Any]] = []
    for t in target_timestamps:
        if t in mapping:
            result.append(mapping[t])
            continue
        idx = bisect.bisect_left(times, t)
        if idx == 0:
            result.append(points[0])
        elif idx >= len(times):
            result.append(points[-1])
        else:
            before = points[idx - 1]
            after = points[idx]
            if abs(before["time"] - t) <= abs(after["time"] - t):
                result.append(before)
            else:
                result.append(after)
    return result


def downsample_aligned(
    primary_curve: Sequence[Any] | None,
    *secondary_curves: Sequence[Any] | None,
    threshold: int = DEFAULT_TARGET,
    skip_if: int = SKIP_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], ...]:
    """Downsample ``primary_curve`` with LTTB and sample secondary curves to match.

    Returns a tuple where every curve shares the exact same timestamps.
    Curves with ``len <= skip_if`` are returned unchanged (after normalization).
    """
    primary = _normalize_points(primary_curve)
    if len(primary) <= skip_if:
        return (primary, *[_normalize_points(s) for s in secondary_curves])

    retained = lttb_downsample(primary, threshold)
    retained_times = [p["time"] for p in retained]

    result: List[List[Dict[str, Any]]] = [retained]
    for curve in secondary_curves:
        result.append(_sample_by_timestamps(curve, retained_times))
    return tuple(result)
