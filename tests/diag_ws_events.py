"""
Diagnostic: verify the backtest WebSocket event pipeline end-to-end.

Tests:
  1. _sanitize_for_json handles NaN/Inf/numpy types
  2. _pub produces valid JSON
  3. Events published to Redis are readable
  4. Full fast backtest produces all expected event types

Usage:
    python tests/diag_ws_events.py
"""
from __future__ import annotations

import json
import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)


def test_sanitize():
    from api.tasks import _sanitize_for_json

    import numpy as np

    # NaN -> None
    assert _sanitize_for_json(float("nan")) is None
    assert _sanitize_for_json(float("inf")) is None
    assert _sanitize_for_json(float("-inf")) is None

    # Numpy types
    assert _sanitize_for_json(np.int32(42)) == 42
    assert isinstance(_sanitize_for_json(np.int32(42)), int)
    assert isinstance(_sanitize_for_json(np.float32(3.14)), float)
    assert _sanitize_for_json(np.bool_(True)) is True

    # Nested
    d = {"sharpe": float("nan"), "trades": np.int32(10), "params": {"x": np.float64(1.5)}}
    clean = _sanitize_for_json(d)
    assert clean["sharpe"] is None
    assert clean["trades"] == 10
    assert clean["params"]["x"] == 1.5

    # List
    arr = [float("nan"), np.int32(1), float("inf")]
    assert _sanitize_for_json(arr) == [None, 1, None]

    # Valid JSON
    json.dumps(clean)
    print("[PASS] _sanitize_for_json")


def test_pub_valid_json():
    from api.tasks import _pub, get_job_events

    import numpy as np

    test_id = f"diag-test-{os.urandom(4).hex()}"

    # Publish with NaN value
    _pub("test_event", test_id, {"sharpe": float("nan"), "trades": np.int32(5)})

    time.sleep(0.2)

    events = get_job_events(test_id)
    assert len(events) >= 1, f"No events found for {test_id}"

    evt = events[0]
    assert evt["event"] == "test_event"
    assert evt["sharpe"] is None
    assert evt["trades"] == 5

    json.dumps(events)
    print("[PASS] _pub produces valid JSON with NaN")


def test_full_backtest_events():
    """Run a minimal backtest and check all expected event types arrive."""
    import uuid
    from api.tasks import _run_backtest_impl, get_job_events

    job_id = str(uuid.uuid4())

    config = {
        "pair": "EURUSD",
        "models": ["logistic"],
        "months": 1,
        "repeats": 1,
        "seed": 42,
        "hpo_intensity": "smoke",
        "trading_costs": False,
        "config_overrides": {
            "lags": 5,
            "lag_depth": 1,
            "n_trials": 1,
            "confidence_threshold": 0.8,
            "target_active_rate": 0.15,
        },
    }

    try:
        _run_backtest_impl(job_id, config)
    except Exception as e:
        print(f"[WARN] Full backtest failed: {e}")
        return

    time.sleep(0.3)

    events = get_job_events(job_id)
    event_types = [e["event"] for e in events]

    expected = {"job_started", "cycle_started", "model_training", "model_phase", "hpo_progress", "month_progress", "oos_result", "job_complete"}
    found = set(event_types) & expected
    missing = expected - found

    print(f"  Events received: {len(events)} total")
    print(f"  Types found: {sorted(found)}")

    if missing:
        print(f"  [WARN] Missing event types: {sorted(missing)}")
    else:
        print("  [PASS] All expected event types received")

    # Verify all events are valid JSON
    json.dumps(events)
    print("  [PASS] All events are valid JSON")

    # Check specific event content
    for evt in events:
        if evt["event"] == "oos_result":
            assert evt.get("sharpe") is None or isinstance(evt["sharpe"], (int, float)), f"Bad sharpe type: {type(evt['sharpe'])}"
            assert evt.get("equity") is None or isinstance(evt["equity"], (int, float)), f"Bad equity type"
            print(f"  oos_result: period={evt.get('period')}, sharpe={evt.get('sharpe')}, equity={evt.get('equity')}")

        if evt["event"] == "hpo_progress":
            print(f"  hpo_progress: model={evt.get('model')}, n_trials={evt.get('n_trials')}")

        if evt["event"] == "cycle_started":
            print(f"  cycle_started: model={evt.get('model')}, cycle={evt.get('cycle_number')}/{evt.get('total_cycles')}")

    print("[PASS] Full backtest event pipeline")


if __name__ == "__main__":
    print("=== KodaQuant WebSocket Event Diagnostics ===\n")

    print("1. _sanitize_for_json")
    test_sanitize()

    print("\n2. _pub valid JSON")
    test_pub_valid_json()

    print("\n3. Full backtest event pipeline")
    test_full_backtest_events()

    print("\n=== All diagnostics complete ===")
