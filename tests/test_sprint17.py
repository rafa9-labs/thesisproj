"""Sprint 17 tests: Real-time backtest monitoring (Phase 1 — backend enrichment)."""
import json
import threading
import time

import pytest


class TestJobEventsBuffer:
    def test_append_and_retrieve(self):
        from api.tasks import _job_events, _job_events_lock, _append_event, get_job_events, clear_job_events

        clear_job_events("test-1")
        _append_event("test-1", {"event": "job_started", "job_id": "test-1", "total_work": 100})
        _append_event("test-1", {"event": "hpo_progress", "job_id": "test-1", "trial": 1})

        events = get_job_events("test-1")
        assert len(events) == 2
        assert events[0]["event"] == "job_started"
        assert events[1]["event"] == "hpo_progress"

        clear_job_events("test-1")

    def test_get_with_after_cursor(self):
        from api.tasks import _append_event, get_job_events, clear_job_events

        clear_job_events("test-2")
        for i in range(5):
            _append_event("test-2", {"event": "tick", "i": i})

        events_after_2 = get_job_events("test-2", after=2)
        assert len(events_after_2) == 3
        assert events_after_2[0]["i"] == 2
        assert events_after_2[2]["i"] == 4

        clear_job_events("test-2")

    def test_max_cap(self):
        from api.tasks import _job_events, _JOB_EVENTS_MAX, _append_event, get_job_events, clear_job_events

        clear_job_events("test-3")
        for i in range(_JOB_EVENTS_MAX + 100):
            _append_event("test-3", {"event": "tick", "i": i})

        events = get_job_events("test-3")
        assert len(events) == _JOB_EVENTS_MAX
        assert events[0]["i"] == 100
        assert events[-1]["i"] == _JOB_EVENTS_MAX + 99

        clear_job_events("test-3")

    def test_thread_safety(self):
        from api.tasks import _append_event, get_job_events, clear_job_events

        clear_job_events("test-4")
        errors = []

        def writer(start):
            try:
                for i in range(200):
                    _append_event("test-4", {"event": "tick", "i": start + i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 200,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        events = get_job_events("test-4")
        assert len(events) == 1000

        clear_job_events("test-4")

    def test_clear_nonexistent(self):
        from api.tasks import clear_job_events
        clear_job_events("nonexistent-id")


class TestProgressCbEnrichment:
    def test_hpo_trial_result_event(self):
        from api.tasks import get_job_events, clear_job_events, _pub

        clear_job_events("hpo-test-1")
        _pub("hpo_trial_result", "hpo-test-1", {
            "model": "xgboost",
            "trial_number": 5,
            "score": 1.23,
            "params": {"lr": 0.01, "max_depth": 6},
            "best_score_so_far": 1.45,
            "trial_state": "COMPLETE",
        })

        events = get_job_events("hpo-test-1")
        assert len(events) == 1
        evt = events[0]
        assert evt["event"] == "hpo_trial_result"
        assert evt["model"] == "xgboost"
        assert evt["trial_number"] == 5
        assert evt["score"] == 1.23
        assert evt["params"] == {"lr": 0.01, "max_depth": 6}
        assert evt["best_score_so_far"] == 1.45

        clear_job_events("hpo-test-1")

    def test_oos_result_event(self):
        from api.tasks import get_job_events, clear_job_events, _pub

        clear_job_events("oos-test-1")
        _pub("oos_result", "oos-test-1", {
            "model": "xgboost",
            "period": 3,
            "total_periods": 12,
            "equity": 1.05,
            "equity_bh": 1.02,
            "sharpe": 1.23,
            "return_pct": 5.0,
            "trades": 15,
            "drawdown": -0.03,
            "win_rate": 0.65,
        })

        events = get_job_events("oos-test-1")
        assert len(events) == 1
        evt = events[0]
        assert evt["event"] == "oos_result"
        assert evt["equity"] == 1.05
        assert evt["equity_bh"] == 1.02
        assert evt["sharpe"] == 1.23
        assert evt["drawdown"] == -0.03

        clear_job_events("oos-test-1")

    def test_pub_also_appends_to_buffer(self):
        from api.tasks import get_job_events, clear_job_events, _pub

        clear_job_events("pub-test-1")
        _pub("job_started", "pub-test-1", {"pair": "EURUSD", "models": ["xgboost"]})
        _pub("month_progress", "pub-test-1", {"model": "xgboost", "period": 1, "sharpe": 1.5})

        events = get_job_events("pub-test-1")
        assert len(events) == 2
        assert events[0]["event"] == "job_started"
        assert events[1]["event"] == "month_progress"

        clear_job_events("pub-test-1")


class TestEventsEndpoint:
    def test_events_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        from api.tasks import _append_event, clear_job_events

        clear_job_events("ep-test-1")
        _append_event("ep-test-1", {"event": "job_started", "job_id": "ep-test-1"})
        _append_event("ep-test-1", {"event": "hpo_progress", "trial": 1})

        client = TestClient(app)
        resp = client.get("/api/v1/backtest/ep-test-1/events")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) == 2
        assert data["total"] == 2

        clear_job_events("ep-test-1")

    def test_events_with_after_param(self):
        from fastapi.testclient import TestClient
        from api.main import app
        from api.tasks import _append_event, clear_job_events

        clear_job_events("ep-test-2")
        for i in range(5):
            _append_event("ep-test-2", {"event": "tick", "i": i})

        client = TestClient(app)
        resp = client.get("/api/v1/backtest/ep-test-2/events?after=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) == 2
        assert data["events"][0]["i"] == 3
        assert data["total"] == 5

        clear_job_events("ep-test-2")