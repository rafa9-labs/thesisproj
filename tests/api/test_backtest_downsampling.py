"""API-level tests for backtest result downsampling."""
from __future__ import annotations

import json
import sys
import uuid

import pytest

sys.path.insert(0, "..")

from starlette.testclient import TestClient

from api.main import app
from api.services import JobManager
from pipeline.data_sqlite import DataStore


@pytest.fixture
def client():
    return TestClient(app)


class TestBacktestResultsDownsampling:
    def setup_method(self):
        self.store = DataStore()
        self.jm = JobManager(self.store)
        for job in self.jm.list_jobs(job_type="backtest", limit=1000):
            self.jm.delete_job(job["id"])

    def _create_job_with_curve(self, n: int):
        job_id = str(uuid.uuid4())
        curve = [{"time": i, "value": 10000.0 + i} for i in range(n)]
        result = {
            "pair": "EURUSD",
            "models": ["logistic"],
            "metrics": [
                {
                    "model": "logistic",
                    "sharpe": 1.0,
                    "equity_curve": curve,
                    "buy_hold_curve": curve,
                    "drawdown_curve": [{"time": p["time"], "value": -(p["value"] % 100)} for p in curve],
                }
            ],
        }
        self.jm.create_job(job_id, "backtest", {"pair": "EURUSD", "models": ["logistic"]})
        self.jm.update_status(job_id, "completed", result=result)
        return job_id

    def test_large_curves_are_downsampled(self, client):
        job_id = self._create_job_with_curve(5000)
        r = client.get(f"/api/v1/backtest/{job_id}/results")
        assert r.status_code == 200
        data = r.json()
        metric = data["metrics"][0]
        assert len(metric["equity_curve"]) == 1000
        assert len(metric["buy_hold_curve"]) == 1000
        assert len(metric["drawdown_curve"]) == 1000
        ec_times = [p["time"] for p in metric["equity_curve"]]
        bhc_times = [p["time"] for p in metric["buy_hold_curve"]]
        ddc_times = [p["time"] for p in metric["drawdown_curve"]]
        assert ec_times == bhc_times == ddc_times

    def test_small_curves_are_unchanged(self, client):
        job_id = self._create_job_with_curve(1000)
        r = client.get(f"/api/v1/backtest/{job_id}/results")
        assert r.status_code == 200
        data = r.json()
        metric = data["metrics"][0]
        assert len(metric["equity_curve"]) == 1000
        assert metric["equity_curve"][0] == {"time": 0, "value": 10000.0}
