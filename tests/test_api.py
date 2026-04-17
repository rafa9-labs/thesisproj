"""Tests for the FastAPI backend."""
import os
import sys
import time
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.fixture
def client():
    from starlette.testclient import TestClient
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"]


class TestPairs:
    def test_list_pairs(self, client):
        r = client.get("/api/v1/pairs")
        assert r.status_code == 200
        data = r.json()
        assert len(data["pairs"]) == 6
        symbols = [p["pair"]["symbol"] for p in data["pairs"]]
        assert "EURUSD" in symbols
        assert "USDJPY" in symbols

    def test_pair_has_pip_value(self, client):
        r = client.get("/api/v1/pairs")
        pairs = r.json()["pairs"]
        usdjpy = next(p for p in pairs if p["pair"]["symbol"] == "USDJPY")
        assert usdjpy["pair"]["pip_value"] == 0.01

    def test_pair_has_timeframes(self, client):
        r = client.get("/api/v1/pairs")
        pairs = r.json()["pairs"]
        eurusd = next(p for p in pairs if p["pair"]["symbol"] == "EURUSD")
        tfs = [t["timeframe"] for t in eurusd["timeframes"]]
        assert "M30" in tfs
        assert "H1" in tfs
        assert "H4" in tfs

    def test_data_range(self, client):
        r = client.get("/api/v1/pairs/EURUSD/data-range")
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "EURUSD"
        assert len(data["timeframes"]) == 3
        m30 = next(t for t in data["timeframes"] if t["timeframe"] == "M30")
        assert m30["rows"] > 100000

    def test_unknown_pair_404(self, client):
        r = client.get("/api/v1/pairs/ZZZZZZ/data-range")
        assert r.status_code == 404


class TestModels:
    def test_list_models(self, client):
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        data = r.json()
        names = [m["name"] for m in data["models"]]
        assert "logistic" in names
        assert "xgboost" in names
        assert "cnn" in names
        assert "lstm" in names

    def test_model_has_category(self, client):
        r = client.get("/api/v1/models")
        models = r.json()["models"]
        cnn = next(m for m in models if m["name"] == "cnn")
        assert cnn["category"] == "deep"


class TestBacktestSubmit:
    def test_submit_valid_backtest(self, client):
        r = client.post("/api/v1/backtest", json={
            "pair": "EURUSD",
            "models": ["logistic"],
            "start_date": "2024-06-01",
            "end_date": "2024-08-01",
            "months": 1,
        })
        assert r.status_code == 202
        data = r.json()
        assert data["job_id"]
        assert data["status"] == "pending"
        assert data["pair"] == "EURUSD"

    def test_submit_unknown_pair_400(self, client):
        r = client.post("/api/v1/backtest", json={
            "pair": "ZZZZZZ",
            "models": ["logistic"],
        })
        assert r.status_code == 400

    def test_submit_unknown_model_400(self, client):
        r = client.post("/api/v1/backtest", json={
            "pair": "EURUSD",
            "models": ["nonexistent_model"],
        })
        assert r.status_code == 400

    def test_list_backtests(self, client):
        r = client.get("/api/v1/backtest")
        assert r.status_code == 200
        assert "jobs" in r.json()

    def test_get_backtest_status(self, client):
        r = client.post("/api/v1/backtest", json={
            "pair": "EURUSD",
            "models": ["logistic"],
            "months": 1,
        })
        job_id = r.json()["job_id"]

        r2 = client.get(f"/api/v1/backtest/{job_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("pending", "running", "completed", "failed")

    def test_get_nonexistent_job_404(self, client):
        r = client.get("/api/v1/backtest/does-not-exist")
        assert r.status_code == 404

    def test_delete_job(self, client):
        r = client.post("/api/v1/backtest", json={
            "pair": "EURUSD",
            "models": ["logistic"],
        })
        job_id = r.json()["job_id"]
        r2 = client.delete(f"/api/v1/backtest/{job_id}")
        assert r2.status_code == 204

    def test_delete_nonexistent_404(self, client):
        r = client.delete("/api/v1/backtest/does-not-exist")
        assert r.status_code == 404


class TestBacktestResults:
    def test_results_for_pending_job(self, client):
        r = client.post("/api/v1/backtest", json={
            "pair": "EURUSD",
            "models": ["logistic"],
        })
        job_id = r.json()["job_id"]
        r2 = client.get(f"/api/v1/backtest/{job_id}/results")
        assert r2.status_code == 400

    def test_results_for_nonexistent_job(self, client):
        r = client.get("/api/v1/backtest/does-not-exist/results")
        assert r.status_code == 404


class TestDataDownload:
    def test_trigger_download(self, client):
        r = client.post("/api/v1/data/download", json={"pair": "EURUSD"})
        assert r.status_code == 202
        data = r.json()
        assert data["job_id"]
        assert data["status"] == "pending"

    def test_download_unknown_pair(self, client):
        r = client.post("/api/v1/data/download", json={"pair": "ZZZZZZ"})
        assert r.status_code == 400


class TestJobManager:
    def test_create_and_get_job(self):
        from api.dependencies import get_data_store
        from api.services import JobManager

        store = get_data_store()
        jm = JobManager(store)

        job_id = str(uuid.uuid4())
        jm.create_job(job_id, "test", {"pair": "EURUSD"})

        job = jm.get_job(job_id)
        assert job is not None
        assert job["id"] == job_id
        assert job["status"] == "pending"
        assert job["config"]["pair"] == "EURUSD"

        jm.delete_job(job_id)
        assert jm.get_job(job_id) is None

    def test_update_job_status(self):
        from api.dependencies import get_data_store
        from api.services import JobManager

        store = get_data_store()
        jm = JobManager(store)

        job_id = str(uuid.uuid4())
        jm.create_job(job_id, "test", {})
        jm.update_status(job_id, "running")
        assert jm.get_job(job_id)["status"] == "running"

        jm.update_status(job_id, "completed", result={"sharpe": 1.5})
        job = jm.get_job(job_id)
        assert job["status"] == "completed"
        assert job["result"]["sharpe"] == 1.5

        jm.delete_job(job_id)

    def test_list_jobs(self):
        from api.dependencies import get_data_store
        from api.services import JobManager

        store = get_data_store()
        jm = JobManager(store)

        jobs_before = len(jm.list_jobs())

        for _ in range(3):
            jm.create_job(str(uuid.uuid4()), "test_list", {})

        jobs = jm.list_jobs(job_type="test_list")
        assert len(jobs) == 3

        for j in jobs:
            jm.delete_job(j["id"])


class TestDataStoreAPI:
    def test_get_candles_returns_dataframe(self):
        from api.dependencies import get_data_store

        store = get_data_store()
        df = store.get_candles("EURUSD", "M30", "2024-01-01", "2024-02-01")
        assert len(df) > 0
        assert "mid_close" in df.columns
        assert "time" in df.columns

    def test_get_pair_returns_metadata(self):
        from api.dependencies import get_data_store

        store = get_data_store()
        pair = store.get_pair("USDJPY")
        assert pair is not None
        assert pair["pip_value"] == 0.01

    def test_get_candle_count(self):
        from api.dependencies import get_data_store

        store = get_data_store()
        count = store.get_candle_count("GBPUSD", "M30")
        assert count > 100000
