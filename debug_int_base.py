"""Try to reproduce the int base error by running a full backtest + fetching results."""
import json, time, sys
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Submit a backtest with sufficient data and HPO
req = {
    "pair": "EURUSD",
    "models": ["logistic"],
    "timeframe": "M30",
    "n_trials": 5,
    "repeats": 1,
    "start_date": "2019-01-01",
    "end_date": "2020-06-01",
    "hpo_intensity": "light",
    "hpo": True,
    "features": ["returns", "rsi", "macd"],
    "lags": 3,
}
r = client.post("/api/v1/backtest", json=req)
print("Backtest:", r.status_code)
data = r.json()
job_id = data.get("job_id")
print("Job ID:", job_id)

# Wait for completion
for i in range(600):
    r = client.get(f"/api/v1/backtest/{job_id}/status")
    st = r.json().get("status")
    if st in ("completed", "failed"):
        print(f"Status: {st}")
        if st == "failed":
            err = r.json().get("error", "")
            print("Error:", err[:500])
            if "int" in err.lower() and "base" in err.lower():
                print("FOUND INT BASE ERROR!")
        break
    if i % 30 == 0:
        r2 = client.get(f"/api/v1/backtest/{job_id}/status")
        prog = r2.json().get("progress", {})
        print(f"  Waiting... ({i*2}s) progress={prog}")
    time.sleep(2)
else:
    print("Timeout")
    sys.exit(1)

# Get results
try:
    r = client.get(f"/api/v1/backtest/{job_id}/results")
    print("Results status:", r.status_code)
    if r.status_code == 200:
        data = r.json()
        print("Models:", data.get("models"))
        for m in data.get("metrics", []):
            model = m.get("model")
            sharpe = m.get("sharpe")
            trades = m.get("total_trades")
            print(f"  {model}: sharpe={sharpe}, trades={trades}")
        print("OK — no int base error")
    else:
        body = r.text[:2000]
        print("Error body:", body)
        if "int" in body.lower() and "base" in body.lower():
            print("FOUND INT BASE ERROR!")
        else:
            print(f"Different error (not int base): status={r.status_code}")
except Exception as e:
    print("Exception:", e)
    import traceback
    traceback.print_exc()
