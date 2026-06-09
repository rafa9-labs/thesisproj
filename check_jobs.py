"""Check previous failed jobs and try to reproduce int base error."""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

for job_id in [
    "1ad4c171-04a8-41e1-ae24-281a9d744d08",
    "86af6767-b183-401d-a6ba-ebc5929e19e1",
]:
    r = client.get(f"/api/v1/backtest/{job_id}/status")
    print(f"Job {job_id}: status={r.status_code}, body={r.json()}")

    r2 = client.get(f"/api/v1/backtest/{job_id}/results")
    print(f"  results: status={r2.status_code}")
    if r2.status_code != 200:
        body = r2.text[:500]
        print(f"  error body: {body}")
        if "int" in body.lower() and "base" in body.lower():
            print("  FOUND INT BASE ERROR!")
    else:
        data = r2.json()
        print(f"  OK: {len(data.get('metrics',[]))} metrics")
