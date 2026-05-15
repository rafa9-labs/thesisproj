"""Verify cross-process event pipeline via SQLite.

Usage:
    python scripts/verify_event_pipeline.py

Expected output: PASS if worker-written events are visible to API server process.
"""
import json
import multiprocessing
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _writer(db_path: str, job_id: str):
    from pipeline.data_sqlite import DataStore

    store = DataStore(db_path)
    for i in range(5):
        store.append_job_event(
            job_id,
            json.dumps({"event": "hpo_progress", "trial": i + 1, "model": "logistic"}),
        )
    print(f"[WRITER] wrote 5 events for {job_id}")


def _reader(db_path: str, job_id: str):
    from pipeline.data_sqlite import DataStore

    store = DataStore(db_path)
    events = store.get_job_events(job_id, after=0)
    print(f"[READER] read {len(events)} events for {job_id}")
    for e in events:
        print(f"  - idx={e.get('_idx')} event={e.get('event')} trial={e.get('trial')}")
    return events


def main():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_events.db"
        job_id = "verify-job-123"

        # Write from one process
        p = multiprocessing.Process(target=_writer, args=(str(db_path), job_id))
        p.start()
        p.join()

        # Read from another process
        events = _reader(str(db_path), job_id)

        # Validate
        passed = True
        if len(events) != 5:
            print(f"FAIL: expected 5 events, got {len(events)}")
            passed = False
        else:
            print("PASS: 5 events written, 5 events read")

        indices = [e.get("_idx") for e in events]
        expected = [0, 1, 2, 3, 4]
        if indices != expected:
            print(f"FAIL: expected indices {expected}, got {indices}")
            passed = False
        else:
            print("PASS: indices are sequential 0-4")

        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
