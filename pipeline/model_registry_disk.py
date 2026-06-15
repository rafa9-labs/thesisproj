"""Model registry on disk — scan, register, activate, and query deployed snapshots.

Works alongside pipeline/model_persistence.py which handles the file I/O.
This module handles the SQLite registry and directory scanning.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.model_persistence import (
    DEPLOY_ROOT, _ensure_deploy_root, read_metadata, validate_snapshot,
    get_active_model_id, set_active_model_id, clear_active_model_id,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_snapshot(snapshot_path: str, db_path: str, parent_job_status: str | None = None) -> str:
    """Validate and register a snapshot in the deployed_models table.

    Returns the model ID (basename of snapshot directory).
    """
    if parent_job_status is not None and parent_job_status != "completed":
        raise ValueError(
            f"Cannot register snapshot: parent job status is '{parent_job_status}', not 'completed'. "
            "Only models from successfully completed backtests can be saved."
        )

    from pipeline.data_sqlite import DataStore

    ok, reason = validate_snapshot(snapshot_path)
    if not ok:
        raise ValueError(f"Invalid snapshot: {reason}")

    meta = read_metadata(snapshot_path)
    model_id = os.path.basename(snapshot_path.rstrip("/\\"))

    store = DataStore(db_path)
    with store._write_cursor() as (conn, cur):
        cur.execute(
            """INSERT OR REPLACE INTO deployed_models
               (id, model_type, snapshot_path, best_sharpe, best_return,
                created_at, status, tags, parent_job_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_id,
                str(meta.get("model_type", "unknown")),
                os.path.abspath(snapshot_path),
                float(meta.get("metrics", {}).get("sharpe", 0.0)),
                float(meta.get("metrics", {}).get("total_return_pct", 0.0)),
                meta.get("created_at_utc", _now()),
                "inactive",
                "[]",
                meta.get("parent_job_id"),
            ),
        )
    return model_id


def get_all_deployed(db_path: str) -> List[Dict[str, Any]]:
    """List all registered models. Verifies disk paths still exist."""
    from pipeline.data_sqlite import DataStore

    store = DataStore(db_path)
    rows: List[Dict[str, Any]] = []
    with store._cursor() as (conn, cur):
        cur.execute("SELECT * FROM deployed_models ORDER BY created_at DESC")
        cols = [d[0] for d in cur.description]
        for r in cur.fetchall():
            row = dict(zip(cols, r))
            if row.get("snapshot_path") and not os.path.isdir(row["snapshot_path"]):
                row["_missing_on_disk"] = True
            try:
                row["tags"] = json.loads(row.get("tags", "[]"))
            except (json.JSONDecodeError, TypeError):
                row["tags"] = []
            rows.append(row)
    return rows


def activate_model(model_id: str, db_path: str) -> bool:
    """Activate a model as the single global active model. Deactivates all others."""
    from pipeline.data_sqlite import DataStore

    store = DataStore(db_path)
    with store._write_cursor() as (conn, cur):
        cur.execute("SELECT id, model_type FROM deployed_models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            return False
        model_type = row[1]

        cur.execute(
            "UPDATE deployed_models SET status = 'inactive' WHERE status = 'active'",
        )
        cur.execute(
            "UPDATE deployed_models SET status = 'active' WHERE id = ?",
            (model_id,),
        )

    set_active_model_id(model_type, model_id)
    return True


def deactivate_model(model_id: str, db_path: str) -> bool:
    """Deactivate a model."""
    from pipeline.data_sqlite import DataStore

    store = DataStore(db_path)
    with store._write_cursor() as (conn, cur):
        cur.execute("SELECT id FROM deployed_models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            return False
        cur.execute("UPDATE deployed_models SET status = 'inactive' WHERE id = ?", (model_id,))
    clear_active_model_id()
    return True


def delete_model(model_id: str, db_path: str) -> Tuple[bool, str]:
    """Remove a model from DB and disk."""
    from pipeline.data_sqlite import DataStore

    store = DataStore(db_path)
    with store._write_cursor() as (conn, cur):
        cur.execute("SELECT id, snapshot_path FROM deployed_models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            return False, "Model not found"
        path = row[1]

        cur.execute("DELETE FROM deployed_models WHERE id = ?", (model_id,))

    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

    clear_active_model_id()
    return True, "ok"


def update_tags(model_id: str, db_path: str, action: str, tag: str) -> Optional[List[str]]:
    """Add or remove a tag from a model. Returns updated tag list or None."""
    from pipeline.data_sqlite import DataStore

    store = DataStore(db_path)
    with store._write_cursor() as (conn, cur):
        cur.execute("SELECT tags FROM deployed_models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if not row:
            return None
        try:
            tags = json.loads(row[0] or "[]")
        except (json.JSONDecodeError, TypeError):
            tags = []

        if action == "add" and tag not in tags:
            tags.append(tag)
        elif action == "remove":
            tags = [t for t in tags if t != tag]

        cur.execute("UPDATE deployed_models SET tags = ? WHERE id = ?", (json.dumps(tags), model_id))
    return tags


def scan_and_repair(db_path: str) -> Dict[str, int]:
    """On startup: scan deployed_models/ dir, register unregistered snapshots,
    clean up stale registry entries for missing directories.

    Returns dict with counts: {registered, cleaned, skipped}.
    """
    from pipeline.data_sqlite import DataStore

    root = _ensure_deploy_root()
    store = DataStore(db_path)

    with store._cursor() as (conn, cur):
        cur.execute("SELECT id, snapshot_path FROM deployed_models")
        db_paths: Dict[str, str] = {r[0]: r[1] for r in cur.fetchall()}

    registered = 0
    cleaned = 0
    skipped = 0

    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if not os.path.isdir(full) or entry.startswith(".") or entry.startswith("_"):
            continue
        existing_id = next((mid for mid, mp in db_paths.items()
                           if os.path.abspath(mp) == os.path.abspath(full)), None)
        if existing_id:
            continue
        ok, _ = validate_snapshot(full)
        if not ok:
            skipped += 1
            continue
        try:
            register_snapshot(full, db_path)
            registered += 1
        except Exception:
            skipped += 1

    for mid, mpath in db_paths.items():
        if not os.path.isdir(mpath):
            with store._write_cursor() as (conn, cur):
                cur.execute("DELETE FROM deployed_models WHERE id = ?", (mid,))
            cleaned += 1

    return {"registered": registered, "cleaned": cleaned, "skipped": skipped}
