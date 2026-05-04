"""Encrypted local storage for license state and sensitive data.

Uses Fernet symmetric encryption with a key derived from APP_SECRET + machine fingerprint.
If APP_SECRET is not set, auto-generates one and stores it next to the database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS licenses (
    license_key TEXT PRIMARY KEY,
    activation_id TEXT,
    machine_fingerprint TEXT,
    plan TEXT NOT NULL DEFAULT 'free',
    activated_at TEXT,
    expires_at TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_name TEXT PRIMARY KEY,
    encrypted_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trial (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    started_at TEXT,
    expires_at TEXT,
    active INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kv_store (
    key TEXT PRIMARY KEY,
    encrypted_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _derive_fernet_key(app_secret: str, machine_id: str) -> bytes:
    hkdf = HKDF(
        algorithm=SHA256(),
        length=32,
        salt=b"kodaquant-secure-storage-v1",
        info=b"fernet-key-derivation",
    )
    raw = hkdf.derive(f"{app_secret}|{machine_id}".encode())
    return urlsafe_b64encode(raw)


def _find_or_create_secret(db_dir: Path) -> str:
    secret_file = db_dir / ".app_secret"
    if secret_file.exists():
        return secret_file.read_text().strip()
    env_secret = os.environ.get("APP_SECRET", "").strip()
    if env_secret:
        secret_file.write_text(env_secret)
        return env_secret
    secret = Fernet.generate_key().decode()
    secret_file.write_text(secret)
    logger.info("Generated new APP_SECRET at %s", secret_file)
    return secret


class SecureStorage:
    def __init__(self, db_path: Optional[str] = None, machine_id: Optional[str] = None):
        if db_path is None:
            data_dir = os.environ.get("FX_DATA_DIR")
            if data_dir:
                db_dir = Path(data_dir)
            else:
                db_dir = Path(os.environ.get("APPDATA", Path.home())) / "KodaQuant" / "data"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "secure.db")
        else:
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self._machine_id = machine_id or "unknown"
        app_secret = _find_or_create_secret(db_dir)
        fernet_key = _derive_fernet_key(app_secret, self._machine_id)
        self._fernet = Fernet(fernet_key)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_V1)

    def _encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def store_license(
        self,
        license_key: str,
        activation_id: str,
        machine_fingerprint: str,
        plan: str,
        activated_at: str,
        expires_at: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        enc_key = self._encrypt(license_key)
        enc_meta = self._encrypt(json.dumps(metadata or {}))
        now = _now_iso()
        self._conn.execute(
            """INSERT INTO licenses (license_key, activation_id, machine_fingerprint, plan,
                   activated_at, expires_at, last_verified, verification_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
               ON CONFLICT(license_key) DO UPDATE SET
                   activation_id=excluded.activation_id,
                   machine_fingerprint=excluded.machine_fingerprint,
                   plan=excluded.plan,
                   activated_at=excluded.activated_at,
                   expires_at=excluded.expires_at,
                   last_verified=excluded.last_verified,
                   verification_count=verification_count+1,
                   metadata=excluded.metadata""",
            (enc_key, activation_id, machine_fingerprint, plan, activated_at, expires_at, now, enc_meta),
        )
        self._conn.commit()

    def get_license(self) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT license_key, activation_id, machine_fingerprint, plan, "
            "activated_at, expires_at, last_verified, verification_count, metadata "
            "FROM licenses LIMIT 1"
        ).fetchone()
        if not row:
            return None
        enc_key, act_id, fp, plan, act_at, exp_at, last_v, count, enc_meta = row
        try:
            meta = json.loads(self._decrypt(enc_meta)) if enc_meta else {}
            return {
                "license_key": self._decrypt(enc_key),
                "activation_id": act_id,
                "machine_fingerprint": fp,
                "plan": plan,
                "activated_at": act_at,
                "expires_at": exp_at,
                "last_verified": last_v,
                "verification_count": count,
                "metadata": meta,
            }
        except Exception:
            logger.warning("Failed to decrypt license data — key mismatch?")
            return None

    def delete_license(self) -> bool:
        cursor = self._conn.execute("DELETE FROM licenses")
        self._conn.commit()
        return cursor.rowcount > 0

    def update_verification(self, last_verified: Optional[str] = None) -> None:
        now = last_verified or _now_iso()
        self._conn.execute(
            "UPDATE licenses SET last_verified=?, verification_count=verification_count+1"
        )
        self._conn.commit()

    def start_trial(self, duration_days: int = 14) -> Dict[str, str]:
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        expires = now + timedelta(days=duration_days)
        self._conn.execute(
            "INSERT INTO trial (id, started_at, expires_at, active) VALUES (1, ?, ?, 1) "
            "ON CONFLICT(id) DO UPDATE SET started_at=excluded.started_at, expires_at=excluded.expires_at, active=1",
            (now.isoformat(), expires.isoformat()),
        )
        self._conn.commit()
        return {"started_at": now.isoformat(), "expires_at": expires.isoformat()}

    def get_trial(self) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT started_at, expires_at, active FROM trial WHERE id=1"
        ).fetchone()
        if not row:
            return None
        started, expires, active = row
        return {"started_at": started, "expires_at": expires, "active": bool(active)}

    def end_trial(self) -> None:
        self._conn.execute("UPDATE trial SET active=0 WHERE id=1")
        self._conn.commit()

    def store_api_key(self, name: str, value: str) -> None:
        enc = self._encrypt(value)
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO api_keys (key_name, encrypted_value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key_name) DO UPDATE SET encrypted_value=excluded.encrypted_value, updated_at=excluded.updated_at",
            (name, enc, now),
        )
        self._conn.commit()

    def get_api_key(self, name: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT encrypted_value FROM api_keys WHERE key_name=?", (name,)
        ).fetchone()
        if not row:
            return None
        try:
            return self._decrypt(row[0])
        except Exception:
            return None

    def delete_api_key(self, name: str) -> bool:
        cursor = self._conn.execute("DELETE FROM api_keys WHERE key_name=?", (name,))
        self._conn.commit()
        return cursor.rowcount > 0

    def set_kv(self, key: str, value: str) -> None:
        enc = self._encrypt(value)
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO kv_store (key, encrypted_value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET encrypted_value=excluded.encrypted_value, updated_at=excluded.updated_at",
            (key, enc, now),
        )
        self._conn.commit()

    def get_kv(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT encrypted_value FROM kv_store WHERE key=?", (key,)
        ).fetchone()
        if not row:
            return None
        try:
            return self._decrypt(row[0])
        except Exception:
            return None

    def close(self) -> None:
        self._conn.close()


def _now_iso() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()