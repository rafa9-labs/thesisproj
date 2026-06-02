"""
Model Store — persist trained committee models with versioning.

Phase E of the Multi-Agent Autonomous Exploration Engine.
Supports the live deployment pipeline by saving/loading sklearn models,
committee configs, feature metadata, and health history.
"""
from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ModelSnapshot:
    """A versioned model artifact with metadata."""
    model_type: str
    model_path: str = ""        # file path within store
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0
    trained_at: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "version": self.version,
        }


@dataclass
class CommitteeSnapshot:
    """A versioned committee deployment package."""
    models: Dict[str, ModelSnapshot]   # model_type → snapshot
    committee_config_json: str         # CommitteeConfig as JSON string
    created_at: str                    # ISO timestamp
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "models": {m: s.to_dict() for m, s in self.models.items()},
            "committee_config_json": self.committee_config_json,
            "created_at": self.created_at,
            "version": self.version,
            "metadata": self.metadata,
        }


class ModelStore:
    """Persistent storage for trained models and committee snapshots.

    Directory structure::

        store_root/
          snapshots/
            v1_2026-06-01T120000/
              manifest.json
              models/
                logistic.pkl
                xgboost.pkl
              committee_config.json
          health/
            logistic.json
            xgboost.json
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(os.path.join(root_dir, "snapshots"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "health"), exist_ok=True)

    # ── Model persistence ────────────────────────────────────────────

    def save_model(self, model: Any, model_type: str, feature_names: List[str],
                   metrics: Optional[Dict[str, float]] = None) -> str:
        """Save a single trained model to the store. Returns the stored path."""
        import joblib
        import pickle

        model_dir = os.path.join(self.root_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        raw_bytes = pickle.dumps(model)
        filename = f"{model_type}_{_hash_bytes(raw_bytes)[:12]}.joblib"
        path = os.path.join(model_dir, filename)

        joblib.dump(model, path)

        # Save metadata
        snapshot = ModelSnapshot(
            model_type=model_type,
            model_path=path,
            feature_names=feature_names,
            n_features=len(feature_names),
            trained_at=datetime.utcnow().isoformat(),
            metrics=metrics or {},
        )
        meta_path = path + ".json"
        with open(meta_path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        return path

    def load_model(self, path: str) -> Any:
        """Load a trained model from the store."""
        import joblib
        return joblib.load(path)

    # ── Committee snapshots ──────────────────────────────────────────

    def save_committee_snapshot(
        self,
        models: Dict[str, Any],
        feature_names: List[str],
        committee_config_json: str,
        model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a complete committee deployment package.

        Parameters
        ----------
        models : dict[str, Any]
            model_type → fitted sklearn model.
        feature_names : list[str]
            Feature columns used by all models.
        committee_config_json : str
            CommitteeConfig serialized as JSON.
        model_metrics : dict, optional
            Per-model metrics (e.g. {"logistic": {"sharpe": 0.5, "trades": 100}}).
        metadata : dict, optional
            Arbitrary metadata.
        """
        now = datetime.utcnow().isoformat().replace(":", "").replace(".", "")
        version_dir = f"v1_{now}"
        snapshot_dir = os.path.join(self.root_dir, "snapshots", version_dir)
        os.makedirs(snapshot_dir, exist_ok=True)
        models_dir = os.path.join(snapshot_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        import joblib

        snapshots: Dict[str, ModelSnapshot] = {}
        for model_type, model in models.items():
            model_path = os.path.join(models_dir, f"{model_type}.pkl")
            joblib.dump(model, model_path)

            snapshots[model_type] = ModelSnapshot(
                model_type=model_type,
                model_path=model_path,
                feature_names=feature_names,
                n_features=len(feature_names),
                trained_at=now,
                metrics=model_metrics.get(model_type, {}) if model_metrics else {},
            )

        # Save committee config
        config_path = os.path.join(snapshot_dir, "committee_config.json")
        with open(config_path, "w") as f:
            f.write(committee_config_json)

        # Save manifest
        cmt = CommitteeSnapshot(
            models=snapshots,
            committee_config_json=committee_config_json,
            created_at=now,
            metadata=metadata or {},
        )
        manifest_path = os.path.join(snapshot_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(cmt.to_dict(), f, indent=2, default=str)

        return snapshot_dir

    def load_committee_snapshot(self, version_dir: str) -> Dict[str, Any]:
        """Load all models from a committee snapshot.

        Returns dict with keys: models (dict), config_json (str), metadata (dict).
        """
        import joblib

        manifest_path = os.path.join(version_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No manifest at {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        models: Dict[str, Any] = {}
        models_dir = os.path.join(version_dir, "models")
        for model_type in manifest.get("models", {}):
            model_path = os.path.join(models_dir, f"{model_type}.pkl")
            if os.path.exists(model_path):
                models[model_type] = joblib.load(model_path)

        return {
            "models": models,
            "config_json": manifest.get("committee_config_json", "{}"),
            "metadata": manifest.get("metadata", {}),
            "feature_names": list(manifest.get("models", {}).values())[0].get("feature_names", [])
                   if manifest.get("models") else [],
        }

    def list_snapshots(self) -> List[str]:
        """Return sorted list of snapshot directory paths."""
        snap_dir = os.path.join(self.root_dir, "snapshots")
        if not os.path.isdir(snap_dir):
            return []
        dirs = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir)
                if os.path.isdir(os.path.join(snap_dir, d))]
        return sorted(dirs)

    # ── Health tracking ──────────────────────────────────────────────

    def save_health(self, model_type: str, health_records: List[Dict[str, Any]]):
        """Append health records for a model."""
        path = os.path.join(self.root_dir, "health", f"{model_type}.json")
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.extend(health_records)
        # Keep last 500 records
        existing = existing[-500:]
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def load_health(self, model_type: str) -> List[Dict[str, Any]]:
        """Load health history for a model."""
        path = os.path.join(self.root_dir, "health", f"{model_type}.json")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
