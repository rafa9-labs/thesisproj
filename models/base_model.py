"""
BaseModel -- abstract interface for all ML models in the pipeline.

Every model wrapper (sklearn, XGBoost, Keras, DQN, Ensemble) must
conform to this interface so that `model_factory_mixin.get_model()`
can use a registry instead of a 10-branch if/elif chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseModel(ABC):
    """Minimal unified interface for pipeline models.

    Subclasses wrap a concrete estimator (sklearn, XGBClassifier,
    keras.Model, DQN agent, etc.) and expose fit/predict/predict_proba.
    """

    # ------------------------------------------------------------------
    # Subclass metadata
    # ------------------------------------------------------------------
    model_type: str = ""           # e.g. "logistic", "cnn", "dqn"
    is_deep: bool = False          # True -> GPU path, subprocess worker
    supports_proba: bool = True    # False -> predict_proba raises

    def __init__(self, **kwargs):
        """Store raw kwargs for reproducibility; subclasses use what they need."""
        self._init_kwargs: dict = kwargs
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, X, y, **kwargs) -> "BaseModel":
        """Train the model. Must return self."""
        ...

    @abstractmethod
    def predict(self, X):
        """Return hard class labels (shape [n])."""
        ...

    def predict_proba(self, X):
        """Return probability array (shape [n, n_classes]).

        Default: raise NotImplementedError.  Subclasses that support
        probabilistic output must override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba"
        )

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        """Return a copy of the init kwargs (sklearn-compatible)."""
        return dict(self._init_kwargs)

    def save(self, path: str) -> None:
        """Persist model to disk. Default: pickle via joblib."""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load a saved model. Default: pickle via joblib."""
        import joblib
        return joblib.load(path)

    def save_artifacts(self, directory: str, *, scaler=None, imputer=None,
                       calibration=None, metadata: dict | None = None) -> str:
        """Persist the full model snapshot to disk.

        Parameters
        ----------
        directory : str
            Output directory (created if missing).
        scaler : object, optional
            Fitted scaler to persist alongside the model.
        imputer : object, optional
            Fitted imputer to persist.
        calibration : object, optional
            Fitted calibration object (e.g. CalibratedClassifierCV).
        metadata : dict, optional
            Arbitrary metadata written as ``metadata.json``.

        Returns
        -------
        str
            The output directory path.
        """
        import hashlib
        import json
        import os
        import joblib as _jl

        os.makedirs(directory, exist_ok=True)

        checksums: dict[str, str] = {}

        def _dump(obj, fname):
            p = os.path.join(directory, fname)
            _jl.dump(obj, p)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            checksums[fname] = h.hexdigest()

        _dump(self, "model.joblib")
        if scaler is not None:
            _dump(scaler, "scaler.joblib")
        if imputer is not None:
            _dump(imputer, "imputer.joblib")
        if calibration is not None:
            _dump(calibration, "calibration.joblib")

        meta = dict(metadata or {})
        meta["schema_version"] = 1
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        with open(os.path.join(directory, "manifest.sha256"), "w") as f:
            for name in sorted(checksums):
                f.write(f"{checksums[name]}  {name}\n")

        return directory

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.model_type!r}, fitted={self._fitted})"