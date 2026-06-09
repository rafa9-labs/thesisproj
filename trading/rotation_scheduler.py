"""
Auto-Rotation Scheduler — periodically checks model health and hot-swaps
underperforming models with better candidates from the committee snapshot.

Runs as a background asyncio task during live committee sessions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RotationScheduler:
    """Background task that monitors model health and rotates unhealthy models.

    Parameters
    ----------
    runner : LiveCommitteeRunner
        The committee runner with rotate_model() and find_replacement().
    snapshot_dir : str or Path or None
        Path to the committee_snapshot directory with {model}.joblib files.
    matrix_path : str or Path or None
        Path to regime_matrix_tuned.json for finding replacements.
    check_interval_hours : float
        How often to check health (default 4h).
    sharpe_threshold : float
        Rolling Sharpe below which a model is flagged unhealthy.
    hitrate_threshold : float
        Rolling hit rate below which a model is flagged unhealthy.
    max_rotations_per_24h : int
        Hard cap on rotations per 24-hour period.
    min_trades_for_health : int
        Minimum trades before health checks activate for a model.
    """

    def __init__(
        self,
        runner: Any,
        snapshot_dir: Optional[str] = None,
        matrix_path: Optional[str] = None,
        check_interval_hours: float = 4.0,
        sharpe_threshold: float = -0.5,
        hitrate_threshold: float = 0.35,
        max_rotations_per_24h: int = 3,
        min_trades_for_health: int = 10,
    ):
        self._runner = runner
        self._snapshot_dir = snapshot_dir
        self._matrix_path = matrix_path
        self.check_interval_hours = check_interval_hours
        self.sharpe_threshold = sharpe_threshold
        self.hitrate_threshold = hitrate_threshold
        self.max_rotations_per_24h = max_rotations_per_24h
        self.min_trades_for_health = min_trades_for_health

        self._rotation_timestamps: List[float] = []
        self._rotation_cooldown_models: Dict[str, float] = {}
        self._stopped = False
        self._events: List[Dict[str, Any]] = []

    @property
    def recent_rotations_24h(self) -> int:
        cutoff = time.time() - 86400
        return sum(1 for ts in self._rotation_timestamps if ts > cutoff)

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    async def run(self, get_session_status, alert_manager=None):
        """Run the scheduler loop. Call with asyncio.create_task()."""
        while not self._stopped:
            await asyncio.sleep(self.check_interval_hours * 3600)

            status = get_session_status()
            if status != "running":
                break

            try:
                self._check_and_rotate(alert_manager)
            except Exception:
                logger.exception("Rotation check failed")

    def stop(self):
        self._stopped = True

    def _check_and_rotate(self, alert_manager=None):
        health = self._runner.get_health_summary()
        if not health:
            return

        matrix = self._load_matrix()
        committee_models = list(self._runner.models.keys())
        current_regime = (
            self._runner.get_recent_regimes(1)[0]
            if self._runner.get_recent_regimes(1)
            else "sideways"
        )

        for model, metrics in health.items():
            if not self._should_rotate(model, metrics, committee_models):
                continue

            replacement = self._find_replacement(model, matrix, current_regime, committee_models)
            if replacement is None:
                self._log("rotation_failed", f"{model}: no replacement candidate")
                if alert_manager:
                    try:
                        alert_manager._send("warning",
                            f"[rotation] Cannot replace {model}: no candidate available")
                    except Exception:
                        pass
                continue

            try:
                new_model_obj = self._load_from_snapshot(replacement)
                if new_model_obj is None:
                    self._log("rotation_failed", f"{model} → {replacement}: snapshot load failed")
                    continue
            except Exception:
                self._log("rotation_failed", f"{model} → {replacement}: load exception")
                continue

            self._runner.rotate_model(model, replacement, new_model_obj)
            self._rotation_timestamps.append(time.time())
            self._rotation_cooldown_models[model] = time.time() + 43200
            self._log("model_rotation", f"{model} → {replacement} (Sharpe: {metrics.get('rolling_sharpe', 'N/A')})")

            msg = (
                f"[rotation] Rotated {model} → {replacement} "
                f"(Sharpe: {metrics.get('rolling_sharpe', 'N/A'):.2f})"
            )
            logger.info(msg)
            if alert_manager:
                try:
                    alert_manager._send("info", msg)
                except Exception:
                    pass

    def _should_rotate(
        self,
        model: str,
        metrics: dict,
        committee_models: List[str],
    ) -> bool:
        # Hard cap on total rotations
        if self.recent_rotations_24h >= self.max_rotations_per_24h:
            return False

        # Cooldown after rotation
        if model in self._rotation_cooldown_models:
            if time.time() < self._rotation_cooldown_models[model]:
                return False

        # Need enough data to judge health
        if metrics.get("total_signals", 0) < self.min_trades_for_health:
            return False

        # Only mark unhealthy if metrics are actually bad
        sr = metrics.get("rolling_sharpe")
        hr = metrics.get("rolling_hit_rate")
        if sr is not None and sr < self.sharpe_threshold:
            return True
        if hr is not None and hr < self.hitrate_threshold:
            return True

        # Don't rotate the last model
        if len(committee_models) <= 1:
            return False
        return False

    def _find_replacement(
        self,
        old_model: str,
        matrix: Any,
        regime: str,
        committee_models: List[str],
    ) -> Optional[str]:
        if matrix is not None:
            from pipeline.regime_utils import _REGIME_NAMES

            try:
                regime_idx = list(_REGIME_NAMES.values()).index(regime)
            except ValueError:
                regime_idx = 6

            models_list = list(matrix.models)
            sharpe_col = matrix.sharpe_matrix[:, regime_idx]
            scored = []
            for i, m in enumerate(models_list):
                if m != old_model and m not in committee_models:
                    if not hasattr(sharpe_col[i], '__isnan__') or not np.isnan(sharpe_col[i]):
                        scored.append((m, float(sharpe_col[i])))
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored:
                return scored[0][0]

        # Fallback: pick any model from snapshot that isn't in the committee
        if self._snapshot_dir:
            import numpy as np
            snap = Path(self._snapshot_dir)
            if snap.exists():
                for f in sorted(snap.glob("*.joblib")):
                    name = f.stem
                    if name != old_model and name not in committee_models:
                        return name
        return None

    def _load_from_snapshot(self, model_type: str) -> Any:
        if not self._snapshot_dir:
            return None
        snap = Path(self._snapshot_dir)
        jl_path = snap / f"{model_type}.joblib"
        tf_path = snap / f"{model_type}_tf"
        try:
            if jl_path.exists():
                import joblib
                return joblib.load(str(jl_path))
            if tf_path.exists():
                import tensorflow as tf
                return tf.keras.models.load_model(str(tf_path))
        except Exception:
            logger.exception("Failed to load snapshot for %s", model_type)
        return None

    def _load_matrix(self) -> Any:
        if not self._matrix_path:
            return None
        try:
            with open(self._matrix_path) as f:
                data = json.load(f)
            from pipeline.expert_profiler import RegimeModelMatrix
            import numpy as np
            return RegimeModelMatrix(
                regimes=data.get("regimes", []),
                models=data.get("models", []),
                sharpe_matrix=np.array(data.get("sharpe", [])),
                trade_matrix=np.array(data.get("trades", [])),
                hitrate_matrix=np.array(data.get("hit_rate", [])),
            )
        except Exception:
            return None

    def _log(self, event: str, detail: str):
        self._events.append({"event": event, "detail": detail, "time": time.time()})
