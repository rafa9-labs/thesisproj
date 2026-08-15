"""
Live Committee Runner — Phase E of the Multi-Agent Autonomous Exploration Engine.

Loads a committee config + trained models, then processes streaming OHLC bars
one at a time, outputting trade signals with confidence scores.

At each new bar:
  1. Compute indicator features from recent price history
  2. Classify current market regime (7-class)
  3. Route to assigned models, get probability predictions
  4. Blend via committee weights
  5. Emit trade signal (-1, 0, 1) with confidence and metadata

Includes health monitoring and model rotation capabilities.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.committee.committee_builder import CommitteeConfig, RegimeAssignment
from pipeline.regime.regime_utils import (
    RegimeConfig,
    _REGIME_NAMES,
)


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class LiveSignal:
    """A trade signal emitted by the committee."""
    timestamp: Any
    signal: int          # -1 (sell), 0 (flat), 1 (buy)
    confidence: float    # 0.0–1.0, max probability of the winning class
    regime: str          # current regime name
    regime_prob: float   # probability of the predicted regime
    blended_probs: Dict[str, float]   # short/flat/long probabilities
    active_models: List[str]          # which models contributed
    model_weights: List[float]        # blending weights used
    is_healthy: bool = True           # False if models are underperforming
    meta_override: bool = False       # True if meta-learner overrode committee
    meta_filtered: bool = False       # P1: True if meta-labeler suppressed trade
    meta_win_prob: float = 0.5        # P1: P(trade_is_winner) from meta-labeler
    throttle_level: str = "full"      # "full", "half", "observe"
    conviction_multiplier: float = 1.0  # 0.5 explorer / 1.0 standard / 1.5 conviction
    bar_vol: float = 0.0              # rolling bar volatility (backtest-aligned sizing)
    atr: float = 0.0                  # ATR proxy in price units (backtest-aligned sizing)

    def to_dict(self) -> dict:
        return {
            "timestamp": str(self.timestamp),
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "regime": self.regime,
            "regime_prob": round(self.regime_prob, 4),
            "blended_probs": {k: round(v, 4) for k, v in self.blended_probs.items()},
            "active_models": self.active_models,
            "model_weights": [round(w, 3) for w in self.model_weights],
            "is_healthy": self.is_healthy,
            "meta_override": self.meta_override,
            "meta_filtered": self.meta_filtered,
            "meta_win_prob": round(self.meta_win_prob, 4),
            "throttle_level": self.throttle_level,
            "conviction_multiplier": round(self.conviction_multiplier, 2),
            "bar_vol": round(self.bar_vol, 8),
            "atr": round(self.atr, 8),
        }


@dataclass
class ModelHealth:
    """Rolling health metrics for a single model."""
    model_type: str
    recent_trades: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    recent_signals: Deque[int] = field(default_factory=lambda: deque(maxlen=100))
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    last_sharpe: float = np.nan
    last_hit_rate: float = np.nan
    is_healthy: bool = True

    def record_trade(self, signal: int, pnl: float):
        self.recent_trades.append(pnl)
        self.recent_signals.append(signal)
        self.total_signals += 1
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1

        # Recompute rolling metrics
        if len(self.recent_trades) >= 3:
            trades_arr = np.array(self.recent_trades)
            mean_ret = np.mean(trades_arr)
            std_ret = np.std(trades_arr, ddof=1)
            self.last_sharpe = float(mean_ret / std_ret * np.sqrt(252 * 24)) if std_ret > 0 else 0.0
            self.last_hit_rate = float((trades_arr > 0).mean())


# ── Main runner ──────────────────────────────────────────────────────

class LiveCommitteeRunner:
    """Streaming committee prediction engine for live trading.

    Parameters
    ----------
    config : CommitteeConfig
        Per-regime model assignments with blending weights.
    models : dict[str, Any]
        model_type → fitted sklearn model (must have predict_proba).
    feature_names : list[str]
        Column names the models were trained on.
    regime_cfg : RegimeConfig
        Thresholds for 7-class regime detection.
    confidence_threshold : float
        Minimum blended probability to emit a non-zero signal.
    lookback_bars : int
        Number of recent bars needed for feature computation (default 100).
    health_window : int
        Rolling window size for health metrics (default 50 trades).
    rotation_sharpe_threshold : float
        If a model's rolling Sharpe drops below this, it's flagged unhealthy.
    rotation_hitrate_threshold : float
        If a model's rolling hit rate drops below this, it's flagged unhealthy.
    """

    def __init__(
        self,
        config: CommitteeConfig,
        models: Dict[str, Any],
        feature_names: List[str],
        regime_cfg: Optional[RegimeConfig] = None,
        confidence_threshold: float = 0.55,
        lookback_bars: int = 100,
        health_window: int = 50,
        rotation_sharpe_threshold: float = -0.5,
        rotation_hitrate_threshold: float = 0.35,
        meta_learner=None,
        meta_labeler=None,
        hmm_detector=None,
        conviction_sizer=None,
    ):
        self.config = config
        self.models = models
        self.feature_names = feature_names
        self.regime_cfg = regime_cfg or RegimeConfig()
        self.confidence_threshold = confidence_threshold
        self.lookback_bars = lookback_bars
        self.health_window = health_window
        self.rotation_sharpe_threshold = rotation_sharpe_threshold
        self.rotation_hitrate_threshold = rotation_hitrate_threshold
        self._meta_learner = meta_learner
        self._meta_labeler = meta_labeler  # P1
        self._hmm_detector = hmm_detector  # P2
        self._conviction_sizer = conviction_sizer  # P3

        # Internal state
        self._bar_buffer: Deque[Dict[str, float]] = deque(maxlen=lookback_bars)
        self._health: Dict[str, ModelHealth] = {
            m: ModelHealth(model_type=m) for m in models
        }
        self._regime_history: Deque[str] = deque(maxlen=100)
        self._signal_history: Deque[LiveSignal] = deque(maxlen=1000)
        self._bar_count: int = 0
        self._start_time: Optional[datetime] = None
        self._is_running: bool = False

    # ── Public API ───────────────────────────────────────────────────

    def start(self):
        """Initialize the runner. Must be called before process_bar()."""
        self._is_running = True
        self._start_time = datetime.utcnow()
        self._bar_count = 0
        self._bar_buffer.clear()
        print(f"[RUNNER] Started with {len(self.models)} models, "
              f"{len(self.feature_names)} features")

    def stop(self) -> Dict[str, Any]:
        """Stop the runner and return session summary."""
        self._is_running = False
        elapsed = (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0
        return {
            "bars_processed": self._bar_count,
            "signals_emitted": len(self._signal_history),
            "non_zero_signals": sum(1 for s in self._signal_history if s.signal != 0),
            "elapsed_seconds": elapsed,
            "model_health": {m: self._health_summary(m) for m in self._health},
        }

    def process_bar(self, bar: Dict[str, float]) -> Optional[LiveSignal]:
        """Process a single OHLC bar and return a trade signal (or None).

        Parameters
        ----------
        bar : dict
            Must contain: mid_c, mid_h, mid_l, mid_o, spread, returns.
            May optionally contain: timestamp.

        Returns
        -------
        LiveSignal or None
            None if insufficient data for feature computation.
        """
        if not self._is_running:
            raise RuntimeError("Runner not started. Call start() first.")

        self._bar_count += 1
        self._bar_buffer.append(bar)

        if len(self._bar_buffer) < self.lookback_bars:
            return None  # Not enough history for features

        # 1. Build feature vector from recent bars
        features = self._build_features()

        # 2. Classify regime
        regime_id, regime_probs, _named_probs = self._classify_regime()
        regime_name = _REGIME_NAMES.get(int(regime_id), "sideways")
        self._regime_history.append(regime_name)

        # 3. Route to models
        assignment = self.config.regime_models(regime_name)
        if assignment is None or not assignment.models:
            return None

        # 4. Get predictions and blend
        blended, active_models, used_weights = self._blend_predictions(
            features, assignment
        )

        if blended is None:
            return None

        # 5. Convert to trade signal
        max_class = np.argmax(blended)
        max_prob = float(blended[max_class])
        signal = 0
        if max_prob >= self.confidence_threshold:
            if max_class == 2:
                signal = 1
            elif max_class == 0:
                signal = -1

        # 5a. Meta-labeler gate (P1: binary P(win) filter)
        meta_filtered = False
        meta_win_prob = 0.5
        if signal != 0 and self._meta_labeler is not None:
            try:
                primary_probs = (float(blended[0]), float(blended[1]), float(blended[2]))
                should_trade, meta_win_prob = self._meta_labeler.should_trade(
                    signal, primary_probs, regime_id=int(regime_id),
                )
                if not should_trade:
                    meta_filtered = True
                    signal = 0
            except Exception:
                pass

        # 5b. Compute conviction multiplier (P3: sigmoid sizing, fallback to tiers)
        conviction_multiplier = 1.0
        if signal != 0:
            if self._conviction_sizer is not None:
                conviction_multiplier = self._conviction_sizer.get_multiplier(max_prob)
            else:
                # Fallback: original 3-tier logic
                if max_prob >= 0.80:
                    conviction_multiplier = 1.5
                elif max_prob >= 0.65:
                    conviction_multiplier = 1.0
                elif max_prob >= 0.55:
                    conviction_multiplier = 0.5

        # 6. Check health
        is_healthy = self._check_health()
        if not is_healthy and signal != 0:
            signal = 0  # suppress trades when unhealthy
            conviction_multiplier = 1.0  # no sizing effect for suppressed signal

        # 7. Meta-learner retired — P1 MetaLabeler now handles trade filtering.
        #    The old CommitteeMetaLearner flipped signal direction (Long→Short),
        #    violating separation of concerns. A secondary model should never
        #    reverse the primary model's direction — only suppress to flat.
        meta_override = False

        # 8. Volatility inputs from the bar buffer (backtest-aligned sizing)
        bar_vol, atr = self._compute_volatility()

        live_signal = LiveSignal(
            timestamp=bar.get("timestamp", self._bar_count),
            signal=signal,
            confidence=max_prob,
            regime=regime_name,
            regime_prob=float(regime_probs.get(int(regime_id), 0.0)),
            blended_probs={
                "short": float(blended[0]),
                "flat": float(blended[1]),
                "long": float(blended[2]),
            },
            active_models=active_models,
            model_weights=used_weights,
            is_healthy=is_healthy,
            meta_override=meta_override,
            meta_filtered=meta_filtered,
            meta_win_prob=meta_win_prob,
            conviction_multiplier=conviction_multiplier,
            bar_vol=bar_vol,
            atr=atr,
        )
        self._signal_history.append(live_signal)
        return live_signal

    def record_trade_outcome(self, signal: LiveSignal, pnl: float):
        """Record the PnL result of a trade signal for health tracking."""
        for model in signal.active_models:
            if model in self._health:
                self._health[model].record_trade(signal.signal, pnl)

    def _compute_volatility(self, vol_window: int = 48, atr_window: int = 14) -> tuple[float, float]:
        """Compute (bar_vol, atr) from the bar buffer (backtest-aligned).

        bar_vol = rolling std of log returns; atr = mean |log return| * price.
        """
        try:
            closes = [
                float(b.get("mid_c", b.get("mid_close", np.nan)))
                for b in list(self._bar_buffer)[-(vol_window + 1):]
                if b.get("mid_c", b.get("mid_close")) is not None
            ]
        except Exception:
            return (0.0, 0.0)
        if len(closes) < 4:
            return (0.0, 0.0)
        closes = np.asarray(closes, dtype=float)
        lrs = np.diff(np.log(np.clip(closes, 1e-9, None)))
        bar_vol = float(np.std(lrs, ddof=1)) if len(lrs) >= 2 else 0.0
        atr_window = min(atr_window, len(lrs))
        atr = float(np.mean(np.abs(lrs[-atr_window:]))) * float(closes[-1]) if atr_window >= 2 else 0.0
        return (bar_vol, atr)

    # ── Feature engineering ──────────────────────────────────────────

    def _build_features(self) -> np.ndarray:
        """Build a single-row feature vector from the bar buffer.

        Delegates to pipeline.features.feature_sweep.compute_feature_matrix so the
        exact same indicator computation used during profiling is applied
        at inference time (zero data-drift between train and production).
        """
        from pipeline.features.feature_sweep import compute_feature_matrix

        df = pd.DataFrame(list(self._bar_buffer))
        feature_df = compute_feature_matrix(df, feature_names=list(self.feature_names),
                                             include_ohlc=False)
        if feature_df.empty or len(feature_df) == 0:
            return np.zeros((1, len(self.feature_names)), dtype=np.float32)
        last = feature_df.iloc[-1]
        result = np.zeros((1, len(self.feature_names)), dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            if name in feature_df.columns:
                val = last[name]
                try:
                    result[0, i] = float(val)
                except (ValueError, TypeError):
                    result[0, i] = 0.0
        return result

    # ── Regime classification ────────────────────────────────────────

    def _classify_regime(self) -> Tuple[int, Dict[int, float], Dict[str, float]]:
        """Classify current bar into a 7-class regime.

        Uses HMM when available (P2), falls back to rule-based ADX/EMA/ATR.
        Returns (regime_id, {regime_id: probability}, {regime_name: probability}).
        """
        df = pd.DataFrame(list(self._bar_buffer)[-50:])

        # ── HMM path (P2) ──
        if self._hmm_detector is not None and self._hmm_detector.is_fitted:
            try:
                regime_ids, _ = self._hmm_detector.predict(df)
                regime_probs_arr = self._hmm_detector.predict_regime_probs(df)
                if len(regime_ids) > 0 and len(regime_probs_arr) > 0:
                    regime = int(regime_ids[-1])
                    probs_arr = regime_probs_arr[-1]
                    probs = {int(i): float(probs_arr[i]) for i in range(7)}
                    named = {_REGIME_NAMES.get(i, f"regime_{i}"): probs[i] for i in range(7)}
                    return regime, probs, named
            except Exception:
                pass  # fall through to rule-based

        # ── Rule-based fallback (existing logic) ──
        cfg = self.regime_cfg

        # Default
        regime = 6  # sideways
        probs = {i: 0.0 for i in range(7)}
        probs[6] = 1.0

        price = df["mid_c"].astype(np.float64)
        if len(price) < 20:
            named = {_REGIME_NAMES.get(i, f"regime_{i}"): probs[i] for i in range(7)}
            return regime, probs, named

        ema = price.ewm(span=20, adjust=False).mean().iloc[-1]
        atr_val = (df["mid_h"] - df["mid_l"]).rolling(14).mean().iloc[-1]

        # Compute simple ADX proxy
        hi, lo = df["mid_h"].astype(np.float64), df["mid_l"].astype(np.float64)
        up_move = hi.diff().clip(lower=0).ewm(alpha=1.0 / 14, adjust=False).mean().iloc[-1]
        down_move = (-lo.diff()).clip(lower=0).ewm(alpha=1.0 / 14, adjust=False).mean().iloc[-1]
        tr_val = (hi - lo).rolling(14).mean().iloc[-1]
        pdi_val = 100.0 * up_move / (tr_val + 1e-10)
        mdi_val = 100.0 * down_move / (tr_val + 1e-10)
        dx_val = 100.0 * abs(pdi_val - mdi_val) / (pdi_val + mdi_val + 1e-10)
        adx_val = dx_val * 0.33 + 15.0  # rough smooth

        # Rules
        is_trend = adx_val >= cfg.adx_thresh
        above_ema = price.iloc[-1] > ema if not np.isnan(ema) else False

        atr_high = atr_val > 0.0008 if not np.isnan(atr_val) else False

        if is_trend and above_ema:
            regime = 1  # trend_up
        elif is_trend and not above_ema:
            regime = 2  # trend_down
        elif atr_high and not is_trend:
            regime = 5  # high_volatile

        # Assign probabilities (soft distribution for fuzzy blending in Phase 4)
        probs = {i: 0.05 for i in range(7)}
        probs[regime] = 0.70

        named = {_REGIME_NAMES.get(i, f"regime_{i}"): probs[i] for i in range(7)}
        return regime, probs, named

    # ── Prediction blending ─────────────────────────────────────────

    def _blend_predictions(
        self,
        features: np.ndarray,
        assignment: RegimeAssignment,
    ) -> Tuple[Optional[np.ndarray], List[str], List[float]]:
        """Get predictions from assigned models and blend by weights.

        Applies dynamic decay to model weights based on rolling hit rate.
        A model below 50% hit rate smoothly loses up to 50% voting power
        before the hard health gate (-0.5 Sharpe) triggers full rotation.
        """
        prob_sum = np.zeros(3, dtype=np.float64)
        weight_sum = 0.0
        active = []
        used_w = []

        for model_name, weight in zip(assignment.models, assignment.weights):
            model = self.models.get(model_name)
            if model is None:
                continue
            try:
                proba = model.predict_proba(features)
                if proba is not None:
                    decay = 1.0
                    health = self._health.get(model_name)
                    if health is not None and health.total_signals >= 5:
                        hit_rate = health.last_hit_rate
                        if not np.isnan(hit_rate) and hit_rate < 0.50:
                            decay = max(0.5, 1.0 - (0.50 - hit_rate) / 0.15)

                    effective_weight = weight * decay
                    n_cols = proba.shape[1]
                    if n_cols >= 3:
                        prob_sum += effective_weight * proba[0, :3]
                        weight_sum += effective_weight
                        active.append(model_name)
                        used_w.append(round(effective_weight, 4))
                    elif n_cols == 2:
                        # Binary classifier: class 0 = short, class 1 = long
                        p_short = proba[0, 0]
                        p_long = proba[0, 1]
                        prob_sum += effective_weight * np.array([p_short, 0.0, p_long])
                        weight_sum += effective_weight
                        active.append(model_name)
                        used_w.append(round(effective_weight, 4))
            except Exception:
                continue

        if weight_sum <= 0:
            return None, [], []

        blended = prob_sum / weight_sum
        return blended, active, used_w

    # ── Health monitoring ────────────────────────────────────────────

    def _check_health(self) -> bool:
        """Check if the committee is healthy enough to trade.

        Returns False if most active models are unhealthy.
        """
        unhealthy = 0
        total = 0
        for model_type, health in self._health.items():
            if health.total_signals < 5:
                continue  # skip models with insufficient history
            total += 1

            is_unhealthy = False
            if not np.isnan(health.last_sharpe) and health.last_sharpe < self.rotation_sharpe_threshold:
                is_unhealthy = True
            if not np.isnan(health.last_hit_rate) and health.last_hit_rate < self.rotation_hitrate_threshold:
                is_unhealthy = True

            health.is_healthy = not is_unhealthy
            if is_unhealthy:
                unhealthy += 1

        if total > 0 and unhealthy >= total * 0.5:
            return False
        return True

    def _health_summary(self, model_type: str) -> Dict[str, Any]:
        """Get health summary for one model."""
        h = self._health.get(model_type)
        if h is None:
            return {}
        return {
            "total_signals": h.total_signals,
            "wins": h.wins,
            "losses": h.losses,
            "rolling_sharpe": round(h.last_sharpe, 3) if not np.isnan(h.last_sharpe) else None,
            "rolling_hit_rate": round(h.last_hit_rate, 3) if not np.isnan(h.last_hit_rate) else None,
            "is_healthy": h.is_healthy,
        }

    def get_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get full health summary for all models."""
        return {m: self._health_summary(m) for m in self._health}

    def get_recent_regimes(self, n: int = 20) -> List[str]:
        return list(self._regime_history)[-n:]

    def get_recent_signals(self, n: int = 10) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in list(self._signal_history)[-n:]]

    # ── Model rotation ───────────────────────────────────────────────

    def rotate_model(
        self,
        old_model: str,
        new_model_name: str,
        new_model: Any,
    ):
        """Replace an underperforming model in the committee.

        This is the hot-swap mechanism: when a model's health decays,
        call this to substitute it without restarting the runner.
        """
        if old_model in self.models:
            del self.models[old_model]
        self.models[new_model_name] = new_model

        # Reset health for the new model
        self._health[new_model_name] = ModelHealth(model_type=new_model_name)
        if old_model in self._health:
            # Keep old health for reference but flag as rotated
            pass

        print(f"[ROTATE] Replaced {old_model} → {new_model_name}")

    def find_replacement(self, performance_matrix: Any,
                         old_model: str, regime: str) -> Optional[str]:
        """Find the best replacement model from a RegimeModelMatrix.

        Looks for the best model (by Sharpe) that isn't the old_model.
        """
        try:
            regime_idx = list(_REGIME_NAMES.values()).index(regime)
        except ValueError:
            regime_idx = 6

        models_list = list(performance_matrix.models)
        sharpe_col = performance_matrix.sharpe_matrix[:, regime_idx]

        scored = []
        for i, m in enumerate(models_list):
            if m != old_model and not np.isnan(sharpe_col[i]):
                scored.append((m, float(sharpe_col[i])))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None
