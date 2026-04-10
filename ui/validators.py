"""
Parameter validation guards for the FX ML Backtester UI.

Prevents user-configurable parameters from contradicting the
quant pipeline execution order:
  1. Load data → 2. Engineer features → 3. Generate labels
  4. Walk-forward split → 5. HPO (Optuna CV) → 6. Final evaluation
"""

from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ValidationResult:
    """Accumulates warnings and errors from parameter validation."""
    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def warn(self, msg: str):
        self.warnings.append(msg)

    def error(self, msg: str):
        self.errors.append(msg)

    def merge(self, other: "ValidationResult"):
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)


def validate_all(params: Dict[str, Any]) -> ValidationResult:
    """Run all validation guards. Returns a ValidationResult."""
    r = ValidationResult()
    r.merge(validate_triple_barrier(params))
    r.merge(validate_features(params))
    r.merge(validate_coverage(params))
    r.merge(validate_model(params))
    r.merge(validate_calibrate(params))
    return r


# ── Triple Barrier & Labels ──────────────────────────────────────────

def validate_triple_barrier(params: Dict[str, Any]) -> ValidationResult:
    """Guard: TB multipliers and label threshold must produce meaningful labels."""
    r = ValidationResult()

    if not params.get("use_triple_barrier", True):
        return r

    pt = float(params.get("tb_pt_mult", 2.0))
    sl = float(params.get("tb_sl_mult", 2.0))
    nz = float(params.get("tb_neutral_zone", 0.5))
    mh = int(params.get("tb_max_holding", 36))
    lt = float(params.get("label_threshold", 0.0005))

    # Guard 1: PT < SL means asymmetric risk-reward (not necessarily wrong, but unusual)
    if pt < sl:
        r.warn(
            f"PT mult ({pt}) < SL mult ({sl}) — strategy risks more than it aims to gain per trade. "
            "This can work but typically reduces win rate."
        )

    # Guard 2: Neutral zone too high → everything times out
    if nz > 1.0:
        r.warn(
            f"tb_neutral_zone={nz:.2f} is high — many trades may expire at the neutral zone "
            "instead of hitting PT/SL, producing mostly class-0 labels."
        )

    # Guard 3: Max holding too short for the timeframe
    if mh < 12:
        r.warn(
            f"tb_max_holding={mh} is very short — may not give trades enough time to reach PT/SL."
        )

    # Guard 4: Label threshold extremes
    if lt < 0.0002:
        r.warn(
            f"label_threshold={lt:.6f} is very low — may label noise as directional, "
            "producing excessive signals with low quality."
        )
    elif lt > 0.001:
        r.warn(
            f"label_threshold={lt:.6f} is high — may produce very few directional labels, "
            "leading to class imbalance (mostly class 0 / timeout)."
        )

    return r


# ── Features & Indicators ────────────────────────────────────────────

def validate_features(params: Dict[str, Any]) -> ValidationResult:
    """Guard: indicator toggles must be internally consistent."""
    r = ValidationResult()

    lags = int(params.get("lags", 14))
    lag_depth = int(params.get("lag_depth", 1))
    feature_count = lags * lag_depth

    # Guard: feature explosion
    if feature_count > 100:
        r.error(
            f"lags ({lags}) × lag_depth ({lag_depth}) = {feature_count} features. "
            f"This exceeds 100 and will likely cause overfitting. "
            f"Reduce lags or lag_depth."
        )
    elif feature_count > 60:
        r.warn(
            f"lags ({lags}) × lag_depth ({lag_depth}) = {feature_count} features. "
            f"Consider reducing for better generalization."
        )

    # Guard: squeeze requires BBands
    if params.get("use_squeeze_breakout") or params.get("use_squeeze_expansion"):
        if not params.get("use_bbands", True):
            r.error(
                "Squeeze detection requires Bollinger Bands (use_bbands=True). "
                "Enable BBands or disable squeeze features."
            )

    # Guard: MACD ATR ratio needs both MACD and ATR
    if params.get("use_macd_atr_ratio", False):
        if not params.get("use_macd", True):
            r.error("use_macd_atr_ratio requires use_macd to be enabled.")
        if not params.get("use_atr", True):
            r.error("use_macd_atr_ratio requires use_atr to be enabled.")

    # Guard: price_ma_z needs at least one MA
    if params.get("use_price_ma_z", False):
        if not params.get("use_sma", True) and not params.get("use_ema", True):
            r.error("use_price_ma_z requires at least one of use_sma or use_ema.")

    # Guard: MTF alignment needs MTF MA
    if params.get("use_mtf_alignment", False) and not params.get("use_mtf_ma", True):
        r.error("use_mtf_alignment requires use_mtf_ma to be enabled.")

    # Guard: fracdiff range
    fd = float(params.get("fracdiff_d", 0.4))
    if params.get("use_fracdiff", True) and (fd < 0.0 or fd > 1.0):
        r.error(f"fracdiff_d must be between 0.0 and 1.0, got {fd}.")

    return r


# ── Coverage Targets ────────────────────────────────────────────────

def validate_coverage(params: Dict[str, Any]) -> ValidationResult:
    """Guard: coverage targets must be consistent."""
    r = ValidationResult()

    tar = float(params.get("target_active_rate", 0.15))
    tc = float(params.get("target_coverage", 0.15))

    # Guard: targets should match
    if abs(tar - tc) > 0.01:
        r.warn(
            f"target_active_rate ({tar:.3f}) != target_coverage ({tc:.3f}). "
            f"They will be synced to target_active_rate={tar:.3f} at runtime."
        )

    # Guard: extreme coverage
    if tar > 0.30:
        r.warn(
            f"target_active_rate={tar:.3f} is very high — the strategy will trade very "
            f"frequently, increasing costs and reducing selectivity."
        )
    if tar < 0.05:
        r.warn(
            f"target_active_rate={tar:.3f} is very low — may produce too few trades "
            f"for statistically meaningful evaluation."
        )

    return r


# ── Model-Specific ──────────────────────────────────────────────────

def validate_model(params: Dict[str, Any]) -> ValidationResult:
    """Guard: model-specific parameter compatibility."""
    r = ValidationResult()
    model = str(params.get("model_type", "logistic")).lower()

    if model == "logistic":
        solver = str(params.get("logit_solver", "lbfgs"))
        penalty = str(params.get("logit_penalty", "l2"))

        # solver ↔ penalty compatibility (sklearn)
        _compat = {
            "lbfgs":    {"l2", "none"},
            "newton-cg": {"l2", "none"},
            "sag":      {"l2", "none"},
            "saga":     {"l1", "l2", "elasticnet", "none"},
            "liblinear": {"l1", "l2"},
        }
        allowed = _compat.get(solver, {"l2"})
        if penalty not in allowed:
            r.error(
                f"logit_solver '{solver}' does not support penalty '{penalty}'. "
                f"Allowed penalties for {solver}: {sorted(allowed)}."
            )

        C = float(params.get("logit_C", 1.0))
        if C < 0.001:
            r.warn(f"logit_C={C:.4f} is very small — very strong regularization.")
        elif C > 10000:
            r.warn(f"logit_C={C:.1f} is very large — almost no regularization.")

    return r


# ── Calibration ─────────────────────────────────────────────────────

def validate_calibrate(params: Dict[str, Any]) -> ValidationResult:
    """Guard: calibration method must be valid."""
    r = ValidationResult()
    cal = str(params.get("calibrate_method", "sigmoid")).strip().lower()
    if cal not in ("sigmoid", "isotonic"):
        r.error(
            f"calibrate_method must be 'sigmoid' or 'isotonic', got '{cal}'."
        )
    return r