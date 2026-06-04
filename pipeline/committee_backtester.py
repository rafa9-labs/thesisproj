"""
Committee Backtester — Phase D of the Multi-Agent Autonomous Exploration Engine.

Takes a CommitteeConfig (from Phase C) and runs walk-forward evaluation with
per-bar regime routing, model selection, and weight blending.

At each bar:
  1. RegimeClassifier determines the current market regime
  2. CommitteeConfig maps regime → (model_list, weights)
  3. Each assigned model produces a probability prediction
  4. Predictions are blended via weighted average
  5. The blended signal is converted to a trade (-1, 0, 1)

Supports both synthetic data (for testing) and real OHLC data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.registry import build_model
from pipeline.committee_builder import CommitteeConfig
from pipeline.regime_utils import (
    detect_regimes,
    detect_regimes_anchored,
    attach_regime_columns,
    RegimeConfig,
    _REGIME_NAMES,
)


@dataclass
class CommitteeFoldResult:
    """Result of a single walk-forward fold evaluation."""
    fold_idx: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    sharpe: float
    trades: int
    active_rate: float
    win_rate: float
    return_val: float
    drawdown: float
    regime_distribution: Dict[str, float] = field(default_factory=dict)
    per_model_active_fraction: Dict[str, float] = field(default_factory=dict)
    num_nan_predictions: int = 0


@dataclass
class CommitteeBacktestResult:
    """Aggregate backtest results for a committee config."""
    config: CommitteeConfig
    folds: List[CommitteeFoldResult]
    models: List[str]
    avg_sharpe: float = 0.0
    avg_trades: float = 0.0
    total_folds: int = 0
    execution_time_s: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def fold_consistency_cv(self) -> float:
        """Coefficient of Variation of fold Sharpes.

        CV = std(fold Sharpes) / |mean(fold Sharpe)|.
        A CV >= 1.0 indicates the committee's performance is dominated by
        a single windfall fold — a red flag for overfitting.
        Returns inf if mean is near zero or fewer than 3 folds exist.
        """
        sharpes = [f.sharpe for f in self.folds if np.isfinite(f.sharpe)]
        if len(sharpes) < 3:
            return float("inf")
        mu = float(np.mean(sharpes))
        if abs(mu) < 1e-8:
            return float("inf")
        return float(np.std(sharpes, ddof=1) / abs(mu))

    @property
    def fold_consistency_pass(self) -> bool:
        return self.fold_consistency_cv < 1.0

    @property
    def per_regime_summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate per-regime metrics across all folds.

        Returns a dict mapping regime_name -> {
            "sharpe": weighted average of fold Sharpes by active fraction,
            "trades": total trades attributed to this regime,
            "folds_active": number of folds where this regime was active,
        }
        """
        regime_data: Dict[str, list] = {}
        for fold in self.folds:
            frac = getattr(fold, "per_model_active_fraction", {}) or {}
            for rname, active_frac in frac.items():
                if active_frac <= 0:
                    continue
                if rname not in regime_data:
                    regime_data[rname] = []
                regime_data[rname].append({
                    "sharpe": fold.sharpe if np.isfinite(fold.sharpe) else 0.0,
                    "trades": int(fold.trades) * active_frac,
                    "active_frac": active_frac,
                })

        summary: Dict[str, Dict[str, float]] = {}
        for rname, entries in regime_data.items():
            total_frac = sum(e["active_frac"] for e in entries)
            if total_frac <= 0:
                continue
            weighted_sharpe = sum(e["sharpe"] * e["active_frac"] for e in entries) / total_frac
            total_trades = sum(e["trades"] for e in entries)
            summary[rname] = {
                "sharpe": round(float(weighted_sharpe), 4),
                "trades": int(round(total_trades, 0)),
                "folds_active": len(entries),
            }
        return summary

    def regime_coverage_report(
        self, min_trades: int = 30, min_sharpe: float = 0.0,
    ) -> Dict[str, Dict]:
        """Evaluate whether each regime meets minimum coverage standards.

        Parameters
        ----------
        min_trades : int
            Minimum number of trades across all folds to consider a regime
            adequately covered (default 30).
        min_sharpe : float
            Minimum weighted Sharpe to consider a regime viable (default 0.0).

        Returns
        -------
        dict
            Mapping regime_name -> {
                "sharpe": float, "trades": int, "folds_active": int,
                "covered": bool (True if both thresholds met),
            }
        """
        summary = self.per_regime_summary
        report: Dict[str, Dict] = {}
        for rname, data in summary.items():
            covered = (
                data["trades"] >= min_trades
                and data["sharpe"] > min_sharpe
            )
            report[rname] = {**data, "covered": covered}
        return report

    @property
    def all_regimes_covered(self) -> bool:
        """True iff every regime in the config has positive coverage."""
        # Only report config regimes, not detected-only regimes
        report = self.regime_coverage_report()
        config_regimes = set(self.config.regimes.keys())
        report_regimes = set(report.keys())
        # Every config regime must appear and be covered
        for rname in config_regimes:
            if rname not in report or not report[rname]["covered"]:
                return False
        return True

    def to_summary_dict(self) -> dict:
        return {
            "models": self.models,
            "folds": self.total_folds,
            "avg_sharpe": round(self.avg_sharpe, 4),
            "avg_trades": round(self.avg_trades, 1),
            "execution_time_s": round(self.execution_time_s, 1),
            "regimes_configured": len(self.config.regimes),
            "fold_consistency_cv": round(self.fold_consistency_cv, 4),
            "fold_consistency_pass": bool(np.isfinite(self.fold_consistency_cv)) and self.fold_consistency_pass,
            "all_regimes_covered": self.all_regimes_covered,
            "warnings": self.warnings,
        }


class CommitteeBacktester:
    """Walk-forward backtester for model committees.

    Parameters
    ----------
    config : CommitteeConfig
        Per-regime model assignments with blending weights.
    regime_cfg : RegimeConfig
        Thresholds for 7-class regime detection.
    confidence_threshold : float
        Minimum blended probability to generate a signal.
    label_threshold : float
        Return threshold for labeling (next-bar threshold).
    model_params : Optional[Dict[str, Dict]]
        Per-model hyperparameters from HPO tuning.
    seq_len : int
        Lookback window size for deep/sequence models (default 30).
    """

    _DEEP_MODEL_TYPES: frozenset = frozenset({
        "cnn", "lstm", "transformer", "gru", "gru_lstm",
    })
    _ENSEMBLE_ADAPTIVE = "ensemble_adaptive_regime"
    _SEQ_FEATURE_COLS = ["mid_o", "mid_h", "mid_l", "mid_c", "returns"]

    def __init__(
        self,
        config: CommitteeConfig,
        regime_clf=None,
        regime_cfg: RegimeConfig = None,
        confidence_threshold: float = 0.6,
        label_threshold: float = 0.0001,
        model_params: Optional[Dict[str, Dict]] = None,
        seq_len: int = 30,
    ):
        self.config = config
        self._regime_clf = regime_clf
        self.regime_cfg = regime_cfg or RegimeConfig()
        self.confidence_threshold = confidence_threshold
        self.label_threshold = label_threshold
        self.model_params = model_params or {}
        self.seq_len = int(seq_len)
        self._trained_models: Dict[str, Any] = {}
        self._regime_clf: Optional[Any] = None
        self._feature_names: List[str] = []
        self._n_seq_features: int = 0

    # ── Main entry point ─────────────────────────────────────────────

    def run_wfo(
        self,
        df: pd.DataFrame,
        train_months: int = 12,
        test_months: int = 1,
        verbose: bool = True,
    ) -> CommitteeBacktestResult:
        """Run walk-forward evaluation of the committee.

        Parameters
        ----------
        df : pd.DataFrame
            OHLC data with columns: mid_c, mid_h, mid_l, mid_o, spread.
            Must have a datetime index.
        train_months : int
            Months of training data per fold.
        test_months : int
            Months of test data per fold.
        verbose : bool
            Print progress.

        Returns
        -------
        CommitteeBacktestResult
        """
        t0 = time.time()
        warnings: List[str] = []

        # Compute features on full dataframe (indicators needed for model training)
        df_full = self._prepare_features(df)

        # Generate WFO splits by month
        splits = self._make_monthly_splits(df_full, train_months, test_months)

        if verbose:
            total_models = len(self.config.all_models())
            print(f"\n[COMMITTEE] WFO: {len(splits)} folds, "
                  f"{train_months}/{test_months} months, "
                  f"{total_models} models in committee")

        fold_results: List[CommitteeFoldResult] = []
        for fold_idx, (train_slice, test_slice) in enumerate(splits):
            try:
                result = self._evaluate_fold(
                    fold_idx, train_slice, test_slice
                )
                if result is not None:
                    fold_results.append(result)
                    if verbose and fold_idx % max(1, len(splits) // 4) == 0:
                        print(f"  Fold {fold_idx}: Sharpe={result.sharpe:.3f}, "
                              f"trades={result.trades}")
            except Exception as e:
                warnings.append(f"Fold {fold_idx}: {e}")

        if not fold_results:
            raise RuntimeError("All folds failed. Check data and committee config.")

        sharpe_vals = [f.sharpe for f in fold_results if not np.isnan(f.sharpe)]
        trade_vals = [f.trades for f in fold_results]

        elapsed = time.time() - t0
        return CommitteeBacktestResult(
            config=self.config,
            folds=fold_results,
            models=self.config.all_models(),
            avg_sharpe=float(np.mean(sharpe_vals)) if sharpe_vals else 0.0,
            avg_trades=float(np.mean(trade_vals)) if trade_vals else 0.0,
            total_folds=len(fold_results),
            execution_time_s=elapsed,
            warnings=warnings,
        )

    # ── Feature preparation ─────────────────────────────────────────
    #
    # NOTE: These feature computations intentionally mirror logic from
    # pipeline/backtester/features_mixin.py. The committee backtester is
    # self-contained for synthetic-OHLC testing (no pipeline dependency).
    # Future: extract a shared FeatureComputer utility.

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicator features needed for regime detection and model training."""
        out = df.copy()
        # Returns
        out["returns"] = np.log(out["mid_c"] / out["mid_c"].shift(1)).astype(np.float32)

        # EMA
        out["ema_20"] = out["mid_c"].ewm(span=20, adjust=False).mean().astype(np.float32)

        # SMA
        out["sma_20"] = out["mid_c"].rolling(20).mean().astype(np.float32)

        # ADX (simplified)
        out["adx_14"] = self._compute_simplified_adx(out, window=14)

        # RSI
        out["rsi_14"] = self._compute_simplified_rsi(out, window=14)

        # Bollinger Bands
        bb_sma = out["mid_c"].rolling(20).mean()
        bb_std = out["mid_c"].rolling(20).std()
        out["bb_upper"] = (bb_sma + 2.0 * bb_std).astype(np.float32)
        out["bb_lower"] = (bb_sma - 2.0 * bb_std).astype(np.float32)
        out["bbw"] = ((out["bb_upper"] - out["bb_lower"]) / out["mid_c"]).astype(np.float32)
        out["bb_pct"] = (
            (out["mid_c"] - out["bb_lower"]) /
            (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
        ).astype(np.float32)

        # ATR
        hl = out["mid_h"] - out["mid_l"]
        out["atr_14"] = hl.rolling(14).mean().astype(np.float32)

        # Realized Vol
        out["rv_48"] = (out["returns"].pow(2).rolling(48).sum().pow(0.5)).astype(np.float32)

        # MACD
        ema12 = out["mid_c"].ewm(span=12, adjust=False).mean()
        ema26 = out["mid_c"].ewm(span=26, adjust=False).mean()
        out["macd_diff"] = (ema12 - ema26).astype(np.float32)

        # Donchian
        out["donchian_up_20"] = out["mid_h"].rolling(20).max().astype(np.float32)
        out["donchian_dn_20"] = out["mid_l"].rolling(20).min().astype(np.float32)
        out["donchian_break_up_20"] = (
            (out["mid_c"] > out["donchian_up_20"].shift(1)).astype(np.int8)
        )
        out["donchian_break_dn_20"] = (
            (out["mid_c"] < out["donchian_dn_20"].shift(1)).astype(np.int8)
        )

        return out

    def _build_sequences(self, df: pd.DataFrame, seq_len: int) -> np.ndarray:
        """Build (n - seq_len + 1, seq_len, n_features) sliding windows.

        Uses OHLC + returns columns for sequence input to deep models.
        Data is normalized per column (zero-mean, unit-variance).
        Returns empty array (0, seq_len, 1) if not enough bars.
        """
        cols = [c for c in self._SEQ_FEATURE_COLS if c in df.columns]
        if not cols:
            return np.empty((0, seq_len, 1), dtype=np.float32)
        data = df[cols].copy()
        data = data.ffill().fillna(0.0).to_numpy(np.float32)
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-10
        data = (data - mean) / std
        n = data.shape[0]
        if n < seq_len:
            return np.empty((0, seq_len, data.shape[1]), dtype=np.float32)
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(data, (seq_len, 1)).reshape(
            n - seq_len + 1, seq_len, data.shape[1])

    # ── Static helpers ───────────────────────────────────────────────

    @staticmethod
    def _compute_simplified_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Simplified ADX — directional movement based."""
        high, low = df["mid_h"].astype(np.float64), df["mid_l"].astype(np.float64)
        up_move = high.diff()
        down_move = -low.diff()
        up_move = up_move.clip(lower=0)
        down_move = down_move.clip(lower=0)

        atr = (high - low).abs().rolling(window, min_periods=1).mean()
        atr = atr.replace(0, np.nan)

        up_smooth = up_move.ewm(alpha=1.0 / window, adjust=False).mean()
        down_smooth = down_move.ewm(alpha=1.0 / window, adjust=False).mean()

        pdi = 100.0 * up_smooth / atr
        mdi = 100.0 * down_smooth / atr
        dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        adx = dx.ewm(alpha=1.0 / window, adjust=False).mean()

        return adx.astype(np.float32)

    @staticmethod
    @staticmethod
    def _compute_simplified_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Simplified RSI."""
        delta = df["mid_c"].diff().astype(np.float64)
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi.astype(np.float32)

    # ── Regime classifier training ──────────────────────────────────

    def _train_regime_classifier(self, df: pd.DataFrame) -> Any:
        """Train a RandomForest regime classifier on labeled data.

        Falls back gracefully if regime_7class column is not present.
        """
        from models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier(
            n_estimators=50, max_depth=6, min_samples_leaf=30,
            random_state=42,
        )
        if "regime_7class" not in df.columns:
            return clf  # return untrained — predict_regimes falls back to rule-based
        clf.fit(df)
        return clf

    # ── WFO split generation ────────────────────────────────────────

    def _make_monthly_splits(
        self,
        df: pd.DataFrame,
        train_months: int,
        test_months: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate (train, test) slices by shifting one month at a time."""
        if df.empty:
            return []

        df_sorted = df.sort_index()
        start_ts = df_sorted.index[0]
        end_ts = df_sorted.index[-1]

        splits = []
        current = start_ts + pd.DateOffset(months=train_months)

        while current + pd.DateOffset(months=test_months) <= end_ts:
            train_end = current
            test_end = current + pd.DateOffset(months=test_months)

            train_slice = df_sorted.loc[start_ts:train_end].iloc[:-1]
            test_slice = df_sorted.loc[train_end:test_end]

            if len(train_slice) >= 50 and len(test_slice) >= 10:
                splits.append((train_slice, test_slice))

            current += pd.DateOffset(months=test_months)

        return splits

    # ── Single fold evaluation ──────────────────────────────────────

    def _evaluate_fold(
        self,
        fold_idx: int,
        train_slice: pd.DataFrame,
        test_slice: pd.DataFrame,
    ) -> Optional[CommitteeFoldResult]:
        """Train committee models on train_slice, evaluate on test_slice with regime routing."""

        # ── 1. Training ──
        self._trained_models = {}
        unique_models = self.config.all_models()

        # Feature columns for training (numeric, no regime/prediction columns)
        exclude = {
            "regime_7class", "regime_name", "regime_id",
            "regime_trend", "regime_sideways", "regime_volatile",
            "time", "returns", "spread", "label",
        }
        feat_cols = [
            c for c in train_slice.columns
            if c not in exclude and np.issubdtype(train_slice[c].dtype, np.number)
        ]

        X_train = train_slice[feat_cols].copy()
        y_train = self._make_labels(train_slice)

        # Align y_train to X_train index (labels are computed on the full slice)
        y_train = y_train.reindex(X_train.index)

        # Drop rows with NaN in features or labels
        valid = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train[valid].to_numpy(np.float32)
        y_train = y_train[valid].to_numpy(np.int32)

        if len(X_train) < 50:
            return None

        # Detect if any committee model needs sequence input
        _needs_seq = any(
            m in self._DEEP_MODEL_TYPES or m == self._ENSEMBLE_ADAPTIVE
            for m in unique_models
        )
        X_seq_train = None
        X_flat_train_aligned = X_train
        y_train_aligned = y_train

        if _needs_seq:
            X_seq_full = self._build_sequences(train_slice, self.seq_len)
            if X_seq_full.shape[0] > 0:
                self._n_seq_features = X_seq_full.shape[2]
                n_seq = X_seq_full.shape[0]
                X_seq_train = X_seq_full
                X_flat_train_aligned = X_train[-n_seq:]
                y_train_aligned = y_train[-n_seq:]
            else:
                _needs_seq = False

        for model_type in unique_models:
            try:
                model_params = self.model_params.get(model_type, {})
                model = self._build_model(model_type, n_features=X_train.shape[1],
                                          params=model_params)

                if _needs_seq and model_type in self._DEEP_MODEL_TYPES:
                    model.fit(X_seq_train, y_train_aligned)
                elif _needs_seq and model_type == self._ENSEMBLE_ADAPTIVE:
                    model.fit(X_seq_train, X_flat_train_aligned, y_train_aligned)
                else:
                    model.fit(X_flat_train_aligned, y_train_aligned)

                self._trained_models[model_type] = model
            except Exception:
                self._trained_models[model_type] = None

        # ── 2. Prediction ──
        X_test = test_slice[feat_cols].fillna(0.0).to_numpy(np.float32)
        X_seq_test = self._build_sequences(test_slice, self.seq_len) if _needs_seq else None

        # Classify regime per bar (per-fold anchored GMM)
        regime_ids = self._predict_regimes(test_slice, df_train=train_slice, fold_idx=fold_idx)

        # Get blended predictions
        blended_probs = self._blend_predictions(X_test, regime_ids, X_seq=X_seq_test)
        preds = self._proba_to_trade(blended_probs)

        # ── 3. Evaluation ──
        eval_df = test_slice[feat_cols].copy()
        eval_df["returns"] = test_slice["returns"].values
        eval_df["spread"] = test_slice.get("spread", pd.Series(0.0, index=test_slice.index)).values

        # Align eval_df to preds length (seq features trim leading warmup bars)
        if _needs_seq and X_seq_test is not None and len(preds) < len(eval_df):
            eval_df = eval_df.iloc[-len(preds):]
            regime_ids = regime_ids[-len(preds):]

        eval_df["pred"] = preds

        # Fill NaN predictions
        nan_count = int(np.isnan(preds).sum())
        eval_df["pred"] = eval_df["pred"].fillna(0.0)

        metrics = self._compute_metrics(eval_df)

        # Regime distribution
        regime_dist = {}
        unique_reg, counts = np.unique(regime_ids, return_counts=True)
        for rid, cnt in zip(unique_reg, counts):
            name = _REGIME_NAMES.get(int(rid), f"regime_{rid}")
            regime_dist[name] = float(cnt) / float(len(regime_ids))

        # Per-model active fraction
        per_model_frac = {}
        for rname, assignment in self.config.regimes.items():
            per_model_frac[rname] = float(
                regime_dist.get(rname, 0.0)
            )

        if metrics is None:
            return None

        sharpe, trades, active_rate, win_rate, ret_val, dd = metrics
        return CommitteeFoldResult(
            fold_idx=fold_idx,
            train_start=train_slice.index[0],
            train_end=train_slice.index[-1],
            test_start=test_slice.index[0],
            test_end=test_slice.index[-1],
            sharpe=float(sharpe) if sharpe is not None else np.nan,
            trades=int(trades) if trades is not None else 0,
            active_rate=float(active_rate) if active_rate is not None else 0.0,
            win_rate=float(win_rate) if win_rate is not None else np.nan,
            return_val=float(ret_val) if ret_val is not None else np.nan,
            drawdown=float(dd) if dd is not None else np.nan,
            regime_distribution=regime_dist,
            per_model_active_fraction=per_model_frac,
            num_nan_predictions=nan_count,
        )

    # ── Labeling ────────────────────────────────────────────────────

    def _make_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create 3-class labels from next-bar returns.

        Returns NaN for the last bar (no next return available).
        """
        returns = df["returns"].shift(-1)
        labels = pd.Series(1.0, index=df.index, dtype=np.float64)  # flat
        labels[returns > self.label_threshold] = 2.0   # long
        labels[returns < -self.label_threshold] = 0.0   # short
        labels.iloc[-1] = np.nan  # last bar has no next return
        return labels

    # ── Model building ──────────────────────────────────────────────

    def _build_model(self, model_type: str, n_features: int = 20,
                     params: Optional[Dict[str, Any]] = None) -> Any:
        """Build a model for committee training.

        Uses sklearn implementations directly for standard models.
        Supports stacking_ensemble and meta_ensemble via their BaseModel interface.
        Optional params override defaults (for HPO-tuned hyperparameters).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        p = dict(params or {})

        if model_type == "logistic":
            return LogisticRegression(
                C=float(p.get("C", 1.0)), max_iter=int(p.get("max_iter", 500)),
                class_weight=p.get("class_weight", "balanced"),
                random_state=42, n_jobs=1,
            )
        elif model_type == "svm":
            from sklearn.svm import SVC
            return SVC(
                C=float(p.get("C", 1.0)), kernel=p.get("kernel", "rbf"),
                gamma=p.get("gamma", "scale"),
                probability=True, class_weight=p.get("class_weight", "balanced"),
                random_state=42,
            )
        elif model_type in ("random_forest", "rf"):
            return RandomForestClassifier(
                n_estimators=int(p.get("n_estimators", 50)),
                max_depth=int(p.get("max_depth", 6)) if p.get("max_depth") else 6,
                min_samples_leaf=int(p.get("min_samples_leaf", 10)),
                class_weight=p.get("class_weight", "balanced_subsample"),
                random_state=42, n_jobs=1,
            )
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=int(p.get("n_estimators", 50)),
                    max_depth=int(p.get("max_depth", 4)),
                    learning_rate=float(p.get("learning_rate", 0.1)),
                    objective="multi:softprob", num_class=3,
                    random_state=42, n_jobs=1, verbosity=0,
                )
            except ImportError:
                return RandomForestClassifier(
                    n_estimators=50, max_depth=6, n_jobs=1, random_state=42,
                )
        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=int(p.get("n_estimators", 50)),
                    max_depth=int(p.get("max_depth", 5)),
                    random_state=42, n_jobs=1, verbose=-1,
                )
            except ImportError:
                return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        elif model_type == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(
                max_depth=int(p.get("max_depth", 6)),
                min_samples_leaf=int(p.get("min_samples_leaf", 10)),
                class_weight=p.get("class_weight", "balanced"),
                random_state=42,
            )
        elif model_type == "stacking_ensemble":
            from models.stacking_ensemble import StackingEnsemble
            sub_types = p.get("stack_sub_models", ["logistic", "xgboost"])
            if isinstance(sub_types, str):
                sub_types = [t.strip() for t in sub_types.split(",") if t.strip()]
            return StackingEnsemble(
                sub_models=[build_model(t) for t in sub_types if build_model(t) is not None],
                stack_sub_models=sub_types,
                method=str(p.get("stack_method", "auto")),
                cv=int(p.get("stack_cv", 3)),
                seed=42,
            )
        elif model_type == "meta_ensemble":
            from models.meta_ensemble import MetaEnsemble
            sub_types = p.get("meta_sub_models", ["logistic", "xgboost"])
            if isinstance(sub_types, str):
                sub_types = [t.strip() for t in sub_types.split(",") if t.strip()]
            method = str(p.get("meta_combination_method", "soft"))
            return MetaEnsemble(
                sub_models=[build_model(t) for t in sub_types if build_model(t) is not None],
                meta_sub_models=sub_types,
                method=method,
                seed=42,
            )
        elif model_type == "cnn":
            from models.cnn import build_cnn
            return build_cnn(
                input_shape=(self.seq_len, max(self._n_seq_features, 5)),
                config=dict(p),
            )
        elif model_type == "lstm":
            from models.lstm import build_lstm
            return build_lstm(
                input_shape=(self.seq_len, max(self._n_seq_features, 5)),
                config=dict(p),
            )
        elif model_type == "transformer":
            from models.transformer import build_transformer
            return build_transformer(
                input_shape=(self.seq_len, max(self._n_seq_features, 5)),
                config=dict(p),
            )
        elif model_type == "gru":
            from models.gru import build_gru
            return build_gru(
                input_shape=(self.seq_len, max(self._n_seq_features, 5)),
                config=dict(p),
            )
        elif model_type == "gru_lstm":
            from models.gru_lstm import build_gru_lstm
            return build_gru_lstm(
                input_shape=(self.seq_len, max(self._n_seq_features, 5)),
                config=dict(p),
            )
        elif model_type == "ensemble_adaptive_regime":
            from models.ensemble_adaptive_regime import AdaptiveRegimeStrategy
            return AdaptiveRegimeStrategy(
                input_shape=(self.seq_len, max(self._n_seq_features, 5)),
                adx_col=p.get("adx_col", "adx_14"),
                vol_col=p.get("vol_col", "rolling_std_20"),
                adx_thresh=float(p.get("adx_thresh", 25)),
                vol_thresh=float(p.get("vol_thresh", 0.002)),
            )
        else:
            return LogisticRegression(
                C=1.0, max_iter=300, class_weight="balanced",
                random_state=42, n_jobs=1,
            )

    # ── Regime routing ──────────────────────────────────────────────

    def _predict_regimes(self, df: pd.DataFrame, df_train: Optional[pd.DataFrame] = None,
                         fold_idx: int = 0) -> np.ndarray:
        """Classify each bar into a 7-class regime.

        Uses anchored GMM detection per fold when df_train is provided (per-fold
        WFO fitting). Falls back to single-fit anchored detection otherwise.
        """
        if df_train is not None:
            return detect_regimes_anchored(
                df, df_train=df_train, window=252,
                random_state=42 + fold_idx,
            )
        return detect_regimes_anchored(df, window=252, random_state=42)

    def _blend_predictions(
        self,
        X_test: np.ndarray,
        regime_ids: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-bar regime routing + model blending.

        For each bar:
          1. Look up regime → assignment in committee config
          2. Get predict_proba from each assigned model (seq or flat)
          3. Weighted average of probabilities
          4. Return (n_samples, 3) blended probability matrix
        """
        has_seq = X_seq is not None and X_seq.shape[0] > 0
        if has_seq:
            n_seq = X_seq.shape[0]
            X_test = X_test[-n_seq:]
            regime_ids = regime_ids[-n_seq:]

        n_samples = X_test.shape[0]
        blended = np.zeros((n_samples, 3), dtype=np.float64)

        for i in range(n_samples):
            regime_name = _REGIME_NAMES.get(int(regime_ids[i]), "sideways")
            assignment = self.config.regime_models(regime_name)

            if assignment is None or len(assignment.models) == 0:
                continue

            prob_sum = np.zeros(3, dtype=np.float64)
            weight_sum = 0.0

            for model_name, weight in zip(assignment.models, assignment.weights):
                model = self._trained_models.get(model_name)
                if model is None:
                    continue
                try:
                    if has_seq and model_name in self._DEEP_MODEL_TYPES:
                        proba = model.predict(X_seq[i:i + 1], verbose=0)
                    elif has_seq and model_name == self._ENSEMBLE_ADAPTIVE:
                        proba = model.predict_proba(
                            X_seq[i:i + 1], X_test[i:i + 1])
                    else:
                        proba = model.predict_proba(X_test[i:i + 1])

                    if proba is not None and proba.shape[1] >= 3:
                        prob_sum += weight * proba[0, :3]
                        weight_sum += weight
                except Exception:
                    continue

            if weight_sum > 0:
                blended[i] = prob_sum / weight_sum

        return blended

    def _proba_to_trade(self, proba: np.ndarray) -> np.ndarray:
        """Convert 3-class probabilities to trade signals (-1, 0, 1).

        Trade when max probability exceeds confidence threshold.
        """
        n = proba.shape[0]
        trades = np.zeros(n, dtype=np.float64)

        for i in range(n):
            p = proba[i]
            # p[0] = short, p[1] = flat, p[2] = long
            max_class = np.argmax(p)
            if p[max_class] >= self.confidence_threshold:
                if max_class == 2:
                    trades[i] = 1.0   # long
                elif max_class == 0:
                    trades[i] = -1.0  # short
                # class 1 (flat) → 0.0

        return trades

    # ── Metrics computation ─────────────────────────────────────────

    def _compute_metrics(self, df: pd.DataFrame) -> Optional[Tuple]:
        """Compute Sharpe, trades, active_rate, win_rate, return, drawdown."""
        if "returns" not in df.columns or "pred" not in df.columns:
            return None

        rets = df["returns"].values.astype(float)
        preds = df["pred"].values.astype(float)

        # Apply 1-bar execution delay
        preds_shifted = np.roll(preds, 1)
        preds_shifted[0] = 0.0

        strategy_returns = rets * preds_shifted

        # Filters: non-zero preds and finite returns
        mask = (preds_shifted != 0) & np.isfinite(strategy_returns)
        active_returns = strategy_returns[mask]

        n_bars = len(df)
        n_trades = int(mask.sum())
        active_rate = n_trades / max(1, n_bars)

        if n_trades < 2 or len(active_returns) < 2:
            return (0.0, n_trades, active_rate, np.nan, np.nan, 0.0)

        mean_ret = np.mean(active_returns)
        std_ret = np.std(active_returns, ddof=1)

        # Annualized Sharpe (assuming H1 bars ≈ 252*24 = 6048 per year)
        annual_factor = np.sqrt(6048.0)
        sharpe = (mean_ret / std_ret) * annual_factor if std_ret > 0 else 0.0

        win_rate = float((active_returns > 0).mean())

        # Cumulative return
        cum_ret = float(np.prod(1.0 + strategy_returns) - 1.0)

        # Max drawdown
        equity = np.cumprod(1.0 + strategy_returns)
        peak = np.maximum.accumulate(equity)
        drawdown = float(np.max((peak - equity) / peak)) if len(equity) > 0 else 0.0

        return (sharpe, n_trades, active_rate, win_rate, cum_ret, drawdown)

    # ── Summary ─────────────────────────────────────────────────────

    def print_summary(self, result: CommitteeBacktestResult):
        """Print human-readable backtest summary."""
        print("\n" + "=" * 72)
        print("  COMMITTEE BACKTESTER — Walk-Forward Validation")
        print("=" * 72)
        print(f"\n  Models: {', '.join(result.models)}")
        print(f"  Folds: {result.total_folds}")
        print(f"  Avg Sharpe: {result.avg_sharpe:.3f}")
        print(f"  Avg Trades: {result.avg_trades:.0f}")
        print(f"  Time: {result.execution_time_s:.0f}s")

        if result.warnings:
            print(f"\n  Warnings ({len(result.warnings)}):")
            for w in result.warnings[:5]:
                print(f"    - {w}")

        print("\n  ── Per-Fold ──")
        for f in result.folds:
            print(f"    Fold {f.fold_idx:2d}: SR={f.sharpe:+.3f}  trades={f.trades:3d}  "
                  f"active={f.active_rate:.3f}  wr={f.win_rate:.2%}")

        if result.folds:
            regimes_used = set()
            for f in result.folds:
                regimes_used.update(f.regime_distribution.keys())
            print(f"\n  Regimes seen: {sorted(regimes_used)}")
            # Show top 2 folds by Sharpe
            sorted_folds = sorted(result.folds, key=lambda f: f.sharpe, reverse=True)
            for best in sorted_folds[:2]:
                top_r = sorted(
                    best.regime_distribution.items(),
                    key=lambda x: x[1], reverse=True
                )[:3]
                regime_str = " ".join(f"{r}={f:.0%}" for r, f in top_r)
                print(f"    Best fold #{best.fold_idx}: {regime_str}")

        print("\n" + "=" * 72)
