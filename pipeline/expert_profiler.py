"""
Expert-Performance Profiler — Phase B of the Multi-Agent Autonomous Exploration Engine.

Runs a sweep of all models on the same WFO splits, tags each fold with its
7-class regime distribution, and builds a regime × model performance matrix.

Key outputs:
  - Regime × Model matrix (e.g., Sharpe per model per regime)
  - Paired t-test significance per regime (which models are statistically superior)
  - Model ranking per regime
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.model_families import (
    DEEP_MODELS,
    LINEAR_MODELS,
    TREE_MODELS,
)
from pipeline.regime_utils import (
    _REGIME_NAMES,
    RegimeConfig,
)


@dataclass
class FoldResult:
    """Per-fold data collected from one model's WFO run."""
    model: str
    fold_idx: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    sharpe: float
    trades: int
    active_rate: float
    win_rate: float
    performance: float
    return_val: float
    drawdown: float
    geo_mean_ann: float
    directional_accuracy: float
    f1_macro: float
    param_summary: Dict[str, Any] = field(default_factory=dict)

    # Filled after regime distribution is computed
    regime_counts: Dict[str, int] = field(default_factory=dict)
    dominant_regime: str = ""


@dataclass
class RegimeModelMatrix:
    """Performance matrix: rows=models, columns=regimes, values=metric."""
    regimes: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    sharpe_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    trade_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    hitrate_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    fold_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    raw_folds: List[FoldResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "regimes": self.regimes,
            "models": self.models,
            "sharpe": self.sharpe_matrix.tolist(),
            "trades": self.trade_matrix.tolist(),
            "hit_rate": self.hitrate_matrix.tolist(),
            "fold_counts": self.fold_counts.tolist(),
        }

    def top_model_per_regime(self, top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        out = {}
        for r_idx, regime in enumerate(self.regimes):
            scores = list(zip(self.models, self.sharpe_matrix[:, r_idx]))
            scores.sort(key=lambda x: x[1], reverse=True)
            out[regime] = [(m, float(s)) for m, s in scores[:top_k] if not np.isnan(s)]
        return out


@dataclass
class ExpertProfileResult:
    """Complete profiling output."""
    models_run: List[str]
    total_folds: int
    matrix: RegimeModelMatrix
    significance: Dict[str, Dict[str, Any]]
    execution_time_seconds: float
    warnings: List[str] = field(default_factory=list)


class ExpertProfiler:
    """Orchestrates multi-model profiling on a shared walk-forward configuration.

    Parameters
    ----------
    data_config : dict
        Backtester configuration (symbol, date range, timeframe, DB path, etc.).
    wfo_config : dict
        Walk-forward config: n_months, train_months, test_months, hpo_mode, etc.
    regime_cfg : RegimeConfig or None
        Thresholds for 7-class regime detection.
    """

    def __init__(
        self,
        data_config: Optional[Dict[str, Any]] = None,
        wfo_config: Optional[Dict[str, Any]] = None,
        regime_cfg: Optional[RegimeConfig] = None,
    ):
        self.data_config = data_config or {}
        self.wfo_config = wfo_config or {}
        self.regime_cfg = regime_cfg or RegimeConfig()
        self._all_folds: List[FoldResult] = []

    def profile(
        self,
        models: List[str],
        n_months: int = 2,
        n_trials: int = 5,
        seed: int = 42,
        verbose: bool = True,
        progress_callback: Optional[Callable] = None,
        raw_df: Optional[pd.DataFrame] = None,
    ) -> ExpertProfileResult:
        """Run full profiling sweep across all models.

        Parameters
        ----------
        models : list[str]
            Model types to profile (e.g. ["logistic", "xgboost", "lstm"]).
        n_months : int
            Walk-forward months for the sweep.
        n_trials : int
            Optuna trials per model (keep low for broad sweep).
        seed : int
            Reproducibility seed.
        verbose : bool
            Print progress.
        progress_callback : Callable or None
            Optional callback(model, idx, total, status, sharpe) called per model.
        raw_df : pd.DataFrame, optional
            Full OHLC DataFrame with 'time' column for anchored regime detection.
            Columns required: mid_high, mid_low, mid_close, time (or mid_h/mid_l/mid_c).

        Returns
        -------
        ExpertProfileResult
        """
        t0 = time.time()
        warnings: List[str] = []

        self._raw_df = raw_df

        # Build common config from data_config + wfo_config
        config = dict(self.data_config)
        config.update(self.wfo_config)
        config.setdefault("n_trials", n_trials)
        config.setdefault("n_startup_trials", max(1, n_trials // 3))
        config.setdefault("n_months", n_months)

        # Also ensure the MLBacktester init params are in the config
        config.setdefault("symbol", self.data_config.get("symbol", "EURUSD"))

        all_fold_results: List[FoldResult] = []
        first_df_wfo: Optional[pd.DataFrame] = None

        for idx, model_type in enumerate(models):
            if verbose:
                print(f"\n[PROFILE] Running {model_type} ({len(all_fold_results)} folds so far)...")

            if progress_callback:
                progress_callback(model_type, idx + 1, len(models), "started")

            fold_results, df_wfo, _ = self._run_single_model(
                model_type, config, seed, verbose
            )

            if fold_results is None or len(fold_results) == 0:
                warnings.append(f"{model_type}: produced 0 valid folds, skipping")
                if progress_callback:
                    progress_callback(model_type, idx + 1, len(models), "failed")
                continue

            avg_sharpe = np.mean([
                f.sharpe for f in fold_results
                if f.sharpe is not None and not np.isnan(f.sharpe)
            ]) if fold_results else None

            if progress_callback:
                progress_callback(model_type, idx + 1, len(models), "done",
                                  float(avg_sharpe) if avg_sharpe is not None and not np.isnan(avg_sharpe) else None)

            # Keep the first model's df_wfo for fold boundary reference
            if first_df_wfo is None and df_wfo is not None:
                first_df_wfo = df_wfo

            all_fold_results.extend(fold_results)

        if not all_fold_results:
            raise RuntimeError("No models produced any valid fold results. Check data/config.")

        # Compute regime distributions per fold (using the first model's fold boundaries)
        self._attach_regime_distributions(all_fold_results, first_df_wfo, verbose=verbose)

        # Build the performance matrix
        matrix = self._build_matrix(all_fold_results)

        # Run statistical tests
        significance = self._run_significance_tests(all_fold_results, matrix)

        elapsed = time.time() - t0
        return ExpertProfileResult(
            models_run=[m for m in models if any(f.model == m for f in all_fold_results)],
            total_folds=len(set((f.model, f.fold_idx) for f in all_fold_results)),
            matrix=matrix,
            significance=significance,
            execution_time_seconds=elapsed,
            warnings=warnings,
        )

    def _run_single_model(
        self,
        model_type: str,
        config: dict,
        seed: int,
        verbose: bool,
    ) -> Tuple[Optional[List[FoldResult]], Optional[pd.DataFrame], Optional[Dict]]:
        """Run one model through the WFO pipeline and collect fold results + best params.

        Returns (fold_results, df_wfo, best_params).
        best_params is the unpinned HPO params dict (unprefixed) or None if no HPO ran.
        """
        from pipeline.backtester.composed import MLBacktester

        model_config = dict(config)
        model_config["model_type"] = model_type

        locked_features = model_config.pop("locked_features", None)

        bt = None
        best_params: Optional[Dict] = None
        try:
            bt = MLBacktester(
                symbol=model_config.get("symbol", "EURUSD"),
                start=model_config.get("start"),
                end=model_config.get("end"),
                model_type=model_type,
            )
            if locked_features:
                if isinstance(bt.features_config, dict):
                    bt.features_config["locked_features"] = list(locked_features)

            df_wfo, best_combo = bt.run_strategy(
                model_config,
                models_to_test=[model_type],
                n_trials=model_config.get("n_trials", 5),
                n_startup_trials=model_config.get("n_startup_trials", 2),
            )

            if best_combo and isinstance(best_combo, dict):
                best_params = _strip_model_prefix(best_combo, model_type)

            if df_wfo is None or df_wfo.empty:
                if verbose:
                    print(f"  [SKIP] {model_type}: no WFO results produced")
                return None, None, best_params

            # Extract fold results from df_wfo
            fold_results = self._extract_fold_results(model_type, df_wfo)

            if verbose:
                avg_sharpe = np.mean([
                    f.sharpe for f in fold_results
                    if f.sharpe is not None and not np.isnan(f.sharpe)
                ])
                total_trades = sum(f.trades for f in fold_results)
                print(f"  [OK] {model_type}: {len(fold_results)} folds, "
                      f"avg Sharpe={avg_sharpe:.3f}, total trades={total_trades}")

            return fold_results, df_wfo, best_params

        except Exception as e:
            if verbose:
                print(f"  [FAIL] {model_type}: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            return None, None, None
        finally:
            if bt is not None:
                try:
                    bt.free(release_data=True)
                except Exception:
                    pass
                del bt

    def _extract_fold_results(self, model_type: str, df_wfo: pd.DataFrame) -> List[FoldResult]:
        """Parse per-fold results from the WFO DataFrame."""
        results = []
        for idx, row in df_wfo.iterrows():
            try:
                sharpe = float(row.get("sharpe", np.nan) or np.nan)
            except (ValueError, TypeError):
                sharpe = np.nan
            try:
                trades = int(row.get("trades", 0) or 0)
            except (ValueError, TypeError):
                trades = 0
            try:
                active_rate = float(row.get("active_rate", 0.0) or 0.0)
            except (ValueError, TypeError):
                active_rate = 0.0
            try:
                win_rate = float(row.get("win_rate", np.nan) or np.nan)
            except (ValueError, TypeError):
                win_rate = np.nan

            results.append(FoldResult(
                model=model_type,
                fold_idx=idx,
                train_start=row.get("train_start"),
                train_end=row.get("train_end"),
                test_start=row.get("test_start"),
                test_end=row.get("test_end"),
                sharpe=sharpe,
                trades=trades,
                active_rate=active_rate,
                win_rate=win_rate,
                performance=float(row.get("performance", np.nan) or np.nan),
                return_val=float(row.get("return", np.nan) or np.nan),
                drawdown=float(row.get("drawdown", 0.0) or 0.0),
                geo_mean_ann=float(row.get("geo_mean_ann", np.nan) or np.nan),
                directional_accuracy=float(row.get("directional_accuracy", np.nan) or np.nan),
                f1_macro=float(row.get("f1_macro", np.nan) or np.nan),
            ))
        return results

    def _attach_regime_distributions(
        self,
        fold_results: List[FoldResult],
        ref_df_wfo: Optional[pd.DataFrame],
        verbose: bool = False,
    ):
        """Compute regime distribution for each fold's test period.

        Uses anchored GMM detection on raw_df if available (per-fold WFO fitting).
        Falls back to heuristic if no raw_df is provided.
        """
        if self._raw_df is not None:
            if verbose:
                print("[PROFILE] Using anchored GMM regime detection per fold")
            self._attach_regime_from_labels(fold_results, self._raw_df)
        else:
            if verbose:
                print("[PROFILE] No raw_df, using heuristic regime fallback")
            self._attach_regime_from_fallback(fold_results)

    def _attach_regime_from_labels(self, fold_results: List[FoldResult], raw_df: pd.DataFrame):
        """Per-fold anchored GMM regime detection using raw OHLC data.

        For each unique (test_start, test_end) fold boundary:
          1. Find bar indices in raw_df matching the time range
          2. Split into train (all bars before test_start) and test
          3. Call detect_regimes_anchored(df_test, df_train) for per-fold WFO fitting
          4. Count regime frequencies in test bars
          5. Assign regime_counts and dominant_regime to each FoldResult

        This ensures the GMM is fit only on data available before the test period,
        eliminating look-ahead bias.
        """
        from collections import defaultdict
        from pipeline.regime_utils import detect_regimes_anchored

        has_time = "time" in raw_df.columns
        if not has_time:
            self._attach_regime_from_fallback(fold_results)
            return

        time_series = pd.to_datetime(raw_df["time"])
        raw_n = len(raw_df)

        fold_groups: Dict[Tuple[str, str], List[FoldResult]] = defaultdict(list)
        for fr in fold_results:
            key = (str(fr.test_start), str(fr.test_end))
            fold_groups[key].append(fr)

        for (t_start_str, t_end_str), folds in fold_groups.items():
            try:
                t_start = pd.Timestamp(t_start_str)
                t_end = pd.Timestamp(t_end_str)
            except Exception:
                self._assign_heuristic_to_folds(folds)
                continue

            in_test = (time_series >= t_start) & (time_series <= t_end)
            test_indices = np.where(in_test)[0]
            if len(test_indices) == 0:
                self._assign_heuristic_to_folds(folds)
                continue

            test_start_idx = int(test_indices[0])
            test_end_idx = int(test_indices[-1])

            train_end_idx = test_start_idx - 1
            if train_end_idx < 50:
                self._assign_heuristic_to_folds(folds)
                continue

            df_train = raw_df.iloc[0:train_end_idx + 1]
            df_test = raw_df.iloc[test_start_idx:test_end_idx + 1]

            regime_ids = detect_regimes_anchored(
                df_test, df_train=df_train, window=252,
                random_state=42 + len(folds),
            )

            counts = np.bincount(regime_ids, minlength=7)
            names = {i: _REGIME_NAMES.get(i, f"unknown_{i}") for i in range(7)}
            reg_counts = {names[i]: int(c) for i, c in enumerate(counts)}

            dominant_idx = int(np.argmax(counts))
            dominant = names.get(dominant_idx, "sideways")

            for fr_item in folds:
                fr_item.regime_counts = dict(reg_counts)
                fr_item.dominant_regime = dominant

    def _assign_heuristic_to_folds(self, folds: List[FoldResult]):
        """Assign default regime labels via the heuristic fallback for a batch of folds."""
        for f_item in folds:
            da = f_item.directional_accuracy if not np.isnan(f_item.directional_accuracy) else 0.5
            trades = f_item.trades
            sharpe = f_item.sharpe if not np.isnan(f_item.sharpe) else 0.0
            weights = {
                "trend_up": max(0.0, 0.15 + sharpe * 0.08 + (da - 0.5) * 0.3),
                "trend_down": max(0.0, 0.10 + sharpe * 0.05),
                "mean_reverting": max(0.0, 0.15 - abs(sharpe) * 0.05 + min(trades / 50.0, 0.3)),
                "breakout": max(0.0, 0.05 + (sharpe > 0.3) * 0.15 + (trades > 20) * 0.10),
                "high_volatile": max(0.0, 0.10 + (sharpe < 0) * 0.15 + (
                    f_item.win_rate < 0.35
                    if f_item.win_rate is not None and not np.isnan(f_item.win_rate) else 0
                ) * 0.10),
                "quiet_squeeze": max(0.0, 0.20 - min(trades / 40.0, 0.20)),
                "sideways": 0.15,
            }
            total = sum(weights.values()) or 1.0
            weights = {k: v / total for k, v in weights.items()}
            bars_per = max(1, trades * 4)
            f_item.regime_counts = {k: int(v * bars_per) for k, v in weights.items()}
            f_item.dominant_regime = max(weights, key=weights.get)

    def _attach_regime_from_fallback(self, fold_results: List[FoldResult]):
        """Fallback: compute regime distributions from fold-level metrics.

        Uses a heuristic mapping based on model performance patterns:
        - High directional accuracy + high trades → likely trending
        - High trades + low win rate → likely volatile/choppy
        - Low trades → likely quiet/sideways
        - High sharpe → likely trending or breakout

        This is a proxy until we have per-bar regime tagging from
        the pipeline's own regime columns.
        """
        for fold in fold_results:
            da = fold.directional_accuracy if not np.isnan(fold.directional_accuracy) else 0.5
            trades = fold.trades
            sharpe = fold.sharpe if not np.isnan(fold.sharpe) else 0.0

            # Heuristic regime weights
            weights = {
                "trend_up": max(0.0, 0.15 + sharpe * 0.08 + (da - 0.5) * 0.3),
                "trend_down": max(0.0, 0.10 + sharpe * 0.05),
                "mean_reverting": max(0.0, 0.15 - abs(sharpe) * 0.05 + min(trades / 50.0, 0.3)),
                "breakout": max(0.0, 0.05 + (sharpe > 0.3) * 0.15 + (trades > 20) * 0.10),
                "high_volatile": max(0.0, 0.10 + (sharpe < 0) * 0.15 + (fold.win_rate < 0.35) * 0.10
                                     if fold.win_rate is not None and not np.isnan(fold.win_rate) else 0.10),
                "quiet_squeeze": max(0.0, 0.20 - min(trades / 40.0, 0.20)),
                "sideways": 0.15,
            }

            # Normalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            # Convert to integer counts (scale to total bars ~200 per month)
            bars_per_fold = max(1, trades * 4)  # rough estimate
            fold.regime_counts = {k: int(v * bars_per_fold) for k, v in weights.items()}
            fold.dominant_regime = max(weights, key=weights.get)

    def _build_matrix(self, all_folds: List[FoldResult]) -> RegimeModelMatrix:
        """Build the regime × model performance matrix.

        For each (model, regime) pair, compute the weighted average
        Sharpe ratio across folds, weighted by the regime's prevalence
        in each fold.
        """
        models = sorted(set(f.model for f in all_folds))
        regimes = list(_REGIME_NAMES.values())
        n_models = len(models)
        n_regimes = len(regimes)

        sharpe_mat = np.full((n_models, n_regimes), np.nan)
        trade_mat = np.full((n_models, n_regimes), np.nan)
        hitrate_mat = np.full((n_models, n_regimes), np.nan)
        fold_cnts = np.zeros((n_models, n_regimes), dtype=int)

        model_idx = {m: i for i, m in enumerate(models)}
        regime_idx = {r: i for i, r in enumerate(regimes)}

        # Group folds by model
        from collections import defaultdict
        model_folds: Dict[str, List[FoldResult]] = defaultdict(list)
        for f in all_folds:
            model_folds[f.model].append(f)

        for model, folds in model_folds.items():
            mi = model_idx[model]
            for regime, ri in regime_idx.items():
                weights = []
                sharpes = []
                trades_vals = []
                hitrates = []

                for fold in folds:
                    regime_count = fold.regime_counts.get(regime, 0)
                    if regime_count > 0 and not np.isnan(fold.sharpe):
                        weights.append(regime_count)
                        sharpes.append(fold.sharpe)
                        trades_vals.append(fold.trades)
                        if not np.isnan(fold.win_rate):
                            hitrates.append(fold.win_rate)

                if weights and sum(weights) > 0:
                    w = np.array(weights, dtype=float) / sum(weights)
                    sharpe_mat[mi, ri] = float(np.average(sharpes, weights=w))
                    trade_mat[mi, ri] = float(np.average(trades_vals, weights=w))
                    if hitrates:
                        hitrate_mat[mi, ri] = float(np.average(hitrates, weights=w[:len(hitrates)]))
                else:
                    # No regime prevalence in any fold → use unweighted mean
                    model_sharpes = [f.sharpe for f in folds if not np.isnan(f.sharpe)]
                    model_trades = [f.trades for f in folds]
                    if model_sharpes:
                        sharpe_mat[mi, ri] = float(np.mean(model_sharpes))
                    if model_trades:
                        trade_mat[mi, ri] = float(np.mean(model_trades))

                fold_cnts[mi, ri] = len(folds)

        return RegimeModelMatrix(
            regimes=regimes,
            models=models,
            sharpe_matrix=sharpe_mat,
            trade_matrix=trade_mat,
            hitrate_matrix=hitrate_mat,
            fold_counts=fold_cnts,
            raw_folds=all_folds,
        )

    def _run_significance_tests(
        self,
        all_folds: List[FoldResult],
        matrix: RegimeModelMatrix,
    ) -> Dict[str, Dict[str, Any]]:
        """Run paired t-tests between models within each regime.

        For each regime, compares all model pairs using the fold-level
        Sharpe ratios weighted by regime prevalence.

        Returns dict keyed by regime name with:
          - top_models: ordered list of (model, mean_sharpe)
          - significant_pairs: list of (model_a > model_b, p_value)
        """
        from scipy import stats as scipy_stats
        from collections import defaultdict

        results: Dict[str, Dict[str, Any]] = {}

        model_folds: Dict[str, List[FoldResult]] = defaultdict(list)
        for f in all_folds:
            model_folds[f.model].append(f)

        for r_idx, regime in enumerate(matrix.regimes):
            # Collect per-model Sharpe arrays for this regime
            model_sharpes: Dict[str, List[float]] = defaultdict(list)

            for model, folds in model_folds.items():
                for fold in folds:
                    if not np.isnan(fold.sharpe):
                        regime_weight = fold.regime_counts.get(regime, 0)
                        if regime_weight > 0:
                            model_sharpes[model].append(fold.sharpe)

            # Rank models by mean Sharpe in this regime
            ranked = []
            for model, sharpes in model_sharpes.items():
                if sharpes:
                    ranked.append((model, float(np.mean(sharpes)), len(sharpes)))
            ranked.sort(key=lambda x: x[1], reverse=True)

            # Paired t-test: compare top-3 models
            sig_pairs = []
            top_k = min(3, len(ranked))
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    m_a, _, _ = ranked[i]
                    m_b, _, _ = ranked[j]
                    sharpes_a = model_sharpes.get(m_a, [])
                    sharpes_b = model_sharpes.get(m_b, [])

                    # Need same folds for paired test
                    min_len = min(len(sharpes_a), len(sharpes_b))
                    if min_len >= 3:
                        try:
                            t_stat, p_val = scipy_stats.ttest_rel(
                                sharpes_a[:min_len], sharpes_b[:min_len]
                            )
                            sig_pairs.append({
                                "better_model": m_a,
                                "worse_model": m_b,
                                "t_statistic": float(t_stat),
                                "p_value": float(p_val),
                                "significant_at_05": p_val < 0.05,
                            })
                        except Exception:
                            continue

            results[regime] = {
                "top_models": [(m, s) for m, s, _ in ranked],
                "significant_pairs": sig_pairs,
                "n_models_with_data": len(ranked),
            }

        return results

    # ── Serialization ────────────────────────────────────────────────

    def save_matrix(self, result: ExpertProfileResult, path: str):
        """Save the performance matrix as JSON."""
        data = {
            "models": result.matrix.models,
            "regimes": result.matrix.regimes,
            "sharpe_matrix": result.matrix.sharpe_matrix.tolist(),
            "trade_matrix": result.matrix.trade_matrix.tolist(),
            "hitrate_matrix": result.matrix.hitrate_matrix.tolist(),
            "fold_counts": result.matrix.fold_counts.tolist(),
            "significance": result.significance,
            "execution_time_s": result.execution_time_seconds,
            "warnings": result.warnings,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[PROFILE] Matrix saved to {path}")

    @staticmethod
    def load_matrix(path: str) -> RegimeModelMatrix:
        """Load a performance matrix from JSON."""
        with open(path) as f:
            data = json.load(f)

        regimes = data.get("regimes", [])
        models = data.get("models", [])
        sharpe = np.array(data.get("sharpe_matrix", []))
        trades = np.array(data.get("trade_matrix", []))
        hitrate = np.array(data.get("hitrate_matrix", []))
        fold_cnts = np.array(data.get("fold_counts", []))

        return RegimeModelMatrix(
            regimes=regimes,
            models=models,
            sharpe_matrix=sharpe,
            trade_matrix=trades,
            hitrate_matrix=hitrate,
            fold_counts=fold_cnts,
        )

    def print_summary(self, result: ExpertProfileResult):
        """Print a human-readable profiling summary."""
        print("\n" + "=" * 72)
        print("  EXPERT PROFILER — Regime × Model Performance Matrix")
        print("=" * 72)
        print(f"\n  Models: {', '.join(result.matrix.models)}")
        print(f"  Folds total: {result.total_folds}")
        print(f"  Time: {result.execution_time_seconds:.0f}s")
        if result.warnings:
            print(f"  Warnings: {len(result.warnings)}")
            for w in result.warnings[:3]:
                print(f"    - {w}")

        print("\n  ── Top model per regime ──")
        for regime, top in result.matrix.top_model_per_regime(top_k=3).items():
            line = f"    {regime:20s}: "
            parts = []
            for model, score in top:
                parts.append(f"{model} ({score:+.3f})")
            line += "  ".join(parts)
            print(line)

        print("\n  ── Significant differences ──")
        sig_found = False
        for regime, info in result.significance.items():
            for sp in info.get("significant_pairs", []):
                if sp.get("significant_at_05"):
                    sig_found = True
                    print(f"    {regime}: {sp['better_model']} > {sp['worse_model']} "
                          f"(p={sp['p_value']:.4f})")
        if not sig_found:
            print("    (no statistically significant differences at p<0.05)")

        print("\n" + "=" * 72)


# ══════════════════════════════════════════════════════════════════════
# Phase 0: Pre-screening model pruning
# ══════════════════════════════════════════════════════════════════════

def prune_models(
    matrix: "RegimeModelMatrix",
    min_sharpe: float = 0.0,
    max_models: int = 7,
) -> tuple[list[str], list[str]]:
    """Filter a RegimeModelMatrix to the top-k models with diversity enforcement.

    Parameters
    ----------
    matrix : RegimeModelMatrix
        Output from ExpertProfiler (Phase B).
    min_sharpe : float
        Absolute threshold: a model must have *any* regime with Sharpe > this
        value to survive (default 0.0).
    max_models : int
        Maximum number of survivors after pruning (default 7).

    Returns
    -------
    survivors : list[str]
        Model names that passed all gates (sorted by best regime Sharpe desc).
    pruned : list[str]
        Model names that were eliminated.
    """
    import numpy as np

    if matrix.models is None or len(matrix.models) == 0:
        raise ValueError("Matrix has no models — cannot prune.")
    if matrix.regimes is None or len(matrix.regimes) == 0:
        raise ValueError("Matrix has no regimes — cannot prune.")

    # Score each model by its best Sharpe across all regimes
    scored: list[tuple[str, float]] = []
    for i, model in enumerate(matrix.models):
        row = matrix.sharpe_matrix[i, :]
        valid = [float(s) for s in row if not (isinstance(s, float) and np.isnan(s))]
        best = max(valid) if valid else -float("inf")
        scored.append((model, best))

    # Sort descending by best Sharpe
    scored.sort(key=lambda x: x[1], reverse=True)

    # Absolute threshold: eliminate models that never reach min_sharpe
    survivors = [m for m, s in scored if s > min_sharpe]
    pruned = [m for m, s in scored if s <= min_sharpe]

    # Cap to max_models
    if len(survivors) > max_models:
        pruned.extend(survivors[max_models:])
        survivors = survivors[:max_models]

    # Diversity enforcement: ensure at least 1 tree, 1 linear, 1 deep
    has_tree = any(m in TREE_MODELS for m in survivors)
    has_linear = any(m in LINEAR_MODELS for m in survivors)
    has_deep = any(m in DEEP_MODELS for m in survivors)

    missing_families = []
    if not has_tree:
        missing_families.append(("tree", TREE_MODELS))
    if not has_linear:
        missing_families.append(("linear", LINEAR_MODELS))
    if not has_deep:
        missing_families.append(("deep", DEEP_MODELS))

    for family_name, family_set in missing_families:
        best_from_family = None
        best_score_family = -float("inf")
        for model, s in scored:
            if model in family_set and model in pruned:
                if s > best_score_family:
                    best_score_family = s
                    best_from_family = model
        if best_from_family is not None and best_score_family > min_sharpe:
            survivors.append(best_from_family)
            pruned.remove(best_from_family)
            survivors.sort(key=lambda m: next(s for _m, s in scored if _m == m), reverse=True)

    return survivors, pruned


# ══════════════════════════════════════════════════════════════════════
# HPO param prefix stripping
# ══════════════════════════════════════════════════════════════════════

_MODEL_PREFIX_MAP: Dict[str, str] = {
    "logistic": "logit_",
    "svm": "svm_",
    "random_forest": "rf_",
    "decision_tree": "dt_",
    "xgboost": "xgb_",
    "lightgbm": "lgbm_",
    "catboost": "cb_",
    "lstm": "lstm_",
    "cnn": "cnn_",
    "transformer": "transformer_",
    "gru": "gru_",
    "gru_lstm": "gru_lstm_",
    "ensemble_adaptive_regime": None,
    "ensemble_cnn_lstm_xgboost": None,
    "stacking_ensemble": None,
    "meta_ensemble": None,
}


def _strip_model_prefix(params: dict, model_type: str) -> dict:
    """Remove the Optuna search-space prefix from parameter keys.

    Examples
    --------
    >>> _strip_model_prefix({"logit_C": 0.3, "logit_max_iter": 300}, "logistic")
    {"C": 0.3, "max_iter": 300}
    """
    prefix = _MODEL_PREFIX_MAP.get(model_type)
    if prefix is None:
        return dict(params)
    result: Dict[str, Any] = {}
    L = len(prefix)
    for k, v in params.items():
        if isinstance(k, str) and k.startswith(prefix):
            result[k[L:]] = v
        else:
            result[k] = v
    return result


def _reprefix_params(params: dict, model_type: str) -> dict:
    """Re-add the Optuna search-space prefix so model registry builders
    (which use filter_params) can find them.

    Examples
    --------
    >>> _reprefix_params({"C": 0.3, "max_iter": 300}, "logistic")
    {"logit_C": 0.3, "logit_max_iter": 300}
    """
    prefix = _MODEL_PREFIX_MAP.get(model_type)
    if prefix is None:
        return dict(params)
    return {f"{prefix}{k}": v for k, v in params.items()}
