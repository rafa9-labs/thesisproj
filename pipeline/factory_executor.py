"""Factory Executor — runs the iteration loop: propose → execute → evaluate → decide."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.committee_builder import CommitteeConfig, RegimeAssignment
from pipeline.committee_backtester import CommitteeBacktester
from pipeline.factory_proposer import ActionProposal, DeterministicProposer
from pipeline.factory_state import FactoryState, IterationRecord
from pipeline.regime_utils import RegimeConfig


class FactoryExecutor:
    def __init__(
        self,
        state: FactoryState,
        proposer: DeterministicProposer = None,
        data_path: str = None,
        confidence_threshold: float = 0.5,
        train_months: int = 6,
        test_months: int = 1,
    ):
        self.state = state
        self.proposer = proposer or DeterministicProposer()
        self.data_path = data_path or os.path.join(
            "csv_data", "EURUSD_10_years_H1_OANDA.csv")
        self.confidence_threshold = confidence_threshold
        self.train_months = train_months
        self.test_months = test_months
        self._ohlc_data: Optional[pd.DataFrame] = None

    def _load_data(self) -> pd.DataFrame:
        if self._ohlc_data is not None:
            return self._ohlc_data
        df = pd.read_csv(self.data_path)
        # Normalize OANDA column names to committee backtester convention
        if "mid_close" in df.columns and "mid_c" not in df.columns:
            df = df.rename(columns={
                "mid_open": "mid_o", "mid_high": "mid_h",
                "mid_low": "mid_l", "mid_close": "mid_c",
            })
        if "time" in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        if "returns" not in df.columns:
            df["returns"] = df["mid_c"].pct_change().fillna(0.0)
        self._ohlc_data = df
        return df

    def apply_action(self, proposal: ActionProposal) -> CommitteeConfig:
        config_dict = self.state.config.to_dict()
        regimes = {}
        for rname, rdata in config_dict.get("regimes", {}).items():
            models = list(rdata.get("models", []))
            weights = list(rdata.get("weights", []))

            if rname == proposal.regime:
                if proposal.type == "swap_model":
                    if proposal.model_remove in models:
                        idx = models.index(proposal.model_remove)
                        models[idx] = proposal.model_add
                elif proposal.type == "add_model":
                    if proposal.model_add not in models:
                        models.append(proposal.model_add)
                        weights.append(0.0)
                elif proposal.type == "remove_model":
                    if proposal.model_remove in models and len(models) > 1:
                        idx = models.index(proposal.model_remove)
                        models.pop(idx)
                        weights.pop(idx)

            # Recompute uniform weights
            if len(models) > 0:
                w = 1.0 / len(models)
                weights = [w] * len(models)

            regimes[rname] = RegimeAssignment(models=models, weights=weights)

        fallback_data = config_dict.get("fallback", {})
        fallback = RegimeAssignment(
            models=fallback_data.get("models", ["logistic"]),
            weights=fallback_data.get("weights", [1.0]),
        )
        return CommitteeConfig(regimes=regimes, fallback=fallback)

    def execute_iteration(
        self, proposal: ActionProposal,
    ) -> Tuple[Optional[IterationRecord], Optional[Any]]:
        if proposal.type == "halt":
            return None, None

        before_config = self.state.config
        before_sharpe = self.state.global_best_sharpe
        if not np.isfinite(before_sharpe):
            before_result = self._run_backtest(before_config)
            before_sharpe = before_result.avg_sharpe if before_result else 0.0
            self.state.global_best_sharpe = before_sharpe
            self.state._last_result = before_result

        new_config = self.apply_action(proposal)
        after_result = self._run_backtest(new_config)
        after_sharpe = after_result.avg_sharpe if after_result else 0.0

        improved = after_sharpe > before_sharpe + 0.005
        if improved:
            self.state.config = new_config
            self.state._last_result = after_result

        per_regime = {}
        if after_result is not None:
            for fold in after_result.folds:
                rd = getattr(fold, "regime_distribution", {}) or {}
                for rname, frac in rd.items():
                    if frac > 0:
                        per_regime[rname] = per_regime.get(rname, 0.0) + getattr(
                            fold, "sharpe", 0.0)

        record = IterationRecord(
            iteration=self.state.iteration,
            action=proposal.to_dict(),
            before_sharpe=before_sharpe,
            after_sharpe=after_sharpe,
            accepted=improved,
            per_regime_delta=per_regime,
            rationale=proposal.rationale,
        )
        self.state.track_iteration(record)
        return record, after_result

    def _run_backtest(self, config: CommitteeConfig):
        df = self._load_data()
        bt = CommitteeBacktester(
            config,
            regime_cfg=RegimeConfig(),
            confidence_threshold=self.confidence_threshold,
        )
        return bt.run_wfo(
            df,
            train_months=self.train_months,
            test_months=self.test_months,
            verbose=False,
        )

    def run_loop(self, verbose: bool = True) -> FactoryState:
        if verbose:
            print("\n[FACTORY] Starting optimization loop")
            print(f"  Initial committee: {len(self.state.config.regimes)} regimes, "
                  f"{len(self.state.config.all_models())} models")
            print(f"  Patience: {self.state.patience}  |  Tolerance: {self.state.stopping_tolerance}")
            print(f"  Max iterations: {self.state.max_iterations}  |  "
                  f"Regime floor: {self.state.regime_sharpe_floor}")

        while True:
            should_stop, reason = self.state.should_stop()
            if should_stop:
                if verbose:
                    print(f"\n[FACTORY] STOP — {reason}")
                break

            proposal = self.proposer.propose(self.state)
            if proposal.type == "halt":
                if verbose:
                    print("\n[FACTORY] HALT — no more untested moves")
                break

            if verbose:
                print(f"\n[FACTORY] Iteration {self.state.iteration + 1}: "
                      f"{proposal.type} — {proposal.regime}")

            record, result = self.execute_iteration(proposal)
            if record is None:
                continue

            if verbose:
                status = "ACCEPTED" if record.accepted else "REJECTED"
                delta = record.after_sharpe - record.before_sharpe
                print(f"  {status}: Sharpe {record.before_sharpe:.4f} -> "
                      f"{record.after_sharpe:.4f} (delta={delta:+.4f})")
                print(f"  Best so far: {self.state.global_best_sharpe:.4f}")

        if verbose:
            print(f"\n[FACTORY] Complete: {self.state.iteration} iterations, "
                  f"best Sharpe = {self.state.global_best_sharpe:.4f}")
            if self.state.global_best_config:
                print(f"  Best config: {json.dumps(self.state.global_best_config, indent=2)}")

        return self.state


def run_factory_from_disk(
    config_path: str = "results/committee/committee_config.json",
    matrix_path: str = "results/profile/regime_model_matrix.json",
    data_path: str = None,
    patience: int = 5,
    tolerance: float = 0.02,
    regime_floor: float = 0.3,
    max_iter: int = 20,
    out_dir: str = "results/factory",
    verbose: bool = True,
    proposer=None,
) -> Optional[FactoryState]:
    from pipeline.factory_state import load_state_from_disk

    state = load_state_from_disk(
        config_path=config_path,
        matrix_path=matrix_path,
        patience=patience,
        tolerance=tolerance,
        floor=regime_floor,
        max_iter=max_iter,
    )
    if state is None:
        print(f"[FACTORY] Failed to load state from {config_path} / {matrix_path}")
        return None

    executor = FactoryExecutor(
        state=state,
        proposer=proposer,
        data_path=data_path,
        train_months=6,
        test_months=1,
    )
    result = executor.run_loop(verbose=verbose)

    # Save results
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "factory_history.json", "w") as f:
        json.dump([r.to_dict() for r in result.history], f, indent=2, default=str)
    if result.global_best_config:
        with open(out / "factory_best_config.json", "w") as f:
            json.dump(result.global_best_config, f, indent=2, default=str)
    with open(out / "factory_summary.json", "w") as f:
        json.dump(result.summary(), f, indent=2, default=str)

    if verbose:
        print(f"\n[FACTORY] Results saved to {out}")
    return result
