"""
Quick script: train a simple logistic model on EURUSD M30 and register
it in the deployed_models table so it appears in the Trading tab.

Run from project root:
    python scripts/train_quick_model.py
"""
from __future__ import annotations
import json
import os
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.backtester.composed import MLBacktester
from pipeline.metrics.metrics_tuples import CLASS_DEFAULTS
from pipeline.models.model_persistence import save_snapshot
from pipeline.models.model_registry_disk import register_snapshot
from api.config import settings

# -- Config ------------------------------------------------------
PAIR = "EURUSD"
TIMEFRAME = "M30"
MODEL_TYPE = "logistic"
MONTHS = 3            # walk-forward window - short = more active
SEED = 42
TRADING_COSTS = 2.0   # pips - low costs = more trades execute
END_DATE = "2026-04-15"
START_DATE = "2022-01-01"   # 37-month warmup pushes first test to ~2025-02

CSV_PATH = str(PROJECT_ROOT / "csv_data" / f"{PAIR}_10_years_{TIMEFRAME}_OANDA.csv")


def main():
    print(f"Training {MODEL_TYPE} on {PAIR} {TIMEFRAME} ...")
    print(f"  Data: {START_DATE} -> {END_DATE}")
    print(f"  Costs: {TRADING_COSTS} pips")

    # Build features config from CLASS_DEFAULTS (same as tasks.py)
    feat_cfg = deepcopy(CLASS_DEFAULTS["features"])
    feat_cfg.update(deepcopy(CLASS_DEFAULTS["cv"]))

    bt = MLBacktester(
        symbol=PAIR,
        start=START_DATE,
        end=END_DATE,
        trading_costs=TRADING_COSTS,
        model_type=MODEL_TYPE,
        features_config=feat_cfg,
        db_path=settings.db_full_path,
    )

    base_cfg = deepcopy(CLASS_DEFAULTS["features"])
    base_cfg.update(deepcopy(CLASS_DEFAULTS["cv"]))
    base_cfg["model_type"] = MODEL_TYPE
    base_cfg["rep"] = 1
    base_cfg["trading_costs"] = TRADING_COSTS
    base_cfg["n_trials"] = 4     # few HPO trials - fast
    base_cfg["seed"] = SEED

    df_sim = bt.real_trading_simulation(
        base_cfg,
        models_to_test=[MODEL_TYPE],
        months=MONTHS,
    )

    # Grab the trained model
    model_obj = getattr(bt, "_last_trained_model", None) or getattr(bt, "model", None)
    if model_obj is None:
        print("ERROR: No model object after training.")
        sys.exit(1)

    feat_names = getattr(bt, "_diagnostics_feature_names", None) or []
    fc = getattr(bt, "features_config", None) or {}
    cov_thr = getattr(bt, "_coverage_conf_thr", None)

    # Build metrics row from simulation results
    metrics_row = {}
    if df_sim is not None and len(df_sim) > 0:
        last = df_sim.iloc[-1] if hasattr(df_sim, "iloc") else {}
        metrics_row = {
            "sharpe": float(last.get("sharpe", 0)),
            "total_return_pct": float(last.get("total_return_pct", 0)),
            "max_drawdown": float(last.get("max_drawdown", 0)),
            "win_rate": float(last.get("win_rate", 0)),
            "total_trades": int(last.get("total_trades", 0)),
            "sortino": float(last.get("sortino", 0)),
            "profit_factor": float(last.get("profit_factor", 0)),
        }
        print(f"\n-- Simulation Metrics --")
        for k, v in metrics_row.items():
            print(f"  {k}: {v}")

    # Save snapshot
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_id = f"{MODEL_TYPE}_{ts}"

    _snap_path = save_snapshot(
        model=model_obj,
        model_type=MODEL_TYPE,
        best_params=fc,
        coverage_conf_thr=float(cov_thr) if cov_thr is not None else None,
        feature_names=list(feat_names) if feat_names else None,
        features_config=fc,
        calibrate_method=fc.get("calibrate_method", "sigmoid"),
        train_start=START_DATE,
        train_end=END_DATE,
        seed=SEED,
        pip_freeze="",
        parent_job_id=f"quick_train_{uuid.uuid4().hex[:8]}",
        metrics=metrics_row,
    )
    print(f"\nSnapshot saved: {_snap_path}")

    # Register in deployed_models DB
    db_path = settings.db_full_path
    registered_id = register_snapshot(_snap_path, db_path, parent_job_status="completed")
    print(f"Registered: {registered_id}")

    # Free memory
    bt.free(release_data=True)
    del bt

    print(f"\n{'='*50}")
    print(f"  Model ready for Trading tab!")
    print(f"  ID:        {registered_id}")
    print(f"  Model:     {MODEL_TYPE}")
    print(f"  Pair:      {PAIR}")
    print(f"  Timeframe: {TIMEFRAME}")
    print(f"  Path:      {_snap_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
