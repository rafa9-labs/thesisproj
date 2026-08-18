# CLAUDE.md

This file provides guidance to AI coding assistants working on the KodaQuant repository.

## Project Overview

KodaQuant is a forex trading research and deployment platform: walk-forward backtesting,
hyperparameter optimization, model committees, and live trading via OANDA. Internally the
code refers to the "forex pipeline" as shorthand — the product name is KodaQuant.

## Commands

- Run tests: `python -m pytest tests -q` (addopts in `pyproject.toml` enable xdist).
- Smoke run: `SMOKE_TEST=1 TRADING_COSTS=1 python -m pipeline.main_cli`
- Start API: `python run_server.py`
- Frontend dev: `cd frontend && npm run dev`

## Architecture

- `pipeline/` — backtest engine, metrics (DSR/PBO/trust), feature selection, tuning
- `api/` — FastAPI routers, JobManager, gatekeeper/VRAM, committee full-cycle runner
- `trading/` — live trading engines (paper/live/committee) with risk gates
- `models/` — model registry; `pipeline/models/model_defaults.py` is the three-tier
  hyperparameter source of truth (SEARCH_SPACE/FIXED_DEFAULTS are derived from it)
- `frontend/` — React/TypeScript desktop UI

## Conventions

- Feature toggles (`use_*`) are user-locked via `UserFixedConfig`; Optuna must never
  sample them when a user config is provided.
- HPO parameter keys are model-prefixed (`logit_C`, `xgb_max_depth`, `lgbm_*`, `cb_*`).
- Risk defaults: max DD 0.15, daily loss 0.05 (mirrored between live and backtest).
