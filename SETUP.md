# KodaQuant Setup Guide

## Requirements

- Python 3.11+ (3.12 recommended)
- Node.js 18+ (frontend only)
- Docker (optional, for the containerized stack)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Starting the application

### Desktop API + worker (development)

```bash
python run_server.py             # FastAPI + Celery worker (no Docker)
```

### Docker compose stack

```bash
docker compose up -d             # api + worker + redis + frontend
```

### Frontend (development)

```bash
cd frontend
npm install
npm run dev
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_LIST` | logistic,xgboost | Models to run |
| `SEEDS` | 33333 | Random seeds |
| `N_MONTHS` | 12 | Walk-forward months |
| `TRADING_COSTS` | 1 | Set 0 for the no-cost ablation |
| `SMOKE_TEST` | 0 | Fast smoke mode |
| `OANDA_ACCESS_TOKEN` | — | Live trading (optional) |
| `OANDA_ACCOUNT_ID` | — | Live trading (optional) |

## Running tests

```bash
python -m pytest tests -q
```

The suite runs in parallel via pytest-xdist. Heavy suites (`test_full_pipeline_e2e`,
`test_phase_d_integration`) are marked `slow`.

## Generating icons

```bash
python scripts/generate_icons.py   # build/ + frontend/public icons
```
