# KodaQuant — Setup & Run Guide

## Prerequisites

| # | Prerequisite | Description |
|---|-------------|-------------|
| 1 | Python 3.10+ | Core runtime. Must be on PATH. |
| 2 | Pip | Python package manager (ships with Python). |
| 3 | Git | Version control (to clone the repo). |
| 4 | WSL2 + Ubuntu 22.04 | Required for GPU-accelerated runs. The .bat launchers delegate to WSL for CUDA/TensorFlow GPU support. |
| 5 | NVIDIA GPU + CUDA | Optional but recommended for deep models (CNN, LSTM, Transformer, DQN). CPU-only works but is 5-10x slower. |
| 6 | Docker Desktop | Required only for the full stack mode (docker-compose). |
| 7 | Redis | Required for the API/Celery worker stack. |
| 8 | Node.js + npm | Required for the React frontend. |
| 9 | OANDA API key | Optional. Only needed if downloading fresh FX data. Sample CSVs are bundled. |

---

## Quick Start

```powershell
git clone https://github.com/rafa9-labs/thesisproj.git
cd thesisproj
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_freeze.txt
copy .env.example .env
python -m tests.smoke_import          # verify imports
.\run_smoke.bat                       # quick backtest test
```

---

## Running KodaQuant

### Desktop App (Electron)

```powershell
# Production build
.\scripts\build_electron.bat

# Development mode
cd frontend; npm run dev:electron
```

### Web Stack (React + FastAPI)

```powershell
# Start Redis
redis-server

# Start FastAPI backend
uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload

# Start Celery worker (separate terminal)
celery -A api.tasks.celery_app worker --loglevel=info --pool=solo -Q celery

# Start React frontend (separate terminal)
cd frontend; npm run dev
```

### CLI (Headless Backtesting)

```powershell
# Single model
python -m pipeline.main_cli --model xgboost --pair EURUSD --tf H1 --months 3

# Smoke test (all models, 1 trial)
.\run_smoke.bat

# GPU smoke test
.\run_smoke_gpu.bat
```

### Model Comparison

```powershell
.\run_comparison.bat smoke    # All models, 1 trial
.\run_comparison.bat full     # Full HPO, 3 months
.\run_comparison.bat gpu      # GPU via WSL
```

### Testing

```powershell
.\run_all_tests.bat           # Full suite (20-60 min)
python -m pytest tests\ -v    # Specific tests
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_LIST` | all | Comma-separated model names |
| `SEEDS` | 42 | Random seeds |
| `N_MONTHS` | 3 | Walk-forward months |
| `SMOKE_TEST` | 0 | Set to 1 for fast mode |
| `OANDA_ACCESS_TOKEN` | - | OANDA API key |
| `NEWSAPI_KEY` | - | NewsAPI key (optional) |
| `PADDLE_API_KEY` | - | Paddle license key |
