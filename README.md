# Forex ML Backtester — Walk-Forward FX Backtesting Platform

A commercial-grade walk-forward backtesting engine for Forex trading strategies. 8 ML models, 6 currency pairs, cost-aware execution, news sentiment, and a full REST API — all with zero data leakage guarantees.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FOREX ML BACKTESTER PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Data Layer                                                          │
│  ├─ 6 FX pairs × 3 timeframes (M30, H1, H4) = 18 datasets          │
│  ├─ EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, GBPJPY                  │
│  ├─ 10 years OANDA historical data per pair                          │
│  └─ SQLite data layer + CSV→SQLite migration tool                    │
│                                                                      │
│  Feature Engineering (pipeline/backtester/features_mixin.py)         │
│  ├─ 80+ TA indicators (RSI, MACD, Bollinger, ATR, ADX, Stochastic)  │
│  ├─ 3-layer cache: in-memory → Parquet disk → fresh compute          │
│  ├─ Cache key: SHA256(data file + size + mtime + feature config)     │
│  ├─ News sentiment features (VADER / finBERT)                        │
│  ├─ Economic event flags (NFP, FOMC, CPI, GDP, PMI, rate decisions)  │
│  └─ All indicators look backward only — zero data leakage            │
│                                                                      │
│  ML Models (models/)                                                 │
│  ├─ Shallow: Logistic, SVM, Random Forest, Decision Tree, XGBoost   │
│  ├─ Deep:    CNN, LSTM, Transformer (TensorFlow/Keras)              │
│  ├─ RL:      Dueling DQN (PyTorch)                                  │
│  ├─ Ensemble: CNN+LSTM+XGBoost hybrid, Adaptive Regime             │
│  └─ All 10 types verified end-to-end                                 │
│                                                                      │
│  Walk-Forward Engine (pipeline/backtester/)                          │
│  ├─ Monthly refit, strict chronological splits (train ≤ test)        │
│  ├─ 1-bar execution delay enforced everywhere                        │
│  ├─ Triple-barrier labeling for exit strategies                      │
│  └─ Optuna HPO with per-model search spaces                          │
│                                                                      │
│  Execution Models (pipeline/execution/)                              │
│  ├─ Position Sizing: Fixed, Fixed Fractional, Kelly, ATR-based       │
│  ├─ Stop-Loss/Take-Profit: Fixed pips, ATR dynamic, Breakeven       │
│  ├─ Trailing Stops: Standard, ATR, Chandelier Exit                   │
│  ├─ Risk Management: DD circuit breaker, loss limits, cooloff        │
│  └─ Cost-Aware: Spread + slippage modeled per bar                    │
│                                                                      │
│  News & Sentiment (news/)                                            │
│  ├─ RSS scraper (Reuters, Bloomberg, ForexLive, Investing.com)       │
│  ├─ NewsAPI integration (opt-in)                                     │
│  ├─ Economic calendar (8 major event types)                          │
│  ├─ VADER (default) / finBERT (opt-in) sentiment scoring             │
│  └─ Walk-forward-safe merge (forward-fill only, no future data)      │
│                                                                      │
│  API Backend (api/) — FastAPI + Celery + WebSocket                   │
│  ├─ POST /api/v1/backtest          — run backtest with config        │
│  ├─ GET  /api/v1/backtest/{id}     — poll progress                   │
│  ├─ GET  /api/v1/backtest/{id}/results — fetch results + metrics     │
│  ├─ WS   /api/v1/backtest/{id}/ws  — real-time progress events       │
│  ├─ GET  /api/v1/models            — list models + registry          │
│  ├─ GET  /api/v1/pairs             — list pairs + timeframes         │
│  ├─ GET  /api/v1/data/download     — data management                 │
│  └─ Background jobs via Celery + Redis                               │
│                                                                      │
│  UI Layer                                                            │
│  ├─ Streamlit (interim dev UI): app.py on port 8501                  │
│  └─ React Frontend (Sprint 8): Vite + TS + TailwindCSS + shadcn/ui  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PRODUCT ARCHITECTURE (Target — Electron Desktop App)       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────────────┐     │
│  │  Electron Shell  │  │  Python Backend (FastAPI)     │     │
│  │  (Sprint 9)      │  │  (Sprint 7 — COMPLETE)        │     │
│  │                  │  │                                │     │
│  │  ├─ BrowserWindow│──│─► REST API (uvicorn)          │     │
│  │  ├─ System Tray  │  │  ├─ Backtest runner            │     │
│  │  ├─ Native Menus │  │  ├─ Model registry             │     │
│  │  ├─ Auto-Update  │  │  ├─ WebSocket progress         │     │
│  │  └─ License UI   │  │  ├─ Celery task queue          │     │
│  │                  │  │  └─ SQLite data layer           │     │
│  │  React App       │  │                                │     │
│  │  (Sprint 8)      │  │  Pipeline Engine               │     │
│  │  ├─ Dashboard    │  │  ├─ Walk-forward backtester    │     │
│  │  ├─ Backtest cfg │  │  ├─ Feature engineering        │     │
│  │  ├─ Results      │  │  ├─ Execution models           │     │
│  │  ├─ Compare      │  │  ├─ HPO (Optuna)               │     │
│  │  └─ Settings     │  │  └─ News & sentiment           │     │
│  └──────────────────┘  └──────────────────────────────┘     │
│                                                              │
│  Desktop App (.exe)    Sold via Paddle                       │
│  Sprints 8-9           Sprint 10 (Licensing)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Features for Quant Research

### Walk-Forward Backtesting
- **Monthly refit cycle** — model retrains each month using all prior data
- **Strict chronological splits** — train data never exceeds test data timestamp
- **1-bar execution delay** — signals generated on bar N are executed on bar N+1 open
- **Triple-barrier labeling** — advanced exit strategy modeling (time, profit, loss barriers)
- **Anti-lookahead guarantees** — all indicators look backward only, verified by 16 integrity tests

### ML Models (10 types)
| Category | Models | Framework |
|----------|--------|-----------|
| **Shallow** | Logistic Regression, SVM, Random Forest, Decision Tree, XGBoost | scikit-learn / XGBoost |
| **Deep Learning** | CNN, LSTM, Transformer | TensorFlow / Keras |
| **Reinforcement Learning** | Dueling DQN | PyTorch |
| **Ensemble** | CNN+LSTM+XGBoost hybrid, Adaptive Regime | Mixed |

### Execution Simulation
- **Position Sizing**: Fixed lot, Fixed Fractional (% equity), Kelly Criterion, ATR-based
- **Stop-Loss / Take-Profit**: Fixed pips, ATR-dynamic, Breakeven management, Partial close (scale-out)
- **Trailing Stops**: Standard (fixed pips), ATR trailing, Chandelier Exit
- **Risk Management**: Max drawdown circuit breaker, consecutive loss limits, daily loss limits, cooloff periods
- **Cost Modeling**: Spread + slippage modeled per bar, configurable via `ExecutionConfig`

### Feature Engineering
- **80+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, CCI, Williams %R, OBV, VWAP, Ichimoku, and more
- **News Sentiment**: VADER (default, zero-dependency) or finBERT (opt-in via HuggingFace)
- **Economic Event Flags**: NFP, FOMC, CPI, GDP, Retail Sales, PMI, ECB Rate, BOE Rate proximity markers
- **3-layer Feature Cache**: In-memory → Parquet disk cache → fresh computation (saves 30-120s per rerun)

### News & Sentiment Pipeline
- **RSS Sources**: Reuters, Bloomberg, ForexLive, Investing.com
- **NewsAPI**: Optional integration with API key
- **Economic Calendar**: 8 major event types with date-based proximity features
- **Deduplication**: Hash-based title deduplication across sources
- **Caching**: Parquet disk cache in `news_cache/`

### Hyperparameter Optimization
- **Optuna-powered** HPO with per-model search spaces
- **7 model-specific search spaces** defined in `config.py`
- **Best configs** stored in `hpo/` directory for reproducibility
- **Smoke mode** (1 trial) vs full mode (all trials) for quick iteration

### Model Comparison & Leaderboard
- **One-command comparison** across all 10 model types
- **Ranked leaderboard** with Sharpe, Sortino, max DD, win rate, profit factor
- **Paired t-test significance testing** between model pairs
- **Cross-pair comparison** with normalized metrics

## Project Structure

```
forex-pipeline/
├── app.py                          # Streamlit dev UI entry point
├── config.py                       # Global config + PIPELINE_CONSTANTS + ExecutionConfig
│
├── pipeline/                       # Core backtesting engine
│   ├── backtester/
│   │   ├── composed.py             # MLBacktester (main engine class)
│   │   ├── execution_patches.py    # Execution loop + PatchConfig + LoopResult
│   │   ├── features_mixin.py       # Feature engineering (TA indicators)
│   │   ├── strategy_mixin.py       # Strategy logic
│   │   ├── model_factory_mixin.py  # Model creation
│   │   ├── deep_mixin.py           # Deep learning model handling
│   │   ├── dqn_mixin.py            # DQN/RL model handling
│   │   ├── ensemble_mixin.py       # Ensemble model logic
│   │   ├── run_mixin.py            # Run orchestration
│   │   └── real_trading_mixin.py   # Real trading simulation
│   ├── execution/
│   │   ├── position_sizing.py      # Kelly, fixed fractional, ATR sizing
│   │   ├── stops.py                # SL/TP management + breakeven + scale-out
│   │   ├── trailing.py             # Standard, ATR, Chandelier trailing stops
│   │   └── risk_manager.py         # DD circuit breaker, loss limits, cooloff
│   ├── tuning/                     # Optuna HPO (sampler, runner, objective)
│   ├── runtime.py                  # GPU detection, CUDA config, thread budgets
│   ├── feature_cache.py            # Parquet disk cache for features
│   ├── model_comparison.py         # Model comparison & leaderboard
│   ├── main_cli.py                 # CLI runner (headless backtesting)
│   ├── metrics_eval.py             # Sharpe, Sortino, max DD, profit factor, etc.
│   ├── plotting.py                 # Chart generation (matplotlib/plotly)
│   └── data_downloader.py          # OANDA multi-pair data download
│
├── models/                         # ML model implementations
│   ├── registry.py                 # Model registry (name → class)
│   ├── base_model.py               # Abstract base model interface
│   ├── logistic.py, svm.py, random_forest.py
│   ├── xgboost_model.py
│   ├── cnn.py, lstm.py, transformer.py
│   ├── ensemble_cnn_lstm_xgboost.py
│   └── ensemble_adaptive_regime.py
│
├── rl/                             # Reinforcement learning
│   ├── dqn_agent.py                # Dueling DQN agent
│   ├── environment.py              # Gym-style trading environment
│   ├── replay_buffer.py            # Experience replay
│   └── wrappers.py                 # Reward processing, cost-aware wrappers
│
├── news/                           # News & sentiment pipeline
│   ├── scraper.py                  # RSS + NewsAPI + economic calendar
│   ├── sentiment.py                # VADER / finBERT scoring
│   └── features.py                 # Walk-forward-safe feature merge
│
├── api/                            # FastAPI backend (Sprint 7)
│   ├── main.py                     # FastAPI app + CORS + lifecycle
│   ├── routers/
│   │   ├── backtest.py             # POST run, GET status, GET results
│   │   ├── models.py               # Model registry endpoint
│   │   ├── pairs.py                # Currency pair listing + data ranges
│   │   ├── ws.py                   # WebSocket real-time progress
│   │   ├── data.py                 # Data download management
│   │   └── health.py               # Health check
│   └── tasks.py                    # Celery background tasks
│
├── ui/                             # Streamlit UI components
│   ├── controls.py                 # Sidebar nav + 6-tab layout
│   ├── state.py                    # AppState + backtest adapter
│   ├── results.py                  # Results display + export
│   ├── dashboard.py                # Dashboard renderer
│   ├── charts.py                   # Plotly chart builders
│   └── validators.py               # Input validation
│
├── schemas/                        # Pydantic-like config validators
│   ├── backtest.py, features.py, hpo.py, settings.py
│
├── csv_data/                       # Historical FX data (18 files)
│   ├── EURUSD_10_years_{M30,H1,H4}_OANDA.csv
│   ├── GBPUSD_10_years_{M30,H1,H4}_OANDA.csv
│   ├── USDJPY_10_years_{M30,H1,H4}_OANDA.csv
│   ├── AUDUSD_10_years_{M30,H1,H4}_OANDA.csv
│   ├── USDCAD_10_years_{M30,H1,H4}_OANDA.csv
│   └── GBPJPY_10_years_{M30,H1,H4}_OANDA.csv
│
├── hpo/                            # Best HPO configs per model
├── tests/                          # 436 tests across 23 test files
│
├── launch_ui.bat                   # Streamlit UI launcher (Windows → WSL)
├── run_smoke.bat                   # Smoke test (all models, 1 trial)
├── run_all_tests.bat               # Full test suite (20-60 min)
├── run_comparison.bat              # Multi-model comparison runner
└── run_smoke_gpu.bat               # GPU smoke test (Windows → WSL)
```

## Installation

### 1. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 2. (Optional) GPU Support for Deep Models
Deep models (CNN, LSTM, Transformer) benefit from CUDA. Install the appropriate TensorFlow:
```powershell
# For CUDA 12.x
pip install tensorflow[and-cuda]
# For CUDA 11.x
pip install tensorflow==2.15.0
```

### 3. (Optional) News Sentiment Backend
```powershell
# VADER is included by default (zero deps)
# For finBERT (better quality, slower):
pip install transformers torch
```

### 4. (Optional) Celery Task Queue
```powershell
pip install celery[redis] redis
# Requires a running Redis instance for background jobs
```

### 5. Verify Installation
```powershell
# Quick smoke test — all 10 models, 1 trial each
.\run_smoke.bat

# Full test suite
.\run_all_tests.bat
```

## Usage

### Running the Streamlit UI

```powershell
# Option 1: Double-click launch_ui.bat

# Option 2: PowerShell
.\launch_ui.bat

# Option 3: Manual (if not using WSL)
streamlit run app.py

# Then open: http://localhost:8501
```

The Streamlit UI provides 6 tabs:
1. **Dashboard** — KPI overview, recent backtests
2. **Backtest** — Configure and run backtests (model, pair, features, execution)
3. **Results** — Equity curves, metrics, trade log, export
4. **Model Comparison** — Side-by-side model leaderboard
5. **News & Sentiment** — Sentiment features, event flags
6. **Settings** — GPU config, data sources, pipeline constants

### Running via CLI (Headless)

```powershell
# Run a single model backtest
python -m pipeline.main_cli --model xgboost --pair EURUSD --tf H1 --months 3

# Run with environment variables
$env:MODEL_LIST="logistic,xgboost,cnn"
$env:SEEDS="42"
$env:N_MONTHS="3"
python -m pipeline.main_cli
```

### Model Comparison & Leaderboard

```powershell
# Smoke mode — all 10 models, 1 trial, 1 month (fast)
.\run_comparison.bat smoke

# Quick mode — logistic + xgboost only
.\run_comparison.bat quick

# Full mode — all models, all trials, 3 months
.\run_comparison.bat full

# Analyze existing results only (no new runs)
.\run_comparison.bat analyze

# GPU mode — run via WSL with CUDA
.\run_comparison.bat gpu
```

### FastAPI Backend

```powershell
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# API docs (interactive Swagger UI)
# Open: http://localhost:8000/docs

# Key endpoints:
# POST   /api/v1/backtest              — Run backtest
# GET    /api/v1/backtest/{job_id}     — Poll progress
# GET    /api/v1/backtest/{job_id}/results — Get results
# WS     /api/v1/backtest/{job_id}/ws  — Real-time progress
# GET    /api/v1/models                — List all models
# GET    /api/v1/pairs                 — List pairs + timeframes
```

### Programmatic Usage

```python
from pipeline.backtester.composed import MLBacktester

backtester = MLBacktester(
    data_file="csv_data/EURUSD_10_years_H1_OANDA.csv",
    model_name="xgboost",
    n_months=3,
)

results = backtester.run()
print(f"Sharpe: {results['sharpe']:.2f}")
print(f"Max DD: {results['max_drawdown']:.1%}")
print(f"Win Rate: {results['win_rate']:.1%}")
```

### Execution Model Configuration

```python
from config import ExecutionConfig

# Configure position sizing + risk management
exec_config = ExecutionConfig(
    sizing_method="kelly",           # fixed, fixed_fractional, kelly, atr
    sl_method="atr",                 # fixed, atr
    tp_method="atr",                 # fixed, atr, breakeven
    trailing_method="chandelier",    # standard, atr, chandelier
    max_drawdown_pct=0.15,           # circuit breaker at 15% DD
    max_consecutive_losses=5,
    daily_loss_limit_pct=0.03,
)
```

### News & Sentiment Features

```python
from news.scraper import NewsScraper
from news.sentiment import SentimentAnalyzer
from news.features import build_news_features

# Scrape news (RSS + optional NewsAPI)
scraper = NewsScraper(cache_dir="news_cache/")
articles = scraper.fetch_all()

# Score sentiment
analyzer = SentimentAnalyzer(backend="vader")  # or "finbert"
scores = analyzer.score_articles(articles)

# Merge into features (walk-forward safe)
features_df = build_news_features(
    price_df=my_price_data,
    articles=articles,
    scores=scores,
    window_bars=[6, 24],
)
```

## Testing

```powershell
# Smoke test — all models end-to-end (2-5 min)
.\run_smoke.bat

# Full test suite — 436 tests (20-60 min via WSL)
.\run_all_tests.bat

# Run specific test file
python -m pytest tests/test_pipeline_validation.py -v

# Run with coverage
python -m pytest tests/ --cov=pipeline --cov=models --cov-report=term-missing

# GPU smoke test (Windows → WSL + CUDA)
.\run_smoke_gpu.bat
```

### Test Coverage
| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_pipeline_validation.py` | 39 | Execution loop, features, E2E (logistic + CNN) |
| `test_execution_bugs.py` | 24 | Sprint A regression tests (9 execution bugs) |
| `test_data_robustness.py` | 8 | Sprint B regression tests (4 data bugs) |
| `test_news.py` | 44 | Scraper, sentiment, features, event flags |
| `test_walk_forward_integrity.py` | 16 | Chronological splits, execution delay |
| `test_schemas.py` | — | Config, features, HPO, settings validators |
| `test_api.py` | — | FastAPI endpoints, WebSocket, job lifecycle |
| `smoke_all_models.py` | — | All 10 model types end-to-end |

## Currency Pairs & Data

| Pair | Timeframes | Years | Source |
|------|-----------|-------|--------|
| EURUSD | M30, H1, H4 | 10 | OANDA |
| GBPUSD | M30, H1, H4 | 10 | OANDA |
| USDJPY | M30, H1, H4 | 10 | OANDA |
| AUDUSD | M30, H1, H4 | 10 | OANDA |
| USDCAD | M30, H1, H4 | 10 | OANDA |
| GBPJPY | M30, H1, H4 | 10 | OANDA |

Download additional data via the OANDA downloader:
```powershell
python -m pipeline.data_downloader --pair XAUUSD --tf H1 --years 10
```

## Configuration

### Pipeline Constants (`config.py`)
All magic numbers centralized in `PIPELINE_CONSTANTS` (26 named constants):
- `vol_window_bars`, `high_vol_q`, `slip_norm_bps`, `gamma_slip_norm`
- Default feature toggles, execution params, walk-forward settings

### Search Spaces (`config.py`)
Per-model HPO search spaces in `SEARCH_SPACE`:
- Logistic, SVM, Random Forest, XGBoost: scikit-learn params
- CNN, LSTM, Transformer: architecture + training params
- Optimized ranges based on literature values

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_LIST` | all | Comma-separated model names to run |
| `SEEDS` | 42 | Random seeds (comma-separated) |
| `REPEATS` | 1 | Number of repeat runs |
| `N_MONTHS` | 3 | Walk-forward months |
| `SMOKE_TEST` | 0 | Set to 1 for fast smoke mode |
| `NEWSAPI_KEY` | — | NewsAPI key (optional) |

## Roadmap

| Sprint | Topic | Status |
|--------|-------|--------|
| **S1** | Model Comparison & Leaderboard | DONE |
| **S2** | Advanced Execution Models | DONE |
| **S3** | Multi-Currency Expansion | DONE |
| **S4** | Docker + CI/CD | PARTIAL (RAM opt done) |
| **S5** | Comprehensive Tests + Benchmarks | IN PROGRESS (436 tests) |
| **S6** | News & Sentiment Features | DONE |
| **S7** | FastAPI Backend | DONE |
| **S8** | React Frontend | NEXT |
| **S9** | Electron Desktop Shell | TODO |
| **S10** | Security & Licensing (Paddle) | TODO |
| **S11** | Installer & Auto-Update | TODO |
| **S12** | Commercial Infrastructure | TODO |
| **S13** | Beta & Launch | TODO |

**Product target**: Commercial Electron desktop app (React + FastAPI + Python), sold via Paddle.
**Pricing**: Hybrid — one-time purchase + annual updates subscription.

See `ROADMAP.md` for full sprint details with sub-tasks and file references.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI (dev)** | Streamlit |
| **UI (product)** | React 18 + TypeScript + TailwindCSS + shadcn/ui |
| **Desktop** | Electron (Sprint 9) |
| **API** | FastAPI + Uvicorn |
| **Task Queue** | Celery + Redis |
| **Data** | Pandas + NumPy + SQLite + Parquet |
| **ML (shallow)** | scikit-learn + XGBoost |
| **ML (deep)** | TensorFlow / Keras |
| **ML (RL)** | PyTorch |
| **HPO** | Optuna |
| **Charts** | Plotly + Matplotlib |
| **News** | feedparser + VADER / HuggingFace Transformers |
| **Testing** | pytest + 436 tests |

## License

Proprietary. All rights reserved.
