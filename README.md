# KodaQuant — Walk-Forward FX Backtesting Platform

A commercial-grade walk-forward backtesting engine for Forex trading strategies. 18 ML models, 6 currency pairs, 4 timeframes, cost-aware execution, news sentiment, LLM-powered analysis, committee-based ensemble trading, live OANDA trading, and a full REST API — all with zero data leakage guarantees.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              KODAQUANT                                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Data Layer                                                               │
│  ├─ 6 FX pairs × 4 timeframes (M15, M30, H1, H4) = 24 datasets          │
│  ├─ EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, GBPJPY                       │
│  ├─ 10 years OANDA historical data per pair                               │
│  ├─ SQLite data layer + CSV→SQLite migration tool                         │
│  └─ OANDA live price streaming (REST + WebSocket)                         │
│                                                                           │
│  Feature Engineering (pipeline/backtester/features_mixin.py)              │
│  ├─ 80+ TA indicators (RSI, MACD, Bollinger, ATR, ADX, Stochastic)       │
│  ├─ 3-layer cache: in-memory → Parquet disk → fresh compute               │
│  ├─ Cache key: SHA256(data file + size + mtime + feature config)          │
│  ├─ BorutaSHAP feature selection (pipeline/features/boruta_sweep.py)      │
│  ├─ Feature importance sweep (pipeline/features/feature_sweep.py)         │
│  ├─ Fractional differentiation (pipeline/features/feature_utils.py)       │
│  ├─ News sentiment features (VADER / finBERT / LLM)                       │
│  ├─ Economic event flags (NFP, FOMC, CPI, GDP, PMI, rate decisions)       │
│  └─ All indicators look backward only — zero data leakage                 │
│                                                                           │
│  ML Models (models/) — 18 registered types                                │
│  ├─ Shallow: Logistic, SVM, Random Forest, Decision Tree, XGBoost         │
│  ├─ Boosting: LightGBM, CatBoost                                         │
│  ├─ Deep: CNN, LSTM, GRU, GRU-LSTM, Transformer (TensorFlow/Keras)       │
│  ├─ RL: Dueling DQN (TensorFlow)                                         │
│  ├─ Ensemble: CNN+LSTM+XGBoost, Adaptive Regime, Stacking, Meta-Ensemble │
│  ├─ Meta: Regime Classifier                                               │
│  └─ All 18 types verified end-to-end                                     │
│                                                                           │
│  Walk-Forward Engine (pipeline/backtester/)                               │
│  ├─ Monthly/Weekly/Daily refit, strict chronological splits               │
│  ├─ 1-bar execution delay enforced everywhere                             │
│  ├─ Triple-barrier labeling for exit strategies                           │
│  ├─ Optuna HPO with per-model search spaces                               │
│  └─ Forward testing engine for saved model validation                     │
│                                                                           │
│  Execution Models (pipeline/execution/)                                   │
│  ├─ Position Sizing: Fixed, Fixed Fractional, Kelly, ATR, Conviction     │
│  ├─ Stop-Loss/Take-Profit: Fixed pips, ATR dynamic, Breakeven            │
│  ├─ Trailing Stops: Standard, ATR, Chandelier Exit                        │
│  ├─ Risk Management: DD circuit breaker, loss limits, cooloff             │
│  └─ Cost-Aware: Spread + slippage modeled per bar                         │
│                                                                           │
│  Committee System (pipeline/committee/)                                   │
│  ├─ Multi-agent ensemble with regime-based model routing                  │
│  ├─ Expert profiler: per-regime model performance tracking                │
│  ├─ UCB1 multi-armed bandit for model selection                           │
│  ├─ LLM-assisted committee exploration                                    │
│  └─ Governor cache for decision deduplication                             │
│                                                                           │
│  Regime Detection (pipeline/regime/)                                      │
│  ├─ Hidden Markov Model regime detection                                  │
│  ├─ Anchored regime detection for walk-forward safety                     │
│  └─ Regime column attachment for feature engineering                      │
│                                                                           │
│  LLM Integration (pipeline/llm/)                                         │
│  ├─ LLM sentiment engine (Ollama / OpenAI / Anthropic)                    │
│  ├─ LLM advisor for backtest diagnostics                                  │
│  ├─ Structured prompts → JSON: direction, confidence, volatility          │
│  └─ SQLite cache for per-article scoring (reuse forever)                  │
│                                                                           │
│  News & Sentiment (news/)                                                 │
│  ├─ RSS scraper (Reuters, Bloomberg, ForexLive, Investing.com)            │
│  ├─ NewsAPI integration (opt-in)                                          │
│  ├─ Economic calendar (8 major event types)                               │
│  ├─ VADER (default) / finBERT / LLM sentiment scoring                     │
│  └─ Walk-forward-safe merge (forward-fill only, no future data)           │
│                                                                           │
│  Model Persistence (pipeline/models/)                                     │
│  ├─ Model snapshots: save/load with metadata + checksums                  │
│  ├─ Deployed models registry with CRUD + tags                             │
│  ├─ Fast retrain for warm-start models                                    │
│  └─ Live prediction bridge (/active/predict endpoints)                    │
│                                                                           │
│  Live Trading (trading/)                                                  │
│  ├─ OANDA v20 REST client (market, limit, stop orders)                    │
│  ├─ Paper trading engine with portfolio tracking                          │
│  ├─ Live trading engine with 4-layer risk architecture                    │
│  ├─ 17 risk gates (pre-trade, post-trade, infra, kill switch)             │
│  ├─ Committee-based live trading with regime switching                    │
│  └─ Model rotation scheduling for live deployment                         │
│                                                                           │
│  API Backend (api/) — FastAPI + Celery + WebSocket                        │
│  ├─ 15 routers: backtest, models, news, prices, trading, committee...    │
│  ├─ POST /api/v1/backtest          — run backtest with config             │
│  ├─ GET  /api/v1/backtest/{id}     — poll progress                        │
│  ├─ GET  /api/v1/backtest/{id}/results — fetch results + metrics          │
│  ├─ WS   /api/v1/backtest/{id}/ws  — real-time progress events            │
│  ├─ GET  /api/v1/models            — list models + registry               │
│  ├─ POST /api/v1/live/deploy       — deploy model for live trading        │
│  ├─ GET  /api/v1/committee/*       — committee builder & profiler         │
│  ├─ GET  /api/v1/prices/*          — historical + live price data         │
│  └─ Background jobs via Celery + Redis                                    │
│                                                                           │
│  UI Layer (frontend/)                                                     │
│  ├─ React 19 + TypeScript 6 + TailwindCSS 4 + shadcn/ui                  │
│  ├─ 10 pages: Dashboard, Backtest, Results, Compare, Committee,           │
│  │   Trading, Models, News, Monitor, Settings                             │
│  ├─ 60+ components (charts, panels, forms, tables)                       │
│  ├─ 7 Zustand stores + React Query cache                                  │
│  └─ Real-time WebSocket for progress, prices, news                        │
│                                                                           │
│  Desktop Shell (electron/)                                                │
│  ├─ Electron with BrowserWindow (min 1280×800)                            │
│  ├─ Python backend lifecycle management                                   │
│  ├─ Auto-update via electron-updater                                      │
│  ├─ System tray + native menus                                            │
│  ├─ License enforcement (Paddle integration)                              │
│  ├─ Anti-debugging protections                                            │
│  └─ Sentry crash reporting (opt-in)                                       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Features for Quant Research

### Walk-Forward Backtesting
- **Flexible refit cycles** — Monthly, Weekly, or Daily refit with all prior data
- **Strict chronological splits** — train data never exceeds test data timestamp
- **1-bar execution delay** — signals generated on bar N are executed on bar N+1 open
- **Triple-barrier labeling** — advanced exit strategy modeling (time, profit, loss barriers)
- **Anti-lookahead guarantees** — all indicators look backward only, verified by 16 integrity tests
- **Forward testing** — validate saved models on out-of-sample data

### ML Models (18 types)
| Category | Models | Framework |
|----------|--------|-----------|
| **Shallow** | Logistic Regression, SVM, Random Forest, Decision Tree | scikit-learn |
| **Boosting** | XGBoost, LightGBM, CatBoost | XGBoost / LightGBM / CatBoost |
| **Deep Learning** | CNN, LSTM, GRU, GRU-LSTM, Transformer | TensorFlow / Keras |
| **Reinforcement Learning** | Dueling DQN | TensorFlow |
| **Ensemble** | CNN+LSTM+XGBoost hybrid, Adaptive Regime, Stacking, Meta-Ensemble | Mixed |
| **Meta** | Regime Classifier | scikit-learn |

### Committee-Based Trading
- **Multi-agent ensemble** — route trades to the best model per market regime
- **Expert profiler** — track per-regime model performance over time
- **UCB1 bandit** — multi-armed bandit for dynamic model selection
- **LLM-assisted exploration** — suggest new committee configurations
- **Governor cache** — deduplicate repeated decisions

### Regime Detection
- **Hidden Markov Model** — detect trending, ranging, high-vol, low-vol regimes
- **Anchored detection** — walk-forward safe regime assignment
- **Regime features** — regime column as model input feature

### Execution Simulation
- **Position Sizing**: Fixed lot, Fixed Fractional (% equity), Kelly Criterion, ATR-based, Conviction-based
- **Stop-Loss / Take-Profit**: Fixed pips, ATR-dynamic, Breakeven management, Partial close (scale-out)
- **Trailing Stops**: Standard (fixed pips), ATR trailing, Chandelier Exit
- **Risk Management**: Max drawdown circuit breaker, consecutive loss limits, daily loss limits, cooloff periods
- **Cost Modeling**: Spread + slippage modeled per bar, configurable via `ExecutionConfig`

### Feature Engineering
- **80+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, CCI, Williams %R, OBV, VWAP, Ichimoku, and more
- **Fractional Differentiation** — stationarity-preserving feature transforms
- **BorutaSHAP Feature Selection** — SHAP-based feature importance with Boruta wrapper
- **News Sentiment**: VADER (default), finBERT (opt-in), or LLM (Ollama/OpenAI/Anthropic)
- **Economic Event Flags**: NFP, FOMC, CPI, GDP, Retail Sales, PMI, ECB Rate, BOE Rate proximity markers
- **3-layer Feature Cache**: In-memory → Parquet disk cache → fresh computation (saves 30-120s per rerun)

### News & Sentiment Pipeline
- **RSS Sources**: Reuters, Bloomberg, ForexLive, Investing.com
- **NewsAPI**: Optional integration with API key
- **Economic Calendar**: 8 major event types with date-based proximity features
- **LLM Sentiment**: Structured prompts → direction + confidence + volatility per article
- **Deduplication**: Hash-based title deduplication across sources
- **Caching**: Parquet disk cache in `news_cache/`

### Hyperparameter Optimization
- **Optuna-powered** HPO with per-model search spaces
- **18 model-specific search spaces** defined in `config.py`
- **Best configs** stored in `hpo/` directory for reproducibility
- **Smoke mode** (1 trial) vs full mode (all trials) for quick iteration
- **Adaptive pruning** with patience and improvement tracking

### Model Persistence & Deployment
- **Model snapshots** — save trained models with metadata, checksums, pip freeze
- **Deployed models registry** — CRUD operations with tags and versioning
- **Fast retrain** — warm-start from saved snapshots
- **Live prediction bridge** — `/active/predict` endpoint for real-time inference

### Live Trading
- **OANDA v20 REST client** — market, limit, stop orders with SL/TP
- **Paper trading** — run models against live prices without real money
- **Live trading** — real execution with 4-layer risk architecture
- **17 risk gates** — pre-trade, post-trade, infrastructure, kill switch
- **Committee-based trading** — regime-switching model routing in production
- **Model rotation** — scheduled model updates for live deployment

### Overfitting Detection & Transparency
- **Train/test score divergence** — detect when model overfits
- **Bootstrap confidence intervals** — Sharpe, return, max DD
- **Walk-forward transparency panel** — per-period train vs test breakdown
- **Feature importance heatmap** — gain-based (tree) or permutation (others)
- **Plain-English summary** — auto-generated natural language backtest report
- **LLM advisor** — AI-powered analysis of backtest results

## Project Structure

```
KodaQuant/
├── config.py                       # Global config + PIPELINE_CONSTANTS + ExecutionConfig + SEARCH_SPACE
├── utilsNoWFO.py                   # Legacy monolith utilities (~1600 lines, gradual migration)
├── run_server.py                   # Production entry point for PyInstaller bundle
├── logging_config.py               # Structured logging + emit_event for Electron progress
│
├── pipeline/                       # Core backtesting engine (~18,000+ lines)
│   ├── backtester/                 # Mixin-composed MLBacktester
│   │   ├── composed.py             # MLBacktester (inherits 11 mixins)
│   │   ├── run_mixin.py            # HPO loop, Optuna study, walk-forward optimization
│   │   ├── real_trading_mixin.py   # Real trading simulation + equity curve
│   │   ├── strategy_mixin.py       # Signal generation + slippage modeling
│   │   ├── features_mixin.py       # Feature engineering (TA indicators)
│   │   ├── ensemble_mixin.py       # Ensemble training/prediction
│   │   ├── deep_mixin.py           # Deep learning model handling
│   │   ├── dqn_mixin.py            # DQN/RL model handling
│   │   ├── model_factory_mixin.py  # Model creation dispatch
│   │   ├── evaluation_mixin.py     # CV evaluation + metrics
│   │   ├── data_mixin.py           # Data loading + date range
│   │   ├── core_mixin.py           # Core state + config merging
│   │   └── execution_patches.py    # Execution loop integration
│   │
│   ├── execution/                  # Trading execution models
│   │   ├── position_sizing.py      # Kelly, fixed fractional, ATR, conviction sizing
│   │   ├── stops.py                # SL/TP management + breakeven + scale-out
│   │   ├── trailing.py             # Standard, ATR, Chandelier trailing stops
│   │   ├── risk_manager.py         # DD circuit breaker, loss limits, cooloff
│   │   ├── conviction_sizer.py     # Confidence-based position sizing
│   │   └── execution_utils.py      # Execution helper utilities
│   │
│   ├── tuning/                     # Optuna HPO
│   │   ├── runner.py               # Full Optuna study lifecycle
│   │   ├── objective.py            # Single-trial objective with retry
│   │   ├── refit.py                # Deep model refit functions
│   │   ├── sampler.py              # Parameter sampling from search spaces
│   │   ├── helpers.py              # HPO helper functions
│   │   ├── adaptive_pruner.py      # Adaptive pruner with patience
│   │   └── fixed_config.py         # Fixed config defaults per model
│   │
│   ├── metrics/                    # Evaluation metrics
│   │   ├── metrics_eval.py         # 16-metric evaluation, confusion matrix, HAC Sharpe
│   │   ├── overfitting.py          # Overfitting detection, DSR, PSR
│   │   ├── diagnostics.py          # CV quality diagnostics
│   │   ├── summary_generator.py    # Plain-English summary generation
│   │   ├── dsr.py                  # Deflated Sharpe Ratio
│   │   ├── pbo.py                  # Probability of Backtest Overfitting
│   │   ├── attribution.py          # Feature attribution analysis
│   │   └── calibration.py          # Probability calibration
│   │
│   ├── features/                   # Feature engineering
│   │   ├── feature_utils.py        # TA indicators, fractional differentiation
│   │   ├── feature_sweep.py        # Feature importance sweep
│   │   ├── boruta_sweep.py         # BorutaSHAP feature selection
│   │   └── feature_cache.py        # Parquet disk cache
│   │
│   ├── committee/                  # Multi-agent committee system
│   │   ├── committee_backtester.py # Walk-forward committee evaluation
│   │   ├── committee_builder.py    # Regime-to-model assignment
│   │   ├── expert_profiler.py      # Per-regime model profiling
│   │   ├── factory_llm.py          # LLM-assisted committee exploration
│   │   ├── factory_ucb.py          # UCB1 multi-armed bandit
│   │   └── governor_cache.py       # Decision deduplication
│   │
│   ├── regime/                     # Regime detection
│   │   ├── hmm_regime.py           # Hidden Markov Model regime detection
│   │   └── regime_utils.py         # Anchored detection, regime attachment
│   │
│   ├── models/                     # Model management
│   │   ├── model_persistence.py    # Save/load snapshots with metadata
│   │   ├── model_registry_disk.py  # Disk-based deployed models registry
│   │   ├── fast_retrain.py         # Warm-start model retraining
│   │   ├── model_defaults.py       # Default hyperparameters per model
│   │   ├── model_comparison.py     # Cross-model comparison and ranking
│   │   └── meta_labeler.py         # Meta-labeler for ensemble filtering
│   │
│   ├── llm/                        # LLM integration
│   │   ├── sentiment.py            # LLM sentiment engine (Ollama/OpenAI/Anthropic)
│   │   ├── advisor.py              # LLM backtest diagnostics advisor
│   │   └── prompts.py              # LLM prompt templates
│   │
│   ├── data/                       # Data layer
│   │   ├── data_sqlite.py          # SQLite store with indexed candle storage
│   │   ├── candle_syncer.py        # OANDA candle sync with gap detection
│   │   ├── data_downloader.py      # Historical data download
│   │   └── pair_config.py          # FX pair configuration
│   │
│   ├── main_cli.py                 # CLI runner (headless backtesting)
│   ├── workers.py                  # Multiprocessing worker pool
│   ├── hardware_profile.py         # CPU/GPU detection + VRAM measurement
│   ├── forward_test.py             # Forward testing engine for saved models
│   └── runtime.py                  # Runtime configuration merging
│
├── models/                         # ML model implementations (18 types)
│   ├── registry.py                 # Model registry with @register_model decorator
│   ├── base_model.py               # Abstract base model interface
│   ├── logistic.py, svm.py, random_forest.py, decision_tree.py
│   ├── xgboost_model.py, lightgbm_proxy.py
│   ├── cnn.py, lstm.py, gru.py, gru_lstm.py, transformer.py
│   ├── ensemble_cnn_lstm_xgboost.py
│   ├── ensemble_adaptive_regime.py
│   ├── stacking_ensemble.py
│   ├── meta_ensemble.py
│   └── regime_classifier.py
│
├── rl/                             # Reinforcement learning
│   ├── dqn_agent.py                # Dueling DQN agent (TensorFlow)
│   ├── environment.py              # Gym-style trading environment
│   ├── replay_buffer.py            # Experience replay
│   └── wrappers.py                 # Reward processing, cost-aware wrappers
│
├── news/                           # News & sentiment pipeline
│   ├── scraper.py                  # RSS + NewsAPI + economic calendar
│   ├── sentiment.py                # VADER / finBERT scoring
│   └── features.py                 # Walk-forward-safe feature merge
│
├── trading/                        # Live trading execution
│   ├── oanda_client.py             # OANDA v20 REST API client
│   ├── live_engine.py              # Live trading engine with risk controls
│   ├── paper_engine.py             # Paper trading engine
│   ├── committee_engine.py         # Committee-based live trading
│   ├── risk_controls.py            # 4-layer risk architecture (17 gates)
│   ├── live_committee_runner.py    # Live committee runner
│   ├── model_store.py              # Model artifact loading for live
│   ├── rotation_scheduler.py       # Model rotation scheduling
│   ├── lean_bridge.py              # QuantConnect LEAN integration
│   └── alerting.py                 # Trade and risk event alerting
│
├── api/                            # FastAPI backend (~10,000+ lines)
│   ├── main.py                     # FastAPI app + 15 routers + lifecycle
│   ├── tasks.py                    # Celery background tasks
│   ├── routers/
│   │   ├── backtest.py             # Backtest job submission + results
│   │   ├── models.py               # Model registry + HPO study management
│   │   ├── committee.py            # Committee builder + profiler + full-cycle
│   │   ├── trading.py              # Paper + live trading endpoints
│   │   ├── live.py                 # Live model deployment
│   │   ├── news.py                 # News + sentiment endpoints
│   │   ├── prices.py               # Historical + live price data
│   │   ├── price_stream.py         # Real-time WebSocket price streaming
│   │   ├── data.py                 # Data management
│   │   ├── pairs.py                # FX pair listing
│   │   ├── ws.py                   # WebSocket real-time progress
│   │   ├── license.py              # License verification (7 endpoints)
│   │   ├── config.py               # Configuration endpoints
│   │   ├── hardware.py             # GPU hardware info
│   │   └── health.py               # Health check
│   ├── licensing/                   # Paddle licensing system
│   │   ├── manager.py              # License state machine
│   │   ├── paddle_client.py        # Paddle v3 API client
│   │   ├── storage.py              # Encrypted license storage
│   │   ├── fingerprint.py          # Machine fingerprinting
│   │   └── gates.py                # Feature gating by license tier
│   ├── middleware/                   # Security middleware
│   │   └── __init__.py             # Security headers + rate limiting
│   └── schemas/                    # Pydantic v2 request/response models
│
├── schemas/                        # Top-level config validators
│   ├── backtest.py                 # BacktestParams, BacktestResult
│   ├── features.py                 # Feature config validation
│   ├── hpo.py                      # HPO config validation
│   └── settings.py                 # Settings validation
│
├── frontend/                       # React 19 app (~35,000+ lines)
│   ├── src/
│   │   ├── pages/                  # 10 pages
│   │   │   ├── Dashboard/          # Live prices, candlestick, Market Pulse, KPIs
│   │   │   ├── Backtest/           # 6-tab config UI (20+ components)
│   │   │   ├── Results/            # Equity curves, trade viz, metrics, export
│   │   │   ├── Compare/            # Leaderboard, significance, cross-pair
│   │   │   ├── Committee/          # Committee builder + profiler + full-cycle
│   │   │   ├── Trading/            # Paper + live trading dashboard
│   │   │   ├── Models/             # Deployed models registry + detail
│   │   │   ├── News/               # Sentiment analysis + economic calendar
│   │   │   ├── Monitor/            # HPO monitoring + trial feed
│   │   │   └── Settings/           # App config + licensing + data sources
│   │   ├── components/             # 60+ shared components
│   │   ├── stores/                 # 7 Zustand stores
│   │   ├── hooks/                  # Custom React hooks
│   │   └── api/                    # REST + WebSocket client layer
│   └── package.json
│
├── electron/                       # Electron desktop shell (~1,045 lines)
│   ├── main.ts                     # Main process + window creation
│   ├── python.ts                   # Python backend lifecycle
│   ├── license.ts                  # License IPC handlers
│   ├── splash.ts                   # Splash screen
│   ├── updater.ts                  # Auto-update via electron-updater
│   ├── sentry.ts                   # Crash reporting (opt-in)
│   ├── anti_debug.ts               # Anti-debugging protections
│   ├── menu.ts                     # Native application menu
│   └── tray.ts                     # System tray icon
│
├── csv_data/                       # Historical FX data (24 CSV files)
│   ├── EURUSD_10_years_{M15,M30,H1,H4}_OANDA.csv
│   ├── GBPUSD_10_years_{M15,M30,H1,H4}_OANDA.csv
│   ├── USDJPY_10_years_{M15,M30,H1,H4}_OANDA.csv
│   ├── AUDUSD_10_years_{M15,M30,H1,H4}_OANDA.csv
│   ├── USDCAD_10_years_{M15,M30,H1,H4}_OANDA.csv
│   └── GBPJPY_10_years_{M15,M30,H1,H4}_OANDA.csv
│
├── hpo/                            # Best HPO configs (20 JSON files)
├── tests/                          # 2,028 tests across 95+ test files
├── scripts/                        # 49 build/utility scripts
├── configs/                        # Feature + DQN grid configs
├── lean/                           # QuantConnect LEAN integration
│
├── run_smoke.bat                   # Smoke test (all models, 1 trial)
├── run_all_tests.bat               # Full test suite
├── run_comparison.bat              # Multi-model comparison runner
└── run_smoke_gpu.bat               # GPU smoke test (Windows → WSL)
```

## Installation

### 1. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 2. (Optional) GPU Support for Deep Models
Deep models (CNN, LSTM, Transformer, GRU) benefit from CUDA. Install the appropriate TensorFlow:
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
# For LLM sentiment (Ollama, OpenAI, or Anthropic):
# Configure llm_backend and llm_api_key in config
```

### 4. (Optional) Celery Task Queue
```powershell
pip install celery[redis] redis
# Requires a running Redis instance for background jobs
```

### 5. Verify Installation
```powershell
# Quick smoke test — all 18 models, 1 trial each
.\run_smoke.bat

# Full test suite — 2,028 tests
.\run_all_tests.bat
```

## Usage

### Running the Desktop App (Electron)

```powershell
# Development mode
cd frontend; npm run dev:electron

# Production build
.\scripts\build_electron.bat
```

The React UI provides pages for:
1. **Dashboard** — Live price ticker, candlestick charts, Market Pulse sentiment, KPIs
2. **Backtest** — Configure and run backtests (6-tab layout: Quick Start, Asset & Model, Study & HPO, Features, Hyperparameters, Execution)
3. **Results** — Equity curves, trade visualization, metrics, walk-forward transparency, training diagnostics, export
4. **Compare** — Model leaderboard, significance testing, cross-pair overlay, parameter sensitivity
5. **Committee** — Multi-agent ensemble builder, expert profiler, regime-based routing
6. **Trading** — Paper + live OANDA trading, position monitor, risk controls, kill switch
7. **Models** — Deployed models registry, model detail, fast retrain
8. **News** — Sentiment analysis, economic calendar, LLM advisor
9. **Monitor** — HPO monitoring, trial feed, diagnostics
10. **Settings** — App config, licensing, data sources

### Running via CLI (Headless)

```powershell
# Run a single model backtest
python -m pipeline.main_cli --model xgboost --pair EURUSD --tf H1 --months 3

# Run with environment variables
$env:MODEL_LIST="logistic,xgboost,cnn,lstm"
$env:SEEDS="42"
$env:N_MONTHS="3"
python -m pipeline.main_cli
```

### Model Comparison & Leaderboard

```powershell
# Smoke mode — all 18 models, 1 trial, 1 month (fast)
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
# POST   /api/v1/models/deploy         — Deploy model for live trading
# POST   /api/v1/live/deploy           — Start paper/live trading
# GET    /api/v1/committee/backtest    — Committee backtesting
# GET    /api/v1/prices/live           — Live price data
# GET    /api/v1/news/events           — Economic calendar events
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
    sizing_method="kelly",           # fixed, fixed_fractional, kelly, atr, conviction
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

### Live Trading

```python
from trading.oanda_client import OandaClient
from trading.paper_engine import PaperTradingEngine
from trading.live_engine import LiveTradingEngine

# Connect to OANDA
client = OandaClient(api_key="YOUR_KEY", account_id="YOUR_ACCOUNT", demo=True)

# Paper trading
paper = PaperTradingEngine(oanda_client=client, model=trained_model)
paper.start(pair="EURUSD", timeframe="H1")

# Live trading (with risk controls)
live = LiveTradingEngine(oanda_client=client, model=trained_model)
live.start(pair="EURUSD", timeframe="H1", max_position_pct=0.02)
```

## Testing

```powershell
# Smoke test — all 18 models end-to-end (2-5 min)
.\run_smoke.bat

# Full test suite — 2,028 tests (20-60 min via WSL)
.\run_all_tests.bat

# Run specific test file
python -m pytest tests/test_pipeline_validation.py -v

# Run with coverage
python -m pytest tests/ --cov=pipeline --cov=models --cov-report=term-missing

# GPU smoke test (Windows → WSL + CUDA)
.\run_smoke_gpu.bat
```

### Test Categories
| Category | Files | Description |
|----------|-------|-------------|
| Pipeline integration | 5 files | E2E pipeline validation, stress tests, wiring |
| Full cycle | 4 files | Full cycle bugs, mock, real data, wiring |
| Committee | 3 files | Committee builder, backtester, engine |
| Models | 3 files | Model training/prediction, defaults, features |
| Trading | 4 files | Trading engine, paper engine, risk controls |
| API | 3 files | API endpoints, E2E, price streaming |
| Data | 3 files | Data robustness, leakage detection, WAL concurrency |
| HPO | 2 files | Pipeline HPO E2E, CV optimization |
| Features | 3 files | Feature sweep, cache, BorutaSHAP |
| Hardware | 3 files | Hardware discovery, GPU detection, VRAM math |
| Security | 2 files | Security audit, licensing |
| Branding | 1 file | 37 branding tests |
| Build | 1 file | 46 build validation tests |
| E2E | 1 file | End-to-end racecar test |
| Smoke | 5 files | Model smoke, full smoke, import smoke |

## Currency Pairs & Data

| Pair | Timeframes | Years | Source |
|------|-----------|-------|--------|
| EURUSD | M15, M30, H1, H4 | 10 | OANDA |
| GBPUSD | M15, M30, H1, H4 | 10 | OANDA |
| USDJPY | M15, M30, H1, H4 | 10 | OANDA |
| AUDUSD | M15, M30, H1, H4 | 10 | OANDA |
| USDCAD | M15, M30, H1, H4 | 10 | OANDA |
| GBPJPY | M15, M30, H1, H4 | 10 | OANDA |

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
- 18 model-specific search spaces with literature-based ranges
- Logistic, SVM, Random Forest, XGBoost, LightGBM, CatBoost: scikit-learn params
- CNN, LSTM, GRU, Transformer: architecture + training params
- Ensemble models: voting weights, regime thresholds

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_LIST` | all | Comma-separated model names to run |
| `SEEDS` | 42 | Random seeds (comma-separated) |
| `REPEATS` | 1 | Number of repeat runs |
| `N_MONTHS` | 3 | Walk-forward months |
| `SMOKE_TEST` | 0 | Set to 1 for fast smoke mode |
| `NEWSAPI_KEY` | — | NewsAPI key (optional) |
| `KODAQUANT_VERBOSE` | 0 | Set to 1 for verbose HPO logging |

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **UI** | React + TypeScript + TailwindCSS + shadcn/ui | 19 / 6 / 4 |
| **Desktop** | Electron | 36 |
| **API** | FastAPI + Uvicorn | — |
| **Task Queue** | Celery + Redis | — |
| **Data** | Pandas + NumPy + SQLite + Parquet | — |
| **ML (shallow)** | scikit-learn + XGBoost + LightGBM + CatBoost | — |
| **ML (deep)** | TensorFlow / Keras | — |
| **HPO** | Optuna | — |
| **Charts** | lightweight-charts + Recharts + Plotly | — |
| **News** | feedparser + VADER / HuggingFace Transformers | — |
| **LLM** | Ollama + OpenAI + Anthropic APIs | — |
| **Testing** | pytest + Vitest | — |
| **Build** | PyInstaller + electron-builder | — |

## Roadmap

| Sprint | Topic | Status |
|--------|-------|--------|
| **S1** | Model Comparison & Leaderboard | DONE |
| **S2** | Advanced Execution Models | DONE |
| **S3** | Multi-Currency Expansion | DONE |
| **S4** | Docker + CI/CD | PARTIAL |
| **S5** | Comprehensive Tests + Benchmarks | DONE |
| **S6** | News & Sentiment Features | DONE |
| **S7** | FastAPI Backend | DONE |
| **S8** | React Frontend | DONE |
| **S8B** | Frontend ↔ API Integration | DONE |
| **S9** | Electron Desktop Shell | DONE |
| **S10** | Security & Licensing (Paddle) | DONE |
| **S11** | Installer & Auto-Update | DONE |
| **S12** | Product Intelligence & UX | DONE |
| **S13** | Beta & Launch | TODO |
| **S14** | Pipeline Enhancements | DONE |
| **S15** | KodaQuant Branding | DONE |
| **S16** | Overfitting & Transparency | DONE |
| **S16.8** | Model Persistence & Deployment | DONE |
| **S16.9** | Forward Test + Live Trading Bridge | IN PROGRESS |
| **S17** | UI Polish & Search | TODO |
| **S18** | Live News & Market Data | PARTIAL |
| **S19** | Ensemble Models & Extensibility | PARTIAL |
| **S20** | LLM / AI Integration | PARTIAL |
| **S21** | Live Trading (OANDA) | DONE |
| **S22** | Commercial Infrastructure | TODO |
| **S23** | Pipeline Stability & Live Monitor UX | DONE |
| **S24** | Historical News as Backtest Features | TODO |

**Product target**: Commercial Electron desktop app (React + FastAPI + Python), sold via Paddle.
**Pricing**: Hybrid — one-time purchase + annual updates subscription.

See `docs/ROADMAP.md` for full sprint details with sub-tasks and file references.

## License

Proprietary. All rights reserved.
