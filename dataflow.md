Here's the complete data architecture — every data source, every functionality, what it fetches and how.
---
The Five Data Stores
┌─────────────────────────────────────────────────────────────────────┐
│  APPLICATION DATA LAYER                                             │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │  forex.db         │  │  secure.db        │  │  news_cache/    │   │
│  │  (SQLite, WAL)    │  │  (Encrypted SQLite)│  │  (Parquet)      │   │
│  │                   │  │                   │  │                 │   │
│  │  • candles        │  │  • api_keys        │  │  articles.parq  │   │
│  │  • pairs          │  │  • licenses        │  │                 │   │
│  │  • jobs           │  │  • trial           │  │  llm_sentiment  │   │
│  │                   │  │  • kv_store        │  │  _cache.db      │   │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘   │
│                                                                     │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐     │
│  │  OANDA API (external) │  │  CSV files (deprecated, legacy) │     │
│  │  • PricingInfo        │  │  csv_data/*.csv                 │     │
│  │  • PricingStream      │  └──────────────────────────────────┘     │
│  │  • InstrumentsCandles │                                          │
│  └──────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
---
Data Store Schemas
1. forex.db (SQLite, WAL mode) — Operational Database
candles — Historical OHLCV bars:
Column	Type	Description
pair	TEXT	e.g. "EURUSD"
timeframe	TEXT	"M15", "M30", "H1", "H4"
ts	TEXT	ISO datetime with timezone (PK)
mid_open, mid_high, mid_low, mid_close	REAL	Mid prices
bid_open, bid_close, ask_open, ask_close	REAL	Bid/ask prices
spread	REAL	Bid-ask spread
volume	INTEGER	Tick volume
Index: (pair, timeframe, ts). PK: (pair, timeframe, ts). Uses INSERT OR REPLACE (upsert).
pairs — Currency pair metadata:
Column	Description
symbol	e.g. "EURUSD" (PK)
oanda_name	e.g. "EUR_USD"
pip_value, lot_size, base_currency, quote_currency, typical_spread_bps	 
jobs — Backtest job tracking:
Column	Description
id	UUID (PK)
type	"backtest" or "download"
status	"pending" → "running" → "completed" / "failed"
config	JSON blob: pair, models, date range, HPO settings, feature toggles
result	JSON blob: full backtest output (metrics, equity curves, trades, HPO data)
error, created_at, updated_at	 
2. secure.db (Encrypted SQLite, Fernet) — Credentials
api_keys:
Column	Description
key_name	"oanda", "openai", "anthropic"
encrypted_value	Fernet-encrypted API key
updated_at	ISO timestamp
licenses, trial, kv_store — license state, trial tracking, generic settings.
3. news_cache/ (Parquet files) — News Articles
articles.parquet — Flat table with one row per article:
title, body, timestamp, source, url, pair_tags, dedup_hash
4. news_cache/llm_sentiment_cache.db (SQLite) — LLM Scores
llm_sentiment_cache:
Column	Description
article_hash	SHA256 of pair:title:body (PK)
direction, confidence, volatility	REAL sentiment scores
currencies_affected	JSON array
pair, created_at	 
---
Complete Data Flow — Per Functionality
DASHBOARD PAGE
DashboardPage
│
├─ PriceTicker (NEW — S12.5)
│   ├─ useLivePrices(["EURUSD","GBPUSD","USDJPY"])
│   │   │
│   │   ▼
│   │   GET /prices/live?pairs=EURUSD,GBPUSD,USDJPY&lookback_bars=50
│   │   │
│   │   ├── Reads from:  secure.db::api_keys["oanda"] ──► OANDA PricingInfo ──► bid/ask/mid
│   │   └── Reads from:  forex.db::candles (last 50 bars per pair, M30)
│   │                    ──► sparkline [{t, v}]
│   │
│   └─ Renders: 3 pair cards (bid/ask/mid/change% + SVG sparkline)
│
├─ CandlestickChart (NEW — S12.5)
│   ├─ useCandles(activePair, "M30", 200)
│   │   │
│   │   ▼
│   │   GET /candles/EURUSD/M30?limit=200
│   │   │
│   │   └── Reads from:  forex.db::candles ──► [{t, o, h, l, c, volume}]
│   │
│   └─ Renders: lightweight-charts CandlestickSeries + volume histogram
│
├─ QuickActions
│   └─ No data — static navigation buttons
│
├─ MarketPulsePanel
│   ├─ useLiveSentiment(activePair)
│   │   │
│   │   ▼
│   │   GET /news/sentiment/live?pair=EURUSD
│   │   │
│   │   ├── Reads from:  news_cache/articles.parquet (all cached articles)
│   │   │                ──► scored by VADER (in-process) ──► vader_sentiment
│   │   ├── Reads from:  pipeline/llm/sentiment.py (Ollama)
│   │   │                ──► llm_sentiment, confidence, volatility
│   │   └── Returns:     blended_sentiment + vader/llm breakdown + top_articles[]
│   │
│   ├─ useNewsStatus()
│   │   │
│   │   ▼
│   │   GET /news/status
│   │   │
│   │   └── Reads from:  news_cache/*.parquet (row count only)
│   │
│   └─ Renders: sentiment gauge (SVG donut) + top 5 articles with VADER scores
│
├─ DashboardKPIs
│   ├─ useJobHistory(50)
│   │   │
│   │   ▼
│   │   GET /backtest?limit=50&offset=0
│   │   │
│   │   └── Reads from:  forex.db::jobs ──► [{job_id, status, pair, models, created_at}]
│   │
│   └─ useQuery(dashboard-aggregate) → fan-out GET /backtest/{id}/results per completed job
│       │
│       └── Reads from:  forex.db::jobs.result (JSON column, full metrics per job)
│                        ──► compute: avgSharpe, avgWinRate, profitableMonthsPct
│
├─ RecentJobsTable (5 items)
│   ├─ Same data as KPIs (equityDataMap from fan-out results)
│   └─ useDeleteJob() → DELETE /backtest/{id} → Writes to: forex.db::jobs
│
└─ (Heatmap → moved to /results page)
BACKTEST PAGE
BacktestPage
│
├─ ModelSelector
│   ├─ useModels() → GET /models
│   │   └── Reads from:  models/registry.py (static, in-code)
│
├─ AssetSelector
│   ├─ usePairs() → GET /pairs
│   │   └── Reads from:  forex.db::pairs + forex.db::candles (timeframe list + date ranges)
│   │
│   └─ GET /backtest/date-ranges?pair=EURUSD&timeframe=H1
│       └── Reads from:  forex.db::candles (MIN/MAX ts per pair+timeframe)
│
├─ FeaturesPanel
│   └─ No data — static toggles (values flow into POST /backtest config_overrides)
│
├─ HpoPanel
│   └─ No data — static inputs (hpo_intensity, max_hpo_duration)
│
├─ RunSummary
│   └─ No data — modal confirmation before submission
│
├─ BacktestProgress
│   ├─ useJobStatus(jobId) → GET /backtest/{jobId} (polls every 2s while running)
│   │   └── Reads from:  forex.db::jobs (status + progress field)
│   │
│   └─ WebSocket WS /backtest/{jobId}/ws
│       └── Reads from:  Redis (pub/sub channel job:{jobId})
│                        Events: job_started, model_training, hpo_progress,
│                        month_progress, job_complete, job_failed
│
└─ Submit (Run Backtest button)
    │
    ▼
    POST /backtest
    │
    ├── Writes to:  forex.db::jobs (create row, status="pending")
    │
    └── Dispatches to:  Celery worker (or synchronous if desktop mode)
                        │
                        ▼
                        _run_backtest_impl(job_id, config)
                        │
                        ├── [1] Data loading
                        │   └── Reads from:  forex.db::candles (via DataStore.get_candles)
                        │                    ──► OHLCV DataFrame for pair+timeframe+date_range
                        │        Alternative: CSV files (legacy, deprecated path)
                        │
                        ├── [2] News injection (if use_news=True)
                        │   ├── Reads from:  news_cache/articles.parquet
                        │   │                ──► scored by VADER ──► news_aggregated
                        │   └── Injects into: bt._news_aggregated, bt._news_economic_events
                        │
                        ├── [3] LLM injection (if llm_sentiment_enabled=True)
                        │   ├── Reads from:  news_cache/articles.parquet
                        │   └── Reads from:  pipeline/llm/sentiment.py (Ollama)
                        │                    ──► llm_aggregated
                        │                    ──► (uses news_cache/llm_sentiment_cache.db for caching)
                        │       Injects into: bt._llm_aggregated
                        │
                        ├── [4] Feature engineering
                        │   └── prepare_features():
                        │       ├── Computes: TA indicators (SMA, EMA, MACD, RSI, BB, ATR, ADX, Stoch) in-process
                        │       ├── Reads from:  pipeline/feature_cache/ (Parquet disk cache — optional)
                        │       │               ──► cached engineered features (cache hit)
                        │       ├── Merges-in:  bt._news_aggregated → sentiment_score, news_volume_*
                        │       └── Merges-in:  bt._llm_aggregated → llm_sentiment, blended_sentiment, etc.
                        │       ──► df_out (500+ columns), features list
                        │
                        ├── [5] HPO (Optuna) — if hpo_intensity != "off"
                        │   ├── Reads from:  hpo/ directory (cached HPO configs per model — optional)
                        │   ├── Writes to:   results/ directory (param_importances.json, optuna dbs)
                        │   └── Runs: CV folds on train data → finds best hyperparameters
                        │
                        ├── [6] Walk-forward simulation
                        │   │
                        │   │  Monthly splits: train on past N months, test on next 1 month
                        │   │  ──► real_trading_simulation()
                        │   │
                        │   └── Per month:
                        │       ├── Model.fit(train_data) → model object
                        │       ├── Model.predict(test_data) → signals
                        │       ├── Execution simulation (slippage, spread, position sizing, stops)
                        │       │   └── Reads from:  bt.data (OHLCV, already loaded)
                        │       └── Produces: trade_log DataFrame per month
                        │
                        ├── [7] Metrics computation
                        │   └── Computes in-process from simulation output:
                        │       sharpe, sortino, max_drawdown, total_return_pct, cagr, calmar,
                        │       win_rate, profit_factor, avg_trade, active_rate, etc.
                        │       + equity_curve, drawdown_curve, monthly_results, trades
                        │
                        ├── [8] Progress events (during execution)
                        │   └── Writes to:  Redis pub/sub (channel job:{job_id})
                        │                  _pub(event, job_id, data)
                        │
                        └── [9] Store results
                            └── Writes to:  forex.db::jobs (UPDATE status, result JSON)
RESULTS PAGE — /results (History Browser)
ResultsHistoryPage
│
├─ useResultsHistory({pair, sort_by, sort_order, limit:100})
│   │
│   ▼
│   GET /backtest/results/summary?pair=EURUSD&sort_by=sharpe&sort_order=desc&limit=100&offset=0
│   │
│   └── Reads from:  forex.db::jobs (config + result JSON, filtered to status="completed")
│                    Iterates ALL 500+ jobs, extracts one row per model per job
│                    ──► [{job_id, pair, models, sharpe, total_return_pct, win_rate, max_drawdown_pct, ...}]
│
├─ PerformanceHeatmapSection (NEW — moved from Dashboard)
│   ├─ useHeatmap()
│   │   │
│   │   ▼
│   │   GET /backtest/heatmap
│   │   │
│   │   └── Reads from:  forex.db::jobs (500 completed jobs, all metrics)
│   │                    ──► {models[], pairs[], cells[{model, pair, sharpe, return, win_rate, max_dd, job_id}]}
│   │
│   └─ Renders: SVG heatmap matrix (model × pair)
│
├─ Search/filter (client-side)
│   └── Filters results[] array in memory by job_id or pair name
│
└─ Export
    ├── CSV: serializes filtered[] rows to CSV blob → browser download
    └── JSON: serializes filtered[] rows to JSON blob → browser download
RESULTS PAGE — /results/:jobId (Detail View)
ResultsPage
│
├─ useJobResults(jobId)
│   │
│   │  ── first: useJobStatus(jobId) → GET /backtest/{jobId}
│   │       Reads from:  forex.db::jobs (status, created_at, error)
│   │
│   │  ── when done: GET /backtest/{jobId}/results
│   │       Reads from:  forex.db::jobs (result JSON: all metrics, curves, trades)
│   │
│   └── Returns: {job_id, pair, models, config, metrics[{model, sharpe, ..., equity_curve, trades, ...}]}
│
├─ MetricsGrid (12 cards)
│   └── Extracted from: metrics[activeModelIdx]
│       Uses: sharpe, sortino, max_drawdown, total_return_pct, cagr, calmar,
│              win_rate, total_trades, profit_factor, avg_trade, active_rate,
│              directional_accuracy, precision_macro, f1_macro
│
├─ EquitySection
│   ├── EquityCurveChart ← metrics[].equity_curve [{time, value}]
│   ├── Buy & Hold overlay ← metrics[].buy_hold_curve [{time, value}]
│   ├── Drawdown histogram ← metrics[].drawdown_curve [{time, value}]
│   └── Events overlay (optional):
│       └─ useNewsEvents(start, end, "high,medium") → GET /news/events?start=X&end=Y
│           └── Reads from:  NewsScraper.economic_calendar_events() (hardcoded calendar)
│
├─ MonthlySection
│   └── metrics[].monthly_results[{month, return_pct, win_rate, trades, sharpe}]
│
├─ TradeLogTable
│   └── metrics[].trades[{trade_id, entry_date, exit_date, direction, entry_price,
│                          exit_price, pips, return_pct, duration_bars, barrier_hit}]
│       Note: field names are from serialized DataFrame (raw columns) —
│       currently has schema mismatch with what frontend expects.
│
├─ BacktestChart (NEW — S12.5)
│   ├── useTradeChartData(jobId, model)
│   │   │
│   │   ▼
│   │   GET /backtest/{jobId}/trades/chart-data?model=xgboost
│   │   │
│   │   └── Reads from:
│   │       ├── forex.db::candles (OHLCV for the backtest date range + pair + timeframe)
│   │       │   ──► [{t, o, h, l, c}] candle bars
│   │       ├── forex.db::jobs.result → trades array
│   │       │   ──► entry/exit timestamps joined with candle close → entry/exit prices
│   │       └── forex.db::jobs.result → equity_curve
│   │           ──► [{time, value}] for overlay
│   │
│   └── Renders: CandlestickSeries + trade markers (▲buy ▼sell) + equity line
│
├─ HpoDiagnostics
│   └── metrics[].hpo_param_importance + metrics[].hpo_trials
│
├─ ConfigViewer
│   └── config (raw JSON tree, collapsible)
│
└─ ExportBar
    ├── CSV: trades[] → CSV file download (client-side)
    ├── PNG: equityCurveChart.takeScreenshot() → canvas.toDataURL() (client-side)
    └── JSON: full results object → JSON file download (client-side)
COMPARE PAGE
ComparePage
│
├── EquityOverlayChart
│    └── Fan-out: GET /backtest/{id}/results per model
│        └── Reads from:  forex.db::jobs.result → equity_curve per model
│
├── LeaderboardTable
│    └── Same data as above, extracted into sortable table
│
├── SignificanceMatrix
│    └── Paired t-test on monthly returns from leaderboard data (client-side)
│
├── CrossPairSection
│    ├── GET /backtest/cross-pair-curves?model=cnn&pairs=EURUSD,GBPUSD
│    │   └── Reads from:  forex.db::jobs (all completed jobs, filter by pair+model)
│    │       ──► equity_curve per pair for the selected model
│    └── Renders: CrossPairOverlayChart
│
└── ParameterSensitivityChart
    └── metrics[].hpo_trials → scatter chart (client-side rendering)
NEWS PAGE
NewsPage
│
├── useNewsStatus() → GET /news/status
│   └── Reads from:  news_cache/*.parquet (row count)
│       Returns: {sentiment_backend, cached_articles, event_types, finbert_available}
│
└── No other data — displays static cards
SETTINGS PAGE
SettingsPage
│
├── usePairs() → GET /pairs
│   └── Reads from:  forex.db::pairs + forex.db::candles (summary)
│
├── useModels() → GET /models
│   └── Reads from:  models/registry.py (static)
│
├── OANDA API Key input
│   └── Writes to:  secure.db::api_keys["oanda"] (encrypted)
│
├── useLicenseStatus() → GET /license/status
│   └── Reads from:  secure.db::licenses + secure.db::trial
│
└── Download Data button
    └── POST /data/download → Celery task
        └── Reads from:  OANDA InstrumentsCandles (historical download)
            Writes to:   forex.db::candles (INSERT OR REPLACE)
            Writes to:   forex.db::pairs (INSERT pairs metadata)
---
Summary Table — Which Data Store Powers What
Data Store	Read By	Write By
forex.db::candles	Dashboard (sparklines, candlesticks), Backtest (MLBacktester.load_data), Results (trade chart), Pairs API (date ranges)	Data downloader (migrate_pair), OANDA download Celery task
forex.db::pairs	Pairs API, Settings page, Backtest asset selector	Data downloader (on pair import)
forex.db::jobs	Dashboard (KPIs, recent activity), Results (history, detail), Compare page, BacktestProgress	POST /backtest (create), Celery worker (update status/result)
secure.db	Settings (license status), Live prices (OANDA key), Backtest (LLM key if configured)	Settings page (save OANDA/OpenAI keys), License activation
news_cache/*.parquet	MarketPulsePanel (articles for VADER), Backtest (news injection), News page (status)	NewsScraper.save_cache() (on fetch_all)
news_cache/llm_sentiment_cache.db	LLMSentimentEngine (cache hits on backtest + live sentiment)	LLMSentimentEngine (cache writes after LLM scoring)
OANDA API	Live price ticker (PricingInfo), Data downloader (InstrumentsCandles)	Nothing (read-only for this app)
Redis pub/sub	WebSocket progress (job events), Frontend (job_started/completed/etc.)	Backend (_pub calls during backtest execution)
pipeline/feature_cache/ (Parquet)	Feature engineering (disk cache hit — optional optimization)	Feature engineering (disk cache write after compute)
hpo/ directory	HPO warm-start (load previous best configs)	HPO execution (save param_importances.json)
results/ directory	Model comparison (post-hoc analysis), HPO diagnostics loading	HPO execution (Optuna studies, parameter importance)