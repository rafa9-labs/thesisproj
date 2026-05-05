# Sprint 12: Product Intelligence & UX Overhaul

> **Branch**: `feature/s12-product-intelligence`
> **Est**: 15-16h
> **Status**: NOT STARTED
> **Priority**: News pipeline fix (P0) → LLM sentiment (P0) → Results history (P1) → Dashboard redesign (P1)

---

## Execution Order

Tasks are ordered by dependency. S12.1 must be done first (it's a prerequisite for S12.2).

```
S12.1 Fix news pipeline ──▶ S12.2a LLM core ──▶ S12.2b Feature integration
                                  │                      │
                                  ▼                      ▼
                            S12.2c Frontend config    S12.2d Tests
                                  │
                                  ▼
                            S12.4 Dashboard redesign ──▶ needs S12.2c API endpoint

S12.3 Results history (independent, can run in parallel with S12.2)
```

---

## S12.1 — Fix Broken News Pipeline Wiring *(2h)*

### Problem
`_news_aggregated` and `_news_economic_events` are never set by the backtester or API tasks, so `use_news=True` silently does nothing. `news_sentiment_backend` config key is never read by Python.

### Files to Change
| File | Change |
|------|--------|
| `api/tasks.py` | Add news fetch+sentiment step before `_run_backtest_impl()`. Inject DataFrames into `bt._news_aggregated` and `bt._news_economic_events` |
| `pipeline/backtester/features_mixin.py` | When `use_news=True` and no data injected, log warning instead of silently skipping |
| `config.py` | Change `use_news` default to `True`. Thread `news_sentiment_backend` through config |
| `news/features.py` | Ensure `merge_news_features()` handles empty/None gracefully with clear logging |
| `tests/test_news.py` | Add integration test: full backtest with `use_news=True` asserting feature columns appear |

### Detailed Steps

1. **`api/tasks.py`** — In the backtest task function, before calling `_run_backtest_impl()`:
   ```python
   # After loading config but before running backtest
   if config_overrides.get("use_news", True):
       from news.scraper import NewsScraper
       from news.sentiment import SentimentAnalyzer
       from news.features import aggregate_to_df
       
       scraper = NewsScraper()
       articles = scraper.fetch_all()
       backend = config_overrides.get("news_sentiment_backend", "vader")
       analyzer = SentimentAnalyzer(backend=backend)
       scored = analyzer.score_articles(articles)
       news_aggregated = aggregate_to_df(scored, freq="1h")
       events = scraper.economic_calendar_events()
       
       bt._news_aggregated = news_aggregated
       bt._news_economic_events = events
   ```

2. **`pipeline/backtester/features_mixin.py`** — Change the news feature block (~line 721):
   ```python
   # OLD: silently skips when None
   use_news = bool(cfg.get("use_news", False))
   if use_news and not base_only:
       from news.features import merge_news_features, get_news_feature_columns
       news_agg = getattr(self, "_news_aggregated", None)
       # ... None means silently skip
   
   # NEW: warn when no data, still default ON
   use_news = bool(cfg.get("use_news", True))
   if use_news and not base_only:
       from news.features import merge_news_features, get_news_feature_columns
       news_agg = getattr(self, "_news_aggregated", None)
       econ_events = getattr(self, "_news_economic_events", None)
       if news_agg is None:
           logger.warning("use_news=True but no news data injected — skipping news features")
       else:
           df_out = merge_news_features(df_out, news_agg, events=econ_events, config=cfg)
           # ... add columns
   ```

3. **`config.py`** — Change `use_news` default:
   ```python
   # In PIPELINE_CONSTANTS
   "use_news": True,  # was False
   "news_sentiment_backend": "vader",  # thread this through to SentimentAnalyzer
   ```

4. **Test** — `tests/test_news.py` add:
   ```python
   def test_news_features_in_backtest(self):
       """Integration test: full backtest with use_news=True produces news feature columns."""
       # ... setup MLBacktester with use_news=True and injected news data
       # Assert 'sentiment_score', 'sentiment_magnitude', 'news_volume_1h' in output columns
   ```

### Acceptance Criteria
- [ ] `use_news=True` backtest (via API) produces `sentiment_score`, `sentiment_magnitude`, `news_volume_1h`, `news_volume_24h` columns in feature output
- [ ] `use_news=True` with no data available logs a warning instead of silently skipping
- [ ] `news_sentiment_backend` config flows from frontend → API task → SentimentAnalyzer
- [ ] Integration test passes with news features in output

---

## S12.2a — LLM Sentiment Engine: Core Module *(3h)*

### Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  News Article│────▶│  LLM Backend  │────▶│  Sentiment Score│
│  (+ pair ctx)│     │  (Ollama)     │     │  per article    │
└─────────────┘     └──────┬───────┘     └───────┬─────────┘
                           │                      │
                    ┌──────┴──────┐        ┌──────┴─────────┐
                    │  Cache       │        │  Feature Columns│
                    │  (SQLite)    │        │  (rolling, etc) │
                    └─────────────┘        └────────────────┘
```

### Files to Create
| File | Purpose |
|------|---------|
| `pipeline/llm/__init__.py` | Package init, expose `LLMSentimentEngine` |
| `pipeline/llm/sentiment.py` | Core LLM sentiment engine with backends + caching |
| `pipeline/llm/prompts.py` | Structured prompts for financial news analysis |

### Files to Modify
| File | Change |
|------|--------|
| `config.py` | Add LLM config keys (`llm_sentiment_enabled`, `llm_backend`, `llm_model`, etc.) |

### Detailed Design

**`pipeline/llm/sentiment.py`**:
```python
class LLMSentimentBackend(Protocol):
    def analyze(self, text: str, pair: str) -> dict: ...

class OllamaBackend:
    """Default. Free, local, private. Requires `ollama serve` running."""
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"): ...
    def analyze(self, text: str, pair: str) -> dict: ...

class OpenAIBackend:
    """Paid, cloud, fast. Requires OPENAI_API_KEY."""
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = ""): ...
    def analyze(self, text: str, pair: str) -> dict: ...

class AnthropicBackend:
    """Paid, cloud. Requires ANTHROPIC_API_KEY."""
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = ""): ...

class LLMSentimentEngine:
    """
    Main engine. Handles:
    - Backend selection (ollama/openai/anthropic)
    - Per-article caching (SQLite llm_sentiment_cache table)
    - Batch processing (collect articles per hour → single LLM call)
    - Fallback to VADER if LLM unavailable
    - Score aggregation → DataFrame ready for feature merge
    """
    def __init__(self, config: dict): ...
    
    def score_articles(self, articles: list[NewsArticle]) -> list[tuple[NewsArticle, dict]]: ...
    def get_live_sentiment(self, pair: str) -> dict: ...
    def _call_llm(self, text: str, pair: str) -> dict: ...  # calls backend
    def _cached_score(self, article_hash: str) -> dict | None: ...  # SQLite lookup
    def _cache_score(self, article_hash: str, scores: dict) -> None: ...  # SQLite write
    def _init_cache_db(self) -> None: ...  # Create table if not exists
```

**`pipeline/llm/prompts.py`**:
```python
SENTIMENT_ANALYSIS_PROMPT = """You are a forex market analyst. Analyze this news article's impact on {pair}.

Respond with ONLY a JSON object (no other text):
{{
  "direction": <float from -1.0 to 1.0, where -1=very bearish, 0=neutral, 1=very bullish>,
  "confidence": <float from 0.0 to 1.0>,
  "volatility": <float from 0.0 to 1.0, where 0=calm, 1=high volatility expected>,
  "currencies_affected": <list of currency codes, e.g. ["USD", "EUR"]>
}}

Article title: {title}
Article body: {body}"""

BATCH_SENTIMENT_PROMPT = """You are a forex market analyst. Analyze these {count} news articles' collective impact on {pair}.

For each article, respond with ONLY a JSON array (no other text):
[{{"direction": <float -1 to 1>, "confidence": <float 0 to 1>, "volatility": <float 0 to 1>, "currencies_affected": [<str>]}}, ...]

Articles:
{articles_text}"""
```

**`config.py`** additions:
```python
PIPELINE_CONSTANTS = {
    # ... existing ...
    "llm_sentiment_enabled": True,
    "llm_backend": "ollama",            # "ollama" | "openai" | "anthropic"
    "llm_model": "llama3",               # model name per backend
    "llm_api_key": "",                    # API key (empty for Ollama)
    "llm_weight": 0.7,                   # blending weight: LLM vs VADER
    "llm_batch_size": 10,                # articles per batch call
    "llm_cache_ttl_hours": 720,          # 30 days cache
    "llm_ollama_url": "http://localhost:11434",
}
```

### Acceptance Criteria
- [ ] `LLMSentimentEngine` with Ollama backend can process articles and return structured scores
- [ ] Per-article caching works: second call for same article returns cached result
- [ ] Fallback to VADER works when Ollama is unavailable
- [ ] OpenAI and Anthropic backends implemented (config-selectable)
- [ ] Config keys present in `config.py` with defaults

---

## S12.2b — LLM Sentiment: Feature Integration + Blending *(2h)*

### Files to Change
| File | Change |
|------|--------|
| `news/features.py` | Add `merge_llm_features()`, `get_llm_feature_columns()`, `compute_blended_sentiment()` |
| `pipeline/backtester/features_mixin.py` | Call `merge_llm_features()` when `llm_sentiment_enabled` |
| `api/tasks.py` | Call LLM analysis after news fetch, before backtester run |

### Feature Columns Added
| Column | Type | Description |
|--------|------|-------------|
| `llm_sentiment` | float32 | LLM directional sentiment [-1, 1] |
| `llm_confidence` | float32 | LLM confidence [0, 1] |
| `llm_volatility` | float32 | LLM volatility expectation [0, 1] |
| `llm_sentiment_ma_6` | float32 | 6-bar rolling mean of llm_sentiment |
| `llm_sentiment_ma_24` | float32 | 24-bar rolling mean of llm_sentiment |
| `blended_sentiment` | float32 | `llm_weight * llm_sentiment + (1 - llm_weight) * vader_sentiment` |

### Blending Formula
```python
blended_sentiment = llm_weight * llm_sentiment + (1 - llm_weight) * vader_sentiment
# Default: 0.7 * llm + 0.3 * vader (favor LLM when available)
# Falls back to pure vader when llm is disabled or unavailable
```

### Walk-Forward Safety
- All LLM features use forward-fill only (no future data)
- Rolling means computed with `min_periods=1` to avoid NaN at start
- If LLM data is missing for a bar, fill with 0 (neutral) and log warning

### Acceptance Criteria
- [ ] `merge_llm_features()` produces all 6 new columns
- [ ] Blending formula produces `blended_sentiment` correctly
- [ ] Walk-forward safe: all features use forward-fill, no look-ahead
- [ ] API task wires LLM analysis before backtest (when `llm_sentiment_enabled=True`)

---

## S12.2c — LLM Sentiment: Frontend Config + API *(2h)*

### Files to Change
| File | Change |
|------|--------|
| `frontend/src/pages/Backtest/FeaturesPanel.tsx` | Add LLM section: toggle, backend dropdown, model input, weight slider |
| `api/routers/news.py` | Add `GET /news/sentiment/live` endpoint |
| `api/schemas/backtest.py` | Add LLM config fields to BacktestParams |
| `frontend/src/api/queries.ts` | Add `useLiveSentiment()` hook |
| `frontend/src/lib/constants.ts` | Add LLM default values |

### API Endpoint Design
```
GET /api/v1/news/sentiment/live
Response: {
  "pairs": {
    "EURUSD": {
      "llm_sentiment": 0.35,
      "vader_sentiment": 0.22,
      "blended_sentiment": 0.31,
      "llm_confidence": 0.82,
      "llm_volatility": 0.45,
      "currencies_affected": ["EUR", "USD"],
      "last_updated": "2026-05-05T14:30:00Z",
      "article_count": 12
    },
    ...
  },
  "backend": "ollama",
  "model": "llama3"
}
```

### Frontend FeaturesPanel Addition
In the "News & Sentiment" section, after the existing toggles:
```tsx
{/* LLM Sentiment */}
{useNews && (
  <div className="space-y-3 pl-4 border-l-2 border-cyan-500/30">
    <Toggle label="LLM Sentiment" value={llmSentimentEnabled} onChange={setLlmSentimentEnabled} />
    {llmSentimentEnabled && (
      <>
        <Select label="Backend" value={llmBackend} options={["ollama", "openai", "anthropic"]} />
        <Input label="Model" value={llmModel} placeholder="llama3" />
        <Slider label="LLM Weight" value={llmWeight} min={0} max={1} step={0.1} />
      </>
    )}
  </div>
)}
```

### Acceptance Criteria
- [ ] FeaturesPanel shows LLM config when news toggle is ON
- [ ] Backend/model/weight values flow through to API task config
- [ ] `GET /news/sentiment/live` returns per-pair LLM+VADER+blended sentiment
- [ ] `useLiveSentiment()` hook fetches and caches live sentiment data

---

## S12.2d — LLM Sentiment: Tests *(1h)*

### Files to Create
| File | Purpose |
|------|---------|
| `tests/test_llm_sentiment.py` | Full test suite for LLM sentiment engine |

### Test Cases
1. **`test_ollama_backend_mock`** — Mock Ollama HTTP response, verify structured output
2. **`test_openai_backend_mock`** — Mock OpenAI API response, verify structured output
3. **`test_per_article_caching`** — Score article → cache hit on second call → verify no duplicate LLM call
4. **`test_batch_mode`** — Multiple articles → single batch call → individual scores extracted
5. **`test_fallback_to_vader`** — Ollama unavailable → falls back silently to VADER
6. **`test_blending_formula`** — Verify `llm_weight * llm + (1 - llm_weight) * vader`
7. **`test_feature_merge_walk_forward`** — LLM features merged with forward-fill, no future data
8. **`test_config_threading`** — LLM config keys flow from frontend → API task → engine

### Acceptance Criteria
- [ ] All 8 tests pass
- [ ] Mock fixtures for Ollama and OpenAI responses
- [ ] Cache database created and queried correctly

---

## S12.3 — Results History Browser *(3h)*

### Problem
`/results` without `jobId` shows empty state. No way to browse all past backtest results.

### Files to Create
| File | Purpose |
|------|---------|
| `frontend/src/pages/Results/ResultsHistoryPage.tsx` | Full results browser with table, sort, filter, search |

### Files to Modify
| File | Change |
|------|--------|
| `api/routers/backtest.py` | Add `GET /backtest/results/summary` endpoint; add `offset`/`limit` pagination |
| `api/schemas/backtest.py` | Add `BacktestSummaryResponse` schema |
| `frontend/src/App.tsx` | Add `/results` route → `ResultsHistoryPage` |
| `frontend/src/api/queries.ts` | Add `useResultsHistory()` hook |

### API Design
```
GET /api/v1/backtest/results/summary?limit=50&offset=0&pair=EURUSD&model=xgboost&sort_by=created_at&sort_order=desc
Response: {
  "results": [
    {
      "job_id": "abc123",
      "created_at": "2026-05-01T10:30:00Z",
      "pair": "EURUSD",
      "timeframe": "H1",
      "models": ["xgboost"],
      "sharpe": 1.42,
      "total_return_pct": 18.3,
      "win_rate": 0.58,
      "max_drawdown_pct": -8.2,
      "total_trades": 245,
      "status": "completed"
    },
    ...
  ],
  "total": 47,
  "limit": 50,
  "offset": 0
}
```

### ResultsHistoryPage Layout
```
┌─────────────────────────────────────────────────────┐
│  Results History                          [Search]  │
│  ┌─────────────────────────────────────────────┐    │
│  │ Filters: [Pair ▾] [Model ▾] [Status ▾]     │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────┐    │
│  │ Date      │ Pair   │ Models  │ Sharpe │ Ret% │ ✓ │
│  │ 05/01     │ EURUSD │ XGBoost│  1.42  │ 18.3 │ □ │
│  │ 04/28     │ GBPUSD │ LSTM   │  0.89  │  5.2 │ □ │
│  │ ...       │ ...    │ ...    │  ...   │ ...  │ □ │
│  └─────────────────────────────────────────────┘    │
│  [Export Selected CSV] [Export Selected JSON]        │
│  Showing 1-50 of 47 results                          │
└─────────────────────────────────────────────────────┘
```

- Click row → navigate to `/results/:jobId`
- Checkbox column → select multiple → bulk export
- Sort by clicking column headers
- Filter dropdowns for pair, model, status

### Acceptance Criteria
- [ ] `/results` shows a table of ALL completed backtests
- [ ] Sortable by date, pair, Sharpe, return, win rate, max DD
- [ ] Filterable by pair, model, status
- [ ] Search by job ID or pair name
- [ ] Click row navigates to `/results/:jobId` detail view
- [ ] Bulk export (CSV + JSON) of selected results
- [ ] Pagination for large result sets

---

## S12.4 — Dashboard Redesign: Live Command Center *(3h)*

### Problem
Dashboard is a static report. User wants 2-3 quick actions and the rest being live data.

### New Layout
```
┌─────────────────────────────────────────────────────┐
│  Quick Actions: [▶ Quick Run] [🔄 Re-run Last]     │
├─────────────────────────────────────────────────────┤
│  KPI Bar (4 metrics: Runs, Best Sharpe, Avg Win,    │
│           Best Return)                                │
├────────────────────────┬────────────────────────────┤
│  Recent Activity (10)   │  Market Pulse Panel         │
│  ┌──────────────────┐   │  ┌────────────────────────┐│
│  │ Job  │ Status │ Δt│   │  │ Sentiment Gauge        ││
│  │ abc  │ ✓ Run  │3m │   │  │  ████████░░ 0.65      ││
│  │ def  │ ⏳ Run  │1m │   │  │  EURUSD: Bullish      ││
│  │ ...  │ ...    │.. │   │  │  Confidence: 0.82      ││
│  └──────────────────┘   │  ├────────────────────────┤│
│                         │  │ Last 5 High-Impact News││
│                         │  │ • NFP: +0.45 (LLM)    ││
│                         │  │ • FOMC: -0.12 (LLM)   ││
│                         │  ├────────────────────────┤│
│                         │  │ Economic Calendar      ││
│                         │  │ ⏰ Wed 14:00 FOMC     ││
│                         │  │ ⏰ Fri 13:30 NFP       ││
│                         │  └────────────────────────┘│
├────────────────────────┴────────────────────────────┤
│  Performance Heatmap (Model × Pair)                    │
└─────────────────────────────────────────────────────┘
```

### Files to Create
| File | Purpose |
|------|---------|
| `frontend/src/pages/Dashboard/QuickActions.tsx` | 2-3 quick action buttons |
| `frontend/src/pages/Dashboard/MarketPulsePanel.tsx` | Live sentiment gauge + news + calendar |

### Files to Modify
| File | Change |
|------|--------|
| `frontend/src/pages/Dashboard/DashboardPage.tsx` | Restructure layout: QuickActions → KPIs → Activity + MarketPulse → Heatmap |
| `api/routers/news.py` | Add `GET /news/sentiment/live` (also needed by S12.2c) |

### QuickActions Component
- **Quick Run**: Pre-filled form with most-used config (pair from last run, default models, default timeframe). Opens backtest page with config pre-filled
- **Re-run Last**: Re-submit the exact same config as the last completed backtest

### MarketPulsePanel Component
- **Sentiment Gauge**: Circular progress showing blended sentiment [-1, 1] for selected/default pair
  - Color: green (bullish > 0.3), amber (neutral -0.3 to 0.3), red (bearish < -0.3)
  - Shows: LLM sentiment, VADER sentiment, blended score
- **News Feed**: Last 5 high-impact news with LLM direction scores
  - Each item: event name, direction arrow, confidence, timestamp
- **Economic Calendar**: Next 7 days of events from existing `economic_calendar_events()`
  - Each item: date, event name, impact level badge

### Acceptance Criteria
- [ ] Dashboard shows 2-3 Quick Action buttons at top
- [ ] KPI bar unchanged (4 metrics)
- [ ] MarketPulsePanel shows live sentiment gauge consuming `useLiveSentiment()` hook
- [ ] MarketPulsePanel shows last 5 high-impact news with LLM scores
- [ ] MarketPulsePanel shows next 7 days economic calendar
- [ ] Layout is responsive: side-by-side on wide, stacked on narrow

---

## Dependencies & Order

```
S12.1 (fix news) ──────▶ S12.2a (LLM core) ──▶ S12.2b (features) ──▶ S12.2c (frontend+API)
                                                                      │
S12.2a ──▶ S12.2d (tests, can be done in parallel with S12.2b)      │
                                                                      ▼
                                                              S12.4 (dashboard, needs S12.2c API)
S12.3 (results history, fully independent, can start anytime)
```

## Validation Checklist (run after each sub-task)

```powershell
# After S12.1
python -m pytest tests/test_news.py -v --timeout=30
python -m pytest tests/test_pipeline_validation.py -v --timeout=60

# After S12.2a-d
python -m pytest tests/test_llm_sentiment.py -v --timeout=30
python -m pytest tests/test_news.py -v --timeout=30

# After S12.3 (frontend)
cd frontend; npm run build; npm run lint

# After S12.4 (frontend)
cd frontend; npm run build; npm run lint
```

## Ollama Setup (prerequisite for S12.2)

```powershell
# Install Ollama
winget install Ollama.Ollama

# Pull default model
ollama pull llama3

# Start server (runs in background)
ollama serve

# Test
ollama run llama3 "Hello, respond with JSON: {status: ok}"
```