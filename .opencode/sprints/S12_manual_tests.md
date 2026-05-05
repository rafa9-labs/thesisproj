# S12 Manual Test Plan

Run these after starting the backend + frontend. Stop if any step fails.

---

## Prerequisites

```powershell
# Terminal 1: Redis
redis-server

# Terminal 2: FastAPI
uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload

# Terminal 3: Celery worker
celery -A api.tasks.celery_app worker --loglevel=info --pool=solo -Q celery

# Terminal 4: React frontend
cd frontend; npm run dev
```

---

## Test 1 — News pipeline actually works (S12.1)

**Step 1.1** — Open browser at `http://localhost:5173`

**Step 1.2** — Go to **Backtest** page. In "News & Sentiment" section, verify:
- [x] "News Features" toggle is **ON** by default (was `false` before)
- [x] Sentiment Engine shows "VADER (Rule-based)"

**Step 1.3** — Configure a minimal backtest:
- Models: check only **Logistic Regression**
- Months: **1**
- HPO: **Light**

**Step 1.4** — Click "Run Backtest". Watch the terminal running FastAPI:
- [x] You should see log lines from `news/scraper.py` fetching RSS feeds
- [x] Should see `news_loaded` event (not `news_skip`)
- [x] Articles count > 0

**Step 1.5** — When completed, go to the Results page (`/results/:jobId`). In the Config Viewer (collapsible JSON at bottom), verify:
- [x] `use_news: true` appears in the config
- [x] The feature list includes `sentiment_score`, `sentiment_magnitude`

**Step 1.6** — If you temporarily disable the "News Features" toggle and re-run:
- [x] Backend terminal should show `news_skip` event
- [x] `sentiment_score` should NOT be in the feature list

---

## Test 2 — LLM sentiment with Ollama (S12.2)

**Prerequisite**: Install and start Ollama first:
```powershell
# Install if not already
winget install Ollama.Ollama

# Pull a model (one-time)
ollama pull llama3.2

# Start the server
ollama serve
```

**Step 2.1** — On the Backtest page, scroll to "News & Sentiment" section:
- [x] "LLM Sentiment" toggle should be visible under "Sentiment Engine"
- [x] Toggle it ON

**Step 2.2** — Expand the LLM config:
- [x] Backend dropdown: Ollama (Local, Free) — selected by default
- [x] Model input shows "llama3"
- [x] Weight slider visible, defaults to 0.7

**Step 2.3** — Change backend to "OpenAI API":
- [x] An "API Key" password field appears
- [x] It hides when switching back to Ollama

**Step 2.4** — Switch back to Ollama. Run a backtest with:
- Models: Logistic Regression only, 1 month, Light HPO
- LLM Sentiment: ON

**Step 2.5** — Watch the FastAPI terminal:
- [x] Should see `llm_loaded` event with article count
- [x] If Ollama is not running, should see `llm_skip` instead (graceful fallback)

**Step 2.6** — After completion, check the feature list in results:
- [x] `llm_sentiment` column present
- [x] `blended_sentiment` column present
- [x] `llm_sentiment_ma_6`, `llm_sentiment_ma_24` columns present

**Step 2.7** — Test live sentiment API directly:
```powershell
curl http://localhost:8001/api/v1/news/sentiment/live?pair=EURUSD
```
- [x] Returns JSON with `pairs.EURUSD.vader_sentiment`
- [x] If Ollama is running, also returns `pairs.EURUSD.llm_sentiment`
- [x] `blended_sentiment` field present

---

## Test 3 — Results history browser (S12.3)

**Step 3.1** — Click "Results" in the sidebar (or `/results`):
- [x] Shows a full-page table, NOT the old "No results to display" empty state
- [x] Table shows all completed backtests with columns:
  - Date, Pair, Models, Sharpe, Return %, Win Rate, Max DD, Actions

**Step 3.2** — If you have at least 2 completed backtests:
- [x] Sorting: click "Sharpe" header → values sort ascending/descending
- [x] Sorting: click "Return" header → same behavior

**Step 3.3** — Filtering:
- [x] Type a job ID in the search box → filters in real-time
- [x] Select a pair from the "All Pairs" dropdown → filters

**Step 3.4** — Select checkboxes:
- [x] Check a few rows → "CSV (N)" and "JSON (N)" export buttons appear
- [x] Click "CSV" → downloads `backtest_results.csv`
- [x] Click "JSON" → downloads `backtest_results.json`

**Step 3.5** — Click a row:
- [x] Navigates to `/results/:jobId` detail view (same as before)

**Step 3.6** — Delete a job:
- [x] Click the trash icon → job is removed from the table

---

## Test 4 — Dashboard redesign (S12.4)

**Step 4.1** — Go to Dashboard (`/`):
- [x] Top section: "Quick Actions" bar with two buttons
  - "New Backtest" (branded color)
  - "Re-run Last" (outlined)

**Step 4.2** — Click "New Backtest":
- [x] Navigates to `/backtest` page

**Step 4.3** — Below Quick Actions:
- [x] 4 KPI cards (same as before): Total Runs, Best Sharpe, Avg Win Rate, Best Return

**Step 4.4** — Middle section (side-by-side on wide screens):
- [x] Left: "Recent Activity" table (same as before)
- [x] Right: "Market Pulse" panel with:
  - **Sentiment gauge**: circular chart with numeric score
  - Bullish/Bearish/Neutral badge
  - Sentiment Details: LLM score, VADER score, Confidence
  - Recent News section
  - Cache status (articles count)

**Step 4.5** — Performance heatmap:
- [x] Still visible below (same as before)

**Step 4.6** — Verify MarketPulse shows live data:
- [x] If Ollama is running, gauge shows "LLM (llama3)" label
- [x] If only VADER, gauge shows "VADER" label
- [x] Cached articles count matches `GET /news/status`

---

## Test 5 — End-to-end with LLM features in model training

**Step 5.1** — Run a backtest with ALL features enabled:
- Models: Logistic Regression
- Months: 3
- HPO: Quick
- News: ON, LLM: ON (Ollama)

**Step 5.2** — After completion, check the Results detail page:
- [x] Equity curve renders
- [x] Metric cards show valid values
- [x] Trade log has entries

**Step 5.3** — In the Config Viewer, verify the feature list contains:
- [x] `sentiment_score`, `sentiment_magnitude` (VADER)
- [x] `news_volume_6bars`, `news_volume_24bars`
- [x] `llm_sentiment`, `llm_confidence`, `llm_volatility`
- [x] `blended_sentiment`
- [x] `llm_sentiment_ma_6`, `llm_sentiment_ma_24`

**Step 5.4** — Verify event flags:
- [x] Run: `curl "http://localhost:8001/api/v1/news/events?start=$(Get-Date -Year 2024 -Month 1 -Day 1 -Format 'yyyy-MM-dd')&end=now&impact=high,medium"` 
- [x] Returns NFP, FOMC, CPI, etc. events

---

## Test 6 — Edge cases

**Step 6.1** — Run a backtest with LLM ON but Ollama NOT running:
- [x] Backend logs `llm_skip` (not a crash)
- [x] Backtest completes successfully
- [x] LLM features are absent but VADER features are present
- [x] `blended_sentiment` equals `sentiment_score` (100% VADER)

**Step 6.2** — Run with News Features OFF:
- [x] No news features in output
- [x] No LLM features in output
- [x] Backtest completes normally

**Step 6.3** — Delete all jobs, then visit `/results`:
- [x] Shows "No results found. Run a backtest to see results here."

---

## Quick API Verification Script

Paste this into PowerShell to run all API checks at once:

```powershell
$base = "http://localhost:8001/api/v1"

Write-Host "=== News Status ==="
Invoke-RestMethod "$base/news/status" | ConvertTo-Json -Depth 2

Write-Host "`n=== Live Sentiment ==="
Invoke-RestMethod "$base/news/sentiment/live?pair=EURUSD" | ConvertTo-Json -Depth 3

Write-Host "`n=== Results Summary ==="
Invoke-RestMethod "$base/backtest/results/summary?limit=5" | ConvertTo-Json -Depth 2

Write-Host "`n=== Backtest List (paginated) ==="
Invoke-RestMethod "$base/backtest?limit=3&offset=0" | ConvertTo-Json -Depth 2

Write-Host "`n=== News Events ==="
Invoke-RestMethod "$base/news/events?start=1719792000&end=1735689600&impact=high" | ConvertTo-Json -Depth 2
```
