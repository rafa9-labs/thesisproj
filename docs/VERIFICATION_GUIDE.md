# Verification Guide — Date Validation + Enriched Metrics

This guide walks you through manually verifying the changes from the backtest date validation and metrics enrichment work.

---

## Prerequisites

- Redis running on `localhost:6379`
- Python env activated with all deps
- Node/NPM available for frontend

---

## Step 1: Start the Backend Stack

Open **3 terminal windows** (PowerShell):

**Terminal 1 — Redis**
```powershell
redis-server
```

**Terminal 2 — Celery Worker**
```powershell
cd C:\Users\rafa\ML_Trading\thesisproj
celery -A api.tasks worker --loglevel=info --pool=solo -Q celery
```

**Terminal 3 — FastAPI**
```powershell
cd C:\Users\rafa\ML_Trading\thesisproj
uvicorn api.main:app --reload --port 8000
```

Verify: Open http://localhost:8000/health — should return `{"status": "ok", ...}`

---

## Step 2: Start the Frontend

**Terminal 4 — Vite Dev Server**
```powershell
cd C:\Users\rafa\ML_Trading\thesisproj\frontend
npm run dev
```

Verify: Open http://localhost:5173 — app should load

---

## Step 3: Test Auto-Fill Dates (AssetSelector)

1. Navigate to **New Backtest** page
2. Select **EURUSD** / **H1**
3. **Verify**: Start Date and End Date fields auto-populate
   - Start should show data minimum (e.g., `2015-12-01`)
   - End should show a date that leaves room for training (roughly data_max - 36 months)
4. Next to each date input, you should see `(min: ...)` and `(max: ...)` hints
5. Change timeframe to **H4** — dates should update for that timeframe's range
6. Change pair to **GBPUSD** — dates should update again

**What to look for:**
- [ ] Dates auto-fill when pair/timeframe changes
- [ ] Min/max range hints visible next to labels
- [ ] No dates initially (empty) → auto-populated after selecting pair

---

## Step 4: Test Date Validation

1. In the End Date field, enter a date **beyond data range** (e.g., `2028-12-31`)
2. **Verify**: The date input border turns red/danger color
3. Below the date row, a red warning appears: "Date range outside available data"
4. Now set End Date to a valid date within range — warning should disappear

**What to look for:**
- [ ] Red border on invalid date input
- [ ] "Date range outside available data" warning text
- [ ] Warning clears when date is corrected

---

## Step 5: Test Backtest Submit (API-level)

Use curl or the browser's Network tab to inspect the POST response:

```powershell
# No dates — should auto-fill
curl -X POST http://localhost:8000/backtest -H "Content-Type: application/json" -d "{\"pair\":\"EURUSD\",\"models\":[\"logistic\"],\"trading_costs\":true,\"months\":1,\"repeats\":1,\"config_overrides\":{\"train_months\":36}}"
```

**Verify the response contains:**
- `warnings` array with messages about auto-filled dates
- `adjusted_start_date` and/or `adjusted_end_date` if dates were auto-set
- `job_id` and `status: "pending"`

```powershell
# Dates beyond range — should clamp
curl -X POST http://localhost:8000/backtest -H "Content-Type: application/json" -d "{\"pair\":\"EURUSD\",\"models\":[\"logistic\"],\"end_date\":\"2028-01-01\",\"trading_costs\":true,\"months\":1,\"repeats\":1,\"config_overrides\":{\"train_months\":36}}"
```

**Verify:**
- [ ] `warnings` contains clamping message
- [ ] `adjusted_end_date` is set to data max (around `2026-04-24`)
- [ ] Job still gets created (202 response)

---

## Step 6: Test Backtest via UI (Full Flow)

1. In the UI, select **EURUSD** / **H1** / **logistic** model
2. Leave dates auto-filled (don't override)
3. Click **Deploy Backtest**
4. **Verify BacktestProgress** shows:
   - Progress bar animating
   - Model status pills (logistic turning green when done)
   - No warnings if trades are produced
5. When complete, navigate to the **Results** page

**If 0 trades** (rare with valid dates, but possible):
- [ ] Warning banner appears in BacktestProgress (orange)
- [ ] Warning banner appears in MetricsGrid (orange)

---

## Step 7: Test Results Page — Enriched Metrics

After a backtest completes, go to the Results page.

**Verify the MetricsGrid shows 12 cards:**

| # | Card | Source Field |
|---|------|-------------|
| 1 | Sharpe | `sharpe` |
| 2 | Sortino | `sortino` |
| 3 | Max Drawdown | `max_drawdown` |
| 4 | CAGR | `cagr` |
| 5 | Total Return | `total_return` |
| 6 | Calmar | `calmar_ratio` |
| 7 | Win Rate | `win_rate` |
| 8 | Profit Factor | `profit_factor` |
| 9 | Total Trades | `total_trades` |
| 10 | Avg Trade | `avg_trade` |
| 11 | Active Rate | `active_rate` |
| 12 | Dir. Accuracy | `directional_accuracy` |

**What to look for:**
- [ ] All 12 cards render (4 columns × 3 rows)
- [ ] Cards that have values show them; cards with `null` show "—"
- [ ] Delta badges appear for Sharpe (Excellent/Good/Weak), Sortino, CAGR, Win Rate, etc.
- [ ] "No trades" delta badge on Total Trades if 0

---

## Step 8: Test Warnings Banner on Results Page

If a model produced 0 trades:

1. The top of the Results page should show an **orange warnings banner** with:
   - Model name and "0 trades" message
   - Available data range line: `Available data: 2015-12-01 → 2026-04-24`
2. The MetricsGrid should also show a smaller orange warning below the model name

**To force a 0-trade scenario**, you can:
```powershell
# Submit with start_date beyond data range (will clamp, but if test window is tiny...)
curl -X POST http://localhost:8000/backtest -H "Content-Type: application/json" -d "{\"pair\":\"EURUSD\",\"models\":[\"logistic\"],\"start_date\":\"2026-04-20\",\"end_date\":\"2026-04-24\",\"months\":1,\"trading_costs\":true,\"repeats\":1,\"config_overrides\":{\"train_months\":36}}"
```

**What to look for:**
- [ ] Orange warnings banner at top of Results page
- [ ] Data range info in the banner
- [ ] Per-model 0-trade warning in MetricsGrid

---

## Step 9: Test Results API Endpoint

```powershell
# Replace JOB_ID with a real completed job ID
curl http://localhost:8000/backtest/JOB_ID/results
```

**Verify:**
- [ ] `warnings` array present (empty if no issues)
- [ ] `data_start` and `data_end` present (e.g., `"2015-12-01"`, `"2026-04-24"`)
- [ ] Each metric object includes new fields: `active_rate`, `directional_accuracy`, `cagr`, `calmar_ratio`, `equity_final`

---

## Step 10: Multi-Model Backtest

1. Select 2-3 models (e.g., logistic, xgboost, random_forest)
2. Run a backtest
3. On Results page, verify the **model switcher buttons** still work
4. Click each model button — MetricsGrid should update with that model's metrics
5. Each button shows `model_name — Sharpe X.XX`

**What to look for:**
- [ ] Model switcher buttons render for each model
- [ ] Clicking a button updates the MetricsGrid
- [ ] Warnings are per-model (only on the 0-trade model)

---

## Cleanup

After testing, you can stop all services:
- Ctrl+C in each terminal
- No data is persisted beyond SQLite job records (safe to leave)

---

## Checklist Summary

| # | Test | Pass? |
|---|------|-------|
| 1 | Dates auto-fill in AssetSelector | ☐ |
| 2 | Min/max date hints visible | ☐ |
| 3 | Red border on out-of-range date | ☐ |
| 4 | Auto-fill warnings in submit API response | ☐ |
| 5 | Date clamping warnings in submit API response | ☐ |
| 6 | BacktestProgress shows live warnings | ☐ |
| 7 | 12 metric cards render in MetricsGrid | ☐ |
| 8 | New fields (CAGR, Calmar, Active Rate, Dir. Accuracy) display | ☐ |
| 9 | 0-trade warning in Results page banner | ☐ |
| 10 | 0-trade warning in MetricsGrid | ☐ |
| 11 | Results API returns data_start/data_end | ☐ |
| 12 | Model switcher works with enriched metrics | ☐ |