# Onboarding + QuickStart Manual Tests

## Prerequisites

```powershell
# Terminal 1 — Backend
uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload

# Terminal 2 — Frontend
cd frontend; npm run dev
# Open: http://localhost:5173
```

---

## Test 1 — First-Visit Onboarding Flow

| Step | Action | Expected |
|------|--------|----------|
| 1.1 | Clear storage: DevTools → Application → Local Storage → delete `fx-datasource-chosen`. Reload the app. | DataSourceModal appears as overlay on the shell (sidebar visible behind it) |
| 1.2 | Verify title | "Connect Your Data" with subtitle "Select how you want to get started" |
| 1.3 | Verify three options visible | **Demo Mode**, **OANDA API**, **CSV Upload** — all with icons and descriptions |
| 1.4 | Click the backdrop (outside the modal) or close | Modal does NOT close (no backdrop click handler) |
| 1.5 | Verify sidebar + top bar visible behind the modal | Shows the main app layout faintly through the dark overlay |

---

## Test 2 — Demo Mode Wizard

| Step | Action | Expected |
|------|--------|----------|
| 2.1 | Click "Demo Mode" | Transitions to Demo Mode info screen |
| 2.2 | Verify header | "Demo Mode" title + "Get started instantly with pre-loaded market data" subtitle |
| 2.3 | Verify back button visible | Chevron-left button in top-right |
| 2.4 | Verify info panel | Cyan left-border box showing 5 bullet points with checkmarks |
| 2.5 | Verify bullet items | Pre-loaded EUR/USD data (M30+H1, 2016–2026), 124K+ candles, Full backtest pipeline, Paper trading pre-enabled, No API key required |
| 2.6 | Click "Start with Demo Data" button | Button shows "Loading Demo Data..." briefly, then modal closes |
| 2.7 | After modal closes | Dashboard loads with "Demo Mode" cyan banner at top |
| 2.8 | Verify demo banner text | "Demo Mode — Using pre-loaded sample market data. Run a real backtest or connect OANDA for live data." |
| 2.9 | Navigate to Backtest page | Page loads normally with QuickStart tab preselected |
| 2.10 | Clear fx-datasource-chosen and reload | Modal reappears |

---

## Test 3 — Back Button on Demo Info

| Step | Action | Expected |
|------|--------|----------|
| 3.1 | Click Demo → goes to demo-info | Back button visible |
| 3.2 | Click back (chevron) | Returns to choice screen with all 3 options |

---

## Test 4 — CSV Upload Wizard

| Step | Action | Expected |
|------|--------|----------|
| 4.1 | From choice screen, click "CSV Upload" | Transitions to CSV Upload screen |
| 4.2 | Verify header | "Upload CSV Data" title + "Import your own historical data files" subtitle |
| 4.3 | Verify drop zone | Dashed border box with upload icon, "Click to select CSV files", "OHLC data with timestamps (.csv)" |
| 4.4 | Click drop zone | File picker opens, filtered to .csv files |
| 4.5 | Select a valid CSV file (e.g. `EURUSD_10_years_M30_OANDA.csv` from `csv_data/`) | File appears in list with name, size in KB, and an X remove button |
| 4.6 | Select a second CSV file | Both files appear in the list |
| 4.7 | Verify button text | "Import & Start (2 files)" |
| 4.8 | Click X on one file | File removed from list; count updates |
| 4.9 | Click "Import & Start" | Button shows "Importing..." then modal closes |
| 4.10 | After import | Dashboard loads (no demo banner since mode is "csv", not "demo") |
| 4.11 | Back button works | Click back → returns to choice screen |

---

## Test 5 — OANDA Key Wizard

| Step | Action | Expected |
|------|--------|----------|
| 5.1 | From choice, click "OANDA API" | Transitions to OANDA key entry screen |
| 5.2 | Verify header | "Enter OANDA API Key" with subtitle |
| 5.3 | Verify two fields | API Key (password, required) + Account ID (text, optional) |
| 5.4 | Leave API key empty, click "Connect" | Red error: "Please enter your OANDA API key" |
| 5.5 | Enter an API key, click "Connect" | Transitions to Download screen |
| 5.6 | Verify download screen | Shows "Connected to OANDA" green banner |
| 5.7 | Verify pair buttons | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD — first 3 selected (cyan) |
| 5.8 | Verify timeframe buttons | M30, H1, H4 — M30+H1 selected |
| 5.9 | Toggle pairs on/off | Click USDJPY → deselects it (grey); click again → reselects (cyan) |
| 5.10 | Verify summary text | Shows "N download(s) queued · 5 years each" |
| 5.11 | Deselect all pairs | Button disabled (grey) |
| 5.12 | Select at least 1 pair + 1 TF | Button enabled again |
| 5.13 | Click "Download & Start" | Button shows "Downloading...", triggers background downloads |
| 5.14 | Verify modal closes after downloads start | Downloads run in background (Celery), modal dismisses immediately |

---

## Test 6 — OANDA Download Cancellation

| Step | Action | Expected |
|------|--------|----------|
| 6.1 | OANDA → key → download screen | Back button visible |
| 6.2 | Click back | Returns to key entry screen |
| 6.3 | Click back again | Returns to choice screen |

---

## Test 7 — QuickStart Tab Layout Changes

| Step | Action | Expected |
|------|--------|----------|
| 7.1 | Navigate to Backtest page | QuickStart tab pre-selected |
| 7.2 | Verify card grid | Cards are in 1-2-3 column responsive grid (1 col mobile, 2 md, 3 xl) |
| 7.3 | Verify card width | Cards are wider than before (3 columns instead of 4) |
| 7.4 | Verify gap between cards | 32px gap (`gap-8`) between cards |
| 7.5 | Verify title + time layout | Title left, estimated time as a green pill badge on the right |
| 7.6 | Verify description | Text with more line-height (leading-6), easier to read |
| 7.7 | Verify model badges | Green-tinted background instead of dark grey (`rgba(168, 224, 99, 0.08)`) |
| 7.8 | Verify spacing in card | Title row → description → model badges have distinct vertical gaps (`mb-3`, `mt-6`) |
| 7.9 | Verify category headers | Left accent bar (2px colored vertical line) before each category name |
| 7.10 | Verify category spacing | More vertical space above headers (`mt-10` vs previous `mt-8`) |

---

## Test 8 — QuickStart Preset Selection

| Step | Action | Expected |
|------|--------|----------|
| 8.1 | Click any preset card (e.g. "Logistic Probe") | Card's hover state triggers (green border + lighter bg) |
| 8.2 | Verify config summary bar updates | Top bar shows the selected asset, model, HPO details |
| 8.3 | Click "Quick Start" tab again | Tab stays selected |

---

## Test 9 — Custom Presets Section

| Step | Action | Expected |
|------|--------|----------|
| 9.1 | No custom presets saved | "My Presets" section not visible |
| 9.2 | Save a preset from a completed backtest setup | Custom preset appears below all category sections |
| 9.3 | Verify custom preset card | Shows name, optional subtitle, save date, delete button |
| 9.4 | Click delete (trash icon) | Preset removed; section hides if no presets left |

---

## Test 10 — No Regression: S21 Trading Page

| Step | Action | Expected |
|------|--------|----------|
| 10.1 | Navigate to `/trading` | Trading page loads (Paper/Live toggle, position monitor, etc.) |
| 10.2 | Verify routes unaffected | `/backtest`, `/results`, `/trading`, `/settings` all work |

---

## Test 11 — No Regression: Dashboard

| Step | Action | Expected |
|------|--------|----------|
| 11.1 | Navigate to Dashboard | Dashboard loads |
| 11.2 | If demo mode is active | Cyan "Demo Mode" banner visible at top |
| 11.3 | If no demo mode | No banner, Dashboard works as before |

---

## Quick Smoke Test

```powershell
# Python backend endpoint check
python -c "import json; from fastapi.testclient import TestClient; from api.main import app; client = TestClient(app); r = client.post('/api/v1/data/seed-demo', json={'pairs':['EURUSD'],'timeframes':['M30']}); d=r.json(); print('Seed:', d['status'], d['total_candles'], 'candles')"

# TypeScript compilation
cd frontend; npx tsc --noEmit; if ($?) { Write-Host "TS: OK" } else { Write-Host "TS: FAIL" }
```
