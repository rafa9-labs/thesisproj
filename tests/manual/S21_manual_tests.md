# S21 Manual Test Suite

Run these after starting:
```powershell
# Terminal 1 — Redis
redis-server

# Terminal 2 — FastAPI
uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload

# Terminal 3 — Celery worker
celery -A api.tasks.celery_app worker --loglevel=info --pool=solo -Q celery

# Terminal 4 — React frontend
cd frontend; npm run dev
# Open: http://localhost:5173
```

---

## Test 1 — Dashboard → Trading Navigation

| Step | Action | Expected |
|------|--------|----------|
| 1.1 | Open `http://localhost:5173` | Dashboard loads |
| 1.2 | Click "Trading" in sidebar (lightning icon) | Navigates to `/trading`, top bar shows "Trading" |
| 1.3 | Verify page title | "Trading" at top-left of header |

---

## Test 2 — API Health Check (Terminal)

| Step | Action | Expected |
|------|--------|----------|
| 2.1 | `curl http://localhost:8001/api/v1/pairs` | Returns JSON list of available pairs |
| 2.2 | `curl http://localhost:8001/api/v1/models/deployed?status=active` | Returns deployed models list |
| 2.3 | `curl http://localhost:8001/api/v1/trading/paper/sessions` | Returns `[]` (empty list) |
| 2.4 | `curl http://localhost:8001/api/v1/trading/live/sessions` | Returns `[]` (empty list) |

---

## Test 3 — Paper Trading Flow (UI)

**Prerequisites**: At least one model must be deployed (run a backtest on Results page, click "Save Model", then activate it on Models page).

| Step | Action | Expected |
|------|--------|----------|
| 3.1 | Navigate to `/trading` | Page loads with "Paper" mode selected (cyan), "Live" mode grey |
| 3.2 | Select pair EURUSD | Dropdown shows EURUSD |
| 3.3 | Select model from dropdown | Shows type + Sharpe ratio |
| 3.4 | Select timeframe H1 | Button goes cyan |
| 3.5 | Set sizing to "Fixed (1 lot)" | Default |
| 3.6 | Set equity to 10000 | Default |
| 3.7 | Click "Deploy" (green button) | Button changes to "Deploying..." then session starts |
| 3.8 | Verify status indicator | "Live" label appears in green |
| 3.9 | Verify price bar appears | Shows mid price + change % + bid/ask |
| 3.10 | Verify position panel | Shows "FLAT" badge, equity, unreal P&L, return %, trades |
| 3.11 | Wait 30-60 seconds | Signals appear in Trade Journal panel |
| 3.12 | Verify trade journal entries | Shows direction (LONG/SHORT), P&L, entry/exit prices |
| 3.13 | Click "Stop" (red button) | Session stops, panel shows "Offline" |
| 3.14 | If paper mode, verify result modal | StopResultModal shows Sharpe, Sortino, Return, Max DD, Win Rate, Trades, Profit Factor, Avg PnL, Final Equity |
| 3.15 | Close modal | Click "Close" or X button |

---

## Test 4 — TradeHistory Filtering

| Step | Action | Expected |
|------|--------|----------|
| 4.1 | With trades visible in journal, click "Wins" filter chip | Only profitable trades shown, count updates |
| 4.2 | Click "Losses" | Only losing trades shown |
| 4.3 | Click "Blocked" | Only risk-blocked trades shown |
| 4.4 | Click "All" | All trades visible again |
| 4.5 | Type in search box (partial trade ID) | List filters to matching trades |
| 4.6 | Clear search | All trades visible |

---

## Test 5 — CSV Export

| Step | Action | Expected |
|------|--------|----------|
| 5.1 | With trades visible, click "CSV" button | Browser downloads a file named `trades_YYYY-MM-DD.csv` |
| 5.2 | Open CSV file | Contains columns: trade_id, direction, size, entry_price, exit_price, pnl, exit_reason, risk_blocked, oanda_order_id |
| 5.3 | Verify no trades → CSV disabled | Button is greyed/disabled when `trades.length === 0` |

---

## Test 6 — Mode Toggle (Paper ↔ Live)

| Step | Action | Expected |
|------|--------|----------|
| 6.1 | Click "Live" on mode toggle | Mode switches to Live (red), "Paper" goes grey |
| 6.2 | Verify [OANDA] badge appears | Red [OANDA] badge next to price |
| 6.3 | Verify "Show Risk Config" button appears | Below controls bar |
| 6.4 | Click "Show Risk Config" | 6 risk sliders appear: Max DD %, Daily Loss %, Consec Losses, Min Confidence, Max Pos %, Daily Trades |
| 6.5 | Adjust sliders | Values change in real-time |
| 6.6 | Click "Hide Risk Config" | Sliders collapse |
| 6.7 | Verify Emergency Kill button hidden | Only shows when Live session is running |
| 6.8 | Switch back to "Paper" | Cyan, no [OANDA] badge, no risk config |

---

## Test 7 — OANDA Key Required State

| Step | Action | Expected |
|------|--------|----------|
| 7.1 | Remove OANDA key from Settings | Or if none configured, skip to 7.3 |
| 7.2 | Navigate to `/trading` | Shows "Add your OANDA API key in Settings" alert with link to Settings |
| 7.3 | Add OANDA key in Settings | Alert disappears, page loads normally |

---

## Test 8 — Paper Session Management (API)

```powershell
# Start paper session
$body = @{
    pair = "EURUSD"
    model_type = "logistic"
    timeframe = "M30"
    initial_equity = 10000
    position_sizing = "fixed"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8001/api/v1/trading/paper/start" -Method Post -Body $body -ContentType "application/json"
Write-Host "Session ID: $($response.session_id)"

# Check status
Invoke-RestMethod -Uri "http://localhost:8001/api/v1/trading/paper/$($response.session_id)/status"

# Wait 10 seconds, then get trades
Start-Sleep -Seconds 10
Invoke-RestMethod -Uri "http://localhost:8001/api/v1/trading/paper/$($response.session_id)/trades"

# Stop session
Invoke-RestMethod -Uri "http://localhost:8001/api/v1/trading/paper/$($response.session_id)/stop" -Method Post

# Get summary
Invoke-RestMethod -Uri "http://localhost:8001/api/v1/trading/paper/$($response.session_id)/summary"
```

| Step | Expected |
|------|----------|
| 8.1 | Start returns session_id, pair, model_type, status="running" |
| 8.2 | Status returns equity, position, trades count |
| 8.3 | After 10s, trades shows trade records |
| 8.4 | Stop returns status="stopped", summary metrics |
| 8.5 | Summary returns sharpe, total_return_pct, max_drawdown_pct, win_rate, profit_factor, final_equity |

---

## Test 9 — Risk Blocking (if Live mode possible)

| Step | Action | Expected |
|------|--------|----------|
| 9.1 | Deploy Live session with `min_confidence: 99` | Almost no signals get through |
| 9.2 | Watch Trade Journal | "BLOCKED" entries appear in orange |
| 9.3 | Verify blocked reason tooltip | Shows `G4:confidence_XX_below_min_99` |
| 9.4 | Stop session | Normal stop |

---

## Test 10 — Kill State Reset

| Step | Action | Expected |
|------|--------|----------|
| 10.1 | Deploy Paper, then Stop | Session stops cleanly |
| 10.2 | Verify Deploy button re-enabled | Can deploy again immediately |
| 10.3 | Deploy again | New session starts with fresh state |
| 10.4 | Verify old trades not visible | Trade journal starts empty for new session |

---

## Test 11 — Concurrent Sessions (Stress)

| Step | Action | Expected |
|------|--------|----------|
| 11.1 | Deploy Paper session | Running normally |
| 11.2 | Start a backtest on Backtest page | Backtest runs in background (Celery) |
| 11.3 | Verify Trading page still responsive | Prices update, signals flow, chart renders |
| 11.4 | Verify no 500 errors or freezes | Both paper session and backtest running concurrently |

---

## Test 12 — WebSocket Events

Open browser DevTools (F12) → Network → WS tab → filter for "ws" connections.

| Step | Action | Expected |
|------|--------|----------|
| 12.1 | Deploy Paper session | WebSocket connects to `/api/v1/trading/paper/{id}/ws` |
| 12.2 | Watch WS frames | See `signal`, `trade_opened`, `trade_closed`, `hold` events |
| 12.3 | Verify heartbeat events every 30s | `{"event":"heartbeat","time":...}` |
| 12.4 | Stop session | Final `stopped` event received, WS closes cleanly |

---

## Test 13 — Python Unit Tests

```powershell
# Run all S21 tests (61 tests)
python -m pytest tests/test_paper_engine.py tests/test_risk_controls.py -v --tb=short

# Run with verbose output
python -m pytest tests/test_paper_engine.py tests/test_risk_controls.py -v -s
```

| Step | Expected |
|------|----------|
| 13.1 | All 61 tests pass (47 risk + 14 paper engine) |
| 13.2 | No skipped, no failed, no errors |

---

## Test 14 — Route Cleanup Verification

| Step | Action | Expected |
|------|--------|----------|
| 14.1 | Navigate to `/live-trading` (old path) | Automatically redirects to `/trading` (or shows page at `/trading`) |
| 14.2 | Check sidebar label | Shows "Trading" not "Live" |
| 14.3 | Check top bar title | Shows "Trading" when on `/trading` |
| 14.4 | Verify old `pages/LiveTrading/` directory deleted | File not found |
| 14.5 | Verify old `pages/Live/` directory cleaned | Orphaned LiveMonitorPage removed |

---

## Quick Smoke Test (One Command)

```powershell
# Single quick check
python -c "
from trading.paper_engine import PaperEngine
from trading.risk_controls import LiveRiskConfig, new_session_state, check_all_pre_trade_gates
engine = PaperEngine()
engine.start({'initial_equity': 20000})
result = engine.process_signal({'direction': 'LONG', 'confidence': 85}, bid=1.0850, ask=1.08502, mid=1.08501)
print('PAPER:', result['position'], result['equity'])
cfg = LiveRiskConfig()
st = new_session_state(cfg)
st.current_equity = 8000
allowed, reason, _ = check_all_pre_trade_gates(st, cfg, {'direction': 'LONG', 'confidence': 85}, 5000, 'FLAT', 0)
print('RISK:', 'PASS' if allowed else reason)
print('OK')
"
```
