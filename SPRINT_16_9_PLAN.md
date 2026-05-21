# Sprint 16.9 — Forward Test + Live Trading Bridge

> **Goal**: Make saved models actionable. Add a Forward Test tab for temporal backtesting (test any saved model on any date range without retraining) and bridge deployed models into live trading (deploy saved models instead of training fresh).
> **Branch**: `feature/sprint16.8-model-persistence`
> **Est**: 5-7h
> **Design decisions**: Forward Test is 7th tab on BacktestPage (not standalone page). Live Trading uses saved models only (no train-fresh option).

---

## Architecture Overview

```
BacktestPage (7th tab: "Forward Test")
    │
    ├── Select saved model from deployed models
    ├── Pick date range / pair / timeframe
    ├── Choose position sizing
    └── "Run Forward Test" → POST /models/{id}/forward-test
                                  │
                                  ▼
                         pipeline/forward_test.py
                            ┌─ load_snapshot() → model + scaler + config
                            ├─ MLBacktester(config) → load data
                            ├─ Inject model: bt.model = loaded_model  
                            ├─ real_trading_simulation(skip_hpo=True, skip_training=True)
                            └─ Return: equity, metrics, trades (same schema as backtest)
                                  │
                                  ▼
                         Monitor → ResultsPage (identical rendering)

LiveTradingPage
    │
    ├── Select deployed model from /models/deployed?status=active
    ├── Show model stats (Sharpe, tags, train range)
    └── "Deploy" → POST /live/deploy { model_id: "logistic_v1_..." }
                       │
                       ▼
                  load_snapshot(model_id) → model_obj
                  Signal loop uses loaded model (no training)
```

---

## Phase 1: Forward Test Engine (Backend)

### File: `pipeline/forward_test.py` (NEW)

```python
def run_forward_test(
    snapshot_path: str,
    pair: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    position_sizing: str = "fixed",
    sizing_config: dict | None = None,
    trading_costs: bool = True,
) -> dict:
    """
    1. Loads snapshot → model + scaler + imputer + metadata
    2. Creates MLBacktester with saved features_config  
    3. Loads market data for [start_date, end_date]
    4. Injects model into backtester
    5. Runs monthly walk-forward with skip_hpo=True, skip_training=True
    6. Returns full results dict (metrics, equity, trades, diagnostics)
    """

def generate_forecast_errors(
    forward_test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-bar predictions vs actual for future DM test (Phase 3)."""
```

Key: The model is loaded ONCE and used for ALL walk-forward months — zero retraining. Feature computation follows saved metadata's `features_config` exactly. Uses existing `real_trading_simulation()` engine — only injects a flag to skip HPO and model training blocks.

### Endpoint: `POST /models/{id}/forward-test` (in `api/routers/models.py`)

```python
class ForwardTestRequest(BaseModel):
    model_id: str
    pair: str = "EURUSD"
    timeframe: str = "H1"
    start_date: str
    end_date: str
    position_sizing: str = "fixed"
    trading_costs: bool = True
    sizing_config: dict = {}

@router.post("/{model_id}/forward-test", status_code=202)
def forward_test_model(model_id: str, req: ForwardTestRequest):
    # 1. Validate model exists in deployed_models
    # 2. Create job via JobManager
    # 3. Launch Celery task _run_forward_test_impl
    # 4. Return { job_id, status: "pending" }
```

### Task: `_run_forward_test_impl()` (in `api/tasks.py`)

Same pattern as `_run_backtest_impl` but simpler:
- Load snapshot
- Call `run_forward_test()`
- Publish WS progress events
- Update job status (running → completed/failed)
- Return result via JobManager

---

## Phase 1: Forward Test Tab (Frontend)

### File: `frontend/src/pages/Backtest/ForwardTestTab.tsx` (NEW)

Components:
1. **SavedModelSelector** — dropdown from `GET /models/deployed`
   - Shows model ID, type, Sharpe, status, tags
   - Grouped by active/inactive
2. **AssetSelector** — reuses existing pair + timeframe selectors (already in BacktestPage store)
3. **DateRangePicker** — start/end date inputs
4. **SizingSelector** — dropdown: Fixed, Fractional, Kelly, ATR, Vol Target
5. **CostsToggle** — checkbox for spread + slippage simulation
6. **"Run Forward Test" button** — validates, submits, navigates to `/monitor`

### Changes to `BacktestPage.tsx`

```tsx
const TABS = [
  { key: "quickstart", label: "Quick Start" },
  { key: "asset", label: "Asset & Model" },
  { key: "study", label: "Study & HPO" },
  { key: "features", label: "Features" },
  { key: "hyperparams", label: "Hyperparameters" },
  { key: "execution", label: "Execution" },
  { key: "forwardtest", label: "Forward Test" },  // NEW
];

{activeTab === "forwardtest" && <ForwardTestTab />}
```

### Changes to `frontend/src/api/queries.ts`

```typescript
// New mutation
export function useSubmitForwardTest() {
  return useMutation({
    mutationFn: (payload: ForwardTestRequest) =>
      apiClient.post(`/models/${payload.model_id}/forward-test`, payload),
  });
}

// Reuses existing  
export function useDeployedModels() { ... }  // GET /models/deployed
```

### Changes to `frontend/src/api/schemas.ts`

```typescript
export interface ForwardTestRequest {
  model_id: string;
  pair: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  position_sizing: string;
  trading_costs: boolean;
  sizing_config?: Record<string, number>;
}

export interface ForwardTestResponse {
  job_id: string;
  status: string;
}
```

---

## Phase 2: Live Trading with Saved Models (Backend)

### Changes to `api/routers/live.py`

```python
class DeployRequest(BaseModel):
    pair: str
    model: str = "logistic"        # kept for backward compat
    timeframe: str = "M30"
    initial_equity: float = 10000.0
    model_id: str | None = None    # NEW — if set, load from snapshot

@router.post("/live/deploy", response_model=SessionInfo)
async def deploy_live_session(req: DeployRequest):
    if req.model_id:
        # PATH A: load saved model
        snapshot = load_snapshot(req.model_id)
        model_obj = snapshot["model"]
        bt = None
        bt_config = snapshot.get("metadata", {})
    else:
        # PATH B (existing): train fresh
        bt_result = _run_backtest_for_model(pair, req.model, req.timeframe)
        model_obj = bt_result.get("model") if bt_result else None
        bt = bt_result.get("backtester") if bt_result else None
    
    # Create session + start signal loop (unchanged)
    session = {
        "model_obj": model_obj,
        "backtester": bt,
        # ...
    }
    asyncio.create_task(_signal_loop(session_id))
```

Zero changes to `_signal_loop()` or `_predict_signal()` — they already work with any `model_obj`.

---

## Phase 2: Live Trading Saved Model Selector (Frontend)

### Changes to `frontend/src/pages/LiveTrading/LiveTradingPage.tsx`

- Replace the model type dropdown with a saved model selector
- Query `GET /models/deployed?status=active` on mount
- Show model cards with type, Sharpe, tags, train range
- If no active models, show "No active models — activate one on the Models page first" with link
- On Deploy: pass `model_id` in the deploy request

```
Live Trading
┌──────────────────────────────────────────────────────────────┐
│ Pair:  [EURUSD ▼]    TF: [M15][M30][H1][H2][H4]             │
│                                                              │
│ Deployed Model (signal source)                               │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ ● logistic_v1    SR +0.80    tags: verified, production  │ │
│ │   trained: 2023-01 → 2025-12                             │ │
│ │                                                          │ │
│ │ ○ xgboost_v2     SR +0.65    tags: verified              │ │
│ │   trained: 2023-06 → 2025-12                             │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ [Deploy]    Status: Offline                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Test Files

### `tests/test_forward_test.py` (NEW — 5 tests)

1. `test_forward_test_engine` — load logistic snapshot, run on test range → valid metrics
2. `test_forward_test_no_trades` — flat model → zero-trade clean result
3. `test_forward_test_endpoint` — POST returns valid job_id + 202
4. `test_forward_test_uses_saved_config` — features match metadata config
5. `test_forecast_errors_generation` — error arrays have correct shape/length

### `tests/test_live_deploy_saved.py` (NEW — 3 tests)

6. `test_live_deploy_with_model_id` — deploy saved model → signal loop starts
7. `test_live_deploy_no_model_id` — backward compat: train-fresh works
8. `test_live_deploy_missing_model_id` — 404 for nonexistent model

---

## File Inventory

| Action | File | Purpose |
|--------|------|---------|
| **CREATE** | `pipeline/forward_test.py` | Forward test engine |
| **MODIFY** | `api/routers/models.py` | Add `POST /{model_id}/forward-test` endpoint |
| **MODIFY** | `api/tasks.py` | Add `_run_forward_test_impl` Celery task |
| **CREATE** | `frontend/src/pages/Backtest/ForwardTestTab.tsx` | Forward Test tab UI |
| **MODIFY** | `frontend/src/pages/Backtest/BacktestPage.tsx` | Add 7th tab |
| **MODIFY** | `api/routers/live.py` | Add `model_id` to `DeployRequest` |
| **MODIFY** | `frontend/src/pages/LiveTrading/LiveTradingPage.tsx` | Saved model selector |
| **MODIFY** | `frontend/src/api/queries.ts` | `useSubmitForwardTest` mutation |
| **MODIFY** | `frontend/src/api/schemas.ts` | Forward test types |
| **CREATE** | `tests/test_forward_test.py` | Forward test unit tests |
| **CREATE** | `tests/test_live_deploy_saved.py` | Live deploy with saved model tests |

---

## Post-Integration Flow

```
BacktestPage ──→ ResultsPage ──→ ModelsPage ──→ LiveTradingPage
    │                │                 │                │
    │ auto-save      │ "Model saved"   │ Activate       │ Deploy saved
    │ snapshot       │ badge           │ Forward Test   │ model → WS
    │                │                 │ Delete         │ signal stream
    │                │ LLM Advisor     │                │
    │                │ "Apply Study"   │                │
    │                │                 │                │
    │  Forward Test tab               │                │
    │  (pick model → any dates)       │                │
    │                │                 │                │
    └── all flow to Monitor → Results (same pipeline) ──┘
```
