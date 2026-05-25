# Sprint 16.8 — Model Persistence, Deployment & Experiment Tracking

> **Goal**: Save trained models as deployable artifacts, track experiment lineage, and enable model sharing across KodaQuant installations.
> **Branch**: `feature/sprint16.8-model-persistence`
> **Est**: 15-19h
> **Dependencies**: Zero. All tools are already imported in the project (joblib, sklearn, SQLite).

---

## Problem Statement

Every month during walk-forward backtesting, the pipeline:
1. Builds a model from scratch using hyperparameters
2. Trains on the month's training data
3. Predicts on the month's test data
4. Discards the trained model object (garbage collected)

**Nothing is persisted** except the hyperparameter dict (`model_*_hpo.json`). For live trading, model sharing, and reproducibility, we need to save the complete trained state: model weights, scaler, imputer, calibration, and all metadata.

---

## Full Model Lifecycle

### 1. SAVE

```
Backtest completes
  → Last month's model is fully trained on all available data
  → pipeline/model_persistence.py:save_snapshot(bt, model_type)
  → Creates: deployed_models/logistic_20260521T143000/
      ├── model.joblib          (sklearn Pipeline: [StandardScaler, LogisticRegression])
      ├── scaler.joblib         (fitted StandardScaler)
      ├── imputer.joblib        (fitted SimpleImputer)
      ├── calibration.joblib    (fitted CalibratedClassifierCV or None)
      ├── metadata.json         (all config + environment + lineage)
      └── manifest.sha256       (checksums of all files)
```

**Per-model sizing:**
| Model Type | Snapshot Size | Format |
|-----------|:---:|--------|
| Logistic, SVM, DT | 5-50 KB | `joblib.dump(Pipeline(...))` |
| Random Forest | 200-500 KB | `joblib.dump(Pipeline(...))` |
| XGBoost | 200 KB - 2 MB | `booster.save_model()` inside joblib wrapper |
| LSTM, CNN, Transformer | 1-5 MB | `model.save_weights("weights.h5")` + architecture re-instantiation |

### 2. DEPLOY

```
User visits Deployed Models page
  → Grid of saved model cards with:
      - Model type badge (blue=classical, purple=deep, orange=ensemble)
      - Best Sharpe from source backtest
      - Date created, tags (editable inline)
      - Status: active / inactive
  → Clicks "Activate" on a model
  → API: POST /models/{id}/activate
  → SQLite: sets status='active', deactivates previous active for same model type
  → File: writes to deployed_models/.active
  → Model is now ready for prediction endpoint
```

**Why file-based active pointer instead of Redis:**
The `.active` file is a 200-byte JSON keyed by model type. Atomic writes via `tempfile.mkstemp()` + `os.replace()`. No Redis dependency. Redis can be added in S21 for cross-process hot-swap if needed.

### 3. PREDICT (Live Use)

```
GET /models/active/predict?pair=EURUSD&tf=H1
  → Read deployed_models/.active → {"logistic": "abc123"}
  → Load snapshot: scaler, imputer, calibration, model
  → Fetch last N bars from SQLite candles table
  → Apply feature engineering (same features_config from metadata)
  → Scale → impute → predict_proba → calibrate → apply coverage gate
  → Return: {class: "BUY", confidence: 0.82, timestamp: "2026-05-21T14:30:00Z"}
  → Log to live_predictions table for auditing
```

**Cold-start times:**
| Model Type | Load + Predict |
|-----------|:---:|
| Logistic, SVM, DT | ~200ms |
| Random Forest | ~500ms |
| XGBoost | ~500ms |
| LSTM, CNN, Transformer | ~2-5s (TF import + model load + warmup) |

**Warm-start optimization:** Pre-load active deep models into memory on app startup, keep warm with periodic no-op predictions.

### 4. EXPORT (Share with Other Users)

**Mode A: Full Snapshot Export**
```
User clicks "Export" on Deployed Models page
  → Packs: deployed_models/logistic_20260521T143000/ → logistic_20260521T143000.koda
  → Format: .zip of the snapshot directory
  → Contains: model.joblib + scaler.joblib + calibration.joblib + metadata.json + manifest.sha256
```

**Mode B: Hyperparams-Only Export (for sharing strategies not weights)**
```
User clicks "Export Config Only"
  → Packs: metadata.json (no model weights)
  → Recipient imports config → runs their own HPO on their own data
  → Same strategy concept, different fit → avoids data leakage, respects data ownership
  → Size: ~2 KB
```

### 5. IMPORT

```
User clicks "Import Model" → selects .koda file
  → API validates manifest.sha256 (tamper detection)
  → API validates metadata.model_type matches registered model types
  → API checks features_config compatibility (warns if feature set differs from current)
  → Extracts to deployed_models/ with new ID
  → Model appears in Deployed Models grid with "imported" tag
  → User can then activate and use for predictions
```

**What's guaranteed when importing:**
- Same predictions on same data (bit-identical) for classical models via joblib + seed
- Same predictions for XGBoost if same version installed
- Approximately same for deep models (floating-point variance ±0.001 on probabilities)
- Feature compatibility check: warns if importer's feature config differs from source
- Tamper detection: manifest.sha256 verified on import — rejects modified files

**What's NOT guaranteed (and shouldn't be):**
- Same performance on different data. A model trained on EURUSD 2019-2025 may fail on GBPUSD or EURUSD 2026.
- Regulatory compliance. If a user sells model weights that encode insider patterns — their problem, not the software's.

---

## Architecture

### New Files

```
pipeline/
  model_persistence.py      — save/load complete model snapshots
  model_registry_disk.py    — scan deployed_models/, validate, register in SQLite

frontend/src/pages/Models/
  DeployedModelsPage.tsx    — grid of saved model cards

frontend/src/components/shared/
  TagEditor.tsx             — inline tag add/remove component
```

### Modified Files

```
models/base_model.py        — wire save()/load() with artifact bundling
models/meta_ensemble.py     — NEW: wraps N models via sklearn VotingClassifier
models/registry.py          — register meta_ensemble
config.py                   — meta_ensemble search space
pipeline/data_sqlite.py     — deployed_models + live_predictions tables
api/services/__init__.py    — deployed models CRUD
api/tasks.py                — auto-save after walk-forward, pip_freeze capture
api/routers/models.py       — deployed models + prediction endpoints
api/routers/backtest.py     — experiment diff + tags endpoints
pipeline/backtester/
  ensemble_mixin.py         — dispatch meta_ensemble evaluation
frontend/src/
  AppShell.tsx              — add "Models" to sidebar nav
  stores/useBacktestStore.ts — parent_job_id for lineage
  pages/Backtest/
    ModelSelector.tsx       — Signal Committee card
  pages/Results/
    ResultsHistoryPage.tsx  — experiment diff modal
```

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  api/tasks.py    │────→│ model_persistence │────→│ deployed_models/ │
│  (save snapshot) │     │ .py               │     │ (disk)           │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
       │                                                   │
       │ pip_freeze                                        │ scan on startup
       ▼                                                   ▼
┌─────────────────┐                              ┌──────────────────┐
│  jobs table      │                              │ model_registry_  │
│  (parent_job_id) │                              │ disk.py          │
└─────────────────┘                              └────────┬─────────┘
       │                                                   │
       │ experiment diff                                   │ register
       ▼                                                   ▼
┌─────────────────┐                              ┌──────────────────┐
│  GET /experiments│                              │ deployed_models  │
│  /{id}/diff      │                              │ table (SQLite)    │
└─────────────────┘                              └──────────────────┘
```

---

## Implementation Order

### Phase A: Model Snapshot System (4-5h)

**A.1 Wire BaseModel.save()/load()** — `models/base_model.py`
- `save()` already defined (line 64-73) using `joblib.dump()`
- `load()` already defined as classmethod using `joblib.load()`
- These currently save the model object alone. Extend `save()` to accept optional `artifact_dir` parameter that bundles scaler, imputer, calibration alongside.
- The `mlb_base_attrs` pattern: model stores a reference to artifacts via `self._artifact_bundle = {...}`. `save()` writes them all, `load()` restores them all.

**A.2 `pipeline/model_persistence.py`** — New file
```python
def save_snapshot(bt, model_type: str, output_dir: str = None) -> str:
    """Save complete model snapshot to deployed_models/{model_type}_{timestamp}/"""
    # 1. Create output directory with timestamp
    # 2. joblib.dump(bt.model) → model.joblib
    # 3. joblib.dump(bt.scaler) → scaler.joblib (if exists, else skip)
    # 4. joblib.dump(bt.imputer) → imputer.joblib (if exists, else skip)
    # 5. joblib.dump(calibration object) → calibration.joblib (if exists)
    # 6. Write metadata.json
    # 7. Compute sha256 of all files → manifest.sha256

def load_snapshot(snapshot_path: str) -> dict:
    """Load complete model snapshot. Returns dict with model, scaler, etc."""
    # 1. Verify manifest.sha256 (tamper check)
    # 2. joblib.load(model.joblib)
    # 3. joblib.load(scaler.joblib) if exists
    # 4. joblib.load(imputer.joblib) if exists
    # 5. joblib.load(calibration.joblib) if exists
    # 6. Load metadata.json
    # 7. Return dict with all artifacts
```

**A.3 Metadata schema**
```json
{
  "schema_version": 1,
  "model_type": "xgboost",
  "best_params": {"lags": 16, "lag_depth": 1, "roll_windows": [20, 60]},
  "feature_names": ["adx_14", "atr_14", "rsi_14", "close_lag_1", ...],
  "features_config_hash": "a92e839f24",
  "coverage_conf_thr": 0.518,
  "calibrate_method": "sigmoid",
  "input_shape": [128],
  "train_start": "2019-01-01",
  "train_end": "2025-01-01",
  "created_at_utc": "2026-05-21T14:30:00Z",
  "seed": 42,
  "pip_freeze": "numpy==1.26.4\nscikit-learn==1.5.1\nxgboost==2.1.0\n...",
  "job_id_parent": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "metrics": {
    "sharpe": 0.72,
    "win_rate": 0.54,
    "total_return_pct": 18.5,
    "max_drawdown": -12.3,
    "total_trades": 142
  }
}
```

**A.4 Environment snapshot** — `api/tasks.py`
- At start of `_run_backtest_impl()`: capture `subprocess.check_output(["pip", "freeze"]).decode()`
- Store in `config["pip_freeze"]` before saving to jobs table
- Included in snapshot metadata from job config

**A.5 Auto-save after walk-forward** — `api/tasks.py` (near line 945)
- After `all_metrics.append(metrics_row)`, call `save_snapshot(bt, model_type)`
- Only save the LAST repeat's model (fully trained on all data)
- Output path: `deployed_models/{model_type}_{timestamp}/`

**A.6 Experiment lineage** — multiple files
- Add `parent_job_id` column to `jobs` table (nullable TEXT)
- When user clicks "Re-run" (ResultsHistoryPage) or "Apply Study" (LLMAdvisor), set `parent_job_id` in the new job's config
- Frontend: `useBacktestStore.toRequestPayload()` includes `parent_job_id` from `activeParentJobId` state

**Phase A Tests:**
```
test_snapshot_save_load_roundtrip:
  Train logistic on synthetic data → save_snapshot → load_snapshot → predict on same X
  Assert: np.allclose(original_preds, loaded_preds)

test_snapshot_metadata_completeness:
  Save snapshot → load metadata.json
  Assert: all required keys present, types correct, no nulls in required fields

test_snapshot_missing_scaler_graceful:
  Train tree-based model (no scaler) → save → load
  Assert: scaler is None, prediction still works

test_pip_freeze_in_config:
  Submit backtest → get job config
  Assert: config["pip_freeze"] exists and contains "scikit-learn" and "xgboost"

test_parent_job_id_lineage:
  Run 3 backtests in chain (B2 parent=A, B3 parent=B2)
  Assert: B2.parent_job_id == A.id, B3.parent_job_id == B2.id
```

---

### Phase B: Deployment Registry & Experiment Tracking (5-6h)

**B.1 SQLite deployed_models table**
```sql
CREATE TABLE IF NOT EXISTS deployed_models (
    id              TEXT PRIMARY KEY,
    model_type      TEXT NOT NULL,
    snapshot_path   TEXT NOT NULL,
    best_sharpe     REAL,
    best_return     REAL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'inactive',
    tags            TEXT DEFAULT '[]',
    parent_job_id   TEXT
);
```

**B.2 `pipeline/model_registry_disk.py`** — New file
- `register_snapshot(snapshot_path)` — validate, extract metadata, INSERT into deployed_models
- `get_all_deployed()` — SELECT from deployed_models, verify path still exists on disk
- `activate_model(id)` — SET status='active', deactivate others of same model_type
- `deactivate_model(id)` — SET status='inactive'
- `delete_model(id)` — DELETE from DB + remove directory from disk
- `get_active_model(model_type)` — return path for currently active model of given type
- `scan_and_repair()` — on startup: scan deployed_models/ dir, register any unregistered snapshots, remove entries for missing directories

**B.3 API endpoints** — `api/routers/models.py`
```
GET    /models/deployed?model_type=logistic&status=active&tags=good
POST   /models/{id}/activate
DELETE /models/{id}
GET    /models/{id}/metadata
```

**B.4 Experiment diff** — `api/routers/backtest.py`
```
GET /experiments/{id}/diff?compare={other_id}
Response: {
  "base": { "job_id": "...", "created_at": "..." },
  "compare": { "job_id": "...", "created_at": "..." },
  "added_keys": { "lags": 30 },
  "removed_keys": {},
  "changed_values": { "conf_threshold": {"from": 0.7, "to": 0.8} },
  "unchanged_count": 142
}
```

**B.5 Experiment tags** — `api/routers/backtest.py`
```
PATCH /experiments/{id}/tags
Body: { "action": "add", "tag": "overfit" }
Body: { "action": "remove", "tag": "good" }
```

**B.6 Frontend: Deployed Models page** — New `DeployedModelsPage.tsx`
- Grid layout: 3 columns at 1200px+
- Each card: model type badge (color-coded), Sharpe, date, tag chips, activate/export/delete buttons
- Filter bar: model type dropdown, status toggle, tag filter
- Active card has highlight border
- Empty state: "No models deployed yet. Run a backtest to save your first model."

**B.7 Frontend: Experiment diff + tags**
- `ResultsHistoryPage.tsx`: Add "Compare" checkbox column → select 2 rows → "Diff" button → modal
- Tag chips visible on each row, click to add/remove
- `TagEditor.tsx`: popover with tag suggestions ("good", "overfit", "failed", "high_sharpe") + free-text input

**B.8 Frontend: Navigation**
- Add "Models" icon to sidebar between "Backtest" and "Results"

**Phase B Tests:**
```
test_deployed_models_crud:
  Register snapshot → GET /models/deployed → verify present
  DELETE → GET → verify absent

test_only_one_active_per_model_type:
  Activate logistic_1 → Activate logistic_2
  GET /models/deployed → logistic_1 is inactive, logistic_2 is active

test_config_diff_correctness:
  Job A: {lags: 10, conf: 0.7}
  Job B: {lags: 20, conf: 0.8}
  GET /experiments/A/diff?compare=B
  Assert: changed_values == {lags: {from:10, to:20}, conf: {from:0.7, to:0.8}}

test_tags_persist:
  PATCH /experiments/A/tags {add: "good"}
  GET /experiments/A → tags contains "good"

test_parent_lineage_visible:
  GET /experiments/child → response includes parent_job_id with parent metadata

test_corrupt_snapshot_skipped:
  Create empty directory in deployed_models/
  Call scan_and_repair() → directory is ignored, no crash
```

---

### Phase C: Multi-Model Signal Engine (4-5h)

**C.1 `models/meta_ensemble.py`** — New file
```python
class MetaEnsemble(BaseModel):
    """Wraps N trained models. Combines signals via voting or weighting."""
    
    def __init__(self, sub_models: list, method: str = "majority",
                 weights: list = None):
        # sub_models: list of BaseModel instances (loaded from snapshots)
        # method: "majority" | "soft" | "weighted"
        # weights: if method="weighted", per-model weight (e.g., OOS Sharpe)
        
    def predict_proba(self, X_seq=None, X_flat=None):
        # Route each sub-model's predict_proba based on its type
        # Classical → X_flat only
        # Deep → X_seq
        # Combine via sklearn.VotingClassifier logic

    def save(self, path):  # Save sub-model references, not weights
    def load(cls, path):   # Load by reference to deployed model IDs
```

Combination methods:
- **Majority vote**: each model votes {-1,0,1}, `sign(∑ votes)`. Ties → 0.
- **Soft voting**: average probabilities → `argmax(mean(probas))`.
- **Weighted**: average weighted by user-configurable weights (default: per-model OOS Sharpe from metadata).

**C.2 Register in MODEL_REGISTRY** — `models/registry.py`
```python
"meta_ensemble": lambda: MetaEnsemble(sub_models=[], method="majority"),
```
Add search space in `config.py`:
```python
"meta_ensemble": {
    "combination_method": ("categorical", ["majority", "soft", "weighted"]),
    "n_members": ("int", 2, 5),
}
```

**C.3 Wire into pipeline** — `pipeline/backtester/ensemble_mixin.py`
- Add `model_type == "meta_ensemble"` branch in `_test_ensemble_strategy_core()`
- Load sub-models from `config["meta_ensemble"]["snapshot_ids"]` — list of deployed model IDs
- Pass through the same train/test cycle as existing ensemble types

**C.4 Frontend** — `ModelSelector.tsx`
- Add "Signal Committee" card in the model selection grid
- Clicking it opens a sub-panel:
  - Checkbox list of deployed models (fetched from `GET /models/deployed`)
  - Radio: majority vote / soft voting / weighted
  - Weight config if "weighted" selected (sliders per model, default from Sharpe)
  - "Add to Selection" button

**Phase C Tests:**
```
test_majority_vote_3_models:
  Model A preds: [1, -1, 1]  # buy, sell, buy
  Model B preds: [1,  1, 1]  # buy, buy, buy
  Model C preds: [-1,-1, 1]  # sell, sell, buy
  Majority: [1, -1, 1]  # 2/3 for each position
  Assert: output matches expected

test_soft_voting_probability_average:
  A: [[0.7,0.2,0.1], ...]  B: [[0.1,0.8,0.1], ...]
  Result: [[0.4,0.5,0.1], ...]
  Assert: allclose(result, expected)

test_confidence_weighted_by_sharpe:
  A (Sharpe=1.0): [[0.9,0.05,0.05]]  weight=1.0
  B (Sharpe=0.5): [[0.1,0.8,0.1]]   weight=0.5
  Expected: heavily influenced toward A's prediction
  Assert: final probability > 0.6 for class 0 (influenced by A)

test_committee_from_deployed_snapshots:
  Deploy 3 models → Create committee referencing their IDs
  Load → verify no re-training (models loaded from disk)
  Predict → 3-class probability output

test_committee_handles_model_disagreement:
  A: [1, -1, 1]  B: [-1, 1, -1]  C: [1, -1, -1]
  No clear majority for first 2 samples
  Assert: ties → 0 (neutral)

test_empty_committee_errors:
  MetaEnsemble(sub_models=[]) → predict()
  Assert: raises clear ValueError with message
```

---

### Phase D: Live Trading Bridge (2-3h)

**D.1 Active model pointer** — `pipeline/model_registry_disk.py`
```python
ACTIVE_POINTER_PATH = "deployed_models/.active"
# Format: {"logistic": "abc123", "xgboost": "def456"}

def get_active_model_id(model_type: str) -> str | None:
    """Read active model ID for given type from .active file."""

def set_active_model_id(model_type: str, model_id: str):
    """Atomically write active model ID. Uses tempfile + os.replace()."""

def clear_active_model_id(model_type: str):
    """Remove entry from .active."""
```

**D.2 Live prediction endpoint** — `api/routers/models.py`
```
GET /models/active/predict?pair=EURUSD&tf=H1
```
1. Read `.active` file → get active model IDs
2. Load each active model's snapshot
3. Fetch last N bars from `candles` table
4. Compute features using saved `features_config`
5. Scale → impute → predict_proba → calibrate → apply coverage gate
6. Log prediction to `live_predictions` table
7. Return JSON: `{timestamp, predictions: {logistic: {class, confidence}, xgboost: {...}}}`

**D.3 `live_predictions` table** — `pipeline/data_sqlite.py`
```sql
CREATE TABLE IF NOT EXISTS live_predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    pair        TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    predicted_class INTEGER NOT NULL,   -- -1, 0, 1
    confidence  REAL NOT NULL,
    signal_used INTEGER NOT NULL DEFAULT 0  -- whether this signal triggered a trade
);
```

**D.4 Live vs backtest comparison** — `api/routers/models.py`
```
GET /models/active/compare?pair=EURUSD
```
Compares last N live predictions against saved backtest metrics for the active model.

**Phase D Tests:**
```
test_active_pointer_atomic:
  Write {"logistic": "abc"} → read → verify
  Write {"logistic": "def"} → read → old value replaced
  Assert: file never corrupted (atomic write)

test_predict_endpoint_returns_valid_shape:
  Deploy model → activate → GET /models/active/predict
  Assert: response contains {class: -1/0/1, confidence: 0.0-1.0, timestamp}

test_live_predictions_logged:
  Call predict → query live_predictions table
  Assert: row exists with correct model_id, pair, tf

test_model_hot_swap_atomic:
  Predict with Model A → change active pointer → Predict with Model B
  Assert: second prediction uses Model B (different model_id in log)

test_compare_backtest_vs_live:
  Create some live predictions → GET /models/active/compare
  Assert: response contains backtest_sharpe, live_accuracy, prediction_count
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Zero new dependencies** | joblib is already imported in `base_model.py`, sklearn in the entire model stack, SQLite everywhere. No pip install needed. |
| **File-based active pointer, not Redis** | 200-byte JSON, atomic writes, no server process. Redis for hot-swap can be added in S21 if cross-process signaling is needed. |
| **`model.save_weights()` for deep models** | Lighter than `model.save()` (1-5 MB vs 5-50 MB SavedModel). Architecture re-instantiated from config. Same predictions within floating-point tolerance. |
| **`.koda` export format** | Simple zip of snapshot directory. `manifest.sha256` provides tamper detection. `.koda` extension distinguishes from generic `.zip`. |
| **Config-only export mode** | For sharing strategy concepts without data leakage. Recipient runs their own HPO → same strategy, different fit. |
| **Committee as a model type** | Fits existing registry → pipeline → evaluation → UI flow. No new special case code paths. |
| **`parent_job_id` for lineage** | Links experiments without needing a separate graph table. Follow parent chain recursively for full history. |

---

## Rollout Strategy

1. **Phase A first** — model snapshots block everything else. Can't deploy or committee without saving models.
2. **Phase B second** — registry makes snapshots discoverable. Experiment diff and tags add immediate research value.
3. **Phase C third** — committee builds on deployed models from Phase B.
4. **Phase D last** — live trading bridge is thin; mostly wiring existing endpoints.

After Phase A+B, users can already run experiments, save models, and browse their collection. Phase C+D add power features that benefit from having a model library to work with.
