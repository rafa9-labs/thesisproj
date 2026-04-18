"""
schemas — Pydantic data models for the FX ML Backtester.

═══════════════════════════════════════════════════════════════════════════════
WHY PYDANTIC? (The Educational Bit)
═══════════════════════════════════════════════════════════════════════════════

BEFORE (current codebase):
  params: Dict[str, Any] = {...}           ← no type safety
  features_config: dict = json.load(f)      ← no validation
  results = {"metrics": {...}}              ← what keys? what types?

  Problems:
    1. A typo like "confidance_threshold" silently flows through 20 functions
       before crashing with an unhelpful NumPy error.
    2. Passing "high" instead of 0.8 for confidence_threshold works at first,
       then explodes deep in the pipeline.
    3. Every function does its own manual validation:
       float(params.get("tb_pt_mult", 2.0))  ← repeated 40+ times

AFTER (with Pydantic):
  params = BacktestParams(confidence_threshold=0.8)
  params.confidence_threshold  ← guaranteed float, guaranteed in [0, 1]

  If you try: BacktestParams(confidence_threshold="high")
  Pydantic raises: ValidationError: Input should be a valid number
  ...at the BOUNDARY, not 20 functions deep.

═══════════════════════════════════════════════════════════════════════════════
HOW PYDANTIC WORKS (Under the Hood)
═══════════════════════════════════════════════════════════════════════════════

1. BaseModel.__init__() is a @classmethod that:
   a. Takes your keyword arguments
   b. Runs them through type validators (int, float, str, etc.)
   c. Runs them through Field constraints (ge, le, pattern, etc.)
   d. Runs any @field_validator or @model_validator methods
   e. Only THEN sets them as instance attributes

2. This happens at the "boundary" — where data enters your system.
   Inside your system, you can trust types 100%.

3. .model_dump() → dict, .model_dump_json() → JSON string,
   ModelClass.model_validate(data) → parse from dict/JSON.
   .model_json_schema() → JSON Schema (for API docs, UI generation).

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE: "Validate at the boundary, trust inside"
═══════════════════════════════════════════════════════════════════════════════

  UI params (raw dict)                    API JSON payload
        │                                       │
        ▼                                       ▼
  BacktestParams.model_validate(dict)    BacktestParams.model_validate_json(str)
        │                                       │
        └─────────────┬─────────────────────────┘
                      ▼
              VALIDATED BacktestParams
                      │
                      ▼
              Pipeline (trusts types)
                      │
                      ▼
              BacktestResult (validated output)

═══════════════════════════════════════════════════════════════════════════════
MODULE MAP
═══════════════════════════════════════════════════════════════════════════════

  schemas.backtest   → BacktestParams, BacktestResult, AggregateMetrics
  schemas.features   → FeaturesConfig, IndicatorWindows, CVConfig
  schemas.hpo        → HPOConfigPayload (for persistence)
  schemas.settings   → PydanticSettingsConfig (future replacement for config.py)

Import everything:
    from schemas import BacktestParams, BacktestResult, FeaturesConfig
"""

# ── Re-export all public models for convenient imports ──
from schemas.backtest import (
    BacktestParams,
    AggregateMetrics,
    BacktestResult,
)
from schemas.features import (
    IndicatorWindows,
    FeaturesConfig,
    CVConfig,
)
from schemas.hpo import (
    HPOConfigPayload,
    ParamImportance,
)
from schemas.settings import (
    PydanticSettingsConfig,
)

__all__ = [
    "BacktestParams",
    "AggregateMetrics",
    "BacktestResult",
    "IndicatorWindows",
    "FeaturesConfig",
    "CVConfig",
    "HPOConfigPayload",
    "ParamImportance",
    "PydanticSettingsConfig",
]