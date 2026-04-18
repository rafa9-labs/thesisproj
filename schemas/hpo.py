"""
schemas.hpo — Pydantic models for HPO (Hyperparameter Optimization) persistence.

═══════════════════════════════════════════════════════════════════════════════
EDUCATIONAL: Why this replaces pipeline/hpo_persistence.py
═══════════════════════════════════════════════════════════════════════════════

BEFORE (pipeline/hpo_persistence.py):
  1. Tries to load JSON from 4+ candidate filenames
  2. Guesses the schema (v1? v2? flat? nested?)
  3. Manually parses each variant with try/except chains
  4. Returns an untyped dict — caller has no idea what's inside

  Total: ~200 lines of filename-guessing and schema-sniffing code.

AFTER:
  config = HPOConfigPayload.model_validate_json(json_string)
  
  One line. Pydantic handles schema validation, type coercion, and
  backward-compatible field aliasing. If the JSON doesn't match,
  you get a clear ValidationError with the exact field and reason.

KEY CONCEPT — model_validate_json():
  Pydantic v2 can parse JSON strings DIRECTLY, without first converting
  to a Python dict via json.loads(). This is:
  1. Faster (Rust-based parser in pydantic-core)
  2. Safer (validates during parsing, not after)
  3. More convenient (one call instead of two)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ══════════════════════════════════════════════════════════════════════════════
# ParamImportance — Optuna parameter importance scores
# ══════════════════════════════════════════════════════════════════════════════

class ParamImportance(BaseModel):
    """
    Importance scores for hyperparameters from an Optuna study.
    
    EDUCATIONAL NOTE — Simple model:
    This is a "flat" model — just two fields, no nested models.
    But even here, Pydantic adds value:
    - param_name is guaranteed to be a non-empty string
    - importance is guaranteed to be a float in [0, 1]
    
    Before Pydantic, a bug could set importance = "high" and it would
    silently pass through until it crashed a plotting function.
    """
    model_config = ConfigDict(extra="ignore")
    
    param_name: str = Field(min_length=1)
    importance: float = Field(ge=0.0, le=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# HPOConfigPayload — The full HPO configuration saved to / loaded from disk
# ══════════════════════════════════════════════════════════════════════════════
#
# This replaces the untyped JSON files in hpo/*.json and the complex
# loading logic in pipeline/hpo_persistence.py.

class HPOConfigPayload(BaseModel):
    """
    Validated HPO configuration payload for persistence.
    
    This model represents the JSON file saved after a successful HPO run.
    It replaces the raw dicts loaded in pipeline/hpo_persistence.py.
    
    EDUCATIONAL NOTE — Field aliases for backward compatibility:
    The existing JSON files use keys like "best_params" and "best_value".
    If we renamed these, all existing saved configs would break.
    
    Pydantic's Field(alias=...) lets us use a DIFFERENT name in Python
    vs. JSON. In Python we use best_params (snake_case), in JSON it's
    still "best_params" (matching existing files).
    
    For future fields, we can use alias to bridge old → new naming:
      model_type: str = Field(alias="model")  # JSON has "model", Python uses "model_type"
    
    model_validate(data, by_alias=False) uses Python names.
    model_validate(data, by_alias=True) uses JSON aliases.
    """
    model_config = ConfigDict(extra="ignore")
    
    # ── Identity ──
    model_type: str = Field(
        default="logistic",
        description="Model type this config was tuned for.",
    )
    
    # ── Optuna results ──
    best_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Best hyperparameters found by Optuna.",
    )
    best_value: Optional[float] = Field(
        default=None,
        description="Best objective value (e.g., Sharpe ratio).",
    )
    n_trials: int = Field(
        default=0,
        ge=0,
        description="Number of Optuna trials completed.",
    )
    
    # ── Metadata ──
    # These are set when the config is saved to disk.
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO format timestamp when config was saved.",
    )
    data_key: Optional[str] = Field(
        default=None,
        description="Dataset identifier used during tuning.",
    )
    features_hash: Optional[str] = Field(
        default=None,
        description="SHA256 hash of the features config (for cache invalidation).",
    )
    
    # ── Importance ──
    param_importances: Optional[List[ParamImportance]] = Field(
        default=None,
        description="Per-hyperparameter importance scores.",
    )
    
    # ── Full study metadata (optional) ──
    study_name: Optional[str] = Field(default=None)
    direction: Optional[str] = Field(default="maximize")
    
    def to_json_file(self, path: str) -> None:
        """
        Save this config to a JSON file.
        
        Replaces save_hpo_config_to_disk() in pipeline/hpo_persistence.py.
        """
        import json
        with open(path, "w") as f:
            json.dump(self.model_dump(exclude_none=True), f, indent=2)
    
    @classmethod
    def from_json_file(cls, path: str) -> "HPOConfigPayload":
        """
        Load and validate a config from a JSON file.
        
        Replaces load_hpo_config_from_disk() in pipeline/hpo_persistence.py.
        
        EDUCATIONAL NOTE:
        This ONE method replaces ~60 lines of filename-guessing and
        schema-sniffing code in hpo_persistence.py. If the JSON doesn't
        match the expected schema, Pydantic raises a ValidationError
        with the exact field and reason — no more silent corruption.
        """
        import json
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)