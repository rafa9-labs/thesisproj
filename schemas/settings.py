"""
schemas.settings — Pydantic replacement for config.py @dataclass settings.

═══════════════════════════════════════════════════════════════════════════════
EDUCATIONAL: Pydantic vs @dataclass for Configuration
═══════════════════════════════════════════════════════════════════════════════

BEFORE (config.py):
  @dataclass
  class Settings:
      data_key: str = "EURUSD_H1"
      log_mode: str = "COMPACT"
      ...
  
  settings = Settings()
  settings.log_mode = os.getenv("LOG_MODE", "COMPACT")
  # What if LOG_MODE is "compact" (lowercase)? No validation!
  # What if someone sets log_mode = 42? No type check!

AFTER (with Pydantic):
  class PydanticSettingsConfig(BaseModel):
      log_mode: str = Field(default="COMPACT", pattern="^(COMPACT|VERBOSE|MINIMAL)$")
  
  config = PydanticSettingsConfig.model_validate(os.environ)
  # If LOG_MODE="compact" → ValidationError (must be uppercase)
  # If LOG_MODE=42 → ValidationError (must be string)

WHY NOT USE pydantic-settings directly?
  The `pydantic-settings` package extends BaseModel with automatic
  env var loading, .env file parsing, etc. It's the "right" way,
  but we keep it simple here to avoid an extra dependency.
  This model can be upgraded to BaseSettings when pydantic-settings
  is added to requirements.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PydanticSettingsConfig(BaseModel):
    """
    Typed global settings — future replacement for config.py Settings dataclass.
    
    This model is NOT YET WIRED into the pipeline. It exists as a typed
    reference for the migration plan. When ready:
    
    1. Replace `from config import settings` with `from schemas.settings import settings`
    2. The pipeline continues working because field names are identical
    3. But now ALL config values are validated on construction
    
    EDUCATIONAL NOTE — env var loading:
    In the current codebase, config.py loads env vars with:
        val = os.getenv("LOG_MODE", default)
    
    With Pydantic, you'd do:
        PydanticSettingsConfig(
            log_mode=os.getenv("LOG_MODE", "COMPACT"),
            ...
        )
    
    Or with pydantic-settings (future):
        class Settings(BaseSettings):  # auto-reads env vars
            log_mode: str = "COMPACT"
            model_config = SettingsConfigDict(env_prefix="APP_")
    """
    model_config = ConfigDict(extra="ignore")
    
    # ── Data ──
    data_key: str = Field(
        default="EURUSD_H1",
        description="Default dataset key (maps to CSV in DATA_FILES dict).",
    )
    
    # ── Logging ──
    log_mode: Literal["COMPACT", "VERBOSE", "MINIMAL"] = Field(
        default="COMPACT",
        description="Log verbosity level.",
    )
    cv_debug: bool = Field(
        default=False,
        description="Enable detailed CV fold debugging output.",
    )
    skip_plots: bool = Field(
        default=False,
        description="Skip all matplotlib plotting (headless mode).",
    )
    
    # ── Computation ──
    tf_force_cpu: bool = Field(
        default=False,
        description="Force TensorFlow to use CPU only.",
    )
    cv_jobs: Optional[int] = Field(
        default=None,
        ge=1,
        le=64,
        description="Number of parallel CV jobs. None = auto-detect.",
    )
    mlb_threads: Optional[int] = Field(
        default=None,
        ge=1,
        le=64,
        description="Thread limit for numpy/sklearn. None = auto-detect.",
    )
    
    # ── Output ──
    results_run_dir: Optional[str] = Field(
        default=None,
        description="Override output directory for results. None = auto-generate.",
    )
    cv_table_mode: Literal["off", "normal", "full"] = Field(
        default="off",
        description="CV fold table printing mode.",
    )
    
    def to_dict(self) -> Dict:
        """
        Convert back to dict for backward compatibility with config.py.
        
        In the current codebase, many modules do: `from config import settings`
        and access settings.data_key. During migration, this method lets us
        create a dict that looks like the old dataclass.
        """
        return self.model_dump()