"""Model hyperparameter metadata schemas."""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel


class HyperparamRange(BaseModel):
    """Numeric range hyperparameter (continuous or step)."""
    type: Literal["float_range", "int_range"] = "float_range"
    low: float
    high: float
    step: Optional[float] = None
    log_scale: bool = False
    default: Optional[float] = None


class HyperparamChoice(BaseModel):
    """Categorical choice hyperparameter."""
    type: Literal["choice"] = "choice"
    values: List[Any]
    default: Optional[Any] = None


class HyperparamFixed(BaseModel):
    """Fixed (non-tunable) parameter shown for reference."""
    type: Literal["fixed"] = "fixed"
    value: Any


HyperparamSpec = Union[HyperparamRange, HyperparamChoice, HyperparamFixed]


class ModelHyperparams(BaseModel):
    """Hyperparameter metadata for one model."""
    model: str
    display_name: str
    category: str
    tunable: bool
    params: dict[str, HyperparamSpec]


class ModelHyperparamsResponse(BaseModel):
    """All models' hyperparameter metadata."""
    models: List[ModelHyperparams]