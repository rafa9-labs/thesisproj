"""Model package — BaseModel ABC + model registry."""
from models.base_model import BaseModel

# Lazy-load registry only when needed (avoids heavy sklearn/TF imports at top level)
def __getattr__(name):
    if name in ("MODEL_REGISTRY", "build_model", "register_model"):
        from models import registry
        return getattr(registry, name)
    raise AttributeError(f"module 'models' has no attribute {name!r}")