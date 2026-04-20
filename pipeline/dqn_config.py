"""
DQN configuration loading and coercion.

Extracted from MLBacktesterNoWFO.py lines 541-662.
"""

from pipeline._imports import *  # noqa: F401,F403

def _load_default_dqn_cfg(path: str) -> dict:
    """
    Load a baseline DQN config from JSON (e.g. dqn_grid_config.json).
    If the file is missing or invalid, return an empty dict so that
    _coerce_dqn_cfg can fill in safe defaults.
    """
    cfg = {}
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                cfg = json.load(f) or {}
                if isinstance(cfg, dict):
                    cfg.setdefault("_cfg_source", "grid_defaults")
            if not isinstance(cfg, dict):
                print(f"⚠️ DQN config at {path} is not a dict; ignoring.")
                cfg = {}
        else:
            print(f"⚠️ DQN default config file not found at {path}; using built-in defaults.")
    except Exception as e:
        print(f"⚠️ Failed to load DQN default config from {path}: {e}")
        cfg = {}

    return cfg


def _coerce_dqn_cfg(cfg: dict, *, strict: bool = False) -> dict:
    """
    Normalize and guardrail DQN config so training actually runs.

    NOTE (your requirement):
    - If config is loaded from dqn_grid_config.json, we treat it as source-of-truth
      and DO NOT override user values (no clamping).
    - We still fill defaults for missing keys so the program runs.
    """
    cfg = dict(cfg or {})

    # Respect JSON defaults as source-of-truth when marked.
    strict = bool(strict) or (cfg.get("_cfg_source") == "grid_defaults")

    # ---- Start from candidate values / defaults ----
    # In strict mode: do not inflate/alter values beyond basic validity.
    # In non-strict mode: allow a bit of "guardrailing" for coherence.
    def _as_int(key: str, default: int, minv: int | None = None) -> int:
        v = cfg.get(key, default)
        try:
            v = int(v)
        except Exception:
            v = default
        if minv is not None:
            v = max(minv, v)
        return v

    bs   = _as_int("batch_size", 64, minv=1)
    buf  = _as_int("buffer_size", 50000, minv=1)
    warm = _as_int("warmup_steps", max(5000, 2 * bs), minv=0)

    # Keep warmup < buffer
    if warm >= buf:
        if strict:
            warm = max(0, buf - 1)
        else:
            buf = max(buf, warm + bs)

    # Batch must fit in buffer
    if bs > buf:
        if strict:
            bs = buf
        else:
            bs = max(32, buf // 2)

    cfg["batch_size"]   = bs
    cfg["buffer_size"]  = buf
    cfg["warmup_steps"] = warm

    # ---- Fill defaults only (never override explicitly provided keys) ----
    cfg.setdefault("gamma", 0.99)
    cfg.setdefault("epsilon", 1.0)
    cfg.setdefault("epsilon_min", 0.05)
    cfg.setdefault("epsilon_decay", 0.999)          # legacy fallback
    cfg.setdefault("epsilon_decay_steps", 200000)   # default horizon
    cfg.setdefault("learning_rate", 0.0005)
    cfg.setdefault("replay_freq", 4)
    cfg.setdefault("target_update_freq", 2000)
    cfg.setdefault("episodes", 50)
    cfg.setdefault("action_size", 3)

    # ---- Type coercion + basic validity (no “policy” overrides) ----
    # episodes: respect exactly if provided (no max(30, ...))
    try:
        cfg["episodes"] = max(1, int(cfg.get("episodes", 50)))
    except Exception:
        cfg["episodes"] = 50

    # epsilon_decay_steps: strict => only ensure >=1; non-strict => ensure a sane lower bound
    try:
        if strict and "epsilon_decay_steps" in cfg:
            cfg["epsilon_decay_steps"] = max(1, int(cfg["epsilon_decay_steps"]))
        else:
            cfg["epsilon_decay_steps"] = max(1, int(cfg.get("epsilon_decay_steps", 200000)))
    except Exception:
        cfg["epsilon_decay_steps"] = 200000

    # epsilon_min: strict => accept as-is (just coerce to float); non-strict => clamp to [0.01, 0.2]
    try:
        eps_min = float(cfg.get("epsilon_min", 0.05))
        if strict and "epsilon_min" in cfg:
            cfg["epsilon_min"] = eps_min
        else:
            cfg["epsilon_min"] = min(0.2, max(0.01, eps_min))
    except Exception:
        cfg["epsilon_min"] = 0.05

    return cfg


# ── Global HPO config helpers ────────────────────────────────────────────────
# Use the same default as utilsNoWFO (repo-root /hpo) to avoid CWD-dependent bugs.
HPO_CONFIG_DIR = os.environ.get(
    "MLB_HPO_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hpo"),
)

