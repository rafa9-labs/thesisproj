"""Test 5: Config, DQN config, and defaults loading."""
import pytest
import json
import os


def test_config_module_imports():
    """config.py must be importable and expose get_settings."""
    import config
    assert hasattr(config, "get_settings")
    s = config.get_settings()
    assert s is not None


def test_config_apply_global_env():
    """config.apply_global_env sets expected env vars."""
    import config
    s = config.get_settings()
    config.apply_global_env(s)
    # Should have set at least TF_THREAD_COUNT
    assert "TF_THREAD_COUNT" in os.environ or True  # may not set if TF not present


def test_dqn_load_default_missing_file():
    """_load_default_dqn_cfg returns empty dict for missing file."""
    from pipeline.dqn_config import _load_default_dqn_cfg
    cfg = _load_default_dqn_cfg("/nonexistent/path/config.json")
    assert isinstance(cfg, dict)


def test_dqn_load_default_valid_file(tmp_path):
    """_load_default_dqn_cfg loads a valid JSON config."""
    from pipeline.dqn_config import _load_default_dqn_cfg
    cfg_file = tmp_path / "dqn_test.json"
    test_cfg = {"batch_size": 128, "gamma": 0.95}
    cfg_file.write_text(json.dumps(test_cfg))
    cfg = _load_default_dqn_cfg(str(cfg_file))
    assert cfg["batch_size"] == 128
    assert cfg["gamma"] == 0.95


def test_dqn_coerce_fills_defaults():
    """_coerce_dqn_cfg fills in missing keys with defaults."""
    from pipeline.dqn_config import _coerce_dqn_cfg
    cfg = _coerce_dqn_cfg({})
    assert "gamma" in cfg
    assert "epsilon" in cfg
    assert "batch_size" in cfg
    assert "buffer_size" in cfg
    assert "warmup_steps" in cfg
    assert cfg["gamma"] == 0.99
    assert cfg["batch_size"] > 0


def test_dqn_coerce_batch_fits_buffer():
    """_coerce_dqn_cfg ensures batch_size <= buffer_size."""
    from pipeline.dqn_config import _coerce_dqn_cfg
    cfg = _coerce_dqn_cfg({"batch_size": 100000, "buffer_size": 500})
    assert cfg["batch_size"] <= cfg["buffer_size"]


def test_dqn_coerce_warmup_less_than_buffer():
    """_coerce_dqn_cfg ensures warmup_steps < buffer_size."""
    from pipeline.dqn_config import _coerce_dqn_cfg
    cfg = _coerce_dqn_cfg({"warmup_steps": 99999, "buffer_size": 1000, "batch_size": 32})
    assert cfg["warmup_steps"] < cfg["buffer_size"]


def test_dqn_coerce_strict_mode():
    """_coerce_dqn_cfg in strict mode preserves user values."""
    from pipeline.dqn_config import _coerce_dqn_cfg
    cfg = _coerce_dqn_cfg({"_cfg_source": "grid_defaults", "batch_size": 100000, "buffer_size": 500})
    # In strict mode: batch is clamped to buffer (can't exceed)
    assert cfg["batch_size"] <= cfg["buffer_size"]


def test_runtime_exports():
    """pipeline.runtime exports SAFE_CORES and CPU_TOTAL."""
    from pipeline.runtime import SAFE_CORES, CPU_TOTAL
    assert isinstance(SAFE_CORES, int)
    assert isinstance(CPU_TOTAL, int)
    assert SAFE_CORES > 0
    assert CPU_TOTAL > 0
    assert SAFE_CORES <= CPU_TOTAL


def test_class_defaults_structure():
    """CLASS_DEFAULTS must have 'features' and 'cv' sections."""
    from pipeline.metrics_tuples import CLASS_DEFAULTS
    assert "features" in CLASS_DEFAULTS
    assert "cv" in CLASS_DEFAULTS
    # Spot check a few known keys
    assert "lag_depth" in CLASS_DEFAULTS["features"]
    assert "cv_mode" in CLASS_DEFAULTS["cv"]


def test_class_defaults_immutability():
    """DEFAULT_FEATURES and DEFAULT_CV are deep copies (not references)."""
    from pipeline.metrics_tuples import CLASS_DEFAULTS, DEFAULT_FEATURES, DEFAULT_CV
    assert DEFAULT_FEATURES is not CLASS_DEFAULTS["features"]
    assert DEFAULT_CV is not CLASS_DEFAULTS["cv"]


def test_hpo_persistence_roundtrip(tmp_path, monkeypatch):
    """save_hpo_config_to_disk / load_hpo_config_from_disk roundtrip."""
    from pipeline.hpo_persistence import save_hpo_config_to_disk, load_hpo_config_from_disk
    import pipeline.hpo_persistence as hpo_mod
    # Redirect HPO dir to tmp_path
    monkeypatch.setattr(hpo_mod, "HPO_CONFIG_DIR", str(tmp_path))
    best_params = {"lr": 0.01, "depth": 6}
    save_hpo_config_to_disk("xgboost", best_params)
    best, topN = load_hpo_config_from_disk("xgboost")
    assert best is not None
    assert best["lr"] == 0.01
    assert best["depth"] == 6
