"""Unit tests for CV geometry optimizations (A: adaptive K, B: val floor, C: suggest params)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ════════════════════════════════════════════════════════════════════
# Adaptive K logic (mirrors run_mixin.py:662)
# ════════════════════════════════════════════════════════════════════

def _adaptive_k(total_len: int, cv_blocks_override: int = None) -> int:
    if cv_blocks_override is not None:
        return cv_blocks_override
    return max(3, min(10, total_len // 4000))


def _val_window_floor(total_len: int) -> int:
    return min(120, max(30, int(round(0.03 * total_len))))


def _val_window(total_len: int, val_frac: float = 0.09) -> int:
    return max(_val_window_floor(total_len), int(round(val_frac * total_len)))


def _max_fit_k(usable: int, val_window: int) -> int:
    return max(1, int(usable // max(1, val_window)) + 1)


class TestAdaptiveK:
    def test_small_data_defaults_to_3(self):
        assert _adaptive_k(500) == 3
        assert _adaptive_k(4000) == 3
        assert _adaptive_k(7999) == 3

    def test_medium_data_proportional(self):
        # 12,000 bars → 3,  20,000 bars → 5,  28,000 bars → 7
        assert _adaptive_k(12000) == 3
        assert _adaptive_k(16000) == 4
        assert _adaptive_k(20000) == 5
        assert _adaptive_k(28000) == 7

    def test_large_data_capped_at_10(self):
        assert _adaptive_k(40000) == 10
        assert _adaptive_k(100000) == 10
        assert _adaptive_k(500000) == 10

    def test_explicit_override_respected(self):
        assert _adaptive_k(20000, cv_blocks_override=7) == 7
        assert _adaptive_k(500, cv_blocks_override=10) == 10
        assert _adaptive_k(100000, cv_blocks_override=2) == 2

    def test_minimum_always_at_least_3(self):
        for n in [30, 100, 500, 1000, 2000, 3999]:
            assert _adaptive_k(n) == 3


class TestValWindowFloor:
    def test_small_data_soft_floor(self):
        assert _val_window_floor(500) == 30
        assert _val_window_floor(1000) == 30
        assert _val_window_floor(2000) == 60  # 0.03 * 2000 = 60
        assert _val_window_floor(4000) == 120  # min(120, 0.03 * 4000 = 120)

    def test_large_data_floor_capped_at_120(self):
        assert _val_window_floor(5000) == 120
        assert _val_window_floor(50000) == 120
        assert _val_window_floor(100000) == 120  # min(120, 3000) = 120

    def test_val_window_respects_val_frac(self):
        assert _val_window(50000, val_frac=0.09) == max(120, 4500)  # 4500
        assert _val_window(2000, val_frac=0.09) == max(60, 180)  # 180

    def test_tiny_dataset_keeps_minimum_val_window(self):
        assert _val_window(500, val_frac=0.09) == max(30, 45)  # 45
        assert _val_window(200, val_frac=0.09) == max(30, 18)  # 30 (floor wins)


class TestFitBasedKCap:
    def test_plenty_of_space_no_cap(self):
        assert _max_fit_k(usable=30000, val_window=2000) == 16
        # K is capped externally by the max(3, min(10,...)) logic

    def test_tight_space_caps_k(self):
        assert _max_fit_k(usable=6000, val_window=5000) == 2
        assert _max_fit_k(usable=6000, val_window=5500) == 2

    def test_barely_any_space(self):
        assert _max_fit_k(usable=1000, val_window=2000) == 1
        assert _max_fit_k(usable=300, val_window=500) == 1


# ════════════════════════════════════════════════════════════════════
# CV params in suggest engine
# ════════════════════════════════════════════════════════════════════

class TestCVParamsInSuggest:
    def test_cv_search_space_defined(self):
        from config import CV_SEARCH_SPACE
        assert "cv_blocks" in CV_SEARCH_SPACE
        assert "cv_val_frac" in CV_SEARCH_SPACE
        assert isinstance(CV_SEARCH_SPACE["cv_blocks"], list)
        assert isinstance(CV_SEARCH_SPACE["cv_val_frac"], list)

    def test_cv_blocks_values_reasonable(self):
        from config import CV_SEARCH_SPACE
        vals = CV_SEARCH_SPACE["cv_blocks"]
        assert all(v >= 3 for v in vals)
        assert all(v <= 10 for v in vals)

    def test_cv_val_frac_values_reasonable(self):
        from config import CV_SEARCH_SPACE
        vals = CV_SEARCH_SPACE["cv_val_frac"]
        assert all(0.03 <= v <= 0.20 for v in vals)

    def test_suggest_sampless_cv_params(self):
        import optuna
        from config import CV_SEARCH_SPACE
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            params = {}
            for key, spec in CV_SEARCH_SPACE.items():
                if isinstance(spec, list):
                    params[key] = trial.suggest_categorical(key, spec)
            return params.get("cv_blocks", 0) + params.get("cv_val_frac", 0)

        study.optimize(objective, n_trials=20)
        trial = study.best_trial
        assert trial.params["cv_blocks"] in CV_SEARCH_SPACE["cv_blocks"]
        assert trial.params["cv_val_frac"] in CV_SEARCH_SPACE["cv_val_frac"]

    def test_cv_param_keys_are_not_prefixed(self):
        import optuna
        from config import CV_SEARCH_SPACE
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            params = {}
            for key, spec in CV_SEARCH_SPACE.items():
                if isinstance(spec, list):
                    params[key] = trial.suggest_categorical(key, spec)
            # Verify the keys don't have model prefixes
            assert "cv_blocks" in params
            assert "xgb_cv_blocks" not in params
            assert "logit_cv_blocks" not in params
            return 0.0

        study.optimize(objective, n_trials=5)


# ════════════════════════════════════════════════════════════════════
# D: Per-family deep CV caps
# ════════════════════════════════════════════════════════════════════

class TestPerFamilyDeepCVCaps:
    def _simulate_caps(self, model_type: str) -> dict:
        """Simulate the setdefault logic from run_mixin.py for a model type."""
        cfg = {}
        cfg["deep_cv_batch_size"] = 256
        if model_type == "transformer":
            cfg.setdefault("deep_cv_max_epochs", 6)
            cfg.setdefault("deep_cv_patience", 4)
            cfg.setdefault("transformer_use_early_stopping", True)
            cfg.setdefault("transformer_patience", 4)
        elif model_type in ("gru", "gru_lstm"):
            cfg.setdefault("deep_cv_max_epochs", 12)
            cfg.setdefault("deep_cv_patience", 8)
            cfg.setdefault("gru_use_early_stopping", True)
            cfg.setdefault("gru_lstm_use_early_stopping", True)
            cfg.setdefault(f"{model_type}_patience", 8)
        elif model_type == "cnn":
            cfg.setdefault("deep_cv_max_epochs", 10)
            cfg.setdefault("deep_cv_patience", 6)
            cfg.setdefault("cnn_use_early_stopping", True)
            cfg.setdefault("cnn_patience", 6)
        elif model_type == "lstm":
            cfg.setdefault("deep_cv_max_epochs", 10)
            cfg.setdefault("deep_cv_patience", 7)
            cfg.setdefault("lstm_use_early_stopping", True)
            cfg.setdefault("lstm_patience", 7)
        else:
            cfg.setdefault("deep_cv_max_epochs", 8)
            cfg.setdefault("deep_cv_patience", 6)
        return cfg

    def test_transformer_gets_tighter_caps(self):
        cfg = self._simulate_caps("transformer")
        assert cfg["deep_cv_max_epochs"] == 6
        assert cfg["deep_cv_patience"] == 4
        assert cfg["transformer_patience"] == 4

    def test_gru_gets_generous_caps(self):
        cfg = self._simulate_caps("gru")
        assert cfg["deep_cv_max_epochs"] == 12
        assert cfg["deep_cv_patience"] == 8

    def test_lstm_caps_in_between(self):
        cfg = self._simulate_caps("lstm")
        assert cfg["deep_cv_max_epochs"] == 10
        assert cfg["deep_cv_patience"] == 7
        assert cfg["lstm_patience"] == 7

    def test_ensemble_keeps_defaults(self):
        cfg = self._simulate_caps("ensemble_adaptive_regime")
        assert cfg["deep_cv_max_epochs"] == 8
        assert cfg["deep_cv_patience"] == 6

    def test_gru_lstm_same_as_gru(self):
        cfg = self._simulate_caps("gru_lstm")
        assert cfg["deep_cv_max_epochs"] == 12
        assert cfg["deep_cv_patience"] == 8
        assert cfg["gru_lstm_patience"] == 8


# ════════════════════════════════════════════════════════════════════
# E: Auto two-phase HPO for models with 4+ tunable params
# ════════════════════════════════════════════════════════════════════

class TestAutoTwoPhase:
    def _count_tunable(self, model_type: str) -> int:
        from config import SEARCH_SPACE
        return sum(1 for v in SEARCH_SPACE.get(model_type, {}).values()
                   if isinstance(v, (list, tuple)))

    def _auto_two_phase(self, model_type: str, explicit: bool = None) -> bool:
        if explicit is not None:
            return explicit
        # v3.0 derived SEARCH_SPACE: deep models have 3 Tier-1 params each.
        return self._count_tunable(model_type) >= 3

    def test_two_phase_auto_enabled_for_deep(self):
        for m in ("cnn", "lstm", "transformer", "gru", "gru_lstm"):
            assert self._auto_two_phase(m) is True, (
                f"{m} has {self._count_tunable(m)} tunable params, expected auto True")

    def test_two_phase_disabled_for_few_params(self):
        assert self._auto_two_phase("logistic") is False
        assert self._auto_two_phase("svm") is False
        assert self._auto_two_phase("meta_ensemble") is False

    def test_two_phase_explicit_override(self):
        assert self._auto_two_phase("lstm", explicit=False) is False
        assert self._auto_two_phase("logistic", explicit=True) is True


# ════════════════════════════════════════════════════════════════════
# F: Early-abort hopeless trials
# ════════════════════════════════════════════════════════════════════

class TestEarlyAbort:
    def _should_abort(self, sharpe_scores: list, folds: int = 3, thr: float = -1.0,
                      relax: float = 1.0) -> bool:
        import numpy as np
        if relax <= 0.0:
            return False  # pruning disabled entirely
        processed = len(sharpe_scores)
        arr = np.asarray(sharpe_scores, dtype=float)
        k_valid = int(np.isfinite(arr).sum())
        if processed < folds or k_valid < folds:
            return False
        valid = arr[np.isfinite(arr)]
        if len(valid) < folds:
            return False
        mean_sharpe = float(np.mean(valid))
        return mean_sharpe < thr * relax

    def test_early_abort_triggers_on_hopeless(self):
        assert self._should_abort([-2.0, -2.5, -1.8]) is True

    def test_early_abort_passes_on_marginal(self):
        assert self._should_abort([-0.5, 0.2, -0.3]) is False

    def test_early_abort_passes_on_positive(self):
        assert self._should_abort([1.2, 0.8, 1.5]) is False

    def test_early_abort_respects_prune_relax(self):
        # relax=0.0 disables all pruning
        assert self._should_abort([-5.0, -4.0, -6.0], relax=0.0) is False

    def test_early_abort_needs_minimum_folds(self):
        # Only 2 folds → not enough to trigger regardless of scores
        assert self._should_abort([-5.0, -5.0], folds=3) is False

    def test_early_abort_ignores_nan_scores(self):
        # NaN scores don't count toward the minimum
        assert self._should_abort([-3.0, float('nan'), -3.0]) is False
