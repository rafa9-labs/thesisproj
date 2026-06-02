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
