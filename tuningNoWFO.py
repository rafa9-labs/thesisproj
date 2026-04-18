"""Compatibility shim — all symbols re-exported from pipeline.tuning."""
from pipeline.tuning.helpers import *
from pipeline.tuning.sampler import sample_param_set, _coerce_ensemble_lags
from pipeline.tuning.refit import (
    refit_cnn_with_overrides,
    refit_lstm_with_overrides,
    refit_transformer_with_overrides,
    refit_ensemble_cnn_lstm_xgb_with_overrides,
    refit_ensemble_adaptive_regime_with_overrides,
    _evaluate_original_no_refit,
    _select_better_result,
    final_refit_if_deep,
    _aggressive_free,
    _assert_free_ram,
)
from pipeline.tuning.objective import optuna_objective
from pipeline.tuning.runner import run_optuna_tuning