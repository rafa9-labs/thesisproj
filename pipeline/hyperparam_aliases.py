"""HYPERPARAM_ALIASES — standalone module for frontend-to-backend key mapping.

Shared by api/tasks.py (for _convert_model_overrides) and
pipeline/tuning/fixed_config.py (for UserFixedConfig.from_api_overrides).
No heavy dependencies so both can import without ImportError.
"""

from typing import Dict

HYPERPARAM_ALIASES: Dict[str, Dict[str, str]] = {
    "logistic": {
        "C": "logit_C",
        "solver": "logit_solver",
        "penalty": "logit_penalty",
        "max_iter": "logit_max_iter",
        "tol": "logit_tol",
        "class_weight": "logit_class_weight",
    },
    "xgboost": {
        "n_estimators": "xgb_n_estimators",
        "max_depth": "xgb_max_depth",
        "learning_rate": "xgb_learning_rate",
        "subsample": "xgb_subsample",
        "colsample_bytree": "xgb_colsample_bytree",
    },
    "svm": {
        "C": "svm_C",
        "gamma": "svm_gamma",
        "kernel": "svm_kernel",
        "class_weight": "svm_class_weight",
    },
    "random_forest": {
        "n_estimators": "rf_n_estimators",
        "max_depth": "rf_max_depth",
        "min_samples_leaf": "rf_min_samples_leaf",
        "max_features": "rf_max_features",
    },
    "decision_tree": {
        "max_depth": "dt_max_depth",
        "min_samples_leaf": "dt_min_samples_leaf",
        "max_features": "dt_max_features",
        "ccp_alpha": "dt_ccp_alpha",
    },
    "lightgbm": {
        "n_estimators": "lgbm_n_estimators",
        "max_depth": "lgbm_max_depth",
        "num_leaves": "lgbm_num_leaves",
        "learning_rate": "lgbm_learning_rate",
        "subsample": "lgbm_subsample",
        "colsample_bytree": "lgbm_colsample_bytree",
        "reg_lambda": "lgbm_reg_lambda",
    },
    "catboost": {
        "iterations": "cb_iterations",
        "depth": "cb_depth",
        "learning_rate": "cb_learning_rate",
        "subsample": "cb_subsample",
        "l2_leaf_reg": "cb_l2_leaf_reg",
    },
    "lstm": {
        "units": "lstm_units",
        "num_layers": "lstm_num_layers",
        "dropout_rate": "lstm_dropout_rate",
        "learning_rate": "lstm_learning_rate",
    },
    "cnn": {
        "filters1": "cnn_filters1",
        "filters2": "cnn_filters2",
        "kernel_size": "cnn_kernel_size",
        "learning_rate": "cnn_learning_rate",
    },
    "transformer": {
        "d_model": "transformer_d_model",
        "num_heads": "transformer_num_heads",
        "dropout_rate": "transformer_dropout_rate",
        "learning_rate": "transformer_learning_rate",
    },
    "gru": {
        "units": "gru_units",
        "num_layers": "gru_num_layers",
        "dropout_rate": "gru_dropout_rate",
        "learning_rate": "gru_learning_rate",
    },
    "gru_lstm": {
        "gru_units": "gru_lstm_gru_units",
        "lstm_units": "gru_lstm_lstm_units",
        "dropout_rate": "gru_lstm_dropout_rate",
        "learning_rate": "gru_lstm_learning_rate",
    },
    "ensemble_adaptive_regime": {
        "lstm_learning_rate": "lstm_learning_rate",
        "rf_max_depth": "rf_max_depth",
        "logit_C": "logit_C",
        "lstm_units": "lstm_units",
        "rf_n_estimators": "rf_n_estimators",
        "adx_thresh": "adx_thresh",
        "vol_thresh": "vol_thresh",
        "logit_solver": "logit_solver",
    },
    "ensemble_cnn_lstm_xgboost": {
        "cnn_filters1": "cnn_filters1",
        "cnn_filters2": "cnn_filters2",
        "cnn_kernel_size": "cnn_kernel_size",
        "cnn_learning_rate": "cnn_learning_rate",
        "lstm_units": "lstm_units",
        "lstm_learning_rate": "lstm_learning_rate",
        "xgb_n_estimators": "xgb_n_estimators",
        "xgb_learning_rate": "xgb_learning_rate",
        "xgb_max_depth": "xgb_max_depth",
    },
    "meta_ensemble": {
        "meta_combination_method": "meta_combination_method",
    },
    "stacking_ensemble": {
        "stack_cv": "stack_cv",
        "stack_method": "stack_method",
    },
}
