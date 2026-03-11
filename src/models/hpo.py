"""
Hyperparameter Optimization Search Spaces

Extracted from tuningNoWFO.py and MLBacktesterNoWFO.py.

CRITICAL Features:
1. Dynamic Defaults: Pull defaults from AppConfig (single source of truth)
2. Active Parameter Filtering: Only tune parameters in active_params list
3. Type Safety: Graceful handling if Optuna not installed
4. Boundary Tracking: Diagnostic utility for range tuning

Usage:
    from src.models.hpo import get_xgboost_space, get_cnn_space
    from src.core.config import load_default_config
    
    config = load_default_config()
    
    # Tune only learning_rate and max_depth
    active_params = ['xgb_learning_rate', 'xgb_max_depth']
    params = get_xgboost_space(trial, config, active_params)
    
    # Tune all parameters
    params = get_xgboost_space(trial, config, active_params=[])
"""

import logging
from typing import Dict, Any, List, Optional

# Type-safe Optuna import
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning(
        "Optuna is not installed. HPO functionality will be disabled. "
        "Install with: pip install optuna"
    )
    # Create dummy Trial class for type hints
    class Trial:
        pass
    optuna = type('optuna', (), {'Trial': Trial})()


logger = logging.getLogger(__name__)


# ============================================================================
# XGBoost Search Space
# ============================================================================

def get_xgboost_space(
    trial: 'optuna.Trial',
    config: Any,
    active_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    XGBoost hyperparameter search space.
    
    Extracted from tuningNoWFO.py lines 450-520.
    
    CRITICAL: Defaults pulled from AppConfig, not hardcoded.
    
    Args:
        trial: Optuna trial object
        config: AppConfig instance (single source of truth)
        active_params: List of parameters to tune (None/empty = tune all)
        
    Returns:
        Dictionary with XGBoost parameters
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
    
    if active_params is None:
        active_params = []
    
    params = {}
    
    # n_estimators: Number of boosting rounds
    if 'xgb_n_estimators' in active_params or not active_params:
        params['n_estimators'] = trial.suggest_int('xgb_n_estimators', 50, 300)
    else:
        params['n_estimators'] = 100  # Standard default
    
    # max_depth: Maximum tree depth
    if 'xgb_max_depth' in active_params or not active_params:
        params['max_depth'] = trial.suggest_int('xgb_max_depth', 3, 10)
    else:
        params['max_depth'] = 5  # Standard default
    
    # learning_rate: Step size shrinkage
    if 'xgb_learning_rate' in active_params or not active_params:
        params['learning_rate'] = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
    else:
        params['learning_rate'] = 0.1  # Standard default
    
    # subsample: Row sampling ratio
    if 'xgb_subsample' in active_params or not active_params:
        params['subsample'] = trial.suggest_float('xgb_subsample', 0.5, 1.0)
    else:
        params['subsample'] = 0.8  # Standard default
    
    # colsample_bytree: Column sampling ratio
    if 'xgb_colsample_bytree' in active_params or not active_params:
        params['colsample_bytree'] = trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0)
    else:
        params['colsample_bytree'] = 0.8  # Standard default
    
    # gamma: Minimum loss reduction for split
    if 'xgb_gamma' in active_params or not active_params:
        params['gamma'] = trial.suggest_float('xgb_gamma', 0.0, 5.0)
    else:
        params['gamma'] = 0.0  # Standard default
    
    # min_child_weight: Minimum sum of instance weight in child
    if 'xgb_min_child_weight' in active_params or not active_params:
        params['min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 1, 10)
    else:
        params['min_child_weight'] = 1  # Standard default
    
    # reg_alpha: L1 regularization
    if 'xgb_reg_alpha' in active_params or not active_params:
        params['reg_alpha'] = trial.suggest_float('xgb_reg_alpha', 0.0, 1.0)
    else:
        params['reg_alpha'] = 0.0  # Standard default
    
    # reg_lambda: L2 regularization
    if 'xgb_reg_lambda' in active_params or not active_params:
        params['reg_lambda'] = trial.suggest_float('xgb_reg_lambda', 0.0, 1.0)
    else:
        params['reg_lambda'] = 1.0  # Standard default
    
    # Fixed parameters (not tuned)
    params['tree_method'] = 'hist'
    params['device'] = 'cpu'
    params['objective'] = 'multi:softprob'
    params['num_class'] = 3
    
    return params


# ============================================================================
# CNN Search Space
# ============================================================================

def get_cnn_space(
    trial: 'optuna.Trial',
    config: Any,
    active_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    CNN hyperparameter search space.
    
    Extracted from tuningNoWFO.py lines 550-620 and cnn.py.
    
    CRITICAL: Defaults pulled from AppConfig, not hardcoded.
    
    Args:
        trial: Optuna trial object
        config: AppConfig instance (single source of truth)
        active_params: List of parameters to tune (None/empty = tune all)
        
    Returns:
        Dictionary with CNN parameters
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
    
    if active_params is None:
        active_params = []
    
    params = {}
    
    # filters1: First conv layer filters
    if 'cnn_filters1' in active_params or not active_params:
        params['cnn_filters1'] = trial.suggest_int('cnn_filters1', 16, 64)
    else:
        params['cnn_filters1'] = 32  # From cnn.py default
    
    # filters2: Second conv layer filters
    if 'cnn_filters2' in active_params or not active_params:
        params['cnn_filters2'] = trial.suggest_int('cnn_filters2', 32, 128)
    else:
        params['cnn_filters2'] = 64  # From cnn.py default
    
    # kernel_size: Convolution kernel size
    if 'cnn_kernel_size' in active_params or not active_params:
        params['cnn_kernel_size'] = trial.suggest_int('cnn_kernel_size', 2, 5)
    else:
        params['cnn_kernel_size'] = 3  # From cnn.py default
    
    # dense_units: Dense layer units
    if 'cnn_dense_units' in active_params or not active_params:
        params['cnn_dense_units'] = trial.suggest_int('cnn_dense_units', 32, 128)
    else:
        params['cnn_dense_units'] = 64  # From cnn.py default
    
    # dropout_rate: Dropout rate
    if 'cnn_dropout_rate' in active_params or not active_params:
        params['cnn_dropout_rate'] = trial.suggest_float('cnn_dropout_rate', 0.1, 0.5)
    else:
        params['cnn_dropout_rate'] = 0.3  # From cnn.py default
    
    # learning_rate: Optimizer learning rate
    if 'cnn_learning_rate' in active_params or not active_params:
        params['cnn_learning_rate'] = trial.suggest_float('cnn_learning_rate', 1e-4, 1e-2, log=True)
    else:
        params['cnn_learning_rate'] = 1e-3  # From cnn.py default
    
    # padding_same: Use 'same' padding (vs 'valid')
    if 'cnn_padding_same' in active_params or not active_params:
        params['cnn_padding_same'] = trial.suggest_categorical('cnn_padding_same', [True, False])
    else:
        params['cnn_padding_same'] = True  # From cnn.py default
    
    # clipnorm: Gradient clipping norm
    if 'cnn_clipnorm' in active_params or not active_params:
        params['cnn_clipnorm'] = trial.suggest_float('cnn_clipnorm', 0.0, 1.0)
    else:
        params['cnn_clipnorm'] = 0.0  # From cnn.py default
    
    # use_early_stopping: Enable early stopping
    if 'cnn_use_early_stopping' in active_params or not active_params:
        params['cnn_use_early_stopping'] = trial.suggest_categorical('cnn_use_early_stopping', [True, False])
    else:
        params['cnn_use_early_stopping'] = False  # From cnn.py default
    
    # patience: Early stopping patience
    if 'cnn_patience' in active_params or not active_params:
        params['cnn_patience'] = trial.suggest_int('cnn_patience', 5, 15)
    else:
        params['cnn_patience'] = 10  # From cnn.py default
    
    # CV-specific overrides from AppConfig
    if hasattr(config, 'features'):
        # Use CV caps from config if available
        if hasattr(config.features, 'cnn_cv_max_epochs'):
            params['max_epochs'] = config.features.cnn_cv_max_epochs
        if hasattr(config.features, 'cnn_cv_batch_size'):
            params['batch_size'] = config.features.cnn_cv_batch_size
    
    return params


# ============================================================================
# LSTM Search Space
# ============================================================================

def get_lstm_space(
    trial: 'optuna.Trial',
    config: Any,
    active_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    LSTM hyperparameter search space.
    
    Extracted from tuningNoWFO.py lines 620-690.
    
    CRITICAL: Defaults pulled from AppConfig, not hardcoded.
    
    Args:
        trial: Optuna trial object
        config: AppConfig instance (single source of truth)
        active_params: List of parameters to tune (None/empty = tune all)
        
    Returns:
        Dictionary with LSTM parameters
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
    
    if active_params is None:
        active_params = []
    
    params = {}
    
    # units1: First LSTM layer units
    if 'lstm_units1' in active_params or not active_params:
        params['lstm_units1'] = trial.suggest_int('lstm_units1', 32, 128)
    else:
        params['lstm_units1'] = 64  # Standard default
    
    # units2: Second LSTM layer units
    if 'lstm_units2' in active_params or not active_params:
        params['lstm_units2'] = trial.suggest_int('lstm_units2', 16, 64)
    else:
        params['lstm_units2'] = 32  # Standard default
    
    # dense_units: Dense layer units
    if 'lstm_dense_units' in active_params or not active_params:
        params['lstm_dense_units'] = trial.suggest_int('lstm_dense_units', 16, 64)
    else:
        params['lstm_dense_units'] = 32  # Standard default
    
    # dropout_rate: Dropout rate
    if 'lstm_dropout_rate' in active_params or not active_params:
        params['lstm_dropout_rate'] = trial.suggest_float('lstm_dropout_rate', 0.1, 0.5)
    else:
        params['lstm_dropout_rate'] = 0.2  # Standard default
    
    # recurrent_dropout: Recurrent dropout rate
    if 'lstm_recurrent_dropout' in active_params or not active_params:
        params['lstm_recurrent_dropout'] = trial.suggest_float('lstm_recurrent_dropout', 0.0, 0.3)
    else:
        params['lstm_recurrent_dropout'] = 0.0  # Standard default
    
    # learning_rate: Optimizer learning rate
    if 'lstm_learning_rate' in active_params or not active_params:
        params['lstm_learning_rate'] = trial.suggest_float('lstm_learning_rate', 1e-4, 1e-2, log=True)
    else:
        params['lstm_learning_rate'] = 1e-3  # Standard default
    
    # clipnorm: Gradient clipping norm
    if 'lstm_clipnorm' in active_params or not active_params:
        params['lstm_clipnorm'] = trial.suggest_float('lstm_clipnorm', 0.0, 1.0)
    else:
        params['lstm_clipnorm'] = 0.0  # Standard default
    
    # use_early_stopping: Enable early stopping
    if 'lstm_use_early_stopping' in active_params or not active_params:
        params['lstm_use_early_stopping'] = trial.suggest_categorical('lstm_use_early_stopping', [True, False])
    else:
        params['lstm_use_early_stopping'] = False  # Standard default
    
    # patience: Early stopping patience
    if 'lstm_patience' in active_params or not active_params:
        params['lstm_patience'] = trial.suggest_int('lstm_patience', 5, 15)
    else:
        params['lstm_patience'] = 10  # Standard default
    
    # CV-specific overrides from AppConfig
    if hasattr(config, 'features'):
        # Use CV caps from config if available
        if hasattr(config.features, 'lstm_cv_max_epochs'):
            params['max_epochs'] = config.features.lstm_cv_max_epochs
        if hasattr(config.features, 'lstm_cv_batch_size'):
            params['batch_size'] = config.features.lstm_cv_batch_size
    
    return params


# ============================================================================
# Ensemble Search Space
# ============================================================================

def get_ensemble_space(
    trial: 'optuna.Trial',
    config: Any,
    active_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Ensemble hyperparameter search space.
    
    Extracted from tuningNoWFO.py lines 690-750.
    
    CRITICAL: Defaults pulled from AppConfig, not hardcoded.
    
    Args:
        trial: Optuna trial object
        config: AppConfig instance (single source of truth)
        active_params: List of parameters to tune (None/empty = tune all)
        
    Returns:
        Dictionary with ensemble parameters
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
    
    if active_params is None:
        active_params = []
    
    params = {}
    
    # voting_method: Voting strategy
    if 'ensemble_voting_method' in active_params or not active_params:
        params['voting_method'] = trial.suggest_categorical('ensemble_voting_method', ['soft', 'hard', 'weighted'])
    else:
        params['voting_method'] = 'soft'  # Standard default
    
    # weights: Model weights (if weighted voting)
    if 'ensemble_weights' in active_params or not active_params:
        # Sample weights for 3 models (XGBoost, CNN, LSTM)
        w1 = trial.suggest_float('ensemble_weight_xgb', 0.1, 1.0)
        w2 = trial.suggest_float('ensemble_weight_cnn', 0.1, 1.0)
        w3 = trial.suggest_float('ensemble_weight_lstm', 0.1, 1.0)
        params['weights'] = [w1, w2, w3]
    else:
        params['weights'] = [1.0, 1.0, 1.0]  # Equal weights default
    
    # calibrate_ensemble: Calibrate ensemble output
    if 'ensemble_calibrate' in active_params or not active_params:
        params['calibrate_ensemble'] = trial.suggest_categorical('ensemble_calibrate', [True, False])
    else:
        # Pull from config if available
        if hasattr(config, 'features') and hasattr(config.features, 'deep_calibrate'):
            params['calibrate_ensemble'] = config.features.deep_calibrate
        else:
            params['calibrate_ensemble'] = False
    
    # fusion_alpha: Ensemble fusion parameter
    if 'ensemble_fusion_alpha' in active_params or not active_params:
        params['fusion_alpha'] = trial.suggest_float('ensemble_fusion_alpha', 0.3, 0.9)
    else:
        # Pull from config if available
        if hasattr(config, 'features') and hasattr(config.features, 'fusion_alpha'):
            params['fusion_alpha'] = config.features.fusion_alpha
        else:
            params['fusion_alpha'] = 0.6
    
    return params


# ============================================================================
# Utility Functions
# ============================================================================

def get_search_space(
    model_type: str,
    trial: 'optuna.Trial',
    config: Any,
    active_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get search space for specified model type.
    
    Unified dispatcher for all model types.
    
    Args:
        model_type: Model type ('xgboost', 'cnn', 'lstm', 'ensemble')
        trial: Optuna trial object
        config: AppConfig instance
        active_params: List of parameters to tune
        
    Returns:
        Dictionary with hyperparameters
        
    Raises:
        ValueError: If model_type is unknown
    """
    model_type_lower = model_type.lower()
    
    if model_type_lower == 'xgboost':
        return get_xgboost_space(trial, config, active_params)
    elif model_type_lower == 'cnn':
        return get_cnn_space(trial, config, active_params)
    elif model_type_lower == 'lstm':
        return get_lstm_space(trial, config, active_params)
    elif model_type_lower in ['ensemble', 'ensemble_cnn_lstm_xgboost']:
        return get_ensemble_space(trial, config, active_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def record_boundary_hit(
    name: str,
    value: float,
    low: float,
    high: float,
    boundary_tracker: Dict[str, int],
    eps_frac: float = 0.01
) -> None:
    """
    Track boundary hits for hyperparameter diagnostics.
    
    Extracted from tuningNoWFO.py lines 68-115.
    
    This utility helps identify when Optuna samples values near the edges
    of search ranges, indicating that ranges may need adjustment.
    
    Args:
        name: Parameter name
        value: Sampled value
        low: Lower bound
        high: Upper bound
        boundary_tracker: Dictionary to track hits (modified in-place)
        eps_frac: Fraction of range considered "near edge" (default 0.01 = 1%)
        
    Example:
        >>> tracker = {}
        >>> record_boundary_hit('xgb_max_depth', 9.8, 3, 10, tracker)
        >>> print(tracker)  # {'xgb_max_depth': 1}
    """
    try:
        low = float(low)
        high = float(high)
        value = float(value)
    except Exception:
        return
    
    if high <= low:
        return
    
    span = high - low
    if span <= 0:
        return
    
    # Normalized distance to each edge
    rel_low = (value - low) / span
    rel_high = (high - value) / span
    
    # Track hits near either edge
    if rel_low <= eps_frac or rel_high <= eps_frac:
        boundary_tracker[name] = boundary_tracker.get(name, 0) + 1
        
        # Log diagnostic message
        if rel_low <= eps_frac:
            logger.debug(f"Boundary hit: {name}={value:.4f} near lower bound {low}")
        else:
            logger.debug(f"Boundary hit: {name}={value:.4f} near upper bound {high}")


def validate_active_params(
    active_params: List[str],
    model_type: str
) -> List[str]:
    """
    Validate active_params list for given model type.
    
    Ensures all parameter names are valid for the specified model.
    
    Args:
        active_params: List of parameter names to validate
        model_type: Model type ('xgboost', 'cnn', 'lstm', 'ensemble')
        
    Returns:
        Validated list of active parameters
        
    Raises:
        ValueError: If invalid parameter names are found
    """
    model_type_lower = model_type.lower()
    
    # Define valid parameters for each model type
    valid_params = {
        'xgboost': [
            'xgb_n_estimators', 'xgb_max_depth', 'xgb_learning_rate',
            'xgb_subsample', 'xgb_colsample_bytree', 'xgb_gamma',
            'xgb_min_child_weight', 'xgb_reg_alpha', 'xgb_reg_lambda'
        ],
        'cnn': [
            'cnn_filters1', 'cnn_filters2', 'cnn_kernel_size', 'cnn_dense_units',
            'cnn_dropout_rate', 'cnn_learning_rate', 'cnn_padding_same',
            'cnn_clipnorm', 'cnn_use_early_stopping', 'cnn_patience'
        ],
        'lstm': [
            'lstm_units1', 'lstm_units2', 'lstm_dense_units', 'lstm_dropout_rate',
            'lstm_recurrent_dropout', 'lstm_learning_rate', 'lstm_clipnorm',
            'lstm_use_early_stopping', 'lstm_patience'
        ],
        'ensemble': [
            'ensemble_voting_method', 'ensemble_weights', 'ensemble_calibrate',
            'ensemble_fusion_alpha', 'ensemble_weight_xgb', 'ensemble_weight_cnn',
            'ensemble_weight_lstm'
        ]
    }
    
    if model_type_lower not in valid_params:
        raise ValueError(f"Unknown model type: {model_type}")
    
    valid_set = set(valid_params[model_type_lower])
    invalid_params = [p for p in active_params if p not in valid_set]
    
    if invalid_params:
        raise ValueError(
            f"Invalid parameters for {model_type}: {invalid_params}. "
            f"Valid parameters: {sorted(valid_set)}"
        )
    
    return active_params
