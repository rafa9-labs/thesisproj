"""
XGBoost model wrapper with GPU/CPU support and early stopping.
Extracted from MLBacktesterNoWFO.py lines 10511-10580.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path

from xgboost import XGBClassifier

from .base import ClassicalMLStrategy


logger = logging.getLogger(__name__)


class XGBoostStrategy(ClassicalMLStrategy):
    """
    XGBoost classifier wrapper with GPU/CPU support and early stopping.
    
    Features:
    - Automatic GPU/CPU device selection with fallback
    - Early stopping on validation set
    - Parameter filtering and validation
    - Strict type hints (np.ndarray)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost strategy.
        
        Args:
            config: Dictionary with XGBoost parameters (prefixed with 'xgb_' or unprefixed)
        """
        super().__init__(config)
        self.model: Optional[XGBClassifier] = None
        self.best_iteration: Optional[int] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> 'XGBoostStrategy':
        """
        Train XGBoost model with optional early stopping.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            eval_set: Optional tuple of (X_val, y_val) for early stopping
            **kwargs: Additional fit arguments
            
        Returns:
            Self for method chaining
        """
        # Extract XGBoost parameters
        xgb_params = self._extract_xgb_params()
        
        # Build model
        self.model = self._build_xgb_model(xgb_params)
        
        # Prepare fit arguments
        fit_kwargs = {}
        
        # Early stopping
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            
            early_stopping_rounds = int(self.config.get(
                'xgb_early_stopping_rounds',
                self.config.get('early_stopping_rounds', 50)
            ))
            
            if early_stopping_rounds > 0:
                fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
                fit_kwargs['verbose'] = False
        
        # Train
        logger.info(f"Training XGBoost: n_samples={len(X)}, n_features={X.shape[1]}")
        
        self.model.fit(X, y, **fit_kwargs)
        
        # Store best iteration if early stopping was used
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
            logger.info(f"XGBoost early stopping at iteration {self.best_iteration}")
        
        self._is_fitted = True
        
        return self
    
    def save(self, path: str) -> None:
        """
        Save XGBoost model to disk.
        
        Args:
            path: File path to save model
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model using pickle (includes all metadata)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'best_iteration': self.best_iteration
            }, f)
        
        logger.info(f"XGBoost model saved to {path}")
    
    def load(self, path: str) -> 'XGBoostStrategy':
        """
        Load XGBoost model from disk.
        
        Args:
            path: File path to load model from
            
        Returns:
            Self for method chaining
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.config = data['config']
        self.best_iteration = data.get('best_iteration')
        self._is_fitted = True
        
        logger.info(f"XGBoost model loaded from {path}")
        
        return self
    
    def _extract_xgb_params(self) -> Dict[str, Any]:
        """
        Extract and filter XGBoost parameters from config.
        
        Returns:
            Dictionary of XGBoost parameters
        """
        xgb_params = {}
        
        # Extract parameters with 'xgb_' prefix or unprefixed
        for key, value in self.config.items():
            if key.startswith('xgb_'):
                param_name = key[4:]  # Remove 'xgb_' prefix
                xgb_params[param_name] = value
            elif key in ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                        'colsample_bytree', 'gamma', 'min_child_weight', 'reg_alpha',
                        'reg_lambda', 'scale_pos_weight', 'random_state', 'n_jobs']:
                xgb_params[key] = value
        
        # Set defaults
        xgb_params.setdefault('objective', 'multi:softprob')
        xgb_params.setdefault('n_estimators', 400)
        xgb_params.setdefault('max_depth', 6)
        xgb_params.setdefault('learning_rate', 0.1)
        xgb_params.setdefault('subsample', 0.8)
        xgb_params.setdefault('colsample_bytree', 0.8)
        xgb_params.setdefault('n_jobs', 3)
        xgb_params.setdefault('random_state', 42)
        
        # Handle regularization parameter naming
        # XGBoost sklearn API uses 'reg_lambda' instead of 'lambda'
        if 'lambda' in xgb_params and 'reg_lambda' not in xgb_params:
            xgb_params['reg_lambda'] = xgb_params.pop('lambda')
        
        if 'alpha' in xgb_params and 'reg_alpha' not in xgb_params:
            xgb_params['reg_alpha'] = xgb_params.pop('alpha')
        
        return xgb_params
    
    def _build_xgb_model(self, xgb_params: Dict[str, Any]) -> XGBClassifier:
        """
        Build XGBoost model with GPU/CPU device selection.
        
        Args:
            xgb_params: XGBoost parameters
            
        Returns:
            XGBClassifier instance
        """
        # GPU configuration
        use_gpu = self.config.get('use_gpu', False)
        
        if use_gpu:
            xgb_params['device'] = 'cuda'
            xgb_params['tree_method'] = 'hist'
            logger.info("XGBoost: GPU mode enabled")
        else:
            xgb_params['device'] = 'cpu'
            xgb_params['tree_method'] = 'hist'
        
        # Remove parameters that shouldn't be passed to XGBClassifier
        params_to_remove = [
            'early_stopping_rounds', 'xgb_early_stopping_rounds',
            'eval_fraction', 'xgb_eval_fraction',
            'use_gpu'
        ]
        for param in params_to_remove:
            xgb_params.pop(param, None)
        
        # Try to build model with GPU, fallback to CPU if it fails
        try:
            model = XGBClassifier(**xgb_params)
            logger.info(f"XGBoost model built: {len(xgb_params)} parameters")
            return model
        except Exception as e:
            if use_gpu:
                logger.warning(f"GPU initialization failed ({e}), falling back to CPU")
                xgb_params['device'] = 'cpu'
                xgb_params['tree_method'] = 'hist'
                model = XGBClassifier(**xgb_params)
                return model
            else:
                raise
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array or None if not fitted
        """
        if not self._is_fitted or self.model is None:
            return None
        
        return self.model.feature_importances_
    
    def get_booster(self):
        """
        Get underlying XGBoost Booster object.
        
        Returns:
            XGBoost Booster or None if not fitted
        """
        if not self._is_fitted or self.model is None:
            return None
        
        return self.model.get_booster()
