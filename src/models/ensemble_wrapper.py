"""
Ensemble model wrapper for combining multiple strategies.
Supports soft/hard voting and weighted averaging.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseStrategy


logger = logging.getLogger(__name__)


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble wrapper combining multiple base strategies.
    
    Features:
    - Parallel training of base models
    - Soft voting (weighted probability averaging)
    - Hard voting (majority vote)
    - Automatic weight learning
    - Memory cleanup for all sub-models
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        models: List[BaseStrategy]
    ):
        """
        Initialize ensemble strategy.
        
        Args:
            config: Ensemble configuration
            models: List of base strategy instances
        """
        super().__init__(config)
        self.models = models
        self.weights: Optional[np.ndarray] = None
        self.voting_method = config.get('voting_method', 'soft')
        
        if self.voting_method not in ['soft', 'hard']:
            raise ValueError(f"voting_method must be 'soft' or 'hard', got {self.voting_method}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parallel: bool = True,
        **kwargs
    ) -> 'EnsembleStrategy':
        """
        Train all base models (optionally in parallel).
        
        Args:
            X: Training features (n_samples, n_features) or (n_samples, timesteps, n_features)
            y: Training labels (n_samples,)
            parallel: Whether to train models in parallel
            **kwargs: Additional fit arguments passed to base models
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training ensemble: {len(self.models)} models, parallel={parallel}")
        
        if parallel:
            self._fit_parallel(X, y, **kwargs)
        else:
            self._fit_sequential(X, y, **kwargs)
        
        # Learn voting weights if requested
        if self.config.get('learn_weights', False):
            self._learn_weights(X, y)
        else:
            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        self._is_fitted = True
        
        logger.info(f"Ensemble training complete: weights={self.weights}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels (n_samples,)
        """
        if self.voting_method == 'soft':
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:  # hard voting
            predictions = np.array([model.predict(X) for model in self.models])
            # Majority vote
            from scipy.stats import mode
            majority, _ = mode(predictions, axis=0, keepdims=False)
            return majority
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted averaging.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities (n_samples, n_classes) as float32
        """
        if not self._is_fitted:
            raise RuntimeError("EnsembleStrategy must be fitted before prediction")
        
        # Get predictions from all models
        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            probas.append(proba)
        
        # Stack predictions
        probas = np.array(probas)  # (n_models, n_samples, n_classes)
        
        # Weighted average
        if self.weights is not None:
            weights = self.weights.reshape(-1, 1, 1)  # (n_models, 1, 1)
            ensemble_proba = (probas * weights).sum(axis=0)
        else:
            ensemble_proba = probas.mean(axis=0)
        
        # Normalize
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        ensemble_proba = ensemble_proba / np.clip(row_sums, 1e-9, None)
        
        return ensemble_proba.astype(np.float32)
    
    def save(self, path: str) -> None:
        """
        Save ensemble and all base models to disk.
        
        Args:
            path: Directory path to save ensemble
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted ensemble")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each base model
        for i, model in enumerate(self.models):
            model_path = save_path / f'model_{i}'
            model.save(str(model_path))
        
        # Save ensemble metadata
        metadata_path = save_path / 'ensemble_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'weights': self.weights,
                'voting_method': self.voting_method,
                'n_models': len(self.models),
                'model_types': [type(m).__name__ for m in self.models]
            }, f)
        
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str) -> 'EnsembleStrategy':
        """
        Load ensemble and all base models from disk.
        
        Args:
            path: Directory path to load ensemble from
            
        Returns:
            Self for method chaining
        """
        load_path = Path(path)
        
        # Load metadata
        metadata_path = load_path / 'ensemble_metadata.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.config = metadata['config']
        self.weights = metadata['weights']
        self.voting_method = metadata['voting_method']
        
        # Load each base model
        n_models = metadata['n_models']
        self.models = []
        
        for i in range(n_models):
            model_path = load_path / f'model_{i}'
            # Note: This requires models to be pre-initialized
            # In practice, you'd need to know the model types
            # For now, assume models are already in self.models
            if i < len(self.models):
                self.models[i].load(str(model_path))
        
        self._is_fitted = True
        
        logger.info(f"Ensemble loaded from {path}")
        
        return self
    
    def _fit_sequential(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train models sequentially.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional fit arguments
        """
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {type(model).__name__}")
            model.fit(X, y, **kwargs)
            
            # Cleanup after each model
            model._cleanup()
    
    def _fit_parallel(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train models in parallel using ThreadPoolExecutor.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional fit arguments
        """
        max_workers = min(len(self.models), self.config.get('max_workers', 3))
        
        def train_model(model_idx: int) -> Tuple[int, BaseStrategy]:
            model = self.models[model_idx]
            logger.info(f"Training model {model_idx+1}/{len(self.models)}: {type(model).__name__}")
            model.fit(X, y, **kwargs)
            model._cleanup()
            return model_idx, model
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(train_model, i): i
                for i in range(len(self.models))
            }
            
            for future in as_completed(futures):
                model_idx, trained_model = future.result()
                self.models[model_idx] = trained_model
                logger.info(f"Model {model_idx+1} training complete")
    
    def _learn_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Learn optimal voting weights using validation performance.
        
        Args:
            X: Validation features
            y: Validation labels
        """
        from sklearn.linear_model import LogisticRegression
        
        # Get predictions from all models
        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            probas.append(proba)
        
        # Stack predictions: (n_samples, n_models * n_classes)
        probas = np.array(probas)  # (n_models, n_samples, n_classes)
        n_models, n_samples, n_classes = probas.shape
        
        # Flatten for meta-learner
        X_meta = probas.transpose(1, 0, 2).reshape(n_samples, -1)
        
        # Train simple logistic regression to learn weights
        meta_learner = LogisticRegression(max_iter=1000)
        meta_learner.fit(X_meta, y)
        
        # Extract weights (average across classes)
        coef = meta_learner.coef_  # (n_classes, n_models * n_classes)
        weights_per_class = coef.reshape(n_classes, n_models, n_classes)
        
        # Average weights across classes and normalize
        self.weights = np.abs(weights_per_class).mean(axis=(0, 2))
        self.weights = self.weights / self.weights.sum()
        
        logger.info(f"Learned ensemble weights: {self.weights}")
    
    def _cleanup(self) -> None:
        """
        Clean up memory for all base models.
        
        CRITICAL: Calls _cleanup() on all sub-models.
        """
        for model in self.models:
            model._cleanup()
        
        super()._cleanup()
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual predictions from each base model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Dictionary mapping model names to predictions
        """
        predictions = {}
        
        for i, model in enumerate(self.models):
            model_name = f"{type(model).__name__}_{i}"
            predictions[model_name] = model.predict_proba(X)
        
        return predictions
