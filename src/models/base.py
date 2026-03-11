"""
BaseStrategy: Abstract base class for all trading models.
Provides standardized interface for fit, predict, save/load, and memory cleanup.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
import gc


logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies/models.
    
    All model wrappers must implement this interface to ensure:
    - Consistent API across classical ML, deep learning, and ensembles
    - Memory cleanup after training/inference
    - Serialization for persistence
    - Type safety with strict np.ndarray/pd.DataFrame hints
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration.
        
        Args:
            config: Dictionary of model hyperparameters and settings
        """
        self.config = config
        self.model = None
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'BaseStrategy':
        """
        Train the model on provided data.
        
        Args:
            X: Training features (n_samples, n_features) or (n_samples, timesteps, n_features)
            y: Training labels (n_samples,)
            **kwargs: Additional training arguments (e.g., eval_set, validation_split)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> 'BaseStrategy':
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
            
        Returns:
            Self for method chaining
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model configuration parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseStrategy':
        """
        Set model configuration parameters.
        
        Args:
            **params: Parameters to update
            
        Returns:
            Self for method chaining
        """
        self.config.update(params)
        return self
    
    def is_fitted(self) -> bool:
        """
        Check if model has been fitted.
        
        Returns:
            True if model is fitted, False otherwise
        """
        return self._is_fitted
    
    def _cleanup(self) -> None:
        """
        Clean up GPU/RAM memory after training or inference.
        
        CRITICAL: This method MUST be called after every HPO trial or WFO fold
        to prevent memory leaks, especially for TensorFlow/Keras models.
        
        Default implementation handles basic cleanup. Subclasses should override
        to add model-specific cleanup (e.g., TensorFlow session clearing).
        """
        # Force garbage collection
        gc.collect()
        
        logger.debug(f"{self.__class__.__name__}._cleanup() called")
    
    def __repr__(self) -> str:
        """String representation of strategy."""
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_status})"


class ClassicalMLStrategy(BaseStrategy):
    """
    Base class for classical ML models (XGBoost, Random Forest, Logistic, etc.).
    Provides common functionality for sklearn-style models.
    """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using sklearn-style model.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before prediction")
        
        if self.model is None:
            raise RuntimeError("Model is None - cannot predict")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using sklearn-style model.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes) as float32
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before prediction")
        
        if self.model is None:
            raise RuntimeError("Model is None - cannot predict")
        
        proba = self.model.predict_proba(X)
        return proba.astype(np.float32)


class DeepLearningStrategy(BaseStrategy):
    """
    Base class for deep learning models (CNN, LSTM, Transformer).
    Provides common functionality for TensorFlow/Keras models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize deep learning strategy.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.scaler = None
        self.input_shape = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using Keras model.
        
        Args:
            X: Features (n_samples, timesteps, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using Keras model.
        
        Args:
            X: Features (n_samples, timesteps, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes) as float32
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before prediction")
        
        if self.model is None:
            raise RuntimeError("Model is None - cannot predict")
        
        # Apply scaling if scaler exists
        X_scaled = X
        if self.scaler is not None:
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled_flat = self.scaler.transform(X_flat)
            X_scaled = X_scaled_flat.reshape(original_shape)
        
        proba = self.model.predict(X_scaled, verbose=0)
        return proba.astype(np.float32)
    
    def _cleanup(self) -> None:
        """
        Clean up TensorFlow/Keras session and GPU memory.
        
        CRITICAL: Must be called after every training/inference cycle.
        """
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            logger.debug(f"{self.__class__.__name__}: TensorFlow session cleared")
        except Exception as e:
            logger.warning(f"TensorFlow cleanup failed: {e}")
        
        # Call parent cleanup for garbage collection
        super()._cleanup()
