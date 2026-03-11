"""
LSTM model wrapper integrating with models/lstm.py.
Handles windowing, scaling, and TensorFlow session cleanup.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from .base import DeepLearningStrategy

# Import from existing models directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from lstm import build_lstm


logger = logging.getLogger(__name__)


class LSTMStrategy(DeepLearningStrategy):
    """
    LSTM classifier wrapper with windowing and scaling.
    
    Features:
    - Automatic windowing for sequential data
    - StandardScaler for feature normalization
    - Bidirectional LSTM support
    - Early stopping support
    - TensorFlow session cleanup via _cleanup()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM strategy.
        
        Args:
            config: Dictionary with LSTM parameters (prefixed with 'lstm_' or unprefixed)
        """
        super().__init__(config)
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.history = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.1,
        **kwargs
    ) -> 'LSTMStrategy':
        """
        Train LSTM model with windowing and scaling.
        
        Args:
            X: Training features (n_samples, n_features) or (n_samples, timesteps, n_features)
            y: Training labels (n_samples,)
            validation_split: Fraction of data for validation
            **kwargs: Additional fit arguments (epochs, batch_size, etc.)
            
        Returns:
            Self for method chaining
        """
        # Reshape if needed
        X_windowed = self._prepare_input(X)
        
        # Fit scaler on training data only
        self.scaler = StandardScaler()
        X_flat = X_windowed.reshape(-1, X_windowed.shape[-1])
        self.scaler.fit(X_flat)
        
        # Transform
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X_windowed.shape)
        
        # Store input shape
        self.input_shape = X_scaled.shape[1:]
        
        # Build model
        self.model = build_lstm(input_shape=self.input_shape, config=self.config)
        
        if self.model is None:
            raise RuntimeError("build_lstm returned None. Check model config or input shape.")
        
        # Training parameters
        epochs = int(kwargs.get('epochs', self.config.get('lstm_epochs', self.config.get('epochs', 50))))
        batch_size = int(kwargs.get('batch_size', self.config.get('lstm_batch_size', self.config.get('batch_size', 32))))
        verbose = int(kwargs.get('verbose', self.config.get('verbose', 0)))
        
        # Prepare callbacks
        callbacks = []
        if hasattr(self.model, 'early_stop_callback') and self.model.early_stop_callback is not None:
            callbacks.append(self.model.early_stop_callback)
        
        # Train
        logger.info(f"Training LSTM: input_shape={self.input_shape}, epochs={epochs}, batch_size={batch_size}")
        
        self.history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self._is_fitted = True
        
        logger.info(f"LSTM training complete: {len(self.history.history['loss'])} epochs")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features (n_samples, n_features) or (n_samples, timesteps, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes) as float32
        """
        if not self._is_fitted:
            raise RuntimeError("LSTMStrategy must be fitted before prediction")
        
        # Reshape if needed
        X_windowed = self._prepare_input(X)
        
        # Scale
        if self.scaler is not None:
            X_flat = X_windowed.reshape(-1, X_windowed.shape[-1])
            X_scaled_flat = self.scaler.transform(X_flat)
            X_scaled = X_scaled_flat.reshape(X_windowed.shape)
        else:
            X_scaled = X_windowed
        
        # Predict
        proba = self.model.predict(X_scaled, verbose=0)
        
        return proba.astype(np.float32)
    
    def save(self, path: str) -> None:
        """
        Save LSTM model and scaler to disk.
        
        Args:
            path: Directory path to save model
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_path = save_path / 'lstm_model.keras'
        self.model.save(model_path)
        
        # Save scaler and config
        metadata_path = save_path / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'config': self.config,
                'input_shape': self.input_shape
            }, f)
        
        logger.info(f"LSTM model saved to {path}")
    
    def load(self, path: str) -> 'LSTMStrategy':
        """
        Load LSTM model and scaler from disk.
        
        Args:
            path: Directory path to load model from
            
        Returns:
            Self for method chaining
        """
        import tensorflow as tf
        
        load_path = Path(path)
        
        # Load Keras model
        model_path = load_path / 'lstm_model.keras'
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scaler and config
        metadata_path = load_path / 'metadata.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.config = metadata['config']
        self.input_shape = metadata['input_shape']
        self._is_fitted = True
        
        logger.info(f"LSTM model loaded from {path}")
        
        return self
    
    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare input for LSTM (ensure 3D shape).
        
        Args:
            X: Input features (n_samples, n_features) or (n_samples, timesteps, n_features)
            
        Returns:
            Windowed features (n_samples, timesteps, n_features)
        """
        if X.ndim == 2:
            # Add timestep dimension
            window_size = int(self.config.get('lstm_window_size', self.config.get('window_size', 10)))
            X_windowed = self._create_windows(X, window_size)
        elif X.ndim == 3:
            X_windowed = X
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")
        
        return X_windowed.astype(np.float32)
    
    @staticmethod
    def _create_windows(X: np.ndarray, window_size: int) -> np.ndarray:
        """
        Create sliding windows from 2D array.
        
        Args:
            X: Input features (n_samples, n_features)
            window_size: Size of sliding window
            
        Returns:
            Windowed features (n_samples - window_size + 1, window_size, n_features)
        """
        from numpy.lib.stride_tricks import sliding_window_view
        
        if len(X) < window_size:
            raise ValueError(f"Input length {len(X)} < window_size {window_size}")
        
        # Create windows using stride tricks
        windowed = sliding_window_view(X, window_shape=window_size, axis=0)
        
        # Reshape to (n_windows, window_size, n_features)
        n_windows = windowed.shape[0]
        n_features = X.shape[1]
        windowed = windowed.reshape(n_windows, window_size, n_features)
        
        return windowed.astype(np.float32)
    
    def get_training_history(self) -> Optional[Dict[str, list]]:
        """
        Get training history.
        
        Returns:
            Dictionary of training metrics or None if not fitted
        """
        if self.history is None:
            return None
        
        return self.history.history
