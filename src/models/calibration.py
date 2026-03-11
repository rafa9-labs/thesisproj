"""
Probability calibration for model outputs.
Extracted from MLBacktesterNoWFO.py and utilsNoWFO.py for modularity.

Includes:
- Temperature scaling (Guo et al., ICML 2017)
- Isotonic/sigmoid calibration (sklearn)
- Conformal prediction for coverage guarantees
"""

import numpy as np
import logging
from typing import Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Temperature scaling for probability calibration.
    
    Optimizes a single scalar T to minimize NLL on calibration set.
    Transforms probabilities as: softmax(logits / T)
    
    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
    """
    
    def __init__(self):
        """Initialize temperature scaler."""
        self.temperature = 1.0
    
    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> float:
        """
        Fit temperature parameter on calibration set.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            y_true: True labels (n_samples,)
            
        Returns:
            Optimal temperature value
        """
        proba = proba.astype(np.float32)
        y_true = y_true.astype(np.int32)
        
        # Grid search for T minimizing NLL
        T_candidates = np.concatenate([
            np.linspace(0.1, 1.0, 10),
            np.linspace(1.0, 2.0, 10),
            np.linspace(2.0, 3.0, 5),
            np.linspace(3.0, 4.0, 5)
        ])
        
        best_T = 1.0
        best_nll = self._nll(proba, y_true)
        
        for T in T_candidates:
            proba_scaled = self._apply_temperature(proba, T)
            nll = self._nll(proba_scaled, y_true)
            if nll < best_nll:
                best_T = float(T)
                best_nll = float(nll)
        
        self.temperature = best_T
        logger.info(f"Temperature scaling fitted: T={self.temperature:.3f}, NLL={best_nll:.4f}")
        
        return self.temperature
    
    def transform(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities (n_samples, n_classes) as float32
        """
        return self._apply_temperature(proba, self.temperature)
    
    def fit_transform(self, proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            y_true: True labels (n_samples,)
            
        Returns:
            Calibrated probabilities (n_samples, n_classes) as float32
        """
        self.fit(proba, y_true)
        return self.transform(proba)
    
    @staticmethod
    def _apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
        """
        Apply temperature scaling transformation.
        
        Args:
            proba: Probabilities (n_samples, n_classes)
            T: Temperature parameter
            
        Returns:
            Scaled probabilities (n_samples, n_classes) as float32
        """
        T = float(max(1e-3, T))
        
        # Convert to log-probabilities
        logp = np.log(np.clip(proba, 1e-7, 1.0)).astype(np.float64)
        
        # Scale by temperature
        z = logp / T
        
        # Numerical stability: subtract max before exp
        z = z - z.max(axis=1, keepdims=True)
        
        # Softmax
        ez = np.exp(z)
        proba_scaled = ez / np.sum(ez, axis=1, keepdims=True)
        
        return proba_scaled.astype(np.float32)
    
    @staticmethod
    def _nll(proba: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute negative log-likelihood.
        
        Args:
            proba: Probabilities (n_samples, n_classes)
            y_true: True labels (n_samples,)
            
        Returns:
            Mean NLL
        """
        idx = (np.arange(len(y_true)), y_true.astype(int))
        p_true = np.clip(proba[idx], 1e-7, 1.0)
        return float(-np.log(p_true).mean())


class IsotonicCalibrator:
    """
    Isotonic or sigmoid calibration using sklearn.
    
    Fits a monotonic mapping from uncalibrated to calibrated probabilities.
    Works well for classical ML models (XGBoost, Random Forest, etc.).
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ("isotonic" or "sigmoid")
        """
        if method not in ["isotonic", "sigmoid"]:
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got {method}")
        
        self.method = method
        self.calibrators = []  # One per class
    
    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit calibration mapping.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            y_true: True labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        proba = proba.astype(np.float32)
        y_true = y_true.astype(np.int32)
        
        n_classes = proba.shape[1]
        self.calibrators = []
        
        for class_idx in range(n_classes):
            # Binary labels for this class
            y_binary = (y_true == class_idx).astype(int)
            
            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(proba[:, class_idx], y_binary)
            else:  # sigmoid
                # Use sklearn's Platt scaling (logistic regression)
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
                calibrator.fit(proba[:, class_idx].reshape(-1, 1), y_binary)
            
            self.calibrators.append(calibrator)
        
        logger.info(f"Isotonic calibration fitted: method={self.method}, n_classes={n_classes}")
        
        return self
    
    def transform(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities (n_samples, n_classes) as float32
        """
        if not self.calibrators:
            raise RuntimeError("Calibrator must be fitted before transform")
        
        proba = proba.astype(np.float32)
        n_samples, n_classes = proba.shape
        
        proba_calibrated = np.zeros_like(proba)
        
        for class_idx, calibrator in enumerate(self.calibrators):
            if self.method == "isotonic":
                proba_calibrated[:, class_idx] = calibrator.transform(proba[:, class_idx])
            else:  # sigmoid
                proba_calibrated[:, class_idx] = calibrator.predict_proba(
                    proba[:, class_idx].reshape(-1, 1)
                )[:, 1]
        
        # Normalize to ensure probabilities sum to 1
        row_sums = proba_calibrated.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 1e-9, None)
        proba_calibrated = proba_calibrated / row_sums
        
        return proba_calibrated.astype(np.float32)
    
    def fit_transform(self, proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            y_true: True labels (n_samples,)
            
        Returns:
            Calibrated probabilities (n_samples, n_classes) as float32
        """
        self.fit(proba, y_true)
        return self.transform(proba)


class ConformalPredictor:
    """
    Split-conformal prediction for multiclass classification.
    
    Provides prediction sets with finite-sample coverage guarantees.
    Nonconformity score: 1 - p_true
    
    Reference: Vovk et al., "Algorithmic Learning in a Random World", 2005
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (1 - alpha = coverage level)
                  e.g., alpha=0.1 gives 90% coverage
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        
        self.alpha = alpha
        self.qhat = None
    
    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> 'ConformalPredictor':
        """
        Fit conformal prediction threshold on calibration set.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            y_true: True labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        proba = proba.astype(np.float32)
        y_true = y_true.astype(np.int32)
        
        # Compute nonconformity scores
        idx = (np.arange(len(y_true)), y_true)
        nc_scores = 1.0 - proba[idx]
        
        # Compute quantile
        n = len(nc_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = float(np.quantile(nc_scores, q_level))
        
        logger.info(f"Conformal predictor fitted: alpha={self.alpha}, qhat={self.qhat:.4f}")
        
        return self
    
    def predict_sets(self, proba: np.ndarray) -> np.ndarray:
        """
        Predict conformal prediction sets.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            
        Returns:
            Binary prediction sets (n_samples, n_classes)
            1 = class included in prediction set, 0 = excluded
        """
        if self.qhat is None:
            raise RuntimeError("Conformal predictor must be fitted before prediction")
        
        proba = proba.astype(np.float32)
        
        # Include class if 1 - p_class <= qhat
        # Equivalently: p_class >= 1 - qhat
        threshold = 1.0 - self.qhat
        prediction_sets = (proba >= threshold).astype(np.int8)
        
        return prediction_sets
    
    def predict_with_sets(self, proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both point predictions and conformal sets.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            
        Returns:
            Tuple of (point_predictions, prediction_sets)
        """
        point_preds = np.argmax(proba, axis=1)
        pred_sets = self.predict_sets(proba)
        
        return point_preds, pred_sets
    
    def get_set_sizes(self, proba: np.ndarray) -> np.ndarray:
        """
        Get size of prediction set for each sample.
        
        Args:
            proba: Predicted probabilities (n_samples, n_classes)
            
        Returns:
            Set sizes (n_samples,)
        """
        pred_sets = self.predict_sets(proba)
        return pred_sets.sum(axis=1)


def calibrate_probabilities(
    proba: np.ndarray,
    y_true: np.ndarray,
    method: str = "temperature",
    **kwargs
) -> Tuple[np.ndarray, object]:
    """
    Convenience function for probability calibration.
    
    Args:
        proba: Predicted probabilities (n_samples, n_classes)
        y_true: True labels (n_samples,)
        method: Calibration method ("temperature", "isotonic", "sigmoid")
        **kwargs: Additional arguments for calibrator
        
    Returns:
        Tuple of (calibrated_proba, calibrator_object)
    """
    if method == "temperature":
        calibrator = TemperatureScaling()
    elif method in ["isotonic", "sigmoid"]:
        calibrator = IsotonicCalibrator(method=method)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    proba_calibrated = calibrator.fit_transform(proba, y_true)
    
    return proba_calibrated, calibrator
