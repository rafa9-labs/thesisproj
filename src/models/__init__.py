"""
Model integration layer: standardized interfaces for training and prediction
"""

from .base import BaseStrategy
from .calibration import TemperatureScaling, IsotonicCalibrator, ConformalPredictor
from .xgboost_wrapper import XGBoostStrategy
from .cnn_wrapper import CNNStrategy
from .lstm_wrapper import LSTMStrategy
from .ensemble_wrapper import EnsembleStrategy
from .trainer import ModelTrainer

__all__ = [
    "BaseStrategy",
    "TemperatureScaling",
    "IsotonicCalibrator",
    "ConformalPredictor",
    "XGBoostStrategy",
    "CNNStrategy",
    "LSTMStrategy",
    "EnsembleStrategy",
    "ModelTrainer",
]
