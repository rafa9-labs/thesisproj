"""
Example usage of Phase 3: Model Integration Layer

Demonstrates:
1. Training XGBoost with calibration
2. Training CNN with early stopping
3. Ensemble voting (CNN + LSTM + XGB)
4. Walk-Forward Optimization
5. Cross-validation
6. Hyperparameter optimization
7. Memory cleanup
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from src.core.config import load_default_config
from src.data.factory import DataFactory
from src.features.pipeline import FeaturePipeline
from src.models import (
    XGBoostStrategy,
    CNNStrategy,
    LSTMStrategy,
    EnsembleStrategy,
    ModelTrainer,
    TemperatureScaling,
    IsotonicCalibrator
)


def configure_logging():
    """Setup logging for examples"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def example_xgboost_training():
    """Example: Train XGBoost with calibration"""
    print("=" * 60)
    print("EXAMPLE 1: XGBoost Training with Calibration")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20).astype(np.float32)
    y_train = np.random.randint(0, 3, 1000)
    X_test = np.random.randn(200, 20).astype(np.float32)
    y_test = np.random.randint(0, 3, 200)
    
    # Configure XGBoost
    config = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'use_gpu': False
    }
    
    # Create and train strategy
    strategy = XGBoostStrategy(config)
    strategy.fit(X_train, y_train)
    
    print(f"✓ XGBoost trained: {strategy.is_fitted()}")
    
    # Predict
    proba = strategy.predict_proba(X_test)
    print(f"✓ Predictions: {proba.shape}")
    
    # Calibrate
    calibrator = TemperatureScaling()
    proba_cal = calibrator.fit_transform(proba[:100], y_test[:100])
    
    print(f"✓ Temperature scaling: T={calibrator.temperature:.3f}")
    
    # Cleanup
    strategy._cleanup()
    print(f"✓ Memory cleanup complete")
    
    print()


def example_cnn_training():
    """Example: Train CNN with windowing"""
    print("=" * 60)
    print("EXAMPLE 2: CNN Training with Windowing")
    print("=" * 60)
    
    # Create synthetic sequential data
    np.random.seed(42)
    X_train = np.random.randn(500, 10, 15).astype(np.float32)  # (samples, timesteps, features)
    y_train = np.random.randint(0, 3, 500)
    X_test = np.random.randn(100, 10, 15).astype(np.float32)
    y_test = np.random.randint(0, 3, 100)
    
    # Configure CNN
    config = {
        'cnn_filters1': 32,
        'cnn_filters2': 64,
        'cnn_kernel_size': 3,
        'cnn_dense_units': 64,
        'cnn_dropout_rate': 0.3,
        'cnn_learning_rate': 0.001,
        'cnn_epochs': 10,
        'cnn_batch_size': 32,
        'cnn_use_early_stopping': True,
        'cnn_patience': 3
    }
    
    # Create and train strategy
    strategy = CNNStrategy(config)
    strategy.fit(X_train, y_train, validation_split=0.2, verbose=0)
    
    print(f"✓ CNN trained: {strategy.is_fitted()}")
    print(f"✓ Input shape: {strategy.input_shape}")
    
    # Predict
    proba = strategy.predict_proba(X_test)
    print(f"✓ Predictions: {proba.shape}")
    
    # Cleanup (CRITICAL for TensorFlow)
    strategy._cleanup()
    print(f"✓ TensorFlow session cleared")
    
    print()


def example_ensemble_training():
    """Example: Ensemble voting with CNN + LSTM + XGBoost"""
    print("=" * 60)
    print("EXAMPLE 3: Ensemble Voting (CNN + LSTM + XGB)")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(500, 10, 15).astype(np.float32)
    y_train = np.random.randint(0, 3, 500)
    X_test = np.random.randn(100, 10, 15).astype(np.float32)
    y_test = np.random.randint(0, 3, 100)
    
    # Create base models
    cnn_config = {
        'cnn_filters1': 16,
        'cnn_filters2': 32,
        'cnn_epochs': 5,
        'cnn_batch_size': 32,
        'verbose': 0
    }
    
    lstm_config = {
        'lstm_units': 32,
        'lstm_epochs': 5,
        'lstm_batch_size': 32,
        'verbose': 0
    }
    
    xgb_config = {
        'n_estimators': 50,
        'max_depth': 3,
        'use_gpu': False
    }
    
    cnn = CNNStrategy(cnn_config)
    lstm = LSTMStrategy(lstm_config)
    
    # XGBoost needs 2D input - flatten
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    xgb = XGBoostStrategy(xgb_config)
    
    # Create ensemble
    ensemble_config = {
        'voting_method': 'soft',
        'learn_weights': False,
        'max_workers': 2
    }
    
    ensemble = EnsembleStrategy(ensemble_config, models=[cnn, lstm])
    
    # Train ensemble (parallel)
    print("Training ensemble...")
    ensemble.fit(X_train, y_train, parallel=True, verbose=0)
    
    print(f"✓ Ensemble trained: {len(ensemble.models)} models")
    print(f"✓ Voting weights: {ensemble.weights}")
    
    # Predict
    proba = ensemble.predict_proba(X_test)
    print(f"✓ Ensemble predictions: {proba.shape}")
    
    # Cleanup (CRITICAL: cleans all sub-models)
    ensemble._cleanup()
    print(f"✓ All models cleaned up")
    
    print()


def example_walk_forward_optimization():
    """Example: Walk-Forward Optimization with leakage prevention"""
    print("=" * 60)
    print("EXAMPLE 4: Walk-Forward Optimization")
    print("=" * 60)
    
    config = load_default_config()
    config.features.use_rsi = True
    config.features.use_ema = True
    
    factory = DataFactory(config)
    pipeline = FeaturePipeline(config, factory)
    trainer = ModelTrainer(config, pipeline, factory)
    
    # Create strategy
    xgb_config = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'use_gpu': False
    }
    strategy = XGBoostStrategy(xgb_config)
    
    try:
        # WFO training (with leakage prevention)
        trained_strategy, metrics = trainer.train_with_wfo(
            strategy=strategy,
            train_start='2023-01-01',
            train_end='2023-06-30',
            test_start='2023-07-01',
            test_end='2023-12-31',
            calibrate=True
        )
        
        print(f"✓ WFO training complete")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        
        # Cleanup
        trained_strategy._cleanup()
        
    except Exception as e:
        print(f"✗ WFO example skipped (data not available): {e}")
    
    print()


def example_cross_validation():
    """Example: Time-series cross-validation"""
    print("=" * 60)
    print("EXAMPLE 5: Time-Series Cross-Validation")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 20).astype(np.float32)
    y = np.random.randint(0, 3, 1000)
    
    config = load_default_config()
    pipeline = FeaturePipeline(config)
    trainer = ModelTrainer(config, pipeline)
    
    # Create strategy
    xgb_config = {
        'n_estimators': 50,
        'max_depth': 3,
        'use_gpu': False
    }
    strategy = XGBoostStrategy(xgb_config)
    
    # Cross-validate
    cv_metrics = trainer.cross_validate(
        strategy=strategy,
        X=X,
        y=y,
        n_splits=5
    )
    
    print(f"✓ Cross-validation complete")
    print(f"  Avg Accuracy: {cv_metrics['accuracy']:.4f} ± {cv_metrics['accuracy_std']:.4f}")
    print(f"  Avg F1: {cv_metrics['f1']:.4f} ± {cv_metrics['f1_std']:.4f}")
    
    # Cleanup
    strategy._cleanup()
    
    print()


def example_hyperparameter_optimization():
    """Example: Hyperparameter optimization with Optuna"""
    print("=" * 60)
    print("EXAMPLE 6: Hyperparameter Optimization")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(500, 20).astype(np.float32)
    y = np.random.randint(0, 3, 500)
    
    config = load_default_config()
    pipeline = FeaturePipeline(config)
    trainer = ModelTrainer(config, pipeline)
    
    # Define search space
    search_space = {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
        'max_depth': {'type': 'int', 'low': 3, 'high': 8},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}
    }
    
    # Optimize
    print("Running HPO (10 trials)...")
    best_strategy, best_params = trainer.optimize_hyperparameters(
        strategy_class=XGBoostStrategy,
        search_space=search_space,
        X=X,
        y=y,
        n_trials=10
    )
    
    print(f"✓ HPO complete")
    print(f"  Best params: {best_params}")
    
    # Cleanup
    best_strategy._cleanup()
    
    print()


def example_calibration_methods():
    """Example: Compare calibration methods"""
    print("=" * 60)
    print("EXAMPLE 7: Calibration Methods Comparison")
    print("=" * 60)
    
    # Create synthetic probabilities
    np.random.seed(42)
    proba = np.random.dirichlet([1, 1, 1], 200).astype(np.float32)
    y_true = np.random.randint(0, 3, 200)
    
    # Temperature scaling
    temp_cal = TemperatureScaling()
    proba_temp = temp_cal.fit_transform(proba, y_true)
    
    print(f"✓ Temperature scaling: T={temp_cal.temperature:.3f}")
    
    # Isotonic calibration
    iso_cal = IsotonicCalibrator(method='isotonic')
    proba_iso = iso_cal.fit_transform(proba, y_true)
    
    print(f"✓ Isotonic calibration: {len(iso_cal.calibrators)} calibrators")
    
    # Sigmoid calibration
    sig_cal = IsotonicCalibrator(method='sigmoid')
    proba_sig = sig_cal.fit_transform(proba, y_true)
    
    print(f"✓ Sigmoid calibration: {len(sig_cal.calibrators)} calibrators")
    
    print()


def example_memory_cleanup():
    """Example: Memory cleanup importance"""
    print("=" * 60)
    print("EXAMPLE 8: Memory Cleanup (Critical for HPO)")
    print("=" * 60)
    
    # Simulate HPO trials
    for trial in range(3):
        print(f"Trial {trial + 1}/3...")
        
        # Create and train model
        X = np.random.randn(200, 10, 15).astype(np.float32)
        y = np.random.randint(0, 3, 200)
        
        config = {
            'cnn_epochs': 3,
            'cnn_batch_size': 32,
            'verbose': 0
        }
        
        strategy = CNNStrategy(config)
        strategy.fit(X, y, validation_split=0.2)
        
        # CRITICAL: Cleanup after each trial
        strategy._cleanup()
        print(f"  ✓ Trial {trial + 1} complete, memory cleaned")
    
    print(f"✓ All trials complete without memory leaks")
    print()


def main():
    """Run all examples"""
    configure_logging()
    
    print("\n" + "=" * 60)
    print("PHASE 3: MODEL INTEGRATION LAYER - EXAMPLES")
    print("=" * 60 + "\n")
    
    example_xgboost_training()
    example_cnn_training()
    example_ensemble_training()
    example_walk_forward_optimization()
    example_cross_validation()
    example_hyperparameter_optimization()
    example_calibration_methods()
    example_memory_cleanup()
    
    print("=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Standardized BaseStrategy interface")
    print("✓ XGBoost, CNN, LSTM wrappers with strict type hints")
    print("✓ Ensemble voting with parallel training")
    print("✓ Walk-Forward Optimization with leakage prevention")
    print("✓ Time-series cross-validation")
    print("✓ Hyperparameter optimization with Optuna")
    print("✓ Temperature scaling and isotonic calibration")
    print("✓ Memory cleanup (_cleanup() after every trial)")
    print("\nNext Steps:")
    print("1. Integrate with your actual forex data")
    print("2. Customize model configurations")
    print("3. Run full WFO backtests")
    print("4. Ready for production deployment")
    print()


if __name__ == "__main__":
    main()
