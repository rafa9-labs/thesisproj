"""
ModelTrainer: Orchestrates model training with Walk-Forward Optimization.

CRITICAL LEAKAGE PREVENTION:
- All feature scaling/transformation fitted ONLY on train_start to train_end
- Validation/test data NEVER used for fitting scalers or calibrators
- Strict temporal ordering in WFO folds
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Type
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ..core.config import AppConfig
from ..data.factory import DataFactory
from ..features.pipeline import FeaturePipeline

from .base import BaseStrategy
from .calibration import TemperatureScaling, IsotonicCalibrator, ConformalPredictor
from .hpo import get_search_space, record_boundary_hit, OPTUNA_AVAILABLE


logger = logging.getLogger(__name__)


@dataclass
class WFOFold:
    """Single Walk-Forward Optimization fold with HPO results."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    hpo_best_params: Dict[str, Any] = field(default_factory=dict)
    hpo_best_score: float = np.nan
    hpo_n_trials: int = 0
    test_metrics: Dict[str, float] = field(default_factory=dict)


class ModelTrainer:
    """
    Model training orchestrator with Walk-Forward Optimization.
    
    Features:
    - WFO with strict temporal ordering
    - Leakage prevention (scalers fitted only on train data)
    - Memory cleanup after each trial/fold
    - Integration with FeaturePipeline and DataFactory
    - Probability calibration
    - Time-series cross-validation
    """
    
    def __init__(
        self,
        config: AppConfig,
        feature_pipeline: FeaturePipeline,
        data_factory: Optional[DataFactory] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            config: Application configuration
            feature_pipeline: Feature engineering pipeline
            data_factory: Optional data factory for loading data
        """
        self.config = config
        self.feature_pipeline = feature_pipeline
        self.data_factory = data_factory
        self._best_params_cache = {}  # Cache for single_time HPO mode
        
        logger.info("ModelTrainer initialized")
    
    def train_with_wfo(
        self,
        strategy: BaseStrategy,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        calibrate: bool = True,
        trial_callback: Optional[Any] = None,
        **kwargs
    ) -> Tuple[BaseStrategy, Dict[str, Any], List[WFOFold], np.ndarray, np.ndarray, List]:
        """
        Train model with Walk-Forward Optimization.
        
        NEW: Returns fold-by-fold HPO logs for transparency.
        NEW: Accepts trial_callback for real-time HPO telemetry.
        NEW: Returns concatenated OOS predictions and probabilities from all folds.
        
        CRITICAL LEAKAGE PREVENTION:
        - Feature scaling fitted ONLY on each fold's train block
        - HPO validation split is chronological (last 20% of train)
        - Test data NEVER used for any fitting
        - Strict position-based indexing prevents off-by-one errors
        
        Args:
            strategy: Model strategy to train
            train_start: Training period start date (YYYY-MM-DD)
            train_end: Training period end date (YYYY-MM-DD)
            test_start: Test period start date (YYYY-MM-DD)
            test_end: Test period end date (YYYY-MM-DD)
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Tuple of (trained_strategy, aggregated_metrics, fold_logs, predictions, probabilities, stitched_indices)
            - predictions: Concatenated OOS predictions from all folds as class indices {0, 1, 2}
            - probabilities: Concatenated OOS probabilities from all folds (n_samples, 3)
            - stitched_indices: Exact DatetimeIndex of test blocks for perfect alignment
            **kwargs: Additional training arguments
            
        Returns:
            Tuple of (trained_strategy, aggregated_metrics, fold_logs)
        """
        logger.info(f"WFO Training: {train_start} to {train_end} -> Test: {test_start} to {test_end}")
        
        # 1) Load full data range for WFO fold generation
        df_full = self._load_data_for_period(train_start, test_end)
        
        if df_full.empty:
            raise ValueError("Empty data")
        
        # 2) Build features for full range
        df_features, features = self.feature_pipeline.build_features(df_full)
        
        if not features:
            raise ValueError("No features generated")
        
        logger.info(f"Features: {len(features)} features built")
        
        # 3) Extract features and labels (full range)
        X_full = df_features[features].values.astype(np.float32)
        y_full = self._extract_labels(df_features)
        
        # 4) Generate WFO folds with strict position-based indexing
        folds = self._generate_wfo_folds(df_features)
        
        if not folds:
            raise ValueError("No WFO folds generated")
        
        logger.info(f"WFO: {len(folds)} folds generated")
        
        # 5) Execute fold lifecycle
        fold_logs = []
        all_test_predictions = []
        all_test_probabilities = []
        all_test_labels = []
        stitched_indices = []  # Track exact DatetimeIndex of test blocks
        
        for fold in folds:
            fold_idx = fold['fold_idx']
            logger.info(f"\nWFO Fold {fold_idx + 1}/{len(folds)}: Train [{fold['train_start']} to {fold['train_end']}], Test [{fold['test_start']} to {fold['test_end']}]")
            
            # Step A: Slice train/test data using STRICT position-based indexing
            train_start_pos = fold['train_start_pos']
            train_end_pos = fold['train_end_pos']
            test_start_pos = fold['test_start_pos']
            test_end_pos = fold['test_end_pos']
            
            # CRITICAL: Use .iloc for position-based slicing (no off-by-one errors)
            X_train = X_full[train_start_pos:train_end_pos]
            y_train = y_full[train_start_pos:train_end_pos]
            X_test = X_full[test_start_pos:test_end_pos]
            y_test = y_full[test_start_pos:test_end_pos]
            
            # Capture exact test block index for perfect alignment
            test_index = df_features.iloc[test_start_pos:test_end_pos].index
            stitched_indices.extend(test_index.tolist())
            
            # Fit scaler ONLY on train block
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
            X_test_scaled = scaler.transform(X_test).astype(np.float32)
            
            # Step B: HPO Phase (if enabled)
            best_params = {}
            best_score = np.nan
            
            if self.config.hpo.enable_hpo:
                model_type = self._infer_model_type(strategy)
                
                # Split train into HPO train/val using hpo_validation_split
                val_split = self.config.wfo.hpo_validation_split
                hpo_split_idx = int(len(X_train_scaled) * (1 - val_split))
                
                X_train_hpo = X_train_scaled[:hpo_split_idx]
                y_train_hpo = y_train[:hpo_split_idx]
                X_val_hpo = X_train_scaled[hpo_split_idx:]
                y_val_hpo = y_train[hpo_split_idx:]
                
                # CRITICAL: Slice df_features for validation set (for BacktestEngine)
                # Calculate absolute positions in df_features
                val_start_pos = train_start_pos + hpo_split_idx
                val_end_pos = train_end_pos
                df_val_hpo = df_features.iloc[val_start_pos:val_end_pos].copy()
                
                # Run HPO based on mode
                if self.config.hpo.hpo_mode == 'single_time':
                    if model_type not in self._best_params_cache:
                        logger.info(f"HPO (single_time): Running optimization")
                        best_params = self._run_optuna_study(
                            type(strategy), X_train_hpo, y_train_hpo, X_val_hpo, y_val_hpo, model_type, df_val_hpo, trial_callback
                        )
                        self._best_params_cache[model_type] = best_params
                        best_score = getattr(self, '_last_study_best_value', np.nan)
                    else:
                        best_params = self._best_params_cache[model_type]
                        logger.info(f"HPO (single_time): Using cached params")
                
                elif self.config.hpo.hpo_mode in ('continuous_wfo', 'mini_folds'):
                    logger.info(f"HPO ({self.config.hpo.hpo_mode}): Running optimization for fold")
                    best_params = self._run_optuna_study(
                        type(strategy), X_train_hpo, y_train_hpo, X_val_hpo, y_val_hpo, model_type, df_val_hpo, trial_callback
                    )
                    best_score = getattr(self, '_last_study_best_value', np.nan)
                
                # Apply best params
                if best_params and hasattr(strategy, 'set_params'):
                    strategy.set_params(**best_params)
            
            # Step C: Train Phase (on FULL train block with best params)
            logger.info(f"Training on {len(X_train_scaled)} samples")
            strategy.fit(X_train_scaled, y_train, **kwargs)
            
            # Step D: Test Phase (predict test block)
            proba_test = strategy.predict_proba(X_test_scaled)
            y_pred_test = np.argmax(proba_test, axis=1)
            
            # Step E: Log fold results
            test_metrics = self._compute_metrics(y_test, y_pred_test, proba_test)
            
            fold_log = WFOFold(
                fold_idx=fold_idx,
                train_start=fold['train_start'],
                train_end=fold['train_end'],
                test_start=fold['test_start'],
                test_end=fold['test_end'],
                train_size=fold['train_size'],
                test_size=fold['test_size'],
                hpo_best_params=best_params,
                hpo_best_score=best_score,
                hpo_n_trials=self.config.hpo.n_trials if self.config.hpo.enable_hpo else 0,
                test_metrics=test_metrics
            )
            fold_logs.append(fold_log)
            
            # Collect test predictions for aggregation
            all_test_predictions.append(y_pred_test)
            all_test_probabilities.append(proba_test)
            all_test_labels.append(y_test)
            
            # Memory cleanup
            strategy._cleanup()
            
            logger.info(f"Fold {fold_idx + 1} complete: accuracy={test_metrics['accuracy']:.4f}")
        
        # 6) Aggregate metrics across folds (CRITICAL: concatenate, don't average)
        all_test_predictions_concat = np.concatenate(all_test_predictions)
        all_test_probabilities_concat = np.vstack(all_test_probabilities)
        all_test_labels_concat = np.concatenate(all_test_labels)
        
        aggregated_metrics = self._compute_metrics(
            all_test_labels_concat,
            all_test_predictions_concat,
            all_test_probabilities_concat
        )
        
        logger.info(f"WFO Training complete: {len(folds)} folds, aggregated accuracy={aggregated_metrics['accuracy']:.4f}")
        
        # Return concatenated OOS predictions, probabilities, and exact indices
        # CRITICAL: These are the TRUE out-of-sample predictions from WFO with perfect index alignment
        return strategy, aggregated_metrics, fold_logs, all_test_predictions_concat, all_test_probabilities_concat, stitched_indices
    
    def cross_validate(
        self,
        strategy: BaseStrategy,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        **kwargs
    ) -> Dict[str, float]:
        """
        Time-series cross-validation.
        
        CRITICAL LEAKAGE PREVENTION:
        - Each fold: scaler fitted ONLY on training portion
        - Strict temporal ordering (no shuffling)
        
        Args:
            strategy: Model strategy to evaluate
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            n_splits: Number of CV splits
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary of averaged CV metrics
        """
        logger.info(f"Time-series CV: {n_splits} splits")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"CV Fold {fold_idx + 1}/{n_splits}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # LEAKAGE PREVENTION: Fit scaler ONLY on training fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold).astype(np.float32)
            X_val_scaled = scaler.transform(X_val_fold).astype(np.float32)
            
            # Train
            strategy.fit(X_train_scaled, y_train_fold, **kwargs)
            
            # Predict
            proba_val = strategy.predict_proba(X_val_scaled)
            y_pred_val = np.argmax(proba_val, axis=1)
            
            # Metrics
            metrics = self._compute_metrics(y_val_fold, y_pred_val, proba_val)
            fold_metrics.append(metrics)
            
            # Cleanup
            strategy._cleanup()
            
            logger.info(f"Fold {fold_idx + 1} accuracy: {metrics['accuracy']:.4f}")
        
        # Average metrics across folds
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in fold_metrics])
            avg_metrics[f"{key}_std"] = np.std([m[key] for m in fold_metrics])
        
        logger.info(f"CV complete: avg_accuracy={avg_metrics['accuracy']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
        
        return avg_metrics
    
    def optimize_hyperparameters(
        self,
        strategy_class: Type[BaseStrategy],
        search_space: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        **kwargs
    ) -> Tuple[BaseStrategy, Dict[str, Any]]:
        """
        Hyperparameter optimization using Optuna.
        
        CRITICAL: Calls _cleanup() after EVERY trial to prevent memory leaks.
        
        Args:
            strategy_class: Strategy class to optimize
            search_space: Optuna search space definition
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            n_trials: Number of Optuna trials
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (best_strategy, best_params)
        """
        import optuna
        
        logger.info(f"HPO: {n_trials} trials for {strategy_class.__name__}")
        
        # Split data for HPO
        n_samples = len(X)
        n_val = int(n_samples * 0.2)
        
        X_train_hpo = X[:-n_val]
        y_train_hpo = y[:-n_val]
        X_val_hpo = X[-n_val:]
        y_val_hpo = y[-n_val:]
        
        # LEAKAGE PREVENTION: Fit scaler once on HPO training data
        scaler_hpo = StandardScaler()
        X_train_hpo_scaled = scaler_hpo.fit_transform(X_train_hpo).astype(np.float32)
        X_val_hpo_scaled = scaler_hpo.transform(X_val_hpo).astype(np.float32)
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Create strategy with sampled params
            strategy = strategy_class(params)
            
            try:
                # Train
                strategy.fit(X_train_hpo_scaled, y_train_hpo)
                
                # Evaluate
                proba_val = strategy.predict_proba(X_val_hpo_scaled)
                y_pred_val = np.argmax(proba_val, axis=1)
                
                # Compute metric
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val_hpo, y_pred_val)
                
            # REMOVED: Silent try/except that was hiding failures
            # Errors now propagate to UI
            
            finally:
                # CRITICAL: Cleanup after EVERY trial
                strategy._cleanup()
            
            return score
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Build best model
        best_params = study.best_params
        best_strategy = strategy_class(best_params)
        
        logger.info(f"HPO complete: best_score={study.best_value:.4f}, best_params={best_params}")
        
        return best_strategy, best_params
    
    def _load_data_for_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load data for specified date range and ensure returns column exists.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            DataFrame with price data and returns column
        """
        if self.data_factory is None:
            raise ValueError("DataFactory not configured")
        
        # Load full data and filter by date range
        df = self.data_factory.get_data(
            source="csv",
            csv_path=self.config.data.csv_path
        )
        
        # Filter by date range if index is datetime
        if hasattr(df.index, 'to_pydatetime'):
            df = df.loc[start_date:end_date]
        
        if df.empty:
            raise ValueError(f"No data found for period {start_date} to {end_date}")
        
        # Ensure returns column exists for labeling
        if 'returns' not in df.columns:
            # Convert price columns to numeric if they're strings
            price_cols = ['close', 'mid_close', 'open', 'mid_open', 'high', 'mid_high', 'low', 'mid_low']
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create returns column from close price
            if 'close' in df.columns:
                df['returns'] = df['close'].pct_change()
            elif 'mid_close' in df.columns:
                df['returns'] = df['mid_close'].pct_change()
            else:
                raise ValueError("No close price column found for returns calculation")
        
        return df
    
    def _extract_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract labels from DataFrame.
        
        CRITICAL: Generates labels from returns column using sign of next-period return.
        Labels are {0, 1, 2} representing {-1, 0, 1} signals.
        
        Args:
            df: DataFrame with returns column
            
        Returns:
            Labels array as class indices {0, 1, 2}
        """
        if 'returns' not in df.columns:
            raise ValueError("'returns' column not found in DataFrame")
        
        # Generate labels from next-period returns
        y = np.sign(df['returns'].shift(-1)).fillna(0).values
        
        # Convert to class labels {0, 1, 2}
        y_class = (y + 1).astype(np.int32)
        
        return y_class
    
    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            log_loss
        )
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        # Log loss (handle potential errors)
        try:
            metrics['log_loss'] = float(log_loss(y_true, proba, labels=[0, 1, 2]))
        except Exception:
            metrics['log_loss'] = np.nan
        
        return metrics
    
    def _generate_wfo_folds(
        self,
        df_full: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Generate WFO fold boundaries with strict position-based indexing.
        
        CRITICAL: Uses .iloc for position-based slicing to prevent off-by-one errors.
        
        Args:
            df_full: Full DataFrame with datetime index
            
        Returns:
            List of fold dictionaries with position indices and date strings
        """
        folds = []
        
        n_total = len(df_full)
        test_period = self.config.wfo.test_period_duration
        min_train = self.config.wfo.min_train_size
        
        # Determine initial train end position
        # For expanding: start with min_train
        # For rolling: start with training_duration
        if self.config.wfo.train_window_type == 'expanding':
            initial_train_end_pos = min_train
        else:
            train_duration = self.config.wfo.training_duration
            initial_train_end_pos = max(min_train, train_duration)
        
        # Ensure we have enough data
        if initial_train_end_pos + test_period > n_total:
            logger.warning(f"Insufficient data for WFO: need {initial_train_end_pos + test_period}, have {n_total}")
            return []
        
        # Generate folds
        test_start_pos = initial_train_end_pos
        fold_idx = 0
        
        while test_start_pos + test_period <= n_total:
            # Test block positions (STRICT: no overlap)
            test_end_pos = test_start_pos + test_period
            
            # Train block positions
            if self.config.wfo.train_window_type == 'expanding':
                # Expanding: train from start to test_start
                train_start_pos = 0
                train_end_pos = test_start_pos
            else:
                # Rolling: train on fixed training_duration
                train_duration = self.config.wfo.training_duration
                train_start_pos = max(0, test_start_pos - train_duration)
                train_end_pos = test_start_pos
            
            # Validate train size
            train_size = train_end_pos - train_start_pos
            if train_size < min_train:
                logger.warning(f"Fold {fold_idx}: Train size {train_size} < min {min_train}, stopping")
                break
            
            # Extract date strings using position-based indexing
            train_start_date = str(df_full.iloc[train_start_pos].name)
            train_end_date = str(df_full.iloc[train_end_pos - 1].name)
            test_start_date = str(df_full.iloc[test_start_pos].name)
            test_end_date = str(df_full.iloc[test_end_pos - 1].name)
            
            folds.append({
                'fold_idx': fold_idx,
                'train_start_pos': train_start_pos,
                'train_end_pos': train_end_pos,
                'test_start_pos': test_start_pos,
                'test_end_pos': test_end_pos,
                'train_start': train_start_date,
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date,
                'train_size': train_size,
                'test_size': test_period
            })
            
            # Move to next fold
            test_start_pos = test_end_pos
            fold_idx += 1
            
            # Stop if we've reached the requested number of folds (NEW: n_mini_folds)
            if self.config.wfo.n_mini_folds and fold_idx >= self.config.wfo.n_mini_folds:
                break
        
        # CRITICAL: Raise error if no folds generated
        if len(folds) == 0:
            raise ValueError(
                f"WFO generated 0 folds. Check your window sizing:\n"
                f"  - Data size: {n_total} bars\n"
                f"  - Training duration: {self.config.wfo.training_duration} bars\n"
                f"  - Test period: {test_period} bars\n"
                f"  - Min train size: {min_train} bars\n"
                f"  - Initial train end: {initial_train_end_pos}\n"
                f"  - Required: {initial_train_end_pos + test_period} bars minimum"
            )
        
        logger.info(f"Generated {len(folds)} WFO folds ({self.config.wfo.train_window_type} window)")
        return folds
    
    @staticmethod
    def _infer_model_type(strategy: BaseStrategy) -> str:
        """
        Infer model type from strategy class name.
        
        Args:
            strategy: Strategy instance
            
        Returns:
            Model type string ('xgboost', 'cnn', 'lstm', 'ensemble')
        """
        class_name = type(strategy).__name__.lower()
        
        if 'xgboost' in class_name or 'xgb' in class_name:
            return 'xgboost'
        elif 'cnn' in class_name:
            return 'cnn'
        elif 'lstm' in class_name:
            return 'lstm'
        elif 'ensemble' in class_name:
            return 'ensemble'
        else:
            # Default to xgboost
            logger.warning(f"Unknown strategy type: {class_name}, defaulting to xgboost")
            return 'xgboost'
    
    def _run_optuna_study(
        self,
        strategy_class: Type[BaseStrategy],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        model_type: str,
        df_val: Optional[pd.DataFrame] = None,
        trial_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run Optuna hyperparameter optimization study.
        
        NEW: Accepts optional trial_callback for real-time UI updates.
        
        CRITICAL Features:
        1. Dynamically fetch search space from hpo.get_search_space()
        2. CHRONOLOGICAL validation split (last 20% of data) - NO random splits
        3. Evaluate using log_loss (primary) or accuracy (fallback)
        4. Call strategy._cleanup() in finally block (prevent TensorFlow OOM)
        5. Track boundary hits if enabled in config
        
        Args:
            strategy_class: Strategy class to optimize
            X_train: Training features (already scaled)
            y_train: Training labels
            X_val: Validation features (already scaled, optional)
            y_val: Validation labels (optional)
            model_type: Model type ('xgboost', 'cnn', 'lstm', 'ensemble')
            trial_callback: Optional callback function called after each trial
            
        Returns:
            Dictionary with best parameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, skipping HPO")
            return {}
        
        import optuna
        
        logger.info(f"HPO: Starting Optuna study for {model_type}")
        
        # CRITICAL: Use chronological split if validation data not provided
        # Last 20% of training data becomes validation (NO random split)
        if X_val is None or y_val is None:
            n_samples = len(X_train)
            n_val = int(n_samples * 0.2)
            
            # CHRONOLOGICAL: Take LAST 20% for validation
            X_train_hpo = X_train[:-n_val]
            y_train_hpo = y_train[:-n_val]
            X_val_hpo = X_train[-n_val:]
            y_val_hpo = y_train[-n_val:]
            
            logger.info(f"HPO: Chronological split - Train: {len(X_train_hpo)}, Val: {len(X_val_hpo)}")
        else:
            X_train_hpo = X_train
            y_train_hpo = y_train
            X_val_hpo = X_val
            y_val_hpo = y_val
            
            logger.info(f"HPO: Using provided validation set - Train: {len(X_train_hpo)}, Val: {len(X_val_hpo)}")
        
        # Boundary tracking
        boundary_tracker = {} if self.config.hpo.track_boundary_hits else None
        
        # Determine optimization direction and metric
        use_log_loss = True
        try:
            from sklearn.metrics import log_loss as _test_log_loss
        except ImportError:
            use_log_loss = False
        
        direction = 'minimize' if use_log_loss else 'maximize'
        
        def objective(trial: optuna.Trial) -> float:
            """
            Optuna objective function using BacktestEngine for realistic Sharpe calculation.
            
            CRITICAL: Optimizes for real trading performance, not just prediction accuracy.
            Accounts for spreads, slippage, and execution delays.
            """
            # Dynamically fetch search space from hpo.py
            params = get_search_space(
                model_type=model_type,
                trial=trial,
                config=self.config,
                active_params=self.config.hpo.active_params
            )
            
            # Track boundary hits if enabled
            if boundary_tracker is not None:
                # Note: Boundary tracking would require access to search ranges
                # For now, we log the sampled params
                pass
            
            # Create strategy with sampled params
            strategy = strategy_class(params)
            
            try:
                # Train
                strategy.fit(X_train_hpo, y_train_hpo)
                
                # Predict on validation set
                proba_val = strategy.predict_proba(X_val_hpo)
                
                # CRITICAL: Convert predictions to trading signals {-1, 0, 1}
                predictions = np.argmax(proba_val, axis=1) - 1  # {0,1,2} -> {-1,0,1}
                
                # Run mini-backtest with BacktestEngine for realistic evaluation
                if df_val is not None:
                    from src.execution.engine import BacktestEngine
                    
                    # Instantiate engine with config
                    engine_config = self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config
                    engine = BacktestEngine(engine_config)
                    
                    # Run backtest on validation set
                    _, metrics = engine.run_backtest(
                        predictions=predictions,
                        probabilities=proba_val,
                        df_data=df_val
                    )
                    
                    # Get Sharpe ratio
                    score = metrics.get('sharpe', -100.0)
                    
                    # CRITICAL: Catch NaN Sharpe and convert to penalty
                    # Prevents Optuna from marking trial as failed
                    if score is None or np.isnan(score):
                        return -100.0
                    
                    # Zero-trade penalty: Optuna shouldn't optimize by staying flat
                    if metrics.get('trades', 0) == 0:
                        return -100.0
                else:
                    # Fallback to accuracy if no df_val provided
                    from sklearn.metrics import accuracy_score
                    y_pred_val = np.argmax(proba_val, axis=1)
                    score = accuracy_score(y_val_hpo, y_pred_val)
                
                return float(score)
            
            finally:
                # CRITICAL: Cleanup after EVERY trial to prevent TensorFlow OOM
                try:
                    strategy._cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup failed: {cleanup_error}")
        
        # Create Optuna study
        sampler_name = self.config.hpo.sampler
        if sampler_name == 'tpe':
            sampler = optuna.samplers.TPESampler()
        elif sampler_name == 'random':
            sampler = optuna.samplers.RandomSampler()
        elif sampler_name == 'grid':
            sampler = optuna.samplers.GridSampler({})
        else:
            sampler = optuna.samplers.TPESampler()
        
        pruner_name = self.config.hpo.pruner
        if pruner_name == 'median':
            pruner = optuna.pruners.MedianPruner()
        elif pruner_name == 'hyperband':
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.NopPruner()
        
        # CRITICAL: Explicitly maximize Sharpe ratio (not minimize)
        study = optuna.create_study(
            direction='maximize',  # Always maximize Sharpe/accuracy
            sampler=sampler,
            pruner=pruner
        )
        
        # Run optimization with optional callback
        n_trials = self.config.hpo.n_trials
        logger.info(f"HPO: Running {n_trials} trials (direction=maximize, metric=Sharpe)")
        
        if trial_callback:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[trial_callback])
        else:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Log results
        best_params = study.best_params
        best_value = study.best_value
        
        # Store best value for fold logging
        self._last_study_best_value = best_value
        
        logger.info(f"HPO complete: best_value={best_value:.4f}, best_params={best_params}")
        
        # Log boundary hits if tracked
        if boundary_tracker and self.config.hpo.verbose:
            logger.info(f"HPO boundary hits: {boundary_tracker}")
        
        return best_params
    
    def save_model(
        self,
        strategy: BaseStrategy,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save trained model with metadata.
        
        Args:
            strategy: Trained strategy
            path: Path to save model
            metadata: Optional metadata to save
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save strategy
        strategy.save(str(save_path))
        
        # Save metadata if provided
        if metadata is not None:
            import json
            metadata_path = save_path.parent / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(
        self,
        strategy: BaseStrategy,
        path: str
    ) -> Tuple[BaseStrategy, Optional[Dict[str, Any]]]:
        """
        Load trained model with metadata.
        
        Args:
            strategy: Strategy instance to load into
            path: Path to load model from
            
        Returns:
            Tuple of (loaded_strategy, metadata)
        """
        load_path = Path(path)
        
        # Load strategy
        strategy.load(str(load_path))
        
        # Load metadata if exists
        metadata = None
        metadata_path = load_path.parent / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Model loaded from {path}")
        
        return strategy, metadata
