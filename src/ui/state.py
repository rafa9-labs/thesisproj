"""
State management with Streamlit caching.

CRITICAL: Uses @st.cache_resource and @st.cache_data to prevent
recalculation on every UI interaction (button clicks, slider changes, etc.).
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging

from src.core.config import load_default_config, AppConfig
from src.data.factory import DataFactory
from src.features.pipeline import FeaturePipeline
from src.models import XGBoostStrategy
from src.execution import BacktestEngine


logger = logging.getLogger(__name__)


class AppState:
    """
    Centralized state management with aggressive caching.
    
    CRITICAL: Prevents recalculation on every UI interaction.
    - @st.cache_resource: For singleton objects (config, factories, pipelines)
    - @st.cache_data: For data loading and computation results
    """
    
    @staticmethod
    @st.cache_resource
    def get_config() -> AppConfig:
        """
        Load configuration (cached as singleton).
        
        Returns:
            AppConfig instance
        """
        logger.info("Loading configuration (cached)")
        return load_default_config()
    
    @staticmethod
    @st.cache_resource
    def get_data_factory(_config: AppConfig) -> DataFactory:
        """
        Initialize DataFactory (cached as singleton).
        
        Args:
            _config: AppConfig instance (underscore prevents hashing)
            
        Returns:
            DataFactory instance
        """
        logger.info("Initializing DataFactory (cached)")
        return DataFactory(_config)
    
    @staticmethod
    @st.cache_resource
    def get_feature_pipeline(_config: AppConfig, _factory: DataFactory) -> FeaturePipeline:
        """
        Initialize FeaturePipeline (cached as singleton).
        
        Args:
            _config: AppConfig instance
            _factory: DataFactory instance
            
        Returns:
            FeaturePipeline instance
        """
        logger.info("Initializing FeaturePipeline (cached)")
        return FeaturePipeline(_config, _factory)
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_historical_data(
        instrument: str,
        start_date: str,
        end_date: str,
        granularity: str
    ) -> pd.DataFrame:
        """
        Load historical data (cached for 1 hour).
        
        CRITICAL: Uses @st.cache_data to prevent reloading on every button click.
        
        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            start_date: Start date string
            end_date: End date string
            granularity: Timeframe (e.g., "H1")
            
        Returns:
            DataFrame with OHLC data
        """
        logger.info(f"Loading data: {instrument} {start_date} to {end_date} ({granularity})")
        
        # Get factory from cache
        config = AppState.get_config()
        factory = AppState.get_data_factory(config)
        
        # Load data
        from datetime import datetime, timezone
        from dateutil.parser import parse as parse_datetime
        
        # Convert string dates to timezone-aware datetime objects (UTC)
        start_dt = parse_datetime(start_date)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        
        end_dt = parse_datetime(end_date)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        
        df = factory.get_data(
            source="oanda",
            instrument=instrument,
            granularity=granularity,
            start=start_dt,
            end=end_dt
        )
        
        logger.info(f"Loaded {len(df)} bars")
        return df
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def build_features(
        _df_data: pd.DataFrame,
        use_rsi: bool,
        use_macd: bool,
        use_ema: bool,
        use_bbands: bool,
        use_atr: bool,
        use_adx: bool
    ) -> Tuple[pd.DataFrame, list]:
        """
        Build features (cached).
        
        Args:
            _df_data: Raw OHLC data
            use_rsi: Enable RSI indicator
            use_macd: Enable MACD indicator
            use_ema: Enable EMA indicator
            use_bbands: Enable Bollinger Bands
            use_atr: Enable ATR indicator
            use_adx: Enable ADX indicator
            
        Returns:
            Tuple of (features_df, feature_list)
        """
        logger.info("Building features (cached)")
        
        # Get config and pipeline from cache
        config = AppState.get_config()
        factory = AppState.get_data_factory(config)
        pipeline = AppState.get_feature_pipeline(config, factory)
        
        # Update config with user selections
        config.features.use_rsi = use_rsi
        config.features.use_macd = use_macd
        config.features.use_ema = use_ema
        config.features.use_bbands = use_bbands
        config.features.use_atr = use_atr
        config.features.use_adx = use_adx
        
        # Build features
        # Ensure returns column exists for labeling
        if 'returns' not in _df_data.columns:
            # Convert price columns to numeric if they're strings
            price_cols = ['close', 'mid_close', 'open', 'mid_open', 'high', 'mid_high', 'low', 'mid_low']
            for col in price_cols:
                if col in _df_data.columns:
                    _df_data[col] = pd.to_numeric(_df_data[col], errors='coerce')
            
            if 'close' in _df_data.columns:
                _df_data['returns'] = _df_data['close'].pct_change()
            elif 'mid_close' in _df_data.columns:
                _df_data['returns'] = _df_data['mid_close'].pct_change()
        
        df_features, features = pipeline.build_features(_df_data)
        
        logger.info(f"Built {len(features)} features")
        return df_features, features
    
    @staticmethod
    def train_and_predict(
        df_features: pd.DataFrame,
        features: List[str],
        train_split: float = 0.8,
        enable_hpo: bool = False,
        hpo_mode: str = 'single_time',
        n_trials: int = 50,
        active_params: tuple = (),
        trial_callback = None
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Train XGBoost model and generate predictions.
        
        CRITICAL: Uses real model predictions, not synthetic data.
        NEW: NO CACHING - executes fresh every time for WFO/HPO.
        
        Args:
            df_features: Features DataFrame
            features: List of feature names
            train_split: Fraction of data for training (default 0.8)
            enable_hpo: Enable hyperparameter optimization
            hpo_mode: HPO mode
            n_trials: Number of Optuna trials
            active_params: Tuple of active parameters
            trial_callback: Optional callback for Optuna trials
            
        Returns:
            Tuple of (predictions, probabilities, stitched_indices)
            - predictions: Trading signals {-1, 0, 1}
            - probabilities: Class probabilities (n_samples, 3)
            - stitched_indices: Exact DatetimeIndex for perfect alignment
        """
        logger.info("Training XGBoost model (fresh execution)")
        
        # Get config and initialize trainer
        config = AppState.get_config()
        factory = AppState.get_data_factory(config)
        pipeline = AppState.get_feature_pipeline(config, factory)
        
        # Update config with HPO settings if enabled
        if enable_hpo:
            from src.core.config import HPOConfig
            hpo_config = HPOConfig(
                enable_hpo=True,
                hpo_mode=hpo_mode,
                n_trials=n_trials,
                active_params=list(active_params) if active_params else []
            )
            config = config.model_copy(update={'hpo': hpo_config})
        
        # Initialize trainer with updated config
        from src.models.trainer import ModelTrainer
        trainer = ModelTrainer(config, pipeline, factory)
        
        # Initialize strategy (HPO will optimize these parameters)
        xgb_config = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'tree_method': 'hist',
            'device': 'cpu'
        }
        strategy = XGBoostStrategy(xgb_config)
        
        # Get date range from df_features
        train_start = str(df_features.index[0])
        split_idx = int(len(df_features) * train_split)
        train_end = str(df_features.index[split_idx - 1])
        test_start = str(df_features.index[split_idx])
        test_end = str(df_features.index[-1])
        
        logger.info(f"WFO Training: {train_start} to {train_end}, Test: {test_start} to {test_end}")
        
        # CRITICAL: Call train_with_wfo with trial_callback for HPO telemetry
        # Returns trained strategy, fold logs, and exact test indices for perfect alignment
        trained_strategy, metrics, fold_logs, _, _, stitched_indices = trainer.train_with_wfo(
            strategy=strategy,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            calibrate=True,
            trial_callback=trial_callback
        )
        
        # Generate predictions for ENTIRE test period using the final trained strategy
        # This ensures predictions length matches test data length
        split_idx = int(len(df_features) * train_split)
        X_test = df_features[features].iloc[split_idx:].values.astype(np.float32)
        
        # Scale test data using the same scaler as training
        from sklearn.preprocessing import StandardScaler
        X_train = df_features[features].iloc[:split_idx].values.astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        
        # Generate predictions for full test period
        probabilities = trained_strategy.predict_proba(X_test_scaled)
        predictions = np.argmax(probabilities, axis=1) - 1  # Convert {0, 1, 2} -> {-1, 0, 1}
        
        # Store fold logs in session state for dashboard rendering
        if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results is not None:
            st.session_state.backtest_results['fold_logs'] = fold_logs
        
        logger.info(f"Generated {len(predictions)} predictions for full test period using final trained strategy")
        logger.info(f"Prediction distribution: {np.bincount(predictions + 1)}")  # Count of each class
        logger.info(f"Stitched indices count: {len(stitched_indices)}")
        
        return predictions, probabilities, stitched_indices
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def run_backtest(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        _df_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.Series]:
        """
        Run backtest (cached).
        
        CRITICAL: Prevents re-running backtest on every UI interaction.
        
        Args:
            predictions: Model predictions (-1, 0, 1)
            probabilities: Model probabilities (n_samples, n_classes)
            _df_test: Test data DataFrame
            
        Returns:
            Tuple of (results_df, metrics_dict, equity_curve)
        """
        logger.info("Running backtest (cached)")
        
        # Get config
        config = AppState.get_config()
        
        # Add execution config with updated sharpe_cap
        execution_config = config.model_dump()
        execution_config.update({
            'sharpe_cap': 100.0,  # Fix the fake -30.00 Sharpe issue
            'min_trades_for_reliability': 30,
            'use_hac': True,
            'hac_max_lag': 'auto'
        })
        
        # Initialize engine
        engine = BacktestEngine(execution_config)
        
        # Run backtest
        results_df, metrics = engine.run_backtest(
            predictions=predictions,
            probabilities=probabilities,
            df_data=_df_test
        )
        
        # Build equity curve (float32 for UI)
        equity_curve = engine.build_equity_curve(results_df)
        
        logger.info(f"Backtest complete: Sharpe={metrics.get('sharpe', 0):.2f}")
        
        return results_df, metrics, equity_curve
