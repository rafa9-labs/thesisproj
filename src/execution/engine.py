"""
BacktestEngine: Main orchestrator for backtesting.

Features:
- Vectorized fast-path (default) for maximum performance
- Bar-by-bar fallback for path-dependent risk management
- Execution delay enforcement (no look-ahead bias)
- Float32 equity curves for UI readiness
- Integration with ModelTrainer output
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import logging

from .simulator import TradeSimulator
from .metrics import PerformanceEvaluator, METRIC_NAMES
from .risk import StaticStopLoss, TrailingStop, PositionSizer


logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting orchestrator with vectorized fast-path.
    
    CRITICAL Features:
    1. Execution Delay: Signals at bar t execute at bar t+1
    2. Vectorized Fast-Path: Default for non-path-dependent strategies
    3. Bar-by-Bar Fallback: Used when trailing stops or complex risk management active
    4. Float32 Equity Curves: Memory-efficient for UI plotting
    
    Integration:
    - Takes ModelTrainer predictions
    - Uses TradeSimulator for costs
    - Computes 16 standard metrics
    - Generates trade logs and equity curves
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        simulator: Optional[TradeSimulator] = None,
        risk_manager: Optional[Any] = None,
        evaluator: Optional[PerformanceEvaluator] = None
    ):
        """
        Initialize backtest engine.
        
        Args:
            config: Configuration dictionary
            simulator: TradeSimulator instance (created if None)
            risk_manager: Optional risk manager (StaticStopLoss, TrailingStop)
            evaluator: PerformanceEvaluator instance (created if None)
        """
        self.config = config
        self.simulator = simulator or TradeSimulator(config)
        self.risk_manager = risk_manager
        self.evaluator = evaluator or PerformanceEvaluator(config)
        
        # Determine execution mode
        self.use_vectorized = self._should_use_vectorized()
        
        logger.info(f"BacktestEngine initialized: vectorized={self.use_vectorized}, "
                   f"risk_manager={type(risk_manager).__name__ if risk_manager else 'None'}")
    
    def run_backtest(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        df_data: pd.DataFrame,
        initial_equity: float = 1.0,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete backtest simulation.
        
        CRITICAL: Execution delay is enforced automatically by TradeSimulator.
        
        Args:
            predictions: Model predictions (-1, 0, 1) shape (n_samples,)
            probabilities: Model probabilities shape (n_samples, n_classes)
            df_data: DataFrame with columns: returns, spread, slippage_bps, price
            initial_equity: Starting equity (default 1.0)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (results_df, metrics_dict)
        """
        logger.info(f"Running backtest: {len(predictions)} bars, vectorized={self.use_vectorized}")
        
        # Validate inputs
        if len(predictions) != len(df_data):
            raise ValueError(f"Predictions length {len(predictions)} != data length {len(df_data)}")
        
        # Ensure required columns
        required_cols = ['returns']
        for col in required_cols:
            if col not in df_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add default columns if missing
        if 'spread' not in df_data.columns:
            df_data['spread'] = 0.0001  # Default 1 pip
        if 'slippage_bps' not in df_data.columns:
            df_data['slippage_bps'] = 0.1  # Default 0.1 bps
        if 'price' not in df_data.columns:
            df_data['price'] = 1.0  # Placeholder
        
        # Convert predictions to signals (-1, 0, 1)
        signals = self._convert_predictions_to_signals(predictions)
        
        # Run simulation (vectorized or bar-by-bar)
        if self.use_vectorized:
            results_df = self._run_vectorized_backtest(
                signals, df_data, initial_equity
            )
        else:
            results_df = self._run_bar_by_bar_backtest(
                signals, df_data, initial_equity
            )
        
        # Add predictions and probabilities to results
        results_df['pred'] = predictions
        results_df['pred_proba_0'] = probabilities[:, 0] if probabilities.shape[1] > 0 else 0.0
        results_df['pred_proba_1'] = probabilities[:, 1] if probabilities.shape[1] > 1 else 0.0
        results_df['pred_proba_2'] = probabilities[:, 2] if probabilities.shape[1] > 2 else 0.0
        
        # Add true direction
        results_df['true_direction'] = np.sign(results_df['returns'])
        
        # Compute metrics
        metrics_dict = self.compute_final_metrics(results_df)
        
        logger.info(f"Backtest complete: Sharpe={metrics_dict.get('sharpe', 0):.2f}, "
                   f"Trades={metrics_dict.get('trades', 0)}")
        
        return results_df, metrics_dict
    
    def _run_vectorized_backtest(
        self,
        signals: np.ndarray,
        df_data: pd.DataFrame,
        initial_equity: float
    ) -> pd.DataFrame:
        """
        Vectorized backtest (fast path).
        
        CRITICAL: Execution delay handled by TradeSimulator.
        
        Args:
            signals: Trading signals (n_bars,)
            df_data: Data DataFrame
            initial_equity: Starting equity
            
        Returns:
            Results DataFrame with positions, costs, equity
        """
        # Extract arrays
        returns = df_data['returns'].values
        spreads = df_data['spread'].values
        slippages = df_data['slippage_bps'].values
        
        # Run vectorized simulation
        positions, costs, equity = self.simulator.vectorized_backtest(
            signals=signals,
            returns=returns,
            spreads=spreads,
            slippages=slippages,
            initial_equity=initial_equity
        )
        
        # Build results DataFrame
        results_df = df_data.copy()
        results_df['position_exec'] = positions
        results_df['costs'] = costs
        results_df['equity'] = equity
        
        # Compute strategy returns
        results_df['strategy'] = np.log(equity / equity[0])
        
        # Continuous equity curves (for carry-over between months)
        results_df['cstrategy_cont'] = equity
        results_df['creturns_cont'] = np.exp(np.cumsum(returns))
        
        return results_df
    
    def _run_bar_by_bar_backtest(
        self,
        signals: np.ndarray,
        df_data: pd.DataFrame,
        initial_equity: float
    ) -> pd.DataFrame:
        """
        Bar-by-bar backtest (fallback for path-dependent risk management).
        
        CRITICAL: Execution delay handled by applying signals with 1-bar lag.
        
        Args:
            signals: Trading signals (n_bars,)
            df_data: Data DataFrame
            initial_equity: Starting equity
            
        Returns:
            Results DataFrame with positions, costs, equity
        """
        n_bars = len(signals)
        
        # Initialize arrays
        positions = np.zeros(n_bars, dtype=np.float32)
        costs = np.zeros(n_bars, dtype=np.float32)
        equity = np.zeros(n_bars, dtype=np.float32)
        
        # Current state
        current_position = 0.0
        current_equity = initial_equity
        
        # Apply execution delay: shift signals by 1
        delayed_signals = np.roll(signals, 1)
        delayed_signals[0] = 0  # No position on first bar
        
        for i in range(n_bars):
            signal = delayed_signals[i]
            returns_bar = df_data['returns'].iloc[i]
            spread = df_data['spread'].iloc[i]
            slippage_bps = df_data['slippage_bps'].iloc[i]
            price = df_data.get('price', pd.Series([1.0] * n_bars)).iloc[i]
            
            # Check risk management
            if self.risk_manager is not None:
                should_exit, reason = self.risk_manager.check_exit(price, current_position)
                
                if should_exit:
                    signal = 0  # Force exit
                    logger.debug(f"Bar {i}: Risk exit - {reason}")
                
                # Update trailing stop
                if hasattr(self.risk_manager, 'update'):
                    self.risk_manager.update(price, current_position)
            
            # Execute trade
            new_position, cost, _ = self.simulator.execute_trade(
                signal=int(signal),
                current_position=current_position,
                price=price,
                spread=spread,
                slippage_bps=slippage_bps
            )
            
            # Record entry for risk management
            if self.risk_manager is not None and new_position != current_position:
                if hasattr(self.risk_manager, 'on_entry'):
                    self.risk_manager.on_entry(price, new_position)
            
            # Compute PnL
            gross_pnl = current_position * returns_bar
            net_pnl = gross_pnl - cost
            
            # Update equity
            current_equity = current_equity * np.exp(net_pnl)
            
            # Record
            positions[i] = new_position
            costs[i] = cost
            equity[i] = current_equity
            
            # Update position
            current_position = new_position
        
        # Build results DataFrame
        results_df = df_data.copy()
        results_df['position_exec'] = positions
        results_df['costs'] = costs
        results_df['equity'] = equity
        
        # Compute strategy returns
        results_df['strategy'] = np.log(equity / equity[0])
        
        # Continuous equity curves
        results_df['cstrategy_cont'] = equity
        results_df['creturns_cont'] = np.exp(np.cumsum(df_data['returns'].values))
        
        return results_df
    
    def build_equity_curve(self, df: pd.DataFrame) -> pd.Series:
        """
        Build cumulative equity curve.
        
        CRITICAL: Returns float32 for UI readiness (memory-efficient plotting).
        
        Args:
            df: Results DataFrame with 'equity' column
            
        Returns:
            Equity curve as float32 Series
        """
        if 'equity' in df.columns:
            equity = df['equity'].astype(np.float32)
        elif 'cstrategy_cont' in df.columns:
            equity = df['cstrategy_cont'].astype(np.float32)
        else:
            # Fallback: build from strategy returns
            equity = np.exp(df['strategy'].cumsum()).astype(np.float32)
        
        return pd.Series(equity, index=df.index, dtype=np.float32)
    
    def generate_trade_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate detailed trade log.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Trade log DataFrame with entry/exit details
        """
        trades = []
        
        positions = df['position_exec'].values
        
        # Find trade entries and exits
        position_changes = np.diff(positions, prepend=0)
        trade_indices = np.where(position_changes != 0)[0]
        
        for i, idx in enumerate(trade_indices):
            if idx >= len(df):
                continue
            
            position = positions[idx]
            
            if abs(position) > 1e-8:
                # Entry
                entry_time = df.index[idx]
                entry_price = df.get('price', pd.Series([np.nan] * len(df))).iloc[idx]
                
                # Find exit
                exit_idx = None
                if i + 1 < len(trade_indices):
                    exit_idx = trade_indices[i + 1]
                elif idx < len(df) - 1:
                    # Check if position closes at end
                    if abs(positions[-1]) < 1e-8:
                        exit_idx = len(df) - 1
                
                if exit_idx is not None:
                    exit_time = df.index[exit_idx]
                    exit_price = df.get('price', pd.Series([np.nan] * len(df))).iloc[exit_idx]
                    
                    # Compute PnL
                    pnl = df['strategy'].iloc[idx:exit_idx+1].sum()
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'duration_bars': exit_idx - idx
                    })
        
        if not trades:
            return pd.DataFrame(columns=['entry_time', 'exit_time', 'position', 
                                        'entry_price', 'exit_price', 'pnl', 'duration_bars'])
        
        return pd.DataFrame(trades)
    
    def compute_final_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute all performance metrics.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Dictionary with all metrics
        """
        # Compute 16 standard metrics
        metrics_tuple = self.evaluator.compute_all_metrics(df)
        
        # Convert to dictionary
        metrics_dict = {
            name: value for name, value in zip(METRIC_NAMES, metrics_tuple)
        }
        
        # Add additional metrics
        metrics_dict['final_equity'] = float(df['equity'].iloc[-1]) if 'equity' in df.columns else 1.0
        metrics_dict['total_return_pct'] = (metrics_dict['final_equity'] - 1.0) * 100
        
        return metrics_dict
    
    def simulate_month(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        df_month: pd.DataFrame,
        initial_equity: float = 1.0,
        initial_position: float = 0.0
    ) -> pd.DataFrame:
        """
        Simulate one month of trading.
        
        Used for month-by-month walk-forward simulation.
        
        Args:
            predictions: Model predictions for month
            probabilities: Model probabilities for month
            df_month: Month data
            initial_equity: Starting equity
            initial_position: Carry-over position from previous month
            
        Returns:
            Results DataFrame for month
        """
        results_df, _ = self.run_backtest(
            predictions=predictions,
            probabilities=probabilities,
            df_data=df_month,
            initial_equity=initial_equity
        )
        
        # Apply initial position carry-over
        if abs(initial_position) > 1e-8:
            # Adjust first bar to account for carry-over
            results_df['position_exec'].iloc[0] = initial_position
        
        return results_df
    
    def _should_use_vectorized(self) -> bool:
        """
        Determine if vectorized fast-path can be used.
        
        Vectorized path is used UNLESS:
        - Trailing stops are active (path-dependent)
        - Complex risk management is active
        
        Returns:
            True if vectorized path should be used
        """
        # If trailing stop, must use bar-by-bar
        if isinstance(self.risk_manager, TrailingStop):
            return False
        
        # If any path-dependent risk manager, use bar-by-bar
        if self.risk_manager is not None:
            if hasattr(self.risk_manager, 'update'):
                return False
        
        # Default: use vectorized
        return True
    
    @staticmethod
    def _convert_predictions_to_signals(predictions: np.ndarray) -> np.ndarray:
        """
        Convert model predictions to trading signals.
        
        Handles various prediction formats:
        - {-1, 0, 1}: Direct signals
        - {0, 1, 2}: Class labels (convert to {-1, 0, 1})
        - Continuous: Threshold to discrete signals
        
        Args:
            predictions: Model predictions
            
        Returns:
            Trading signals (-1, 0, 1)
        """
        preds = np.asarray(predictions, dtype=float)
        
        # Check if already in {-1, 0, 1}
        unique_vals = np.unique(preds[np.isfinite(preds)])
        
        if set(unique_vals).issubset({-1, 0, 1}):
            return preds.astype(np.int8)
        
        # Check if in {0, 1, 2} (class labels)
        if set(unique_vals).issubset({0, 1, 2}):
            # Convert: 0 -> -1, 1 -> 0, 2 -> 1
            signals = preds - 1
            return signals.astype(np.int8)
        
        # Continuous predictions: threshold
        signals = np.where(preds > 0.5, 1, np.where(preds < -0.5, -1, 0))
        
        return signals.astype(np.int8)
