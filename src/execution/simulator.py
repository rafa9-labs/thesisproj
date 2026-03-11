"""
TradeSimulator: Execution engine with cost modeling.

Handles:
- Spread costs (capped)
- Volatility-aware slippage (two-regime model)
- Commission modeling
- Execution delay (next bar open)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class TradeSimulator:
    """
    Trade execution simulator with realistic cost modeling.
    
    CRITICAL: Execution Delay
    - If signal generated at bar close, execution happens at NEXT bar open
    - This prevents look-ahead bias (trading on prices we don't have yet)
    
    Features:
    - Spread cost (capped to prevent outlier impact)
    - Volatility-aware slippage (two-regime: low/high vol)
    - Optional commission
    - Cost breakdown tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trade simulator.
        
        Args:
            config: Dictionary with execution parameters:
                - spread_cap: Maximum spread (default 0.0004)
                - slippage_factor: Slippage multiplier (default 1.0)
                - slippage_bps_lo: Low volatility slippage in bps (default 0.08)
                - slippage_bps_med: Medium volatility slippage in bps (default 0.16)
                - commission_bps: Commission in bps (default 0.0)
                - use_execution_delay: Apply 1-bar delay (default True)
        """
        self.config = config
        
        # Spread parameters
        self.spread_cap = float(config.get('spread_cap', 0.0004))
        
        # Slippage parameters
        self.slippage_factor = float(config.get('slippage_factor', 1.0))
        self.slippage_bps_lo = float(config.get('slippage_bps_lo', 0.08))
        self.slippage_bps_med = float(config.get('slippage_bps_med', 0.16))
        
        # Commission
        self.commission_bps = float(config.get('commission_bps', 0.0))
        
        # Execution delay
        self.use_execution_delay = bool(config.get('use_execution_delay', True))
        
        logger.info(f"TradeSimulator initialized: spread_cap={self.spread_cap}, "
                   f"slippage_factor={self.slippage_factor}, "
                   f"execution_delay={self.use_execution_delay}")
    
    def execute_trade(
        self,
        signal: int,
        current_position: float,
        price: float,
        spread: float,
        slippage_bps: float,
        **kwargs
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Execute trade and compute costs.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            current_position: Current position size
            price: Execution price
            spread: Bid-ask spread
            slippage_bps: Slippage in basis points
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (new_position, total_cost, cost_breakdown)
        """
        # Target position from signal
        target_position = float(signal)
        
        # Position change
        position_change = target_position - current_position
        
        if abs(position_change) < 1e-8:
            # No trade
            return current_position, 0.0, {
                'spread_cost': 0.0,
                'slippage_cost': 0.0,
                'commission': 0.0,
                'total_cost': 0.0
            }
        
        # Compute individual cost components
        spread_cost = self.compute_spread_cost(position_change, spread)
        slippage_cost = self.compute_slippage_cost(position_change, slippage_bps)
        commission = self.compute_commission(position_change)
        
        total_cost = spread_cost + slippage_cost + commission
        
        cost_breakdown = {
            'spread_cost': float(spread_cost),
            'slippage_cost': float(slippage_cost),
            'commission': float(commission),
            'total_cost': float(total_cost)
        }
        
        return target_position, total_cost, cost_breakdown
    
    def compute_spread_cost(self, position_change: float, spread: float) -> float:
        """
        Compute spread cost for position change.
        
        Spread cost = |position_change| * min(spread, spread_cap)
        
        Args:
            position_change: Change in position size
            spread: Bid-ask spread
            
        Returns:
            Spread cost (always positive)
        """
        # Cap spread to prevent outlier impact
        spread_capped = min(abs(spread), self.spread_cap)
        
        # Cost proportional to position change
        cost = abs(position_change) * spread_capped
        
        return float(cost)
    
    def compute_slippage_cost(self, position_change: float, slippage_bps: float) -> float:
        """
        Compute slippage cost (volatility-aware).
        
        Uses two-regime model:
        - Low volatility: slippage_bps_lo
        - High volatility: slippage_bps_med
        
        Args:
            position_change: Change in position size
            slippage_bps: Slippage in basis points (from data)
            
        Returns:
            Slippage cost (always positive)
        """
        # Apply slippage factor
        effective_slippage_bps = slippage_bps * self.slippage_factor
        
        # Convert bps to fraction
        slippage_fraction = effective_slippage_bps / 10000.0
        
        # Cost proportional to position change
        cost = abs(position_change) * slippage_fraction
        
        return float(cost)
    
    def compute_commission(self, position_change: float) -> float:
        """
        Compute commission cost.
        
        Args:
            position_change: Change in position size
            
        Returns:
            Commission cost (always positive)
        """
        if self.commission_bps <= 0:
            return 0.0
        
        commission_fraction = self.commission_bps / 10000.0
        cost = abs(position_change) * commission_fraction
        
        return float(cost)
    
    def compute_pnl(
        self,
        position: float,
        returns: float,
        costs: float
    ) -> float:
        """
        Compute PnL for bar.
        
        PnL = position * returns - costs
        
        Args:
            position: Position size
            returns: Bar returns (log returns)
            costs: Total costs for bar
            
        Returns:
            Net PnL (can be negative)
        """
        gross_pnl = position * returns
        net_pnl = gross_pnl - costs
        
        return float(net_pnl)
    
    def apply_execution_delay(
        self,
        signals: np.ndarray,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply execution delay: signal at bar t executes at bar t+1.
        
        CRITICAL: This prevents look-ahead bias.
        - Signal generated at close of bar t
        - Execution happens at open of bar t+1
        - Returns earned from bar t+1
        
        Args:
            signals: Trading signals (n_bars,)
            returns: Bar returns (n_bars,)
            
        Returns:
            Tuple of (delayed_signals, aligned_returns)
        """
        if not self.use_execution_delay:
            return signals, returns
        
        # Shift signals forward by 1 bar
        delayed_signals = np.roll(signals, 1)
        delayed_signals[0] = 0  # No position on first bar
        
        return delayed_signals, returns
    
    def vectorized_backtest(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        spreads: np.ndarray,
        slippages: np.ndarray,
        initial_equity: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized backtest (fast path).
        
        CRITICAL: Applies execution delay automatically.
        
        Args:
            signals: Trading signals (n_bars,)
            returns: Bar returns (n_bars,)
            spreads: Bid-ask spreads (n_bars,)
            slippages: Slippage in bps (n_bars,)
            initial_equity: Starting equity
            
        Returns:
            Tuple of (positions, costs, equity_curve)
        """
        n_bars = len(signals)
        
        # Apply execution delay
        delayed_signals, aligned_returns = self.apply_execution_delay(signals, returns)
        
        # Initialize arrays
        positions = np.zeros(n_bars, dtype=np.float32)
        costs = np.zeros(n_bars, dtype=np.float32)
        equity = np.zeros(n_bars, dtype=np.float32)
        
        # First bar
        positions[0] = delayed_signals[0]
        equity[0] = initial_equity
        
        # Vectorized position changes
        position_changes = np.diff(delayed_signals, prepend=0)
        
        # Vectorized costs
        spread_costs = np.abs(position_changes) * np.minimum(spreads, self.spread_cap)
        slippage_costs = np.abs(position_changes) * (slippages * self.slippage_factor / 10000.0)
        commission_costs = np.abs(position_changes) * (self.commission_bps / 10000.0)
        
        costs = spread_costs + slippage_costs + commission_costs
        
        # Vectorized PnL
        gross_pnl = delayed_signals * aligned_returns
        net_pnl = gross_pnl - costs
        
        # Cumulative equity
        equity = initial_equity * np.exp(np.cumsum(net_pnl))
        
        # Update positions array
        positions = delayed_signals.astype(np.float32)
        
        return positions, costs.astype(np.float32), equity.astype(np.float32)
