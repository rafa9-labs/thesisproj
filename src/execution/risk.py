"""
Risk management: Stop Loss, Take Profit, Trailing Stops, and Position Sizing.

Provides modular risk management classes for backtesting and live trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class StaticStopLoss:
    """
    Static Stop Loss and Take Profit manager.
    
    Features:
    - Fixed SL/TP in pips
    - Symmetric or asymmetric levels
    - Per-position tracking
    """
    
    def __init__(self, sl_pips: float = 0.0, tp_pips: float = 0.0):
        """
        Initialize static SL/TP manager.
        
        Args:
            sl_pips: Stop loss in pips (0 = disabled)
            tp_pips: Take profit in pips (0 = disabled)
        """
        self.sl_pips = float(sl_pips)
        self.tp_pips = float(tp_pips)
        self.entry_price = None
        self.entry_position = None
        
        logger.info(f"StaticStopLoss initialized: SL={self.sl_pips} pips, TP={self.tp_pips} pips")
    
    def on_entry(self, entry_price: float, position: float) -> None:
        """
        Record entry for SL/TP tracking.
        
        Args:
            entry_price: Entry price
            position: Position size (-1, 0, 1)
        """
        self.entry_price = float(entry_price)
        self.entry_position = float(position)
    
    def check_exit(
        self,
        current_price: float,
        current_position: float
    ) -> Tuple[bool, str]:
        """
        Check if SL or TP hit.
        
        Args:
            current_price: Current market price
            current_position: Current position size
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.entry_price is None or abs(current_position) < 1e-8:
            return False, ""
        
        # Only check if position hasn't changed
        if abs(current_position - self.entry_position) > 1e-8:
            return False, ""
        
        price_change_pips = (current_price - self.entry_price) * 10000
        
        # Long position
        if current_position > 0:
            # Stop loss check
            if self.sl_pips > 0 and price_change_pips <= -self.sl_pips:
                return True, "SL_HIT"
            
            # Take profit check
            if self.tp_pips > 0 and price_change_pips >= self.tp_pips:
                return True, "TP_HIT"
        
        # Short position
        elif current_position < 0:
            # Stop loss check (inverse for short)
            if self.sl_pips > 0 and price_change_pips >= self.sl_pips:
                return True, "SL_HIT"
            
            # Take profit check (inverse for short)
            if self.tp_pips > 0 and price_change_pips <= -self.tp_pips:
                return True, "TP_HIT"
        
        return False, ""
    
    def reset(self) -> None:
        """Reset entry tracking."""
        self.entry_price = None
        self.entry_position = None


class TrailingStop:
    """
    Trailing stop manager.
    
    Features:
    - Activation threshold (profit before trailing starts)
    - Trail distance in pips
    - Automatic level updates
    - Per-position tracking
    
    NOTE: Trailing stops require bar-by-bar simulation (path-dependent).
    """
    
    def __init__(
        self,
        trail_pips: float = 0.0,
        activation_pips: float = 0.0
    ):
        """
        Initialize trailing stop manager.
        
        Args:
            trail_pips: Trailing distance in pips (0 = disabled)
            activation_pips: Profit threshold before trailing starts (0 = immediate)
        """
        self.trail_pips = float(trail_pips)
        self.activation_pips = float(activation_pips)
        self.trail_level = None
        self.entry_price = None
        self.entry_position = None
        self.activated = False
        
        logger.info(f"TrailingStop initialized: trail={self.trail_pips} pips, "
                   f"activation={self.activation_pips} pips")
    
    def on_entry(self, entry_price: float, position: float) -> None:
        """
        Record entry for trailing stop tracking.
        
        Args:
            entry_price: Entry price
            position: Position size (-1, 0, 1)
        """
        self.entry_price = float(entry_price)
        self.entry_position = float(position)
        self.trail_level = None
        self.activated = False
    
    def update(self, current_price: float, current_position: float) -> None:
        """
        Update trailing stop level.
        
        Args:
            current_price: Current market price
            current_position: Current position size
        """
        if self.entry_price is None or abs(current_position) < 1e-8:
            return
        
        # Only trail if position hasn't changed
        if abs(current_position - self.entry_position) > 1e-8:
            return
        
        price_change_pips = (current_price - self.entry_price) * 10000
        
        # Check activation
        if not self.activated:
            if abs(price_change_pips) >= self.activation_pips:
                self.activated = True
            else:
                return
        
        # Long position: trail stop upward
        if current_position > 0:
            new_trail_level = current_price - (self.trail_pips / 10000)
            
            if self.trail_level is None:
                self.trail_level = new_trail_level
            else:
                # Only move stop up, never down
                self.trail_level = max(self.trail_level, new_trail_level)
        
        # Short position: trail stop downward
        elif current_position < 0:
            new_trail_level = current_price + (self.trail_pips / 10000)
            
            if self.trail_level is None:
                self.trail_level = new_trail_level
            else:
                # Only move stop down, never up
                self.trail_level = min(self.trail_level, new_trail_level)
    
    def check_exit(
        self,
        current_price: float,
        current_position: float
    ) -> Tuple[bool, str]:
        """
        Check if trailing stop hit.
        
        Args:
            current_price: Current market price
            current_position: Current position size
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.trail_level is None or not self.activated:
            return False, ""
        
        # Long position: exit if price drops below trail level
        if current_position > 0:
            if current_price <= self.trail_level:
                return True, "TRAIL_STOP_HIT"
        
        # Short position: exit if price rises above trail level
        elif current_position < 0:
            if current_price >= self.trail_level:
                return True, "TRAIL_STOP_HIT"
        
        return False, ""
    
    def reset(self) -> None:
        """Reset trailing stop state."""
        self.trail_level = None
        self.entry_price = None
        self.entry_position = None
        self.activated = False


class PositionSizer:
    """
    Position sizing calculator.
    
    Methods:
    - Fixed: Constant position size
    - Risk Percentage: Size based on account risk and stop loss
    - Volatility-based: Size inversely proportional to ATR
    """
    
    def __init__(self, method: str = "fixed", **kwargs):
        """
        Initialize position sizer.
        
        Args:
            method: Sizing method ("fixed", "risk_pct", "volatility")
            **kwargs: Method-specific parameters:
                - fixed_size: Position size for "fixed" method (default 1.0)
                - risk_pct: Risk percentage for "risk_pct" method (default 0.02)
                - vol_target: Volatility target for "volatility" method (default 0.01)
        """
        self.method = method.lower()
        
        # Fixed size parameters
        self.fixed_size = float(kwargs.get('fixed_size', 1.0))
        
        # Risk percentage parameters
        self.risk_pct = float(kwargs.get('risk_pct', 0.02))
        
        # Volatility-based parameters
        self.vol_target = float(kwargs.get('vol_target', 0.01))
        
        logger.info(f"PositionSizer initialized: method={self.method}")
    
    def calculate_size(
        self,
        signal: int,
        account_equity: float = 1.0,
        atr: Optional[float] = None,
        sl_pips: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate position size based on method.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            account_equity: Current account equity
            atr: Average True Range (for volatility-based sizing)
            sl_pips: Stop loss in pips (for risk-based sizing)
            **kwargs: Additional parameters
            
        Returns:
            Position size (signed: -1, 0, 1 or fractional)
        """
        if signal == 0:
            return 0.0
        
        if self.method == "fixed":
            return float(signal) * self.fixed_size
        
        elif self.method == "risk_pct":
            return self._calculate_risk_based_size(signal, account_equity, sl_pips)
        
        elif self.method == "volatility":
            return self._calculate_volatility_based_size(signal, atr)
        
        else:
            logger.warning(f"Unknown sizing method: {self.method}, using fixed")
            return float(signal) * self.fixed_size
    
    def _calculate_risk_based_size(
        self,
        signal: int,
        account_equity: float,
        sl_pips: Optional[float]
    ) -> float:
        """
        Calculate position size based on risk percentage.
        
        Size = (Account Equity * Risk %) / Stop Loss
        
        Args:
            signal: Trading signal
            account_equity: Current account equity
            sl_pips: Stop loss in pips
            
        Returns:
            Position size
        """
        if sl_pips is None or sl_pips <= 0:
            # No SL defined, use fixed size
            return float(signal) * self.fixed_size
        
        # Risk amount in account currency
        risk_amount = account_equity * self.risk_pct
        
        # Stop loss in price units
        sl_price = sl_pips / 10000.0
        
        # Position size
        size = risk_amount / sl_price
        
        # Apply signal direction
        return float(signal) * size
    
    def _calculate_volatility_based_size(
        self,
        signal: int,
        atr: Optional[float]
    ) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Size = Volatility Target / ATR
        
        Args:
            signal: Trading signal
            atr: Average True Range
            
        Returns:
            Position size
        """
        if atr is None or atr <= 0:
            # No ATR available, use fixed size
            return float(signal) * self.fixed_size
        
        # Size inversely proportional to volatility
        size = self.vol_target / atr
        
        # Clip to reasonable range
        size = np.clip(size, 0.1, 10.0)
        
        # Apply signal direction
        return float(signal) * size
