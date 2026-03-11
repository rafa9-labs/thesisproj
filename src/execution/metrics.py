"""
PerformanceEvaluator: Compute 16 standard trading metrics.

Extracted from utilsNoWFO.py for modularity and testability.

Metrics (in order):
1. cstrategy - Cumulative strategy return
2. outperformance - Strategy vs. buy-and-hold
3. creturns - Cumulative buy-and-hold return
4. sharpe - Annualized Sharpe ratio (HAC-adjusted)
5. drawdown - Maximum drawdown
6. trades - Number of trades
7. geo_mean_ann - Annualized geometric mean return
8. directional_accuracy - Hit rate
9. precision_macro - Macro-averaged precision
10. f1_macro - Macro-averaged F1 score
11. active_rate - Fraction of time in market
12. profit_per_hit - Profit per correct prediction
13. return_per_trade - Average return per trade
14. win_rate - Fraction of winning trades
15. strategy_volatility - Strategy volatility
16. kurtosis - Excess kurtosis
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging
from scipy.stats import kurtosis
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)


# Canonical metric names (exactly 16)
METRIC_NAMES = [
    "cstrategy", "outperformance", "creturns", "sharpe", "drawdown", "trades",
    "geo_mean_ann", "directional_accuracy", "precision_macro", "f1_macro",
    "active_rate", "profit_per_hit", "return_per_trade", "win_rate",
    "strategy_volatility", "kurtosis",
]

N_METRICS = 16


class PerformanceEvaluator:
    """
    Performance metrics calculator for trading strategies.
    
    Computes all 16 standard metrics with robust guards:
    - HAC-adjusted Sharpe ratio
    - Reliability thresholds (min trades)
    - Annualization based on data frequency
    - Classification metrics (precision, F1)
    """
    
    def __init__(self, config: dict):
        """
        Initialize performance evaluator.
        
        Args:
            config: Dictionary with evaluation parameters:
                - sharpe_cap: Maximum Sharpe ratio (default 100.0)
                - min_trades_for_reliability: Min trades for valid Sharpe (default 30)
                - use_hac: Use HAC-adjusted std for Sharpe (default True)
                - hac_max_lag: HAC lag selection ("auto", "andrews", or int)
        """
        self.config = config
        self.sharpe_cap = float(config.get('sharpe_cap', 100.0))
        self.min_trades_for_reliability = int(config.get('min_trades_for_reliability', 30))
        self.use_hac = bool(config.get('use_hac', True))
        self.hac_max_lag = config.get('hac_max_lag', 'auto')
        
        logger.info(f"PerformanceEvaluator initialized: sharpe_cap={self.sharpe_cap}, "
                   f"min_trades={self.min_trades_for_reliability}")
    
    def compute_all_metrics(
        self,
        df: pd.DataFrame,
        returns_col: str = "strategy",
        positions_col: str = "position_exec",
        predictions_col: str = "pred",
        true_direction_col: str = "true_direction"
    ) -> Tuple[float, ...]:
        """
        Compute all 16 standard metrics.
        
        Args:
            df: DataFrame with backtest results
            returns_col: Column name for strategy returns
            positions_col: Column name for executed positions
            predictions_col: Column name for predictions
            true_direction_col: Column name for true direction
            
        Returns:
            Tuple of 16 metrics in canonical order
        """
        # Extract series
        strategy_returns = df[returns_col].dropna()
        market_returns = df.get('returns', pd.Series([], dtype=float)).dropna()
        positions = df.get(positions_col, pd.Series([], dtype=float))
        predictions = df.get(predictions_col, pd.Series([], dtype=float))
        true_direction = df.get(true_direction_col, pd.Series([], dtype=float))
        
        # 1-3: Cumulative returns
        cstrategy = float(np.exp(strategy_returns.sum()))
        creturns = float(np.exp(market_returns.sum())) if len(market_returns) > 0 else 1.0
        outperformance = cstrategy / creturns if creturns > 0 else 1.0
        
        # 4-6: Sharpe, drawdown, trades
        sharpe, drawdown, trades = self.compute_sharpe_drawdown_trades(
            strategy_returns, positions
        )
        
        # 7: Geometric mean annualized
        geo_mean_ann = self.compute_geometric_mean_annualized(strategy_returns)
        
        # 8-10: Classification metrics
        directional_accuracy, precision_macro, f1_macro = self.compute_classification_metrics(
            true_direction, predictions
        )
        
        # 11: Active rate
        active_rate = self.compute_active_rate(positions)
        
        # 12: Profit per hit
        profit_per_hit = self.compute_profit_per_hit(df, true_direction, predictions, returns_col)
        
        # 13: Return per trade
        return_per_trade = (cstrategy - 1.0) / trades if trades > 0 else 0.0
        
        # 14: Win rate
        win_rate = self.compute_win_rate(df, predictions, returns_col)
        
        # 15: Strategy volatility
        strategy_volatility = float(np.std(strategy_returns))
        
        # 16: Excess kurtosis
        excess_kurtosis = float(kurtosis(strategy_returns, fisher=True))
        
        return (
            round(cstrategy, 6),
            round(outperformance, 6),
            round(creturns, 6),
            sharpe,
            drawdown,
            trades,
            round(geo_mean_ann, 6),
            round(directional_accuracy, 4),
            round(precision_macro, 4),
            round(f1_macro, 4),
            round(active_rate, 4),
            round(profit_per_hit, 6),
            round(return_per_trade, 6),
            round(win_rate, 4),
            round(strategy_volatility, 6),
            round(excess_kurtosis, 4)
        )
    
    def compute_sharpe_drawdown_trades(
        self,
        returns: pd.Series,
        positions: pd.Series
    ) -> Tuple[float, float, int]:
        """
        Compute Sharpe ratio, max drawdown, and trade count.
        
        Args:
            returns: Strategy returns
            positions: Position series
            
        Returns:
            Tuple of (sharpe, drawdown, trades)
        """
        # Sharpe ratio
        sharpe = self.compute_sharpe(returns)
        
        # Max drawdown
        drawdown = self.compute_drawdown(returns)
        
        # Trade count
        trades = self.compute_trade_count(positions)
        
        # Reliability guard: if too few trades, Sharpe is unreliable
        if trades < self.min_trades_for_reliability:
            sharpe = float('nan')
        
        return sharpe, drawdown, trades
    
    def compute_sharpe(
        self,
        returns: pd.Series,
        use_hac: Optional[bool] = None
    ) -> float:
        """
        Compute annualized Sharpe ratio with HAC adjustment.
        
        Args:
            returns: Strategy returns
            use_hac: Use HAC-adjusted std (default from config)
            
        Returns:
            Annualized Sharpe ratio
        """
        if use_hac is None:
            use_hac = self.use_hac
        
        returns = returns.dropna()
        
        # Estimate frequency
        try:
            frequency_per_year = self._estimate_frequency_per_year(returns.index)
        except Exception:
            frequency_per_year = 252.0
        
        ann_factor = float(np.sqrt(max(1.0, frequency_per_year)))
        
        # Filter active returns
        active = returns[np.abs(returns) > 1e-12]
        n_active = len(active)
        
        if n_active < 25:
            return 0.0
        
        # Compute std
        if use_hac:
            std = self._hac_std(active, max_lag=self.hac_max_lag)
        else:
            std = float(active.std(ddof=1))
        
        mean = float(active.mean())
        
        if not np.isfinite(std) or std < 1e-8:
            return 0.0
        
        sharpe = (mean / std) * ann_factor
        
        # Apply cap
        if self.sharpe_cap > 0:
            sharpe = float(np.clip(sharpe, -self.sharpe_cap, self.sharpe_cap))
        
        return round(sharpe, 2)
    
    def compute_drawdown(self, returns: pd.Series) -> float:
        """
        Compute maximum drawdown.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Maximum drawdown (negative value)
        """
        cum = returns.cumsum().apply(np.exp)
        
        if cum.empty:
            return 0.0
        
        drawdown = (cum / cum.cummax() - 1).min()
        
        return round(float(drawdown), 4)
    
    def compute_trade_count(self, positions: pd.Series) -> int:
        """
        Compute number of directional trades.
        
        Args:
            positions: Position series
            
        Returns:
            Number of trades
        """
        try:
            if positions is None or len(positions) == 0:
                return 0
            
            p = positions.fillna(0).values
            
            if len(p) <= 1:
                return 0
            
            # Count directional changes
            p_dir = np.sign(p)
            trades = int(np.sum(np.abs(np.diff(p_dir))))
            
            return trades
        except Exception:
            return 0
    
    def compute_geometric_mean_annualized(self, returns: pd.Series) -> float:
        """
        Compute annualized geometric mean return.
        
        Args:
            returns: Strategy returns (log returns)
            
        Returns:
            Annualized geometric mean
        """
        n = len(returns)
        if n == 0:
            return 0.0
        
        compounded = np.exp(returns.sum())
        
        try:
            bars_per_year = self._estimate_frequency_per_year(returns.index)
        except Exception:
            bars_per_year = 252.0
        
        annual_factor = bars_per_year / max(1, n)
        
        return float(compounded ** annual_factor - 1)
    
    def compute_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Compute directional accuracy, precision, and F1.
        
        Args:
            y_true: True direction
            y_pred: Predicted direction
            
        Returns:
            Tuple of (directional_accuracy, precision_macro, f1_macro)
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0, 0.0, 0.0
        
        # Coerce to discrete labels
        y_true_discrete = self._coerce_direction_labels(y_true.values)
        y_pred_discrete = self._coerce_direction_labels(y_pred.values)
        
        # Directional accuracy
        hit_mask = (y_pred_discrete == y_true_discrete)
        directional_accuracy = float(hit_mask.mean())
        
        # Precision and F1 (macro-averaged)
        precision_macro, f1_macro = self._macro_prec_f1_from_confusion(
            y_true_discrete, y_pred_discrete
        )
        
        return directional_accuracy, precision_macro, f1_macro
    
    def compute_active_rate(self, positions: pd.Series) -> float:
        """
        Compute fraction of time in market.
        
        Args:
            positions: Position series
            
        Returns:
            Active rate (0 to 1)
        """
        if positions is None or len(positions) == 0:
            return 0.0
        
        active = (positions.fillna(0).values != 0)
        return float(active.mean())
    
    def compute_profit_per_hit(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_pred: pd.Series,
        returns_col: str
    ) -> float:
        """
        Compute profit per correct prediction.
        
        Args:
            df: DataFrame with returns
            y_true: True direction
            y_pred: Predicted direction
            returns_col: Column name for returns
            
        Returns:
            Profit per hit
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        
        y_true_discrete = self._coerce_direction_labels(y_true.values)
        y_pred_discrete = self._coerce_direction_labels(y_pred.values)
        
        hit_mask = pd.Series(y_pred_discrete == y_true_discrete, index=df.index)
        
        correct_returns = float(df.loc[hit_mask, returns_col].sum()) if len(df) > 0 else 0.0
        hits = int(hit_mask.sum())
        
        return float(correct_returns) / float(hits if hits > 0 else 1)
    
    def compute_win_rate(
        self,
        df: pd.DataFrame,
        predictions: pd.Series,
        returns_col: str
    ) -> float:
        """
        Compute win rate from trade returns.
        
        Args:
            df: DataFrame with returns
            predictions: Prediction series
            returns_col: Column name for returns
            
        Returns:
            Win rate (0 to 1)
        """
        if len(predictions) == 0:
            return 0.0
        
        pred_discrete = self._coerce_direction_labels(predictions.values)
        pred_series = pd.Series(pred_discrete, index=df.index)
        
        # Identify trade edges
        trade_edge = pred_series.diff().fillna(0) != 0
        trade_returns = df[returns_col][trade_edge]
        
        if len(trade_returns) == 0:
            return 0.0
        
        num_wins = int((trade_returns > 0).sum())
        num_trades = len(trade_returns)
        
        return float(num_wins) / float(num_trades)
    
    @staticmethod
    def _coerce_direction_labels(arr, deadzone: float = 0.5):
        """Coerce predictions to discrete {-1, 0, 1} labels."""
        a = np.asarray(arr)
        if a.size == 0:
            return a.astype(int)
        
        if a.dtype.kind in ('f', 'c'):
            a = np.where(a > deadzone, 1, np.where(a < -deadzone, -1, 0))
        else:
            try:
                a = a.astype(int, copy=False)
            except Exception:
                a = a.astype(float)
                a = np.where(a > deadzone, 1, np.where(a < -deadzone, -1, 0))
        
        # Sanitize unexpected labels
        valid = np.isin(a, np.array([-1, 0, 1], dtype=int))
        if not np.all(valid):
            a = np.where(valid, a, 0)
        
        return a.astype(int, copy=False)
    
    @staticmethod
    def _macro_prec_f1_from_confusion(y_true, y_pred):
        """Compute macro-averaged precision and F1 from confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
            
            # Per-class precision and recall
            precisions = []
            recalls = []
            
            for i in range(3):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                precisions.append(prec)
                recalls.append(rec)
            
            # Macro averages
            precision_macro = float(np.mean(precisions))
            recall_macro = float(np.mean(recalls))
            
            # F1 from macro precision and recall
            if precision_macro + recall_macro > 0:
                f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
            else:
                f1_macro = 0.0
            
            return precision_macro, f1_macro
        except Exception:
            return 0.0, 0.0
    
    @staticmethod
    def _estimate_frequency_per_year(index) -> float:
        """Estimate bars per year from DateTimeIndex."""
        if not hasattr(index, 'tz'):
            try:
                index = pd.to_datetime(index, utc=True, errors='coerce')
            except Exception:
                return 252.0
        
        if len(index) < 3:
            return 252.0
        
        by_day = pd.Series(1.0, index=index).groupby(index.floor('D')).count()
        if by_day.empty:
            return 252.0
        
        bars_per_day = float(by_day.median())
        
        # Detect weekends
        days = pd.Index(by_day.index)
        weekend_days = int(((days.dayofweek == 5) | (days.dayofweek == 6)).sum())
        frac_weekend = weekend_days / max(1, len(days))
        days_per_year = 365.0 if frac_weekend > 0.10 else 252.0
        
        return max(1.0, bars_per_day * days_per_year)
    
    @staticmethod
    def _hac_std(x, max_lag='auto') -> float:
        """Newey-West HAC standard deviation."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = x.size
        
        if n <= 1:
            return 0.0
        
        x = x - np.mean(x)
        g0 = np.dot(x, x) / n
        
        # Determine lag
        if isinstance(max_lag, str):
            if max_lag == 'auto':
                q = int(np.floor(np.sqrt(n)))
            else:
                q = int(np.floor(np.sqrt(n)))
        else:
            q = int(max(0, max_lag))
        
        if q == 0:
            var = g0
        else:
            var = g0
            for k in range(1, q + 1):
                w = 1.0 - k / (q + 1.0)
                gamma_k = np.dot(x[:-k], x[k:]) / n
                var += 2.0 * w * gamma_k
        
        return float(np.sqrt(max(var, 0.0)))
