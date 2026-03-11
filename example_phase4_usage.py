"""
Example usage of Phase 4: Execution & Backtesting Engine

Demonstrates:
1. Trade execution with cost breakdown
2. Static SL/TP risk management
3. Trailing stop functionality
4. Position sizing (fixed vs. risk-based)
5. Vectorized backtest (fast path)
6. Bar-by-bar backtest (with trailing stops)
7. Full backtest with 16 metrics
8. Integration with Phase 3 ModelTrainer
9. Execution delay enforcement
10. Float32 equity curves for UI
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.execution import (
    TradeSimulator,
    StaticStopLoss,
    TrailingStop,
    PositionSizer,
    PerformanceEvaluator,
    METRIC_NAMES,
    BacktestEngine
)


def configure_logging():
    """Setup logging for examples"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def example_trade_execution():
    """Example: Trade execution with cost breakdown"""
    print("=" * 60)
    print("EXAMPLE 1: Trade Execution with Cost Breakdown")
    print("=" * 60)
    
    config = {
        'spread_cap': 0.0004,
        'slippage_factor': 1.0,
        'slippage_bps_lo': 0.08,
        'commission_bps': 0.5,
        'use_execution_delay': True
    }
    
    simulator = TradeSimulator(config)
    
    # Execute a trade
    signal = 1  # Long
    current_position = 0.0
    price = 1.1000
    spread = 0.0002
    slippage_bps = 0.1
    
    new_position, total_cost, cost_breakdown = simulator.execute_trade(
        signal=signal,
        current_position=current_position,
        price=price,
        spread=spread,
        slippage_bps=slippage_bps
    )
    
    print(f"✓ Trade executed:")
    print(f"  Signal: {signal}")
    print(f"  New position: {new_position}")
    print(f"  Total cost: {total_cost:.6f}")
    print(f"  Breakdown:")
    print(f"    - Spread cost: {cost_breakdown['spread_cost']:.6f}")
    print(f"    - Slippage cost: {cost_breakdown['slippage_cost']:.6f}")
    print(f"    - Commission: {cost_breakdown['commission']:.6f}")
    print()


def example_static_sl_tp():
    """Example: Static Stop Loss and Take Profit"""
    print("=" * 60)
    print("EXAMPLE 2: Static SL/TP Risk Management")
    print("=" * 60)
    
    sl_tp = StaticStopLoss(sl_pips=20.0, tp_pips=40.0)
    
    # Entry
    entry_price = 1.1000
    position = 1.0  # Long
    sl_tp.on_entry(entry_price, position)
    
    print(f"✓ Entered long at {entry_price}")
    print(f"  SL: {entry_price - 20/10000:.4f} (-20 pips)")
    print(f"  TP: {entry_price + 40/10000:.4f} (+40 pips)")
    
    # Check various prices
    test_prices = [1.0980, 1.1020, 1.1040]
    for price in test_prices:
        should_exit, reason = sl_tp.check_exit(price, position)
        pips_change = (price - entry_price) * 10000
        
        if should_exit:
            print(f"  Price {price:.4f} ({pips_change:+.1f} pips): EXIT - {reason}")
        else:
            print(f"  Price {price:.4f} ({pips_change:+.1f} pips): Hold")
    
    print()


def example_trailing_stop():
    """Example: Trailing stop functionality"""
    print("=" * 60)
    print("EXAMPLE 3: Trailing Stop")
    print("=" * 60)
    
    trail_stop = TrailingStop(trail_pips=15.0, activation_pips=20.0)
    
    # Entry
    entry_price = 1.1000
    position = 1.0  # Long
    trail_stop.on_entry(entry_price, position)
    
    print(f"✓ Entered long at {entry_price}")
    print(f"  Trail: 15 pips, Activation: 20 pips")
    
    # Simulate price movement
    price_sequence = [1.1010, 1.1025, 1.1030, 1.1020, 1.1010]
    
    for price in price_sequence:
        # Update trailing stop
        trail_stop.update(price, position)
        
        # Check exit
        should_exit, reason = trail_stop.check_exit(price, position)
        
        pips_change = (price - entry_price) * 10000
        trail_level = trail_stop.trail_level
        
        if should_exit:
            print(f"  Price {price:.4f} ({pips_change:+.1f} pips): EXIT - {reason}")
            break
        else:
            if trail_level:
                print(f"  Price {price:.4f} ({pips_change:+.1f} pips): Trail at {trail_level:.4f}")
            else:
                print(f"  Price {price:.4f} ({pips_change:+.1f} pips): Not activated yet")
    
    print()


def example_position_sizing():
    """Example: Position sizing methods"""
    print("=" * 60)
    print("EXAMPLE 4: Position Sizing")
    print("=" * 60)
    
    # Fixed size
    sizer_fixed = PositionSizer(method='fixed', fixed_size=1.0)
    size = sizer_fixed.calculate_size(signal=1, account_equity=10000)
    print(f"✓ Fixed sizing: {size}")
    
    # Risk percentage
    sizer_risk = PositionSizer(method='risk_pct', risk_pct=0.02)
    size = sizer_risk.calculate_size(signal=1, account_equity=10000, sl_pips=20.0)
    print(f"✓ Risk-based sizing (2% risk, 20 pip SL): {size:.2f}")
    
    # Volatility-based
    sizer_vol = PositionSizer(method='volatility', vol_target=0.01)
    size = sizer_vol.calculate_size(signal=1, atr=0.0015)
    print(f"✓ Volatility-based sizing (ATR=0.0015): {size:.2f}")
    
    print()


def example_vectorized_backtest():
    """Example: Vectorized backtest (fast path)"""
    print("=" * 60)
    print("EXAMPLE 5: Vectorized Backtest (Fast Path)")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_bars = 1000
    
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='h')
    returns = np.random.randn(n_bars) * 0.001
    spreads = np.full(n_bars, 0.0001)
    slippages = np.full(n_bars, 0.1)
    
    df_data = pd.DataFrame({
        'returns': returns,
        'spread': spreads,
        'slippage_bps': slippages,
        'price': 1.1 + np.cumsum(returns)
    }, index=dates)
    
    # Create predictions (simple trend-following)
    predictions = np.where(returns > 0, 1, np.where(returns < 0, -1, 0))
    probabilities = np.random.dirichlet([1, 1, 1], n_bars).astype(np.float32)
    
    # Run backtest
    config = {
        'spread_cap': 0.0004,
        'slippage_factor': 1.0,
        'use_execution_delay': True,
        'sharpe_cap': 30.0,
        'min_trades_for_reliability': 30
    }
    
    engine = BacktestEngine(config)
    
    results_df, metrics = engine.run_backtest(
        predictions=predictions,
        probabilities=probabilities,
        df_data=df_data
    )
    
    print(f"✓ Vectorized backtest complete:")
    print(f"  Bars: {len(results_df)}")
    print(f"  Final equity: {metrics['final_equity']:.4f}")
    print(f"  Sharpe ratio: {metrics['sharpe']:.2f}")
    print(f"  Max drawdown: {metrics['drawdown']:.2%}")
    print(f"  Trades: {metrics['trades']}")
    print(f"  Win rate: {metrics['win_rate']:.2%}")
    print(f"  Active rate: {metrics['active_rate']:.2%}")
    
    # Check equity curve type
    equity_curve = engine.build_equity_curve(results_df)
    print(f"  Equity curve dtype: {equity_curve.dtype} (float32 for UI)")
    
    print()


def example_bar_by_bar_with_trailing():
    """Example: Bar-by-bar backtest with trailing stops"""
    print("=" * 60)
    print("EXAMPLE 6: Bar-by-Bar with Trailing Stops")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_bars = 500
    
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='h')
    returns = np.random.randn(n_bars) * 0.001
    
    df_data = pd.DataFrame({
        'returns': returns,
        'spread': 0.0001,
        'slippage_bps': 0.1,
        'price': 1.1 + np.cumsum(returns)
    }, index=dates)
    
    # Create predictions
    predictions = np.where(returns > 0, 1, np.where(returns < 0, -1, 0))
    probabilities = np.random.dirichlet([1, 1, 1], n_bars).astype(np.float32)
    
    # Create trailing stop
    trail_stop = TrailingStop(trail_pips=15.0, activation_pips=10.0)
    
    # Run backtest with trailing stop
    config = {
        'spread_cap': 0.0004,
        'slippage_factor': 1.0,
        'use_execution_delay': True
    }
    
    engine = BacktestEngine(config, risk_manager=trail_stop)
    
    print(f"✓ Execution mode: {'Vectorized' if engine.use_vectorized else 'Bar-by-bar'}")
    
    results_df, metrics = engine.run_backtest(
        predictions=predictions,
        probabilities=probabilities,
        df_data=df_data
    )
    
    print(f"✓ Backtest with trailing stops complete:")
    print(f"  Final equity: {metrics['final_equity']:.4f}")
    print(f"  Sharpe ratio: {metrics['sharpe']:.2f}")
    print(f"  Trades: {metrics['trades']}")
    
    print()


def example_full_metrics():
    """Example: All 16 standard metrics"""
    print("=" * 60)
    print("EXAMPLE 7: All 16 Standard Metrics")
    print("=" * 60)
    
    # Create synthetic backtest results
    np.random.seed(42)
    n_bars = 1000
    
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='h')
    returns = np.random.randn(n_bars) * 0.001
    
    df_results = pd.DataFrame({
        'returns': returns,
        'strategy': returns * 0.8,  # Strategy slightly worse than market
        'position_exec': np.random.choice([-1, 0, 1], n_bars),
        'pred': np.random.choice([-1, 0, 1], n_bars),
        'true_direction': np.sign(returns)
    }, index=dates)
    
    # Compute metrics
    config = {'sharpe_cap': 30.0, 'min_trades_for_reliability': 30}
    evaluator = PerformanceEvaluator(config)
    
    metrics_tuple = evaluator.compute_all_metrics(df_results)
    
    print(f"✓ All 16 metrics computed:")
    for name, value in zip(METRIC_NAMES, metrics_tuple):
        if isinstance(value, float):
            if abs(value) < 0.01:
                print(f"  {name:25s}: {value:.6f}")
            else:
                print(f"  {name:25s}: {value:.4f}")
        else:
            print(f"  {name:25s}: {value}")
    
    print()


def example_execution_delay():
    """Example: Execution delay enforcement"""
    print("=" * 60)
    print("EXAMPLE 8: Execution Delay (No Look-Ahead Bias)")
    print("=" * 60)
    
    # Create simple data
    signals = np.array([0, 1, 1, 0, -1, -1, 0])
    returns = np.array([0.01, 0.02, -0.01, 0.015, -0.02, 0.01, 0.005])
    
    config = {'use_execution_delay': True}
    simulator = TradeSimulator(config)
    
    # Apply execution delay
    delayed_signals, aligned_returns = simulator.apply_execution_delay(signals, returns)
    
    print(f"✓ Execution delay applied:")
    print(f"  Original signals:  {signals}")
    print(f"  Delayed signals:   {delayed_signals}")
    print(f"  Returns:           {returns}")
    print()
    print(f"  CRITICAL: Signal at bar t executes at bar t+1")
    print(f"  This prevents trading on prices we don't have yet!")
    
    print()


def example_trade_log():
    """Example: Trade log generation"""
    print("=" * 60)
    print("EXAMPLE 9: Trade Log Generation")
    print("=" * 60)
    
    # Create synthetic backtest results
    np.random.seed(42)
    n_bars = 100
    
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='h')
    
    # Create position sequence with some trades
    positions = np.zeros(n_bars)
    positions[10:30] = 1.0  # Long trade
    positions[40:60] = -1.0  # Short trade
    positions[70:85] = 1.0  # Another long
    
    returns = np.random.randn(n_bars) * 0.001
    
    df_results = pd.DataFrame({
        'position_exec': positions,
        'strategy': np.cumsum(positions * returns),
        'price': 1.1 + np.cumsum(returns)
    }, index=dates)
    
    # Generate trade log
    config = {}
    engine = BacktestEngine(config)
    
    trade_log = engine.generate_trade_log(df_results)
    
    print(f"✓ Trade log generated: {len(trade_log)} trades")
    print(trade_log.to_string())
    
    print()


def example_integration_with_phase3():
    """Example: Integration with Phase 3 ModelTrainer"""
    print("=" * 60)
    print("EXAMPLE 10: Integration with Phase 3 ModelTrainer")
    print("=" * 60)
    
    print("Complete pipeline demonstration:")
    print()
    
    # Simulate Phase 3 output
    print("1. Phase 3: Train model with ModelTrainer")
    print("   → strategy.predict_proba(X_test) → probabilities")
    print()
    
    # Create synthetic model output
    np.random.seed(42)
    n_bars = 500
    
    probabilities = np.random.dirichlet([1, 1, 1], n_bars).astype(np.float32)
    predictions = np.argmax(probabilities, axis=1) - 1  # Convert to {-1, 0, 1}
    
    print(f"2. Model predictions: {predictions[:10]}")
    print(f"   Shape: {predictions.shape}")
    print()
    
    # Create data
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='h')
    returns = np.random.randn(n_bars) * 0.001
    
    df_data = pd.DataFrame({
        'returns': returns,
        'spread': 0.0001,
        'slippage_bps': 0.1,
        'price': 1.1 + np.cumsum(returns)
    }, index=dates)
    
    print("3. Phase 4: Run backtest with BacktestEngine")
    
    config = {
        'spread_cap': 0.0004,
        'slippage_factor': 1.0,
        'use_execution_delay': True,
        'sharpe_cap': 30.0
    }
    
    engine = BacktestEngine(config)
    
    results_df, metrics = engine.run_backtest(
        predictions=predictions,
        probabilities=probabilities,
        df_data=df_data
    )
    
    print(f"   ✓ Backtest complete")
    print()
    
    print("4. Results:")
    print(f"   Final equity: {metrics['final_equity']:.4f}")
    print(f"   Total return: {metrics['total_return_pct']:.2f}%")
    print(f"   Sharpe ratio: {metrics['sharpe']:.2f}")
    print(f"   Max drawdown: {metrics['drawdown']:.2%}")
    print(f"   Trades: {metrics['trades']}")
    print(f"   Win rate: {metrics['win_rate']:.2%}")
    print()
    
    print("5. Float32 equity curve for UI:")
    equity_curve = engine.build_equity_curve(results_df)
    print(f"   Dtype: {equity_curve.dtype}")
    print(f"   Memory: {equity_curve.memory_usage() / 1024:.2f} KB")
    print(f"   (vs float64: {equity_curve.astype(np.float64).memory_usage() / 1024:.2f} KB)")
    
    print()


def main():
    """Run all examples"""
    configure_logging()
    
    print("\n" + "=" * 60)
    print("PHASE 4: EXECUTION & BACKTESTING ENGINE - EXAMPLES")
    print("=" * 60 + "\n")
    
    example_trade_execution()
    example_static_sl_tp()
    example_trailing_stop()
    example_position_sizing()
    example_vectorized_backtest()
    example_bar_by_bar_with_trailing()
    example_full_metrics()
    example_execution_delay()
    example_trade_log()
    example_integration_with_phase3()
    
    print("=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Trade execution with cost breakdown (spread, slippage, commission)")
    print("✓ Static SL/TP risk management")
    print("✓ Trailing stops (path-dependent)")
    print("✓ Position sizing (fixed, risk-based, volatility-based)")
    print("✓ Vectorized backtest (fast path)")
    print("✓ Bar-by-bar backtest (for trailing stops)")
    print("✓ All 16 standard metrics (Sharpe, drawdown, win rate, etc.)")
    print("✓ Execution delay enforcement (no look-ahead bias)")
    print("✓ Trade log generation")
    print("✓ Float32 equity curves (UI-ready)")
    print("✓ Integration with Phase 3 ModelTrainer")
    print("\nNext Steps:")
    print("1. Integrate with your actual forex data")
    print("2. Run full backtests with trained models")
    print("3. Optimize risk management parameters")
    print("4. Ready for production deployment")
    print()


if __name__ == "__main__":
    main()
