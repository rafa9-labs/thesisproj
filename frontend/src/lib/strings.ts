const STRINGS = {
  nav: {
    dashboard: "DASHBOARD",
    backtest: "BACKTEST",
    results: "RESULTS",
    compare: "COMPARE",
    news: "NEWS",
    settings: "SETTINGS",
  },
  actions: {
    deploy: "DEPLOY BACKTEST",
    terminate: "TERMINATE RUN",
    export: "EXPORT",
    load: "LOAD",
    runFirst: "Run First Backtest",
  },
  labels: {
    totalRuns: "Total Runs",
    bestSharpe: "Best Sharpe",
    avgWinRate: "Avg Win Rate",
    bestReturn: "Best Return",
    sharpe: "Sharpe Ratio",
    maxDD: "Max Drawdown",
    totalReturn: "Total Return",
    winRate: "Win Rate",
    trades: "Total Trades",
    activeRate: "Active Rate",
    f1Macro: "F1 Macro",
    outperformance: "Outperformance",
  },
  status: {
    pending: "Pending",
    running: "Running",
    completed: "Completed",
    failed: "Failed",
  },
  empty: {
    noRuns: "No backtests yet",
    noResults: "No results to display",
    noComparison: "No comparison data",
  },
} as const;

export default STRINGS;
