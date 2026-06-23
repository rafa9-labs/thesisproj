import { describe, it, expect } from "vitest";
import type {
  Metrics,
  TradeRecord,
  MonthlyResult,
  HpoParamImportance,
  HpoTrial,
} from "@/api/schemas";

describe("API Schema types — runtime shape validation", () => {
  it("Metrics accepts full object", () => {
    const m: Metrics = {
      model: "xgboost",
      sharpe: 1.47,
      sortino: 2.31,
      max_drawdown: -0.085,
      total_return_pct: 0.124,
      cagr: null,
      calmar_ratio: null,
      win_rate: 0.552,
      total_trades: 320,
      profit_factor: 2.1,
      avg_trade: 0.04,
      active_rate: null,
      directional_accuracy: null,
      precision_macro: null,
      f1_macro: null,
      equity_curve: null,
      buy_hold_curve: null,
      drawdown_curve: null,
      monthly_results: null,
      trades: null,
      hpo_param_importance: null,
      hpo_trials: null,
      best_study: null,
      hpo_study_meta: null,
      hpo_learning_summary: null,
      hpo_sensitivity: null,
      overfitting: null,
      walkforward_periods: null,
      diagnostics: null,
      summary_text: null,
    };
    expect(m.model).toBe("xgboost");
    expect(m.sharpe).toBeCloseTo(1.47);
  });

  it("Metrics accepts all-null optional fields", () => {
    const m: Metrics = {
      model: "test",
      sharpe: null,
      sortino: null,
      max_drawdown: null,
      total_return_pct: null,
      cagr: null,
      calmar_ratio: null,
      win_rate: null,
      total_trades: null,
      profit_factor: null,
      avg_trade: null,
      active_rate: null,
      directional_accuracy: null,
      precision_macro: null,
      f1_macro: null,
      equity_curve: null,
      buy_hold_curve: null,
      drawdown_curve: null,
      monthly_results: null,
      trades: null,
      hpo_param_importance: null,
      hpo_trials: null,
      best_study: null,
      hpo_study_meta: null,
      hpo_learning_summary: null,
      hpo_sensitivity: null,
      overfitting: null,
      walkforward_periods: null,
      diagnostics: null,
      summary_text: null,
    };
    expect(m.sharpe).toBeNull();
    expect(m.total_trades).toBeNull();
  });

  it("TradeRecord has correct shape", () => {
    const t: TradeRecord = {
      trade_id: 1,
      entry_date: "2024-01-15T10:00:00",
      exit_date: "2024-01-17T14:00:00",
      direction: "BUY",
      entry_price: 1.0842,
      exit_price: 1.0854,
      pips: 12,
      return_pct: 0.12,
      duration_bars: 8,
      barrier_hit: "tp",
    };
    expect(t.direction).toBe("BUY");
    expect(t.pips).toBe(12);
  });

  it("TradeRecord allows null barrier_hit", () => {
    const t: TradeRecord = {
      trade_id: 2,
      entry_date: "2024-01-20",
      exit_date: "2024-01-21",
      direction: "SELL",
      entry_price: 1.09,
      exit_price: 1.0892,
      pips: -8,
      return_pct: -0.08,
      duration_bars: 4,
      barrier_hit: null,
    };
    expect(t.barrier_hit).toBeNull();
  });

  it("MonthlyResult has correct shape", () => {
    const m: MonthlyResult = {
      month: "2024-01",
      return_pct: 0.012,
      win_rate: 0.55,
      trades: 28,
      sharpe: 1.2,
    };
    expect(m.month).toBe("2024-01");
    expect(m.sharpe).toBe(1.2);
  });

  it("HpoParamImportance sorts correctly", () => {
    const params: HpoParamImportance[] = [
      { param: "lags", importance: 0.8 },
      { param: "threshold", importance: 0.95 },
      { param: "fracdiff_d", importance: 0.3 },
    ];
    const sorted = params.sort((a, b) => b.importance - a.importance);
    expect(sorted[0].param).toBe("threshold");
    expect(sorted[2].param).toBe("fracdiff_d");
  });

  it("HpoTrial tracks optimization progress", () => {
    const trials: HpoTrial[] = [
      { trial_number: 1, value: -0.5, params: { lags: 10 } },
      { trial_number: 2, value: -0.8, params: { lags: 14 } },
      { trial_number: 3, value: -0.3, params: { lags: 20 } },
    ];
    const bestValue = Math.max(...trials.map((t) => t.value));
    expect(bestValue).toBe(-0.3);
  });
});
