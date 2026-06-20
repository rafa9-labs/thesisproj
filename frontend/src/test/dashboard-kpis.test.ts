import { describe, it, expect } from "vitest";
import { computeDashboardKPIs } from "@/pages/Dashboard/DashboardKPIs";
import type { JobResults } from "@/api/schemas";

const nullExtra = {
  cagr: null,
  calmar_ratio: null,
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
};

const makeResults = (overrides: Partial<JobResults> = {}): JobResults => ({
  job_id: "test-job-1",
  pair: "EURUSD",
  models: ["xgboost"],
  config: null,
  metrics: [],
  ...overrides,
});

describe("computeDashboardKPIs", () => {
  it("returns nulls for empty array", () => {
    const kpis = computeDashboardKPIs([]);
    expect(kpis).toEqual({
      winRate: null,
      totalProfit: null,
      profitFactor: null,
      maxDrawdown: null,
    });
  });

  it("computes single job with single model", () => {
    const results = [
      makeResults({
        metrics: [
          {
            model: "xgboost",
            sharpe: 1.47,
            sortino: 2.31,
            max_drawdown: -0.085,
            total_return_pct: 0.124,
            win_rate: 0.552,
            total_trades: 320,
            profit_factor: 2.1,
            avg_trade: 0.04,
            ...nullExtra,
          },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.winRate).toBeCloseTo(0.552);
    expect(kpis.totalProfit).toBeCloseTo(0.124);
    expect(kpis.profitFactor).toBeCloseTo(2.1);
    expect(kpis.maxDrawdown).toBeCloseTo(-0.085);
  });

  it("computes across multiple jobs with multiple models", () => {
    const results = [
      makeResults({
        metrics: [
          {
            model: "xgboost",
            sharpe: 1.47,
            sortino: 2.3,
            max_drawdown: -0.08,
            total_return_pct: 0.12,
            win_rate: 0.55,
            total_trades: 320,
            profit_factor: 2.1,
            avg_trade: 0.04,
            ...nullExtra,
          },
          {
            model: "lstm",
            sharpe: 0.89,
            sortino: 1.5,
            max_drawdown: -0.11,
            total_return_pct: 0.06,
            win_rate: 0.52,
            total_trades: 280,
            profit_factor: 1.6,
            avg_trade: 0.02,
            ...nullExtra,
          },
        ],
      }),
      makeResults({
        job_id: "test-job-2",
        metrics: [
          {
            model: "cnn",
            sharpe: 1.12,
            sortino: 1.89,
            max_drawdown: -0.098,
            total_return_pct: 0.09,
            win_rate: 0.54,
            total_trades: 300,
            profit_factor: 1.9,
            avg_trade: 0.03,
            ...nullExtra,
          },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.winRate).toBeCloseTo((0.55 + 0.52 + 0.54) / 3);
    expect(kpis.totalProfit).toBeCloseTo(0.12 + 0.06 + 0.09);
    expect(kpis.maxDrawdown).toBeCloseTo(-0.11);
  });

  it("handles null metrics gracefully", () => {
    const results = [
      makeResults({
        metrics: [
          {
            model: "xgboost",
            sharpe: null,
            sortino: null,
            max_drawdown: null,
            total_return_pct: null,
            win_rate: null,
            total_trades: null,
            profit_factor: null,
            avg_trade: null,
            ...nullExtra,
          },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.winRate).toBeNull();
    expect(kpis.totalProfit).toBeNull();
    expect(kpis.profitFactor).toBeNull();
  });

  it("handles mix of null and present values", () => {
    const results = [
      makeResults({
        metrics: [
          {
            model: "a",
            sharpe: 1.0,
            sortino: null,
            max_drawdown: null,
            total_return_pct: 0.05,
            win_rate: 0.5,
            total_trades: null,
            profit_factor: null,
            avg_trade: null,
            ...nullExtra,
          },
          {
            model: "b",
            sharpe: null,
            sortino: null,
            max_drawdown: null,
            total_return_pct: null,
            win_rate: 0.6,
            total_trades: null,
            profit_factor: null,
            avg_trade: null,
            ...nullExtra,
          },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.winRate).toBeCloseTo(0.55);
    expect(kpis.totalProfit).toBeCloseTo(0.05);
  });
});
