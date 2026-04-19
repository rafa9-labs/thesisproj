import { describe, it, expect } from "vitest";
import { computeDashboardKPIs } from "@/pages/Dashboard/DashboardKPIs";
import type { JobResults } from "@/api/schemas";

const makeResults = (overrides: Partial<JobResults> = {}): JobResults => ({
  job_id: "test-job-1",
  pair: "EURUSD",
  models: ["xgboost"],
  config: null,
  metrics: [],
  monthly_results: null,
  trades: null,
  equity_curve: null,
  buy_hold_curve: null,
  drawdown_curve: null,
  hpo_param_importance: null,
  hpo_trials: null,
  ...overrides,
});

describe("computeDashboardKPIs", () => {
  it("returns zeros for empty array", () => {
    const kpis = computeDashboardKPIs([]);
    expect(kpis).toEqual({
      totalRuns: 0,
      bestSharpe: null,
      avgWinRate: null,
      bestReturn: null,
    });
  });

  it("computes single job with single model", () => {
    const results = [
      makeResults({
        metrics: [
          { model: "xgboost", sharpe: 1.47, sortino: 2.31, max_drawdown: -0.085, total_return: 0.124, win_rate: 0.552, total_trades: 320, profit_factor: 2.1, avg_trade: 0.04 },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.totalRuns).toBe(1);
    expect(kpis.bestSharpe).toBe(1.47);
    expect(kpis.avgWinRate).toBeCloseTo(0.552);
    expect(kpis.bestReturn).toBeCloseTo(0.124);
  });

  it("computes across multiple jobs with multiple models", () => {
    const results = [
      makeResults({
        metrics: [
          { model: "xgboost", sharpe: 1.47, sortino: 2.3, max_drawdown: -0.08, total_return: 0.12, win_rate: 0.55, total_trades: 320, profit_factor: 2.1, avg_trade: 0.04 },
          { model: "lstm", sharpe: 0.89, sortino: 1.5, max_drawdown: -0.11, total_return: 0.06, win_rate: 0.52, total_trades: 280, profit_factor: 1.6, avg_trade: 0.02 },
        ],
      }),
      makeResults({
        job_id: "test-job-2",
        metrics: [
          { model: "cnn", sharpe: 1.12, sortino: 1.89, max_drawdown: -0.098, total_return: 0.09, win_rate: 0.54, total_trades: 300, profit_factor: 1.9, avg_trade: 0.03 },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.totalRuns).toBe(3);
    expect(kpis.bestSharpe).toBe(1.47);
    expect(kpis.avgWinRate).toBeCloseTo((0.55 + 0.52 + 0.54) / 3);
    expect(kpis.bestReturn).toBeCloseTo(0.12);
  });

  it("handles null metrics gracefully", () => {
    const results = [
      makeResults({
        metrics: [
          { model: "xgboost", sharpe: null, sortino: null, max_drawdown: null, total_return: null, win_rate: null, total_trades: null, profit_factor: null, avg_trade: null },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.totalRuns).toBe(1);
    expect(kpis.bestSharpe).toBeNull();
    expect(kpis.avgWinRate).toBeNull();
    expect(kpis.bestReturn).toBeNull();
  });

  it("handles mix of null and present values", () => {
    const results = [
      makeResults({
        metrics: [
          { model: "a", sharpe: 1.0, sortino: null, max_drawdown: null, total_return: 0.05, win_rate: 0.5, total_trades: null, profit_factor: null, avg_trade: null },
          { model: "b", sharpe: null, sortino: null, max_drawdown: null, total_return: null, win_rate: 0.6, total_trades: null, profit_factor: null, avg_trade: null },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.totalRuns).toBe(2);
    expect(kpis.bestSharpe).toBe(1.0);
    expect(kpis.avgWinRate).toBeCloseTo(0.55);
    expect(kpis.bestReturn).toBeCloseTo(0.05);
  });

  it("picks best Sharpe from negative values", () => {
    const results = [
      makeResults({
        metrics: [
          { model: "a", sharpe: -0.5, sortino: null, max_drawdown: null, total_return: -0.1, win_rate: 0.4, total_trades: 10, profit_factor: 0.8, avg_trade: -0.01 },
          { model: "b", sharpe: -0.2, sortino: null, max_drawdown: null, total_return: -0.02, win_rate: 0.48, total_trades: 10, profit_factor: 0.9, avg_trade: -0.002 },
        ],
      }),
    ];
    const kpis = computeDashboardKPIs(results);
    expect(kpis.bestSharpe).toBe(-0.2);
    expect(kpis.bestReturn).toBeCloseTo(-0.02);
  });
});
