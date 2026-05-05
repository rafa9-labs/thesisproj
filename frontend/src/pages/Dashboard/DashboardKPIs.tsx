import { useMemo } from "react";
import type { JobResults } from "@/api/schemas";

export interface DashboardKPIValues {
  avgSharpe: number | null;
  avgWinRate: number | null;
  profitableMonthsPct: number | null;
}

export function computeDashboardKPIs(allResults: JobResults[]): DashboardKPIValues {
  if (allResults.length === 0) {
    return { avgSharpe: null, avgWinRate: null, profitableMonthsPct: null };
  }

  let sharpeSum = 0;
  let sharpeCount = 0;
  let winRateSum = 0;
  let winRateCount = 0;
  let profitableMonths = 0;
  let totalMonths = 0;

  for (const result of allResults) {
    for (const m of result.metrics ?? []) {
      if (m.sharpe != null) {
        sharpeSum += m.sharpe;
        sharpeCount++;
      }
      if (m.win_rate != null) {
        winRateSum += m.win_rate;
        winRateCount++;
      }
    }
    for (const m of result.metrics ?? []) {
      const monthly = m.monthly_results;
      if (monthly) {
        for (const month of monthly) {
          totalMonths++;
          if (month.return_pct != null && month.return_pct > 0) {
            profitableMonths++;
          }
        }
      }
    }
  }

  return {
    avgSharpe: sharpeCount > 0 ? sharpeSum / sharpeCount : null,
    avgWinRate: winRateCount > 0 ? winRateSum / winRateCount : null,
    profitableMonthsPct: totalMonths > 0 ? profitableMonths / totalMonths : null,
  };
}

export function useDashboardKPIs(allResults: JobResults[]): DashboardKPIValues {
  return useMemo(() => computeDashboardKPIs(allResults), [allResults]);
}