import { useMemo } from "react";
import type { JobResults } from "@/api/schemas";

export interface DashboardKPIValues {
  totalRuns: number;
  bestSharpe: number | null;
  avgWinRate: number | null;
  bestReturn: number | null;
}

export function computeDashboardKPIs(allResults: JobResults[]): DashboardKPIValues {
  if (allResults.length === 0) {
    return { totalRuns: 0, bestSharpe: null, avgWinRate: null, bestReturn: null };
  }

  let totalRuns = 0;
  let bestSharpe: number | null = null;
  let winRateSum = 0;
  let winRateCount = 0;
  let bestReturn: number | null = null;

  for (const result of allResults) {
    for (const m of result.metrics ?? []) {
      totalRuns++;
      if (m.sharpe != null) {
        if (bestSharpe === null || m.sharpe > bestSharpe) bestSharpe = m.sharpe;
      }
      if (m.win_rate != null) {
        winRateSum += m.win_rate;
        winRateCount++;
      }
      if (m.total_return_pct != null) {
        if (bestReturn === null || m.total_return_pct > bestReturn) bestReturn = m.total_return_pct;
      }
    }
  }

  return {
    totalRuns,
    bestSharpe,
    avgWinRate: winRateCount > 0 ? winRateSum / winRateCount : null,
    bestReturn,
  };
}

export function useDashboardKPIs(allResults: JobResults[]): DashboardKPIValues {
  return useMemo(() => computeDashboardKPIs(allResults), [allResults]);
}
