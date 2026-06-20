import type { JobResults, FullCycleHistoryEntry } from "@/api/schemas";

export interface DashboardKPIValues {
  winRate: number | null;
  totalProfit: number | null;
  profitFactor: number | null;
  maxDrawdown: number | null;
}

export interface CommitteeKPIValues {
  avgSharpe: number | null;
  trustScore: number | null;
  survivors: number | null;
  factorySharpe: number | null;
}

export function computeDashboardKPIs(allResults: JobResults[]): DashboardKPIValues {
  if (allResults.length === 0) {
    return { winRate: null, totalProfit: null, profitFactor: null, maxDrawdown: null };
  }

  let winRateSum = 0;
  let winRateCount = 0;
  let profitSum = 0;
  let profitCount = 0;
  let pfNumerator = 0;
  let pfDenominator = 0;
  let worstDrawdown: number | null = null;

  for (const result of allResults) {
    for (const m of result.metrics ?? []) {
      if (m.win_rate != null) {
        winRateSum += m.win_rate;
        winRateCount++;
      }
      if (m.total_return_pct != null) {
        profitSum += m.total_return_pct;
        profitCount++;
      }
      if (m.profit_factor != null) {
        pfNumerator += m.profit_factor * (m.total_trades ?? 1);
        pfDenominator += m.total_trades ?? 1;
      }
      if (m.max_drawdown != null) {
        if (worstDrawdown === null || m.max_drawdown < worstDrawdown) {
          worstDrawdown = m.max_drawdown;
        }
      }
    }
  }

  return {
    winRate: winRateCount > 0 ? winRateSum / winRateCount : null,
    totalProfit: profitCount > 0 ? profitSum : null,
    profitFactor: pfDenominator > 0 ? pfNumerator / pfDenominator : null,
    maxDrawdown: worstDrawdown,
  };
}

export function computeCommitteeKPIs(entries: FullCycleHistoryEntry[]): CommitteeKPIValues {
  const completed = entries.filter((e) => e.status === "completed");
  if (completed.length === 0) {
    return { avgSharpe: null, trustScore: null, survivors: null, factorySharpe: null };
  }

  let sharpeSum = 0;
  let sharpeCount = 0;
  let bestTrust: number | null = null;
  let totalSurvivors = 0;
  let bestFactory: number | null = null;

  for (const e of completed) {
    if (e.avg_sharpe != null && isFinite(e.avg_sharpe)) {
      sharpeSum += e.avg_sharpe;
      sharpeCount++;
    }
    if (e.trust_score != null && isFinite(e.trust_score)) {
      if (bestTrust === null || e.trust_score > bestTrust) {
        bestTrust = e.trust_score;
      }
    }
    totalSurvivors += e.survivors_count ?? 0;
    if (e.factory_best_sharpe != null && isFinite(e.factory_best_sharpe)) {
      if (bestFactory === null || e.factory_best_sharpe > bestFactory) {
        bestFactory = e.factory_best_sharpe;
      }
    }
  }

  return {
    avgSharpe: sharpeCount > 0 ? sharpeSum / sharpeCount : null,
    trustScore: bestTrust,
    survivors: totalSurvivors > 0 ? totalSurvivors : null,
    factorySharpe: bestFactory,
  };
}
