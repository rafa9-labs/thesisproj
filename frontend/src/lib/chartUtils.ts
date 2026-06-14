import type { EquityPoint, TradeRecord, MonthlyResult } from "@/api/schemas";

export function normalizeEquityCurve(raw: EquityPoint[] | null | undefined): EquityPoint[] | null {
  if (!raw || !Array.isArray(raw) || raw.length === 0) return raw ?? null;
  const first = raw[0];
  if (typeof first === "number") {
    return (raw as number[]).map((v, i) => ({ time: i, value: v }));
  }
  if (typeof first === "object" && first !== null && "time" in first && "value" in first) {
    return raw as EquityPoint[];
  }
  return raw as EquityPoint[];
}

export function computeRollingSharpe(
  curve: EquityPoint[],
  windowSize: number = 30,
): { time: number; sharpe: number }[] {
  if (curve.length < windowSize + 1) return [];

  const returns: number[] = [];
  for (let i = 1; i < curve.length; i++) {
    const prev = curve[i - 1].value ?? 0;
    const curr = curve[i].value ?? 0;
    returns.push(prev !== 0 ? (curr - prev) / prev : 0);
  }

  const result: { time: number; sharpe: number }[] = [];
  for (let i = windowSize; i <= returns.length; i++) {
    const slice = returns.slice(i - windowSize, i);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const std = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length);
    const sharpe = std > 1e-9 ? (mean / std) * Math.sqrt(252) : 0;
    result.push({ time: curve[i]?.time ?? curve[curve.length - 1].time, sharpe });
  }
  return result;
}

export function computeRollingReturn(
  curve: EquityPoint[],
  windowSize: number = 30,
): { time: number; returnPct: number }[] {
  if (curve.length < windowSize) return [];

  const result: { time: number; returnPct: number }[] = [];
  for (let i = windowSize - 1; i < curve.length; i++) {
    const start = curve[i - windowSize + 1].value ?? 0;
    const end = curve[i].value ?? 0;
    const ret = start !== 0 ? (end - start) / start : 0;
    result.push({ time: curve[i].time, returnPct: ret * 100 });
  }
  return result;
}

export function binTradeReturns(
  trades: TradeRecord[],
  numBins?: number,
): { bin: string; count: number; positive: boolean }[] {
  if (trades.length === 0) return [];

  const returns = trades.map((t) => t.return_pct).filter((r) => r != null);
  if (returns.length === 0) return [];

  const min = Math.min(...returns);
  const max = Math.max(...returns);
  const range = max - min;
  const binCount = numBins ?? Math.max(Math.ceil(Math.sqrt(returns.length)), 8);
  const binWidth = range > 0 ? range / binCount : 1;

  const bins: { bin: string; count: number; positive: boolean }[] = [];
  for (let i = 0; i < binCount; i++) {
    const lo = min + i * binWidth;
    const hi = lo + binWidth;
    const mid = (lo + hi) / 2;
    bins.push({
      bin: mid >= 0 ? `+${mid.toFixed(2)}` : mid.toFixed(2),
      count: 0,
      positive: mid >= 0,
    });
  }

  for (const r of returns) {
    let idx = range > 0 ? Math.floor((r - min) / binWidth) : 0;
    if (idx >= binCount) idx = binCount - 1;
    if (idx < 0) idx = 0;
    bins[idx].count++;
  }

  return bins;
}

export function cumulativePnlFromTrades(
  trades: TradeRecord[],
): { tradeNum: number; cumPnl: number }[] {
  const sorted = [...trades]
    .filter((t) => t.entry_date && t.return_pct != null)
    .sort((a, b) => a.entry_date.localeCompare(b.entry_date));

  let cumSum = 0;
  return sorted.map((t, i) => {
    cumSum += t.return_pct;
    return { tradeNum: i + 1, cumPnl: cumSum * 100 };
  });
}

export function monthlySparklineData(
  monthly: MonthlyResult[] | null,
  key: keyof MonthlyResult,
): number[] {
  if (!monthly || monthly.length === 0) return [];
  return monthly.map((m) => {
    const v = m[key];
    return typeof v === "number" ? v : 0;
  });
}
