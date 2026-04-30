import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Target,
  Activity,
  Percent,
  Award,
  Zap,
  Gauge,
  Compass,
  ShieldCheck,
  Landmark,
} from "lucide-react";
import { useMemo } from "react";
import { MetricCard } from "@/components/shared/MetricCard";
import { ZeroLeakageBadge } from "@/components/shared/ZeroLeakageBadge";
import { formatMetric, formatPercent, formatInt } from "@/lib/formatters";
import { monthlySparklineData } from "@/lib/chartUtils";
import type { Metrics, MonthlyResult } from "@/api/schemas";

interface MetricsGridProps {
  metrics: Metrics;
  modelName: string;
  warnings?: string[];
  monthlyResults?: MonthlyResult[] | null;
}

export function MetricsGrid({ metrics, modelName, warnings, monthlyResults }: MetricsGridProps) {
  const sparkSharpe = useMemo(() => monthlySparklineData(monthlyResults ?? null, "sharpe"), [monthlyResults]);
  const sparkWinRate = useMemo(() => monthlySparklineData(monthlyResults ?? null, "win_rate"), [monthlyResults]);
  const sparkReturn = useMemo(() => monthlySparklineData(monthlyResults ?? null, "return_pct"), [monthlyResults]);
  const cards = [
    {
      label: "Sharpe",
      value: formatMetric(metrics.sharpe),
      icon: <TrendingUp size={14} />,
      delta: metrics.sharpe !== null ? (metrics.sharpe >= 1 ? "Excellent" : metrics.sharpe >= 0.5 ? "Good" : "Weak") : null,
      deltaType: (metrics.sharpe ?? 0) >= 1 ? "positive" as const : (metrics.sharpe ?? 0) >= 0.5 ? "neutral" as const : "negative" as const,
      sparklineData: sparkSharpe,
    },
    {
      label: "Sortino",
      value: formatMetric(metrics.sortino),
      icon: <Award size={14} />,
      delta: metrics.sortino !== null ? (metrics.sortino >= 1.5 ? "Excellent" : metrics.sortino >= 0.8 ? "Good" : "Weak") : null,
      deltaType: (metrics.sortino ?? 0) >= 1.5 ? "positive" as const : (metrics.sortino ?? 0) >= 0.8 ? "neutral" as const : "negative" as const,
    },
    {
      label: "Max Drawdown",
      value: metrics.max_drawdown !== null ? formatPercent(metrics.max_drawdown) : "—",
      icon: <TrendingDown size={14} />,
      delta: metrics.max_drawdown !== null ? (Math.abs(metrics.max_drawdown) < 0.1 ? "Low risk" : Math.abs(metrics.max_drawdown) < 0.2 ? "Moderate" : "High risk") : null,
      deltaType: metrics.max_drawdown !== null ? (Math.abs(metrics.max_drawdown) < 0.1 ? "positive" as const : Math.abs(metrics.max_drawdown) < 0.2 ? "neutral" as const : "negative" as const) : "neutral" as const,
    },
    {
      label: "CAGR",
      value: formatPercent(metrics.cagr),
      icon: <BarChart3 size={14} />,
      deltaType: (metrics.cagr ?? 0) >= 0 ? "positive" as const : "negative" as const,
    },
    {
      label: "Total Return",
      value: formatPercent(metrics.total_return_pct),
      icon: <Landmark size={14} />,
      deltaType: (metrics.total_return_pct ?? 0) >= 0 ? "positive" as const : "negative" as const,
      sparklineData: sparkReturn,
    },
    {
      label: "Calmar",
      value: formatMetric(metrics.calmar_ratio),
      icon: <ShieldCheck size={14} />,
      delta: metrics.calmar_ratio !== null ? (metrics.calmar_ratio >= 3 ? "Excellent" : metrics.calmar_ratio >= 1 ? "Good" : "Weak") : null,
      deltaType: (metrics.calmar_ratio ?? 0) >= 3 ? "positive" as const : (metrics.calmar_ratio ?? 0) >= 1 ? "neutral" as const : "negative" as const,
    },
    {
      label: "Win Rate",
      value: formatPercent(metrics.win_rate, 1),
      icon: <Target size={14} />,
      delta: metrics.win_rate !== null ? (metrics.win_rate >= 0.55 ? "Above avg" : metrics.win_rate >= 0.5 ? "Avg" : "Below avg") : null,
      deltaType: (metrics.win_rate ?? 0) >= 0.55 ? "positive" as const : (metrics.win_rate ?? 0) >= 0.5 ? "neutral" as const : "negative" as const,
      sparklineData: sparkWinRate,
    },
    {
      label: "Profit Factor",
      value: formatMetric(metrics.profit_factor),
      icon: <Percent size={14} />,
      delta: metrics.profit_factor !== null ? (metrics.profit_factor >= 2 ? "Strong" : metrics.profit_factor >= 1.5 ? "Good" : "Weak") : null,
      deltaType: (metrics.profit_factor ?? 0) >= 2 ? "positive" as const : (metrics.profit_factor ?? 0) >= 1.5 ? "neutral" as const : "negative" as const,
    },
    {
      label: "Total Trades",
      value: formatInt(metrics.total_trades),
      icon: <Activity size={14} />,
      delta: metrics.total_trades !== null && metrics.total_trades === 0 ? "No trades" : null,
      deltaType: "negative" as const,
    },
    {
      label: "Avg Trade",
      value: formatPercent(metrics.avg_trade),
      icon: <Zap size={14} />,
      deltaType: (metrics.avg_trade ?? 0) >= 0 ? "positive" as const : "negative" as const,
    },
    {
      label: "Active Rate",
      value: formatPercent(metrics.active_rate),
      icon: <Gauge size={14} />,
    },
    {
      label: "Dir. Accuracy",
      value: formatPercent(metrics.directional_accuracy),
      icon: <Compass size={14} />,
      delta: metrics.directional_accuracy !== null ? (metrics.directional_accuracy >= 0.55 ? "Above avg" : "Below avg") : null,
      deltaType: (metrics.directional_accuracy ?? 0) >= 0.55 ? "positive" as const : "negative" as const,
    },
  ];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span
            className="text-lg font-bold"
            style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
          >
            {modelName}
          </span>
          <ZeroLeakageBadge />
        </div>
      </div>
      {warnings && warnings.length > 0 && (
        <div
          className="rounded-md border px-3 py-2 text-xs"
          style={{
            backgroundColor: "rgba(255, 167, 38, 0.08)",
            borderColor: "var(--color-accent-warning, #ffa726)",
            color: "var(--color-accent-warning, #ffa726)",
          }}
        >
          {warnings.map((w, i) => (
            <div key={i}>{w}</div>
          ))}
        </div>
      )}
      <div className="grid grid-cols-4 gap-3">
        {cards.map((card) => (
          <MetricCard
            key={card.label}
            label={card.label}
            value={card.value}
            icon={card.icon}
            delta={card.delta}
            deltaType={card.deltaType}
            sparklineData={card.sparklineData}
          />
        ))}
      </div>
    </div>
  );
}
