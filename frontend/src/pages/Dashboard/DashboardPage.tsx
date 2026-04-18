import { useNavigate } from "react-router-dom";
import { MetricCard } from "@/components/shared/MetricCard";
import { EmptyState } from "@/components/shared/EmptyState";
import { BarChart3, TrendingUp, Activity, Trophy } from "lucide-react";

export function DashboardPage() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col gap-6">
      <h2
        className="text-base font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Dashboard
      </h2>

      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Total Runs" value="—" icon={<BarChart3 size={16} />} />
        <MetricCard label="Best Sharpe" value="—" icon={<Trophy size={16} />} />
        <MetricCard label="Avg Win Rate" value="—" icon={<Activity size={16} />} />
        <MetricCard label="Best Return" value="—" icon={<TrendingUp size={16} />} />
      </div>

      <EmptyState
        title="No backtests yet"
        description="Run your first backtest to see performance data, model comparisons, and historical results here."
        actionLabel="Run First Backtest"
        onAction={() => navigate("/backtest")}
      />
    </div>
  );
}
