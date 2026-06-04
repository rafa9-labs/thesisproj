import { useMemo } from "react";
import type { Metrics } from "@/api/schemas";
import { ParallelCoordinates } from "@/components/charts/ParallelCoordinates";
import { ContourPlot } from "@/components/charts/ContourPlot";
import { ChartCard } from "@/components/charts/ChartCard";
import { Activity, TrendingUp } from "lucide-react";

interface Props {
  metrics: Metrics | null;
}

export function ParameterExplorer({ metrics }: Props) {
  const trials = useMemo(() => {
    if (!metrics?.hpo_trials || !Array.isArray(metrics.hpo_trials)) return null;
    return metrics.hpo_trials as Array<{
      trial_number: number;
      value: number | null;
      params: Record<string, unknown>;
    }>;
  }, [metrics?.hpo_trials]);

  if (!trials || trials.length === 0) {
    return null;
  }

  const validCount = trials.filter((t) => t.value != null).length;

  return (
    <div
      className="rounded-sm border p-5"
      style={{
        borderColor: "var(--color-glass-border)",
        backgroundColor: "var(--color-glass)",
      }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity size={16} style={{ color: "var(--color-accent)" }} />
          <h3
            className="text-xs font-semibold uppercase tracking-[0.1em]"
            style={{ color: "var(--color-text-secondary)" }}
          >
            Parameter Explorer
          </h3>
        </div>
        <span className="text-[10px] font-mono" style={{ color: "var(--color-text-muted)" }}>
          {validCount}/{trials.length} trials
        </span>
      </div>

      <div className="flex flex-col gap-4">
        {/* Parallel Coordinates */}
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <TrendingUp size={11} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Parallel Coordinates
            </span>
          </div>
          <ChartCard title="" subtitle="" height={200}>
            <ParallelCoordinates trials={trials} />
          </ChartCard>
        </div>

        {/* Contour Plot */}
        <div>
          <div className="flex items-center gap-1.5 mb-2">
            <TrendingUp size={11} style={{ color: "var(--color-text-muted)" }} />
            <span className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
              Contour (Param × Param vs Score)
            </span>
          </div>
          <ChartCard title="" subtitle="" height={250}>
            <ContourPlot trials={trials} />
          </ChartCard>
        </div>
      </div>
    </div>
  );
}
