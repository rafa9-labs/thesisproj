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
    <div className="rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-(--color-accent)" />
          <h3 className="text-xs font-semibold tracking-[0.1em] text-(--color-text-secondary) uppercase">
            Parameter Explorer
          </h3>
        </div>
        <span className="font-mono text-[10px] text-(--color-text-muted)">
          {validCount}/{trials.length} trials
        </span>
      </div>

      <div className="flex flex-col gap-4">
        {/* Parallel Coordinates */}
        <div>
          <div className="mb-2 flex items-center gap-1.5">
            <TrendingUp size={11} className="text-(--color-text-muted)" />
            <span className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
              Parallel Coordinates
            </span>
          </div>
          <ChartCard title="" subtitle="" height={200}>
            <ParallelCoordinates trials={trials} />
          </ChartCard>
        </div>

        {/* Contour Plot */}
        <div>
          <div className="mb-2 flex items-center gap-1.5">
            <TrendingUp size={11} className="text-(--color-text-muted)" />
            <span className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
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
