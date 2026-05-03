import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { SharpeHeatmap } from "@/components/charts/SharpeHeatmap";
import type { HeatmapCellData } from "@/components/charts/SharpeHeatmap";
import type { HeatmapResponse } from "@/api/schemas";

type MetricKey = "sharpe" | "total_return_pct" | "win_rate" | "max_drawdown";

interface PerformanceHeatmapSectionProps {
  data: HeatmapResponse | undefined;
  isLoading: boolean;
}

const METRIC_OPTIONS: { key: MetricKey; label: string }[] = [
  { key: "sharpe", label: "Sharpe" },
  { key: "total_return_pct", label: "Return %" },
  { key: "win_rate", label: "Win Rate" },
  { key: "max_drawdown", label: "Max DD" },
];

export function PerformanceHeatmapSection({ data, isLoading }: PerformanceHeatmapSectionProps) {
  const navigate = useNavigate();
  const [metric, setMetric] = useState<MetricKey>("sharpe");

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Model × Pair Performance
        </h3>
        <div
          className="h-[200px] rounded-lg border animate-pulse"
          style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
        />
      </div>
    );
  }

  if (!data || data.models.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Model × Pair Performance
        </h3>
        <SharpeHeatmap models={[]} pairs={[]} cells={[]} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Model × Pair Performance
        </h3>
        <div className="flex gap-1">
          {METRIC_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              onClick={() => setMetric(opt.key)}
              className="rounded-md border px-2 py-0.5 text-[10px] transition-colors"
              style={{
                borderColor: metric === opt.key ? "var(--color-primary)" : "var(--color-border)",
                backgroundColor: metric === opt.key ? "var(--color-primary-glow)" : "var(--color-surface)",
                color: metric === opt.key ? "var(--color-primary)" : "var(--color-text-secondary)",
                cursor: "pointer",
                fontFamily: "var(--font-mono)",
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
      <SharpeHeatmap
        models={data.models}
        pairs={data.pairs}
        cells={data.cells as unknown as HeatmapCellData[]}
        onCellClick={(jobId) => navigate(`/results/${jobId}`)}
        metric={metric}
      />
    </div>
  );
}