import { useState, useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ZAxis,
} from "recharts";
import { ChartCard } from "./ChartCard";
import type { HpoTrial } from "@/api/schemas";

const PARAM_COLORS = [
  "#2962FF",
  "#089981",
  "#F59E0B",
  "#EC4899",
  "#06B6D4",
  "#7C3AED",
  "#FF9800",
  "#E91E63",
];

interface ParameterSensitivityChartProps {
  trials: HpoTrial[] | null;
}

function NumericParamSelector({
  params,
  selected,
  onSelect,
}: {
  params: string[];
  selected: string;
  onSelect: (p: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {params.map((p) => (
        <button
          key={p}
          onClick={() => onSelect(p)}
          className="rounded-md border px-2 py-0.5 font-mono text-[10px] transition-colors"
          style={{
            borderColor: p === selected ? "var(--color-accent)" : "var(--color-border)",
            backgroundColor: p === selected ? "rgba(41,98,255,0.1)" : "var(--color-surface)",
            color: p === selected ? "var(--color-accent)" : "var(--color-text-secondary)",
            cursor: "pointer",
          }}
        >
          {p}
        </button>
      ))}
    </div>
  );
}

export function ParameterSensitivityChart({ trials }: ParameterSensitivityChartProps) {
  const trialData = trials ?? [];

  const numericParams = useMemo(() => {
    if (trialData.length === 0) return [];
    const paramSet = new Set<string>();
    for (const t of trialData) {
      for (const [key, val] of Object.entries(t.params ?? {})) {
        if (typeof val === "number") {
          paramSet.add(key);
        }
      }
    }
    return Array.from(paramSet).sort();
  }, [trialData]);

  const [selectedParam, setSelectedParam] = useState<string>(() => numericParams[0] ?? "");

  const scatterData = useMemo(() => {
    if (!selectedParam || trialData.length === 0) return [];
    return trialData
      .filter((t) => typeof (t.params ?? {})[selectedParam] === "number")
      .map((t, i) => ({
        trial: i + 1,
        paramValue: (t.params ?? {})[selectedParam] as number,
        objective: t.value,
      }));
  }, [trialData, selectedParam]);

  if (trialData.length < 3) {
    return (
      <ChartCard title="Parameter Sensitivity" subtitle={`${trialData.length} trials`}>
        <div className="flex items-center justify-center py-8 font-mono text-(--color-text-muted)">
          <span className="text-xs">
            Requires at least 3 HPO trials (current: {trialData.length})
          </span>
        </div>
      </ChartCard>
    );
  }

  if (numericParams.length === 0) {
    return (
      <ChartCard title="Parameter Sensitivity" subtitle={`${trialData.length} trials`}>
        <div className="flex items-center justify-center py-8 font-mono text-(--color-text-muted)">
          <span className="text-xs">No numeric hyperparameters found in trial data</span>
        </div>
      </ChartCard>
    );
  }

  const effectiveParam = numericParams.includes(selectedParam) ? selectedParam : numericParams[0];

  return (
    <ChartCard
      title="Parameter Sensitivity"
      subtitle={`${trialData.length} trials · ${numericParams.length} params`}
    >
      <div className="flex flex-col gap-3">
        <NumericParamSelector
          params={numericParams}
          selected={effectiveParam}
          onSelect={setSelectedParam}
        />
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-elevated)" />
            <XAxis
              type="number"
              dataKey="paramValue"
              name={effectiveParam}
              tick={{
                fill: "var(--color-text-muted)",
                fontSize: 10,
                fontFamily: "var(--font-mono)",
              }}
              label={{
                value: effectiveParam,
                position: "insideBottomRight",
                offset: -5,
                style: { fill: "var(--color-text-muted)", fontSize: 10 },
              }}
            />
            <YAxis
              type="number"
              dataKey="objective"
              name="Objective"
              tick={{
                fill: "var(--color-text-muted)",
                fontSize: 10,
                fontFamily: "var(--font-mono)",
              }}
              label={{
                value: "Objective (Sharpe)",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                style: { fill: "var(--color-text-muted)", fontSize: 10 },
              }}
            />
            <ZAxis range={[36, 36]} />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-elevated)",
                border: "1px solid var(--color-glass-border)",
                borderRadius: 6,
                fontSize: 12,
                fontFamily: "var(--font-mono)",
              }}
              labelStyle={{ color: "var(--color-text-muted)" }}
              formatter={(value: number, name: string) => [
                typeof value === "number" ? value.toFixed(4) : value,
                name === "paramValue" ? effectiveParam : "Objective",
              ]}
            />
            <Scatter
              data={scatterData}
              fill={PARAM_COLORS[numericParams.indexOf(effectiveParam) % PARAM_COLORS.length]}
              opacity={0.8}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}
