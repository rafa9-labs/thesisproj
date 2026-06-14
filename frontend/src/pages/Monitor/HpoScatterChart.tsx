import { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  ZAxis,
} from "recharts";
import type { HpoTrialRow } from "@/api/schemas";

const MODEL_COLORS = ["#06b6d4", "#089981", "#f59e0b", "#a78bfa", "#ec4899", "#22d3ee"];

interface Props {
  allTrials: { model: string; trial: HpoTrialRow }[];
  filterModel?: string | null;
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: { trial: number; score: number; model: string } }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div className="rounded-md border border-(--color-glass-border) bg-(--color-surface) px-3 py-2 font-mono text-xs shadow-2xl">
      <div className="text-[10px] text-(--color-text-dim)">{d.model}</div>
      <div>
        Trial <span className="text-(--color-brand)">#{d.trial}</span>
      </div>
      <div>
        Score <span className="text-(--color-accent-success)">{d.score.toFixed(4)}</span>
      </div>
    </div>
  );
}

export function HpoScatterChart({ allTrials, filterModel }: Props) {
  const filtered = useMemo(
    () => (filterModel ? allTrials.filter((t) => t.model === filterModel) : allTrials),
    [allTrials, filterModel],
  );

  const data = useMemo(() => {
    return filtered.map(({ model, trial }) => ({
      trial: trial.trial_number,
      score: trial.score,
      model,
      isBest: false,
    }));
  }, [filtered]);

  const modelGroups = useMemo(() => {
    const map = new Map<string, typeof data>();
    data.forEach((d) => {
      if (!map.has(d.model)) map.set(d.model, []);
      map.get(d.model)!.push(d);
    });
    return [...map.entries()];
  }, [data]);

  const bestScore = useMemo(() => {
    let best = -Infinity;
    let bestIdx = -1;
    data.forEach((d, i) => {
      if (d.score > best) {
        best = d.score;
        bestIdx = i;
      }
    });
    if (bestIdx >= 0) data[bestIdx].isBest = true;
    return bestIdx;
  }, [data]);

  const bestTrials = bestScore >= 0 ? [data[bestScore]] : [];

  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-[10px] text-(--color-text-muted)">
        No HPO trials yet
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 8, right: 8, bottom: 16, left: 0 }}>
        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="trial"
          name="Trial"
          tick={{ fontSize: 9, fontFamily: "JetBrains Mono", fill: "#64748b" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          dataKey="score"
          name="Score"
          tick={{ fontSize: 9, fontFamily: "JetBrains Mono", fill: "#64748b" }}
          axisLine={false}
          tickLine={false}
          domain={[0, "auto"]}
        />
        <ZAxis range={[14, 14]} />
        <Tooltip content={<CustomTooltip />} />
        {modelGroups.map(([model, points], i) => (
          <Scatter
            key={`model-${model}`}
            data={points}
            fill={MODEL_COLORS[i % MODEL_COLORS.length]}
            opacity={0.6}
          />
        ))}
        <Scatter data={bestTrials} fill="#fbbf24" opacity={1} name="Best" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
