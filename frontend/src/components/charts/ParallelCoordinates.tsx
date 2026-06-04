import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

interface HpoTrial {
  trial_number: number;
  value: number | null;
  params: Record<string, unknown>;
}

interface Props {
  trials: HpoTrial[] | null;
}

export function ParallelCoordinates({ trials }: Props) {
  const { data, paramKeys } = useMemo(() => {
    if (!trials || trials.length === 0) return { data: [], paramKeys: [] };

    // Find numeric parameters across all trials
    const paramSet = new Set<string>();
    for (const t of trials) {
      if (!t.params) continue;
      for (const [k, v] of Object.entries(t.params)) {
        if (typeof v === "number" || typeof v === "boolean") {
          paramSet.add(k);
        }
      }
    }

    const keys = Array.from(paramSet);
    if (keys.length === 0) return { data: [], paramKeys: [] };

    // Compute min/max per param for normalization
    const ranges: Record<string, { min: number; max: number }> = {};
    for (const k of keys) {
      const vals: number[] = [];
      for (const t of trials) {
        const v = t.params?.[k];
        if (typeof v === "number") vals.push(v);
        else if (typeof v === "boolean") vals.push(v ? 1 : 0);
      }
      if (vals.length > 0) {
        ranges[k] = { min: Math.min(...vals), max: Math.max(...vals) };
      }
    }

    // Only include trials with a valid score
    const valid = trials.filter((t) => t.value != null);

    // Filter to top 80% by score
    const sorted = [...valid].sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
    const cutoff = Math.max(5, Math.floor(sorted.length * 0.8));
    const topTrials = sorted.slice(0, cutoff);

    // Transform to PCP format: each position on X axis = one param, Y = normalized value
    const rows = topTrials.map((t) => {
      const row: Record<string, number | string> = { trial: `#${t.trial_number}` };
      for (const k of keys) {
        const v = t.params?.[k];
        if (typeof v === "number") {
          const r = ranges[k];
          row[k] = r.max > r.min ? (v - r.min) / (r.max - r.min) : 0.5;
        } else if (typeof v === "boolean") {
          row[k] = v ? 1 : 0;
        }
      }
      return row;
    });

    // If too many trials, sample
    const sampled = rows.length > 50
      ? rows.filter((_, i) => i % Math.ceil(rows.length / 50) === 0)
      : rows;

    return { data: sampled, paramKeys: keys.slice(0, 10) };
  }, [trials]);

  // Use a single line with all data points flattened
  const chartData = useMemo(() => {
    const result: { param: string; trial: string; value: number }[] = [];
    for (const row of data) {
      for (const k of paramKeys) {
        if (row[k] !== undefined) {
          result.push({
            param: k,
            trial: row.trial as string,
            value: row[k] as number,
          });
        }
      }
    }
    return result;
  }, [data, paramKeys]);

  if (paramKeys.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-sm p-8" style={{ color: "var(--color-text-muted)" }}>
        <span className="text-xs">No numeric parameter data available for visualization.</span>
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: 200 }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: 4, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
          <XAxis
            dataKey="param"
            tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
            angle={-45}
            textAnchor="end"
            height={50}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--color-elevated)",
              border: "1px solid var(--color-glass-border)",
              borderRadius: 6,
              fontSize: 10,
              fontFamily: "var(--font-mono)",
            }}
          />
          {[...new Set(chartData.map((d) => d.trial))]
            .slice(0, 30)
            .map((trial) => (
              <Line
                key={trial}
                type="linear"
                dataKey="value"
                data={chartData.filter((d) => d.trial === trial)}
                stroke="var(--color-brand)"
                strokeWidth={0.5}
                dot={false}
                opacity={0.3}
                isAnimationActive={false}
                connectNulls
              />
            ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
