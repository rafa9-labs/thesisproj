import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

interface Props {
  foldResults: Array<{ sharpe: number; trades: number; return_val: number; fold_idx: number }> | null;
  foldCv: number;
  foldCvPass?: boolean;
  avgSharpe?: number;
}

export function CommitteeFoldChart({ foldResults, foldCv, foldCvPass, avgSharpe }: Props) {
  const chartData = useMemo(() => {
    if (!foldResults || foldResults.length === 0) return [];
    return foldResults.map((f, i) => ({
      fold: `F${(f.fold_idx ?? i) + 1}`,
      sharpe: f.sharpe ?? 0,
      trades: f.trades ?? 0,
    }));
  }, [foldResults]);

  if (chartData.length === 0) {
    return (
      <p className="text-[11px] text-(--color-text-dim)">No per-fold data available</p>
    );
  }

  const cvColor = foldCv < 0.5
    ? "var(--color-accent-success)"
    : foldCv < 1.0
      ? "var(--color-accent-warning)"
      : "var(--color-accent-danger)";
  const cvLabel = foldCv < 0.5 ? "Consistent" : foldCv < 1.0 ? "Moderate" : "Unstable";

  return (
    <div>
      <div className="mb-2 flex items-center gap-3">
        <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
          Fold Sharpe Consistency
        </span>
        <span className="font-mono text-xs font-bold" style={{ color: cvColor }}>
          CV = {foldCv.toFixed(3)}
        </span>
        <span
          className="rounded px-1.5 py-0.5 text-[9px] font-semibold tracking-[0.06em] uppercase"
          style={{
            backgroundColor: `${cvColor}1A`,
            color: cvColor,
          }}
        >
          {cvLabel}
        </span>
        {foldCvPass !== undefined && (
          <span
            className="text-[9px]"
            style={{ color: foldCvPass ? "var(--color-accent-success)" : "var(--color-accent-danger)" }}
          >
            {foldCvPass ? "PASS" : "FAIL"}
          </span>
        )}
      </div>
      <div style={{ height: 180 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" vertical={false} />
            <XAxis
              dataKey="fold"
              tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
            />
            <YAxis
              tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-elevated)",
                border: "1px solid var(--color-glass-border)",
                borderRadius: 6,
                fontSize: 11,
                fontFamily: "var(--font-mono)",
              }}
              formatter={(value: number, name: string) => {
                if (name === "sharpe") return [value.toFixed(3), "Sharpe"];
                if (name === "trades") return [value, "Trades"];
                return [value, name];
              }}
            />
            {avgSharpe != null && (
              <ReferenceLine
                y={avgSharpe}
                stroke="var(--color-text-muted)"
                strokeDasharray="4 4"
                label={{
                  value: `avg ${avgSharpe.toFixed(2)}`,
                  position: "right",
                  fill: "var(--color-text-muted)",
                  fontSize: 9,
                  fontFamily: "var(--font-mono)",
                }}
              />
            )}
            <ReferenceLine y={0} stroke="var(--color-glass-border)" />
            <Bar dataKey="sharpe" radius={[3, 3, 0, 0]} maxBarSize={32}>
              {chartData.map((entry) => (
                <Cell
                  key={entry.fold}
                  fill={
                    entry.sharpe >= 0.5
                      ? "var(--color-accent-success)"
                      : entry.sharpe >= 0
                        ? "var(--color-accent-warning)"
                        : "var(--color-accent-danger)"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
