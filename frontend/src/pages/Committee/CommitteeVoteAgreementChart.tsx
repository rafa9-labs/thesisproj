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
  LabelList,
} from "recharts";

interface VoteBin {
  count: number;
  pct: number;
  win_rate: number;
  avg_return: number;
}

interface Props {
  voteAgreement: Record<string, VoteBin> | undefined;
}

const CATEGORIES = ["unanimous", "supermajority", "majority", "split"] as const;
const LABELS: Record<string, string> = {
  unanimous: "100%",
  supermajority: "75%+",
  majority: "50-75%",
  split: "<50%",
};

export function CommitteeVoteAgreementChart({ voteAgreement }: Props) {
  const chartData = useMemo(() => {
    if (!voteAgreement) return [];
    return CATEGORIES
      .map((key) => {
        const bin = voteAgreement[key];
        if (!bin) return null;
        return {
          name: LABELS[key] || key,
          key,
          pct: bin.pct,
          winRate: bin.win_rate * 100,
          avgReturn: bin.avg_return * 100,
          count: bin.count,
        };
      })
      .filter(Boolean) as { name: string; key: string; pct: number; winRate: number; avgReturn: number; count: number }[];
  }, [voteAgreement]);

  if (chartData.length === 0) {
    return (
      <p className="text-[11px] text-(--color-text-dim)">No vote agreement data</p>
    );
  }

  return (
    <div>
      <div style={{ height: 160 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            />
            <YAxis hide />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-elevated)",
                border: "1px solid var(--color-glass-border)",
                borderRadius: 6,
                fontSize: 11,
                fontFamily: "var(--font-mono)",
              }}
              formatter={(value: number, name: string) => {
                if (name === "pct") return [`${value.toFixed(1)}%`, "Share of bars"];
                if (name === "winRate") return [`${value.toFixed(1)}%`, "Win Rate"];
                return [value, name];
              }}
            />
            <Bar dataKey="pct" radius={[3, 3, 0, 0]} maxBarSize={60}>
              {chartData.map((entry) => (
                <Cell
                  key={entry.key}
                  fill={
                    entry.winRate >= 55
                      ? "var(--color-accent-success)"
                      : entry.winRate >= 48
                        ? "var(--color-accent-warning)"
                        : "var(--color-accent-danger)"
                  }
                />
              ))}
              <LabelList
                dataKey="winRate"
                position="top"
                style={{
                  fill: "var(--color-text-muted)",
                  fontSize: 9,
                  fontFamily: "var(--font-mono)",
                }}
                formatter={(v: number) => `${v.toFixed(0)}% wr`}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex items-center gap-4 font-mono text-[9px] text-(--color-text-muted)">
        {chartData.map((d) => (
          <span key={d.key}>{d.name} Vote: {d.pct.toFixed(0)}%</span>
        ))}
      </div>
    </div>
  );
}
