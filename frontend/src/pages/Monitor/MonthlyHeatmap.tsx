import type { OosPeriodResult } from "@/api/schemas";

interface Props {
  periods: OosPeriodResult[];
}

export function MonthlyHeatmap({ periods }: Props) {
  if (periods.length === 0) return null;

  return (
    <div className="flex h-6 gap-0.5 overflow-x-auto rounded border border-(--color-glass-border) bg-(--color-glass-hover) px-1">
      {periods.map((p, idx) => {
        const hasData = p.sharpe != null;
        const sharpe = p.sharpe ?? 0;
        const color = !hasData
          ? "var(--color-glass-border)"
          : sharpe >= 1
            ? "#089981"
            : sharpe >= 0.5
              ? "#0ea868"
              : sharpe >= 0
                ? "#f59e0b"
                : sharpe >= -0.5
                  ? "#f23645"
                  : "#991b1b";

        return (
          <div
            key={`${p.period}-${p.model ?? idx}`}
            className="flex h-full min-w-[14px] items-center justify-center rounded-sm transition-opacity hover:opacity-80"
            style={{ backgroundColor: color, opacity: hasData ? 1 : 0.2 }}
            title={`M${p.period}: Sharpe ${sharpe.toFixed(2)}, Return ${(p.return_pct ?? 0).toFixed(1)}%`}
          >
            <span className="font-mono text-[7px] font-bold text-white/70">{p.period}</span>
          </div>
        );
      })}
    </div>
  );
}
