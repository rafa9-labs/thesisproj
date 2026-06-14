import { MonthlyReturnsChart } from "@/components/charts/MonthlyReturnsChart";
import type { MonthlyResult } from "@/api/schemas";

interface MonthlySectionProps {
  monthlyResults: MonthlyResult[] | null;
}

export function MonthlySection({ monthlyResults }: MonthlySectionProps) {
  const data = monthlyResults ?? [];

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-sm border border-(--color-border) bg-(--color-surface) p-8 text-(--color-text-muted)">
        <span className="font-mono text-sm">No monthly breakdown available</span>
      </div>
    );
  }

  const startMonth = data[0]?.month ?? "";
  const endMonth = data[data.length - 1]?.month ?? "";
  const positiveMonths = data.filter((m) => (m.return_pct ?? 0) > 0).length;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] text-(--color-text-muted)">
          {startMonth} &rarr; {endMonth} &middot; {positiveMonths}/{data.length} positive
        </span>
      </div>
      <MonthlyReturnsChart monthlyResults={monthlyResults} height={300} />
    </div>
  );
}
