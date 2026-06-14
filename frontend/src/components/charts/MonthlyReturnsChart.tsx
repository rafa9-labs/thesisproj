import { useMemo, useState } from "react";
import type { MonthlyResult } from "@/api/schemas";

interface MonthlyReturnsChartProps {
  monthlyResults: MonthlyResult[] | null;
  height?: number;
}

const MONTH_LABELS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/** Maps a return % (−1..+1 raw fraction, or number) to a muted institutional color */
function returnToColor(ret: number | null | undefined): string {
  if (ret == null) return "var(--color-surface)"; // empty cell
  const pct = ret * 100; // convert fraction to %
  if (pct >= 3) return "rgba(8,153,129,0.85)";
  if (pct >= 1.5) return "rgba(8,153,129,0.60)";
  if (pct >= 0.5) return "rgba(8,153,129,0.38)";
  if (pct >= 0) return "rgba(8,153,129,0.18)";
  if (pct >= -0.5) return "rgba(242,54,69,0.18)";
  if (pct >= -1.5) return "rgba(242,54,69,0.38)";
  if (pct >= -3) return "rgba(242,54,69,0.60)";
  return "rgba(242,54,69,0.85)";
}

function returnToTextColor(ret: number | null | undefined): string {
  if (ret == null) return "var(--color-text-dim)";
  return ret >= 0 ? "#089981" : "#F23645";
}

function formatPct(v: number | null | undefined): string {
  if (v == null) return "";
  const pct = v * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

interface CalCell {
  year: number;
  month: number; // 0-indexed
  return_pct: number | null;
  win_rate: number | null;
  trades: number | null;
  sharpe: number | null;
}

export function MonthlyReturnsChart({ monthlyResults }: MonthlyReturnsChartProps) {
  const [hoveredCell, setHoveredCell] = useState<CalCell | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);

  const { years, grid } = useMemo(() => {
    if (!monthlyResults || monthlyResults.length === 0) {
      return { years: [], grid: new Map<number, Map<number, CalCell>>() };
    }

    const yearSet = new Set<number>();
    const g = new Map<number, Map<number, CalCell>>();

    for (const m of monthlyResults) {
      const raw = m.month ?? "";
      // Expected format: "YYYY-MM" or "YYYY-MM-DD"
      const parts = raw.split("-");
      const yr = parseInt(parts[0], 10);
      const mo = parseInt(parts[1], 10) - 1; // 0-indexed

      if (isNaN(yr) || isNaN(mo)) continue;

      yearSet.add(yr);
      if (!g.has(yr)) g.set(yr, new Map());
      g.get(yr)!.set(mo, {
        year: yr,
        month: mo,
        return_pct: m.return_pct ?? null,
        win_rate: m.win_rate ?? null,
        trades: m.trades ?? null,
        sharpe: m.sharpe ?? null,
      });
    }

    const sortedYears = Array.from(yearSet).sort((a, b) => a - b);
    return { years: sortedYears, grid: g };
  }, [monthlyResults]);

  if (years.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-sm bg-(--color-surface) p-8 font-mono text-[12px] text-(--color-text-dim)">
        No monthly data available
      </div>
    );
  }

  return (
    <div className="relative select-none">
      <div className="overflow-hidden rounded-sm border border-[#2A2E39] bg-(--color-surface)">
        {/* Header row: months */}
        <div
          className="grid border-b border-[#2A2E39]"
          style={{ gridTemplateColumns: "52px repeat(12, minmax(0, 1fr))" }}
        >
          <div className="p-[4px_6px]" />
          {MONTH_LABELS.map((m) => (
            <div
              key={m}
              className="p-[4px_2px] text-center font-sans text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase"
            >
              {m}
            </div>
          ))}
        </div>

        {/* Data rows: one per year */}
        {years.map((yr, yi) => (
          <div
            key={yr}
            className="grid"
            style={{
              gridTemplateColumns: "52px repeat(12, minmax(0, 1fr))",
              borderBottom: yi < years.length - 1 ? "1px solid #131722" : "none",
            }}
          >
            {/* Year label */}
            <div className="flex items-center justify-end border-r border-[#2A2E39] pr-2 font-mono text-[10px] text-(--color-text-muted)">
              {yr}
            </div>

            {/* 12 month cells */}
            {Array.from({ length: 12 }, (_, mo) => {
              const cell = grid.get(yr)?.get(mo) ?? null;
              const bg = returnToColor(cell?.return_pct);
              const tc = returnToTextColor(cell?.return_pct);
              return (
                <div
                  key={mo}
                  className="flex h-[30px] cursor-default items-center justify-center font-mono text-[9px] font-semibold outline-offset-[-1px] transition-all duration-100"
                  style={{
                    backgroundColor: bg,
                    color: tc,
                    borderLeft: mo > 0 ? "1px solid rgba(19,23,34,0.5)" : "none",
                    outline:
                      hoveredCell?.year === yr && hoveredCell?.month === mo
                        ? "1px solid rgba(255,255,255,0.15)"
                        : "none",
                  }}
                  onMouseEnter={(e) => {
                    if (!cell) return;
                    setHoveredCell(cell);
                    setTooltipPos({ x: e.clientX, y: e.clientY });
                  }}
                  onMouseMove={(e) => {
                    if (!cell) return;
                    setTooltipPos({ x: e.clientX, y: e.clientY });
                  }}
                  onMouseLeave={() => {
                    setHoveredCell(null);
                    setTooltipPos(null);
                  }}
                >
                  {cell ? formatPct(cell.return_pct) : ""}
                </div>
              );
            })}
          </div>
        ))}

        {/* Legend */}
        <div className="flex items-center gap-2 border-t border-[#2A2E39] px-3 py-2 font-sans text-[9px] text-(--color-text-muted)">
          <span className="tracking-[0.06em] uppercase">Return</span>
          {[
            { bg: "rgba(242,54,69,0.85)", label: "< -3%" },
            { bg: "rgba(242,54,69,0.60)", label: "-3 – -1.5%" },
            { bg: "rgba(242,54,69,0.18)", label: "-1.5 – 0%" },
            { bg: "rgba(8,153,129,0.18)", label: "0 – 1.5%" },
            { bg: "rgba(8,153,129,0.60)", label: "1.5 – 3%" },
            { bg: "rgba(8,153,129,0.85)", label: "> 3%" },
          ].map((s) => (
            <div key={s.label} className="flex items-center gap-1">
              <div className="h-[10px] w-[10px] rounded-[2px]" style={{ backgroundColor: s.bg }} />
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Floating tooltip */}
      {hoveredCell && tooltipPos && (
        <div
          className="pointer-events-none fixed z-50 min-w-[160px] rounded-sm border border-[#2A2E39] bg-(--color-elevated) px-3 py-2 font-mono text-[11px] text-(--color-text-primary) shadow-[0_8px_24px_rgba(0,0,0,0.5)]"
          style={{
            left: tooltipPos.x + 12,
            top: tooltipPos.y - 80,
          }}
        >
          <div className="mb-1.5 border-b border-[#2A2E39] pb-1.5 text-[10px] font-semibold tracking-wider text-(--color-text-muted) uppercase">
            {MONTH_LABELS[hoveredCell.month]} {hoveredCell.year}
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex justify-between gap-4">
              <span className="text-(--color-text-muted)">Return</span>
              <span
                className="font-bold"
                style={{ color: returnToTextColor(hoveredCell.return_pct) }}
              >
                {formatPct(hoveredCell.return_pct)}
              </span>
            </div>
            {hoveredCell.win_rate != null && (
              <div className="flex justify-between gap-4">
                <span className="text-(--color-text-muted)">Win Rate</span>
                <span>{formatPct(hoveredCell.win_rate)}</span>
              </div>
            )}
            {hoveredCell.trades != null && (
              <div className="flex justify-between gap-4">
                <span className="text-(--color-text-muted)">Trades</span>
                <span>{hoveredCell.trades}</span>
              </div>
            )}
            {hoveredCell.sharpe != null && (
              <div className="flex justify-between gap-4">
                <span className="text-(--color-text-muted)">Sharpe</span>
                <span
                  style={{
                    color: (hoveredCell.sharpe ?? 0) >= 1 ? "#089981" : "var(--color-text-primary)",
                  }}
                >
                  {hoveredCell.sharpe.toFixed(2)}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
