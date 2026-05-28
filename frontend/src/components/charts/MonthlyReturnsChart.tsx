import { useMemo, useState } from "react";
import type { MonthlyResult } from "@/api/schemas";

interface MonthlyReturnsChartProps {
  monthlyResults: MonthlyResult[] | null;
  height?: number;
}

const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Maps a return % (−1..+1 raw fraction, or number) to a muted institutional color */
function returnToColor(ret: number | null | undefined): string {
  if (ret == null) return "#1E222D"; // empty cell
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
  if (ret == null) return "#4A5568";
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
      <div
        className="flex items-center justify-center rounded-lg p-8"
        style={{ backgroundColor: "#1E222D", color: "#4A5568", fontSize: 12, fontFamily: "JetBrains Mono, monospace" }}
      >
        No monthly data available
      </div>
    );
  }

  return (
    <div className="relative select-none">
      <div
        className="rounded-lg overflow-hidden"
        style={{ backgroundColor: "#1E222D", border: "1px solid #2A2E39" }}
      >
        {/* Header row: months */}
        <div
          className="grid"
          style={{
            gridTemplateColumns: "52px repeat(12, minmax(0, 1fr))",
            borderBottom: "1px solid #2A2E39",
          }}
        >
          <div style={{ padding: "4px 6px" }} />
          {MONTH_LABELS.map((m) => (
            <div
              key={m}
              className="text-center"
              style={{
                padding: "4px 2px",
                fontSize: 10,
                fontFamily: "Inter, sans-serif",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                color: "#787B86",
              }}
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
            <div
              className="flex items-center justify-end pr-2"
              style={{
                fontSize: 10,
                fontFamily: "JetBrains Mono, monospace",
                color: "#787B86",
                borderRight: "1px solid #2A2E39",
              }}
            >
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
                  className="flex items-center justify-center cursor-default transition-all duration-100"
                  style={{
                    height: 30,
                    backgroundColor: bg,
                    fontSize: 9,
                    fontFamily: "JetBrains Mono, monospace",
                    fontWeight: 600,
                    color: tc,
                    borderLeft: mo > 0 ? "1px solid rgba(19,23,34,0.5)" : "none",
                    outline: hoveredCell?.year === yr && hoveredCell?.month === mo ? "1px solid rgba(255,255,255,0.15)" : "none",
                    outlineOffset: -1,
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
        <div
          className="flex items-center gap-2 px-3 py-2"
          style={{ borderTop: "1px solid #2A2E39", fontSize: 9, fontFamily: "Inter, sans-serif", color: "#787B86" }}
        >
          <span className="uppercase tracking-[0.06em]">Return</span>
          {[
            { bg: "rgba(242,54,69,0.85)", label: "< -3%" },
            { bg: "rgba(242,54,69,0.60)", label: "-3 – -1.5%" },
            { bg: "rgba(242,54,69,0.18)", label: "-1.5 – 0%" },
            { bg: "rgba(8,153,129,0.18)", label: "0 – 1.5%" },
            { bg: "rgba(8,153,129,0.60)", label: "1.5 – 3%" },
            { bg: "rgba(8,153,129,0.85)", label: "> 3%" },
          ].map((s) => (
            <div key={s.label} className="flex items-center gap-1">
              <div style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: s.bg }} />
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Floating tooltip */}
      {hoveredCell && tooltipPos && (
        <div
          className="fixed z-50 pointer-events-none rounded-lg px-3 py-2 text-[11px]"
          style={{
            left: tooltipPos.x + 12,
            top: tooltipPos.y - 80,
            backgroundColor: "#252934",
            border: "1px solid #2A2E39",
            fontFamily: "JetBrains Mono, monospace",
            boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
            minWidth: 160,
            color: "#E8ECF1",
          }}
        >
          <div
            className="font-semibold mb-1.5 pb-1.5 text-[10px] uppercase tracking-wider"
            style={{ color: "#787B86", borderBottom: "1px solid #2A2E39" }}
          >
            {MONTH_LABELS[hoveredCell.month]} {hoveredCell.year}
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex justify-between gap-4">
              <span style={{ color: "#787B86" }}>Return</span>
              <span style={{ color: returnToTextColor(hoveredCell.return_pct), fontWeight: 700 }}>
                {formatPct(hoveredCell.return_pct)}
              </span>
            </div>
            {hoveredCell.win_rate != null && (
              <div className="flex justify-between gap-4">
                <span style={{ color: "#787B86" }}>Win Rate</span>
                <span>{formatPct(hoveredCell.win_rate)}</span>
              </div>
            )}
            {hoveredCell.trades != null && (
              <div className="flex justify-between gap-4">
                <span style={{ color: "#787B86" }}>Trades</span>
                <span>{hoveredCell.trades}</span>
              </div>
            )}
            {hoveredCell.sharpe != null && (
              <div className="flex justify-between gap-4">
                <span style={{ color: "#787B86" }}>Sharpe</span>
                <span style={{ color: (hoveredCell.sharpe ?? 0) >= 1 ? "#089981" : "#E8ECF1" }}>
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
