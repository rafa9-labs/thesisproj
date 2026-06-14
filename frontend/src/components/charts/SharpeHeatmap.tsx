export interface HeatmapCellData {
  model: string;
  pair: string;
  sharpe: number | null;
  total_return_pct: number | null;
  win_rate: number | null;
  max_drawdown: number | null;
  total_trades: number | null;
  job_id: string | null;
}

interface SharpeHeatmapProps {
  models: string[];
  pairs: string[];
  cells: HeatmapCellData[];
  onCellClick?: (jobId: string) => void;
  metric?: "sharpe" | "total_return_pct" | "win_rate" | "max_drawdown";
}

const METRIC_LABELS: Record<string, string> = {
  sharpe: "Sharpe",
  total_return_pct: "Return %",
  win_rate: "Win Rate",
  max_drawdown: "Max DD",
};

function cellColor(val: number | null | undefined, metric: string): string {
  if (val == null) return "var(--color-elevated)";

  if (metric === "max_drawdown") {
    if (val > -5) return "rgba(8,153,129,0.85)";
    if (val > -15) return "rgba(8,153,129,0.5)";
    if (val > -30) return "rgba(255,152,0,0.45)";
    return "rgba(242,54,69,0.55)";
  }

  if (metric === "win_rate") {
    if (val > 0.55) return "rgba(8,153,129,0.85)";
    if (val > 0.5) return "rgba(8,153,129,0.5)";
    if (val > 0.45) return "rgba(255,152,0,0.45)";
    return "rgba(242,54,69,0.55)";
  }

  if (val > 1) return "rgba(8,153,129,0.85)";
  if (val > 0) return "rgba(8,153,129,0.5)";
  if (val > -0.5) return "rgba(255,152,0,0.45)";
  return "rgba(242,54,69,0.55)";
}

function formatCellVal(val: number | null | undefined, metric: string): string {
  if (val == null) return "—";
  if (metric === "win_rate") return `${(val * 100).toFixed(1)}%`;
  if (metric === "max_drawdown") return `${val.toFixed(1)}%`;
  if (metric === "total_return_pct") return `${val.toFixed(1)}%`;
  return val.toFixed(2);
}

export function SharpeHeatmap({
  models,
  pairs,
  cells,
  onCellClick,
  metric = "sharpe",
}: SharpeHeatmapProps) {
  const cellMap = new Map<string, HeatmapCellData>();
  for (const c of cells) {
    cellMap.set(`${c.model}::${c.pair}`, c);
  }

  const cellWidth = 88;
  const cellHeight = 28;
  const labelWidth = 120;
  const headerHeight = 50;
  const totalW = labelWidth + pairs.length * cellWidth + 4;
  const totalH = headerHeight + models.length * cellHeight + 4;

  if (models.length === 0 || pairs.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-sm border border-(--color-border) bg-(--color-surface) p-8 text-(--color-text-muted)">
        <span className="font-mono text-sm">
          Run backtests across multiple pairs and models to populate the heatmap
        </span>
      </div>
    );
  }

  return (
    <div className="overflow-auto rounded-sm border border-(--color-border) bg-(--color-surface) p-3">
      <svg width={totalW} height={totalH} viewBox={`0 0 ${totalW} ${totalH}`}>
        {pairs.map((pair, ci) => (
          <g key={`h-${pair}`}>
            <text
              x={labelWidth + ci * cellWidth + cellWidth / 2}
              y={headerHeight - 10}
              textAnchor="middle"
              fill="var(--color-text-muted)"
              fontSize={9}
              fontFamily="var(--font-mono)"
              transform={`rotate(-30, ${labelWidth + ci * cellWidth + cellWidth / 2}, ${headerHeight - 10})`}
            >
              {pair}
            </text>
          </g>
        ))}
        {models.map((model, ri) => (
          <g key={`r-${model}`}>
            <text
              x={labelWidth - 6}
              y={headerHeight + ri * cellHeight + cellHeight / 2 + 4}
              textAnchor="end"
              fill="var(--color-text-muted)"
              fontSize={9}
              fontFamily="var(--font-mono)"
            >
              {model.length > 16 ? model.slice(0, 16) + "…" : model}
            </text>
          </g>
        ))}
        {models.map((model, ri) =>
          pairs.map((pair, ci) => {
            const cell = cellMap.get(`${model}::${pair}`);
            const val = cell ? (cell[metric as keyof HeatmapCellData] as number | null) : null;
            const fill = cellColor(val, metric);
            return (
              <g
                key={`${ri}-${ci}`}
                onClick={() => {
                  if (cell?.job_id && onCellClick) onCellClick(cell.job_id);
                }}
                style={{ cursor: cell?.job_id ? "pointer" : "default" }}
              >
                <rect
                  x={labelWidth + ci * cellWidth + 2}
                  y={headerHeight + ri * cellHeight + 2}
                  width={cellWidth - 4}
                  height={cellHeight - 4}
                  rx={4}
                  fill={fill}
                />
                <text
                  x={labelWidth + ci * cellWidth + cellWidth / 2}
                  y={headerHeight + ri * cellHeight + cellHeight / 2 + 3}
                  textAnchor="middle"
                  fill="var(--color-text-primary)"
                  fontSize={9}
                  fontFamily="var(--font-mono)"
                  fontWeight={600}
                >
                  {formatCellVal(val, metric)}
                </text>
              </g>
            );
          }),
        )}
      </svg>
      <div className="mt-2 flex items-center gap-4 text-[10px] text-(--color-text-muted)">
        <span>Metric: {METRIC_LABELS[metric]}</span>
        <span>Click cell to view results</span>
      </div>
    </div>
  );
}
