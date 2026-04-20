import { useMemo } from "react";
import { AgGridReact } from "ag-grid-react";
import type { ColDef, ValueFormatterParams } from "ag-grid-community";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import type { Metrics } from "@/api/schemas";
import { formatMetric, formatPercent, formatInt } from "@/lib/formatters";
import { modelCategories } from "@/lib/tokens";

ModuleRegistry.registerModules([AllCommunityModule]);

interface LeaderboardTableProps {
  metrics: Metrics[];
  sortMetric?: keyof Metrics;
}

function CategoryDotRenderer({ value }: { value: string }) {
  let color = "var(--color-text-muted)";
  for (const cat of Object.values(modelCategories)) {
    if ((cat.models as readonly string[]).includes(value)) {
      color = cat.color;
      break;
    }
  }
  return (
    <div className="flex items-center gap-2">
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      <span style={{ fontSize: 12 }}>{value}</span>
    </div>
  );
}

function SharpeRenderer({ value }: { value: number | null }) {
  const color =
    value == null
      ? "var(--color-text-muted)"
      : value >= 1
        ? "var(--color-accent-success)"
        : value >= 0.5
          ? "var(--color-text-primary)"
          : "var(--color-accent-danger)";
  return (
    <span style={{ color, fontFamily: "var(--font-mono)", fontSize: 12 }}>
      {formatMetric(value)}
    </span>
  );
}

function ReturnRenderer({ value }: { value: number | null }) {
  if (value == null) return <span style={{ color: "var(--color-text-muted)" }}>—</span>;
  const color = value >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  return (
    <span style={{ color, fontFamily: "var(--font-mono)", fontSize: 12 }}>
      {formatPercent(value)}
    </span>
  );
}

function DrawdownRenderer({ value }: { value: number | null }) {
  if (value == null) return <span style={{ color: "var(--color-text-muted)" }}>—</span>;
  const severity =
    Math.abs(value) < 0.1 ? "var(--color-accent-success)" : Math.abs(value) < 0.2 ? "var(--color-accent-warning)" : "var(--color-accent-danger)";
  return (
    <span style={{ color: severity, fontFamily: "var(--font-mono)", fontSize: 12 }}>
      {formatPercent(value)}
    </span>
  );
}

const MONO = { fontFamily: "var(--font-mono)", fontSize: 12 };

export function LeaderboardTable({ metrics = [], sortMetric = "sharpe" }: LeaderboardTableProps) {
  const sortedMetrics = useMemo(() => {
    const arr = [...metrics];
    arr.sort((a, b) => {
      const aVal = a[sortMetric] as number | null;
      const bVal = b[sortMetric] as number | null;
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;
      return bVal - aVal;
    });
    return arr;
  }, [metrics, sortMetric]);

  const columnDefs = useMemo<ColDef<Metrics>[]>(
    () => [
      {
        headerName: "#",
        valueGetter: (params) => {
          const idx = sortedMetrics.findIndex((m) => m.model === params.data?.model);
          return idx + 1;
        },
        width: 55,
        sortable: false,
        suppressMovable: true,
        cellStyle: MONO,
        cellRenderer: ({ value }: { value: number }) => (
          <span
            className="inline-flex items-center justify-center rounded-full text-[10px] font-bold"
            style={{
              width: 22,
              height: 22,
              backgroundColor: value <= 3 ? "rgba(41,98,255,0.15)" : "transparent",
              color: value <= 3 ? "var(--color-accent)" : "var(--color-text-muted)",
            }}
          >
            {value}
          </span>
        ),
      },
      {
        headerName: "Model",
        field: "model",
        width: 150,
        sortable: true,
        suppressMovable: true,
        cellRenderer: CategoryDotRenderer,
      },
      {
        headerName: "Sharpe",
        field: "sharpe",
        width: 90,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellRenderer: SharpeRenderer,
      },
      {
        headerName: "Sortino",
        field: "sortino",
        width: 90,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: MONO,
        valueFormatter: (p: ValueFormatterParams) => formatMetric(p.value),
      },
      {
        headerName: "Max DD",
        field: "max_drawdown",
        width: 90,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellRenderer: DrawdownRenderer,
      },
      {
        headerName: "Win Rate",
        field: "win_rate",
        width: 90,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: MONO,
        valueFormatter: (p: ValueFormatterParams) => formatPercent(p.value, 1),
      },
      {
        headerName: "PF",
        field: "profit_factor",
        width: 70,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: MONO,
        valueFormatter: (p: ValueFormatterParams) => formatMetric(p.value),
      },
      {
        headerName: "Return",
        field: "total_return",
        width: 95,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellRenderer: ReturnRenderer,
      },
      {
        headerName: "Trades",
        field: "total_trades",
        width: 70,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: MONO,
        valueFormatter: (p: ValueFormatterParams) => formatInt(p.value),
      },
    ],
    [sortedMetrics],
  );

  const defaultColDef = useMemo<ColDef>(() => ({ resizable: true, minWidth: 50 }), []);

  if (metrics.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border p-8"
        style={{
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-border)",
          color: "var(--color-text-muted)",
        }}
      >
        <span className="text-sm" style={{ fontFamily: "var(--font-mono)" }}>
          No metrics data available
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <h3
        className="text-xs font-semibold uppercase tracking-[0.08em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Leaderboard
      </h3>
      <div
        className="ag-theme-alpine-dark rounded-lg border overflow-hidden"
        style={{
          height: Math.min(metrics.length * 40 + 44, 320),
          borderColor: "var(--color-border)",
          backgroundColor: "var(--color-surface)",
          "--ag-background-color": "#1E222D",
          "--ag-header-background-color": "#2A2E39",
          "--ag-odd-row-background-color": "#1a1e29",
          "--ag-row-hover-color": "rgba(41,98,255,0.08)",
          "--ag-selected-row-background-color": "rgba(41,98,255,0.15)",
          "--ag-range-selection-border-color": "#2962FF",
          "--ag-border-color": "#363A45",
          "--ag-header-foreground-color": "#80899F",
          "--ag-foreground-color": "#EDEFF5",
          "--ag-row-border-color": "#2A2E39",
          "--ag-font-size": "12px",
          "--ag-font-family": "Inter, sans-serif",
          "--ag-grid-size": "4px",
          "--ag-header-height": "40px",
          "--ag-row-height": "36px",
        } as React.CSSProperties}
      >
        <AgGridReact<Metrics>
          rowData={sortedMetrics}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          suppressCellFocus
          animateRows={false}
        />
      </div>
    </div>
  );
}
