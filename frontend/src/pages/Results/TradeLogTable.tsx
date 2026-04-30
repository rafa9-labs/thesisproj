import { useMemo, useCallback } from "react";
import { AgGridReact } from "ag-grid-react";
import type { ColDef } from "ag-grid-community";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import type { TradeRecord } from "@/api/schemas";

ModuleRegistry.registerModules([AllCommunityModule]);

interface TradeLogTableProps {
  trades: TradeRecord[] | null;
  onTradeSelect?: (trade: TradeRecord) => void;
}

function DirectionCellRenderer({ value }: { value: string }) {
  const isBuy = value === "BUY";
  return (
    <span
      className="inline-flex items-center justify-center rounded px-2 py-0.5 text-[10px] font-bold uppercase"
      style={{
        backgroundColor: isBuy ? "rgba(8,153,129,0.15)" : "rgba(242,54,69,0.15)",
        color: isBuy ? "var(--color-accent-success)" : "var(--color-accent-danger)",
      }}
    >
      {value}
    </span>
  );
}

function ReturnCellRenderer({ value }: { value: number }) {
  if (value == null) return <span style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>—</span>;
  const color = value >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  const sign = value >= 0 ? "+" : "";
  return (
    <span style={{ color, fontFamily: "var(--font-mono)", fontSize: 12 }}>
      {sign}
      {value.toFixed(2)}%
    </span>
  );
}

function PipsCellRenderer({ value }: { value: number }) {
  if (value == null) return <span style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>—</span>;
  const color = value >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  const sign = value >= 0 ? "+" : "";
  return (
    <span style={{ color, fontFamily: "var(--font-mono)", fontSize: 12 }}>
      {sign}
      {value.toFixed(1)}
    </span>
  );
}

export function TradeLogTable({ trades, onTradeSelect }: TradeLogTableProps) {
  const data = trades ?? [];

  const columnDefs = useMemo<ColDef<TradeRecord>[]>(
    () => [
      {
        headerName: "#",
        field: "trade_id",
        width: 60,
        sortable: true,
        suppressMovable: true,
        cellStyle: { fontFamily: "var(--font-mono)", fontSize: 12 },
      },
      {
        headerName: "Entry Date",
        field: "entry_date",
        width: 130,
        sortable: true,
        suppressMovable: true,
        cellStyle: { fontFamily: "var(--font-mono)", fontSize: 12 },
      },
      {
        headerName: "Exit Date",
        field: "exit_date",
        width: 130,
        sortable: true,
        suppressMovable: true,
        cellStyle: { fontFamily: "var(--font-mono)", fontSize: 12 },
      },
      {
        headerName: "Dir",
        field: "direction",
        width: 80,
        sortable: true,
        suppressMovable: true,
        cellRenderer: DirectionCellRenderer,
      },
      {
        headerName: "Entry",
        field: "entry_price",
        width: 100,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: { fontFamily: "var(--font-mono)", fontSize: 12 },
        valueFormatter: (p) => (p.value != null ? p.value.toFixed(5) : ""),
      },
      {
        headerName: "Exit",
        field: "exit_price",
        width: 100,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: { fontFamily: "var(--font-mono)", fontSize: 12 },
        valueFormatter: (p) => (p.value != null ? p.value.toFixed(5) : ""),
      },
      {
        headerName: "Pips",
        field: "pips",
        width: 80,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellRenderer: PipsCellRenderer,
      },
      {
        headerName: "Return",
        field: "return_pct",
        width: 90,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellRenderer: ReturnCellRenderer,
      },
      {
        headerName: "Bars",
        field: "duration_bars",
        width: 70,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: { fontFamily: "var(--font-mono)", fontSize: 12 },
      },
      {
        headerName: "Barrier",
        field: "barrier_hit",
        width: 100,
        sortable: true,
        suppressMovable: true,
        cellStyle: { fontSize: 12, fontFamily: "var(--font-mono)" },
        valueFormatter: (p) => p.value ?? "—",
      },
    ],
    [],
  );

  const defaultColDef = useMemo<ColDef>(
    () => ({
      resizable: true,
      minWidth: 50,
    }),
    [],
  );

  const onRowClicked = useCallback(
    (event: { data: TradeRecord | undefined }) => {
      if (event.data && onTradeSelect) {
        onTradeSelect(event.data);
      }
    },
    [onTradeSelect],
  );

  if (data.length === 0) {
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
          No trade data available
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Trade Log
        </h3>
        <span
          className="text-xs"
          style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
        >
          {data.length.toLocaleString()} trades
        </span>
      </div>
      <div
        className="ag-theme-alpine-dark rounded-lg border overflow-hidden"
        style={{
          height: 360,
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
        <AgGridReact<TradeRecord>
          rowData={data}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          onRowClicked={onRowClicked}
          rowSelection="single"
          suppressCellFocus={false}
          suppressRowClickSelection={true}
          animateRows={false}
        />
      </div>
    </div>
  );
}
