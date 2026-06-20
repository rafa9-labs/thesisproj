import { useMemo, useCallback } from "react";
import { AgGridReact } from "ag-grid-react";
import type { ColDef } from "ag-grid-community";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { TradeRecord } from "@/api/schemas";

ModuleRegistry.registerModules([AllCommunityModule]);

interface TradeLogTableProps {
  trades: TradeRecord[] | null;
  onTradeSelect?: (trade: TradeRecord) => void;
  title: string;
  open: boolean;
  onToggle: () => void;
}

// ── Cell renderers ──────────────────────────────────────────────────────

function DirectionCellRenderer({ value }: { value: string }) {
  const isBuy = value === "BUY";
  return (
    <span
      className="inline-flex h-full items-center justify-center"
    >
      <span
        className="rounded px-2 py-0.5 text-[10px] font-bold uppercase"
        style={{
          backgroundColor: isBuy ? "rgba(8,153,129,0.15)" : "rgba(242,54,69,0.15)",
          color: isBuy ? "var(--color-accent-success)" : "var(--color-accent-danger)",
        }}
      >
        {value}
      </span>
    </span>
  );
}

function ReturnCellRenderer({ value }: { value: number }) {
  if (value == null)
    return <span className="font-mono text-xs text-(--color-text-muted) tabular-nums">&mdash;</span>;
  const pct = value * 100;
  const color = pct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  return (
    <span className="font-mono text-xs font-medium tabular-nums" style={{ color }}>
      {pct >= 0 ? "+" : ""}
      {pct.toFixed(2)}%
    </span>
  );
}

function PipsCellRenderer({ value }: { value: number }) {
  if (value == null)
    return <span className="font-mono text-xs text-(--color-text-muted) tabular-nums">&mdash;</span>;
  const color = value >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  return (
    <span className="font-mono text-xs font-medium tabular-nums" style={{ color }}>
      {value >= 0 ? "+" : ""}
      {value.toFixed(1)}
    </span>
  );
}

// ── Base cell style (applied via defaultColDef) ─────────────────────────

const baseCellStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
};

const monoCellStyle: React.CSSProperties = {
  ...baseCellStyle,
  fontFamily: "var(--font-mono)",
  fontSize: 12,
};

// Total column widths ≈ 935 px — wrapper minWidth is 960
export function TradeLogTable({ trades, onTradeSelect, title, open, onToggle }: TradeLogTableProps) {
  const data = trades ?? [];
  const showAll = open;
  const previewData = showAll ? data : data.slice(-5);
  const hiddenCount = data.length - 5;

  const columnDefs = useMemo<ColDef<TradeRecord>[]>(
    () => [
      {
        headerName: "#",
        field: "trade_id",
        flex: 0.5,
        minWidth: 50,
        sortable: true,
        suppressMovable: true,
        cellStyle: { ...monoCellStyle, justifyContent: "flex-end" },
      },
      {
        headerName: "Entry Date",
        field: "entry_date",
        flex: 1.4,
        minWidth: 100,
        sortable: true,
        suppressMovable: true,
        cellStyle: monoCellStyle,
      },
      {
        headerName: "Exit Date",
        field: "exit_date",
        flex: 1.4,
        minWidth: 100,
        sortable: true,
        suppressMovable: true,
        cellStyle: monoCellStyle,
      },
      {
        headerName: "Dir",
        field: "direction",
        flex: 0.7,
        minWidth: 55,
        sortable: true,
        suppressMovable: true,
        cellRenderer: DirectionCellRenderer,
        cellStyle: { ...baseCellStyle, justifyContent: "center" },
      },
      {
        headerName: "Entry",
        field: "entry_price",
        flex: 1,
        minWidth: 75,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: { ...monoCellStyle, justifyContent: "flex-end" },
        valueFormatter: (p) => (p.value != null ? p.value.toFixed(5) : "—"),
      },
      {
        headerName: "Exit",
        field: "exit_price",
        flex: 1,
        minWidth: 75,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: { ...monoCellStyle, justifyContent: "flex-end" },
        valueFormatter: (p) => (p.value != null ? p.value.toFixed(5) : "—"),
      },
      {
        headerName: "Pips",
        field: "pips",
        flex: 0.8,
        minWidth: 60,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellRenderer: PipsCellRenderer,
        cellStyle: { ...baseCellStyle, justifyContent: "flex-end" },
      },
      {
        headerName: "Return",
        field: "return_pct",
        flex: 0.8,
        minWidth: 65,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellRenderer: ReturnCellRenderer,
        cellStyle: { ...baseCellStyle, justifyContent: "flex-end" },
      },
      {
        headerName: "Bars",
        field: "duration_bars",
        flex: 0.6,
        minWidth: 48,
        sortable: true,
        suppressMovable: true,
        type: "numericColumn",
        cellStyle: { ...monoCellStyle, justifyContent: "flex-end" },
      },
      {
        headerName: "Barrier",
        field: "barrier_hit",
        flex: 1,
        minWidth: 75,
        sortable: true,
        suppressMovable: true,
        cellStyle: monoCellStyle,
        valueFormatter: (p) => p.value ?? "—",
      },
    ],
    [],
  );

  const defaultColDef = useMemo<ColDef>(
    () => ({
      resizable: true,
      sortable: false,
      cellStyle: baseCellStyle,
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

  return (
    <div className="rounded-sm border border-(--color-glass-border)">
      {/* ── Embedded title bar ────────────────────────────────────── */}
      <div className="flex items-center justify-between px-3 py-2">
        <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
          {title}
        </span>
        {data.length > 5 && (
          <button
            onClick={onToggle}
            className="flex cursor-pointer items-center gap-1 text-(--color-text-muted)"
            style={{ border: "none", background: "none" }}
            aria-label={showAll ? "Show less" : "Show all"}
          >
            <span className="font-mono text-[10px]">
              {showAll ? "Show less" : `Show all ${data.length} trades`}
            </span>
            {showAll ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        )}
      </div>

      {/* ── Table body ─────────────────────────────────────────────── */}
      <div className="border-t border-(--color-glass-border)">
        {data.length === 0 ? (
          <div className="flex h-[120px] items-center justify-center text-(--color-text-muted)">
            <span className="font-mono text-xs">No trade data available</span>
          </div>
        ) : (
          <div
            className="ag-theme-alpine-dark w-full overflow-x-auto"
            style={
              {
                height: showAll ? 420 : undefined,
                backgroundColor: "var(--color-surface)",
                "--ag-background-color": "#1E222D",
                "--ag-header-background-color": "#1E293B",
                "--ag-odd-row-background-color": "rgba(255,255,255,0.012)",
                "--ag-row-hover-color": "rgba(41,98,255,0.08)",
                "--ag-selected-row-background-color": "rgba(41,98,255,0.13)",
                "--ag-range-selection-border-color": "#2962FF",
                "--ag-border-color": "#2A2E39",
                "--ag-header-foreground-color": "#80899F",
                "--ag-foreground-color": "#EDEFF5",
                "--ag-row-border-color": "rgba(42,46,57,0.6)",
                "--ag-font-size": "12px",
                "--ag-font-family": "Inter, sans-serif",
                "--ag-grid-size": "3px",
                "--ag-header-height": "38px",
                "--ag-row-height": "40px",
                "--ag-cell-horizontal-padding": "10px",
              } as React.CSSProperties
            }
          >
            <AgGridReact<TradeRecord>
              rowData={previewData}
              columnDefs={columnDefs}
              defaultColDef={defaultColDef}
              onRowClicked={onRowClicked}
              rowSelection="single"
              suppressCellFocus={false}
              suppressRowClickSelection={true}
              animateRows={false}
              headerHeight={38}
              rowHeight={40}
              domLayout={showAll ? "normal" : "autoHeight"}
            />
          </div>
        )}
      </div>
    </div>
  );
}
