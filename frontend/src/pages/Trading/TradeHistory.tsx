import { useState, useMemo } from "react";
import { FileSpreadsheet, Search } from "lucide-react";
import type { SignalDirection } from "./PositionMonitor";

export interface TradeRecord {
  trade_id: string;
  direction: SignalDirection;
  size: number;
  entry_price: number;
  exit_price: number | null;
  pnl: number;
  exit_reason: string;
  time: number;
  risk_blocked?: boolean;
  risk_reason?: string;
  oanda_order_id?: string;
}

type FilterMode = "all" | "wins" | "losses" | "blocked";

interface TradeHistoryProps {
  trades: TradeRecord[];
}

function downloadCsv(trades: TradeRecord[]) {
  const header = "trade_id,direction,size,entry_price,exit_price,pnl,exit_reason,risk_blocked,oanda_order_id";
  const rows = trades.map((t) =>
    [
      t.trade_id,
      t.direction,
      t.size.toFixed(4),
      t.entry_price.toFixed(5),
      t.exit_price?.toFixed(5) ?? "",
      t.pnl.toFixed(4),
      t.exit_reason,
      t.risk_blocked ? "true" : "false",
      t.oanda_order_id ?? "",
    ].join(","),
  );
  const blob = new Blob([header + "\n" + rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function TradeHistory({ trades }: TradeHistoryProps) {
  const [filter, setFilter] = useState<FilterMode>("all");
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    let list = trades;
    if (filter === "wins") list = list.filter((t) => t.pnl > 0);
    if (filter === "losses") list = list.filter((t) => t.pnl <= 0 && !t.risk_blocked);
    if (filter === "blocked") list = list.filter((t) => t.risk_blocked);
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (t) =>
          t.trade_id.toLowerCase().includes(q) ||
          (t.oanda_order_id ?? "").toLowerCase().includes(q),
      );
    }
    return list.reverse();
  }, [trades, filter, search]);

  const filters: { key: FilterMode; label: string }[] = [
    { key: "all", label: "All" },
    { key: "wins", label: "Wins" },
    { key: "losses", label: "Losses" },
    { key: "blocked", label: "Blocked" },
  ];

  return (
    <div
      className="rounded-sm border p-3 flex flex-col flex-1 min-h-0"
      style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <h4
          className="text-[10px] font-medium uppercase tracking-[0.12em]"
          style={{ color: "var(--color-text-muted)" }}
        >
          Trade Journal
        </h4>
        <div className="flex items-center gap-1">
          <button
            onClick={() => downloadCsv(trades)}
            disabled={trades.length === 0}
            className="flex items-center gap-1 rounded-md border px-2 py-0.5 text-[9px] font-semibold uppercase transition hover:border-[var(--color-border-active)] disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              borderColor: "var(--color-glass-border)",
              color: "var(--color-text-muted)",
            }}
            title="Export CSV"
          >
            <FileSpreadsheet size={10} />
            CSV
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="flex items-center gap-1.5 rounded-md border px-2 py-1 mb-2" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "rgba(0,0,0,0.15)" }}>
        <Search size={10} style={{ color: "var(--color-text-muted)", flexShrink: 0 }} />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search ID…"
          className="bg-transparent text-[9px] flex-1 outline-none"
          style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
        />
      </div>

      {/* Filter chips */}
      <div className="flex items-center gap-1 mb-2">
        {filters.map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className="rounded-md border px-2 py-0.5 text-[9px] font-semibold uppercase transition-all"
            style={{
              borderColor: filter === f.key ? "var(--color-brand)" : "var(--color-glass-border)",
              backgroundColor: filter === f.key ? "var(--color-brand-glow)" : "transparent",
              color: filter === f.key ? "var(--color-brand)" : "var(--color-text-muted)",
            }}
          >
            {f.label}
          </button>
        ))}
        <span className="text-[9px] ml-auto" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {filtered.length}
        </span>
      </div>

      {/* Trade list */}
      <div className="flex flex-col gap-1.5 overflow-y-auto flex-1 min-h-0" style={{ scrollbarWidth: "thin" }}>
        {filtered.length === 0 ? (
          <div className="text-[10px] text-center py-4" style={{ color: "var(--color-text-muted)" }}>
            No trades yet
          </div>
        ) : (
          filtered.map((t) => (
            <TradeRow key={t.trade_id} trade={t} />
          ))
        )}
      </div>
    </div>
  );
}

function TradeRow({ trade: t }: { trade: TradeRecord }) {
  if (t.risk_blocked) {
    return (
      <div className="rounded px-2 py-1.5" style={{ backgroundColor: "rgba(245,158,11,0.06)" }}>
        <div className="flex items-center justify-between" style={{ fontSize: "10px" }}>
          <span className="font-semibold" style={{ color: "var(--color-accent-warning)", fontFamily: "var(--font-mono)" }}>
            BLOCKED
          </span>
          <span className="font-medium" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
            {t.risk_reason?.slice(0, 40)}
          </span>
        </div>
      </div>
    );
  }
  const isWin = t.pnl > 0;
  const pnlColor = isWin ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  const pnlDisplay = `${isWin ? "+" : ""}${t.pnl.toFixed(2)}`;
  return (
    <div
      className="rounded px-2 py-1.5"
      style={{ backgroundColor: isWin ? "rgba(34,197,94,0.06)" : "rgba(239,68,68,0.06)" }}
    >
      <div className="flex items-center justify-between" style={{ fontSize: "10px" }}>
        <span
          className="font-semibold"
          style={{
            color: t.direction === "LONG" ? "var(--color-accent-success)" : "var(--color-accent-danger)",
            fontFamily: "var(--font-mono)",
          }}
        >
          {t.direction}
        </span>
        <span className="font-medium" style={{ color: pnlColor, fontFamily: "var(--font-mono)" }}>
          {pnlDisplay}
        </span>
        <span className="font-medium" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {t.exit_reason}
        </span>
      </div>
      <div className="flex items-center justify-between mt-0.5" style={{ fontSize: "9px" }}>
        <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          Entry {t.entry_price.toFixed(5)}
        </span>
        <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          Exit {t.exit_price?.toFixed(5) ?? "—"}
        </span>
        <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          x{t.size.toFixed(2)}
        </span>
      </div>
      {t.oanda_order_id && (
        <div className="text-[8px] mt-0.5" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          OANDA: {t.oanda_order_id.slice(0, 14)}
        </div>
      )}
    </div>
  );
}
