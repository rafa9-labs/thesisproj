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
  const header =
    "trade_id,direction,size,entry_price,exit_price,pnl,exit_reason,risk_blocked,oanda_order_id";
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
    <div className="flex min-h-0 flex-1 flex-col rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-3">
      {/* Header row */}
      <div className="mb-2 flex items-center justify-between">
        <h4 className="text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
          Trade Journal
        </h4>
        <div className="flex items-center gap-1">
          <button
            onClick={() => downloadCsv(trades)}
            disabled={trades.length === 0}
            className="flex items-center gap-1 rounded-md border border-(--color-glass-border) px-2 py-0.5 text-[9px] font-semibold text-(--color-text-muted) uppercase transition hover:border-[var(--color-border-active)] disabled:cursor-not-allowed disabled:opacity-30"
            title="Export CSV"
          >
            <FileSpreadsheet size={10} />
            CSV
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="mb-2 flex items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-[rgba(0,0,0,0.15)] px-2 py-1">
        <Search size={10} className="shrink-0 text-(--color-text-muted)" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search ID…"
          className="flex-1 bg-transparent font-mono text-[9px] text-(--color-text-primary) outline-none"
        />
      </div>

      {/* Filter chips */}
      <div className="mb-2 flex items-center gap-1">
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
        <span className="ml-auto font-mono text-[9px] text-(--color-text-muted)">
          {filtered.length}
        </span>
      </div>

      {/* Trade list */}
      <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto [scrollbar-width:thin]">
        {filtered.length === 0 ? (
          <div className="py-4 text-center text-[10px] text-(--color-text-muted)">
            No trades yet
          </div>
        ) : (
          filtered.map((t) => <TradeRow key={t.trade_id} trade={t} />)
        )}
      </div>
    </div>
  );
}

function TradeRow({ trade: t }: { trade: TradeRecord }) {
  if (t.risk_blocked) {
    return (
      <div className="rounded bg-[rgba(245,158,11,0.06)] px-2 py-1.5">
        <div className="flex items-center justify-between text-[10px]">
          <span className="font-mono font-semibold text-(--color-accent-warning)">BLOCKED</span>
          <span className="font-mono font-medium text-(--color-text-muted)">
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
          className="font-mono font-semibold"
          style={{
            color:
              t.direction === "LONG" ? "var(--color-accent-success)" : "var(--color-accent-danger)",
          }}
        >
          {t.direction}
        </span>
        <span className="font-mono font-medium" style={{ color: pnlColor }}>
          {pnlDisplay}
        </span>
        <span className="font-mono font-medium text-(--color-text-muted)">{t.exit_reason}</span>
      </div>
      <div className="mt-0.5 flex items-center justify-between text-[9px]">
        <span className="font-mono text-[9px] text-(--color-text-muted)">
          Entry {t.entry_price.toFixed(5)}
        </span>
        <span className="font-mono text-[9px] text-(--color-text-muted)">
          Exit {t.exit_price?.toFixed(5) ?? "—"}
        </span>
        <span className="font-mono text-[9px] text-(--color-text-muted)">x{t.size.toFixed(2)}</span>
      </div>
      {t.oanda_order_id && (
        <div className="mt-0.5 font-mono text-[8px] text-(--color-text-muted)">
          OANDA: {t.oanda_order_id.slice(0, 14)}
        </div>
      )}
    </div>
  );
}
