import { useState, useMemo } from "react";
import type { TradeRecord } from "./TradeHistory";

interface TradeTableProps {
  trades: TradeRecord[];
  selectedPair: string;
  className?: string;
  style?: React.CSSProperties;
}

type TabKey = "active" | "history" | "orders";

const TABLE_COLS = ["Symbol", "Type", "Size", "Entry", "Current", "S/L", "T/P", "P&L"];

export function TradeTable({ trades, selectedPair, className = "", style }: TradeTableProps) {
  const [tab, setTab] = useState<TabKey>("active");

  const openTrades = useMemo(() => trades.filter((t) => t.exit_price == null), [trades]);

  const closedTrades = useMemo(() => trades.filter((t) => t.exit_price != null), [trades]);

  const tabs: { key: TabKey; label: string; count: number }[] = [
    { key: "active", label: "Active Trades", count: openTrades.length },
    { key: "history", label: "History", count: closedTrades.length },
    { key: "orders", label: "Orders", count: 0 },
  ];

  const rows = tab === "active" ? openTrades : tab === "history" ? closedTrades : [];

  return (
    <div
      className={`flex min-h-0 flex-col overflow-hidden border-t border-(--color-border-subtle) ${className}`}
      style={style}
    >
      <div className="flex shrink-0 items-center border-b border-(--color-glass-border) px-4">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className="relative flex items-center gap-1.5 px-3 py-2 text-[10px] font-medium uppercase transition"
            style={{
              color: tab === t.key ? "var(--color-text-primary)" : "var(--color-text-muted)",
            }}
          >
            {t.label}
            <span className="font-mono text-[9px] opacity-50">({t.count})</span>
            {tab === t.key && (
              <span className="absolute right-0 bottom-0 left-0 h-0.5 bg-(--color-brand)" />
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto">
        {rows.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-[10px] text-(--color-text-muted)">
            {tab === "active"
              ? "No open positions"
              : tab === "history"
                ? "No trade history yet"
                : "No pending orders"}
          </div>
        ) : (
          <div className="min-w-[640px]">
            <table className="w-full">
              <thead>
                <tr className="border-b border-(--color-glass-border)">
                  {TABLE_COLS.map((col) => (
                    <th
                      key={col}
                      className="px-3 py-1.5 text-left text-[9px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((t, i) => (
                  <TradeRow key={t.trade_id || i} trade={t} pair={selectedPair} tab={tab} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function TradeRow({ trade: t, pair, tab }: { trade: TradeRecord; pair: string; tab: TabKey }) {
  const isWin = t.pnl > 0;
  const isOpen = t.exit_price == null;
  const pnlColor = isWin ? "var(--color-accent-success)" : "var(--color-accent-danger)";

  return (
    <tr
      className="border-b border-[rgba(42,46,57,0.4)] text-[10px] transition hover:bg-(--color-glass-hover)"
      style={{
        backgroundColor: isOpen
          ? "transparent"
          : isWin
            ? "rgba(8,153,129,0.04)"
            : "rgba(242,54,69,0.04)",
      }}
    >
      <td className="px-3 py-1.5 font-mono font-medium text-(--color-text-primary)">{pair}</td>
      <td className="px-3 py-1.5">
        <span
          className="font-mono font-semibold"
          style={{
            color:
              t.direction === "LONG" ? "var(--color-accent-success)" : "var(--color-accent-danger)",
          }}
        >
          {t.direction}
        </span>
      </td>
      <td className="px-3 py-1.5 font-mono text-(--color-text-secondary) tabular-nums">
        {t.size.toFixed(2)}
      </td>
      <td className="px-3 py-1.5 font-mono text-(--color-text-secondary) tabular-nums">
        {t.entry_price.toFixed(5)}
      </td>
      <td className="px-3 py-1.5 font-mono text-(--color-text-secondary) tabular-nums">
        {isOpen ? "\u2014" : (t.exit_price ?? 0).toFixed(5)}
      </td>
      <td className="px-3 py-1.5 font-mono text-(--color-text-muted) tabular-nums">\u2014</td>
      <td className="px-3 py-1.5 font-mono text-(--color-text-muted) tabular-nums">\u2014</td>
      <td className="px-3 py-1.5 font-mono font-semibold tabular-nums">
        <span style={{ color: pnlColor }}>
          {isOpen ? "\u2014" : `${isWin ? "+" : ""}${t.pnl.toFixed(2)}`}
        </span>
      </td>
    </tr>
  );
}
