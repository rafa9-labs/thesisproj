import { TrendingUp, TrendingDown, Minus } from "lucide-react";

export type SignalDirection = "LONG" | "SHORT" | "FLAT";

interface PositionMonitorProps {
  position: SignalDirection;
  equity: number;
  unrealizedPnl: number;
  initialEquity: number;
  tradeCount: number;
}

export function PositionBadge({ position }: { position: SignalDirection }) {
  const isLong = position === "LONG";
  const isShort = position === "SHORT";
  const color = isLong
    ? "var(--color-accent-success)"
    : isShort
      ? "var(--color-accent-danger)"
      : "var(--color-text-muted)";
  const icon = isLong ? (
    <TrendingUp size={14} />
  ) : isShort ? (
    <TrendingDown size={14} />
  ) : (
    <Minus size={14} />
  );
  const label = isLong ? "LONG" : isShort ? "SHORT" : "FLAT";
  return (
    <div
      className="flex items-center gap-1.5 rounded-md border px-3 py-1.5"
      style={{
        borderColor: color,
        backgroundColor: isLong
          ? "rgba(34,197,94,0.1)"
          : isShort
            ? "rgba(239,68,68,0.1)"
            : "transparent",
      }}
    >
      {icon}
      <span className="font-mono text-[11px] font-semibold uppercase">{label}</span>
    </div>
  );
}

export function PositionMonitor({
  position,
  equity,
  unrealizedPnl,
  initialEquity,
  tradeCount,
}: PositionMonitorProps) {
  const returnPct = initialEquity > 0 ? ((equity - initialEquity) / initialEquity) * 100 : 0;
  return (
    <div className="rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-3">
      <h4 className="mb-3 text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
        Position
      </h4>
      <div className="flex flex-col gap-2">
        <PositionBadge position={position} />
        <Row label="Equity" value={equity.toFixed(2)} />
        <Row
          label="Unreal P&L"
          value={`${unrealizedPnl >= 0 ? "+" : ""}${unrealizedPnl.toFixed(2)}`}
          color={unrealizedPnl >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)"}
        />
        <Row
          label="Return"
          value={`${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(2)}%`}
          color={returnPct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)"}
        />
        <Row label="Trades" value={String(tradeCount)} />
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  color = "var(--color-text-primary)",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex justify-between">
      <span className="text-[10px] text-(--color-text-muted)">{label}</span>
      <span className="font-mono text-[11px] font-medium">{value}</span>
    </div>
  );
}
