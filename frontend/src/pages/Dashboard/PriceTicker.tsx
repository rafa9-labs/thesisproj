import { useLivePrices } from "@/api/queries";
import { useNavigate } from "react-router-dom";
import { Settings } from "lucide-react";
import type { LivePrice } from "@/api/schemas";

function Sparkline({ points }: { points: { t: number; v: number }[] }) {
  if (!points || points.length < 2) return <div className="h-[26px] w-full" style={{ backgroundColor: "var(--color-glass-hover)" }} />;

  const values = points.map((p) => p.v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const width = 120;
  const height = 26;
  const isUp = values[values.length - 1] >= values[0];
  const color = isUp ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  const path = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} className="flex-shrink-0">
      <path d={path} fill="none" stroke={color} strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PriceCard({ price }: { price: LivePrice }) {
  const isUp = (price.change_pct ?? 0) >= 0;
  const changeColor = isUp ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  const arrow = isUp ? "▲" : "▼";

  return (
    <div
      className="flex-1 rounded-lg border px-3 py-2.5 min-w-0"
      style={{
        borderColor: "var(--color-glass-border)",
        backgroundColor: "var(--color-glass)",
        backdropFilter: "blur(12px)",
      }}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[12px] font-semibold" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
          {price.symbol}
        </span>
        {price.change_pct != null && (
          <span className="text-[10px] font-medium" style={{ color: changeColor, fontFamily: "var(--font-mono)" }}>
            {arrow} {price.change_pct > 0 ? "+" : ""}{price.change_pct.toFixed(2)}%
          </span>
        )}
      </div>

      <div className="flex items-baseline gap-2 mb-2">
        <div className="flex flex-col">
          <span className="text-[8px] leading-none" style={{ color: "var(--color-text-muted)" }}>BID</span>
          <span className="text-base font-bold leading-none" style={{ color: "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}>
            {price.bid != null ? price.bid.toFixed(5) : "—"}
          </span>
        </div>
        <div className="flex flex-col">
          <span className="text-[8px] leading-none" style={{ color: "var(--color-text-muted)" }}>ASK</span>
          <span className="text-base font-bold leading-none" style={{ color: "var(--color-accent-success)", fontFamily: "var(--font-mono)" }}>
            {price.ask != null ? price.ask.toFixed(5) : "—"}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-1 text-[8px]" style={{ color: "var(--color-text-muted)" }}>
        <span>S: {price.spread_pips != null ? `${price.spread_pips}p` : "—"}</span>
        <span style={{ fontFamily: "var(--font-mono)" }}>
          M: {price.mid != null ? price.mid.toFixed(5) : "—"}
        </span>
      </div>

      {price.sparkline && (
        <div className="mt-1.5">
          <Sparkline points={price.sparkline} />
        </div>
      )}
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="flex-1 rounded-lg border px-3 py-2.5 animate-pulse min-w-0"
      style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
      <div className="h-3 w-14 rounded mb-2" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      <div className="flex gap-2 mb-2">
        <div className="h-8 w-16 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
        <div className="h-8 w-16 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      </div>
      <div className="h-3 w-24 rounded mb-1.5" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      <div className="h-[26px] w-full rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
    </div>
  );
}

export function PriceTicker({ pairs }: { pairs: string[] }) {
  const navigate = useNavigate();
  if (!pairs || pairs.length === 0) return null;
  const displayPairs = pairs.slice(0, 3);
  const { data, isLoading } = useLivePrices(displayPairs);

  if (data?.source === "key_required") {
    return (
      <div
        className="flex items-center gap-3 rounded-lg border px-4 py-3"
        style={{
          borderColor: "var(--color-glass-border)",
          backgroundColor: "var(--color-glass)",
          backdropFilter: "blur(12px)",
        }}
      >
        <Settings size={16} style={{ color: "var(--color-text-muted)" }} />
        <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
          {data.message}
        </span>
        <button
          onClick={() => navigate("/settings")}
          className="text-[11px] font-medium rounded px-2 py-0.5 transition hover:underline"
          style={{ color: "var(--color-brand)" }}
        >
          Settings
        </button>
      </div>
    );
  }

  if (data?.source === "unavailable") {
    return (
      <div
        className="flex items-center gap-3 rounded-lg border px-4 py-3"
        style={{
          borderColor: "var(--color-glass-border)",
          backgroundColor: "var(--color-glass)",
        }}
      >
        <span className="text-[11px]" style={{ color: "var(--color-accent-warning)" }}>
          OANDA API unreachable. Check your connection.
        </span>
      </div>
    );
  }

  return (
    <div className="flex gap-2">
      {isLoading
        ? displayPairs.map((p) => <SkeletonCard key={p} />)
        : (data?.prices ?? displayPairs.map((s) => ({ symbol: s, bid: null, ask: null, mid: null, spread_pips: null, change_pct: null, timestamp: "", sparkline: [] }))).map((price) => (
            <PriceCard key={price.symbol} price={price} />
          ))}
    </div>
  );
}
